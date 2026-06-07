"""
diffusion_model.py - Simplified Diffusion Policy for state input.
Based on Chi et al. 2023.

Updated: uses DDIM sampling for robust subsampled inference.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============== NOISE SCHEDULER ==============

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    f_t = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = f_t / f_t[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(betas, 0.0001, 0.9999)


class NoiseScheduler:
    """DDPM noise scheduler with DDIM-style deterministic inference."""
    
    def __init__(self, num_train_timesteps=100):
        self.num_train_timesteps = num_train_timesteps
        betas = cosine_beta_schedule(num_train_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    
    def to(self, device):
        for attr in ['betas', 'alphas', 'alphas_cumprod',
                     'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod']:
            setattr(self, attr, getattr(self, attr).to(device))
        return self
    
    def add_noise(self, x0, t, noise):
        """Forward diffusion."""
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t]
        while sqrt_alpha.dim() < x0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
        return sqrt_alpha * x0 + sqrt_one_minus * noise
    
    @torch.no_grad()
    def ddim_step(self, predicted_noise, t_curr, t_next, x_t):
        """
        DDIM deterministic update (eta=0).
        Handles subsampled inference correctly.
        
        Args:
            predicted_noise: model output
            t_curr: current timestep index
            t_next: next (smaller) timestep index, or -1 for final
            x_t: current noisy sample
        Returns:
            x_t_next
        """
        alpha_bar_curr = self.alphas_cumprod[t_curr]
        
        # Predict x_0 from x_t and predicted noise
        # x_0 = (x_t - sqrt(1 - alpha_bar_t) * noise) / sqrt(alpha_bar_t)
        sqrt_one_minus = torch.sqrt(1.0 - alpha_bar_curr)
        sqrt_alpha_bar = torch.sqrt(alpha_bar_curr)
        pred_x0 = (x_t - sqrt_one_minus * predicted_noise) / sqrt_alpha_bar
        
        # Clip predicted x_0 for stability (since training data is normalized to ~[-3, 3])
        pred_x0 = torch.clamp(pred_x0, -5.0, 5.0)
        
        if t_next < 0:
            # Final step: return pred_x0 directly
            return pred_x0
        
        # DDIM forward to x_{t_next} from pred_x0
        alpha_bar_next = self.alphas_cumprod[t_next]
        sqrt_alpha_bar_next = torch.sqrt(alpha_bar_next)
        sqrt_one_minus_next = torch.sqrt(1.0 - alpha_bar_next)
        
        # Deterministic DDIM: x_{t_next} = sqrt(alpha_bar_next) * x_0 + sqrt(1 - alpha_bar_next) * noise
        x_t_next = sqrt_alpha_bar_next * pred_x0 + sqrt_one_minus_next * predicted_noise
        return x_t_next


# ============== EMBEDDINGS ==============

class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


# ============== U-NET BLOCKS ==============

class FiLMConditionedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.norm = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.cond_proj = nn.Linear(cond_dim, 2 * out_ch)
        self.activation = nn.Mish()
    
    def forward(self, x, cond):
        x = self.conv(x)
        x = self.norm(x)
        cond_out = self.cond_proj(cond)
        scale, shift = cond_out.chunk(2, dim=-1)
        x = x * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        x = self.activation(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, downsample=True):
        super().__init__()
        self.block1 = FiLMConditionedConvBlock(in_ch, out_ch, cond_dim)
        self.block2 = FiLMConditionedConvBlock(out_ch, out_ch, cond_dim)
        if downsample:
            self.downsample = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)
        else:
            self.downsample = nn.Identity()
    
    def forward(self, x, cond):
        x = self.block1(x, cond)
        x = self.block2(x, cond)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, cond_dim, upsample=True):
        super().__init__()
        if upsample:
            self.upsample = nn.ConvTranspose1d(in_ch, in_ch, kernel_size=4, stride=2, padding=1)
        else:
            self.upsample = nn.Identity()
        self.block1 = FiLMConditionedConvBlock(in_ch + skip_ch, out_ch, cond_dim)
        self.block2 = FiLMConditionedConvBlock(out_ch, out_ch, cond_dim)
    
    def forward(self, x, skip, cond):
        x = self.upsample(x)
        if x.shape[-1] != skip.shape[-1]:
            x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, cond)
        x = self.block2(x, cond)
        return x


# ============== MAIN DIFFUSION MODEL ==============

class DiffusionPolicy(nn.Module):
    def __init__(self, obs_dim=27, act_dim=6, chunk_size=6,
                 time_emb_dim=128, obs_emb_dim=128,
                 down_dims=(128, 256)):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.chunk_size = chunk_size
        cond_dim = time_emb_dim + obs_emb_dim
        
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.Mish(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        self.obs_embed = nn.Sequential(
            nn.Linear(obs_dim, obs_emb_dim * 2),
            nn.Mish(),
            nn.Linear(obs_emb_dim * 2, obs_emb_dim),
        )
        
        self.input_proj = nn.Conv1d(act_dim, down_dims[0], kernel_size=1)
        
        self.down_blocks = nn.ModuleList()
        ch_in = down_dims[0]
        for i, ch_out in enumerate(down_dims):
            is_last = i == len(down_dims) - 1
            self.down_blocks.append(
                DownBlock(ch_in, ch_out, cond_dim, downsample=not is_last)
            )
            ch_in = ch_out
        
        self.mid1 = FiLMConditionedConvBlock(down_dims[-1], down_dims[-1], cond_dim)
        self.mid2 = FiLMConditionedConvBlock(down_dims[-1], down_dims[-1], cond_dim)
        
        self.up_blocks = nn.ModuleList()
        for i, ch_out in enumerate(reversed(down_dims[:-1])):
            is_first = i == 0
            ch_in_up = down_dims[-1] if is_first else down_dims[-1 - i + 1]
            self.up_blocks.append(
                UpBlock(ch_in_up, ch_out, ch_out, cond_dim, upsample=True)
            )
        
        self.output_proj = nn.Conv1d(down_dims[0], act_dim, kernel_size=1)
    
    def forward(self, noisy_chunk, timestep, obs):
        t_emb = self.time_embed(timestep)
        o_emb = self.obs_embed(obs)
        cond = torch.cat([t_emb, o_emb], dim=-1)
        
        x = noisy_chunk.transpose(1, 2)
        x = self.input_proj(x)
        
        skips = []
        for block in self.down_blocks:
            x, skip = block(x, cond)
            skips.append(skip)
        
        x = self.mid1(x, cond)
        x = self.mid2(x, cond)
        
        for i, block in enumerate(self.up_blocks):
            skip = skips[-2 - i]
            x = block(x, skip, cond)
        
        x = self.output_proj(x)
        return x.transpose(1, 2)


# ============== INFERENCE (DDIM denoising) ==============

@torch.no_grad()
def sample_action_chunk(model, scheduler, obs, num_inference_steps=50, device='cuda'):
    """
    Generate action chunk via DDIM denoising (deterministic, robust to subsampling).
    """
    B = obs.shape[0]
    model.eval()
    
    # Start from pure noise
    x_t = torch.randn(B, model.chunk_size, model.act_dim, device=device)
    
    # Pick timesteps to use (descending from num_train - 1 down to 0)
    timesteps = torch.linspace(
        scheduler.num_train_timesteps - 1, 0, num_inference_steps + 1, device=device
    ).long()
    
    for i in range(num_inference_steps):
        t_curr = timesteps[i].item()
        t_next = timesteps[i + 1].item() if i + 1 < num_inference_steps else -1
        
        t_batch = torch.full((B,), t_curr, dtype=torch.long, device=device)
        predicted_noise = model(x_t, t_batch, obs)
        x_t = scheduler.ddim_step(predicted_noise, t_curr, t_next, x_t)
    
    x_t = torch.clamp(x_t, -2.0, 2.0)
    return x_t


# ============== SANITY CHECK ==============

if __name__ == "__main__":
    print("Testing DiffusionPolicy model construction and forward pass...")
    
    model = DiffusionPolicy(obs_dim=27, act_dim=6, chunk_size=6)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
    
    scheduler = NoiseScheduler(num_train_timesteps=100)
    
    B = 4
    obs = torch.randn(B, 27)
    chunk = torch.randn(B, 6, 6)
    
    t = torch.randint(0, 100, (B,))
    noise = torch.randn_like(chunk)
    noisy_chunk = scheduler.add_noise(chunk, t, noise)
    pred_noise = model(noisy_chunk, t, obs)
    
    print(f"Training forward: noisy {noisy_chunk.shape}, pred_noise {pred_noise.shape}")
    loss = F.mse_loss(pred_noise, noise)
    print(f"Initial noise-prediction loss: {loss.item():.4f}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    scheduler = scheduler.to(device)
    obs = obs.to(device)
    action_chunk = sample_action_chunk(model, scheduler, obs,
                                        num_inference_steps=20, device=device)
    print(f"Inference forward (full denoising): {action_chunk.shape}")
    
    print("Sanity check passed!")
