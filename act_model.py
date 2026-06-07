"""
act_model.py - Simplified ACT (Action Chunking with Transformers) for state input.

Based on Zhao et al. 2023 "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
Simplified for low-dimensional state observations (no image encoders).

Architecture:
  - Encoder (CVAE encoder, training only):
      Input:  observation (27D) + ground-truth action chunk (K x 6D)
      Output: latent z parameters (mean, logvar)
  
  - Decoder (policy, used at training and inference):
      Input:  observation (27D) + latent z (32D)
      Output: action chunk (K x 6D)

At inference, latent z = 0 (deterministic mean of prior N(0, I)).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """Simple transformer block: self-attention + feedforward, with pre-norm."""
    
    def __init__(self, dim, n_heads, ff_mult=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x, mask=None):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class DecoderBlock(nn.Module):
    """Cross-attention decoder block: self-attn on queries + cross-attn to memory + FF."""
    
    def __init__(self, dim, n_heads, ff_mult=4, dropout=0.1):
        super().__init__()
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, q, memory):
        q_norm = self.norm_self(q)
        sa, _ = self.self_attn(q_norm, q_norm, q_norm, need_weights=False)
        q = q + sa
        
        q_norm = self.norm_cross(q)
        ca, _ = self.cross_attn(q_norm, memory, memory, need_weights=False)
        q = q + ca
        
        q = q + self.ff(self.norm_ff(q))
        return q


class ACTEncoder(nn.Module):
    """CVAE encoder. Used only during training."""
    
    def __init__(self, obs_dim=27, act_dim=6, chunk_size=6, hidden=256,
                 latent_dim=32, n_layers=2, n_heads=4):
        super().__init__()
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        
        self.obs_proj = nn.Linear(obs_dim, hidden)
        self.act_proj = nn.Linear(act_dim, hidden)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.normal_(self.cls_token, std=0.02)
        
        seq_len = 1 + 1 + chunk_size  # [CLS] + obs + chunk_size acts
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden))
        nn.init.normal_(self.pos_embed, std=0.02)
        
        self.layers = nn.ModuleList([
            TransformerBlock(hidden, n_heads) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden)
        
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
    
    def forward(self, obs, action_chunk):
        B = obs.shape[0]
        obs_emb = self.obs_proj(obs).unsqueeze(1)
        act_emb = self.act_proj(action_chunk)
        cls = self.cls_token.expand(B, -1, -1)
        
        x = torch.cat([cls, obs_emb, act_emb], dim=1)
        x = x + self.pos_embed
        
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        
        cls_out = x[:, 0]
        mu = self.fc_mu(cls_out)
        logvar = self.fc_logvar(cls_out)
        return mu, logvar


class ACTDecoder(nn.Module):
    """CVAE decoder / policy. Used at training AND inference."""
    
    def __init__(self, obs_dim=27, act_dim=6, chunk_size=6, hidden=256,
                 latent_dim=32, n_layers=2, n_heads=4):
        super().__init__()
        self.chunk_size = chunk_size
        self.hidden = hidden
        
        self.obs_proj = nn.Linear(obs_dim, hidden)
        self.z_proj = nn.Linear(latent_dim, hidden)
        
        self.pos_queries = nn.Parameter(torch.zeros(1, chunk_size, hidden))
        nn.init.normal_(self.pos_queries, std=0.02)
        
        self.layers = nn.ModuleList([
            DecoderBlock(hidden, n_heads) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden)
        
        self.out = nn.Linear(hidden, act_dim)
    
    def forward(self, obs, z):
        B = obs.shape[0]
        obs_tok = self.obs_proj(obs).unsqueeze(1)
        z_tok = self.z_proj(z).unsqueeze(1)
        memory = torch.cat([obs_tok, z_tok], dim=1)
        
        q = self.pos_queries.expand(B, -1, -1)
        
        for layer in self.layers:
            q = layer(q, memory)
        q = self.norm(q)
        
        action_chunk = self.out(q)
        return action_chunk


class ACT(nn.Module):
    """Full ACT model combining encoder and decoder."""
    
    def __init__(self, obs_dim=27, act_dim=6, chunk_size=6, hidden=256,
                 latent_dim=32, n_enc_layers=2, n_dec_layers=2, n_heads=4):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        
        self.encoder = ACTEncoder(
            obs_dim, act_dim, chunk_size, hidden, latent_dim, n_enc_layers, n_heads
        )
        self.decoder = ACTDecoder(
            obs_dim, act_dim, chunk_size, hidden, latent_dim, n_dec_layers, n_heads
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def training_forward(self, obs, action_chunk):
        mu, logvar = self.encoder(obs, action_chunk)
        z = self.reparameterize(mu, logvar)
        pred_chunk = self.decoder(obs, z)
        return pred_chunk, mu, logvar
    
    def inference_forward(self, obs):
        B = obs.shape[0]
        z = torch.zeros(B, self.latent_dim, device=obs.device, dtype=obs.dtype)
        pred_chunk = self.decoder(obs, z)
        return pred_chunk
    
    def forward(self, obs, action_chunk=None):
        if action_chunk is not None:
            return self.training_forward(obs, action_chunk)
        else:
            return self.inference_forward(obs)


def act_loss(pred_chunk, gt_chunk, mu, logvar, kl_weight=10.0):
    """
    ACT loss = MSE reconstruction + KL divergence (β-weighted).
    
    Returns dict with 'total', 'recon', 'kl' for logging.
    """
    recon = F.mse_loss(pred_chunk, gt_chunk)
    # KL(N(mu, sigma^2) || N(0, I))
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    total = recon + kl_weight * kl
    return {"total": total, "recon": recon, "kl": kl}


if __name__ == "__main__":
    print("Testing ACT model construction and forward pass...")
    
    model = ACT(obs_dim=27, act_dim=6, chunk_size=6, hidden=256, latent_dim=32)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
    
    B = 4
    obs = torch.randn(B, 27)
    chunk = torch.randn(B, 6, 6)
    
    pred, mu, logvar = model.training_forward(obs, chunk)
    print(f"Training pass: pred {pred.shape}, mu {mu.shape}, logvar {logvar.shape}")
    
    losses = act_loss(pred, chunk, mu, logvar)
    print(f"Loss: total={losses['total'].item():.4f}, "
          f"recon={losses['recon'].item():.4f}, "
          f"kl={losses['kl'].item():.4f}")
    
    pred_inf = model.inference_forward(obs)
    print(f"Inference pass: pred {pred_inf.shape}")
    
    print("Sanity check passed!")
