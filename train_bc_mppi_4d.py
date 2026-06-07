"""
train_bc_mppi_4d.py
====================
BC with 4D Cartesian-delta output, trained on MPPI Phase 2 data.
"""
import pickle, glob, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


DEMO_GLOB = "demonstrations/normal_*_seed*_mppi.pkl"
CHECKPOINT = "checkpoints/bc_mppi_4d.pt"
EPOCHS = 1500
BATCH = 256
LR = 3e-4
VAL_FRAC = 0.1


class BCPolicy(nn.Module):
    """27D obs -> 4D Cartesian delta (dx, dy, dz, dw)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(27, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 4),
        )
    def forward(self, x):
        return self.net(x)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    files = sorted(glob.glob(DEMO_GLOB))
    print(f"Found {len(files)} demos")
    
    obs_eps = []
    act_eps = []
    n_p1_dropped = 0
    
    for f in files:
        with open(f, 'rb') as fp:
            d = pickle.load(fp)
        obs = d["observations"]
        act = d["actions"]
        
        # Phase 1 = leading zero actions. Drop them.
        is_zero = np.all(np.abs(act) < 1e-7, axis=1)
        if is_zero.all():
            continue
        phase2_start = np.argmax(~is_zero)
        
        # Optionally also drop trailing zeros if any (rarely)
        last_nonzero = len(act) - np.argmax(~is_zero[::-1])
        
        n_p1_dropped += phase2_start
        
        obs_p2 = obs[phase2_start:last_nonzero].astype(np.float32)
        act_p2 = act[phase2_start:last_nonzero].astype(np.float32)
        
        if len(obs_p2) < 30:
            continue
        
        obs_eps.append(obs_p2)
        act_eps.append(act_p2)
    
    print(f"Loaded {len(obs_eps)} demos after dropping Phase 1")
    print(f"Phase 1 timesteps dropped: {n_p1_dropped:,}")
    
    if not obs_eps:
        print("No demos loaded!"); return
    
    # Train/val split by episode
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(obs_eps))
    n_val = max(1, int(len(obs_eps) * VAL_FRAC))
    val_set = set(indices[:n_val].tolist())
    
    train_o = np.concatenate([obs_eps[i] for i in range(len(obs_eps)) if i not in val_set])
    train_a = np.concatenate([act_eps[i] for i in range(len(act_eps)) if i not in val_set])
    val_o = np.concatenate([obs_eps[i] for i in range(len(obs_eps)) if i in val_set])
    val_a = np.concatenate([act_eps[i] for i in range(len(act_eps)) if i in val_set])
    
    print(f"Train samples: {len(train_o):,}")
    print(f"Val   samples: {len(val_o):,}")
    
    # Normalize
    obs_mean = train_o.mean(0)
    obs_std = train_o.std(0) + 1e-6
    act_mean = train_a.mean(0)
    act_std = train_a.std(0) + 1e-6
    
    print(f"\nAction statistics (Phase 2 only):")
    print(f"  Mean: {act_mean.round(5)}")
    print(f"  Std:  {act_std.round(5)}")
    print(f"  % zero actions: {(np.all(np.abs(train_a) < 1e-7, axis=1)).sum() / len(train_a) * 100:.2f}%")
    
    train_on = (train_o - obs_mean) / obs_std
    train_an = (train_a - act_mean) / act_std
    val_on = (val_o - obs_mean) / obs_std
    val_an = (val_a - act_mean) / act_std
    
    train_dl = DataLoader(
        TensorDataset(torch.tensor(train_on), torch.tensor(train_an)),
        batch_size=BATCH, shuffle=True, num_workers=0,
    )
    val_dl = DataLoader(
        TensorDataset(torch.tensor(val_on), torch.tensor(val_an)),
        batch_size=BATCH, shuffle=False, num_workers=0,
    )
    
    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    
    policy = BCPolicy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    
    print(f"\nTraining for {EPOCHS} epochs...")
    start = time.time()
    best_val = float('inf')
    
    for epoch in range(EPOCHS):
        policy.train()
        for x, y in train_dl:
            x = x.to(device); y = y.to(device)
            loss = ((policy(x) - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
        sched.step()
        
        if epoch % 25 == 0 or epoch == EPOCHS - 1:
            policy.eval()
            val_loss = 0.0; n = 0
            with torch.no_grad():
                for x, y in val_dl:
                    x = x.to(device); y = y.to(device)
                    val_loss += ((policy(x) - y) ** 2).mean().item()
                    n += 1
            val_loss /= max(n, 1)
            
            elapsed = time.time() - start
            print(f"  Epoch {epoch:4d}/{EPOCHS}  val={val_loss:.6f}  best={min(val_loss, best_val):.6f}  [{elapsed:.0f}s]")
            
            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "policy": policy.state_dict(),
                    "obs_mean": obs_mean, "obs_std": obs_std,
                    "act_mean": act_mean, "act_std": act_std,
                    "epoch": epoch, "val_loss": val_loss,
                    "config": {"obs_dim": 27, "act_dim": 4, "hidden": 256,
                               "action_format": "cartesian_delta_dxdydzdw"},
                }, CHECKPOINT)
    
    print(f"\nDone. Best val: {best_val:.6f}")
    print(f"Saved: {CHECKPOINT}")
    
    # Sanity check
    print(f"\nSanity check on 5 samples:")
    policy.eval()
    sample_idx = rng.choice(len(train_o), 5, replace=False)
    with torch.no_grad():
        for i in sample_idx:
            obs_i = train_o[i]
            act_gt = train_a[i]
            norm_obs = torch.tensor((obs_i - obs_mean) / obs_std,
                                    dtype=torch.float32, device=device).unsqueeze(0)
            norm_pred = policy(norm_obs).squeeze(0).cpu().numpy()
            pred = norm_pred * act_std + act_mean
            err = np.linalg.norm(act_gt - pred)
            print(f"  Sample {i}: GT={act_gt.round(4)}, pred={pred.round(4)}, err={err:.5f}")


if __name__ == "__main__":
    main()
