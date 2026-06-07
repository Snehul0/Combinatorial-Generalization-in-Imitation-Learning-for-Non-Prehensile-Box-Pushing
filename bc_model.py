"""
bc_model.py - Simple MLP behavioral cloning policy.
Configurable obs_dim for both 27D and 51D experiments.
"""

import torch
import torch.nn as nn


class BCPolicy(nn.Module):
    def __init__(self, obs_dim=27, act_dim=6, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, act_dim),
        )
    
    def forward(self, obs):
        return self.net(obs)


if __name__ == "__main__":
    for obs_dim in [27, 51]:
        m = BCPolicy(obs_dim=obs_dim)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"obs_dim={obs_dim}: {n_params:,} parameters")
