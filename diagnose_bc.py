# diagnose_bc.py
import numpy as np
import torch
import torch.nn as nn
import pickle, glob

from Teleop_collector import RobotScene, BOXES, get_obs, move_to_home, setup_scene

# Load checkpoint
ckpt = torch.load('checkpoints/bc_best.pt', map_location='cpu', weights_only=False)

class BCPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(27, 256), nn.ReLU(), nn.Dropout(0.0),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.0),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 6),
        )
    def forward(self, x): return self.net(x)

policy = BCPolicy()
policy.load_state_dict(ckpt["policy"])
policy.eval()

obs_mean = ckpt["obs_mean"]; obs_std = ckpt["obs_std"]
act_mean = ckpt["act_mean"]; act_std = ckpt["act_std"]

# TEST 1: Load a REAL training observation and see what BC predicts
print("="*70)
print("TEST 1: BC prediction on REAL training observations")
print("="*70)

# Need to download a few demos from Drive to test
# For now, simulate: use a fake observation from your scene
rs = RobotScene('scene.xml')
scene_boxes = {"A1": BOXES["A1"][1].copy()}
goal = BOXES["A1"][1].copy()
goal[0] += 0.25
setup_scene(rs, scene_boxes, goal)
move_to_home(rs, steps=200)

obs = get_obs(rs, rs.data, "A1", goal)
print(f"\nObservation from current MuJoCo state (arm at home, box at A1 default):")
print(f"  obs[0:6]  (joint pos):   {obs[0:6].round(3)}")
print(f"  obs[6:12] (joint vel):   {obs[6:12].round(3)}")
print(f"  obs[12:15] (tip):        {obs[12:15].round(3)}")
print(f"  obs[15:18] (box):        {obs[15:18].round(3)}")
print(f"  obs[18:21] (box vel):    {obs[18:21].round(3)}")
print(f"  obs[21:24] (goal):       {obs[21:24].round(3)}")
print(f"  obs[24:27] (goal-box):   {obs[24:27].round(3)}")

# Normalize and predict
norm_obs = (obs - obs_mean) / obs_std
print(f"\nNormalized obs range: [{norm_obs.min():.2f}, {norm_obs.max():.2f}]")

with torch.no_grad():
    norm_pred = policy(torch.tensor(norm_obs, dtype=torch.float32)).numpy()
pred = norm_pred * act_std + act_mean

print(f"\nBC prediction (denormalized):")
print(f"  pred: {pred.round(3)}")
print(f"  current joints: {obs[0:6].round(3)}")
print(f"  pred - current: {(pred - obs[0:6]).round(3)}")

# TEST 2: How does the obs from MuJoCo compare to obs in training?
print("\n" + "="*70)
print("TEST 2: How similar is THIS obs to training observations?")
print("="*70)

# Try to find demos
demo_paths = glob.glob('/home/smartslab/cse579project/.../demonstrations_aggressive_decimated/*.pkl')
if not demo_paths:
    # Try other paths
    demo_paths = glob.glob('demonstrations_aggressive_decimated/*.pkl')
if demo_paths:
    print(f"Found {len(demo_paths)} demos.")
    with open(demo_paths[0], 'rb') as f:
        d = pickle.load(f)
    print(f"\nFirst demo, first observation:")
    print(f"  obs[0:6]:  {d['observations'][0, 0:6].round(3)}")
    print(f"  obs[12:15]: {d['observations'][0, 12:15].round(3)}")
    print(f"  obs[15:18]: {d['observations'][0, 15:18].round(3)}")
    
    print(f"\nDistance from MuJoCo current obs to training obs:")
    dist = np.linalg.norm(obs - d['observations'][0])
    print(f"  Euclidean distance: {dist:.4f}")
else:
    print("Can't find training demos locally to compare.")
    print("(You'd need to download some from Drive)")

# TEST 3: Check obs_mean and obs_std are sensible
print("\n" + "="*70)
print("TEST 3: Normalization statistics")
print("="*70)
print(f"obs_mean: {obs_mean.round(3)}")
print(f"obs_std:  {obs_std.round(3)}")
print(f"act_mean: {act_mean.round(3)}")
print(f"act_std:  {act_std.round(3)}")

# Check if any std is suspiciously small or zero
small_stds = obs_std < 0.001
if small_stds.any():
    print(f"\nWARNING: obs_std has tiny values at indices: {np.where(small_stds)[0]}")
small_act_stds = act_std < 0.001
if small_act_stds.any():
    print(f"WARNING: act_std has tiny values at indices: {np.where(small_act_stds)[0]}")
