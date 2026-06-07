"""inspect_mppi.py — see what's in the MPPI demos."""
import pickle, glob, numpy as np

files = sorted(glob.glob("demonstrations/normal_*_seed*_mppi.pkl"))
print(f"Found {len(files)} normal MPPI demos")
print()

box_starts = []
box_ends = []
demo_lengths = []
trivial_pcts = []

for f in files:
    with open(f, 'rb') as fp:
        d = pickle.load(fp)
    
    obs = d["observations"]
    act = d["actions"]
    T = len(obs)
    
    # Trivial action %
    delta = act - obs[:, :6]
    within_01 = np.all(np.abs(delta) < 0.01, axis=1).sum()
    trivial_pct = 100 * within_01 / T
    
    box_start = obs[0, 15:18]
    box_end = obs[-1, 15:18]
    push_dist = np.linalg.norm(box_end[:2] - box_start[:2])
    
    print(f"  {f.split('/')[-1]:30s} T={T:4d}  push={push_dist:.3f}m  trivial={trivial_pct:.1f}%")
    
    box_starts.append(box_start)
    box_ends.append(box_end)
    demo_lengths.append(T)
    trivial_pcts.append(trivial_pct)

box_starts = np.array(box_starts)
box_ends = np.array(box_ends)
print()
print(f"Summary across {len(files)} demos:")
print(f"  Demo length: mean={np.mean(demo_lengths):.0f}, range [{min(demo_lengths)}, {max(demo_lengths)}]")
print(f"  Trivial-action %: mean={np.mean(trivial_pcts):.1f}%")
print(f"  Box start range: x ∈ [{box_starts[:, 0].min():.3f}, {box_starts[:, 0].max():.3f}]")
print(f"                   y ∈ [{box_starts[:, 1].min():.3f}, {box_starts[:, 1].max():.3f}]")
print(f"  Box end range:   x ∈ [{box_ends[:, 0].min():.3f}, {box_ends[:, 0].max():.3f}]")
print(f"                   y ∈ [{box_ends[:, 1].min():.3f}, {box_ends[:, 1].max():.3f}]")

# Check obs structure
with open(files[0], 'rb') as fp:
    d = pickle.load(fp)
print()
print(f"Obs shape: {d['observations'].shape}")
print(f"Act shape: {d['actions'].shape}")
print(f"Sample obs[0]: {d['observations'][0]}")
print(f"Sample act[0]: {d['actions'][0]}")
print(f"Info keys: {list(d.get('info', {}).keys())}")
print(f"Info: {d.get('info', {})}")	
