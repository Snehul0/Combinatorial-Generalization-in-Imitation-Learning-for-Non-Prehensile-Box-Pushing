"""inspect_phase2.py — analyze Phase 2 (non-zero actions) of MPPI demos."""
import pickle, glob
import numpy as np

files = sorted(glob.glob("demonstrations/normal_*_seed*_mppi.pkl"))
print(f"Found {len(files)} normal MPPI demos\n")

phase1_lengths = []
phase2_lengths = []
push_distances = []
push_directions = []

for f in files:
    with open(f, 'rb') as fp:
        d = pickle.load(fp)
    
    obs = d["observations"]
    act = d["actions"]
    info = d["info"]
    
    # Find where Phase 2 starts (first non-zero action)
    is_zero = np.all(np.abs(act) < 1e-7, axis=1)
    phase1_end = np.argmax(~is_zero) if not is_zero.all() else len(act)
    phase1_len = phase1_end
    phase2_len = len(act) - phase1_end
    
    box_start = np.array(info["box_start"])
    goal = np.array(info["goal"])
    push_dir = np.array(info["push_dir"])
    push_dist = np.linalg.norm(goal[:2] - box_start[:2])
    
    box_actual_end = obs[-1, 15:18]
    actual_push = np.linalg.norm(box_actual_end[:2] - box_start[:2])
    
    print(f"  {f.split('/')[-1]:35s} "
          f"T={len(obs):4d}  P1={phase1_len:4d}  P2={phase2_len:4d}  "
          f"push_dir=[{push_dir[0]:.2f},{push_dir[1]:.2f}]  "
          f"actual={actual_push:.3f}m")
    
    phase1_lengths.append(phase1_len)
    phase2_lengths.append(phase2_len)
    push_distances.append(push_dist)
    push_directions.append(push_dir)

print(f"\nSummary:")
print(f"  Phase 1 length: mean={np.mean(phase1_lengths):.0f}, range [{min(phase1_lengths)}, {max(phase1_lengths)}]")
print(f"  Phase 2 length: mean={np.mean(phase2_lengths):.0f}, range [{min(phase2_lengths)}, {max(phase2_lengths)}]")
print(f"  Total Phase 2 samples: {sum(phase2_lengths):,}")
print(f"  Push directions:")
push_directions = np.array(push_directions)
unique_dirs = np.unique(np.round(push_directions, 2), axis=0)
for d in unique_dirs:
    count = np.sum(np.all(np.abs(push_directions - d) < 0.05, axis=1))
    print(f"    [{d[0]:+.2f}, {d[1]:+.2f}, {d[2]:+.2f}]: {count} demos")
