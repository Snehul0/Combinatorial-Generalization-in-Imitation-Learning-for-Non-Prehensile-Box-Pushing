"""inspect_mppi_v2.py — see the full structure of one MPPI demo."""
import pickle, glob
import numpy as np

files = sorted(glob.glob("demonstrations/normal_*_seed*_mppi.pkl"))
if not files:
    print("No MPPI demos found")
    exit()

# Inspect the first demo deeply
f = files[0]
print(f"Inspecting: {f}\n")

with open(f, 'rb') as fp:
    d = pickle.load(fp)

print(f"Top-level keys: {list(d.keys())}")
print()

for key, val in d.items():
    if isinstance(val, np.ndarray):
        print(f"  {key:20s}: array shape={val.shape}, dtype={val.dtype}")
        if val.size > 0:
            print(f"  {key:20s}  first row: {val[0]}")
    elif isinstance(val, dict):
        print(f"  {key:20s}: dict")
        for k, v in val.items():
            if isinstance(v, np.ndarray):
                print(f"    .{k}: array shape={v.shape}")
            elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], (int, float)):
                print(f"    .{k}: list/tuple of {len(v)} numbers, first 5 = {list(v[:5])}")
            else:
                print(f"    .{k}: {v}")
    elif isinstance(val, list):
        print(f"  {key:20s}: list of {len(val)}")
        if val:
            print(f"  {key:20s}  first: {val[0] if not hasattr(val[0], 'shape') else f'array shape {val[0].shape}'}")
    else:
        print(f"  {key:20s}: {type(val).__name__} = {val}")

print()
print("Observation breakdown (assuming obs[t] = [???]):")
obs = d.get("observations")
if obs is not None and len(obs) > 0:
    print(f"  obs[0]: {obs[0]}")
    print(f"  obs[0].shape: {obs[0].shape}")
    print(f"  obs dim: {obs.shape[1] if obs.ndim == 2 else '?'}")

print()
print("Action breakdown:")
act = d.get("actions")
if act is not None and len(act) > 0:
    print(f"  act[0]: {act[0]}")
    print(f"  act[10]: {act[10] if len(act) > 10 else 'N/A'}")
    print(f"  act[100]: {act[100] if len(act) > 100 else 'N/A'}")
    print(f"  act mean: {act.mean(0)}")
    print(f"  act std: {act.std(0)}")
    print(f"  act range: [{act.min(0)}, {act.max(0)}]")
