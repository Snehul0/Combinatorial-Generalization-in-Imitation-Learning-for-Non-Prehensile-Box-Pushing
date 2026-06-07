"""
make_datasets_27d_51d.py
========================

Slices the 51D demos into two parallel datasets:
  - multibox_27d_demos.pkl: each obs is just the first 27 dims (no blocker info)
  - multibox_51d_demos.pkl: each obs is the full 51D (with blocker info)

Same trajectories, same actions, same dones — only the observation width differs.
This is the cleanest test of "does representation matter."
"""

import os
import pickle
import glob
import numpy as np


SOURCE_SUCCESS_DIR = "demonstrations_teleop_51d"
SOURCE_HER_DIR = "demonstrations_her_51d"
OUTPUT_DIR = "datasets_27d_vs_51d"

INCLUDE_HER = True  # set False to use only SPACE successes


def load_all_demos():
    """Load all 51D demos from disk, returning list of demo dicts."""
    success_files = sorted(glob.glob(f"{SOURCE_SUCCESS_DIR}/*.pkl"))
    print(f"Found {len(success_files)} success demos")
    
    demos = []
    for f in success_files:
        with open(f, "rb") as fp:
            d = pickle.load(fp)
        demos.append(d)
    
    if INCLUDE_HER:
        her_files = sorted(glob.glob(f"{SOURCE_HER_DIR}/*.pkl"))
        print(f"Found {len(her_files)} HER demos")
        for f in her_files:
            with open(f, "rb") as fp:
                d = pickle.load(fp)
            
            # HER relabeling: replace original goal with achieved goal in obs
            achieved = np.array(d["info"]["achieved_goal"], dtype=np.float32)
            new_obs = d["observations"].copy()
            # In the 27D base, goal is at indices 21,22,23 and goal-box at 24,25,26
            # box is at indices 15,16,17
            for t in range(len(new_obs)):
                box = new_obs[t, 15:18]
                new_obs[t, 21:24] = achieved
                new_obs[t, 24:27] = achieved - box
            d_relabeled = dict(d)
            d_relabeled["observations"] = new_obs
            d_relabeled["info"] = dict(d["info"])
            d_relabeled["info"]["goal"] = achieved.tolist()
            d_relabeled["info"]["her_relabeled"] = True
            demos.append(d_relabeled)
    
    print(f"Total demos: {len(demos)}")
    return demos


def slice_obs_27d(demos):
    """Create copies of demos with obs sliced to first 27 dims."""
    out = []
    for d in demos:
        d_new = dict(d)
        d_new["observations"] = d["observations"][:, :27].copy()
        d_new["info"] = dict(d["info"])
        d_new["info"]["obs_dim"] = 27
        out.append(d_new)
    return out


def slice_obs_51d(demos):
    """Keep full 51D (just copy)."""
    out = []
    for d in demos:
        d_new = dict(d)
        d_new["observations"] = d["observations"].copy()
        d_new["info"] = dict(d["info"])
        d_new["info"]["obs_dim"] = 51
        out.append(d_new)
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    demos = load_all_demos()
    
    # Total sample count
    total_samples = sum(len(d["observations"]) for d in demos)
    print(f"Total timestep samples across all demos: {total_samples}")
    
    # Create 27D dataset
    demos_27d = slice_obs_27d(demos)
    out_27d = f"{OUTPUT_DIR}/multibox_27d_demos.pkl"
    with open(out_27d, "wb") as f:
        pickle.dump(demos_27d, f)
    
    sample_obs_27d = demos_27d[0]["observations"][0]
    print(f"\n27D dataset saved: {out_27d}")
    print(f"  num demos: {len(demos_27d)}")
    print(f"  first demo obs shape: {demos_27d[0]['observations'].shape}")
    print(f"  first obs sample: {sample_obs_27d.round(2)}")
    print(f"  first obs length: {len(sample_obs_27d)} (should be 27)")
    
    # Create 51D dataset
    demos_51d = slice_obs_51d(demos)
    out_51d = f"{OUTPUT_DIR}/multibox_51d_demos.pkl"
    with open(out_51d, "wb") as f:
        pickle.dump(demos_51d, f)
    
    sample_obs_51d = demos_51d[0]["observations"][0]
    print(f"\n51D dataset saved: {out_51d}")
    print(f"  num demos: {len(demos_51d)}")
    print(f"  first demo obs shape: {demos_51d[0]['observations'].shape}")
    print(f"  first obs length: {len(sample_obs_51d)} (should be 51)")
    print(f"  last 6 of first obs (blocker info): {sample_obs_51d[-6:].round(2)}")
    
    # Sanity check: 27D should equal first 27 dims of 51D
    obs_27 = demos_27d[0]["observations"]
    obs_51 = demos_51d[0]["observations"]
    if np.allclose(obs_27, obs_51[:, :27]):
        print("\nSANITY CHECK PASSED: 27D dataset == first 27 dims of 51D dataset")
    else:
        print("\nSANITY CHECK FAILED — slicing mismatch!")
    
    # Distribution summary
    scenarios = {}
    styles = {}
    her_count = 0
    for d in demos:
        s = d["info"]["scenario"]
        st = d["info"].get("style_name", "?")
        scenarios[s] = scenarios.get(s, 0) + 1
        styles[st] = styles.get(st, 0) + 1
        if d["info"].get("her_relabeled", False):
            her_count += 1
    
    print(f"\nDataset composition:")
    print(f"  By scenario: {scenarios}")
    print(f"  By style:    {styles}")
    print(f"  HER relabeled: {her_count}")


if __name__ == "__main__":
    main()
