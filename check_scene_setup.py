"""check_scene_setup.py — Verify starting distances are well above the 7cm threshold."""
import json
import numpy as np

with open("bc_results.json") as f:
    results = json.load(f)

print("Starting distance from box to goal in each scenario:\n")

for k in ["in_dist", "ood_n3", "ood_n4", "ood_n5"]:
    starting_dists = []
    for r in results[k]:
        start = np.array(r["box_start"])
        goal = np.array(r["goal"])
        d = np.linalg.norm(start - goal)
        starting_dists.append(d)
    
    print(f"{k:12s}: mean_start_dist={np.mean(starting_dists):.3f}m, "
          f"min={np.min(starting_dists):.3f}m, max={np.max(starting_dists):.3f}m")

print("\nFor reference: success threshold = 0.07m (7cm)")
print()

# Now show the SUCCESSES with detailed metrics
print("\nAll trials marked as success:\n")
for k in ["in_dist", "ood_n3", "ood_n4", "ood_n5"]:
    for r in results[k]:
        if r["success"]:
            start = np.array(r["box_start"])
            end = np.array(r["box_end"])
            goal = np.array(r["goal"])
            start_dist = np.linalg.norm(start - goal)
            end_dist = np.linalg.norm(end - goal)
            box_moved = np.linalg.norm(end - start)
            print(f"  {k} seed={r['seed']}: "
                  f"start_dist={start_dist:.3f}, "
                  f"end_dist={end_dist:.3f}, "
                  f"best_dist={r['best_dist']:.3f}, "
                  f"box_moved={box_moved:.3f}, "
                  f"disturb={r['disturbance']:.3f}")
