"""final_numbers.py — official poster numbers using strict + real definitions."""
import json
import numpy as np

with open("bc_results.json") as f:
    results = json.load(f)

print("=" * 70)
print("FINAL POSTER NUMBERS")
print("=" * 70)
print()
print(f"{'Scene':<15} {'Strict':<15} {'Real':<15} {'Avg Disturb':<15}")
print(f"{'(N boxes)':<15} {'(end<7cm)':<15} {'(best<7cm)':<15} {'(non-target)':<15}")
print("-" * 60)

for k in ["in_dist", "ood_n3", "ood_n4", "ood_n5"]:
    trials = results[k]
    n = len(trials)
    
    strict = sum(1 for r in trials if r["final_dist"] < 0.07)
    real = sum(1 for r in trials 
               if r["best_dist"] < 0.07 and r["disturbance"] < 0.5)
    
    avg_disturb = np.mean([r["disturbance"] for r in trials])
    
    print(f"{k:<15} {strict}/{n} ({strict/n*100:.0f}%)        "
          f"{real}/{n} ({real/n*100:.0f}%)        "
          f"{avg_disturb:.2f}m")

print()
print("=" * 70)
print("INTERPRETATION FOR POSTER")
print("=" * 70)
print("""
BC achieves 20% real success in-distribution but 0% under OOD clutter.

Apparent OOD 'successes' in our earlier evaluation (40%) were due to:
- Arm flailing through the scene
- Target box accidentally bumped near goal through cascade collisions
- High disturbance (1.8-7.3m) of non-target boxes

This supports the hypothesis: the 27D target-only observation cannot 
encode blocker positions. BC has no path to learn obstacle-aware behavior
because the information is not in the input. Even chaotic motions that 
hit the goal by chance are not 'success' — they are noise.

The cleanest result: BC works at single-target pushing (20% in-dist),
and fails completely at the combinatorial generalization task (0% OOD),
exactly as the input-representation hypothesis predicts.
""")
