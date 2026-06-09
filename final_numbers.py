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

