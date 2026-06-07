"""make_plots.py — generate poster plots from bc_results.json."""
import json
import numpy as np
import matplotlib.pyplot as plt

with open("bc_results.json") as f:
    results = json.load(f)

test_sets = ["in_dist", "ood_n3", "ood_n4", "ood_n5"]
test_labels = ["In-Dist\n(N=1)", "OOD\nN=3", "OOD\nN=4", "OOD\nN=5"]
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

# Plot 1: Success rate
fig, ax = plt.subplots(figsize=(9, 6))
sr = [sum(t["success"] for t in results[k]) / len(results[k]) * 100 for k in test_sets]
bars = ax.bar(test_labels, sr, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel("Success Rate (%)", fontsize=13, fontweight='bold')
ax.set_title("BC Success Rate by Scene Type\n(Goal threshold: 7cm)",
             fontsize=14, fontweight='bold')
ax.set_ylim(0, 60)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, sr):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1.5,
            f'{val:.0f}%', ha='center', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig("plot1_success.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved plot1_success.png")

# Plot 2: Best dist distribution (closest approach)
fig, ax = plt.subplots(figsize=(9, 6))
data_per_set = [[t["best_dist"] for t in results[k]] for k in test_sets]
bp = ax.boxplot(data_per_set, labels=test_labels, patch_artist=True, widths=0.6)
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.axhline(y=0.07, color='red', linestyle='--', linewidth=2, label='Success threshold (7cm)')
ax.set_ylabel("Closest Approach to Goal (m)", fontsize=13, fontweight='bold')
ax.set_title("Closest Distance Achieved by BC", fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("plot2_best_dist.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved plot2_best_dist.png")

# Plot 3: Disturbance growth with N
fig, ax = plt.subplots(figsize=(9, 6))
avg_disturb = []
for k in test_sets:
    d = [t["disturbance"] for t in results[k]]
    avg_disturb.append(np.mean(d))
bars = ax.bar(test_labels, avg_disturb, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel("Average Disturbance (m)", fontsize=13, fontweight='bold')
ax.set_title("Disturbance to Non-Target Boxes Grows with Clutter\n"
             "(BC cannot see blockers in 27D obs)",
             fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, avg_disturb):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
            f'{val:.2f}m', ha='center', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig("plot3_disturbance.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved plot3_disturbance.png")

# Plot 4: Push distance distribution
fig, ax = plt.subplots(figsize=(9, 6))
data_per_set = [[t["push_distance"] for t in results[k]] for k in test_sets]
bp = ax.boxplot(data_per_set, labels=test_labels, patch_artist=True, widths=0.6)
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.set_ylabel("Total Push Distance (m)", fontsize=13, fontweight='bold')
ax.set_title("BC Pushes Box But Overshoots Goal\n"
             "(target push: ~0.22m, observed: 0.62-1.57m)",
             fontsize=14, fontweight='bold')
ax.axhline(y=0.22, color='green', linestyle='--', linewidth=2, label='Target push distance')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("plot4_push_dist.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved plot4_push_dist.png")

# Summary numbers
print("\n=== POSTER NUMBERS ===")
for k, label in zip(test_sets, test_labels):
    trials = results[k]
    sr = sum(t["success"] for t in trials) / len(trials) * 100
    best = np.mean([t["best_dist"] for t in trials])
    push = np.mean([t["push_distance"] for t in trials])
    disturb = np.mean([t["disturbance"] for t in trials])
    print(f"{label.replace(chr(10), ' '):15s}: "
          f"success={sr:.0f}% | "
          f"avg_closest={best*100:.1f}cm | "
          f"avg_push={push*100:.0f}cm | "
          f"disturb={disturb*100:.0f}cm")

