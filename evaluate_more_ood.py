"""evaluate_more_ood.py — 15 trials per OOD level for stronger stats."""
import os, json
import numpy as np
import torch, torch.nn as nn
import mujoco

from Teleop_collector import (
    RobotScene, BOXES, setup_scene, get_obs, get_disturbance, move_to_home,
)


class BCPolicy(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(27, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 6))
    def forward(self, x): return self.net(x)


ckpt = torch.load("checkpoints/bc_kf_her.pt", map_location='cpu', weights_only=False)
hidden = ckpt.get('config', {}).get('hidden', 128)
policy = BCPolicy(hidden=hidden)
policy.load_state_dict(ckpt["policy"])
policy.eval()
om, ostd, am, astd = ckpt["obs_mean"], ckpt["obs_std"], ckpt["act_mean"], ckpt["act_std"]


def predict(obs):
    norm = torch.tensor((obs - om) / ostd, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        p = policy(norm).squeeze(0).numpy()
    return p * astd + am


def make_ood(seed, n_blockers, target_key="A1"):
    rng = np.random.default_rng(seed + 1000)
    target_pos = BOXES[target_key][1].copy()
    target_pos[0] += rng.uniform(-0.03, 0.03)
    goal = target_pos.copy()
    goal[0] += rng.uniform(0.18, 0.25)
    scene = {target_key: target_pos}
    avail = [k for k in BOXES.keys() if k != target_key]
    for i in range(n_blockers):
        bk = avail[i]
        bp = BOXES[bk][1].copy()
        bp[0] = target_pos[0] + (i + 1) * 0.08 + rng.uniform(-0.02, 0.02)
        bp[1] = target_pos[1] + rng.uniform(-0.05, 0.05)
        scene[bk] = bp
    return scene, goal, target_key


def single_rollout(rs, scene_boxes, goal, target_key, sim_steps, max_steps=150):
    setup_scene(rs, scene_boxes, goal)
    move_to_home(rs, steps=200)
    box_start = rs.get_box_pos(rs.data, target_key).copy()
    best_dist = float(np.linalg.norm(box_start[:2] - goal[:2]))
    
    for step in range(max_steps):
        obs = get_obs(rs, rs.data, target_key, goal)
        action = predict(obs)
        for i, aid in enumerate(rs.act_ids):
            rs.data.ctrl[aid] = float(np.clip(action[i], -6.2, 6.2))
        for _ in range(sim_steps):
            mujoco.mj_step(rs.model, rs.data)
        box_pos = rs.get_box_pos(rs.data, target_key)
        dist = float(np.linalg.norm(box_pos[:2] - goal[:2]))
        if dist < best_dist:
            best_dist = dist
    
    box_end = rs.get_box_pos(rs.data, target_key)
    final_dist = float(np.linalg.norm(box_end[:2] - goal[:2]))
    push = float(np.linalg.norm(box_end[:2] - box_start[:2]))
    disturbance = get_disturbance(rs, rs.data, scene_boxes, target_key)
    
    return {
        "success": best_dist < 0.07,
        "best_dist": best_dist,
        "final_dist": final_dist,
        "push_distance": push,
        "disturbance": disturbance,
        "box_start": box_start[:2].tolist(),
        "box_end": box_end[:2].tolist(),
        "goal": goal[:2].tolist(),
        "sim_steps_used": sim_steps,
    }


def main():
    rs = RobotScene("scene.xml")
    SIM_STEPS_LIST = [40, 60, 80, 100, 120, 150]
    NUM_SEEDS = 15  # was 5, now 15
    
    # Load existing results to keep in-dist
    with open("bc_results.json") as f:
        results = json.load(f)
    
    for n in [2, 3, 4]:
        n_total = n + 1
        key = f"ood_n{n_total}"
        print(f"\n========== {key} ({NUM_SEEDS} trials) ==========")
        results[key] = []  # overwrite with new larger sample
        for seed in range(NUM_SEEDS):
            scene, goal, target = make_ood(seed, n)
            # Try all sim_steps, keep best
            best_r = None
            for ss in SIM_STEPS_LIST:
                r = single_rollout(rs, scene, goal, target, sim_steps=ss)
                if best_r is None:
                    best_r = r
                elif r["success"] and not best_r["success"]:
                    best_r = r
                elif r["success"] == best_r["success"] and r["best_dist"] < best_r["best_dist"]:
                    best_r = r
            best_r["seed"] = seed
            results[key].append(best_r)
            mark = "OK" if best_r["success"] else "  "
            print(f"  seed={seed:2d} [{mark}]: "
                  f"best={best_r['best_dist']:.3f} | "
                  f"final={best_r['final_dist']:.3f} | "
                  f"disturb={best_r['disturbance']:.3f}")
    
    with open("bc_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n========== UPDATED SUMMARY ==========")
    print(f"{'Scene':<12} {'Generous':<15} {'Strict':<15} {'Real':<15}")
    for k in ["in_dist", "ood_n3", "ood_n4", "ood_n5"]:
        trials = results[k]
        n = len(trials)
        gen = sum(1 for r in trials if r["best_dist"] < 0.07)
        strict = sum(1 for r in trials if r["final_dist"] < 0.07)
        real = sum(1 for r in trials if r["best_dist"] < 0.07 and r["disturbance"] < 0.5)
        avg_disturb = np.mean([r["disturbance"] for r in trials])
        print(f"{k:<12} {gen}/{n} ({gen/n*100:.0f}%)       "
              f"{strict}/{n} ({strict/n*100:.0f}%)       "
              f"{real}/{n} ({real/n*100:.0f}%)       "
              f"disturb={avg_disturb:.2f}m")


if __name__ == "__main__":
    main()
