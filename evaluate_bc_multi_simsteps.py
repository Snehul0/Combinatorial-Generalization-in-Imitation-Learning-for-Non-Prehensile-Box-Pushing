"""
evaluate_bc_multi_simsteps.py — Evaluate BC across multiple sim_steps per scene.

For each scene, try sim_steps in {40, 60, 80, 100, 120, 150} and report the BEST result.
This gives BC the most generous evaluation.
"""
import os, json, time
import numpy as np
import torch
import torch.nn as nn
import mujoco

from Teleop_collector import (
    RobotScene, BOXES, HOME_QPOS,
    setup_scene, get_obs, get_disturbance, move_to_home,
)


class BCPolicy(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(27, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 6))
    def forward(self, x): return self.net(x)


CKPT = "checkpoints/bc_kf_her.pt"
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
hidden = ckpt.get('config', {}).get('hidden', 128)
policy = BCPolicy(hidden=hidden)
policy.load_state_dict(ckpt["policy"])
policy.eval()
om, ostd = ckpt["obs_mean"], ckpt["obs_std"]
am, astd = ckpt["act_mean"], ckpt["act_std"]


def predict(obs):
    norm = torch.tensor((obs - om) / ostd, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        p = policy(norm).squeeze(0).numpy()
    return p * astd + am


def make_in_dist(seed, target_key="A1"):
    rng = np.random.default_rng(seed)
    target_pos = BOXES[target_key][1].copy()
    target_pos[0] += rng.uniform(-0.05, 0.05)
    target_pos[1] += rng.uniform(-0.03, 0.03)
    goal = target_pos.copy()
    goal[0] += rng.uniform(0.18, 0.25)
    return {target_key: target_pos}, goal, target_key


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
    push_distance = float(np.linalg.norm(box_end[:2] - box_start[:2]))
    disturbance = get_disturbance(rs, rs.data, scene_boxes, target_key)
    
    return {
        "success": best_dist < 0.07,
        "best_dist": best_dist,
        "final_dist": final_dist,
        "push_distance": push_distance,
        "disturbance": disturbance,
        "box_start": box_start[:2].tolist(),
        "box_end": box_end[:2].tolist(),
        "goal": goal[:2].tolist(),
        "sim_steps_used": sim_steps,
    }


def evaluate_scene(rs, scene_boxes, goal, target_key, sim_steps_list):
    """Try multiple sim_steps for one scene, return best result."""
    best_result = None
    for ss in sim_steps_list:
        r = single_rollout(rs, scene_boxes, goal, target_key, ss)
        # Score: success > best_dist (lower better)
        if best_result is None:
            best_result = r
        elif r["success"] and not best_result["success"]:
            best_result = r
        elif r["success"] == best_result["success"] and r["best_dist"] < best_result["best_dist"]:
            best_result = r
    return best_result


def main():
    rs = RobotScene("scene.xml")
    
    # Try these sim_steps for each scene
    SIM_STEPS_LIST = [40, 60, 80, 100, 120, 150]
    
    print(f"BC checkpoint: {CKPT}")
    print(f"Trying sim_steps: {SIM_STEPS_LIST}")
    print(f"Each scene tested with {len(SIM_STEPS_LIST)} different settings, best result kept\n")
    
    results = {}
    start_time = time.time()
    
    # === IN-DISTRIBUTION ===
    print("=" * 60)
    print("IN-DISTRIBUTION")
    print("=" * 60)
    results["in_dist"] = []
    for seed in range(10):
        for tk in ["A1", "A4"]:
            scene, goal, target = make_in_dist(seed, tk)
            r = evaluate_scene(rs, scene, goal, target, SIM_STEPS_LIST)
            r["seed"] = seed
            r["target"] = tk
            results["in_dist"].append(r)
            mark = "OK" if r["success"] else "  "
            print(f"  seed={seed} tk={tk} [{mark}]: "
                  f"best_dist={r['best_dist']:.3f} | "
                  f"final={r['final_dist']:.3f} | "
                  f"push={r['push_distance']:.3f} | "
                  f"sim_steps={r['sim_steps_used']}")
    
    # === OOD ===
    for n in [2, 3, 4]:
        n_total = n + 1
        key = f"ood_n{n_total}"
        print(f"\n{'=' * 60}")
        print(f"{key.upper()} (target + {n} blockers)")
        print("=" * 60)
        results[key] = []
        for seed in range(5):
            scene, goal, target = make_ood(seed, n)
            r = evaluate_scene(rs, scene, goal, target, SIM_STEPS_LIST)
            r["seed"] = seed
            results[key].append(r)
            mark = "OK" if r["success"] else "  "
            print(f"  seed={seed} [{mark}]: "
                  f"best_dist={r['best_dist']:.3f} | "
                  f"final={r['final_dist']:.3f} | "
                  f"push={r['push_distance']:.3f} | "
                  f"disturb={r['disturbance']:.3f} | "
                  f"sim_steps={r['sim_steps_used']}")
    
    # === SAVE ===
    with open("bc_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # === SUMMARY ===
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    for k, trials in results.items():
        sr = sum(t["success"] for t in trials) / len(trials) * 100
        avg_final = np.mean([t["final_dist"] for t in trials])
        avg_best = np.mean([t["best_dist"] for t in trials])
        avg_push = np.mean([t["push_distance"] for t in trials])
        avg_disturb = np.mean([t["disturbance"] for t in trials])
        print(f"  {k:12s}: success={sr:.0f}% | "
              f"best_dist={avg_best:.3f} | "
              f"final_dist={avg_final:.3f} | "
              f"push={avg_push:.3f} | "
              f"disturb={avg_disturb:.3f}")
    
    print(f"\nTotal time: {elapsed:.0f}s")
    print(f"Results saved to bc_results.json")


if __name__ == "__main__":
    main()
