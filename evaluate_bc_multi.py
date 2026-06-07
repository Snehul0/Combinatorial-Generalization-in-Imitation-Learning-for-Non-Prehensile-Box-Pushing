"""
evaluate_bc_multi.py - Evaluate BC-27D-multi or BC-51D-multi on 35 OOD scenes.

Usage:
    python evaluate_bc_multi.py --obs_dim 27 --ckpt checkpoints/bc_27d_multi.pt
    python evaluate_bc_multi.py --obs_dim 51 --ckpt checkpoints/bc_51d_multi.pt
"""

import argparse
import json
import time
import numpy as np
import torch
import mujoco

from bc_model import BCPolicy
from Teleop_collector_51d_auto import (
    RobotScene, BOXES, setup_scene, get_obs_51d, get_disturbance, move_to_home,
)


def make_in_dist(seed, target_key="A1"):
    """In-distribution: just target box and goal, no blockers."""
    rng = np.random.default_rng(seed)
    target_pos = BOXES[target_key][1].copy()
    target_pos[0] += float(rng.uniform(-0.05, 0.05))
    target_pos[1] += float(rng.uniform(-0.03, 0.03))
    goal = target_pos.copy()
    goal[0] += float(rng.uniform(0.18, 0.25))
    return {target_key: target_pos}, goal, target_key


def make_ood(seed, n_blockers, target_key="A1"):
    """OOD: same straight-line blocker layout as your existing BC/ACT/Diffusion eval."""
    rng = np.random.default_rng(seed + 1000)
    target_pos = BOXES[target_key][1].copy()
    target_pos[0] += float(rng.uniform(-0.03, 0.03))
    goal = target_pos.copy()
    goal[0] += float(rng.uniform(0.18, 0.25))
    scene = {target_key: target_pos}
    avail = [k for k in BOXES.keys() if k != target_key]
    for i in range(n_blockers):
        bk = avail[i]
        bp = BOXES[bk][1].copy()
        bp[0] = target_pos[0] + (i + 1) * 0.08 + float(rng.uniform(-0.02, 0.02))
        bp[1] = target_pos[1] + float(rng.uniform(-0.05, 0.05))
        scene[bk] = bp
    return scene, goal, target_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs_dim", type=int, required=True, choices=[27, 51])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--xml", default="scene.xml")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    
    if args.out is None:
        args.out = f"bc_{args.obs_dim}d_multi_results.json"
    
    print(f"Loading: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = BCPolicy(obs_dim=cfg["obs_dim"], act_dim=cfg["act_dim"],
                     hidden_dim=cfg["hidden_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    om = ckpt["obs_mean"]
    os_ = ckpt["obs_std"]
    am = ckpt["act_mean"]
    as_ = ckpt["act_std"]
    
    print(f"BC obs_dim: {cfg['obs_dim']}, best val loss: {ckpt.get('val_loss', 'N/A')}")
    
    rs = RobotScene(args.xml)
    
    def predict_action(scene_boxes, target_key, goal):
        # Always compute 51D obs, then slice if needed
        obs_51d = get_obs_51d(rs, rs.data, target_key, goal, scene_boxes)
        if args.obs_dim == 27:
            obs = obs_51d[:27]
        else:
            obs = obs_51d
        norm = ((obs - om) / os_).astype(np.float32)
        with torch.no_grad():
            t = torch.from_numpy(norm).unsqueeze(0)
            a_norm = model(t).squeeze(0).numpy()
        return a_norm * as_ + am
    
    def single_rollout(scene_boxes, goal, target_key, sim_steps=80, max_outer=200):
        setup_scene(rs, scene_boxes, goal)
        move_to_home(rs, steps=200)
        
        box_start = rs.get_box_pos(rs.data, target_key).copy()
        best_dist = float(np.linalg.norm(box_start[:2] - goal[:2]))
        
        for step in range(max_outer):
            action = predict_action(scene_boxes, target_key, goal)
            for j, aid in enumerate(rs.act_ids):
                rs.data.ctrl[aid] = float(np.clip(action[j], -6.2, 6.2))
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
        }
    
    results = {}
    start = time.time()
    
    print("\n=== IN-DISTRIBUTION ===")
    results["in_dist"] = []
    for seed in range(10):
        for tk in ["A1", "A4"]:
            scene, goal, target = make_in_dist(seed, tk)
            r = single_rollout(scene, goal, target)
            r["seed"] = seed
            r["target"] = tk
            results["in_dist"].append(r)
            mark = "OK" if r["success"] else "  "
            elapsed = time.time() - start
            print(f"  seed={seed} tk={tk} [{mark}]: best={r['best_dist']:.3f} | "
                  f"disturb={r['disturbance']:.3f} [{elapsed:.0f}s]")
    
    for n in [2, 3, 4]:
        n_total = n + 1
        key = f"ood_n{n_total}"
        print(f"\n=== {key.upper()} ===")
        results[key] = []
        for seed in range(5):
            scene, goal, target = make_ood(seed, n)
            r = single_rollout(scene, goal, target)
            r["seed"] = seed
            results[key].append(r)
            mark = "OK" if r["success"] else "  "
            elapsed = time.time() - start
            print(f"  seed={seed} [{mark}]: best={r['best_dist']:.3f} | "
                  f"disturb={r['disturbance']:.3f} [{elapsed:.0f}s]")
    
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== SUMMARY (BC-{args.obs_dim}D-multi) ===")
    print(f"{'Scene':<14} {'Generous':<16} {'Strict':<16} {'Real':<16} {'Avg Disturb'}")
    for k in ["in_dist", "ood_n3", "ood_n4", "ood_n5"]:
        trials = results[k]
        n = len(trials)
        gen = sum(1 for r in trials if r["best_dist"] < 0.07)
        strict = sum(1 for r in trials if r["final_dist"] < 0.07)
        real = sum(1 for r in trials if r["best_dist"] < 0.07 and r["disturbance"] < 0.5)
        avg_d = np.mean([r["disturbance"] for r in trials])
        print(f"{k:<14} {gen}/{n} ({gen/n*100:.0f}%)        "
              f"{strict}/{n} ({strict/n*100:.0f}%)        "
              f"{real}/{n} ({real/n*100:.0f}%)        "
              f"{avg_d:.3f}m")
    
    print(f"\nTotal time: {(time.time()-start)/60:.1f} min")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
