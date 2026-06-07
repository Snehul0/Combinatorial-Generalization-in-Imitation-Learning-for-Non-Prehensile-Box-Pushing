"""
evaluate_diffusion_multi.py - Evaluate Diffusion Policy on dual OOD.

Usage:
    python evaluate_diffusion_multi.py --obs_dim 27 --ckpt checkpoints/diffusion_27d_multi_v1.pt
    (etc)
"""

import argparse
import json
import time
import numpy as np
import torch
import mujoco

from diffusion_model import DiffusionPolicy, NoiseScheduler
from Teleop_collector_51d_auto import (
    RobotScene, BOXES, setup_scene, get_obs_51d, get_disturbance, move_to_home,
    make_scenario_auto,
)


N_DDIM_STEPS = 30  # number of DDIM inference steps


def make_in_dist(seed, target_key="A1"):
    rng = np.random.default_rng(seed)
    target_pos = BOXES[target_key][1].copy()
    target_pos[0] += float(rng.uniform(-0.05, 0.05))
    target_pos[1] += float(rng.uniform(-0.03, 0.03))
    goal = target_pos.copy()
    goal[0] += float(rng.uniform(0.18, 0.25))
    return {target_key: target_pos}, goal, target_key


def make_ood_straight(seed, n_blockers, target_key="A1"):
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


def make_ood_trainstyle(seed, n_blockers, target_key="A1"):
    scenario = {2: "N3", 3: "N4", 4: "N5"}[n_blockers]
    eval_seed = 500 + seed
    scene, goal, target, _ = make_scenario_auto(scenario, eval_seed)
    return scene, goal, target


def ddim_sample(model, noise_scheduler, obs_tensor, chunk_size, act_dim, n_steps, device):
    """Run DDIM sampling. obs_tensor: (1, obs_dim). Returns (1, K, act_dim)."""
    # Initialize from random noise
    x = torch.randn(1, act_dim, chunk_size, device=device)
    
    # Build subsampled timestep sequence
    timesteps = torch.linspace(noise_scheduler.num_train_timesteps - 1, 0, n_steps + 1).long().to(device)
    
    for i in range(n_steps):
        t_curr = timesteps[i].item()
        t_next = timesteps[i + 1].item() if i < n_steps - 1 else -1
        t_batch = torch.tensor([t_curr], device=device, dtype=torch.long)
        with torch.no_grad():
            pred_noise = model(x, t_batch, obs_tensor)
            x = noise_scheduler.ddim_step(pred_noise, t_curr, t_next, x)
    
    # Final clip
    x = torch.clamp(x, -2.0, 2.0)
    # x is (1, act_dim, K) — transpose to (1, K, act_dim)
    return x.permute(0, 2, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs_dim", type=int, required=True, choices=[27, 51])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--xml", default="scene.xml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--exec_steps", type=int, default=3)
    args = parser.parse_args()
    
    if args.out is None:
        base = args.ckpt.replace(".pt", "").split("/")[-1]
        args.out = f"{base}_DUAL_results.json"
    
    device = "cpu"
    print(f"Loading: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    
    model = DiffusionPolicy(
        obs_dim=cfg["obs_dim"], act_dim=cfg["act_dim"],
        chunk_size=cfg["chunk_size"],
        time_emb_dim=cfg["time_emb_dim"], obs_emb_dim=cfg["obs_emb_dim"],
        down_dims=tuple(cfg["down_dims"]),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    noise_scheduler = NoiseScheduler(num_train_timesteps=cfg["num_train_timesteps"]).to(device)
    
    om = ckpt["obs_mean"]; os_ = ckpt["obs_std"]
    am = ckpt["act_mean"]; as_ = ckpt["act_std"]
    
    print(f"Diffusion obs_dim: {cfg['obs_dim']}, val_noise_loss: {ckpt.get('val_noise_loss', 'N/A')}")
    
    rs = RobotScene(args.xml)
    
    def predict_chunk(scene_boxes, target_key, goal):
        obs_51d = get_obs_51d(rs, rs.data, target_key, goal, scene_boxes)
        if args.obs_dim == 27:
            obs = obs_51d[:27]
        else:
            obs = obs_51d
        norm = ((obs - om) / os_).astype(np.float32)
        obs_t = torch.from_numpy(norm).unsqueeze(0).to(device)
        chunk = ddim_sample(model, noise_scheduler, obs_t,
                            cfg["chunk_size"], cfg["act_dim"], N_DDIM_STEPS, device)
        chunk_np = chunk.squeeze(0).cpu().numpy()  # (K, act_dim)
        return chunk_np * as_ + am
    
    def single_rollout(scene_boxes, goal, target_key, sim_steps=80, max_outer=80):
        setup_scene(rs, scene_boxes, goal)
        move_to_home(rs, steps=200)
        
        box_start = rs.get_box_pos(rs.data, target_key).copy()
        best_dist = float(np.linalg.norm(box_start[:2] - goal[:2]))
        
        for outer in range(max_outer):
            chunk = predict_chunk(scene_boxes, target_key, goal)
            for k in range(min(args.exec_steps, len(chunk))):
                action = chunk[k]
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
            "best_dist": best_dist, "final_dist": final_dist,
            "push_distance": push, "disturbance": disturbance,
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
            r["seed"] = seed; r["target"] = tk
            results["in_dist"].append(r)
            mark = "OK" if r["success"] else "  "
            print(f"  seed={seed} tk={tk} [{mark}]: best={r['best_dist']:.3f} | disturb={r['disturbance']:.3f} [{time.time()-start:.0f}s]")
    
    for n in [2, 3, 4]:
        n_total = n + 1
        key = f"ood_straight_n{n_total}"
        print(f"\n=== {key.upper()} ===")
        results[key] = []
        for seed in range(5):
            scene, goal, target = make_ood_straight(seed, n)
            r = single_rollout(scene, goal, target)
            r["seed"] = seed; r["layout"] = "straight"
            results[key].append(r)
            mark = "OK" if r["success"] else "  "
            print(f"  seed={seed} [{mark}]: best={r['best_dist']:.3f} | disturb={r['disturbance']:.3f} [{time.time()-start:.0f}s]")
    
    for n in [2, 3, 4]:
        n_total = n + 1
        key = f"ood_trainstyle_n{n_total}"
        print(f"\n=== {key.upper()} ===")
        results[key] = []
        for seed in range(5):
            scene, goal, target = make_ood_trainstyle(seed, n)
            r = single_rollout(scene, goal, target)
            r["seed"] = 500 + seed; r["layout"] = "trainstyle"
            results[key].append(r)
            mark = "OK" if r["success"] else "  "
            print(f"  seed={500+seed} [{mark}]: best={r['best_dist']:.3f} | disturb={r['disturbance']:.3f} [{time.time()-start:.0f}s]")
    
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"SUMMARY: Diffusion obs_dim={args.obs_dim}, ckpt={args.ckpt.split('/')[-1]}")
    print(f"{'='*70}")
    
    def summarize(trials, name):
        n = len(trials)
        if n == 0:
            return f"{name:<22} 0/0"
        gen = sum(1 for r in trials if r["best_dist"] < 0.07)
        strict = sum(1 for r in trials if r["final_dist"] < 0.07)
        real = sum(1 for r in trials if r["best_dist"] < 0.07 and r["disturbance"] < 0.5)
        avg_d = np.mean([r["disturbance"] for r in trials])
        avg_best = np.mean([r["best_dist"] for r in trials])
        return (f"{name:<22} Generous {gen}/{n} ({gen/n*100:3.0f}%)   "
                f"Real {real}/{n} ({real/n*100:3.0f}%)   "
                f"avg_best={avg_best:.3f}m   avg_disturb={avg_d:.3f}m")
    
    print("\n-- In-distribution --")
    print(summarize(results["in_dist"], "in_dist"))
    print("\n-- STRAIGHT-LINE OOD --")
    for n in [3, 4, 5]:
        print(summarize(results[f"ood_straight_n{n}"], f"ood_straight_n{n}"))
    print("\n-- TRAIN-STYLE OOD --")
    for n in [3, 4, 5]:
        print(summarize(results[f"ood_trainstyle_n{n}"], f"ood_trainstyle_n{n}"))
    
    print(f"\nTotal time: {(time.time()-start)/60:.1f} min")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
