"""
record_multi_videos.py — Universal video recorder for BC/ACT/Diffusion checkpoints
trained on multi-box demos (v1 or v2).

Usage:
    python record_multi_videos.py --method BC --obs_dim 51 --ckpt checkpoints/bc_51d_multi_v1.pt --label bc_51d_v1
    python record_multi_videos.py --method ACT --obs_dim 27 --ckpt checkpoints/act_27d_multi_v2.pt --label act_27d_v2
    python record_multi_videos.py --method Diffusion --obs_dim 27 --ckpt checkpoints/diffusion_27d_multi_v2.pt --label diff_27d_v2

Records 35 videos per model: 20 in-dist + 5 each for OOD straight N=3,4,5 + 5 each for OOD train-style N=3,4,5
Total: 20 + 15 + 15 = 50 scenes (will use 35 default).
"""

import argparse
import os
import numpy as np
import torch
import mujoco
import imageio

from bc_model import BCPolicy
from act_model import ACT
from diffusion_model import DiffusionPolicy, NoiseScheduler

from Teleop_collector_51d_auto import (
    RobotScene, BOXES, setup_scene, get_obs_51d, get_disturbance, move_to_home,
    make_scenario_auto,
)


WIDTH, HEIGHT = 640, 480
FPS = 30
N_DDIM_STEPS = 30


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
    """Run DDIM sampling for Diffusion."""
    x = torch.randn(1, act_dim, chunk_size, device=device)
    timesteps = torch.linspace(noise_scheduler.num_train_timesteps - 1, 0, n_steps + 1).long().to(device)
    for i in range(n_steps):
        t_curr = timesteps[i].item()
        t_next = timesteps[i + 1].item() if i < n_steps - 1 else -1
        t_batch = torch.tensor([t_curr], device=device, dtype=torch.long)
        with torch.no_grad():
            pred_noise = model(x, t_batch, obs_tensor)
            x = noise_scheduler.ddim_step(pred_noise, t_curr, t_next, x)
    x = torch.clamp(x, -2.0, 2.0)
    return x.permute(0, 2, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["BC", "ACT", "Diffusion"])
    parser.add_argument("--obs_dim", type=int, required=True, choices=[27, 51])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--label", required=True, help="Used for video filename prefix")
    parser.add_argument("--xml", default="scene.xml")
    parser.add_argument("--video_dir", default=None, help="Output dir (default: videos_<label>)")
    parser.add_argument("--exec_steps", type=int, default=3, help="Chunked methods: actions per inference")
    parser.add_argument("--max_outer", type=int, default=80, help="Max outer rollout steps for chunked methods")
    args = parser.parse_args()
    
    if args.video_dir is None:
        args.video_dir = f"videos_{args.label}"
    os.makedirs(args.video_dir, exist_ok=True)
    
    print(f"Method: {args.method}, obs_dim: {args.obs_dim}")
    print(f"Loading: {args.ckpt}")
    print(f"Output dir: {args.video_dir}")
    
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    
    if args.method == "BC":
        model = BCPolicy(obs_dim=cfg["obs_dim"], act_dim=cfg["act_dim"], hidden_dim=cfg["hidden_dim"])
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        noise_scheduler = None
    elif args.method == "ACT":
        model = ACT(
            obs_dim=cfg["obs_dim"], act_dim=cfg["act_dim"], chunk_size=cfg["chunk_size"],
            hidden=cfg["hidden"], latent_dim=cfg["latent_dim"],
            n_enc_layers=cfg["n_enc_layers"], n_dec_layers=cfg["n_dec_layers"],
            n_heads=cfg["n_heads"],
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        noise_scheduler = None
    else:  # Diffusion
        model = DiffusionPolicy(
            obs_dim=cfg["obs_dim"], act_dim=cfg["act_dim"], chunk_size=cfg["chunk_size"],
            time_emb_dim=cfg["time_emb_dim"], obs_emb_dim=cfg["obs_emb_dim"],
            down_dims=tuple(cfg["down_dims"]),
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        noise_scheduler = NoiseScheduler(num_train_timesteps=cfg["num_train_timesteps"])
    
    om = ckpt["obs_mean"]; os_ = ckpt["obs_std"]
    am = ckpt["act_mean"]; as_ = ckpt["act_std"]
    
    rs = RobotScene(args.xml)
    renderer = mujoco.Renderer(rs.model, height=HEIGHT, width=WIDTH)
    
    def predict_action_or_chunk(scene_boxes, target_key, goal):
        """Returns either a single action (BC) or a chunk of actions (ACT/Diffusion)."""
        obs_51d = get_obs_51d(rs, rs.data, target_key, goal, scene_boxes)
        if args.obs_dim == 27:
            obs = obs_51d[:27]
        else:
            obs = obs_51d
        norm = ((obs - om) / os_).astype(np.float32)
        
        if args.method == "BC":
            with torch.no_grad():
                t = torch.from_numpy(norm).unsqueeze(0)
                a_norm = model(t).squeeze(0).numpy()
            return a_norm * as_ + am  # single action (6,)
        elif args.method == "ACT":
            with torch.no_grad():
                t = torch.from_numpy(norm).unsqueeze(0)
                chunk = model.inference_forward(t).squeeze(0).numpy()
            return chunk * as_ + am  # chunk (K, 6)
        else:  # Diffusion
            obs_t = torch.from_numpy(norm).unsqueeze(0)
            chunk = ddim_sample(model, noise_scheduler, obs_t,
                                cfg["chunk_size"], cfg["act_dim"], N_DDIM_STEPS, "cpu")
            chunk_np = chunk.squeeze(0).numpy()
            return chunk_np * as_ + am  # chunk (K, 6)
    
    def record_rollout(scene_boxes, goal, target_key, video_path, label, sim_steps=80):
        setup_scene(rs, scene_boxes, goal)
        move_to_home(rs, steps=200)
        
        frames = []
        box_start = rs.get_box_pos(rs.data, target_key).copy()
        best_dist = float(np.linalg.norm(box_start[:2] - goal[:2]))
        
        is_chunked = args.method in ["ACT", "Diffusion"]
        max_outer = args.max_outer if is_chunked else 200
        
        for outer in range(max_outer):
            result = predict_action_or_chunk(scene_boxes, target_key, goal)
            
            if is_chunked:
                # Result is (K, 6), execute first `exec_steps`
                for k in range(min(args.exec_steps, len(result))):
                    action = result[k]
                    for j, aid in enumerate(rs.act_ids):
                        rs.data.ctrl[aid] = float(np.clip(action[j], -6.2, 6.2))
                    for _ in range(sim_steps):
                        mujoco.mj_step(rs.model, rs.data)
                    box_pos = rs.get_box_pos(rs.data, target_key)
                    dist = float(np.linalg.norm(box_pos[:2] - goal[:2]))
                    if dist < best_dist:
                        best_dist = dist
                    # Render frame
                    renderer.update_scene(rs.data, camera=-1)
                    frames.append(renderer.render())
            else:
                # BC: single action
                action = result
                for j, aid in enumerate(rs.act_ids):
                    rs.data.ctrl[aid] = float(np.clip(action[j], -6.2, 6.2))
                for _ in range(sim_steps):
                    mujoco.mj_step(rs.model, rs.data)
                box_pos = rs.get_box_pos(rs.data, target_key)
                dist = float(np.linalg.norm(box_pos[:2] - goal[:2]))
                if dist < best_dist:
                    best_dist = dist
                # Render frame
                renderer.update_scene(rs.data, camera=-1)
                frames.append(renderer.render())
        
        box_end = rs.get_box_pos(rs.data, target_key)
        final_dist = float(np.linalg.norm(box_end[:2] - goal[:2]))
        disturbance = get_disturbance(rs, rs.data, scene_boxes, target_key)
        
        # Save video
        with imageio.get_writer(video_path, fps=FPS, codec='libx264', quality=8) as writer:
            for frame in frames:
                writer.append_data(frame)
        
        success = best_dist < 0.07 and disturbance < 0.5
        return success, best_dist, disturbance, len(frames)
    
    import time
    t0 = time.time()
    
    print(f"\n=== Recording {args.label} ===")
    
    # In-distribution: 10 seeds × 2 targets = 20 videos
    print("\n-- In-distribution --")
    for seed in range(10):
        for tk in ["A1", "A4"]:
            scene, goal, target = make_in_dist(seed, tk)
            path = f"{args.video_dir}/in_dist_seed{seed}_{tk}.mp4"
            success, best, disturb, n_frames = record_rollout(scene, goal, target, path, f"in_dist_{seed}_{tk}")
            mark = "OK" if success else "  "
            elapsed = time.time() - t0
            print(f"  seed={seed} tk={tk} [{mark}]: best={best:.3f} disturb={disturb:.3f} frames={n_frames} [{elapsed:.0f}s]")
    
    # OOD straight-line: 5 seeds × 3 N levels = 15 videos
    print("\n-- OOD straight-line --")
    for n in [2, 3, 4]:
        n_total = n + 1
        for seed in range(5):
            scene, goal, target = make_ood_straight(seed, n)
            path = f"{args.video_dir}/ood_straight_n{n_total}_seed{seed}.mp4"
            success, best, disturb, n_frames = record_rollout(scene, goal, target, path, f"ood_str_n{n_total}_{seed}")
            mark = "OK" if success else "  "
            elapsed = time.time() - t0
            print(f"  N={n_total} seed={seed} [{mark}]: best={best:.3f} disturb={disturb:.3f} frames={n_frames} [{elapsed:.0f}s]")
    
    # OOD train-style: 5 seeds × 3 N levels = 15 videos
    print("\n-- OOD train-style --")
    for n in [2, 3, 4]:
        n_total = n + 1
        for seed in range(5):
            scene, goal, target = make_ood_trainstyle(seed, n)
            path = f"{args.video_dir}/ood_trainstyle_n{n_total}_seed{seed}.mp4"
            success, best, disturb, n_frames = record_rollout(scene, goal, target, path, f"ood_ts_n{n_total}_{seed}")
            mark = "OK" if success else "  "
            elapsed = time.time() - t0
            print(f"  N={n_total} seed={500+seed} [{mark}]: best={best:.3f} disturb={disturb:.3f} frames={n_frames} [{elapsed:.0f}s]")
    
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Total: 50 videos in {elapsed/60:.1f} min")
    print(f"Saved to: {args.video_dir}/")


if __name__ == "__main__":
    main()
