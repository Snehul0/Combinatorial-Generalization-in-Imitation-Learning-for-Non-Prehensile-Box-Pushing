"""
record_bc_videos.py — Run BC evaluations and record videos using MuJoCo offscreen renderer.

Saves videos to videos/ directory:
  in_dist_seedX_tkY.mp4
  ood_nN_seedX.mp4
"""
import os, json
import numpy as np
import torch
import torch.nn as nn
import mujoco
import imageio

from Teleop_collector import (
    RobotScene, BOXES, HOME_QPOS,
    setup_scene, get_obs, get_disturbance, move_to_home,
)


VIDEO_DIR = "videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

WIDTH, HEIGHT = 640, 480
FPS = 30


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


def record_rollout(rs, renderer, scene_boxes, goal, target_key,
                   sim_steps, max_steps, video_path, label):
    setup_scene(rs, scene_boxes, goal)
    move_to_home(rs, steps=200)
    
    box_start = rs.get_box_pos(rs.data, target_key).copy()
    best_dist = float(np.linalg.norm(box_start[:2] - goal[:2]))
    
    frames = []
    
    for step in range(max_steps):
        obs = get_obs(rs, rs.data, target_key, goal)
        action = predict(obs)
        for i, aid in enumerate(rs.act_ids):
            rs.data.ctrl[aid] = float(np.clip(action[i], -6.2, 6.2))
        for _ in range(sim_steps):
            mujoco.mj_step(rs.model, rs.data)
        
        # Render frame
        renderer.update_scene(rs.data, camera="fixed" if "fixed" in [
            mujoco.mj_id2name(rs.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(rs.model.ncam)
        ] else -1)
        frame = renderer.render()
        frames.append(frame)
        
        box_pos = rs.get_box_pos(rs.data, target_key)
        dist = float(np.linalg.norm(box_pos[:2] - goal[:2]))
        if dist < best_dist:
            best_dist = dist
    
    box_end = rs.get_box_pos(rs.data, target_key)
    final_dist = float(np.linalg.norm(box_end[:2] - goal[:2]))
    push = float(np.linalg.norm(box_end[:2] - box_start[:2]))
    disturbance = get_disturbance(rs, rs.data, scene_boxes, target_key)
    success = best_dist < 0.07
    
    # Write video
    print(f"  Writing {video_path} ({len(frames)} frames)...")
    imageio.mimsave(video_path, frames, fps=FPS, codec='libx264', quality=6)
    
    return {
        "success": success,
        "best_dist": best_dist,
        "final_dist": final_dist,
        "push_distance": push,
        "disturbance": disturbance,
        "label": label,
        "video": video_path,
    }


def main():
    rs = RobotScene("scene.xml")
    
    # Create offscreen renderer
    renderer = mujoco.Renderer(rs.model, height=HEIGHT, width=WIDTH)
    
    # Load earlier results to use the best sim_steps per scene
    with open("bc_results.json") as f:
        prev = json.load(f)
    
    all_results = {}
    
    # ===== IN-DIST =====
    print("=" * 60)
    print("Recording IN-DISTRIBUTION videos")
    print("=" * 60)
    all_results["in_dist"] = []
    for r in prev["in_dist"]:
        seed = r["seed"]
        tk = r["target"]
        ss = r["sim_steps_used"]
        scene, goal, target = make_in_dist(seed, tk)
        video_path = f"{VIDEO_DIR}/in_dist_seed{seed}_{tk}.mp4"
        label = f"In-Dist seed={seed} target={tk} sim_steps={ss}"
        result = record_rollout(rs, renderer, scene, goal, target,
                                sim_steps=ss, max_steps=150,
                                video_path=video_path, label=label)
        all_results["in_dist"].append(result)
        mark = "OK" if result["success"] else "  "
        print(f"  [{mark}] {label}: best_dist={result['best_dist']:.3f} -> {video_path}")
    
    # ===== OOD =====
    for n_blk in [2, 3, 4]:
        n_total = n_blk + 1
        key = f"ood_n{n_total}"
        print(f"\n{'=' * 60}")
        print(f"Recording {key.upper()} videos")
        print("=" * 60)
        all_results[key] = []
        for r in prev[key]:
            seed = r["seed"]
            ss = r["sim_steps_used"]
            scene, goal, target = make_ood(seed, n_blk)
            video_path = f"{VIDEO_DIR}/{key}_seed{seed}.mp4"
            label = f"{key} seed={seed} sim_steps={ss}"
            result = record_rollout(rs, renderer, scene, goal, target,
                                    sim_steps=ss, max_steps=150,
                                    video_path=video_path, label=label)
            all_results[key].append(result)
            mark = "OK" if result["success"] else "  "
            print(f"  [{mark}] {label}: best_dist={result['best_dist']:.3f} "
                  f"disturb={result['disturbance']:.3f} -> {video_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, trials in all_results.items():
        sr = sum(t["success"] for t in trials) / len(trials) * 100
        avg_best = np.mean([t["best_dist"] for t in trials])
        print(f"  {k:12s}: success={sr:.0f}% | best_dist={avg_best:.3f} | "
              f"{len(trials)} videos saved")
    
    print(f"\nAll videos saved in {VIDEO_DIR}/")


if __name__ == "__main__":
    main()
