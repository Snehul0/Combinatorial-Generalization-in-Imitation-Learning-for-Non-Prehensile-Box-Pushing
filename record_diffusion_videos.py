"""record_diffusion_videos.py — Record videos of Diffusion rollouts."""
import os, json
import numpy as np
import torch
import mujoco
import imageio

from diffusion_model import DiffusionPolicy, NoiseScheduler, sample_action_chunk
from Teleop_collector import (
    RobotScene, BOXES, setup_scene, get_obs, get_disturbance, move_to_home,
)


VIDEO_DIR = "videos_diffusion"
os.makedirs(VIDEO_DIR, exist_ok=True)

WIDTH, HEIGHT = 640, 480
FPS = 30
INF_STEPS = 30

CKPT = "checkpoints/diffusion_kf_her.pt"
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
cfg = ckpt["config"]
cfg['down_dims'] = tuple(cfg['down_dims'])

device = 'cpu'
model = DiffusionPolicy(
    obs_dim=cfg['obs_dim'], act_dim=cfg['act_dim'],
    chunk_size=cfg['chunk_size'],
    time_emb_dim=cfg['time_emb_dim'], obs_emb_dim=cfg['obs_emb_dim'],
    down_dims=cfg['down_dims'],
).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
scheduler = NoiseScheduler(num_train_timesteps=cfg['num_train_timesteps']).to(device)

om, ostd = ckpt["obs_mean"], ckpt["obs_std"]
am, astd = ckpt["act_mean"], ckpt["act_std"]
CHUNK = cfg["chunk_size"]


def predict_chunk(obs):
    norm = torch.tensor((obs - om) / ostd, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        chunk_norm = sample_action_chunk(model, scheduler, norm,
                                          num_inference_steps=INF_STEPS,
                                          device=device).squeeze(0).cpu().numpy()
    return chunk_norm * astd + am


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


def record(rs, renderer, scene_boxes, goal, target_key, sim_steps, exec_steps,
           video_path, max_outer=50):
    setup_scene(rs, scene_boxes, goal)
    move_to_home(rs, steps=200)
    box_start = rs.get_box_pos(rs.data, target_key).copy()
    best_dist = float(np.linalg.norm(box_start[:2] - goal[:2]))
    
    frames = []
    
    for outer in range(max_outer):
        obs = get_obs(rs, rs.data, target_key, goal)
        chunk = predict_chunk(obs)
        
        for i in range(min(exec_steps, CHUNK)):
            action = chunk[i]
            for j, aid in enumerate(rs.act_ids):
                rs.data.ctrl[aid] = float(np.clip(action[j], -6.2, 6.2))
            for _ in range(sim_steps):
                mujoco.mj_step(rs.model, rs.data)
            
            renderer.update_scene(rs.data)
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
    
    imageio.mimsave(video_path, frames, fps=FPS, codec='libx264', quality=6)
    
    return {
        "success": best_dist < 0.07,
        "best_dist": best_dist,
        "final_dist": final_dist,
        "push_distance": push,
        "disturbance": disturbance,
    }


def main():
    rs = RobotScene("scene.xml")
    renderer = mujoco.Renderer(rs.model, height=HEIGHT, width=WIDTH)
    
    with open("diffusion_results.json") as f:
        prev = json.load(f)
    
    print("=" * 60)
    print("Recording Diffusion in-distribution videos")
    print("=" * 60)
    for r in prev["in_dist"]:
        seed = r["seed"]
        tk = r["target"]
        ss = r["sim_steps_used"]
        es = r["exec_steps_used"]
        scene, goal, target = make_in_dist(seed, tk)
        video_path = f"{VIDEO_DIR}/in_dist_seed{seed}_{tk}.mp4"
        result = record(rs, renderer, scene, goal, target, ss, es, video_path)
        mark = "OK" if result["success"] else "  "
        print(f"  [{mark}] seed={seed} tk={tk} (ss={ss}, es={es}): "
              f"best={result['best_dist']:.3f} -> {video_path}")
    
    for n_blk in [2, 3, 4]:
        n_total = n_blk + 1
        key = f"ood_n{n_total}"
        print(f"\n{'=' * 60}")
        print(f"Recording {key.upper()} videos")
        print("=" * 60)
        for r in prev[key]:
            seed = r["seed"]
            ss = r["sim_steps_used"]
            es = r["exec_steps_used"]
            scene, goal, target = make_ood(seed, n_blk)
            video_path = f"{VIDEO_DIR}/{key}_seed{seed}.mp4"
            result = record(rs, renderer, scene, goal, target, ss, es, video_path)
            mark = "OK" if result["success"] else "  "
            print(f"  [{mark}] seed={seed} (ss={ss}, es={es}): "
                  f"best={result['best_dist']:.3f} disturb={result['disturbance']:.3f}")
    
    print(f"\nAll videos saved to {VIDEO_DIR}/")


if __name__ == "__main__":
    main()
