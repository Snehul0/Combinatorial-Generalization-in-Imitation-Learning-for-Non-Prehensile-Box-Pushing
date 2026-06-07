"""test_diffusion.py — Test trained Diffusion policy in MuJoCo with viewer."""

import argparse, time
import numpy as np
import torch
import mujoco, mujoco.viewer

from diffusion_model import DiffusionPolicy, NoiseScheduler, sample_action_chunk
from Teleop_collector import (
    RobotScene, BOXES, setup_scene, get_obs, get_disturbance, move_to_home,
)


def load_diffusion(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ckpt["config"]
    cfg['down_dims'] = tuple(cfg['down_dims'])
    model = DiffusionPolicy(
        obs_dim=cfg['obs_dim'], act_dim=cfg['act_dim'],
        chunk_size=cfg['chunk_size'],
        time_emb_dim=cfg['time_emb_dim'], obs_emb_dim=cfg['obs_emb_dim'],
        down_dims=cfg['down_dims'],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    scheduler = NoiseScheduler(num_train_timesteps=cfg['num_train_timesteps'])
    return (model, scheduler, ckpt["obs_mean"], ckpt["obs_std"],
            ckpt["act_mean"], ckpt["act_std"], cfg["chunk_size"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/diffusion_kf_her.pt")
    ap.add_argument("--xml", default="scene.xml")
    ap.add_argument("--target", default="A1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--sim-steps", type=int, default=80)
    ap.add_argument("--exec-steps", type=int, default=4)
    ap.add_argument("--inf-steps", type=int, default=50,
                    help="Denoising steps at inference")
    args = ap.parse_args()
    
    rs = RobotScene(args.xml)
    
    device = 'cpu'  # CPU is fine for inference on this small model
    model, scheduler, om, ostd, am, astd, chunk_size = load_diffusion(args.ckpt)
    model = model.to(device)
    scheduler = scheduler.to(device)
    print(f"Loaded Diffusion: chunk_size={chunk_size}, inf_steps={args.inf_steps}")
    
    rng = np.random.default_rng(args.seed)
    target_pos = BOXES[args.target][1].copy()
    target_pos[0] += rng.uniform(-0.05, 0.05)
    target_pos[1] += rng.uniform(-0.03, 0.03)
    goal = target_pos.copy()
    goal[0] += rng.uniform(0.18, 0.25)
    scene_boxes = {args.target: target_pos}
    
    print(f"\nScene: target {args.target} at {target_pos[:2].round(3)}, "
          f"goal at {goal[:2].round(3)}")
    print(f"Sim steps per action: {args.sim_steps}")
    print(f"Actions executed per chunk: {args.exec_steps}/{chunk_size}\n")
    
    setup_scene(rs, scene_boxes, goal)
    move_to_home(rs, steps=200)
    
    with mujoco.viewer.launch_passive(rs.model, rs.data) as viewer:
        step = 0
        max_outer = args.max_steps // args.exec_steps
        for outer in range(max_outer):
            if not viewer.is_running():
                break
            
            obs = get_obs(rs, rs.data, args.target, goal)
            obs_norm = (obs - om) / ostd
            obs_tensor = torch.tensor(obs_norm, dtype=torch.float32,
                                       device=device).unsqueeze(0)
            
            chunk_norm = sample_action_chunk(model, scheduler, obs_tensor,
                                              num_inference_steps=args.inf_steps,
                                              device=device)
            chunk = chunk_norm.squeeze(0).cpu().numpy() * astd + am
            
            for i in range(min(args.exec_steps, chunk_size)):
                action = chunk[i]
                for j, aid in enumerate(rs.act_ids):
                    rs.data.ctrl[aid] = float(np.clip(action[j], -6.2, 6.2))
                for _ in range(args.sim_steps):
                    mujoco.mj_step(rs.model, rs.data)
                viewer.sync()
                
                box_pos = rs.get_box_pos(rs.data, args.target)
                dist = float(np.linalg.norm(box_pos[:2] - goal[:2]))
                
                if step % 10 == 0:
                    tip = rs.get_tip(rs.data)
                    print(f"  Step {step:4d}: box {box_pos[:2].round(3)}, "
                          f"tip {tip[:2].round(3)}, dist {dist:.3f}m, "
                          f"action[0:3]={action[:3].round(2)}")
                step += 1
                
                if dist < 0.07:
                    print(f"\n*** SUCCESS at step {step}! ***")
                    box_pos = rs.get_box_pos(rs.data, args.target)
                    final_dist = float(np.linalg.norm(box_pos[:2] - goal[:2]))
                    disturbance = get_disturbance(rs, rs.data, scene_boxes, args.target)
                    print(f"Final: dist={final_dist:.3f}m, "
                          f"disturbance={disturbance:.3f}m")
                    while viewer.is_running():
                        time.sleep(0.1)
                    return
        
        box_pos = rs.get_box_pos(rs.data, args.target)
        final_dist = float(np.linalg.norm(box_pos[:2] - goal[:2]))
        disturbance = get_disturbance(rs, rs.data, scene_boxes, args.target)
        print(f"\nFinal: success=False, dist={final_dist:.3f}m, "
              f"disturbance={disturbance:.3f}m")
        print(f"Box: {box_pos[:2].round(3)}, Goal: {goal[:2].round(3)}")
        
        while viewer.is_running():
            time.sleep(0.1)


if __name__ == "__main__":
    main()
