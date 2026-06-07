"""test_diffusion_slow.py — Test Diffusion policy with slowed visualization."""

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
    ap.add_argument("--inf-steps", type=int, default=50)
    
    # NEW: speed controls
    ap.add_argument("--sync-every", type=int, default=4,
                    help="Sync viewer every N physics steps (lower = slower visual)")
    ap.add_argument("--sleep-per-sync", type=float, default=0.02,
                    help="Real-time sleep between syncs in seconds")
    ap.add_argument("--pause-between-chunks", type=float, default=0.5,
                    help="Pause between chunk re-planning to make it visible")
    args = ap.parse_args()
    
    rs = RobotScene(args.xml)
    
    device = 'cpu'
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
    print(f"Slow mode: sync every {args.sync_every} physics steps, "
          f"sleep {args.sleep_per_sync*1000:.0f}ms per sync\n")
    
    setup_scene(rs, scene_boxes, goal)
    print("Moving to home pose...")
    move_to_home(rs, steps=200)
    
    print("Starting rollout. Take your time to watch.\n")
    
    with mujoco.viewer.launch_passive(rs.model, rs.data) as viewer:
        step = 0
        max_outer = args.max_steps // args.exec_steps
        
        for outer in range(max_outer):
            if not viewer.is_running():
                break
            
            # Predict chunk
            obs = get_obs(rs, rs.data, args.target, goal)
            obs_norm = (obs - om) / ostd
            obs_tensor = torch.tensor(obs_norm, dtype=torch.float32,
                                       device=device).unsqueeze(0)
            
            chunk_norm = sample_action_chunk(model, scheduler, obs_tensor,
                                              num_inference_steps=args.inf_steps,
                                              device=device)
            chunk = chunk_norm.squeeze(0).cpu().numpy() * astd + am
            
            print(f"\n[Chunk {outer}] Predicted chunk (first action): "
                  f"{chunk[0].round(2)}")
            
            # Brief pause so you can see chunks happening
            if args.pause_between_chunks > 0:
                t_pause = time.time()
                while time.time() - t_pause < args.pause_between_chunks:
                    viewer.sync()
                    time.sleep(0.01)
            
            # Execute first EXEC_STEPS actions
            for i in range(min(args.exec_steps, chunk_size)):
                action = chunk[i]
                for j, aid in enumerate(rs.act_ids):
                    rs.data.ctrl[aid] = float(np.clip(action[j], -6.2, 6.2))
                
                # Run physics in small batches, syncing visualizer between batches
                for phys in range(args.sim_steps):
                    mujoco.mj_step(rs.model, rs.data)
                    
                    # Sync and sleep periodically for smooth visualization
                    if (phys + 1) % args.sync_every == 0:
                        viewer.sync()
                        time.sleep(args.sleep_per_sync)
                
                # Always sync at end of action
                viewer.sync()
                
                box_pos = rs.get_box_pos(rs.data, args.target)
                dist = float(np.linalg.norm(box_pos[:2] - goal[:2]))
                
                tip = rs.get_tip(rs.data)
                print(f"  Step {step:3d} (act {i+1}/{args.exec_steps}): "
                      f"box {box_pos[:2].round(3)}, "
                      f"tip {tip[:2].round(3)}, "
                      f"dist {dist:.3f}m")
                step += 1
                
                if dist < 0.07:
                    print(f"\n*** SUCCESS at step {step}! ***")
                    box_pos = rs.get_box_pos(rs.data, args.target)
                    final_dist = float(np.linalg.norm(box_pos[:2] - goal[:2]))
                    disturbance = get_disturbance(rs, rs.data, scene_boxes,
                                                   args.target)
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
        
        while viewer.is_running():
            time.sleep(0.1)


if __name__ == "__main__":
    main()
