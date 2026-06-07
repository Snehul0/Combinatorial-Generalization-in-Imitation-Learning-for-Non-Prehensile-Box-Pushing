"""test_bc_kf_her.py — Test the HER-augmented keyframe BC in MuJoCo."""
import argparse, time
import numpy as np
import torch
import torch.nn as nn
import mujoco, mujoco.viewer

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/bc_kf_her.pt")
    ap.add_argument("--xml", default="scene.xml")
    ap.add_argument("--target", default="A1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--sim-steps", type=int, default=20)
    args = ap.parse_args()
    
    rs = RobotScene(args.xml)
    
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    hidden = ckpt.get('config', {}).get('hidden', 128)
    policy = BCPolicy(hidden=hidden)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    obs_mean = ckpt["obs_mean"]; obs_std = ckpt["obs_std"]
    act_mean = ckpt["act_mean"]; act_std = ckpt["act_std"]
    print(f"Loaded BC: epoch {ckpt.get('epoch')}, val_loss {ckpt.get('val_loss'):.4f}")
    
    rng = np.random.default_rng(args.seed)
    target_pos = BOXES[args.target][1].copy()
    target_pos[0] += rng.uniform(-0.05, 0.05)
    target_pos[1] += rng.uniform(-0.03, 0.03)
    
    goal = target_pos.copy()
    goal[0] += rng.uniform(0.18, 0.25)
    
    scene_boxes = {args.target: target_pos}
    
    print(f"\nScene: target {args.target} at {target_pos[:2].round(3)}, goal at {goal[:2].round(3)}")
    print(f"Expected push: {(goal[:2] - target_pos[:2]).round(3)}m")
    
    setup_scene(rs, scene_boxes, goal)
    move_to_home(rs, steps=200)
    
    print(f"\nStarting BC with sim_steps={args.sim_steps}\n")
    
    with mujoco.viewer.launch_passive(rs.model, rs.data) as viewer:
        success = False
        for step in range(args.max_steps):
            if not viewer.is_running():
                break
            
            obs = get_obs(rs, rs.data, args.target, goal)
            norm_obs = torch.tensor((obs - obs_mean) / obs_std,
                                    dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                norm_pred = policy(norm_obs).squeeze(0).numpy()
            action = norm_pred * act_std + act_mean
            
            for i, aid in enumerate(rs.act_ids):
                rs.data.ctrl[aid] = float(np.clip(action[i], -6.2, 6.2))
            
            for _ in range(args.sim_steps):
                mujoco.mj_step(rs.model, rs.data)
            
            viewer.sync()
            
            box_pos = rs.get_box_pos(rs.data, args.target)
            dist = float(np.linalg.norm(box_pos[:2] - goal[:2]))
            
            if step % 10 == 0:
                tip = rs.get_tip(rs.data)
                print(f"  Step {step:4d}: box {box_pos[:2].round(3)}, "
                      f"tip {tip[:2].round(3)}, dist {dist:.3f}m, "
                      f"action {action[:3].round(2)}")
            
            if dist < 0.07:
                print(f"\n*** SUCCESS at step {step}! ***")
                success = True
                break
        
        box_pos = rs.get_box_pos(rs.data, args.target)
        final_dist = float(np.linalg.norm(box_pos[:2] - goal[:2]))
        disturbance = get_disturbance(rs, rs.data, scene_boxes, args.target)
        
        print(f"\nFinal: success={success}, dist={final_dist:.3f}m, disturbance={disturbance:.3f}m")
        print(f"Box: {box_pos[:2].round(3)}, Goal: {goal[:2].round(3)}")
        
        while viewer.is_running():
            time.sleep(0.1)


if __name__ == "__main__":
    main()
