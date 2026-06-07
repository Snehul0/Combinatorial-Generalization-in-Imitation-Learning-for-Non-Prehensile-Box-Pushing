"""test_bc_mppi_4d.py — Test 4D-Cartesian BC in MuJoCo."""
import argparse, time
import numpy as np
import torch
import torch.nn as nn
import mujoco, mujoco.viewer

from Teleop_collector import (
    RobotScene, BOXES, HOME_QPOS,
    setup_scene, get_obs, get_disturbance, move_to_home,
    apply_cartesian,
    SIM_STEPS_PER_ACTION,
)


class BCPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(27, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 4),
        )
    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/bc_mppi_4d.pt")
    ap.add_argument("--xml", default="scene.xml")
    ap.add_argument("--target", default="A1", choices=["A1", "A4", "2BB"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=2000)
    args = ap.parse_args()
    
    rs = RobotScene(args.xml)
    
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    policy = BCPolicy()
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    obs_mean = ckpt["obs_mean"]; obs_std = ckpt["obs_std"]
    act_mean = ckpt["act_mean"]; act_std = ckpt["act_std"]
    print(f"Loaded BC: epoch {ckpt.get('epoch')}, val_loss {ckpt.get('val_loss'):.6f}")
    print(f"Action format: {ckpt['config'].get('action_format', 'cartesian_delta')}")
    
    # Use default box positions like MPPI did
    rng = np.random.default_rng(args.seed)
    target_pos = BOXES[args.target][1].copy()
    target_pos[0] += rng.uniform(-0.02, 0.02)
    
    # Goal +X 20cm push (matches MPPI training)
    goal = target_pos.copy()
    goal[0] += 0.20
    
    scene_boxes = {args.target: target_pos}
    
    print(f"\nScene:")
    print(f"  Target: {args.target} at {target_pos[:2].round(3)}")
    print(f"  Goal:                  {goal[:2].round(3)}")
    print(f"  Expected push: +X {0.20}m")
    
    setup_scene(rs, scene_boxes, goal)
    move_to_home(rs, steps=200)
    
    print(f"\nStarting BC. Will use apply_cartesian for action execution.\n")
    
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
            action = norm_pred * act_std + act_mean   # 4D Cartesian delta
            
            dx, dy, dz, dw = action
            
            # Clip to safe bounds
            dx = float(np.clip(dx, -0.020, 0.020))
            dy = float(np.clip(dy, -0.020, 0.020))
            dz = float(np.clip(dz, -0.010, 0.010))
            dw = float(np.clip(dw, -0.050, 0.050))
            
            # apply_cartesian does the Jacobian IK + sim step internally
            apply_cartesian(rs, dx, dy, dz, dw, sim_steps=SIM_STEPS_PER_ACTION)
            
            viewer.sync()
            
            box_pos = rs.get_box_pos(rs.data, args.target)
            dist = float(np.linalg.norm(box_pos[:2] - goal[:2]))
            
            if step % 20 == 0:
                tip = rs.get_tip(rs.data)
                print(f"  Step {step:4d}: box {box_pos[:2].round(3)}, "
                      f"tip {tip[:2].round(3)}, dist {dist:.3f}m, "
                      f"action [{dx:+.4f}, {dy:+.4f}, {dz:+.4f}, {dw:+.4f}]")
            
            if dist < 0.07:
                print(f"\n*** SUCCESS at step {step}! ***")
                success = True
                break
        
        box_pos = rs.get_box_pos(rs.data, args.target)
        final_dist = float(np.linalg.norm(box_pos[:2] - goal[:2]))
        print(f"\nFinal: success={success}, dist={final_dist:.3f}m")
        print(f"Box at: {box_pos[:2].round(3)}, Goal: {goal[:2].round(3)}")
        
        while viewer.is_running():
            time.sleep(0.1)


if __name__ == "__main__":
    main()
