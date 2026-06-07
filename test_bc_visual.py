"""
test_bc_visual.py — Watch BC in MuJoCo with viewer.
"""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import mujoco
import mujoco.viewer

from Teleop_collector import (
    RobotScene, BOXES, HOME_QPOS,
    setup_scene, get_obs, get_disturbance, move_to_home,
    SIM_STEPS_PER_ACTION,
)


class BCPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(27, 256), nn.ReLU(), nn.Dropout(0.0),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.0),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 6),
        )
    def forward(self, x):
        return self.net(x)


class BCInference:
    def __init__(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        self.net = BCPolicy()
        self.net.load_state_dict(ckpt["policy"])
        self.net.eval()
        self.obs_mean = ckpt["obs_mean"]
        self.obs_std = ckpt["obs_std"]
        self.act_mean = ckpt["act_mean"]
        self.act_std = ckpt["act_std"]
        print(f"Loaded BC: epoch {ckpt.get('epoch')}, val_loss {ckpt.get('val_loss'):.4f}")

    @torch.no_grad()
    def predict(self, obs):
        norm_obs = torch.tensor((obs - self.obs_mean) / self.obs_std,
                                dtype=torch.float32).unsqueeze(0)
        norm_pred = self.net(norm_obs).squeeze(0).numpy()
        return norm_pred * self.act_std + self.act_mean


def make_simple_scene(seed, target_key="A1"):
    rng = np.random.default_rng(seed)
    target_pos = BOXES[target_key][1].copy()
    target_pos[0] += rng.uniform(-0.05, 0.05)
    target_pos[1] += rng.uniform(-0.03, 0.03)

    goal = target_pos.copy()
    angle = rng.uniform(-np.pi/6, np.pi/6)
    dist = rng.uniform(0.20, 0.30)
    goal[0] += dist * np.cos(angle)
    goal[1] += dist * np.sin(angle)

    scene_boxes = {target_key: target_pos}
    return scene_boxes, goal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="scene.xml")
    ap.add_argument("--ckpt", default="checkpoints/bc_best.pt")
    ap.add_argument("--target", default="A1", choices=["A1", "A4", "1A9", "2BB", "B0"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--sim-steps", type=int, default=10,
                    help="Physics steps per BC call")
    ap.add_argument("--slow", action="store_true")
    args = ap.parse_args()

    print(f"Loading scene from {args.xml}...")
    rs = RobotScene(args.xml)

    print(f"Loading BC from {args.ckpt}...")
    policy = BCInference(args.ckpt)

    scene_boxes, goal = make_simple_scene(args.seed, args.target)

    print(f"\nScene setup:")
    print(f"  Target box:    {args.target}")
    print(f"  Box starts at: {scene_boxes[args.target][:2].round(3)}")
    print(f"  Goal at:       {goal[:2].round(3)}")
    expected = goal[:2] - scene_boxes[args.target][:2]
    print(f"  Expected push: {expected.round(3)} ({np.linalg.norm(expected):.3f}m)")

    # Use the same setup_scene function that teleop used
    setup_scene(rs, scene_boxes, goal)

    # Move arm to home (same as teleop init)
    move_to_home(rs, steps=200)

    print(f"\nStarting BC rollout. Close viewer to exit.")
    print(f"Sim steps per BC call: {args.sim_steps}\n")

    with mujoco.viewer.launch_passive(rs.model, rs.data) as viewer:
        success = False
        for step in range(args.max_steps):
            if not viewer.is_running():
                print("Viewer closed.")
                break

            # BC inference
            obs = get_obs(rs, rs.data, args.target, goal)
            action = policy.predict(obs)

            # Apply action to actuators
            for i, aid in enumerate(rs.act_ids):
                rs.data.ctrl[aid] = float(np.clip(action[i], -6.2, 6.2))

            # Step physics
            for _ in range(args.sim_steps):
                mujoco.mj_step(rs.model, rs.data)

            viewer.sync()
            if args.slow:
                time.sleep(0.05)

            # Check success
            box_pos = rs.get_box_pos(rs.data, args.target)
            dist_to_goal = float(np.linalg.norm(box_pos[:2] - goal[:2]))

            if step % 20 == 0:
                print(f"  Step {step:4d}: box {box_pos[:2].round(3)}, "
                      f"dist {dist_to_goal:.3f}m, "
                      f"action[0:3] {action[:3].round(2)}")

            if dist_to_goal < 0.07:
                print(f"\n*** SUCCESS at step {step}! Box within 7cm of goal. ***")
                success = True
                break

        # Final report
        box_pos = rs.get_box_pos(rs.data, args.target)
        final_dist = float(np.linalg.norm(box_pos[:2] - goal[:2]))
        disturbance = get_disturbance(rs, rs.data, scene_boxes, args.target)

        print(f"\n{'='*50}")
        print(f"FINAL RESULT")
        print(f"{'='*50}")
        print(f"  Success:        {success}")
        print(f"  Final dist:     {final_dist:.3f}m  (threshold: 0.07m)")
        print(f"  Disturbance:    {disturbance:.3f}m")
        print(f"  Box final pos:  {box_pos[:2].round(3)}")
        print(f"  Goal pos:       {goal[:2].round(3)}")

        print(f"\nKeep viewer open to inspect, close when done.")
        while viewer.is_running():
            time.sleep(0.1)


if __name__ == "__main__":
    main()
