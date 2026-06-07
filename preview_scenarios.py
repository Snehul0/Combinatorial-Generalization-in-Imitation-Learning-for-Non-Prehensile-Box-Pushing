"""preview_scenarios.py — Render preview images of all teleop scenarios."""
import os
import numpy as np
import mujoco
import imageio

# Import setup from your teleop collector
from Teleo_collector_51d import (
    RobotScene, BOXES, make_scenario, setup_scene, move_to_home, HOME_QPOS
)

OUT_DIR = "scenario_previews"
os.makedirs(OUT_DIR, exist_ok=True)

WIDTH, HEIGHT = 640, 480

scenarios = [
    ("N3", list(range(100, 107))),  # seeds 100-106 (7 demos)
    ("N4", list(range(200, 208))),  # seeds 200-207 (8 demos)
    ("N5", list(range(300, 305))),  # seeds 300-304 (5 demos)
]

rs = RobotScene("scene.xml")
renderer = mujoco.Renderer(rs.model, height=HEIGHT, width=WIDTH)

for scenario, seeds in scenarios:
    print(f"\n=== Rendering {scenario} scenarios ===")
    for seed in seeds:
        scene_boxes, goal, target_key = make_scenario(scenario, seed)
        setup_scene(rs, scene_boxes, goal)
        
        # Move arm to home so it's not blocking the view
        move_to_home(rs, steps=80)
        
        # Step physics a few times to let things settle
        for _ in range(20):
            mujoco.mj_step(rs.model, rs.data)
        
        renderer.update_scene(rs.data)
        frame = renderer.render()
        
        out_path = f"{OUT_DIR}/{scenario}_seed{seed}.png"
        imageio.imwrite(out_path, frame)
        
        # Print summary
        blocker_keys = [k for k in scene_boxes.keys() if k != target_key]
        target_xy = scene_boxes[target_key][:2]
        goal_xy = goal[:2]
        push_dist = float(np.linalg.norm(goal_xy - target_xy))
        
        print(f"  {scenario} seed={seed}:")
        print(f"    target {target_key} at ({target_xy[0]:+.3f}, {target_xy[1]:+.3f})")
        print(f"    goal at ({goal_xy[0]:+.3f}, {goal_xy[1]:+.3f}), push distance {push_dist:.3f}m")
        for bk in blocker_keys:
            bp = scene_boxes[bk][:2]
            d = float(np.linalg.norm(bp - target_xy))
            print(f"    blocker {bk} at ({bp[0]:+.3f}, {bp[1]:+.3f}), {d:.3f}m from target")
        print(f"    -> {out_path}")

print(f"\nAll previews saved to {OUT_DIR}/")
print("Open the PNGs to see each scenario.")
