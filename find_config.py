"""
find_configs.py
===============
Step 1: Run this script to find good pre-push joint configurations.
It opens the MuJoCo viewer. You manually pose the arm to a good
pre-push position for each box, then press ENTER to save the config.

This gives you hardcoded joint configs that ALWAYS work — no IK failures.

Usage:
  python find_configs.py --xml scene.xml --box A4
"""
import argparse
import numpy as np
import mujoco
import mujoco.viewer
import time

HOME_QPOS = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

BOXES = {
    "A1":  ("amazon_box_small_A1",  np.array([-0.50, 0.60, 0.346])),
    "A4":  ("amazon_box_medium_A4", np.array([-0.10, 0.60, 0.365])),
    "2BB": ("amazon_box_extra_large_2BB", np.array([0.33, 0.60, 0.395])),
    "1A9": ("amazon_box_medium_tall_1A9", np.array([-0.30, 0.60, 0.806])),
    "B0":  ("amazon_box_large_B0",  np.array([0.20, 0.60, 0.815])),
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", required=True)
    p.add_argument("--box", required=True, choices=list(BOXES.keys()))
    args = p.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data  = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)
    data.qpos[:6] = HOME_QPOS
    data.ctrl[:6] = HOME_QPOS
    mujoco.mj_forward(model, data)

    box_name, box_pos = BOXES[args.box]
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pusher_tip_site")

    print(f"\nBox: {args.box} at {box_pos}")
    print("Open viewer. Use the joint sliders to pose the arm.")
    print("Goal: pusher tip 3-5 cm BEHIND the box on the robot side (-Y)")
    print("Press CTRL+C in terminal to print current joint config.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        try:
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.01)
        except KeyboardInterrupt:
            tip = data.site_xpos[tip_id].copy()
            print(f"\n{'='*50}")
            print(f"Joint config: {np.round(data.qpos[:6], 4).tolist()}")
            print(f"Tip position: {np.round(tip, 4)}")
            print(f"Box position: {box_pos}")
            print(f"Tip-to-box:   {np.round(tip - box_pos, 4)}")
            print(f"Distance:     {np.linalg.norm(tip - box_pos):.4f} m")
            print(f"{'='*50}")
            print("\nCopy the joint config above into PREPUSH_CONFIGS in demo_collector.py")

if __name__ == "__main__":
    main()
