import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

print(f"\n=== {model.nbody} bodies in scene ===\n")
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "(unnamed)"
    pos = data.xpos[i]
    if "box" in name or "shelf" in name or "base" in name or "wrist" in name or "pusher" in name:
        print(f"  {name:35s} pos=[{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]")

# Robot base location matters for reach calculations
base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
if base >= 0:
    print(f"\nRobot base at: {data.xpos[base]}")

# Tip site location
tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pusher_tip_site")
if tip >= 0:
    print(f"Pusher tip   : {data.site_xpos[tip]}")
