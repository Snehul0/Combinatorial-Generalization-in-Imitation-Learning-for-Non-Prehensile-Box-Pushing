import numpy as np
import mujoco

MODEL_PATH = "scene.xml"

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

TIP_SITE = "pusher_tip_site"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

joint_ids = [model.joint(name).id for name in JOINT_NAMES]
qpos_ids = [model.jnt_qposadr[jid] for jid in joint_ids]

tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TIP_SITE)

# A4 box is at [-0.10, 0.60, 0.365].
# For +Y pushing, good pre-contact is slightly in front of box:
# y = 0.60 - box_half_y - clearance ≈ 0.60 - 0.065 - 0.025 = 0.510
target_tip = np.array([-0.10, 0.51, 0.385])

rng = np.random.default_rng(0)

best = []

# Joint search ranges.
# These are broad but not insane.
ranges = [
    (-3.14, 3.14),   # shoulder_pan
    (-2.6, -0.3),    # shoulder_lift
    (0.3, 2.8),      # elbow
    (-2.8, 0.2),     # wrist_1
    (-2.4, 0.2),     # wrist_2
    (-3.14, 3.14),   # wrist_3
]

N = 20000

for _ in range(N):
    q = np.array([rng.uniform(lo, hi) for lo, hi in ranges])

    for i, qid in enumerate(qpos_ids):
        data.qpos[qid] = q[i]

    mujoco.mj_forward(model, data)

    tip = data.site_xpos[tip_id].copy()
    diff = tip - target_tip

    # Weighted cost: prioritize y and z because those were wrong before.
    cost = (
        1.0 * diff[0] ** 2
        + 3.0 * diff[1] ** 2
        + 3.0 * diff[2] ** 2
    )

    best.append((cost, q.copy(), tip.copy(), diff.copy()))

best.sort(key=lambda x: x[0])

print("Target tip:", target_tip)
print("\nTop 10 FK seed candidates:")

for i in range(10):
    cost, q, tip, diff = best[i]
    print("\nCandidate", i)
    print("q =", np.round(q, 4).tolist())
    print("tip =", np.round(tip, 4).tolist())
    print("diff =", np.round(diff, 4).tolist())
    print("distance =", float(np.linalg.norm(diff)))
    print("weighted cost =", float(cost))

print("\nUse Candidate 0 first as SHELF_SEED_LOWER.")
