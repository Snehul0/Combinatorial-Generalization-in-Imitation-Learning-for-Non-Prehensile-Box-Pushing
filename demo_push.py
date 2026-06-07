import time
import numpy as np
import mujoco
import mujoco.viewer


MODEL_PATH = "scene.xml"   # change to your XML filename if needed


model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)


JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

ACTUATOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]

SITE_NAME = "pusher_tip_site"
BOX_BODY_NAME = "amazon_box_medium_A4"
BOX_GEOM_NAME = "amazon_box_medium_A4_geom"


joint_ids = [model.joint(name).id for name in JOINT_NAMES]
qpos_ids = [model.jnt_qposadr[jid] for jid in joint_ids]
dof_ids = [model.jnt_dofadr[jid] for jid in joint_ids]
act_ids = [model.actuator(name).id for name in ACTUATOR_NAMES]

site_id = model.site(SITE_NAME).id
box_body_id = model.body(BOX_BODY_NAME).id
box_geom_id = model.geom(BOX_GEOM_NAME).id


def set_robot_qpos(q):
    for i, qid in enumerate(qpos_ids):
        data.qpos[qid] = q[i]
    for i, aid in enumerate(act_ids):
        data.ctrl[aid] = q[i]
    mujoco.mj_forward(model, data)


def get_robot_qpos():
    return np.array([data.qpos[qid] for qid in qpos_ids])


def solve_ik(target_pos, q_init=None, max_iter=120, step_size=0.45, tol=1e-3):
    """
    Simple Jacobian IK for pusher_tip_site position.
    It solves only position, not orientation.
    This is enough for a visual reaching/pushing demo.
    """
    if q_init is None:
        q = get_robot_qpos().copy()
    else:
        q = q_init.copy()

    for _ in range(max_iter):
        for i, qid in enumerate(qpos_ids):
            data.qpos[qid] = q[i]

        mujoco.mj_forward(model, data)

        site_pos = data.site_xpos[site_id].copy()
        err = target_pos - site_pos

        if np.linalg.norm(err) < tol:
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

        J = jacp[:, dof_ids]

        # Damped least squares for stability
        damping = 1e-3
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), err)

        q = q + step_size * dq

        # Clip to joint limits
        for i, jid in enumerate(joint_ids):
            low, high = model.jnt_range[jid]
            q[i] = np.clip(q[i], low, high)

    return q


def move_to_q(viewer, q_target, steps=250, pause=0.005):
    q_start = get_robot_qpos()

    for k in range(steps):
        alpha = k / max(steps - 1, 1)
        q_cmd = (1 - alpha) * q_start + alpha * q_target

        for i, aid in enumerate(act_ids):
            data.ctrl[aid] = q_cmd[i]

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(pause)


def move_pusher_to(viewer, target_pos, steps=250):
    q_now = get_robot_qpos()
    q_target = solve_ik(target_pos, q_init=q_now)
    move_to_q(viewer, q_target, steps=steps)


# Better starting pose
home_q = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])


with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Resetting robot to home pose...")
    set_robot_qpos(home_q)

    for _ in range(200):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.005)

    # Read current box position
    mujoco.mj_forward(model, data)
    box_pos = data.xpos[box_body_id].copy()
    box_size = model.geom_size[box_geom_id].copy()

    print("Box position:", box_pos)
    print("Box half-size:", box_size)

    # We push along +Y direction into the rack.
    # The robot approaches from the front side, lower Y.
    x = box_pos[0]
    z = box_pos[2]

    box_front_y = box_pos[1] - box_size[1]

    pre_reach = np.array([x, box_front_y - 0.18, z])
    contact = np.array([x, box_front_y - 0.02, z])
    push_1 = np.array([x, box_pos[1] + 0.10, z])
    push_2 = np.array([x, box_pos[1] + 0.22, z])
    retreat = np.array([x, box_front_y - 0.20, z + 0.12])

    print("Moving near box...")
    move_pusher_to(viewer, pre_reach, steps=300)

    print("Contacting box...")
    move_pusher_to(viewer, contact, steps=250)

    print("Pushing box...")
    move_pusher_to(viewer, push_1, steps=300)
    move_pusher_to(viewer, push_2, steps=350)

    print("Retreating...")
    move_pusher_to(viewer, retreat, steps=300)

    print("Demo complete. Viewer will stay open.")
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.005)
