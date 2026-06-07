import os
import json
import time
import numpy as np
from tqdm import tqdm

import mujoco
import mujoco.viewer


MODEL_PATH = "ur5e.xml"

TARGET_BOX = "amazon_box_medium_A4"
TIP_SITE = "pusher_tip_site"
GOAL_GEOM = "goal_marker"

# Start with this height.
# If the pusher is too high and misses the box, try 0.395.
# If the pusher hits the shelf, try 0.420 or 0.430.
SAFE_PUSH_Z = 0.405

BOX_HALF_X = 0.100
BOX_HALF_Y = 0.065
BOX_HALF_Z = 0.055

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

BOX_NAMES = [
    "amazon_box_small_A1",
    "amazon_box_medium_A4",
    "amazon_box_extra_large_2BB",
    "amazon_box_medium_tall_1A9",
    "amazon_box_large_B0",
]


class NormalPushEnv:
    def __init__(self, seed=0):
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)

        self.target_box = TARGET_BOX
        self.goal_pos = np.array([0.25, 0.85, 0.365], dtype=np.float64)

        self.home_q = np.array(
            [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
            dtype=np.float64,
        )

        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES
        ]

        self.joint_qpos_addr = [
            self.model.jnt_qposadr[jid] for jid in self.joint_ids
        ]

        self.joint_dof_addr = [
            self.model.jnt_dofadr[jid] for jid in self.joint_ids
        ]

        self.tip_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            TIP_SITE,
        )

        self.goal_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            GOAL_GEOM,
        )

        self.box_body_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in BOX_NAMES
        }

        self.box_joint_addr = {}

        for name in BOX_NAMES:
            body_id = self.box_body_ids[name]
            joint_id = self.model.body_jntadr[body_id]
            self.box_joint_addr[name] = self.model.jnt_qposadr[joint_id]

    def set_goal_marker(self):
        self.model.geom_pos[self.goal_geom_id, 0] = self.goal_pos[0]
        self.model.geom_pos[self.goal_geom_id, 1] = self.goal_pos[1]
        self.model.geom_pos[self.goal_geom_id, 2] = self.goal_pos[2]

    def reset_robot_home(self):
        for i, addr in enumerate(self.joint_qpos_addr):
            self.data.qpos[addr] = self.home_q[i]
            self.data.ctrl[i] = self.home_q[i]

        mujoco.mj_forward(self.model, self.data)

        for _ in range(200):
            mujoco.mj_step(self.model, self.data)

    def set_box_pose(self, name, pos, yaw=0.0):
        addr = self.box_joint_addr[name]

        self.data.qpos[addr + 0] = pos[0]
        self.data.qpos[addr + 1] = pos[1]
        self.data.qpos[addr + 2] = pos[2]

        self.data.qpos[addr + 3] = np.cos(yaw / 2.0)
        self.data.qpos[addr + 4] = 0.0
        self.data.qpos[addr + 5] = 0.0
        self.data.qpos[addr + 6] = np.sin(yaw / 2.0)

    def reset_normal(self):
        mujoco.mj_resetData(self.model, self.data)

        self.reset_robot_home()

        # Correct target location on shelf.
        self.set_box_pose(TARGET_BOX, [-0.10, 0.85, 0.365])

        # Move other boxes far away.
        self.set_box_pose("amazon_box_small_A1", [2.0, 2.0, 0.35])
        self.set_box_pose("amazon_box_extra_large_2BB", [2.4, 2.0, 0.40])
        self.set_box_pose("amazon_box_medium_tall_1A9", [2.8, 2.0, 0.80])
        self.set_box_pose("amazon_box_large_B0", [3.2, 2.0, 0.80])

        # Correct goal location on same shelf row.
        self.goal_pos = np.array([0.25, 0.85, 0.365], dtype=np.float64)
        self.set_goal_marker()

        mujoco.mj_forward(self.model, self.data)

        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

    def get_tip_pos(self, data=None):
        if data is None:
            data = self.data
        return data.site_xpos[self.tip_site_id].copy()

    def get_box_pos(self, data=None):
        if data is None:
            data = self.data
        body_id = self.box_body_ids[TARGET_BOX]
        return data.xpos[body_id].copy()

    def get_state(self):
        tip = self.get_tip_pos()
        box = self.get_box_pos()
        goal = self.goal_pos
        box_size = np.array([BOX_HALF_X, BOX_HALF_Y, BOX_HALF_Z], dtype=np.float64)

        # State dim = 21:
        # tip(3), target box pos(3), goal(3), box size(3), 3 padded neighbor boxes(9)
        state = np.concatenate(
            [
                tip,
                box,
                goal,
                box_size,
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
            ]
        )

        return state.astype(np.float32)

    def set_ctrl_q(self, data, q):
        for i in range(6):
            data.ctrl[i] = q[i]

    def ik_step(self, data, desired_tip_pos, damping=1e-3, step_scale=0.25):
        current_tip = data.site_xpos[self.tip_site_id].copy()
        err = desired_tip_pos - current_tip

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))

        mujoco.mj_jacSite(
            self.model,
            data,
            jacp,
            jacr,
            self.tip_site_id,
        )

        J = jacp[:, self.joint_dof_addr]
        JT = J.T

        dq = JT @ np.linalg.solve(J @ JT + damping * np.eye(3), err)
        dq = step_scale * dq

        q = np.array([data.qpos[addr] for addr in self.joint_qpos_addr])
        q_new = q + dq

        q_new = np.clip(q_new, -6.0, 6.0)

        return q_new

    def apply_cartesian_action(self, data, action, sim_steps=20):
        """
        2D Cartesian action:
            action = [dx, dy]

        z is fixed at SAFE_PUSH_Z.
        This prevents MPC from choosing low actions that hit the shelf.
        """
        action = np.asarray(action, dtype=np.float64)

        if action.shape[0] < 2:
            raise ValueError("Action must have at least 2 values: [dx, dy].")

        dx = np.clip(action[0], -0.015, 0.015)
        dy = np.clip(action[1], -0.015, 0.015)

        desired_tip = data.site_xpos[self.tip_site_id].copy()
        desired_tip[0] += dx
        desired_tip[1] += dy
        desired_tip[2] = SAFE_PUSH_Z

        for _ in range(sim_steps):
            q_target = self.ik_step(data, desired_tip)
            self.set_ctrl_q(data, q_target)
            mujoco.mj_step(self.model, data)

    def move_tip_to_position(self, desired_tip_pos, max_steps=250, tol=0.018):
        """
        Move pusher tip toward desired x-y position.
        z is controlled by SAFE_PUSH_Z.
        """
        actions = []

        print("\nMoving tip to:", desired_tip_pos)

        for step in range(max_steps):
            current_tip = self.get_tip_pos()
            delta = desired_tip_pos - current_tip
            err_xy = np.linalg.norm(delta[:2])

            if step % 20 == 0:
                print(
                    "  approach step:",
                    step,
                    "current_tip:",
                    current_tip,
                    "desired:",
                    desired_tip_pos,
                    "xy error:",
                    err_xy,
                )

            if err_xy < tol:
                print("  Reached desired XY. Final XY error:", err_xy)
                break

            action = np.clip(delta[:2], -0.015, 0.015)

            before = self.get_tip_pos().copy()
            self.apply_cartesian_action(self.data, action, sim_steps=35)
            after = self.get_tip_pos().copy()

            actual_action = after[:2] - before[:2]
            actions.append(actual_action.tolist())

        final_tip = self.get_tip_pos()
        final_error_xy = np.linalg.norm(desired_tip_pos[:2] - final_tip[:2])

        print("  Final tip:", final_tip)
        print("  Final desired:", desired_tip_pos)
        print("  Final XY error:", final_error_xy)

        if final_error_xy > 0.05:
            print("  WARNING: IK did not reach desired XY point.")

        return actions

    def move_behind_and_contact(self):
        """
        Normal pushing:
        target box at [-0.10, 0.85, 0.365]
        goal at [0.25, 0.85, 0.365]

        Goal is in +X direction, so pusher should contact from -X side.
        """
        box = self.get_box_pos()
        goal = self.goal_pos

        left_face_x = box[0] - BOX_HALF_X

        pre_reach = np.array(
            [
                left_face_x - 0.20,
                box[1],
                SAFE_PUSH_Z,
            ],
            dtype=np.float64,
        )

        contact = np.array(
            [
                left_face_x - 0.020,
                box[1],
                SAFE_PUSH_Z,
            ],
            dtype=np.float64,
        )

        print("Box:", box)
        print("Goal:", goal)
        print("Left face x:", left_face_x)
        print("Pre-reach:", pre_reach)
        print("Contact point:", contact)

        self.move_tip_to_position(pre_reach, max_steps=280)
        self.move_tip_to_position(contact, max_steps=220)

    def snapshot(self):
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "time": float(self.data.time),
        }

    def restore(self, data, snap):
        data.qpos[:] = snap["qpos"]
        data.qvel[:] = snap["qvel"]
        data.ctrl[:] = snap["ctrl"]
        data.time = snap["time"]
        mujoco.mj_forward(self.model, data)

    def contact_info(self, data=None):
        if data is None:
            data = self.data

        has_shelf_contact = False
        has_target_contact = False
        contact_pairs = []

        for i in range(data.ncon):
            c = data.contact[i]
            g1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
            g2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""

            pair = f"{g1} {g2}"
            contact_pairs.append(pair)

            if "pusher_tip" in pair and "shelf_surface_bottom" in pair:
                has_shelf_contact = True

            if "pusher_tip" in pair and "amazon_box_medium_A4_geom" in pair:
                has_target_contact = True

        return has_shelf_contact, has_target_contact, contact_pairs

    def success(self):
        box = self.get_box_pos()
        return np.linalg.norm(box[:2] - self.goal_pos[:2]) < 0.08

    def distance_to_goal(self):
        box = self.get_box_pos()
        return float(np.linalg.norm(box[:2] - self.goal_pos[:2]))


class NormalMPC:
    def __init__(
        self,
        env,
        horizon=8,
        num_samples=128,
        num_elites=16,
        cem_iters=2,
        seed=0,
    ):
        self.env = env
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.cem_iters = cem_iters
        self.rng = np.random.default_rng(seed)

    def rollout_cost(self, action_seq, snap):
        data = mujoco.MjData(self.env.model)
        self.env.restore(data, snap)

        prev_a = np.zeros(2)
        control_cost = 0.0
        smooth_cost = 0.0
        stage_goal_cost = 0.0
        shelf_collision_cost = 0.0
        target_contact_bonus = 0.0

        for a in action_seq:
            self.env.apply_cartesian_action(data, a, sim_steps=15)

            control_cost += float(np.sum(a ** 2))
            smooth_cost += float(np.sum((a - prev_a) ** 2))
            prev_a = a.copy()

            box_now = self.env.get_box_pos(data)
            stage_goal_cost += float(
                np.sum((box_now[:2] - self.env.goal_pos[:2]) ** 2)
            )

            has_shelf_contact, has_target_contact, _ = self.env.contact_info(data)

            if has_shelf_contact:
                shelf_collision_cost += 1.0

            if has_target_contact:
                target_contact_bonus += 1.0

        box_final = self.env.get_box_pos(data)
        tip_final = self.env.get_tip_pos(data)

        goal_cost = float(np.sum((box_final[:2] - self.env.goal_pos[:2]) ** 2))
        contact_cost = float(np.sum((tip_final[:2] - box_final[:2]) ** 2))

        total_cost = (
            80.0 * goal_cost
            + 3.0 * stage_goal_cost
            + 2.0 * contact_cost
            + 0.2 * control_cost
            + 0.4 * smooth_cost
            + 150.0 * shelf_collision_cost
            - 2.0 * target_contact_bonus
        )

        return total_cost

    def plan(self):
        snap = self.env.snapshot()

        box = self.env.get_box_pos()
        goal = self.env.goal_pos

        push_dir = goal - box
        push_dir[2] = 0.0
        push_dir = push_dir / (np.linalg.norm(push_dir[:2]) + 1e-8)

        # 2D action sequence: [dx, dy]
        mean = np.zeros((self.horizon, 2), dtype=np.float64)

        for h in range(self.horizon):
            mean[h, 0] = 0.010 * push_dir[0]
            mean[h, 1] = 0.010 * push_dir[1]

        std = np.ones((self.horizon, 2), dtype=np.float64) * 0.006

        best_seq = None
        best_cost = np.inf

        for _ in range(self.cem_iters):
            samples = self.rng.normal(
                mean,
                std,
                size=(self.num_samples, self.horizon, 2),
            )

            samples = np.clip(samples, -0.015, 0.015)

            costs = np.zeros(self.num_samples)

            for i in range(self.num_samples):
                costs[i] = self.rollout_cost(samples[i], snap)

            elite_idx = np.argsort(costs)[: self.num_elites]
            elites = samples[elite_idx]

            mean = elites.mean(axis=0)
            std = elites.std(axis=0) + 1e-4

            if costs[elite_idx[0]] < best_cost:
                best_cost = float(costs[elite_idx[0]])
                best_seq = samples[elite_idx[0]].copy()

        return best_seq, best_cost


def make_future_chunks(actions, H=8):
    chunks = []

    for t in range(len(actions)):
        chunk = []

        for h in range(H):
            idx = min(t + h, len(actions) - 1)
            chunk.append(actions[idx])

        chunks.append(chunk)

    return chunks


def collect_normal_demo(
    num_episodes=3,
    max_steps=80,
    save_path="data/normal_mpc_demo.json",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    env = NormalPushEnv(seed=0)
    planner = NormalMPC(
        env,
        horizon=8,
        num_samples=128,
        num_elites=16,
        cem_iters=2,
        seed=0,
    )

    all_samples = []

    for ep in tqdm(range(num_episodes), desc="normal"):
        env.reset_normal()

        print("Initial distance:", env.distance_to_goal())

        env.move_behind_and_contact()

        print("Distance after contact:", env.distance_to_goal())

        ep_samples = []
        ep_actions = []

        for t in range(max_steps):
            state = env.get_state()
            tip = env.get_tip_pos()
            box = env.get_box_pos()

            best_seq, best_cost = planner.plan()
            action = best_seq[0]

            sample = {
                "scenario": "normal",
                "num_active_boxes": 1,
                "active_boxes": [TARGET_BOX],
                "target_box": TARGET_BOX,
                "state": state.tolist(),
                "action": action.tolist(),
                "tip_pos": tip.tolist(),
                "target_box_pos": box.tolist(),
                "goal_pos": env.goal_pos.tolist(),
                "mpc_cost": float(best_cost),
                "distance_to_goal": env.distance_to_goal(),
            }

            ep_samples.append(sample)
            ep_actions.append(action.tolist())

            env.apply_cartesian_action(env.data, action, sim_steps=20)

            has_shelf_contact, has_target_contact, _ = env.contact_info()

            print(
                "step:",
                t,
                "cost:",
                round(best_cost, 4),
                "distance:",
                round(env.distance_to_goal(), 4),
                "action:",
                action,
                "target_contact:",
                has_target_contact,
                "shelf_contact:",
                has_shelf_contact,
            )

            if env.success():
                print("SUCCESS")
                break

        chunks = make_future_chunks(ep_actions, H=8)

        for i, sample in enumerate(ep_samples):
            sample["future_actions"] = chunks[i]
            all_samples.append(sample)

    with open(save_path, "w") as f:
        json.dump(all_samples, f)

    print("Saved samples:", len(all_samples))
    print("Path:", save_path)


def view_normal_mpc():
    env = NormalPushEnv(seed=0)
    planner = NormalMPC(
        env,
        horizon=8,
        num_samples=128,
        num_elites=16,
        cem_iters=2,
        seed=0,
    )

    env.reset_normal()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        print("Initial distance:", env.distance_to_goal())

        env.move_behind_and_contact()
        viewer.sync()
        time.sleep(0.5)

        print("Tip after contact phase:", env.get_tip_pos())
        print("Box after contact phase:", env.get_box_pos())
        print("Distance after contact:", env.distance_to_goal())

        # For first debug, stop after reaching.
        # If pusher reaches left face correctly, comment out this return.
        # return

        for t in range(100):
            best_seq, best_cost = planner.plan()
            action = best_seq[0]

            env.apply_cartesian_action(env.data, action, sim_steps=20)

            has_shelf_contact, has_target_contact, contact_pairs = env.contact_info()

            for pair in contact_pairs:
                if (
                    "pusher" in pair
                    or "amazon_box_medium_A4" in pair
                    or "shelf_surface_bottom" in pair
                ):
                    print("CONTACT:", pair)

            print(
                "step:",
                t,
                "cost:",
                round(best_cost, 4),
                "distance:",
                round(env.distance_to_goal(), 4),
                "action:",
                action,
                "target_contact:",
                has_target_contact,
                "shelf_contact:",
                has_shelf_contact,
            )

            viewer.sync()
            time.sleep(0.02)

            if env.success():
                print("SUCCESS")
                break

        time.sleep(5)


if __name__ == "__main__":
    # First visually debug:
    view_normal_mpc()

    # After visual behavior works, comment view_normal_mpc()
    # and uncomment this:
    # collect_normal_demo(num_episodes=3)
