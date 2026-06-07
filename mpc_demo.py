import os
import json
import time
import numpy as np

import mujoco
import mujoco.viewer


# Use scene.xml because your working demo_push.py used scene.xml.
# If your active XML is ur5e.xml, change this to "ur5e.xml".
MODEL_PATH = "scene.xml"

TARGET_BOX = "amazon_box_medium_A4"
TARGET_GEOM = "amazon_box_medium_A4_geom"
TIP_SITE = "pusher_tip_site"
GOAL_GEOM = "goal_marker"

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

HOME_Q = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])


class NormalPushDemo:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)

        self.joint_ids = [self.model.joint(name).id for name in JOINT_NAMES]
        self.qpos_ids = [self.model.jnt_qposadr[jid] for jid in self.joint_ids]
        self.dof_ids = [self.model.jnt_dofadr[jid] for jid in self.joint_ids]
        self.act_ids = [self.model.actuator(name).id for name in ACTUATOR_NAMES]

        self.site_id = self.model.site(TIP_SITE).id
        self.box_body_id = self.model.body(TARGET_BOX).id
        self.box_geom_id = self.model.geom(TARGET_GEOM).id

        self.goal_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, GOAL_GEOM
        )

        self.box_joint_addr = {}
        for name in BOX_NAMES:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid == -1:
                continue
            jid = self.model.body_jntadr[bid]
            self.box_joint_addr[name] = self.model.jnt_qposadr[jid]

        self.goal_pos = None
        self.safe_push_z = None

    # -----------------------------
    # Basic helpers
    # -----------------------------

    def set_robot_qpos(self, q):
        for i, qid in enumerate(self.qpos_ids):
            self.data.qpos[qid] = q[i]

        for i, aid in enumerate(self.act_ids):
            self.data.ctrl[aid] = q[i]

        mujoco.mj_forward(self.model, self.data)

    def get_robot_qpos(self):
        return np.array([self.data.qpos[qid] for qid in self.qpos_ids])

    def set_box_pose(self, name, pos, yaw=0.0):
        if name not in self.box_joint_addr:
            return

        addr = self.box_joint_addr[name]

        self.data.qpos[addr + 0] = pos[0]
        self.data.qpos[addr + 1] = pos[1]
        self.data.qpos[addr + 2] = pos[2]

        self.data.qpos[addr + 3] = np.cos(yaw / 2.0)
        self.data.qpos[addr + 4] = 0.0
        self.data.qpos[addr + 5] = 0.0
        self.data.qpos[addr + 6] = np.sin(yaw / 2.0)

    def get_tip_pos(self):
        return self.data.site_xpos[self.site_id].copy()

    def get_box_pos(self):
        return self.data.xpos[self.box_body_id].copy()

    def get_box_size(self):
        return self.model.geom_size[self.box_geom_id].copy()

    def set_goal_marker(self):
        if self.goal_geom_id != -1 and self.goal_pos is not None:
            self.model.geom_pos[self.goal_geom_id, 0] = self.goal_pos[0]
            self.model.geom_pos[self.goal_geom_id, 1] = self.goal_pos[1]
            self.model.geom_pos[self.goal_geom_id, 2] = self.goal_pos[2]

    def distance_to_goal(self):
        box = self.get_box_pos()
        return float(np.linalg.norm(box[:2] - self.goal_pos[:2]))

    def get_state(self):
        tip = self.get_tip_pos()
        box = self.get_box_pos()
        box_size = self.get_box_size()

        state = np.concatenate(
            [
                tip,                  # 3
                box,                  # 3
                self.goal_pos,         # 3
                box_size,              # 3
                np.zeros(3),           # neighbor padding
                np.zeros(3),
                np.zeros(3),
            ]
        )

        return state.astype(np.float32)

    # -----------------------------
    # Reset scene
    # -----------------------------

    def reset_scene(self):
        mujoco.mj_resetData(self.model, self.data)

        self.set_robot_qpos(HOME_Q)

        for _ in range(150):
            mujoco.mj_step(self.model, self.data)

        # Keep only one target box on rack.
        self.set_box_pose(TARGET_BOX, [-0.10, 0.85, 0.365])

        # Move other boxes far away without changing XML.
        self.set_box_pose("amazon_box_small_A1", [5.0, 5.0, 0.35])
        self.set_box_pose("amazon_box_extra_large_2BB", [5.5, 5.0, 0.40])
        self.set_box_pose("amazon_box_medium_tall_1A9", [6.0, 5.0, 0.80])
        self.set_box_pose("amazon_box_large_B0", [6.5, 5.0, 0.80])

        mujoco.mj_forward(self.model, self.data)

        for _ in range(150):
            mujoco.mj_step(self.model, self.data)

        box = self.get_box_pos()
        box_size = self.get_box_size()

        # Push along +Y because your working reaching script approached from lower Y.
        self.goal_pos = np.array([box[0], box[1] + 0.22, box[2]], dtype=np.float64)

        # Push near box center / slightly above center.
        self.safe_push_z = box[2] + 0.02

        self.set_goal_marker()
        mujoco.mj_forward(self.model, self.data)

    # -----------------------------
    # Contact checking
    # -----------------------------

    def contact_info(self):
        has_target_contact = False
        has_shelf_contact = False
        pairs = []

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
            g2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
            pair = f"{g1} {g2}"
            pairs.append(pair)

            if ("pusher" in pair or "ee_pusher" in pair) and TARGET_GEOM in pair:
                has_target_contact = True

            if ("pusher" in pair or "ee_pusher" in pair) and "shelf_surface_bottom" in pair:
                has_shelf_contact = True

        return has_target_contact, has_shelf_contact, pairs

    # -----------------------------
    # IK solver
    # -----------------------------

    def solve_ik(self, target_pos, q_init=None, max_iter=160, step_size=0.45, tol=1e-3):
        if q_init is None:
            q = self.get_robot_qpos().copy()
        else:
            q = q_init.copy()

        for _ in range(max_iter):
            for i, qid in enumerate(self.qpos_ids):
                self.data.qpos[qid] = q[i]

            mujoco.mj_forward(self.model, self.data)

            site_pos = self.get_tip_pos()
            err = target_pos - site_pos

            if np.linalg.norm(err) < tol:
                break

            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)

            J = jacp[:, self.dof_ids]

            damping = 1e-3
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), err)

            q = q + step_size * dq

            for i, jid in enumerate(self.joint_ids):
                low, high = self.model.jnt_range[jid]
                if low < high:
                    q[i] = np.clip(q[i], low, high)
                else:
                    q[i] = np.clip(q[i], -6.0, 6.0)

        return q

    def move_to_q(self, viewer, q_target, steps=250, pause=0.004):
        q_start = self.get_robot_qpos()

        for k in range(steps):
            alpha = k / max(steps - 1, 1)
            q_cmd = (1.0 - alpha) * q_start + alpha * q_target

            for i, aid in enumerate(self.act_ids):
                self.data.ctrl[aid] = q_cmd[i]

            mujoco.mj_step(self.model, self.data)

            if viewer is not None:
                viewer.sync()
                time.sleep(pause)

    def move_pusher_to(self, viewer, target_pos, steps=250):
        q_now = self.get_robot_qpos()
        q_target = self.solve_ik(target_pos, q_init=q_now)

        print("IK target:", target_pos)

        # Show expected final tip from IK.
        for i, qid in enumerate(self.qpos_ids):
            self.data.qpos[qid] = q_target[i]
        mujoco.mj_forward(self.model, self.data)
        print("IK final tip:", self.get_tip_pos())
        print("IK final error:", np.linalg.norm(self.get_tip_pos() - target_pos))

        self.move_to_q(viewer, q_target, steps=steps)

    # -----------------------------
    # Reach contact
    # -----------------------------

    def reach_contact(self, viewer=None):
        box = self.get_box_pos()
        box_size = self.get_box_size()

        x = box[0]
        z = self.safe_push_z
        box_front_y = box[1] - box_size[1]

        pre_reach = np.array([x, box_front_y - 0.18, z], dtype=np.float64)
        contact = np.array([x, box_front_y - 0.020, z], dtype=np.float64)

        print("Box position:", box)
        print("Box half-size:", box_size)
        print("Goal:", self.goal_pos)
        print("Pre-reach:", pre_reach)
        print("Contact:", contact)

        print("Moving near box...")
        self.move_pusher_to(viewer, pre_reach, steps=300)

        print("Moving to contact...")
        self.move_pusher_to(viewer, contact, steps=250)

        # Creep forward until actual contact is detected.
        print("Creeping forward until pusher-box contact...")
        for k in range(30):
            has_target_contact, has_shelf_contact, pairs = self.contact_info()

            if has_target_contact:
                print("Contact achieved at creep step:", k)
                break

            current_tip = self.get_tip_pos()
            target_tip = current_tip + np.array([0.0, 0.003, 0.0])
            target_tip[2] = self.safe_push_z

            q_target = self.solve_ik(target_tip, q_init=self.get_robot_qpos(), max_iter=80, step_size=0.25)
            self.move_to_q(viewer, q_target, steps=25, pause=0.003)

        has_target_contact, has_shelf_contact, pairs = self.contact_info()

        for p in pairs:
            if "pusher" in p or TARGET_GEOM in p:
                print("CONTACT:", p)

        print("target_contact:", has_target_contact)
        print("shelf_contact:", has_shelf_contact)
        print("Tip after contact:", self.get_tip_pos())
        print("Box after contact:", self.get_box_pos())
        print("Distance after contact:", self.distance_to_goal())

    # -----------------------------
    # Incremental pushing
    # -----------------------------

    def incremental_push_to_goal(
        self,
        viewer=None,
        max_steps=80,
        step_size=0.004,
        save_samples=False,
    ):
        samples = []
        actions = []

        for t in range(max_steps):
            state = self.get_state()
            tip_before = self.get_tip_pos()
            box_before = self.get_box_pos()
            dist_before = self.distance_to_goal()

            if dist_before < 0.08:
                print("SUCCESS")
                break

            direction = self.goal_pos[:2] - box_before[:2]
            direction = direction / (np.linalg.norm(direction) + 1e-8)

            # Mostly push along +Y.
            action_xy = step_size * direction
            action_xy[0] = np.clip(action_xy[0], -0.0015, 0.0015)
            action_xy[1] = np.clip(action_xy[1], 0.0, step_size)

            target_tip = tip_before.copy()
            target_tip[0] += action_xy[0]
            target_tip[1] += action_xy[1]
            target_tip[2] = self.safe_push_z

            q_target = self.solve_ik(target_tip, q_init=self.get_robot_qpos(), max_iter=80, step_size=0.25)
            self.move_to_q(viewer, q_target, steps=25, pause=0.003)

            has_target_contact, has_shelf_contact, _ = self.contact_info()
            dist_after = self.distance_to_goal()
            box_after = self.get_box_pos()

            if save_samples:
                sample = {
                    "scenario": "normal",
                    "expert": "ik_reach_incremental_push",
                    "target_box": TARGET_BOX,
                    "state": state.tolist(),
                    "action": action_xy.tolist(),
                    "tip_pos": tip_before.tolist(),
                    "target_box_pos": box_before.tolist(),
                    "goal_pos": self.goal_pos.tolist(),
                    "distance_to_goal": dist_before,
                }
                samples.append(sample)
                actions.append(action_xy.tolist())

            print(
                "step:",
                t,
                "distance:",
                round(dist_after, 4),
                "box:",
                np.round(box_after, 3),
                "action:",
                np.round(action_xy, 4),
                "target_contact:",
                has_target_contact,
                "shelf_contact:",
                has_shelf_contact,
            )

            if viewer is not None:
                viewer.sync()
                time.sleep(0.02)

        return samples, actions


def make_future_chunks(actions, H=8):
    chunks = []

    for t in range(len(actions)):
        chunk = []
        for h in range(H):
            idx = min(t + h, len(actions) - 1)
            chunk.append(actions[idx])
        chunks.append(chunk)

    return chunks


def view_demo():
    env = NormalPushDemo()
    env.reset_scene()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        print("Initial tip:", env.get_tip_pos())
        print("Initial box:", env.get_box_pos())
        print("Goal:", env.goal_pos)
        print("Initial distance:", env.distance_to_goal())

        env.reach_contact(viewer=viewer)

        print("Starting incremental push...")
        env.incremental_push_to_goal(
            viewer=viewer,
            max_steps=80,
            step_size=0.004,
            save_samples=False,
        )

        print("Demo complete. Viewer will stay open for 5 seconds.")
        time.sleep(5)


def collect_demo(num_episodes=5, save_path="data/normal_incremental_demo.json"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    all_samples = []

    for ep in range(num_episodes):
        print("\nEpisode:", ep)

        env = NormalPushDemo()
        env.reset_scene()

        env.reach_contact(viewer=None)

        samples, actions = env.incremental_push_to_goal(
            viewer=None,
            max_steps=80,
            step_size=0.004,
            save_samples=True,
        )

        chunks = make_future_chunks(actions, H=8)

        for i, sample in enumerate(samples):
            sample["future_actions"] = chunks[i]
            all_samples.append(sample)

    with open(save_path, "w") as f:
        json.dump(all_samples, f)

    print("Saved samples:", len(all_samples))
    print("Path:", save_path)


if __name__ == "__main__":
    view_demo()

    # After visual demo works, comment view_demo()
    # and uncomment:
    # collect_demo(num_episodes=5)
