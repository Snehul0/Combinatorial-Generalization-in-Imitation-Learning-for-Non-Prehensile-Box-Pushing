"""
teleop_collector.py
===================

Interactive teleop demo collector for UR5e + flat pusher in MuJoCo.

WORKFLOW PER DEMO:
  STAGE 1: Place TARGET box (arrow keys + ENTER)
  STAGE 2: Place each BLOCKER box (arrow keys + ENTER)
  STAGE 3: Place GOAL (arrow keys + ENTER)
  STAGE 4: Teleop the arm via keyboard
  SAVE:    SPACE

Action: 6D joint positions (ACT/Diffusion/LeRobot compatible)
Observation: 27D state vector
"""

import argparse
import os
import pickle
import time
import numpy as np

import mujoco
import mujoco.viewer


HOME_QPOS = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

ACTUATOR_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow",
    "wrist_1", "wrist_2", "wrist_3",
]

BOXES = {
    "A1":  ("amazon_box_small_A1",        np.array([-0.30, 0.60, 0.346]),
                                          np.array([0.080, 0.055, 0.040])),
    "A4":  ("amazon_box_medium_A4",       np.array([-0.10, 0.60, 0.365]),
                                          np.array([0.100, 0.065, 0.055])),
    "2BB": ("amazon_box_extra_large_2BB", np.array([+0.33, 0.60, 0.395]),
                                          np.array([0.130, 0.080, 0.085])),
    "1A9": ("amazon_box_medium_tall_1A9", np.array([-0.30, 0.60, 0.806]),
                                          np.array([0.100, 0.075, 0.080])),
    "B0":  ("amazon_box_large_B0",        np.array([+0.20, 0.60, 0.400]),
                                          np.array([0.120, 0.075, 0.090])),
}

PARK_POS = np.array([0.0, 0.0, -5.0])

# Workspace bounds for placement (approximate reachable region)
PLACE_X_MIN, PLACE_X_MAX = -0.60, +0.60
PLACE_Y_MIN, PLACE_Y_MAX = +0.40, +0.75
PLACE_STEP = 0.02

# Teleop step sizes
STEP_XY = 0.004
STEP_Z = 0.002
STEP_W = 0.020
STEP_ORIENT = 0.025
STEP_JOINT = 0.020
SIM_STEPS_PER_ACTION = 8

CLIP_XY = 0.020
CLIP_Z = 0.008
CLIP_W = 0.060

MODE_CARTESIAN = "CARTESIAN"
MODE_ORIENTATION = "ORIENTATION"
MODE_JOINT = "JOINT"
MODES_ORDER = [MODE_CARTESIAN, MODE_ORIENTATION, MODE_JOINT]


class RobotScene:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
        self.qpos_ids = [self.model.jnt_qposadr[jid] for jid in self.joint_ids]
        self.dof_ids = [self.model.jnt_dofadr[jid] for jid in self.joint_ids]
        self.act_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACTUATOR_NAMES]
        self.tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pusher_tip_site")
        self.goal_marker_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal_marker")

    def set_robot_q(self, data, q):
        for i, qid in enumerate(self.qpos_ids): data.qpos[qid] = q[i]
        for i, aid in enumerate(self.act_ids): data.ctrl[aid] = q[i]
        mujoco.mj_forward(self.model, data)

    def get_robot_q(self, data):
        return np.array([data.qpos[qid] for qid in self.qpos_ids])

    def set_ctrl_q(self, data, q):
        for i, aid in enumerate(self.act_ids): data.ctrl[aid] = q[i]

    def get_tip(self, data):
        return data.site_xpos[self.tip_id].copy()

    def body_id(self, box_key):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, BOXES[box_key][0])

    def get_box_pos(self, data, box_key):
        return data.xpos[self.body_id(box_key)].copy()

    def set_box_qpos(self, data, box_key, pos):
        """Directly set a box's qpos in the simulation."""
        bid = self.body_id(box_key)
        if bid < 0: return
        jid = self.model.body_jntadr[bid]
        if jid < 0: return
        qadr = self.model.jnt_qposadr[jid]
        vadr = self.model.jnt_dofadr[jid]
        data.qpos[qadr:qadr+3] = pos
        data.qpos[qadr+3:qadr+7] = [1, 0, 0, 0]
        data.qvel[vadr:vadr+6] = 0.0

    def set_goal_marker(self, goal):
        if self.goal_marker_id != -1:
            self.model.geom_pos[self.goal_marker_id, 0] = goal[0]
            self.model.geom_pos[self.goal_marker_id, 1] = goal[1]
            self.model.geom_pos[self.goal_marker_id, 2] = goal[2]


def park_all_boxes(rs):
    """Park all boxes off-shelf initially. We'll place them one by one."""
    data = rs.data
    for key, (body_name, _, _) in BOXES.items():
        bid = mujoco.mj_name2id(rs.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0: continue
        jid = rs.model.body_jntadr[bid]
        if jid < 0: continue
        qadr = rs.model.jnt_qposadr[jid]
        vadr = rs.model.jnt_dofadr[jid]
        data.qpos[qadr:qadr+3] = PARK_POS
        data.qpos[qadr+3:qadr+7] = [1, 0, 0, 0]
        data.qvel[vadr:vadr+6] = 0.0
    mujoco.mj_forward(rs.model, data)


def initial_scene_reset(rs):
    """Reset to home, park all boxes."""
    mujoco.mj_resetData(rs.model, rs.data)
    rs.data.qpos[:6] = HOME_QPOS
    rs.data.ctrl[:6] = HOME_QPOS
    park_all_boxes(rs)
    mujoco.mj_forward(rs.model, rs.data)


def settle_physics(rs, n_steps=80):
    """Let physics settle after placing boxes."""
    for _ in range(n_steps):
        mujoco.mj_step(rs.model, rs.data)


def get_obs(rs, data, target_key, goal):
    model = rs.model
    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BOXES[target_key][0])
    qpos = data.qpos[:6].copy()
    qvel = data.qvel[:6].copy()
    tip = rs.get_tip(data)
    box = data.xpos[box_id].copy()
    vel6 = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, box_id, vel6, 0)
    box_linvel = vel6[3:].copy()
    return np.concatenate([qpos, qvel, tip, box, box_linvel, goal, goal - box], dtype=np.float32)


def get_disturbance(rs, data, scene_boxes, target_key):
    total = 0.0
    for key, start_pos in scene_boxes.items():
        if key == target_key: continue
        bid = rs.body_id(key)
        if bid >= 0: total += float(np.linalg.norm(data.xpos[bid] - start_pos))
    return total


def contact_info(rs, data, target_key):
    target_geom = BOXES[target_key][0] + "_geom"
    has_target = False
    has_bad = False
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = mujoco.mj_id2name(rs.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
        g2 = mujoco.mj_id2name(rs.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
        pair = f"{g1} {g2}"
        if ("pusher" in pair or "ee_pusher" in pair) and target_geom in pair:
            has_target = True
        if ("pusher" in pair or "wrist" in pair or "forearm" in pair or "upperarm" in pair) and \
                ("shelf" in pair or "beam" in pair or "wall" in pair or "floor" in pair):
            has_bad = True
    return has_target, has_bad


class KeyboardState:
    """Tracks pressed keys. Behavior changes based on current stage."""

    STAGE_PLACE = "PLACE"   # placing boxes or goal
    STAGE_TELEOP = "TELEOP" # teleoperating the arm

    def __init__(self):
        self.keys_held = set()
        self.signals = {
            'success': False, 'cancel': False, 'pause': False, 'quit': False,
            'mode_switch': False, 'reset_orientation': False, 'lock': False,
        }
        self.mode_index = 0
        self.selected_joint = 0
        self.stage = self.STAGE_PLACE

    @property
    def mode(self):
        return MODES_ORDER[self.mode_index]

    def on_key(self, keycode):
        # ENTER is universal lock signal
        if keycode == 257:
            self.signals['lock'] = True
            return

        # ESC quit
        if keycode == 256:
            self.signals['quit'] = True
            return

        if self.stage == self.STAGE_PLACE:
            # Arrow keys for placement
            if keycode == 262: self._toggle('right')
            if keycode == 263: self._toggle('left')
            if keycode == 264: self._toggle('back')
            if keycode == 265: self._toggle('forward')
            if keycode == ord('Q'): self._toggle('up')
            if keycode == ord('Z'): self._toggle('down')
            return

        # TELEOP stage
        if keycode in (ord('D'), 262): self._toggle('right')
        if keycode in (ord('A'), 263): self._toggle('left')
        if keycode in (ord('S'), 264): self._toggle('back')
        if keycode in (ord('W'), 265): self._toggle('forward')
        if keycode == ord('Q'): self._toggle('up')
        if keycode == ord('Z'): self._toggle('down')
        if keycode == ord('R'): self._toggle('wrist+')
        if keycode == ord('F'): self._toggle('wrist-')
        if keycode == ord('I'): self._toggle('pitch+')
        if keycode == ord('K'): self._toggle('pitch-')
        if keycode == ord('J'): self._toggle('yaw+')
        if keycode == ord('L'): self._toggle('yaw-')
        if keycode == ord('U'): self._toggle('roll+')
        if keycode == ord('O'): self._toggle('roll-')
        if ord('1') <= keycode <= ord('6'):
            self.selected_joint = keycode - ord('1')
            print(f"\n  [JOINT] selected {self.selected_joint + 1}: {JOINT_NAMES[self.selected_joint]}")
        if keycode == ord('['): self._toggle('joint-')
        if keycode == ord(']'): self._toggle('joint+')
        if keycode == 32: self.signals['success'] = True
        if keycode == ord('X'): self.signals['cancel'] = True
        if keycode == ord('P'): self.signals['pause'] = not self.signals['pause']
        if keycode == 258:
            self.mode_index = (self.mode_index + 1) % len(MODES_ORDER)
            self.keys_held.clear()
            self.signals['mode_switch'] = True
        if keycode == ord('B'): self.signals['reset_orientation'] = True

    def _toggle(self, key):
        if key in self.keys_held: self.keys_held.discard(key)
        else: self.keys_held.add(key)

    def place_delta(self):
        """Returns dx, dy, dz for box/goal placement."""
        dx = dy = dz = 0.0
        if 'forward' in self.keys_held: dx += PLACE_STEP
        if 'back' in self.keys_held: dx -= PLACE_STEP
        if 'right' in self.keys_held: dy += PLACE_STEP
        if 'left' in self.keys_held: dy -= PLACE_STEP
        if 'up' in self.keys_held: dz += PLACE_STEP / 2
        if 'down' in self.keys_held: dz -= PLACE_STEP / 2
        return dx, dy, dz

    def cartesian_delta(self):
        dx = dy = dz = dw = 0.0
        if 'forward' in self.keys_held: dx += STEP_XY
        if 'back' in self.keys_held: dx -= STEP_XY
        if 'right' in self.keys_held: dy += STEP_XY
        if 'left' in self.keys_held: dy -= STEP_XY
        if 'up' in self.keys_held: dz += STEP_Z
        if 'down' in self.keys_held: dz -= STEP_Z
        if 'wrist+' in self.keys_held: dw += STEP_W
        if 'wrist-' in self.keys_held: dw -= STEP_W
        return dx, dy, dz, dw

    def orientation_delta(self):
        droll = dpitch = dyaw = 0.0
        if 'pitch+' in self.keys_held: dpitch += STEP_ORIENT
        if 'pitch-' in self.keys_held: dpitch -= STEP_ORIENT
        if 'yaw+' in self.keys_held: dyaw += STEP_ORIENT
        if 'yaw-' in self.keys_held: dyaw -= STEP_ORIENT
        if 'roll+' in self.keys_held: droll += STEP_ORIENT
        if 'roll-' in self.keys_held: droll -= STEP_ORIENT
        return droll, dpitch, dyaw

    def joint_delta(self):
        dq = 0.0
        if 'joint+' in self.keys_held: dq += STEP_JOINT
        if 'joint-' in self.keys_held: dq -= STEP_JOINT
        return self.selected_joint, dq


def apply_cartesian(rs, dx, dy, dz, dw):
    model, data = rs.model, rs.data
    desired_tip = rs.get_tip(data).copy()
    desired_tip[0] += dx
    desired_tip[1] += dy
    desired_tip[2] += dz
    for _ in range(SIM_STEPS_PER_ACTION):
        tip = rs.get_tip(data)
        err = desired_tip - tip
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, rs.tip_id)
        J = jacp[:, rs.dof_ids]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err)
        q = rs.get_robot_q(data)
        q_new = q + 0.35 * dq
        q_new[5] += dw
        q_new = np.clip(q_new, -6.2, 6.2)
        rs.set_ctrl_q(data, q_new)
        mujoco.mj_step(model, data)


def apply_orientation(rs, droll, dpitch, dyaw):
    model, data = rs.model, rs.data
    tip_target = rs.get_tip(data).copy()
    omega = np.array([droll, dpitch, dyaw])
    for _ in range(SIM_STEPS_PER_ACTION):
        tip = rs.get_tip(data)
        pos_err = tip_target - tip
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, rs.tip_id)
        Jp = jacp[:, rs.dof_ids]
        Jr = jacr[:, rs.dof_ids]
        J_full = np.vstack([Jp, Jr])
        v_desired = np.concatenate([pos_err * 5.0, omega])
        dq = J_full.T @ np.linalg.solve(J_full @ J_full.T + 1e-3 * np.eye(6), v_desired)
        q = rs.get_robot_q(data)
        q_new = q + 0.2 * dq
        q_new = np.clip(q_new, -6.2, 6.2)
        rs.set_ctrl_q(data, q_new)
        mujoco.mj_step(model, data)


def apply_joint(rs, joint_idx, delta):
    model, data = rs.model, rs.data
    q = rs.get_robot_q(data)
    q_new = q.copy()
    q_new[joint_idx] += delta
    q_new = np.clip(q_new, -6.2, 6.2)
    for _ in range(SIM_STEPS_PER_ACTION):
        rs.set_ctrl_q(data, q_new)
        mujoco.mj_step(model, data)


def move_to_home(rs, render_cb=None, steps=200):
    data = rs.data
    model = rs.model
    q_start = data.qpos[:6].copy()
    for k in range(steps):
        alpha = k / max(steps - 1, 1)
        alpha_smooth = 3 * alpha**2 - 2 * alpha**3
        q_cmd = (1.0 - alpha_smooth) * q_start + alpha_smooth * HOME_QPOS
        data.ctrl[:6] = q_cmd
        mujoco.mj_step(model, data)
        if render_cb: render_cb()


def place_object_interactive(rs, viewer, keyboard, label, initial_pos, set_fn,
                              z_min=0.30, z_max=0.50):
    """Generic placement: arrow keys move the object, ENTER locks position.

    set_fn(pos) is called every step to update the object's position
    (either box qpos or goal marker).
    """
    print()
    print("=" * 70)
    print(f"PLACE {label}  (arrow keys = move, Q/Z = up/down, ENTER = lock, ESC = quit)")
    print("=" * 70)

    pos = initial_pos.copy()
    set_fn(pos)

    keyboard.signals['lock'] = False  # ensure not already triggered
    keyboard.keys_held.clear()

    while not keyboard.signals['lock']:
        if keyboard.signals['quit']:
            return None
        dx, dy, dz = keyboard.place_delta()
        if abs(dx) + abs(dy) + abs(dz) > 0:
            pos[0] += dx
            pos[1] += dy
            pos[2] += dz
            pos[0] = float(np.clip(pos[0], PLACE_X_MIN, PLACE_X_MAX))
            pos[1] = float(np.clip(pos[1], PLACE_Y_MIN, PLACE_Y_MAX))
            pos[2] = float(np.clip(pos[2], z_min, z_max))
            set_fn(pos)
            mujoco.mj_forward(rs.model, rs.data)
            print(f"\r  {label}: ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) [ENTER to lock] ",
                  end='', flush=True)
        viewer.sync()
        time.sleep(0.05)

    print(f"\n  {label} locked at {np.round(pos, 3)}")
    keyboard.signals['lock'] = False
    keyboard.keys_held.clear()
    return pos


def print_teleop_controls():
    print()
    print("=" * 70)
    print("STAGE 4: TELEOP THE ARM  (TAB cycles modes)")
    print("=" * 70)
    print("  CARTESIAN: W/S A/D Q/Z R/F")
    print("  ORIENTATION: I/K J/L U/O (B = reset wrist)")
    print("  JOINT: 1-6 select, ]/[ +/-")
    print("  SPACE = save success, X = cancel, P = pause, ESC = quit")
    print("=" * 70)
    print()


def collect_demo(rs, target_key, blocker_keys, max_steps=3000):
    initial_scene_reset(rs)
    keyboard = KeyboardState()

    print(f"\nTarget: {target_key}")
    print(f"Blockers to place: {blocker_keys if blocker_keys else 'none'}")

    scene_boxes = {}
    goal = None

    with mujoco.viewer.launch_passive(rs.model, rs.data, key_callback=keyboard.on_key) as viewer:

        # Move arm to home
        print("\nMoving to home...")
        move_to_home(rs, render_cb=viewer.sync, steps=200)
        time.sleep(0.5)

        # ─── STAGE 1: Place target box ───
        keyboard.stage = KeyboardState.STAGE_PLACE
        target_z = BOXES[target_key][1][2]
        target_initial = np.array([0.0, 0.60, target_z])

        def set_target(p):
            rs.set_box_qpos(rs.data, target_key, p)

        target_pos = place_object_interactive(
            rs, viewer, keyboard,
            f"TARGET ({target_key})",
            target_initial, set_target,
            z_min=target_z - 0.01, z_max=target_z + 0.01,
        )
        if target_pos is None: return None
        scene_boxes[target_key] = target_pos.copy()
        settle_physics(rs)

        # ─── STAGE 2: Place each blocker ───
        for i, blocker_key in enumerate(blocker_keys):
            if blocker_key == target_key: continue
            blocker_z = BOXES[blocker_key][1][2]
            # Start blocker near target but offset
            blocker_initial = target_pos.copy()
            blocker_initial[0] += 0.20
            blocker_initial[1] += 0.05 * ((-1) ** i)  # alternate sides
            blocker_initial[2] = blocker_z

            def set_blocker(p, bk=blocker_key):
                rs.set_box_qpos(rs.data, bk, p)

            blocker_pos = place_object_interactive(
                rs, viewer, keyboard,
                f"BLOCKER {i+1}/{len(blocker_keys)} ({blocker_key})",
                blocker_initial, set_blocker,
                z_min=blocker_z - 0.01, z_max=blocker_z + 0.01,
            )
            if blocker_pos is None: return None
            scene_boxes[blocker_key] = blocker_pos.copy()
            settle_physics(rs)

        # ─── STAGE 3: Place goal ───
        # Goal Z at target's Z is fine
        goal_initial = target_pos.copy()
        goal_initial[0] += 0.25

        def set_goal(p):
            rs.set_goal_marker(p)

        goal = place_object_interactive(
            rs, viewer, keyboard,
            "GOAL",
            goal_initial, set_goal,
            z_min=0.30, z_max=0.50,
        )
        if goal is None: return None

        # ─── STAGE 4: Teleop ───
        keyboard.stage = KeyboardState.STAGE_TELEOP
        print_teleop_controls()
        print(f"Ready. Mode: {keyboard.mode}\n")

        obs_list, act_list, rew_list, done_list, mode_list = [], [], [], [], []
        step_count = 0
        start_time = time.time()

        while step_count < max_steps:
            if keyboard.signals['quit']:
                print("\n  [QUIT]")
                return None
            if keyboard.signals['cancel']:
                print("\n  [CANCEL]")
                return None
            if keyboard.signals['success']:
                print(f"\n  [SUCCESS] step {step_count}")
                obs = get_obs(rs, rs.data, target_key, goal)
                obs_list.append(obs)
                act_list.append(rs.get_robot_q(rs.data).astype(np.float32))
                rew_list.append(np.float32(0.0))
                done_list.append(np.float32(True))
                mode_list.append(keyboard.mode)
                break
            if keyboard.signals['mode_switch']:
                print(f"\n  [MODE] -> {keyboard.mode}" + " " * 40)
                keyboard.signals['mode_switch'] = False
            if keyboard.signals['reset_orientation']:
                print("\n  [RESET wrist]" + " " * 30)
                q = rs.get_robot_q(rs.data).copy()
                q_target = q.copy()
                q_target[3:6] = HOME_QPOS[3:6]
                for _ in range(20):
                    q_interp = 0.9 * rs.get_robot_q(rs.data) + 0.1 * q_target
                    rs.set_ctrl_q(rs.data, q_interp)
                    for _ in range(2):
                        mujoco.mj_step(rs.model, rs.data)
                    viewer.sync()
                keyboard.signals['reset_orientation'] = False
                continue
            if keyboard.signals['pause']:
                viewer.sync()
                time.sleep(0.05)
                continue

            obs = get_obs(rs, rs.data, target_key, goal)
            obs_list.append(obs)

            mode = keyboard.mode
            if mode == MODE_CARTESIAN:
                dx, dy, dz, dw = keyboard.cartesian_delta()
                dx = float(np.clip(dx, -CLIP_XY, CLIP_XY))
                dy = float(np.clip(dy, -CLIP_XY, CLIP_XY))
                dz = float(np.clip(dz, -CLIP_Z, CLIP_Z))
                dw = float(np.clip(dw, -CLIP_W, CLIP_W))
                apply_cartesian(rs, dx, dy, dz, dw)
            elif mode == MODE_ORIENTATION:
                droll, dpitch, dyaw = keyboard.orientation_delta()
                if abs(droll) + abs(dpitch) + abs(dyaw) > 0:
                    apply_orientation(rs, droll, dpitch, dyaw)
                else:
                    for _ in range(SIM_STEPS_PER_ACTION):
                        mujoco.mj_step(rs.model, rs.data)
            elif mode == MODE_JOINT:
                joint_idx, delta = keyboard.joint_delta()
                if abs(delta) > 0:
                    apply_joint(rs, joint_idx, delta)
                else:
                    for _ in range(SIM_STEPS_PER_ACTION):
                        mujoco.mj_step(rs.model, rs.data)

            current_q = rs.get_robot_q(rs.data).astype(np.float32)
            act_list.append(current_q)
            rew_list.append(np.float32(0.0))
            done_list.append(np.float32(False))
            mode_list.append(mode)

            box = rs.get_box_pos(rs.data, target_key)
            box_to_goal = float(np.linalg.norm(box[:2] - goal[:2]))
            has_contact, bad_contact = contact_info(rs, rs.data, target_key)
            disturbance = get_disturbance(rs, rs.data, scene_boxes, target_key)

            if step_count % 5 == 0:
                elapsed = time.time() - start_time
                mode_str = mode[:4]
                extra = f" J{keyboard.selected_joint + 1}" if mode == MODE_JOINT else ""
                print(f"\r  step {step_count:4d} | {elapsed:5.1f}s | [{mode_str}{extra}] | "
                      f"box->goal {box_to_goal:.3f} | contact {'Y' if has_contact else '.'} | "
                      f"bad {'Y' if bad_contact else '.'} | dist {disturbance:.3f}    ",
                      end='', flush=True)

            viewer.sync()
            time.sleep(0.001)
            step_count += 1
        else:
            print(f"\n  [TIMEOUT]")

    print(f"\n  Steps: {step_count}, Recorded: {len(obs_list)}")
    success = bool(keyboard.signals['success'])
    final_box = rs.get_box_pos(rs.data, target_key)
    final_dist = float(np.linalg.norm(final_box[:2] - goal[:2]))
    disturbance = get_disturbance(rs, rs.data, scene_boxes, target_key)

    return {
        "observations": np.stack(obs_list),
        "actions": np.stack(act_list),
        "rewards": np.array(rew_list, dtype=np.float32),
        "dones": np.array(done_list, dtype=np.float32),
        "modes": mode_list,
        "info": {
            "target_box": target_key,
            "scene_boxes": {k: v.tolist() for k, v in scene_boxes.items()},
            "goal": goal.tolist(),
            "success": success,
            "final_dist": final_dist,
            "disturbance": disturbance,
            "demo_length": len(obs_list),
            "collection_method": "keyboard_teleop_interactive_placement",
            "obs_dim": 27,
            "act_dim": 6,
            "act_format": "joint_position",
        },
    }


def save_demo(demo, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    pkl_path = os.path.join(out_dir, f"{filename}.pkl")
    npz_path = os.path.join(out_dir, f"{filename}.npz")
    with open(pkl_path, "wb") as f:
        pickle.dump(demo, f)
    np.savez_compressed(npz_path,
        observations=demo["observations"], actions=demo["actions"],
        rewards=demo["rewards"], dones=demo["dones"])
    print(f"\nSaved:\n  {pkl_path}\n  {npz_path}")
    print(f"  obs: {demo['observations'].shape}, act: {demo['actions'].shape}")
    print(f"  success: {demo['info']['success']}, dist: {demo['info']['final_dist']:.4f}, "
          f"disturb: {demo['info']['disturbance']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True)
    parser.add_argument("--target", required=True, choices=list(BOXES.keys()))
    parser.add_argument("--blockers", nargs="*", default=[], choices=list(BOXES.keys()))
    parser.add_argument("--out_dir", default="demonstrations_teleop")
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    rs = RobotScene(args.xml)
    print(f"\nTarget: {args.target}")
    print(f"Blockers to place interactively: {args.blockers if args.blockers else 'none'}")

    demo = collect_demo(rs, args.target, args.blockers, max_steps=args.max_steps)
    if demo is None:
        print("\nDemo canceled.")
        return
    save_demo(demo, args.out_dir, args.name)


if __name__ == "__main__":
    main()
