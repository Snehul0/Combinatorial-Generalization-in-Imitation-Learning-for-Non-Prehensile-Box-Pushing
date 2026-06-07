"""
Teleop_collector.py
===================

Interactive keyboard teleoperation demo collector for UR5e + flat pusher in MuJoCo.

WORKFLOW PER DEMO:
  1. Launch -> arm moves to home
  2. TARGET PLACEMENT MODE: arrow keys/Q/Z move target box one step per click, ENTER locks
  3. GOAL PLACEMENT MODE: arrow keys/Q/Z move goal marker one step per click, ENTER locks
  4. TELEOP MODE: Cartesian keys move arm/pusher
     SPACE saves success, H saves failed attempt for HER

Action format: 6D joint positions
Observation format: 27D state vector
"""

import argparse
import os
import pickle
import time
import numpy as np

import mujoco
import mujoco.viewer


# ============================================================
# CONFIGURATION
# ============================================================

HOME_QPOS = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

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

BOXES = {
    "A1":  ("amazon_box_small_A1",        np.array([-0.30, 0.45, 0.346]), np.array([0.080, 0.055, 0.040])),
    "A4":  ("amazon_box_medium_A4",       np.array([-0.10, 0.45, 0.365]), np.array([0.100, 0.065, 0.055])),
    "2BB": ("amazon_box_extra_large_2BB", np.array([+0.33, 0.45, 0.395]), np.array([0.130, 0.080, 0.085])),
    "1A9": ("amazon_box_medium_tall_1A9", np.array([-0.30, 0.45, 0.390]), np.array([0.100, 0.075, 0.080])),
    "B0":  ("amazon_box_large_B0",        np.array([+0.20, 0.45, 0.400]), np.array([0.120, 0.075, 0.090])),
}

PARK_POS = np.array([0.0, 0.0, -5.0])

# Placement bounds and step sizes
PLACE_X_MIN, PLACE_X_MAX = -0.85, +0.85
PLACE_Y_MIN, PLACE_Y_MAX = +0.10, +0.80
PLACE_STEP = 0.02

# Backward-compatible names used by old goal placement code
GOAL_STEP = PLACE_STEP
GOAL_X_MIN, GOAL_X_MAX = PLACE_X_MIN, PLACE_X_MAX
GOAL_Y_MIN, GOAL_Y_MAX = PLACE_Y_MIN, PLACE_Y_MAX

# Teleop step sizes
STEP_XY = 0.020
STEP_Z = 0.010
STEP_W = 0.050
STEP_ORIENT = 0.025
STEP_JOINT = 0.020
SIM_STEPS_PER_ACTION = 3

CLIP_XY = 0.030
CLIP_Z = 0.020
CLIP_W = 0.100

MODE_CARTESIAN = "CARTESIAN"
MODE_ORIENTATION = "ORIENTATION"
MODE_JOINT = "JOINT"
MODES_ORDER = [MODE_CARTESIAN, MODE_ORIENTATION, MODE_JOINT]


# ============================================================
# ROBOT SCENE
# ============================================================

class RobotScene:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES
        ]
        self.qpos_ids = [self.model.jnt_qposadr[jid] for jid in self.joint_ids]
        self.dof_ids = [self.model.jnt_dofadr[jid] for jid in self.joint_ids]
        self.act_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ACTUATOR_NAMES
        ]
        self.tip_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "pusher_tip_site"
        )
        self.goal_marker_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal_marker"
        )

        print("\n===== ROBOT DEBUG INFO =====")
        for name, jid, qid, did in zip(JOINT_NAMES, self.joint_ids, self.qpos_ids, self.dof_ids):
            print(f"{name:25s} joint_id={jid}, qpos_id={qid}, dof_id={did}")
        for name, aid in zip(ACTUATOR_NAMES, self.act_ids):
            print(f"{name:25s} actuator_id={aid}")
        print("tip_id =", self.tip_id)
        print("goal_marker_id =", self.goal_marker_id)
        print("nq =", self.model.nq, "nv =", self.model.nv, "nu =", self.model.nu)
        print("============================\n")

    def set_robot_q(self, data, q):
        """Set joint positions AND control targets to the same value.
        This is the right way to initialize -- positions and ctrl in sync,
        velocities zero. Used only at reset, not during teleop."""
        for i, qid in enumerate(self.qpos_ids):
            data.qpos[qid] = q[i]
        for dof in self.dof_ids:
            data.qvel[dof] = 0.0
        for i, aid in enumerate(self.act_ids):
            if aid >= 0:
                data.ctrl[aid] = q[i]
        mujoco.mj_forward(self.model, data)

    def get_robot_q(self, data):
        return np.array([data.qpos[qid] for qid in self.qpos_ids])

    def set_ctrl_q(self, data, q):
        for i, aid in enumerate(self.act_ids):
            if aid >= 0:
                data.ctrl[aid] = q[i]

    def get_tip(self, data):
        return data.site_xpos[self.tip_id].copy()

    def body_id(self, box_key):
        body_name = BOXES[box_key][0]
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)

    def get_box_pos(self, data, box_key):
        return data.xpos[self.body_id(box_key)].copy()

    def set_goal_marker(self, goal):
        if self.goal_marker_id != -1:
            self.model.geom_pos[self.goal_marker_id, 0] = goal[0]
            self.model.geom_pos[self.goal_marker_id, 1] = goal[1]
            self.model.geom_pos[self.goal_marker_id, 2] = goal[2]
            mujoco.mj_forward(self.model, self.data)

    def set_box_qpos(self, data, box_key, pos):
        """Directly place a box at pos = [x, y, z]."""
        bid = self.body_id(box_key)
        if bid < 0:
            return

        jid = self.model.body_jntadr[bid]
        if jid < 0:
            return

        qadr = self.model.jnt_qposadr[jid]
        vadr = self.model.jnt_dofadr[jid]

        data.qpos[qadr:qadr + 3] = pos
        data.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]
        data.qvel[vadr:vadr + 6] = 0.0

        mujoco.mj_forward(self.model, data)


# ============================================================
# SCENE SETUP AND OBSERVATION
# ============================================================

def setup_initial_scene(target_key, blockers, jitter_seed=None):
    """Create initial scene dictionary. Target is still manually placed later."""
    target_pos = BOXES[target_key][1].copy()
    if jitter_seed is not None:
        rng = np.random.default_rng(jitter_seed)
        target_pos[0] += float(rng.uniform(-0.05, 0.05))
        target_pos[1] += float(rng.uniform(-0.03, 0.03))

    scene_boxes = {target_key: target_pos.copy()}

    for blocker_key in blockers:
        if blocker_key == target_key:
            continue
        blocker_pos = BOXES[blocker_key][1].copy()
        if jitter_seed is not None:
            rng = np.random.default_rng(jitter_seed + 100)
            blocker_pos[0] += float(rng.uniform(-0.05, 0.05))
            blocker_pos[1] += float(rng.uniform(-0.03, 0.03))
        scene_boxes[blocker_key] = blocker_pos

    return scene_boxes


def setup_scene(rs, scene_boxes, goal):
    model = rs.model
    data = rs.data
    mujoco.mj_resetData(model, data)

    # Put robot at home and park boxes not in scene.
    rs.set_robot_q(data, HOME_QPOS)

    for key, (body_name, _, _) in BOXES.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            continue
        jid = model.body_jntadr[bid]
        if jid < 0:
            continue
        qadr = model.jnt_qposadr[jid]
        vadr = model.jnt_dofadr[jid]
        pos = scene_boxes[key] if key in scene_boxes else PARK_POS
        data.qpos[qadr:qadr + 3] = pos
        data.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]
        data.qvel[vadr:vadr + 6] = 0.0

    rs.set_goal_marker(goal)
    rs.set_robot_q(data, HOME_QPOS)
    mujoco.mj_forward(model, data)


def get_obs(rs, data, target_key, goal):
    """27D observation."""
    model = rs.model
    box_name = BOXES[target_key][0]
    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, box_name)

    qpos = rs.get_robot_q(data).copy()
    qvel = np.array([data.qvel[dof] for dof in rs.dof_ids])
    tip = rs.get_tip(data)
    box = data.xpos[box_id].copy()

    vel6 = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, box_id, vel6, 0)
    box_linvel = vel6[3:].copy()

    return np.concatenate(
        [qpos, qvel, tip, box, box_linvel, goal, goal - box],
        dtype=np.float32,
    )


def get_disturbance(rs, data, scene_boxes, target_key):
    total = 0.0
    for key, start_pos in scene_boxes.items():
        if key == target_key:
            continue
        bid = rs.body_id(key)
        if bid >= 0:
            total += float(np.linalg.norm(data.xpos[bid] - start_pos))
    return total


def contact_info(rs, data, target_key):
    target_geom = BOXES[target_key][0] + "_geom"
    has_target_contact = False
    has_bad_contact = False
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = mujoco.mj_id2name(rs.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
        g2 = mujoco.mj_id2name(rs.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
        pair = f"{g1} {g2}"
        if ("pusher" in pair or "ee_pusher" in pair) and target_geom in pair:
            has_target_contact = True
        if ("pusher" in pair or "wrist" in pair or "forearm" in pair or "upperarm" in pair) and \
                ("shelf" in pair or "beam" in pair or "wall" in pair or "floor" in pair):
            has_bad_contact = True
    return has_target_contact, has_bad_contact


# ============================================================
# KEYBOARD STATE
# ============================================================

class KeyboardState:
    def __init__(self):
        self.keys_held = set()           # legacy, unused for teleop now
        self.place_nudge = np.zeros(3, dtype=float)
        # One-shot accumulators for teleop. Each keypress adds a delta.
        # The delta is consumed (zeroed) every control step.
        self.cart_nudge = np.zeros(4, dtype=float)    # dx, dy, dz, dw
        self.orient_nudge = np.zeros(3, dtype=float)  # droll, dpitch, dyaw
        self.joint_nudge = 0.0                        # delta for selected joint
        self.signals = {
            'success': False,
            'her_save': False,
            'cancel': False,
            'pause': False,
            'quit': False,
            'mode_switch': False,
            'reset_orientation': False,
            'lock_placement': False,
        }
        self.mode_index = 0
        self.selected_joint = 0
        self.in_placement = True

    @property
    def mode(self):
        return MODES_ORDER[self.mode_index]

    def on_key(self, keycode):
        # ENTER locks target/goal placement.
        if keycode == 257:
            self.signals['lock_placement'] = True
            return

        # ESC quits.
        if keycode == 256:
            self.signals['quit'] = True
            return

        # Placement mode: one key press = one step.
        if self.in_placement:
            if keycode == 265:        # Arrow Up: +X
                self.place_nudge[0] += PLACE_STEP
            elif keycode == 264:      # Arrow Down: -X
                self.place_nudge[0] -= PLACE_STEP
            elif keycode == 262:      # Arrow Right: +Y
                self.place_nudge[1] += PLACE_STEP
            elif keycode == 263:      # Arrow Left: -Y
                self.place_nudge[1] -= PLACE_STEP
            elif keycode == ord('Q'):
                self.place_nudge[2] += PLACE_STEP / 2
            elif keycode == ord('Z'):
                self.place_nudge[2] -= PLACE_STEP / 2
            return

        # Teleop mode -- each keypress adds ONE delta to the accumulator.
        # CARTESIAN nudges
        if keycode in (ord('D'), 262):   self.cart_nudge[1] += STEP_XY  # +Y
        if keycode in (ord('A'), 263):   self.cart_nudge[1] -= STEP_XY  # -Y
        if keycode in (ord('S'), 264):   self.cart_nudge[0] -= STEP_XY  # -X
        if keycode in (ord('W'), 265):   self.cart_nudge[0] += STEP_XY  # +X
        if keycode == ord('Q'):          self.cart_nudge[2] += STEP_Z   # +Z
        if keycode == ord('Z'):          self.cart_nudge[2] -= STEP_Z   # -Z
        if keycode == ord('R'):          self.cart_nudge[3] += STEP_W   # wrist+
        if keycode == ord('F'):          self.cart_nudge[3] -= STEP_W   # wrist-

        # ORIENTATION nudges (droll, dpitch, dyaw)
        if keycode == ord('I'):          self.orient_nudge[1] += STEP_ORIENT  # pitch+
        if keycode == ord('K'):          self.orient_nudge[1] -= STEP_ORIENT  # pitch-
        if keycode == ord('J'):          self.orient_nudge[2] += STEP_ORIENT  # yaw+
        if keycode == ord('L'):          self.orient_nudge[2] -= STEP_ORIENT  # yaw-
        if keycode == ord('U'):          self.orient_nudge[0] += STEP_ORIENT  # roll+
        if keycode == ord('O'):          self.orient_nudge[0] -= STEP_ORIENT  # roll-

        # JOINT nudges
        if ord('1') <= keycode <= ord('6'):
            self.selected_joint = keycode - ord('1')
            print(f"\n  [JOINT] selected joint {self.selected_joint + 1}: {JOINT_NAMES[self.selected_joint]}")

        if keycode == ord('N'):          self.joint_nudge -= STEP_JOINT
        if keycode == ord('M'):          self.joint_nudge += STEP_JOINT
        if keycode == 32:
            self.signals['success'] = True

        # H = save this rollout as a failed-but-useful HER attempt.
        # Later, training can relabel the goal as the final achieved box position.
        if keycode == ord('H'):
            self.signals['her_save'] = True

        if keycode == ord('X'):
            self.signals['cancel'] = True
        if keycode == ord('P'):
            self.signals['pause'] = not self.signals['pause']

        if keycode == 258:  # TAB
            self.mode_index = (self.mode_index + 1) % len(MODES_ORDER)
            self.keys_held.clear()
            self.signals['mode_switch'] = True

        if keycode == ord('B'):
            self.signals['reset_orientation'] = True

    def _press_once(self, key):
        self.keys_held.clear()
        self.keys_held.add(key)

    def placement_delta(self):
        dx, dy, dz = self.place_nudge.copy()
        self.place_nudge[:] = 0.0
        return dx, dy, dz

    # Backward-compatible name.
    def goal_delta(self):
        return self.placement_delta()

    def cartesian_delta(self):
        # Consume the one-shot accumulator: read it, zero it.
        # Cap at single-step magnitude so OS key-autorepeat can't queue
        # up multi-step motions on what the user thinks is a single press.
        dx = float(np.clip(self.cart_nudge[0], -STEP_XY, STEP_XY))
        dy = float(np.clip(self.cart_nudge[1], -STEP_XY, STEP_XY))
        dz = float(np.clip(self.cart_nudge[2], -STEP_Z,  STEP_Z))
        dw = float(np.clip(self.cart_nudge[3], -STEP_W,  STEP_W))
        self.cart_nudge[:] = 0.0
        return dx, dy, dz, dw

    def orientation_delta(self):
        droll  = float(np.clip(self.orient_nudge[0], -STEP_ORIENT, STEP_ORIENT))
        dpitch = float(np.clip(self.orient_nudge[1], -STEP_ORIENT, STEP_ORIENT))
        dyaw   = float(np.clip(self.orient_nudge[2], -STEP_ORIENT, STEP_ORIENT))
        self.orient_nudge[:] = 0.0
        return droll, dpitch, dyaw

    def joint_delta(self):
        dq = float(np.clip(self.joint_nudge, -STEP_JOINT, STEP_JOINT))
        self.joint_nudge = 0.0
        return self.selected_joint, dq


# ============================================================
# CONTROL APPLICATION
# ============================================================

def apply_cartesian(rs, dx, dy, dz, dw, sim_steps=SIM_STEPS_PER_ACTION):
    """IK-driven Cartesian control. Computes target joint config once,
    then holds it via actuators while physics resolves contacts."""
    model = rs.model
    data = rs.data

    # Compute target tip position
    desired_tip = rs.get_tip(data).copy()
    desired_tip[0] += dx
    desired_tip[1] += dy
    desired_tip[2] += dz

    # Compute target joint configuration via Jacobian IK (one shot,
    # based on CURRENT joint positions, not drooping ones).
    tip = rs.get_tip(data)
    err = desired_tip - tip

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, rs.tip_id)

    J = jacp[:, rs.dof_ids]
    dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err)

    # Use the commanded control as the base, not the actual (drooping) qpos.
    # This prevents gravity-induced drift from compounding.
    q_target = np.array([data.ctrl[aid] for aid in rs.act_ids])
    q_target = q_target + 0.8 * dq
    q_target[5] += dw
    q_target = np.clip(q_target, -6.2, 6.2)

    # Hold this target while physics runs multiple steps.
    # PD actuators will track to it and resist gravity.
    rs.set_ctrl_q(data, q_target)
    for _ in range(sim_steps):
        mujoco.mj_step(model, data)


def apply_orientation(rs, droll, dpitch, dyaw, sim_steps=SIM_STEPS_PER_ACTION):
    model = rs.model
    data = rs.data

    tip_target = rs.get_tip(data).copy()
    omega = np.array([droll, dpitch, dyaw])

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

    # Use commanded ctrl as base to prevent gravity drift
    q_target = np.array([data.ctrl[aid] for aid in rs.act_ids])
    q_target = q_target + 0.5 * dq
    q_target = np.clip(q_target, -6.2, 6.2)

    rs.set_ctrl_q(data, q_target)
    for _ in range(sim_steps):
        mujoco.mj_step(model, data)


def apply_joint(rs, joint_idx, delta, sim_steps=SIM_STEPS_PER_ACTION):
    model = rs.model
    data = rs.data

    # Base on COMMANDED position, not actual (drooping) position
    q_target = np.array([data.ctrl[aid] for aid in rs.act_ids])
    q_target[joint_idx] += delta
    q_target = np.clip(q_target, -6.2, 6.2)

    rs.set_ctrl_q(data, q_target)
    for _ in range(sim_steps):
        mujoco.mj_step(model, data)


def move_to_home(rs, render_cb=None, steps=200):
    """Smoothly move the arm to home, then hold there with extra settle steps."""
    data = rs.data
    model = rs.model

    q_start = rs.get_robot_q(data)
    q_target = HOME_QPOS.copy()

    # Smooth ramp from start to home
    for k in range(steps):
        alpha = k / max(steps - 1, 1)
        alpha_smooth = 3 * alpha**2 - 2 * alpha**3
        q_cmd = (1.0 - alpha_smooth) * q_start + alpha_smooth * q_target
        rs.set_ctrl_q(data, q_cmd)
        mujoco.mj_step(model, data)
        if render_cb:
            render_cb()

    # Extra settle: hold home command for a bit so the arm fully stabilizes
    rs.set_ctrl_q(data, q_target)
    for _ in range(100):
        mujoco.mj_step(model, data)
        if render_cb:
            render_cb()


def hold_robot_pose(rs):
    # Just step physics. The actuators already have the last commanded target,
    # so they'll actively hold the arm against gravity.
    # DO NOT read current qpos and re-command -- that creates drift.
    mujoco.mj_step(rs.model, rs.data)


# ============================================================
# PLACEMENT CONTROLS AND FUNCTIONS
# ============================================================

def print_target_controls(target_key):
    print()
    print("=" * 70)
    print(f"STEP 1: PLACE TARGET BOX ({target_key})")
    print("=" * 70)
    print("Use arrow keys to move the target box.")
    print("  ↑  Move box +X")
    print("  ↓  Move box -X")
    print("  →  Move box +Y")
    print("  ←  Move box -Y")
    print("  Q  Raise box Z")
    print("  Z  Lower box Z")
    print("  ENTER  Lock target box position")
    print("  ESC    Quit without saving")
    print("One key press = one small step.")
    print("=" * 70)
    print()
    
    
def print_blocker_controls(blocker_key):
    print()
    print("=" * 70)
    print(f"STEP 2: PLACE BLOCKER BOX ({blocker_key})")
    print("=" * 70)
    print("Use arrow keys to move the blocker box.")
    print("  ↑  Move blocker +X")
    print("  ↓  Move blocker -X")
    print("  →  Move blocker +Y")
    print("  ←  Move blocker -Y")
    print("  Q  Raise blocker Z")
    print("  Z  Lower blocker Z")
    print("  ENTER  Lock blocker box position")
    print("  ESC    Quit without saving")
    print("One key press = one small step.")
    print("=" * 70)
    print()


def print_goal_controls():
    print()
    print("=" * 70)
    print("STEP 2: PLACE THE GOAL")
    print("=" * 70)
    print("Use arrow keys to move the green goal marker.")
    print("  ↑  Move goal +X")
    print("  ↓  Move goal -X")
    print("  →  Move goal +Y")
    print("  ←  Move goal -Y")
    print("  Q  Raise goal Z")
    print("  Z  Lower goal Z")
    print("  ENTER  Lock goal position")
    print("  ESC    Quit without saving")
    print("One key press = one small step.")
    print("=" * 70)
    print()


def print_teleop_controls():
    print()
    print("=" * 70)
    print("STEP 3: PUSH THE TARGET BOX TO THE GOAL")
    print("=" * 70)
    print("MODE: TAB cycles  CARTESIAN -> ORIENTATION -> JOINT")
    print("CARTESIAN:                ORIENTATION:           JOINT:")
    print("  W/S  forward/back         I/K  pitch +/-         1-6  select joint")
    print("  A/D  left/right           J/L  yaw +/-           M    increase")
    print("  Q/Z  up/down              U/O  roll +/-          N    decrease")
    print("  R/F  wrist3 +/-           B    reset wrist")
    print("SAVE/QUIT:")
    print("  SPACE  save successful demo")
    print("  H      save failed attempt for HER")
    print("  X      cancel without saving")
    print("  P      pause/resume")
    print("  ESC    quit without saving")
    print("=" * 70)
    print()


def place_target_interactive(rs, target_key, viewer, keyboard):
    target_z = BOXES[target_key][1][2]
    pos = BOXES[target_key][1].copy()

    print_target_controls(target_key)
    print(f"Initial target placement: {np.round(pos, 3)}")

    keyboard.signals['lock_placement'] = False
    keyboard.in_placement = True
    keyboard.keys_held.clear()
    keyboard.place_nudge[:] = 0.0

    while not keyboard.signals['lock_placement']:
        if keyboard.signals['quit']:
            return None

        dx, dy, dz = keyboard.placement_delta()
        if abs(dx) + abs(dy) + abs(dz) > 0:
            pos[0] += dx
            pos[1] += dy
            pos[2] += dz

            pos[0] = float(np.clip(pos[0], PLACE_X_MIN, PLACE_X_MAX))
            pos[1] = float(np.clip(pos[1], PLACE_Y_MIN, PLACE_Y_MAX))
            pos[2] = float(np.clip(pos[2], target_z - 0.01, target_z + 0.01))

            rs.set_box_qpos(rs.data, target_key, pos)

            print(
                f"\r  TARGET {target_key}: ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) [ENTER to lock]    ",
                end="",
                flush=True,
            )

            keyboard.keys_held.clear()
            keyboard.place_nudge[:] = 0.0

        hold_robot_pose(rs)
        viewer.sync()
        time.sleep(0.05)

    print()
    print(f"\n  Target locked at {np.round(pos, 3)}")

    keyboard.signals['lock_placement'] = False
    keyboard.keys_held.clear()
    keyboard.place_nudge[:] = 0.0

    return pos
    
    
    
def place_blocker_interactive(rs, blocker_key, viewer, keyboard):
    blocker_z = BOXES[blocker_key][1][2]
    pos = BOXES[blocker_key][1].copy()

    print_blocker_controls(blocker_key)
    print(f"Initial blocker placement: {np.round(pos, 3)}")

    keyboard.signals['lock_placement'] = False
    keyboard.in_placement = True
    keyboard.keys_held.clear()
    keyboard.place_nudge[:] = 0.0

    # Put blocker at its initial placement before user moves it
    rs.set_box_qpos(rs.data, blocker_key, pos)

    while not keyboard.signals['lock_placement']:
        if keyboard.signals['quit']:
            return None

        dx, dy, dz = keyboard.placement_delta()
        if abs(dx) + abs(dy) + abs(dz) > 0:
            pos[0] += dx
            pos[1] += dy
            pos[2] += dz

            pos[0] = float(np.clip(pos[0], PLACE_X_MIN, PLACE_X_MAX))
            pos[1] = float(np.clip(pos[1], PLACE_Y_MIN, PLACE_Y_MAX))
            pos[2] = float(np.clip(pos[2], blocker_z - 0.01, blocker_z + 0.01))

            rs.set_box_qpos(rs.data, blocker_key, pos)

            print(
                f"\r  BLOCKER {blocker_key}: ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) [ENTER to lock]    ",
                end="",
                flush=True,
            )

            keyboard.keys_held.clear()
            keyboard.place_nudge[:] = 0.0

        hold_robot_pose(rs)
        viewer.sync()
        time.sleep(0.05)

    print()
    print(f"\n  Blocker {blocker_key} locked at {np.round(pos, 3)}")

    keyboard.signals['lock_placement'] = False
    keyboard.keys_held.clear()
    keyboard.place_nudge[:] = 0.0

    return pos


def place_goal_interactive(rs, scene_boxes, target_key, viewer, keyboard):
    target_pos = rs.get_box_pos(rs.data, target_key)
    goal = target_pos.copy()
    goal[0] += 0.25

    print_goal_controls()
    print(f"Initial goal placement: {np.round(goal, 3)}")

    keyboard.signals['lock_placement'] = False
    keyboard.in_placement = True
    keyboard.keys_held.clear()
    keyboard.place_nudge[:] = 0.0

    while not keyboard.signals['lock_placement']:
        if keyboard.signals['quit']:
            return None

        dx, dy, dz = keyboard.placement_delta()
        if abs(dx) + abs(dy) + abs(dz) > 0:
            goal[0] += dx
            goal[1] += dy
            goal[2] += dz

            goal[0] = float(np.clip(goal[0], PLACE_X_MIN, PLACE_X_MAX))
            goal[1] = float(np.clip(goal[1], PLACE_Y_MIN, PLACE_Y_MAX))
            goal[2] = float(np.clip(goal[2], 0.30, 0.50))

            rs.set_goal_marker(goal)
            mujoco.mj_forward(rs.model, rs.data)

            print(
                f"\r  GOAL: ({goal[0]:+.3f}, {goal[1]:+.3f}, {goal[2]:+.3f}) [ENTER to lock]    ",
                end="",
                flush=True,
            )

            keyboard.keys_held.clear()
            keyboard.place_nudge[:] = 0.0

        hold_robot_pose(rs)
        viewer.sync()
        time.sleep(0.05)

    print()
    print(f"\n  Goal locked at {np.round(goal, 3)}")

    keyboard.signals['lock_placement'] = False
    keyboard.in_placement = False
    keyboard.keys_held.clear()
    keyboard.place_nudge[:] = 0.0

    return goal


# ============================================================
# TELEOP DEMO COLLECTION
# ============================================================

def collect_demo(rs, scene_boxes, target_key, max_steps=3000):
    temp_goal = BOXES[target_key][1].copy()
    temp_goal[0] += 0.20
    setup_scene(rs, scene_boxes, temp_goal)

    keyboard = KeyboardState()

    print()
    print("=" * 70)
    print("TELEOPERATION DEMO")
    print("=" * 70)
    print(f"  Target:      {target_key}")
    print(f"  Scene boxes: {list(scene_boxes.keys())}")

    obs_list = []
    act_list = []
    rew_list = []
    done_list = []
    mode_list = []
    goal = None

    with mujoco.viewer.launch_passive(
        rs.model, rs.data, key_callback=keyboard.on_key,
    ) as viewer:
        print("\nMoving to home position...")
        move_to_home(rs, render_cb=viewer.sync, steps=200)
        time.sleep(0.5)

        # STEP 1: Target placement
        target_pos = place_target_interactive(rs, target_key, viewer, keyboard)
        if target_pos is None:
            print("\n  [QUIT during target placement]")
            return None

        scene_boxes[target_key] = target_pos.copy()
        hold_robot_pose(rs)
        viewer.sync()

        blocker_keys = [k for k in scene_boxes.keys() if k != target_key]

        if len(blocker_keys) > 1:
            print("\n  [WARNING] More than one blocker was provided.")
            print("  This collector is designed for one target + one blocker + one goal.")
            print(f"  Blockers provided: {blocker_keys}")

        for blocker_key in blocker_keys:
           blocker_pos = place_blocker_interactive(rs, blocker_key, viewer, keyboard)
           if blocker_pos is None:
               print("\n  [QUIT during blocker placement]")
               return None

           scene_boxes[blocker_key] = blocker_pos.copy()
           hold_robot_pose(rs)
           viewer.sync()

        # STEP 2: Goal placement
        goal = place_goal_interactive(rs, scene_boxes, target_key, viewer, keyboard)
        if goal is None:
            print("\n  [QUIT during goal placement]")
            return None

        hold_robot_pose(rs)
        viewer.sync()

        # STEP 3: Teleop
        keyboard.in_placement = False
        print_teleop_controls()
        print(f"Ready. Mode: {keyboard.mode}\n")

        step_count = 0
        start_time = time.time()

        while step_count < max_steps:
            if keyboard.signals['quit']:
                print("\n  [QUIT]")
                return None
            if keyboard.signals['cancel']:
                print("\n  [CANCEL]")
                return None
            # Save and end rollout.
            # SPACE -> successful demonstration.
            # H     -> failed attempt saved for Hindsight Experience Replay (HER).
            if keyboard.signals['success'] or keyboard.signals['her_save']:
                if keyboard.signals['success']:
                    print(f"\n  [SUCCESS] step {step_count}")
                else:
                    print(f"\n  [HER FAILED ATTEMPT] step {step_count}")

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
                    rs.set_robot_q(rs.data, q_interp)
                    viewer.sync()
                keyboard.signals['reset_orientation'] = False
                continue
            if keyboard.signals['pause']:
                hold_robot_pose(rs)
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

                if abs(dx) + abs(dy) + abs(dz) + abs(dw) > 0:
                    apply_cartesian(rs, dx, dy, dz, dw)
                    keyboard.keys_held.clear()
                else:
                    hold_robot_pose(rs)

            elif mode == MODE_ORIENTATION:
                droll, dpitch, dyaw = keyboard.orientation_delta()
                if abs(droll) + abs(dpitch) + abs(dyaw) > 0:
                    apply_orientation(rs, droll, dpitch, dyaw)
                    keyboard.keys_held.clear()
                else:
                    hold_robot_pose(rs)

            elif mode == MODE_JOINT:
                joint_idx, delta = keyboard.joint_delta()
                if abs(delta) > 0:
                    apply_joint(rs, joint_idx, delta)
                    keyboard.keys_held.clear()
                else:
                    hold_robot_pose(rs)

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
                print(
                    f"\r  step {step_count:4d} | {elapsed:5.1f}s | [{mode_str}{extra}] | "
                    f"box->goal {box_to_goal:.3f} | "
                    f"contact {'Y' if has_contact else '.'} | "
                    f"bad {'Y' if bad_contact else '.'} | "
                    f"dist {disturbance:.3f}    ",
                    end='',
                    flush=True,
                )

            viewer.sync()
            time.sleep(0.02)
            step_count += 1
        else:
            print(f"\n  [TIMEOUT at {max_steps} steps]")

    print(f"\n  Steps: {step_count}, Recorded obs: {len(obs_list)}")

    success = bool(keyboard.signals['success'])
    her_candidate = bool(keyboard.signals['her_save'])
    save_type = "success" if success else ("her_failed_attempt" if her_candidate else "unknown")

    final_box = rs.get_box_pos(rs.data, target_key)
    achieved_goal = final_box.copy()

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

            # Original goal is the green marker you intended to reach.
            "goal": goal.tolist(),
            "original_goal": goal.tolist(),

            # Achieved goal is where the target box ended up.
            # HER can relabel original_goal <- achieved_goal later.
            "achieved_goal": achieved_goal.tolist(),

            "success": success,
            "her_candidate": her_candidate,
            "save_type": save_type,

            "final_dist": final_dist,
            "disturbance": disturbance,
            "demo_length": len(obs_list),
            "collection_method": "keyboard_teleop_interactive_target_and_goal",
            "obs_dim": 27,
            "act_dim": 6,
            "act_format": "joint_position",
        },
    }


def next_her_filename(out_dir, filename):
    """
    Return a unique filename for HER failed attempts so pressing H multiple
    times across repeated runs never overwrites an older failed demo.

    Example:
        filename = scenario1_A4_demo017
        -> scenario1_A4_demo017_failHER_001
        -> scenario1_A4_demo017_failHER_002
    """
    os.makedirs(out_dir, exist_ok=True)

    if "her" in filename.lower():
        base = filename
    else:
        base = f"{filename}_failHER"

    idx = 1
    while True:
        candidate = f"{base}_{idx:03d}"
        pkl_path = os.path.join(out_dir, f"{candidate}.pkl")
        npz_path = os.path.join(out_dir, f"{candidate}.npz")
        if not os.path.exists(pkl_path) and not os.path.exists(npz_path):
            return candidate
        idx += 1


def save_demo(demo, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    pkl_path = os.path.join(out_dir, f"{filename}.pkl")
    npz_path = os.path.join(out_dir, f"{filename}.npz")

    with open(pkl_path, "wb") as f:
        pickle.dump(demo, f)

    np.savez_compressed(
        npz_path,
        observations=demo["observations"],
        actions=demo["actions"],
        rewards=demo["rewards"],
        dones=demo["dones"],
    )

    print("\nSaved:")
    print(f"  {pkl_path}")
    print(f"  {npz_path}")
    print(f"  obs:     {demo['observations'].shape}")
    print(f"  act:     {demo['actions'].shape} (6D joint positions)")
    print(f"  success: {demo['info']['success']}")
    print(f"  HER:     {demo['info'].get('her_candidate', False)}")
    print(f"  type:    {demo['info'].get('save_type', 'unknown')}")
    print(f"  dist:    {demo['info']['final_dist']:.4f}")
    print(f"  disturb: {demo['info']['disturbance']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True)
    parser.add_argument("--target", required=True, choices=list(BOXES.keys()))
    parser.add_argument("--blockers", nargs="*", default=[], choices=list(BOXES.keys()))
    parser.add_argument("--jitter_seed", type=int, default=None,
                        help="If set, randomize box positions slightly before manual placement")
    parser.add_argument("--out_dir", default="demonstrations_teleop",
                        help="Folder for successful/regular teleop demos")
    parser.add_argument("--her_out_dir", default="demonstrations_her",
                        help="Folder for failed attempts saved with H for HER")
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--name", required=True,
                        help="Filename for the saved demo, e.g. demo_001")
    args = parser.parse_args()

    rs = RobotScene(args.xml)
    print(f"\nTarget:   {args.target}")
    print(f"Blockers: {args.blockers if args.blockers else 'none'}")
    if args.jitter_seed is not None:
        print(f"Jitter:   seed={args.jitter_seed}")

    scene_boxes = setup_initial_scene(args.target, args.blockers, args.jitter_seed)
    demo = collect_demo(rs, scene_boxes, args.target, max_steps=args.max_steps)

    if demo is None:
        print("\nDemo canceled.")
        return

    # Save successful demos and HER failed attempts in different folders.
    # If H was pressed, automatically create a unique HER filename so repeated
    # failed attempts with the same --name are not overwritten.
    if demo["info"].get("her_candidate", False):
        her_name = next_her_filename(args.her_out_dir, args.name)
        print("\nSaving HER failed attempt to:", args.her_out_dir)
        print("HER filename:", her_name)
        save_demo(demo, args.her_out_dir, her_name)
    else:
        print("\nSaving successful/regular demo to:", args.out_dir)
        save_demo(demo, args.out_dir, args.name)


if __name__ == "__main__":
    main()
