"""
demo_collector_mppi.py
======================

Staged Cartesian MPPI demo collector for non-prehensile box pushing.

Main idea:
  Phase 1:
    Automatically find/reach a good pre-contact pose near the box.
    This uses multistart Jacobian IK, not manual sliders.

  Phase 2:
    Use staged MPPI:
      Stage A: move pusher to correct contact face of box.
      Stage B: after contact/near-contact, push box toward final goal.

MPPI action:
  a_t = [dx, dy, dz, d_wrist3]

This is better than raw 6D joint-space MPPI because MPPI searches in
meaningful pusher-tip motion space, while Jacobian IK converts pusher
motion into UR5 joint commands.

Usage:
  python demo_collector_mppi.py --xml scene.xml --box A4 --scenario normal --render
  python demo_collector_mppi.py --xml scene.xml --box A4 --scenario normal
"""

import argparse
import os
import pickle
import time
import numpy as np

import mujoco
import mujoco.viewer

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class PushStage:
    """One stage of a multi-stage push plan."""
    target_box: str
    waypoint: np.ndarray
    push_dir: np.ndarray
    max_steps: int = 350
    success_thresh: float = 0.07
    name: str = "stage"

    def __post_init__(self):
        norm = np.linalg.norm(self.push_dir)
        if norm < 1e-8:
            raise ValueError("Push direction must be non-zero")
        self.push_dir = np.asarray(self.push_dir, dtype=float) / norm

@dataclass
class StagedPlan:
    """A multi-stage push plan with final goal for metrics."""
    stages: List[PushStage]
    final_goal: np.ndarray
    target_key: str
    scenario_desc: str = ""

# ============================================================
# CONSTANTS
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
    "A1": (
        "amazon_box_small_A1",
        np.array([-0.30, 0.60, 0.346]),
        np.array([0.080, 0.055, 0.040]),
    ),
    "A4": (
        "amazon_box_medium_A4",
        np.array([-0.10, 0.60, 0.365]),
        np.array([0.100, 0.065, 0.055]),
    ),
    "2BB": (
        "amazon_box_extra_large_2BB",
        np.array([0.33, 0.60, 0.395]),
        np.array([0.130, 0.080, 0.085]),
    ),
    "1A9": (
        "amazon_box_medium_tall_1A9",
        np.array([-0.30, 0.60, 0.806]),
        np.array([0.100, 0.075, 0.080]),
    ),
    "B0": (
        "amazon_box_large_B0",
        np.array([0.20, 0.60, 0.400]),
        np.array([0.120, 0.075, 0.090]),
    ),
}

UPPER_SHELF_KEYS = {"1A9", "B0"}

PARK_POS = np.array([0.0, 0.0, -5.0])

SHELF_LEFT_X = -0.73
SHELF_RIGHT_X = 0.73

PUSH_DIST = 0.18
PUSHER_DEPTH = 0.010


# ============================================================
# SCENARIO
# ============================================================

def build_scenario(scenario, target_key, direction="+x", seed=0):
    """Build a scenario. Returns (scene_boxes, staged_plan)."""
    _, init_pos, half = BOXES[target_key]

    if scenario == "normal":
        scene_boxes = {target_key: init_pos.copy()}
        dir_map = {
            "+x": np.array([1.0, 0.0, 0.0]),
            "-x": np.array([-1.0, 0.0, 0.0]),
            "+y": np.array([0.0, 1.0, 0.0]),
            "-y": np.array([0.0, -1.0, 0.0]),
        }
        push_vec = dir_map[direction]
        goal = init_pos.copy() + push_vec * PUSH_DIST

        stages = [PushStage(
            target_box=target_key,
            waypoint=goal,
            push_dir=push_vec,
            max_steps=400,
            success_thresh=0.07,
            name=f"normal_push_{direction}",
        )]
        return scene_boxes, StagedPlan(
            stages=stages,
            final_goal=goal,
            target_key=target_key,
            scenario_desc=f"Normal push {direction}",
        )

    elif scenario == "blocked":
        variant = seed % 4
        if variant == 0:
            return _blocked_case_move_blocker(+1)
        elif variant == 1:
            return _blocked_case_curved_bypass()
        elif variant == 2:
            return _blocked_case_push_and_realign()
        else:
            return _blocked_case_move_blocker(-1)

    else:
        raise ValueError(f"Unknown scenario: {scenario}")


def _blocked_case_move_blocker(blocker_dy_sign):
    """Case 0/3: move blocker out of the way, then push target."""
    target_key  = "A1"
    blocker_key = "A4"
    target_pos  = BOXES[target_key][1].copy()
    blocker_natural_z = BOXES[blocker_key][1][2]

    blocker_pos = np.array([
        target_pos[0] + 0.20,
        target_pos[1] + 0.06 * blocker_dy_sign,
        blocker_natural_z,
    ])

    
    scene_boxes = {
        target_key:  target_pos.copy(),
        blocker_key: blocker_pos.copy(),
    }

    final_goal = target_pos.copy()
    final_goal[0] += 0.43

    stage1_waypoint = blocker_pos.copy()
    stage1_waypoint[1] += 0.12 * blocker_dy_sign
    stage1 = PushStage(
        target_box=blocker_key,
        waypoint=stage1_waypoint,
        push_dir=np.array([0.0, 1.0 * blocker_dy_sign, 0.0]),
        max_steps=300,
        success_thresh=0.05,
        name=f"push_blocker_{'+y' if blocker_dy_sign > 0 else '-y'}",
    )

    stage2 = PushStage(
        target_box=target_key,
        waypoint=final_goal,
        push_dir=np.array([1.0, 0.0, 0.0]),
        max_steps=400,
        success_thresh=0.07,
        name="push_target_+x",
    )

    return scene_boxes, StagedPlan(
        stages=[stage1, stage2],
        final_goal=final_goal,
        target_key=target_key,
        scenario_desc=f"Move-blocker-first (blocker_dy_sign={blocker_dy_sign:+d})",
    )


def _blocked_case_curved_bypass():
    """Case 1: blocker spaced enough that curved bypass works in single stage."""
    target_key  = "A1"
    blocker_key = "A4"
    target_pos  = BOXES[target_key][1].copy()
    blocker_natural_z = BOXES[blocker_key][1][2]

    blocker_pos = np.array([
        target_pos[0] + 0.25,
        target_pos[1] + 0.06,
        blocker_natural_z,
    ])
    
    scene_boxes = {
        target_key:  target_pos.copy(),
        blocker_key: blocker_pos.copy(),
    }

    final_goal = target_pos.copy()
    final_goal[0] += 0.45
    final_goal[1] -= 0.05

    push_dir = final_goal - target_pos

    stage1 = PushStage(
        target_box=target_key,
        waypoint=final_goal,
        push_dir=push_dir,
        max_steps=500,
        success_thresh=0.08,
        name="curved_push",
    )

    return scene_boxes, StagedPlan(
        stages=[stage1],
        final_goal=final_goal,
        target_key=target_key,
        scenario_desc="Curved bypass (single stage)",
    )


def _blocked_case_push_and_realign():
    """Case 2: push +x past blocker, then realign -y."""
    target_key  = "A4"
    blocker_key = "2BB"
    target_pos  = BOXES[target_key][1].copy()
    blocker_pos = BOXES[blocker_key][1].copy()

    scene_boxes = {
        target_key:  target_pos.copy(),
        blocker_key: blocker_pos.copy(),
    }

    final_goal = target_pos.copy()
    final_goal[0] += 0.55
    final_goal[1] -= 0.06

    stage1_waypoint = target_pos.copy()
    stage1_waypoint[0] += 0.55
    stage1 = PushStage(
        target_box=target_key,
        waypoint=stage1_waypoint,
        push_dir=np.array([1.0, 0.0, 0.0]),
        max_steps=500,
        success_thresh=0.07,
        name="push_+x_past_blocker",
    )

    stage2 = PushStage(
        target_box=target_key,
        waypoint=final_goal,
        push_dir=np.array([0.0, -1.0, 0.0]),
        max_steps=200,
        success_thresh=0.05,
        name="realign_-y",
    )

    return scene_boxes, StagedPlan(
        stages=[stage1, stage2],
        final_goal=final_goal,
        target_key=target_key,
        scenario_desc="Push-and-realign (multi-direction)",
    )

# ============================================================
# PUSH PLAN
# ============================================================

class PushPlan:
    def __init__(self, box_pos, goal, half_sizes, push_dir=None):
        self.goal = goal.copy()
        self.half_sizes = half_sizes.copy()
        self.explicit_push_dir = push_dir
        self.update(box_pos)

    def update(self, box_pos):
        self.box_pos = box_pos.copy()

        if self.explicit_push_dir is not None:
            self.push_dir = self.explicit_push_dir / (np.linalg.norm(self.explicit_push_dir) + 1e-9)
        else:
            delta = self.goal - box_pos
            delta[2] = 0.0
            norm = np.linalg.norm(delta[:2])
            if norm < 1e-8:
                self.push_dir = np.array([1.0, 0.0, 0.0])
            else:
                self.push_dir = delta / norm

        self.anti_dir = -self.push_dir

        self.face_dist = (
            abs(self.push_dir[0]) * self.half_sizes[0]
            + abs(self.push_dir[1]) * self.half_sizes[1]
        )

        self.contact_pt = box_pos.copy()
        self.contact_pt += self.anti_dir * (self.face_dist + PUSHER_DEPTH + 0.005)
        self.contact_pt[2] = box_pos[2] + 0.020

        self.approach_pt = box_pos.copy()
        self.approach_pt += self.anti_dir * (self.face_dist + PUSHER_DEPTH + 0.045)
        self.approach_pt[2] = box_pos[2] + 0.020

        self.contact_thresh = self.face_dist + PUSHER_DEPTH + 0.018

    def print(self):
        print(f"    box_pos      : {np.round(self.box_pos, 3)}")
        print(f"    goal         : {np.round(self.goal, 3)}")
        print(f"    push_dir     : [{self.push_dir[0]:+.3f}, {self.push_dir[1]:+.3f}]")
        print(f"    approach_pt  : {np.round(self.approach_pt, 3)}")
        print(f"    contact_pt   : {np.round(self.contact_pt, 3)}")
        print(f"    contact_thresh: {self.contact_thresh:.4f} m")


# ============================================================
# MODEL HELPERS
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

    def set_robot_q(self, data, q):
        for i, qid in enumerate(self.qpos_ids):
            data.qpos[qid] = q[i]
        for i, aid in enumerate(self.act_ids):
            data.ctrl[aid] = q[i]
        mujoco.mj_forward(self.model, data)

    def get_robot_q(self, data):
        return np.array([data.qpos[qid] for qid in self.qpos_ids])

    def set_ctrl_q(self, data, q):
        for i, aid in enumerate(self.act_ids):
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


# ============================================================
# SCENE SETUP
# ============================================================

def setup_scene(rs, scene_boxes, goal):
    model = rs.model
    data = rs.data

    mujoco.mj_resetData(model, data)

    data.qpos[:6] = HOME_QPOS
    data.ctrl[:6] = HOME_QPOS

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
    mujoco.mj_forward(model, data)

    for _ in range(100):
        mujoco.mj_step(model, data)


# ============================================================
# OBSERVATION
# ============================================================

def get_obs(rs, data, target_key, goal):
    model = rs.model

    box_name = BOXES[target_key][0]
    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, box_name)

    qpos = data.qpos[:6].copy()
    qvel = data.qvel[:6].copy()
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


# ============================================================
# CONTACT CHECK
# ============================================================

def contact_info(rs, data, target_key):
    target_geom = BOXES[target_key][0] + "_geom"

    has_target_contact = False
    has_bad_contact = False
    pairs = []

    for i in range(data.ncon):
        c = data.contact[i]

        g1 = mujoco.mj_id2name(rs.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
        g2 = mujoco.mj_id2name(rs.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""

        pair = f"{g1} {g2}"
        pairs.append(pair)

        if ("pusher" in pair or "ee_pusher" in pair) and target_geom in pair:
            has_target_contact = True

        if ("pusher" in pair or "wrist" in pair or "forearm" in pair or "upperarm" in pair) and (
            "shelf" in pair or "beam" in pair or "wall" in pair or "floor" in pair
        ):
            has_bad_contact = True

    return has_target_contact, has_bad_contact, pairs


# ============================================================
# JACOBIAN IK
# ============================================================

def solve_ik(rs, data, target_pos, q_init, max_iter=250, step_size=0.40, damping=1e-3):
    q = q_init.copy()

    for _ in range(max_iter):
        rs.set_robot_q(data, q)

        tip = rs.get_tip(data)
        err = target_pos - tip

        if np.linalg.norm(err) < 0.005:
            break

        jacp = np.zeros((3, rs.model.nv))
        jacr = np.zeros((3, rs.model.nv))

        mujoco.mj_jacSite(rs.model, data, jacp, jacr, rs.tip_id)

        J = jacp[:, rs.dof_ids]
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), err)

        q = q + step_size * dq
        q = np.clip(q, -6.2, 6.2)

    rs.set_robot_q(data, q)

    tip = rs.get_tip(data)
    err = np.linalg.norm(target_pos - tip)

    return q, tip, err


def auto_find_seed(rs, target_key, target_pos, trials=350, seed=0):
    """
    Automatically find a seed configuration near the desired approach/contact point.
    This replaces manual find_configs.py.
    """
    rng = np.random.default_rng(seed)

    base_seeds = [
        HOME_QPOS,
        np.array([-1.4694, -1.4618, 1.6743, -1.5564, -2.2838, 0.0]),  # good A4 front-ish seed
        np.array([0.0, -1.40, 1.60, -1.75, -1.5708, 0.0]),
        np.array([0.0, -1.25, 1.45, -1.80, -1.5708, 0.0]),
        np.array([0.0, -1.55, 1.80, -1.90, -1.5708, 0.0]),
    ]

    best = None

    for trial in range(trials):
        d = mujoco.MjData(rs.model)
        mujoco.mj_resetData(rs.model, d)

        if trial < len(base_seeds):
            q0 = base_seeds[trial].copy()
        else:
            center = base_seeds[1]
            noise = np.array([
                rng.normal(0.0, 0.65),
                rng.normal(0.0, 0.45),
                rng.normal(0.0, 0.45),
                rng.normal(0.0, 0.45),
                rng.normal(0.0, 0.35),
                rng.normal(0.0, 0.65),
            ])
            q0 = center + noise

        q0 = np.clip(q0, -6.2, 6.2)

        q_sol, tip, err = solve_ik(rs, d, target_pos, q0)

        _, bad_contact, _ = contact_info(rs, d, target_key)
        bad_pen = 10.0 if bad_contact else 0.0

        # Prefer wrist_2/wrist_3 values not too crazy, but do not force them.
        wrist_pen = 0.03 * abs(q_sol[4] + 1.5708) + 0.02 * abs(q_sol[5])

        score = err + bad_pen + wrist_pen

        if best is None or score < best["score"]:
            best = {
                "score": score,
                "q": q_sol.copy(),
                "tip": tip.copy(),
                "err": err,
                "bad": bad_contact,
            }

            if err < 0.012 and not bad_contact:
                break

    print("  [auto seed] target:", np.round(target_pos, 4))
    print("  [auto seed] tip   :", np.round(best["tip"], 4))
    print("  [auto seed] err   :", best["err"])
    print("  [auto seed] q     :", np.round(best["q"], 4).tolist())

    return best["q"]

def move_to_q_smooth(rs, q_target, render_cb=None, steps=300, sleep_steps=True):
    """
    Visibly move robot from current joint configuration to q_target.
    This replaces teleporting to the seed configuration.
    """
    data = rs.data
    model = rs.model

    q_start = data.qpos[:6].copy()

    for k in range(steps):
        alpha = k / max(steps - 1, 1)

        # Smooth interpolation: slow start and slow end
        alpha_smooth = 3 * alpha**2 - 2 * alpha**3

        q_cmd = (1.0 - alpha_smooth) * q_start + alpha_smooth * q_target

        data.ctrl[:6] = q_cmd
        mujoco.mj_step(model, data)

        if render_cb:
            render_cb()

# ============================================================
# PHASE 1: AUTO SEEDED IK APPROACH
# ============================================================

def phase1_auto_ik(rs, target_key, plan, render_cb=None, seed=0):
    data = rs.data
    model = rs.model

    box_name = BOXES[target_key][0]
    obs_list = []
    act_list = []

    print("=" * 50)
    print("[Phase 1] Auto-seeded IK approach")
    print("=" * 50)

    # First target: approach point.
    seed_q = auto_find_seed(rs, target_key, plan.approach_pt, trials=350, seed=seed)

    # Move current sim to seed.
   # data.qpos[:6] = seed_q.copy()
    #data.ctrl[:6] = seed_q.copy()
    #mujoco.mj_forward(model, data)
     # Move from home/current pose to seed smoothly.
    print("  Moving from home to seed configuration smoothly...")
    move_to_q_smooth(rs, seed_q, render_cb=render_cb, steps=350)

    # Now fine-tune to approach point and then contact point.
    targets = [
        ("approach", plan.approach_pt, 0.010, 300, 2.5),
        ("contact", plan.contact_pt, 0.006, 300, 1.5),
    ]

    for tag, target, tol, max_steps, gain in targets:
        print(f"  [{tag}] target: {np.round(target, 4)}")

        for step in range(max_steps):
            obs_list.append(get_obs(rs, data, target_key, plan.goal))

            tip = rs.get_tip(data)
            err = target - tip
            dist = np.linalg.norm(err)

            if dist < tol:
                print(f"  [{tag}] converged in {step} steps, err={dist:.4f}")
                break

            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, rs.tip_id)

            J = jacp[:, rs.dof_ids]
            dq = J.T @ np.linalg.solve(J @ J.T + 0.01 * np.eye(3), err)

            ctrl = np.clip(data.qpos[:6] + gain * dq, -6.2, 6.2)

            act_list.append(ctrl.astype(np.float32))

            data.ctrl[:6] = ctrl
            mujoco.mj_step(model, data)

            if render_cb:
                render_cb()
        else:
            print(f"  [{tag}] WARNING: did not converge, final err={dist:.4f}")

    box_now = rs.get_box_pos(data, target_key)
    tip_now = rs.get_tip(data)
    plan.update(box_now)

    tip_to_contact = np.linalg.norm(tip_now[:2] - plan.contact_pt[:2])
    tip_to_box = np.linalg.norm(tip_now - box_now)

    print("\n  [Phase1 done]")
    print("    tip        :", np.round(tip_now, 4))
    print("    box        :", np.round(box_now, 4))
    print("    contact_pt :", np.round(plan.contact_pt, 4))
    print("    tip-contact xy:", tip_to_contact)
    print("    tip-box    :", tip_to_box)

    return obs_list, act_list


# ============================================================
# PHASE 2: STAGED CARTESIAN MPPI
# ============================================================

class CartesianMPPI:
    def __init__(
        self,
        rs,
        target_key,
        scene_boxes,
        plan,
        horizon=14,
        n_samples=96,
        temperature=0.08,
        noise_sigma=np.array([0.008, 0.008, 0.004, 0.020]),
        seed=0,
    ):
        self.rs = rs
        self.target_key = target_key
        self.scene_boxes = scene_boxes
        self.plan = plan

        self.horizon = horizon
        self.n_samples = n_samples
        self.temperature = temperature
        self.noise_sigma = noise_sigma.astype(np.float64)

        self.rng = np.random.default_rng(seed)

        # Action: dx, dy, dz, d_wrist3
        self.U = np.zeros((self.horizon, 4), dtype=np.float64)

    def snapshot(self, data):
        return {
            "qpos": data.qpos.copy(),
            "qvel": data.qvel.copy(),
            "ctrl": data.ctrl.copy(),
            "time": float(data.time),
        }

    def restore(self, data, snap):
        data.qpos[:] = snap["qpos"]
        data.qvel[:] = snap["qvel"]
        data.ctrl[:] = snap["ctrl"]
        data.time = snap["time"]
        mujoco.mj_forward(self.rs.model, data)

    def nominal_action(self, data):
        tip = self.rs.get_tip(data)
        box = self.rs.get_box_pos(data, self.target_key)

        local_plan = PushPlan(box, self.plan.goal, BOXES[self.target_key][2], push_dir=self.plan.explicit_push_dir)
        has_contact, _, _ = contact_info(self.rs, data, self.target_key)

        dist_contact = np.linalg.norm(tip[:2] - local_plan.contact_pt[:2])

        if (not has_contact) and dist_contact > 0.025:
            # Stage A: move toward correct contact point.
            delta = local_plan.contact_pt - tip
            norm = np.linalg.norm(delta) + 1e-8
            direction = delta / norm

            nominal = np.array([
                0.010 * direction[0],
                0.010 * direction[1],
                0.004 * direction[2],
                0.0,
            ])
        else:
            # Stage B: push along box-to-goal direction.
            nominal = np.array([
                0.007 * local_plan.push_dir[0],
                0.007 * local_plan.push_dir[1],
                0.0,
                0.0,
            ])

        return nominal

    def ik_step_for_data(self, data, desired_tip, d_wrist3):
        tip = self.rs.get_tip(data)
        err = desired_tip - tip

        jacp = np.zeros((3, self.rs.model.nv))
        jacr = np.zeros((3, self.rs.model.nv))
        mujoco.mj_jacSite(self.rs.model, data, jacp, jacr, self.rs.tip_id)

        J = jacp[:, self.rs.dof_ids]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err)

        q = self.rs.get_robot_q(data)
        q_new = q + 0.35 * dq

        # Let MPPI slightly change wrist_3 to adjust pusher orientation.
        q_new[5] += d_wrist3

        q_new = np.clip(q_new, -6.2, 6.2)
        return q_new

    def apply_action(self, data, action, sim_steps=10):
        dx = np.clip(action[0], -0.015, 0.015)
        dy = np.clip(action[1], -0.015, 0.015)
        dz = np.clip(action[2], -0.006, 0.006)
        dw = np.clip(action[3], -0.050, 0.050)

        #desired_tip = self.rs.get_tip(data).copy()
        #desired_tip += np.array([dx, dy, dz])

        # Keep tip around shelf/box height.
        #box = self.rs.get_box_pos(data, self.target_key)
        #desired_tip[2] = np.clip(desired_tip[2], box[2] - 0.015, box[2] + 0.055)
        
        
        box = self.rs.get_box_pos(data, self.target_key)
        desired_tip = self.rs.get_tip(data).copy()
        desired_tip[0] += dx
        desired_tip[1] += dy
        desired_tip[2] = box[2] + 0.020   # servo Z, don't integrate

        for _ in range(sim_steps):
            q_target = self.ik_step_for_data(data, desired_tip, dw)
            self.rs.set_ctrl_q(data, q_target)
            mujoco.mj_step(self.rs.model, data)

    def rollout_cost(self, action_seq, snap):
        data = mujoco.MjData(self.rs.model)
        self.restore(data, snap)

        total = 0.0
        prev_a = np.zeros(4)
        had_contact_ever = False
        
        for t, a in enumerate(action_seq):
            self.apply_action(data, a, sim_steps=8)

            tip = self.rs.get_tip(data)
            box = self.rs.get_box_pos(data, self.target_key)

            local_plan = PushPlan(box, self.plan.goal, BOXES[self.target_key][2], push_dir=self.plan.explicit_push_dir)

            dist_goal = np.linalg.norm(box[:2] - self.plan.goal[:2])
            dist_contact = np.linalg.norm(tip[:2] - local_plan.contact_pt[:2])

            has_contact, bad_contact, _ = contact_info(self.rs, data, self.target_key)


        # NEW early-terminate on bad contact with terminal penalty
            if bad_contact:
                remaining = len(action_seq) - t
                total += 5000.0 * remaining          # terminal-ish penalty
                total += 500.0 * dist_goal ** 2      # still account for where box is
                return total

        # NEW contact hysteresis — punish losing contact once gained
            if has_contact:
                had_contact_ever = True
            detach_pen = 200.0 if (had_contact_ever and not has_contact) else 0.0


            if has_contact or dist_contact < 0.025:
                # Push stage.
                w_contact = 30.0
                w_goal = 220.0
                contact_reward = -15.0 if has_contact else 0.0
            else:
                # Contact-seeking stage.
                w_contact = 180.0
                w_goal = 15.0
                contact_reward = -5.0 if has_contact else 0.0

            action_cost = np.sum(a ** 2)
            smooth_cost = np.sum((a - prev_a) ** 2)

            #bad_pen = 100.0 if bad_contact else 0.0
            height_cost = (tip[2] - local_plan.contact_pt[2]) ** 2
  
            non_target_disturbance = 0.0
            for key, start_pos in self.scene_boxes.items():
                if key == self.target_key:
                    continue
                bid = self.rs.body_id(key)
                if bid >= 0:
                    cur = data.xpos[bid].copy()
                    non_target_disturbance += np.linalg.norm(cur[:2] - start_pos[:2])

  
            total += (
                w_contact * dist_contact ** 2
                + w_goal * dist_goal ** 2
                + 0.20 * action_cost
                + 0.50 * smooth_cost
                + 40.0 * height_cost
                #+ bad_pen
                + detach_pen
                + contact_reward
                + 800.0 * non_target_disturbance 
            )

            prev_a = a.copy()

        # Terminal costs.
        final_box = self.rs.get_box_pos(data, self.target_key)
        final_tip = self.rs.get_tip(data)
        final_plan = PushPlan(final_box, self.plan.goal, BOXES[self.target_key][2], push_dir=self.plan.explicit_push_dir)

        final_goal_dist = np.linalg.norm(final_box[:2] - self.plan.goal[:2])
        final_contact_dist = np.linalg.norm(final_tip[:2] - final_plan.contact_pt[:2])

        total += 300.0 * final_goal_dist ** 2
        total += 100.0 * final_contact_dist ** 2

        return total

    def act(self, data):
        snap = self.snapshot(data)

        nominal = self.nominal_action(data)

        for h in range(self.horizon):
            self.U[h] = 0.75 * self.U[h] + 0.25 * nominal

        noise = self.rng.normal(
            0.0,
            self.noise_sigma,
            size=(self.n_samples, self.horizon, 4),
        )

        samples = self.U[None, :, :] + noise

        samples[:, :, 0] = np.clip(samples[:, :, 0], -0.015, 0.015)
        samples[:, :, 1] = np.clip(samples[:, :, 1], -0.015, 0.015)
        samples[:, :, 2] = np.clip(samples[:, :, 2], -0.006, 0.006)
        samples[:, :, 3] = np.clip(samples[:, :, 3], -0.050, 0.050)

        costs = np.zeros(self.n_samples)

        for i in range(self.n_samples):
            costs[i] = self.rollout_cost(samples[i], snap)

        beta = np.min(costs)
        weights = np.exp(-(costs - beta) / self.temperature)
        weights /= np.sum(weights) + 1e-8

        self.U = np.sum(weights[:, None, None] * samples, axis=0)

        action = self.U[0].copy()
        best_cost = float(np.min(costs))

        # Shift warm start.
        self.U[:-1] = self.U[1:]
        self.U[-1] = self.U[-2]

        return action, best_cost


def phase2_mppi_push(
    rs,
    target_key,
    scene_boxes,
    plan,
    render_cb=None,
    max_steps=220,
    horizon=14,
    n_samples=96,
    seed=0,
    success_thresh=0.07,
):
    data = rs.data

    mppi = CartesianMPPI(
        rs,
        target_key,
        scene_boxes,
        plan,
        horizon=horizon,
        n_samples=n_samples,
        temperature=0.08,
        seed=seed,
    )

    obs_list = []
    act_list = []
    rew_list = []
    done_list = []

    success = False
    final_dist = float("inf")
    
    bad_streak = 0
    BAD_STREAK_LIMIT = 5


    print("=" * 50)
    print("[Phase 2] Staged Cartesian MPPI push")
    print("=" * 50)

    for step in range(max_steps):
        obs = get_obs(rs, data, target_key, plan.goal)

        box = rs.get_box_pos(data, target_key)
        tip = rs.get_tip(data)

        plan.update(box)

        final_dist = float(np.linalg.norm(box[:2] - plan.goal[:2]))
        done = final_dist < success_thresh

        if done:
            success = True
            obs_list.append(obs)
            act_list.append(np.zeros(4, dtype=np.float32))
            rew_list.append(np.float32(-final_dist))
            done_list.append(np.float32(True))
            print(f"  [MPPI] SUCCESS in {step} steps, box→goal={final_dist:.4f}")
            break

        action, cost = mppi.act(data)

        obs_list.append(obs)
        act_list.append(action.astype(np.float32))
        rew_list.append(np.float32(-final_dist))
        done_list.append(np.float32(False))

        mppi.apply_action(data, action, sim_steps=10)

        has_contact, bad_contact, _ = contact_info(rs, data, target_key)

        if render_cb:
            render_cb()

        if step % 10 == 0:
            print(
                f"  step {step:4d} | cost {cost:8.3f} | "
                f"box→goal {final_dist:.4f} | "
                f"box {np.round(box, 3)} | tip {np.round(tip, 3)} | "
                f"action {np.round(action, 4)} | "
                f"contact {has_contact} | bad {bad_contact}"
            )

        if bad_contact:
            bad_streak += 1
        else:
            bad_streak = 0

        if bad_streak >= BAD_STREAK_LIMIT:
            print(f"  [MPPI] EARLY STOP at step {step}: bad contact persistent "
                  f"({BAD_STREAK_LIMIT} consecutive steps). "
                  f"final box→goal={final_dist:.4f}")
            break

        if step % 10 == 0:
            print(
                f"  step {step:4d} | cost {cost:8.3f} | "
                f"box→goal {final_dist:.4f} | "
                f"box {np.round(box, 3)} | tip {np.round(tip, 3)} | "
                f"action {np.round(action, 4)} | "
                f"contact {has_contact} | bad {bad_contact}"
            )

    else:
        print(f"  [MPPI] TIMEOUT, final box→goal={final_dist:.4f}")

    disturbance = get_disturbance(rs, data, scene_boxes, target_key)

    return obs_list, act_list, rew_list, done_list, success, disturbance
    
    
def execute_staged_plan(
    rs,
    staged_plan,
    scene_boxes,
    render_cb=None,
    horizon=16,
    n_samples=192,
    seed=0,
):
    """Execute a multi-stage push plan, calling Phase 1 + Phase 2 for each stage."""

    print("=" * 60)
    print(f"STAGED PLAN: {staged_plan.scenario_desc}")
    print(f"  {len(staged_plan.stages)} stages")
    for i, stage in enumerate(staged_plan.stages):
        print(f"    Stage {i+1}: push {stage.target_box} to {np.round(stage.waypoint, 3)} "
              f"along {np.round(stage.push_dir, 3)}")
    print("=" * 60)

    all_obs = []
    all_acts = []
    all_rewards = []
    all_dones = []

    overall_success = True

    for i, stage in enumerate(staged_plan.stages):
        print()
        print("#" * 60)
        print(f"# Stage {i+1}/{len(staged_plan.stages)}: {stage.name}")
        print(f"#   Pushing {stage.target_box} to {np.round(stage.waypoint, 3)}")
        print("#" * 60)

        stage_target_pos = rs.get_box_pos(rs.data, stage.target_box)
        stage_half = BOXES[stage.target_box][2]
        stage_push_plan = PushPlan(
            box_pos=stage_target_pos,
            goal=stage.waypoint,
            half_sizes=stage_half,
            push_dir=stage.push_dir,
        )

        o1, a1 = phase1_auto_ik(
            rs, stage.target_box, stage_push_plan,
            render_cb=render_cb,
            seed=seed + i * 13,    # different seed per stage
        )
        all_obs += o1
        all_acts += [np.zeros(4, dtype=np.float32) for _ in o1]
        all_rewards += [np.float32(0.0)] * len(o1)
        all_dones += [np.float32(0.0)] * len(o1)

        o2, a2, r2, d2, stage_success, _ = phase2_mppi_push(
            rs, stage.target_box, scene_boxes, stage_push_plan,
            render_cb=render_cb,
            max_steps=stage.max_steps,
            horizon=horizon,
            n_samples=n_samples,
            seed=seed + i * 7,
            success_thresh=stage.success_thresh,
        )
        all_obs += o2
        all_acts += a2
        all_rewards += r2
        all_dones += d2

        if not stage_success:
            print(f"  [Stage {i+1}] FAILED. Aborting remaining stages.")
            overall_success = False
            break
        else:
            print(f"  [Stage {i+1}] SUCCESS.")
            # Return arm to home between stages for clean re-approach in next stage
            if i < len(staged_plan.stages) - 1:
                print(f"  [Inter-stage] Returning arm to home before next stage...")
                move_to_q_smooth(rs, HOME_QPOS, render_cb=render_cb, steps=200)

    final_target_pos = rs.get_box_pos(rs.data, staged_plan.target_key)
    final_dist = float(np.linalg.norm(final_target_pos[:2] - staged_plan.final_goal[:2]))
    disturbance = get_disturbance(rs, rs.data, scene_boxes, staged_plan.target_key)
    overall_success = overall_success and (final_dist < 0.10)

    print()
    print("=" * 60)
    print(f"STAGED PLAN COMPLETE")
    print(f"  Overall success: {overall_success}")
    print(f"  Final target→goal: {final_dist:.4f}")
    print(f"  Disturbance: {disturbance:.4f}")
    print("=" * 60)

    return all_obs, all_acts, all_rewards, all_dones, overall_success, disturbance, final_dist


# ============================================================
# SAVE
# ============================================================

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

    print(f"\nSaved:")
    print(f"  {pkl_path}")
    print(f"  {npz_path}")
    print(f"  obs: {demo['observations'].shape}")
    print(f"  act: {demo['actions'].shape}")
    print(f"  success: {demo['info']['success']}")
    print(f"  disturbance: {demo['info']['disturbance']:.4f}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True)
    parser.add_argument("--box", required=True, choices=list(BOXES.keys()))
    parser.add_argument(
        "--scenario",
        required=True,
        choices=["normal", "blocked", "cluttered", "wedged"],
    )
    parser.add_argument("--out_dir", default="demonstrations")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--max_steps", type=int, default=220)
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--n_samples", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--direction",
        default="+x",
        choices=["+x", "-x", "+y", "-y"],
        help="Push direction for normal scenario. Ignored for blocked.",
    )  
  
    args = parser.parse_args()

    rs = RobotScene(args.xml)
    print(f"\nScenario : {args.scenario.upper()}")
    print(f"Box      : {args.box}")
    print(f"Seed     : {args.seed}")
   

    def run(render_cb=None):
        scene_boxes, staged_plan = build_scenario(
            args.scenario, args.box, args.direction, args.seed
        )

        setup_scene(rs, scene_boxes, staged_plan.final_goal)
        rs.set_goal_marker(staged_plan.final_goal)

        all_obs, all_acts, all_rewards, all_dones, success, disturbance, final_dist = \
            execute_staged_plan(
                rs, staged_plan, scene_boxes,
                render_cb=render_cb,
                horizon=args.horizon,
                n_samples=args.n_samples,
                seed=args.seed,
            )

        demo = {
            "observations": np.stack(all_obs),
            "actions": np.stack(all_acts),
            "rewards": np.array(all_rewards, dtype=np.float32),
            "dones": np.array(all_dones, dtype=np.float32),
            "info": {
                "scenario": args.scenario,
                "box": args.box,
                "seed": args.seed,
                "scenario_desc": staged_plan.scenario_desc,
                "n_stages": len(staged_plan.stages),
                "scene_boxes": {k: v.tolist() for k, v in scene_boxes.items()},
                "final_goal": staged_plan.final_goal.tolist(),
                "success": success,
                "final_dist": final_dist,
                "disturbance": disturbance,
                "note": "Multi-stage staged plan: Phase 1 IK + Phase 2 MPPI per stage.",
            },
        }

        filename = f"{args.scenario}_{args.box}_seed{args.seed}_mppi"
        save_demo(demo, args.out_dir, filename)


    if args.render:
        with mujoco.viewer.launch_passive(rs.model, rs.data) as viewer:
            def sync():
                viewer.sync()
                time.sleep(0.001)

            run(render_cb=sync)
    else:
        run(render_cb=None)


if __name__ == "__main__":
    main()
