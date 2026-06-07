"""
Teleop_collector_51d.py - 51D obs, no idle frame saves, no action cap.
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
    "A1":  ("amazon_box_small_A1",        np.array([-0.30, 0.45, 0.346]), np.array([0.080, 0.055, 0.040])),
    "A4":  ("amazon_box_medium_A4",       np.array([-0.10, 0.45, 0.365]), np.array([0.100, 0.065, 0.055])),
    "2BB": ("amazon_box_extra_large_2BB", np.array([+0.33, 0.45, 0.395]), np.array([0.130, 0.080, 0.085])),
    "1A9": ("amazon_box_medium_tall_1A9", np.array([-0.30, 0.45, 0.390]), np.array([0.100, 0.075, 0.080])),
    "B0":  ("amazon_box_large_B0",        np.array([+0.20, 0.45, 0.400]), np.array([0.120, 0.075, 0.090])),
}

PARK_POS = np.array([0.0, 0.0, -5.0])

MAX_BLOCKERS = 4
SENTINEL = np.array([5.0, 5.0, 5.0], dtype=np.float32)

# Take as many actions as you need.
MAX_SAVED_ACTIONS = 500
SIM_STEPS_PER_ACTION = 25

STEP_XY = 0.030
STEP_Z = 0.015
STEP_W = 0.080
STEP_ORIENT = 0.040
STEP_JOINT = 0.040
CLIP_XY = 0.040
CLIP_Z = 0.030
CLIP_W = 0.150

MODE_CARTESIAN = "CARTESIAN"
MODE_ORIENTATION = "ORIENTATION"
MODE_JOINT = "JOINT"
MODES_ORDER = [MODE_CARTESIAN, MODE_ORIENTATION, MODE_JOINT]


class RobotScene:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES]
        self.qpos_ids = [self.model.jnt_qposadr[jid] for jid in self.joint_ids]
        self.dof_ids = [self.model.jnt_dofadr[jid] for jid in self.joint_ids]
        self.act_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ACTUATOR_NAMES]
        self.tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pusher_tip_site")
        self.goal_marker_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal_marker")
        print(f"\nLoaded scene: nq={self.model.nq}, nu={self.model.nu}\n")

    def set_robot_q(self, data, q):
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
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, BOXES[box_key][0])

    def get_box_pos(self, data, box_key):
        return data.xpos[self.body_id(box_key)].copy()

    def set_goal_marker(self, goal):
        if self.goal_marker_id != -1:
            self.model.geom_pos[self.goal_marker_id, 0] = goal[0]
            self.model.geom_pos[self.goal_marker_id, 1] = goal[1]
            self.model.geom_pos[self.goal_marker_id, 2] = goal[2]
            mujoco.mj_forward(self.model, self.data)


def get_obs_51d(rs, data, target_key, goal, scene_boxes):
    model = rs.model
    box_id = rs.body_id(target_key)
    qpos = rs.get_robot_q(data).copy()
    qvel = np.array([data.qvel[dof] for dof in rs.dof_ids])
    tip = rs.get_tip(data)
    box = data.xpos[box_id].copy()
    vel6 = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, box_id, vel6, 0)
    box_linvel = vel6[3:].copy()
    base_27 = np.concatenate([qpos, qvel, tip, box, box_linvel, goal, goal - box], dtype=np.float32)

    blocker_keys = [k for k in scene_boxes.keys() if k != target_key]
    target_pos_2d = box[:2]
    blocker_distances = []
    for bk in blocker_keys:
        bp = rs.get_box_pos(data, bk)
        dist = float(np.linalg.norm(bp[:2] - target_pos_2d))
        blocker_distances.append((dist, bk, bp))
    blocker_distances.sort(key=lambda x: x[0])

    blocker_section = []
    for slot in range(MAX_BLOCKERS):
        if slot < len(blocker_distances):
            _, bk, bp = blocker_distances[slot]
            blocker_pos = bp.astype(np.float32)
            blocker_rel = (bp - box).astype(np.float32)
        else:
            blocker_pos = SENTINEL.copy()
            blocker_rel = SENTINEL.copy()
        blocker_section.append(blocker_pos)
        blocker_section.append(blocker_rel)
    blocker_array = np.concatenate(blocker_section, dtype=np.float32)
    return np.concatenate([base_27, blocker_array], dtype=np.float32)


def get_disturbance(rs, data, scene_boxes, target_key):
    total = 0.0
    for key, start_pos in scene_boxes.items():
        if key == target_key:
            continue
        bid = rs.body_id(key)
        if bid >= 0:
            total += float(np.linalg.norm(data.xpos[bid] - start_pos))
    return total


def make_scenario(scenario, seed):
    rng = np.random.default_rng(seed)
    target_key = "A1"
    n_blockers = {"N3": 2, "N4": 3, "N5": 4}[scenario]
    target_pos = BOXES[target_key][1].copy()
    target_pos[0] += float(rng.uniform(-0.03, 0.03))
    goal = target_pos.copy()
    goal[0] += float(rng.uniform(0.18, 0.25))
    scene_boxes = {target_key: target_pos}
    avail_keys = [k for k in BOXES.keys() if k != target_key]
    for i in range(n_blockers):
        bk = avail_keys[i]
        bp = BOXES[bk][1].copy()
        bp[0] = target_pos[0] + (i + 1) * 0.08 + float(rng.uniform(-0.02, 0.02))
        bp[1] = target_pos[1] + float(rng.uniform(-0.05, 0.05))
        scene_boxes[bk] = bp
    return scene_boxes, goal, target_key


def setup_scene(rs, scene_boxes, goal):
    model = rs.model
    data = rs.data
    mujoco.mj_resetData(model, data)
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


def move_to_home(rs, render_cb=None, steps=150):
    data = rs.data
    model = rs.model
    q_start = rs.get_robot_q(data)
    q_target = HOME_QPOS.copy()
    for k in range(steps):
        alpha = k / max(steps - 1, 1)
        alpha_smooth = 3 * alpha**2 - 2 * alpha**3
        q_cmd = (1.0 - alpha_smooth) * q_start + alpha_smooth * q_target
        rs.set_ctrl_q(data, q_cmd)
        mujoco.mj_step(model, data)
        if render_cb:
            render_cb()
    rs.set_ctrl_q(data, q_target)
    for _ in range(60):
        mujoco.mj_step(model, data)
        if render_cb:
            render_cb()


class KeyboardState:
    def __init__(self):
        self.cart_nudge = np.zeros(4, dtype=float)
        self.orient_nudge = np.zeros(3, dtype=float)
        self.joint_nudge = 0.0
        self.selected_joint = 0
        self.mode_index = 0
        self.signals = {
            'success': False, 'her_save': False, 'cancel': False,
            'quit': False, 'mode_switch': False, 'reset_orientation': False,
        }

    @property
    def mode(self):
        return MODES_ORDER[self.mode_index]

    def on_key(self, keycode):
        if keycode == 256:
            self.signals['quit'] = True
            return
        if keycode in (ord('D'), 262): self.cart_nudge[1] += STEP_XY
        if keycode in (ord('A'), 263): self.cart_nudge[1] -= STEP_XY
        if keycode in (ord('S'), 264): self.cart_nudge[0] -= STEP_XY
        if keycode in (ord('W'), 265): self.cart_nudge[0] += STEP_XY
        if keycode == ord('Q'): self.cart_nudge[2] += STEP_Z
        if keycode == ord('Z'): self.cart_nudge[2] -= STEP_Z
        if keycode == ord('R'): self.cart_nudge[3] += STEP_W
        if keycode == ord('F'): self.cart_nudge[3] -= STEP_W
        if keycode == ord('I'): self.orient_nudge[1] += STEP_ORIENT
        if keycode == ord('K'): self.orient_nudge[1] -= STEP_ORIENT
        if keycode == ord('J'): self.orient_nudge[2] += STEP_ORIENT
        if keycode == ord('L'): self.orient_nudge[2] -= STEP_ORIENT
        if keycode == ord('U'): self.orient_nudge[0] += STEP_ORIENT
        if keycode == ord('O'): self.orient_nudge[0] -= STEP_ORIENT
        if ord('1') <= keycode <= ord('6'):
            self.selected_joint = keycode - ord('1')
            print(f"\n  [JOINT] selected {self.selected_joint + 1}")
        if keycode == ord('N'): self.joint_nudge -= STEP_JOINT
        if keycode == ord('M'): self.joint_nudge += STEP_JOINT
        if keycode == 32: self.signals['success'] = True
        if keycode == ord('H'): self.signals['her_save'] = True
        if keycode == ord('X'): self.signals['cancel'] = True
        if keycode == 258:
            self.mode_index = (self.mode_index + 1) % len(MODES_ORDER)
            self.signals['mode_switch'] = True
            # Clear all nudge accumulators on mode switch
            self.cart_nudge[:] = 0.0
            self.orient_nudge[:] = 0.0
            self.joint_nudge = 0.0

    def cartesian_delta(self):
        dx = float(np.clip(self.cart_nudge[0], -STEP_XY, STEP_XY))
        dy = float(np.clip(self.cart_nudge[1], -STEP_XY, STEP_XY))
        dz = float(np.clip(self.cart_nudge[2], -STEP_Z, STEP_Z))
        dw = float(np.clip(self.cart_nudge[3], -STEP_W, STEP_W))
        self.cart_nudge[:] = 0.0
        return dx, dy, dz, dw

    def orientation_delta(self):
        droll = float(np.clip(self.orient_nudge[0], -STEP_ORIENT, STEP_ORIENT))
        dpitch = float(np.clip(self.orient_nudge[1], -STEP_ORIENT, STEP_ORIENT))
        dyaw = float(np.clip(self.orient_nudge[2], -STEP_ORIENT, STEP_ORIENT))
        self.orient_nudge[:] = 0.0
        return droll, dpitch, dyaw

    def joint_delta(self):
        dq = float(np.clip(self.joint_nudge, -STEP_JOINT, STEP_JOINT))
        self.joint_nudge = 0.0
        return self.selected_joint, dq

    def has_any_input(self):
        return (np.any(np.abs(self.cart_nudge) > 0) or
                np.any(np.abs(self.orient_nudge) > 0) or
                abs(self.joint_nudge) > 0)


def apply_cartesian(rs, dx, dy, dz, dw, sim_steps=SIM_STEPS_PER_ACTION):
    model = rs.model
    data = rs.data
    desired_tip = rs.get_tip(data).copy()
    desired_tip[0] += dx
    desired_tip[1] += dy
    desired_tip[2] += dz
    tip = rs.get_tip(data)
    err = desired_tip - tip
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, rs.tip_id)
    J = jacp[:, rs.dof_ids]
    dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err)
    q_target = np.array([data.ctrl[aid] for aid in rs.act_ids])
    q_target = q_target + 0.8 * dq
    q_target[5] += dw
    q_target = np.clip(q_target, -6.2, 6.2)
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
    q_target = np.array([data.ctrl[aid] for aid in rs.act_ids])
    q_target = q_target + 0.5 * dq
    q_target = np.clip(q_target, -6.2, 6.2)
    rs.set_ctrl_q(data, q_target)
    for _ in range(sim_steps):
        mujoco.mj_step(model, data)


def apply_joint(rs, joint_idx, delta, sim_steps=SIM_STEPS_PER_ACTION):
    model = rs.model
    data = rs.data
    q_target = np.array([data.ctrl[aid] for aid in rs.act_ids])
    q_target[joint_idx] += delta
    q_target = np.clip(q_target, -6.2, 6.2)
    rs.set_ctrl_q(data, q_target)
    for _ in range(sim_steps):
        mujoco.mj_step(model, data)


def hold_robot_pose(rs):
    mujoco.mj_step(rs.model, rs.data)


def collect_demo(rs, scene_boxes, goal, target_key, scenario, seed):
    setup_scene(rs, scene_boxes, goal)
    keyboard = KeyboardState()
    obs_list = []
    act_list = []
    rew_list = []
    done_list = []
    mode_list = []

    with mujoco.viewer.launch_passive(rs.model, rs.data, key_callback=keyboard.on_key) as viewer:
        print("\nMoving to home...")
        move_to_home(rs, render_cb=viewer.sync, steps=150)
        time.sleep(0.3)

        print()
        print("=" * 70)
        print(f"TELEOPERATION - {scenario}, seed {seed}")
        print("=" * 70)
        print(f"Target: {target_key} at {scene_boxes[target_key][:2].round(3)}")
        print(f"Goal:   {goal[:2].round(3)}")
        print(f"Blockers: {[k for k in scene_boxes.keys() if k != target_key]}")
        print()
        print("Controls:")
        print("  TAB: switch CARTESIAN / ORIENTATION / JOINT mode")
        print("  W/S/A/D: move arm forward/back/left/right (cartesian)")
        print("  Q/Z: arm up/down")
        print("  SPACE: save successful demo")
        print("  H: save failed attempt for HER")
        print("  X: cancel without saving")
        print("  ESC: quit")
        print()
        print("Take as many actions as needed. Idle frames are not saved.")
        print()

        action_count = 0

        while True:
            if keyboard.signals['quit']:
                print("\n  [QUIT]")
                return None
            if keyboard.signals['cancel']:
                print("\n  [CANCEL]")
                return None
            if keyboard.signals['success'] or keyboard.signals['her_save']:
                if keyboard.signals['success']:
                    print(f"\n  [SUCCESS] {action_count} actions")
                else:
                    print(f"\n  [HER FAILED ATTEMPT] {action_count} actions")
                obs = get_obs_51d(rs, rs.data, target_key, goal, scene_boxes)
                obs_list.append(obs)
                act_list.append(rs.get_robot_q(rs.data).astype(np.float32))
                rew_list.append(np.float32(0.0))
                done_list.append(np.float32(True))
                mode_list.append(keyboard.mode)
                break

            if keyboard.signals['mode_switch']:
                print(f"\n  [MODE] -> {keyboard.mode}")
                keyboard.signals['mode_switch'] = False

# Check if the CURRENT mode has actual input (mode-specific check)
            mode = keyboard.mode
            mode_has_input = False
            if mode == MODE_CARTESIAN:
                mode_has_input = np.any(np.abs(keyboard.cart_nudge) > 0)
            elif mode == MODE_ORIENTATION:
                mode_has_input = np.any(np.abs(keyboard.orient_nudge) > 0)
            elif mode == MODE_JOINT:
                mode_has_input = abs(keyboard.joint_nudge) > 0

            if mode_has_input:
                obs = get_obs_51d(rs, rs.data, target_key, goal, scene_boxes)
                obs_list.append(obs)

                if mode == MODE_CARTESIAN:
                    dx, dy, dz, dw = keyboard.cartesian_delta()
                    dx = float(np.clip(dx, -CLIP_XY, CLIP_XY))
                    dy = float(np.clip(dy, -CLIP_XY, CLIP_XY))
                    dz = float(np.clip(dz, -CLIP_Z, CLIP_Z))
                    dw = float(np.clip(dw, -CLIP_W, CLIP_W))
                    apply_cartesian(rs, dx, dy, dz, dw)
                elif mode == MODE_ORIENTATION:
                    droll, dpitch, dyaw = keyboard.orientation_delta()
                    apply_orientation(rs, droll, dpitch, dyaw)
                elif mode == MODE_JOINT:
                    joint_idx, delta = keyboard.joint_delta()
                    apply_joint(rs, joint_idx, delta)
                
                # Clear ALL nudge accumulators to prevent leftover from previous mode
                keyboard.cart_nudge[:] = 0.0
                keyboard.orient_nudge[:] = 0.0
                keyboard.joint_nudge = 0.0

                current_q = rs.get_robot_q(rs.data).astype(np.float32)
                act_list.append(current_q)
                rew_list.append(np.float32(0.0))
                done_list.append(np.float32(False))
                mode_list.append(mode)

                action_count += 1
                box = rs.get_box_pos(rs.data, target_key)
                box_to_goal = float(np.linalg.norm(box[:2] - goal[:2]))
                disturbance = get_disturbance(rs, rs.data, scene_boxes, target_key)
                print(f"  [ACT {action_count:3d}] [{mode[:4]}] box->goal {box_to_goal:.3f}m | disturb {disturbance:.3f}m")
            else:
                hold_robot_pose(rs)

            viewer.sync()
            time.sleep(0.03)

    success = bool(keyboard.signals['success'])
    her_candidate = bool(keyboard.signals['her_save'])
    save_type = "success" if success else ("her_failed_attempt" if her_candidate else "unknown")
    final_box = rs.get_box_pos(rs.data, target_key)
    final_dist = float(np.linalg.norm(final_box[:2] - goal[:2]))
    disturbance = get_disturbance(rs, rs.data, scene_boxes, target_key)
    print(f"\n  Saved {len(obs_list)} timesteps")
    print(f"  final_dist={final_dist:.3f}, disturbance={disturbance:.3f}")

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
            "original_goal": goal.tolist(),
            "achieved_goal": final_box.tolist(),
            "scenario": scenario,
            "seed": seed,
            "success": success,
            "her_candidate": her_candidate,
            "save_type": save_type,
            "final_dist": final_dist,
            "disturbance": disturbance,
            "demo_length": len(obs_list),
            "n_actions_saved": len(obs_list),
            "obs_dim": 51,
            "act_dim": 6,
            "act_format": "joint_position",
        },
    }


def save_demo(demo, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    pkl_path = os.path.join(out_dir, f"{filename}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(demo, f)
    print(f"\nSaved: {pkl_path}")
    print(f"  obs: {demo['observations'].shape}")
    print(f"  act: {demo['actions'].shape}")
    print(f"  success: {demo['info']['success']}, HER: {demo['info']['her_candidate']}")


def next_her_filename(out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    base = filename if "her" in filename.lower() else f"{filename}_failHER"
    idx = 1
    while True:
        candidate = f"{base}_{idx:03d}"
        if not os.path.exists(os.path.join(out_dir, f"{candidate}.pkl")):
            return candidate
        idx += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True)
    parser.add_argument("--scenario", required=True, choices=["N3", "N4", "N5"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out_dir", default="demonstrations_teleop_51d")
    parser.add_argument("--her_out_dir", default="demonstrations_her_51d")
    args = parser.parse_args()

    rs = RobotScene(args.xml)
    scene_boxes, goal, target_key = make_scenario(args.scenario, args.seed)
    print(f"\nScenario: {args.scenario}, Seed: {args.seed}")
    print(f"Target: {target_key} at {scene_boxes[target_key][:2].round(3)}")
    print(f"Goal: {goal[:2].round(3)}")

    demo = collect_demo(rs, scene_boxes, goal, target_key, args.scenario, args.seed)

    if demo is None:
        print("\nDemo canceled.")
        return

    if demo["info"].get("her_candidate", False):
        her_name = next_her_filename(args.her_out_dir, args.name)
        save_demo(demo, args.her_out_dir, her_name)
    else:
        save_demo(demo, args.out_dir, args.name)


if __name__ == "__main__":
    main()
