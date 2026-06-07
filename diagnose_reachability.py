"""
Single-Box MPPI Pushing Demo — Demonstration Collector
=======================================================

Calibrated to ur5e.xml + scene.xml:
  - Box positions and half-sizes taken directly from XML
  - Goal = push box forward along +Y (into rack back wall) by 0.18 m
  - Unified MPPI: one controller handles approach AND push via sigmoid gate
  - No separate IK phase — MPPI plans everything from home qpos

Observation (27-dim):
  [0:6]   joint positions
  [6:12]  joint velocities
  [12:15] pusher tip position (world)
  [15:18] box position (world)
  [18:21] box linear velocity
  [21:24] goal position
  [24:27] box → goal vector

Usage:
  python mpc_push_demo.py --xml scene.xml --box A1 --goal_dir right
  python mpc_push_demo.py --xml scene.xml --box A4 --goal_dir left --render
  python mpc_push_demo.py --xml scene.xml --box 2BB --goal_dir right --render
"""

import argparse
import os
import pickle
import time

import mujoco
import mujoco.viewer
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# BOX REGISTRY  — positions and sizes taken verbatim from ur5e.xml
# ─────────────────────────────────────────────────────────────────────────────
#
#  Half-sizes (x, y, z) are the <geom size="..."> values from the XML.
#  init_pos is the <body pos="..."> world position from the XML.
#
#  World frame: X = right, Y = forward (into rack), Z = up
#
# key → (body_name, geom_name, init_pos, half_sizes_xyz)

HOME_QPOS = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

BOXES = {
    # Lower shelf ─────────────────────────────────────────────────────────────
    "A1":  ("amazon_box_small_A1",
            "amazon_box_small_A1_geom",
            np.array([-0.50, 0.60, 0.346]),
            np.array([0.080, 0.055, 0.040])),

    "A4":  ("amazon_box_medium_A4",
            "amazon_box_medium_A4_geom",
            np.array([-0.10, 0.60, 0.365]),
            np.array([0.100, 0.065, 0.055])),

    "2BB": ("amazon_box_extra_large_2BB",
            "amazon_box_extra_large_2BB_geom",
            np.array([ 0.33, 0.60, 0.395]),
            np.array([0.130, 0.080, 0.085])),

    # Upper shelf ─────────────────────────────────────────────────────────────
    "1A9": ("amazon_box_medium_tall_1A9",
            "amazon_box_medium_tall_1A9_geom",
            np.array([-0.30, 0.60, 0.806]),
            np.array([0.100, 0.075, 0.080])),

    "B0":  ("amazon_box_large_B0",
            "amazon_box_large_B0_geom",
            np.array([ 0.20, 0.60, 0.815]),
            np.array([0.120, 0.075, 0.090])),
}

# Park non-target boxes far underground
PARK_POS = np.array([0.0, 0.0, -5.0])

# Push distance along +Y (into rack)
PUSH_DIST_Y = 0.18   # metres


# ─────────────────────────────────────────────────────────────────────────────
# MPPI CONFIG
# ─────────────────────────────────────────────────────────────────────────────

class MPPIConfig:
    horizon         = 20
    n_samples       = 512
    noise_sigma     = 0.15
    temperature     = 0.05
    discount        = 0.98
    max_steps       = 600
    success_tol     = 0.06      # metres — box-to-goal threshold for success

    # ── Two-goal cost weights ─────────────────────────────────────────────────
    #
    #  approach_cost = w_approach  * d(tip, box)          — always active
    #  push_cost     = gate * w_push_goal * d(box, goal)  — activates at contact
    #
    #  Keeping w_approach always-on means the arm stays near the box even
    #  after contact — it won't fly away mid-push.
    #
    w_approach      = 3.0
    w_push_goal     = 12.0
    w_ctrl          = 0.002

    # ── Sigmoid gate ──────────────────────────────────────────────────────────
    #
    #  gate = sigmoid((contact_thresh - d_tip_box) / gate_sharpness)
    #       → 0  when tip is FAR from box   (approach term drives arm to box)
    #       → 1  when tip is AT/PAST box    (push term drives box to goal)
    #
    #  contact_thresh is auto-calibrated per box in contact_thresh_for_box().
    #  gate_sharpness=0.02 gives a smooth ~4 cm transition band.
    #
    contact_thresh  = 0.09      # default; overridden per-box at runtime
    gate_sharpness  = 0.02


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_goal(target_key):
    """Goal = box centre moved +PUSH_DIST_Y along Y (into rack)."""
    _, _, init_pos, _ = BOXES[target_key]
    goal = init_pos.copy()
    goal[1] += PUSH_DIST_Y
    return goal


def contact_thresh_for_box(target_key):
    """
    Auto-calibrate contact threshold from XML geometry:
        thresh = box_half_y + pusher_tip_half_depth + margin

    pusher_tip geom (from ur5e.xml): size="0.090 0.010 0.060"
        → half-depth along Y = 0.010 m
    """
    _, _, _, half_sizes = BOXES[target_key]
    half_y            = float(half_sizes[1])
    pusher_half_depth = 0.010   # from XML: pusher_tip size y
    margin            = 0.025
    return half_y + pusher_half_depth + margin


# ─────────────────────────────────────────────────────────────────────────────
# SCENE SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_scene(model, data, target_key):
    """
    Reset sim to home pose.
    Place the target box at its shelf position.
    Park all other boxes underground (freejoint teleport).
    """
    mujoco.mj_resetData(model, data)
    data.qpos[:6] = HOME_QPOS
    data.ctrl[:6] = HOME_QPOS

    for key, (body_name, _, init_pos, _) in BOXES.items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            print(f"  [WARN] body '{body_name}' not found — skipping.")
            continue
        jnt_id = model.body_jntadr[body_id]
        if jnt_id < 0:
            print(f"  [WARN] body '{body_name}' has no freejoint — skipping.")
            continue
        qadr = model.jnt_qposadr[jnt_id]
        vadr = model.jnt_dofadr[jnt_id]

        pos = init_pos if key == target_key else PARK_POS
        data.qpos[qadr:qadr+3]   = pos
        data.qpos[qadr+3:qadr+7] = [1, 0, 0, 0]   # identity quaternion
        data.qvel[vadr:vadr+6]   = 0.0

    mujoco.mj_forward(model, data)


# ─────────────────────────────────────────────────────────────────────────────
# OBSERVATION  (27-dim, float32)
# ─────────────────────────────────────────────────────────────────────────────

def get_obs(model, data, box_body, goal):
    """
    27-dim observation:
      [0:6]   joint positions  (robot arm)
      [6:12]  joint velocities (robot arm)
      [12:15] pusher tip position   (world frame)
      [15:18] box centre position   (world frame)
      [18:21] box linear velocity   (world frame)
      [21:24] goal position
      [24:27] box → goal vector
    """
    qpos = data.qpos[:6].copy()
    qvel = data.qvel[:6].copy()

    tip_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pusher_tip_site")
    tip_pos = data.site_xpos[tip_id].copy()

    box_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, box_body)
    box_pos = data.xpos[box_id].copy()

    vel6 = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, box_id, vel6, 0)
    box_linvel = vel6[3:].copy()

    to_goal = goal - box_pos
    return np.concatenate([qpos, qvel, tip_pos, box_pos, box_linvel, goal, to_goal],
                          dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED MPPI ROLLOUT COST
# ─────────────────────────────────────────────────────────────────────────────

def rollout_cost_unified(model, data_ref, action_seq, box_id, tip_id, goal, cfg):
    """
    Simulate one candidate trajectory and return its total discounted cost.

    Per-step cost:
        gate          = sigmoid((contact_thresh - d_tip_box) / gate_sharpness)
        step_cost     = w_approach  * d(tip, box)              [always active]
                      + gate * w_push_goal * d(box, goal)      [gated]
                      + w_ctrl * ||u_t||^2                     [regulariser]
        total_cost   += gamma^t * step_cost

    MPPI samples many such trajectories; lower-cost ones get higher weight.
    The optimal weighted average becomes the new control mean.
    """
    d = mujoco.MjData(model)
    d.qpos[:] = data_ref.qpos
    d.qvel[:] = data_ref.qvel
    d.ctrl[:]  = data_ref.ctrl
    mujoco.mj_forward(model, d)

    cost  = 0.0
    gamma = 1.0

    for t in range(cfg.horizon):
        d.ctrl[:6] = action_seq[t]
        mujoco.mj_step(model, d)

        tip_pos = d.site_xpos[tip_id].copy()
        box_pos = d.xpos[box_id].copy()

        d_tip_box  = np.linalg.norm(tip_pos - box_pos)
        d_box_goal = np.linalg.norm(box_pos  - goal)

        # Sigmoid gate: 0 = far (approach phase), 1 = near/contact (push phase)
        contact_gate = 1.0 / (
            1.0 + np.exp((d_tip_box - cfg.contact_thresh) / cfg.gate_sharpness)
        )

        approach_cost = cfg.w_approach  * d_tip_box
        push_cost     = contact_gate    * cfg.w_push_goal * d_box_goal
        ctrl_cost     = cfg.w_ctrl      * float(np.sum(action_seq[t] ** 2))

        cost += gamma * (approach_cost + push_cost + ctrl_cost)
        gamma *= cfg.discount

    return cost


# ─────────────────────────────────────────────────────────────────────────────
# MPPI CONTROLLER
# ─────────────────────────────────────────────────────────────────────────────

class MPPIController:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg   = cfg
        self.U     = np.zeros((cfg.horizon, 6))   # warm-started mean sequence

    def reset(self):
        self.U[:] = 0.0

    def act(self, data, box_id, tip_id, goal):
        cfg = self.cfg

        # Perturb mean sequence: N_samples x horizon x 6
        eps = np.random.randn(cfg.n_samples, cfg.horizon, 6) * cfg.noise_sigma
        U_s = np.clip(self.U[None] + eps, -6.2, 6.2)

        # Evaluate all candidate trajectories
        costs = np.array([
            rollout_cost_unified(self.model, data, U_s[s], box_id, tip_id, goal, cfg)
            for s in range(cfg.n_samples)
        ])

        # MPPI weight update
        beta = costs.min()
        w    = np.exp(-(costs - beta) / cfg.temperature)
        w   /= w.sum() + 1e-8

        # Weighted average of samples → new mean sequence
        self.U = np.einsum("s,shd->hd", w, U_s)

        # Extract first action, then shift horizon (warm-start next step)
        action      = self.U[0].copy()
        self.U[:-1] = self.U[1:]
        self.U[-1]  = 0.0

        return action


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED MPPI EXECUTION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_unified_mppi(model, data, box_body, goal, cfg, render_cb=None):
    """
    Execute the unified MPPI controller.

    The controller starts with the robot at home pose and must:
      1. Move the pusher tip to the box   (approach cost drives this)
      2. Push the box to the goal         (push cost activates on contact)

    Progress is printed every 50 steps:
      step | tip→box dist | gate value | box→goal dist

    Returns:
      obs_list, act_list, rew_list, done_list, success (bool)
    """
    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, box_body)
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pusher_tip_site")

    if box_id < 0:
        raise ValueError(f"Body '{box_body}' not found in model.")
    if tip_id < 0:
        raise ValueError("Site 'pusher_tip_site' not found in model.")

    controller = MPPIController(model, cfg)
    obs_list, act_list, rew_list, done_list = [], [], [], []
    success = False
    dist    = float("inf")

    for step in range(cfg.max_steps):
        obs    = get_obs(model, data, box_body, goal)
        action = controller.act(data, box_id, tip_id, goal)

        data.ctrl[:6] = np.clip(action, -6.2, 6.2)
        mujoco.mj_step(model, data)
        if render_cb:
            render_cb()

        box_pos  = data.xpos[box_id].copy()
        tip_pos  = data.site_xpos[tip_id].copy()
        dist     = float(np.linalg.norm(box_pos - goal))
        done     = dist < cfg.success_tol

        # Gate value (diagnostic only — not used for control here)
        d_tip_box    = float(np.linalg.norm(tip_pos - box_pos))
        contact_gate = 1.0 / (
            1.0 + np.exp((d_tip_box - cfg.contact_thresh) / cfg.gate_sharpness)
        )

        obs_list.append(obs)
        act_list.append(action.astype(np.float32))
        rew_list.append(np.float32(-dist))
        done_list.append(np.float32(done))

        if step % 50 == 0:
            print(f"  step {step:4d} | tip→box: {d_tip_box:.3f} m | "
                  f"gate: {contact_gate:.2f} | box→goal: {dist:.3f} m")

        if done:
            success = True
            print(f"\n  [MPPI] SUCCESS in {step+1} steps  "
                  f"(final box→goal = {dist:.4f} m)")
            break

    if not success:
        print(f"\n  [MPPI] TIMEOUT after {cfg.max_steps} steps.  "
              f"Final box→goal = {dist:.3f} m")

    return obs_list, act_list, rew_list, done_list, success


# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save_demo(demo, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)

    pkl_path = os.path.join(out_dir, f"{filename}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(demo, f)

    npz_path = os.path.join(out_dir, f"{filename}.npz")
    np.savez_compressed(
        npz_path,
        observations = demo["observations"],
        actions      = demo["actions"],
        rewards      = demo["rewards"],
        dones        = demo["dones"],
    )

    print(f"\n  Saved → {pkl_path}")
    print(f"  Saved → {npz_path}")
    print(f"  Total transitions : {len(demo['observations'])}")
    print(f"  Obs shape         : {demo['observations'].shape}")
    print(f"  Action shape      : {demo['actions'].shape}")
    print(f"  Success           : {demo['info']['success']}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collect MPPI pushing demonstrations — UR5e shelf scene."
    )
    parser.add_argument("--xml",       required=True,
                        help="Path to scene.xml")
    parser.add_argument("--box",       required=True,
                        choices=list(BOXES.keys()),
                        help="Box to push: A1 | A4 | 2BB | 1A9 | B0")
    parser.add_argument("--goal_dir",  required=True,
                        choices=["left", "right"],
                        help="Kept for filename/CLI compatibility; "
                             "push is always along +Y into rack")
    parser.add_argument("--out_dir",   default="demonstrations")
    parser.add_argument("--render",    action="store_true")
    # MPPI knobs
    parser.add_argument("--n_samples",      type=int,   default=512)
    parser.add_argument("--horizon",        type=int,   default=20)
    parser.add_argument("--max_steps",      type=int,   default=600)
    parser.add_argument("--noise_sigma",    type=float, default=None)
    parser.add_argument("--temperature",    type=float, default=None)
    # Cost knobs
    parser.add_argument("--contact_thresh", type=float, default=None,
                        help="Override auto-calibrated contact threshold (m). "
                             "Auto = box_half_y + 0.010 + 0.025")
    parser.add_argument("--w_approach",     type=float, default=None)
    parser.add_argument("--w_push_goal",    type=float, default=None)
    parser.add_argument("--gate_sharpness", type=float, default=None)
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    model = mujoco.MjModel.from_xml_path(args.xml)
    data  = mujoco.MjData(model)

    # ── Config ────────────────────────────────────────────────────────────────
    cfg           = MPPIConfig()
    cfg.n_samples = args.n_samples
    cfg.horizon   = args.horizon
    cfg.max_steps = args.max_steps

    # Auto-calibrate contact_thresh from XML box geometry
    cfg.contact_thresh = contact_thresh_for_box(args.box)

    # CLI overrides
    if args.contact_thresh  is not None: cfg.contact_thresh = args.contact_thresh
    if args.w_approach      is not None: cfg.w_approach     = args.w_approach
    if args.w_push_goal     is not None: cfg.w_push_goal    = args.w_push_goal
    if args.gate_sharpness  is not None: cfg.gate_sharpness = args.gate_sharpness
    if args.noise_sigma     is not None: cfg.noise_sigma    = args.noise_sigma
    if args.temperature     is not None: cfg.temperature    = args.temperature

    # ── Info print ────────────────────────────────────────────────────────────
    box_name, _, init_pos, half_sizes = BOXES[args.box]
    goal = compute_goal(args.box)

    print(f"\n{'='*62}")
    print(f"  Box             : {args.box}  ({box_name})")
    print(f"  Start pos       : {init_pos}")
    print(f"  Half sizes (xyz): {half_sizes}")
    print(f"  Goal            : {goal}  (+{PUSH_DIST_Y:.2f} m in Y)")
    print(f"  contact_thresh  : {cfg.contact_thresh:.4f} m")
    print(f"    = half_y {half_sizes[1]:.3f} + tip_depth 0.010 + margin 0.025")
    print(f"  gate_sharpness  : {cfg.gate_sharpness}")
    print(f"  w_approach      : {cfg.w_approach}")
    print(f"  w_push_goal     : {cfg.w_push_goal}")
    print(f"  n_samples       : {cfg.n_samples}")
    print(f"  horizon         : {cfg.horizon}")
    print(f"  max_steps       : {cfg.max_steps}")
    print(f"{'='*62}\n")

    # ── Setup scene ───────────────────────────────────────────────────────────
    setup_scene(model, data, args.box)

    filename = f"{args.box}_{args.goal_dir}"

    def run(render_cb=None):
        print("[Unified MPPI] Starting approach + push ...\n")
        obs_list, act_list, rew_list, done_list, success = run_unified_mppi(
            model, data, box_name, goal, cfg, render_cb=render_cb
        )

        demo = {
            "observations": np.stack(obs_list),
            "actions":      np.stack(act_list),
            "rewards":      np.array(rew_list,  dtype=np.float32),
            "dones":        np.array(done_list, dtype=np.float32),
            "info": {
                "box":            args.box,
                "box_body":       box_name,
                "box_init_pos":   init_pos.tolist(),
                "box_half_sizes": half_sizes.tolist(),
                "goal":           goal.tolist(),
                "goal_dir":       args.goal_dir,
                "mppi_steps":     len(obs_list),
                "success":        success,
                "cfg": {
                    "contact_thresh": cfg.contact_thresh,
                    "gate_sharpness": cfg.gate_sharpness,
                    "w_approach":     cfg.w_approach,
                    "w_push_goal":    cfg.w_push_goal,
                    "w_ctrl":         cfg.w_ctrl,
                    "n_samples":      cfg.n_samples,
                    "horizon":        cfg.horizon,
                    "noise_sigma":    cfg.noise_sigma,
                    "temperature":    cfg.temperature,
                },
            },
        }

        print("\n── Saving demonstration ──")
        save_demo(demo, args.out_dir, filename)

    # ── Launch ────────────────────────────────────────────────────────────────
    if args.render:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            def sync():
                viewer.sync()
                time.sleep(model.opt.timestep)
            run(render_cb=sync)
    else:
        run()


if __name__ == "__main__":
    main()
