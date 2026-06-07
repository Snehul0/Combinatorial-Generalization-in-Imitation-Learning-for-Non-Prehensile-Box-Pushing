import argparse
import numpy as np
import mujoco


HOME_QPOS = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

BOXES = {
    "A1":  ("amazon_box_small_A1",        np.array([-0.50, 0.60, 0.346]), np.array([0.080, 0.055, 0.040])),
    "A4":  ("amazon_box_medium_A4",       np.array([-0.10, 0.60, 0.365]), np.array([0.100, 0.065, 0.055])),
    "2BB": ("amazon_box_extra_large_2BB", np.array([ 0.33, 0.60, 0.395]), np.array([0.130, 0.080, 0.085])),
    "1A9": ("amazon_box_medium_tall_1A9", np.array([-0.30, 0.60, 0.806]), np.array([0.100, 0.075, 0.080])),
    "B0":  ("amazon_box_large_B0",        np.array([ 0.20, 0.60, 0.815]), np.array([0.120, 0.075, 0.090])),
}


JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def get_ids(model):
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
    qpos_ids = [model.jnt_qposadr[j] for j in joint_ids]
    dof_ids = [model.jnt_dofadr[j] for j in joint_ids]
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pusher_tip_site")
    return joint_ids, qpos_ids, dof_ids, site_id


def set_robot_q(model, data, q, qpos_ids):
    for i, qid in enumerate(qpos_ids):
        data.qpos[qid] = q[i]
    data.ctrl[:6] = q
    mujoco.mj_forward(model, data)


def solve_ik(model, data, target, q_init, qpos_ids, dof_ids, site_id,
             max_iter=250, step_size=0.45, damping=1e-3):
    q = q_init.copy()

    for _ in range(max_iter):
        set_robot_q(model, data, q, qpos_ids)

        tip = data.site_xpos[site_id].copy()
        err = target - tip

        if np.linalg.norm(err) < 0.005:
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

        J = jacp[:, dof_ids]
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), err)

        q = q + step_size * dq
        q = np.clip(q, -6.2, 6.2)

    set_robot_q(model, data, q, qpos_ids)
    tip = data.site_xpos[site_id].copy()
    err_norm = np.linalg.norm(target - tip)

    return q, tip, err_norm


def collision_penalty(model, data):
    penalty = 0.0
    bad_pairs = []

    for i in range(data.ncon):
        c = data.contact[i]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
        pair = f"{g1} {g2}"

        # Penalize robot/pusher hitting rack/shelf/floor.
        if ("pusher" in pair or "wrist" in pair or "forearm" in pair or "upperarm" in pair) and (
            "shelf" in pair or "rack" in pair or "beam" in pair or "wall" in pair or "floor" in pair
        ):
            penalty += 10.0
            bad_pairs.append(pair)

    return penalty, bad_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True)
    parser.add_argument("--box", required=True, choices=list(BOXES.keys()))
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    joint_ids, qpos_ids, dof_ids, site_id = get_ids(model)

    box_name, box_pos, half = BOXES[args.box]

    # For normal scenario, push +Y.
    # Contact point is behind/front of the box on robot side.
    target = box_pos.copy()
    target[0] = box_pos[0]
    target[1] = box_pos[1] - half[1] - 0.035
    target[2] = box_pos[2] + 0.020

    print("\nTarget box:", args.box, box_name)
    print("Box pos:", box_pos)
    print("Desired pre-push tip target:", target)

    # Good rough seeds around shelf-reaching posture.
    base_seeds = [
        HOME_QPOS,
        np.array([0.0, -1.40, 1.60, -1.75, -1.5708, 0.0]),
        np.array([0.0, -1.25, 1.45, -1.80, -1.5708, 0.0]),
        np.array([0.0, -1.10, 1.35, -1.85, -1.5708, 0.0]),
        np.array([0.0, -1.55, 1.80, -1.90, -1.5708, 0.0]),
    ]

    best = None

    for trial in range(args.trials):
        mujoco.mj_resetData(model, data)

        if trial < len(base_seeds):
            q0 = base_seeds[trial].copy()
        else:
            center = base_seeds[1]
            noise = np.array([
                rng.normal(0.0, 0.45),  # shoulder pan
                rng.normal(0.0, 0.45),  # shoulder lift
                rng.normal(0.0, 0.45),  # elbow
                rng.normal(0.0, 0.45),  # wrist 1
                rng.normal(0.0, 0.20),  # wrist 2
                rng.normal(0.0, 0.40),  # wrist 3
            ])
            q0 = center + noise

        q0 = np.clip(q0, -6.2, 6.2)

        q_sol, tip, err = solve_ik(
            model, data, target, q0, qpos_ids, dof_ids, site_id
        )

        pen, bad_pairs = collision_penalty(model, data)

        # Keep wrist_2 near -pi/2 to avoid weird tool orientation.
        orient_pen = 0.1 * abs(q_sol[4] + 1.5708)

        score = err + pen + orient_pen

        if best is None or score < best["score"]:
            best = {
                "score": score,
                "err": err,
                "pen": pen,
                "q": q_sol.copy(),
                "tip": tip.copy(),
                "bad_pairs": bad_pairs,
            }

            print("\nNew best at trial", trial)
            print("  score:", best["score"])
            print("  err:", best["err"])
            print("  collision_penalty:", best["pen"])
            print("  q:", np.round(best["q"], 4).tolist())
            print("  tip:", np.round(best["tip"], 4))
            print("  tip-target:", np.round(best["tip"] - target, 4))

            if best["err"] < 0.015 and best["pen"] == 0.0:
                print("\nGood seed found.")
                break

    print("\n" + "=" * 70)
    print("BEST RESULT")
    print("=" * 70)
    print("Error:", best["err"])
    print("Collision penalty:", best["pen"])
    print("Tip:", np.round(best["tip"], 4))
    print("Target:", np.round(target, 4))
    print("Tip-target:", np.round(best["tip"] - target, 4))
    print("\nCopy this into demo_collector.py:")
    print("SHELF_SEED_LOWER = np.array(" + repr(np.round(best["q"], 4).tolist()) + ")")
    print("=" * 70)


if __name__ == "__main__":
    main()
