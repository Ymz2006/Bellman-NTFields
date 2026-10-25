"""
Sample collision-free joint configurations via FK (uniform C-space sampling) — UR5e version.

Instead of IK-based sampling (which can bias the C-space distribution),
this script uniformly samples joint angles within joint limits, runs FK to
get the EE position and collision spheres, then filters by:
  1. EE position within the table workspace bounds
  2. Collision-free (obstacle + self-collision)
  NOTE: UR5e has NO torso obstacle; torso handling is disabled.

Usage (inside curobo container):
    python sample_fk_ur5e.py [--n_points 1500000] [--height 0.3] [--device cuda:0] [--seed 42]
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

DATA_DIR = Path("/outputs/sampler-cache")

# UR5e-specific constants (replaces G1 imports from visualize.py)
URDF_PATH = "/opt/ur5e_description/ur5e_robotiq_2f_140.urdf"
ROBOT_CFG_PATH = "ur5e_arm.yml"
SPHERE_YML_PATH = "spheres_ur5e_arm.yml"
COLLISION_LINK_NAMES = [
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
    "tool0",
    "robotiq_arg2f_base_link",
    "left_outer_knuckle",
    "left_inner_knuckle",
    "left_outer_finger",
    "left_inner_finger",
    "left_inner_finger_pad",
    "right_outer_knuckle",
    "right_inner_knuckle",
    "right_outer_finger",
    "right_inner_finger",
    "right_inner_finger_pad",
]
# UR5e has NO torso obstacle — set to None to disable torso collision filtering
TORSO_OBSTACLE_PATH = None
TORSO_IGNORE_SPHERE_INDICES = []

from visualize import init_curobo


def load_scene_yaml(scene_path):
    with open(scene_path) as f:
        return yaml.safe_load(f)


def get_table_top_z(cuboids):
    """Get table top Z from a list of cuboid dicts or a scene_data dict."""
    items = cuboids.get("cuboids", cuboids) if isinstance(cuboids, dict) else cuboids
    for c in items:
        if "table" in c["name"]:
            return c["pose"][2] + c["dims"][2] / 2.0
    return 0.025  # default
from collision_utils import (
    load_cuboid_geometry,
    check_collision_full,
    build_self_collision_mask,
)
from constraint_utils import (
    PALM_AXIS_COLUMN,
)
from ur5e_task_axes import (
    axis_label,
    load_task_axes,
    task_axes_json,
    task_axis_arrays,
    tilt_deg_from_quat,
)


SCENE_PATH = "scene_ur5e.yml"

DOF = 6
Q_MIN = torch.tensor([-6.2832, -6.2832, -3.1416, -6.2832, -6.2832, -6.2832])
Q_MAX = torch.tensor([ 6.2832,  6.2832,  3.1416,  6.2832,  6.2832,  6.2832])


def sample_fk(
    robot_model, cuboids, n_points=1500000, height=0.3,
    device="cuda:0", batch_size=50000,
    cuboid_centers=None, cuboid_half=None,
    torso_centers=None, torso_half=None,
    self_coll_mask=None,
    max_tilt_deg=None,   # None = no tilt filter (legacy behavior)
    n_ref=None,          # legacy metadata only; scene task_axes define references
    axis_col=PALM_AXIS_COLUMN,  # legacy CLI compatibility
    pouring_mode=False,  # if True, use the configured pour-axis constraint
    x_tilt_max_deg=20.0, # max configured pour-axis tilt (historical option name)
    orient_filter_margin=0.0, # NEW: extra degrees beyond x_tilt_max_deg for FK filter
    max_pour_deg=180.0,  # legacy filename tag for pouring pools
    y_lo_override=None,  # override workspace Y lower bound (for zone-based sampling)
    y_hi_override=None,  # override workspace Y upper bound (for zone-based sampling)
    sampling_bounds=None,  # optional {"x":[lo,hi],"y":[lo,hi]} from scene `sampling:`
                           # block — DECOUPLES the FK box from the (large) visual table
    task_axes=None,
):
    """Sample collision-free configs via uniform FK sampling with workspace filtering.

    1. Uniform random joint angles in [Q_MIN, Q_MAX]
    2. FK → EE position + collision spheres
    3. Filter: EE in workspace + collision-free
    4. Repeat until n_points collected
    """
    t_start = time.time()
    table_top_z = get_table_top_z(cuboids)
    table_cuboid = next((c for c in cuboids if "table" in c["name"]), None)
    if table_cuboid is None:
        raise RuntimeError("No table cuboid found in scene")

    tx, ty = table_cuboid["pose"][0], table_cuboid["pose"][1]
    half_dx = table_cuboid["dims"][0] / 2.0
    half_dy = table_cuboid["dims"][1] / 2.0

    # Workspace bounds. Prefer an explicit `sampling:` region (DECOUPLED from a large
    # visual table); otherwise fall back to the table-derived box (legacy behavior).
    if sampling_bounds is not None:
        x_lo, x_hi = float(sampling_bounds["x"][0]), float(sampling_bounds["x"][1])
        y_lo, y_hi = float(sampling_bounds["y"][0]), float(sampling_bounds["y"][1])
        print(f"  [sampling] using explicit `sampling:` bounds "
              f"x=[{x_lo:.2f},{x_hi:.2f}] y=[{y_lo:.2f},{y_hi:.2f}] (table dims ignored)")
    else:
        x_lo, x_hi = tx - half_dx * 0.9, tx + half_dx * 0.9
        y_lo, y_hi = ty - half_dy * 0.9, ty + half_dy * 0.9
    z_lo, z_hi = table_top_z, table_top_z + max(height, 0.01)

    # Allow callers to restrict Y range (for dual-zone sampling)
    if y_lo_override is not None:
        y_lo = max(y_lo, y_lo_override)
    if y_hi_override is not None:
        y_hi = min(y_hi, y_hi_override)

    q_min = Q_MIN.to(device)
    q_max = Q_MAX.to(device)
    q_range = q_max - q_min

    do_collision_filter = cuboid_centers is not None

    all_positions = []
    all_q = []
    collected = 0
    round_num = 0
    total_sampled = 0

    pbar = tqdm(total=n_points, desc="FK samples", unit="pts")

    while collected < n_points:
        round_num += 1

        # Uniform random joint angles in [Q_MIN, Q_MAX]
        rand_01 = torch.rand(batch_size, DOF, device=device)
        q_batch = q_min + rand_01 * q_range
        total_sampled += batch_size

        # FK: get EE position and collision spheres
        state = robot_model.get_state(q_batch)
        ee_pos = state.ee_pose.position  # (B, 3)

        # Filter 1: EE in workspace bounds
        ee_mask = (
            (ee_pos[:, 0] >= x_lo) & (ee_pos[:, 0] <= x_hi) &
            (ee_pos[:, 1] >= y_lo) & (ee_pos[:, 1] <= y_hi) &
            (ee_pos[:, 2] >= z_lo) & (ee_pos[:, 2] <= z_hi)
        )
        n_in_ws = ee_mask.sum().item()

        if n_in_ws == 0:
            pbar.set_postfix(ws=0, sampled=total_sampled)
            continue

        q_ws = q_batch[ee_mask]
        ee_ws = ee_pos[ee_mask].cpu().numpy()

        # Filter 1b (optional): tilt constraint or pouring constraint
        if pouring_mode:
            state_ws = robot_model.get_state(q_ws)
            quat_ws = state_ws.ee_pose.quaternion  # (B, 4) wxyz
            local_axis, world_axis = task_axis_arrays(task_axes or load_task_axes(None), "pour")
            pour_tilt = tilt_deg_from_quat(quat_ws, local_axis, world_axis)
            fk_tilt_limit = x_tilt_max_deg + orient_filter_margin
            tilt_ok = pour_tilt <= fk_tilt_limit
            n_tilt_ok = tilt_ok.sum().item()
            if n_tilt_ok == 0:
                pbar.set_postfix(ws=n_in_ws, tilt=0, sampled=total_sampled)
                continue
            q_ws = q_ws[tilt_ok]
            ee_ws = ee_ws[tilt_ok.cpu().numpy()]
        elif max_tilt_deg is not None:
            state_ws = robot_model.get_state(q_ws)
            quat_ws = state_ws.ee_pose.quaternion  # (B, 4) wxyz
            local_axis, world_axis = task_axis_arrays(task_axes or load_task_axes(None), "cup")
            cup_tilt = tilt_deg_from_quat(quat_ws, local_axis, world_axis)
            tilt_ok = cup_tilt <= max_tilt_deg
            n_tilt_ok = tilt_ok.sum().item()
            if n_tilt_ok == 0:
                pbar.set_postfix(ws=n_in_ws, tilt=0, sampled=total_sampled)
                continue
            q_ws = q_ws[tilt_ok]
            ee_ws = ee_ws[tilt_ok.cpu().numpy()]
        else:
            n_tilt_ok = None  # sentinel to skip printing

        # Filter 2: collision-free (obstacle + torso + self-collision)
        if do_collision_filter:
            in_coll, _ = check_collision_full(
                robot_model, q_ws, cuboid_centers, cuboid_half,
                torso_cuboid_centers=torso_centers,
                torso_cuboid_half=torso_half,
                self_collision_mask=self_coll_mask,
                torso_ignore_sphere_indices=TORSO_IGNORE_SPHERE_INDICES,
                device=device,
            )
            free_mask = ~in_coll.cpu().numpy()
            q_free = q_ws[free_mask].cpu().numpy()
            ee_free = ee_ws[free_mask]
        else:
            q_free = q_ws.cpu().numpy()
            ee_free = ee_ws

        n_free = q_free.shape[0]
        if n_free == 0:
            pbar.set_postfix(ws=n_in_ws, free=0, sampled=total_sampled)
            continue

        # Take only what we need
        take = min(n_free, n_points - collected)
        all_q.append(q_free[:take])
        all_positions.append(ee_free[:take])
        collected += take
        pbar.update(take)
        postfix = {
            "ws_rate": f"{n_in_ws/batch_size*100:.1f}%",
            "free_rate": f"{n_free/n_in_ws*100:.1f}%",
            "sampled": total_sampled,
        }
        if n_tilt_ok is not None:
            postfix["tilt_rate"] = f"{n_tilt_ok/n_in_ws*100:.1f}%"
        pbar.set_postfix(**postfix)

    pbar.close()

    q_solutions = np.concatenate(all_q)[:n_points]
    positions_out = np.concatenate(all_positions)[:n_points]

    t_total = time.time() - t_start
    accept_rate = n_points / total_sampled * 100
    print(f"  Collected {n_points} FK samples in {round_num} rounds, {t_total:.1f}s")
    print(f"  Total sampled: {total_sampled}, acceptance rate: {accept_rate:.2f}%")

    return {
        "positions": positions_out,
        "q_solutions": q_solutions,
        "success": np.ones(n_points, dtype=bool),
        "n_success": n_points,
        "n_total": total_sampled,
        "n_requested": n_points,
        "height": height,
        "table_top_z": table_top_z,
        "sampling_method": "fk",
        "timing": {"total_s": t_total},
        "max_tilt_deg": max_tilt_deg,  # NEW, None if disabled
        "n_ref": n_ref.cpu().numpy() if n_ref is not None else None,  # NEW
    }


def main():
    parser = argparse.ArgumentParser(description="Sample FK configs and save pkl")
    parser.add_argument("--n_points", type=int, default=1500000)
    parser.add_argument("--height", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional random seed for repeatable FK sampling.")
    parser.add_argument("--max_tilt_deg", type=float, default=None,
                        help="If set, filter configs to palm tilt <= this (deg). "
                             "Enables roll-constrained sampling.")
    parser.add_argument("--q_default", type=str, default=None,
                        help="Comma-separated 6 joint angles (rad) defining the "
                             "reference 'cupping' pose.")
    parser.add_argument("--task", type=str, default="cupping",
                        choices=["cupping", "pouring"],
                        help="Task type: 'cupping' or 'pouring'; orientation axes come from scene task_axes.")
    parser.add_argument("--max_pour_deg", type=float, default=180.0,
                        help="Legacy pouring pool filename tag; filtering uses scene task_axes + --x_tilt_max_deg.")
    parser.add_argument("--x_tilt_max_deg", type=float, default=20.0,
                        help="Configured pour-axis tilt threshold for pouring (degrees)")
    parser.add_argument("--orient_filter_margin", type=float, default=0.0,
                        help="Extra degrees beyond x_tilt_max_deg for FK filter. "
                             "E.g. 15 with x_tilt_max_deg=10 → filter at 25°, "
                             "keeping boundary configs for orientation speed field.")
    parser.add_argument("--scene", type=str, default=SCENE_PATH,
                        help="Scene YAML file (default: scene_obstacle.yml)")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional output pkl path. If omitted, use the legacy generated name.")
    parser.add_argument("--no_table_collision", action="store_true",
                        help="Drop table-named cuboids from the COLLISION filter while "
                             "keeping them for z-floor/workspace derivation. Used for the "
                             "true free-space (no-table) sample: configs that would graze "
                             "the table are now allowed, but the workspace box/z-range is "
                             "unchanged so the task stays comparable.")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"  Random seed: {args.seed}")

    # Init cuRobo
    print("Initializing cuRobo...")
    cfg_dict, robot_cfg, robot_model, tensor_args = init_curobo(
        ROBOT_CFG_PATH, URDF_PATH, args.device
    )
    print("  cuRobo ready.")

    scene_path = args.scene
    scene_data = load_scene_yaml(scene_path)
    task_axes = load_task_axes(scene_path)
    cup_local, cup_world = task_axis_arrays(task_axes, "cup")
    pour_local, pour_world = task_axis_arrays(task_axes, "pour")
    print(
        "  Task axes: "
        f"cup local {axis_label(cup_local)} -> world {axis_label(cup_world)}; "
        f"pour local {axis_label(pour_local)} -> world {axis_label(pour_world)}"
    )

    # Determine constraint based on task.
    # UR5e grasp_frame axes are SWAPPED vs G1 (validated 2026-06-14):
    #   cup-up   = grasp_frame +X  → col 0   (G1 used col 2)
    #   forward  = grasp_frame +Z  → col 2   (G1 used col 0)
    CUP_AXIS_COL_UR5E = 0
    pouring_mode = (args.task == "pouring")
    if pouring_mode:
        axis_col = CUP_AXIS_COL_UR5E  # unused in pouring_mode (kept for compat)
        fk_limit = args.x_tilt_max_deg + args.orient_filter_margin
        print(
            f"  Task: pouring ({axis_label(pour_local)} -> {axis_label(pour_world)} "
            f"within ±{args.x_tilt_max_deg}°, FK filter at {fk_limit}°)"
        )
    else:
        axis_col = CUP_AXIS_COL_UR5E  # 0 = EE +X (UR5e cup-up axis)

    n_ref_tensor = None
    if not pouring_mode and args.max_tilt_deg is not None:
        if args.q_default is not None:
            print("  [warning] --q_default is ignored; scene task_axes define the reference axis.")
        else:
            print(
                f"  Task: cupping ({axis_label(cup_local)} -> {axis_label(cup_world)})"
            )
        print(f"  Task: {args.task}, axis_col={axis_col}")
        print(f"  Roll constraint enabled: max_tilt_deg={args.max_tilt_deg}")
        if n_ref_tensor is not None:
            print(f"  n_ref     = {n_ref_tensor.cpu().numpy()}")

    # Load scene cuboids
    cuboids = scene_data.get("cuboids", scene_data) if isinstance(scene_data, dict) else scene_data
    # Optional explicit FK sampling region (decoupled from a large visual table).
    sampling_bounds = scene_data.get("sampling") if isinstance(scene_data, dict) else None

    # Load collision geometry. For the true free-space sample, drop table-named
    # cuboids from COLLISION only (they remain in `cuboids` for z-floor/workspace).
    if args.no_table_collision:
        coll_cuboids = [c for c in cuboids if "table" not in c["name"]]
        if len(coll_cuboids) == 0:
            cuboid_centers, cuboid_half = None, None
            print("  [no_table_collision] table dropped from collision; "
                  "NO obstacle collision filter (self-collision only).")
        else:
            cc = torch.tensor([c["pose"][:3] for c in coll_cuboids],
                              dtype=torch.float32, device=args.device)
            ch = torch.tensor([[d / 2.0 for d in c["dims"]] for c in coll_cuboids],
                              dtype=torch.float32, device=args.device)
            cuboid_centers, cuboid_half = cc, ch
            print(f"  [no_table_collision] table dropped from collision; "
                  f"{len(coll_cuboids)} non-table cuboid(s) remain.")
    else:
        cuboid_centers, cuboid_half = load_cuboid_geometry(scene_path, args.device)
    # UR5e: no torso obstacle — torso_centers/torso_half stay None
    if TORSO_OBSTACLE_PATH is not None:
        torso_centers, torso_half = load_cuboid_geometry(TORSO_OBSTACLE_PATH, args.device)
    else:
        torso_centers, torso_half = None, None
    with open(ROBOT_CFG_PATH) as f:
        rcfg = yaml.safe_load(f)
    self_coll_ignore = rcfg["robot_cfg"]["kinematics"].get("self_collision_ignore", {})
    self_coll_mask, _ = build_self_collision_mask(
        SPHERE_YML_PATH, self_coll_ignore, COLLISION_LINK_NAMES, device=args.device
    )
    print("  Collision filter ready (obstacle + self-collision; no torso for UR5e).")

    # Sample
    print(f"FK sampling: n_points={args.n_points}, height={args.height}, batch_size={args.batch_size}")
    result = sample_fk(
        robot_model=robot_model,
        cuboids=cuboids,
        n_points=args.n_points,
        height=args.height,
        device=args.device,
        batch_size=args.batch_size,
        cuboid_centers=cuboid_centers,
        cuboid_half=cuboid_half,
        torso_centers=torso_centers,
        torso_half=torso_half,
        self_coll_mask=self_coll_mask,
        max_tilt_deg=args.max_tilt_deg if not pouring_mode else None,
        n_ref=n_ref_tensor,
        axis_col=axis_col,
        pouring_mode=pouring_mode,
        x_tilt_max_deg=args.x_tilt_max_deg,
        orient_filter_margin=args.orient_filter_margin,
        max_pour_deg=args.max_pour_deg,
        sampling_bounds=sampling_bounds,
        task_axes=task_axes,
    )

    t_total = result["timing"]["total_s"]
    print(f"  Result: {result['n_success']} solutions, {t_total:.1f}s")
    result["scene"] = args.scene
    result["task"] = args.task
    result["task_axes_json"] = task_axes_json(task_axes)

    # Save pkl
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    h = result.get("height", 0)
    n = result.get("n_success", 0)
    if pouring_mode:
        tilt_str = f"_pour{int(round(args.max_pour_deg))}"
    elif args.max_tilt_deg is not None:
        tilt_str = f"_tilt{int(round(args.max_tilt_deg))}"
    else:
        tilt_str = ""
    task_str = f"_{args.task}" if args.task != "cupping" else ""
    save_path = Path(args.output) if args.output else DATA_DIR / f"table_fk_{n}pts_h{h:.2f}{tilt_str}{task_str}.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(result, f)
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()
