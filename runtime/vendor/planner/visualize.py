"""
NTFields-G1 Viser Web GUI.

Provides 3D visualization of G1 right arm with:
- URDF mesh rendering via yourdfpy
- Scene obstacle rendering
- Joint control sliders
- Multi-task training data visualization (load from disk)
- Trajectory planning and playback via cuRobo MotionGen

Usage:
    python visualize.py --port 8888
    # Open http://localhost:8888 in browser
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
import viser
import yaml
import yourdfpy

from viser_utils import (
    load_urdf,
    add_robot_to_scene,
    update_robot_pose,
    load_scene_yaml,
    add_scene_to_viser,
    add_zone_planes_to_viser,
    load_collision_spheres,
    get_link_transform,
)
from collision_utils import (
    build_self_collision_mask,
)


# ── Constants ──────────────────────────────────────────────────────────────
URDF_PATH = "/opt/g1_description/g1_29dof.urdf"
ASSET_ROOT = "/opt/g1_description"
ROBOT_CFG_PATH = "g1_right_arm.yml"
SPHERE_YML_PATH = "spheres_g1_arm_edited.yml"
SCENE_DIR = Path(".")

# Pelvis at origin, table below
PELVIS_HEIGHT = 0.0

TASK_SCENE_DEFAULTS = {
    "unconstrained": "scene_obstacle",
    "cupping": "scene_obstacle",
    "pouring": "scene_table_only",
    "cupping+pouring": "scene_tabletop",
}

JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

COLLISION_LINK_NAMES = [
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "right_rubber_hand",
]

TORSO_OBSTACLE_PATH = "torso_obstacle.yml"
TORSO_IGNORE_SPHERE_INDICES = [0]


# ── URDF Rendering ────────────────────────────────────────────────────────
# load_urdf, add_robot_to_scene, update_robot_pose imported from viser_utils

# ── Scene (Obstacles) ─────────────────────────────────────────────────────
# load_scene_yaml, add_scene_to_viser imported from viser_utils



# ── Collision Spheres ─────────────────────────────────────────────────────

# load_collision_spheres imported from viser_utils


def compute_sphere_cuboid_distances_np(sphere_centers, sphere_radii, cuboid_centers, cuboid_half):
    """Compute per-sphere signed distances to cuboids (CPU/numpy).
    Returns: min_distances (S,), nearest_points (S, 3), nearest_cuboid_idx (S,)
    """
    S = sphere_centers.shape[0]
    K = cuboid_centers.shape[0]
    min_distances = np.full(S, np.inf)
    nearest_points = np.zeros((S, 3))
    nearest_cuboid_idx = np.zeros(S, dtype=int)

    for s in range(S):
        sc = sphere_centers[s]
        sr = sphere_radii[s]
        for k in range(K):
            cc = cuboid_centers[k]
            ch = cuboid_half[k]
            clamped = np.clip(sc, cc - ch, cc + ch)
            delta = sc - clamped
            dist_to_surface = np.linalg.norm(delta)
            if dist_to_surface < 1e-10:
                face_dists = np.concatenate([(cc + ch) - sc, sc - (cc - ch)])
                min_face_idx = np.argmin(face_dists)
                signed_dist = -(face_dists[min_face_idx] + sr)
                nearest_pt = sc.copy()
                axis = min_face_idx % 3
                if min_face_idx < 3:
                    nearest_pt[axis] = cc[axis] + ch[axis]
                else:
                    nearest_pt[axis] = cc[axis] - ch[axis]
            else:
                signed_dist = dist_to_surface - sr
                nearest_pt = clamped
            if signed_dist < min_distances[s]:
                min_distances[s] = signed_dist
                nearest_points[s] = nearest_pt
                nearest_cuboid_idx[s] = k
    return min_distances, nearest_points, nearest_cuboid_idx


def compute_sphere_sphere_distances_np(sphere_centers, sphere_radii, ignore_mask_np):
    """Compute pairwise self-collision distances (CPU).
    Returns: min_distance, (i, j) pair, full distance matrix
    """
    S = sphere_centers.shape[0]
    diff = sphere_centers[:, None, :] - sphere_centers[None, :, :]
    center_dist = np.linalg.norm(diff, axis=-1)
    signed = center_dist - sphere_radii[:, None] - sphere_radii[None, :]
    np.fill_diagonal(signed, np.inf)
    signed[ignore_mask_np] = np.inf
    min_flat = np.argmin(signed)
    min_i, min_j = np.unravel_index(min_flat, (S, S))
    return signed[min_i, min_j], (min_i, min_j), signed


def get_all_sphere_world_positions(urdf, collision_spheres):
    """Get world positions and radii for all collision spheres.
    Returns: centers (S, 3), radii (S,), labels list
    """
    centers, radii, labels = [], [], []
    for link_name in COLLISION_LINK_NAMES:
        if link_name not in collision_spheres:
            continue
        transform = get_link_transform(urdf, link_name)
        if transform is None:
            continue
        R, t = transform[:3, :3], transform[:3, 3]
        for i, sph in enumerate(collision_spheres[link_name]):
            local_center = np.array(sph["center"])
            centers.append(R @ local_center + t)
            radii.append(sph["radius"])
            labels.append(f"{link_name}/{i}")
    return np.array(centers), np.array(radii), labels


def distance_to_color(d, threshold_safe=0.05, threshold_warn=0.005):
    """Map signed distance to RGB: green (safe) -> yellow (warn) -> red (collision)."""
    if d > threshold_safe:
        return (0, 200, 0)
    elif d > threshold_warn:
        t = (d - threshold_warn) / (threshold_safe - threshold_warn)
        r = int(255 * (1 - t))
        g = int(200 * t + 100 * (1 - t))
        return (r, g, 0)
    elif d > 0:
        return (255, 100, 0)
    else:
        return (255, 0, 0)


def add_collision_spheres_to_scene(
    server: viser.ViserServer,
    collision_spheres: dict[str, list],
) -> tuple[dict[str, list], dict[str, list]]:
    """Add collision spheres and distance labels. Returns (sphere_handles, label_handles)."""
    handles = {}
    label_handles = {}
    for link_name, spheres in collision_spheres.items():
        handles[link_name] = []
        label_handles[link_name] = []
        for i, sph in enumerate(spheres):
            center = sph["center"]
            radius = sph["radius"]
            h = server.scene.add_icosphere(
                f"/collision_spheres/{link_name}/{i}",
                radius=radius,
                color=(128, 128, 128),
                opacity=0.4,
                position=(center[0], center[1], center[2]),
            )
            handles[link_name].append(h)
            lbl = server.scene.add_label(
                f"/collision_labels/{link_name}/{i}",
                text="",
                position=(center[0], center[1], center[2]),
            )
            label_handles[link_name].append(lbl)
    return handles, label_handles


def update_collision_spheres(
    urdf: yourdfpy.URDF,
    collision_spheres: dict[str, list],
    sphere_handles: dict[str, list],
):
    """Update collision sphere positions based on current URDF config."""
    for link_name, spheres in collision_spheres.items():
        transform = get_link_transform(urdf, link_name)
        if transform is None:
            continue
        for i, sph in enumerate(spheres):
            center = np.array(sph["center"])
            world_pos = transform[:3, :3] @ center + transform[:3, 3]
            sphere_handles[link_name][i].position = world_pos
            quat = trimesh.transformations.quaternion_from_matrix(transform)
            sphere_handles[link_name][i].wxyz = (quat[0], quat[1], quat[2], quat[3])


# ── cuRobo Integration ────────────────────────────────────────────────────

def init_curobo(robot_cfg_path: str, urdf_path: str, device: str = "cuda:0"):
    """Initialize cuRobo robot model. Returns (cfg_dict, robot_cfg, robot_model, tensor_args)."""
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel

    tensor_args = TensorDeviceType(device=torch.device(device))

    with open(robot_cfg_path) as f:
        cfg_dict = yaml.safe_load(f)
    p = Path(urdf_path).parent
    cfg_dict["robot_cfg"]["kinematics"]["urdf_path"] = urdf_path
    cfg_dict["robot_cfg"]["kinematics"]["asset_root_path"] = str(p)

    robot_cfg = RobotConfig.from_dict(cfg_dict, tensor_args)
    robot_model = CudaRobotModel(robot_cfg.kinematics)

    return cfg_dict, robot_cfg, robot_model, tensor_args


def curobo_fk(robot_model, q: np.ndarray, device: str = "cuda:0"):
    """Run FK via cuRobo. Returns ee_pos (3,), ee_quat (4,) as numpy."""
    q_tensor = torch.tensor(q, dtype=torch.float32, device=device).unsqueeze(0)
    state = robot_model.get_state(q_tensor)
    ee_pos = state.ee_pose.position[0].cpu().numpy()
    ee_quat = state.ee_pose.quaternion[0].cpu().numpy()
    return ee_pos, ee_quat


def plan_trajectory(
    robot_cfg_dict: dict,
    start_q: np.ndarray,
    goal_position: np.ndarray,
    goal_quaternion: np.ndarray,
    scene_path: str,
    device: str = "cuda:0",
) -> dict:
    """Plan trajectory using cuRobo MotionGen. Returns dict with success, waypoints, dt, error."""
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import JointState
    from curobo.geom.types import Cuboid, WorldConfig
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

    tensor_args = TensorDeviceType(device=torch.device(device))

    with open(scene_path) as f:
        scene_data = yaml.safe_load(f)
    cuboids = []
    for obj in scene_data.get("cuboids", []):
        cuboids.append(Cuboid(name=obj["name"], pose=obj["pose"], dims=obj["dims"]))
    world_cfg = WorldConfig(cuboid=cuboids)

    cfg_dict = copy.deepcopy(robot_cfg_dict)
    p = Path(URDF_PATH).parent
    cfg_dict["robot_cfg"]["kinematics"]["urdf_path"] = URDF_PATH
    cfg_dict["robot_cfg"]["kinematics"]["asset_root_path"] = str(p)

    try:
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            cfg_dict, world_cfg, tensor_args=tensor_args, interpolation_dt=0.02,
        )
        motion_gen = MotionGen(motion_gen_config)
        motion_gen.warmup(warmup_js_trajopt=False)

        start_state = JointState.from_position(
            torch.tensor(start_q, dtype=torch.float32, device=device).unsqueeze(0)
        )
        goal_pose = Pose(
            position=torch.tensor(goal_position, dtype=torch.float32, device=device).unsqueeze(0),
            quaternion=torch.tensor(goal_quaternion, dtype=torch.float32, device=device).unsqueeze(0),
        )

        plan_config = MotionGenPlanConfig(enable_graph=True, max_attempts=5)
        result = motion_gen.plan_single(start_state, goal_pose, plan_config)

        if result.success.item():
            waypoints = result.interpolated_plan.position.cpu().numpy()
            return {"success": True, "waypoints": waypoints, "dt": 0.02, "error": None}
        else:
            return {"success": False, "waypoints": None, "dt": 0.02, "error": "MotionGen planning failed"}
    except Exception as e:
        return {"success": False, "waypoints": None, "dt": 0.02, "error": str(e)}


# ── GUI Setup ─────────────────────────────────────────────────────────────

def setup_gui(server: viser.ViserServer, urdf: yourdfpy.URDF) -> dict:
    """Set up GUI controls. Returns dict of GUI handles."""
    gui = {}

    with server.gui.add_folder("Scene"):
        scene_files = sorted(SCENE_DIR.glob("scene_*.yml"))
        scene_names = [f.stem for f in scene_files]
        gui["scene_dropdown"] = server.gui.add_dropdown(
            "Scene", options=scene_names, initial_value=scene_names[0] if scene_names else "scene_empty",
        )
        gui["task_dropdown"] = server.gui.add_dropdown(
            "Task",
            options=["unconstrained", "cupping", "pouring", "cupping+pouring"],
            initial_value="unconstrained",
        )
        gui["show_mesh"] = server.gui.add_checkbox("Show Robot Mesh", initial_value=True)
        gui["show_spheres"] = server.gui.add_checkbox("Show Collision Spheres", initial_value=True)
        gui["show_labels"] = server.gui.add_checkbox("Show Distance Labels", initial_value=False)
        gui["collision_info"] = server.gui.add_markdown("*Move joints to see collision info*")

    with server.gui.add_folder("Joint Control"):
        gui["joint_sliders"] = {}
        for jname in JOINT_NAMES:
            joint = urdf.joint_map.get(jname)
            if joint is not None and joint.limit is not None:
                lo = float(joint.limit.lower)
                hi = float(joint.limit.upper)
            else:
                lo, hi = -3.14, 3.14
            short_name = jname.replace("right_", "R_").replace("_joint", "")
            gui["joint_sliders"][jname] = server.gui.add_slider(
                short_name, min=lo, max=hi, step=0.01, initial_value=0.0,
            )
        gui["reset_joints_btn"] = server.gui.add_button("Reset Joints", color="gray")
        gui["ee_info"] = server.gui.add_markdown("*EE: --*")

    with server.gui.add_folder("Multi-Task Training Data"):
        # --- Scan for available multi-task dataset directories ---
        _mtd_data_root = Path("datasets/arm")
        _mtd_data_dirs = sorted([
            d.name for d in _mtd_data_root.iterdir()
            if d.is_dir() and (d / "speed_cupping.npy").exists()
        ]) if _mtd_data_root.exists() else []
        # Also include dirs with the old single speed.npy format
        _mtd_data_dirs_old = sorted([
            d.name for d in _mtd_data_root.iterdir()
            if d.is_dir() and (d / "speed.npy").exists() and d.name not in _mtd_data_dirs
        ]) if _mtd_data_root.exists() else []
        _mtd_all_dirs = _mtd_data_dirs + _mtd_data_dirs_old
        if not _mtd_all_dirs:
            _mtd_all_dirs = ["G1_mt_v2"]
        _mtd_default_dir = "G1_mt_v2" if "G1_mt_v2" in _mtd_all_dirs else _mtd_all_dirs[0]

        gui["mtd_data_dir"] = server.gui.add_dropdown(
            "Dataset Dir", options=_mtd_all_dirs,
            initial_value=_mtd_default_dir,
        )
        gui["mtd_n_show"] = server.gui.add_slider(
            "Show Points", min=10, max=5000, step=10, initial_value=500,
        )
        gui["mtd_show_mode"] = server.gui.add_dropdown(
            "Display", options=["Speed (viridis)", "Pairs (lines)"],
            initial_value="Speed (viridis)",
        )
        gui["mtd_speed_task"] = server.gui.add_dropdown(
            "Speed Task", options=["Cupping speed", "Pouring speed", "Min of both"],
            initial_value="Cupping speed",
        )
        gui["mtd_point_filter"] = server.gui.add_dropdown(
            "Show Points", options=["Start + Goal", "Start only", "Goal only"],
            initial_value="Start + Goal",
        )
        gui["mtd_pair_filter"] = server.gui.add_dropdown(
            "Pair Filter", options=["All", "Intra-cupping", "Intra-pouring", "Cross-zone"],
            initial_value="All",
        )
        gui["mtd_zone_filter"] = server.gui.add_dropdown(
            "Zone Filter", options=["All", "Cupping only", "Pouring only"],
            initial_value="All",
        )
        gui["mtd_load_btn"] = server.gui.add_button("Load from Disk", color="blue")
        gui["mtd_clear_btn"] = server.gui.add_button("Clear", color="gray")

        # --- Speed recomputation sliders (active when d_orient files are loaded) ---
        gui["mtd_alpha"] = server.gui.add_slider(
            "Alpha (orient blend)", min=0.0, max=1.0, step=0.05, initial_value=0.65,
        )
        gui["mtd_margin"] = server.gui.add_slider(
            "Margin (m)", min=0.005, max=0.15, step=0.005, initial_value=0.05,
        )
        gui["mtd_offset"] = server.gui.add_slider(
            "Offset (m)", min=0.001, max=0.02, step=0.001, initial_value=0.005,
        )
        gui["mtd_recompute_btn"] = server.gui.add_button("Recompute Speed", color="green")
        # Initially disable the sliders (enabled after loading d_orient files)
        gui["mtd_alpha"].disabled = True
        gui["mtd_margin"].disabled = True
        gui["mtd_offset"].disabled = True
        gui["mtd_recompute_btn"].disabled = True

        gui["mtd_status"] = server.gui.add_markdown("*Click Load from Disk to load training data*")

    with server.gui.add_folder("Trajectory Planning"):
        gui["plan_btn"] = server.gui.add_button("Plan Trajectory", color="green")
        gui["play_btn"] = server.gui.add_button("Play", color="blue")
        gui["stop_btn"] = server.gui.add_button("Stop", color="red")
        gui["speed_slider"] = server.gui.add_slider(
            "Playback Speed", min=0.1, max=5.0, step=0.1, initial_value=1.0,
        )
        gui["plan_stats"] = server.gui.add_markdown("*No trajectory*")

    return gui



# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NTFields-G1 Viser Web GUI")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    print("Initializing cuRobo...")
    robot_cfg_dict, robot_cfg, robot_model, tensor_args = init_curobo(
        ROBOT_CFG_PATH, URDF_PATH, args.device
    )
    print("  cuRobo ready.")

    default_scene = "scene_empty"

    print("Loading URDF for visualization...")
    urdf = load_urdf(URDF_PATH)
    print(f"  Loaded {len(urdf.joint_map)} joints, {len(urdf.link_map)} links")

    print(f"Starting viser server on port {args.port}...")
    server = viser.ViserServer(host="0.0.0.0", port=args.port)

    # Lighting
    server.scene.add_light_ambient("/light/ambient", intensity=0.4)
    server.scene.add_light_directional(
        "/light/dir", color=(255, 255, 255), intensity=0.8,
        wxyz=(0.85, 0.35, 0.35, 0.15),
    )
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", width=2.0, height=2.0, cell_size=0.1, plane="xy")

    # Robot — raise pelvis to standing height
    server.scene.add_frame(
        "/robot",
        show_axes=False,
        position=(0.0, 0.0, PELVIS_HEIGHT),
    )
    mesh_handles = add_robot_to_scene(server, urdf)
    print(f"  Added {len(mesh_handles)} mesh handles")

    # Collision spheres (with distance labels)
    collision_spheres_data = load_collision_spheres("/workspace/spheres_g1_arm_edited.yml")
    sphere_handles, sphere_label_handles = add_collision_spheres_to_scene(server, collision_spheres_data)
    print(f"  Added collision spheres for {len(collision_spheres_data)} links")

    # Scene visuals
    scene_cuboids = load_scene_yaml(f"{default_scene}.yml")
    scene_handles = add_scene_to_viser(server, scene_cuboids)

    # Zone visualization (cupping / overlap / pouring)
    add_zone_planes_to_viser(server)

    # Torso obstacle
    torso_cuboids_raw = load_scene_yaml(TORSO_OBSTACLE_PATH)
    for c in torso_cuboids_raw:
        pose = c["pose"]
        dims = c["dims"]
        server.scene.add_box(
            name=f"/torso/{c['name']}",
            dimensions=(dims[0], dims[1], dims[2]),
            position=(pose[0], pose[1], pose[2]),
            wxyz=(pose[3], pose[4], pose[5], pose[6]),
            color=(80, 80, 200),
            opacity=0.25,
        )

    # Precompute cuboid geometry (CPU, for distance visualization)
    scene_cuboid_centers_np = np.array([[c["pose"][0], c["pose"][1], c["pose"][2]] for c in scene_cuboids])
    scene_cuboid_half_np = np.array([[c["dims"][0]/2, c["dims"][1]/2, c["dims"][2]/2] for c in scene_cuboids])
    torso_centers_np = np.array([[c["pose"][0], c["pose"][1], c["pose"][2]] for c in torso_cuboids_raw])
    torso_half_np = np.array([[c["dims"][0]/2, c["dims"][1]/2, c["dims"][2]/2] for c in torso_cuboids_raw])

    # Build self-collision mask
    with open(ROBOT_CFG_PATH) as _f:
        _rcfg = yaml.safe_load(_f)
    self_coll_ignore = _rcfg["robot_cfg"]["kinematics"].get("self_collision_ignore", {})
    self_coll_mask_gpu, _ = build_self_collision_mask(
        SPHERE_YML_PATH, self_coll_ignore, COLLISION_LINK_NAMES, device=args.device
    )
    self_coll_mask_np = self_coll_mask_gpu.cpu().numpy()

    # Distance visualization handles
    min_dist_line = server.scene.add_line_segments(
        "/min_distance_line",
        points=np.zeros((1, 2, 3)),
        colors=np.array([[255, 0, 0]], dtype=np.uint8),
        line_width=3.0,
    )
    min_dist_label = server.scene.add_label("/min_distance_label", text="", position=(0, 0, 0))
    self_coll_line = server.scene.add_line_segments(
        "/self_collision_line",
        points=np.zeros((1, 2, 3)),
        colors=np.array([[200, 0, 200]], dtype=np.uint8),
        line_width=3.0,
    )
    self_coll_label = server.scene.add_label("/self_collision_label", text="", position=(0, 0, 0))
    torso_coll_line = server.scene.add_line_segments(
        "/torso_collision_line",
        points=np.zeros((1, 2, 3)),
        colors=np.array([[0, 100, 255]], dtype=np.uint8),
        line_width=3.0,
    )
    torso_coll_label = server.scene.add_label("/torso_collision_label", text="", position=(0, 0, 0))

    # GUI
    gui = setup_gui(server, urdf)

    # State
    current_scene_path = f"{default_scene}.yml"
    trajectory_line_handle = None
    planned_waypoints = None
    playing = False

    # Lock joints from robot config
    with open(ROBOT_CFG_PATH) as f:
        rcfg = yaml.safe_load(f)
    lock_joints = rcfg.get("robot_cfg", {}).get("kinematics", {}).get("lock_joints", {})

    def get_current_joint_values() -> dict[str, float]:
        vals = {}
        for jname, slider in gui["joint_sliders"].items():
            vals[jname] = slider.value
        vals.update(lock_joints)
        return vals

    def get_current_q() -> np.ndarray:
        return np.array([gui["joint_sliders"][jn].value for jn in JOINT_NAMES], dtype=np.float32)

    def update_collision_visualization():
        """Compute distances and color spheres by collision status."""
        if not gui["show_spheres"].value:
            return

        sphere_centers, sphere_radii, sphere_labels = get_all_sphere_world_positions(
            urdf, collision_spheres_data
        )
        if len(sphere_centers) == 0:
            return

        # Compute obstacle distances (only if scene has cuboids)
        if len(scene_cuboid_centers_np) > 0:
            obs_distances, nearest_pts, nearest_cuboid = compute_sphere_cuboid_distances_np(
                sphere_centers, sphere_radii, scene_cuboid_centers_np, scene_cuboid_half_np
            )
        else:
            obs_distances = np.full(len(sphere_centers), np.inf)
            nearest_pts = np.zeros_like(sphere_centers)
            nearest_cuboid = np.zeros(len(sphere_centers), dtype=int)

        # Torso distances
        torso_distances, torso_nearest_pts, _ = compute_sphere_cuboid_distances_np(
            sphere_centers, sphere_radii, torso_centers_np, torso_half_np
        )
        torso_distances_masked = torso_distances.copy()
        for si in TORSO_IGNORE_SPHERE_INDICES:
            if si < len(torso_distances_masked):
                torso_distances_masked[si] = np.inf

        # Self-collision
        self_d_min, self_pair, _ = compute_sphere_sphere_distances_np(
            sphere_centers, sphere_radii, self_coll_mask_np
        )

        # Per-sphere min distance across obstacle + torso
        distances = np.minimum(obs_distances, torso_distances_masked)
        min_idx = np.argmin(distances)

        # Overall minimums
        overall_obs_min = np.min(obs_distances) if len(obs_distances) > 0 else np.inf
        overall_torso_min = np.min(torso_distances_masked)
        overall_self_min = self_d_min

        # Update sphere colors
        s_idx = 0
        for link_name in COLLISION_LINK_NAMES:
            if link_name not in collision_spheres_data:
                continue
            for i in range(len(collision_spheres_data[link_name])):
                if s_idx >= len(distances):
                    break
                d = distances[s_idx]
                pos = sphere_centers[s_idx]

                is_self_pair = (self_d_min < d and
                                (s_idx == self_pair[0] or s_idx == self_pair[1]))
                if s_idx == min_idx and d <= self_d_min:
                    color, opacity = (255, 255, 255), 0.9
                elif is_self_pair:
                    color, opacity = (255, 0, 255), 0.9
                else:
                    color, opacity = distance_to_color(d), 0.4

                sphere_handles[link_name][i].position = pos
                sphere_handles[link_name][i].color = color
                sphere_handles[link_name][i].opacity = opacity

                if gui["show_labels"].value:
                    sphere_label_handles[link_name][i].text = f"{d*1000:.1f}mm"
                    sphere_label_handles[link_name][i].position = pos + np.array([0, 0, 0.04])
                    sphere_label_handles[link_name][i].visible = True
                else:
                    sphere_label_handles[link_name][i].visible = False
                s_idx += 1

        # Min obstacle/torso distance line
        min_d = distances[min_idx]
        min_sphere_pos = sphere_centers[min_idx]
        if torso_distances_masked[min_idx] < obs_distances[min_idx]:
            min_nearest_pt = torso_nearest_pts[min_idx]
        else:
            min_nearest_pt = nearest_pts[min_idx]
        min_dist_line.points = np.array([[min_sphere_pos, min_nearest_pt]])
        min_dist_line.colors = np.array([distance_to_color(min_d)[:3]], dtype=np.uint8)
        midpt = (min_sphere_pos + min_nearest_pt) / 2
        min_dist_label.text = f"OBS: {min_d*1000:.1f}mm"
        min_dist_label.position = midpt + np.array([0, 0, 0.02])

        # Self-collision line
        if self_d_min < np.inf:
            si, sj = self_pair
            p1, p2 = sphere_centers[si], sphere_centers[sj]
            self_coll_line.points = np.array([[p1, p2]])
            self_coll_line.colors = np.array([[200, 0, 200]], dtype=np.uint8)
            self_coll_label.text = f"SELF: {self_d_min*1000:.1f}mm"
            self_coll_label.position = (p1 + p2) / 2 + np.array([0, 0, 0.03])
        else:
            self_coll_line.points = np.zeros((1, 2, 3))
            self_coll_label.text = ""

        # Torso line
        torso_min_idx = np.argmin(torso_distances_masked)
        torso_min_d = torso_distances_masked[torso_min_idx]
        if torso_min_d < np.inf:
            tp1 = sphere_centers[torso_min_idx]
            tp2 = torso_nearest_pts[torso_min_idx]
            torso_coll_line.points = np.array([[tp1, tp2]])
            torso_color = (0, 100, 255) if torso_min_d > 0 else (255, 0, 100)
            torso_coll_line.colors = np.array([torso_color], dtype=np.uint8)
            torso_coll_label.text = f"TORSO: {torso_min_d*1000:.1f}mm"
            torso_coll_label.position = (tp1 + tp2) / 2 + np.array([0, 0, 0.02])
        else:
            torso_coll_line.points = np.zeros((1, 2, 3))
            torso_coll_label.text = ""

        # Collision status info
        obs_status = "COLL" if overall_obs_min * 1000 < 5.0 else "OK"
        torso_status = "COLL" if overall_torso_min * 1000 < 5.0 else "OK"
        self_status = "COLL" if overall_self_min * 1000 < 2.0 else "OK"
        overall = "COLLISION" if "COLL" in (obs_status, torso_status, self_status) else "SAFE"
        status_icon = "\U0001f534" if overall == "COLLISION" else "\U0001f7e2"
        gui["collision_info"].content = (
            f"## {status_icon} {overall}\n"
            f"| Check | Dist | Status |\n|---|---|---|\n"
            f"| Obstacle | {overall_obs_min*1000:.1f}mm | {obs_status} |\n"
            f"| Torso | {overall_torso_min*1000:.1f}mm | {torso_status} |\n"
            f"| Self | {overall_self_min*1000:.1f}mm | {self_status} |"
        )

    def on_joint_change(_):
        joint_values = get_current_joint_values()
        update_robot_pose(urdf, mesh_handles, joint_values)
        # Update collision sphere positions and colors
        update_collision_visualization()
        # Update EE info
        q = get_current_q()
        ee_pos, ee_quat = curobo_fk(robot_model, q, args.device)
        gui["ee_info"].content = f"**EE:** ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})"

    for slider in gui["joint_sliders"].values():
        slider.on_update(on_joint_change)

    @gui["reset_joints_btn"].on_click
    def _(_):
        for slider in gui["joint_sliders"].values():
            slider.value = 0.0
        on_joint_change(None)

    @gui["show_mesh"].on_update
    def _(_):
        visible = gui["show_mesh"].value
        for h in mesh_handles.values():
            h.visible = visible

    @gui["show_spheres"].on_update
    def _(_):
        visible = gui["show_spheres"].value
        for link_handles in sphere_handles.values():
            for h in link_handles:
                h.visible = visible
        for link_labels in sphere_label_handles.values():
            for lbl in link_labels:
                lbl.visible = visible and gui["show_labels"].value
        # Hide distance lines when spheres hidden
        if not visible:
            min_dist_line.points = np.zeros((1, 2, 3))
            min_dist_label.text = ""
            self_coll_line.points = np.zeros((1, 2, 3))
            self_coll_label.text = ""
            torso_coll_line.points = np.zeros((1, 2, 3))
            torso_coll_label.text = ""
        else:
            update_collision_visualization()

    @gui["show_labels"].on_update
    def _(_):
        update_collision_visualization()

    @gui["scene_dropdown"].on_update
    def _(event):
        nonlocal scene_handles, current_scene_path, scene_cuboids
        nonlocal scene_cuboid_centers_np, scene_cuboid_half_np
        for h in scene_handles:
            h.remove()
        scene_name = gui["scene_dropdown"].value
        current_scene_path = f"{scene_name}.yml"
        scene_cuboids = load_scene_yaml(current_scene_path)
        scene_handles = add_scene_to_viser(server, scene_cuboids)
        # Update cuboid geometry for collision visualization
        if len(scene_cuboids) > 0:
            scene_cuboid_centers_np = np.array([[c["pose"][0], c["pose"][1], c["pose"][2]] for c in scene_cuboids])
            scene_cuboid_half_np = np.array([[c["dims"][0]/2, c["dims"][1]/2, c["dims"][2]/2] for c in scene_cuboids])
        else:
            scene_cuboid_centers_np = np.zeros((0, 3))
            scene_cuboid_half_np = np.zeros((0, 3))
        update_collision_visualization()
        print(f"  Scene switched: {scene_name}")

    # Goal pose transform controls
    goal_tf = server.scene.add_transform_controls(
        "/goal_pose", scale=0.15, depth_test=True, opacity=0.8,
    )
    goal_tf.position = (0.3, -0.15, -0.0)
    server.scene.add_frame(
        "/goal_pose/frame", show_axes=True, axes_length=0.08, axes_radius=0.004,
    )

    # Task dropdown callback
    @gui["task_dropdown"].on_update
    def _(_):
        task = gui["task_dropdown"].value
        if task in TASK_SCENE_DEFAULTS:
            default_scene = TASK_SCENE_DEFAULTS[task]
            if default_scene in gui["scene_dropdown"].options:
                gui["scene_dropdown"].value = default_scene

    # ── Multi-Task Training Data ────────────────────────────────────────
    mtd_state = None  # training data cache
    mtd_point_handle = None
    mtd_line_handle = None

    def _height_to_viridis(h_norm: np.ndarray) -> np.ndarray:
        """Map normalized height [0,1] to viridis RGB (N,3) uint8."""
        from matplotlib import colormaps
        t = np.clip(h_norm, 0.0, 1.0)
        rgba = colormaps["viridis"](t)
        return (rgba[:, :3] * 255).astype(np.uint8)

    def _mtd_update_viz():
        """Show training data points colored by speed or pairs as lines."""
        nonlocal mtd_point_handle, mtd_line_handle
        if mtd_point_handle is not None:
            mtd_point_handle.remove()
            mtd_point_handle = None
        if mtd_line_handle is not None:
            mtd_line_handle.remove()
            mtd_line_handle = None
        if mtd_state is None:
            return

        n_pairs = mtd_state["n_pairs"]

        # --- Zone filter (legacy: uses per-config zone labels) ---
        zone_filter = gui["mtd_zone_filter"].value
        if "zones" in mtd_state:
            zones = mtd_state["zones"]
            starts = mtd_state["starts"]
            goals = mtd_state["goals"]
            if zone_filter == "Cupping only":
                zone_mask = (zones[starts] == 0) | (zones[goals] == 0)
            elif zone_filter == "Pouring only":
                zone_mask = (zones[starts] == 1) | (zones[goals] == 1)
            else:
                zone_mask = np.ones(n_pairs, dtype=bool)
        elif "zone_id" in mtd_state:
            # Loaded from disk: zone_id is (N,2) per-pair
            zone_id = mtd_state["zone_id"]
            if zone_filter == "Cupping only":
                zone_mask = (zone_id[:, 0] == 0) | (zone_id[:, 1] == 0)
            elif zone_filter == "Pouring only":
                zone_mask = (zone_id[:, 0] == 1) | (zone_id[:, 1] == 1)
            else:
                zone_mask = np.ones(n_pairs, dtype=bool)
        else:
            zone_mask = np.ones(n_pairs, dtype=bool)

        # --- Pair type filter (only for disk-loaded data with pair_type) ---
        pair_filter = gui["mtd_pair_filter"].value
        if "pair_type" in mtd_state and pair_filter != "All":
            pt = mtd_state["pair_type"]
            if pair_filter == "Intra-cupping":
                type_mask = pt == 0
            elif pair_filter == "Intra-pouring":
                type_mask = pt == 1
            elif pair_filter == "Cross-zone":
                type_mask = pt == 2
            else:
                type_mask = np.ones(n_pairs, dtype=bool)
        else:
            type_mask = np.ones(n_pairs, dtype=bool)

        pair_mask = zone_mask & type_mask
        valid_idx = np.where(pair_mask)[0]
        if len(valid_idx) == 0:
            return
        n_show = min(int(gui["mtd_n_show"].value), len(valid_idx))
        idx = np.random.choice(valid_idx, n_show, replace=False) if n_show < len(valid_idx) else valid_idx

        mode = gui["mtd_show_mode"].value
        point_filter = gui["mtd_point_filter"].value

        # --- Select speed source based on mtd_speed_task dropdown ---
        speed_task = gui["mtd_speed_task"].value
        if speed_task == "Pouring speed" and "speed_pouring_start" in mtd_state:
            spd_start_arr = mtd_state["speed_pouring_start"]
            spd_goal_arr = mtd_state["speed_pouring_goal"]
        elif speed_task == "Min of both" and "speed_pouring_start" in mtd_state:
            spd_start_arr = np.minimum(
                mtd_state["speed_cupping_start"], mtd_state["speed_pouring_start"]
            )
            spd_goal_arr = np.minimum(
                mtd_state["speed_cupping_goal"], mtd_state["speed_pouring_goal"]
            )
        elif "speed_cupping_start" in mtd_state:
            # Cupping speed (default) or fallback when multi-task keys exist
            spd_start_arr = mtd_state["speed_cupping_start"]
            spd_goal_arr = mtd_state["speed_cupping_goal"]
        else:
            # Legacy single-speed format
            spd_start_arr = mtd_state["speed_start"]
            spd_goal_arr = mtd_state["speed_goal"]

        if "Speed" in mode:
            # Collect points based on start/goal filter
            pts_list = []
            spd_list = []
            if point_filter in ("Start + Goal", "Start only"):
                pts_list.append(mtd_state["ee_start"][idx])
                spd_list.append(spd_start_arr[idx])
            if point_filter in ("Start + Goal", "Goal only"):
                pts_list.append(mtd_state["ee_goal"][idx])
                spd_list.append(spd_goal_arr[idx])
            pts = np.concatenate(pts_list)
            spd = np.concatenate(spd_list)
            colors = _height_to_viridis(spd)  # viridis: 0=purple(slow), 1=yellow(fast)
            mtd_point_handle = server.scene.add_point_cloud(
                "/mtd/points", points=pts.astype(np.float32),
                colors=colors, point_size=0.005, point_shape="rounded",
            )
        else:
            # Pairs as lines
            ee_start = mtd_state["ee_start"][idx]
            ee_goal = mtd_state["ee_goal"][idx]
            points = np.zeros((len(idx), 2, 3), dtype=np.float32)
            points[:, 0, :] = ee_start
            points[:, 1, :] = ee_goal
            line_colors = np.full((len(idx), 3), [100, 200, 255], dtype=np.uint8)
            mtd_line_handle = server.scene.add_line_segments(
                "/mtd/lines", points=points, colors=line_colors, line_width=1.5,
            )

    @gui["mtd_load_btn"].on_click
    def _(_):
        nonlocal mtd_state
        dir_name = gui["mtd_data_dir"].value
        data_dir = Path("datasets/arm") / dir_name
        if not data_dir.exists():
            gui["mtd_status"].content = f"**Error:** Directory `{data_dir}` not found"
            return

        # Check for required files
        ee_path = data_dir / "ee_positions.npy"
        sp_path = data_dir / "sampled_points.npy"
        if not ee_path.exists() or not sp_path.exists():
            gui["mtd_status"].content = f"**Error:** Missing ee_positions.npy or sampled_points.npy in `{data_dir}`"
            return

        gui["mtd_status"].content = f"*Loading from `{data_dir}`...*"

        try:
            ee_positions = np.load(ee_path)       # (N, 6)
            sampled_points = np.load(sp_path)     # (N, 14)
            N = len(ee_positions)

            # Build base mtd_state
            mtd_state = {
                "n_pairs": N,
                "ee_start": ee_positions[:, :3],
                "ee_goal": ee_positions[:, 3:],
                "starts": np.arange(N),
                "goals": np.arange(N),
                "from_disk": True,
            }

            # Load multi-task speed files (cupping / pouring)
            speed_cup_path = data_dir / "speed_cupping.npy"
            speed_pour_path = data_dir / "speed_pouring.npy"
            if speed_cup_path.exists():
                speed_cupping = np.load(speed_cup_path)  # (N, 2)
                mtd_state["speed_cupping_start"] = speed_cupping[:, 0]
                mtd_state["speed_cupping_goal"] = speed_cupping[:, 1]
                # Default speed_start/goal to cupping for backward compat
                mtd_state["speed_start"] = speed_cupping[:, 0]
                mtd_state["speed_goal"] = speed_cupping[:, 1]
            if speed_pour_path.exists():
                speed_pouring = np.load(speed_pour_path)  # (N, 2)
                mtd_state["speed_pouring_start"] = speed_pouring[:, 0]
                mtd_state["speed_pouring_goal"] = speed_pouring[:, 1]

            # Fallback: old single-speed format
            speed_path = data_dir / "speed.npy"
            if "speed_start" not in mtd_state and speed_path.exists():
                speed = np.load(speed_path)  # (N, 2)
                mtd_state["speed_start"] = speed[:, 0]
                mtd_state["speed_goal"] = speed[:, 1]

            # Load zone_id and pair_type if available
            zone_id_path = data_dir / "zone_id.npy"
            if zone_id_path.exists():
                mtd_state["zone_id"] = np.load(zone_id_path)  # (N, 2) int8

            pair_type_path = data_dir / "pair_type.npy"
            if pair_type_path.exists():
                mtd_state["pair_type"] = np.load(pair_type_path)  # (N,) int8

            # Load raw distance files for speed recomputation
            d_obs_path = data_dir / "d_obstacle.npy"
            d_self_path = data_dir / "d_self.npy"
            d_orient_cup_path = data_dir / "d_orient_cupping.npy"
            d_orient_pour_path = data_dir / "d_orient_pouring.npy"
            if d_obs_path.exists():
                mtd_state["d_obstacle"] = np.load(d_obs_path)  # (N, 2)
            if d_self_path.exists():
                mtd_state["d_self"] = np.load(d_self_path)  # (N, 2)
            if d_orient_cup_path.exists():
                mtd_state["d_orient_cupping"] = np.load(d_orient_cup_path)  # (N, 2)
            if d_orient_pour_path.exists():
                mtd_state["d_orient_pouring"] = np.load(d_orient_pour_path)  # (N, 2)

            # Enable/disable speed recomputation sliders
            has_orient = "d_orient_cupping" in mtd_state and "d_obstacle" in mtd_state
            gui["mtd_alpha"].disabled = not has_orient
            gui["mtd_margin"].disabled = not has_orient
            gui["mtd_offset"].disabled = not has_orient
            gui["mtd_recompute_btn"].disabled = not has_orient

            # Set slider defaults from pipeline_config if available
            config_path_pre = data_dir / "pipeline_config.json"
            if has_orient and config_path_pre.exists():
                with open(config_path_pre, "r") as f:
                    cfg_pre = json.load(f)
                if "alpha" in cfg_pre:
                    gui["mtd_alpha"].value = cfg_pre["alpha"]
                if "margin" in cfg_pre:
                    gui["mtd_margin"].value = cfg_pre["margin"]
                if "offset" in cfg_pre:
                    gui["mtd_offset"].value = cfg_pre["offset"]

            # Build status message
            status_parts = [f"**Loaded {N:,} pairs** from `{data_dir}`"]

            # Try loading pipeline_config.json for extra stats
            config_path = data_dir / "pipeline_config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    cfg = json.load(f)
                if "margin" in cfg:
                    status_parts.append(f"- Margin: {cfg['margin']*1000:.0f}mm")
                if "alpha" in cfg:
                    status_parts.append(f"- Alpha: {cfg['alpha']:.2f}")
                if "n_configs_cupping" in cfg:
                    status_parts.append(f"- Cupping configs: {cfg['n_configs_cupping']:,}")
                if "n_configs_pouring" in cfg:
                    status_parts.append(f"- Pouring configs: {cfg['n_configs_pouring']:,}")

            # Pair type stats
            if "pair_type" in mtd_state:
                pt = mtd_state["pair_type"]
                n_cup = int((pt == 0).sum())
                n_pour = int((pt == 1).sum())
                n_cross = int((pt == 2).sum())
                status_parts.append(f"- Intra-cupping: {n_cup:,}, Intra-pouring: {n_pour:,}, Cross: {n_cross:,}")

            # Speed range
            if "speed_start" in mtd_state:
                all_spd = np.concatenate([mtd_state["speed_start"], mtd_state["speed_goal"]])
                status_parts.append(f"- Speed range: [{all_spd.min():.3f}, {all_spd.max():.3f}]")

            gui["mtd_status"].content = "\n\n".join(status_parts)

            _mtd_update_viz()

        except Exception as e:
            gui["mtd_status"].content = f"**Error loading:** {e}"

    @gui["mtd_clear_btn"].on_click
    def _(_):
        nonlocal mtd_state, mtd_point_handle, mtd_line_handle
        if mtd_point_handle is not None:
            mtd_point_handle.remove()
            mtd_point_handle = None
        if mtd_line_handle is not None:
            mtd_line_handle.remove()
            mtd_line_handle = None
        mtd_state = None
        gui["mtd_alpha"].disabled = True
        gui["mtd_margin"].disabled = True
        gui["mtd_offset"].disabled = True
        gui["mtd_recompute_btn"].disabled = True
        gui["mtd_status"].content = "*Generate FK samples first*"

    def _mtd_recompute_speed():
        """Recompute speed from raw distances using current slider values."""
        if mtd_state is None:
            return
        if "d_obstacle" not in mtd_state or "d_orient_cupping" not in mtd_state:
            return

        alpha = gui["mtd_alpha"].value
        margin = gui["mtd_margin"].value
        offset = gui["mtd_offset"].value

        # d_phys = min(d_obstacle, d_self) per endpoint, matching pipeline logic
        d_obs = mtd_state["d_obstacle"]  # (N, 2)
        if "d_self" in mtd_state:
            d_phys = np.minimum(d_obs, mtd_state["d_self"])
        else:
            d_phys = d_obs

        # Recompute cupping speed: clip(alpha * d_orient + (1-alpha) * d_phys, offset, margin) / margin
        d_orient_cup = mtd_state["d_orient_cupping"]  # (N, 2)
        d_combined_cup = alpha * d_orient_cup + (1.0 - alpha) * d_phys
        speed_cup = np.clip(d_combined_cup, offset, margin) / margin

        mtd_state["speed_cupping_start"] = speed_cup[:, 0]
        mtd_state["speed_cupping_goal"] = speed_cup[:, 1]
        mtd_state["speed_start"] = speed_cup[:, 0]
        mtd_state["speed_goal"] = speed_cup[:, 1]

        # Recompute pouring speed if orientation distances are available
        if "d_orient_pouring" in mtd_state:
            d_orient_pour = mtd_state["d_orient_pouring"]  # (N, 2)
            d_combined_pour = alpha * d_orient_pour + (1.0 - alpha) * d_phys
            speed_pour = np.clip(d_combined_pour, offset, margin) / margin
            mtd_state["speed_pouring_start"] = speed_pour[:, 0]
            mtd_state["speed_pouring_goal"] = speed_pour[:, 1]

        # Update status with new speed stats
        all_cup = np.concatenate([speed_cup[:, 0], speed_cup[:, 1]])
        n_slow = (all_cup < 1.0).sum()
        status = (
            f"**Recomputed** alpha={alpha:.2f}, margin={margin*1000:.0f}mm, offset={offset*1000:.0f}mm\n\n"
            f"- Cupping speed: [{all_cup.min():.3f}, {all_cup.max():.3f}]\n\n"
            f"- Near-obstacle (speed<1): {n_slow}/{len(all_cup)} ({100*n_slow/len(all_cup):.1f}%)"
        )
        if "d_orient_pouring" in mtd_state:
            all_pour = np.concatenate([speed_pour[:, 0], speed_pour[:, 1]])
            n_slow_p = (all_pour < 1.0).sum()
            status += (
                f"\n\n- Pouring speed: [{all_pour.min():.3f}, {all_pour.max():.3f}]"
                f"\n\n- Near-obstacle (speed<1): {n_slow_p}/{len(all_pour)} ({100*n_slow_p/len(all_pour):.1f}%)"
            )
        gui["mtd_status"].content = status

        _mtd_update_viz()

    @gui["mtd_recompute_btn"].on_click
    def _(_):
        _mtd_recompute_speed()

    @gui["mtd_show_mode"].on_update
    def _(_):
        _mtd_update_viz()

    @gui["mtd_n_show"].on_update
    def _(_):
        _mtd_update_viz()

    @gui["mtd_point_filter"].on_update
    def _(_):
        _mtd_update_viz()

    @gui["mtd_zone_filter"].on_update
    def _(_):
        _mtd_update_viz()

    @gui["mtd_speed_task"].on_update
    def _(_):
        _mtd_update_viz()

    @gui["mtd_pair_filter"].on_update
    def _(_):
        _mtd_update_viz()

    @gui["plan_btn"].on_click
    def _(_):
        nonlocal planned_waypoints, trajectory_line_handle
        gui["plan_stats"].content = "*Planning...*"

        start_q = get_current_q()
        goal_pos = np.array(goal_tf.position, dtype=np.float32)
        goal_quat = np.array(goal_tf.wxyz, dtype=np.float32)

        result = plan_trajectory(
            robot_cfg_dict, start_q, goal_pos, goal_quat,
            current_scene_path, args.device,
        )

        if result["success"]:
            planned_waypoints = result["waypoints"]
            n_wp = planned_waypoints.shape[0]
            total_time = n_wp * result["dt"]

            ee_path = []
            for i in range(0, n_wp, max(1, n_wp // 200)):
                pos, _ = curobo_fk(robot_model, planned_waypoints[i], args.device)
                ee_path.append(pos)
            ee_path = np.array(ee_path, dtype=np.float32)

            if trajectory_line_handle is not None:
                trajectory_line_handle.remove()

            if len(ee_path) > 1:
                points = np.zeros((2 * (len(ee_path) - 1), 3), dtype=np.float32)
                for i in range(len(ee_path) - 1):
                    points[2*i] = ee_path[i]
                    points[2*i+1] = ee_path[i+1]
                trajectory_line_handle = server.scene.add_line_segments(
                    "/trajectory", points=points, colors=(0, 200, 255), line_width=3.0,
                )

            gui["plan_stats"].content = (
                f"**Plan OK**\n\n"
                f"- Waypoints: {n_wp}\n"
                f"- Duration: {total_time:.2f}s\n"
                f"- Press Play to animate"
            )
        else:
            planned_waypoints = None
            gui["plan_stats"].content = f"**Planning failed**\n\n{result['error']}"

    @gui["play_btn"].on_click
    def _(_):
        nonlocal playing
        if planned_waypoints is None:
            gui["plan_stats"].content = "*No trajectory to play*"
            return
        playing = True
        speed = gui["speed_slider"].value
        dt = 0.02 / speed
        n_wp = planned_waypoints.shape[0]

        for i in range(n_wp):
            if not playing:
                break
            for j, jname in enumerate(JOINT_NAMES):
                gui["joint_sliders"][jname].value = float(planned_waypoints[i, j])
            on_joint_change(None)
            time.sleep(dt)

        playing = False

    @gui["stop_btn"].on_click
    def _(_):
        nonlocal playing
        playing = False

    # Initial pose
    on_joint_change(None)

    print(f"\n  GUI ready at http://localhost:{args.port}")
    print("  Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
