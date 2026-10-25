"""
UR5e NTField multi-task evaluation: MPPI planning with constraint-aware routing.

Loads a dual-head MultiNN model and runs MPPI with routing info (ee_y/ee_xy,
d_orient_cup, d_orient_pour) computed at each step via FK.

Usage:
    python ur5e_plan_eval_multi.py \
        --checkpoint Experiments/multi/UR5e_R02_slerp_balanced1M/.../Best_Model.pt \
        --scene scene_ur5e.yml \
        --norm_file datasets/arm/UR5e_multi_balanced_1M/joint_normalization.npz \
        --eval_set datasets/arm/eval_ur5e_multi_v1_100.npz
"""

import sys
sys.path.append('.')

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_NTRL_ROOT = _REPO_ROOT / "ntrl"
if str(_NTRL_ROOT) not in sys.path:
    sys.path.insert(0, str(_NTRL_ROOT))

# Multi-task model imports
from train.train_g1_multi import MultiModel

# cuRobo imports
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

from collision_utils import (
    load_cuboid_geometry,
    check_collision_full,
    check_trajectory_collision_full as check_traj_collision_full,
    build_self_collision_mask,
    sphere_cuboid_distance,
    sphere_sphere_distance,
)
from constraint_utils import (
    _quat_wxyz_to_rotmat,
)

try:
    from models.metric_arm.zone_router import (
        combine_overlap_tilt_pass_torch,
        corner_masks_torch,
        load_zone_config,
    )
except ImportError:
    from metric_arm.zone_router import (
        combine_overlap_tilt_pass_torch,
        corner_masks_torch,
        load_zone_config,
    )
from ur5e_task_axes import (
    axis_alignment_from_quat,
    axis_label,
    load_task_axes,
    task_axis_arrays,
    tilt_deg_from_quat,
)
from mppi_update_utils import budget_feasible_update, constant_arc_update
from morph_orient import corner_blend_param
from tabletop_eval_groups import derive_query_groups


DOF = 6

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
TORSO_OBSTACLE_PATH = None
TORSO_IGNORE_SPHERE_INDICES = []
SPHERE_YML_PATH = "spheres_ur5e_arm.yml"

FK_FEATURE_NAMES = [
    "ee_x",
    "ee_y",
    "ee_z",
    "cup_axis_dot",
    "pour_axis_dot",
    "cup_tilt_deg_norm",
    "pour_tilt_deg_norm",
    "tilt_handover_ratio",
]


def _np_scalar(value):
    return np.asarray(value).reshape(()).item()


def _load_eval_metadata(eval_set):
    if eval_set is None:
        return {}
    data = np.load(eval_set, allow_pickle=True)
    meta = {}
    for key in (
        "pair_type", "ee_y_goal", "start_zone", "goal_zone",
        "direct_line_strict_feasible", "query_group_label", "scenario_label",
        "cup_tilt_deg", "pour_tilt_deg", "corner_overlap_deg",
        "handover_accept", "handover_mono_tol", "overlap_logic",
        "task_axes_json",
    ):
        if key in data.files:
            meta[key] = data[key]
    return meta


def _parse_json_scalar(value):
    if value is None:
        return None
    raw = _np_scalar(value) if np.asarray(value).shape == () else value
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def _axes_close(a, b, atol=1e-6):
    for task in ("cup", "pour"):
        for key in ("local_axis", "world_axis"):
            av = np.asarray(a[task][key], dtype=np.float64)
            bv = np.asarray(b[task][key], dtype=np.float64)
            if not np.allclose(av, bv, atol=atol):
                return False
    return True


def _sync_eval_args_from_metadata(args, eval_meta, task_axes):
    """Keep fixed-eval validation parameters tied to the evalset provenance."""
    if not eval_meta:
        return

    eval_axes = _parse_json_scalar(eval_meta.get("task_axes_json"))
    if eval_axes is not None and not _axes_close(eval_axes, task_axes):
        raise ValueError(
            "Evalset task_axes_json does not match scene task_axes. "
            f"eval={eval_axes}, scene={task_axes}")

    if args.zone_eval:
        if "cup_tilt_deg" in eval_meta:
            args.cup_eval_tilt = float(_np_scalar(eval_meta["cup_tilt_deg"]))
        if "pour_tilt_deg" in eval_meta:
            args.pour_eval_tilt = float(_np_scalar(eval_meta["pour_tilt_deg"]))
        if "corner_overlap_deg" in eval_meta:
            args.corner_overlap_deg = float(_np_scalar(eval_meta["corner_overlap_deg"]))
        if "handover_mono_tol" in eval_meta:
            args.handover_mono_tol = float(_np_scalar(eval_meta["handover_mono_tol"]))
        if "handover_accept" in eval_meta:
            args.handover_accept = bool(_np_scalar(eval_meta["handover_accept"]))
        print("  Fixed eval metadata synced:"
              f" cup_eval_tilt={args.cup_eval_tilt},"
              f" pour_eval_tilt={args.pour_eval_tilt},"
              f" corner_overlap_deg={args.corner_overlap_deg},"
              f" handover_accept={args.handover_accept},"
              f" handover_mono_tol={args.handover_mono_tol}")


class _PureHJBNetworkAdapter(torch.nn.Module):
    """Expose a pure one-output HJB checkpoint to the existing MPPI loop."""

    def __init__(self, value_model):
        super().__init__()
        self.value_model = value_model
        self.pure_hjb = True
        self.boundary_conditioned = bool(
            getattr(value_model, "boundary_conditioned", False))
        self.endpoint_speed_conditioning = bool(
            getattr(value_model, "endpoint_speed_conditioning", False))
        self.fk_feature_dim = 0
        self.lifted_phase = False
        self.use_orient_head = False
        self.add_orient_to_tau = False
        self.y_divide = 0.0
        self.overlap = 0.0
        self.route_temp = 1.0
        self.weight_mode = "single_scalar_value"
        self.zone_mode = "corner_table"
        self.corner_zones = None
        self.corner_spatial_temp = 0.05

    def out(
            self, endpoint_pairs, boundary_context=None,
            scalar_speed=None, **_unused):
        if endpoint_pairs.ndim != 2 or endpoint_pairs.shape[1] != 2 * DOF:
            raise ValueError(
                f"pure HJB input must have shape (B, {2 * DOF})")
        if self.boundary_conditioned:
            if boundary_context is None:
                raise ValueError(
                    "boundary-conditioned HJB requires immutable context")
            if self.endpoint_speed_conditioning:
                if scalar_speed is None:
                    raise ValueError(
                        "speed-conditioned HJB requires exact scalar speed")
                scalar_speed = torch.as_tensor(
                    scalar_speed,
                    device=endpoint_pairs.device,
                    dtype=endpoint_pairs.dtype)
                if scalar_speed.shape == (2 * len(endpoint_pairs),):
                    state_speed = scalar_speed[:len(endpoint_pairs)]
                    goal_speed = scalar_speed[len(endpoint_pairs):]
                elif scalar_speed.shape == (len(endpoint_pairs), 2):
                    state_speed = scalar_speed[:, 0]
                    goal_speed = scalar_speed[:, 1]
                else:
                    raise ValueError(
                        "scalar speed must have shape (2B,) or (B, 2)")
                value = self.value_model(
                    endpoint_pairs[:, :DOF],
                    endpoint_pairs[:, DOF:],
                    boundary_context,
                    state_speed,
                    goal_speed)
            else:
                value = self.value_model(
                    endpoint_pairs[:, :DOF],
                    endpoint_pairs[:, DOF:],
                    boundary_context)
        else:
            value = self.value_model(
                endpoint_pairs[:, :DOF], endpoint_pairs[:, DOF:])
        # Only the first item is consumed by raw learned-tau MPPI.  The
        # placeholders preserve the historical interface without introducing
        # another score, head, or routing decision.
        return value, None, endpoint_pairs, None, None

    def compute_window_weights(
            self, ee_y, d_orient_cup, d_orient_pour, ee_xy=None):
        del d_orient_cup, d_orient_pour, ee_xy
        one = torch.ones_like(ee_y)
        return one, torch.zeros_like(one)


class _PureHJBModelAdapter:
    def __init__(self, value_model):
        self.dim = DOF
        self.network = _PureHJBNetworkAdapter(value_model)


def _load_pure_hjb_model(checkpoint_path, device):
    """Return a pure adapter when the file declares the auditable contract."""
    payload = torch.load(
        checkpoint_path, map_location=device, weights_only=False)
    model_class = payload.get("model_class")
    if model_class not in (
            "SymmetricGoalValue",
            "GeodesicEmbeddingValue",
            "BoundaryConditionedGoalValue"):
        return None
    contract = payload.get("training_contract", {})
    if int(contract.get("learned_models", -1)) != 1:
        raise ValueError("pure HJB checkpoint must declare learned_models=1")
    if contract.get("learned_outputs") != [
            "scalar_goal_conditioned_value"]:
        raise ValueError(
            "pure HJB checkpoint must have one scalar value output")
    from pure_ur5e_hjb import (
        GeodesicEmbeddingValue,
        SymmetricGoalValue,
    )
    if model_class == "BoundaryConditionedGoalValue":
        from pure_ur5e_context_hjb import (
            BoundaryConditionedGoalValue,
        )
        model_type = BoundaryConditionedGoalValue
    elif model_class == "SymmetricGoalValue":
        model_type = SymmetricGoalValue
    else:
        model_type = GeodesicEmbeddingValue
    value_model = model_type(**payload["model_kwargs"]).to(device)
    value_model.load_state_dict(payload["model_state_dict"], strict=True)
    value_model.eval()
    return _PureHJBModelAdapter(value_model)


def _augment_lifted_phase_pairs(physical_pairs, model, pair_types=None,
                                start_zones=None, goal_zones=None):
    """Attach deterministic boundary states for one lifted scalar field.

    The seventh coordinate is part of the model state, not an inference-time
    score or rescue.  Cupping endpoints use p=0, pouring endpoints p=1, and
    cross endpoints follow their declared start/goal zones.
    """
    network = model.network
    if not getattr(network, "lifted_phase", False):
        return physical_pairs
    physical_dim = int(network.lifted_phase_physical_dim)
    model_dim = int(model.dim)
    if model_dim != physical_dim + 1:
        raise ValueError(
            "lifted checkpoint must have exactly one non-physical coordinate")
    if physical_pairs.ndim != 2 or physical_pairs.shape[1] != 2 * physical_dim:
        raise ValueError(
            f"physical pairs must have width {2 * physical_dim}, got "
            f"{tuple(physical_pairs.shape)}")

    batch = physical_pairs.shape[0]
    device = physical_pairs.device
    dtype = physical_pairs.dtype
    if start_zones is not None and goal_zones is not None:
        p_start = torch.as_tensor(start_zones, device=device, dtype=dtype)
        p_goal = torch.as_tensor(goal_zones, device=device, dtype=dtype)
    elif pair_types is not None:
        pair_types = torch.as_tensor(pair_types, device=device)
        # Evaluation cross pairs are cup -> pour unless explicit endpoint-zone
        # metadata above says otherwise.
        p_start = (pair_types == 1).to(dtype)
        p_goal = (pair_types != 0).to(dtype)
    else:
        raise ValueError(
            "lifted phase evaluation requires pair_type or endpoint zones")
    if p_start.numel() != batch or p_goal.numel() != batch:
        raise ValueError("lifted phase metadata length does not match batch")
    if (not torch.all((p_start == 0) | (p_start == 1))
            or not torch.all((p_goal == 0) | (p_goal == 1))):
        raise ValueError("lifted phase endpoint zones must be binary")

    phase_scale = float(network.lifted_phase_scale)
    z_start = phase_scale * (p_start.reshape(batch, 1) - 0.5)
    z_goal = phase_scale * (p_goal.reshape(batch, 1) - 0.5)
    return torch.cat([
        physical_pairs[:, :physical_dim], z_start,
        physical_pairs[:, physical_dim:], z_goal,
    ], dim=1)


def load_scene_from_yaml(path: str) -> WorldConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    cuboids = []
    for obj in data.get("cuboids", []):
        cuboids.append(Cuboid(
            name=obj["name"],
            pose=obj["pose"],
            dims=obj["dims"],
        ))
    return WorldConfig(cuboid=cuboids)


def setup_robot_world(robot_cfg_path: str, scene: WorldConfig, device: str = "cuda:0",
                      urdf_dir: str = None):
    with open(robot_cfg_path) as f:
        robot_cfg_dict = yaml.safe_load(f)
    if urdf_dir:
        from pathlib import Path as P
        p = P(urdf_dir)
        urdf_file = p / "urdf" / "ur5e_robotiq_2f_140.urdf"
        if not urdf_file.exists():
            urdf_file = p / "ur5e_robotiq_2f_140.urdf"
        robot_cfg_dict["robot_cfg"]["kinematics"]["urdf_path"] = str(urdf_file)
        robot_cfg_dict["robot_cfg"]["kinematics"]["asset_root_path"] = str(p)
    tensor_args = TensorDeviceType(device=torch.device(device))
    robot_cfg = RobotConfig.from_dict(robot_cfg_dict, tensor_args)
    robot_world_cfg = RobotWorldConfig.load_from_config(
        robot_cfg, world_model=scene, tensor_args=tensor_args,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
    )
    robot_world = RobotWorld(robot_world_cfg)
    robot_model = CudaRobotModel(robot_cfg.kinematics)
    return robot_world, robot_model


# ── Orientation distance replication (must match pkl_to_training_data_multi.py) ──

def cupping_orient_distance(q_batch, robot_model, margin, tilt_max, tilt_soft,
                            task_axes=None):
    """UR5e cupping orientation distance from scene task_axes."""
    state = robot_model.get_state(q_batch)
    quat = state.ee_pose.quaternion
    local_axis, world_axis = task_axis_arrays(task_axes or load_task_axes(None), "cup")
    tilt = tilt_deg_from_quat(quat, local_axis, world_axis)
    excess = (tilt - tilt_max).clamp(min=0.0)
    d_orient = margin * (1.0 - excess / tilt_soft)
    return d_orient.clamp(min=-margin)


def pouring_orient_distance(q_batch, robot_model, margin, tilt_max, tilt_soft,
                            rzy_safe=0.15, rzy_soft=0.30,
                            task_axes=None):
    """UR5e pouring orientation distance from scene task_axes."""
    state = robot_model.get_state(q_batch)
    quat = state.ee_pose.quaternion
    local_axis, world_axis = task_axis_arrays(task_axes or load_task_axes(None), "pour")
    x_tilt = tilt_deg_from_quat(quat, local_axis, world_axis)
    x_excess = (x_tilt - tilt_max).clamp(min=0.0)
    d_x = margin * (1.0 - x_excess / tilt_soft)
    return d_x.clamp(min=-margin)


def fk_feature_tensor(ee_pos, quat, cup_tilt, pour_tilt, cup_dot, pour_dot):
    """Compact endpoint FK features used by FK-conditioned model variants."""
    denom = (cup_tilt + pour_tilt).clamp(min=1e-3)
    handover = cup_tilt / denom
    return torch.stack([
        ee_pos[:, 0],
        ee_pos[:, 1],
        ee_pos[:, 2],
        cup_dot,
        pour_dot,
        cup_tilt / 180.0,
        pour_tilt / 180.0,
        handover,
    ], dim=1)


def compute_routing_info(q_rad_flat, robot_model, margin,
                         cup_tilt_max, cup_tilt_soft,
                         pour_tilt_max, pour_tilt_soft,
                         pour_rzy_safe=0.15, pour_rzy_soft=0.30,
                         return_xy=False,
                         return_fk_features=False,
                         task_axes=None):
    """Compute ee_y, d_orient_cup, d_orient_pour for a flat batch of joint configs.

    Args:
        q_rad_flat: (N, 7) joint angles in radians
        robot_model: CudaRobotModel
        margin: float, must match training margin
        cup_tilt_max, cup_tilt_soft: cupping orientation params
        pour_tilt_max, pour_tilt_soft: pouring orientation params

    Returns:
        ee_y: (N,) EE Y-position
        d_orient_cup: (N,) cupping orientation distance
        d_orient_pour: (N,) pouring orientation distance
    """
    task_axes = task_axes or load_task_axes(None)
    state = robot_model.get_state(q_rad_flat)
    ee_pos = state.ee_pose.position  # (N, 3)
    ee_y = ee_pos[:, 1]
    quat = state.ee_pose.quaternion

    cup_local, cup_world = task_axis_arrays(task_axes, "cup")
    pour_local, pour_world = task_axis_arrays(task_axes, "pour")
    cup_dot = axis_alignment_from_quat(quat, cup_local, cup_world)
    pour_dot = axis_alignment_from_quat(quat, pour_local, pour_world)
    cup_tilt = torch.acos(torch.clamp(cup_dot, -1.0, 1.0)) * (180.0 / 3.14159265)
    pour_tilt = torch.acos(torch.clamp(pour_dot, -1.0, 1.0)) * (180.0 / 3.14159265)

    cup_excess = (cup_tilt - cup_tilt_max).clamp(min=0.0)
    d_orient_cup = (margin * (1.0 - cup_excess / cup_tilt_soft)).clamp(min=-margin)
    pour_excess = (pour_tilt - pour_tilt_max).clamp(min=0.0)
    d_orient_pour = (margin * (1.0 - pour_excess / pour_tilt_soft)).clamp(min=-margin)

    fk_features = None
    if return_fk_features:
        fk_features = fk_feature_tensor(
            ee_pos, quat, cup_tilt, pour_tilt, cup_dot, pour_dot)

    if return_xy and return_fk_features:
        return ee_y, ee_pos[:, :2], d_orient_cup, d_orient_pour, fk_features
    if return_xy:
        return ee_y, ee_pos[:, :2], d_orient_cup, d_orient_pour
    if return_fk_features:
        return ee_y, d_orient_cup, d_orient_pour, fk_features
    return ee_y, d_orient_cup, d_orient_pour


def _configure_live_scalar_fk_runtime(
        model, robot_model, q_center, q_range):
    """Attach FK to any serialized q-only scalar that declares live geometry.

    The original runtime check keyed on the presence of learned full-semantic
    replicas.  Analytic continuous and headwise scalar modes use the same
    immutable FK buffers without owning those replicas, so checking the
    serialized normalization buffers is the mode-independent contract.
    """
    network = model.network
    required = (
        "configure_fullsemantic_robot_model",
        "speed_constraint_fullsemantic_q_center",
        "speed_constraint_fullsemantic_q_range",
    )
    if not all(hasattr(network, name) for name in required):
        return False
    network.configure_fullsemantic_robot_model(
        robot_model, q_center, q_range)
    return True


def travel_times_with_routing(model, XP, robot_model, q_center, q_range,
                              margin, cup_tilt_max, cup_tilt_soft,
                              pour_tilt_max, pour_tilt_soft,
                              pour_rzy_safe=0.15, pour_rzy_soft=0.30,
                              task_axes=None, physical_dim=None):
    """Compute travel times with constraint-aware routing.

    Args:
        model: MultiModel (must have .network with .out())
        XP: (B, 2*dim) normalized start-goal pairs
        robot_model: CudaRobotModel for FK
        q_center, q_range: (physical_dim,) tensors for un-normalization
        margin: float, orient distance margin (must match training)
        cup_tilt_max, cup_tilt_soft: cupping params
        pour_tilt_max, pour_tilt_soft: pouring params

    Returns:
        (B,) travel times
    """
    dim = XP.shape[1] // 2
    _configure_live_scalar_fk_runtime(
        model, robot_model, q_center, q_range)
    if physical_dim is None:
        physical_dim = int(q_center.numel())
    if q_center.numel() != physical_dim or q_range.numel() != physical_dim:
        raise ValueError("physical normalization dimension mismatch")
    B = XP.shape[0]

    # Un-normalize both start and goal to radians
    q_start_rad = XP[:, :physical_dim] * q_range + q_center
    q_goal_rad = (
        XP[:, dim:dim + physical_dim] * q_range + q_center)
    q_all_rad = torch.cat([q_start_rad, q_goal_rad], dim=0)

    # Compute routing info via FK
    need_fk_features = getattr(model.network, "fk_feature_dim", 0) > 0
    need_handover_phase = bool(getattr(
        model.network, "viscosity_handover_phase_context", False)) or bool(
            getattr(model.network,
                    "causal_handover_phase_context", False)) or float(
            getattr(model.network, "handover_phase_embedding_scale", 0.0)) > 0.0
    need_task_fk = need_fk_features or need_handover_phase
    routing = compute_routing_info(
        q_all_rad, robot_model, margin,
        cup_tilt_max, cup_tilt_soft,
        pour_tilt_max, pour_tilt_soft,
        pour_rzy_safe=pour_rzy_safe, pour_rzy_soft=pour_rzy_soft,
        return_xy=True,
        return_fk_features=need_task_fk,
        task_axes=task_axes)
    if need_task_fk:
        ee_y, ee_xy, d_orient_cup, d_orient_pour, task_fk = routing
    else:
        ee_y, ee_xy, d_orient_cup, d_orient_pour = routing
        task_fk = None
    fk_features = task_fk if need_fk_features else None
    handover_phase = (
        task_fk[:, 7] if need_handover_phase else None)

    # Forward pass with routing: ee_y is already (2*B,) stacked as [starts; goals]
    tau, w, coords, _, _ = model.network.out(
        XP, ee_y=ee_y, ee_xy=ee_xy,
        d_orient_cup=d_orient_cup, d_orient_pour=d_orient_pour,
        fk_features=fk_features, handover_phase=handover_phase)
    return tau[:, 0]


def _exact_context_scalar_speed(
        q_rad, boundary_context, robot_model,
        cuboid_centers, cuboid_half, self_coll_mask,
        corner_zones, task_axes,
        cup_tilt_max=20.0, pour_tilt_max=20.0,
        corner_overlap_deg=45.0,
        signed_boundary_speed=1.0e-3,
        signed_distance_scale=0.01):
    """Evaluate the training scalar F(q,c) without a learned field or cache."""
    if cuboid_centers is None or cuboid_half is None:
        raise ValueError(
            "speed-conditioned HJB requires obstacle geometry")
    if self_coll_mask is None:
        raise ValueError(
            "speed-conditioned HJB requires the training self-collision mask")
    if corner_zones is None:
        raise ValueError(
            "speed-conditioned HJB requires immutable corner zones")
    context = torch.as_tensor(
        boundary_context,
        device=q_rad.device,
        dtype=q_rad.dtype).reshape(-1)
    if len(context) != len(q_rad):
        raise ValueError("one scalar context is required per configuration")

    state = robot_model.get_state(q_rad)
    spheres = state.link_spheres_tensor
    centers = spheres[:, :, :3]
    radii = spheres[:, :, 3]
    d_obstacle = sphere_cuboid_distance(
        centers, radii, cuboid_centers, cuboid_half)
    d_self = sphere_sphere_distance(
        centers, radii, self_coll_mask)
    ee_pos = state.ee_pose.position
    quat = state.ee_pose.quaternion
    cup_local, cup_world = task_axis_arrays(task_axes, "cup")
    pour_local, pour_world = task_axis_arrays(task_axes, "pour")
    cup_tilt = tilt_deg_from_quat(
        quat, cup_local, cup_world)
    pour_tilt = tilt_deg_from_quat(
        quat, pour_local, pour_world)
    cup_clearance = (
        0.05 * (float(cup_tilt_max) - cup_tilt)
        / max(float(cup_tilt_max), 1.0e-8))
    pour_clearance = (
        0.05 * (float(pour_tilt_max) - pour_tilt)
        / max(float(pour_tilt_max), 1.0e-8))

    progress = corner_blend_param(
        ee_pos[:, 0], ee_pos[:, 1],
        overlap_deg=corner_overlap_deg)
    open_corner = torch.full_like(cup_tilt, 0.05)
    open_corner = torch.where(
        progress <= 1.0e-6, cup_clearance, open_corner)
    open_corner = torch.where(
        progress >= 1.0 - 1.0e-6,
        pour_clearance, open_corner).clamp(-0.05, 0.05)

    cup_mask, pour_mask, _ = corner_masks_torch(
        ee_pos[:, :2], corner_zones, assign_outside=True)
    pure_zone = torch.full_like(cup_tilt, 0.05)
    pure_zone = torch.where(
        cup_mask, cup_clearance, pure_zone)
    pure_zone = torch.where(
        pour_mask,
        torch.minimum(pure_zone, pour_clearance),
        pure_zone).clamp(-0.05, 0.05)
    orientation = torch.where(
        context >= 0.5, open_corner, pure_zone)

    invalid = torch.minimum(
        torch.minimum(
            d_obstacle - 0.005,
            2.5 * (d_self - 0.002)),
        orientation)
    exponent = (
        invalid / float(signed_distance_scale)).clamp(
            min=-20.0, max=20.0)
    return (
        float(signed_boundary_speed)
        * torch.exp(exponent)).clamp(
            min=1.0e-5, max=1.0)


def _check_pouring_tilt(traj_rad, robot_model, x_tilt_max_deg=20.0,
                        task_axes=None):
    """Check UR5e pouring constraint from scene task_axes."""
    state = robot_model.get_state(traj_rad)
    quat = state.ee_pose.quaternion
    local_axis, world_axis = task_axis_arrays(task_axes or load_task_axes(None), "pour")
    x_tilt = tilt_deg_from_quat(quat, local_axis, world_axis)

    max_x_tilt = x_tilt.max().item()
    x_ok = max_x_tilt <= x_tilt_max_deg

    return x_ok, max_x_tilt


def _check_cupping_tilt(traj_rad, robot_model, cup_tilt_max_deg=20.0,
                        task_axes=None):
    """Check UR5e cupping constraint from scene task_axes."""
    state = robot_model.get_state(traj_rad)
    quat = state.ee_pose.quaternion
    local_axis, world_axis = task_axis_arrays(task_axes or load_task_axes(None), "cup")
    cup_tilt = tilt_deg_from_quat(quat, local_axis, world_axis)

    max_tilt = cup_tilt.max().item()
    return max_tilt <= cup_tilt_max_deg, max_tilt


def _densify_traj(traj_rad, max_step):
    """Insert waypoints so no joint moves more than `max_step` rad between samples.

    The tilt gate is a per-waypoint predicate, so its verdict is only as fine as the
    trajectory's own discretisation.  MPPI emits steps of ~0.01 in normalised space
    (~0.06 rad), which is coarse enough to step over a tilt excursion entirely.
    """
    t = traj_rad if torch.is_tensor(traj_rad) else torch.as_tensor(traj_rad)
    if t.shape[0] < 2:
        return t
    seg = (t[1:] - t[:-1])
    n = (seg.abs().max(dim=1).values / max_step).ceil().clamp(min=1).to(torch.long)
    pieces = []
    for i in range(t.shape[0] - 1):
        k = int(n[i].item())
        alphas = torch.arange(k, device=t.device, dtype=t.dtype).unsqueeze(1) / k
        pieces.append(t[i].unsqueeze(0) + alphas * seg[i].unsqueeze(0))
    pieces.append(t[-1].unsqueeze(0))
    return torch.cat(pieces, dim=0)


def _check_tilt_per_waypoint_batch(trajs_rad, robot_model, y_divide,
                                    cup_tilt_threshold, pour_tilt_threshold,
                                    n_ref_cup, overlap=0.0,
                                    morph_orient=False, morph_mode="linear",
                                    zone_mode="y", corner_zones=None,
                                    corner_overlap_deg=45.0,
                                    handover_accept=False,
                                    handover_mono_tol=0.10,
                                    pair_types=None,
                                    task_axes=None,
                                    overlap_logic="or"):
    """GPU-batched per-waypoint zone-aware tilt check for B trajectories.

    Each waypoint is assigned to a zone based on its FK EE Y-position:
      - overlap=0 (default): hard split at y_divide
      - overlap>0 (OR logic): three zones:
          cup-exclusive (ee_y > y_divide + overlap): only cupping check
          overlap zone (y_divide-overlap <= ee_y <= y_divide+overlap): EITHER check (OR)
          pour-exclusive (ee_y < y_divide - overlap): only pouring check

    Args:
        trajs_rad: list of B tensors, each (T_i, 7) in radians
        robot_model: CudaRobotModel
        y_divide: zone boundary Y-coordinate
        cup_tilt_threshold: max cupping tilt in degrees
        pour_tilt_threshold: max pouring x-tilt in degrees
        n_ref_cup: (3,) reference direction for cupping (world +Z)
        overlap: half-width of overlap zone for OR-logic (0 = legacy hard split)
        overlap_logic: acceptance rule inside an explicit overlap band. The
            historical default ``"or"`` accepts either local constraint;
            ``"both"`` requires both cup and pour constraints.

    Returns:
        list of B tuples: (tilt_ok, max_cup_tilt, max_pour_tilt, n_cup, n_pour)
    """
    if overlap_logic not in ("or", "both"):
        raise ValueError(
            f"overlap_logic must be 'or' or 'both', got {overlap_logic!r}")

    B = len(trajs_rad)
    lengths = [t.shape[0] for t in trajs_rad]
    T_max = max(lengths)
    device = trajs_rad[0].device

    # Pad & stack: repeat last waypoint for padding
    padded = []
    for t in trajs_rad:
        if t.shape[0] < T_max:
            pad = t[-1:].expand(T_max - t.shape[0], -1)
            padded.append(torch.cat([t, pad], dim=0))
        else:
            padded.append(t)
    stacked = torch.stack(padded, dim=0)  # (B, T_max, 7)

    # Valid mask: True for real waypoints, False for padding
    valid_mask = torch.zeros(B, T_max, dtype=torch.bool, device=device)
    for i, L in enumerate(lengths):
        valid_mask[i, :L] = True

    # Single FK call for all B*T_max waypoints
    all_wpts = stacked.reshape(B * T_max, DOF)
    state = robot_model.get_state(all_wpts)
    ee_pos = state.ee_pose.position.clone().reshape(B, T_max, 3)
    ee_y = ee_pos[:, :, 1]
    quat = state.ee_pose.quaternion.clone()

    # Zone assignment per waypoint
    if zone_mode == "corner_table":
        cup_mask, pour_mask, overlap_mask = corner_masks_torch(
            ee_pos[:, :, :2], corner_zones,
            valid_mask=valid_mask, assign_outside=True)
    elif overlap > 0:
        # OR-logic: three zones
        lower = y_divide - overlap
        upper = y_divide + overlap
        cup_only = (ee_y > upper) & valid_mask     # cup-exclusive (+Y)
        pour_only = (ee_y < lower) & valid_mask    # pour-exclusive (-Y)
        overlap_mask = (ee_y >= lower) & (ee_y <= upper) & valid_mask  # overlap zone
        cup_mask = cup_only | overlap_mask   # cupping check applies here
        pour_mask = pour_only | overlap_mask  # pouring check applies here
        # In overlap zone: both masks True, but combined with OR below
    else:
        # Legacy: hard split at y_divide
        cup_mask = (ee_y >= y_divide) & valid_mask   # (B, T_max)
        pour_mask = (ee_y < y_divide) & valid_mask    # (B, T_max)

    cup_local, cup_world = task_axis_arrays(task_axes or load_task_axes(None), "cup")
    pour_local, pour_world = task_axis_arrays(task_axes or load_task_axes(None), "pour")
    cup_tilt_deg = tilt_deg_from_quat(quat, cup_local, cup_world).reshape(B, T_max)
    # Cupping waypoints must satisfy tilt <= threshold; non-cupping waypoints auto-pass
    cup_ok = (cup_tilt_deg <= cup_tilt_threshold) | ~cup_mask  # (B, T_max)

    pour_x_tilt = tilt_deg_from_quat(quat, pour_local, pour_world).reshape(B, T_max)
    pour_ok = (pour_x_tilt <= pour_tilt_threshold) | ~pour_mask  # (B, T_max)

    # Combined check per waypoint:
    # - cup-exclusive zone: must pass cupping (cup_ok handles via ~cup_mask auto-pass)
    # - pour-exclusive zone: must pass pouring (pour_ok handles via ~pour_mask auto-pass)
    # - overlap zone: pass if EITHER cupping OR pouring constraint is met
    if handover_accept and zone_mode == "corner_table":
        # MONOTONIC HANDOVER acceptance (2026-06-30): the oracle showed the
        # corner has NO consistent orientation target (0/33 LOO at every band),
        # so pinning a per-waypoint mid-corner angle is ill-posed. Instead:
        #   - WINGS (s~0 / s~1): keep the strict pure single-axis tilt <= thr.
        #   - CORNER: don't pin an angle; require the handover coordinate
        #       h = cup_tilt / (cup_tilt + pour_tilt)   in [0,1]
        #     to be MONOTONIC non-decreasing along the path (no snapping back),
        #     i.e. the wrist continuously rotates cup->pour through the corner.
        from morph_orient import corner_blend_param
        ee_x_wp = ee_pos[:, :, 0]
        s_wp = corner_blend_param(ee_x_wp, ee_y, overlap_deg=corner_overlap_deg)  # (B,T)
        # wing membership: pure cup wing (s<=0) or pure pour wing (s>=1) after clamp.
        cup_wing = (s_wp <= 1e-6) & valid_mask
        pour_wing = (s_wp >= 1.0 - 1e-6) & valid_mask
        corner_wp = valid_mask & ~cup_wing & ~pour_wing
        # (1) wing waypoints: strict pure single-axis tilt
        wing_ok = (~cup_wing | (cup_tilt_deg <= cup_tilt_threshold)) & \
                  (~pour_wing | (pour_x_tilt <= pour_tilt_threshold))
        # (2) corner waypoints: no per-waypoint angle pin -> auto-pass the band
        #     part; monotonicity is enforced as a trajectory-level gate below.
        all_ok = wing_ok | ~valid_mask
        # handover coordinate per waypoint (safe divide)
        denom = (cup_tilt_deg + pour_x_tilt).clamp(min=1e-3)
        h = cup_tilt_deg / denom                                   # (B,T) in [0,1]
        # trajectory-level monotonicity over CORNER waypoints, ordered by path.
        # For each path, take h at corner waypoints in sequence; require each step
        # h[t+1] >= h[t] - tol. We compute via masked diffs.
        h_masked = torch.where(corner_wp, h, torch.full_like(h, float("nan")))
        mono_ok = torch.ones(h.shape[0], dtype=torch.bool, device=h.device)
        for bi in range(h.shape[0]):
            hv = h_masked[bi]
            hv = hv[~torch.isnan(hv)]
            if hv.numel() <= 1:
                continue
            steps = hv[1:] - hv[:-1]
            mono_ok[bi] = bool((steps >= -handover_mono_tol).all().item())
        handover_ok_batch = all_ok.all(dim=1) & mono_ok
        if pair_types is None:
            handover_mask = torch.ones(B, dtype=torch.bool, device=device)
        else:
            pair_types_t = torch.as_tensor(pair_types, device=device)
            handover_mask = pair_types_t.reshape(-1).to(torch.long) == 2
            if handover_mask.numel() != B:
                raise ValueError(
                    f"pair_types must have {B} entries, got {handover_mask.numel()}")
        pure_ok_batch = (cup_ok & pour_ok).all(dim=1)
        tilt_ok_batch = torch.where(handover_mask, handover_ok_batch, pure_ok_batch)
        # also require valid endpoints already covered by wing_ok at s~0/1.
        results = []
        for i in range(B):
            cm = cup_mask[i]; pm = pour_mask[i]
            n_cup = cm.sum().item(); n_pour = pm.sum().item()
            max_cup = cup_tilt_deg[i, cm].max().item() if n_cup > 0 else float('nan')
            max_pour = pour_x_tilt[i, pm].max().item() if n_pour > 0 else float('nan')
            results.append((tilt_ok_batch[i].item(), max_cup, max_pour, n_cup, n_pour))
        return results
    if morph_orient and zone_mode == "corner_table":
        # L-shape corner morph acceptance: a single morphed orientation target by
        # EE polar bearing, evaluated at EVERY waypoint. On the wings the morph
        # saturates to pure cup (s=0) or pure pour (s=1), so wing waypoints get the
        # strict single-task check; only the corner morphs. This MATCHES the
        # training label (morph_tilt_deg_corner), so train==eval acceptance.
        from morph_orient import morph_tilt_deg_corner
        ee_x_wp = ee_pos[:, :, 0]
        mt = morph_tilt_deg_corner(cup_tilt_deg, pour_x_tilt, ee_x_wp, ee_y,
                                   morph_mode, overlap_deg=corner_overlap_deg)
        # unified band: cup threshold == pour threshold (both 20deg). Only judge
        # real waypoints; padding auto-passes.
        all_ok = (mt <= cup_tilt_threshold) | ~valid_mask
    elif zone_mode == "corner_table" or overlap > 0:
        cup_pass = (cup_tilt_deg <= cup_tilt_threshold)
        pour_pass = pour_x_tilt <= pour_tilt_threshold
        if morph_orient and zone_mode != "corner_table":
            from morph_orient import morph_tilt_deg
            mt = morph_tilt_deg(cup_tilt_deg, pour_x_tilt, ee_y, y_divide, overlap, morph_mode)
            morph_pass = mt <= cup_tilt_threshold   # unified band: cup threshold == pour threshold
            all_ok = torch.where(overlap_mask, morph_pass, cup_ok & pour_ok)
        else:
            # Outside overlap: AND of cup_ok & pour_ok (one auto-passes via
            # ~mask). Keep the historical OR rule by default, while allowing
            # the pure softmin-zone contract to require both constraints.
            all_ok = combine_overlap_tilt_pass_torch(
                cup_ok,
                pour_ok,
                cup_pass,
                pour_pass,
                overlap_mask,
                overlap_logic,
            )
    else:
        all_ok = cup_ok & pour_ok  # (B, T_max)
    tilt_ok_batch = all_ok.all(dim=1)  # (B,)

    # Per-query stats
    results = []
    for i in range(B):
        cm = cup_mask[i]
        pm = pour_mask[i]
        n_cup = cm.sum().item()
        n_pour = pm.sum().item()
        max_cup = cup_tilt_deg[i, cm].max().item() if n_cup > 0 else float('nan')
        max_pour = pour_x_tilt[i, pm].max().item() if n_pour > 0 else float('nan')
        results.append((tilt_ok_batch[i].item(), max_cup, max_pour, n_cup, n_pour))

    return results


def _zone_tilts_batch(q_rad, robot_model, return_pos=False, task_axes=None):
    """Return (cup_tilt_deg, pour_tilt_deg, ee_y) for a flat batch of configs (radians).

    Uses the scene task_axes convention.
    """
    st = robot_model.get_state(q_rad)
    quat = st.ee_pose.quaternion
    ee_pos = st.ee_pose.position
    ee_y = ee_pos[:, 1]
    cup_local, cup_world = task_axis_arrays(task_axes or load_task_axes(None), "cup")
    pour_local, pour_world = task_axis_arrays(task_axes or load_task_axes(None), "pour")
    cup = tilt_deg_from_quat(quat, cup_local, cup_world)
    pour = tilt_deg_from_quat(quat, pour_local, pour_world)
    if return_pos:
        return cup, pour, ee_y, ee_pos
    return cup, pour, ee_y


def build_both_constraint_waypoints(pkl_cup, pkl_pour, robot_model,
                                    cuboid_centers, cuboid_half,
                                    tilt_cap, n_pool, q_center_np, q_range_np,
                                    device="cuda:0", seed=0, task_axes=None):
    """Build a both-constraint waypoint pool (cup-tilt<=cap AND pour-tilt<=cap AND
    collision-free) from the two FK pkls, plus a cKDTree on normalized configs.

    Returns (Wn, wtree) where Wn is (M, dim) normalized configs and wtree is a cKDTree.
    Mirrors r58_twostage_probe.py.
    """
    import pickle
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    pool = []
    for p in (pkl_cup, pkl_pour):
        with open(p, "rb") as f:
            pool.append(np.asarray(pickle.load(f)["q_solutions"], dtype=np.float32))
    q_pool = np.concatenate(pool, axis=0)
    sub = rng.choice(len(q_pool), size=min(n_pool, len(q_pool)), replace=False)
    q_pool = q_pool[sub]
    qt = torch.tensor(q_pool, dtype=torch.float32, device=device)

    both_keep = []
    for s in range(0, len(q_pool), 20000):
        e = min(s + 20000, len(q_pool))
        cup, pour, _ = _zone_tilts_batch(qt[s:e], robot_model, task_axes=task_axes)
        ok = (cup <= tilt_cap) & (pour <= tilt_cap)
        _, dr = check_collision_full(
            robot_model, qt[s:e], cuboid_centers, cuboid_half,
            torso_cuboid_centers=None, torso_cuboid_half=None,
            self_collision_mask=None, margin=0.005, device=device)
        ok &= (dr["obstacle"] > 0) & (dr["self"] > 0.002)
        both_keep.append(ok.cpu().numpy())
    both_keep = np.concatenate(both_keep)
    W = q_pool[both_keep]
    print(f"[2stage] both-constraint (cup&pour<={tilt_cap}deg) collision-free "
          f"waypoints: {len(W)} (from pool of {len(q_pool)})")
    if len(W) == 0:
        return None, None
    Wn = (W - q_center_np[None, :]) / q_range_np[None, :]
    wtree = cKDTree(Wn)
    return Wn.astype(np.float32), wtree


class _GraphFidelityError(RuntimeError):
    """Raised only when a captured graph's output differs from eager.

    Kept distinct from capture failure: a graph that cannot be captured is a
    performance matter and must fall back silently, but a graph that captures and
    then returns different numbers is a correctness failure and must never be
    silently used."""


_GRAPH_COST_CACHE = {}


class _GraphCost:
    """Replay the unchanged cost network with a CUDA Graph.

    Adapted from the colleague's runtime/graph_cost.py.  The cost forward is a pure
    tensor function of fixed-shape inputs, so once captured it can be replayed by
    copying new values into the static buffers.  Only pays off when the launch
    overhead is a large fraction of the step -- i.e. at small batch sizes; at
    batch20 x 1536 it measured 1.01x and is not worth enabling.

    Capture requires the cost forward to make no host-side allocation, which is why
    the quadrature constants had to be cached first (model_network_multi.py).

    Fidelity gate: the replayed output is compared bitwise against a fresh eager
    forward at capture time and capture is aborted if they differ.
    """

    def __init__(self, forward, max_shapes=4):
        self.forward = forward
        self.entries = {}
        self.max_shapes = max_shapes
        self.disabled = False

    @torch.no_grad()
    def __call__(self, coords, **kwargs):
        if self.disabled:
            return self.forward(coords, **kwargs)
        values = {'coords': coords, **kwargs}
        try:
            key = tuple(
                (n, (tuple(v.shape), v.dtype, v.device) if torch.is_tensor(v)
                 else (None if v is None else type(v).__name__))
                for n, v in values.items())
        except Exception:
            return self.forward(coords, **kwargs)
        if key not in self.entries:
            # Shapes keep changing (e.g. the active set is shrinking): capture would
            # thrash, so fall back to eager for the rest of the run.
            if len(self.entries) >= self.max_shapes:
                self.disabled = True
                print("Graph cost: too many distinct shapes, falling back to eager",
                      flush=True)
                return self.forward(coords, **kwargs)
            buffers = {n: (v.clone() if torch.is_tensor(v) else v)
                       for n, v in values.items()}
            try:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self.forward(**buffers)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs = self.forward(**buffers)
                graph.replay()
                reference = self.forward(**values)
                if not torch.equal(outputs[0], reference[0]):
                    err = float((outputs[0] - reference[0]).abs().max())
                    raise _GraphFidelityError(
                        f"Graph cost differs from eager (max {err}); refusing")
                print("Graph cost CUDA Graph audit passed (bitwise)", flush=True)
            except _GraphFidelityError:
                raise
            except Exception as exc:
                self.disabled = True
                print(f"Graph cost: capture unavailable ({str(exc)[:70]}), "
                      f"using eager", flush=True)
                return self.forward(coords, **kwargs)
            self.entries[key] = (buffers, graph, outputs)
        buffers, graph, outputs = self.entries[key]
        for n, v in values.items():
            if torch.is_tensor(v):
                buffers[n].copy_(v)
        graph.replay()
        return outputs


def MPPI_batch_multi(model, XP_batch, robot_model, q_center, q_range,
                     margin, cup_tilt_max, cup_tilt_soft,
                     pour_tilt_max, pour_tilt_soft,
                     q_norm_lo=None, q_norm_hi=None,
                     pour_rzy_safe=0.15, pour_rzy_soft=0.30,
                     dim=7, physical_dim=None,
                     steps=200, sample_num=50, horizon=5,
                     step_size=0.015, convergence_threshold=0.01,
                     pair_type_batch=None, force_intra_routing=False,
                     pair_type_context_batch=None,
                     boundary_context_batch=None,
                     safety_rerank=False, safety_select_best=False,
                     y_divide=0.0, safety_tilt_weight=0.02,
                     graded_tilt_weight=0.0,
                     safety_collision_weight=20.0,
                     safety_collision_margin=0.005,
                     safety_hard_filter=False,
                     safety_hard_penalty=10.0,
                     zone_mode="y",
                     corner_zones=None,
                     cuboid_centers=None, cuboid_half=None,
                     torso_centers=None, torso_half=None,
                     self_coll_mask=None, no_self_collision=False,
                     line_guide=False,
                     goal_cost_weight=0.0,
                     task_adaptive_objective=False,
                     orient_mppi_weight=0.0,
                     handover_mppi_weight=0.0,
                     handover_mppi_tol=0.10,
                     corner_overlap_deg=45.0,
                     handover_accept=False,
                     cross_line_blend=0.0,
                     task_axes=None,
                     ensemble_models=None,
                     ensemble_mode="mean",
                     tau_cost_mode="first_last",
                     tau_first_weight=10.0,
                     mppi_cost_beta=50.0,
                     learned_select_best=False,
                     constant_arc_step=False,
                     budget_feasible_step=False,
                     ):
    """Batched MPPI planner with multi-task constraint-aware routing.

    At each MPPI step, computes FK at trajectory endpoints to obtain
    ee_y, d_orient_cup, d_orient_pour for dynamic routing through the
    dual-head network.

    Args:
        model: MultiModel with .network (MultiNN)
        XP_batch: (B, 2*dim) normalized start-goal pairs
        robot_model: CudaRobotModel for FK
        q_center, q_range: (physical_dim,) tensors for un-normalization
        margin: float, orient distance margin (must match training)
        cup_tilt_max, cup_tilt_soft: cupping tilt parameters
        pour_tilt_max, pour_tilt_soft: pouring tilt parameters
        pour_rzy_safe, pour_rzy_soft: pouring Rz_y parameters
        dim, steps, sample_num, horizon, step_size, convergence_threshold: MPPI params

    Returns:
        trajectories: list of B lists of physical (1, physical_dim) tensors
        n_iters: (B,) tensor of iteration counts
    """
    B = XP_batch.shape[0]
    device = XP_batch.device
    if XP_batch.shape[1] != 2 * dim:
        raise ValueError(
            f"XP_batch has width {XP_batch.shape[1]}, expected {2 * dim}")
    if physical_dim is None:
        physical_dim = int(q_center.numel())
    if not 1 <= physical_dim <= dim:
        raise ValueError(
            f"physical_dim must lie in [1, {dim}], got {physical_dim}")
    if q_center.numel() != physical_dim or q_range.numel() != physical_dim:
        raise ValueError("physical normalization dimension mismatch")
    boundary_conditioned = bool(
        getattr(model.network, "boundary_conditioned", False))
    if boundary_conditioned:
        if boundary_context_batch is None:
            raise ValueError(
                "boundary-conditioned HJB requires one immutable context "
                "per planning query")
        boundary_context_batch = torch.as_tensor(
            boundary_context_batch,
            dtype=XP_batch.dtype,
            device=device).reshape(-1)
        if boundary_context_batch.numel() != B:
            raise ValueError(
                f"boundary_context_batch must have {B} entries")
        if torch.any(
                (boundary_context_batch < 0.0)
                | (boundary_context_batch > 1.0)):
            raise ValueError(
                "boundary context must lie in [0, 1]")
    lifted_phase = bool(
        dim == physical_dim + 1
        and getattr(model.network, "lifted_phase", False))
    lifted_phase_causal = bool(
        lifted_phase
        and getattr(model.network, "lifted_phase_causal", False))
    phase_bound = None
    phase_static = None
    if lifted_phase:
        phase_bound = 0.5 * float(model.network.lifted_phase_scale)
        if phase_bound <= 0.0:
            raise ValueError("lifted phase bound must be positive")
        phase_static = (
            XP_batch[:, physical_dim]
            - XP_batch[:, dim + physical_dim]).abs() <= 1e-7

    dP_prior = torch.zeros((B, 1, dim), device=device)
    trajectories = [[XP_batch[i:i+1, :physical_dim].clone()]
                    for i in range(B)]

    # SPEEDUP (adapted from the colleague's runtime/cached_goal_fk.py): within one
    # query the GOAL config never changes, yet its FK/routing was recomputed for every
    # candidate on every iteration -- half of all routing work, discarded and redone
    # ~400 times.  Compute it once per query and reuse.  Bitwise-identical by
    # construction; audited against the original on the first iteration.
    # Unlike the upstream version this is batch-size agnostic and carries no serial8.
    cached_goal_routing = None
    cache_goal_fk = os.environ.get('NTF_CACHE_GOAL_FK', '1') == '1'

    # Optional CUDA-Graph replay of the cost network.  Default OFF: it only helps
    # when launch-bound (small batch), and measured 1.01x at batch20 x 1536.
    # Keyed by network identity and kept across calls: with batch_size 1 this
    # function runs once per query, so a per-call cache would recapture 180 times.
    _graph_cost = None
    if os.environ.get('NTF_GRAPH_COST', '0') == '1':
        _key = id(model.network)
        if _key not in _GRAPH_COST_CACHE:
            _GRAPH_COST_CACHE[_key] = _GraphCost(model.network.out)
        _graph_cost = _GRAPH_COST_CACHE[_key]
        model.network.out = _graph_cost

    # JOINT LIMITS (2026-09-13): MPPI integrates in normalized space with no bound
    # on the physical joints, so the state could leave the robot's range entirely --
    # measured overshoots up to 0.6 rad on the cross-pour suite, 11-12 failed queries.
    # RRT-Connect plans in a bounded joint space (ob.RealVectorBounds), so bounding
    # ours is parity, not a crutch.  It is also in-distribution: the normalized limits
    # are exactly the [-0.5, 0.5] box the training pairs live in, so an out-of-range
    # state is one the network was never trained on and its tau there is meaningless.
    # Candidates are clamped BEFORE they are scored, so MPPI never selects a move it
    # is not allowed to make, and the committed state is clamped again.
    _lim_lo = _lim_hi = None
    if q_norm_lo is not None and q_norm_hi is not None:
        _lim_lo = torch.as_tensor(
            q_norm_lo, device=device, dtype=XP_batch.dtype).reshape(1, -1)[:, :physical_dim]
        _lim_hi = torch.as_tensor(
            q_norm_hi, device=device, dtype=XP_batch.dtype).reshape(1, -1)[:, :physical_dim]
        XP_batch[:, :physical_dim] = XP_batch[:, :physical_dim].clamp(
            min=_lim_lo, max=_lim_hi)

    active = torch.ones(B, dtype=torch.bool, device=device)
    n_iters = torch.full((B,), steps - 1, dtype=torch.long, device=device)

    for it in range(steps):
        n_active = active.sum().item()
        if n_active == 0:
            break

        active_idx = torch.where(active)[0]
        A = len(active_idx)

        XP_active = XP_batch[active_idx]
        dP_prior_active = dP_prior[active_idx]

        XP_tmp = XP_active.unsqueeze(1).unsqueeze(1).expand(A, sample_num, horizon, -1).clone()

        dP = step_size * torch.randn(A, sample_num, 1, dim, device=device) \
           + step_size * torch.randn(A, sample_num, horizon, dim, device=device)
        dP = dP + 2 * dP_prior_active.unsqueeze(1)
        if lifted_phase:
            # p is a task-state coordinate, not a freely drifting seventh
            # joint.  Within-task queries have identical endpoint phase and
            # therefore zero phase dynamics.  In the causal architecture,
            # cross-task queries may advance the handover clock but cannot
            # run it backward.  The learned scalar-speed tube around h(q)
            # then turns this state-space causality into cup->pour wrist
            # causality without adding a planner-side score.
            static_active = phase_static[active_idx]
            if static_active.any():
                dP[static_active, :, :, physical_dim] = 0.0
            if lifted_phase_causal and (~static_active).any():
                dP[~static_active, :, :, physical_dim].clamp_(min=0.0)
        dP_norm = torch.norm(dP, dim=3, keepdim=True)
        dP = dP / (torch.clamp(dP_norm, min=step_size) / step_size)

        if line_guide and sample_num > 0:
            to_goal = XP_active[:, dim:] - XP_active[:, :dim]
            dist = torch.norm(to_goal, dim=1, keepdim=True).clamp(min=1e-6)
            step_len = torch.minimum(
                dist, torch.full_like(dist, step_size))
            line_step = to_goal / dist * step_len
            dP[:, 0, :, :] = line_step.unsqueeze(1).expand(A, horizon, dim)

        dP_cumsum = torch.cumsum(dP, dim=2)
        XP_tmp[..., :dim] = XP_tmp[..., :dim] + dP_cumsum
        if _lim_lo is not None:
            XP_tmp[..., :physical_dim] = XP_tmp[..., :physical_dim].clamp(
                min=_lim_lo, max=_lim_hi)
        if lifted_phase:
            XP_tmp[..., physical_dim].clamp_(
                min=-phase_bound, max=phase_bound)

        # Evaluate learned travel times at candidate horizon points. The original
        # planner used only the first and last horizon point; alternate modes keep
        # the cost learned-tau-only while letting MPPI see more of the horizon.
        if tau_cost_mode == "first_last":
            tau_points = XP_tmp[:, :, [0, -1], :]
        elif tau_cost_mode == "last":
            tau_points = XP_tmp[:, :, [-1], :]
        elif tau_cost_mode in ("mean", "sum", "max", "first_mean",
                               "first_max", "discount"):
            tau_points = XP_tmp
        else:
            raise ValueError(f"unknown tau_cost_mode={tau_cost_mode!r}")
        n_tau_points = tau_points.shape[2]
        endpoints_flat = tau_points.reshape(-1, 2 * dim)

        # Dynamic routing: compute FK at every step
        q_start_norm = endpoints_flat[:, :physical_dim]
        q_goal_norm = endpoints_flat[:, dim:dim + physical_dim]
        q_start_rad = (
            q_start_norm * q_range.view(1, physical_dim)
            + q_center.view(1, physical_dim))
        q_goal_rad = (
            q_goal_norm * q_range.view(1, physical_dim)
            + q_center.view(1, physical_dim))
        q_all_rad = torch.cat([q_start_rad, q_goal_rad], dim=0)

        need_fk_features = getattr(model.network, "fk_feature_dim", 0) > 0
        need_handover_phase = bool(getattr(
            model.network, "viscosity_handover_phase_context", False)) or bool(
                getattr(model.network,
                        "causal_handover_phase_context", False)) or float(
                getattr(
                    model.network, "handover_phase_embedding_scale", 0.0)) > 0.0
        need_task_fk = need_fk_features or need_handover_phase
        if getattr(model.network, "pure_hjb", False):
            # The pure value consumes q and goal only.  Avoid leaking exact FK
            # features into inference; FK remains available later solely to
            # score the frozen trajectory under the external evaluator.
            routing_rows = q_all_rad.shape[0]
            ee_y = torch.zeros(
                routing_rows, device=device, dtype=q_all_rad.dtype)
            ee_xy = torch.zeros(
                routing_rows, 2, device=device, dtype=q_all_rad.dtype)
            d_cup = torch.zeros_like(ee_y)
            d_pour = torch.zeros_like(ee_y)
            task_fk = None
        else:
            if cache_goal_fk:
                if cached_goal_routing is None:
                    goal_rad = (
                        XP_batch[:, dim:dim + physical_dim]
                        * q_range.view(1, physical_dim)
                        + q_center.view(1, physical_dim))
                    cached_goal_routing = tuple(
                        v.clone() for v in compute_routing_info(
                            goal_rad, robot_model, margin,
                            cup_tilt_max, cup_tilt_soft,
                            pour_tilt_max, pour_tilt_soft,
                            pour_rzy_safe=pour_rzy_safe,
                            pour_rzy_soft=pour_rzy_soft,
                            return_xy=True,
                            return_fk_features=need_task_fk,
                            task_axes=task_axes))
                start_routing = compute_routing_info(
                    q_start_rad, robot_model, margin,
                    cup_tilt_max, cup_tilt_soft,
                    pour_tilt_max, pour_tilt_soft,
                    pour_rzy_safe=pour_rzy_safe, pour_rzy_soft=pour_rzy_soft,
                    return_xy=True,
                    return_fk_features=need_task_fk,
                    task_axes=task_axes)
                routing = tuple(
                    torch.cat((cur, fixed[active_idx].repeat_interleave(
                        sample_num * n_tau_points, dim=0)), dim=0)
                    for cur, fixed in zip(start_routing, cached_goal_routing))
                if it == 0:
                    ref = compute_routing_info(
                        q_all_rad, robot_model, margin,
                        cup_tilt_max, cup_tilt_soft,
                        pour_tilt_max, pour_tilt_soft,
                        pour_rzy_safe=pour_rzy_safe,
                        pour_rzy_soft=pour_rzy_soft,
                        return_xy=True,
                        return_fk_features=need_task_fk,
                        task_axes=task_axes)
                    if not all(torch.equal(x, y)
                               for x, y in zip(routing, ref)):
                        errs = [float((x - y).abs().max())
                                for x, y in zip(routing, ref)]
                        raise RuntimeError(
                            f"Cached goal FK is not bitwise equal: {errs}")
                    print("Cached goal FK exact audit passed", flush=True)
            else:
                routing = compute_routing_info(
                    q_all_rad, robot_model, margin,
                    cup_tilt_max, cup_tilt_soft,
                    pour_tilt_max, pour_tilt_soft,
                    pour_rzy_safe=pour_rzy_safe, pour_rzy_soft=pour_rzy_soft,
                    return_xy=True,
                    return_fk_features=need_task_fk,
                    task_axes=task_axes)
            if need_task_fk:
                ee_y, ee_xy, d_cup, d_pour, task_fk = routing
            else:
                ee_y, ee_xy, d_cup, d_pour = routing
                task_fk = None

        n_rows = endpoints_flat.shape[0]
        pt_rows = None
        if force_intra_routing and pair_type_batch is not None:
            pt_rows = pair_type_batch[active_idx].repeat_interleave(
                sample_num * n_tau_points)
        pair_context_rows = None
        if pair_type_context_batch is not None:
            pair_context_rows = pair_type_context_batch[active_idx].repeat_interleave(
                sample_num * n_tau_points)
        boundary_context_rows = None
        if boundary_context_batch is not None:
            boundary_context_rows = boundary_context_batch[
                active_idx].repeat_interleave(
                    sample_num * n_tau_points).reshape(-1, 1)
        scalar_speed_rows = None
        speed_models = [model, *(ensemble_models or [])]
        if any(getattr(
                item.network,
                "endpoint_speed_conditioning",
                False) for item in speed_models):
            if boundary_context_rows is None:
                raise ValueError(
                    "speed-conditioned HJB requires boundary context")
            scalar_context = torch.cat([
                boundary_context_rows.reshape(-1),
                boundary_context_rows.reshape(-1),
            ], dim=0)
            with torch.no_grad():
                scalar_speed_rows = _exact_context_scalar_speed(
                    q_all_rad,
                    scalar_context,
                    robot_model,
                    cuboid_centers,
                    cuboid_half,
                    self_coll_mask,
                    corner_zones,
                    task_axes,
                    cup_tilt_max=cup_tilt_max,
                    pour_tilt_max=pour_tilt_max,
                    corner_overlap_deg=corner_overlap_deg)

        def _travel_times_for(eval_model):
            context_kwargs = {}
            if getattr(eval_model.network, "boundary_conditioned", False):
                context_kwargs["boundary_context"] = boundary_context_rows
            if getattr(
                    eval_model.network,
                    "endpoint_speed_conditioning", False):
                context_kwargs["scalar_speed"] = scalar_speed_rows
            eval_need_fk = getattr(eval_model.network, "fk_feature_dim", 0) > 0
            eval_need_phase = bool(getattr(
                eval_model.network,
                "viscosity_handover_phase_context", False)) or bool(getattr(
                    eval_model.network,
                    "causal_handover_phase_context", False)) or float(getattr(
                    eval_model.network,
                    "handover_phase_embedding_scale", 0.0)) > 0.0
            eval_need_task_fk = eval_need_fk or eval_need_phase
            if eval_need_task_fk and task_fk is None:
                eval_routing = compute_routing_info(
                    q_all_rad, robot_model, margin,
                    cup_tilt_max, cup_tilt_soft,
                    pour_tilt_max, pour_tilt_soft,
                    pour_rzy_safe=pour_rzy_safe, pour_rzy_soft=pour_rzy_soft,
                    return_xy=True,
                    return_fk_features=True,
                    task_axes=task_axes)
                (eval_ee_y, eval_ee_xy, eval_d_cup, eval_d_pour,
                 eval_task_fk) = eval_routing
            else:
                eval_ee_y, eval_ee_xy = ee_y, ee_xy
                eval_d_cup, eval_d_pour = d_cup, d_pour
                eval_task_fk = task_fk
            eval_fk = eval_task_fk if eval_need_fk else None
            eval_handover_phase = (
                eval_task_fk[:, 7] if eval_need_phase else None)

            if force_intra_routing and pair_type_batch is not None:
                w_cup, w_pour = eval_model.network.compute_window_weights(
                    eval_ee_y, eval_d_cup, eval_d_pour, ee_xy=eval_ee_xy)
                row_cup = (pt_rows == 0)
                row_pour = (pt_rows == 1)
                for offset in (0, n_rows):
                    w_cup[offset:offset + n_rows][row_cup] = 1.0
                    w_pour[offset:offset + n_rows][row_cup] = 0.0
                    w_cup[offset:offset + n_rows][row_pour] = 0.0
                    w_pour[offset:offset + n_rows][row_pour] = 1.0
                tau_eval, _, _, _, _ = eval_model.network.out(
                    endpoints_flat, w_cup_override=w_cup, w_pour_override=w_pour,
                    fk_features=eval_fk, pair_type_context=pair_context_rows,
                    handover_phase=eval_handover_phase, **context_kwargs)
            else:
                tau_eval, _, _, _, _ = eval_model.network.out(
                    endpoints_flat,
                    ee_y=eval_ee_y, ee_xy=eval_ee_xy,
                    d_orient_cup=eval_d_cup, d_orient_pour=eval_d_pour,
                    fk_features=eval_fk, pair_type_context=pair_context_rows,
                    handover_phase=eval_handover_phase, **context_kwargs)
            return tau_eval

        if force_intra_routing and pair_type_batch is not None:
            tau = _travel_times_for(model)
            w_cup = w_pour = None
        else:
            tau = _travel_times_for(model)

        if ensemble_models:
            tau_stack = [tau]
            for eval_model in ensemble_models:
                tau_stack.append(_travel_times_for(eval_model))
            tau_all = torch.stack(tau_stack, dim=0)
            if ensemble_mode == "min":
                tau = tau_all.min(dim=0).values
            elif ensemble_mode == "max":
                tau = tau_all.max(dim=0).values
            elif ensemble_mode == "mean":
                tau = tau_all.mean(dim=0)
            else:
                raise ValueError(f"unknown ensemble_mode={ensemble_mode!r}")

        if force_intra_routing and pair_type_batch is not None:
            # Recompute primary-model weights only for optional orientation-head
            # cost below; learned-tau ensemble cost above is already finalized.
            w_cup, w_pour = model.network.compute_window_weights(ee_y, d_cup, d_pour, ee_xy=ee_xy)
            row_cup = (pt_rows == 0)
            row_pour = (pt_rows == 1)
            for offset in (0, n_rows):
                w_cup[offset:offset + n_rows][row_cup] = 1.0
                w_pour[offset:offset + n_rows][row_cup] = 0.0
                w_cup[offset:offset + n_rows][row_pour] = 0.0
                w_pour[offset:offset + n_rows][row_pour] = 1.0

        tau_cost = tau[:, 0].reshape(A, sample_num, n_tau_points)
        if tau_cost_mode == "first_last":
            cost = tau_first_weight * tau_cost[:, :, 0] + tau_cost[:, :, 1]
        elif tau_cost_mode == "last":
            cost = tau_cost[:, :, 0]
        elif tau_cost_mode == "mean":
            cost = tau_cost.mean(dim=2)
        elif tau_cost_mode == "sum":
            cost = tau_cost.sum(dim=2)
        elif tau_cost_mode == "max":
            cost = tau_cost.amax(dim=2)
        elif tau_cost_mode == "first_mean":
            cost = tau_first_weight * tau_cost[:, :, 0] + tau_cost.mean(dim=2)
        elif tau_cost_mode == "first_max":
            cost = tau_first_weight * tau_cost[:, :, 0] + tau_cost.amax(dim=2)
        elif tau_cost_mode == "discount":
            weights = torch.linspace(
                1.0, 0.25, n_tau_points, device=device,
                dtype=tau_cost.dtype)
            cost = (tau_cost * weights.view(1, 1, -1)).sum(dim=2) / weights.sum()
        else:
            raise ValueError(f"unknown tau_cost_mode={tau_cost_mode!r}")

        # R58F: decoupled additive orientation cost from the FIELD itself.
        # Phi_orient(q) is the per-config orientation potential the network learned;
        # adding its current-config (start slot) component to the MPPI cost makes
        # sample selection prefer low-tilt configs that the field scores low.
        if orient_mppi_weight > 0.0 and getattr(model.network, 'use_orient_head', False):
            if force_intra_routing and pair_type_batch is not None:
                phi_start, _ = model.network.orient_cost(
                    endpoints_flat, w_cup_override=w_cup, w_pour_override=w_pour)
            else:
                phi_start, _ = model.network.orient_cost(
                    endpoints_flat,
                    ee_y=ee_y, ee_xy=ee_xy,
                    d_orient_cup=d_cup, d_orient_pour=d_pour)
            # phi_start is the orientation cost of the *current/moving* (start) config.
            phi_cur = phi_start.reshape(A, sample_num, 2)[:, :, 0]
            cost = cost + orient_mppi_weight * phi_cur

        if goal_cost_weight > 0.0:
            terminal_norm = XP_tmp[:, :, -1, :dim]
            goal_norm = XP_active[:, dim:].unsqueeze(1)
            terminal_dist = torch.norm(terminal_norm - goal_norm, dim=2)
            if task_adaptive_objective and pair_type_batch is not None:
                objective_mask = (pair_type_batch[active_idx] != 0).float().unsqueeze(1)
                cost = cost + objective_mask * goal_cost_weight * terminal_dist
            else:
                cost = cost + goal_cost_weight * terminal_dist

        dP_step0 = dP[:, :, 0, :]

        if safety_rerank or graded_tilt_weight > 0.0 or handover_mppi_weight > 0.0:
            q_horizon_norm = XP_tmp[..., :physical_dim]
            q_horizon_rad = (
                q_horizon_norm * q_range.view(1, 1, 1, physical_dim)
                + q_center.view(1, 1, 1, physical_dim)
            )
            q_horizon_flat = q_horizon_rad.reshape(-1, physical_dim)

            state = robot_model.get_state(q_horizon_flat)
            ee_pos_h = state.ee_pose.position
            ee_y_h = ee_pos_h[:, 1]
            quat = state.ee_pose.quaternion
            cup_local, cup_world = task_axis_arrays(task_axes or load_task_axes(None), "cup")
            pour_local, pour_world = task_axis_arrays(task_axes or load_task_axes(None), "pour")
            cup_tilt = tilt_deg_from_quat(quat, cup_local, cup_world)
            pour_tilt = tilt_deg_from_quat(quat, pour_local, pour_world)

            if zone_mode == "corner_table":
                valid_flat = torch.ones_like(cup_tilt, dtype=torch.bool)
                cup_mask, pour_mask, overlap_mask = corner_masks_torch(
                    ee_pos_h[:, :2], corner_zones,
                    valid_mask=valid_flat, assign_outside=True)
                cup_v = (cup_tilt - cup_tilt_max).clamp(min=0.0)
                pour_v = (pour_tilt - pour_tilt_max).clamp(min=0.0)
                cup_violation = torch.where(cup_mask, cup_v, torch.zeros_like(cup_v))
                pour_violation = torch.where(pour_mask, pour_v, torch.zeros_like(pour_v))
                tilt_violation = torch.maximum(cup_violation, pour_violation)
                # OR transition: a waypoint is invalid only if both task tilts violate.
                tilt_violation = torch.where(
                    overlap_mask, torch.minimum(cup_v, pour_v), tilt_violation)
            else:
                cup_violation = torch.where(
                    ee_y_h >= y_divide,
                    (cup_tilt - cup_tilt_max).clamp(min=0.0),
                    torch.zeros_like(cup_tilt),
                )
                pour_violation = torch.where(
                    ee_y_h < y_divide,
                    (pour_tilt - pour_tilt_max).clamp(min=0.0),
                    torch.zeros_like(pour_tilt),
                )
                tilt_violation = torch.maximum(cup_violation, pour_violation)
            tilt_penalty = tilt_violation.reshape(A, sample_num, horizon).amax(dim=2)
            if task_adaptive_objective and pair_type_batch is not None:
                objective_mask = (pair_type_batch[active_idx] != 0).float().unsqueeze(1)
            else:
                objective_mask = None

            # R58F proof-of-concept (decoupled orientation): a GRADED, SMOOTH orientation
            # cost active INSIDE the cone too (not just over-limit), added to the MPPI cost
            # without hard-filtering. This emulates an additive orientation potential
            # Phi_orient(q) whose gradient is independent of the position-reaching tau, to test
            # whether a balanced additive orientation term recovers cross WITHOUT the
            # convergence collapse the over-limit hard-rerank caused. Uses the zone-appropriate
            # raw tilt (deg), mean over the horizon so it shapes the whole step.
            if graded_tilt_weight > 0.0:
                if zone_mode == "corner_table":
                    graded_tilt = torch.where(cup_mask, cup_tilt, pour_tilt)
                    graded_tilt = torch.where(
                        overlap_mask, torch.minimum(cup_tilt, pour_tilt), graded_tilt)
                else:
                    graded_tilt = torch.where(ee_y_h >= y_divide, cup_tilt, pour_tilt)
                graded_pen = graded_tilt.reshape(A, sample_num, horizon).mean(dim=2)
                if objective_mask is None:
                    cost = cost + graded_tilt_weight * graded_pen
                else:
                    cost = cost + objective_mask * graded_tilt_weight * graded_pen

            if safety_rerank:
                if objective_mask is None:
                    cost = cost + safety_tilt_weight * tilt_penalty
                else:
                    cost = cost + objective_mask * safety_tilt_weight * tilt_penalty
            hard_invalid = tilt_penalty > 0.0

            if (handover_mppi_weight > 0.0 and handover_accept
                    and zone_mode == "corner_table"):
                from morph_orient import corner_blend_param
                ee_x_h = ee_pos_h[:, 0].reshape(A, sample_num, horizon)
                ee_y_h_seq = ee_y_h.reshape(A, sample_num, horizon)
                s_h = corner_blend_param(
                    ee_x_h, ee_y_h_seq, overlap_deg=corner_overlap_deg)
                corner_h = (s_h > 1e-6) & (s_h < 1.0 - 1e-6)
                h_ratio = cup_tilt / (cup_tilt + pour_tilt).clamp(min=1e-3)
                h_ratio = h_ratio.reshape(A, sample_num, horizon)

                mono_penalty = torch.zeros(
                    (A, sample_num), device=device, dtype=cost.dtype)
                if horizon > 1:
                    adj_mask = corner_h[:, :, 1:] & corner_h[:, :, :-1]
                    drops = h_ratio[:, :, :-1] - h_ratio[:, :, 1:]
                    internal_violation = torch.relu(
                        drops - handover_mppi_tol) * adj_mask.float()
                    mono_penalty = torch.maximum(
                        mono_penalty, internal_violation.amax(dim=2))

                q_cur_rad = (
                    XP_active[:, :physical_dim]
                    * q_range.view(1, physical_dim)
                    + q_center.view(1, physical_dim)
                )
                cur_state = robot_model.get_state(q_cur_rad)
                cur_pos = cur_state.ee_pose.position
                cur_quat = cur_state.ee_pose.quaternion
                cup_local, cup_world = task_axis_arrays(
                    task_axes or load_task_axes(None), "cup")
                pour_local, pour_world = task_axis_arrays(
                    task_axes or load_task_axes(None), "pour")
                cup_cur = tilt_deg_from_quat(cur_quat, cup_local, cup_world)
                pour_cur = tilt_deg_from_quat(cur_quat, pour_local, pour_world)
                cur_s = corner_blend_param(
                    cur_pos[:, 0], cur_pos[:, 1], overlap_deg=corner_overlap_deg)
                cur_corner = (cur_s > 1e-6) & (cur_s < 1.0 - 1e-6)
                cur_h = cup_cur / (cup_cur + pour_cur).clamp(min=1e-3)
                first_corner = corner_h[:, :, 0] & cur_corner.unsqueeze(1)
                first_drop = cur_h.unsqueeze(1) - h_ratio[:, :, 0]
                first_violation = (
                    torch.relu(first_drop - handover_mppi_tol)
                    * first_corner.float())
                mono_penalty = torch.maximum(mono_penalty, first_violation)

                if task_adaptive_objective and pair_type_batch is not None:
                    objective_mask = (
                        pair_type_batch[active_idx] == 2).float().unsqueeze(1)
                    cost = cost + objective_mask * handover_mppi_weight * mono_penalty
                else:
                    cost = cost + handover_mppi_weight * mono_penalty

            if cuboid_centers is not None and cuboid_half is not None:
                if no_self_collision:
                    _, d_result = check_collision_full(
                        robot_model, q_horizon_flat, cuboid_centers, cuboid_half,
                        torso_cuboid_centers=None, torso_cuboid_half=None,
                        self_collision_mask=None,
                        margin=safety_collision_margin, device=device)
                else:
                    _, d_result = check_collision_full(
                        robot_model, q_horizon_flat, cuboid_centers, cuboid_half,
                        torso_cuboid_centers=torso_centers,
                        torso_cuboid_half=torso_half,
                        self_collision_mask=self_coll_mask,
                        torso_ignore_sphere_indices=TORSO_IGNORE_SPHERE_INDICES,
                        margin=safety_collision_margin, device=device)

                obstacle_violation = (
                    safety_collision_margin - d_result["obstacle"]).clamp(min=0.0)
                torso_violation = (
                    safety_collision_margin - d_result["torso"]).clamp(min=0.0)
                self_violation = (0.002 - d_result["self"]).clamp(min=0.0)
                collision_violation = torch.maximum(
                    torch.maximum(obstacle_violation, torso_violation),
                    self_violation,
                )
                collision_penalty = collision_violation.reshape(
                    A, sample_num, horizon).amax(dim=2)
                if objective_mask is None:
                    cost = cost + safety_collision_weight * collision_penalty
                else:
                    cost = cost + objective_mask * safety_collision_weight * collision_penalty
                hard_invalid = hard_invalid | (collision_penalty > 0.0)

            if safety_hard_filter:
                if objective_mask is None:
                    cost = cost + safety_hard_penalty * hard_invalid.float()
                else:
                    cost = cost + objective_mask * safety_hard_penalty * hard_invalid.float()

        if learned_select_best or (safety_rerank and safety_select_best):
            best_idx = cost.argmin(dim=1)
            dP_best = dP_step0[torch.arange(A, device=device), best_idx]
        else:
            weight = torch.softmax(-float(mppi_cost_beta) * cost, dim=1)
            dP_best = torch.einsum('as,asd->ad', weight, dP_step0)

        if constant_arc_step:
            # Every sampled control has arc length <= step_size, but their
            # weighted Euclidean mean can have an arbitrarily smaller norm.
            # That makes query reachability depend on sample disagreement and
            # creates an undocumented distance horizon.  NTFields supplies a
            # descent direction; integrate that direction at the configured
            # arc-length resolution, clipping only at the goal.
            dP_best = constant_arc_update(
                dP_best, XP_active[:, :dim], XP_active[:, dim:], step_size)
        elif budget_feasible_step:
            # Preserve MPPI's confidence-dependent update magnitude unless it
            # makes convergence impossible within the remaining integration
            # budget.  The lower bound is derived, not tuned: a remaining path
            # of length d needs average arc d / remaining_steps.  Never exceed
            # the configured local resolution or overshoot the goal.
            dP_best = budget_feasible_update(
                dP_best, XP_active[:, :dim], XP_active[:, dim:], step_size,
                remaining_steps=max(steps - it, 1))

        # R58H hybrid: blend the field step with a straight-line-to-goal step. The tau field
        # is monotone in VALUE along the straight cross path but its GRADIENT saturates in the
        # far-field, so MPPI random-walks. Adding a straight-line prior gives a usable
        # direction where the field is flat. Applied only to cross pairs (pair_type==2).
        if cross_line_blend > 0.0 and pair_type_batch is not None:
            cur = XP_active[:, :dim]
            goal = XP_active[:, dim:]
            to_goal = goal - cur
            tg_norm = torch.norm(to_goal, dim=1, keepdim=True).clamp(min=1e-6)
            line_step = to_goal / tg_norm * step_size            # straight-line step, step_size long
            is_cross = (pair_type_batch[active_idx] == 2).float().unsqueeze(1)
            blended = (1.0 - cross_line_blend) * dP_best + cross_line_blend * line_step
            dP_best = is_cross * blended + (1.0 - is_cross) * dP_best

        next_state = dP_best + XP_active[:, :dim]
        if _lim_lo is not None:
            next_state[:, :physical_dim] = next_state[:, :physical_dim].clamp(
                min=_lim_lo, max=_lim_hi)
            dP_best = next_state - XP_active[:, :dim]
        if lifted_phase:
            static_active = phase_static[active_idx]
            if lifted_phase_causal and (~static_active).any():
                next_state[~static_active, physical_dim] = torch.maximum(
                    next_state[~static_active, physical_dim],
                    XP_active[~static_active, physical_dim])
            next_state[:, physical_dim].clamp_(
                min=-phase_bound, max=phase_bound)
            if static_active.any():
                next_state[static_active, physical_dim] = (
                    XP_batch[active_idx[static_active], dim + physical_dim])
            # Store the actually applied bounded control as the MPPI prior.
            dP_best = next_state - XP_active[:, :dim]
        XP_batch[active_idx, :dim] = next_state
        dP_prior[active_idx, 0, :] = dP_best

        for j, bi in enumerate(active_idx.tolist()):
            trajectories[bi].append(
                XP_batch[bi:bi+1, :physical_dim].clone())

        dis = torch.norm(XP_batch[active_idx, dim:] - XP_batch[active_idx, :dim], dim=1)
        newly_converged = dis < convergence_threshold
        if newly_converged.any():
            converged_global = active_idx[newly_converged]
            n_iters[converged_global] = it
            active[converged_global] = False

    if _graph_cost is not None:
        model.network.out = _graph_cost.forward

    for i in range(B):
        trajectories[i].append(
            XP_batch[i:i+1, dim:dim + physical_dim].clone())

    return trajectories, n_iters


def characteristic_batch_multi(
        model, XP_batch, robot_model, q_center, q_range,
        margin, cup_tilt_max, cup_tilt_soft,
        pour_tilt_max, pour_tilt_soft,
        *, dim=6, steps=400, step_size=0.0025,
        convergence_threshold=0.01, task_axes=None,
        budget_feasible_step=False):
    """Integrate the sole learned scalar's normalized descent characteristic.

    This planner performs no sampling, collision query, task reranking,
    fallback, graph lookup, or auxiliary policy evaluation.  FK-derived values
    are detached routing compatibility inputs exactly as in raw MPPI; a pure
    q-only checkpoint ignores them in its scalar value.
    """
    if XP_batch.shape[1] != 2 * dim:
        raise ValueError("characteristic planner expects physical q--goal pairs")
    _vc = getattr(model.network, "value_condition", "none")
    if (getattr(model.network, "endpoint_dual_value_heads", False)
            or getattr(model.network, "use_task_adapters", False)
            or int(getattr(model.network, "handover_k", 0)) != 0):
        raise ValueError("characteristic backend is restricted to pure scalar models")
    if _vc != "none":
        # The routing/conditioning inputs are recomputed per step but DETACHED, so
        # the gradient below omits d(tau)/d(cond) * d(cond)/dq -- it is the partial
        # derivative w.r.t. q at fixed conditioning.  That is precisely the descent
        # direction a gradient planner would follow, so allow it for diagnosis while
        # being explicit that it is partial.
        if os.environ.get("ALLOW_CONDITIONED_CHARACTERISTIC", "0") != "1":
            raise ValueError("characteristic backend is restricted to pure scalar models")
        print(f"  [characteristic] value_condition={_vc!r}: using PARTIAL grad "
              f"(conditioning detached)")
    device = XP_batch.device
    current = XP_batch[:, :dim].detach().clone()
    goal = XP_batch[:, dim:].detach().clone()
    batch = len(current)
    trajectories = [[current[i:i + 1].clone()] for i in range(batch)]
    active = torch.ones(batch, dtype=torch.bool, device=device)
    n_iters = torch.full(
        (batch,), steps - 1, dtype=torch.long, device=device)

    for iteration in range(steps):
        remaining = goal - current
        remaining_distance = torch.linalg.vector_norm(
            remaining, dim=1)
        reached = active & (remaining_distance <= convergence_threshold)
        n_iters[reached] = iteration
        active = active & ~reached
        if not bool(active.any()):
            break
        active_index = torch.nonzero(active, as_tuple=False).reshape(-1)
        with torch.enable_grad():
            state = current[active_index].detach().requires_grad_(True)
            active_goal = goal[active_index].detach()
            coords = torch.cat([state, active_goal], dim=1)
            q_start_rad = state * q_range.view(1, dim) + q_center.view(1, dim)
            q_goal_rad = (
                active_goal * q_range.view(1, dim) + q_center.view(1, dim))
            q_all_rad = torch.cat([q_start_rad, q_goal_rad], dim=0)
            routing = compute_routing_info(
                q_all_rad, robot_model, margin,
                cup_tilt_max, cup_tilt_soft,
                pour_tilt_max, pour_tilt_soft,
                return_xy=True, return_fk_features=False,
                task_axes=task_axes)
            ee_y, ee_xy, d_cup, d_pour = routing
            tau, _, _, _, _ = model.network.out(
                coords,
                ee_y=ee_y.detach(), ee_xy=ee_xy.detach(),
                d_orient_cup=d_cup.detach(),
                d_orient_pour=d_pour.detach(),
                detach_input=False)
            gradient = torch.autograd.grad(tau[:, 0].sum(), state)[0]
            direction = -gradient
            direction = direction / torch.linalg.vector_norm(
                direction, dim=1, keepdim=True).clamp_min(1.0e-8)

        active_remaining = remaining_distance[active_index]
        arc = torch.full_like(active_remaining, float(step_size))
        if budget_feasible_step:
            steps_left = max(steps - iteration, 1)
            arc = torch.maximum(arc, active_remaining / float(steps_left))
        arc = torch.minimum(arc, active_remaining)
        next_state = (state.detach() + arc[:, None] * direction.detach()).clamp(
            min=-0.5, max=0.5)
        current[active_index] = next_state
        for local, global_index in enumerate(active_index.tolist()):
            trajectories[global_index].append(next_state[local:local + 1].clone())

    return trajectories, n_iters


def plan_cross_two_stage(model, cross_XP, Wn, wtree, robot_model, q_center, q_range,
                         mppi_kwargs, k_candidates, tilt_cap,
                         cuboid_centers, cuboid_half, self_coll_mask,
                         no_self_collision, collision_margin, device,
                         leg_reach_thresh=0.08, ts_y_divide=0.0, ts_overlap=0.15,
                         zone_mode="y", corner_zones=None):
    """Plan a batch of CROSS pairs via two-stage waypoint routing.

    For each cross pair, retrieve its K nearest both-constraint waypoints (to the
    start/goal midpoint). For candidate k (k=0..K-1), plan leg1 (start->W_k) and leg2
    (W_k->goal) with the EXISTING batched MPPI, concatenate, and accept the first
    candidate whose full 2-leg path is collision-free AND every waypoint's zone-tilt
    <= tilt_cap AND both legs reach. Once a pair is accepted, it is frozen.

    Args:
        cross_XP: (C, 2*dim) normalized start-goal pairs (cross only)
        Wn: (M, dim) normalized both-constraint waypoints
        wtree: cKDTree over Wn
        mppi_kwargs: dict of keyword args forwarded to MPPI_batch_multi (the leg planner).
                     pair_type / cross-specific objective flags should NOT be set here so
                     each leg is planned as a plain reach (intra-like) plan.

    Returns:
        (trajs, reached) where trajs is a list of C trajectories (each a list of
        (1, dim) norm-space tensors, matching MPPI_batch_multi format) and reached is
        a (C,) bool array: True if the chosen 2-leg path ends at the original goal.
    """
    C = cross_XP.shape[0]
    dim = cross_XP.shape[1] // 2
    qr = q_range.detach().cpu().numpy()
    qc = q_center.detach().cpu().numpy()
    task_axes = mppi_kwargs.get("task_axes")

    starts = cross_XP[:, :dim]
    goals = cross_XP[:, dim:]

    # K nearest both-constraint waypoints to each pair midpoint (normalized C-space)
    mids = (0.5 * (starts + goals)).detach().cpu().numpy()
    K = max(1, k_candidates)
    _, wis = wtree.query(mids, k=K)  # (C, K) or (C,) when K==1
    wis = np.atleast_2d(wis.reshape(C, K))

    # Best-so-far trajectory per cross pair (defaults to a degenerate start-only path;
    # will be overwritten by the first accepted candidate). If none accepted, the last
    # attempted leg-concatenation is kept so the strict check fails honestly.
    best_traj = [None] * C
    accepted = np.zeros(C, dtype=bool)
    reached = np.zeros(C, dtype=bool)

    for k in range(K):
        pending = np.where(~accepted)[0]
        if len(pending) == 0:
            break
        pidx = torch.tensor(pending, dtype=torch.long, device=device)
        w_k = torch.tensor(Wn[wis[pending, k]], dtype=torch.float32, device=device)  # (P, dim)
        s_p = starts[pidx]
        g_p = goals[pidx]

        # ── Leg 1: start -> W ──
        leg1_XP = torch.cat([s_p, w_k], dim=1).clone()
        with torch.no_grad():
            leg1_trajs, _ = MPPI_batch_multi(
                model, leg1_XP, robot_model, q_center, q_range,
                dim=dim, **mppi_kwargs)

        # ── Leg 2: W -> goal ──
        leg2_XP = torch.cat([w_k, g_p], dim=1).clone()
        with torch.no_grad():
            leg2_trajs, _ = MPPI_batch_multi(
                model, leg2_XP, robot_model, q_center, q_range,
                dim=dim, **mppi_kwargs)

        for j, gi in enumerate(pending):
            t1 = torch.cat(leg1_trajs[j], dim=0)  # (T1, dim) norm
            t2 = torch.cat(leg2_trajs[j], dim=0)  # (T2, dim) norm
            full = torch.cat([t1, t2], dim=0)     # (T1+T2, dim)

            # store as list of (1, dim) tensors to match MPPI_batch_multi format
            full_list = [full[t:t+1].clone() for t in range(full.shape[0])]

            # leg-reach test (t2[-1] is the appended goal slot; use the planned
            # second-to-last waypoint to judge whether leg2 actually converged)
            w_reached = (t1[-1] - w_k[j]).norm().item() < leg_reach_thresh
            g_node = t2[-2] if t2.shape[0] >= 2 else t2[-1]
            g_reached = (g_node - g_p[j]).norm().item() < leg_reach_thresh

            if best_traj[gi] is None:
                best_traj[gi] = full_list  # keep last attempt as fallback
                reached[gi] = bool(w_reached and g_reached)

            if not (w_reached and g_reached):
                continue

            # per-waypoint zone-tilt <= cap, matching the eval's strict check:
            # hard zone split outside the overlap band, OR-logic (either head valid)
            # inside [y_divide-overlap, y_divide+overlap]. The old hard ee_y>=0 split
            # rejected boundary waypoints the eval would accept (cost ~1 pair on v7).
            q_rad = full * q_range.unsqueeze(0) + q_center.unsqueeze(0)
            if zone_mode == "corner_table":
                cup, pour, _, ee_pos = _zone_tilts_batch(
                    q_rad, robot_model, return_pos=True, task_axes=task_axes)
                valid = torch.ones_like(cup, dtype=torch.bool)
                cup_mask, pour_mask, in_band = corner_masks_torch(
                    ee_pos[:, :2], corner_zones,
                    valid_mask=valid, assign_outside=True)
                zt_hard = torch.where(cup_mask, cup, pour)
                wp_ok = torch.where(
                    in_band,
                    (torch.minimum(cup, pour) <= tilt_cap),
                    (zt_hard <= tilt_cap))
            else:
                cup, pour, ee_y = _zone_tilts_batch(
                    q_rad, robot_model, task_axes=task_axes)
                zt_hard = torch.where(ee_y >= ts_y_divide, cup, pour)
                in_band = (ee_y - ts_y_divide).abs() <= ts_overlap
                wp_ok = torch.where(
                    in_band,
                    (torch.minimum(cup, pour) <= tilt_cap),   # overlap: either head valid
                    (zt_hard <= tilt_cap))                      # outside: zone-appropriate
            tilt_ok = wp_ok.all().item()
            if not tilt_ok:
                continue

            # full path collision-free
            if no_self_collision:
                _, dr = check_collision_full(
                    robot_model, q_rad, cuboid_centers, cuboid_half,
                    torso_cuboid_centers=None, torso_cuboid_half=None,
                    self_collision_mask=None,
                    margin=collision_margin, device=device)
                coll_free = (dr["obstacle"] > 0).all().item()
            else:
                _, dr = check_collision_full(
                    robot_model, q_rad, cuboid_centers, cuboid_half,
                    torso_cuboid_centers=None, torso_cuboid_half=None,
                    self_collision_mask=self_coll_mask,
                    margin=collision_margin, device=device)
                coll_free = ((dr["obstacle"] > 0) & (dr["self"] > 0.002)).all().item()
            if not coll_free:
                continue

            # accepted: freeze this candidate's path
            best_traj[gi] = full_list
            accepted[gi] = True
            reached[gi] = True

    print(f"[2stage] cross pairs accepted (valid 2-leg path found): "
          f"{int(accepted.sum())}/{C}")
    return best_traj, reached


def main():
    parser = argparse.ArgumentParser(description="UR5e NTField Multi-Task Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained multi-task model checkpoint")
    parser.add_argument("--ensemble_checkpoints", type=str, default="",
                        help="Optional comma-separated extra checkpoints. MPPI cost "
                             "uses base checkpoint plus these learned fields.")
    parser.add_argument("--ensemble_mode", type=str, default="mean",
                        choices=["mean", "min", "max"],
                        help="How to combine learned tau values for ensemble MPPI.")
    parser.add_argument("--tau_cost_mode", type=str, default="first_last",
                        choices=["first_last", "last", "mean", "sum", "max",
                                 "first_mean", "first_max", "discount"],
                        help="Learned-tau-only MPPI candidate cost aggregation. "
                             "first_last preserves the original 10*tau(first)+tau(last).")
    parser.add_argument("--tau_first_weight", type=float, default=10.0,
                        help="Weight on the first horizon learned tau for first_last "
                             "and first_mean cost modes. Default 10 preserves prior behavior.")
    parser.add_argument("--mppi_cost_beta", type=float, default=50.0,
                        help="Inverse temperature for learned-cost MPPI sample weighting. "
                             "Default 50 preserves prior behavior.")
    parser.add_argument("--learned_select_best", action="store_true",
                        help="Select the argmin learned-tau MPPI sample at each step "
                             "instead of softmax averaging samples. This uses only "
                             "learned field cost, not FK safety reranking.")
    parser.add_argument("--constant_arc_step", action="store_true",
                        help="Normalize the learned MPPI update to the configured "
                             "step_size (clipped at the goal). This preserves the "
                             "learned direction while removing the accidental "
                             "distance horizon caused by vector averaging.")
    parser.add_argument("--budget_feasible_step", action="store_true",
                        help="Preserve the learned MPPI update magnitude, but raise "
                             "it to remaining_distance / remaining_steps when "
                             "otherwise convergence is numerically impossible.")
    parser.add_argument("--scene", type=str, default="scene_ur5e.yml")
    parser.add_argument("--robot_cfg", type=str,
                        default=str(Path(__file__).parent / "ur5e_arm.yml"))
    parser.add_argument("--urdf", type=str, default=None,
                        help="Path to ur5e_description dir (patches URDF paths in robot_cfg)")
    parser.add_argument("--norm_file", type=str,
                        default="datasets/arm/UR5e_multi_balanced_1M/joint_normalization.npz",
                        help="Joint normalization params from preprocessing")
    parser.add_argument("--n_queries", type=int, default=20,
                        help="Number of start-goal pairs to test")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--step_size", type=float, default=0.015)
    parser.add_argument("--sample_num", type=int, default=50)
    parser.add_argument("--mppi_steps", type=int, default=200)
    parser.add_argument("--planner", type=str, default="mppi",
                        choices=["mppi", "endpoint_line", "characteristic"],
                        help="Planner backend. endpoint_line uses direct joint-space "
                             "interpolation; characteristic integrates normalized "
                             "descent of the sole learned scalar.")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Batch size for MPPI (all queries at once by default)")
    parser.add_argument("--no_self_collision", action="store_true",
                        help="Disable self-collision and torso checking (obstacle only)")
    parser.add_argument("--warmup_iters", type=int, default=2,
                        help="discarded solves before timing starts, to keep "
                             "generation time free of one-off CUDA/graph setup "
                             "(0 restores the pre-2026-09-13 cold-start timing)")
    parser.add_argument("--warmup_steps", type=int, default=12,
                        help="MPPI iterations per warmup solve (shapes matter, "
                             "iteration count does not)")
    parser.add_argument("--no_joint_limit_clamp", action="store_true",
                        help="disable bounding MPPI to the robot's joint limits "
                             "(restores the pre-2026-09-13 unbounded behaviour)")
    parser.add_argument("--save_trajectories", type=str, default=None,
                        help="Save trajectories to .npz for visualization")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for query sampling (default 42)")
    parser.add_argument("--eval_pairs_dir", type=str, default=None,
                        help="Optional directory containing sampled_points.npy "
                             "+ joint_normalization.npz to use as the eval pair pool.")
    parser.add_argument("--eval_set", type=str, default=None,
                        help="Path to fixed eval set .npz with starts_rad, goals_rad, "
                             "starts_norm, goals_norm. Overrides --eval_pairs_dir.")
    parser.add_argument("--horizon", type=int, default=5,
                        help="MPPI lookahead horizon (default 5)")
    parser.add_argument("--tilt_interp_step", type=float, default=0.005,
                        help="Max joint step (rad) between tilt samples. The collision "
                             "checker already densifies; without this the tilt gate only "
                             "saw the raw MPPI waypoints and missed mid-segment "
                             "excursions. 0 disables (legacy, unsound).")
    parser.add_argument("--collision_margin", type=float, default=0.005,
                        help="Collision safety margin in meters (default 0.005 = 5mm)")

    # Multi-task specific args
    parser.add_argument("--y_divide", type=float, default=0.0,
                        help="Y-coordinate zone boundary (default 0.0)")
    parser.add_argument("--overlap", type=float, default=0.05,
                        help="Half-width of overlap zone (default 0.05)")
    parser.add_argument("--route_temp", type=float, default=0.01,
                        help="Softmax temperature for constraint-aware routing (default 0.01)")
    parser.add_argument("--zone_mode", type=str, default="auto",
                        choices=["auto", "y", "corner_table"],
                        help="Routing geometry. auto uses checkpoint zone_params, falling "
                             "back to scene zones.type.")
    parser.add_argument("--corner_spatial_temp", type=float, default=0.05,
                        help="Spatial fallback temperature for corner-table routing.")
    parser.add_argument("--margin", type=float, default=0.05,
                        help="Orient distance margin (must match training, default 0.05)")
    parser.add_argument("--cup_tilt_max", type=float, default=20.0,
                        help="Cupping max tilt from vertical in degrees (must match training)")
    parser.add_argument("--cup_tilt_soft", type=float, default=10.0,
                        help="Cupping soft margin beyond tilt_max in degrees")
    parser.add_argument("--pour_tilt_max", type=float, default=20.0,
                        help="Pouring max X-tilt in degrees (must match training)")
    parser.add_argument("--pour_tilt_soft", type=float, default=10.0,
                        help="Pouring soft margin beyond tilt_max in degrees")
    parser.add_argument("--pour_rzy_safe", type=float, default=0.15,
                        help="Rz_y threshold above which cup-left-only is safe")
    parser.add_argument("--pour_rzy_soft", type=float, default=0.30,
                        help="Rz_y soft transition range")

    # Tilt checking (post-hoc validation)
    parser.add_argument("--max_tilt_deg", type=float, default=None,
                        help="If set, check trajectory tilt for strict success metric.")
    parser.add_argument("--task", type=str, default="cupping",
                        choices=["cupping", "pouring"],
                        help="Task type for tilt checking (default cupping)")

    # Zone-aware tilt evaluation (multi-task)
    parser.add_argument("--zone_eval", action="store_true",
                        help="Enable zone-aware tilt checking (overrides --max_tilt_deg/--task)")
    parser.add_argument("--cup_eval_tilt", type=float, default=20.0,
                        help="Cupping tilt threshold for zone eval (degrees, default 20)")
    parser.add_argument("--pour_eval_tilt", type=float, default=20.0,
                        help="Pouring tilt threshold for zone eval (degrees, default 20)")
    parser.add_argument("--morph_orient", action="store_true",
                        help="R60: overlap-band acceptance uses morphed cup->pour tilt.")
    parser.add_argument("--morph_mode", type=str, default="linear", choices=["linear", "max"])
    parser.add_argument("--handover_accept", action="store_true",
                        help="corner-table: replace the per-waypoint mid-corner "
                             "angle pin with a monotonic cup->pour handover "
                             "constraint (wings stay strict; no pinned mid angle).")
    parser.add_argument("--handover_mono_tol", type=float, default=0.10,
                        help="tolerance for the handover-coordinate monotonicity "
                             "(allows small non-monotone noise).")
    parser.add_argument("--corner_overlap_deg", type=float, default=45.0,
                        help="L-shape corner morph: angular half-width (deg) of the "
                             "OR-band around the inner diagonal for acceptance. Must "
                             "match the value used in training data generation.")
    parser.add_argument("--tilt_overlap", type=float, default=0.0,
                        help="OR-logic tilt check: half-width of overlap zone where EITHER "
                             "cupping or pouring constraint suffices. "
                             "0 = legacy hard split at y_divide (default). "
                             "E.g. 0.05 means overlap zone is [y_divide-0.05, y_divide+0.05].")
    parser.add_argument(
        "--overlap_both", action="store_true",
        help=("Require BOTH cupping and pouring tilt constraints in the "
              "explicit corner overlap zone. This is the pure scalar "
              "contract; without it, a positive --tilt_overlap retains "
              "the historical OR rule."))
    parser.add_argument("--force_intra_routing", action="store_true",
                        help="Diagnostic: for fixed eval sets with pair_type metadata, "
                             "force intra-cupping pairs through the cup head and "
                             "intra-pouring pairs through the pour head during MPPI. "
                             "Cross pairs still use dynamic routing.")
    parser.add_argument("--linear_fallback", action="store_true",
                        help="If an NTField trajectory fails strict validation, try a "
                             "straight-line joint-space trajectory and use it only when "
                             "it is collision-free and tilt-valid.")
    parser.add_argument("--linear_fallback_waypoints", type=int, default=101,
                        help="Number of waypoints for --linear_fallback (default 101).")
    parser.add_argument("--safety_rerank", action="store_true",
                        help="Add zone-aware tilt and collision penalties to MPPI "
                             "candidate costs during sampling.")
    parser.add_argument("--safety_select_best", action="store_true",
                        help="With --safety_rerank, take the lowest-cost sampled first "
                             "step instead of a softmax-weighted average.")
    parser.add_argument("--safety_tilt_weight", type=float, default=0.02,
                        help="Cost weight per degree of candidate zone-tilt violation.")
    parser.add_argument("--orient_tau_lambda", type=float, default=None,
                        help="R58R: override trained orient_tau_lambda (Phi-potential weight in "
                             "tau_total=tau+lambda*Phi) at eval. Sweep without retraining.")
    parser.add_argument("--orient_mppi_weight", type=float, default=0.0,
                        help="R58F: weight on the field's decoupled orientation cost "
                             "Phi_orient(current config) added to the MPPI cost. "
                             "0=off; auto-enabled to 1.0 if the loaded model has the "
                             "orient head and this is left at 0. Ignored if model has no head.")
    parser.add_argument("--graded_tilt_weight", type=float, default=0.0,
                        help="R58F PoC: cost weight per degree of RAW zone-tilt (graded, "
                             "active inside the cone too, smooth, no hard-filter). Emulates an "
                             "additive decoupled orientation potential. 0=off.")
    parser.add_argument("--handover_mppi_weight", type=float, default=0.0,
                        help="Cost weight for corner-table handover monotonicity "
                             "inside MPPI candidates. Matches --handover_accept's "
                             "h=cup/(cup+pour) nondecreasing check; 0=off.")
    parser.add_argument("--handover_mppi_tol", type=float, default=None,
                        help="Tolerance for --handover_mppi_weight. Defaults to "
                             "--handover_mono_tol.")
    parser.add_argument("--safety_collision_weight", type=float, default=20.0,
                        help="Cost weight per meter of candidate clearance violation.")
    parser.add_argument("--safety_hard_filter", action="store_true",
                        help="With --safety_rerank, add a large penalty to candidate "
                             "horizons that violate zone tilt or collision margin.")
    parser.add_argument("--safety_hard_penalty", type=float, default=10.0,
                        help="Penalty used by --safety_hard_filter.")
    parser.add_argument("--line_guide", action="store_true",
                        help="Inject one deterministic sample that moves directly toward "
                             "the goal in joint space at each MPPI step.")
    parser.add_argument("--cross_line_blend", type=float, default=0.0,
                        help="R58H: blend weight [0,1] of a straight-line-to-goal step into "
                             "the MPPI step for CROSS pairs (0=pure field, 1=pure straight "
                             "line). The cross tau field is monotone in value but gradient-"
                             "saturated in the far-field; this supplies direction where it is "
                             "flat. Applied only to pair_type==2.")
    parser.add_argument("--goal_cost_weight", type=float, default=0.0,
                        help="Cost weight for normalized terminal distance to the goal "
                             "inside MPPI. Default 0 preserves the original learned-tau-only planner.")
    parser.add_argument("--task_adaptive_objective", action="store_true",
                        help="For fixed multi-task eval sets with pair_type metadata, apply "
                             "goal/safety objective terms only to non-cupping pairs. "
                             "Intra-cupping keeps the original learned-tau-only objective.")

    # R58 two-stage waypoint planning for CROSS pairs
    parser.add_argument("--two_stage_waypoint", action="store_true",
                        help="R58: route each CROSS (cup->pour) pair through an intermediate "
                             "both-constraint boundary waypoint. Each pair is planned as two "
                             "sequential batched-MPPI legs (start->W, W->goal); the concatenated "
                             "2-leg path is then run through the SAME strict check. INTRA "
                             "(cupping/pouring) pairs are unaffected (single MPPI pass).")
    parser.add_argument("--ts_tilt_cap", type=float, default=20.0,
                        help="Two-stage: max tilt (deg) for BOTH cup AND pour at a waypoint, "
                             "and the per-waypoint zone-tilt cap used to accept a 2-leg path.")
    parser.add_argument("--ts_k_candidates", type=int, default=15,
                        help="Two-stage: try the K nearest-to-midpoint both-constraint "
                             "waypoints per cross pair; keep the first that yields a valid "
                             "2-leg path.")
    parser.add_argument("--ts_wp_pool", type=int, default=200000,
                        help="Two-stage: number of FK configs to subsample when building the "
                             "both-constraint waypoint pool.")
    parser.add_argument("--ts_pkl_cup", type=str,
                        default="data/table_fk_2000000pts_h0.30_tilt20.pkl",
                        help="Two-stage: cupping FK pkl (key 'q_solutions') for waypoint pool.")
    parser.add_argument("--ts_pkl_pour", type=str,
                        default="data/table_fk_2000000pts_h0.30_pour90_pouring.pkl",
                        help="Two-stage: pouring FK pkl (key 'q_solutions') for waypoint pool.")

    args = parser.parse_args()

    if args.constant_arc_step and args.budget_feasible_step:
        parser.error(
            "--constant_arc_step and --budget_feasible_step are mutually exclusive")

    device = args.device
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── 1. Load joint normalization ───────────────────────────────────────────
    norm = np.load(args.norm_file)
    # Joint limits for the strict-success check.  Absent from older norm files, in
    # which case the check is skipped and a warning is printed rather than silently
    # passing everything.
    q_limit_min = q_limit_max = None
    if "q_min" in norm and "q_max" in norm:
        q_limit_min = torch.as_tensor(norm["q_min"], dtype=torch.float32, device=device)
        q_limit_max = torch.as_tensor(norm["q_max"], dtype=torch.float32, device=device)
    else:
        print("WARNING: norm file has no q_min/q_max -- joint-limit check DISABLED")
    q_center = torch.tensor(norm["q_center"], dtype=torch.float32, device=device)
    q_range = torch.tensor(norm["q_range"], dtype=torch.float32, device=device)
    q_min = torch.tensor(norm["q_min"], dtype=torch.float32, device=device)
    q_max = torch.tensor(norm["q_max"], dtype=torch.float32, device=device)
    # The same limits expressed in the normalized coordinates MPPI integrates in,
    # so the planner can be bounded the way a sampling planner's state space is.
    q_norm_lo = (q_min - q_center) / q_range
    q_norm_hi = (q_max - q_center) / q_range
    # One warmup for the whole evaluation, not one per batch.
    _warmed_up = {'done': False}

    print(f"Joint normalization loaded from {args.norm_file}")

    # ── 2. Load multi-task NTField model ─────────────────────────────────────
    print(f"Loading multi-task model from {args.checkpoint}...")
    modelPath = './Experiments/multi'
    dataPath = './datasets/arm/UR5e_multi_balanced_1M'
    source = [0.0] * DOF

    model = _load_pure_hjb_model(args.checkpoint, device)
    pure_hjb = model is not None
    if model is None:
        model = MultiModel(modelPath, dataPath, DOF, source, device=device,
                           y_divide=args.y_divide, overlap=args.overlap,
                           route_temp=args.route_temp, cup_side=1)
        model.load(args.checkpoint)
        # R63: a checkpoint trained with FK-recomputed interior cond MUST be
        # evaluated the same way, or tau at inference is computed from an
        # interpolated interior the network never saw.  Not a tuning knob -- it
        # follows the checkpoint.
        if getattr(model.network, 'interior_routing_requested', False):
            from train.train_g1_multi import (
                configure_interior_routing_from_dataset)
            configure_interior_routing_from_dataset(
                model.network, str(Path(args.norm_file).parent), device=device)
    model.network.eval()
    model_dim = int(model.dim)
    lifted_phase = bool(getattr(model.network, "lifted_phase", False))
    model_physical_dim = int(getattr(
        model.network, "lifted_phase_physical_dim", model_dim))
    if lifted_phase and model_physical_dim != DOF:
        raise ValueError(
            f"lifted checkpoint physical_dim={model_physical_dim}, "
            f"but UR5e DOF={DOF}")
    if not lifted_phase and model_dim != DOF:
        raise ValueError(
            f"non-lifted UR5e checkpoint dim={model_dim}, expected {DOF}")
    print(
          f"Model loaded ({'pure_hjb' if pure_hjb else 'legacy_multi'}). "
          f"Zone params: y_divide={model.network.y_divide}, "
          f"overlap={model.network.overlap}, route_temp={model.network.route_temp}, "
          f"weight_mode={model.network.weight_mode}")
    # R62: model.load() restores route_temp from the checkpoint's zone_params, so the
    # constructor arg is discarded and --route_temp had no effect. Re-apply it here so
    # the routing softmax temperature can be swept without retraining. The effective
    # temperature in DEGREES is tilt_band * route_temp / margin (3.0 deg at 0.01).
    _old_rt = getattr(model.network, "route_temp", None)
    if (getattr(args, "route_temp", None) is not None
            and _old_rt is not None
            and float(args.route_temp) != float(_old_rt)):
        # Changing route_temp at eval MUTATES THE MODEL: the two task heads
        # specialised under the trained temperature, and it also shaped the anchor's
        # regression target during training. Decoding at a different temperature is a
        # train/test mismatch, not a planner setting -- the same class of error as
        # overriding orient_tau_lambda (measured 2026-09-03: 54/180 as trained vs
        # 141/180 with the override). To test a different value, RETRAIN at it.
        if os.environ.get("ALLOW_ROUTE_TEMP_OVERRIDE", "0") != "1":
            raise ValueError(
                f"--route_temp {args.route_temp} differs from the checkpoint's "
                f"{_old_rt}. Overriding it at eval mutates the model and makes the "
                f"result a train/test mismatch. Retrain at the value you want, or set "
                f"ALLOW_ROUTE_TEMP_OVERRIDE=1 for a deliberate diagnostic.")
        model.network.route_temp = float(args.route_temp)
        print(f"  [R62] DIAGNOSTIC route_temp override: {_old_rt} -> "
              f"{model.network.route_temp}  (train/test MISMATCH, not a clean result)")
    print(
        f"Model state: model_dim={model_dim}, physical_dim={DOF}, "
        f"lifted_phase={lifted_phase}, lifted_phase_causal="
        f"{bool(getattr(model.network, 'lifted_phase_causal', False))}")
    scene_zone_mode, scene_corner_zones = load_zone_config(args.scene, "auto")
    task_axes = load_task_axes(args.scene)
    eval_meta = _load_eval_metadata(args.eval_set)
    _sync_eval_args_from_metadata(args, eval_meta, task_axes)
    overlap_logic = "both" if args.overlap_both else "or"
    print(f"Tilt overlap logic: {overlap_logic.upper()}")
    ckpt_zone_mode = getattr(model.network, "zone_mode", "y")
    if args.zone_mode == "auto":
        zone_mode = ckpt_zone_mode if ckpt_zone_mode != "y" else scene_zone_mode
    else:
        zone_mode = args.zone_mode
    corner_zones = getattr(model.network, "corner_zones", None)
    if zone_mode == "corner_table" and corner_zones is None:
        corner_zones = scene_corner_zones
    if zone_mode == "corner_table" and corner_zones is None:
        raise ValueError("corner_table eval requires corner_zones in checkpoint or scene yaml")
    model.network.zone_mode = zone_mode
    model.network.corner_zones = corner_zones
    model.network.corner_spatial_temp = args.corner_spatial_temp

    # VALIDITY FIX (2026-09-11): the SCORING zones must come from the BENCHMARK, not
    # from the model being scored.  Previously the tilt gate used the checkpoint's own
    # corner_zones, so a model trained with a wide overlap band was graded with a wide
    # band and one with a narrow band graded narrowly -- different exams.  Nothing
    # stopped a model from declaring a lenient band and "improving" by construction.
    #
    # The model's ROUTING geometry (corner_zones, above) still follows the checkpoint:
    # the network was trained with it and must keep it.  Only SCORING is pinned.
    def _norm_zones(z):
        # checkpoint stores lists (via JSON), the scene loader yields tuples -- compare
        # on value, not on container type, or every run prints a spurious mismatch.
        if not isinstance(z, dict):
            return z
        return {k: (list(v) if isinstance(v, (list, tuple)) else v)
                for k, v in z.items()}

    # The SCENE is the task definition: zones are a property of the workspace geometry,
    # not of a query list.  The suite's own metadata can be stale relative to the scene
    # it was generated for -- this suite still records overlap 0.05 while the scene (and
    # all training since) uses 0.10.  Prefer the scene, but say so loudly if the suite
    # disagrees, because that disagreement is exactly how a benchmark silently drifts.
    score_corner_zones, score_zones_src = corner_zones, "checkpoint (LAST RESORT)"
    if scene_corner_zones is not None:
        score_corner_zones, score_zones_src = scene_corner_zones, "scene yaml"
    _suite_raw = None
    try:
        _sz = np.load(args.eval_set, allow_pickle=True)
        if "corner_zones_json" in _sz.files:
            _suite_raw = json.loads(str(_sz["corner_zones_json"]))
    except Exception:
        pass
    print(f"Scoring geometry: from {score_zones_src}")
    if _suite_raw is not None and _norm_zones(_suite_raw) != _norm_zones(score_corner_zones):
        print(f"  WARNING: the eval suite records DIFFERENT zones than the scene.")
        print(f"    suite : {_suite_raw}")
        print(f"    using : {score_corner_zones}")
        print(f"    (the scene is authoritative; the suite metadata may be stale)")
    if (score_corner_zones is not None
            and _norm_zones(score_corner_zones) != _norm_zones(corner_zones)):
        print(f"  NOTE: scoring zones differ from the model's routing zones")
        print(f"    scoring : {score_corner_zones}")
        print(f"    routing : {corner_zones}")
    print(f"Routing geometry: zone_mode={zone_mode}")
    if corner_zones is not None:
        print(f"  corner_zones={corner_zones}")
    cup_local, cup_world = task_axis_arrays(task_axes, "cup")
    pour_local, pour_world = task_axis_arrays(task_axes, "pour")
    print(f"Task axes: cup local {axis_label(cup_local)} -> world {axis_label(cup_world)}; "
          f"pour local {axis_label(pour_local)} -> world {axis_label(pour_world)}")
    # R58F: auto-enable the orientation-cost MPPI term if the model has the head
    # but the user left the weight at 0 (so head models get the term by default,
    # while head-less models are completely unaffected).
    has_orient_head = getattr(model.network, 'use_orient_head', False)
    # R58G: when the orientation cost is baked into tau (add_orient_to_tau), out()
    # already returns tau_total = tau + lambda*Phi during MPPI. Adding the separate
    # orient_mppi_weight cost term on top would DOUBLE-COUNT Phi, so we keep it off
    # (unless the user explicitly passed a non-zero --orient_mppi_weight).
    add_orient_to_tau = getattr(model.network, 'add_orient_to_tau', False)
    # R58R: allow overriding the trained orient_tau_lambda at eval to sweep how strongly
    # the Phi(start) potential bends tau, without retraining (Phi is a fixed learned fn).
    if getattr(args, 'orient_tau_lambda', None) is not None and args.orient_tau_lambda >= 0:
        old_lam = getattr(model.network, 'orient_tau_lambda', None)
        model.network.orient_tau_lambda = float(args.orient_tau_lambda)
        print(f"  [R58R] orient_tau_lambda override: {old_lam} -> {model.network.orient_tau_lambda}")
    orient_mppi_weight = args.orient_mppi_weight
    if has_orient_head and not add_orient_to_tau and orient_mppi_weight == 0.0:
        orient_mppi_weight = 1.0
    if not has_orient_head:
        orient_mppi_weight = 0.0
    print(f"R58F orient head: present={has_orient_head}, "
          f"add_orient_to_tau={add_orient_to_tau}, "
          f"orient_mppi_weight={orient_mppi_weight}")
    print(f"Planner: {args.planner}")
    # Warn if CLI args disagree with checkpoint zone params
    if args.y_divide != model.network.y_divide:
        print(f"  WARNING: --y_divide={args.y_divide} differs from checkpoint "
              f"y_divide={model.network.y_divide}. Using CLI value.")
    if args.overlap != model.network.overlap:
        print(f"  WARNING: --overlap={args.overlap} differs from checkpoint "
              f"overlap={model.network.overlap}. Using CLI value.")

    ensemble_models = []
    ensemble_paths = [
        p.strip() for p in args.ensemble_checkpoints.split(",") if p.strip()
    ]
    for ensemble_path in ensemble_paths:
        print(f"Loading ensemble model from {ensemble_path}...")
        ensemble_model = MultiModel(
            modelPath, dataPath, DOF, source, device=device,
            y_divide=args.y_divide, overlap=args.overlap,
            route_temp=args.route_temp, cup_side=1)
        ensemble_model.load(ensemble_path)
        ensemble_model.network.eval()
        ensemble_model.network.zone_mode = zone_mode
        if getattr(ensemble_model.network, "corner_zones", None) is None:
            ensemble_model.network.corner_zones = corner_zones
        ensemble_model.network.corner_spatial_temp = args.corner_spatial_temp
        ensemble_models.append(ensemble_model)
    if ensemble_models:
        print(f"Ensemble MPPI: base + {len(ensemble_models)} extra fields, "
              f"mode={args.ensemble_mode}")

    # ── 3. Setup cuRobo for collision checking ────────────────────────────────
    print(f"Setting up cuRobo RobotWorld ({args.scene})...")
    scene = load_scene_from_yaml(args.scene)
    robot_world, robot_model = setup_robot_world(args.robot_cfg, scene, device, urdf_dir=args.urdf)
    for semantic_model in [model, *ensemble_models]:
        if _configure_live_scalar_fk_runtime(
                semantic_model, robot_model, q_center, q_range):
            print(
                "  Live-FK scalar relation gate: pinned FK runtime "
                "configured")
    cuboid_centers, cuboid_half = load_cuboid_geometry(args.scene, device)

    # UR5e has no torso obstacle.
    if TORSO_OBSTACLE_PATH is not None:
        torso_scene = str(Path(args.scene).parent / TORSO_OBSTACLE_PATH) if '/' in args.scene else TORSO_OBSTACLE_PATH
        torso_centers, torso_half = load_cuboid_geometry(torso_scene, device)
    else:
        torso_centers, torso_half = None, None

    # Build self-collision mask
    with open(args.robot_cfg) as f:
        rcfg = yaml.safe_load(f)
    self_coll_ignore = rcfg["robot_cfg"]["kinematics"].get("self_collision_ignore", {})
    self_coll_mask, _ = build_self_collision_mask(
        SPHERE_YML_PATH, self_coll_ignore, COLLISION_LINK_NAMES, device=device
    )

    # ── Tilt constraint setup (for post-hoc validation) ──────────────────────
    if args.max_tilt_deg is not None:
        print(f"  Task: {args.task}")
        print(f"  Tilt constraint active: max_tilt_deg={args.max_tilt_deg}")

    # Zone-aware tilt: UR5e checks are quaternion/matrix based, no reference pose needed.
    n_ref_cup = None
    if args.zone_eval:
        print(f"  Zone-aware tilt eval enabled:")
        print(f"    cupping threshold: {args.cup_eval_tilt} deg")
        print(f"    pouring threshold: {args.pour_eval_tilt} deg")
        if zone_mode == "corner_table":
            print("    routing: corner_table regions from scene/checkpoint")
        elif args.tilt_overlap > 0:
            lower = args.y_divide - args.tilt_overlap
            upper = args.y_divide + args.tilt_overlap
            print(f"    OR-logic overlap: [{lower:.3f}, {upper:.3f}] (half-width={args.tilt_overlap})")
            print(f"      cup-exclusive: ee_y > {upper:.3f}")
            print(f"      overlap (BOTH): {lower:.3f} <= ee_y <= {upper:.3f}")
            print(f"      pour-exclusive: ee_y < {lower:.3f}")

    # Load eval set metadata for per-query zone info. This is independent of
    # --zone_eval because planner options such as --cross_line_blend also need
    # pair_type to identify cross-zone queries.
    eval_pair_types = eval_meta.get("pair_type")
    eval_direct = eval_meta.get("direct_line_strict_feasible")
    eval_query_groups = (
        eval_meta.get("query_group_label", eval_meta.get("scenario_label")))
    if eval_query_groups is None and eval_pair_types is not None:
        eval_query_groups = derive_query_groups(eval_pair_types, eval_direct)
    eval_ee_y_goal = eval_meta.get("ee_y_goal")
    eval_start_zones = eval_meta.get("start_zone")
    eval_goal_zones = eval_meta.get("goal_zone")

    # ── 4. Sample start-goal pairs ────────────────────────────────────────────
    if args.eval_set is not None:
        print(f"\nLoading fixed eval set: {args.eval_set}")
        eval_data = np.load(args.eval_set)
        n_avail = len(eval_data['starts_rad'])
        n_queries = min(args.n_queries, n_avail)
        starts_norm_list = []
        goals_norm_list = []
        for i in range(n_queries):
            starts_norm_list.append(torch.tensor(eval_data['starts_norm'][i], dtype=torch.float32, device=device).unsqueeze(0))
            goals_norm_list.append(torch.tensor(eval_data['goals_norm'][i], dtype=torch.float32, device=device).unsqueeze(0))
        selected = np.concatenate([eval_data['starts_norm'][:n_queries],
                                   eval_data['goals_norm'][:n_queries]], axis=1)
        print(f"Loaded {n_queries} fixed pairs.")
    elif args.eval_pairs_dir is not None:
        pairs_dir = Path(args.eval_pairs_dir)
        print(f"\nSampling {args.n_queries} start-goal pairs from {pairs_dir}...")
        pair_norm = np.load(str(pairs_dir / "joint_normalization.npz"))
        src_center = pair_norm["q_center"]
        src_range = pair_norm["q_range"]
        tgt_center = norm["q_center"]
        tgt_range = norm["q_range"]

        sampled_points_norm = np.load(str(pairs_dir / "sampled_points.npy"))
        start_src = sampled_points_norm[:, :DOF]
        goal_src = sampled_points_norm[:, DOF:]
        start_rad = start_src * src_range[None, :] + src_center[None, :]
        goal_rad = goal_src * src_range[None, :] + src_center[None, :]
        start_tgt = (start_rad - tgt_center[None, :]) / tgt_range[None, :]
        goal_tgt = (goal_rad - tgt_center[None, :]) / tgt_range[None, :]
        sampled_points = np.concatenate([start_tgt, goal_tgt], axis=1).astype(np.float32)
        print(f"  Loaded {len(sampled_points)} pairs from pool")
        if np.allclose(src_center, tgt_center) and np.allclose(src_range, tgt_range):
            print(f"  (source/target norms identical)")
        else:
            print(f"  (re-normalized pairs from source norm to model norm)")
    else:
        print(f"\nSampling {args.n_queries} start-goal pairs from training data...")
        sampled_points = np.load(
            str(Path(args.norm_file).parent / "sampled_points.npy"))
        print(f"  Loaded {len(sampled_points)} pairs from pool")

    # Random selection (skip for fixed eval set)
    if args.eval_set is None:
        rng = np.random.RandomState(args.seed)
        indices = rng.choice(len(sampled_points), size=args.n_queries, replace=False)
        selected = sampled_points[indices]
        print(f"  Selected {args.n_queries} pairs (seed={args.seed})")

    n_queries = min(args.n_queries, len(selected))

    results = {
        "success": [],
        "collision_free": [],
        "n_steps": [],
        "path_length": [],
        "planning_time": [],
        "tilt_ok": [],
        "max_tilt": [],
        "max_cup_tilt": [],
        "max_pour_tilt": [],
        "n_cup_wpts": [],
        "n_pour_wpts": [],
        "zone_label": [],
        "pair_type_label": [],
        "used_fallback": [],
        "interp_collision_free": [],
        "interp_tilt_ok": [],
        "interp_strict": [],
    }
    all_trajectories = []

    # ── R58 two-stage waypoint pool (built once) ─────────────────────────────
    ts_Wn = None
    ts_wtree = None
    if args.two_stage_waypoint:
        if lifted_phase:
            raise ValueError(
                "two-stage waypoint rescue is intentionally unavailable for "
                "the lifted single-field evaluation")
        if eval_pair_types is None:
            print("  WARNING: --two_stage_waypoint requires an eval set with pair_type "
                  "metadata (and --zone_eval). Disabling two-stage.")
            args.two_stage_waypoint = False
        else:
            print(f"\n[2stage] Building both-constraint waypoint pool "
                  f"(cap={args.ts_tilt_cap}deg, pool={args.ts_wp_pool})...")
            ts_Wn, ts_wtree = build_both_constraint_waypoints(
                args.ts_pkl_cup, args.ts_pkl_pour, robot_model,
                cuboid_centers, cuboid_half,
                tilt_cap=args.ts_tilt_cap, n_pool=args.ts_wp_pool,
                q_center_np=norm["q_center"], q_range_np=norm["q_range"],
                device=device, seed=0, task_axes=task_axes)
            if ts_Wn is None:
                print("  WARNING: no both-constraint waypoint exists at this cap; "
                      "disabling two-stage.")
                args.two_stage_waypoint = False

    # ── Batched MPPI planning ────────────────────────────────────────────────
    batch_size = min(args.batch_size, n_queries)

    for batch_start in range(0, n_queries, batch_size):
        batch_end = min(batch_start + batch_size, n_queries)
        B = batch_end - batch_start

        XP_physical_batch = torch.tensor(
            selected[batch_start:batch_end],
            dtype=torch.float32, device=device)
        batch_pair_types_meta = (
            np.asarray(eval_pair_types)[batch_start:batch_end]
            if eval_pair_types is not None else None)
        batch_start_zones = (
            np.asarray(eval_start_zones)[batch_start:batch_end]
            if eval_start_zones is not None else None)
        batch_goal_zones = (
            np.asarray(eval_goal_zones)[batch_start:batch_end]
            if eval_goal_zones is not None else None)
        batch_boundary_context = None
        if getattr(model.network, "boundary_conditioned", False):
            if batch_start_zones is None or batch_goal_zones is None:
                raise ValueError(
                    "boundary-conditioned HJB requires deterministic "
                    "start_zone and goal_zone query metadata")
            # This is an immutable, direction-free boundary parameter, not
            # pair_type and not trajectory memory.  Crossing means that the
            # two original endpoints belong to different zones, irrespective
            # of whether the query is cup->pour or pour->cup.  It is computed
            # before planning and remains fixed for every MPPI sample and
            # iteration.
            batch_boundary_context = torch.as_tensor(
                (np.asarray(batch_start_zones) < 0.5)
                != (np.asarray(batch_goal_zones) < 0.5),
                dtype=torch.float32,
                device=device)
        XP_batch = _augment_lifted_phase_pairs(
            XP_physical_batch, model,
            pair_types=batch_pair_types_meta,
            start_zones=batch_start_zones,
            goal_zones=batch_goal_zones)
        query_safety_context = any(os.environ.get(name) is not None for name in (
            "R58_SAFETY_TAU_QUERY_LAMBDA_CUP",
            "R58_SAFETY_TAU_QUERY_LAMBDA_POUR",
            "R58_SAFETY_TAU_QUERY_LAMBDA_CROSS",
        ))
        query_orient_context = any(os.environ.get(name) is not None for name in (
            "R58_ORIENT_TAU_QUERY_LAMBDA_CUP",
            "R58_ORIENT_TAU_QUERY_LAMBDA_POUR",
            "R58_ORIENT_TAU_QUERY_LAMBDA_CROSS",
        ))
        pair_type_batch = None
        if (args.force_intra_routing or args.task_adaptive_objective
                or args.cross_line_blend > 0.0
                or args.handover_mppi_weight > 0.0
                or query_safety_context or query_orient_context) and eval_pair_types is not None:
            pair_type_batch = torch.tensor(
                eval_pair_types[batch_start:batch_end],
                dtype=torch.long, device=device)

        # WARMUP (2026-09-13): the first timed solve otherwise absorbs one-off CUDA
        # context init, cuDNN autotuning, lazy allocator growth and -- with
        # NTF_GRAPH_COST=1 -- the CUDA-Graph capture itself.  Measured on the mixed
        # suite: query 0 took 0.665 s against 0.145 s for query 1.  The baselines we
        # compare against exclude warmup from their generation time (their harness
        # does three synthetic short-motion solves), so timing a cold first query
        # here would compare warm numbers against a cold one.
        #
        # The warmup runs the SAME shapes as the real solve (same batch, sample_num
        # and horizon) so it primes the identical kernels and graph entry, and it is
        # driven from the current batch's own start states with a tiny displacement,
        # so no extra geometry is touched.  Results are discarded.
        if not _warmed_up['done'] and args.warmup_iters > 0:
            # The warmup draws from torch.randn, so without this the timed run would
            # see a different RNG stream than a --warmup_iters 0 run and results would
            # silently depend on whether warmup happened (measured: 17/100 trajectories
            # changed, though no outcome flipped).  Snapshot and restore so warmup is
            # purely a timing device and cannot move a single number.
            _rng_cpu = torch.get_rng_state()
            _rng_gpu = (torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available() else None)
            with torch.no_grad():
                XP_warm = XP_batch.clone()
                XP_warm[:, model_dim:] = (
                    XP_warm[:, :model_dim] + 1.0e-3)
                for _ in range(args.warmup_iters):
                    MPPI_batch_multi(
                        model, XP_warm.clone(), robot_model, q_center, q_range,
                        q_norm_lo=(None if args.no_joint_limit_clamp else q_norm_lo),
                        q_norm_hi=(None if args.no_joint_limit_clamp else q_norm_hi),
                        margin=args.margin,
                        cup_tilt_max=args.cup_tilt_max, cup_tilt_soft=args.cup_tilt_soft,
                        pour_tilt_max=args.pour_tilt_max, pour_tilt_soft=args.pour_tilt_soft,
                        pour_rzy_safe=args.pour_rzy_safe, pour_rzy_soft=args.pour_rzy_soft,
                        dim=model_dim, physical_dim=DOF,
                        steps=args.warmup_steps, sample_num=args.sample_num,
                        horizon=args.horizon, step_size=args.step_size)
            torch.cuda.synchronize()
            torch.set_rng_state(_rng_cpu)
            if _rng_gpu is not None:
                torch.cuda.set_rng_state_all(_rng_gpu)
            _warmed_up['done'] = True
            print(f"Timing warmup: {args.warmup_iters} discarded solve(s) "
                  f"at the evaluation shapes", flush=True)

        t0 = time.time()
        with torch.no_grad():
            if args.planner == "endpoint_line":
                n_line = max(int(args.linear_fallback_waypoints), 2)
                alpha = torch.linspace(
                    0.0, 1.0, n_line,
                    dtype=XP_batch.dtype, device=device).view(-1, 1)
                batch_trajs = []
                for row in XP_physical_batch:
                    line = row[:DOF].unsqueeze(0) + alpha * (
                        row[DOF:] - row[:DOF]).unsqueeze(0)
                    batch_trajs.append([line[k:k + 1] for k in range(n_line)])
                iter_count = min(n_line - 1, max(args.mppi_steps - 2, 0))
                batch_iters = torch.full(
                    (B,), iter_count, dtype=torch.long, device=device)
            elif args.planner == "characteristic":
                batch_trajs, batch_iters = characteristic_batch_multi(
                    model, XP_batch.clone(), robot_model, q_center, q_range,
                    margin=args.margin,
                    cup_tilt_max=args.cup_tilt_max,
                    cup_tilt_soft=args.cup_tilt_soft,
                    pour_tilt_max=args.pour_tilt_max,
                    pour_tilt_soft=args.pour_tilt_soft,
                    dim=model_dim, steps=args.mppi_steps,
                    step_size=args.step_size,
                    task_axes=task_axes,
                    budget_feasible_step=args.budget_feasible_step)
            else:
                batch_trajs, batch_iters = MPPI_batch_multi(
                    model, XP_batch.clone(), robot_model, q_center, q_range,
                    q_norm_lo=(None if args.no_joint_limit_clamp else q_norm_lo),
                    q_norm_hi=(None if args.no_joint_limit_clamp else q_norm_hi),
                    margin=args.margin,
                    cup_tilt_max=args.cup_tilt_max, cup_tilt_soft=args.cup_tilt_soft,
                    pour_tilt_max=args.pour_tilt_max, pour_tilt_soft=args.pour_tilt_soft,
                    pour_rzy_safe=args.pour_rzy_safe, pour_rzy_soft=args.pour_rzy_soft,
                    dim=model_dim, physical_dim=DOF,
                    steps=args.mppi_steps, sample_num=args.sample_num,
                    horizon=args.horizon, step_size=args.step_size,
                    pair_type_batch=pair_type_batch,
                    force_intra_routing=args.force_intra_routing,
                    pair_type_context_batch=(
                        pair_type_batch
                        if (query_safety_context or query_orient_context)
                        else None),
                    boundary_context_batch=batch_boundary_context,
                    safety_rerank=args.safety_rerank,
                    safety_select_best=args.safety_select_best,
                    y_divide=args.y_divide,
                    safety_tilt_weight=args.safety_tilt_weight,
                    graded_tilt_weight=args.graded_tilt_weight,
                    safety_collision_weight=args.safety_collision_weight,
                    safety_collision_margin=args.collision_margin,
                    safety_hard_filter=args.safety_hard_filter,
                    safety_hard_penalty=args.safety_hard_penalty,
                    zone_mode=zone_mode,
                    corner_zones=corner_zones,
                    cuboid_centers=cuboid_centers,
                    cuboid_half=cuboid_half,
                    torso_centers=torso_centers,
                    torso_half=torso_half,
                    self_coll_mask=self_coll_mask,
                    no_self_collision=args.no_self_collision,
                    line_guide=args.line_guide,
                    goal_cost_weight=args.goal_cost_weight,
                    task_adaptive_objective=args.task_adaptive_objective,
                    orient_mppi_weight=orient_mppi_weight,
                    handover_mppi_weight=args.handover_mppi_weight,
                    handover_mppi_tol=(
                        args.handover_mppi_tol
                        if args.handover_mppi_tol is not None
                        else args.handover_mono_tol),
                    corner_overlap_deg=args.corner_overlap_deg,
                    handover_accept=args.handover_accept,
                    cross_line_blend=args.cross_line_blend,
                    task_axes=task_axes,
                    ensemble_models=ensemble_models,
                    ensemble_mode=args.ensemble_mode,
                    tau_cost_mode=args.tau_cost_mode,
                    tau_first_weight=args.tau_first_weight,
                    mppi_cost_beta=args.mppi_cost_beta,
                    learned_select_best=args.learned_select_best,
                    constant_arc_step=args.constant_arc_step,
                    budget_feasible_step=args.budget_feasible_step,
                )
        # ── R58 two-stage: replace CROSS pairs with two-leg waypoint-routed paths ──
        # INTRA pairs keep their single-pass MPPI trajectory from above.
        if args.planner == "mppi" and args.two_stage_waypoint and eval_pair_types is not None:
            pt_b = eval_pair_types[batch_start:batch_end]
            cross_local = np.where(pt_b == 2)[0]
            if len(cross_local) > 0:
                cross_XP = XP_batch[torch.tensor(cross_local, dtype=torch.long,
                                                 device=device)]
                # Plain reach legs: no cross-specific objective terms / pair_type.
                leg_kwargs = dict(
                    margin=args.margin,
                    cup_tilt_max=args.cup_tilt_max, cup_tilt_soft=args.cup_tilt_soft,
                    pour_tilt_max=args.pour_tilt_max, pour_tilt_soft=args.pour_tilt_soft,
                    pour_rzy_safe=args.pour_rzy_safe, pour_rzy_soft=args.pour_rzy_soft,
                    steps=args.mppi_steps, sample_num=args.sample_num,
                    horizon=args.horizon, step_size=args.step_size,
                    safety_rerank=args.safety_rerank,
                    safety_select_best=args.safety_select_best,
                    y_divide=args.y_divide,
                    safety_tilt_weight=args.safety_tilt_weight,
                    graded_tilt_weight=args.graded_tilt_weight,
                    safety_collision_weight=args.safety_collision_weight,
                    safety_collision_margin=args.collision_margin,
                    safety_hard_filter=args.safety_hard_filter,
                    safety_hard_penalty=args.safety_hard_penalty,
                    zone_mode=zone_mode,
                    corner_zones=corner_zones,
                    cuboid_centers=cuboid_centers, cuboid_half=cuboid_half,
                    torso_centers=torso_centers, torso_half=torso_half,
                    self_coll_mask=self_coll_mask,
                    no_self_collision=args.no_self_collision,
                    line_guide=args.line_guide,
                    handover_mppi_weight=args.handover_mppi_weight,
                    handover_mppi_tol=(
                        args.handover_mppi_tol
                        if args.handover_mppi_tol is not None
                        else args.handover_mono_tol),
                    corner_overlap_deg=args.corner_overlap_deg,
                    handover_accept=args.handover_accept,
                    task_axes=task_axes,
                    learned_select_best=args.learned_select_best,
                    constant_arc_step=args.constant_arc_step,
                    budget_feasible_step=args.budget_feasible_step,
                )
                cross_trajs, cross_reached = plan_cross_two_stage(
                    model, cross_XP.clone(), ts_Wn, ts_wtree,
                    robot_model, q_center, q_range,
                    leg_kwargs, args.ts_k_candidates, args.ts_tilt_cap,
                    cuboid_centers, cuboid_half, self_coll_mask,
                    args.no_self_collision, args.collision_margin, device,
                    ts_y_divide=args.y_divide, ts_overlap=args.overlap,
                    zone_mode=zone_mode, corner_zones=corner_zones)

                for li, ci in enumerate(cross_local):
                    batch_trajs[ci] = cross_trajs[li]
                    # Convergence == the 2-leg path actually reached the goal. If a leg
                    # never reached, leave it failing the strict check honestly.
                    batch_iters[ci] = (0 if cross_reached[li]
                                       else (args.mppi_steps - 1))

        planning_time = time.time() - t0
        per_query_time = planning_time / B

        # Pre-compute trajectories in radians for all queries in the batch
        all_trajs_norm = []
        all_trajs_rad = []
        for j in range(B):
            tn = torch.cat(batch_trajs[j], dim=0)  # (T, 7)
            tr = tn * q_range.unsqueeze(0) + q_center.unsqueeze(0)
            all_trajs_norm.append(tn)
            all_trajs_rad.append(tr)

        # GPU-batched per-waypoint tilt check (one FK call for all B trajectories)
        batch_tilt_results = None
        if args.zone_eval:
            cup_tilt_threshold = args.cup_eval_tilt
            pour_tilt_threshold = args.pour_eval_tilt
            batch_pair_types = None
            if eval_pair_types is not None:
                batch_pair_types = np.asarray(eval_pair_types)[batch_start:batch_start + B]
            # VALIDITY FIX (2026-09-09, reported by a collaborator): the collision
            # checker densifies between waypoints but the tilt check did not, so an
            # orientation excursion BETWEEN two compliant waypoints was invisible.
            # Measured on wall25_corner020 seed 0: 3 of 144 "successes" reach 22.4-27.2
            # deg mid-segment against a 20 deg gate.  Densify at the same resolution
            # the collision path uses so both constraints see the same trajectory.
            tilt_input_trajs = all_trajs_rad
            if args.tilt_interp_step > 0:
                tilt_input_trajs = [
                    _densify_traj(t, args.tilt_interp_step) for t in all_trajs_rad]
            with torch.no_grad():
                batch_tilt_results = _check_tilt_per_waypoint_batch(
                    tilt_input_trajs, robot_model, args.y_divide,
                    cup_tilt_threshold, pour_tilt_threshold, n_ref_cup,
                    overlap=args.tilt_overlap,
                    morph_orient=args.morph_orient, morph_mode=args.morph_mode,
                    corner_overlap_deg=args.corner_overlap_deg,
                    handover_accept=args.handover_accept,
                    handover_mono_tol=args.handover_mono_tol,
                    pair_types=batch_pair_types,
                    zone_mode=zone_mode, corner_zones=score_corner_zones,
                    task_axes=task_axes,
                    overlap_logic=overlap_logic)

        # Per-trajectory: collision check + result assembly
        for j in range(B):
            qi = batch_start + j
            n_iter = batch_iters[j].item()
            converged = n_iter < args.mppi_steps - 1

            traj_norm = all_trajs_norm[j]
            traj_rad = all_trajs_rad[j]

            if args.no_self_collision:
                in_coll, d_result = check_traj_collision_full(
                    robot_model, traj_rad, cuboid_centers, cuboid_half,
                    torso_cuboid_centers=None, torso_cuboid_half=None,
                    self_collision_mask=None,
                    margin=args.collision_margin, device=device)
            else:
                in_coll, d_result = check_traj_collision_full(
                    robot_model, traj_rad, cuboid_centers, cuboid_half,
                    torso_cuboid_centers=torso_centers, torso_cuboid_half=torso_half,
                    self_collision_mask=self_coll_mask,
                    torso_ignore_sphere_indices=TORSO_IGNORE_SPHERE_INDICES,
                    margin=args.collision_margin, device=device)
            is_collision_free = not in_coll.any().item()

            # Per-trajectory tilt check
            zone_label = "unknown"
            pair_type_label = "unknown"
            max_tilt_val = float("nan")
            tilt_ok = True
            if eval_pair_types is not None and qi < len(eval_pair_types):
                pt_val = int(eval_pair_types[qi])
                pair_type_label = {0: "cupping", 1: "pouring", 2: "cross"}.get(
                    pt_val, f"type{pt_val}")
                zone_label = pair_type_label

            if args.zone_eval:
                # Read pre-computed per-waypoint tilt results
                tilt_ok, max_cup_tilt, max_pour_tilt, n_cup, n_pour = batch_tilt_results[j]
                max_tilt_val = max(
                    max_cup_tilt if not math.isnan(max_cup_tilt) else 0.0,
                    max_pour_tilt if not math.isnan(max_pour_tilt) else 0.0)

                # Zone label from eval set metadata (reporting only)
                if eval_pair_types is None or qi >= len(eval_pair_types):
                    zone_label = "cupping" if n_cup >= n_pour else "pouring"
                    pair_type_label = zone_label

            elif args.max_tilt_deg is not None:
                # Legacy single-task tilt check
                with torch.no_grad():
                    if args.task == "pouring":
                        tilt_ok, max_tilt_val = _check_pouring_tilt(
                            traj_rad, robot_model, x_tilt_max_deg=args.max_tilt_deg,
                            task_axes=task_axes)
                    else:
                        tilt_ok, max_tilt_val = _check_cupping_tilt(
                            traj_rad, robot_model, cup_tilt_max_deg=args.max_tilt_deg,
                            task_axes=task_axes)
                zone_label = args.task
                pair_type_label = args.task

            # VALIDITY FIX (2026-09-09): strict success never checked joint bounds,
            # so trajectories leaving the URDF limits counted as successes.  Measured
            # on wall25_corner020 seed 0: 4 of 144.
            joint_ok = True
            if q_limit_min is not None:
                _t = (traj_rad if torch.is_tensor(traj_rad)
                      else torch.as_tensor(traj_rad))
                _t = _t.to(q_limit_min.device)
                joint_ok = bool(((_t >= q_limit_min - 1e-6)
                                 & (_t <= q_limit_max + 1e-6)).all().item())
            strict_ok = converged and is_collision_free and tilt_ok and joint_ok
            used_fallback = False

            alpha_interp = torch.linspace(
                0.0, 1.0, args.linear_fallback_waypoints,
                dtype=traj_norm.dtype, device=device).unsqueeze(1)
            interp_start_norm = XP_physical_batch[j, :DOF]
            interp_goal_norm = XP_physical_batch[j, DOF:]
            interp_traj_norm = (
                interp_start_norm.unsqueeze(0)
                + alpha_interp * (interp_goal_norm - interp_start_norm).unsqueeze(0)
            )
            interp_traj_rad = (
                interp_traj_norm * q_range.unsqueeze(0) + q_center.unsqueeze(0)
            )
            if args.no_self_collision:
                interp_in_coll, _ = check_traj_collision_full(
                    robot_model, interp_traj_rad, cuboid_centers, cuboid_half,
                    torso_cuboid_centers=None,
                    torso_cuboid_half=None,
                    self_collision_mask=None,
                    margin=args.collision_margin, device=device)
            else:
                interp_in_coll, _ = check_traj_collision_full(
                    robot_model, interp_traj_rad, cuboid_centers, cuboid_half,
                    torso_cuboid_centers=torso_centers, torso_cuboid_half=torso_half,
                    self_collision_mask=self_coll_mask,
                    torso_ignore_sphere_indices=TORSO_IGNORE_SPHERE_INDICES,
                    margin=args.collision_margin, device=device)
            interp_collision_free = not interp_in_coll.any().item()
            interp_tilt_ok = True
            if args.zone_eval:
                interp_tilt_ok = _check_tilt_per_waypoint_batch(
                    [interp_traj_rad], robot_model, args.y_divide,
                    args.cup_eval_tilt, args.pour_eval_tilt, n_ref_cup,
                    overlap=args.tilt_overlap,
                    morph_orient=args.morph_orient, morph_mode=args.morph_mode,
                    corner_overlap_deg=args.corner_overlap_deg,
                    handover_accept=args.handover_accept,
                    handover_mono_tol=args.handover_mono_tol,
                    pair_types=(
                        [int(eval_pair_types[qi])]
                        if eval_pair_types is not None else None),
                    zone_mode=zone_mode, corner_zones=corner_zones,
                    task_axes=task_axes,
                    overlap_logic=overlap_logic)[0][0]
            elif args.max_tilt_deg is not None:
                with torch.no_grad():
                    if args.task == "pouring":
                        interp_tilt_ok = _check_pouring_tilt(
                            interp_traj_rad, robot_model,
                            x_tilt_max_deg=args.max_tilt_deg,
                            task_axes=task_axes)[0]
                    else:
                        interp_tilt_ok = _check_cupping_tilt(
                            interp_traj_rad, robot_model,
                            cup_tilt_max_deg=args.max_tilt_deg,
                            task_axes=task_axes)[0]
            interp_strict_ok = interp_collision_free and interp_tilt_ok

            if args.linear_fallback and not strict_ok:
                alpha = torch.linspace(
                    0.0, 1.0, args.linear_fallback_waypoints,
                    dtype=traj_norm.dtype, device=device).unsqueeze(1)
                start_norm = XP_physical_batch[j, :DOF]
                goal_norm = XP_physical_batch[j, DOF:]
                fb_traj_norm = (
                    start_norm.unsqueeze(0)
                    + alpha * (goal_norm - start_norm).unsqueeze(0)
                )
                fb_traj_rad = fb_traj_norm * q_range.unsqueeze(0) + q_center.unsqueeze(0)

                if args.no_self_collision:
                    fb_in_coll, fb_d_result = check_traj_collision_full(
                        robot_model, fb_traj_rad, cuboid_centers, cuboid_half,
                        torso_cuboid_centers=None,
                        torso_cuboid_half=None,
                        self_collision_mask=None,
                        margin=args.collision_margin, device=device)
                else:
                    fb_in_coll, fb_d_result = check_traj_collision_full(
                        robot_model, fb_traj_rad, cuboid_centers, cuboid_half,
                        torso_cuboid_centers=torso_centers, torso_cuboid_half=torso_half,
                        self_collision_mask=self_coll_mask,
                        torso_ignore_sphere_indices=TORSO_IGNORE_SPHERE_INDICES,
                        margin=args.collision_margin, device=device)
                fb_collision_free = not fb_in_coll.any().item()

                fb_tilt_ok = True
                fb_max_tilt = float("nan")
                fb_max_cup_tilt = float("nan")
                fb_max_pour_tilt = float("nan")
                fb_n_cup = 0
                fb_n_pour = 0
                if args.zone_eval:
                    (
                        fb_tilt_ok,
                        fb_max_cup_tilt,
                        fb_max_pour_tilt,
                        fb_n_cup,
                        fb_n_pour,
                    ) = _check_tilt_per_waypoint_batch(
                        [fb_traj_rad], robot_model, args.y_divide,
                        args.cup_eval_tilt, args.pour_eval_tilt, n_ref_cup,
                        overlap=args.tilt_overlap,
                        morph_orient=args.morph_orient, morph_mode=args.morph_mode,
                        corner_overlap_deg=args.corner_overlap_deg,
                        handover_accept=args.handover_accept,
                        handover_mono_tol=args.handover_mono_tol,
                        pair_types=(
                            [int(eval_pair_types[qi])]
                            if eval_pair_types is not None else None),
                        zone_mode=zone_mode, corner_zones=corner_zones,
                        task_axes=task_axes,
                        overlap_logic=overlap_logic)[0]
                    fb_max_tilt = max(
                        fb_max_cup_tilt if not math.isnan(fb_max_cup_tilt) else 0.0,
                        fb_max_pour_tilt if not math.isnan(fb_max_pour_tilt) else 0.0)
                elif args.max_tilt_deg is not None:
                    with torch.no_grad():
                        if args.task == "pouring":
                            fb_tilt_ok, fb_max_tilt = _check_pouring_tilt(
                                fb_traj_rad, robot_model,
                                x_tilt_max_deg=args.max_tilt_deg,
                                task_axes=task_axes)
                        else:
                            fb_tilt_ok, fb_max_tilt = _check_cupping_tilt(
                                fb_traj_rad, robot_model,
                                cup_tilt_max_deg=args.max_tilt_deg,
                                task_axes=task_axes)

                if fb_collision_free and fb_tilt_ok:
                    traj_norm = fb_traj_norm
                    traj_rad = fb_traj_rad
                    converged = True
                    n_iter = max(args.linear_fallback_waypoints - 1, 0)
                    in_coll = fb_in_coll
                    d_result = fb_d_result
                    is_collision_free = fb_collision_free
                    tilt_ok = fb_tilt_ok
                    max_tilt_val = fb_max_tilt
                    if args.zone_eval:
                        max_cup_tilt = fb_max_cup_tilt
                        max_pour_tilt = fb_max_pour_tilt
                        n_cup = fb_n_cup
                        n_pour = fb_n_pour
                    strict_ok = True
                    used_fallback = True

            traj_norm_cpu = traj_norm.cpu()
            if traj_norm_cpu.shape[0] > 1:
                diffs = traj_norm_cpu[1:] - traj_norm_cpu[:-1]
                path_length = torch.norm(diffs, dim=1).sum().item()
            else:
                path_length = 0.0

            results["success"].append(converged)
            results["collision_free"].append(is_collision_free)
            results["n_steps"].append(n_iter + 1)
            results["path_length"].append(path_length)
            results["planning_time"].append(per_query_time)
            results["tilt_ok"].append(tilt_ok)
            results["max_tilt"].append(max_tilt_val)
            if args.zone_eval:
                results["max_cup_tilt"].append(max_cup_tilt)
                results["max_pour_tilt"].append(max_pour_tilt)
                results["n_cup_wpts"].append(n_cup)
                results["n_pour_wpts"].append(n_pour)
            else:
                results["max_cup_tilt"].append(float('nan'))
                results["max_pour_tilt"].append(float('nan'))
                results["n_cup_wpts"].append(0)
                results["n_pour_wpts"].append(0)
            results["zone_label"].append(zone_label)
            results["pair_type_label"].append(pair_type_label)
            results["used_fallback"].append(used_fallback)
            results["interp_collision_free"].append(interp_collision_free)
            results["interp_tilt_ok"].append(interp_tilt_ok)
            results["interp_strict"].append(interp_strict_ok)

            all_trajectories.append(traj_rad.cpu().numpy())

            status = "OK" if strict_ok else "FAIL"
            fallback_str = "+FB" if used_fallback else ""
            coll_str = "free" if is_collision_free else f"COLL({in_coll.sum().item()})"
            tilt_active = args.zone_eval or args.max_tilt_deg is not None
            if args.zone_eval:
                tilt_str = (
                    f"  tilt={max_tilt_val:5.1f}deg{'OK' if tilt_ok else 'X'}"
                    f" [{pair_type_label}|{n_cup}c+{n_pour}p]"
                )
            elif tilt_active:
                tilt_str = (
                    f"  tilt={max_tilt_val:5.1f}deg{'OK' if tilt_ok else 'X'}"
                    f" [{zone_label}]"
                )
            else:
                tilt_str = ""
            print(f"  [{qi+1}/{n_queries}] {status}{fallback_str}  "
                  f"converged={converged}  {coll_str}  "
                  f"steps={n_iter+1}  len={path_length:.3f}  "
                  f"time={per_query_time:.3f}s  "
                  f"d_min={d_result['min'].min():.4f}{tilt_str}")

        print(f"  Batch [{batch_start+1}-{batch_end}]: {planning_time:.3f}s total")

    # ── 5. Summary ────────────────────────────────────────────────────────────
    n = n_queries
    n_success = sum(results["success"])
    n_cf = sum(results["collision_free"])
    n_both = sum(s and c for s, c in zip(results["success"], results["collision_free"]))
    n_interp = sum(results["interp_strict"])

    print(f"\n{'='*60}")
    print(f"Results ({n} queries, multi-task model):")
    print(f"  Convergence rate:     {n_success}/{n} ({n_success/n:.0%})")
    print(f"  Collision-free rate:  {n_cf}/{n} ({n_cf/n:.0%})")
    print(f"  Full success rate:    {n_both}/{n} ({n_both/n:.0%})")
    print(f"  Interp-only SR:       {n_interp}/{n} ({n_interp/n:.0%})  "
          f"(joint-space linear, collision-free + tilt-valid)")

    tilt_active = args.zone_eval or args.max_tilt_deg is not None
    if tilt_active:
        n_tilt = sum(results["tilt_ok"])
        n_strict = sum(
            s and c and t for s, c, t in zip(
                results["success"], results["collision_free"], results["tilt_ok"]
            )
        )
        if args.zone_eval:
            print(f"  Tilt-ok rate:         {n_tilt}/{n} ({n_tilt/n:.0%})  "
                  f"(zone-aware: cup={args.cup_eval_tilt}deg, pour={args.pour_eval_tilt}deg)")
        else:
            print(f"  Tilt-ok rate:         {n_tilt}/{n} ({n_tilt/n:.0%})  "
                  f"(max_tilt_deg={args.max_tilt_deg})")
        print(f"  Strict success rate:  {n_strict}/{n} ({n_strict/n:.0%})")
        print(f"  Avg max tilt:         {np.nanmean(results['max_tilt']):.2f}deg")

    if args.zone_eval:
        # Per-waypoint zone stats
        avg_cup_wpts = np.mean(results["n_cup_wpts"])
        avg_pour_wpts = np.mean(results["n_pour_wpts"])
        avg_cup_tilt = np.nanmean(results["max_cup_tilt"])
        avg_pour_tilt = np.nanmean(results["max_pour_tilt"])
        print(f"  Per-waypoint zones:   avg {avg_cup_wpts:.0f} cup + {avg_pour_wpts:.0f} pour wpts/traj")
        print(f"  Avg max cup tilt:     {avg_cup_tilt:.1f}deg  |  Avg max pour tilt: {avg_pour_tilt:.1f}deg")

        # Per-zone breakdown
        print(f"\n  Per-zone breakdown (per-waypoint tilt):")
        zone_labels = results["zone_label"]
        pt_labels = results["pair_type_label"]

        # Group by pair_type_label (cupping/pouring/cross) for the primary breakdown
        for pt_name in ["cupping", "pouring", "cross"]:
            idxs = [i for i, pl in enumerate(pt_labels) if pl == pt_name]
            if not idxs:
                continue
            nz = len(idxs)
            nz_strict = sum(
                1 for i in idxs
                if results["success"][i] and results["collision_free"][i]
                and results["tilt_ok"][i]
            )
            nz_conv = sum(1 for i in idxs if results["success"][i])
            nz_cf = sum(1 for i in idxs if results["collision_free"][i])
            nz_tilt = sum(1 for i in idxs if results["tilt_ok"][i])
            nz_interp = sum(1 for i in idxs if results["interp_strict"][i])
            # Average max tilt for this zone (how close to passing)
            zone_tilts = [results["max_tilt"][i] for i in idxs
                          if not math.isnan(results["max_tilt"][i])]
            avg_tilt_str = f"  avg_tilt={np.mean(zone_tilts):.1f}deg" if zone_tilts else ""
            # Path-sanity: avg normalized path length per zone. Clean convergence => short
            # path; chaotic 'converged-by-luck-after-thrashing' => large path. A huge cross
            # path_len next to small cup/pour exposes the misleading conv metric.
            zone_plen = [results["path_length"][i] for i in idxs]
            plen_str = f"  path_len={np.mean(zone_plen):.1f}" if zone_plen else ""
            print(f"    {pt_name.capitalize():12s} ({nz:3d} queries):  "
                  f"strict={nz_strict}/{nz} ({nz_strict/nz:.0%})  "
                  f"conv={nz_conv}/{nz}  coll_free={nz_cf}/{nz}  "
                  f"tilt_ok={nz_tilt}/{nz}  interp={nz_interp}/{nz}"
                  f"{avg_tilt_str}{plen_str}")

    if args.linear_fallback:
        n_fb = sum(results["used_fallback"])
        print(f"  Linear fallback used: {n_fb}/{n}")

    print(f"  Avg planning time:    {np.mean(results['planning_time']):.3f}s")
    print(f"  Avg path length:      {np.mean(results['path_length']):.3f}")
    print(f"  Avg steps:            {np.mean(results['n_steps']):.1f}")
    print(f"  Planner:              {args.planner}")
    print(f"  Zone params:          zone_mode={zone_mode}, y_divide={args.y_divide}, "
          f"overlap={args.overlap}, route_temp={args.route_temp}")
    print(f"  Orient params:        margin={args.margin}, "
          f"cup_tilt={args.cup_tilt_max}/{args.cup_tilt_soft}, "
          f"pour_tilt={args.pour_tilt_max}/{args.pour_tilt_soft}")
    print(f"  Task axes:            cup {axis_label(cup_local)}->{axis_label(cup_world)}, "
          f"pour {axis_label(pour_local)}->{axis_label(pour_world)}")
    print(f"{'='*60}")

    # ── 6. Save trajectories ──────────────────────────────────────────────────
    if args.save_trajectories:
        out_path = Path(args.save_trajectories)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        success_arr = np.array(results["success"])
        collision_free_arr = np.array(results["collision_free"])
        tilt_ok_arr = np.array(results["tilt_ok"])
        strict_success_arr = success_arr & collision_free_arr & tilt_ok_arr
        np.savez(out_path,
                 trajectories=np.array(all_trajectories, dtype=object),
                 success=success_arr,
                 collision_free=collision_free_arr,
                 strict_success=strict_success_arr,
                 n_steps=np.array(results["n_steps"]),
                 path_length=np.array(results["path_length"]),
                 planning_time=np.array(results["planning_time"]),
                 tilt_ok=tilt_ok_arr,
                 max_tilt=np.array(results["max_tilt"]),
                 max_cup_tilt=np.array(results["max_cup_tilt"]),
                 max_pour_tilt=np.array(results["max_pour_tilt"]),
                 n_cup_wpts=np.array(results["n_cup_wpts"]),
                 n_pour_wpts=np.array(results["n_pour_wpts"]),
                 zone_label=np.array(results["zone_label"]),
                 pair_type_label=np.array(results["pair_type_label"]),
                 query_group_label=(
                     np.asarray(eval_query_groups)[:n]
                     if eval_query_groups is not None
                     else np.array(results["pair_type_label"])),
                 direct_line_strict_feasible=(
                     np.asarray(eval_direct, dtype=bool)[:n]
                     if eval_direct is not None
                     else np.zeros(n, dtype=bool)),
                 used_fallback=np.array(results["used_fallback"]),
                 interp_collision_free=np.array(results["interp_collision_free"]),
                 interp_tilt_ok=np.array(results["interp_tilt_ok"]),
                 interp_strict=np.array(results["interp_strict"]),
                 planner=args.planner,
                 checkpoint_path=args.checkpoint,
                 model_label=np.array(
                     f"{Path(args.checkpoint).parent.parent.name}/"
                     f"{Path(args.checkpoint).stem}"),
                 ensemble_checkpoints=args.ensemble_checkpoints,
                 eval_set_path=args.eval_set or "",
                 norm_file_path=args.norm_file,
                 seed=np.int64(args.seed),
                 configured_n_queries=np.int64(args.n_queries),
                 configured_batch_size=np.int64(args.batch_size),
                 configured_sample_num=np.int64(args.sample_num),
                 configured_horizon=np.int64(args.horizon),
                 configured_mppi_steps=np.int64(args.mppi_steps),
                 configured_step_size=np.float64(args.step_size),
                 characteristic_convergence_threshold=np.float64(0.01),
                 budget_feasible_step=np.bool_(args.budget_feasible_step),
                 linear_fallback_enabled=np.bool_(args.linear_fallback),
                 safety_rerank_enabled=np.bool_(args.safety_rerank),
                 safety_select_best_enabled=np.bool_(args.safety_select_best),
                 two_stage_waypoint_enabled=np.bool_(args.two_stage_waypoint),
                 handover_accept=np.bool_(args.handover_accept),
                 handover_mono_tol=np.float64(args.handover_mono_tol),
                 collision_margin=np.float64(args.collision_margin),
                 cup_eval_tilt=np.float64(args.cup_eval_tilt),
                 pour_eval_tilt=np.float64(args.pour_eval_tilt),
                 corner_overlap_deg=np.float64(args.corner_overlap_deg),
                 zone_mode=zone_mode,
                 corner_zones_json=json.dumps(corner_zones) if corner_zones is not None else "",
                 overlap_logic=overlap_logic,
                 task_axes_json=json.dumps(task_axes))
        print(f"Trajectories saved to {out_path}")


if __name__ == "__main__":
    main()
