"""
Multi-task training data generator — UR5e (DOF 6) fork of pkl_to_training_data_multi_v2.py.

Loads TWO pre-filtered FK samples:
  - Cupping FK sample (tilt-filtered): used for cupping-zone configs
  - Pouring FK sample (pour-filtered): used for pouring-zone configs

UR5e differences vs the G1 source:
  - DOF = 6; UR5e joint limits / collision links / sphere file; NO torso obstacle.
  - ee_link = grasp_frame (Robotiq 2F-140). Task local/world axes are read from
    the scene's task_axes block (default preserves the older cup +X->world +Z,
    pour +Z->world +X convention; the corner-table scene overrides pour to world
    -Y).
  - Zone convention: legacy scenes use cup_side=+1 / cupping on +y with
    --y_divide defaulting to 0.0. Corner-table scenes set zones.type=corner_table
    and use scene cup_region/pour_region/overlap_region for XY routing.

Usage (inside curobo container):
    python pkl_to_training_data_multi_ur5e.py \\
        --pkl_cupping data/table_fk_8000000pts_h0.30_tilt20.pkl \\
        --pkl_pouring data/table_fk_8000000pts_h0.30_pour90_pouring.pkl \\
        --scene scene_ur5e.yml --y_divide 0.0 --n_pairs 2000000 \\
        --out_dir datasets/arm/UR5e_multi
"""

import argparse
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm
try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None

# UR5e robot paths (the G1 source imported these from visualize.py).
URDF_PATH = "/opt/ur5e_description/ur5e_robotiq_2f_140.urdf"
ROBOT_CFG_PATH = "ur5e_arm.yml"
from visualize import init_curobo
from collision_utils import (
    load_cuboid_geometry,
    sphere_cuboid_distance as _sphere_cuboid_distance,
    sphere_sphere_distance,
    build_self_collision_mask,
    check_collision_full,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_NTRL_ROOT = _REPO_ROOT / "ntrl"
if str(_NTRL_ROOT) not in sys.path:
    sys.path.insert(0, str(_NTRL_ROOT))
from models.metric_arm.zone_router import (
    assign_corner_masks_np,
    assign_corner_zone_ids_np,
    corner_transition_mask_np,
    load_zone_config,
)
from ur5e_task_axes import (
    axis_label,
    load_task_axes,
    task_axes_json,
    task_axis_arrays,
    tilt_deg_from_quat,
)

# ── Constants (UR5e) ─────────────────────────────────────────────────────────

DOF = 6

# UR5e joint limits (matches sample_fk_ur5e.py Q_MIN/Q_MAX).
Q_MIN = np.array([-6.2832, -6.2832, -3.1416, -6.2832, -6.2832, -6.2832], dtype=np.float32)
Q_MAX = np.array([ 6.2832,  6.2832,  3.1416,  6.2832,  6.2832,  6.2832], dtype=np.float32)

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
# UR5e has NO torso obstacle.
TORSO_OBSTACLE_PATH = None
TORSO_IGNORE_SPHERE_INDICES = []
SPHERE_YML_PATH = "spheres_ur5e_arm.yml"

# The 2 shoulder_link spheres (idx 0,1; r=0.105 at the robot base) are the static
# mount: they sit ~15mm from the table edge for EVERY config and would otherwise
# pin the obstacle distance, flattening the speed field at ~0.31 (0% fully-free).
# Exclude them from the SPEED-FIELD obstacle distance ONLY (they still count for
# collision rejection). Excluding them, arm clearance median ~74mm, ~64% reach
# speed=1.0. Mirrors G1's torso_ignore_sphere_indices for structural geometry.
OBSTACLE_IGNORE_SPHERE_INDICES = [0, 1]

ZONE_CUPPING = 0
ZONE_POURING = 1

PAIR_INTRA_CUPPING = 0
PAIR_INTRA_POURING = 1
PAIR_CROSS = 2


# ── Geometry helpers (reused from pkl_to_training_data_multi.py) ──────────

def load_scene_cuboids(path: str) -> list[dict]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("cuboids", [])


def cuboid_half_extents(cuboids: list[dict], device: str = "cuda:0"):
    centers, half_dims = [], []
    for c in cuboids:
        p = c["pose"]
        d = c["dims"]
        centers.append([p[0], p[1], p[2]])
        half_dims.append([d[0] / 2, d[1] / 2, d[2] / 2])
    return (
        torch.tensor(centers, dtype=torch.float32, device=device),
        torch.tensor(half_dims, dtype=torch.float32, device=device),
    )


def sphere_cuboid_distance(sphere_centers, sphere_radii, cuboid_centers, cuboid_half,
                           ignore_sphere_indices=None):
    """Min signed sphere→cuboid distance per batch element.

    `ignore_sphere_indices` (e.g. the static base/shoulder spheres) are excluded
    from the min — used for the SPEED-FIELD distance so the robot's own mount, which
    is permanently ~15mm from the table edge, does not flatten the field. These
    spheres are still used for COLLISION REJECTION (see check_collision_full).
    """
    B, S, _ = sphere_centers.shape
    K = cuboid_centers.shape[0]
    delta = sphere_centers.unsqueeze(2) - cuboid_centers.unsqueeze(0).unsqueeze(0)
    clamped = torch.clamp(torch.abs(delta) - cuboid_half.unsqueeze(0).unsqueeze(0), min=0.0)
    dist_to_surface = torch.norm(clamped, dim=-1)
    if sphere_radii.dim() == 1:
        sphere_radii = sphere_radii.unsqueeze(0).expand(B, -1)
    signed_dist = dist_to_surface - sphere_radii.unsqueeze(2)   # (B, S, K)
    if ignore_sphere_indices:
        for si in ignore_sphere_indices:
            if si < S:
                signed_dist[:, si, :] = float('inf')
    min_dist, _ = signed_dist.reshape(B, -1).min(dim=1)
    return min_dist


# ── Distance computation ─────────────────────────────────────────────────

def compute_physical_distance(robot_model, q_rad, cuboid_centers, cuboid_half,
                              torso_centers=None, torso_half=None,
                              self_coll_mask=None,
                              workspace_bounds=None,
                              device="cuda:0", batch_size=2000):
    """Min signed distance from collision spheres to all physical obstacles.

    Returns (N,) d_obs, (N,) d_self, and (N,) d_workspace.

    ``workspace_bounds`` is an optional (3, 2) XYZ box that the end effector
    must stay *inside* -- the inverse of an obstacle, which it must stay
    outside. The evaluator enforces this box at every densified waypoint but
    it is absent from the scene cuboids, so without this term the speed field
    carries no signal about it at all. Distance is signed: positive inside,
    negative once the EE leaves the box, so it composes with the obstacle and
    self-collision terms under the same min()/clip() as everything else.
    Returns +inf when no box is supplied, which is a no-op under min().
    """
    N = q_rad.shape[0]
    d_obs_list = []
    d_self_list = []
    d_ws_list = []
    ws_t = None
    if workspace_bounds is not None:
        ws_t = torch.as_tensor(np.asarray(workspace_bounds),
                               dtype=torch.float32, device=device)
    for i in tqdm(range(0, N, batch_size), desc="Computing physical distances",
                  total=(N + batch_size - 1) // batch_size):
        end = min(i + batch_size, N)
        q_batch = torch.tensor(q_rad[i:end], dtype=torch.float32, device=device)
        state = robot_model.get_state(q_batch)
        spheres = state.link_spheres_tensor
        centers = spheres[:, :, :3]
        radii = spheres[:, :, 3]

        d_min = sphere_cuboid_distance(centers, radii, cuboid_centers, cuboid_half,
                                       ignore_sphere_indices=OBSTACLE_IGNORE_SPHERE_INDICES)

        if torso_centers is not None and torso_half is not None:
            B_cur = centers.shape[0]
            delta = centers.unsqueeze(2) - torso_centers.unsqueeze(0).unsqueeze(0)
            clamped_t = torch.clamp(torch.abs(delta) - torso_half.unsqueeze(0).unsqueeze(0), min=0.0)
            dist_t = torch.norm(clamped_t, dim=-1)
            r = radii if radii.dim() == 2 else radii.unsqueeze(0).expand(B_cur, -1)
            signed_torso = dist_t - r.unsqueeze(2)
            for si in TORSO_IGNORE_SPHERE_INDICES:
                signed_torso[:, si, :] = float('inf')
            d_torso, _ = signed_torso.reshape(B_cur, -1).min(dim=1)
            d_min = torch.min(d_min, d_torso)

        d_self_batch = torch.full((end - i,), float('inf'), device=device)
        if self_coll_mask is not None:
            d_self_batch = sphere_sphere_distance(centers, radii, self_coll_mask)

        # Workspace containment on the end effector, matching the evaluator,
        # which checks ee_pose.position only rather than every sphere.
        d_ws_batch = torch.full((end - i,), float('inf'), device=device)
        if ws_t is not None:
            ee = state.ee_pose.position
            inside = torch.minimum(ee - ws_t[:, 0], ws_t[:, 1] - ee)
            d_ws_batch = inside.min(dim=-1).values

        d_obs_list.append(d_min.cpu().numpy())
        d_self_list.append(d_self_batch.cpu().numpy())
        d_ws_list.append(d_ws_batch.cpu().numpy())

    return (np.concatenate(d_obs_list), np.concatenate(d_self_list),
            np.concatenate(d_ws_list))


# ── Orientation distance functions ───────────────────────────────────────

# R58: graded orientation cost toggle. Set from --graded_orient / --tilt_grade in main().
# When _GRADED_ORIENT is True, d_orient ramps continuously from upright (tilt=0) instead
# of being flat (==margin) across the whole valid cone. This is the Phase-0 fix: it gives
# the speed field a "stay-oriented" gradient mid-transition AND widens the C-space ramp so
# it is learnable (Phase-0 measured tilt~106deg/unit-C-space, so the old 10deg tilt_soft
# ramp spanned only 0.094 normalized units — a near-vertical cliff).
_GRADED_ORIENT = False
_TILT_GRADE = 50.0   # degrees over which d_orient ramps margin -> 0 (graded mode)
# R62: signed angular distance to the constraint surface. The direct analogue of the
# obstacle term, which is a signed distance to the obstacle SURFACE. The legacy flat
# mode measures excess PAST the limit, so its whole ramp lives outside the feasible
# set and d == margin (full speed) at the limit itself; the graded mode measures
# distance from upright, putting its zero crossing at _TILT_GRADE rather than at the
# constraint. Signed mode puts d == 0 exactly AT tilt_max.
_SIGNED_ORIENT = False
_TILT_BAND = 20.0    # degrees over which d_orient spans +margin -> -margin
_TASK_AXES = load_task_axes(None)


def _task_axes(task_axes, task):
    axes = task_axes if task_axes is not None else _TASK_AXES
    return task_axis_arrays(axes, task)


def _orient_d_from_tilt(tilt, margin, tilt_max, tilt_soft):
    """Map a tilt-angle tensor (deg) to an orientation distance.

    Legacy (flat) mode: d = margin*(1 - max(tilt-tilt_max,0)/tilt_soft). Flat==margin
        for all tilt<=tilt_max (no gradient inside the cone).
    Graded mode (R58): d = margin*(1 - tilt/_TILT_GRADE). Decreases continuously from
        upright, giving a stay-oriented gradient across the whole cone and beyond, with
        a much wider (learnable) C-space ramp. Clamped to [-margin, margin].
    """
    if _SIGNED_ORIENT:
        # d = margin at tilt_max - band (comfortably inside), 0 AT tilt_max,
        # -margin at tilt_max + band. Gradient is non-zero on BOTH sides, so the
        # speed field carries a stay-oriented signal inside the feasible cone.
        d = margin * ((tilt_max - tilt) / _TILT_BAND)
        return d.clamp(min=-margin, max=margin)
    if _GRADED_ORIENT:
        d = margin * (1.0 - tilt / _TILT_GRADE)
        return d.clamp(min=-margin, max=margin)
    excess = (tilt - tilt_max).clamp(min=0.0)
    d = margin * (1.0 - excess / tilt_soft)
    return d.clamp(min=-margin)


def _cupping_orient_distance_batch(q_batch, robot_model, margin, tilt_max, tilt_soft,
                                   task_axes=None):
    state = robot_model.get_state(q_batch)
    quat = state.ee_pose.quaternion
    local_axis, world_axis = _task_axes(task_axes, "cup")
    tilt = tilt_deg_from_quat(quat, local_axis, world_axis)
    return _orient_d_from_tilt(tilt, margin, tilt_max, tilt_soft)


def _pouring_orient_distance_batch(q_batch, robot_model, margin, tilt_max, tilt_soft,
                                   rzy_safe=0.15, rzy_soft=0.30, task_axes=None):
    state = robot_model.get_state(q_batch)
    quat = state.ee_pose.quaternion
    local_axis, world_axis = _task_axes(task_axes, "pour")
    x_tilt = tilt_deg_from_quat(quat, local_axis, world_axis)
    return _orient_d_from_tilt(x_tilt, margin, tilt_max, tilt_soft)


def _morph_orient_distance_batch(q_batch, robot_model, margin, tilt_max, tilt_soft,
                                 y_divide, overlap, mode,
                                 zone_mode="y", corner_overlap_deg=45.0,
                                 task_axes=None):
    """R60: morphed orientation distance. Computes BOTH cup and pour tilt, blends them,
    grades the blended tilt into a distance via the same _orient_d_from_tilt grading
    (so the unified 20->30 band still applies).

    zone_mode='y' (default): 1D blend along ee_y across [y_divide+-overlap].
    zone_mode='corner_table': 2D angular blend through the L-shape inner corner
    (cup wing -X -> pour wing -Y), with corner_overlap_deg the angular OR-band
    half-width around the inner diagonal. The wings keep pure single-task targets;
    only the corner morphs -- the L-shape analogue of the y-divide overlap region."""
    from morph_orient import morph_tilt_deg, morph_tilt_deg_corner
    state = robot_model.get_state(q_batch)
    quat = state.ee_pose.quaternion
    ee_pos = state.ee_pose.position
    ee_x = ee_pos[:, 0]
    ee_y = ee_pos[:, 1]
    cup_local, cup_world = _task_axes(task_axes, "cup")
    pour_local, pour_world = _task_axes(task_axes, "pour")
    cup_tilt = tilt_deg_from_quat(quat, cup_local, cup_world)
    pour_tilt = tilt_deg_from_quat(quat, pour_local, pour_world)
    if zone_mode == "corner_table":
        mt = morph_tilt_deg_corner(cup_tilt, pour_tilt, ee_x, ee_y, mode,
                                   overlap_deg=corner_overlap_deg)
    else:
        mt = morph_tilt_deg(cup_tilt, pour_tilt, ee_y, y_divide, overlap, mode)
    return _orient_d_from_tilt(mt, margin, tilt_max, tilt_soft)


def _zone_tilt_rad(q_batch, robot_model, zone_mode="y", y_divide=0.0,
                   corner_zones=None, task_axes=None):
    """Zone-appropriate wrist tilt (deg) for a batch."""
    st = robot_model.get_state(q_batch)
    quat = st.ee_pose.quaternion
    ee_pos = st.ee_pose.position
    eey = ee_pos[:, 1]
    cup_local, cup_world = _task_axes(task_axes, "cup")
    pour_local, pour_world = _task_axes(task_axes, "pour")
    cup = tilt_deg_from_quat(quat, cup_local, cup_world)
    pour = tilt_deg_from_quat(quat, pour_local, pour_world)
    if zone_mode == "corner_table" and corner_zones is not None:
        from models.metric_arm.zone_router import corner_masks_torch
        ee_xy = ee_pos[:, :2]
        cup_mask, pour_mask, overlap_mask = corner_masks_torch(
            ee_xy, corner_zones, assign_outside=True)
        zone_tilt = torch.where(cup_mask, cup, pour)
        # OR policy in overlap: either constraint can carry the transition.
        zone_tilt = torch.where(overlap_mask, torch.minimum(cup, pour), zone_tilt)
        return zone_tilt
    return torch.where(eey >= y_divide, cup, pour)


def compute_tilt_gradient(robot_model, q_rad, device="cuda:0", eps=2e-4, batch_size=2048,
                          zone_mode="y", y_divide=0.0, corner_zones=None,
                          task_axes=None):
    """R58U: clean per-config DESCENT direction for zone tilt: normalized -grad_q tilt_FK.

    This is the gold orientation-descent target the learned Phi head's -gradient should match.
    Computed offline via FK finite-diff (finer eps than the min-blended `normal`, and NO
    min-switch with d_phys), so it points reliably down-tilt. FK is used only here (offline);
    inference never calls FK. Returns (N, dim) unit vectors in RAD joint space.
    """
    N, DOF = q_rad.shape
    out = np.zeros((N, DOF), dtype=np.float32)
    for i in tqdm(range(0, N, batch_size), desc="Tilt gradient",
                  total=(N + batch_size - 1) // batch_size):
        end = min(i + batch_size, N)
        q = torch.tensor(q_rad[i:end], dtype=torch.float32, device=device)
        g = torch.zeros(end - i, DOF, device=device)
        for j in range(DOF):
            qp = q.clone(); qp[:, j] += eps
            qm = q.clone(); qm[:, j] -= eps
            g[:, j] = (
                _zone_tilt_rad(qp, robot_model, zone_mode, y_divide, corner_zones, task_axes)
                - _zone_tilt_rad(qm, robot_model, zone_mode, y_divide, corner_zones, task_axes)
            ) / (2 * eps)
        desc = -g  # descent = lower tilt
        desc = desc / desc.norm(dim=1, keepdim=True).clamp(min=1e-8)
        out[i:end] = desc.cpu().numpy()
    return out


def compute_orient_distance(robot_model, q_rad, task, margin, tilt_max, tilt_soft,
                            device="cuda:0", batch_size=4096,
                            morph=False, morph_mode="linear",
                            y_divide=0.0, overlap=0.15,
                            zone_mode="y", corner_overlap_deg=45.0,
                            task_axes=None, **kwargs):
    N = q_rad.shape[0]
    d_orient = np.empty(N, dtype=np.float32)
    for i in tqdm(range(0, N, batch_size),
                  desc=f"Orient dist ({task}{'+morph' if morph else ''})",
                  total=(N + batch_size - 1) // batch_size):
        end = min(i + batch_size, N)
        q_batch = torch.tensor(q_rad[i:end], dtype=torch.float32, device=device)
        if morph:
            d_batch = _morph_orient_distance_batch(
                q_batch, robot_model, margin, tilt_max, tilt_soft,
                y_divide, overlap, morph_mode,
                zone_mode=zone_mode, corner_overlap_deg=corner_overlap_deg,
                task_axes=task_axes)
        elif task == "cupping":
            d_batch = _cupping_orient_distance_batch(
                q_batch, robot_model, margin, tilt_max, tilt_soft,
                task_axes=task_axes)
        else:
            d_batch = _pouring_orient_distance_batch(
                q_batch, robot_model, margin, tilt_max, tilt_soft,
                rzy_safe=kwargs.get("rzy_safe", 0.15),
                rzy_soft=kwargs.get("rzy_soft", 0.30),
                task_axes=task_axes,
            )
        d_orient[i:end] = d_batch.cpu().numpy()
    return d_orient


# ── Speed computation ────────────────────────────────────────────────────

def compute_speed(d_phys, d_orient, alpha, margin, offset, blend_mode="min"):
    """Compute speed from physical and orientation distances.

    blend_mode='min':    d = min(d_phys, d_orient)
    blend_mode='alpha':  d = alpha*d_orient + (1-alpha)*d_phys
    blend_mode='product': d = d_phys * d_orient / margin
    blend_mode='harmonic': d = 2*d_phys*d_orient / (d_phys+d_orient)
    blend_mode='quadrature': combine "excess danger beyond the margin
        boundary" (1/d - 1/margin, zero once a term is at or beyond margin)
        in quadrature, then map back to a distance: d = 1 / (1/margin +
        sqrt(excess_phys^2 + excess_orient^2)). Properly normalized so that
        d_combined == margin (full speed) when EITHER term alone is at or
        beyond margin -- unlike a naive 1/d^2 sum, which never reaches
        margin even when both terms are exactly at the safe boundary. When
        only one term is marginal (the other >= margin) this reduces to
        min(d_phys, d_orient) exactly. When both are simultaneously
        marginal, it degrades faster than min() (compounding risk) but
        gentler than product's degradation in that regime.
    """
    if blend_mode == "min":
        d_combined = np.minimum(d_phys, d_orient)
    elif blend_mode == "product":
        d_combined = d_phys * d_orient / np.maximum(margin, 1e-8)
    elif blend_mode == "harmonic":
        denom = np.maximum(d_phys + d_orient, 1e-8)
        d_combined = 2.0 * d_phys * d_orient / denom
    elif blend_mode == "quadrature":
        dp = np.maximum(d_phys, 1e-6)
        do = np.maximum(d_orient, 1e-6)
        excess_p = np.maximum(1.0 / dp - 1.0 / margin, 0.0)
        excess_o = np.maximum(1.0 / do - 1.0 / margin, 0.0)
        excess = np.sqrt(excess_p**2 + excess_o**2)
        d_combined = 1.0 / (1.0 / margin + excess)
        # A true constraint violation (d_orient < 0, e.g. past the tilt
        # limit) must still floor to offset like the other blend modes do.
        d_combined = np.where((d_phys < 0) | (d_orient < 0), offset, d_combined)
    else:  # alpha
        d_combined = alpha * d_orient + (1.0 - alpha) * d_phys
    return np.clip(d_combined, offset, margin) / margin


# ── Gradient computation ─────────────────────────────────────────────────

def _torch_blend(d_phys, d_orient, blend_mode, alpha, margin):
    """Torch version of distance blending for gradient computation."""
    if blend_mode == "min":
        return torch.min(d_phys, d_orient)
    elif blend_mode == "product":
        return d_phys * d_orient / max(margin, 1e-8)
    elif blend_mode == "harmonic":
        return 2.0 * d_phys * d_orient / (d_phys + d_orient).clamp(min=1e-8)
    elif blend_mode == "quadrature":
        dp = d_phys.clamp(min=1e-6)
        do = d_orient.clamp(min=1e-6)
        excess_p = (1.0 / dp - 1.0 / margin).clamp(min=0.0)
        excess_o = (1.0 / do - 1.0 / margin).clamp(min=0.0)
        excess = torch.sqrt(excess_p**2 + excess_o**2)
        d_combined = 1.0 / (1.0 / margin + excess)
        violated = (d_phys < 0) | (d_orient < 0)
        return torch.where(violated, torch.zeros_like(d_combined), d_combined)
    else:
        return alpha * d_orient + (1.0 - alpha) * d_phys


def compute_gradient_task(robot_model, q_rad, cuboid_centers, cuboid_half,
                          torso_centers, torso_half, self_coll_mask,
                          task, alpha, margin, tilt_max, tilt_soft,
                          device="cuda:0", eps=1e-3, batch_size=500,
                          blend_mode="min", self_speed_scale=1.0,
                          morph=False, morph_mode="linear",
                          y_divide=0.0, overlap=0.15,
                          zone_mode="y", corner_overlap_deg=45.0,
                          task_axes=None,
                          **orient_kwargs):
    N = q_rad.shape[0]
    grad_all = []

    def _orient_dist(q_b, task):
        if morph:
            return _morph_orient_distance_batch(
                q_b, robot_model, margin, tilt_max, tilt_soft,
                y_divide, overlap, morph_mode,
                zone_mode=zone_mode, corner_overlap_deg=corner_overlap_deg,
                task_axes=task_axes)
        if task == "cupping":
            return _cupping_orient_distance_batch(
                q_b, robot_model, margin, tilt_max, tilt_soft,
                task_axes=task_axes)
        else:
            return _pouring_orient_distance_batch(
                q_b, robot_model, margin, tilt_max, tilt_soft,
                rzy_safe=orient_kwargs.get("rzy_safe", 0.15),
                rzy_soft=orient_kwargs.get("rzy_soft", 0.30),
                task_axes=task_axes,
            )

    def _phys_dist(centers, radii):
        d = sphere_cuboid_distance(centers, radii, cuboid_centers, cuboid_half,
                                   ignore_sphere_indices=OBSTACLE_IGNORE_SPHERE_INDICES)
        if torso_centers is not None:
            B_cur = centers.shape[0]
            delta = centers.unsqueeze(2) - torso_centers.unsqueeze(0).unsqueeze(0)
            clamped_t = torch.clamp(torch.abs(delta) - torso_half.unsqueeze(0).unsqueeze(0), min=0.0)
            dist_t = torch.norm(clamped_t, dim=-1)
            r = radii if radii.dim() == 2 else radii.unsqueeze(0).expand(B_cur, -1)
            signed_torso = dist_t - r.unsqueeze(2)
            for si in TORSO_IGNORE_SPHERE_INDICES:
                signed_torso[:, si, :] = float('inf')
            d_torso, _ = signed_torso.reshape(B_cur, -1).min(dim=1)
            d = torch.min(d, d_torso)
        if self_coll_mask is not None:
            d_self = sphere_sphere_distance(centers, radii, self_coll_mask)
            if self_speed_scale != 1.0:
                d_self = d_self * self_speed_scale
            d = torch.min(d, d_self)
        return d

    for i in tqdm(range(0, N, batch_size),
                  desc=f"Gradients ({task})",
                  total=(N + batch_size - 1) // batch_size):
        end = min(i + batch_size, N)
        q_batch = torch.tensor(q_rad[i:end], dtype=torch.float32, device=device)
        B = q_batch.shape[0]
        grad = torch.zeros(B, DOF, device=device)

        for j in range(DOF):
            q_plus = q_batch.clone()
            q_minus = q_batch.clone()
            q_plus[:, j] += eps
            q_minus[:, j] -= eps

            sp = robot_model.get_state(q_plus).link_spheres_tensor
            d_p_phys = _phys_dist(sp[:, :, :3], sp[:, :, 3])
            d_p_orient = _orient_dist(q_plus, task)
            d_p = _torch_blend(d_p_phys, d_p_orient, blend_mode, alpha, margin)

            sm = robot_model.get_state(q_minus).link_spheres_tensor
            d_m_phys = _phys_dist(sm[:, :, :3], sm[:, :, 3])
            d_m_orient = _orient_dist(q_minus, task)
            d_m = _torch_blend(d_m_phys, d_m_orient, blend_mode, alpha, margin)

            grad[:, j] = (d_p - d_m) / (2 * eps)

        grad_all.append(grad.cpu().numpy())

    grad_all = np.concatenate(grad_all)
    norms = np.linalg.norm(grad_all, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return grad_all / norms


# ── EE position computation ─────────────────────────────────────────────

def compute_ee_positions(robot_model, q_rad, device="cuda:0", batch_size=4096):
    N = q_rad.shape[0]
    positions = []
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        q_batch = torch.tensor(q_rad[i:end], dtype=torch.float32, device=device)
        pos = robot_model.get_state(q_batch).ee_pose.position.cpu().numpy()
        positions.append(pos)
    return np.concatenate(positions)


# ── Zone assignment ──────────────────────────────────────────────────────

def assign_zones(ee_y, y_divide, overlap, ee_x=None,
                 zone_mode="y", corner_zones=None, include_overlap=True):
    if zone_mode == "corner_table":
        if ee_x is None or corner_zones is None:
            raise ValueError("corner_table zone assignment requires ee_x and corner_zones")
        ee_xy = np.stack([ee_x, ee_y], axis=1)
        return assign_corner_masks_np(
            ee_xy, corner_zones, include_overlap=include_overlap)

    # UR5e: scene_ur5e.yml has cup_side=+1 → CUPPING on the HIGHER-y (+y/LEFT) side,
    # POURING on the LOWER-y (-y/RIGHT) side. This is FLIPPED vs the G1 source
    # (which had cupping on lower y). Keep this consistent with the scene's cup_side.
    cupping_mask = ee_y >= (y_divide - overlap)
    pouring_mask = ee_y <= (y_divide + overlap)
    overlap_mask = cupping_mask & pouring_mask
    return cupping_mask, pouring_mask, overlap_mask


# ── Pair sampling ────────────────────────────────────────────────────────

def sample_pairs(cupping_indices, pouring_indices, n_pairs, intra_ratio, rng):
    """Legacy zone-based sampling (not source-aware). Kept for backward compatibility."""
    n_intra = int(n_pairs * intra_ratio)
    n_intra_cup = n_intra // 2
    n_intra_pour = n_intra - n_intra_cup
    n_cross = n_pairs - n_intra

    start_all = []
    goal_all = []
    ptype_all = []
    szone_all = []
    gzone_all = []

    def _sample_intra(pool, n, pair_type_val, zone_val):
        if len(pool) < 2:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64), \
                   np.array([], dtype=np.int64), np.array([], dtype=np.int64), \
                   np.array([], dtype=np.int64)
        idx = rng.choice(pool, size=2 * n, replace=True)
        s = idx[:n]
        g = idx[n:]
        dup = s == g
        while dup.any():
            g[dup] = rng.choice(pool, size=dup.sum(), replace=True)
            dup = s == g
        return (s, g,
                np.full(n, pair_type_val, dtype=np.int64),
                np.full(n, zone_val, dtype=np.int64),
                np.full(n, zone_val, dtype=np.int64))

    s, g, pt, sz, gz = _sample_intra(cupping_indices, n_intra_cup, PAIR_INTRA_CUPPING, ZONE_CUPPING)
    start_all.append(s); goal_all.append(g)
    ptype_all.append(pt); szone_all.append(sz); gzone_all.append(gz)

    s, g, pt, sz, gz = _sample_intra(pouring_indices, n_intra_pour, PAIR_INTRA_POURING, ZONE_POURING)
    start_all.append(s); goal_all.append(g)
    ptype_all.append(pt); szone_all.append(sz); gzone_all.append(gz)

    n_c2p = n_cross // 2
    n_p2c = n_cross - n_c2p

    if len(cupping_indices) > 0 and len(pouring_indices) > 0:
        s_c2p = rng.choice(cupping_indices, size=n_c2p, replace=True)
        g_c2p = rng.choice(pouring_indices, size=n_c2p, replace=True)
        start_all.append(s_c2p); goal_all.append(g_c2p)
        ptype_all.append(np.full(n_c2p, PAIR_CROSS, dtype=np.int64))
        szone_all.append(np.full(n_c2p, ZONE_CUPPING, dtype=np.int64))
        gzone_all.append(np.full(n_c2p, ZONE_POURING, dtype=np.int64))

        s_p2c = rng.choice(pouring_indices, size=n_p2c, replace=True)
        g_p2c = rng.choice(cupping_indices, size=n_p2c, replace=True)
        start_all.append(s_p2c); goal_all.append(g_p2c)
        ptype_all.append(np.full(n_p2c, PAIR_CROSS, dtype=np.int64))
        szone_all.append(np.full(n_p2c, ZONE_POURING, dtype=np.int64))
        gzone_all.append(np.full(n_p2c, ZONE_CUPPING, dtype=np.int64))

    start_idx = np.concatenate(start_all)
    goal_idx = np.concatenate(goal_all)
    pair_type = np.concatenate(ptype_all)
    start_zone = np.concatenate(szone_all)
    goal_zone = np.concatenate(gzone_all)

    perm = rng.permutation(len(start_idx))
    return start_idx[perm], goal_idx[perm], pair_type[perm], start_zone[perm], goal_zone[perm]


def sample_pairs_source_aware(cupping_mask, pouring_mask, source_id, n_pairs, intra_ratio, rng,
                               source_mix=0.0, ee_y=None, y_divide=0.0,
                               ee_pos=None, zone_mode="y", corner_zones=None,
                               q_norm=None, boundary_cross_frac=0.0,
                               boundary_cross_band=0.10, local_cross_k=8,
                               min_cross_dist=0.0, max_cross_dist=0.0,
                               cross_dist_oversample=6,
                               corner_cross_frac=0.0,
                               corner_box=(-0.35, 0.15, -0.35, 0.15)):
    """Source-aware pair sampling with optional mixed orientation-negative examples.

    Main pools:
      Cupping intra-zone: both endpoints from cupping-pkl configs in cupping zone.
      Pouring intra-zone: both endpoints from pouring-pkl configs in pouring zone.
      Cross-zone: cupping-pkl <-> pouring-pkl.

    source_mix (0.0-1.0): fraction of intra-zone pairs where ONE endpoint comes from
    the other source pkl. These provide orientation-negative examples:
    - In cupping pairs, a pouring-pkl endpoint has bad cupping orientation -> low cupping speed
    - In pouring pairs, a cupping-pkl endpoint has bad pouring orientation -> low pouring speed
    The pair_type remains the same (intra-cupping/pouring), so the training loss treats
    them as same-zone pairs with appropriate speed labels.

    boundary_cross_frac: fraction of cross pairs sampled as boundary-local nearest
    neighbors in normalized joint space. These pairs supervise the transition manifold
    used by local MPPI steps, without changing the planner objective.
    """
    n_intra = int(n_pairs * intra_ratio)
    n_intra_cup = n_intra // 2
    n_intra_pour = n_intra - n_intra_cup
    n_cross = n_pairs - n_intra

    # Source-filtered pools:
    cup_pool = np.where(cupping_mask & (source_id == 0))[0]      # cup-pkl in cup zone
    pour_pool = np.where(pouring_mask & (source_id == 1))[0]      # pour-pkl in pour zone
    # "Negative" pools: opposite source in each zone
    cup_neg_pool = np.where(cupping_mask & (source_id == 1))[0]   # pour-pkl in cup zone
    pour_neg_pool = np.where(pouring_mask & (source_id == 0))[0]  # cup-pkl in pour zone

    # Exclusive (non-overlap) pools for cross pairs: genuine zone-crossing, matching eval.
    cup_excl_mask  = cupping_mask & ~pouring_mask   # ee_y > y_divide + overlap (true cup side)
    pour_excl_mask = pouring_mask & ~cupping_mask   # ee_y < y_divide - overlap (true pour side)
    cup_cross_pool  = np.where(cup_excl_mask  & (source_id == 0))[0]
    pour_cross_pool = np.where(pour_excl_mask & (source_id == 1))[0]
    print(f"  cross pools (exclusive zones): cup_cross={len(cup_cross_pool)} pour_cross={len(pour_cross_pool)}")

    # Corner-endpoint cross pool (R5 code fix): configs whose EE lands in the inner
    # corner box, drawn from the FULL pool (both sources). boundary_cross_frac
    # saturates corner coverage ~5%; this forces cross GOALS into the corner so the
    # field learns a proper cost-to-go / clearance for the transition region it must
    # traverse. Legitimate coverage (from FK pools), not eval curation.
    corner_goal_pool = np.empty(0, dtype=np.int64)
    # R67: the default corner_box (-0.35,0.15,-0.35,0.15) is left over from the old
    # corner-table geometry and does not intersect this workspace (x in [0.29,0.85]),
    # so it silently yields an EMPTY pool.  Derive the box from the zone config's
    # overlap_region instead -- that IS the transition band, and it is exactly the
    # region cross pairs traverse but never terminate in.
    #
    # Measured 2026-09-08: of 200,000 cross pairs, ZERO have an endpoint inside the
    # band, while intra pairs contribute 520k band endpoints.  So the corner is well
    # covered by single-task poses but no cross query is ever taught what to do there.
    if corner_zones and corner_zones.get('overlap_region') is not None:
        _ov = corner_zones['overlap_region']
        corner_box = (float(_ov[0]), float(_ov[1]), float(_ov[2]), float(_ov[3]))
        print(f"  [R67] corner_box derived from overlap_region: {corner_box}")
    if ee_pos is not None:
        cxlo, cxhi, cylo, cyhi = corner_box
        _cx, _cy = ee_pos[:, 0], ee_pos[:, 1]
        corner_mask = (_cx > cxlo) & (_cx < cxhi) & (_cy > cylo) & (_cy < cyhi)
        corner_goal_pool = np.where(corner_mask)[0]
        print(f"  corner-endpoint pool: {len(corner_goal_pool)} configs in {corner_box}")
        if corner_cross_frac > 0 and len(corner_goal_pool) == 0:
            raise ValueError(
                "--corner_cross_frac was requested but the corner pool is EMPTY; "
                f"corner_box={corner_box} does not intersect the config EE range. "
                "Refusing to train on a silently inert setting.")

    print(f"  Source-aware pools (mix={source_mix:.0%}):")
    print(f"    Cupping pool (cup_pkl in cup_zone): {len(cup_pool)}")
    print(f"    Cupping neg  (pour_pkl in cup_zone): {len(cup_neg_pool)}")
    print(f"    Pouring pool (pour_pkl in pour_zone): {len(pour_pool)}")
    print(f"    Pouring neg  (cup_pkl in pour_zone): {len(pour_neg_pool)}")

    start_all = []
    goal_all = []
    ptype_all = []
    szone_all = []
    gzone_all = []

    def _sample_intra(pool, n, pair_type_val, zone_val):
        if len(pool) < 2:
            print(f"    WARNING: pool too small ({len(pool)}) for {n} pairs")
            return (np.array([], dtype=np.int64),) * 5
        idx = rng.choice(pool, size=2 * n, replace=True)
        s = idx[:n]
        g = idx[n:]
        dup = s == g
        while dup.any():
            g[dup] = rng.choice(pool, size=dup.sum(), replace=True)
            dup = s == g
        return (s, g,
                np.full(n, pair_type_val, dtype=np.int64),
                np.full(n, zone_val, dtype=np.int64),
                np.full(n, zone_val, dtype=np.int64))

    # Split intra pairs into pure + mixed
    n_cup_mixed = int(n_intra_cup * source_mix) if len(cup_neg_pool) > 0 else 0
    n_cup_pure = n_intra_cup - n_cup_mixed
    n_pour_mixed = int(n_intra_pour * source_mix) if len(pour_neg_pool) > 0 else 0
    n_pour_pure = n_intra_pour - n_pour_mixed

    # Pure intra-cupping: both from cup_pool
    s, g, pt, sz, gz = _sample_intra(cup_pool, n_cup_pure, PAIR_INTRA_CUPPING, ZONE_CUPPING)
    start_all.append(s); goal_all.append(g)
    ptype_all.append(pt); szone_all.append(sz); gzone_all.append(gz)

    # Mixed intra-cupping: start from cup_pool, goal from cup_neg_pool (or vice versa randomly)
    if n_cup_mixed > 0:
        s_mix = rng.choice(cup_pool, size=n_cup_mixed, replace=True)
        g_mix = rng.choice(cup_neg_pool, size=n_cup_mixed, replace=True)
        # Randomly swap start/goal so orientation-negative can be at either endpoint
        swap = rng.random(n_cup_mixed) < 0.5
        s_final = np.where(swap, g_mix, s_mix)
        g_final = np.where(swap, s_mix, g_mix)
        start_all.append(s_final); goal_all.append(g_final)
        ptype_all.append(np.full(n_cup_mixed, PAIR_INTRA_CUPPING, dtype=np.int64))
        szone_all.append(np.full(n_cup_mixed, ZONE_CUPPING, dtype=np.int64))
        gzone_all.append(np.full(n_cup_mixed, ZONE_CUPPING, dtype=np.int64))

    # Pure intra-pouring: both from pour_pool
    s, g, pt, sz, gz = _sample_intra(pour_pool, n_pour_pure, PAIR_INTRA_POURING, ZONE_POURING)
    start_all.append(s); goal_all.append(g)
    ptype_all.append(pt); szone_all.append(sz); gzone_all.append(gz)

    # Mixed intra-pouring: start from pour_pool, goal from pour_neg_pool
    if n_pour_mixed > 0:
        s_mix = rng.choice(pour_pool, size=n_pour_mixed, replace=True)
        g_mix = rng.choice(pour_neg_pool, size=n_pour_mixed, replace=True)
        swap = rng.random(n_pour_mixed) < 0.5
        s_final = np.where(swap, g_mix, s_mix)
        g_final = np.where(swap, s_mix, g_mix)
        start_all.append(s_final); goal_all.append(g_final)
        ptype_all.append(np.full(n_pour_mixed, PAIR_INTRA_POURING, dtype=np.int64))
        szone_all.append(np.full(n_pour_mixed, ZONE_POURING, dtype=np.int64))
        gzone_all.append(np.full(n_pour_mixed, ZONE_POURING, dtype=np.int64))

    print(f"    Cupping pairs: {n_cup_pure} pure + {n_cup_mixed} mixed = {n_intra_cup}")
    print(f"    Pouring pairs: {n_pour_pure} pure + {n_pour_mixed} mixed = {n_intra_pour}")

    # Cross-zone: cupping-pkl <-> pouring-pkl.
    n_boundary_cross = int(n_cross * boundary_cross_frac)
    n_corner_cross = int(n_cross * corner_cross_frac) if len(corner_goal_pool) > 0 else 0
    n_random_cross = n_cross - n_boundary_cross - n_corner_cross
    n_random_c2p = n_random_cross // 2
    n_random_p2c = n_random_cross - n_random_c2p

    def _dist_gated_pairs(src_pool, dst_pool, n, lo, hi, oversample, rng, q_norm):
        """Random src->dst pairs with normalized cspace dist in [lo, hi]. Rejection sample."""
        if n <= 0:
            return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
        acc_s, acc_g = [], []
        need = n
        tries = 0
        while need > 0 and tries < 50:
            m = max(need * oversample, 1024)
            s = rng.choice(src_pool, size=m, replace=True)
            g = rng.choice(dst_pool, size=m, replace=True)
            d = np.linalg.norm(q_norm[s] - q_norm[g], axis=1)
            ok = np.ones(m, dtype=bool)
            if lo > 0: ok &= (d >= lo)
            if hi > 0: ok &= (d <= hi)
            s, g = s[ok], g[ok]
            take = min(need, len(s))
            acc_s.append(s[:take]); acc_g.append(g[:take])
            need -= take
            tries += 1
        s_out = np.concatenate(acc_s) if acc_s else np.empty(0, dtype=np.int64)
        g_out = np.concatenate(acc_g) if acc_g else np.empty(0, dtype=np.int64)
        return s_out, g_out

    if len(cup_cross_pool) > 0 and len(pour_cross_pool) > 0 and n_random_cross > 0:
        _use_gate = (min_cross_dist > 0 or max_cross_dist > 0)
        if _use_gate and q_norm is None:
            print("    WARNING: distance-gated cross requested but q_norm unavailable; "
                  "falling back to plain random cross")
            _use_gate = False
        if _use_gate:
            s_c2p, g_c2p = _dist_gated_pairs(cup_cross_pool, pour_cross_pool, n_random_c2p,
                                             min_cross_dist, max_cross_dist,
                                             cross_dist_oversample, rng, q_norm)
        else:
            s_c2p = rng.choice(cup_cross_pool, size=n_random_c2p, replace=True)
            g_c2p = rng.choice(pour_cross_pool, size=n_random_c2p, replace=True)
        n_random_c2p = len(s_c2p)
        start_all.append(s_c2p); goal_all.append(g_c2p)
        ptype_all.append(np.full(n_random_c2p, PAIR_CROSS, dtype=np.int64))
        szone_all.append(np.full(n_random_c2p, ZONE_CUPPING, dtype=np.int64))
        gzone_all.append(np.full(n_random_c2p, ZONE_POURING, dtype=np.int64))

        if _use_gate:
            s_p2c, g_p2c = _dist_gated_pairs(pour_cross_pool, cup_cross_pool, n_random_p2c,
                                             min_cross_dist, max_cross_dist,
                                             cross_dist_oversample, rng, q_norm)
        else:
            s_p2c = rng.choice(pour_cross_pool, size=n_random_p2c, replace=True)
            g_p2c = rng.choice(cup_cross_pool, size=n_random_p2c, replace=True)
        n_random_p2c = len(s_p2c)
        start_all.append(s_p2c); goal_all.append(g_p2c)
        ptype_all.append(np.full(n_random_p2c, PAIR_CROSS, dtype=np.int64))
        szone_all.append(np.full(n_random_p2c, ZONE_POURING, dtype=np.int64))
        gzone_all.append(np.full(n_random_p2c, ZONE_CUPPING, dtype=np.int64))

    # Corner-endpoint cross pairs: wing start -> corner goal (and corner -> wing).
    # Forces the field to learn cost-to-go INTO the corner transition region.
    if n_corner_cross > 0:
        half = n_corner_cross // 2
        # cup-wing start -> corner goal
        s_cc1 = rng.choice(cup_cross_pool, size=half, replace=True)
        g_cc1 = rng.choice(corner_goal_pool, size=half, replace=True)
        start_all.append(s_cc1); goal_all.append(g_cc1)
        ptype_all.append(np.full(half, PAIR_CROSS, dtype=np.int64))
        szone_all.append(np.full(half, ZONE_CUPPING, dtype=np.int64))
        gzone_all.append(np.full(half, ZONE_POURING, dtype=np.int64))
        # corner start -> pour-wing goal
        rest = n_corner_cross - half
        s_cc2 = rng.choice(corner_goal_pool, size=rest, replace=True)
        g_cc2 = rng.choice(pour_cross_pool, size=rest, replace=True)
        start_all.append(s_cc2); goal_all.append(g_cc2)
        ptype_all.append(np.full(rest, PAIR_CROSS, dtype=np.int64))
        szone_all.append(np.full(rest, ZONE_CUPPING, dtype=np.int64))
        gzone_all.append(np.full(rest, ZONE_POURING, dtype=np.int64))
        print(f"    Corner-endpoint cross pairs: {n_corner_cross} "
              f"({half} wing->corner + {rest} corner->wing)")

    if n_boundary_cross > 0:
        if q_norm is None or cKDTree is None:
            print("    WARNING: boundary-local cross requested but q_norm/scipy is unavailable; skipping")
        else:
            if zone_mode == "corner_table":
                if ee_pos is None or corner_zones is None:
                    print("    WARNING: corner boundary-local cross requested but ee_pos/corner_zones unavailable; skipping")
                    cup_boundary = np.array([], dtype=np.int64)
                    pour_boundary = np.array([], dtype=np.int64)
                else:
                    transition = corner_transition_mask_np(
                        ee_pos[:, :2], corner_zones, boundary_cross_band)
                    cup_boundary = cup_pool[transition[cup_pool]]
                    pour_boundary = pour_pool[transition[pour_pool]]
            else:
                if ee_y is None:
                    print("    WARNING: boundary-local cross requested but ee_y unavailable; skipping")
                    cup_boundary = np.array([], dtype=np.int64)
                    pour_boundary = np.array([], dtype=np.int64)
                else:
                    cup_boundary = cup_pool[np.abs(ee_y[cup_pool] - y_divide) <= boundary_cross_band]
                    pour_boundary = pour_pool[np.abs(ee_y[pour_pool] - y_divide) <= boundary_cross_band]
            print(f"    Boundary-local cross pools (band={boundary_cross_band:.3f}): "
                  f"cup={len(cup_boundary)}, pour={len(pour_boundary)}")
            if len(cup_boundary) > 0 and len(pour_boundary) > 0:
                def _nearest_pairs(src_pool, dst_pool, n):
                    src = rng.choice(src_pool, size=n, replace=True)
                    k = max(1, min(int(local_cross_k), len(dst_pool)))
                    tree = cKDTree(q_norm[dst_pool])
                    _, nn = tree.query(q_norm[src], k=k, workers=-1)
                    if k == 1:
                        dst = dst_pool[nn]
                    else:
                        pick = rng.integers(0, k, size=n)
                        dst = dst_pool[nn[np.arange(n), pick]]
                    return src, dst

                n_local_c2p = n_boundary_cross // 2
                n_local_p2c = n_boundary_cross - n_local_c2p
                s_c2p, g_c2p = _nearest_pairs(cup_boundary, pour_boundary, n_local_c2p)
                start_all.append(s_c2p); goal_all.append(g_c2p)
                ptype_all.append(np.full(n_local_c2p, PAIR_CROSS, dtype=np.int64))
                szone_all.append(np.full(n_local_c2p, ZONE_CUPPING, dtype=np.int64))
                gzone_all.append(np.full(n_local_c2p, ZONE_POURING, dtype=np.int64))

                s_p2c, g_p2c = _nearest_pairs(pour_boundary, cup_boundary, n_local_p2c)
                start_all.append(s_p2c); goal_all.append(g_p2c)
                ptype_all.append(np.full(n_local_p2c, PAIR_CROSS, dtype=np.int64))
                szone_all.append(np.full(n_local_p2c, ZONE_POURING, dtype=np.int64))
                gzone_all.append(np.full(n_local_p2c, ZONE_CUPPING, dtype=np.int64))
                print(f"    Cross pairs: {n_random_cross} random + {n_boundary_cross} boundary-local")

    start_idx = np.concatenate(start_all)
    goal_idx = np.concatenate(goal_all)
    pair_type = np.concatenate(ptype_all)
    start_zone = np.concatenate(szone_all)
    goal_zone = np.concatenate(gzone_all)

    # diagnostic: realized cross-pair distance
    if q_norm is not None:
        _cross = (pair_type == PAIR_CROSS)
        if _cross.any():
            _d = np.linalg.norm(q_norm[start_idx[_cross]] - q_norm[goal_idx[_cross]], axis=1)
            print(f"  [cross-dist] n={_cross.sum()} normalized: "
                  f"p10={np.percentile(_d,10):.3f} p50={np.percentile(_d,50):.3f} "
                  f"p90={np.percentile(_d,90):.3f} max={_d.max():.3f}  "
                  f"(v5-hard cross band 0.167-0.332)")

    perm = rng.permutation(len(start_idx))
    return start_idx[perm], goal_idx[perm], pair_type[perm], start_zone[perm], goal_zone[perm]


def _pair_counts(pair_type):
    return (
        int((pair_type == PAIR_INTRA_CUPPING).sum()),
        int((pair_type == PAIR_INTRA_POURING).sum()),
        int((pair_type == PAIR_CROSS).sum()),
    )


def _zone_tilt_deg(q_rad_t, robot_model, ee_y, task_axes=None):
    """Zone-appropriate wrist tilt (deg) for a batch of rad configs.

    The concrete local/world axes come from scene task_axes.
    """
    st = robot_model.get_state(q_rad_t)
    quat = st.ee_pose.quaternion
    cup_local, cup_world = _task_axes(task_axes, "cup")
    pour_local, pour_world = _task_axes(task_axes, "pour")
    cup = tilt_deg_from_quat(quat, cup_local, cup_world)
    pour = tilt_deg_from_quat(quat, pour_local, pour_world)
    return torch.where(torch.as_tensor(ee_y, device=q_rad_t.device) >= 0, cup, pour)


def build_corridor_waypoints(robot_model, q_all_rad, start_idx, goal_idx, pair_type,
                             goal_zone_arr,
                             cuboid_centers, cuboid_half, torso_centers, torso_half,
                             self_coll_mask, mask_for_speed, self_speed_scale,
                             args, offset, cup_blend, pour_blend,
                             cup_alpha, pour_alpha, y_divide):
    """R58N root fix: expand cross pairs into high-tilt corridor (waypoint -> goal) pairs.

    For a fraction of cross pairs, take the straight-line joint interpolation, keep the
    interior waypoints whose zone-tilt > corridor_min_tilt (the wrist-flip region the
    field is currently blind to), compute their full per-config labels (d_phys, d_orient,
    speed, normal, ee, zone) with the SAME functions the main pipeline uses, and return
    them plus new (waypoint -> original goal) cross pairs. The waypoint's speed is low by
    construction (high tilt -> low d_orient -> floor speed), so the field learns a speed
    valley there and its Eikonal geodesic routes around the flip.

    Returns dict with the new per-config arrays (to append to the pool) and the new
    pair index arrays (referencing pool positions OFFSET by the current pool size).
    """
    is_cross = pair_type == PAIR_CROSS
    cross_rows = np.where(is_cross)[0]
    if len(cross_rows) == 0 or args.corridor_inject_frac <= 0:
        return None

    rng = np.random.default_rng(args.seed + 777)
    n_sel = max(1, int(len(cross_rows) * args.corridor_inject_frac))
    sel = rng.choice(cross_rows, size=n_sel, replace=False)
    s_rad = q_all_rad[start_idx[sel]]          # (n_sel, dim)
    g_rad = q_all_rad[goal_idx[sel]]           # (n_sel, dim)
    g_pool_idx = goal_idx[sel]                 # keep original goal pool index per pair
    g_zone_sel = goal_zone_arr[sel]            # keep original goal zone per pair
    dim = q_all_rad.shape[1]
    device = args.device

    # interior interpolation fractions (strictly between endpoints)
    K = max(1, int(args.corridor_waypoints))
    fracs = np.linspace(0.0, 1.0, K + 2)[1:-1]  # K interior points

    wp_rad = []        # collected waypoint rad configs
    wp_goal_idx = []   # original goal pool index for each waypoint
    wp_goal_zone = []  # original goal zone for each waypoint
    for f in fracs:
        wp = s_rad + f * (g_rad - s_rad)        # (n_sel, dim) linear in joint space
        wp_rad.append(wp)
        wp_goal_idx.append(g_pool_idx)
        wp_goal_zone.append(g_zone_sel)
    wp_rad = np.concatenate(wp_rad, axis=0).astype(np.float32)
    wp_goal_idx = np.concatenate(wp_goal_idx, axis=0)
    wp_goal_zone = np.concatenate(wp_goal_zone, axis=0)

    # zone (by waypoint ee_y) and zone-tilt; keep only high-tilt waypoints
    wp_t = torch.tensor(wp_rad, dtype=torch.float32, device=device)
    ee_pos = compute_ee_positions(robot_model, wp_rad, device=device)  # (M,6)? -> only need start xyz
    wp_ee_y = ee_pos[:, 1] if ee_pos.shape[1] == 3 else ee_pos[:, 1]
    tilt = _zone_tilt_deg(
        wp_t, robot_model, wp_ee_y,
        task_axes=getattr(args, "task_axes_resolved", None)).cpu().numpy()
    keep = tilt > args.corridor_min_tilt
    if keep.sum() == 0:
        print(f"  [corridor] no waypoints above {args.corridor_min_tilt:.0f}deg tilt; skipping")
        return None
    wp_rad = wp_rad[keep]; wp_goal_idx = wp_goal_idx[keep]
    wp_goal_zone = wp_goal_zone[keep]
    wp_ee_y = wp_ee_y[keep]
    if getattr(args, "zone_mode_resolved", "y") == "corner_table":
        wp_zone = assign_corner_zone_ids_np(
            ee_pos[keep][:, :2], args.corner_zones_resolved).astype(np.int64)
    else:
        wp_zone = np.where(wp_ee_y >= y_divide, ZONE_CUPPING, ZONE_POURING).astype(np.int64)

    # per-config labels for the waypoints, reusing the pipeline's own functions
    d_obs_w, d_self_w, _d_ws_w = compute_physical_distance(
        robot_model, wp_rad, cuboid_centers, cuboid_half,
        torso_centers, torso_half,
        self_coll_mask=(self_coll_mask if not args.no_self_in_speed else None),
        device=device)
    d_phys_w = d_obs_w.copy()
    if not args.no_self_in_speed:
        d_phys_w = np.minimum(d_phys_w, d_self_w * self_speed_scale)
    d_orient_cup_w = compute_orient_distance(
        robot_model, wp_rad, "cupping", args.margin,
        args.cup_tilt_deg, args.cup_tilt_soft, device=device,
        morph=args.morph_orient, morph_mode=args.morph_mode,
        zone_mode=getattr(args, "zone_mode_resolved", "y"),
        corner_overlap_deg=args.corner_overlap_deg,
        y_divide=args.y_divide, overlap=args.overlap,
        task_axes=getattr(args, "task_axes_resolved", None))
    d_orient_pour_w = compute_orient_distance(
        robot_model, wp_rad, "pouring", args.margin,
        args.pour_tilt_deg, args.pour_tilt_soft, device=device,
        morph=args.morph_orient, morph_mode=args.morph_mode,
        zone_mode=getattr(args, "zone_mode_resolved", "y"),
        corner_overlap_deg=args.corner_overlap_deg,
        y_divide=args.y_divide, overlap=args.overlap,
        rzy_safe=args.pour_rzy_safe, rzy_soft=args.pour_rzy_soft,
        task_axes=getattr(args, "task_axes_resolved", None))
    speed_cup_w = compute_speed(d_phys_w, d_orient_cup_w, cup_alpha, args.margin, offset, blend_mode=cup_blend)
    speed_pour_w = compute_speed(d_phys_w, d_orient_pour_w, pour_alpha, args.margin, offset, blend_mode=pour_blend)
    # R58Q: optionally drive the high-tilt corridor speed to a deeper floor (near-impassable)
    # so the Eikonal geodesic routes AROUND it instead of just slowing through. The normal
    # floor (offset/margin) only realized ~42% slowdown in the learned field; this deepens it.
    if getattr(args, "corridor_speed_floor", 0.0) and args.corridor_speed_floor > 0:
        cf = float(args.corridor_speed_floor)
        speed_cup_w = np.minimum(speed_cup_w, cf)
        speed_pour_w = np.minimum(speed_pour_w, cf)
    normal_cup_w = compute_gradient_task(
        robot_model, wp_rad, cuboid_centers, cuboid_half,
        torso_centers, torso_half, mask_for_speed,
        "cupping", cup_alpha, args.margin, args.cup_tilt_deg, args.cup_tilt_soft,
        device=device, blend_mode=cup_blend,
        self_speed_scale=self_speed_scale,
        morph=args.morph_orient, morph_mode=args.morph_mode,
        zone_mode=getattr(args, "zone_mode_resolved", "y"),
        corner_overlap_deg=args.corner_overlap_deg,
        y_divide=args.y_divide, overlap=args.overlap,
        task_axes=getattr(args, "task_axes_resolved", None))
    normal_pour_w = compute_gradient_task(
        robot_model, wp_rad, cuboid_centers, cuboid_half,
        torso_centers, torso_half, mask_for_speed,
        "pouring", pour_alpha, args.margin, args.pour_tilt_deg, args.pour_tilt_soft,
        device=device, blend_mode=pour_blend,
        self_speed_scale=self_speed_scale,
        morph=args.morph_orient, morph_mode=args.morph_mode,
        zone_mode=getattr(args, "zone_mode_resolved", "y"),
        corner_overlap_deg=args.corner_overlap_deg,
        y_divide=args.y_divide, overlap=args.overlap,
        rzy_safe=args.pour_rzy_safe, rzy_soft=args.pour_rzy_soft,
        task_axes=getattr(args, "task_axes_resolved", None))

    print(f"  [corridor] injected {len(wp_rad)} high-tilt waypoints "
          f"(>{args.corridor_min_tilt:.0f}deg) from {n_sel} cross pairs x {K} interior pts; "
          f"speed_cup mean={speed_cup_w.mean():.3f} speed_pour mean={speed_pour_w.mean():.3f} "
          f"(floor={offset/args.margin:.2f})")

    # R58U: clean tilt-descent gradient for the waypoints too (so the pool stays consistent)
    tilt_grad_w = compute_tilt_gradient(
        robot_model, wp_rad, device=device,
        zone_mode=getattr(args, "zone_mode_resolved", "y"),
        y_divide=args.y_divide,
        corner_zones=getattr(args, "corner_zones_resolved", None),
        task_axes=getattr(args, "task_axes_resolved", None))

    return dict(
        wp_rad=wp_rad, wp_ee=ee_pos[keep], wp_goal_idx=wp_goal_idx, wp_zone=wp_zone,
        wp_goal_zone=wp_goal_zone, tilt_grad=tilt_grad_w,
        d_obs=d_obs_w, d_self=d_self_w,
        d_orient_cup=d_orient_cup_w, d_orient_pour=d_orient_pour_w,
        speed_cup=speed_cup_w, speed_pour=speed_pour_w,
        normal_cup=normal_cup_w, normal_pour=normal_pour_w,
    )


def filter_cross_pairs_by_segment(robot_model, q_all, start_idx, goal_idx, pair_type,
                                  cuboid_centers, cuboid_half, torso_centers, torso_half,
                                  self_coll_mask, args):
    """Drop cross pairs whose straight interpolation is physically inconsistent.

    This is an offline label-quality filter: it prevents endpoint-valid but
    internally invalid cross pairs from becoming positive training examples.
    It is not used by the runtime planner or eval.
    """
    cross_rows = np.where(pair_type == PAIR_CROSS)[0]
    if len(cross_rows) == 0:
        return np.ones(len(pair_type), dtype=bool), {
            "checked": 0,
            "kept": 0,
            "rejected_collision": 0,
            "rejected_tilt": 0,
        }

    waypoints = max(2, int(args.cross_segment_waypoints))
    batch_pairs = max(1, int(args.cross_segment_batch_size))
    alphas = torch.linspace(0.0, 1.0, waypoints, device=args.device).view(1, waypoints, 1)
    keep = np.ones(len(pair_type), dtype=bool)

    n_kept = 0
    n_rej_coll = 0
    n_rej_tilt = 0

    for i in tqdm(range(0, len(cross_rows), batch_pairs),
                  desc="Filtering cross segments",
                  total=(len(cross_rows) + batch_pairs - 1) // batch_pairs):
        rows = cross_rows[i:i + batch_pairs]
        q_start = torch.tensor(q_all[start_idx[rows]], dtype=torch.float32, device=args.device)
        q_goal = torch.tensor(q_all[goal_idx[rows]], dtype=torch.float32, device=args.device)
        q_interp = q_start.unsqueeze(1) + alphas * (q_goal - q_start).unsqueeze(1)
        flat = q_interp.reshape(-1, DOF)
        B = len(rows)

        in_coll, _ = check_collision_full(
            robot_model, flat, cuboid_centers, cuboid_half,
            torso_cuboid_centers=torso_centers,
            torso_cuboid_half=torso_half,
            self_collision_mask=self_coll_mask,
            torso_ignore_sphere_indices=TORSO_IGNORE_SPHERE_INDICES,
            margin=args.cross_segment_collision_margin,
            self_collision_margin=args.cross_segment_self_margin,
            device=args.device,
        )
        collision_ok = ~in_coll.reshape(B, waypoints).any(dim=1)

        state = robot_model.get_state(flat)
        ee_y_seg = state.ee_pose.position[:, 1].reshape(B, waypoints)
        quat = state.ee_pose.quaternion
        cup_local, cup_world = _task_axes(getattr(args, "task_axes_resolved", None), "cup")
        pour_local, pour_world = _task_axes(getattr(args, "task_axes_resolved", None), "pour")
        cup_tilt = tilt_deg_from_quat(quat, cup_local, cup_world)
        cup_tilt = cup_tilt.reshape(B, waypoints)
        pour_tilt = tilt_deg_from_quat(quat, pour_local, pour_world)
        pour_tilt = pour_tilt.reshape(B, waypoints)

        if args.cross_segment_tilt_overlap > 0:
            lower = args.y_divide - args.cross_segment_tilt_overlap
            upper = args.y_divide + args.cross_segment_tilt_overlap
            cup_only = ee_y_seg > upper
            pour_only = ee_y_seg < lower
            overlap = (ee_y_seg >= lower) & (ee_y_seg <= upper)
            cup_pass = cup_tilt <= args.cross_segment_cup_tilt_deg
            pour_pass = pour_tilt <= args.cross_segment_pour_tilt_deg
            tilt_waypoint_ok = torch.where(
                overlap,
                cup_pass | pour_pass,
                (cup_pass | ~cup_only) & (pour_pass | ~pour_only),
            )
        else:
            cup_mask = ee_y_seg >= args.y_divide
            pour_mask = ee_y_seg < args.y_divide
            cup_ok = (cup_tilt <= args.cross_segment_cup_tilt_deg) | ~cup_mask
            pour_ok = (pour_tilt <= args.cross_segment_pour_tilt_deg) | ~pour_mask
            tilt_waypoint_ok = cup_ok & pour_ok

        tilt_ok = tilt_waypoint_ok.all(dim=1)
        chunk_keep = collision_ok & tilt_ok

        keep[rows] = chunk_keep.detach().cpu().numpy()
        n_kept += int(chunk_keep.sum().item())
        n_rej_coll += int((~collision_ok).sum().item())
        n_rej_tilt += int((~tilt_ok).sum().item())

    stats = {
        "checked": int(len(cross_rows)),
        "kept": int(n_kept),
        "rejected_collision": int(n_rej_coll),
        "rejected_tilt": int(n_rej_tilt),
        "waypoints": int(waypoints),
        "collision_margin": float(args.cross_segment_collision_margin),
        "self_margin": float(args.cross_segment_self_margin),
        "cup_tilt_deg": float(args.cross_segment_cup_tilt_deg),
        "pour_tilt_deg": float(args.cross_segment_pour_tilt_deg),
        "tilt_overlap": float(args.cross_segment_tilt_overlap),
    }
    return keep, stats


# ── Safe-neighbor index builder ──────────────────────────────────────────

def build_safe_index_standalone(args, out_dir, robot_model, cuboid_centers, cuboid_half,
                                q_center, q_range):
    """Build safe_index.npz: normalized collision-free C-space cloud + per-config
    d_obs, tilt_cup, tilt_pour, ee_x, ee_y. Used by a later path-loss to retrieve
    a known collision-free config near any query point.

    Reads the SAME pkls as the full pipeline, filters to collision-free (obstacle
    distance >= 0, using the same OBSTACLE_IGNORE_SPHERE_INDICES convention as the
    speed field), and saves the index into out_dir. Tilt formulas must match the
    scene task_axes convention.
    """
    import pickle
    qs = []
    for pkl_path in [args.pkl_cupping, args.pkl_pouring]:
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        q = d["q_solutions"]
        if "success" in d:
            q = q[d["success"]]
        qs.append(q)
    q_all = np.concatenate(qs, axis=0).astype(np.float32)            # (N,6)
    bs = 8192
    keep = np.zeros(q_all.shape[0], dtype=bool)
    d_obs_all = np.empty(q_all.shape[0], dtype=np.float32)
    tcup = np.empty(q_all.shape[0], dtype=np.float32)
    tpour = np.empty(q_all.shape[0], dtype=np.float32)
    eex = np.empty(q_all.shape[0], dtype=np.float32)
    eey = np.empty(q_all.shape[0], dtype=np.float32)
    for i in tqdm(range(0, q_all.shape[0], bs), desc="Building safe index",
                  total=(q_all.shape[0] + bs - 1) // bs):
        e = min(i + bs, q_all.shape[0])
        qb = torch.tensor(q_all[i:e], dtype=torch.float32, device=args.device)
        st = robot_model.get_state(qb)
        sp = st.link_spheres_tensor
        dd = sphere_cuboid_distance(sp[:, :, :3], sp[:, :, 3], cuboid_centers, cuboid_half,
                                    ignore_sphere_indices=OBSTACLE_IGNORE_SPHERE_INDICES)
        qt = st.ee_pose.quaternion
        cup_local, cup_world = _task_axes(getattr(args, "task_axes_resolved", None), "cup")
        pour_local, pour_world = _task_axes(getattr(args, "task_axes_resolved", None), "pour")
        tc = tilt_deg_from_quat(qt, cup_local, cup_world)
        tp = tilt_deg_from_quat(qt, pour_local, pour_world)
        d_obs_all[i:e] = dd.detach().cpu().numpy()
        tcup[i:e] = tc.detach().cpu().numpy()
        tpour[i:e] = tp.detach().cpu().numpy()
        eex[i:e] = st.ee_pose.position[:, 0].detach().cpu().numpy()
        eey[i:e] = st.ee_pose.position[:, 1].detach().cpu().numpy()
        keep[i:e] = (dd >= 0.0).detach().cpu().numpy()
    q_all, d_obs_all = q_all[keep], d_obs_all[keep]
    tcup, tpour, eex, eey = tcup[keep], tpour[keep], eex[keep], eey[keep]
    qc = q_center.cpu().numpy() if hasattr(q_center, "cpu") else np.asarray(q_center)
    qr = q_range.cpu().numpy() if hasattr(q_range, "cpu") else np.asarray(q_range)
    q_norm_full = (q_all - qc) / qr
    np.savez(out_dir / "safe_index.npz",
             q_norm=q_norm_full.astype(np.float32), d_obs=d_obs_all,
             tilt_cup=tcup, tilt_pour=tpour, ee_x=eex, ee_y=eey)
    print(f"  Saved safe_index.npz ({q_norm_full.shape[0]} collision-free configs)")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-task training data v2: task-specific FK sampling")
    parser.add_argument("--pkl_cupping", type=str, required=True,
                        help="Cupping FK pkl (tilt-filtered)")
    parser.add_argument("--pkl_pouring", type=str, required=True,
                        help="Pouring FK pkl (pour-filtered)")
    parser.add_argument("--scene", type=str, default="scene_ur5e.yml")
    parser.add_argument("--n_pairs", type=int, default=2000000)
    parser.add_argument("--y_divide", type=float, default=0.0,
                        help="Zone boundary in EE Y-space (UR5e: 0.0; cupping is +y side)")
    parser.add_argument("--overlap", type=float, default=0.05,
                        help="Overlap width in meters on each side of y_divide")
    parser.add_argument("--zone_mode", type=str, default="auto",
                        choices=["auto", "y", "corner_table"],
                        help="Routing geometry for task zones. auto reads zones.type from --scene; "
                             "corner_table uses scene cup_region/pour_region/overlap_region.")
    parser.add_argument("--intra_ratio", type=float, default=0.8,
                        help="Fraction of intra-zone pairs (rest are cross-zone)")
    parser.add_argument("--margin", type=float, default=0.05,
                        help="Distance at which speed=1.0 (fully free)")
    parser.add_argument("--offset", type=float, default=None,
                        help="Distance at which speed is minimum (default: margin/10)")
    parser.add_argument("--alpha", type=float, default=0.65,
                        help="Orient/collision blend: speed = alpha*d_orient + (1-alpha)*d_phys (only for --blend_mode alpha)")
    parser.add_argument("--blend_mode", type=str, default="min",
                        choices=["min", "alpha", "product", "harmonic", "quadrature"],
                        help="How to combine d_obs and d_orient for BOTH tasks. "
                             "Use --cup_blend_mode/--pour_blend_mode for per-zone override.")
    parser.add_argument("--cup_blend_mode", type=str, default=None,
                        choices=["min", "alpha", "product", "harmonic", "quadrature"],
                        help="Override blend mode for cupping speed (default: use --blend_mode)")
    parser.add_argument("--pour_blend_mode", type=str, default=None,
                        choices=["min", "alpha", "product", "harmonic", "quadrature"],
                        help="Override blend mode for pouring speed (default: use --blend_mode)")
    parser.add_argument("--cup_alpha", type=float, default=None,
                        help="Override alpha for cupping (default: use --alpha)")
    parser.add_argument("--pour_alpha", type=float, default=None,
                        help="Override alpha for pouring (default: use --alpha)")
    parser.add_argument("--self_margin", type=float, default=0.005,
                        help="Filter configs with self-collision distance below this")
    parser.add_argument("--self_speed_margin", type=float, default=None,
                        help="Self-collision clearance mapped to free speed. "
                             "Default reuses --margin.")
    parser.add_argument("--workspace_height", type=float, default=None,
                        help="Enable workspace containment in the speed field. Height (m) "
                             "above the tabletop; the XYZ box is derived from the scene's "
                             "sampling bounds, matching the evaluator's workspace gate. "
                             "Omit to disable (default: no workspace term).")
    parser.add_argument("--workspace_speed_margin", type=float, default=None,
                        help="Distance (m) from the workspace boundary at which speed is "
                             "minimum. Smaller than --margin sharpens the boundary. "
                             "Defaults to --margin.")
    parser.add_argument("--no_self_in_speed", action="store_true",
                        help="Exclude self-collision from speed/gradient computation")
    parser.add_argument("--cup_tilt_deg", type=float, default=15.0,
                        help="Cupping max tilt from vertical (degrees)")
    parser.add_argument("--cup_tilt_soft", type=float, default=10.0,
                        help="Cupping soft margin beyond tilt_max (degrees)")
    parser.add_argument("--pour_tilt_deg", type=float, default=90.0,
                        help="Pouring max X-tilt (degrees)")
    parser.add_argument("--pour_tilt_soft", type=float, default=15.0,
                        help="Pouring soft margin beyond tilt_max (degrees)")
    parser.add_argument("--pour_rzy_safe", type=float, default=0.15,
                        help="Rz_y threshold above which cup-left-only is safe")
    parser.add_argument("--pour_rzy_soft", type=float, default=0.30,
                        help="Rz_y soft transition range")
    parser.add_argument("--graded_orient", action="store_true",
                        help="R58: graded orientation cost. d_orient ramps continuously from "
                             "upright (tilt=0) instead of being flat==margin across the valid "
                             "cone. Fixes the Phase-0 label flaw (no orientation gradient "
                             "mid-transition) and widens the C-space ramp so it is learnable.")
    parser.add_argument("--signed_orient", action="store_true",
                        help="R62: d_orient is the SIGNED angular distance to the tilt "
                             "constraint surface -- margin*(tilt_max - tilt)/tilt_band, "
                             "clipped to [-margin, margin]. Zero exactly AT tilt_max, "
                             "positive inside, negative outside. Direct analogue of the "
                             "obstacle distance term. Overrides --graded_orient.")
    parser.add_argument("--tilt_band", type=float, default=20.0,
                        help="With --signed_orient: degrees over which d_orient spans "
                             "+margin -> -margin around tilt_max.")
    parser.add_argument("--tilt_grade", type=float, default=50.0,
                        help="With --graded_orient: degrees over which d_orient ramps "
                             "margin->0. Larger = wider/softer ramp. Default 50 (~2.5x a 20deg "
                             "limit, so speed at the limit ~0.6*margin and keeps dropping past).")
    parser.add_argument("--out_dir", type=str, default="datasets/arm/UR5e_multi")
    parser.add_argument("--rebuild_index_only", action="store_true",
                        help="Fast path: build safe_index.npz on the existing dataset "
                             "(reads pkls, filters collision-free, computes per-config "
                             "d_obs/tilt/ee_y, saves safe_index.npz) then exit. Skips the "
                             "expensive pair-sampling / speed-field / gradient pipeline.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for pair sampling")
    parser.add_argument("--n_cupping", type=int, default=None,
                        help="Max configs from cupping pkl (default: use all)")
    parser.add_argument("--n_pouring", type=int, default=None,
                        help="Max configs from pouring pkl (default: use all)")
    parser.add_argument("--source_aware", action="store_true",
                        help="Source-aware pair sampling: cupping pairs only from "
                             "cupping-pkl configs, pouring pairs only from pouring-pkl configs. "
                             "Eliminates floor-speed contamination from wrong-source configs.")
    parser.add_argument("--n_synth_colliding", type=int, default=0,
                        help="R66: synthesise this many configs INSIDE obstacles (by "
                             "perturbing collision-free ones until they penetrate) so "
                             "the speed field gets negative-distance supervision. The "
                             "source pkls are already collision-filtered, so without "
                             "this the field never sees d_obstacle < 0. 0 = off.")
    parser.add_argument("--keep_colliding", type=float, default=0.0,
                        help="R66: fraction [0,1] of OBSTACLE-colliding configs to keep "
                             "so the speed field gets negative-distance supervision "
                             "inside obstacles. Self-collision rejects are never kept. "
                             "0.0 = legacy behaviour (no config inside any obstacle).")
    parser.add_argument("--keep_contaminated", type=float, default=0.0,
                        help="R64: fraction [0,1] of wrong-source configs in exclusive "
                             "zones to KEEP instead of deleting. They carry correct, "
                             "slow labels for the zone's task, which is the supervision "
                             "our cross failures need. 0.0 = legacy behaviour.")
    parser.add_argument("--source_mix", type=float, default=0.0,
                        help="When --source_aware, fraction of intra-zone pairs that include "
                             "one endpoint from the OTHER source pkl (orientation negative examples). "
                             "E.g. 0.2 means 20%% of cupping pairs have one endpoint from pour_pkl.")
    parser.add_argument("--filter_overlap_dual", action="store_true",
                        help="In the overlap zone, only keep configs where BOTH d_orient_cup > threshold "
                             "AND d_orient_pour > threshold. Removes configs with incompatible gradients.")
    parser.add_argument("--dual_threshold", type=float, default=0.0,
                        help="Threshold for --filter_overlap_dual: keep configs where "
                             "d_orient_cup > threshold AND d_orient_pour > threshold. "
                             "0.0 = just positive (default), 0.02 = stricter.")
    parser.add_argument("--cross_dual_filter", action="store_true",
                        help="Only filter CROSS pair endpoints (not intra pairs) to dual-constraint. "
                             "Cross pairs keep only endpoints where d_orient_cup > 0 AND d_orient_pour > 0.")
    parser.add_argument("--corner_cross_frac", type=float, default=0.0,
                        help="Fraction of cross pairs with an endpoint drawn from "
                             "the inner-corner box (wing<->corner). Forces corner "
                             "transition coverage past the boundary_cross_frac ~5%% cap.")
    parser.add_argument("--boundary_cross_frac", type=float, default=0.0,
                        help="Fraction of cross pairs sampled as boundary-local nearest neighbors "
                             "in normalized joint space. Default 0 preserves random cross sampling.")
    parser.add_argument("--boundary_cross_band", type=float, default=0.10,
                        help="EE-y half-band around y_divide for boundary-local cross pools.")
    parser.add_argument("--local_cross_k", type=int, default=8,
                        help="Randomly pick among the k nearest opposite-task boundary configs.")
    parser.add_argument("--min_cross_dist", type=float, default=0.0,
                        help="Min NORMALIZED cspace dist for a cross pair (0 = no gate)")
    parser.add_argument("--max_cross_dist", type=float, default=0.0,
                        help="Max NORMALIZED cspace dist for a cross pair (0 = no cap)")
    parser.add_argument("--cross_dist_oversample", type=int, default=6,
                        help="Candidate multiplier for distance-gated rejection sampling")
    parser.add_argument("--cross_segment_filter", action="store_true",
                        help="Offline label-quality filter for cross pairs: keep only pairs whose "
                             "linear joint interpolation is collision-free and zone-tilt valid.")
    parser.add_argument("--cross_segment_waypoints", type=int, default=21,
                        help="Number of interpolation waypoints for --cross_segment_filter.")
    parser.add_argument("--cross_segment_batch_size", type=int, default=2048,
                        help="Cross pairs per GPU batch for --cross_segment_filter.")
    parser.add_argument("--cross_segment_collision_margin", type=float, default=0.005,
                        help="Obstacle/torso margin for --cross_segment_filter.")
    parser.add_argument("--cross_segment_self_margin", type=float, default=0.002,
                        help="Self-collision margin for --cross_segment_filter.")
    parser.add_argument("--cross_segment_cup_tilt_deg", type=float, default=20.0,
                        help="Cupping tilt threshold for --cross_segment_filter.")
    parser.add_argument("--cross_segment_pour_tilt_deg", type=float, default=25.0,
                        help="Pouring tilt threshold for --cross_segment_filter.")
    parser.add_argument("--cross_segment_tilt_overlap", type=float, default=0.0,
                        help="OR-logic EE-y overlap for --cross_segment_filter; default 0.0 "
                             "matches strict zone_eval.")
    parser.add_argument("--cross_segment_refill_cross", action="store_true",
                        help="After --cross_segment_filter, resample kept clean cross pairs with "
                             "replacement to restore the pre-filter cross count.")
    # R58N: corridor-waypoint injection. Adds straight-line interpolation waypoints of
    # cross pairs as STANDALONE (waypoint -> goal) training configs, so the field learns
    # the (correctly low) speed of the high-tilt mid-corridor and its Eikonal geodesic
    # routes AROUND the wrist-flip. PROVEN root cause: 0% of cross configs have tilt>45deg,
    # so the trained field only slows ~14% at high tilt instead of the ~85% the label gives.
    parser.add_argument("--corridor_inject_frac", type=float, default=0.0,
                        help="R58N: fraction of cross pairs to expand into corridor-waypoint "
                             "(waypoint->goal) training pairs. 0 disables (default).")
    parser.add_argument("--corridor_waypoints", type=int, default=5,
                        help="R58N: interior interpolation waypoints per selected cross pair "
                             "(evenly spaced strictly between start and goal).")
    parser.add_argument("--corridor_min_tilt", type=float, default=30.0,
                        help="R58N: only inject waypoints whose zone-tilt exceeds this (deg); "
                             "low-tilt waypoints add nothing new. Keeps the high-tilt corridor.")
    parser.add_argument("--compute_tilt_grad", action="store_true",
                        help="R58U: save tilt_grad.npy = normalized -grad_q tilt_FK per config "
                             "(clean offline FK tilt-descent target for orient grad-supervision; "
                             "inference never calls FK).")
    parser.add_argument("--tilt_grad_eps", type=float, default=2e-4,
                        help="R58U: finite-diff eps for the FK tilt gradient (finer than the "
                             "min-blended normal's 1e-3, which was unreliable for orientation).")
    parser.add_argument("--corridor_speed_floor", type=float, default=0.0,
                        help="R58Q: if >0, override the injected corridor waypoints' BOTH-head "
                             "speed to this value (e.g. 0.02), making the high-tilt corridor "
                             "near-impassable so the Eikonal geodesic routes AROUND it. The "
                             "normal speed floor (offset/margin) only realized a 42%% slowdown; "
                             "this forces a deeper valley. 0 disables (use computed speed).")
    parser.add_argument("--morph_orient", action="store_true",
                        help="R60: in the overlap band, morph the orientation tilt "
                             "continuously from cupping to pouring (physically correct "
                             "gradual reorientation) instead of per-zone pure tilt.")
    parser.add_argument("--morph_mode", type=str, default="linear",
                        choices=["linear", "max"],
                        help="R60 morph blend mode for overlap band.")
    parser.add_argument("--corner_overlap_deg", type=float, default=45.0,
                        help="L-shape corner morph: angular half-width (deg) of the "
                             "OR-band around the inner diagonal. Outside it the wings "
                             "keep pure cup/pour targets; inside the corner morphs. "
                             "45=full-arc ramp, 25=pure wings beyond 25deg. Only used "
                             "with --morph_orient on a corner_table scene.")
    args = parser.parse_args()

    # R58: activate graded orientation cost globally (read by the _orient_* batch fns).
    global _GRADED_ORIENT, _TILT_GRADE, _SIGNED_ORIENT, _TILT_BAND
    _GRADED_ORIENT = bool(args.graded_orient)
    _TILT_GRADE = float(args.tilt_grade)
    _SIGNED_ORIENT = bool(args.signed_orient)
    _TILT_BAND = float(args.tilt_band)
    if _SIGNED_ORIENT:
        print(f"[R62] SIGNED orientation distance ON: "
              f"d_orient = margin*(tilt_max - tilt)/{_TILT_BAND:.1f}, clipped to +-margin")
        if _GRADED_ORIENT:
            print("[R62] --graded_orient is ignored while --signed_orient is set.")
            _GRADED_ORIENT = False
    if _GRADED_ORIENT:
        print(f"[R58] GRADED orientation cost ON: d_orient = margin*(1 - tilt/{_TILT_GRADE:.1f})")

    offset = args.offset if args.offset is not None else args.margin / 10.0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_total_start = time.time()

    # ── Step 1: Load BOTH pkl files ─────────────────────────────────────────
    print(f"Loading cupping pkl: {args.pkl_cupping}")
    with open(args.pkl_cupping, "rb") as f:
        pkl_cup = pickle.load(f)
    q_cup = pkl_cup["q_solutions"]
    if "success" in pkl_cup:
        q_cup = q_cup[pkl_cup["success"]]
    if args.n_cupping and len(q_cup) > args.n_cupping:
        q_cup = q_cup[:args.n_cupping]
    N_cup_raw = q_cup.shape[0]
    print(f"  Loaded {N_cup_raw} cupping configs")

    print(f"Loading pouring pkl: {args.pkl_pouring}")
    with open(args.pkl_pouring, "rb") as f:
        pkl_pour = pickle.load(f)
    q_pour = pkl_pour["q_solutions"]
    if "success" in pkl_pour:
        q_pour = q_pour[pkl_pour["success"]]
    if args.n_pouring and len(q_pour) > args.n_pouring:
        q_pour = q_pour[:args.n_pouring]
    N_pour_raw = q_pour.shape[0]
    print(f"  Loaded {N_pour_raw} pouring configs")

    # Mark source for each config
    # source_id: 0=cupping_pkl, 1=pouring_pkl
    source_id = np.concatenate([
        np.zeros(N_cup_raw, dtype=np.int8),
        np.ones(N_pour_raw, dtype=np.int8),
    ])

    # Merge all configs
    q_all = np.concatenate([q_cup, q_pour], axis=0)
    N_raw = q_all.shape[0]
    print(f"\n  Total merged: {N_raw} configs (cup={N_cup_raw}, pour={N_pour_raw})")

    del q_cup, q_pour  # free memory

    # ── Step 2: Joint normalization ──────────────────────────────────────────
    q_center = (Q_MAX + Q_MIN) / 2.0
    q_range = Q_MAX - Q_MIN
    np.savez(out_dir / "joint_normalization.npz",
             q_center=q_center, q_range=q_range, q_min=Q_MIN, q_max=Q_MAX)
    print(f"  Saved joint_normalization.npz")

    # ── Step 3: Init robot + scene ───────────────────────────────────────────
    print("Initializing cuRobo robot model...")
    _, _, robot_model, _ = init_curobo(ROBOT_CFG_PATH, URDF_PATH, args.device)

    print(f"Loading scene cuboids: {args.scene}")
    cuboids = load_scene_cuboids(args.scene)
    cuboid_centers, cuboid_half = cuboid_half_extents(cuboids, args.device)
    print(f"  {len(cuboids)} cuboids: {[c['name'] for c in cuboids]}")
    zone_mode, corner_zones = load_zone_config(args.scene, args.zone_mode)
    args.zone_mode_resolved = zone_mode
    args.corner_zones_resolved = corner_zones
    task_axes = load_task_axes(args.scene)
    args.task_axes_resolved = task_axes
    print(f"  zone_mode: {zone_mode}")
    if corner_zones is not None:
        print(f"  corner_zones: {corner_zones}")
    cup_local, cup_world = task_axis_arrays(task_axes, "cup")
    pour_local, pour_world = task_axis_arrays(task_axes, "pour")
    print(f"  task_axes: cup local {axis_label(cup_local)} -> world {axis_label(cup_world)}; "
          f"pour local {axis_label(pour_local)} -> world {axis_label(pour_world)}")

    # ── Fast path: build only the safe-neighbor index, then exit ─────────────
    # All inputs the builder needs (out_dir, robot_model, cuboid geometry,
    # q_center/q_range) are ready here, BEFORE the expensive collision filter /
    # pair sampling / speed-field / gradient stages below.
    if args.rebuild_index_only:
        print("\n── Rebuild-index-only: building safe_index.npz ──")
        build_safe_index_standalone(args, out_dir, robot_model,
                                    cuboid_centers, cuboid_half, q_center, q_range)
        return

    # UR5e has NO torso obstacle (TORSO_OBSTACLE_PATH is None) → skip the torso load.
    if TORSO_OBSTACLE_PATH is not None:
        torso_scene = (str(Path(args.scene).parent / TORSO_OBSTACLE_PATH)
                       if '/' in args.scene else TORSO_OBSTACLE_PATH)
        torso_centers, torso_half = load_cuboid_geometry(torso_scene, args.device)
    else:
        torso_centers, torso_half = None, None

    with open(ROBOT_CFG_PATH) as f:
        rcfg = yaml.safe_load(f)
    self_coll_ignore = rcfg["robot_cfg"]["kinematics"].get("self_collision_ignore", {})
    self_coll_mask, _ = build_self_collision_mask(
        SPHERE_YML_PATH, self_coll_ignore, COLLISION_LINK_NAMES, device=args.device
    )

    # ── Step 4: Collision filtering ──────────────────────────────────────────
    print(f"Filtering collisions (self_margin={args.self_margin:.4f}m)...")
    t0 = time.time()
    in_coll_all = []
    obs_coll_all = []
    filter_bs = 2000
    for i in tqdm(range(0, N_raw, filter_bs), desc="Collision filtering",
                  total=(N_raw + filter_bs - 1) // filter_bs):
        end = min(i + filter_bs, N_raw)
        q_batch = torch.tensor(q_all[i:end], dtype=torch.float32, device=args.device)
        in_coll, d_result = check_collision_full(
            robot_model, q_batch, cuboid_centers, cuboid_half,
            torso_cuboid_centers=torso_centers, torso_cuboid_half=torso_half,
            self_collision_mask=self_coll_mask,
            torso_ignore_sphere_indices=TORSO_IGNORE_SPHERE_INDICES,
            margin=0.0, self_collision_margin=0.0, device=args.device)
        self_too_close = d_result['self'] < args.self_margin
        reject = in_coll | self_too_close
        # R66: track obstacle-only rejects separately.  Self-collision configs are
        # physically impossible and must always go; obstacle-colliding ones are the
        # only supervision the field can ever get at NEGATIVE obstacle distance.
        obs_only = in_coll & ~self_too_close
        in_coll_all.append(reject.cpu().numpy())
        obs_coll_all.append(obs_only.cpu().numpy())
    in_coll_all = np.concatenate(in_coll_all)
    obs_coll_all = np.concatenate(obs_coll_all)
    # R66: measured 2026-09-08 -- with every colliding config filtered out, the whole
    # training set has min(d_obstacle) = +0.0050 (exactly the offset) and ZERO configs
    # inside an obstacle.  The field therefore extrapolates "free" into the table
    # interior: probed slowness at inside-table waypoints was 2.93 (free space = 1.0,
    # a real contact should be 27.7), and MPPI drives the upper_arm elbow 7.5 cm
    # through a 5 cm slab for a median of 9 consecutive waypoints.  Retaining some
    # obstacle-colliding configs is the only way to supply negative-distance labels;
    # compute_physical_distance already returns SIGNED distance and clip(d, offset,
    # margin) already maps negatives to the speed floor, so nothing downstream changes.
    keep_coll = float(getattr(args, 'keep_colliding', 0.0) or 0.0)
    if keep_coll > 0.0:
        rng_coll = np.random.default_rng(args.seed + 977)
        cand = np.flatnonzero(obs_coll_all & in_coll_all)
        n_keep = int(round(keep_coll * cand.size))
        if n_keep > 0:
            sel = rng_coll.choice(cand, size=n_keep, replace=False)
            in_coll_all[sel] = False
        print(f"  [R66] retaining {n_keep}/{cand.size} obstacle-colliding configs "
              f"(keep_colliding={keep_coll}) to supply d_obstacle < 0 labels")
    n_coll = in_coll_all.sum()
    n_cup_coll = in_coll_all[:N_cup_raw].sum()
    n_pour_coll = in_coll_all[N_cup_raw:].sum()
    print(f"  {n_coll}/{N_raw} configs rejected (cup={n_cup_coll}, pour={n_pour_coll})")
    q_all = q_all[~in_coll_all]
    source_id = source_id[~in_coll_all]
    N = q_all.shape[0]
    print(f"  {N} configs remaining after collision filter ({time.time()-t0:.1f}s)")
    print(f"    From cupping pkl: {(source_id == 0).sum()}")
    print(f"    From pouring pkl: {(source_id == 1).sum()}")

    # ── R66: synthesise configs INSIDE obstacles ─────────────────────────────
    # The source pkls come from sample_fk.py, which already discards colliding poses,
    # so this pipeline never sees a config with d_obstacle < 0 -- min over the whole
    # 2M-endpoint training set is exactly +0.0050, the offset.  The field therefore
    # extrapolates "free" into the table interior.  Probed 2026-09-08 on the deployed
    # 140.3 model: slowness at inside-table waypoints is 2.93 where free space is 1.0
    # and a real contact should be 27.7, and MPPI drives the upper_arm elbow (sphere 2)
    # 7.5 cm through a 5 cm slab for a median of 9 CONSECUTIVE waypoints -- verified
    # not to be a tunnelling artifact (40.7% of waypoints strictly inside).
    #
    # We perturb collision-free configs until they penetrate rather than sampling
    # uniformly, so the synthesised set concentrates just inside the boundary, which
    # is where the planner actually strays.  Self-colliding samples are always
    # discarded -- those configurations are physically impossible.
    n_synth = int(getattr(args, 'n_synth_colliding', 0) or 0)
    if n_synth > 0:
        print(f"[R66] synthesising up to {n_synth} configs inside obstacles...")
        t_syn = time.time()
        rng_s = np.random.default_rng(args.seed + 1013)
        got_q, got_src = [], []
        n_have, attempts = 0, 0
        while n_have < n_synth and attempts < 400:
            attempts += 1
            sigma = float(rng_s.choice([0.05, 0.10, 0.20, 0.35]))
            pick = rng_s.integers(0, N, size=8192)
            qb = q_all[pick] + rng_s.normal(0.0, sigma, size=(8192, q_all.shape[1]))
            qb = np.clip(qb, Q_MIN, Q_MAX).astype(np.float32)
            qt = torch.tensor(qb, dtype=torch.float32, device=args.device)
            in_c, dres = check_collision_full(
                robot_model, qt, cuboid_centers, cuboid_half,
                torso_cuboid_centers=torso_centers, torso_cuboid_half=torso_half,
                self_collision_mask=self_coll_mask,
                torso_ignore_sphere_indices=TORSO_IGNORE_SPHERE_INDICES,
                margin=0.0, self_collision_margin=0.0, device=args.device)
            ok = (in_c & (dres['self'] >= args.self_margin)).cpu().numpy()
            if ok.any():
                got_q.append(qb[ok]); got_src.append(source_id[pick][ok])
                n_have += int(ok.sum())
        if got_q:
            add_q = np.concatenate(got_q, axis=0)[:n_synth]
            add_src = np.concatenate(got_src, axis=0)[:n_synth]
            q_all = np.concatenate([q_all, add_q], axis=0)
            source_id = np.concatenate([source_id, add_src], axis=0)
            N = q_all.shape[0]
            print(f"  [R66] added {add_q.shape[0]} inside-obstacle configs in "
                  f"{attempts} rounds ({time.time()-t_syn:.1f}s); pool now {N}")
        else:
            print("  [R66] WARNING: synthesised none -- perturbation never collided")

    # ── Step 5: Compute EE positions + zone assignment ───────────────────────
    print("Computing EE positions for zone assignment...")
    ee_pos_all = compute_ee_positions(robot_model, q_all, device=args.device)
    ee_y = ee_pos_all[:, 1]

    cupping_mask, pouring_mask, overlap_mask = assign_zones(
        ee_y, args.y_divide, args.overlap,
        ee_x=ee_pos_all[:, 0], zone_mode=zone_mode, corner_zones=corner_zones)

    # ── Task-affinity filtering: remove cross-contaminated configs ──────────
    # In exclusive zones, only keep configs from the matching FK source.
    # In overlap zone, keep ALL configs (both tasks valid, routing handles it).
    cupping_exclusive = cupping_mask & ~pouring_mask   # Y > y_divide + overlap
    pouring_exclusive = pouring_mask & ~cupping_mask   # Y < y_divide - overlap

    # Before filtering: cross-tabulate source x zone
    cup_excl_from_cup = ((source_id == 0) & cupping_exclusive).sum()
    cup_excl_from_pour = ((source_id == 1) & cupping_exclusive).sum()
    pour_excl_from_cup = ((source_id == 0) & pouring_exclusive).sum()
    pour_excl_from_pour = ((source_id == 1) & pouring_exclusive).sum()
    overlap_from_cup = ((source_id == 0) & overlap_mask).sum()
    overlap_from_pour = ((source_id == 1) & overlap_mask).sum()

    print(f"\n  BEFORE task-affinity filtering:")
    print(f"    Cupping-exclusive zone: {cup_excl_from_cup} from cup_pkl + {cup_excl_from_pour} from pour_pkl")
    print(f"    Pouring-exclusive zone: {pour_excl_from_cup} from cup_pkl + {pour_excl_from_pour} from pour_pkl")
    print(f"    Overlap zone: {overlap_from_cup} from cup_pkl + {overlap_from_pour} from pour_pkl")

    # Contaminated configs: wrong-source in exclusive zones
    contaminated = (
        (cupping_exclusive & (source_id == 1)) |  # pouring-pkl in cupping-exclusive
        (pouring_exclusive & (source_id == 0))     # cupping-pkl in pouring-exclusive
    )
    n_contaminated = contaminated.sum()
    n_contam_cup_excl = (cupping_exclusive & (source_id == 1)).sum()
    n_contam_pour_excl = (pouring_exclusive & (source_id == 0)).sum()
    print(f"\n  Removing {n_contaminated} contaminated configs:")
    print(f"    {n_contam_cup_excl} pouring-pkl configs in cupping-exclusive zone")
    print(f"    {n_contam_pour_excl} cupping-pkl configs in pouring-exclusive zone")

    # R64: optionally KEEP a fraction of the "contaminated" configs.
    #
    # These are not mislabelled -- they are wrong-SOURCE configs, i.e. pour-oriented
    # poses that happen to sit in the cupping-exclusive zone (and vice versa).  Labels
    # are computed AFTER this filter, so a retained one gets its true, correctly SLOW
    # d_orient for the zone's task.  Deleting them removes exactly the supervision that
    # says "holding pour orientation here is expensive".
    #
    # Measured 2026-09-08: our cross failures are precisely this configuration --
    # cup-exclusive zone, wrist rotated toward pour, ~9deg over the gate at path
    # fraction 0.19-0.25 -- and the field charges only 4.66 slowness there against a
    # 26.03 target.  See [[cross-ceiling-and-detour]].
    keep = ~contaminated
    keep_frac = float(getattr(args, 'keep_contaminated', 0.0) or 0.0)
    if keep_frac > 0.0:
        rng_keep = np.random.default_rng(args.seed + 991)
        contam_idx = np.flatnonzero(contaminated)
        n_keep = int(round(keep_frac * contam_idx.size))
        if n_keep > 0:
            sel = rng_keep.choice(contam_idx, size=n_keep, replace=False)
            keep[sel] = True
        print(f"  [R64] keeping {n_keep}/{contam_idx.size} contaminated configs "
              f"(keep_contaminated={keep_frac})")
    q_all = q_all[keep]
    source_id = source_id[keep]
    ee_pos_all = ee_pos_all[keep]
    N = q_all.shape[0]
    ee_y = ee_pos_all[:, 1]
    print(f"  {N} configs remaining after task-affinity filter")

    # Recompute zones after filtering
    cupping_mask, pouring_mask, overlap_mask = assign_zones(
        ee_y, args.y_divide, args.overlap,
        ee_x=ee_pos_all[:, 0], zone_mode=zone_mode, corner_zones=corner_zones)
    cupping_indices = np.where(cupping_mask)[0]
    pouring_indices = np.where(pouring_mask)[0]
    overlap_indices = np.where(overlap_mask)[0]

    n_cup = len(cupping_indices)
    n_pour = len(pouring_indices)
    n_overlap = len(overlap_indices)
    n_cup_only = n_cup - n_overlap
    n_pour_only = n_pour - n_overlap

    print(f"\n  Zone assignment AFTER filtering (zone_mode={zone_mode}, y_divide={args.y_divide}, overlap={args.overlap}):")
    if zone_mode == "corner_table":
        print(f"    Cupping zone (corner cup_region + overlap): {n_cup} configs")
        print(f"    Pouring zone (corner pour_region + overlap): {n_pour} configs")
    else:
        print(f"    Cupping zone (Y >= {args.y_divide - args.overlap:.2f}): {n_cup} configs")
        print(f"    Pouring zone (Y <= {args.y_divide + args.overlap:.2f}): {n_pour} configs")
    print(f"    Overlap zone: {n_overlap} configs")
    print(f"    Cupping-only: {n_cup_only}, Pouring-only: {n_pour_only}")
    print(f"    EE X range: [{ee_pos_all[:, 0].min():.3f}, {ee_pos_all[:, 0].max():.3f}]")
    print(f"    EE Y range: [{ee_y.min():.3f}, {ee_y.max():.3f}]")

    # Cross-tabulate source x zone (after filtering)
    cup_from_cup = ((source_id == 0) & cupping_mask).sum()
    cup_from_pour = ((source_id == 1) & cupping_mask).sum()
    pour_from_cup = ((source_id == 0) & pouring_mask).sum()
    pour_from_pour = ((source_id == 1) & pouring_mask).sum()
    print(f"\n  Source x Zone cross-tabulation (AFTER filtering):")
    print(f"    Cupping zone: {cup_from_cup} from cup_pkl + {cup_from_pour} from pour_pkl")
    print(f"    Pouring zone: {pour_from_cup} from cup_pkl + {pour_from_pour} from pour_pkl")

    # Verify: exclusive zones should have ZERO cross-source configs -- unless R64
    # --keep_contaminated deliberately retained some, in which case report instead.
    cupping_exclusive_post = cupping_mask & ~pouring_mask
    pouring_exclusive_post = pouring_mask & ~cupping_mask
    n_cross_cup = int(((source_id == 1) & cupping_exclusive_post).sum())
    n_cross_pour = int(((source_id == 0) & pouring_exclusive_post).sum())
    if float(getattr(args, 'keep_contaminated', 0.0) or 0.0) > 0.0:
        print(f"  [R64] retained cross-source configs in exclusive zones: "
              f"{n_cross_cup} pour-in-cup, {n_cross_pour} cup-in-pour "
              f"(intentional -- they carry the slow labels)")
    else:
        assert n_cross_cup == 0, \
            "BUG: pouring-pkl configs still in cupping-exclusive zone!"
        assert n_cross_pour == 0, \
            "BUG: cupping-pkl configs still in pouring-exclusive zone!"
        print(f"  ✓ No cross-contamination in exclusive zones")

    if n_cup == 0 or n_pour == 0:
        print("ERROR: One zone has 0 configs. Check y_divide and overlap settings.")
        return

    # ── Step 6: Compute physical distances ───────────────────────────────────
    include_self = not args.no_self_in_speed
    mask_for_speed = self_coll_mask if include_self else None
    print(f"\nComputing physical distances (include_self={include_self})...")
    t0 = time.time()
    workspace_bounds = None
    if args.workspace_height is not None:
        from build_witness_certified_eval_suite import scene_sampling_workspace
        workspace_bounds = scene_sampling_workspace(
            args.scene, args.workspace_height)
        print(f"  Workspace containment ON: x={workspace_bounds[0]} "
              f"y={workspace_bounds[1]} z={workspace_bounds[2]}")

    d_obs, d_self, d_ws = compute_physical_distance(
        robot_model, q_all, cuboid_centers, cuboid_half,
        torso_centers=torso_centers, torso_half=torso_half,
        self_coll_mask=mask_for_speed, workspace_bounds=workspace_bounds,
        device=args.device)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  d_obs stats: min={d_obs.min():.4f}, max={d_obs.max():.4f}, mean={d_obs.mean():.4f}")

    d_phys = d_obs.copy()
    if include_self:
        self_speed_margin = args.self_speed_margin or args.margin
        self_speed_scale = args.margin / max(self_speed_margin, 1e-8)
        d_self_speed = d_self * self_speed_scale
        print(f"  d_self stats: min={d_self.min():.4f}, max={d_self.max():.4f}, mean={d_self.mean():.4f}")
        if self_speed_scale != 1.0:
            print(f"  Self speed scale: margin={args.margin:.4f} / "
                  f"self_speed_margin={self_speed_margin:.4f} = {self_speed_scale:.3f}")
        d_phys = np.minimum(d_phys, d_self_speed)
    else:
        self_speed_margin = None
        self_speed_scale = 1.0

    if workspace_bounds is not None:
        ws_speed_margin = args.workspace_speed_margin or args.margin
        ws_speed_scale = args.margin / max(ws_speed_margin, 1e-8)
        print(f"  d_workspace stats: min={d_ws.min():.4f}, max={d_ws.max():.4f}, "
              f"mean={d_ws.mean():.4f}, outside={(d_ws < 0).sum()}")
        if ws_speed_scale != 1.0:
            print(f"  Workspace speed scale: margin={args.margin:.4f} / "
                  f"workspace_speed_margin={ws_speed_margin:.4f} = {ws_speed_scale:.3f}")
        d_phys = np.minimum(d_phys, d_ws * ws_speed_scale)
    else:
        ws_speed_margin = None
        ws_speed_scale = 1.0

    # ── Step 7: Compute orientation distances (both tasks) ───────────────────
    print(f"\nComputing cupping orientation distance (tilt_max={args.cup_tilt_deg}deg)...")
    d_orient_cup = compute_orient_distance(
        robot_model, q_all, "cupping", args.margin,
        args.cup_tilt_deg, args.cup_tilt_soft, device=args.device,
        morph=args.morph_orient, morph_mode=args.morph_mode,
        zone_mode=getattr(args, "zone_mode_resolved", "y"),
        corner_overlap_deg=args.corner_overlap_deg,
        y_divide=args.y_divide, overlap=args.overlap,
        task_axes=args.task_axes_resolved)

    print(f"  d_orient_cup stats: min={d_orient_cup.min():.4f}, max={d_orient_cup.max():.4f}")
    print(f"  Violating cupping: {(d_orient_cup < 0).sum()}/{N} "
          f"({100*(d_orient_cup < 0).sum()/N:.1f}%)")

    # Per-source breakdown
    cup_viol_from_cup = ((d_orient_cup < 0) & (source_id == 0)).sum()
    cup_viol_from_pour = ((d_orient_cup < 0) & (source_id == 1)).sum()
    cup_ok_from_cup = ((d_orient_cup >= 0) & (source_id == 0)).sum()
    cup_ok_from_pour = ((d_orient_cup >= 0) & (source_id == 1)).sum()
    print(f"    From cup_pkl: {cup_ok_from_cup} OK, {cup_viol_from_cup} violating")
    print(f"    From pour_pkl: {cup_ok_from_pour} OK, {cup_viol_from_pour} violating")

    print(f"\nComputing pouring orientation distance (tilt_max={args.pour_tilt_deg}deg)...")
    d_orient_pour = compute_orient_distance(
        robot_model, q_all, "pouring", args.margin,
        args.pour_tilt_deg, args.pour_tilt_soft, device=args.device,
        morph=args.morph_orient, morph_mode=args.morph_mode,
        zone_mode=getattr(args, "zone_mode_resolved", "y"),
        corner_overlap_deg=args.corner_overlap_deg,
        y_divide=args.y_divide, overlap=args.overlap,
        rzy_safe=args.pour_rzy_safe, rzy_soft=args.pour_rzy_soft,
        task_axes=args.task_axes_resolved)

    print(f"  d_orient_pour stats: min={d_orient_pour.min():.4f}, max={d_orient_pour.max():.4f}")
    print(f"  Violating pouring: {(d_orient_pour < 0).sum()}/{N} "
          f"({100*(d_orient_pour < 0).sum()/N:.1f}%)")

    pour_viol_from_cup = ((d_orient_pour < 0) & (source_id == 0)).sum()
    pour_viol_from_pour = ((d_orient_pour < 0) & (source_id == 1)).sum()
    pour_ok_from_cup = ((d_orient_pour >= 0) & (source_id == 0)).sum()
    pour_ok_from_pour = ((d_orient_pour >= 0) & (source_id == 1)).sum()
    print(f"    From cup_pkl: {pour_ok_from_cup} OK, {pour_viol_from_cup} violating")
    print(f"    From pour_pkl: {pour_ok_from_pour} OK, {pour_viol_from_pour} violating")

    # ── Step 7b: Dual-constraint overlap filtering ──────────────────────────
    if args.filter_overlap_dual:
        thr = args.dual_threshold
        print(f"\n── Dual-constraint overlap filtering (threshold={thr:.3f}) ──")
        # Identify overlap zone configs
        n_overlap_before = overlap_mask.sum()
        dual_ok = (d_orient_cup > thr) & (d_orient_pour > thr)
        # Only filter overlap zone configs; keep all exclusive-zone configs
        dual_reject = overlap_mask & ~dual_ok
        n_dual_reject = dual_reject.sum()
        n_dual_keep = (overlap_mask & dual_ok).sum()
        print(f"  Overlap zone: {n_overlap_before} configs total")
        print(f"  Both d_orient > {thr}: {n_dual_keep} ({100*n_dual_keep/max(n_overlap_before,1):.1f}%)")
        print(f"  Rejected (violates one or both): {n_dual_reject}")

        # Per-source breakdown in overlap
        for src_name, src_val in [("cup_pkl", 0), ("pour_pkl", 1)]:
            in_overlap_src = overlap_mask & (source_id == src_val)
            dual_ok_src = in_overlap_src & dual_ok
            dual_rej_src = in_overlap_src & ~dual_ok
            print(f"    {src_name}: {in_overlap_src.sum()} overlap, "
                  f"{dual_ok_src.sum()} keep ({100*dual_ok_src.sum()/max(in_overlap_src.sum(),1):.1f}%), "
                  f"{dual_rej_src.sum()} reject")

        # Apply filter
        keep_dual = ~dual_reject
        q_all = q_all[keep_dual]
        source_id = source_id[keep_dual]
        ee_pos_all = ee_pos_all[keep_dual]
        d_obs = d_obs[keep_dual]
        d_self = d_self[keep_dual]
        d_phys = d_phys[keep_dual]
        d_orient_cup = d_orient_cup[keep_dual]
        d_orient_pour = d_orient_pour[keep_dual]
        N = q_all.shape[0]
        ee_y = ee_pos_all[:, 1]
        print(f"  {N} configs remaining after dual-constraint filter")

        # Recompute zones
        cupping_mask, pouring_mask, overlap_mask = assign_zones(
            ee_y, args.y_divide, args.overlap,
            ee_x=ee_pos_all[:, 0], zone_mode=zone_mode, corner_zones=corner_zones)
        cupping_indices = np.where(cupping_mask)[0]
        pouring_indices = np.where(pouring_mask)[0]
        overlap_indices = np.where(overlap_mask)[0]
        n_cup = len(cupping_indices)
        n_pour = len(pouring_indices)
        n_overlap = len(overlap_indices)
        n_cup_only = n_cup - n_overlap
        n_pour_only = n_pour - n_overlap
        print(f"  Zones after filter: cup={n_cup}, pour={n_pour}, overlap={n_overlap}")
        print(f"  Cupping-only: {n_cup_only}, Pouring-only: {n_pour_only}")

        # Speed stats preview for filtered overlap
        if n_overlap > 0:
            ov_speed_cup = np.clip(np.minimum(d_phys[overlap_mask], d_orient_cup[overlap_mask]),
                                   offset, args.margin) / args.margin
            ov_speed_pour = np.clip(np.minimum(d_phys[overlap_mask], d_orient_pour[overlap_mask]),
                                    offset, args.margin) / args.margin
            print(f"  Filtered overlap speed_cup: mean={ov_speed_cup.mean():.3f}, "
                  f"min={ov_speed_cup.min():.3f}")
            print(f"  Filtered overlap speed_pour: mean={ov_speed_pour.mean():.3f}, "
                  f"min={ov_speed_pour.min():.3f}")

    # ── Step 8: Compute speeds (both tasks) ──────────────────────────────────
    alpha = args.alpha
    blend_mode = args.blend_mode
    cup_blend = args.cup_blend_mode or blend_mode
    pour_blend = args.pour_blend_mode or blend_mode
    cup_alpha = args.cup_alpha if args.cup_alpha is not None else alpha
    pour_alpha = args.pour_alpha if args.pour_alpha is not None else alpha
    print(f"\nComputing speeds:")
    print(f"  Cup: blend={cup_blend}, alpha={cup_alpha}, margin={args.margin}")
    print(f"  Pour: blend={pour_blend}, alpha={pour_alpha}, margin={args.margin}")

    speed_cupping = compute_speed(d_phys, d_orient_cup, cup_alpha, args.margin, offset, blend_mode=cup_blend)
    speed_pouring = compute_speed(d_phys, d_orient_pour, pour_alpha, args.margin, offset, blend_mode=pour_blend)

    print(f"  Cupping speed stats: min={speed_cupping.min():.4f}, max={speed_cupping.max():.4f}")
    print(f"  Pouring speed stats: min={speed_pouring.min():.4f}, max={speed_pouring.max():.4f}")

    # Speed distribution analysis per zone
    sp_floor = 0.11  # floor threshold
    for zone_name, mask in [("cupping", cupping_mask), ("pouring", pouring_mask)]:
        cup_floor_pct = (speed_cupping[mask] < sp_floor).mean() * 100
        pour_floor_pct = (speed_pouring[mask] < sp_floor).mean() * 100
        cup_free_pct = (speed_cupping[mask] >= 1.0).mean() * 100
        pour_free_pct = (speed_pouring[mask] >= 1.0).mean() * 100
        print(f"\n  {zone_name.capitalize()} zone ({mask.sum()} configs):")
        print(f"    Cupping speed: {cup_floor_pct:.1f}% at floor, {cup_free_pct:.1f}% free")
        print(f"    Pouring speed: {pour_floor_pct:.1f}% at floor, {pour_free_pct:.1f}% free")

    n_cup_near = (speed_cupping < 1.0).sum()
    n_pour_near = (speed_pouring < 1.0).sum()
    print(f"\n  Overall cupping near-obs (speed<1): {n_cup_near}/{N} ({100*n_cup_near/N:.1f}%)")
    print(f"  Overall pouring near-obs (speed<1): {n_pour_near}/{N} ({100*n_pour_near/N:.1f}%)")

    # ── Step 9: Compute gradients (both tasks) ───────────────────────────────
    print(f"\nComputing cupping gradients (blend_mode={blend_mode})...")
    t0 = time.time()
    normal_cupping = compute_gradient_task(
        robot_model, q_all, cuboid_centers, cuboid_half,
        torso_centers, torso_half, mask_for_speed,
        "cupping", cup_alpha, args.margin, args.cup_tilt_deg, args.cup_tilt_soft,
        device=args.device, blend_mode=cup_blend,
        self_speed_scale=self_speed_scale,
        morph=args.morph_orient, morph_mode=args.morph_mode,
        zone_mode=getattr(args, "zone_mode_resolved", "y"),
        corner_overlap_deg=args.corner_overlap_deg,
        y_divide=args.y_divide, overlap=args.overlap,
        task_axes=args.task_axes_resolved)
    print(f"  Done in {time.time()-t0:.1f}s")

    print(f"Computing pouring gradients (blend={pour_blend}, alpha={pour_alpha})...")
    t0 = time.time()
    normal_pouring = compute_gradient_task(
        robot_model, q_all, cuboid_centers, cuboid_half,
        torso_centers, torso_half, mask_for_speed,
        "pouring", pour_alpha, args.margin, args.pour_tilt_deg, args.pour_tilt_soft,
        device=args.device, blend_mode=pour_blend,
        self_speed_scale=self_speed_scale,
        morph=args.morph_orient, morph_mode=args.morph_mode,
        zone_mode=getattr(args, "zone_mode_resolved", "y"),
        corner_overlap_deg=args.corner_overlap_deg,
        y_divide=args.y_divide, overlap=args.overlap,
        rzy_safe=args.pour_rzy_safe, rzy_soft=args.pour_rzy_soft,
        task_axes=args.task_axes_resolved)
    print(f"  Done in {time.time()-t0:.1f}s")

    # NaN filter
    valid = (~np.any(np.isnan(normal_cupping), axis=1) &
             ~np.any(np.isnan(normal_pouring), axis=1))
    n_nan = (~valid).sum()
    if n_nan > 0:
        print(f"  WARNING: {n_nan} NaN normals, filtering out")
        q_all = q_all[valid]
        ee_pos_all = ee_pos_all[valid]
        d_obs = d_obs[valid]
        d_self = d_self[valid]
        d_phys = d_phys[valid]
        d_orient_cup = d_orient_cup[valid]
        d_orient_pour = d_orient_pour[valid]
        speed_cupping = speed_cupping[valid]
        speed_pouring = speed_pouring[valid]
        normal_cupping = normal_cupping[valid]
        normal_pouring = normal_pouring[valid]
        source_id = source_id[valid]
        N = q_all.shape[0]
        ee_y = ee_pos_all[:, 1]
        cupping_mask, pouring_mask, overlap_mask = assign_zones(
            ee_y, args.y_divide, args.overlap,
            ee_x=ee_pos_all[:, 0], zone_mode=zone_mode, corner_zones=corner_zones)
        cupping_indices = np.where(cupping_mask)[0]
        pouring_indices = np.where(pouring_mask)[0]
        n_cup = cupping_mask.sum()
        n_pour = pouring_mask.sum()
        n_overlap = overlap_mask.sum()
        n_cup_only = n_cup - n_overlap
        n_pour_only = n_pour - n_overlap
        print(f"  {N} configs remaining after NaN filter")

    # ── Step 9b (R58U): clean per-config tilt-descent target ─────────────────
    # normalized -grad_q tilt_FK (zone-appropriate), for supervising the learned
    # orientation potential's gradient. Offline FK only; inference never calls FK.
    tilt_grad = None
    if getattr(args, "compute_tilt_grad", False):
        print("\nComputing clean FK tilt-descent gradient (R58U target)...")
        tilt_grad = compute_tilt_gradient(robot_model, q_all, device=args.device,
                                          eps=args.tilt_grad_eps,
                                          zone_mode=zone_mode,
                                          y_divide=args.y_divide,
                                          corner_zones=corner_zones,
                                          task_axes=args.task_axes_resolved)
        print(f"  tilt_grad shape {tilt_grad.shape}")

    # ── Step 10: Sample pairs ────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    q_norm_for_sampling = (q_all - q_center) / q_range

    if args.source_aware:
        print(f"\nSampling {args.n_pairs} SOURCE-AWARE pairs (intra_ratio={args.intra_ratio}, "
              f"source_mix={args.source_mix:.0%})...")
        start_idx, goal_idx, pair_type, start_zone, goal_zone = sample_pairs_source_aware(
            cupping_mask, pouring_mask, source_id, args.n_pairs, args.intra_ratio, rng,
            source_mix=args.source_mix,
            ee_y=ee_y,
            y_divide=args.y_divide,
            ee_pos=ee_pos_all,
            zone_mode=zone_mode,
            corner_zones=corner_zones,
            q_norm=q_norm_for_sampling,
            boundary_cross_frac=args.boundary_cross_frac,
            corner_cross_frac=args.corner_cross_frac,
            boundary_cross_band=args.boundary_cross_band,
            local_cross_k=args.local_cross_k,
            min_cross_dist=args.min_cross_dist,
            max_cross_dist=args.max_cross_dist,
            cross_dist_oversample=args.cross_dist_oversample)
    else:
        print(f"\nSampling {args.n_pairs} pairs (intra_ratio={args.intra_ratio})...")
        start_idx, goal_idx, pair_type, start_zone, goal_zone = sample_pairs(
            cupping_indices, pouring_indices, args.n_pairs, args.intra_ratio, rng)
    n_actual = len(start_idx)

    n_ic = (pair_type == PAIR_INTRA_CUPPING).sum()
    n_ip = (pair_type == PAIR_INTRA_POURING).sum()
    n_cr = (pair_type == PAIR_CROSS).sum()
    print(f"  Sampled {n_actual} pairs:")
    print(f"    Intra-cupping: {n_ic} ({100*n_ic/n_actual:.1f}%)")
    print(f"    Intra-pouring: {n_ip} ({100*n_ip/n_actual:.1f}%)")
    print(f"    Cross-zone:    {n_cr} ({100*n_cr/n_actual:.1f}%)")

    # ── Optional: filter cross pairs to dual-constraint endpoints ────────
    if getattr(args, 'cross_dual_filter', False):
        print(f"\n  [cross_dual_filter] Filtering cross pairs to dual-constraint endpoints...")
        is_cross = pair_type == PAIR_CROSS
        if is_cross.any():
            cross_idx = np.where(is_cross)[0]
            s_cup_ok = d_orient_cup[start_idx[cross_idx]] > 0
            s_pour_ok = d_orient_pour[start_idx[cross_idx]] > 0
            g_cup_ok = d_orient_cup[goal_idx[cross_idx]] > 0
            g_pour_ok = d_orient_pour[goal_idx[cross_idx]] > 0
            both_ok = s_cup_ok & s_pour_ok & g_cup_ok & g_pour_ok
            n_before = len(cross_idx)
            n_kept = both_ok.sum()
            print(f"    Cross pairs: {n_before} -> {n_kept} ({100*n_kept/n_before:.1f}% kept)")
            if n_kept < n_before:
                remove_mask = np.zeros(n_actual, dtype=bool)
                remove_mask[cross_idx[~both_ok]] = True
                keep = ~remove_mask
                start_idx = start_idx[keep]
                goal_idx = goal_idx[keep]
                pair_type = pair_type[keep]
                start_zone = start_zone[keep]
                goal_zone = goal_zone[keep]
                n_actual = len(start_idx)
                n_cr = (pair_type == PAIR_CROSS).sum()
                print(f"    After filter: {n_actual} total pairs, {n_cr} cross")

    cross_segment_stats = None
    if args.cross_segment_filter:
        print(f"\n  [cross_segment_filter] Filtering cross pairs by internal feasibility...")
        keep, cross_segment_stats = filter_cross_pairs_by_segment(
            robot_model, q_all, start_idx, goal_idx, pair_type,
            cuboid_centers, cuboid_half, torso_centers, torso_half,
            self_coll_mask, args)
        n_before = len(start_idx)
        n_cross_before = int((pair_type == PAIR_CROSS).sum())
        n_removed = int((~keep).sum())
        if n_removed > 0:
            start_idx = start_idx[keep]
            goal_idx = goal_idx[keep]
            pair_type = pair_type[keep]
            start_zone = start_zone[keep]
            goal_zone = goal_zone[keep]
        n_actual = len(start_idx)
        n_ic, n_ip, n_cr = _pair_counts(pair_type)
        kept_pct = 100.0 * n_cr / max(n_cross_before, 1)
        print(f"    Total pairs: {n_before} -> {n_actual} (removed {n_removed})")
        print(f"    Cross pairs: {n_cross_before} -> {n_cr} ({kept_pct:.1f}% kept)")
        print(f"    Segment rejection stats: collision={cross_segment_stats['rejected_collision']}, "
              f"tilt={cross_segment_stats['rejected_tilt']}")

        if args.cross_segment_refill_cross and 0 < n_cr < n_cross_before:
            cross_rows = np.where(pair_type == PAIR_CROSS)[0]
            n_extra = n_cross_before - n_cr
            extra_rows = rng.choice(cross_rows, size=n_extra, replace=True)
            start_idx = np.concatenate([start_idx, start_idx[extra_rows]])
            goal_idx = np.concatenate([goal_idx, goal_idx[extra_rows]])
            pair_type = np.concatenate([pair_type, pair_type[extra_rows]])
            start_zone = np.concatenate([start_zone, start_zone[extra_rows]])
            goal_zone = np.concatenate([goal_zone, goal_zone[extra_rows]])
            perm = rng.permutation(len(start_idx))
            start_idx = start_idx[perm]
            goal_idx = goal_idx[perm]
            pair_type = pair_type[perm]
            start_zone = start_zone[perm]
            goal_zone = goal_zone[perm]
            n_actual = len(start_idx)
            n_ic, n_ip, n_cr = _pair_counts(pair_type)
            cross_segment_stats["refilled_cross_to"] = int(n_cr)
            cross_segment_stats["refill_duplicates_added"] = int(n_extra)
            print(f"    Refilled clean cross pairs: +{n_extra} duplicates -> {n_cr} cross")

    n_ic, n_ip, n_cr = _pair_counts(pair_type)

    # ── Step 10b (R58N): inject high-tilt corridor (waypoint -> goal) cross pairs ──
    # PROVEN root cause: the high-tilt mid-corridor is never sampled, so the trained
    # field is tilt-blind there (~14% slowdown vs the ~85% the label gives). We append
    # interpolation waypoints as standalone configs (with their correctly-low labels)
    # and (waypoint -> original goal) cross pairs, so the field learns a speed valley
    # at high tilt and its geodesic routes around the wrist-flip.
    if args.corridor_inject_frac > 0:
        print(f"\n  [R58N corridor inject] frac={args.corridor_inject_frac} "
              f"waypoints={args.corridor_waypoints} min_tilt={args.corridor_min_tilt}deg")
        cw = build_corridor_waypoints(
            robot_model, q_all, start_idx, goal_idx, pair_type, goal_zone,
            cuboid_centers, cuboid_half, torso_centers, torso_half,
            self_coll_mask, mask_for_speed, self_speed_scale,
            args, offset, cup_blend, pour_blend, cup_alpha, pour_alpha, args.y_divide)
        if cw is not None:
            base = q_all.shape[0]                       # pool offset for new configs
            n_new = cw["wp_rad"].shape[0]
            # append new configs to every per-config pool array
            q_all = np.concatenate([q_all, cw["wp_rad"]], axis=0)
            q_norm_for_sampling = np.concatenate(
                [q_norm_for_sampling, (cw["wp_rad"] - q_center) / q_range], axis=0)
            ee_pos_all = np.concatenate([ee_pos_all, cw["wp_ee"]], axis=0)
            d_obs = np.concatenate([d_obs, cw["d_obs"]], axis=0)
            d_self = np.concatenate([d_self, cw["d_self"]], axis=0)
            d_orient_cup = np.concatenate([d_orient_cup, cw["d_orient_cup"]], axis=0)
            d_orient_pour = np.concatenate([d_orient_pour, cw["d_orient_pour"]], axis=0)
            speed_cupping = np.concatenate([speed_cupping, cw["speed_cup"]], axis=0)
            speed_pouring = np.concatenate([speed_pouring, cw["speed_pour"]], axis=0)
            normal_cupping = np.concatenate([normal_cupping, cw["normal_cup"]], axis=0)
            normal_pouring = np.concatenate([normal_pouring, cw["normal_pour"]], axis=0)
            if tilt_grad is not None:
                tilt_grad = np.concatenate([tilt_grad, cw["tilt_grad"]], axis=0)
            if 'source_id' in dir():
                source_id = np.concatenate(
                    [source_id, np.full(n_new, -1, dtype=source_id.dtype)], axis=0)
            # new pairs: waypoint(start) -> original goal, labeled CROSS
            wp_start = np.arange(base, base + n_new, dtype=start_idx.dtype)
            start_idx = np.concatenate([start_idx, wp_start])
            goal_idx = np.concatenate([goal_idx, cw["wp_goal_idx"].astype(goal_idx.dtype)])
            pair_type = np.concatenate(
                [pair_type, np.full(n_new, PAIR_CROSS, dtype=pair_type.dtype)])
            start_zone = np.concatenate(
                [start_zone, cw["wp_zone"].astype(start_zone.dtype)])
            goal_zone = np.concatenate(
                [goal_zone, cw["wp_goal_zone"].astype(goal_zone.dtype)])
            # reshuffle so injected pairs are not all at the tail
            perm = rng.permutation(len(start_idx))
            start_idx = start_idx[perm]; goal_idx = goal_idx[perm]
            pair_type = pair_type[perm]
            start_zone = start_zone[perm]; goal_zone = goal_zone[perm]
            n_actual = len(start_idx)
            n_ic, n_ip, n_cr = _pair_counts(pair_type)
            print(f"  [R58N] pool {base} -> {q_all.shape[0]} configs (+{n_new}); "
                  f"pairs -> {n_actual} ({n_cr} cross)")

    # ── Step 11: Assign speed labels ─────────────────────────────────────────
    speed_cup_pairs = np.zeros((n_actual, 2), dtype=np.float32)
    speed_pour_pairs = np.zeros((n_actual, 2), dtype=np.float32)
    speed_cup_pairs[:, 0] = speed_cupping[start_idx]
    speed_cup_pairs[:, 1] = speed_cupping[goal_idx]
    speed_pour_pairs[:, 0] = speed_pouring[start_idx]
    speed_pour_pairs[:, 1] = speed_pouring[goal_idx]

    # Speed distribution in sampled pairs
    print(f"\n  Speed distributions in sampled pairs:")
    for ptype_name, ptype_val in [("intra-cup", 0), ("intra-pour", 1), ("cross", 2)]:
        pmask = pair_type == ptype_val
        if not pmask.any():
            continue
        cup_floor = (speed_cup_pairs[pmask] < sp_floor).mean() * 100
        pour_floor = (speed_pour_pairs[pmask] < sp_floor).mean() * 100
        cup_mean = speed_cup_pairs[pmask].mean()
        pour_mean = speed_pour_pairs[pmask].mean()
        print(f"    {ptype_name}: cupping {cup_floor:.1f}% floor (mean={cup_mean:.3f}), "
              f"pouring {pour_floor:.1f}% floor (mean={pour_mean:.3f})")

    # Source-aware breakdown for intra-cupping pairs
    if args.source_aware:
        cup_pmask = pair_type == PAIR_INTRA_CUPPING
        if cup_pmask.any():
            # All endpoints should be from cupping pkl
            s_src = source_id[start_idx[cup_pmask]]
            g_src = source_id[goal_idx[cup_pmask]]
            print(f"\n  Source verification (intra-cupping):")
            print(f"    Start from cup_pkl: {(s_src==0).sum()}/{cup_pmask.sum()} "
                  f"({100*(s_src==0).mean():.1f}%)")
            print(f"    Goal from cup_pkl: {(g_src==0).sum()}/{cup_pmask.sum()} "
                  f"({100*(g_src==0).mean():.1f}%)")
        pour_pmask = pair_type == PAIR_INTRA_POURING
        if pour_pmask.any():
            s_src = source_id[start_idx[pour_pmask]]
            g_src = source_id[goal_idx[pour_pmask]]
            print(f"  Source verification (intra-pouring):")
            print(f"    Start from pour_pkl: {(s_src==1).sum()}/{pour_pmask.sum()} "
                  f"({100*(s_src==1).mean():.1f}%)")
            print(f"    Goal from pour_pkl: {(g_src==1).sum()}/{pour_pmask.sum()} "
                  f"({100*(g_src==1).mean():.1f}%)")

    # ── Step 12: Build output arrays ─────────────────────────────────────────
    q_norm_all = q_norm_for_sampling

    sampled_points = np.concatenate([q_norm_all[start_idx], q_norm_all[goal_idx]], axis=1)
    normal_cup_pairs = np.concatenate([normal_cupping[start_idx], normal_cupping[goal_idx]], axis=1)
    normal_pour_pairs = np.concatenate([normal_pouring[start_idx], normal_pouring[goal_idx]], axis=1)
    # R58U: per-pair clean tilt-descent gradient [start | goal]
    tilt_grad_pairs = None
    if tilt_grad is not None:
        tilt_grad_pairs = np.concatenate([tilt_grad[start_idx], tilt_grad[goal_idx]], axis=1)
    ee_positions = np.concatenate([ee_pos_all[start_idx], ee_pos_all[goal_idx]], axis=1)
    d_obs_pairs = np.stack([d_obs[start_idx], d_obs[goal_idx]], axis=1)
    d_self_pairs = np.stack([d_self[start_idx], d_self[goal_idx]], axis=1)
    d_orient_cup_pairs = np.stack([d_orient_cup[start_idx], d_orient_cup[goal_idx]], axis=1)
    d_orient_pour_pairs = np.stack([d_orient_pour[start_idx], d_orient_pour[goal_idx]], axis=1)
    zone_id = np.stack([start_zone, goal_zone], axis=1)

    # Final NaN filter on assembled pairs
    valid_pairs = (~np.any(np.isnan(normal_cup_pairs), axis=1) &
                   ~np.any(np.isnan(normal_pour_pairs), axis=1))
    n_nan_pairs = (~valid_pairs).sum()
    if n_nan_pairs > 0:
        print(f"  Filtering {n_nan_pairs} NaN pair rows")
        sampled_points = sampled_points[valid_pairs]
        speed_cup_pairs = speed_cup_pairs[valid_pairs]
        speed_pour_pairs = speed_pour_pairs[valid_pairs]
        normal_cup_pairs = normal_cup_pairs[valid_pairs]
        normal_pour_pairs = normal_pour_pairs[valid_pairs]
        if tilt_grad_pairs is not None:
            tilt_grad_pairs = tilt_grad_pairs[valid_pairs]
        ee_positions = ee_positions[valid_pairs]
        d_obs_pairs = d_obs_pairs[valid_pairs]
        d_self_pairs = d_self_pairs[valid_pairs]
        d_orient_cup_pairs = d_orient_cup_pairs[valid_pairs]
        d_orient_pour_pairs = d_orient_pour_pairs[valid_pairs]
        zone_id = zone_id[valid_pairs]
        pair_type = pair_type[valid_pairs]

    M = sampled_points.shape[0]
    print(f"\n  Final: {M} pairs")

    # ── Step 13: Save ────────────────────────────────────────────────────────
    print(f"\nSaving to {out_dir}/...")
    np.save(out_dir / "sampled_points.npy", sampled_points.astype(np.float32))
    np.save(out_dir / "speed_cupping.npy", speed_cup_pairs.astype(np.float32))
    np.save(out_dir / "speed_pouring.npy", speed_pour_pairs.astype(np.float32))
    np.save(out_dir / "normal_cupping.npy", normal_cup_pairs.astype(np.float32))
    np.save(out_dir / "normal_pouring.npy", normal_pour_pairs.astype(np.float32))
    if tilt_grad_pairs is not None:
        np.save(out_dir / "tilt_grad.npy", tilt_grad_pairs.astype(np.float32))
        print(f"  tilt_grad.npy:       {tilt_grad_pairs.shape}  (R58U clean tilt-descent target)")
    np.save(out_dir / "ee_positions.npy", ee_positions.astype(np.float32))
    np.save(out_dir / "zone_id.npy", zone_id.astype(np.int8))
    np.save(out_dir / "pair_type.npy", pair_type.astype(np.int8))
    np.save(out_dir / "d_obstacle.npy", d_obs_pairs.astype(np.float32))
    np.save(out_dir / "d_self.npy", d_self_pairs.astype(np.float32))
    np.save(out_dir / "d_orient_cupping.npy", d_orient_cup_pairs.astype(np.float32))
    np.save(out_dir / "d_orient_pouring.npy", d_orient_pour_pairs.astype(np.float32))

    # Pipeline config
    config = {
        "timestamp": datetime.now().isoformat(),
        "version": (
            "v5_cross_segment_filter" if args.cross_segment_filter else
            ("v4_dual_constraint" if args.filter_overlap_dual else
             ("v3_source_aware" if args.source_aware else "v2b_decontaminated"))
        ),
        "filter_overlap_dual": args.filter_overlap_dual,
        "dual_threshold": args.dual_threshold if args.filter_overlap_dual else None,
        "pkl_cupping": str(args.pkl_cupping),
        "pkl_pouring": str(args.pkl_pouring),
        "scene": args.scene,
        "task_axes": task_axes,
        "task_axes_json": task_axes_json(task_axes),
        "n_pairs_requested": args.n_pairs,
        "n_pairs_actual": M,
        "n_configs_raw_cupping": N_cup_raw,
        "n_configs_raw_pouring": N_pour_raw,
        "n_configs_total_raw": N_raw,
        "n_configs_after_collision_filter": int(N + n_contaminated),
        "n_contaminated_removed": int(n_contaminated),
        "n_contam_pour_in_cup_excl": int(n_contam_cup_excl),
        "n_contam_cup_in_pour_excl": int(n_contam_pour_excl),
        "keep_contaminated": float(getattr(args, "keep_contaminated", 0.0) or 0.0),
        "keep_colliding": float(getattr(args, "keep_colliding", 0.0) or 0.0),
        "n_synth_colliding": int(getattr(args, "n_synth_colliding", 0) or 0),
        "n_configs_after_filter": N,
        "y_divide": args.y_divide,
        "overlap": args.overlap,
        "zone_mode": zone_mode,
        "corner_zones": corner_zones,
        "intra_ratio": args.intra_ratio,
        "margin": args.margin,
        "offset": offset,
        "alpha": args.alpha,
        "blend_mode": args.blend_mode,
        "self_margin": args.self_margin,
        "self_speed_margin": self_speed_margin,
        "self_speed_scale": self_speed_scale,
        "workspace_height": args.workspace_height,
        "workspace_speed_margin": ws_speed_margin,
        "workspace_speed_scale": ws_speed_scale,
        "no_self_in_speed": args.no_self_in_speed,
        "source_aware": args.source_aware,
        "source_mix": args.source_mix if args.source_aware else None,
        "boundary_cross_frac": args.boundary_cross_frac if args.source_aware else None,
        # R67: record the corner-sampling setting -- it was absent, so a
        # corner-sampled dataset was indistinguishable from a baseline one.
        "corner_cross_frac": args.corner_cross_frac if args.source_aware else None,
        "boundary_cross_band": args.boundary_cross_band if args.source_aware else None,
        "local_cross_k": args.local_cross_k if args.source_aware else None,
        "cross_segment_filter": args.cross_segment_filter,
        "cross_segment_refill_cross": args.cross_segment_refill_cross,
        "cross_segment_stats": cross_segment_stats,
        "signed_orient": bool(args.signed_orient),
        "tilt_band": float(args.tilt_band),
        "graded_orient": bool(args.graded_orient),
        "tilt_grade": float(args.tilt_grade),
        "morph_orient": args.morph_orient,
        "morph_mode": args.morph_mode,
        "corner_overlap_deg": args.corner_overlap_deg,
        "cup_tilt_deg": args.cup_tilt_deg,
        "cup_tilt_soft": args.cup_tilt_soft,
        "pour_tilt_deg": args.pour_tilt_deg,
        "pour_tilt_soft": args.pour_tilt_soft,
        "pour_rzy_safe": args.pour_rzy_safe,
        "pour_rzy_soft": args.pour_rzy_soft,
        "seed": args.seed,
        "device": args.device,
        "out_dir": str(args.out_dir),
        "zone_stats": {
            "n_cupping": int(n_cup),
            "n_pouring": int(n_pour),
            "n_overlap": int(n_overlap),
            "n_cupping_only": int(n_cup_only),
            "n_pouring_only": int(n_pour_only),
            "ee_x_range": [float(ee_pos_all[:, 0].min()), float(ee_pos_all[:, 0].max())],
            "ee_y_range": [float(ee_y.min()), float(ee_y.max())],
        },
        "pair_stats": {
            "n_intra_cupping": int(n_ic),
            "n_intra_pouring": int(n_ip),
            "n_cross": int(n_cr),
        },
        "speed_stats": {
            "cupping_min": float(speed_cupping.min()),
            "cupping_max": float(speed_cupping.max()),
            "pouring_min": float(speed_pouring.min()),
            "pouring_max": float(speed_pouring.max()),
        },
    }
    with open(out_dir / "pipeline_config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(out_dir / "routing_config.json", "w") as f:
        json.dump({
            "zone_mode": zone_mode,
            "y_divide": args.y_divide,
            "overlap": args.overlap,
            "corner_zones": corner_zones,
            "task_axes": task_axes,
            "task_axes_json": task_axes_json(task_axes),
        }, f, indent=2)

    # Print summary
    print(f"\n  sampled_points.npy:  {sampled_points.shape}")
    print(f"  speed_cupping.npy:   {speed_cup_pairs.shape}")
    print(f"  speed_pouring.npy:   {speed_pour_pairs.shape}")
    print(f"  normal_cupping.npy:  {normal_cup_pairs.shape}")
    print(f"  normal_pouring.npy:  {normal_pour_pairs.shape}")
    print(f"  ee_positions.npy:    {ee_positions.shape}")
    print(f"  zone_id.npy:         {zone_id.shape}")
    print(f"  pair_type.npy:       {pair_type.shape}")
    print(f"  d_obstacle.npy:      {d_obs_pairs.shape}")
    print(f"  d_self.npy:          {d_self_pairs.shape}")
    print(f"  d_orient_cupping.npy:{d_orient_cup_pairs.shape}")
    print(f"  d_orient_pouring.npy:{d_orient_pour_pairs.shape}")
    print(f"  pipeline_config.json")
    print(f"  routing_config.json")

    # ── Verification summary ─────────────────────────────────────────────────
    print("\n── Verification ──")
    print(f"  sampled_points range: [{sampled_points.min():.4f}, {sampled_points.max():.4f}]")
    print(f"  speed_cupping range:  [{speed_cup_pairs.min():.4f}, {speed_cup_pairs.max():.4f}]")
    print(f"  speed_pouring range:  [{speed_pour_pairs.min():.4f}, {speed_pour_pairs.max():.4f}]")
    nc = np.linalg.norm(normal_cup_pairs[:, :DOF], axis=1)
    np_ = np.linalg.norm(normal_pour_pairs[:, :DOF], axis=1)
    print(f"  normal_cup[:,:{DOF}] norms: [{nc.min():.4f}, {nc.max():.4f}]")
    print(f"  normal_pour[:,:{DOF}] norms: [{np_.min():.4f}, {np_.max():.4f}]")
    print(f"  Zone 0 (cupping) count in zone_id: {(zone_id == 0).sum()}")
    print(f"  Zone 1 (pouring) count in zone_id: {(zone_id == 1).sum()}")
    print(f"  Pair types: cup={n_ic}, pour={n_ip}, cross={n_cr}")
    print(f"  All shapes match: {sampled_points.shape[0] == speed_cup_pairs.shape[0] == M}")

    t_total = time.time() - t_total_start
    print(f"\nDone! Total time: {t_total:.1f}s")


if __name__ == "__main__":
    main()
