"""
Orientation-constraint utilities for manipulation tasks.

Supported tasks:
  - "cupping water": palm normal (EE +Z axis) must point UP (world +Z)
  - "pouring water": finger axis (EE +X axis) must point DOWN (world -Z)

The constraint is defined by:
  1. An axis column of the EE rotation matrix (which physical direction to constrain)
  2. A reference direction n_ref in the world frame (the target direction)
  3. A max tilt angle (half-cone around n_ref)

This module provides:
  - compute_ee_axis(q, robot_model, axis_col): batched FK -> axis vector (B, 3)
  - compute_palm_normal(q, robot_model): shortcut for axis_col=2 (palm normal)
  - compute_tilt_deg(n_cur, n_ref): batched angle in degrees (B,)
  - compute_tilt_mask(q, robot_model, n_ref, max_tilt_deg, axis_col): bool mask
  - compute_trajectory_tilt(traj, robot_model, n_ref, axis_col): per-step tilt

All functions are GPU-batched. Input q is (B, 7) torch tensor in radians.
"""

from __future__ import annotations

import torch

# Hard-coded default "cupping water" pose for the G1 right arm (radians).
# Joint order matches Q_MIN/Q_MAX in sample_fk.py:
#   [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
#    wrist_roll, wrist_pitch, wrist_yaw]
# NOTE: This config is only used as a fallback if n_ref is not specified
# directly. The preferred way to define the constraint is via N_REF_UP below.
Q_DEFAULT_CUPPING = torch.tensor(
    [0.0, 0.3, 0.0, 1.2, 0.0, -0.3, 0.0], dtype=torch.float32
)

# ---------------------------------------------------------------------------
# Task-specific constants
# ---------------------------------------------------------------------------

# "Cupping water": EE Z-axis (palm normal) should point world +Z (up).
N_REF_UP = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

# "Pouring water": EE X-axis (finger/cup-opening direction) should point
# world -Z (down), i.e. the cup is tipped upside-down so water pours out.
# Mathematical representation:
#   Let R_ee be the 3x3 rotation matrix of the EE frame in world coords.
#   The constrained axis is a = R_ee[:, 0]  (column 0 = EE +X axis).
#   For pouring: finger (EE +X) must stay FORWARD, i.e. close to world +X [1,0,0].
#   The check uses Rx_x = R_ee[0,0] = dot(EE_X, world_X), tilt = arccos(Rx_x) ≤ 20°.
#   Additionally, Rz_y ≥ -0.05 ensures the cup tilts LEFT only (not backward).
N_REF_DOWN = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)  # legacy, kept for compat
POUR_AXIS_COLUMN = 0  # EE +X axis = finger direction, must stay FORWARD for pouring


def _quat_wxyz_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """
    Batched quaternion (B, 4) in (w, x, y, z) -> rotation matrix (B, 3, 3).

    Matches gen_pointcloud.py:quat_to_rotation_matrix but torch-batched.
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    B = quat.shape[0]
    R = torch.empty(B, 3, 3, dtype=quat.dtype, device=quat.device)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


# Which column of the EE rotation matrix is the palm normal.
# For the G1 right hand with the URDF shipped in g1_description, +Z of the
# end-effector frame points out of the palm. If future inspection shows this
# is wrong, change PALM_AXIS_COLUMN.
PALM_AXIS_COLUMN = 2  # 0=X, 1=Y, 2=Z


def compute_ee_axis(
    q: torch.Tensor, robot_model, axis_col: int = PALM_AXIS_COLUMN
) -> torch.Tensor:
    """
    Batched FK -> unit vector along a specified EE frame axis.

    Args:
        q: (B, 7) joint configs in radians, on the robot_model device.
        robot_model: cuRobo CudaRobotModel.
        axis_col: which column of R_ee to extract.
                  0 = EE +X (finger direction, used for pouring)
                  1 = EE +Y
                  2 = EE +Z (palm normal, used for cupping)

    Returns:
        a: (B, 3) unit vectors (already unit length from rotation matrix).
    """
    state = robot_model.get_state(q)
    quat = state.ee_pose.quaternion  # (B, 4) wxyz
    R = _quat_wxyz_to_rotmat(quat)   # (B, 3, 3)
    return R[:, :, axis_col]          # (B, 3)


def compute_palm_normal(q: torch.Tensor, robot_model) -> torch.Tensor:
    """
    Batched FK -> palm normal unit vector (EE +Z axis).

    Shortcut for compute_ee_axis(q, robot_model, axis_col=PALM_AXIS_COLUMN).
    """
    return compute_ee_axis(q, robot_model, axis_col=PALM_AXIS_COLUMN)


def compute_tilt_deg(n_cur: torch.Tensor, n_ref: torch.Tensor) -> torch.Tensor:
    """
    Batched angle (degrees) between current and reference palm normals.

    Args:
        n_cur: (B, 3) unit vectors.
        n_ref: (3,) unit vector, on same device.

    Returns:
        tilt: (B,) angles in degrees, range [0, 180].
    """
    dot = (n_cur * n_ref).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(dot))


def compute_tilt_mask(
    q: torch.Tensor, robot_model, n_ref: torch.Tensor, max_tilt_deg: float,
    axis_col: int = PALM_AXIS_COLUMN,
) -> torch.Tensor:
    """
    Boolean mask: True where tilt(q) <= max_tilt_deg.

    Args:
        axis_col: which EE axis to measure (default: Z = palm normal).
                  Use POUR_AXIS_COLUMN (0) for the pouring constraint.
    """
    n_cur = compute_ee_axis(q, robot_model, axis_col=axis_col)
    tilt = compute_tilt_deg(n_cur, n_ref)
    return tilt <= max_tilt_deg


def compute_reference_normal(
    q_default: torch.Tensor = None, robot_model=None, task: str = "cupping"
) -> torch.Tensor:
    """
    Return the reference normal for a given task.

    Args:
        q_default: Ignored (kept for backward compat).
        robot_model: Ignored.
        task: "cupping" → N_REF_UP [0,0,1], "pouring" → N_REF_DOWN [0,0,-1].

    Returns:
        n_ref: (3,) tensor on the device of q_default (or CPU).
    """
    device = q_default.device if q_default is not None else "cpu"
    if task == "pouring":
        return N_REF_DOWN.to(device)
    return N_REF_UP.to(device)


def compute_trajectory_tilt(
    traj: torch.Tensor, robot_model, n_ref: torch.Tensor,
    axis_col: int = PALM_AXIS_COLUMN,
) -> torch.Tensor:
    """
    Per-step tilt along a single trajectory.

    Args:
        traj: (T, 7) joint configs in radians.
        robot_model: cuRobo CudaRobotModel.
        n_ref: (3,) reference direction.
        axis_col: which EE axis to measure (default: Z = palm normal).

    Returns:
        tilt: (T,) tilt in degrees at each step.
    """
    n_cur = compute_ee_axis(traj, robot_model, axis_col=axis_col)  # (T, 3)
    return compute_tilt_deg(n_cur, n_ref)


def _rotmat_from_vectors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Batched rotation matrix that rotates unit vector a onto unit vector b.

    Uses Rodrigues' formula. Handles a == b (identity) and a == -b (180 deg
    around arbitrary perpendicular axis) edge cases.

    Args:
        a: (B, 3) unit vectors.
        b: (B, 3) unit vectors.

    Returns:
        R: (B, 3, 3) rotation matrices with R @ a = b.
    """
    B = a.shape[0]
    device = a.device
    dtype = a.dtype

    v = torch.cross(a, b, dim=-1)  # (B, 3) axis (not unit)
    c = (a * b).sum(dim=-1, keepdim=True)  # (B, 1) cosine
    s2 = (v * v).sum(dim=-1, keepdim=True)  # (B, 1) sin^2

    # Skew-symmetric cross-product matrix for each row of v
    zero = torch.zeros(B, device=device, dtype=dtype)
    K = torch.stack([
        torch.stack([zero, -v[:, 2], v[:, 1]], dim=-1),
        torch.stack([v[:, 2], zero, -v[:, 0]], dim=-1),
        torch.stack([-v[:, 1], v[:, 0], zero], dim=-1),
    ], dim=-2)  # (B, 3, 3)

    I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)

    # Standard Rodrigues: R = I + K + K^2 * (1 - c) / s^2
    # s^2 = 1 - c^2 when a, b are unit vectors; near a == -b, s2 -> 0
    denom = torch.clamp(s2, min=1e-8)
    R = I + K + torch.bmm(K, K) * ((1.0 - c) / denom).unsqueeze(-1)

    # Handle antipodal case (a == -b): Rodrigues above is singular.
    # Replace with 180-degree rotation around any axis perpendicular to a.
    antipodal = (c.squeeze(-1) < -0.9999)
    if antipodal.any():
        # Pick an axis perpendicular to a: cross a with world X or Y (whichever
        # is not parallel)
        a_bad = a[antipodal]  # (M, 3)
        ref_axis = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        perp = torch.cross(a_bad, ref_axis.expand_as(a_bad), dim=-1)
        # If a is parallel to X, fall back to Y
        perp_norm = torch.linalg.norm(perp, dim=-1, keepdim=True)
        fallback = (perp_norm.squeeze(-1) < 1e-4)
        if fallback.any():
            y_axis = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
            perp[fallback] = torch.cross(a_bad[fallback], y_axis.expand_as(a_bad[fallback]), dim=-1)
        perp = perp / torch.clamp(torch.linalg.norm(perp, dim=-1, keepdim=True), min=1e-8)
        # 180 deg rotation around perp: R = 2*outer(perp, perp) - I
        R_anti = 2.0 * torch.einsum("bi,bj->bij", perp, perp) - I[antipodal]
        R[antipodal] = R_anti

    return R


def _rotmat_to_quat_wxyz(R: torch.Tensor) -> torch.Tensor:
    """
    Batched rotation matrix (B, 3, 3) -> quaternion (B, 4) in (w, x, y, z).

    Uses Shepperd's method (numerically stable branching on matrix trace).
    """
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]

    trace = m00 + m11 + m22
    B = R.shape[0]
    q = torch.empty(B, 4, device=R.device, dtype=R.dtype)

    # Branch 1: trace > 0
    cond1 = trace > 0
    s1 = torch.sqrt(torch.clamp(trace[cond1] + 1.0, min=1e-12)) * 2.0  # s1 = 4*qw
    q[cond1, 0] = 0.25 * s1
    q[cond1, 1] = (m21[cond1] - m12[cond1]) / s1
    q[cond1, 2] = (m02[cond1] - m20[cond1]) / s1
    q[cond1, 3] = (m10[cond1] - m01[cond1]) / s1

    # Branch 2: m00 largest diagonal and trace <= 0
    cond2 = (~cond1) & (m00 >= m11) & (m00 >= m22)
    s2 = torch.sqrt(torch.clamp(1.0 + m00[cond2] - m11[cond2] - m22[cond2], min=1e-12)) * 2.0
    q[cond2, 0] = (m21[cond2] - m12[cond2]) / s2
    q[cond2, 1] = 0.25 * s2
    q[cond2, 2] = (m01[cond2] + m10[cond2]) / s2
    q[cond2, 3] = (m02[cond2] + m20[cond2]) / s2

    # Branch 3: m11 largest
    cond3 = (~cond1) & (~cond2) & (m11 >= m22)
    s3 = torch.sqrt(torch.clamp(1.0 + m11[cond3] - m00[cond3] - m22[cond3], min=1e-12)) * 2.0
    q[cond3, 0] = (m02[cond3] - m20[cond3]) / s3
    q[cond3, 1] = (m01[cond3] + m10[cond3]) / s3
    q[cond3, 2] = 0.25 * s3
    q[cond3, 3] = (m12[cond3] + m21[cond3]) / s3

    # Branch 4: m22 largest
    cond4 = (~cond1) & (~cond2) & (~cond3)
    s4 = torch.sqrt(torch.clamp(1.0 + m22[cond4] - m00[cond4] - m11[cond4], min=1e-12)) * 2.0
    q[cond4, 0] = (m10[cond4] - m01[cond4]) / s4
    q[cond4, 1] = (m02[cond4] + m20[cond4]) / s4
    q[cond4, 2] = (m12[cond4] + m21[cond4]) / s4
    q[cond4, 3] = 0.25 * s4

    return q


# ---------------------------------------------------------------------------
# Pouring-task convenience API
# ---------------------------------------------------------------------------

# Default tilt threshold for pouring: ±20° from ideal (EE +X perfectly down).
# Rationale: pouring needs to be loose enough for the arm to reach many configs
# (the wrist must rotate significantly), but tight enough that water actually
# flows out. 20° is a starting point; may tighten to 15° after data inspection.
POUR_MAX_TILT_DEG = 20.0


def compute_pour_axis(q: torch.Tensor, robot_model) -> torch.Tensor:
    """Batched FK -> pouring axis (EE +X) unit vector. Shape (B, 3)."""
    return compute_ee_axis(q, robot_model, axis_col=POUR_AXIS_COLUMN)


def validate_pouring_orientation(
    q: torch.Tensor,
    robot_model,
    max_tilt_deg: float = POUR_MAX_TILT_DEG,
    n_ref: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Check whether each config satisfies the pouring constraint.

    Pouring constraint definition:
        The EE +X axis (finger/cup-opening direction) must point within
        max_tilt_deg of world -Z (downward). Formally:

            a = R_ee[:, 0]            (EE X-axis in world frame)
            n_ref = [0, 0, -1]        (target direction = world down)
            θ = arccos(a · n_ref)     (tilt angle in [0°, 180°])
            valid ⟺ θ ≤ max_tilt_deg

    Args:
        q: (B, 7) joint configs in radians.
        robot_model: cuRobo CudaRobotModel.
        max_tilt_deg: half-cone angle threshold (default 20°).
        n_ref: (3,) override reference direction (default: N_REF_DOWN).

    Returns:
        valid: (B,) bool — True where constraint is satisfied.
        tilt_deg: (B,) float — actual tilt angle in degrees.
    """
    if n_ref is None:
        n_ref = N_REF_DOWN.to(q.device)
    a = compute_ee_axis(q, robot_model, axis_col=POUR_AXIS_COLUMN)
    tilt_deg = compute_tilt_deg(a, n_ref)
    valid = tilt_deg <= max_tilt_deg
    return valid, tilt_deg


def sample_orientations_in_cone(
    n_ref: torch.Tensor,
    n_samples: int,
    max_tilt_deg: float,
    canonical_axis: int = PALM_AXIS_COLUMN,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Sample random SO(3) orientations whose canonical palm axis lies within a
    spherical cone of half-angle max_tilt_deg around n_ref. Free rotation
    around the palm axis is also sampled uniformly.

    The "canonical palm axis" is the PALM_AXIS_COLUMN-th column of the
    identity rotation matrix, i.e. the world axis the palm normal would point
    along if R = I. For PALM_AXIS_COLUMN=2 this is [0,0,1].

    Args:
        n_ref: (3,) unit reference palm normal.
        n_samples: number of orientations.
        max_tilt_deg: half-angle of the cone, degrees.
        canonical_axis: which column of R represents the palm axis (default
                        matches PALM_AXIS_COLUMN module constant).
        device: torch device; defaults to n_ref.device.

    Returns:
        quat_wxyz: (n_samples, 4) quaternions in (w, x, y, z) convention.
    """
    if device is None:
        device = n_ref.device
    dtype = n_ref.dtype

    # 1. Sample target palm normals uniformly in the spherical cap around n_ref.
    # Use the standard trick: sample cos(theta) uniformly in [cos(max_angle), 1]
    # and phi uniformly in [0, 2*pi] in a frame where n_ref is +Z, then rotate
    # back to the world frame.
    max_angle = torch.deg2rad(torch.tensor(max_tilt_deg, device=device, dtype=dtype))
    cos_min = torch.cos(max_angle)

    u = torch.rand(n_samples, device=device, dtype=dtype)
    cos_theta = cos_min + (1.0 - cos_min) * u  # (N,) in [cos_min, 1]
    sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta * cos_theta, min=0.0))
    phi = 2.0 * torch.pi * torch.rand(n_samples, device=device, dtype=dtype)

    # Local frame: n_local = (sin_theta cos_phi, sin_theta sin_phi, cos_theta)
    n_local = torch.stack(
        [sin_theta * torch.cos(phi), sin_theta * torch.sin(phi), cos_theta],
        dim=-1,
    )  # (N, 3)

    # Rotate n_local so that local +Z aligns with n_ref
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).expand(n_samples, -1)
    R_align = _rotmat_from_vectors(z_axis, n_ref.expand(n_samples, -1))  # (N, 3, 3)
    n_target = torch.bmm(R_align, n_local.unsqueeze(-1)).squeeze(-1)  # (N, 3)

    # 2. Build a rotation that maps the canonical palm axis (column
    # canonical_axis of identity) to n_target.
    canonical = torch.zeros(3, device=device, dtype=dtype)
    canonical[canonical_axis] = 1.0
    R_palm = _rotmat_from_vectors(
        canonical.expand(n_samples, -1), n_target
    )  # (N, 3, 3) — now R_palm @ canonical == n_target

    # 3. Add a random rotation around n_target (free "spin" around palm axis).
    spin_angle = 2.0 * torch.pi * torch.rand(n_samples, device=device, dtype=dtype)
    # Rotation of spin_angle around n_target (Rodrigues)
    c_s = torch.cos(spin_angle)
    s_s = torch.sin(spin_angle)
    nx, ny, nz = n_target[:, 0], n_target[:, 1], n_target[:, 2]
    # K is skew(n_target)
    zero = torch.zeros(n_samples, device=device, dtype=dtype)
    K = torch.stack([
        torch.stack([zero, -nz, ny], dim=-1),
        torch.stack([nz, zero, -nx], dim=-1),
        torch.stack([-ny, nx, zero], dim=-1),
    ], dim=-2)  # (N, 3, 3)
    I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(n_samples, -1, -1)
    R_spin = I + s_s.view(-1, 1, 1) * K + (1.0 - c_s).view(-1, 1, 1) * torch.bmm(K, K)

    # 4. Final rotation: spin after aligning palm axis to n_target
    R_final = torch.bmm(R_spin, R_palm)  # (N, 3, 3)

    return _rotmat_to_quat_wxyz(R_final)
