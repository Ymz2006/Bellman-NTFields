"""
Collision checking utilities using true signed distances.

Provides correct sphere-cuboid distance computation for collision checking,
replacing the cuRobo cost-based API which returns activation-clamped costs
(not signed distances).

Key functions:
    sphere_cuboid_distance:  per-batch min signed distance (positive=free, negative=penetration)
    check_collision:         boolean collision check for a batch of joint configs
    check_trajectory_collision: collision check with waypoint interpolation

Usage:
    from collision_utils import check_trajectory_collision, load_cuboid_geometry

    cuboid_centers, cuboid_half = load_cuboid_geometry("scene_obstacle.yml", device)
    in_coll, d_min = check_trajectory_collision(robot_model, traj_rad, cuboid_centers, cuboid_half)
"""

from pathlib import Path

import torch
import yaml


def load_cuboid_geometry(scene_path: str, device: str = "cuda:0"):
    """Load cuboid obstacle geometry from scene YAML.

    Returns:
        cuboid_centers: (K, 3) tensor of cuboid center positions
        cuboid_half:    (K, 3) tensor of cuboid half-extents

    Note: Assumes axis-aligned cuboids (quaternion rotation ignored).
    """
    with open(scene_path) as f:
        data = yaml.safe_load(f)
    centers = []
    half_dims = []
    for c in data.get("cuboids", []):
        p = c["pose"]
        d = c["dims"]
        centers.append([p[0], p[1], p[2]])
        half_dims.append([d[0] / 2, d[1] / 2, d[2] / 2])
    return (
        torch.tensor(centers, dtype=torch.float32, device=device),
        torch.tensor(half_dims, dtype=torch.float32, device=device),
    )


def sphere_cuboid_distance(sphere_centers, sphere_radii, cuboid_centers, cuboid_half):
    """Compute signed distance from collision spheres to axis-aligned cuboids.

    Args:
        sphere_centers: (B, S, 3) sphere center positions in world frame
        sphere_radii:   (B, S) or (S,) sphere radii
        cuboid_centers: (K, 3) cuboid center positions
        cuboid_half:    (K, 3) cuboid half-extents

    Returns:
        (B,) min signed distance across all sphere-cuboid pairs per batch element.
        Positive = free space, negative = penetration.
    """
    B, S, _ = sphere_centers.shape

    # (B, S, 1, 3) - (1, 1, K, 3) → (B, S, K, 3)
    delta = sphere_centers.unsqueeze(2) - cuboid_centers.unsqueeze(0).unsqueeze(0)

    # Distance from point to AABB surface (0 if inside)
    clamped = torch.clamp(torch.abs(delta) - cuboid_half.unsqueeze(0).unsqueeze(0), min=0.0)
    dist_to_surface = torch.norm(clamped, dim=-1)  # (B, S, K)

    # Subtract sphere radius
    if sphere_radii.dim() == 1:
        sphere_radii = sphere_radii.unsqueeze(0).expand(B, -1)
    signed_dist = dist_to_surface - sphere_radii.unsqueeze(2)  # (B, S, K)

    # Min over all spheres and cuboids
    min_dist, _ = signed_dist.reshape(B, -1).min(dim=1)  # (B,)
    return min_dist


def sphere_sphere_distance(centers, radii, ignore_mask=None):
    """Compute pairwise signed distances between collision spheres.

    Args:
        centers: (B, S, 3) sphere center positions
        radii:   (B, S) or (S,) sphere radii
        ignore_mask: (S, S) bool tensor — True means ignore that pair

    Returns:
        (B,) min signed distance across all valid pairs.
        Positive = separated, negative = overlapping.
    """
    B, S, _ = centers.shape
    if radii.dim() == 1:
        radii = radii.unsqueeze(0).expand(B, -1)

    # (B, S, 1, 3) - (B, 1, S, 3) → (B, S, S, 3)
    diff = centers.unsqueeze(2) - centers.unsqueeze(1)
    dist = torch.norm(diff, dim=-1)  # (B, S, S)

    # signed distance = center distance - r_i - r_j
    signed = dist - radii.unsqueeze(2) - radii.unsqueeze(1)  # (B, S, S)

    # Mask: ignore self-pairs (diagonal) and specified ignore pairs
    mask = torch.eye(S, dtype=torch.bool, device=centers.device).unsqueeze(0)  # (1, S, S)
    if ignore_mask is not None:
        mask = mask | ignore_mask.unsqueeze(0)

    signed = signed.masked_fill(mask, float('inf'))

    # Min over all valid pairs
    min_dist, _ = signed.reshape(B, -1).min(dim=1)  # (B,)
    return min_dist


def build_self_collision_mask(sphere_yaml_path, self_collision_ignore, collision_link_names,
                              extra_ignore_gap=2, device="cuda:0"):
    """Build (S, S) ignore mask for self-collision checking.

    Masks out:
        - Same-link sphere pairs (always ignored)
        - Adjacent-link sphere pairs (from self_collision_ignore config)
        - Links within extra_ignore_gap steps in the kinematic chain
          (because closely-spaced links on a serial arm always overlap)

    Args:
        sphere_yaml_path: path to sphere YAML file
        self_collision_ignore: dict mapping link_name → [ignored_link_names]
        collision_link_names: ordered list of collision link names
        extra_ignore_gap: ignore link pairs within this many steps (default 2)

    Returns:
        ignore_mask: (S, S) bool tensor — True = ignore this pair
        sphere_link_indices: (S,) int tensor — link index for each sphere
    """
    with open(sphere_yaml_path) as f:
        data = yaml.safe_load(f)
    spheres = data["collision_spheres"]

    # Build sphere-to-link index mapping
    sphere_link_idx = []
    for link_idx, link_name in enumerate(collision_link_names):
        if link_name in spheres:
            n = len(spheres[link_name])
            sphere_link_idx.extend([link_idx] * n)
    sphere_link_idx = torch.tensor(sphere_link_idx, device=device)
    S = len(sphere_link_idx)

    # Same-link mask
    mask = sphere_link_idx.unsqueeze(0) == sphere_link_idx.unsqueeze(1)  # (S, S)

    # Adjacent-link mask from self_collision_ignore config
    link_name_to_idx = {name: i for i, name in enumerate(collision_link_names)}
    for link_a, ignored_links in self_collision_ignore.items():
        if link_a not in link_name_to_idx:
            continue
        idx_a = link_name_to_idx[link_a]
        for link_b in ignored_links:
            if link_b not in link_name_to_idx:
                continue
            idx_b = link_name_to_idx[link_b]
            pair_mask = ((sphere_link_idx.unsqueeze(0) == idx_a) &
                         (sphere_link_idx.unsqueeze(1) == idx_b))
            pair_mask = pair_mask | pair_mask.T
            mask = mask | pair_mask

    # Also ignore link pairs within extra_ignore_gap steps in the chain.
    # On a serial arm, links 2 steps apart often have overlapping spheres
    # at zero config (e.g., elbow/2 ↔ wrist_pitch/0).
    if extra_ignore_gap >= 2:
        n_links = len(collision_link_names)
        for gap in range(2, extra_ignore_gap + 1):
            for li in range(n_links - gap):
                idx_a = li
                idx_b = li + gap
                pair_mask = ((sphere_link_idx.unsqueeze(0) == idx_a) &
                             (sphere_link_idx.unsqueeze(1) == idx_b))
                pair_mask = pair_mask | pair_mask.T
                mask = mask | pair_mask

    return mask, sphere_link_idx


def check_collision(robot_model, q_rad, cuboid_centers, cuboid_half,
                    margin=0.005, device="cuda:0"):
    """Check joint configs for collision using true signed distances.

    Args:
        robot_model: cuRobo CudaRobotModel instance
        q_rad: (B, 7) joint angles in radians
        cuboid_centers: (K, 3) cuboid centers
        cuboid_half: (K, 3) cuboid half-extents
        margin: safety margin in meters (configs with d < margin are collisions)

    Returns:
        in_collision: (B,) bool tensor
        min_distances: (B,) signed distance tensor
    """
    q = q_rad.to(device)
    if q.dim() == 1:
        q = q.unsqueeze(0)

    state = robot_model.get_state(q)
    spheres = state.link_spheres_tensor  # (B, S, 4)
    centers = spheres[:, :, :3]
    radii = spheres[:, :, 3]

    d = sphere_cuboid_distance(centers, radii, cuboid_centers, cuboid_half)
    return d < margin, d


def check_trajectory_collision(robot_model, trajectory_rad, cuboid_centers, cuboid_half,
                               margin=0.005, interpolation_steps=10, device="cuda:0"):
    """Check trajectory for collisions with waypoint interpolation.

    Interpolates between consecutive waypoints to catch collisions
    that occur between sparse MPPI steps.

    Args:
        robot_model: cuRobo CudaRobotModel instance
        trajectory_rad: (T, 7) tensor in radians
        cuboid_centers: (K, 3) cuboid centers
        cuboid_half: (K, 3) cuboid half-extents
        margin: safety margin in meters
        interpolation_steps: number of intermediate configs between waypoints

    Returns:
        in_collision: (N,) bool tensor for all interpolated configs
        distances: (N,) signed distances for all interpolated configs
    """
    q = trajectory_rad.to(device)
    n_pts = q.shape[0]

    if n_pts < 2 or interpolation_steps <= 1:
        return check_collision(robot_model, q, cuboid_centers, cuboid_half, margin, device)

    # Interpolate between consecutive waypoints (vectorized)
    alphas = torch.linspace(0, 1, interpolation_steps + 1, device=device)[1:]  # (S,)
    q_from = q[:-1]  # (n_pts-1, D)
    q_to = q[1:]     # (n_pts-1, D)
    interp_segments = q_from.unsqueeze(1) + alphas.reshape(1, -1, 1) * (q_to - q_from).unsqueeze(1)
    interp_q = torch.cat([q[0:1], interp_segments.reshape(-1, q.shape[1])], dim=0)

    return check_collision(robot_model, interp_q, cuboid_centers, cuboid_half, margin, device)


def check_collision_full(robot_model, q_rad, cuboid_centers, cuboid_half,
                         torso_cuboid_centers=None, torso_cuboid_half=None,
                         self_collision_mask=None,
                         torso_ignore_sphere_indices=None,
                         margin=0.005, self_collision_margin=0.002,
                         device="cuda:0"):
    """Check collision against obstacles, torso, and self-collision.

    Args:
        robot_model: cuRobo CudaRobotModel instance
        q_rad: (B, 7) joint angles in radians
        cuboid_centers: (K, 3) environment cuboid centers
        cuboid_half: (K, 3) environment cuboid half-extents
        torso_cuboid_centers: (K2, 3) torso cuboid centers (optional)
        torso_cuboid_half: (K2, 3) torso cuboid half-extents (optional)
        self_collision_mask: (S, S) bool ignore mask (optional)
        torso_ignore_sphere_indices: list of sphere indices to ignore vs torso
            (e.g., [0] for right_shoulder_pitch_link attached to torso)
        margin: safety margin for obstacle/torso in meters (default 5mm)
        self_collision_margin: safety margin for self-collision in meters (default 2mm)

    Returns:
        in_collision: (B,) bool tensor
        distances: dict with 'obstacle', 'torso', 'self', 'min' keys, each (B,)
    """
    q = q_rad.to(device)
    if q.dim() == 1:
        q = q.unsqueeze(0)
    B = q.shape[0]

    state = robot_model.get_state(q)
    spheres = state.link_spheres_tensor  # (B, S, 4)
    centers = spheres[:, :, :3]
    radii = spheres[:, :, 3]

    result = {}

    # 1. Obstacle distance (existing)
    d_obs = sphere_cuboid_distance(centers, radii, cuboid_centers, cuboid_half)
    result['obstacle'] = d_obs

    # 2. Torso distance
    if torso_cuboid_centers is not None and torso_cuboid_half is not None:
        S = centers.shape[1]
        K2 = torso_cuboid_centers.shape[0]

        # Compute full distance matrix (B, S, K2)
        delta = centers.unsqueeze(2) - torso_cuboid_centers.unsqueeze(0).unsqueeze(0)
        clamped = torch.clamp(torch.abs(delta) - torso_cuboid_half.unsqueeze(0).unsqueeze(0), min=0.0)
        dist_to_surface = torch.norm(clamped, dim=-1)  # (B, S, K2)

        r = radii if radii.dim() == 2 else radii.unsqueeze(0).expand(B, -1)
        signed_torso = dist_to_surface - r.unsqueeze(2)  # (B, S, K2)

        # Mask out spheres that are attached to the torso
        if torso_ignore_sphere_indices:
            for si in torso_ignore_sphere_indices:
                signed_torso[:, si, :] = float('inf')

        d_torso, _ = signed_torso.reshape(B, -1).min(dim=1)
        result['torso'] = d_torso
    else:
        result['torso'] = torch.full((B,), float('inf'), device=device)

    # 3. Self-collision distance
    if self_collision_mask is not None:
        d_self = sphere_sphere_distance(centers, radii, self_collision_mask)
        result['self'] = d_self
    else:
        result['self'] = torch.full((B,), float('inf'), device=device)

    # Combined: check each type against its own margin
    obs_coll = result['obstacle'] < margin
    torso_coll = result['torso'] < margin
    self_coll = result['self'] < self_collision_margin
    in_collision = obs_coll | torso_coll | self_coll

    d_min = torch.min(torch.min(result['obstacle'], result['torso']), result['self'])
    result['min'] = d_min

    return in_collision, result


def check_trajectory_collision_full(robot_model, trajectory_rad, cuboid_centers, cuboid_half,
                                    torso_cuboid_centers=None, torso_cuboid_half=None,
                                    self_collision_mask=None,
                                    torso_ignore_sphere_indices=None,
                                    margin=0.005, interpolation_steps=10, device="cuda:0"):
    """Check trajectory for collisions with full collision checking.

    Like check_trajectory_collision but includes torso and self-collision.
    """
    q = trajectory_rad.to(device)
    n_pts = q.shape[0]

    if n_pts < 2 or interpolation_steps <= 1:
        return check_collision_full(
            robot_model, q, cuboid_centers, cuboid_half,
            torso_cuboid_centers, torso_cuboid_half,
            self_collision_mask, torso_ignore_sphere_indices,
            margin=margin, device=device)

    # Interpolate between consecutive waypoints (vectorized)
    alphas = torch.linspace(0, 1, interpolation_steps + 1, device=device)[1:]  # (S,)
    q_from = q[:-1]  # (n_pts-1, D)
    q_to = q[1:]     # (n_pts-1, D)
    interp_segments = q_from.unsqueeze(1) + alphas.reshape(1, -1, 1) * (q_to - q_from).unsqueeze(1)
    interp_q = torch.cat([q[0:1], interp_segments.reshape(-1, q.shape[1])], dim=0)

    return check_collision_full(
        robot_model, interp_q, cuboid_centers, cuboid_half,
        torso_cuboid_centers, torso_cuboid_half,
        self_collision_mask, torso_ignore_sphere_indices,
        margin=margin, device=device)
