"""Deterministic q-only geometry features for the Phase-DF scalar gate.

The learned module in this file never predicts a travel time.  It consumes
fixed forward-kinematic relations of one configuration and returns four
internal semantic logits.  The parent scalar network reduces those logits to
one approval coefficient and still exposes exactly one ``F(q)``.
"""

from __future__ import annotations

import math
import struct

import torch
from torch import nn


FULLSEMANTIC_REPLICA_COUNT = 3
FULLSEMANTIC_HEAD_COUNT = 4
FULLSEMANTIC_SPHERE_COUNT = 53
FULLSEMANTIC_SELF_RELATION_COUNT = 731
FULLSEMANTIC_Q_FREQUENCIES = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
FULLSEMANTIC_ATTENTION_TEMPERATURES_M = (
    0.00025, 0.0005, 0.001, 0.002, 0.005)
FULLSEMANTIC_FREE_DISTANCE_M = 0.05
FULLSEMANTIC_SELF_SCALE = 2.5
FULLSEMANTIC_ORIENTATION_MARGIN_SCALE = 0.2
FULLSEMANTIC_ZONE_WIDTH_M = 0.005
ANALYTIC_CONTINUOUS_PERMISSION_TAU = 0.01


class _ExactSqrtSafeGradient(torch.autograd.Function):
    """Exact sqrt forward with a finite zero-distance backward subgradient."""

    @staticmethod
    def forward(ctx, squared: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(squared)
        return torch.sqrt(squared)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (squared,) = ctx.saved_tensors
        safe = squared.clamp_min(1.0e-12)
        return grad_output * (0.5 / torch.sqrt(safe))


class _ExactAcosSafeGradient(torch.autograd.Function):
    """Exact clamped-acos forward with finite gradients at aligned poses."""

    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(value)
        return torch.acos(value.clamp(-1.0, 1.0))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (value,) = ctx.saved_tensors
        safe = value.clamp(-1.0 + 1.0e-7, 1.0 - 1.0e-7)
        derivative = -torch.rsqrt((1.0 - safe.square()).clamp_min(1.0e-12))
        inside = (value >= -1.0) & (value <= 1.0)
        return torch.where(inside, grad_output * derivative, 0.0)


def _exact_norm_safe_gradient(
        value: torch.Tensor, dim: int) -> torch.Tensor:
    squared = torch.sum(value.square(), dim=dim)
    return _ExactSqrtSafeGradient.apply(squared)


def fullsemantic_q_fourier_features(
        q_standardized: torch.Tensor) -> torch.Tensor:
    """Return q plus the six registered sine/cosine frequency bands."""
    if q_standardized.ndim != 2 or q_standardized.shape[1] != 6:
        raise ValueError("standardized q must have shape (N,6)")
    frequencies = q_standardized.new_tensor(FULLSEMANTIC_Q_FREQUENCIES)
    phase = (
        math.pi * q_standardized[:, :, None]
        * frequencies[None, None, :])
    return torch.cat([
        q_standardized,
        torch.sin(phase).flatten(start_dim=1),
        torch.cos(phase).flatten(start_dim=1),
    ], dim=1)


def fullsemantic_multi_temperature_attention(
        primitive_distances: torch.Tensor,
        temperatures: torch.Tensor) -> torch.Tensor:
    """Return one scalar QK/V distance summary per temperature."""
    if primitive_distances.ndim != 2 or temperatures.ndim != 1:
        raise ValueError("primitive distances/temperatures must be rank 2/1")
    if primitive_distances.shape[1] == 0 or torch.any(temperatures <= 0):
        raise ValueError("relations and temperatures must be nonempty/positive")
    logits = (
        -primitive_distances[:, None, :]
        / temperatures[None, :, None])
    weights = torch.softmax(logits, dim=2)
    return torch.sum(
        weights * primitive_distances[:, None, :], dim=2)


def fullsemantic_relation_features(
        sphere_centers_flat: torch.Tensor, radii: torch.Tensor,
        cuboid_centers: torch.Tensor, cuboid_half: torch.Tensor,
        self_pairs: torch.Tensor, temperatures: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the registered obstacle and self-relation attention summaries."""
    if (sphere_centers_flat.ndim != 2
            or sphere_centers_flat.shape[1]
            != 3 * FULLSEMANTIC_SPHERE_COUNT):
        raise ValueError("sphere centers must have shape (N,159)")
    if radii.shape != (FULLSEMANTIC_SPHERE_COUNT,):
        raise ValueError("radii must have shape (53,)")
    if cuboid_centers.shape != (2, 3) or cuboid_half.shape != (2, 3):
        raise ValueError("the pinned scene must contain two axis-aligned cuboids")
    if self_pairs.shape != (FULLSEMANTIC_SELF_RELATION_COUNT, 2):
        raise ValueError("self pairs must have shape (731,2)")
    centers = sphere_centers_flat.reshape(
        -1, FULLSEMANTIC_SPHERE_COUNT, 3)
    delta = (
        centers[:, :, None, :] - cuboid_centers[None, None, :, :])
    obstacle = _exact_norm_safe_gradient(
        torch.relu(torch.abs(delta) - cuboid_half[None, None, :, :]),
        dim=-1) - radii[None, :, None]
    obstacle = obstacle.flatten(1)
    relative = (
        centers[:, self_pairs[:, 0], :]
        - centers[:, self_pairs[:, 1], :])
    self_distance = (
        _exact_norm_safe_gradient(relative, dim=-1)
        - radii[self_pairs[:, 0]][None, :]
        - radii[self_pairs[:, 1]][None, :])
    return (
        fullsemantic_multi_temperature_attention(obstacle, temperatures),
        fullsemantic_multi_temperature_attention(self_distance, temperatures),
    )


def fullsemantic_relation_minima(
        sphere_centers_flat: torch.Tensor, radii: torch.Tensor,
        cuboid_centers: torch.Tensor, cuboid_half: torch.Tensor,
        self_pairs: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact obstacle/self primitive minima with safe gradients."""
    if (sphere_centers_flat.ndim != 2
            or sphere_centers_flat.shape[1]
            != 3 * FULLSEMANTIC_SPHERE_COUNT):
        raise ValueError("sphere centers must have shape (N,159)")
    if radii.shape != (FULLSEMANTIC_SPHERE_COUNT,):
        raise ValueError("radii must have shape (53,)")
    if cuboid_centers.shape != (2, 3) or cuboid_half.shape != (2, 3):
        raise ValueError("the pinned scene must contain two axis-aligned cuboids")
    if self_pairs.shape != (FULLSEMANTIC_SELF_RELATION_COUNT, 2):
        raise ValueError("self pairs must have shape (731,2)")
    centers = sphere_centers_flat.reshape(
        -1, FULLSEMANTIC_SPHERE_COUNT, 3)
    delta = (
        centers[:, :, None, :] - cuboid_centers[None, None, :, :])
    obstacle = _exact_norm_safe_gradient(
        torch.relu(torch.abs(delta) - cuboid_half[None, None, :, :]),
        dim=-1) - radii[None, :, None]
    relative = (
        centers[:, self_pairs[:, 0], :]
        - centers[:, self_pairs[:, 1], :])
    self_distance = (
        _exact_norm_safe_gradient(relative, dim=-1)
        - radii[self_pairs[:, 0]][None, :]
        - radii[self_pairs[:, 1]][None, :])
    return (
        obstacle.flatten(1).min(dim=1).values,
        self_distance.min(dim=1).values,
    )


def analytic_continuous_approval(
        weakest_margin: torch.Tensor,
        tau: torch.Tensor | float = ANALYTIC_CONTINUOUS_PERMISSION_TAU,
        ) -> torch.Tensor:
    """Map one signed component margin to zero-sided scalar permission."""
    tau_value = torch.as_tensor(
        tau, device=weakest_margin.device, dtype=weakest_margin.dtype)
    if tau_value.numel() != 1 or bool(tau_value.detach() <= 0.0):
        raise ValueError("analytic permission tau must be one positive scalar")
    return -torch.expm1(-torch.relu(weakest_margin) / tau_value)


def _signed_box_margin(
        x: torch.Tensor, y: torch.Tensor,
        box: tuple[float, float, float, float] | list[float],
        ) -> torch.Tensor:
    x0, x1, y0, y1 = [float(value) for value in box]
    return torch.minimum(
        torch.minimum(x - x0, x1 - x),
        torch.minimum(y - y0, y1 - y))


def _box_distance(
        x: torch.Tensor, y: torch.Tensor,
        box: tuple[float, float, float, float] | list[float],
        ) -> torch.Tensor:
    x0, x1, y0, y1 = [float(value) for value in box]
    dx = torch.maximum(
        torch.maximum(x.new_tensor(x0) - x, x - x.new_tensor(x1)),
        torch.zeros_like(x))
    dy = torch.maximum(
        torch.maximum(y.new_tensor(y0) - y, y - y.new_tensor(y1)),
        torch.zeros_like(y))
    return _ExactSqrtSafeGradient.apply(dx.square() + dy.square())


def fullsemantic_zone_weights(
        ee_xy: torch.Tensor, corner_zones: dict,
        zone_width: float = FULLSEMANTIC_ZONE_WIDTH_M,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch replay of the frozen smooth q-only cup/pour activation."""
    if ee_xy.ndim != 2 or ee_xy.shape[1] != 2:
        raise ValueError("ee_xy must have shape (N,2)")
    if zone_width <= 0.0:
        raise ValueError("zone_width must be positive")
    cup_box = corner_zones["cup_region"]
    pour_box = corner_zones["pour_region"]
    overlap_box = corner_zones.get("overlap_region")
    x, y = ee_xy[:, 0], ee_xy[:, 1]
    cup_membership = torch.sigmoid(
        _signed_box_margin(x, y, cup_box) / float(zone_width))
    pour_membership = torch.sigmoid(
        _signed_box_margin(x, y, pour_box) / float(zone_width))
    if overlap_box is None:
        overlap_membership = torch.zeros_like(cup_membership)
    else:
        overlap_membership = torch.sigmoid(
            _signed_box_margin(x, y, overlap_box) / float(zone_width))
    nearest = torch.softmax(torch.stack([
        -_box_distance(x, y, cup_box) / float(zone_width),
        -_box_distance(x, y, pour_box) / float(zone_width),
    ], dim=1), dim=1)
    outside = (1.0 - cup_membership) * (1.0 - pour_membership)
    cup = cup_membership + outside * nearest[:, 0]
    pour = pour_membership + outside * nearest[:, 1]
    cup = 1.0 - (1.0 - cup) * (1.0 - overlap_membership)
    pour = 1.0 - (1.0 - pour) * (1.0 - overlap_membership)
    return cup, pour


def _float32_scalar(value: float) -> float:
    """Return the IEEE-754 float32 rounding of one Python scalar."""
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _numpy_float32_divide(
        numerator: torch.Tensor, denominator: float) -> torch.Tensor:
    """Replay NumPy float32 scalar division with a differentiable backward.

    CUDA commonly lowers float32 division to a reciprocal multiply.  That is
    algebraically equivalent but can differ by one ULP from the CPU NumPy
    label path; division by the 5 mm zone width then exposes that difference
    near the cup/pour bisector.  A float64 quotient followed by a float32 cast
    reproduces the correctly rounded float32 operation while retaining normal
    autograd through the cast.
    """
    if numerator.dtype != torch.float32:
        raise ValueError("analytic label replay requires float32 tensors")
    denominator_f32 = _float32_scalar(denominator)
    if denominator_f32 <= 0.0:
        raise ValueError("analytic label-replay denominator must be positive")
    return (numerator.to(torch.float64) / denominator_f32).to(torch.float32)


def analytic_continuous_zone_weights(
        ee_xy: torch.Tensor, corner_zones: dict,
        zone_width: float = FULLSEMANTIC_ZONE_WIDTH_M,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiably replay the immutable NumPy float32 zone arithmetic."""
    if ee_xy.ndim != 2 or ee_xy.shape[1] != 2:
        raise ValueError("ee_xy must have shape (N,2)")
    if ee_xy.dtype != torch.float32:
        raise ValueError("analytic label replay requires float32 ee_xy")
    if zone_width <= 0.0:
        raise ValueError("zone_width must be positive")
    cup_box = corner_zones["cup_region"]
    pour_box = corner_zones["pour_region"]
    overlap_box = corner_zones.get("overlap_region")
    x, y = ee_xy[:, 0], ee_xy[:, 1]

    def membership(box):
        scaled = _numpy_float32_divide(
            _signed_box_margin(x, y, box), zone_width)
        # The label helper explicitly promotes its sigmoid input to float64
        # and casts the result back to float32.
        return torch.sigmoid(scaled.to(torch.float64)).to(torch.float32)

    cup_membership = membership(cup_box)
    pour_membership = membership(pour_box)
    if overlap_box is None:
        overlap_membership = torch.zeros_like(cup_membership)
    else:
        overlap_membership = membership(overlap_box)

    logits = torch.stack([
        _numpy_float32_divide(-_box_distance(x, y, cup_box), zone_width),
        _numpy_float32_divide(-_box_distance(x, y, pour_box), zone_width),
    ], dim=1)
    centered = logits - logits.max(dim=1, keepdim=True).values
    # NumPy's exp receives float32 and returns float32.  Computing exp in
    # float64 before the cast avoids CUDA fast-math drift while preserving the
    # label path's float32 normalization order.
    exponent = torch.exp(centered.to(torch.float64)).to(torch.float32)
    nearest = exponent / exponent.sum(dim=1, keepdim=True)
    outside = (1.0 - cup_membership) * (1.0 - pour_membership)
    cup = cup_membership + outside * nearest[:, 0]
    pour = pour_membership + outside * nearest[:, 1]
    cup = 1.0 - (1.0 - cup) * (1.0 - overlap_membership)
    pour = 1.0 - (1.0 - pour) * (1.0 - overlap_membership)
    return cup, pour


def fullsemantic_fk8_from_pose(
        ee_position: torch.Tensor,
        quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """Reconstruct the exact registered FK8 coordinate from one EE pose."""
    if ee_position.ndim != 2 or ee_position.shape[1] != 3:
        raise ValueError("EE positions must have shape (N,3)")
    if quaternion_wxyz.shape != (ee_position.shape[0], 4):
        raise ValueError("EE quaternions must have shape (N,4)")
    # Replay the exact operation order used by the frozen q-feature builder:
    # construct R, apply normalized task axes with einsum, then express tilt
    # in degrees before normalizing.  This is algebraically identical to the
    # old closed form but avoids amplifying a one-ULP dot-product difference
    # through the 5 mm zone sigmoid.
    w, x, y, z = quaternion_wxyz.unbind(dim=1)
    rotation = torch.empty(
        (quaternion_wxyz.shape[0], 3, 3),
        dtype=quaternion_wxyz.dtype, device=quaternion_wxyz.device)
    rotation[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotation[:, 0, 1] = 2 * (x * y - z * w)
    rotation[:, 0, 2] = 2 * (x * z + y * w)
    rotation[:, 1, 0] = 2 * (x * y + z * w)
    rotation[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotation[:, 1, 2] = 2 * (y * z - x * w)
    rotation[:, 2, 0] = 2 * (x * z - y * w)
    rotation[:, 2, 1] = 2 * (y * z + x * w)
    rotation[:, 2, 2] = 1 - 2 * (x * x + y * y)

    cup_local = quaternion_wxyz.new_tensor([1.0, 0.0, 0.0])
    cup_world = quaternion_wxyz.new_tensor([0.0, 0.0, 1.0])
    pour_local = quaternion_wxyz.new_tensor([0.0, 0.0, 1.0])
    pour_world = quaternion_wxyz.new_tensor([0.0, -1.0, 0.0])
    cup_local = cup_local / cup_local.norm().clamp_min(1.0e-8)
    cup_world = cup_world / cup_world.norm().clamp_min(1.0e-8)
    pour_local = pour_local / pour_local.norm().clamp_min(1.0e-8)
    pour_world = pour_world / pour_world.norm().clamp_min(1.0e-8)
    cup_axis_world = torch.einsum("bij,j->bi", rotation, cup_local)
    pour_axis_world = torch.einsum("bij,j->bi", rotation, pour_local)
    cup_dot = torch.einsum("bi,i->b", cup_axis_world, cup_world)
    pour_dot = torch.einsum("bi,i->b", pour_axis_world, pour_world)
    degree_scale = 180.0 / 3.14159265
    cup_tilt = _ExactAcosSafeGradient.apply(cup_dot) * degree_scale
    pour_tilt = _ExactAcosSafeGradient.apply(pour_dot) * degree_scale
    cup_norm = cup_tilt / 180.0
    pour_norm = pour_tilt / 180.0
    handover = cup_tilt / (cup_tilt + pour_tilt).clamp_min(1.0e-3)
    return torch.stack([
        ee_position[:, 0], ee_position[:, 1], ee_position[:, 2],
        cup_dot, pour_dot, cup_norm, pour_norm, handover,
    ], dim=1)


def fullsemantic_orientation_relations(
        fk8: torch.Tensor, corner_zones: dict,
        component_boundary: torch.Tensor | float,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return active cup/pour component margins used by Phase DD/DF."""
    if fk8.ndim != 2 or fk8.shape[1] != 8:
        raise ValueError("FK8 must have shape (N,8)")
    cup_weight, pour_weight = fullsemantic_zone_weights(
        fk8[:, :2], corner_zones)
    free = FULLSEMANTIC_FREE_DISTANCE_M
    cup_tilt_deg = 180.0 * fk8[:, 5]
    pour_tilt_deg = 180.0 * fk8[:, 6]
    cup_clearance = (
        free * (20.0 - cup_tilt_deg) / 20.0).clamp(-free, free)
    pour_clearance = (
        free * (20.0 - pour_tilt_deg) / 20.0).clamp(-free, free)
    cup_component = (
        free + cup_weight * (cup_clearance - free)) / free
    pour_component = (
        free + pour_weight * (pour_clearance - free)) / free
    boundary = torch.as_tensor(
        component_boundary, device=fk8.device, dtype=fk8.dtype)
    return (
        ((cup_component - boundary)
         / FULLSEMANTIC_ORIENTATION_MARGIN_SCALE)[:, None],
        ((pour_component - boundary)
         / FULLSEMANTIC_ORIENTATION_MARGIN_SCALE)[:, None],
    )


def analytic_continuous_orientation_margins(
        fk8: torch.Tensor, corner_zones: dict,
        component_boundary: torch.Tensor | float,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay the exact-label cup/pour components as two scalar margins.

    Unlike the learned Phase-DF relation inputs, the analytic gate is audited
    directly against the immutable NumPy component labels.  It therefore
    uses their float32 zone arithmetic and reconstructs tilt from the stored
    axis dot, avoiding a lossy normalized-tilt multiply by 180.
    """
    if fk8.ndim != 2 or fk8.shape[1] != 8:
        raise ValueError("FK8 must have shape (N,8)")
    if fk8.dtype != torch.float32:
        raise ValueError("analytic label replay requires float32 FK8")
    cup_weight, pour_weight = analytic_continuous_zone_weights(
        fk8[:, :2], corner_zones)
    degree_scale = 180.0 / 3.14159265
    cup_tilt_deg = _ExactAcosSafeGradient.apply(fk8[:, 3]) * degree_scale
    pour_tilt_deg = _ExactAcosSafeGradient.apply(fk8[:, 4]) * degree_scale
    free = FULLSEMANTIC_FREE_DISTANCE_M
    cup_clearance = (
        free * (20.0 - cup_tilt_deg) / 20.0).clamp(-free, free)
    pour_clearance = (
        free * (20.0 - pour_tilt_deg) / 20.0).clamp(-free, free)
    cup_component = (
        free + cup_weight * (cup_clearance - free)) / free
    pour_component = (
        free + pour_weight * (pour_clearance - free)) / free
    boundary = torch.as_tensor(
        component_boundary, device=fk8.device, dtype=fk8.dtype)
    return cup_component - boundary, pour_component - boundary


class FullSemanticRelationReplica(nn.Module):
    """One frozen Phase-DF four-head semantic-relation classifier."""

    def __init__(self, anchor_dim: int = 512, context_dim: int = 8):
        super().__init__()
        self.anchor_branch = nn.Sequential(
            nn.Linear(anchor_dim, 128), nn.SiLU())
        self.q_branch = nn.Sequential(
            nn.Linear(78, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU())
        self.context_branch = nn.Sequential(
            nn.Linear(context_dim, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU())
        self.fk8_branch = nn.Sequential(
            nn.Linear(8, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU())
        self.shared_fusion = nn.Sequential(
            nn.Linear(512, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU())
        relation_dim = len(FULLSEMANTIC_ATTENTION_TEMPERATURES_M)
        self.obstacle_relation_branch = nn.Sequential(
            nn.Linear(relation_dim, 64), nn.SiLU(),
            nn.Linear(64, 128), nn.SiLU())
        self.self_relation_branch = nn.Sequential(
            nn.Linear(relation_dim, 64), nn.SiLU(),
            nn.Linear(64, 128), nn.SiLU())
        self.orientation_relation_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 64), nn.SiLU(),
                nn.Linear(64, 128), nn.SiLU())
            for _ in range(2)
        ])
        self.safety_private_tails = nn.ModuleList([
            nn.Sequential(
                nn.Linear(384, 128), nn.SiLU(),
                nn.Linear(128, 64), nn.SiLU(),
                nn.Linear(64, 1))
            for _ in range(2)
        ])
        self.orientation_private_tails = nn.ModuleList([
            nn.Sequential(
                nn.Linear(384, 128), nn.SiLU(),
                nn.Linear(128, 64), nn.SiLU(),
                nn.Linear(64, 1))
            for _ in range(2)
        ])

    def forward(
            self, anchor_hidden: torch.Tensor,
            q_standardized: torch.Tensor, context: torch.Tensor,
            fk8_standardized: torch.Tensor,
            obstacle_relations: torch.Tensor,
            self_relations: torch.Tensor,
            cup_relation: torch.Tensor,
            pour_relation: torch.Tensor) -> torch.Tensor:
        shared = self.shared_fusion(torch.cat([
            self.anchor_branch(anchor_hidden),
            self.q_branch(fullsemantic_q_fourier_features(q_standardized)),
            self.context_branch(context),
            self.fk8_branch(fk8_standardized),
        ], dim=1))
        private = [
            self.obstacle_relation_branch(obstacle_relations),
            self.self_relation_branch(self_relations),
            self.orientation_relation_branches[0](cup_relation),
            self.orientation_relation_branches[1](pour_relation),
        ]
        tails = [
            *self.safety_private_tails,
            *self.orientation_private_tails,
        ]
        return torch.cat([
            tail(torch.cat([shared, semantic], dim=1))
            for tail, semantic in zip(tails, private)
        ], dim=1)
