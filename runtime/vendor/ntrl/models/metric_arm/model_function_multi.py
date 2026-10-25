import matplotlib
import numpy as np
import math
import random
import time
import contextlib
import os

import torch
import torch.nn.functional as F

from torch.nn import Linear
from torch import Tensor
from torch.nn import Conv3d
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable, grad
from torch.cuda.amp import autocast
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
try:
    import igl
except ImportError:
    igl = None
import copy

import matplotlib
import matplotlib.pylab as plt

from timeit import default_timer as timer

try:
    from .model_network_multi import (
        ROUTE_CHART_CONSENSUS_MODE,
        ROUTE_CHART_UNION_CONSENSUS_MODE,
        ROUTE_CHART_FOURIER_RISK_CONSENSUS_MODE,
        ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_MARGIN_RESCUE_CONSENSUS_MODE,
        ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_SEMANTIC_ANCHOR_RESCUE_CONSENSUS_MODE,
        ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_REPLICA_MEDIAN_ANCHOR_RESCUE_CONSENSUS_MODE,
        ROUTE_CHART_FOURIER_RISK_MODES,
        ROUTE_CHART_UNION_MODES,
        ROUTE_CHART_MODES,
        ROUTE_CHART_COUNT,
        ROUTE_CHART_FOURIER_RISK_HEAD_COUNT,
        monotone_release_veto,
        margin_signed_release_rescue,
        scalar_speed_to_release,
        semantic_approved_anchor_release,
        replica_median_positive_anchor_release,
        neutral_log_odds_fusion,
        normalized_logit_softmin_probability,
        normalized_release_smoothmax,
    )
except ImportError:
    from model_network_multi import (
        ROUTE_CHART_CONSENSUS_MODE,
        ROUTE_CHART_UNION_CONSENSUS_MODE,
        ROUTE_CHART_FOURIER_RISK_CONSENSUS_MODE,
        ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_MARGIN_RESCUE_CONSENSUS_MODE,
        ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_SEMANTIC_ANCHOR_RESCUE_CONSENSUS_MODE,
        ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_REPLICA_MEDIAN_ANCHOR_RESCUE_CONSENSUS_MODE,
        ROUTE_CHART_FOURIER_RISK_MODES,
        ROUTE_CHART_UNION_MODES,
        ROUTE_CHART_MODES,
        ROUTE_CHART_COUNT,
        ROUTE_CHART_FOURIER_RISK_HEAD_COUNT,
        monotone_release_veto,
        margin_signed_release_rescue,
        scalar_speed_to_release,
        semantic_approved_anchor_release,
        replica_median_positive_anchor_release,
        neutral_log_odds_fusion,
        normalized_logit_softmin_probability,
        normalized_release_smoothmax,
    )

torch.backends.cudnn.benchmark = True


def _pair_type_mask(pair_type, spec):
    """Return a bool tensor selecting pair types from a cup/pour/cross spec."""
    if pair_type is None:
        return None
    spec = (spec or "").strip().lower()
    if not spec or spec in ("all", "*"):
        return torch.ones_like(pair_type, dtype=torch.bool)
    aliases = {
        "cup": 0, "cupping": 0, "0": 0,
        "pour": 1, "pouring": 1, "1": 1,
        "cross": 2, "handover": 2, "2": 2,
    }
    keep = torch.zeros_like(pair_type, dtype=torch.bool)
    for raw in spec.replace(",", " ").split():
        if raw not in aliases:
            raise ValueError(f"unknown pair type '{raw}' in path loss spec {spec!r}")
        keep |= (pair_type == aliases[raw])
    return keep


def _coordinate_component_repair_target(
        network, components, gap, independent_rho, target_speed,
        iterations=24):
    """Scale a per-factor repair proposal to respect its exact scalar target.

    The aggregate is monotone in the common scale beta, so a fixed bisection
    finds the largest proposal on the segment from the raw semantic factors to
    their independently repaired values without overshooting F*.  This uses
    only the existing q-only aggregate/component labels.
    """
    if (components.ndim != 2 or gap.shape != components.shape
            or independent_rho.shape != components.shape):
        raise ValueError(
            "components, gap, and independent_rho must share shape (N,H)")
    target = target_speed.reshape(-1, 1)
    if target.shape[0] != components.shape[0]:
        raise ValueError("target_speed must contain one scalar per row")
    if iterations < 1:
        raise ValueError("repair coordination iterations must be positive")
    semantic = network._aggregate_speed_constraint_components(components)
    proposal = network._aggregate_speed_constraint_components(
        components + independent_rho * gap)
    lower = torch.zeros_like(target)
    upper = torch.ones_like(target)
    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        midpoint_speed = network._aggregate_speed_constraint_components(
            components + midpoint * independent_rho * gap)
        lower = torch.where(midpoint_speed < target, midpoint, lower)
        upper = torch.where(midpoint_speed >= target, midpoint, upper)
    beta = 0.5 * (lower + upper)
    beta = torch.where(target >= proposal, torch.ones_like(beta), beta)
    # Prefer no repair when both endpoints are numerically identical (most
    # often because the normalized softmin is clamped at the speed floor).
    # The proposal has no current effect there, while a positive target would
    # give the hurdle head a misleading release label.
    beta = torch.where(target <= semantic, torch.zeros_like(beta), beta)
    return independent_rho * beta, beta


def _aggregate_corridor_consensus_target(
        repaired_semantic, direct, target_speed, threshold=0.2):
    """Identify a one-sided direct-scalar rescue at a fixed trust boundary.

    Only an exact high state that the repaired semantic scalar calls low is a
    positive.  The inherited direct scalar must itself cross the same boundary
    and lie above the semantic scalar.  Magnitude is the exact reachable
    convex coefficient; all other rows are deterministic zero targets.
    """
    semantic = repaired_semantic.reshape(-1)
    direct = direct.reshape(-1)
    target = target_speed.reshape(-1)
    if not (semantic.shape == direct.shape == target.shape):
        raise ValueError(
            "repaired_semantic, direct, and target_speed must share shape")
    gap = torch.relu(direct - semantic)
    identified = gap > 1e-8
    support = (
        identified
        & (semantic < threshold)
        & (target >= threshold)
        & (direct >= threshold))
    safe_gap = torch.where(identified, gap, torch.ones_like(gap))
    alpha = torch.clamp(
        (target - semantic) / safe_gap, min=0.0, max=1.0)
    alpha = torch.where(support, alpha, torch.zeros_like(alpha))
    return alpha, support, identified


def _release_compensated_magnitude_target(
        alpha_target, release_prediction, support=None, eps=1e-6):
    """Undo a fixed hurdle release when supervising its magnitude tail.

    The deployed consensus remains ``alpha = release * magnitude``.  Dividing
    the exact alpha by the detached release asks the magnitude tail to produce
    the reachable optimum: after multiplication the result is exactly alpha
    when release is large enough, and release otherwise.  No deployment-time
    oracle or extra output is introduced.
    """
    if alpha_target.shape != release_prediction.shape:
        raise ValueError(
            "alpha_target and release_prediction must share shape")
    if eps <= 0.0:
        raise ValueError("release compensation epsilon must be positive")
    if support is None:
        support = alpha_target > 0.0
    if support.shape != alpha_target.shape:
        raise ValueError("support must share alpha_target shape")
    safe_release = release_prediction.detach().clamp_min(eps)
    target = torch.clamp(alpha_target / safe_release, min=0.0, max=1.0)
    return torch.where(support, target, torch.zeros_like(target))


def _magnitude_compensated_release_target(
        alpha_target, magnitude_prediction, support=None, eps=1e-6):
    """Undo a fixed magnitude when supervising effective semantic release.

    For ``alpha = effective_release * magnitude``, this is the reachable
    release target ``clip(alpha*/magnitude, 0, 1)``.  Magnitude is detached so
    a private semantic correction cannot reduce its loss by changing the
    inherited tail.  Dangerous rows receive the explicit zero target in the
    balanced objective that consumes this helper.
    """
    if alpha_target.shape != magnitude_prediction.shape:
        raise ValueError(
            "alpha_target and magnitude_prediction must share shape")
    if eps <= 0.0:
        raise ValueError("magnitude compensation epsilon must be positive")
    if support is None:
        support = alpha_target > 0.0
    if support.shape != alpha_target.shape:
        raise ValueError("support must share alpha_target shape")
    safe_magnitude = magnitude_prediction.detach().clamp_min(eps)
    target = torch.clamp(alpha_target / safe_magnitude, min=0.0, max=1.0)
    return torch.where(support, target, torch.zeros_like(target))


def _corridor_dangerous_negative_mask(
        repaired_semantic, direct, target_speed, threshold=0.2):
    """Select lows that an open upward corridor can flip at the boundary."""
    semantic = repaired_semantic.reshape(-1)
    direct = direct.reshape(-1)
    target = target_speed.reshape(-1)
    if not (semantic.shape == direct.shape == target.shape):
        raise ValueError(
            "repaired_semantic, direct, and target_speed must share shape")
    identified = torch.relu(direct - semantic) > 1e-8
    return (
        identified
        & (target < threshold)
        & (semantic < threshold)
        & (direct >= threshold))


def _balanced_index_pool_sample(positive_pool, negative_pool, total_count):
    """Draw an equal, shuffled minibatch from two fixed index pools.

    Both pools are defined entirely from the training scalar field and frozen
    scalar experts.  Sampling cycles through a fresh random permutation before
    repeating a pool, so rare strata are exposed without silently changing the
    equal-class objective.
    """
    if positive_pool.ndim != 1 or negative_pool.ndim != 1:
        raise ValueError("balanced index pools must be one-dimensional")
    if positive_pool.device != negative_pool.device:
        raise ValueError("balanced index pools must share a device")
    if positive_pool.numel() < 1 or negative_pool.numel() < 1:
        raise ValueError("balanced index pools must both be non-empty")
    if total_count < 2:
        raise ValueError("balanced index sampling requires at least two rows")

    def _draw(pool, count):
        pieces = []
        remaining = int(count)
        while remaining > 0:
            order = torch.randperm(pool.numel(), device=pool.device)
            take = min(remaining, pool.numel())
            pieces.append(pool[order[:take]])
            remaining -= take
        return torch.cat(pieces, dim=0)

    positive_count = total_count // 2
    negative_count = total_count - positive_count
    sampled = torch.cat([
        _draw(positive_pool, positive_count),
        _draw(negative_pool, negative_count),
    ], dim=0)
    return sampled[torch.randperm(sampled.numel(), device=sampled.device)]


def _normalized_softmin_component_permission_target(
        component_target, aggregate_threshold, temperature):
    """Return exact per-component permissions for a normalized soft minimum.

    For ``A(c) = -tau log(mean(exp(-c/tau)))`` with ``H`` components,
    ``A(c) >= theta`` implies every component is at least
    ``theta - tau log(H)``.  The resulting binary targets therefore split an
    aggregate veto into physical private heads without inventing a fitted
    threshold or reading any endpoint/value/evaluation label.
    """
    if component_target.ndim != 2 or component_target.shape[1] < 2:
        raise ValueError(
            "component_target must have shape (N,H) with H >= 2")
    if temperature <= 0.0:
        raise ValueError("normalized softmin temperature must be positive")
    component_threshold = (
        float(aggregate_threshold)
        - float(temperature) * math.log(component_target.shape[1]))
    permission = component_target >= component_threshold
    return permission.to(component_target.dtype), component_threshold


class MultiFunction():
    """Multi-task loss function for dual-head NTField network.

    Based on model_function_metric.Function but routes speed/normal through
    the same zone-aware blending weights as the network features.
    """

    def __init__(self, path, device, network, dim,
                 use_temporal_loss=True, use_eikonal_loss=True, use_normal_loss=True,
                 eikonal_form='legacy_sqrt',
                 eikonal_balance_power=0.0,
                 eikonal_min_speed=0.0,
                 speed_loss_weight=0.0,
                 speed_loss_balance_power=0.0,
                 speed_loss_form='speed',
                 collocation_speed_loss_weight=0.0,
                 collocation_speed_n=0,
                 collocation_speed_loss_balance_power=0.0,
                 collocation_speed_loss_form='inherit',
                 collocation_speed_component_loss_weight=0.0,
                 collocation_speed_uncomposed_loss_weight=0.0,
                 collocation_speed_consensus_identification_weight=0.0,
                 collocation_speed_sampling='uniform',
                 collocation_speed_consensus_negative_weight=1.0,
                 collocation_speed_consensus_head_component_weight=0.0,
                 collocation_speed_repair_identification_weight=0.0,
                 collocation_speed_repair_coordination=False,
                 clearance_loss_weight=0.0,
                 clearance_self_scale=2.5,
                 hopf_cole_weight=0.0,
                 hopf_cole_n=0,
                 hopf_epsilon_start=0.20,
                 hopf_epsilon_end=0.02,
                 hopf_decay_epochs=600,
                 hopf_goal_radius=0.06,
                 hopf_goal_weight=1.0,
                 hopf_adaptive=False,
                 hopf_adaptive_speed_power=2.0,
                 hopf_adaptive_sharpness_weight=1.0,
                 hopf_characteristic_fraction=0.5,
                 hopf_characteristic_step=0.02,
                 hopf_state_constraint_weight=0.0,
                 hopf_boundary_speed=0.35,
                 hopf_speed_mode='predicted',
                 hopf_residual_form='travel_time',
                 hopf_ritz_normalize=False,
                 hopf_shellwise_normalize=False,
                 hopf_shell_bins=8,
                 hopf_endpoint_measure=False,
                 hopf_goal_shell_measure=False,
                 hopf_shell_radius_min=0.03,
                 hopf_shell_radius_max=1.0,
                 viscosity_weight=0.0,
                 viscosity_n=0,
                 viscosity_dirs=16,
                 viscosity_action_mode='discrete_min',
                 viscosity_softmin_temperature=0.05,
                 viscosity_epsilon=0.005,
                 viscosity_rollout_steps=1,
                 viscosity_intrinsic_budget=0.0,
                 viscosity_step=0.02,
                 viscosity_step_max=0.0,
                 viscosity_start_epoch=1,
                 viscosity_include_descent=True,
                 viscosity_include_speed_ascent=False,
                 viscosity_causal_target=False,
                 viscosity_causal_order='euclidean',
                 viscosity_candidate_routing='cup',
                 viscosity_hard_domain=False,
                 viscosity_update_mode='symmetric',
                 viscosity_collocation_pairs=False,
                 viscosity_global_cloud=False,
                 viscosity_local_boundary_fraction=0.0,
                 viscosity_boundary_min_radius=0.002,
                 viscosity_boundary_max_radius=0.03,
                 fermat_path_weight=0.0,
                 fermat_loss_transform='linear',
                 use_align_loss=False, cross_mode='default',
                 use_boundary_align=False, boundary_align_margin=0.05,
                 boundary_align_weight=1e-3,
                 and_overlap=0.0, and_blend='min',
                 cup_loss_weight=1.0, pour_loss_weight=1.0,
                 cross_loss_weight=1.0,
                 pure_scalar_q_only=False):
        self.path = path
        self.device = device
        self.dim = dim
        self.network = network
        self.pure_scalar_q_only = bool(pure_scalar_q_only)
        if self.pure_scalar_q_only:
            if not getattr(network, 'endpoint_task_siamese', False):
                raise ValueError(
                    "pure_scalar_q_only requires endpoint_task_siamese")
            if getattr(network, 'endpoint_dual_value_heads', False):
                raise ValueError(
                    "pure_scalar_q_only forbids endpoint_dual_value_heads")
            if getattr(network, 'value_condition', None) != 'none':
                raise ValueError(
                    "pure_scalar_q_only requires value_condition='none'")

        self.use_temporal_loss = use_temporal_loss
        self.use_eikonal_loss = use_eikonal_loss
        self.use_normal_loss = use_normal_loss
        self.eikonal_form = str(eikonal_form)
        if self.eikonal_form not in ('legacy_sqrt', 'standard'):
            raise ValueError(
                f"unknown eikonal_form={self.eikonal_form!r}")
        self.eikonal_balance_power = float(eikonal_balance_power)
        self.eikonal_min_speed = float(eikonal_min_speed)
        if not 0.0 <= self.eikonal_min_speed < 1.0:
            raise ValueError(
                "eikonal_min_speed must lie in [0, 1)")
        # Optional coherent-goal collocation for the same scalar Eikonal PDE.
        # Environment controls keep the default training/config interface
        # byte-compatible; launchers record every nonzero setting explicitly.
        self.coherent_eikonal_weight = float(
            os.environ.get("R58_COHERENT_EIKONAL_WEIGHT", "0"))
        self.coherent_eikonal_n = int(
            os.environ.get("R58_COHERENT_EIKONAL_N", "0"))
        self.coherent_goal_bank = int(
            os.environ.get("R58_COHERENT_GOAL_BANK", "32"))
        self.coherent_goals_per_batch = int(
            os.environ.get("R58_COHERENT_GOALS_PER_BATCH", "4"))
        self.coherent_viscosity_epsilon = float(
            os.environ.get("R58_COHERENT_VISCOSITY_EPSILON", "0"))
        self.coherent_subsolution_reward = float(
            os.environ.get("R58_COHERENT_SUBSOLUTION_REWARD", "0"))
        self.coherent_discount_lambda = float(
            os.environ.get("R58_COHERENT_DISCOUNT_LAMBDA", "0"))
        if self.coherent_eikonal_weight < 0.0:
            raise ValueError("R58_COHERENT_EIKONAL_WEIGHT must be non-negative")
        if self.coherent_viscosity_epsilon < 0.0:
            raise ValueError(
                "R58_COHERENT_VISCOSITY_EPSILON must be non-negative")
        if self.coherent_subsolution_reward < 0.0:
            raise ValueError(
                "R58_COHERENT_SUBSOLUTION_REWARD must be non-negative")
        if self.coherent_discount_lambda < 0.0:
            raise ValueError(
                "R58_COHERENT_DISCOUNT_LAMBDA must be non-negative")
        coherent_selector_count = sum((
            self.coherent_subsolution_reward > 0.0,
            self.coherent_viscosity_epsilon > 0.0,
            self.coherent_discount_lambda > 0.0,
        ))
        if coherent_selector_count > 1:
            raise ValueError(
                "coherent max-subsolution, second-order viscosity, and "
                "discounted proper-PDE "
                "controls are isolated experiments")
        if self.coherent_eikonal_weight > 0.0:
            if not self.pure_scalar_q_only:
                raise ValueError(
                    "coherent Eikonal collocation requires pure_scalar_q_only")
            if self.coherent_eikonal_n < 2:
                raise ValueError("R58_COHERENT_EIKONAL_N must be at least 2")
            if self.coherent_goal_bank < 1:
                raise ValueError("R58_COHERENT_GOAL_BANK must be positive")
            if not 1 <= self.coherent_goals_per_batch <= self.coherent_goal_bank:
                raise ValueError(
                    "R58_COHERENT_GOALS_PER_BATCH must lie in [1, goal_bank]")
        self._coherent_step = 0
        self._coherent_goal_indices = None
        # Training-only monotone Godunov stencil.  This repeatedly evaluates
        # the same deployed scalar T at q +/- h e_i; it stores no grid/value
        # state and consumes no Bellman/DP target, path, or teacher signal.
        self.monotone_fd_weight = float(
            os.environ.get("R58_MONOTONE_FD_WEIGHT", "0"))
        self.monotone_fd_n = int(
            os.environ.get("R58_MONOTONE_FD_N", "256"))
        self.monotone_fd_step = float(
            os.environ.get("R58_MONOTONE_FD_STEP", "0.01"))
        self.monotone_fd_domain_speed = float(
            os.environ.get("R58_MONOTONE_FD_DOMAIN_SPEED", "0.2"))
        self.monotone_fd_goal_bank = int(
            os.environ.get("R58_MONOTONE_FD_GOAL_BANK", "32"))
        self.monotone_fd_goals_per_batch = int(
            os.environ.get("R58_MONOTONE_FD_GOALS_PER_BATCH", "4"))
        self.monotone_fd_sampling_power = float(
            os.environ.get("R58_MONOTONE_FD_SAMPLING_POWER", "0"))
        self.monotone_fd_rar_factor = int(
            os.environ.get("R58_MONOTONE_FD_RAR_FACTOR", "1"))
        if self.monotone_fd_weight < 0.0:
            raise ValueError("R58_MONOTONE_FD_WEIGHT must be non-negative")
        if self.monotone_fd_weight > 0.0:
            if not self.pure_scalar_q_only:
                raise ValueError(
                    "monotone finite differences require pure_scalar_q_only")
            if not getattr(network, 'use_speed_head', False):
                raise ValueError(
                    "monotone finite differences require one scalar F head")
            if self.monotone_fd_n < 2:
                raise ValueError("R58_MONOTONE_FD_N must be at least two")
            if not 0.0 < self.monotone_fd_step < 0.5:
                raise ValueError("R58_MONOTONE_FD_STEP must lie in (0,0.5)")
            if not 0.0 < self.monotone_fd_domain_speed <= 1.0:
                raise ValueError(
                    "R58_MONOTONE_FD_DOMAIN_SPEED must lie in (0,1]")
            if self.monotone_fd_goal_bank < 1:
                raise ValueError(
                    "R58_MONOTONE_FD_GOAL_BANK must be positive")
            if not (1 <= self.monotone_fd_goals_per_batch
                    <= self.monotone_fd_goal_bank):
                raise ValueError(
                    "R58_MONOTONE_FD_GOALS_PER_BATCH must lie in goal bank")
            if self.monotone_fd_sampling_power < 0.0:
                raise ValueError(
                    "R58_MONOTONE_FD_SAMPLING_POWER must be non-negative")
            if self.monotone_fd_rar_factor < 1:
                raise ValueError(
                    "R58_MONOTONE_FD_RAR_FACTOR must be at least one")
        self._monotone_fd_step_count = 0
        self._monotone_fd_goal_indices = None
        self.trusted_anchor_maximal_weight = float(
            os.environ.get("R58_TRUSTED_ANCHOR_MAXIMAL_WEIGHT", "0"))
        self.trusted_anchor_n = int(
            os.environ.get("R58_TRUSTED_ANCHOR_N", "128"))
        self.trusted_anchor_count = int(
            os.environ.get("R58_TRUSTED_ANCHOR_COUNT", "16"))
        self.trusted_anchor_quadrature = int(
            os.environ.get("R58_TRUSTED_ANCHOR_QUADRATURE", "9"))
        self.trusted_anchor_min_speed = float(
            os.environ.get("R58_TRUSTED_ANCHOR_MIN_SPEED", "0.2"))
        self.trusted_anchor_upper_weight = float(
            os.environ.get("R58_TRUSTED_ANCHOR_UPPER_WEIGHT", "100"))
        self.trusted_anchor_optimize_steps = int(
            os.environ.get("R58_TRUSTED_ANCHOR_OPTIMIZE_STEPS", "0"))
        self.trusted_anchor_optimize_lr = float(
            os.environ.get("R58_TRUSTED_ANCHOR_OPTIMIZE_LR", "0.05"))
        self.trusted_anchor_softmin_temperature = float(
            os.environ.get("R58_TRUSTED_ANCHOR_SOFTMIN_TEMPERATURE", "0.02"))
        self.trusted_anchor_barrier_weight = float(
            os.environ.get("R58_TRUSTED_ANCHOR_BARRIER_WEIGHT", "10"))
        if self.trusted_anchor_maximal_weight < 0.0:
            raise ValueError(
                "R58_TRUSTED_ANCHOR_MAXIMAL_WEIGHT must be non-negative")
        if self.trusted_anchor_maximal_weight > 0.0:
            if not self.pure_scalar_q_only:
                raise ValueError(
                    "trusted-anchor maximality requires pure_scalar_q_only")
            if not getattr(network, 'use_speed_head', False):
                raise ValueError(
                    "trusted-anchor maximality requires one scalar speed head")
            if self.trusted_anchor_n < 2 or self.trusted_anchor_count < 1:
                raise ValueError("invalid trusted-anchor pair/anchor count")
            if self.trusted_anchor_quadrature < 2:
                raise ValueError("trusted-anchor quadrature must be at least two")
            if not 0.0 < self.trusted_anchor_min_speed <= 1.0:
                raise ValueError("trusted-anchor minimum speed must lie in (0,1]")
            if self.trusted_anchor_upper_weight <= 0.0:
                raise ValueError("trusted-anchor upper weight must be positive")
            if (self.trusted_anchor_optimize_steps < 0
                    or self.trusted_anchor_optimize_lr <= 0.0
                    or self.trusted_anchor_softmin_temperature <= 0.0
                    or self.trusted_anchor_barrier_weight < 0.0):
                raise ValueError(
                    "invalid trusted-anchor inner optimizer settings")
        self.speed_loss_weight = float(speed_loss_weight)
        self.speed_loss_balance_power = float(speed_loss_balance_power)
        self.speed_loss_form = str(speed_loss_form)
        if self.speed_loss_form not in ('speed', 'log_speed', 'slowness'):
            raise ValueError(
                f"unknown speed_loss_form={self.speed_loss_form!r}")
        self.collocation_speed_loss_weight = float(
            collocation_speed_loss_weight)
        self.collocation_speed_n = int(collocation_speed_n)
        self.collocation_speed_loss_balance_power = float(
            collocation_speed_loss_balance_power)
        self.collocation_speed_loss_form = str(collocation_speed_loss_form)
        self.collocation_speed_component_loss_weight = float(
            collocation_speed_component_loss_weight)
        self.collocation_speed_uncomposed_loss_weight = float(
            collocation_speed_uncomposed_loss_weight)
        self.collocation_speed_consensus_identification_weight = float(
            collocation_speed_consensus_identification_weight)
        self.collocation_speed_sampling = str(collocation_speed_sampling)
        self.collocation_speed_consensus_negative_weight = float(
            collocation_speed_consensus_negative_weight)
        self.collocation_speed_consensus_head_component_weight = float(
            collocation_speed_consensus_head_component_weight)
        self.collocation_speed_repair_identification_weight = float(
            collocation_speed_repair_identification_weight)
        self.collocation_speed_repair_coordination = bool(
            collocation_speed_repair_coordination)
        if self.collocation_speed_loss_form not in (
                'inherit', 'speed', 'log_speed', 'slowness'):
            raise ValueError(
                "collocation_speed_loss_form must be 'inherit', 'speed', "
                "'log_speed', or 'slowness'")
        if self.collocation_speed_loss_weight < 0.0:
            raise ValueError(
                "collocation_speed_loss_weight must be non-negative")
        if self.collocation_speed_n < 0:
            raise ValueError("collocation_speed_n must be non-negative")
        if self.collocation_speed_loss_balance_power < 0.0:
            raise ValueError(
                "collocation_speed_loss_balance_power must be non-negative")
        if self.collocation_speed_component_loss_weight < 0.0:
            raise ValueError(
                "collocation_speed_component_loss_weight must be non-negative")
        if self.collocation_speed_uncomposed_loss_weight < 0.0:
            raise ValueError(
                "collocation_speed_uncomposed_loss_weight must be non-negative")
        if self.collocation_speed_consensus_identification_weight < 0.0:
            raise ValueError(
                "collocation_speed_consensus_identification_weight must be "
                "non-negative")
        if self.collocation_speed_sampling not in (
                'uniform', 'consensus_balanced'):
            raise ValueError(
                "collocation_speed_sampling must be 'uniform' or "
                "'consensus_balanced'")
        if self.collocation_speed_consensus_negative_weight <= 0.0:
            raise ValueError(
                "collocation_speed_consensus_negative_weight must be "
                "positive")
        if self.collocation_speed_consensus_head_component_weight < 0.0:
            raise ValueError(
                "collocation_speed_consensus_head_component_weight must be "
                "non-negative")
        if (self.collocation_speed_sampling == 'consensus_balanced'
                and self.collocation_speed_consensus_identification_weight
                <= 0.0):
            raise ValueError(
                "consensus-balanced sampling requires consensus "
                "identification supervision")
        if self.collocation_speed_repair_identification_weight < 0.0:
            raise ValueError(
                "collocation_speed_repair_identification_weight must be "
                "non-negative")
        if (self.collocation_speed_component_loss_weight > 0.0
                and getattr(network, 'speed_constraint_head_count', 0) < 2):
            raise ValueError(
                "component supervision requires scalar constraint heads")
        if (self.collocation_speed_uncomposed_loss_weight > 0.0
                and getattr(network, 'speed_constraint_consensus_width', 0)
                < 1):
            raise ValueError(
                "uncomposed speed supervision requires scalar consensus")
        if (self.collocation_speed_consensus_identification_weight > 0.0
                and getattr(network, 'speed_constraint_consensus_width', 0)
                < 1):
            raise ValueError(
                "consensus identification requires scalar consensus")
        if (self.collocation_speed_repair_identification_weight > 0.0
                and getattr(network, 'speed_constraint_repair_width', 0) < 1):
            raise ValueError(
                "repair identification requires scalar component repair")
        if (self.collocation_speed_repair_coordination
                and self.collocation_speed_repair_identification_weight <= 0.0):
            raise ValueError(
                "repair coordination requires repair identification")
        if ((self.collocation_speed_loss_weight > 0.0
             or self.collocation_speed_component_loss_weight > 0.0
             or self.collocation_speed_uncomposed_loss_weight > 0.0
             or self.collocation_speed_consensus_identification_weight > 0.0
             or self.collocation_speed_repair_identification_weight > 0.0)
                and self.collocation_speed_n < 1):
            raise ValueError(
                "collocation_speed_n must be positive when collocation "
                "speed supervision is enabled")
        self.clearance_loss_weight = float(clearance_loss_weight)
        self.clearance_self_scale = float(clearance_self_scale)
        if self.clearance_self_scale <= 0.0:
            raise ValueError("clearance_self_scale must be positive")
        self.hopf_cole_weight = float(hopf_cole_weight)
        self.hopf_cole_n = int(hopf_cole_n)
        self.hopf_epsilon_start = float(hopf_epsilon_start)
        self.hopf_epsilon_end = float(hopf_epsilon_end)
        self.hopf_decay_epochs = int(hopf_decay_epochs)
        self.hopf_goal_radius = float(hopf_goal_radius)
        self.hopf_goal_weight = float(hopf_goal_weight)
        if self.hopf_goal_weight <= 0.0:
            raise ValueError("hopf_goal_weight must be positive")
        self.hopf_adaptive = bool(hopf_adaptive)
        self.hopf_adaptive_speed_power = float(
            hopf_adaptive_speed_power)
        self.hopf_adaptive_sharpness_weight = float(
            hopf_adaptive_sharpness_weight)
        self.hopf_characteristic_fraction = float(
            hopf_characteristic_fraction)
        self.hopf_characteristic_step = float(hopf_characteristic_step)
        self.hopf_state_constraint_weight = float(
            hopf_state_constraint_weight)
        self.hopf_boundary_speed = float(hopf_boundary_speed)
        self.hopf_speed_mode = str(hopf_speed_mode)
        if self.hopf_speed_mode not in ('predicted', 'target'):
            raise ValueError(
                "hopf_speed_mode must be 'predicted' or 'target'")
        self.hopf_residual_form = str(hopf_residual_form)
        if self.hopf_residual_form not in (
                'travel_time', 'desirability_ritz'):
            raise ValueError(
                "hopf_residual_form must be 'travel_time' or "
                "'desirability_ritz'")
        self.hopf_ritz_normalize = bool(hopf_ritz_normalize)
        self.hopf_shellwise_normalize = bool(hopf_shellwise_normalize)
        self.hopf_shell_bins = int(hopf_shell_bins)
        self.hopf_endpoint_measure = bool(hopf_endpoint_measure)
        self.hopf_goal_shell_measure = bool(hopf_goal_shell_measure)
        self.hopf_shell_radius_min = float(hopf_shell_radius_min)
        self.hopf_shell_radius_max = float(hopf_shell_radius_max)
        if self.hopf_endpoint_measure and self.hopf_goal_shell_measure:
            raise ValueError("Hopf endpoint and goal-shell measures are exclusive")
        if self.hopf_shell_bins < 1:
            raise ValueError("hopf_shell_bins must be positive")
        if (self.hopf_shellwise_normalize
                and not self.hopf_goal_shell_measure):
            raise ValueError(
                "hopf_shellwise_normalize requires hopf_goal_shell_measure")
        if (self.hopf_shellwise_normalize
                and self.hopf_residual_form != 'desirability_ritz'):
            raise ValueError(
                "hopf_shellwise_normalize requires desirability_ritz")
        if not (0.0 < self.hopf_shell_radius_min
                <= self.hopf_shell_radius_max):
            raise ValueError("invalid Hopf goal-shell radius range")
        if (self.hopf_speed_mode == 'target'
                and self.hopf_adaptive
                and self.hopf_characteristic_fraction > 0.0
                and self.hopf_characteristic_step > 0.0):
            raise ValueError(
                "hopf_speed_mode='target' cannot use displaced "
                "characteristic collocation points without target labels "
                "at the displaced states")
        if self.hopf_cole_weight < 0.0 or self.hopf_cole_n < 0:
            raise ValueError("Hopf--Cole weight and sample count must be non-negative")
        if not 0.0 < self.hopf_epsilon_end <= self.hopf_epsilon_start:
            raise ValueError("require 0 < hopf_epsilon_end <= hopf_epsilon_start")
        if self.hopf_decay_epochs < 1 or self.hopf_goal_radius <= 0.0:
            raise ValueError("invalid Hopf--Cole decay or goal radius")
        if self.hopf_adaptive_speed_power < 0.0:
            raise ValueError("Hopf adaptive speed power must be non-negative")
        if self.hopf_adaptive_sharpness_weight < 0.0:
            raise ValueError("Hopf adaptive sharpness weight must be non-negative")
        if not 0.0 <= self.hopf_characteristic_fraction <= 1.0:
            raise ValueError("Hopf characteristic fraction must lie in [0, 1]")
        if self.hopf_characteristic_step < 0.0:
            raise ValueError("Hopf characteristic step must be non-negative")
        if self.hopf_state_constraint_weight < 0.0:
            raise ValueError("Hopf state-constraint weight must be non-negative")
        if not 0.0 < self.hopf_boundary_speed <= 1.0:
            raise ValueError("Hopf boundary speed must lie in (0, 1]")
        self.viscosity_weight = float(viscosity_weight)
        self.viscosity_n = int(viscosity_n)
        self.viscosity_dirs = int(viscosity_dirs)
        self.viscosity_action_mode = str(viscosity_action_mode)
        if self.viscosity_action_mode not in (
                'discrete_min', 'softmin', 'stochastic_vv'):
            raise ValueError(
                f"unknown viscosity_action_mode="
                f"{self.viscosity_action_mode!r}")
        self.viscosity_softmin_temperature = float(
            viscosity_softmin_temperature)
        if self.viscosity_softmin_temperature <= 0.0:
            raise ValueError(
                "viscosity_softmin_temperature must be positive")
        self.viscosity_epsilon = float(viscosity_epsilon)
        if self.viscosity_epsilon <= 0.0:
            raise ValueError("viscosity_epsilon must be positive")
        if (self.viscosity_action_mode == 'stochastic_vv'
                and (self.viscosity_dirs < 2
                     or self.viscosity_dirs % 2 != 0)):
            raise ValueError(
                "stochastic_vv requires an even viscosity_dirs >= 2")
        self.viscosity_rollout_steps = int(viscosity_rollout_steps)
        if self.viscosity_rollout_steps < 1:
            raise ValueError("viscosity_rollout_steps must be >= 1")
        self.viscosity_intrinsic_budget = float(
            viscosity_intrinsic_budget)
        if self.viscosity_intrinsic_budget < 0.0:
            raise ValueError(
                "viscosity_intrinsic_budget must be non-negative")
        if (self.viscosity_rollout_steps > 1
                and self.viscosity_action_mode not in (
                    'stochastic_vv', 'softmin')):
            raise ValueError(
                "multi-step viscosity rollout requires stochastic_vv or "
                "softmin")
        if (self.viscosity_rollout_steps > 1
                and self.viscosity_action_mode == 'stochastic_vv'
                and self.viscosity_intrinsic_budget <= 0.0):
            raise ValueError(
                "multi-step viscosity rollout requires a positive "
                "intrinsic budget")
        self.viscosity_step = float(viscosity_step)
        self.viscosity_step_max = float(viscosity_step_max)
        self.viscosity_start_epoch = int(viscosity_start_epoch)
        self.viscosity_include_descent = bool(viscosity_include_descent)
        self.viscosity_include_speed_ascent = bool(
            viscosity_include_speed_ascent)
        self.viscosity_causal_target = bool(viscosity_causal_target)
        self.viscosity_causal_order = str(viscosity_causal_order)
        if self.viscosity_causal_order not in ('euclidean', 'value', 'none'):
            raise ValueError(
                f"unknown viscosity_causal_order="
                f"{self.viscosity_causal_order!r}")
        self.viscosity_candidate_routing = str(
            viscosity_candidate_routing)
        if self.viscosity_candidate_routing not in ('cup', 'local'):
            raise ValueError(
                f"unknown viscosity_candidate_routing="
                f"{self.viscosity_candidate_routing!r}")
        self.viscosity_hard_domain = bool(viscosity_hard_domain)
        if self.viscosity_hard_domain and self.eikonal_min_speed <= 0:
            raise ValueError(
                "viscosity_hard_domain requires eikonal_min_speed > 0")
        self.viscosity_update_mode = str(viscosity_update_mode)
        if self.viscosity_update_mode not in ('symmetric', 'raise_only'):
            raise ValueError(
                f"unknown viscosity_update_mode="
                f"{self.viscosity_update_mode!r}")
        self.viscosity_collocation_pairs = bool(
            viscosity_collocation_pairs)
        if (self.viscosity_collocation_pairs
                and not self.pure_scalar_q_only):
            raise ValueError(
                "viscosity_collocation_pairs is restricted to the pure "
                "q-only scalar PDE")
        self.viscosity_global_cloud = bool(viscosity_global_cloud)
        if (self.viscosity_global_cloud
                and not self.viscosity_collocation_pairs):
            raise ValueError(
                "viscosity_global_cloud requires independent scalar "
                "viscosity_collocation_pairs")
        if (self.viscosity_global_cloud
                and self.viscosity_action_mode != 'softmin'):
            raise ValueError(
                "viscosity_global_cloud requires the smooth softmin "
                "Bellman operator")
        # A finite point-source cell supplies the missing boundary closure
        # for an independent scalar PDE measure.  These synthetic pairs are
        # generated from q-only collocation states and an isotropic local
        # displacement; no endpoint task/pair metadata is consulted.
        self.viscosity_local_boundary_fraction = float(
            viscosity_local_boundary_fraction)
        self.viscosity_boundary_min_radius = float(
            viscosity_boundary_min_radius)
        self.viscosity_boundary_max_radius = float(
            viscosity_boundary_max_radius)
        if not 0.0 <= self.viscosity_local_boundary_fraction <= 1.0:
            raise ValueError(
                "viscosity_local_boundary_fraction must lie in [0, 1]")
        if (self.viscosity_boundary_min_radius <= 0.0
                or self.viscosity_boundary_max_radius
                < self.viscosity_boundary_min_radius):
            raise ValueError(
                "invalid viscosity local-boundary radius range")
        if (self.viscosity_local_boundary_fraction > 0.0
                and not self.viscosity_collocation_pairs):
            raise ValueError(
                "viscosity_local_boundary_fraction requires "
                "viscosity_collocation_pairs")
        self.fermat_path_weight = float(fermat_path_weight)
        self.fermat_loss_transform = str(fermat_loss_transform)
        if self.fermat_loss_transform not in ('linear', 'log'):
            raise ValueError(
                f"unknown fermat_loss_transform="
                f"{self.fermat_loss_transform!r}")
        self.use_align_loss = use_align_loss
        self.cross_mode = cross_mode
        self.use_boundary_align = use_boundary_align
        self.boundary_align_margin = boundary_align_margin
        self.boundary_align_weight = boundary_align_weight
        self.and_overlap = and_overlap
        self.and_blend = and_blend
        self.cup_loss_weight = cup_loss_weight
        self.pour_loss_weight = pour_loss_weight
        self.cross_loss_weight = cross_loss_weight

        # Path-level self-supervision hooks (populated by the training loop; None = off)
        self._oracle = None            # TrainCollisionOracle or None
        self._safe_neighbor = None     # callable (q_norm)->(q_safe, found) or None
        self._segment_barrier_safe_q = None  # optional safe q pool for local validity anchors
        self._line_preserve_safe_q = None  # optional safe q pool for direct-line preservation
        self._q_min_norm = None        # (dim,) tensor normalized lower bound
        self._q_max_norm = None        # (dim,) tensor normalized upper bound

        self.total_train_loss = []
        self.total_val_loss = []

        self.alpha = 1.025
        limit = 0.5
        self.margin = limit / 15.0
        self.offset = self.margin / 10.0
        self._consensus_support_index_pool = None
        self._consensus_danger_index_pool = None
        self._consensus_stratified_pool_size = None

    def gradient(self, y, x, create_graph=True):
        grad_y = torch.ones_like(y)
        grad_x = torch.autograd.grad(y, x, grad_y, only_inputs=True,
                                     retain_graph=True, create_graph=create_graph)[0]
        return grad_x

    def _build_consensus_stratified_pools(
            self, independent_pool, independent_labels, threshold,
            chunk_size=65536):
        """Cache exact support/danger strata from frozen scalar experts."""
        pool_size = int(independent_pool.shape[0])
        if (self._consensus_support_index_pool is not None
                and self._consensus_danger_index_pool is not None
                and self._consensus_stratified_pool_size == pool_size):
            return

        consensus_mode = getattr(
            self.network, 'speed_constraint_consensus_mode', 'single')
        if consensus_mode not in (
                'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8',
                *ROUTE_CHART_MODES):
            raise ValueError(
                "consensus-balanced sampling is registered only for the "
                "head-wise semantic correction or configuration-chart "
                "route experts")

        support_parts = []
        danger_parts = []
        with torch.no_grad():
            for start in range(0, pool_size, chunk_size):
                stop = min(start + chunk_size, pool_size)
                states = independent_pool[start:stop]
                targets = independent_labels[start:stop]
                direct = self.network.predict_uncomposed_speed(states)[:, 0]
                components = self.network.predict_repaired_speed_components(
                    states)
                semantic = (
                    self.network._aggregate_speed_constraint_components(
                        components)[:, 0])
                _, support, _ = _aggregate_corridor_consensus_target(
                    semantic, direct, targets, threshold=threshold)
                danger = _corridor_dangerous_negative_mask(
                    semantic, direct, targets, threshold=threshold)
                if bool(support.any()):
                    support_parts.append(
                        torch.nonzero(support, as_tuple=False)[:, 0] + start)
                if bool(danger.any()):
                    danger_parts.append(
                        torch.nonzero(danger, as_tuple=False)[:, 0] + start)

        if not support_parts or not danger_parts:
            raise RuntimeError(
                "consensus-balanced scalar sampler found an empty support "
                "or dangerous stratum")
        self._consensus_support_index_pool = torch.cat(support_parts, dim=0)
        self._consensus_danger_index_pool = torch.cat(danger_parts, dim=0)
        self._consensus_stratified_pool_size = pool_size
        if consensus_mode in ROUTE_CHART_MODES:
            chart_stats = self.network.fit_speed_constraint_route_charts(
                independent_pool[self._consensus_support_index_pool])
            print(
                "[scalar_consensus_sampler] fitted training-only "
                "configuration charts: "
                f"counts={chart_stats['counts']} "
                f"temperature={chart_stats['temperature']:.9g}")
        print(
            "[scalar_consensus_sampler] cached training-only strata: "
            f"support={self._consensus_support_index_pool.numel()} "
            f"dangerous={self._consensus_danger_index_pool.numel()} "
            f"total={pool_size}")

    def Loss(self, points, speed_cup, speed_pour, normal_cup, normal_pour,
             ee_y, d_orient_cup, d_orient_pour, zone_id, beta, gamma, epoch,
             ee_xy=None,
             tilt_grad=None,
             sample_weight=None,
             handover_grad=None,
             fk_features=None,
             d_obstacle=None,
             d_self=None,
             collocation_points=None,
             collocation_speed_labels=None,
             collocation_speed_component_labels=None,
             pair_type=None, path_loss_cfg=None,
             segment_barrier_cfg=None,
             line_preserve_cfg=None):
        """Multi-task eikonal loss with zone-masked routing.

        Zone-masked loss: each subnet only learns from data in its region.
        - pair_type=0 (intra-cupping): force w_cup=1, w_pour=0, use speed_cup
        - pair_type=1 (intra-pouring): force w_cup=0, w_pour=1, use speed_pour
        - pair_type=2 (cross): use spatial routing weights, zone-appropriate speed

        Args:
            points: (B, 2*dim) start-goal joint config pairs
            speed_cup: (B, 2) cupping speed for [start, goal]
            speed_pour: (B, 2) pouring speed for [start, goal]
            normal_cup: (B, 2*dim) cupping distance field gradient
            normal_pour: (B, 2*dim) pouring distance field gradient
            ee_y: (B, 2) EE Y-position for [start, goal]
            d_orient_cup: (B, 2) cupping orientation distance for [start, goal]
            d_orient_pour: (B, 2) pouring orientation distance for [start, goal]
            zone_id: (B, 2) zone assignment (unused for now, routing is soft)
            beta: scalar loss scaling
            gamma: scalar (unused legacy)
            epoch: current epoch number
            pair_type: (B,) int tensor, 0=intra-cup, 1=intra-pour, 2=cross
            collocation_speed_labels: (M,) scalar speed labels matching the
                independent collocation cloud, when available
            collocation_speed_component_labels: (M,H) signed q-only factors
                whose analytic normalized soft minimum is the scalar label

        Returns:
            loss, loss_n, diff (same interface as single-task Function.Loss)
        """
        import os as _nohack_os
        bad_teacher_env = []
        for key, value in _nohack_os.environ.items():
            if not key.startswith("R58_TEACHER_"):
                continue
            raw = str(value).strip()
            if raw == "" or raw.lower() in {"0", "0.0", "false", "none", "null"}:
                continue
            bad_teacher_env.append(f"{key}={value}")
        if bad_teacher_env:
            raise RuntimeError(
                "Teacher trajectory/direction losses are disabled by "
                "reports/ur5e-no-hack-policy.md. Remove: "
                + ", ".join(sorted(bad_teacher_env)))
        B = points.shape[0]

        # High-order PDE terms can encounter an isolated exploding derivative
        # before the network output itself becomes non-finite.  Keep the
        # default path bit-for-bit compatible (clip=0); a retest may opt into
        # a finite local derivative cap as a numerical stabilizer.  The cap is
        # never a data target and is recorded in the training command/report.
        derivative_clip_raw = _nohack_os.environ.get(
            'R58_DERIVATIVE_CLIP', '0')
        try:
            derivative_clip = float(derivative_clip_raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"R58_DERIVATIVE_CLIP must be numeric, got "
                f"{derivative_clip_raw!r}")

        def _finite_derivative(value):
            if derivative_clip <= 0.0:
                return value
            return torch.nan_to_num(
                value, nan=0.0, posinf=derivative_clip,
                neginf=-derivative_clip).clamp(
                    -derivative_clip, derivative_clip)


        # A pure scalar run deliberately drops pair/task routing from the loss.
        # The compatibility cup/pour speed arrays are identical copies, so the
        # only remaining target is one q-only scalar F(q).
        if self.pure_scalar_q_only:
            pair_type = None

        # --- Per-point routing weights (2B: B starts + B goals) ---
        ee_y_stacked = torch.cat([ee_y[:, 0], ee_y[:, 1]], dim=0)  # (2B,)
        ee_xy_stacked = None
        if ee_xy is not None:
            # ee_xy layout: [start_x, start_y, goal_x, goal_y]
            ee_xy_stacked = torch.cat([ee_xy[:, 0:2], ee_xy[:, 2:4]], dim=0)
        d_cup_stacked = torch.cat([d_orient_cup[:, 0], d_orient_cup[:, 1]], dim=0)
        d_pour_stacked = torch.cat([d_orient_pour[:, 0], d_orient_pour[:, 1]], dim=0)

        if self.pure_scalar_q_only:
            w_cup_2B = torch.ones(
                2 * B, device=points.device, dtype=points.dtype)
            w_pour_2B = torch.zeros_like(w_cup_2B)
        else:
            w_cup_2B, w_pour_2B = self.network.compute_window_weights(
                ee_y_stacked, d_cup_stacked, d_pour_stacked,
                ee_xy=ee_xy_stacked)

        if pair_type is not None:
            is_cup = (pair_type == 0)
            is_pour = (pair_type == 1)
            is_cross = (pair_type == 2)
            # Override both start and goal for intra-zone pairs
            w_cup_2B[:B][is_cup] = 1.0;  w_pour_2B[:B][is_cup] = 0.0
            w_cup_2B[B:][is_cup] = 1.0;  w_pour_2B[B:][is_cup] = 0.0
            w_cup_2B[:B][is_pour] = 0.0;  w_pour_2B[:B][is_pour] = 1.0
            w_cup_2B[B:][is_pour] = 0.0;  w_pour_2B[B:][is_pour] = 1.0

            # Cross-zone routing modes
            if self.cross_mode == 'hard_cross':
                # Force start=cup, goal=pour for cross pairs
                w_cup_2B[:B][is_cross] = 1.0;  w_pour_2B[:B][is_cross] = 0.0
                w_cup_2B[B:][is_cross] = 0.0;  w_pour_2B[B:][is_cross] = 1.0
            elif self.cross_mode == 'sharp':
                # Very sharp transition (sigmoid scale=0.02) for cross pairs
                # This diagnostic remains Y-boundary based; corner-table runs should
                # prefer the default or zone_cross modes.
                y_div = self.network.y_divide
                ee_y_s = ee_y[:, 0]
                ee_y_g = ee_y[:, 1]
                sharp_scale = 0.02
                cup_side = getattr(self.network, 'cup_side', -1.0)
                w_s = torch.sigmoid(cup_side * (ee_y_s - y_div) / sharp_scale)
                w_g = torch.sigmoid(cup_side * (ee_y_g - y_div) / sharp_scale)
                w_cup_2B[:B][is_cross] = w_s[is_cross]
                w_pour_2B[:B][is_cross] = 1.0 - w_s[is_cross]
                w_cup_2B[B:][is_cross] = w_g[is_cross]
                w_pour_2B[B:][is_cross] = 1.0 - w_g[is_cross]
            elif self.cross_mode == 'half_cross':
                # Start uses routing, goal forced to pour
                w_cup_2B[B:][is_cross] = 0.0;  w_pour_2B[B:][is_cross] = 1.0
            elif self.cross_mode == 'cross_equal_weights':
                # Force w_cup=w_pour=0.5 at BOTH endpoints for cross pairs
                # Both encoders contribute equally → teaches both to handle transitions
                w_cup_2B[:B][is_cross] = 0.5;  w_pour_2B[:B][is_cross] = 0.5
                w_cup_2B[B:][is_cross] = 0.5;  w_pour_2B[B:][is_cross] = 0.5
            elif self.cross_mode == 'zone_cross':
                # Cross pairs can be cup->pour or pour->cup. Use the per-endpoint
                # zone labels so each endpoint is supervised by its active task.
                start_zone = zone_id[:, 0].long()
                goal_zone = zone_id[:, 1].long()
                start_cup = is_cross & (start_zone == 0)
                start_pour = is_cross & (start_zone == 1)
                goal_cup = is_cross & (goal_zone == 0)
                goal_pour = is_cross & (goal_zone == 1)
                w_cup_2B[:B][start_cup] = 1.0;  w_pour_2B[:B][start_cup] = 0.0
                w_cup_2B[:B][start_pour] = 0.0; w_pour_2B[:B][start_pour] = 1.0
                w_cup_2B[B:][goal_cup] = 1.0;   w_pour_2B[B:][goal_cup] = 0.0
                w_cup_2B[B:][goal_pour] = 0.0;  w_pour_2B[B:][goal_pour] = 1.0
            # 'default': use compute_window_weights as-is (no override for cross)

            # --- Cross speed override modes ---
            # These modify the TARGET speed for cross pairs, not the routing
            if self.cross_mode == 'cross_maxspeed':
                # Use max(speed_cup, speed_pour) for cross pairs → relax orientation
                cross_speed_s = torch.max(speed_cup[:, 0][is_cross], speed_pour[:, 0][is_cross])
                cross_speed_g = torch.max(speed_cup[:, 1][is_cross], speed_pour[:, 1][is_cross])
                speed_cup[:, 0][is_cross] = cross_speed_s
                speed_pour[:, 0][is_cross] = cross_speed_s
                speed_cup[:, 1][is_cross] = cross_speed_g
                speed_pour[:, 1][is_cross] = cross_speed_g
            elif self.cross_mode == 'cross_minspeed':
                # Use min(speed_cup, speed_pour) for cross pairs → enforce BOTH constraints
                # Only configs satisfying both cupping AND pouring get high speed
                cross_speed_s = torch.min(speed_cup[:, 0][is_cross], speed_pour[:, 0][is_cross])
                cross_speed_g = torch.min(speed_cup[:, 1][is_cross], speed_pour[:, 1][is_cross])
                speed_cup[:, 0][is_cross] = cross_speed_s
                speed_pour[:, 0][is_cross] = cross_speed_s
                speed_cup[:, 1][is_cross] = cross_speed_g
                speed_pour[:, 1][is_cross] = cross_speed_g
            elif self.cross_mode == 'cross_highspeed':
                # Force speed=1.0 for cross pairs → pure obstacle avoidance in blend
                speed_cup[:, 0][is_cross] = 1.0; speed_cup[:, 1][is_cross] = 1.0
                speed_pour[:, 0][is_cross] = 1.0; speed_pour[:, 1][is_cross] = 1.0
            elif self.cross_mode == 'cross_downweight':
                # Don't modify speed, but mark cross pairs for loss downweighting
                # (handled below in loss computation)
                pass
            elif self.cross_mode == 'no_cross':
                # Skip cross pairs entirely — zero out their loss contribution
                pass

        # Extract start weights for speed/normal blending
        w_cup_start = w_cup_2B[:B]
        w_pour_start = w_pour_2B[:B]

        # Forward pass with per-point (2B) routing
        tau, w, Xp, f_cup, f_pour = self.network.out(
            points, ee_y=ee_y_stacked,
            d_orient_cup=d_cup_stacked, d_orient_pour=d_pour_stacked,
            ee_xy=ee_xy_stacked,
            w_cup_override=w_cup_2B, w_pour_override=w_pour_2B,
            fk_features=fk_features)
        # R62: grab the endpoint slowness from THIS forward pass immediately --
        # Loss calls self.network.out() 21 times (temporal, coherent, stencil,
        # characteristic ...), so reading the cache later picks up a different
        # batch of points and regresses them against the wrong labels.
        _ep_slowness = getattr(self.network, "_last_endpoint_slowness", None)
        # R65: capture the INTERIOR cache here too, for exactly the reason above --
        # reading it later would pick up one of the other 20 out() calls, i.e. the
        # wrong points and a graph that carries no useful gradient.  (Measured
        # 2026-09-08: reading it at loss-assembly time made the term inert, and two
        # different weights produced byte-identical evals at baseline level.)
        _in_slowness = getattr(self.network, "_last_interior_slowness", None)
        _in_q = getattr(self.network, "_last_interior_q", None)
        dtau = _finite_derivative(self.gradient(tau, Xp))

        # Blend speed using start weights (speed is per-endpoint)
        # Blend speed: start uses start weights, goal uses goal weights
        w_cup_goal = w_cup_2B[B:]
        w_pour_goal = w_pour_2B[B:]
        Yobs = torch.zeros_like(speed_cup)
        Yobs[:, 0] = w_cup_start * speed_cup[:, 0] + w_pour_start * speed_pour[:, 0]
        Yobs[:, 1] = w_cup_goal * speed_cup[:, 1] + w_pour_goal * speed_pour[:, 1]

        # AND-logic: in overlap zone, override speed target for cross pairs
        # and_blend controls how speed_cup and speed_pour are combined
        if self.and_overlap > 0 and pair_type is not None:
            y_div = self.network.y_divide
            lower = y_div - self.and_overlap
            upper = y_div + self.and_overlap
            is_cross = (pair_type == 2)
            blend = getattr(self, 'and_blend', 'min')
            for col, ey_col in [(0, ee_y[:, 0]), (1, ee_y[:, 1])]:
                mask = is_cross & (ey_col >= lower) & (ey_col <= upper)
                if not mask.any():
                    continue
                sc = speed_cup[:, col][mask]
                sp = speed_pour[:, col][mask]
                if blend == 'min':
                    Yobs[:, col][mask] = torch.min(sc, sp)
                elif blend == 'geomean':
                    Yobs[:, col][mask] = torch.sqrt(sc * sp + 1e-8)
                elif blend == 'avg':
                    Yobs[:, col][mask] = 0.5 * sc + 0.5 * sp
                elif blend == 'harmonic':
                    Yobs[:, col][mask] = 2.0 * sc * sp / (sc + sp + 1e-8)
                elif blend == 'max':
                    # Optimistic: go fast if EITHER constraint is satisfied
                    Yobs[:, col][mask] = torch.max(sc, sp)
                elif blend == 'const1':
                    # Treat overlap as free space for cross pairs
                    Yobs[:, col][mask] = 1.0
                elif blend == 'const05':
                    # Fixed medium speed in overlap
                    Yobs[:, col][mask] = 0.5
                elif blend == 'max_floor03':
                    # min but floored at 0.3
                    Yobs[:, col][mask] = torch.clamp(torch.min(sc, sp), min=0.3)
                elif blend == 'y_interp':
                    # Smooth position-based: cup_speed near cup side, pour_speed near pour side
                    t = (ey_col[mask] - lower) / (upper - lower + 1e-8)
                    t = torch.clamp(t, 0.0, 1.0)
                    Yobs[:, col][mask] = (1.0 - t) * sc + t * sp

        clearance_target = None
        if (getattr(self.network, 'value_arch', None) ==
                'bounded_chord_clearance_eikonal_symmetric'):
            if d_obstacle is None or d_self is None:
                raise ValueError(
                    "clearance-Eikonal propagation requires scalar "
                    "d_obstacle and d_self arrays")
            # The only new supervision is scalar signed clearance.  Convert
            # it to the same positive speed used by the model, then intersect
            # it with the existing task speed.  No normal, metric tensor,
            # collision critic, or trajectory label enters the PDE.
            physical_clearance = torch.minimum(
                d_obstacle, self.clearance_self_scale * d_self)
            clearance_target = (
                physical_clearance
                / float(self.network.clearance_scale)).clamp(-1.0, 1.0)
            clearance_speed_target = self.network.clearance_to_speed(
                clearance_target)
            Yobs = torch.minimum(Yobs, clearance_speed_target)

        if getattr(self.network, 'lifted_phase', False):
            # The lifted dataset may store only the scalar speed of the new
            # phase restriction.  The deployed field is its monotone
            # intersection with the inherited q-only field, so every loss
            # that represents the same Eikonal problem must use that same
            # intersection -- not just the pointwise speed-regression term.
            # In particular, using phase-only labels below for the PDE would
            # incorrectly declare inherited physical obstacles to be free.
            # The base prediction is detached because it is a fixed boundary
            # condition for the new phase capacity, not a second teacher or
            # an additional deployed output.
            endpoint_points_for_target = torch.cat([
                Xp[:, :self.dim], Xp[:, self.dim:]], dim=0)
            with torch.no_grad():
                base_speed_2B = self.network.predict_base_speed(
                    endpoint_points_for_target, features=f_cup)[:, 0]
            Yobs = torch.minimum(
                Yobs,
                torch.stack([
                    base_speed_2B[:B], base_speed_2B[B:]], dim=1))

        # Blend normal: start uses start weights, goal uses goal weights
        normal = (w_cup_start.unsqueeze(1) * normal_cup[:, :self.dim] +
                  w_pour_start.unsqueeze(1) * normal_pour[:, :self.dim])
        normal_goal = (w_cup_goal.unsqueeze(1) * normal_cup[:, self.dim:] +
                       w_pour_goal.unsqueeze(1) * normal_pour[:, self.dim:])
        normal_blended = torch.cat([normal, normal_goal], dim=1)  # (B, 2*dim)

        speed_supervision_loss = torch.zeros(B, device=tau.device, dtype=tau.dtype)
        clearance_supervision_loss = torch.zeros(
            B, device=tau.device, dtype=tau.dtype)
        collocation_speed_loss = torch.zeros(
            (), device=tau.device, dtype=tau.dtype)
        collocation_speed_mae = torch.zeros_like(collocation_speed_loss)
        collocation_speed_rmse = torch.zeros_like(collocation_speed_loss)
        collocation_speed_corr = torch.zeros_like(collocation_speed_loss)
        collocation_speed_slow_recall = torch.zeros_like(
            collocation_speed_loss)
        collocation_speed_component_loss = torch.zeros_like(
            collocation_speed_loss)
        collocation_speed_component_mae = torch.zeros_like(
            collocation_speed_loss)
        collocation_speed_component_corr_mean = torch.zeros_like(
            collocation_speed_loss)
        collocation_speed_component_corr_min = torch.zeros_like(
            collocation_speed_loss)
        collocation_speed_uncomposed_loss = torch.zeros_like(
            collocation_speed_loss)
        collocation_speed_uncomposed_mae = torch.zeros_like(
            collocation_speed_loss)
        collocation_speed_uncomposed_corr = torch.zeros_like(
            collocation_speed_loss)
        collocation_speed_consensus_identification_loss = torch.zeros_like(
            collocation_speed_loss)
        collocation_speed_consensus_alpha_mae = torch.zeros_like(
            collocation_speed_loss)
        collocation_speed_consensus_alpha_target_mean = torch.zeros_like(
            collocation_speed_loss)
        collocation_speed_consensus_alpha_prediction_mean = torch.zeros_like(
            collocation_speed_loss)
        collocation_speed_consensus_identified_fraction = torch.zeros_like(
            collocation_speed_loss)
        collocation_speed_consensus_release_bce = torch.zeros_like(
            collocation_speed_loss)
        collocation_speed_consensus_magnitude_mse = torch.zeros_like(
            collocation_speed_loss)
        collocation_speed_consensus_negative_magnitude_mse = torch.zeros_like(
            collocation_speed_loss)
        collocation_speed_consensus_specificity_bce = torch.zeros_like(
            collocation_speed_loss)
        collocation_speed_consensus_release_target_mean = torch.zeros_like(
            collocation_speed_loss)
        collocation_speed_consensus_release_prediction_mean = torch.zeros_like(
            collocation_speed_loss)
        collocation_speed_consensus_magnitude_target_mean = torch.zeros_like(
            collocation_speed_loss)
        collocation_speed_consensus_magnitude_prediction_mean = (
            torch.zeros_like(collocation_speed_loss))
        collocation_speed_consensus_negative_fraction = torch.zeros_like(
            collocation_speed_loss)
        collocation_speed_consensus_negative_magnitude_prediction_mean = (
            torch.zeros_like(collocation_speed_loss))
        collocation_speed_consensus_effective_release_mse = (
            torch.zeros_like(collocation_speed_loss))
        collocation_speed_consensus_effective_release_target_mean = (
            torch.zeros_like(collocation_speed_loss))
        collocation_speed_consensus_effective_release_prediction_mean = (
            torch.zeros_like(collocation_speed_loss))
        collocation_speed_consensus_negative_effective_release_mse = (
            torch.zeros_like(collocation_speed_loss))
        collocation_speed_consensus_negative_effective_release_prediction_mean = (
            torch.zeros_like(collocation_speed_loss))
        collocation_speed_consensus_specificity_positive_mean = (
            torch.zeros_like(collocation_speed_loss))
        collocation_speed_consensus_specificity_negative_mean = (
            torch.zeros_like(collocation_speed_loss))
        collocation_speed_repair_identification_loss = torch.zeros_like(
            collocation_speed_loss)
        collocation_speed_repair_rho_mae = torch.zeros_like(
            collocation_speed_loss)
        collocation_speed_repair_release_bce = torch.zeros_like(
            collocation_speed_loss)
        collocation_speed_repair_magnitude_mse = torch.zeros_like(
            collocation_speed_loss)
        collocation_speed_repair_support_target_mean = torch.zeros_like(
            collocation_speed_loss)
        collocation_speed_repair_release_prediction_mean = torch.zeros_like(
            collocation_speed_loss)
        collocation_speed_repair_magnitude_target_mean = torch.zeros_like(
            collocation_speed_loss)
        collocation_speed_repair_magnitude_prediction_mean = torch.zeros_like(
            collocation_speed_loss)
        collocation_speed_repair_coordination_beta_mean = torch.zeros_like(
            collocation_speed_loss)
        collocation_speed_repair_coordination_zero_fraction = torch.zeros_like(
            collocation_speed_loss)
        speed_pred_2B = None
        speed_floor = float(getattr(self.network, 'speed_floor', 0.1))

        def _speed_regression_coordinate(value, form=None):
            """Map scalar F to the coordinate used by its regression loss."""
            selected_form = self.speed_loss_form if form in (None, 'inherit') else form
            value = value.clamp_min(speed_floor)
            if selected_form == 'log_speed':
                scale = -math.log(max(speed_floor, 1e-8))
                return -torch.log(value) / max(scale, 1e-8)
            if selected_form == 'slowness':
                scale = 1.0 / max(speed_floor, 1e-8) - 1.0
                return (value.reciprocal() - 1.0) / max(scale, 1e-8)
            return value

        if getattr(self.network, 'use_speed_head', False):
            endpoint_points = torch.cat([
                Xp[:, :self.dim], Xp[:, self.dim:]], dim=0)
            speed_pred_2B = self.network.predict_speed(
                endpoint_points, features=f_cup)
            if self.speed_loss_weight > 0:
                if self.speed_loss_balance_power > 0:
                    speed_weight_start = Yobs[:, 0].clamp_min(speed_floor).pow(
                        -self.speed_loss_balance_power)
                    speed_weight_goal = Yobs[:, 1].clamp_min(speed_floor).pow(
                        -self.speed_loss_balance_power)
                    weight_mean = 0.5 * (
                        speed_weight_start.mean()
                        + speed_weight_goal.mean())
                    speed_weight_start = speed_weight_start / weight_mean
                    speed_weight_goal = speed_weight_goal / weight_mean
                else:
                    speed_weight_start = 1.0
                    speed_weight_goal = 1.0

                pred_coordinate = _speed_regression_coordinate(
                    speed_pred_2B[:, 0])
                target_speed_start = Yobs[:, 0]
                target_speed_goal = Yobs[:, 1]
                target_start = _speed_regression_coordinate(
                    target_speed_start)
                target_goal = _speed_regression_coordinate(
                    target_speed_goal)
                speed_supervision_loss = self.speed_loss_weight * (
                    speed_weight_start
                    * (pred_coordinate[:B] - target_start).square()
                    + speed_weight_goal
                    * (pred_coordinate[B:] - target_goal).square()
                )

        # Independent scalar-speed supervision.  The collocation cloud is a
        # q-only measure, so its labels are not attached to an endpoint pair
        # and must be sampled explicitly here.  This is deliberately kept
        # separate from path/value supervision: it adds no trajectory,
        # waypoint, critic, or evaluator signal.
        if (self.collocation_speed_loss_weight > 0.0
                or self.collocation_speed_component_loss_weight > 0.0
                or self.collocation_speed_uncomposed_loss_weight > 0.0
                or self.collocation_speed_consensus_identification_weight
                > 0.0
                or self.collocation_speed_repair_identification_weight
                > 0.0):
            if not getattr(self.network, 'use_speed_head', False):
                raise RuntimeError(
                    "collocation speed supervision requires the scalar "
                    "speed head")
            if collocation_points is None or collocation_speed_labels is None:
                raise RuntimeError(
                    "collocation speed supervision requires both "
                    "collocation_points and collocation_speed.npy")
            independent_pool = collocation_points.to(
                device=tau.device, dtype=Xp.dtype)
            independent_labels = collocation_speed_labels.to(
                device=tau.device, dtype=Xp.dtype).reshape(-1)
            if (independent_pool.ndim != 2
                    or independent_pool.shape[1] != self.dim
                    or independent_pool.shape[0] != independent_labels.numel()):
                raise ValueError(
                    "collocation_points and collocation_speed labels must "
                    "have shapes (M, dim) and (M,)")
            n_collocation_speed = min(
                self.collocation_speed_n, independent_pool.shape[0])
            if n_collocation_speed < 1:
                raise RuntimeError("collocation speed cloud is empty")
            # Optional scalar-only importance sampling for the narrow
            # state-constraint boundary.  The default is the historical
            # uniform measure.  When enabled, rows are sampled with a smooth
            # inverse-speed power; it uses only the same scalar target F(q),
            # never pair/task labels, FK, paths, or a collision oracle.
            hard_sampling_power = float(_nohack_os.environ.get(
                "PURE_SCALAR_COLLOCATION_HARD_POWER", "0.0"))
            if self.collocation_speed_sampling == 'consensus_balanced':
                if hard_sampling_power > 0.0:
                    raise ValueError(
                        "consensus-balanced and inverse-speed collocation "
                        "sampling are isolated protocols")
                threshold = max(
                    2.0 * speed_floor, speed_floor + 1e-6)
                self._build_consensus_stratified_pools(
                    independent_pool, independent_labels, threshold)
                speed_idx = _balanced_index_pool_sample(
                    self._consensus_support_index_pool,
                    self._consensus_danger_index_pool,
                    n_collocation_speed)
            elif hard_sampling_power > 0.0:
                sampling_speed = independent_labels.clamp_min(speed_floor)
                sampling_weight = (
                    speed_floor / sampling_speed).pow(hard_sampling_power)
                sampling_weight = sampling_weight / sampling_weight.sum()
                speed_idx = torch.multinomial(
                    sampling_weight, n_collocation_speed, replacement=False)
            else:
                speed_idx = torch.randperm(
                    independent_pool.shape[0], device=tau.device
                )[:n_collocation_speed]
            speed_states = independent_pool[speed_idx].detach()
            speed_target = independent_labels[speed_idx].clamp_min(speed_floor)
            speed_pred = self.network.predict_propagation_speed(
                speed_states)[:, 0]
            speed_pred_coord = _speed_regression_coordinate(
                speed_pred, self.collocation_speed_loss_form)
            speed_target_coord = _speed_regression_coordinate(
                speed_target, self.collocation_speed_loss_form)
            if self.collocation_speed_loss_balance_power > 0.0:
                speed_weight = speed_target.clamp_min(speed_floor).pow(
                    -self.collocation_speed_loss_balance_power)
                speed_weight = speed_weight / speed_weight.mean().clamp_min(1e-8)
            else:
                speed_weight = torch.ones_like(speed_target)
            collocation_speed_loss = self.collocation_speed_loss_weight * (
                speed_weight * (speed_pred_coord - speed_target_coord).square()
            ).mean()
            with torch.no_grad():
                error = speed_pred - speed_target
                collocation_speed_mae = error.abs().mean()
                collocation_speed_rmse = error.square().mean().sqrt()
                pred_centered = speed_pred - speed_pred.mean()
                target_centered = speed_target - speed_target.mean()
                corr_den = (
                    pred_centered.square().sum().sqrt()
                    * target_centered.square().sum().sqrt())
                collocation_speed_corr = torch.where(
                    corr_den > 1e-8,
                    (pred_centered * target_centered).sum() / corr_den,
                    torch.zeros_like(corr_den))
                slow_threshold = max(2.0 * speed_floor, speed_floor + 1e-6)
                slow_target = speed_target <= slow_threshold
                collocation_speed_slow_recall = torch.where(
                    slow_target.any(),
                    ((speed_pred <= slow_threshold) & slow_target).sum()
                    / slow_target.sum().clamp_min(1),
                    torch.zeros_like(collocation_speed_loss))

            # Identify the direct estimate independently of the consensus
            # alpha.  It uses the same exact aggregate F(q) labels, sampled
            # rows, regression coordinate, and weights as the deployed field.
            # predict_uncomposed_speed bypasses both semantic composition and
            # alpha, so this term cannot be minimized by moving the gate.
            if self.collocation_speed_uncomposed_loss_weight > 0.0:
                uncomposed_pred = self.network.predict_uncomposed_speed(
                    speed_states)[:, 0]
                uncomposed_pred_coord = _speed_regression_coordinate(
                    uncomposed_pred, self.collocation_speed_loss_form)
                collocation_speed_uncomposed_loss = (
                    self.collocation_speed_uncomposed_loss_weight
                    * (speed_weight * (
                        uncomposed_pred_coord - speed_target_coord
                    ).square()).mean())
                with torch.no_grad():
                    uncomposed_error = uncomposed_pred - speed_target
                    collocation_speed_uncomposed_mae = (
                        uncomposed_error.abs().mean())
                    uncomposed_centered = (
                        uncomposed_pred - uncomposed_pred.mean())
                    uncomposed_target_centered = (
                        speed_target - speed_target.mean())
                    uncomposed_corr_den = (
                        uncomposed_centered.square().sum().sqrt()
                        * uncomposed_target_centered.square().sum().sqrt())
                    collocation_speed_uncomposed_corr = torch.where(
                        uncomposed_corr_den > 1e-8,
                        (uncomposed_centered
                         * uncomposed_target_centered).sum()
                        / uncomposed_corr_den,
                        torch.zeros_like(uncomposed_corr_den))

            # When the two frozen scalar experts disagree, their exact best
            # convex coefficient is algebraically identified by the same
            # aggregate F(q) label already used above:
            #
            #   alpha* = clip((F* - F_sem) / (F_direct - F_sem), 0, 1).
            #
            # Ordinary final-output MSE implicitly multiplies alpha error by
            # the squared expert gap, which makes rare routing decisions easy
            # to ignore.  This term removes only that accidental conditioning;
            # it introduces no teacher, path, planner, or evaluation label.
            if self.collocation_speed_consensus_identification_weight > 0.0:
                consensus_mode = getattr(
                    self.network, 'speed_constraint_consensus_mode', 'single')
                with torch.no_grad():
                    consensus_direct = self.network.predict_uncomposed_speed(
                        speed_states)[:, 0]
                    if consensus_mode in (
                            'corridor_hurdle',
                            'corridor_hurdle_compensated',
                            'corridor_hurdle_compensated_balanced',
                            'corridor_hurdle_specificity',
                            'corridor_hurdle_specificity_odds',
                            'corridor_hurdle_specificity_orthogonal',
                            'corridor_hurdle_semantic_veto_min_orthogonal',
                            'corridor_hurdle_semantic_veto_min_odds',
                            'corridor_hurdle_semantic_veto_min_odds_residual',
                            'corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend',
                            'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8',
                            *ROUTE_CHART_MODES):
                        consensus_components = (
                            self.network.predict_repaired_speed_components(
                                speed_states))
                    else:
                        consensus_components = (
                            self.network.predict_speed_components(
                                speed_states))
                    consensus_semantic = (
                        self.network._aggregate_speed_constraint_components(
                            consensus_components)[:, 0])
                    if consensus_mode in (
                            'corridor_hurdle',
                            'corridor_hurdle_compensated',
                            'corridor_hurdle_compensated_balanced',
                            'corridor_hurdle_specificity',
                            'corridor_hurdle_specificity_odds',
                            'corridor_hurdle_specificity_orthogonal',
                            'corridor_hurdle_semantic_veto_min_orthogonal',
                            'corridor_hurdle_semantic_veto_min_odds',
                            'corridor_hurdle_semantic_veto_min_odds_residual',
                            'corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend',
                            'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8',
                            *ROUTE_CHART_MODES):
                        (consensus_alpha_target, _,
                         consensus_identified) = (
                            _aggregate_corridor_consensus_target(
                                consensus_semantic, consensus_direct,
                                speed_target,
                                threshold=max(
                                    2.0 * speed_floor,
                                    speed_floor + 1e-6)))
                        consensus_dangerous_negative = (
                            _corridor_dangerous_negative_mask(
                                consensus_semantic, consensus_direct,
                                speed_target,
                                threshold=max(
                                    2.0 * speed_floor,
                                    speed_floor + 1e-6)))
                    else:
                        consensus_gap = (
                            consensus_direct - consensus_semantic)
                        consensus_identified = consensus_gap.abs() > 1e-8
                        safe_gap = torch.where(
                            consensus_identified,
                            consensus_gap,
                            torch.ones_like(consensus_gap))
                        consensus_alpha_target = torch.clamp(
                            (speed_target - consensus_semantic) / safe_gap,
                            min=0.0,
                            max=1.0)
                if consensus_mode in (
                        'hurdle', 'corridor_hurdle',
                        'corridor_hurdle_compensated',
                        'corridor_hurdle_compensated_balanced',
                        'corridor_hurdle_specificity',
                        'corridor_hurdle_specificity_odds',
                        'corridor_hurdle_specificity_orthogonal',
                        'corridor_hurdle_semantic_veto_min_orthogonal',
                        'corridor_hurdle_semantic_veto_min_odds',
                        'corridor_hurdle_semantic_veto_min_odds_residual',
                        'corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend',
                        'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8',
                        *ROUTE_CHART_MODES):
                    consensus_factors = (
                        self.network.predict_speed_consensus_factors(
                            speed_states))
                    consensus_release_prediction = consensus_factors[:, 0]
                    if consensus_mode in (
                            'corridor_hurdle_specificity',
                            'corridor_hurdle_specificity_odds',
                            'corridor_hurdle_specificity_orthogonal',
                            'corridor_hurdle_semantic_veto_min_orthogonal',
                            'corridor_hurdle_semantic_veto_min_odds',
                            'corridor_hurdle_semantic_veto_min_odds_residual',
                            'corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend',
                            'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8',
                            *ROUTE_CHART_MODES):
                        if consensus_mode in (
                                'corridor_hurdle_semantic_veto_min_orthogonal',
                                'corridor_hurdle_semantic_veto_min_odds',
                                'corridor_hurdle_semantic_veto_min_odds_residual',
                                'corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend',
                                'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8',
                                *ROUTE_CHART_MODES):
                            specificity_count = int(getattr(
                                self.network,
                                'speed_constraint_head_count', 0))
                            consensus_specificity_prediction = (
                                consensus_factors[
                                    :, 1:1 + specificity_count])
                            consensus_magnitude_prediction = (
                                consensus_factors[:, 1 + specificity_count])
                            weakest_permission = (
                                consensus_specificity_prediction.min(
                                    dim=1).values)
                            if consensus_mode in (
                                    'corridor_hurdle_semantic_veto_min_odds',
                                    'corridor_hurdle_semantic_veto_min_odds_residual',
                                    'corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend',
                                    'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8',
                                    *ROUTE_CHART_MODES):
                                consensus_effective_release = (
                                    neutral_log_odds_fusion(
                                        consensus_release_prediction,
                                        weakest_permission))
                                if consensus_mode in ROUTE_CHART_MODES:
                                    route_offset = 2 + specificity_count
                                    consensus_route_release_prediction = (
                                        consensus_factors[
                                            :,
                                            route_offset:
                                            route_offset + ROUTE_CHART_COUNT])
                                    consensus_route_gate = (
                                        consensus_factors[
                                            :,
                                            route_offset + ROUTE_CHART_COUNT:
                                            route_offset
                                            + 2 * ROUTE_CHART_COUNT])
                                    if consensus_mode in ROUTE_CHART_UNION_MODES:
                                        consensus_effective_release = (
                                            normalized_release_smoothmax(
                                                consensus_route_release_prediction
                                            )[:, 0])
                                    else:
                                        consensus_effective_release = (
                                            consensus_route_gate
                                            * consensus_route_release_prediction
                                        ).sum(dim=1)
                                    if (consensus_mode
                                            in ROUTE_CHART_FOURIER_RISK_MODES):
                                        semantic_offset = (
                                            route_offset
                                            + 2 * ROUTE_CHART_COUNT)
                                        semantic_logits = consensus_factors[
                                            :,
                                            semantic_offset:
                                            semantic_offset
                                            + ROUTE_CHART_FOURIER_RISK_HEAD_COUNT]
                                        semantic_evidence = semantic_logits.min(
                                            dim=1).values
                                        if consensus_mode == (
                                                ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_REPLICA_MEDIAN_ANCHOR_RESCUE_CONSENSUS_MODE):
                                            ch_release = (
                                                margin_signed_release_rescue(
                                                    consensus_effective_release,
                                                    semantic_evidence))
                                            anchor_speed = (
                                                self.network
                                                ._predict_speed_constraint_anchor(
                                                    speed_states)[:, 0])
                                            anchor_release = (
                                                scalar_speed_to_release(
                                                    anchor_speed,
                                                    consensus_semantic,
                                                    consensus_direct,
                                                    consensus_magnitude_prediction))
                                            reliability_offset = (
                                                semantic_offset
                                                + ROUTE_CHART_FOURIER_RISK_HEAD_COUNT)
                                            replica_logits = consensus_factors[
                                                :,
                                                reliability_offset:
                                                reliability_offset + 15]
                                            replica_logits = replica_logits.reshape(
                                                replica_logits.shape[0],
                                                3, 5).permute(1, 0, 2)
                                            consensus_effective_release, _, _ = (
                                                replica_median_positive_anchor_release(
                                                    ch_release[:, None],
                                                    anchor_release[:, None],
                                                    replica_logits))
                                            consensus_effective_release = (
                                                consensus_effective_release[:, 0])
                                        elif (consensus_mode ==
                                                ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_SEMANTIC_ANCHOR_RESCUE_CONSENSUS_MODE):
                                            anchor_speed = (
                                                self.network
                                                ._predict_speed_constraint_anchor(
                                                    speed_states)[:, 0])
                                            anchor_release = (
                                                scalar_speed_to_release(
                                                    anchor_speed,
                                                    consensus_semantic,
                                                    consensus_direct,
                                                    consensus_magnitude_prediction))
                                            consensus_effective_release = (
                                                semantic_approved_anchor_release(
                                                    consensus_effective_release,
                                                    anchor_release,
                                                    semantic_evidence))
                                            consensus_effective_release = (
                                                margin_signed_release_rescue(
                                                    consensus_effective_release,
                                                    semantic_evidence))
                                        elif (consensus_mode
                                                == ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_MARGIN_RESCUE_CONSENSUS_MODE):
                                            consensus_effective_release = (
                                                margin_signed_release_rescue(
                                                    consensus_effective_release,
                                                    semantic_evidence))
                                        else:
                                            consensus_effective_release = (
                                                monotone_release_veto(
                                                    consensus_effective_release,
                                                    semantic_evidence))
                                elif (consensus_mode
                                        == 'corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend'):
                                    smooth_permission = (
                                        normalized_logit_softmin_probability(
                                            consensus_specificity_prediction,
                                            temperature=1.0)[:, 0])
                                    smooth_release = neutral_log_odds_fusion(
                                        consensus_release_prediction,
                                        smooth_permission)
                                    consensus_evidence_blend_prediction = (
                                        consensus_factors[
                                            :, 2 + specificity_count])
                                    consensus_effective_release = torch.clamp(
                                        consensus_effective_release
                                        + consensus_evidence_blend_prediction
                                        * (smooth_release
                                           - consensus_effective_release),
                                        min=0.0, max=1.0)
                                elif (consensus_mode
                                        == 'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8'):
                                    consensus_head_logit_residual_prediction = (
                                        consensus_factors[
                                            :,
                                            2 + specificity_count:
                                            2 + 2 * specificity_count])
                                    corrected_specificity = torch.sigmoid(
                                        torch.logit(
                                            consensus_specificity_prediction.clamp(
                                                1e-6, 1.0 - 1e-6))
                                        + consensus_head_logit_residual_prediction)
                                    corrected_weakest_permission = (
                                        corrected_specificity.min(
                                            dim=1).values)
                                    consensus_effective_release = (
                                        neutral_log_odds_fusion(
                                            consensus_release_prediction,
                                            corrected_weakest_permission))
                            else:
                                consensus_effective_release = (
                                    consensus_release_prediction
                                    * weakest_permission)
                        else:
                            consensus_specificity_prediction = (
                                consensus_factors[:, 1])
                            consensus_magnitude_prediction = (
                                consensus_factors[:, 2])
                        if (consensus_mode
                                == 'corridor_hurdle_specificity_odds'):
                            consensus_effective_release = (
                                neutral_log_odds_fusion(
                                    consensus_release_prediction,
                                    consensus_specificity_prediction))
                        elif consensus_mode not in (
                                'corridor_hurdle_semantic_veto_min_orthogonal',
                                'corridor_hurdle_semantic_veto_min_odds',
                                'corridor_hurdle_semantic_veto_min_odds_residual',
                                'corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend',
                                'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8',
                                *ROUTE_CHART_MODES):
                            consensus_effective_release = (
                                consensus_release_prediction
                                * consensus_specificity_prediction)
                        consensus_alpha_prediction = (
                            consensus_effective_release
                            * consensus_magnitude_prediction)
                    else:
                        consensus_specificity_prediction = None
                        consensus_magnitude_prediction = consensus_factors[:, 1]
                        consensus_alpha_prediction = (
                            consensus_release_prediction
                            * consensus_magnitude_prediction)
                else:
                    consensus_release_prediction = None
                    consensus_specificity_prediction = None
                    consensus_magnitude_prediction = None
                    consensus_alpha_prediction = (
                        self.network.predict_speed_consensus_alpha(
                            speed_states)[:, 0])
                alpha_square_error = (
                    consensus_alpha_prediction
                    - consensus_alpha_target).square()
                if bool(consensus_identified.any()):
                    identified_error = alpha_square_error[
                        consensus_identified]
                    if consensus_mode in (
                            'hurdle', 'corridor_hurdle',
                            'corridor_hurdle_compensated',
                            'corridor_hurdle_compensated_balanced',
                            'corridor_hurdle_specificity',
                            'corridor_hurdle_specificity_odds',
                            'corridor_hurdle_specificity_orthogonal',
                            'corridor_hurdle_semantic_veto_min_orthogonal',
                            'corridor_hurdle_semantic_veto_min_odds',
                            'corridor_hurdle_semantic_veto_min_odds_residual',
                            'corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend',
                            'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8',
                            *ROUTE_CHART_MODES):
                        consensus_release_target = (
                            consensus_alpha_target > 0.0).to(
                                consensus_alpha_target.dtype)
                        identified_release_prediction = (
                            consensus_release_prediction[
                                consensus_identified])
                        identified_release_target = consensus_release_target[
                            consensus_identified]
                        collocation_speed_consensus_release_bce = (
                            F.binary_cross_entropy(
                                identified_release_prediction,
                                identified_release_target))
                        consensus_positive = (
                            consensus_identified
                            & (consensus_alpha_target > 0.0))
                        if bool(consensus_positive.any()):
                            positive_magnitude_prediction = (
                                consensus_magnitude_prediction[
                                    consensus_positive])
                            if consensus_mode in (
                                    'corridor_hurdle_compensated',
                                    'corridor_hurdle_compensated_balanced',
                                    'corridor_hurdle_specificity',
                                    'corridor_hurdle_specificity_odds',
                                    'corridor_hurdle_specificity_orthogonal',
                                    'corridor_hurdle_semantic_veto_min_orthogonal',
                                    'corridor_hurdle_semantic_veto_min_odds',
                                    'corridor_hurdle_semantic_veto_min_odds_residual',
                                    'corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend',
                                    'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8',
                                    *ROUTE_CHART_MODES):
                                compensation_factor = (
                                    consensus_release_prediction)
                                if (consensus_mode
                                        == 'corridor_hurdle_specificity'):
                                    compensation_factor = (
                                        compensation_factor
                                        * consensus_specificity_prediction)
                                elif (consensus_mode
                                        == 'corridor_hurdle_specificity_odds'):
                                    compensation_factor = (
                                        neutral_log_odds_fusion(
                                            compensation_factor,
                                            consensus_specificity_prediction))
                                elif consensus_mode in (
                                        'corridor_hurdle_semantic_veto_min_odds',
                                        'corridor_hurdle_semantic_veto_min_odds_residual'):
                                    compensation_factor = (
                                        neutral_log_odds_fusion(
                                            compensation_factor,
                                            consensus_specificity_prediction.min(
                                                dim=1).values))
                                elif (consensus_mode
                                        == 'corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend'):
                                    compensation_factor = (
                                        consensus_effective_release)
                                elif (consensus_mode
                                        == 'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8'):
                                    compensation_factor = (
                                        consensus_effective_release)
                                elif consensus_mode in ROUTE_CHART_MODES:
                                    compensation_factor = (
                                        consensus_effective_release)
                                compensated_magnitude_target = (
                                    _release_compensated_magnitude_target(
                                        consensus_alpha_target,
                                        compensation_factor,
                                        consensus_positive))
                                positive_magnitude_target = (
                                    compensated_magnitude_target[
                                        consensus_positive])
                            else:
                                positive_magnitude_target = (
                                    consensus_alpha_target[
                                        consensus_positive])
                            collocation_speed_consensus_magnitude_mse = (
                                (positive_magnitude_prediction
                                 - positive_magnitude_target).square().mean())
                            with torch.no_grad():
                                collocation_speed_consensus_magnitude_target_mean = (
                                    positive_magnitude_target.mean())
                                collocation_speed_consensus_magnitude_prediction_mean = (
                                    positive_magnitude_prediction.mean())
                        if consensus_mode in (
                                'corridor_hurdle_specificity',
                                'corridor_hurdle_specificity_odds',
                                'corridor_hurdle_specificity_orthogonal',
                                'corridor_hurdle_semantic_veto_min_orthogonal',
                                'corridor_hurdle_semantic_veto_min_odds',
                                'corridor_hurdle_semantic_veto_min_odds_residual',
                                'corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend',
                                'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8',
                                *ROUTE_CHART_MODES):
                            specificity_terms = []
                            if consensus_mode in (
                                    'corridor_hurdle_semantic_veto_min_orthogonal',
                                    'corridor_hurdle_semantic_veto_min_odds',
                                    'corridor_hurdle_semantic_veto_min_odds_residual',
                                    'corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend',
                                    'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8',
                                    *ROUTE_CHART_MODES):
                                if collocation_speed_component_labels is None:
                                    raise RuntimeError(
                                        "semantic specificity requires "
                                        "collocation_speed_components.npy")
                                semantic_component_labels = (
                                    collocation_speed_component_labels.to(
                                        device=tau.device, dtype=Xp.dtype))
                                specificity_count = int(getattr(
                                    self.network,
                                    'speed_constraint_head_count', 0))
                                if (semantic_component_labels.ndim != 2
                                        or semantic_component_labels.shape[0]
                                        != independent_pool.shape[0]
                                        or semantic_component_labels.shape[1]
                                        != specificity_count):
                                    raise ValueError(
                                        "semantic specificity component "
                                        "labels must have shape "
                                        f"(M,{specificity_count}), got "
                                        f"{tuple(semantic_component_labels.shape)}")
                                component_permission_target, _ = (
                                    _normalized_softmin_component_permission_target(
                                        semantic_component_labels[speed_idx],
                                        max(
                                            2.0 * speed_floor,
                                            speed_floor + 1e-6),
                                        float(getattr(
                                            self.network,
                                            'speed_constraint_softmin_temperature',
                                            0.04))))
                                component_permission = (
                                    component_permission_target > 0.5)
                                specificity_for_identification = (
                                    consensus_specificity_prediction)
                                if (consensus_mode
                                        == 'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8'):
                                    # The inherited semantic heads are frozen;
                                    # identify the three new corrections with
                                    # the exact component permissions rather
                                    # than reporting BCE on the unchanged base.
                                    head_residual = consensus_factors[
                                        :, 2 + specificity_count:
                                        2 + 2 * specificity_count]
                                    specificity_for_identification = (
                                        torch.sigmoid(
                                            torch.logit(
                                                consensus_specificity_prediction.clamp(
                                                    1e-6, 1.0 - 1e-6))
                                            + head_residual))
                                per_head_terms = []
                                for head_index in range(specificity_count):
                                    head_prediction = (
                                        specificity_for_identification[
                                            :, head_index])
                                    head_permission = (
                                        component_permission[:, head_index])
                                    class_terms = []
                                    if bool(head_permission.any()):
                                        positive_specificity = (
                                            head_prediction[head_permission])
                                        class_terms.append(
                                            F.binary_cross_entropy(
                                                positive_specificity,
                                                torch.ones_like(
                                                    positive_specificity)))
                                    head_veto = ~head_permission
                                    if bool(head_veto.any()):
                                        negative_specificity = (
                                            head_prediction[head_veto])
                                        class_terms.append(
                                            F.binary_cross_entropy(
                                                negative_specificity,
                                                torch.zeros_like(
                                                    negative_specificity)))
                                    if class_terms:
                                        per_head_terms.append(
                                            sum(class_terms) / len(class_terms))
                                if per_head_terms:
                                    specificity_terms.append(
                                        sum(per_head_terms)
                                        / len(per_head_terms))
                                with torch.no_grad():
                                    if bool(component_permission.any()):
                                        collocation_speed_consensus_specificity_positive_mean = (
                                            specificity_for_identification[
                                                component_permission].mean())
                                    component_veto = ~component_permission
                                    if bool(component_veto.any()):
                                        collocation_speed_consensus_specificity_negative_mean = (
                                            specificity_for_identification[
                                                component_veto].mean())
                                    collocation_speed_consensus_negative_fraction = (
                                        component_veto.to(
                                            speed_target.dtype).mean())
                            else:
                                if bool(consensus_positive.any()):
                                    positive_specificity = (
                                        consensus_specificity_prediction[
                                            consensus_positive])
                                    specificity_terms.append(
                                        F.binary_cross_entropy(
                                            positive_specificity,
                                            torch.ones_like(
                                                positive_specificity)))
                                    with torch.no_grad():
                                        collocation_speed_consensus_specificity_positive_mean = (
                                            positive_specificity.mean())
                                if bool(consensus_dangerous_negative.any()):
                                    negative_specificity = (
                                        consensus_specificity_prediction[
                                            consensus_dangerous_negative])
                                    specificity_terms.append(
                                        F.binary_cross_entropy(
                                            negative_specificity,
                                            torch.zeros_like(
                                                negative_specificity)))
                                    with torch.no_grad():
                                        collocation_speed_consensus_specificity_negative_mean = (
                                            negative_specificity.mean())
                                        collocation_speed_consensus_negative_fraction = (
                                            consensus_dangerous_negative.to(
                                                speed_target.dtype).mean())
                            if specificity_terms:
                                collocation_speed_consensus_specificity_bce = (
                                    sum(specificity_terms)
                                    / len(specificity_terms))
                            if (consensus_mode
                                    == 'corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend'):
                                # The inherited magnitude and both fusion
                                # endpoints are frozen.  The new signed blend
                                # head is therefore identified directly by
                                # exact support alpha, with dangerous lows
                                # receiving the same balanced zero-alpha term.
                                balanced_alpha_terms = []
                                if bool(consensus_positive.any()):
                                    balanced_alpha_terms.append(
                                        alpha_square_error[
                                            consensus_positive].mean())
                                if bool(consensus_dangerous_negative.any()):
                                    negative_alpha_prediction = (
                                        consensus_alpha_prediction[
                                            consensus_dangerous_negative])
                                    collocation_speed_consensus_negative_magnitude_mse = (
                                        negative_alpha_prediction.square().mean())
                                    balanced_alpha_terms.append(
                                        collocation_speed_consensus_negative_magnitude_mse)
                                    with torch.no_grad():
                                        collocation_speed_consensus_negative_magnitude_prediction_mean = (
                                            negative_alpha_prediction.mean())
                                if balanced_alpha_terms:
                                    collocation_speed_consensus_identification_loss = (
                                        self.collocation_speed_consensus_identification_weight
                                        * sum(balanced_alpha_terms)
                                        / len(balanced_alpha_terms))
                                with torch.no_grad():
                                    collocation_speed_consensus_negative_fraction = (
                                        consensus_dangerous_negative.to(
                                            speed_target.dtype).mean())
                            elif consensus_mode in ROUTE_CHART_MODES:
                                # Each support row trains only the nearest
                                # configuration-chart expert, while every
                                # expert sees every threshold-dangerous low.
                                # The deployed soft gate remains a single
                                # differentiable q-only effective release.
                                with torch.no_grad():
                                    release_target = (
                                        _magnitude_compensated_release_target(
                                            consensus_alpha_target,
                                            consensus_magnitude_prediction,
                                            consensus_positive))
                                    route_assignment = (
                                        consensus_route_gate.argmax(dim=1))
                                expert_losses = []
                                for route_index in range(ROUTE_CHART_COUNT):
                                    route_support = (
                                        consensus_positive
                                        & (route_assignment == route_index))
                                    route_loss_sum = torch.zeros_like(
                                        collocation_speed_loss)
                                    route_loss_weight = 0.0
                                    if bool(route_support.any()):
                                        route_loss_sum = (
                                            route_loss_sum
                                            + (consensus_route_release_prediction[
                                                route_support, route_index]
                                               - release_target[route_support]
                                               ).square().mean())
                                        route_loss_weight += 1.0
                                    if bool(consensus_dangerous_negative.any()):
                                        negative_weight = (
                                            self.collocation_speed_consensus_negative_weight)
                                        route_loss_sum = (
                                            route_loss_sum
                                            + negative_weight
                                            * consensus_route_release_prediction[
                                                consensus_dangerous_negative,
                                                route_index].square().mean())
                                        route_loss_weight += negative_weight
                                    if route_loss_weight > 0.0:
                                        expert_losses.append(
                                            route_loss_sum / route_loss_weight)
                                if expert_losses:
                                    collocation_speed_consensus_identification_loss = (
                                        self.collocation_speed_consensus_identification_weight
                                        * sum(expert_losses)
                                        / len(expert_losses))
                                if bool(consensus_positive.any()):
                                    collocation_speed_consensus_effective_release_mse = (
                                        (consensus_effective_release[
                                            consensus_positive]
                                         - release_target[
                                             consensus_positive]
                                         ).square().mean())
                                    with torch.no_grad():
                                        collocation_speed_consensus_effective_release_target_mean = (
                                            release_target[
                                                consensus_positive].mean())
                                        collocation_speed_consensus_effective_release_prediction_mean = (
                                            consensus_effective_release[
                                                consensus_positive].mean())
                                if bool(consensus_dangerous_negative.any()):
                                    collocation_speed_consensus_negative_effective_release_mse = (
                                        consensus_effective_release[
                                            consensus_dangerous_negative]
                                        .square().mean())
                                    with torch.no_grad():
                                        collocation_speed_consensus_negative_effective_release_prediction_mean = (
                                            consensus_effective_release[
                                                consensus_dangerous_negative]
                                            .mean())
                                        collocation_speed_consensus_negative_fraction = (
                                            consensus_dangerous_negative.to(
                                                speed_target.dtype).mean())
                            elif (consensus_mode
                                    == 'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8'):
                                # The inherited magnitude is frozen.  Dividing
                                # exact support alpha by that magnitude gives
                                # the identifiable effective-release target.
                                # Supervising release directly prevents a low
                                # magnitude from hiding an unsafe semantic
                                # permission correction on dangerous rows.
                                balanced_release_sum = torch.zeros_like(
                                    collocation_speed_loss)
                                balanced_release_weight = 0.0
                                if bool(consensus_positive.any()):
                                    with torch.no_grad():
                                        positive_effective_release_target = (
                                            _magnitude_compensated_release_target(
                                                consensus_alpha_target,
                                                consensus_magnitude_prediction,
                                                consensus_positive)[
                                                    consensus_positive])
                                    positive_effective_release_prediction = (
                                        consensus_effective_release[
                                            consensus_positive])
                                    collocation_speed_consensus_effective_release_mse = (
                                        (positive_effective_release_prediction
                                         - positive_effective_release_target
                                         ).square().mean())
                                    balanced_release_sum = (
                                        balanced_release_sum
                                        + collocation_speed_consensus_effective_release_mse)
                                    balanced_release_weight += 1.0
                                    with torch.no_grad():
                                        collocation_speed_consensus_effective_release_target_mean = (
                                            positive_effective_release_target.mean())
                                        collocation_speed_consensus_effective_release_prediction_mean = (
                                            positive_effective_release_prediction.mean())
                                if bool(consensus_dangerous_negative.any()):
                                    negative_effective_release_prediction = (
                                        consensus_effective_release[
                                            consensus_dangerous_negative])
                                    collocation_speed_consensus_negative_effective_release_mse = (
                                        negative_effective_release_prediction.square().mean())
                                    negative_weight = (
                                        self.collocation_speed_consensus_negative_weight)
                                    balanced_release_sum = (
                                        balanced_release_sum
                                        + negative_weight
                                        * collocation_speed_consensus_negative_effective_release_mse)
                                    balanced_release_weight += negative_weight
                                    with torch.no_grad():
                                        collocation_speed_consensus_negative_effective_release_prediction_mean = (
                                            negative_effective_release_prediction.mean())
                                if balanced_release_weight > 0.0:
                                    balanced_release_loss = (
                                        balanced_release_sum
                                        / balanced_release_weight)
                                    component_weight = (
                                        self.collocation_speed_consensus_head_component_weight)
                                    if component_weight > 0.0:
                                        balanced_release_loss = (
                                            balanced_release_loss
                                            + component_weight
                                            * collocation_speed_consensus_specificity_bce
                                        ) / (1.0 + component_weight)
                                    collocation_speed_consensus_identification_loss = (
                                        self.collocation_speed_consensus_identification_weight
                                        * balanced_release_loss)
                                with torch.no_grad():
                                    collocation_speed_consensus_negative_fraction = (
                                        consensus_dangerous_negative.to(
                                            speed_target.dtype).mean())
                            elif consensus_mode in (
                                    'corridor_hurdle_semantic_veto_min_odds',
                                    'corridor_hurdle_semantic_veto_min_odds_residual'):
                                # Phase AQ freezes the three semantic heads.
                                # Their BCE remains a diagnostic, while the
                                # only optimized tail gets balanced positive
                                # rescue and threshold-flippable low examples.
                                # This is required because odds fusion raises
                                # support capacity and dangerous capacity at
                                # the same time; magnitude must learn which of
                                # those reachable rows should actually fire.
                                balanced_magnitude_terms = []
                                if bool(consensus_positive.any()):
                                    balanced_magnitude_terms.append(
                                        collocation_speed_consensus_magnitude_mse)
                                if bool(consensus_dangerous_negative.any()):
                                    negative_magnitude_prediction = (
                                        consensus_magnitude_prediction[
                                            consensus_dangerous_negative])
                                    collocation_speed_consensus_negative_magnitude_mse = (
                                        negative_magnitude_prediction.square().mean())
                                    balanced_magnitude_terms.append(
                                        collocation_speed_consensus_negative_magnitude_mse)
                                    with torch.no_grad():
                                        collocation_speed_consensus_negative_magnitude_prediction_mean = (
                                            negative_magnitude_prediction.mean())
                                if balanced_magnitude_terms:
                                    collocation_speed_consensus_identification_loss = (
                                        self.collocation_speed_consensus_identification_weight
                                        * sum(balanced_magnitude_terms)
                                        / len(balanced_magnitude_terms))
                                with torch.no_grad():
                                    collocation_speed_consensus_negative_fraction = (
                                        consensus_dangerous_negative.to(
                                            speed_target.dtype).mean())
                            else:
                                specificity_magnitude_terms = []
                                if specificity_terms:
                                    specificity_magnitude_terms.append(
                                        collocation_speed_consensus_specificity_bce)
                                if bool(consensus_positive.any()):
                                    specificity_magnitude_terms.append(
                                        collocation_speed_consensus_magnitude_mse)
                                if specificity_magnitude_terms:
                                    collocation_speed_consensus_identification_loss = (
                                        self.collocation_speed_consensus_identification_weight
                                        * sum(specificity_magnitude_terms)
                                        / len(specificity_magnitude_terms))
                        elif (consensus_mode
                                == 'corridor_hurdle_compensated_balanced'):
                            balanced_magnitude_terms = []
                            if bool(consensus_positive.any()):
                                balanced_magnitude_terms.append(
                                    collocation_speed_consensus_magnitude_mse)
                            if bool(consensus_dangerous_negative.any()):
                                negative_magnitude_prediction = (
                                    consensus_magnitude_prediction[
                                        consensus_dangerous_negative])
                                collocation_speed_consensus_negative_magnitude_mse = (
                                    negative_magnitude_prediction.square().mean())
                                balanced_magnitude_terms.append(
                                    collocation_speed_consensus_negative_magnitude_mse)
                                with torch.no_grad():
                                    collocation_speed_consensus_negative_magnitude_prediction_mean = (
                                        negative_magnitude_prediction.mean())
                            if balanced_magnitude_terms:
                                collocation_speed_consensus_identification_loss = (
                                    self.collocation_speed_consensus_identification_weight
                                    * sum(balanced_magnitude_terms)
                                    / len(balanced_magnitude_terms))
                            with torch.no_grad():
                                collocation_speed_consensus_negative_fraction = (
                                    consensus_dangerous_negative.to(
                                        speed_target.dtype).mean())
                        elif (consensus_mode
                                == 'corridor_hurdle_compensated'):
                            collocation_speed_consensus_identification_loss = (
                                self.collocation_speed_consensus_identification_weight
                                * collocation_speed_consensus_magnitude_mse)
                        else:
                            collocation_speed_consensus_identification_loss = (
                                0.5
                                * self.collocation_speed_consensus_identification_weight
                                * (collocation_speed_consensus_release_bce
                                   + collocation_speed_consensus_magnitude_mse))
                        with torch.no_grad():
                            collocation_speed_consensus_release_target_mean = (
                                identified_release_target.mean())
                            collocation_speed_consensus_release_prediction_mean = (
                                identified_release_prediction.mean())
                    else:
                        collocation_speed_consensus_identification_loss = (
                            self.collocation_speed_consensus_identification_weight
                            * identified_error.mean())
                    with torch.no_grad():
                        collocation_speed_consensus_alpha_mae = (
                            identified_error.sqrt().mean())
                        collocation_speed_consensus_alpha_target_mean = (
                            consensus_alpha_target[
                                consensus_identified].mean())
                        collocation_speed_consensus_alpha_prediction_mean = (
                            consensus_alpha_prediction[
                                consensus_identified].mean())
                collocation_speed_consensus_identified_fraction = (
                    consensus_identified.to(speed_target.dtype).mean())

            # Repair each semantic bottleneck before the final soft minimum.
            # The frozen direct scalar is only an upper endpoint for a
            # one-sided convex repair, never a teacher target:
            #
            #   gap_i  = relu(F_direct - C_i)
            #   rho_i* = clip((C_i* - C_i) / gap_i, 0, 1)
            #   C'_i   = C_i + rho_i gap_i.
            #
            # As with the global hurdle, support and conditional magnitude are
            # separated because most exact rho targets are zero.
            if self.collocation_speed_repair_identification_weight > 0.0:
                if collocation_speed_component_labels is None:
                    raise RuntimeError(
                        "repair identification requires "
                        "collocation_speed_components.npy")
                repair_component_labels = (
                    collocation_speed_component_labels.to(
                        device=tau.device, dtype=Xp.dtype))
                expected_heads = int(getattr(
                    self.network, 'speed_constraint_head_count', 0))
                if (repair_component_labels.ndim != 2
                        or repair_component_labels.shape[0]
                        != independent_pool.shape[0]
                        or repair_component_labels.shape[1] != expected_heads):
                    raise ValueError(
                        "collocation component labels must have shape "
                        f"(M,{expected_heads}), got "
                        f"{tuple(repair_component_labels.shape)}")
                repair_component_target = repair_component_labels[speed_idx]
                with torch.no_grad():
                    repair_direct = self.network.predict_uncomposed_speed(
                        speed_states)
                    repair_components = (
                        self.network.predict_speed_components(speed_states))
                    repair_gap = torch.relu(
                        repair_direct - repair_components)
                    repair_identified = repair_gap > 1e-8
                    repair_safe_gap = torch.where(
                        repair_identified,
                        repair_gap,
                        torch.ones_like(repair_gap))
                    repair_rho_target = torch.clamp(
                        (repair_component_target - repair_components)
                        / repair_safe_gap,
                        min=0.0,
                        max=1.0)
                    repair_rho_target = torch.where(
                        repair_identified,
                        repair_rho_target,
                        torch.zeros_like(repair_rho_target))
                    if self.collocation_speed_repair_coordination:
                        repair_rho_target, repair_coordination_beta = (
                            _coordinate_component_repair_target(
                                self.network, repair_components, repair_gap,
                                repair_rho_target, speed_target,
                                iterations=24))
                        collocation_speed_repair_coordination_beta_mean = (
                            repair_coordination_beta.mean())
                        collocation_speed_repair_coordination_zero_fraction = (
                            (repair_coordination_beta <= 1e-7).to(
                                repair_coordination_beta.dtype).mean())
                    repair_support_target = (
                        repair_rho_target > 0.0).to(
                            repair_rho_target.dtype)
                repair_factors = (
                    self.network.predict_speed_constraint_repair_factors(
                        speed_states))
                repair_release_prediction = repair_factors[:, :, 0]
                repair_magnitude_prediction = repair_factors[:, :, 1]
                repair_rho_prediction = (
                    repair_release_prediction
                    * repair_magnitude_prediction)
                if bool(repair_identified.any()):
                    identified_release = repair_release_prediction[
                        repair_identified]
                    identified_support = repair_support_target[
                        repair_identified]
                    collocation_speed_repair_release_bce = (
                        F.binary_cross_entropy(
                            identified_release, identified_support))
                    repair_positive = (
                        repair_identified & (repair_rho_target > 0.0))
                    if bool(repair_positive.any()):
                        positive_magnitude = repair_magnitude_prediction[
                            repair_positive]
                        positive_target = repair_rho_target[repair_positive]
                        collocation_speed_repair_magnitude_mse = (
                            (positive_magnitude
                             - positive_target).square().mean())
                        with torch.no_grad():
                            collocation_speed_repair_magnitude_target_mean = (
                                positive_target.mean())
                            collocation_speed_repair_magnitude_prediction_mean = (
                                positive_magnitude.mean())
                    collocation_speed_repair_identification_loss = (
                        0.5
                        * self.collocation_speed_repair_identification_weight
                        * (collocation_speed_repair_release_bce
                           + collocation_speed_repair_magnitude_mse))
                    with torch.no_grad():
                        collocation_speed_repair_rho_mae = (
                            (repair_rho_prediction[repair_identified]
                             - repair_rho_target[
                                 repair_identified]).abs().mean())
                        collocation_speed_repair_support_target_mean = (
                            identified_support.mean())
                        collocation_speed_repair_release_prediction_mean = (
                            identified_release.mean())

            if self.collocation_speed_component_loss_weight > 0.0:
                if collocation_speed_component_labels is None:
                    raise RuntimeError(
                        "component supervision requires "
                        "collocation_speed_components.npy")
                component_labels = collocation_speed_component_labels.to(
                    device=tau.device, dtype=Xp.dtype)
                expected_heads = int(getattr(
                    self.network, 'speed_constraint_head_count', 0))
                if (component_labels.ndim != 2
                        or component_labels.shape[0]
                        != independent_pool.shape[0]
                        or component_labels.shape[1] != expected_heads):
                    raise ValueError(
                        "collocation component labels must have shape "
                        f"(M,{expected_heads}), got "
                        f"{tuple(component_labels.shape)}")
                component_target = component_labels[speed_idx]
                component_pred = self.network.predict_speed_components(
                    speed_states)
                collocation_speed_component_loss = (
                    self.collocation_speed_component_loss_weight
                    * (component_pred - component_target).square().mean())
                with torch.no_grad():
                    component_error = component_pred - component_target
                    collocation_speed_component_mae = (
                        component_error.abs().mean())
                    pred_centered = (
                        component_pred - component_pred.mean(dim=0))
                    target_centered = (
                        component_target - component_target.mean(dim=0))
                    corr_den = (
                        pred_centered.square().sum(dim=0).sqrt()
                        * target_centered.square().sum(dim=0).sqrt())
                    component_corr = torch.where(
                        corr_den > 1e-8,
                        (pred_centered * target_centered).sum(dim=0)
                        / corr_den,
                        torch.zeros_like(corr_den))
                    collocation_speed_component_corr_mean = (
                        component_corr.mean())
                    collocation_speed_component_corr_min = (
                        component_corr.min())

        if (getattr(self.network, 'use_clearance_head', False)
                and self.clearance_loss_weight > 0.0):
            if d_obstacle is None or d_self is None:
                raise ValueError(
                    "clearance supervision requires scalar d_obstacle and "
                    "d_self arrays")
            # Both quantities are scalar distances.  Scaling self clearance
            # by 0.05/0.02=2.5 places its strict 2 mm boundary at the same
            # normalized 0.1 coordinate as the 5 mm obstacle boundary.
            if clearance_target is None:
                physical_clearance = torch.minimum(
                    d_obstacle, self.clearance_self_scale * d_self)
                clearance_target = (
                    physical_clearance
                    / float(self.network.clearance_scale)).clamp(-1.0, 1.0)
            endpoint_points = torch.cat([
                Xp[:, :self.dim], Xp[:, self.dim:]], dim=0)
            clearance_pred = self.network.predict_clearance(
                endpoint_points)[:, 0]
            clearance_supervision_loss = self.clearance_loss_weight * (
                (clearance_pred[:B] - clearance_target[:, 0]).square()
                + (clearance_pred[B:] - clearance_target[:, 1]).square())

        # --- From here, identical to single-task Loss ---
        D = torch.norm(Xp[:, self.dim:] - Xp[:, :self.dim], p=2, dim=1)

        DT0 = dtau[:, :self.dim]
        DT1 = dtau[:, self.dim:]

        S0 = torch.einsum('ij,ij->i', DT0, DT0)
        S1 = torch.einsum('ij,ij->i', DT1, DT1)

        # --- Temporal loss ---
        td_weight = 1e-3
        with torch.no_grad():
            length0 = (0.02) / (Yobs[:, 0]).unsqueeze(1)
            Dir0 = length0 * (DT0 * Yobs[:, 0].unsqueeze(1) ** 2).clone().detach()
            Xp_new0 = Xp.clone().detach()
            Xp_new0[:, :self.dim] = Xp_new0[:, :self.dim] - Dir0

            # Preserve per-endpoint routing. Passing only start weights would
            # repeat them for the goal endpoint inside network.out(), which is
            # wrong for cross-zone pairs.
            tau_new0, w_tmp, Xp_new0, _, _ = self.network.out(
                Xp_new0, w_cup_override=w_cup_2B, w_pour_override=w_pour_2B,
                ee_y=ee_y_stacked,
                d_orient_cup=d_cup_stacked,
                d_orient_pour=d_pour_stacked,
                ee_xy=ee_xy_stacked,
                fk_features=fk_features)
            tau_new1 = length0
            del Xp_new0, Dir0

        temporal_loss0 = td_weight * ((tau - (tau_new0 + tau_new1)) ** 2).squeeze()

        with torch.no_grad():
            length1 = (0.02) / (Yobs[:, 1]).unsqueeze(1)
            Dir1 = length1 * (DT1 * Yobs[:, 1].unsqueeze(1) ** 2).clone().detach()
            Xp_new0 = Xp.clone().detach()
            Xp_new0[:, self.dim:] = Xp_new0[:, self.dim:] - Dir1

            tau_new0, w_tmp, Xp_new0, _, _ = self.network.out(
                Xp_new0, w_cup_override=w_cup_2B, w_pour_override=w_pour_2B,
                ee_y=ee_y_stacked,
                d_orient_cup=d_cup_stacked,
                d_orient_pour=d_pour_stacked,
                ee_xy=ee_xy_stacked,
                fk_features=fk_features)
            tau_new1 = length1
            del Xp_new0, Dir1

        temporal_loss1 = td_weight * ((tau - (tau_new0 + tau_new1)) ** 2).squeeze()

        where_d0 = (tau[:, 0] < length0.squeeze())
        where_d1 = (tau[:, 0] < length1.squeeze())
        temporal_loss0[where_d0] = 0
        temporal_loss1[where_d1] = 0

        temporal_loss = temporal_loss0 + temporal_loss1

        # --- Eikonal loss ---
        Ypred0 = torch.sqrt(S0 + 1e-8)
        Ypred1 = torch.sqrt(S1 + 1e-8)

        Ypred0_visco = Ypred0
        Ypred1_visco = Ypred1

        sq_Ypred0 = (Ypred0_visco)
        sq_Ypred1 = (Ypred1_visco)

        sq_Yobs0 = (Yobs[:, 0])
        sq_Yobs1 = (Yobs[:, 1])

        l0 = sq_Yobs0 * sq_Ypred0
        l1 = sq_Yobs1 * sq_Ypred1

        # Standard isotropic Eikonal equation: F(q)||grad T|| = 1.  Keep the
        # historical square-root residual available only for reproducibility;
        # it has the same zero set but weakens large PDE violations.
        if self.eikonal_form == 'standard':
            l0_2 = l0
            l1_2 = l1
        else:
            l0_2 = torch.sqrt(l0)
            l1_2 = torch.sqrt(l1)

        loss_weight = 1e-2
        loss0 = loss_weight * (l0_2 - 1) ** 2
        loss1 = loss_weight * (l1_2 - 1) ** 2
        eikonal_valid0 = None
        eikonal_valid1 = None
        if self.eikonal_min_speed > 0:
            # The Eikonal PDE is solved on the feasible domain.  Speeds below
            # this threshold represent the obstacle interior, where F -> 0
            # makes |grad T|=1/F singular.  Keep those speeds in chord and
            # Bellman costs as a barrier, but do not force a finite network to
            # fit the PDE inside the excluded domain.
            # T(q, g) is a free-domain value only when both q and its boundary
            # condition g are feasible.  An invalid goal has no finite
            # free-space arrival-time solution, so exclude the whole pair.
            eikonal_valid_pair = (
                (sq_Yobs0 >= self.eikonal_min_speed)
                & (sq_Yobs1 >= self.eikonal_min_speed))
            eikonal_valid0 = eikonal_valid_pair
            eikonal_valid1 = eikonal_valid_pair
        if self.eikonal_balance_power > 0:
            # F=1 free space dominates uniform batches.  Correct that
            # sampling imbalance using only the supplied scalar speed, so
            # narrow low-speed bands contribute enough PDE measure without
            # normals, collision labels, or a teacher trajectory.
            eik_weight0 = sq_Yobs0.clamp_min(speed_floor).pow(
                -self.eikonal_balance_power)
            eik_weight1 = sq_Yobs1.clamp_min(speed_floor).pow(
                -self.eikonal_balance_power)
            if eikonal_valid0 is not None:
                valid0 = eikonal_valid0.to(eik_weight0.dtype)
                valid1 = eikonal_valid1.to(eik_weight1.dtype)
                eik_weight_mean = (
                    (eik_weight0 * valid0).sum()
                    + (eik_weight1 * valid1).sum()
                ) / (valid0.sum() + valid1.sum()).clamp_min(1.0)
                eik_weight_mean = eik_weight_mean.clamp_min(1e-8)
            else:
                eik_weight_mean = 0.5 * (
                    eik_weight0.mean() + eik_weight1.mean())
            loss0 = loss0 * (eik_weight0 / eik_weight_mean)
            loss1 = loss1 * (eik_weight1 / eik_weight_mean)
        if eikonal_valid0 is not None:
            loss0 = loss0 * eikonal_valid0.to(loss0.dtype)
            loss1 = loss1 * eikonal_valid1.to(loss1.dtype)

        T = tau[:, 0]
        diff = loss0 + loss1

        # Coherent-goal Eikonal collocation.  Random endpoint pairs almost
        # never revisit the same boundary-value problem in six dimensions.
        # Here a small fixed bank of strict-feasible scalar-cloud states acts
        # only as repeated point goals; many independently sampled q states
        # share each selected goal in an update.  Supervision remains exactly
        # F(q)|grad_q T(q,g)|=1 with the scalar F labels already present in
        # collocation_speed.npy.  No route/task label, path, graph, teacher,
        # normal, value target, or evaluator state is consulted.
        coherent_eikonal_loss = tau.new_zeros(())
        coherent_eikonal_residual = tau.new_zeros(())
        coherent_first_order_residual = tau.new_zeros(())
        coherent_viscosity_laplacian_rms = tau.new_zeros(())
        coherent_subsolution_violation = tau.new_zeros(())
        coherent_subsolution_value = tau.new_zeros(())
        coherent_discount_value = tau.new_zeros(())
        coherent_goal_count = tau.new_zeros(())
        trusted_anchor_loss = tau.new_zeros(())
        trusted_anchor_stats = {
            "hard_count": tau.new_zeros(()),
            "direct_hard_count": tau.new_zeros(()),
            "route_pair_count": tau.new_zeros(()),
            "trusted_anchor_mean": tau.new_zeros(()),
            "ratio_mean": tau.new_zeros(()),
            "violation_fraction": tau.new_zeros(()),
            "violation_p95": tau.new_zeros(()),
            "direct_cost_mean": tau.new_zeros(()),
        }
        monotone_fd_loss = tau.new_zeros(())
        monotone_fd_residual = tau.new_zeros(())
        monotone_fd_valid_rate = tau.new_zeros(())
        monotone_fd_norm_mean = tau.new_zeros(())
        if self.coherent_eikonal_weight > 0.0:
            if collocation_points is None or collocation_speed_labels is None:
                raise RuntimeError(
                    "coherent Eikonal loss requires collocation_points.npy "
                    "and collocation_speed.npy")
            cloud = collocation_points.to(device=tau.device, dtype=Xp.dtype)
            cloud_speed = collocation_speed_labels.to(
                device=tau.device, dtype=Xp.dtype).reshape(-1)
            if (cloud.ndim != 2 or cloud.shape[1] != self.dim
                    or cloud.shape[0] != cloud_speed.numel()):
                raise ValueError(
                    "coherent Eikonal cloud must have shapes (M,dim) and (M,)")
            valid = torch.isfinite(cloud_speed)
            if self.eikonal_min_speed > 0.0:
                valid &= cloud_speed >= self.eikonal_min_speed
            valid_idx = torch.nonzero(valid, as_tuple=False).reshape(-1)
            if valid_idx.numel() < self.coherent_goal_bank:
                raise RuntimeError(
                    "coherent Eikonal cloud has fewer strict-feasible states "
                    "than the requested goal bank")
            if (self._coherent_goal_indices is None
                    or self._coherent_goal_indices.device != tau.device
                    or int(self._coherent_goal_indices.max())
                    >= cloud.shape[0]):
                # The source cloud is generated in a seeded random order.
                # Even spacing gives a deterministic subset without reading
                # geometry, pair labels, or evaluation states.
                positions = torch.linspace(
                    0, valid_idx.numel() - 1,
                    self.coherent_goal_bank,
                    device=tau.device).round().long()
                self._coherent_goal_indices = valid_idx[positions]

            goal_count = min(
                self.coherent_goals_per_batch,
                self.coherent_eikonal_n)
            per_goal = self.coherent_eikonal_n // goal_count
            n_coherent = per_goal * goal_count
            goal_slots = (
                torch.arange(goal_count, device=tau.device)
                + self._coherent_step * goal_count
            ) % self.coherent_goal_bank
            selected_goals = cloud[
                self._coherent_goal_indices[goal_slots]].detach()
            goals = selected_goals.repeat_interleave(per_goal, dim=0)
            q_slots = torch.randint(
                valid_idx.numel(), (n_coherent,), device=tau.device)
            q_idx = valid_idx[q_slots]
            q_states = cloud[q_idx].detach()
            q_speed = cloud_speed[q_idx].detach()
            coherent_pairs = torch.cat([q_states, goals], dim=1)
            coherent_pairs.requires_grad_(True)
            scalar_ones = torch.ones(
                2 * n_coherent, device=tau.device, dtype=tau.dtype)
            scalar_zeros = torch.zeros_like(scalar_ones)
            coherent_tau, _, coherent_x, _, _ = self.network.out(
                coherent_pairs, detach_input=False,
                w_cup_override=scalar_ones,
                w_pour_override=scalar_zeros)
            coherent_grad = self.gradient(
                coherent_tau, coherent_x)[:, :self.dim]
            coherent_norm = torch.linalg.vector_norm(
                coherent_grad, dim=1)
            coherent_first_order = q_speed * coherent_norm - 1.0
            coherent_residual = coherent_first_order
            if self.coherent_discount_lambda > 0.0:
                # The discounted Eikonal equation
                #   F(q)|grad U(q,g)| + lambda U(q,g) - 1 = 0
                # is a proper Hamilton--Jacobi equation: its positive
                # zero-order term supplies a comparison principle without a
                # neighbor stencil, path/control representation, target
                # value, or additional deployed field. Its viscosity solution
                # is the monotone compression
                #   U=(1-exp(-lambda*d_F))/lambda,
                # so geodesic descent directions are preserved.
                coherent_residual = (
                    coherent_first_order
                    + self.coherent_discount_lambda * coherent_tau[:, 0])
                coherent_discount_value = coherent_tau[:, 0].mean()
            if self.coherent_viscosity_epsilon > 0.0:
                # Hutchinson trace of the q Hessian.  The antithetic sign gives
                # the same v^T H v, so one seeded Rademacher probe is the
                # minimum unbiased trace estimator.  This realizes the
                # vanishing-viscosity boundary-value PDE
                #   F|grad T| - 1 - eps F Delta(T) = 0
                # without selecting a route, neighbor, graph, or teacher.
                probe = torch.empty_like(coherent_grad).bernoulli_(0.5)
                probe = probe.mul_(2.0).sub_(1.0)
                directional_grad = (coherent_grad * probe).sum()
                hessian_vector = torch.autograd.grad(
                    directional_grad, coherent_x,
                    create_graph=True, retain_graph=True)[0][:, :self.dim]
                laplacian = (hessian_vector * probe).sum(dim=1)
                coherent_residual = (
                    coherent_first_order
                    - self.coherent_viscosity_epsilon * q_speed * laplacian)
                coherent_viscosity_laplacian_rms = (
                    laplacian.square().mean().sqrt())
            off_diagonal = torch.linalg.vector_norm(
                q_states - goals, dim=1) > 1e-5
            finite = torch.isfinite(coherent_residual) & off_diagonal
            if finite.any():
                coherent_first_order_residual = (
                    coherent_first_order[finite].square().mean())
                coherent_eikonal_residual = (
                    coherent_residual[finite].square().mean())
                if self.coherent_subsolution_reward > 0.0:
                    # The geodesic viscosity distance is the maximal
                    # subsolution with T(g,g)=0.  The deployed architectures
                    # enforce the diagonal boundary structurally; penalize
                    # only F|grad T|>1 and maximize the same scalar T over the
                    # coherent domain.  This is a global variational selector,
                    # not a neighbor/path/graph target or monotone rollout.
                    coherent_subsolution_violation = (
                        coherent_first_order[finite].clamp_min(0.0)
                        .square().mean())
                    coherent_subsolution_value = coherent_tau[:, 0][finite].mean()
                    coherent_eikonal_loss = (
                        self.coherent_eikonal_weight
                        * coherent_subsolution_violation
                        - self.coherent_subsolution_reward
                        * coherent_subsolution_value)
                else:
                    coherent_eikonal_loss = (
                        self.coherent_eikonal_weight
                        * coherent_eikonal_residual)
            coherent_goal_count = tau.new_tensor(float(goal_count))
            self._coherent_step += 1

        if self.monotone_fd_weight > 0.0:
            if collocation_points is None or collocation_speed_labels is None:
                raise RuntimeError(
                    "monotone finite differences require the independent "
                    "scalar collocation cloud")
            cloud = collocation_points.to(device=tau.device, dtype=Xp.dtype)
            cloud_labels = collocation_speed_labels.to(
                device=tau.device, dtype=Xp.dtype).reshape(-1)
            if (cloud.ndim != 2 or cloud.shape[1] != self.dim
                    or cloud.shape[0] != cloud_labels.numel()):
                raise ValueError(
                    "monotone finite-difference cloud must have shapes "
                    "(M,dim) and (M,)")
            h = self.monotone_fd_step
            eligible = (
                torch.isfinite(cloud_labels)
                & (cloud_labels >= self.monotone_fd_domain_speed)
                & (cloud.abs() <= 0.5 - h).all(dim=1))
            eligible_idx = torch.nonzero(
                eligible, as_tuple=False).reshape(-1)
            if eligible_idx.numel() < self.monotone_fd_goal_bank:
                raise RuntimeError(
                    "too few interior feasible states for monotone stencil")
            if (self._monotone_fd_goal_indices is None
                    or self._monotone_fd_goal_indices.device != tau.device
                    or int(self._monotone_fd_goal_indices.max())
                    >= cloud.shape[0]):
                positions = torch.linspace(
                    0, eligible_idx.numel() - 1,
                    self.monotone_fd_goal_bank,
                    device=tau.device).round().long()
                self._monotone_fd_goal_indices = eligible_idx[positions]

            goal_count = min(
                self.monotone_fd_goals_per_batch, self.monotone_fd_n)
            per_goal = self.monotone_fd_n // goal_count
            n_fd = per_goal * goal_count
            goal_slots = (
                torch.arange(goal_count, device=tau.device)
                + self._monotone_fd_step_count * goal_count
            ) % self.monotone_fd_goal_bank
            goal_base = cloud[
                self._monotone_fd_goal_indices[goal_slots]
            ].detach()
            goals = goal_base.repeat_interleave(per_goal, dim=0)
            if self.monotone_fd_rar_factor > 1:
                # Residual-based adaptive refinement (RAR): score a fixed
                # larger cloud with the same local monotone operator, detach
                # the selection, and optimize only the worst current
                # residuals.  This is ephemeral collocation, not persistent
                # replay/value state or evaluator-driven hard mining.
                candidate_count = n_fd * self.monotone_fd_rar_factor
                if candidate_count > eligible_idx.numel():
                    raise RuntimeError(
                        "monotone RAR candidate count exceeds eligible cloud")
                candidate_slots = torch.randperm(
                    eligible_idx.numel(), device=tau.device
                )[:candidate_count]
                candidate_states = cloud[
                    eligible_idx[candidate_slots]].detach()
                candidate_goals = goal_base.repeat_interleave(
                    per_goal * self.monotone_fd_rar_factor, dim=0)
                from .monotone_fd import axis_stencil, godunov_upwind_norm
                with torch.no_grad():
                    cand_minus, cand_plus = axis_stencil(
                        candidate_states, h)
                    cand_repeated_goals = candidate_goals.repeat_interleave(
                        self.dim, dim=0)
                    cand_states_all = torch.cat([
                        candidate_states, cand_minus, cand_plus], dim=0)
                    cand_goals_all = torch.cat([
                        candidate_goals, cand_repeated_goals,
                        cand_repeated_goals], dim=0)
                    cand_pairs = torch.cat(
                        [cand_states_all, cand_goals_all], dim=1)
                    cand_ones = torch.ones(
                        2 * cand_pairs.shape[0], device=tau.device,
                        dtype=tau.dtype)
                    cand_zeros = torch.zeros_like(cand_ones)
                    cand_tau, _, _, _, _ = self.network.out(
                        cand_pairs, detach_input=False,
                        w_cup_override=cand_ones,
                        w_pour_override=cand_zeros)
                    cand_value = cand_tau[:, 0]
                    cand_minus_value = cand_value[
                        candidate_count:
                        candidate_count * (self.dim + 1)
                    ].reshape(candidate_count, self.dim)
                    cand_plus_value = cand_value[
                        candidate_count * (self.dim + 1):
                    ].reshape(candidate_count, self.dim)
                    cand_norm = godunov_upwind_norm(
                        cand_value[:candidate_count], cand_minus_value,
                        cand_plus_value, h)
                    cand_speed = self.network.predict_propagation_speed(
                        cand_states_all)[:, 0]
                    cand_center_speed = cand_speed[:candidate_count]
                    cand_minus_speed = cand_speed[
                        candidate_count:
                        candidate_count * (self.dim + 1)
                    ].reshape(candidate_count, self.dim)
                    cand_plus_speed = cand_speed[
                        candidate_count * (self.dim + 1):
                    ].reshape(candidate_count, self.dim)
                    cand_valid = (
                        (cand_center_speed
                         >= self.monotone_fd_domain_speed)
                        & (cand_minus_speed
                           >= self.monotone_fd_domain_speed).all(dim=1)
                        & (cand_plus_speed
                           >= self.monotone_fd_domain_speed).all(dim=1)
                        & (torch.linalg.vector_norm(
                            candidate_states - candidate_goals, dim=1)
                           > 2.0 * h))
                    cand_residual = cand_center_speed * cand_norm - 1.0
                    cand_valid &= torch.isfinite(cand_residual)
                    valid_candidate_idx = torch.nonzero(
                        cand_valid, as_tuple=False).reshape(-1)
                    if valid_candidate_idx.numel() < n_fd:
                        raise RuntimeError(
                            "monotone RAR has fewer valid candidates than "
                            "the requested training stencil count")
                    selected_local = torch.topk(
                        cand_residual[valid_candidate_idx].abs(),
                        k=n_fd, largest=True, sorted=False).indices
                    selected = valid_candidate_idx[selected_local]
                states = candidate_states[selected]
                goals = candidate_goals[selected]
            elif self.monotone_fd_sampling_power > 0.0:
                eligible_speed = cloud_labels[eligible_idx].clamp_min(
                    self.monotone_fd_domain_speed)
                sampling_weight = (
                    self.monotone_fd_domain_speed / eligible_speed
                ).pow(self.monotone_fd_sampling_power)
                q_slots = torch.multinomial(
                    sampling_weight, n_fd, replacement=False)
                states = cloud[eligible_idx[q_slots]].detach()
            else:
                q_slots = torch.randint(
                    eligible_idx.numel(), (n_fd,), device=tau.device)
                states = cloud[eligible_idx[q_slots]].detach()

            from .monotone_fd import axis_stencil, godunov_upwind_norm
            minus_states, plus_states = axis_stencil(states, h)
            repeated_goals = goals.repeat_interleave(self.dim, dim=0)
            stencil_states = torch.cat([
                states, minus_states, plus_states], dim=0)
            stencil_goals = torch.cat([
                goals, repeated_goals, repeated_goals], dim=0)
            stencil_pairs = torch.cat(
                [stencil_states, stencil_goals], dim=1)
            scalar_ones = torch.ones(
                2 * stencil_pairs.shape[0],
                device=tau.device, dtype=tau.dtype)
            scalar_zeros = torch.zeros_like(scalar_ones)
            stencil_tau, _, _, _, _ = self.network.out(
                stencil_pairs, detach_input=False,
                w_cup_override=scalar_ones,
                w_pour_override=scalar_zeros)
            stencil_value = stencil_tau[:, 0]
            center_value = stencil_value[:n_fd]
            minus_value = stencil_value[
                n_fd:n_fd * (self.dim + 1)].reshape(n_fd, self.dim)
            plus_value = stencil_value[
                n_fd * (self.dim + 1):].reshape(n_fd, self.dim)
            upwind_norm = godunov_upwind_norm(
                center_value, minus_value, plus_value, h)

            # The coefficient and domain mask query the same deployed q-only
            # scalar F.  They are detached so this experiment changes T only.
            with torch.no_grad():
                stencil_speed = self.network.predict_propagation_speed(
                    stencil_states)[:, 0]
            center_speed = stencil_speed[:n_fd]
            minus_speed = stencil_speed[
                n_fd:n_fd * (self.dim + 1)].reshape(n_fd, self.dim)
            plus_speed = stencil_speed[
                n_fd * (self.dim + 1):].reshape(n_fd, self.dim)
            domain_valid = (
                (center_speed >= self.monotone_fd_domain_speed)
                & (minus_speed >= self.monotone_fd_domain_speed).all(dim=1)
                & (plus_speed >= self.monotone_fd_domain_speed).all(dim=1))
            off_source = torch.linalg.vector_norm(
                states - goals, dim=1) > 2.0 * h
            residual = center_speed * upwind_norm - 1.0
            finite = (
                domain_valid & off_source & torch.isfinite(residual))
            if not bool(finite.any()):
                raise RuntimeError(
                    "monotone finite-difference batch has no valid stencil")
            monotone_fd_residual = residual[finite].square().mean()
            monotone_fd_loss = (
                self.monotone_fd_weight * monotone_fd_residual)
            monotone_fd_valid_rate = finite.to(tau.dtype).mean()
            monotone_fd_norm_mean = upwind_norm[finite].mean()
            self._monotone_fd_step_count += 1

        if self.trusted_anchor_maximal_weight > 0.0:
            if collocation_points is None or collocation_speed_labels is None:
                raise RuntimeError(
                    "trusted-anchor maximality requires collocation_points.npy "
                    "and collocation_speed.npy")
            anchor_cloud = collocation_points.to(
                device=tau.device, dtype=Xp.dtype)
            anchor_cloud_speed = collocation_speed_labels.to(
                device=tau.device, dtype=Xp.dtype).reshape(-1)
            from .trusted_anchor_loss import trusted_anchor_maximal_loss

            def _trusted_speed(points_in):
                return self.network.predict_propagation_speed(points_in)[:, 0]

            def _trusted_value(starts_in, goals_in):
                scalar_pairs = torch.cat([starts_in, goals_in], dim=1)
                count = scalar_pairs.shape[0]
                scalar_ones = torch.ones(
                    2 * count, device=tau.device, dtype=tau.dtype)
                scalar_zeros = torch.zeros_like(scalar_ones)
                value, _, _, _, _ = self.network.out(
                    scalar_pairs, detach_input=False,
                    w_cup_override=scalar_ones,
                    w_pour_override=scalar_zeros)
                return value[:, 0]

            raw_trusted_loss, trusted_anchor_stats = (
                trusted_anchor_maximal_loss(
                    _trusted_value,
                    _trusted_speed,
                    anchor_cloud,
                    anchor_cloud_speed,
                    pair_count=self.trusted_anchor_n,
                    anchor_count=self.trusted_anchor_count,
                    quadrature_points=self.trusted_anchor_quadrature,
                    trust_speed=self.trusted_anchor_min_speed,
                    upper_weight=self.trusted_anchor_upper_weight,
                    optimize_steps=self.trusted_anchor_optimize_steps,
                    optimize_learning_rate=self.trusted_anchor_optimize_lr,
                    optimize_softmin_temperature=
                        self.trusted_anchor_softmin_temperature,
                    optimize_barrier_weight=
                        self.trusted_anchor_barrier_weight))
            trusted_anchor_loss = (
                self.trusted_anchor_maximal_weight * raw_trusted_loss)

        # Hopf--Cole / desirability continuation.  Unlike characteristic
        # rollout, the Laplacian communicates through every local direction
        # and does not ask the current value field to select its own branch.
        # Exact symmetry lets start-endpoint collocation cover both roles.
        hopf_cole_loss = torch.zeros_like(diff)
        hopf_boundary_fraction = torch.zeros(
            (), device=tau.device, dtype=tau.dtype)
        hopf_characteristic_count = torch.zeros(
            (), device=tau.device, dtype=tau.dtype)
        hopf_state_violation_rate = torch.zeros(
            (), device=tau.device, dtype=tau.dtype)
        # The state-constraint normal condition is also useful as a clean
        # first-order boundary ablation on its own.  Historically it was
        # nested under the Hopf--Cole branch, which forced every boundary
        # experiment to pay the noisy second-order Laplacian and finite goal
        # cell.  Allow a zero Hopf weight with a positive state weight to
        # isolate the scalar state-constraint condition:
        #   grad_q T dot grad_q F <= 0  on the low-speed boundary.
        # This remains q-only and uses no collision/normal/path target.
        if ((self.hopf_cole_weight > 0.0
             or self.hopf_state_constraint_weight > 0.0)
                and self.hopf_cole_n > 0):
            if not getattr(self.network, 'use_speed_head', False):
                raise RuntimeError(
                    "Hopf--Cole training requires the shared scalar speed head")
            if getattr(self.network, 'value_condition', None) != 'none':
                raise RuntimeError(
                    "Hopf--Cole training requires one q-only scalar field")
            independent_pool = None
            independent_pool_speed = None
            if (collocation_points is not None
                    and not self.hopf_endpoint_measure
                    and not self.hopf_goal_shell_measure):
                independent_pool = collocation_points.to(
                    device=tau.device, dtype=Xp.dtype)
                if (independent_pool.ndim != 2
                        or independent_pool.shape[1] != self.dim
                        or independent_pool.shape[0] < 1):
                    raise ValueError(
                        "collocation_points must have shape (M, dim) with M>0")
                if collocation_speed_labels is not None:
                    independent_pool_speed = collocation_speed_labels.to(
                        device=tau.device, dtype=Xp.dtype).reshape(-1)
                    if independent_pool_speed.shape[0] != independent_pool.shape[0]:
                        raise ValueError(
                            "collocation_speed_labels must match "
                            "collocation_points")
            if (self.hopf_speed_mode == 'target'
                    and not self.hopf_endpoint_measure):
                if self.hopf_goal_shell_measure:
                    raise RuntimeError(
                        "goal-shell Hopf measure requires predicted scalar speed")
                if independent_pool is None or independent_pool_speed is None:
                    raise RuntimeError(
                        "hopf_speed_mode='target' requires the independent "
                        "collocation cloud and collocation_speed.npy")
            n_hopf = min(
                self.hopf_cole_n, B,
                independent_pool.shape[0] if independent_pool is not None else B)
            independent_measure = independent_pool is not None
            independent_speed_targets = None
            shell_bin = None
            # These slots provide endpoint goals for the independent q states
            # and determine where the measure-only loss is accumulated.
            hopf_idx = torch.randperm(B, device=tau.device)[:n_hopf]
            if self.hopf_goal_shell_measure:
                goal_states = Xp[hopf_idx, self.dim:].detach()
                shell_dir = torch.randn_like(goal_states)
                shell_dir = shell_dir / torch.linalg.vector_norm(
                    shell_dir, dim=1, keepdim=True).clamp_min(1e-8)
                log_min = math.log(self.hopf_shell_radius_min)
                log_max = math.log(self.hopf_shell_radius_max)
                if self.hopf_shellwise_normalize:
                    # Stratify log radius so every shell contributes an
                    # equally populated PDE measure.  The random offset
                    # keeps sampling continuous inside each shell; shell_bin
                    # is used only to rescale the same positive Ritz energy.
                    shell_bin = (
                        torch.arange(n_hopf, device=tau.device)
                        % self.hopf_shell_bins)
                    shell_bin = shell_bin[torch.randperm(
                        n_hopf, device=tau.device)]
                    shell_u = (
                        shell_bin.to(tau.dtype)
                        + torch.rand(
                            n_hopf, device=tau.device, dtype=tau.dtype)
                    ) / float(self.hopf_shell_bins)
                else:
                    shell_u = torch.rand(
                        n_hopf, device=tau.device, dtype=tau.dtype)
                shell_radius = torch.exp(
                    log_min + (log_max - log_min) * shell_u[:, None])
                shell_states = (goal_states + shell_radius * shell_dir).clamp(
                    -0.5, 0.5)
                collocation_pairs = torch.cat(
                    [shell_states, goal_states], dim=1)
            elif self.hopf_adaptive and independent_measure:
                # Score only the learned scalar speed and its local sharpness.
                # No stored collocation speed is used as a target.
                candidate_n = min(
                    independent_pool.shape[0], max(4 * n_hopf, n_hopf))
                candidate_idx = torch.randperm(
                    independent_pool.shape[0], device=tau.device)[:candidate_n]
                candidate_states = independent_pool[candidate_idx].detach().clone()
                candidate_states.requires_grad_(True)
                speed_for_score = self.network.predict_propagation_speed(
                    candidate_states)[:, 0]
                speed_score_grad = torch.autograd.grad(
                    speed_for_score.sum(), candidate_states, create_graph=False,
                    retain_graph=True)[0]
                speed_for_score = torch.nan_to_num(
                    speed_for_score.detach(), nan=1.0, posinf=1.0,
                    neginf=speed_floor)
                speed_score_grad = torch.nan_to_num(
                    speed_score_grad.detach(), nan=0.0, posinf=0.0,
                    neginf=0.0)
                sharpness = torch.linalg.vector_norm(
                    speed_score_grad, dim=1)
                sharpness = sharpness / sharpness.mean().clamp_min(1e-8)
                boundary_score = speed_for_score.clamp_min(
                    speed_floor).pow(-self.hopf_adaptive_speed_power)
                boundary_score = boundary_score / boundary_score.mean().clamp_min(1e-8)
                sample_score = boundary_score + (
                    self.hopf_adaptive_sharpness_weight
                    * sharpness.detach().clamp(max=10.0))
                sample_score = torch.nan_to_num(
                    sample_score, nan=1.0, posinf=1.0, neginf=0.0)
                selected = torch.multinomial(
                    sample_score.clamp_min(1e-8), n_hopf, replacement=False)
                independent_states = candidate_states[selected].detach()
                if independent_pool_speed is not None:
                    independent_speed_targets = (
                        independent_pool_speed[candidate_idx[selected]].detach())
                collocation_pairs = torch.cat([
                    independent_states, Xp[hopf_idx, self.dim:].detach()], dim=1)
            elif independent_measure:
                independent_idx = torch.randperm(
                    independent_pool.shape[0], device=tau.device)[:n_hopf]
                independent_states = independent_pool[independent_idx].detach()
                if independent_pool_speed is not None:
                    independent_speed_targets = (
                        independent_pool_speed[independent_idx].detach())
                collocation_pairs = torch.cat([
                    independent_states, Xp[hopf_idx, self.dim:].detach()], dim=1)
            elif self.hopf_adaptive:
                # Use only the supplied scalar speed and the learned scalar
                # speed's local variation to concentrate elliptic measure near
                # thin boundaries.  The score is detached: it chooses PDE
                # points but is never a supervision target.
                speed_for_score = self.network.predict_propagation_speed(
                    Xp[:, :self.dim])[:, 0]
                speed_score_grad = torch.autograd.grad(
                    speed_for_score.sum(), Xp, create_graph=False,
                    retain_graph=True)[0][:, :self.dim]
                speed_for_score = torch.nan_to_num(
                    speed_for_score.detach(), nan=1.0, posinf=1.0,
                    neginf=speed_floor)
                speed_score_grad = torch.nan_to_num(
                    speed_score_grad.detach(), nan=0.0, posinf=0.0,
                    neginf=0.0)
                sharpness = torch.linalg.vector_norm(
                    speed_score_grad, dim=1)
                sharpness = sharpness / sharpness.mean().clamp_min(1e-8)
                boundary_score = torch.nan_to_num(
                    Yobs[:, 0].detach(), nan=1.0, posinf=1.0,
                    neginf=speed_floor).clamp_min(speed_floor).pow(
                    -self.hopf_adaptive_speed_power)
                boundary_score = (
                    boundary_score / boundary_score.mean().clamp_min(1e-8))
                sample_score = boundary_score + (
                    self.hopf_adaptive_sharpness_weight
                    * sharpness.clamp(max=10.0))
                sample_score = torch.nan_to_num(
                    sample_score, nan=1.0, posinf=1.0, neginf=0.0)
                hopf_idx = torch.multinomial(
                    sample_score.clamp_min(1e-8), n_hopf,
                    replacement=False)
                collocation_pairs = Xp[hopf_idx].detach().clone()
            else:
                collocation_pairs = Xp[hopf_idx].detach().clone()
            if (self.hopf_endpoint_measure
                    and self.hopf_speed_mode == 'target'):
                # Endpoint-pair collocation uses the scalar start-speed label
                # already attached to that same q state.  Pair type, routing
                # label, trajectories, and oracle information remain unused.
                independent_speed_targets = Yobs[hopf_idx, 0].detach()
            n_characteristic = 0
            if (self.hopf_adaptive
                    and self.hopf_characteristic_step > 0.0):
                n_characteristic = min(
                    n_hopf,
                    int(round(
                        n_hopf * self.hopf_characteristic_fraction)))
                if n_characteristic > 0:
                    # One local characteristic displacement changes the
                    # collocation measure only.  Its endpoint is not used as
                    # a path, value target, policy, or Bellman target.
                    if independent_measure:
                        # Independent collocation starts from the frozen
                        # measure-only cloud.  Recompute the local direction
                        # at those states; never borrow an endpoint-pair
                        # direction, which would couple the characteristic
                        # sampler back to the training-row population.
                        char_pairs = collocation_pairs[:n_characteristic].detach().clone()
                        char_pairs.requires_grad_(True)
                        char_tau, _, _, _, _ = self.network.out(
                            char_pairs, detach_input=False)
                        char_grad = self.gradient(
                            char_tau, char_pairs, create_graph=False)
                        descent = -char_grad[:, :self.dim].detach()
                    else:
                        descent = -DT0[hopf_idx[:n_characteristic]].detach()
                    descent = descent / torch.linalg.vector_norm(
                        descent, dim=1, keepdim=True).clamp_min(1e-8)
                    collocation_pairs[:n_characteristic, :self.dim] = (
                        collocation_pairs[
                            :n_characteristic, :self.dim]
                        + self.hopf_characteristic_step * descent
                    ).clamp(-0.5, 0.5)
            hopf_characteristic_count = torch.as_tensor(
                float(n_characteristic), device=tau.device, dtype=tau.dtype)

            collocation_tau, _, collocation_pairs, _, _ = self.network.out(
                collocation_pairs)
            collocation_grad = _finite_derivative(self.gradient(
                collocation_tau, collocation_pairs))
            collocation_grad0 = collocation_grad[:, :self.dim]
            collocation_s0 = collocation_grad0.square().sum(dim=1)
            laplacian = torch.zeros(
                n_hopf, device=tau.device, dtype=tau.dtype)
            if (self.hopf_cole_weight > 0.0
                    and self.hopf_residual_form == 'travel_time'):
                for axis in range(self.dim):
                    second = torch.autograd.grad(
                        collocation_grad0[:, axis].sum(), collocation_pairs,
                        create_graph=True, retain_graph=True)[0]
                    second = _finite_derivative(second)
                    laplacian = laplacian + second[:, axis]
            fraction = min(max(
                (float(epoch) - 1.0)
                / max(float(self.hopf_decay_epochs - 1), 1.0),
                0.0), 1.0)
            epsilon = math.exp(
                (1.0 - fraction) * math.log(self.hopf_epsilon_start)
                + fraction * math.log(self.hopf_epsilon_end))
            collocation_state = collocation_pairs[:, :self.dim]
            if self.hopf_speed_mode == 'target':
                # The target coefficient is detached deliberately.  This arm
                # identifies whether the value PDE can select the correct
                # branch when the scalar field is known, without allowing the
                # speed head and value field to co-adapt around one another.
                collocation_speed = independent_speed_targets.to(
                    device=tau.device, dtype=tau.dtype)
            else:
                collocation_speed = self.network.predict_propagation_speed(
                    collocation_state)[:, 0]
            # A rare exploding second derivative must not poison the whole
            # optimizer state.  Non-finite PDE samples are discarded from
            # this local measure (they are not converted into supervision
            # targets); training logs should still be checked for any
            # finite-value warning emitted by the surrounding loop.
            collocation_speed = torch.nan_to_num(
                collocation_speed, nan=1.0, posinf=1.0, neginf=speed_floor
            ).clamp_min(speed_floor)
            hopf_interior_loss = torch.zeros_like(collocation_s0)
            if (self.hopf_cole_weight > 0.0
                    and self.hopf_residual_form == 'travel_time'):
                hopf_residual = collocation_speed.square() * (
                    collocation_s0 - 2.0 * epsilon * laplacian) - 1.0
                hopf_residual = torch.nan_to_num(
                    hopf_residual, nan=0.0, posinf=100.0, neginf=-100.0)
                hopf_interior_loss = hopf_residual.square()
            elif (self.hopf_cole_weight > 0.0
                  and self.hopf_residual_form == 'desirability_ritz'):
                # Hopf--Cole turns the viscous Eikonal equation into the
                # linear elliptic boundary-value problem
                #
                #   4 eps^2 F(q)^2 Delta z - z = 0,
                #   z = exp(-T / (2 eps)).
                #
                # Dividing by F^2 gives the Euler equation of the positive
                # Dirichlet energy below.  Multiplication by 8 eps^2 keeps
                # its numerical scale O(1) without changing the minimizer:
                #
                #   E = 4 eps^2 |grad z|^2 + z^2 / F^2.
                #
                # This weak form propagates the finite goal-cell boundary in
                # every direction and needs only first derivatives.  It is
                # therefore distinct from squaring the nonlinear strong-form
                # residual in T, which can select a locally Eikonal but
                # globally Euclidean branch in high dimension.
                desirability = torch.exp(
                    -collocation_tau[:, 0].clamp_min(0.0)
                    / (2.0 * epsilon))
                desirability_grad = (
                    -desirability[:, None] * collocation_grad0
                    / (2.0 * epsilon))
                hopf_interior_loss = (
                    4.0 * epsilon * epsilon
                    * desirability_grad.square().sum(dim=1)
                    + desirability.square()
                    / collocation_speed.square().clamp_min(speed_floor ** 2))
                if self.hopf_shellwise_normalize:
                    # A single global z^2 normalization is still dominated
                    # exponentially by the nearest goal shell.  Normalize
                    # each log-distance shell independently, then average the
                    # equally populated shells.  Each detached factor is a
                    # positive constant on that shell, so this changes the
                    # collocation conditioning without introducing a value
                    # target, route, graph, or second field.
                    shell_scaled = torch.empty_like(hopf_interior_loss)
                    for shell_index in range(self.hopf_shell_bins):
                        shell_mask = shell_bin == shell_index
                        if not bool(shell_mask.any()):
                            continue
                        shell_scale = desirability[shell_mask].square().mean(
                        ).detach().clamp_min(1e-12)
                        shell_scaled[shell_mask] = (
                            hopf_interior_loss[shell_mask] / shell_scale)
                    hopf_interior_loss = shell_scaled
                elif self.hopf_ritz_normalize:
                    # Small-epsilon desirability is exponentially tiny at
                    # ordinary pair distances.  A detached scalar rescales
                    # the positive energy without changing its stationary
                    # solution; cap amplification at 1e6.
                    ritz_scale = (
                        desirability.square().mean().detach().clamp_min(1e-6))
                    hopf_interior_loss = hopf_interior_loss / ritz_scale
                hopf_interior_loss = torch.nan_to_num(
                    hopf_interior_loss, nan=0.0, posinf=10000.0,
                    neginf=0.0)
                hopf_residual = torch.zeros_like(collocation_s0)
            else:
                # Do not spend second-order autograd or inject a residual when
                # this is the first-order state-constraint-only ablation.
                hopf_residual = torch.zeros_like(collocation_s0)

            state_constraint = torch.zeros_like(hopf_residual)
            if self.hopf_state_constraint_weight > 0.0:
                # grad(F) points from the low-speed boundary toward free
                # space.  Eikonal descent -grad(T) must not cross in the
                # opposite direction.  Detaching grad(F) prevents this term
                # from reshaping the separately speed-supervised scalar F.
                boundary_speed = self.network.predict_propagation_speed(
                    collocation_state)[:, 0]
                speed_gradient = torch.autograd.grad(
                    boundary_speed.sum(), collocation_pairs,
                    create_graph=False, retain_graph=True)[0][:, :self.dim]
                speed_normal = speed_gradient.detach()
                speed_normal = speed_normal / torch.linalg.vector_norm(
                    speed_normal, dim=1, keepdim=True).clamp_min(1e-8)
                value_normal = collocation_grad0 / torch.linalg.vector_norm(
                    collocation_grad0, dim=1, keepdim=True).clamp_min(1e-8)
                outward = (value_normal * speed_normal).sum(dim=1)
                boundary_mask = (
                    collocation_speed.detach() <= self.hopf_boundary_speed)
                # Keep the local penalty unweighted here; the common loss
                # assembly below applies hopf_state_constraint_weight once.
                # This lets state-only mode use the same term without the
                # historical accidental weight-squared behavior.
                state_constraint = (
                    torch.relu(outward).square()
                    * boundary_mask.to(outward.dtype))
                state_constraint = torch.nan_to_num(
                    state_constraint, nan=0.0, posinf=100.0, neginf=0.0)
                hopf_boundary_fraction = boundary_mask.to(tau.dtype).mean()
                if bool(boundary_mask.any()):
                    hopf_state_violation_rate = (
                        (outward[boundary_mask] > 0).to(tau.dtype).mean())

            # A point is a zero-capacity boundary for an elliptic PDE in 6D.
            # Give the goal one finite cell and close it using only the same
            # learned scalar speed's local distance/F asymptotic.
            goal_anchor = Xp[hopf_idx, self.dim:].detach()
            offset = torch.randn_like(goal_anchor)
            offset = offset / torch.linalg.vector_norm(
                offset, dim=1, keepdim=True).clamp_min(1e-8)
            radius = self.hopf_goal_radius * torch.rand(
                n_hopf, 1, device=tau.device, dtype=tau.dtype).pow(
                    1.0 / float(self.dim))
            near_state = (goal_anchor + radius * offset).clamp(-0.5, 0.5)
            near_pairs = torch.cat([near_state, goal_anchor], dim=1)
            near_tau, _, _, _, _ = self.network.out(near_pairs)
            midpoint = 0.5 * (near_state + goal_anchor)
            with torch.no_grad():
                midpoint_speed = self.network.predict_propagation_speed(
                    midpoint)[:, 0]
                midpoint_speed = torch.nan_to_num(
                    midpoint_speed, nan=1.0, posinf=1.0,
                    neginf=speed_floor).clamp_min(speed_floor)
                local_target = torch.linalg.vector_norm(
                    near_state - goal_anchor, dim=1) / midpoint_speed
            if self.hopf_residual_form == 'desirability_ritz':
                near_desirability = torch.exp(
                    -near_tau[:, 0].clamp_min(0.0) / (2.0 * epsilon))
                target_desirability = torch.exp(
                    -local_target.clamp_min(0.0) / (2.0 * epsilon))
                local_residual = near_desirability - target_desirability
            else:
                local_residual = near_tau[:, 0] - local_target
            local_residual = torch.nan_to_num(
                local_residual, nan=0.0, posinf=100.0, neginf=-100.0)
            hopf_cole_loss[hopf_idx] = (
                self.hopf_cole_weight * (
                    hopf_interior_loss
                    + self.hopf_goal_weight * local_residual.square())
                + self.hopf_state_constraint_weight * state_constraint)
            hopf_cole_loss = torch.nan_to_num(
                hopf_cole_loss, nan=0.0, posinf=10000.0, neginf=0.0)

        # Semi-Lagrangian dynamic programming for the viscosity solution:
        #   T(q,g) = min_u [ ds/F(q) + T(q + ds*u, g) ].
        # Its one-step cost uses only scalar speed: either the model's scalar
        # speed head, or (for the vanilla metric architecture) the supplied
        # endpoint speed label.  No trajectory, collision oracle, normal,
        # teacher, or external critic is queried.  Finite-step consistency
        # rejects locally valid but globally wrong Eikonal branches.
        viscosity_loss = torch.zeros_like(diff)
        # Independent collocation q--goal pairs have no corresponding row in
        # the endpoint minibatch. Keep their Bellman residual as a scalar so
        # it is not diluted by the (usually much larger) endpoint batch.
        viscosity_cloud_loss = torch.zeros(
            (), device=tau.device, dtype=tau.dtype)
        viscosity_domain_valid_rate = torch.zeros(
            (), device=tau.device, dtype=tau.dtype)
        viscosity_upwind_rate = torch.zeros(
            (), device=tau.device, dtype=tau.dtype)
        viscosity_signed_residual = torch.zeros(
            (), device=tau.device, dtype=tau.dtype)
        viscosity_target_above_rate = torch.zeros(
            (), device=tau.device, dtype=tau.dtype)
        viscosity_local_boundary_rate = torch.zeros(
            (), device=tau.device, dtype=tau.dtype)
        viscosity_pool = None
        if self.eikonal_min_speed > 0:
            viscosity_pool = torch.nonzero(
                (sq_Yobs0 >= self.eikonal_min_speed)
                & (sq_Yobs1 >= self.eikonal_min_speed),
                as_tuple=False).reshape(-1)
        viscosity_active = (
            self.viscosity_weight > 0
            and self.viscosity_n > 0
            and epoch >= self.viscosity_start_epoch
            and (
                speed_pred_2B is not None
                or getattr(self.network, 'value_arch', None) == 'metric')
            and (
                getattr(self.network, 'value_condition', None) == 'none'
                or getattr(self.network, 'value_arch', None) == 'metric'
                or self.viscosity_action_mode == 'stochastic_vv')
            and getattr(self.network, 'value_arch', None) in (
                'metric', 'global_residual', 'global_residual_symmetric',
                'bounded_chord_symmetric',
                'bounded_chord_fermat_curve_symmetric',
                'bounded_chord_endpoint_fermat_curve_symmetric',
                'bounded_chord_endpoint_speed_fermat_curve_symmetric',
                'bounded_chord_log_symmetric',
                'chord_integral_symmetric',
                'bounded_chord_hazard_symmetric',
                'bounded_chord_hazard_floor_symmetric',
                'bounded_chord_eikonal_symmetric',
                'bounded_chord_eikonal_refine_symmetric',
                'bounded_chord_endpoint_eikonal_symmetric',
                'factored_source_eikonal_symmetric',
                'bounded_chord_local_endpoint_eikonal_symmetric',
                'bounded_chord_speed_barrier_symmetric',
                'bounded_chord_local_speed_barrier_symmetric',
                'bounded_chord_global_excess_symmetric',
                'bounded_chord_smooth_excess_symmetric',
                'bounded_chord_neural_viscosity_symmetric',
                'bounded_chord_signed_clearance_symmetric',
                'bounded_chord_clearance_eikonal_symmetric',
                'bounded_chord_clearance_state_symmetric',
                'endpoint_eikonal_residual_symmetric',
                'local_endpoint_eikonal_residual_symmetric',
                'bounded_chord_profile_symmetric',
                'fermat_waypoint_symmetric',
                'fermat_curve_symmetric')
            and (viscosity_pool is None or viscosity_pool.numel() > 0)
        )
        if viscosity_active:
            target_network = getattr(
                self, 'viscosity_target_network', self.network)
            use_lagged_target = target_network is not self.network
            use_independent_pairs = bool(self.viscosity_collocation_pairs)
            if viscosity_pool is not None:
                if use_independent_pairs:
                    # The independent collocation cloud is a q-only measure;
                    # its scalar speed labels are used only to keep both
                    # endpoints inside the strict PDE domain.  No task/zone
                    # or pair label is consulted.
                    if (collocation_points is None
                            or collocation_speed_labels is None):
                        raise RuntimeError(
                            "viscosity_collocation_pairs requires the "
                            "independent scalar-speed collocation cloud")
                    pde_pool = collocation_points.to(
                        device=tau.device, dtype=Xp.dtype)
                    pde_labels = collocation_speed_labels.to(
                        device=tau.device, dtype=Xp.dtype).reshape(-1)
                    if (pde_pool.ndim != 2
                            or pde_pool.shape[1] != self.dim
                            or pde_pool.shape[0] != pde_labels.numel()):
                        raise ValueError(
                            "viscosity collocation cloud and labels must "
                            "have shapes (M, dim) and (M,)")
                    pde_valid = torch.isfinite(pde_labels)
                    if self.eikonal_min_speed > 0.0:
                        pde_valid &= (
                            pde_labels >= self.eikonal_min_speed)
                    pde_pool = pde_pool[pde_valid]
                    if pde_pool.shape[0] < 2:
                        raise RuntimeError(
                            "viscosity collocation cloud has fewer than "
                            "two strict-feasible states")
                    # Draw paired states and their exact reversals.  This is
                    # an explicitly symmetric q<->goal measure, while the
                    # deployed value remains one scalar field.  A controlled
                    # fraction may instead be a local point-source closure:
                    # one strict-feasible cloud state is the goal and the
                    # start is an isotropic q-only displacement.  The local
                    # cell is deliberately generated without a task label,
                    # route label, path, or collision query.
                    n = min(self.viscosity_n, int(pde_pool.shape[0]))
                    n = max(2, n - (n % 2))
                    m = n // 2
                    local_m = int(round(
                        m * self.viscosity_local_boundary_fraction))
                    local_m = min(max(local_m, 0), m)
                    global_m = m - local_m
                    if global_m:
                        q_perm = torch.randperm(
                            pde_pool.shape[0], device=tau.device)[:global_m]
                        g_perm = torch.randperm(
                            pde_pool.shape[0], device=tau.device)[:global_m]
                        q_global = pde_pool[q_perm].detach()
                        g_global = pde_pool[g_perm].detach()
                    else:
                        q_global = pde_pool[:0].detach()
                        g_global = pde_pool[:0].detach()
                    if local_m:
                        local_goal_idx = torch.randperm(
                            pde_pool.shape[0], device=tau.device)[:local_m]
                        g_local = pde_pool[local_goal_idx].detach()
                        local_direction = torch.randn(
                            local_m, self.dim,
                            device=tau.device, dtype=tau.dtype)
                        local_direction = local_direction / (
                            torch.linalg.vector_norm(
                                local_direction, dim=1, keepdim=True
                            ).clamp_min(1e-8))
                        local_radius = torch.empty(
                            local_m, 1, device=tau.device, dtype=tau.dtype
                        ).uniform_(
                            self.viscosity_boundary_min_radius,
                            self.viscosity_boundary_max_radius)
                        q_local = g_local + local_radius * local_direction
                        # Reflecting box is the same scalar-domain convention
                        # used by the Bellman stencil below.  It avoids a
                        # clipping pile-up at +/-0.5 while preserving the
                        # isotropic local measure.
                        phase = torch.remainder(q_local + 0.5, 2.0)
                        q_local = -0.5 + torch.where(
                            phase <= 1.0, phase, 2.0 - phase)
                    else:
                        q_local = pde_pool[:0].detach()
                        g_local = pde_pool[:0].detach()
                    q_half = torch.cat([q_global, q_local], dim=0)
                    g_half = torch.cat([g_global, g_local], dim=0)
                    if q_half.shape[0] != m:
                        raise RuntimeError(
                            "invalid independent scalar PDE pair count")
                    local_boundary_fraction = torch.as_tensor(
                        float(local_m) / float(max(m, 1)),
                        device=tau.device, dtype=tau.dtype)
                    viscosity_local_boundary_rate = local_boundary_fraction
                    q = torch.cat([q_half, g_half], dim=0)
                    goal = torch.cat([g_half, q_half], dim=0)
                    independent_pairs = torch.cat([q, goal], dim=1)
                    independent_pairs.requires_grad_(True)
                    ones = torch.ones(
                        2 * n, device=tau.device, dtype=tau.dtype)
                    zeros = torch.zeros_like(ones)
                    tau_visc, _, X_visc, _, _ = self.network.out(
                        independent_pairs,
                        w_cup_override=ones,
                        w_pour_override=zeros,
                        detach_input=False)
                    dtau_visc = _finite_derivative(
                        self.gradient(tau_visc, X_visc))
                    tau_here = tau_visc[:, 0]
                    DT_q = dtau_visc[:, :self.dim]
                    idx = None
                else:
                    n = min(self.viscosity_n, int(viscosity_pool.numel()))
                    idx = viscosity_pool[
                        torch.randperm(
                            viscosity_pool.numel(), device=tau.device)[:n]]
                    q = Xp[idx, :self.dim].detach()
                    goal = Xp[idx, self.dim:].detach()
                    tau_here = tau[idx, 0]
                    DT_q = DT0[idx]
            else:
                n = min(self.viscosity_n, B)
                idx = torch.randperm(B, device=tau.device)[:n]
                q = Xp[idx, :self.dim].detach()
                goal = Xp[idx, self.dim:].detach()
                tau_here = tau[idx, 0]
                DT_q = DT0[idx]
            # A local Eikonal residual cannot distinguish globally wrong
            # characteristic branches.  The optional global-cloud operator
            # is a mixed local/nonlocal softmin discretization of the same
            # scalar Bellman equation: half the candidates retain the local
            # stencil and half are sampled from the independent positive-
            # speed q cloud. Every transition cost is integrated from this
            # model's scalar F(q), with no task/pair/path/teacher label.
            global_cloud_rollout = bool(self.viscosity_global_cloud)
            if global_cloud_rollout:
                if not use_independent_pairs:
                    raise RuntimeError(
                        "global scalar-cloud Bellman requires "
                        "viscosity_collocation_pairs")
                if self.viscosity_rollout_steps != 1:
                    raise ValueError(
                        "global scalar-cloud Bellman does not combine with "
                        "multi-stage local rollout")
                k = min(self.viscosity_dirs, int(pde_pool.shape[0]))
                k_global = max(1, k // 2)
                k_local = max(1, k - k_global)
                # Retain a local stencil alongside the nonlocal cloud.  The
                # local half supplies the source/upwind consistency that a
                # cloud-only transition lacks; the cloud half is the only
                # new global information channel.
                goal_dir = goal - q
                goal_dir = goal_dir / (
                    torch.linalg.vector_norm(
                        goal_dir, dim=1, keepdim=True) + 1e-8)
                local_parts = [goal_dir[:, None, :]]
                if self.viscosity_include_descent:
                    descent = -DT_q.detach()
                    descent = descent / (
                        torch.linalg.vector_norm(
                            descent, dim=1, keepdim=True) + 1e-8)
                    local_parts.insert(0, descent[:, None, :])
                eye = torch.eye(
                    self.dim, device=tau.device, dtype=tau.dtype)
                axes = torch.cat([eye, -eye], dim=0)[None].expand(n, -1, -1)
                local_parts.append(axes)
                local_directions = torch.cat(local_parts, dim=1)
                if local_directions.shape[1] < k_local:
                    random_dirs = torch.randn(
                        n, k_local - local_directions.shape[1], self.dim,
                        device=tau.device, dtype=tau.dtype)
                    random_dirs = random_dirs / torch.linalg.vector_norm(
                        random_dirs, dim=2, keepdim=True).clamp_min(1e-8)
                    local_directions = torch.cat(
                        [local_directions, random_dirs], dim=1)
                local_directions = local_directions[:, :k_local]
                if self.viscosity_step_max > self.viscosity_step:
                    local_step = torch.exp(torch.empty(
                        n, k_local, device=tau.device, dtype=tau.dtype
                    ).uniform_(
                        math.log(self.viscosity_step),
                        math.log(self.viscosity_step_max)))
                else:
                    local_step = torch.full(
                        (n, k_local), self.viscosity_step,
                        device=tau.device, dtype=tau.dtype)
                q_local = q[:, None, :] + local_step[:, :, None] * local_directions
                candidate_idx = torch.randint(
                    pde_pool.shape[0], (n, k_global), device=tau.device)
                q_global = pde_pool[candidate_idx].detach()
                global_step = torch.linalg.vector_norm(
                    q_global - q[:, None, :], dim=2).clamp_min(1e-4)
                q_next = torch.cat([q_local, q_global], dim=1)
                step = torch.cat([local_step, global_step], dim=1)
                directions = None
            else:
                if self.viscosity_step_max > self.viscosity_step:
                    lo = math.log(self.viscosity_step)
                    hi = math.log(self.viscosity_step_max)
                    step = torch.exp(torch.empty(
                        n, device=tau.device, dtype=tau.dtype).uniform_(lo, hi))
                else:
                    step = torch.full(
                        (n,), self.viscosity_step,
                        device=tau.device, dtype=tau.dtype)

                goal_dir = goal - q
                goal_dir = goal_dir / (
                    torch.linalg.vector_norm(
                        goal_dir, dim=1, keepdim=True) + 1e-8)
                direction_parts = [goal_dir[:, None, :]]
                if self.viscosity_include_speed_ascent:
                    # Fermat rays bend according to the transverse gradient
                    # of the learned scalar F(q), not a normal or path label.
                    q_speed = q.detach().requires_grad_(True)
                    speed_at_q = target_network.predict_propagation_speed(
                        q_speed)[:, 0]
                    speed_grad = torch.autograd.grad(
                        speed_at_q.sum(), q_speed,
                        create_graph=False, retain_graph=False)[0].detach()
                    speed_grad_norm = torch.linalg.vector_norm(
                        speed_grad, dim=1, keepdim=True)
                    clearance_dir = torch.where(
                        speed_grad_norm > 1e-6,
                        speed_grad / speed_grad_norm.clamp_min(1e-8),
                        goal_dir)
                    transverse = speed_grad - (
                        speed_grad * goal_dir).sum(
                            dim=1, keepdim=True) * goal_dir
                    transverse_norm = torch.linalg.vector_norm(
                        transverse, dim=1, keepdim=True)
                    transverse_dir = torch.where(
                        transverse_norm > 1e-6,
                        transverse / transverse_norm.clamp_min(1e-8),
                        torch.zeros_like(transverse))
                    bent_dir = goal_dir + transverse_dir
                    bent_dir = bent_dir / torch.linalg.vector_norm(
                        bent_dir, dim=1, keepdim=True).clamp_min(1e-8)
                    direction_parts.extend([
                        clearance_dir[:, None, :],
                        bent_dir[:, None, :],
                    ])
                if self.viscosity_include_descent:
                    # Kept for the local operator; the global cloud has no
                    # self-generated direction bank.
                    descent = -DT_q.detach()
                    descent = descent / (
                        torch.linalg.vector_norm(
                            descent, dim=1, keepdim=True) + 1e-8)
                    direction_parts.insert(0, descent[:, None, :])

                eye = torch.eye(self.dim, device=tau.device, dtype=tau.dtype)
                axes = torch.cat([eye, -eye], dim=0)
                axes = axes[None].expand(n, -1, -1)
                direction_parts.append(axes)
                directions = torch.cat(direction_parts, dim=1)
                if directions.shape[1] < self.viscosity_dirs:
                    n_random = self.viscosity_dirs - directions.shape[1]
                    random_dirs = torch.randn(
                        n, n_random, self.dim,
                        device=tau.device, dtype=tau.dtype)
                    random_dirs = random_dirs / (
                        torch.linalg.vector_norm(
                            random_dirs, dim=2, keepdim=True) + 1e-8)
                    directions = torch.cat([directions, random_dirs], dim=1)
                directions = directions[:, :self.viscosity_dirs]
                k = directions.shape[1]
                q_next = q[:, None, :] + step[:, None, None] * directions
            rollout_stage = None
            multi_rollout_stage = None
            multi_rollout_valid = None
            deterministic_rollout = (
                self.viscosity_action_mode == 'softmin'
                and self.viscosity_rollout_steps > 1)
            if deterministic_rollout:
                if self.viscosity_rollout_steps != 2:
                    raise ValueError(
                        "deterministic softmin currently supports exactly "
                        "two Bellman stages")
                if speed_pred_2B is None:
                    raise RuntimeError(
                        "multi-stage softmin requires the shared scalar "
                        "speed head")
                # Compose two isotropic first-order stencils before taking
                # the smooth minimum.  The final state is a genuine bend
                # q -> q_1 -> q_2, not a larger straight chord; all stage
                # costs are integrated from the same q-only scalar F.  This
                # is a local numerical Bellman operator, not a decoded path
                # or evaluator candidate.
                first_k = k
                q_first = q_next
                q_second = (
                    q_first[:, :, None, :]
                    + step[:, None, None, None]
                    * directions[:, None, :, :])
                first_valid = (q_first.abs() <= 0.5).all(dim=2)
                second_valid = (q_second.abs() <= 0.5).all(dim=3)
                multi_rollout_valid = (
                    first_valid[:, :, None] & second_valid)
                with torch.no_grad():
                    speed_q = target_network.predict_propagation_speed(
                        q)[:, 0]
                    speed_first = target_network.predict_propagation_speed(
                        q_first.reshape(n * first_k, self.dim)
                    )[:, 0].reshape(n, first_k)
                    q_mid_first = 0.5 * (
                        q[:, None, :] + q_first)
                    speed_mid_first = (
                        target_network.predict_propagation_speed(
                            q_mid_first.reshape(n * first_k, self.dim)
                        )[:, 0].reshape(n, first_k))
                    q_mid_second = 0.5 * (
                        q_first[:, :, None, :] + q_second)
                    speed_mid_second = (
                        target_network.predict_propagation_speed(
                            q_mid_second.reshape(
                                n * first_k * first_k, self.dim)
                        )[:, 0].reshape(n, first_k, first_k))
                    speed_second = (
                        target_network.predict_propagation_speed(
                            q_second.reshape(
                                n * first_k * first_k, self.dim)
                        )[:, 0].reshape(n, first_k, first_k))
                    safe_q = speed_q.clamp_min(speed_floor)
                    safe_first = speed_first.clamp_min(speed_floor)
                    safe_mid_first = speed_mid_first.clamp_min(
                        speed_floor)
                    safe_mid_second = speed_mid_second.clamp_min(
                        speed_floor)
                    safe_second = speed_second.clamp_min(speed_floor)
                    stage_first = (step[:, None] / 6.0) * (
                        safe_q[:, None].reciprocal()
                        + 4.0 * safe_mid_first.reciprocal()
                        + safe_first.reciprocal())
                    stage_second = (step[:, None, None] / 6.0) * (
                        safe_first[:, :, None].reciprocal()
                        + 4.0 * safe_mid_second.reciprocal()
                        + safe_second.reciprocal())
                    multi_rollout_stage = (
                        stage_first[:, :, None] + stage_second
                    ).reshape(n, first_k * first_k)
                q_next = q_second.reshape(n, first_k * first_k, self.dim)
                k = first_k * first_k
            if self.viscosity_action_mode == 'stochastic_vv':
                # Vanishing-viscosity Eikonal update.  The deterministic drift
                # is the unique continuous characteristic; isotropic Brownian
                # diffusion lets information cross a developing shock before
                # epsilon -> 0 selects the viscosity solution.  Antithetic
                # walkers have exactly zero sample-mean noise and introduce no
                # direction bank, policy, graph, or path label.
                characteristic = DT_q.detach()
                characteristic = characteristic / torch.linalg.vector_norm(
                    characteristic, dim=1, keepdim=True).clamp_min(1e-8)
                half = k // 2
                if self.viscosity_rollout_steps == 1:
                    noise = torch.randn(
                        n, half, self.dim,
                        device=tau.device, dtype=tau.dtype)
                    noise = torch.cat([noise, -noise], dim=1)
                    diffusion = torch.sqrt(
                        2.0 * self.viscosity_epsilon * step
                    )[:, None, None]
                    q_next = (
                        q[:, None, :]
                        - step[:, None, None]
                        * characteristic[:, None, :]
                        + diffusion * noise)
                else:
                    if speed_pred_2B is None:
                        raise RuntimeError(
                            "multi-step stochastic_vv requires the shared "
                            "continuous speed head")
                    # Repeated Feynman--Kac propagation through the same
                    # scalar field.  The stopping clock is intrinsic time
                    # sum(ds/F), so free regions cover more configuration
                    # distance while slow regions automatically take shorter
                    # spatial steps.  K is only a numerical safety ceiling.
                    states = q[:, None, :].expand(-1, k, -1).clone()
                    goals = goal[:, None, :].expand(-1, k, -1)
                    accumulated = torch.zeros(
                        n, k, device=tau.device, dtype=tau.dtype)
                    active = torch.ones(
                        n, k, device=tau.device, dtype=torch.bool)
                    if use_independent_pairs:
                        with torch.no_grad():
                            state_speed = target_network.predict_propagation_speed(
                                q)[:, 0].detach()
                    else:
                        state_speed = speed_pred_2B[idx, 0].detach()
                    state_speed = state_speed[:, None].expand(-1, k)
                    state_characteristic = characteristic[:, None, :].expand(
                        -1, k, -1)

                    if use_independent_pairs:
                        rollout_w_cup = torch.ones(
                            2 * n * k, device=tau.device, dtype=tau.dtype)
                        rollout_w_pour = torch.zeros_like(rollout_w_cup)
                    else:
                        rollout_w_cup = torch.cat([
                            w_cup_2B[idx, None].expand(-1, k).reshape(-1),
                            w_cup_2B[B + idx, None].expand(-1, k).reshape(-1),
                        ]).detach()
                        rollout_w_pour = torch.cat([
                            w_pour_2B[idx, None].expand(-1, k).reshape(-1),
                            w_pour_2B[B + idx, None].expand(-1, k).reshape(-1),
                        ]).detach()

                    for rollout_index in range(
                            self.viscosity_rollout_steps):
                        if rollout_index > 0:
                            walker_pairs = torch.cat(
                                [states, goals], dim=2
                            ).reshape(n * k, 2 * self.dim)
                            walker_tau, _, walker_xp, walker_features, _ = (
                                target_network.out(
                                    walker_pairs,
                                    w_cup_override=rollout_w_cup,
                                    w_pour_override=rollout_w_pour))
                            walker_grad = self.gradient(
                                walker_tau, walker_xp,
                                create_graph=False)[:, :self.dim].detach()
                            state_characteristic = walker_grad.reshape(
                                n, k, self.dim)
                            state_characteristic = state_characteristic / (
                                torch.linalg.vector_norm(
                                    state_characteristic, dim=2,
                                    keepdim=True).clamp_min(1e-8))
                            with torch.no_grad():
                                state_speed = (
                                    target_network.predict_propagation_speed(
                                        states.reshape(n * k, self.dim),
                                        features=walker_features[:n * k])
                                    [:, 0].reshape(n, k))

                        speed_safe = state_speed.clamp_min(speed_floor)
                        time_left = (
                            self.viscosity_intrinsic_budget
                            - accumulated).clamp_min(0.0)
                        local_step = torch.minimum(
                            step[:, None].expand(-1, k),
                            time_left * speed_safe)
                        distance_to_goal = torch.linalg.vector_norm(
                            goals - states, dim=2)
                        reaches_goal = active & (
                            distance_to_goal <= local_step)
                        moving = active & ~reaches_goal & (local_step > 1e-9)

                        accumulated = accumulated + torch.where(
                            reaches_goal,
                            distance_to_goal / speed_safe,
                            torch.where(
                                moving, local_step / speed_safe,
                                torch.zeros_like(local_step)))

                        noise = torch.randn(
                            n, half, self.dim,
                            device=tau.device, dtype=tau.dtype)
                        noise = torch.cat([noise, -noise], dim=1)
                        proposal = (
                            states
                            - local_step[:, :, None] * state_characteristic
                            + torch.sqrt(
                                2.0 * self.viscosity_epsilon * local_step
                            )[:, :, None] * noise)
                        phase = torch.remainder(proposal + 0.5, 2.0)
                        proposal = -0.5 + torch.where(
                            phase <= 1.0, phase, 2.0 - phase)
                        states = torch.where(
                            moving[:, :, None], proposal, states)
                        states = torch.where(
                            reaches_goal[:, :, None], goals, states)
                        active = moving & (
                            accumulated
                            < self.viscosity_intrinsic_budget - 1e-7)

                    q_next = states
                    rollout_stage = accumulated

                # Reflecting state boundary, matching the Neumann boundary of
                # the viscous PDE without rejecting or clipping actions.
                phase = torch.remainder(q_next + 0.5, 2.0)
                q_next = -0.5 + torch.where(
                    phase <= 1.0, phase, 2.0 - phase)
            q_mid = 0.5 * (q[:, None, :] + q_next)
            if getattr(self.network, 'lifted_phase', False):
                # Physical normalized joints live in [-0.5, 0.5], whereas
                # the lifted coordinate is deliberately metric-scaled to
                # z in [-phase_scale/2, phase_scale/2].  Treating z as an
                # ordinary joint would let Bellman actions leave the phase
                # domain by up to 10x and bootstrap through an artificial
                # flat extension of p=0 or p=1.
                physical_dim = int(
                    self.network.lifted_phase_physical_dim)
                physical_in_bounds = (
                    q_next[:, :, :physical_dim].abs() <= 0.5).all(dim=2)
                phase_bound = 0.5 * float(
                    self.network.lifted_phase_scale)
                phase_in_bounds = (
                    q_next[:, :, physical_dim:].abs()
                    <= phase_bound + 1e-7).all(dim=2)
                in_bounds = physical_in_bounds & phase_in_bounds
            else:
                in_bounds = (q_next.abs() <= 0.5).all(dim=2)
            if multi_rollout_valid is not None:
                in_bounds = in_bounds & multi_rollout_valid.reshape(n, k)
            if (self.viscosity_causal_target
                    and self.viscosity_causal_order == 'euclidean'):
                # Historical approximation to causality.  It is useful for
                # reproducibility, but Euclidean proximity is not the upwind
                # ordering of an obstacle-aware Eikonal solution: a valid
                # geodesic may first move sideways or away from the goal.
                dist_here = torch.linalg.vector_norm(
                    goal - q, dim=1, keepdim=True)
                dist_next = torch.linalg.vector_norm(
                    goal[:, None, :] - q_next, dim=2)
                in_bounds = in_bounds & (dist_next < dist_here)
            candidate_pairs = torch.cat([
                q_next,
                goal[:, None, :].expand(-1, k, -1),
            ], dim=2).reshape(n * k, 2 * self.dim)
            if (self.viscosity_candidate_routing == 'local'
                    and not use_independent_pairs):
                # A Bellman equation must bootstrap within the same value
                # field.  The training PDE treats routing weights as locally
                # fixed when differentiating q, so use that identical local
                # convention for q_next and preserve the original goal's
                # routing.  The historical all-cup override mixed different
                # fields on the two sides of the Bellman identity.
                candidate_w_cup = torch.cat([
                    w_cup_2B[idx, None].expand(-1, k).reshape(-1),
                    w_cup_2B[B + idx, None].expand(-1, k).reshape(-1),
                ]).detach()
                candidate_w_pour = torch.cat([
                    w_pour_2B[idx, None].expand(-1, k).reshape(-1),
                    w_pour_2B[B + idx, None].expand(-1, k).reshape(-1),
                ]).detach()
            else:
                candidate_w_cup = torch.ones(
                    2 * n * k, device=tau.device, dtype=tau.dtype)
                candidate_w_pour = torch.zeros_like(candidate_w_cup)
            # With fitted value iteration, the right-hand side is evaluated
            # by a frozen, periodically synchronized training-only copy.  The
            # deployed/online network remains the sole left-hand prediction.
            target_context = (
                torch.no_grad()
                if (use_lagged_target or self.viscosity_causal_target
                    or self.viscosity_action_mode == 'stochastic_vv')
                else contextlib.nullcontext())
            with target_context:
                tau_next, _, _, candidate_features, _ = target_network.out(
                    candidate_pairs,
                    w_cup_override=candidate_w_cup,
                    w_pour_override=candidate_w_pour)
            tau_next_matrix = tau_next[:, 0].reshape(n, k)
            if (self.viscosity_causal_target
                    and self.viscosity_causal_order == 'value'):
                # The Eikonal equation is causal in arrival time, not in
                # straight-line distance to the goal.  Accept any geometric
                # direction -- including a temporary detour -- provided its
                # detached value is genuinely upwind (lower T).  This uses no
                # normal, trajectory, collision oracle, or teacher label.
                tau_here = tau_here.detach()[:, None]
                in_bounds = in_bounds & (
                    tau_next_matrix.detach() < tau_here)
            candidate_points = torch.cat([
                candidate_pairs[:, :self.dim],
                candidate_pairs[:, self.dim:]], dim=0)
            if speed_pred_2B is None:
                # Clean vanilla NTFields experiment: F(q) is already the only
                # scalar supervision supplied by the dataset.  Holding that
                # local coefficient constant over a first-order stencil gives
                # ds/F(q); neighbor values propagate its effect globally.
                # This deliberately adds neither a speed head nor an off-grid
                # geometry query.
                if use_independent_pairs:
                    with torch.no_grad():
                        speed_here = target_network.predict_propagation_speed(
                            q)[:, 0].reshape(n, 1)
                else:
                    speed_here = Yobs[idx, 0].detach()[:, None]
                speed_next = speed_here.expand(-1, k)
            else:
                with torch.no_grad():
                    speed_next = target_network.predict_propagation_speed(
                        candidate_points,
                        features=candidate_features)[:n * k, 0].reshape(n, k)
                    if use_independent_pairs or use_lagged_target:
                        speed_here = target_network.predict_propagation_speed(
                            q)[:, 0].reshape(n, 1)
                    else:
                        speed_here = speed_pred_2B[idx, 0][:, None]
            if multi_rollout_stage is not None:
                stage = multi_rollout_stage
            elif rollout_stage is not None:
                # The repeated rollout already integrated ds/F at every
                # visited state.  Only the final continuation value remains.
                stage = rollout_stage
            elif self.viscosity_hard_domain:
                # State-constraint Eikonal problem: obstacle interiors are
                # outside the PDE domain, not merely finite-cost slow space.
                # Probe three interior points so a large Bellman step cannot
                # jump across a thin forbidden band.  Boole's five-point rule
                # also gives a higher-order stage-cost integral using only the
                # same learned scalar speed F(q).
                fractions = torch.tensor(
                    [0.25, 0.5, 0.75], device=tau.device,
                    dtype=tau.dtype)
                q_interior = (
                    q[:, None, None, :]
                    + fractions[None, None, :, None]
                    * (q_next - q[:, None, :])[:, :, None, :])
                with torch.no_grad():
                    speed_interior = target_network.predict_propagation_speed(
                        q_interior.reshape(n * k * 3, self.dim)
                    )[:, 0].reshape(n, k, 3)
                speed_samples = torch.cat([
                    speed_here.expand(-1, k)[:, :, None],
                    speed_interior,
                    speed_next[:, :, None],
                ], dim=2)
                domain_valid = (
                    speed_samples >= self.eikonal_min_speed).all(dim=2)
                in_bounds = in_bounds & domain_valid
                viscosity_domain_valid_rate = domain_valid.float().mean()
                boole_weights = torch.tensor(
                    [7.0, 32.0, 12.0, 32.0, 7.0],
                    device=tau.device, dtype=tau.dtype)
                stage_factor = step if global_cloud_rollout else step[:, None]
                stage = (stage_factor / 90.0) * (
                    speed_samples.clamp_min(speed_floor).reciprocal()
                    * boole_weights[None, None, :]).sum(dim=2)
            elif speed_pred_2B is None:
                # First-order monotone semi-Lagrangian stage cost.  Candidate
                # values, rather than an invented off-grid speed target, choose
                # which direction carries information toward the current state.
                stage_factor = step if global_cloud_rollout else step[:, None]
                stage = stage_factor / speed_here[:, None].clamp_min(
                    speed_floor)
            else:
                with torch.no_grad():
                    speed_mid = target_network.predict_propagation_speed(
                        q_mid.reshape(n * k, self.dim))[:, 0].reshape(n, k)
                # Simpson integration of slowness along each candidate
                # segment.  This is the historical finite-speed formulation.
                stage_factor = step if global_cloud_rollout else step[:, None]
                speed_here_inv = speed_here.clamp_min(
                    speed_floor).reciprocal()
                # ``speed_here`` is stored as (n,1) for both endpoint and
                # independent-cloud pairs.  Keep that singleton axis: adding
                # another one here would broadcast the stage to (n,n,k),
                # making the Bellman target incompatible with tau(q,g).
                stage = (stage_factor / 6.0) * (
                    speed_here_inv
                    + 4.0 * speed_mid.clamp_min(speed_floor).reciprocal()
                    + speed_next.clamp_min(speed_floor).reciprocal())
            candidate_cost = stage + tau_next_matrix
            # Random global cloud transitions can land on a numerically
            # unresolved endpoint of the warm-start field.  Treat such a
            # transition as inadmissible, exactly like an out-of-domain
            # candidate, rather than allowing NaN/Inf to enter log-sum-exp.
            finite_candidate = torch.isfinite(candidate_cost)
            in_bounds = in_bounds & finite_candidate
            candidate_cost = torch.nan_to_num(
                candidate_cost, nan=1e6, posinf=1e6, neginf=0.0)
            candidate_cost = candidate_cost.masked_fill(~in_bounds, 1e6)
            has_upwind_candidate = in_bounds.any(dim=1)
            if self.viscosity_action_mode == 'stochastic_vv':
                # Feynman--Kac expectation replaces the discrete action min.
                # Direction selection is supplied by the vanishing-viscosity
                # limit, not by choosing among hand-written proposals.  A
                # state-constraint boundary rejects invalid proposals; they
                # must therefore be removed from the quadrature measure before
                # normalizing.  Averaging the 1e6 sentinel (the old behavior)
                # made one invalid direction dominate the entire residual and
                # was a numerical artifact, not a viscosity PDE.
                valid_count = in_bounds.sum(dim=1).clamp_min(1).to(
                    candidate_cost.dtype)
                valid_cost = candidate_cost.masked_fill(~in_bounds, 0.0)
                bellman_target = valid_cost.sum(dim=1) / valid_count
                bellman_target = torch.where(
                    has_upwind_candidate,
                    bellman_target,
                    torch.zeros_like(bellman_target))
            elif self.viscosity_action_mode == 'softmin':
                # Smooth monotone approximation to the control minimum.  The
                # log-mean-exp normalization makes the operator invariant to
                # the number of admissible directions (adding samples must
                # refine the same Hamiltonian, not lower travel time merely
                # because the action bank is larger).  As temperature tends
                # to zero this converges to the hard Bellman minimum.
                soft_temperature = max(
                    float(self.viscosity_softmin_temperature), 1e-6)
                valid_count = in_bounds.sum(dim=1).clamp_min(1).to(
                    candidate_cost.dtype)
                soft_cost = candidate_cost.masked_fill(
                    ~in_bounds, float('inf'))
                log_mean_exp = torch.logsumexp(
                    -soft_cost / soft_temperature, dim=1
                ) - torch.log(valid_count)
                bellman_target = -soft_temperature * log_mean_exp
                bellman_target = torch.where(
                    has_upwind_candidate,
                    bellman_target,
                    torch.zeros_like(bellman_target))
            else:
                bellman_target = candidate_cost.min(dim=1).values
            viscosity_upwind_rate = has_upwind_candidate.float().mean()
            if (self.viscosity_causal_target
                    or self.viscosity_action_mode == 'stochastic_vv'):
                # The accepted upwind state is a one-way bootstrap target, not
                # a second prediction to move in lockstep with this state.
                bellman_target = bellman_target.detach()

            if global_cloud_rollout:
                # Nonlocal transitions are not filtered by their sampled
                # jump length; only the source-cell neighborhood is exempt
                # from the Bellman residual because the point-source closure
                # already supplies that local PDE boundary condition.
                far = torch.linalg.vector_norm(
                    goal - q, dim=1) > self.viscosity_step
            else:
                far_limit = 4.0 * step if deterministic_rollout else 2.0 * step
                far = torch.linalg.vector_norm(goal - q, dim=1) > far_limit
            residual = tau_here - bellman_target
            # Rows with no admissible upwind action carry the 1e6 masking
            # sentinel in ``bellman_target`` and are multiplied out of the
            # training loss below.  Exclude those rows from diagnostics too;
            # otherwise a harmless one-percent invalid rate looks like a
            # catastrophic O(1e4) Bellman residual in the progress log.
            valid_residual = residual.detach()[has_upwind_candidate]
            if valid_residual.numel() > 0:
                viscosity_signed_residual = valid_residual.mean()
                viscosity_target_above_rate = (
                    valid_residual < 0).to(tau.dtype).mean()
            if self.viscosity_update_mode == 'raise_only':
                # Monotone fitted value iteration from a lower solution:
                # only a Bellman target above the current value may update it.
                # A spuriously small sampled minimum is ignored instead of
                # being propagated through later target synchronizations.
                residual = residual.clamp_max(0.0)
            abs_residual = residual.abs()
            loss_step = (step.mean(dim=1)
                         if global_cloud_rollout else step)
            local = torch.where(
                abs_residual < loss_step,
                0.5 * residual.square() / loss_step.clamp_min(1e-6),
                abs_residual - 0.5 * loss_step)
            weighted_local = (
                local * far.to(local.dtype)
                * has_upwind_candidate.to(local.dtype))
            if use_independent_pairs:
                viscosity_cloud_loss = self.viscosity_weight * (
                    weighted_local.mean())
            else:
                viscosity_loss[idx] = self.viscosity_weight * weighted_local

        # Fermat's variational principle: with the scalar speed field fixed,
        # the geodesic is the path that minimizes integral(ds / F).  For the
        # one-waypoint architecture tau is exactly that integral, so this term
        # gives its path decoder a direct, label-free learning signal.  In the
        # intended staged use --freeze_base_for_value_head keeps the already
        # learned scalar field fixed and trains only the waypoint network.
        fermat_path_loss = torch.zeros_like(diff)
        if (self.fermat_path_weight > 0
                and getattr(self.network, 'value_arch', None)
                in ('fermat_waypoint_symmetric',
                    'fermat_curve_symmetric')):
            # Fermat's problem is defined only for endpoints in the feasible
            # Eikonal domain.  Boundary/interior samples remain useful for
            # fitting F(q), but asking a finite curve to connect an invalid
            # endpoint teaches the path decoder the wrong problem.
            if self.eikonal_min_speed > 0:
                feasible_pair = (
                    (sq_Yobs0 >= self.eikonal_min_speed)
                    & (sq_Yobs1 >= self.eikonal_min_speed))
            else:
                feasible_pair = torch.ones_like(T, dtype=torch.bool)
            path_objective = T
            if self.fermat_loss_transform == 'log':
                # log1p is strictly increasing, so it has exactly the same
                # minimizing geodesic while conditioning the very large costs
                # produced by near-zero obstacle speed.
                path_objective = torch.log1p(path_objective)
            feasible_scale = (
                feasible_pair.numel()
                / feasible_pair.float().sum().clamp_min(1.0))
            fermat_path_loss = (
                self.fermat_path_weight * path_objective
                * feasible_pair.to(path_objective.dtype) * feasible_scale)

        # --- Normal loss ---
        # R58E: orientation-amplification. The normal loss is the only term that constrains
        # the DIRECTION of grad-tau (aligns it with -normal/speed). At the default 1e-3 it is
        # 10x weaker than the eikonal (1e-2), so the learned orientation gradient is ~3x
        # weaker than position (measured via P0b). Allow up-weighting via R58_NORMAL_WEIGHT to
        # bring ||grad-tau_orient|| toward parity with position. Default unchanged (1e-3).
        import os as _os
        normal_weight = float(_os.environ.get("R58_NORMAL_WEIGHT", "1e-3"))
        normal0 = normal_blended[:, :self.dim]
        normal1 = normal_blended[:, self.dim:]
        n_loss0 = (1.01 - Yobs[:, 0].unsqueeze(1)) * (Yobs[:, 0].unsqueeze(1) * DT0 + normal0) ** 2
        n_loss1 = (1.01 - Yobs[:, 1].unsqueeze(1)) * (Yobs[:, 1].unsqueeze(1) * DT1 + normal1) ** 2
        n_loss = normal_weight * (torch.sum(n_loss0, dim=1) + torch.sum(n_loss1, dim=1))

        # Loss ablation
        if not self.use_temporal_loss:
            temporal_loss = torch.zeros_like(temporal_loss)
        if not self.use_eikonal_loss:
            diff = torch.zeros_like(diff)
        if not self.use_normal_loss:
            n_loss = torch.zeros_like(n_loss)

        # --- Feature alignment loss for cross-zone pairs ---
        align_loss = torch.zeros_like(diff)
        if self.use_align_loss and pair_type is not None:
            cross_mask = (pair_type == 2)
            if cross_mask.any():
                # f_cup, f_pour are (2*B, h_size); use both start and goal features
                f_cup_start = f_cup[:B][cross_mask]
                f_cup_goal = f_cup[B:][cross_mask]
                f_pour_start = f_pour[:B][cross_mask]
                f_pour_goal = f_pour[B:][cross_mask]
                align_weight = 1e-3
                align_per_point = (torch.sum((f_cup_start - f_pour_start) ** 2, dim=1) +
                                   torch.sum((f_cup_goal - f_pour_goal) ** 2, dim=1))
                align_loss[cross_mask] = align_weight * align_per_point

        # --- Boundary feature alignment: align f_cup and f_pour for points near y_divide ---
        boundary_align_loss = torch.zeros_like(diff)
        if self.use_boundary_align:
            if (getattr(self.network, 'zone_mode', 'y') == 'corner_table'
                    and ee_xy is not None
                    and self.network.corner_zones is not None):
                # The G1 loss aligned heads near a one-dimensional Y split.
                # For the L-shaped scene, use the configured 2-D overlap box;
                # applying the legacy y_divide would align an unrelated strip.
                from models.metric_arm.zone_router import corner_masks_torch
                _, _, overlap_start = corner_masks_torch(
                    ee_xy[:, 0:2], self.network.corner_zones,
                    assign_outside=False)
                _, _, overlap_goal = corner_masks_torch(
                    ee_xy[:, 2:4], self.network.corner_zones,
                    assign_outside=False)
                near_boundary = overlap_start | overlap_goal
            else:
                y_div = self.network.y_divide
                margin = self.boundary_align_margin
                # Check both start and goal ee_y positions.
                ee_y_start = ee_y[:, 0]
                ee_y_goal = ee_y[:, 1]
                near_boundary = (
                    (torch.abs(ee_y_start - y_div) < margin)
                    | (torch.abs(ee_y_goal - y_div) < margin))
            if near_boundary.any():
                # Align features at boundary points
                f_cup_s = f_cup[:B][near_boundary]
                f_cup_g = f_cup[B:][near_boundary]
                f_pour_s = f_pour[:B][near_boundary]
                f_pour_g = f_pour[B:][near_boundary]
                ba_per_point = (torch.sum((f_cup_s - f_pour_s) ** 2, dim=1) +
                                torch.sum((f_cup_g - f_pour_g) ** 2, dim=1))
                boundary_align_loss[near_boundary] = self.boundary_align_weight * ba_per_point

        # --- R58F decoupled orientation head supervision ---
        # Phi_orient(config) regresses a target derived from the config's
        # zone-appropriate orientation distance (lower d_orient = more tilt =
        # higher cost). Because Phi reads only the orientation-bearing feature,
        # its gradient gives MPPI a dedicated orientation-restoring signal that
        # is not diluted by the position-reaching geometry of tau.
        orient_loss = torch.zeros_like(diff)
        if getattr(self.network, 'use_orient_head', False):
            import os as _os2
            orient_loss_weight = float(_os2.environ.get("R58_ORIENT_WEIGHT", "1e-2"))
            orient_margin = 0.05
            phi_start, phi_goal = self.network.orient_cost(
                points, w_cup_override=w_cup_2B, w_pour_override=w_pour_2B)
            # Zone-appropriate d_orient per endpoint: zone_id 0=cup, 1=pour.
            zone_start = zone_id[:, 0].long()
            zone_goal = zone_id[:, 1].long()
            d_orient_start = torch.where(
                zone_start == 0, d_orient_cup[:, 0], d_orient_pour[:, 0])
            d_orient_goal = torch.where(
                zone_goal == 0, d_orient_cup[:, 1], d_orient_pour[:, 1])
            target_start = torch.clamp(
                (orient_margin - d_orient_start) / orient_margin, min=0.0)
            target_goal = torch.clamp(
                (orient_margin - d_orient_goal) / orient_margin, min=0.0)
            orient_loss = orient_loss_weight * (
                (phi_start.squeeze(1) - target_start) ** 2
                + (phi_goal.squeeze(1) - target_goal) ** 2)

            # --- R58T: GRADIENT supervision of the orientation potential ---
            # Proven (R58R/S): supervising Phi's VALUE alone gives either an erratic
            # gradient (high-freq basis) or an inaccurate/inverted one (low-freq). Fix:
            # also supervise -grad_q Phi(start) to align with the orientation-descent
            # direction. At HIGH-tilt configs the blended distance min(d_phys,d_orient)
            # == d_orient, so the precomputed `normal` (its normalized C-space gradient,
            # pointing toward HIGHER d_orient == LOWER tilt) IS the orientation gradient.
            # We want Phi HIGH at high tilt, so -grad Phi should point toward lower tilt,
            # i.e. ALIGN with `normal`. FK-free at inference (uses only the learned Phi's
            # autograd gradient); FK entered only offline when `normal` was built.
            orient_grad_weight = float(_os2.environ.get("R58_ORIENT_GRAD_WEIGHT", "0.0"))
            if orient_grad_weight > 0.0:
                x_start = points[:, :self.network.dim]
                x_start = x_start.detach().requires_grad_(True)
                x_start_f = self.network.input_mapping(x_start, self.network.B_orient)
                phi_s = self.network.orient_head(x_start_f)  # (B,1)
                grad_phi = torch.autograd.grad(
                    phi_s.sum(), x_start, create_graph=True)[0]  # (B, dim)
                neg_grad = -grad_phi
                ng = neg_grad / neg_grad.norm(dim=1, keepdim=True).clamp(min=1e-8)
                # R58U: prefer the CLEAN FK tilt-descent target (tilt_grad). The stored
                # `normal` (grad of min(d_phys,d_orient)) aligns with true tilt-descent at
                # only cos~0.05, so it is a bad target; tilt_grad points reliably down-tilt.
                if tilt_grad is not None:
                    descent_dir = tilt_grad[:, :self.network.dim]
                    grad_mask = torch.norm(descent_dir, dim=1) > 1e-6
                    descent_dir = descent_dir / torch.norm(
                        descent_dir, dim=1, keepdim=True).clamp(min=1e-8)
                else:
                    descent_dir = torch.where(
                        (zone_start == 0).unsqueeze(1),
                        normal_cup[:, :self.network.dim], normal_pour[:, :self.network.dim])
                    grad_mask = torch.ones_like(target_start, dtype=torch.bool)
                    descent_dir = descent_dir / torch.norm(
                        descent_dir, dim=1, keepdim=True).clamp(min=1e-8)
                # supervise everywhere tilt_grad is defined; weight high-tilt more (where it
                # matters) but DO keep low-tilt configs so the basin is smooth and oriented.
                cos = (ng * descent_dir).sum(dim=1)
                orient_grad_loss = (
                    orient_grad_weight * grad_mask.float() * (1.0 - cos))
                orient_loss = orient_loss + orient_grad_loss

        safety_loss = torch.zeros_like(diff)
        if getattr(self.network, 'use_safety_head', False):
            import os as _os_safe
            safety_loss_weight = float(
                _os_safe.environ.get("R58_SAFETY_WEIGHT", "1e-2"))
            obs_margin = float(_os_safe.environ.get("R58_SAFETY_MARGIN", "0.03"))
            self_margin = float(_os_safe.environ.get("R58_SAFETY_SELF_MARGIN", "0.002"))
            pos_weight = float(_os_safe.environ.get("R58_SAFETY_POS_WEIGHT", "1.0"))
            if d_obstacle is not None and d_self is not None:
                phi_s, phi_g = self.network.safety_cost(points)
                obs_bad = torch.clamp(
                    (obs_margin - d_obstacle) / max(obs_margin, 1e-8), min=0.0)
                self_bad = torch.clamp(
                    (self_margin - d_self) / max(self_margin, 1e-8), min=0.0)
                target = torch.maximum(obs_bad, self_bad)
                point_weight = 1.0 + (pos_weight - 1.0) * (target > 0.0).float()
                safety_loss = safety_loss_weight * (
                    point_weight[:, 0] * (phi_s.squeeze(1) - target[:, 0]) ** 2
                    + point_weight[:, 1] * (phi_g.squeeze(1) - target[:, 1]) ** 2)

        para = getattr(self, 'travel_time_weight', 0.5)
        per_sample_loss = (
            diff + n_loss + temporal_loss + align_loss + boundary_align_loss
            + orient_loss + safety_loss
        ) * torch.exp(-para * T)
        per_sample_loss = (
            per_sample_loss + speed_supervision_loss
            + clearance_supervision_loss + hopf_cole_loss + viscosity_loss
            + fermat_path_loss)
        if sample_weight is not None:
            per_sample_loss = per_sample_loss * sample_weight.reshape(-1)

        # Pair-type loss modifiers. Defaults are 1.0, preserving legacy runs.
        if pair_type is not None:
            if self.cup_loss_weight != 1.0:
                per_sample_loss[pair_type == 0] *= self.cup_loss_weight
            if self.pour_loss_weight != 1.0:
                per_sample_loss[pair_type == 1] *= self.pour_loss_weight
            if self.cross_loss_weight != 1.0:
                per_sample_loss[pair_type == 2] *= self.cross_loss_weight

        if pair_type is not None and self.cross_mode == 'cross_downweight':
            is_cross = (pair_type == 2)
            per_sample_loss[is_cross] *= 0.1  # 10x downweight
        elif pair_type is not None and self.cross_mode == 'no_cross':
            is_cross = (pair_type == 2)
            per_sample_loss[is_cross] = 0.0  # zero out

        # Keep the independent speed-cloud term in the same scalar objective
        # as endpoint supervision so the existing beta normalization remains
        # comparable across runs.  It is a cloud mean, not duplicated as a
        # separate endpoint label.
        loss_n = (torch.sum(per_sample_loss) / Yobs.shape[0]
                  + collocation_speed_loss
                  + collocation_speed_component_loss
                  + collocation_speed_uncomposed_loss
                  + collocation_speed_consensus_identification_loss
                  + collocation_speed_repair_identification_loss
                  + coherent_eikonal_loss
                  + monotone_fd_loss
                  + trusted_anchor_loss
                  + viscosity_cloud_loss)

        # --- Optional path-level self-supervision ---
        # Xp is (B, 2*dim): first dim cols = q_start, next dim cols = q_goal
        # (confirmed by D = ||Xp[:, dim:] - Xp[:, :dim]|| above).
        path_eik_term = tau.new_zeros(())
        rank_term = tau.new_zeros(())
        path_handover_mono_term = tau.new_zeros(())
        segment_barrier_term = tau.new_zeros(())
        segment_invalid_rate = tau.new_zeros(())
        line_preserve_term = tau.new_zeros(())
        line_preserve_valid_rate = tau.new_zeros(())
        tilt_grad_tau_term = tau.new_zeros(())
        handover_grad_term = tau.new_zeros(())
        align_dir_term = tau.new_zeros(())

        def _tau_with_oracle_routing(points_in):
            """Tau for arbitrary normalized pairs with train-time FK routing."""
            if self._oracle is None:
                t, _, _, _, _ = self.network.out(
                    points_in, detach_input=False)
                return t.reshape(-1)

            q_s_eval = points_in[:, :self.dim].detach()
            q_g_eval = points_in[:, self.dim:].detach()
            info_s = self._oracle.query(q_s_eval)
            info_g = self._oracle.query(q_g_eval)
            ee_y_eval = torch.cat([info_s["ee_y"], info_g["ee_y"]], dim=0)
            ee_xy_eval = torch.cat([
                torch.stack([info_s["ee_x"], info_s["ee_y"]], dim=1),
                torch.stack([info_g["ee_x"], info_g["ee_y"]], dim=1),
            ], dim=0)
            d_cup_eval = torch.cat([
                info_s["d_orient_cup"], info_g["d_orient_cup"]], dim=0)
            d_pour_eval = torch.cat([
                info_s["d_orient_pour"], info_g["d_orient_pour"]], dim=0)
            t, _, _, _, _ = self.network.out(
                points_in, detach_input=False,
                ee_y=ee_y_eval,
                ee_xy=ee_xy_eval,
                d_orient_cup=d_cup_eval,
                d_orient_pour=d_pour_eval)
            return t.reshape(-1)

        if (path_loss_cfg is not None and self._oracle is not None
                and epoch >= path_loss_cfg["start_epoch"]):
            from .path_loss import rollout_and_path_loss
            pool_start = getattr(self, "_path_pair_pool_start", None)
            pool_goal = getattr(self, "_path_pair_pool_goal", None)
            pool_pair = getattr(self, "_path_pair_pool_pair_type", None)
            q_s = None
            q_g = None
            cap = int(path_loss_cfg["rollout_n"])
            if (pool_start is not None and pool_goal is not None
                    and pool_pair is not None and pool_start.numel() > 0):
                pair_mask = _pair_type_mask(
                    pool_pair, path_loss_cfg.get("pair_types", "cross"))
                if pair_mask.any():
                    idx = torch.nonzero(pair_mask, as_tuple=False).reshape(-1)
                    if idx.numel() > cap:
                        perm = torch.randperm(
                            idx.numel(), device=idx.device)[:cap]
                        idx = idx[perm]
                    q_s = pool_start[idx].detach()
                    q_g = pool_goal[idx].detach()
            elif pair_type is not None:
                pair_mask = _pair_type_mask(
                    pair_type, path_loss_cfg.get("pair_types", "cross"))
                if pair_mask.any():
                    idx = torch.nonzero(pair_mask, as_tuple=False).reshape(-1)
                    if idx.numel() > cap:
                        perm = torch.randperm(
                            idx.numel(), device=idx.device)[:cap]
                        idx = idx[perm]
                    q_s = Xp[idx, :self.dim].detach()
                    q_g = Xp[idx, self.dim:].detach()

            if q_s is not None and q_s.numel() > 0:

                def tau_fn(points_in):
                    # detach_input=False keeps tau differentiable w.r.t. points_in.
                    return _tau_with_oracle_routing(points_in)

                use_rank = epoch >= path_loss_cfg["rank_start_epoch"]
                out = rollout_and_path_loss(
                    tau_fn, q_s, q_g, self._oracle.query, self._safe_neighbor,
                    steps=path_loss_cfg["steps"], step_size=path_loss_cfg["step_size"],
                    dim=self.dim, margin=path_loss_cfg["margin"],
                    cup_tilt_max=path_loss_cfg["cup_tilt_max"],
                    pour_tilt_max=path_loss_cfg["pour_tilt_max"],
                    rank_margin=path_loss_cfg["rank_margin"],
                    path_eik_weight=path_loss_cfg["path_eik_weight"],
                    rank_weight=path_loss_cfg["rank_weight"],
                    use_rank=use_rank, use_path_eik=True,
                    q_min=self._q_min_norm, q_max=self._q_max_norm,
                    offset=path_loss_cfg.get("offset"),
                    handover_accept=path_loss_cfg.get("handover_accept", False),
                    corner_overlap_deg=path_loss_cfg.get("corner_overlap_deg", 45.0),
                    handover_mono_tol=path_loss_cfg.get("handover_mono_tol", 0.10),
                    handover_weight=path_loss_cfg.get("handover_weight", 0.0),
                    self_collision_margin=path_loss_cfg.get(
                        "self_collision_margin", 0.002),
                    collision_only=path_loss_cfg.get("collision_only", False),
                )
                path_eik_term = out["path_eik"]
                rank_term = out["rank"]
                path_handover_mono_term = out.get(
                    "handover_mono", path_handover_mono_term)

        if (segment_barrier_cfg is not None and self._oracle is not None
                and pair_type is not None
                and epoch >= segment_barrier_cfg["start_epoch"]):
            from .path_loss import local_segment_barrier_loss
            if (segment_barrier_cfg.get("safe_pool", False)
                    and self._segment_barrier_safe_q is not None
                    and self._segment_barrier_safe_q.numel() > 0):
                cap = int(segment_barrier_cfg["anchor_n"])
                safe_q = self._segment_barrier_safe_q
                take = min(cap, int(safe_q.shape[0]))
                pool_idx = torch.randint(
                    safe_q.shape[0], (take,), device=safe_q.device)
                q_anchor = safe_q[pool_idx].to(device=Xp.device, dtype=Xp.dtype)
                out = local_segment_barrier_loss(
                    _tau_with_oracle_routing,
                    q_anchor,
                    self._oracle.query,
                    dim=self.dim,
                    step_size=segment_barrier_cfg["step_size"],
                    n_dirs=segment_barrier_cfg["n_dirs"],
                    margin=segment_barrier_cfg["margin"],
                    cup_tilt_max=segment_barrier_cfg["cup_tilt_max"],
                    pour_tilt_max=segment_barrier_cfg["pour_tilt_max"],
                    rank_margin=segment_barrier_cfg["rank_margin"],
                    weight=segment_barrier_cfg["weight"],
                    self_collision_margin=segment_barrier_cfg.get(
                        "self_collision_margin", 0.002),
                    q_min=self._q_min_norm,
                    q_max=self._q_max_norm,
                    handover_accept=segment_barrier_cfg.get(
                        "handover_accept", False),
                    corner_overlap_deg=segment_barrier_cfg.get(
                        "corner_overlap_deg", 45.0),
                    check_midpoint=segment_barrier_cfg.get(
                        "check_midpoint", True),
                    collision_only=segment_barrier_cfg.get(
                        "collision_only", False),
                )
                segment_barrier_term = out["segment_barrier"]
                segment_invalid_rate = out.get(
                    "invalid_rate", segment_invalid_rate)
            else:
                pair_mask = _pair_type_mask(
                    pair_type, segment_barrier_cfg.get("pair_types", "cross"))
                if pair_mask.any():
                    idx = torch.nonzero(pair_mask, as_tuple=False).reshape(-1)
                    cap = segment_barrier_cfg["anchor_n"]
                    if idx.numel() > cap:
                        perm = torch.randperm(idx.numel(), device=idx.device)[:cap]
                        idx = idx[perm]
                    mode = segment_barrier_cfg.get("anchor_mode", "both")
                    anchors = []
                    if mode in ("start", "both", "all"):
                        anchors.append(Xp[idx, :self.dim].detach())
                    if mode in ("goal", "both", "all"):
                        anchors.append(Xp[idx, self.dim:].detach())
                    if mode in ("line", "all"):
                        q0 = Xp[idx, :self.dim].detach()
                        q1 = Xp[idx, self.dim:].detach()
                        u = torch.rand(
                            q0.shape[0], 1, device=q0.device, dtype=q0.dtype)
                        anchors.append(q0 + u * (q1 - q0))
                    if not anchors:
                        raise ValueError(
                            f"unknown segment_barrier anchor_mode={mode!r}")
                    q_anchor = torch.cat(anchors, dim=0)
                    out = local_segment_barrier_loss(
                        _tau_with_oracle_routing,
                        q_anchor,
                        self._oracle.query,
                        dim=self.dim,
                        step_size=segment_barrier_cfg["step_size"],
                        n_dirs=segment_barrier_cfg["n_dirs"],
                        margin=segment_barrier_cfg["margin"],
                        cup_tilt_max=segment_barrier_cfg["cup_tilt_max"],
                        pour_tilt_max=segment_barrier_cfg["pour_tilt_max"],
                        rank_margin=segment_barrier_cfg["rank_margin"],
                        weight=segment_barrier_cfg["weight"],
                        self_collision_margin=segment_barrier_cfg.get(
                            "self_collision_margin", 0.002),
                        q_min=self._q_min_norm,
                        q_max=self._q_max_norm,
                        handover_accept=segment_barrier_cfg.get(
                            "handover_accept", False),
                        corner_overlap_deg=segment_barrier_cfg.get(
                            "corner_overlap_deg", 45.0),
                        check_midpoint=segment_barrier_cfg.get(
                            "check_midpoint", True),
                        collision_only=segment_barrier_cfg.get(
                            "collision_only", False),
                    )
                    segment_barrier_term = out["segment_barrier"]
                    segment_invalid_rate = out.get(
                        "invalid_rate", segment_invalid_rate)

        import os as _os3

        def _pair_filter_mask(env_name, default_spec=""):
            """Return bool mask for pair_type based on a comma/space env spec."""
            if pair_type is None:
                return None
            spec = _os3.environ.get(env_name, default_spec)
            spec = (spec or "").strip().lower()
            if not spec or spec in ("all", "*"):
                return None
            aliases = {
                "cup": 0, "cupping": 0, "0": 0,
                "pour": 1, "pouring": 1, "1": 1,
                "cross": 2, "handover": 2, "2": 2,
            }
            ids = []
            for raw in spec.replace(",", " ").split():
                if raw not in aliases:
                    raise ValueError(
                        f"unknown pair type '{raw}' in {env_name}={spec!r}")
                ids.append(aliases[raw])
            keep = torch.zeros_like(pair_type, dtype=torch.bool)
            for pid in ids:
                keep |= (pair_type == pid)
            return keep

        def _apply_pair_filter(mask, env_name, default_spec=""):
            keep = _pair_filter_mask(env_name, default_spec)
            if keep is not None:
                mask = mask & keep
            return mask

        def _weighted_mean(vec, idx, weight):
            if sample_weight is not None and idx is not None:
                sw = sample_weight[idx].detach().reshape(-1)
                return weight * (vec * sw).sum() / sw.sum().clamp(min=1e-8)
            return weight * vec.mean()

        def _line_preserve_from_pairs(q0, q1, batch_idx=None):
            n = q0.shape[0]
            if n == 0:
                z = tau.new_zeros(())
                return z, z
            n_checks = max(2, int(line_preserve_cfg["checks"]))
            ts = torch.linspace(
                0.0, 1.0, n_checks, device=Xp.device,
                dtype=Xp.dtype).view(1, n_checks, 1)
            q_line = q0[:, None, :] + ts * (q1 - q0)[:, None, :]
            flat_line = q_line.reshape(-1, self.dim)
            info = self._oracle.query(flat_line)
            d_obs = info["d_obs"].reshape(n, n_checks)
            d_self_raw = info.get("d_self")
            if d_self_raw is not None:
                d_self_line = d_self_raw.reshape(n, n_checks)
                collision_bad = (
                    (d_obs < line_preserve_cfg["margin"])
                    | (d_self_line < line_preserve_cfg["self_collision_margin"]))
            else:
                collision_bad = d_obs < line_preserve_cfg["margin"]

            from .path_loss import _corner_tilt_violate
            tilt_bad = _corner_tilt_violate(
                info,
                line_preserve_cfg["cup_tilt_max"],
                line_preserve_cfg["pour_tilt_max"],
                handover_accept=line_preserve_cfg.get(
                    "handover_accept", False),
                corner_overlap_deg=line_preserve_cfg.get(
                    "corner_overlap_deg", 45.0),
            ).reshape(n, n_checks)
            valid_line = ~(collision_bad | tilt_bad).any(dim=1)
            valid_rate = valid_line.float().mean().detach()

            if not valid_line.any():
                return tau.new_zeros(()), valid_rate

            q0v = q0[valid_line].detach()
            q1v = q1[valid_line].detach()
            valid_idx = (
                batch_idx[valid_line] if batch_idx is not None else None)
            term = tau.new_zeros(())

            align_weight = line_preserve_cfg.get("align_weight", 0.0)
            if align_weight > 0.0:
                coords_line = torch.cat([q0v, q1v], dim=1).detach()
                coords_line = coords_line.requires_grad_(True)
                tau_line = _tau_with_oracle_routing(coords_line)
                g = torch.autograd.grad(
                    tau_line.sum(), coords_line,
                    create_graph=True, retain_graph=True)[0]
                u = q1v - q0v
                u = u / u.norm(dim=1, keepdim=True).clamp(min=1e-8)
                desc = -g[:, :self.dim]
                desc = desc / desc.norm(dim=1, keepdim=True).clamp(min=1e-8)
                align_vec = 1.0 - torch.sum(desc * u, dim=1)
                term = term + _weighted_mean(align_vec, valid_idx, align_weight)

            mono_weight = line_preserve_cfg.get("mono_weight", 0.0)
            if mono_weight > 0.0:
                step_frac = float(line_preserve_cfg.get("step_frac", 0.20))
                step_frac = min(max(step_frac, 1e-3), 1.0)
                t0 = torch.rand(
                    q0v.shape[0], 1, device=Xp.device,
                    dtype=Xp.dtype) * (1.0 - step_frac)
                t1 = t0 + step_frac
                q_curr = q0v + t0 * (q1v - q0v)
                q_next = q0v + t1 * (q1v - q0v)
                tau_curr = _tau_with_oracle_routing(
                    torch.cat([q_curr, q1v], dim=1))
                tau_next = _tau_with_oracle_routing(
                    torch.cat([q_next, q1v], dim=1))
                progress = (q_next - q_curr).norm(dim=1)
                margin = float(line_preserve_cfg.get("mono_margin", 0.0))
                mono_vec = torch.relu(
                    tau_next - tau_curr + margin * progress)
                term = term + _weighted_mean(mono_vec, valid_idx, mono_weight)

            return term, valid_rate

        if (line_preserve_cfg is not None and self._oracle is not None
                and epoch >= line_preserve_cfg["start_epoch"]):
            line_pool_start = getattr(self, "_line_pair_pool_start", None)
            line_pool_goal = getattr(self, "_line_pair_pool_goal", None)
            line_pool_pair = getattr(self, "_line_pair_pool_pair_type", None)
            if (line_pool_start is not None and line_pool_goal is not None
                    and line_pool_pair is not None
                    and line_pool_start.numel() > 0):
                pair_mask = _pair_type_mask(
                    line_pool_pair, line_preserve_cfg.get("pair_types", "all"))
                if pair_mask.any():
                    idx = torch.nonzero(pair_mask, as_tuple=False).reshape(-1)
                    cap = int(line_preserve_cfg["sample_n"])
                    if idx.numel() > cap:
                        perm = torch.randperm(
                            idx.numel(), device=idx.device)[:cap]
                        idx = idx[perm]
                    q0 = line_pool_start[idx].detach()
                    q1 = line_pool_goal[idx].detach()
                    line_preserve_term, line_preserve_valid_rate = (
                        _line_preserve_from_pairs(q0, q1))
            elif (line_preserve_cfg.get("safe_pool", False)
                    and self._line_preserve_safe_q is not None
                    and self._line_preserve_safe_q.numel() > 0):
                cap = line_preserve_cfg["sample_n"]
                safe_q = self._line_preserve_safe_q
                take = min(int(cap), int(safe_q.shape[0]))
                pool_idx = torch.randint(
                    safe_q.shape[0], (take,), device=safe_q.device)
                q0 = safe_q[pool_idx].to(device=Xp.device, dtype=Xp.dtype)
                dirs = torch.randn(
                    take, self.dim, device=Xp.device, dtype=Xp.dtype)
                dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp(min=1e-8)
                step_min = float(line_preserve_cfg.get(
                    "local_step_min", line_preserve_cfg.get("local_step", 0.03)))
                step_max = float(line_preserve_cfg.get(
                    "local_step_max", step_min))
                if step_max < step_min:
                    step_max = step_min
                if step_max > step_min:
                    step = step_min + (step_max - step_min) * torch.rand(
                        take, 1, device=Xp.device, dtype=Xp.dtype)
                else:
                    step = torch.full(
                        (take, 1), step_min, device=Xp.device, dtype=Xp.dtype)
                q1 = q0 + step * dirs
                if self._q_min_norm is not None:
                    q1 = q1.clamp(
                        min=self._q_min_norm.to(dtype=Xp.dtype),
                        max=self._q_max_norm.to(dtype=Xp.dtype))
                line_preserve_term, line_preserve_valid_rate = (
                    _line_preserve_from_pairs(q0, q1))
            elif pair_type is not None:
                pair_mask = _pair_type_mask(
                    pair_type, line_preserve_cfg.get("pair_types", "all"))
                if pair_mask.any():
                    idx = torch.nonzero(pair_mask, as_tuple=False).reshape(-1)
                    cap = line_preserve_cfg["sample_n"]
                    if idx.numel() > cap:
                        perm = torch.randperm(
                            idx.numel(), device=idx.device)[:cap]
                        idx = idx[perm]

                    q0 = Xp[idx, :self.dim].detach()
                    q1 = Xp[idx, self.dim:].detach()
                    line_preserve_term, line_preserve_valid_rate = (
                        _line_preserve_from_pairs(q0, q1, idx))

        # --- FK tilt-gradient descent alignment for tau ---
        # tilt_grad.npy is an offline finite-difference target: a unit C-space
        # direction that lowers the zone-appropriate wrist tilt. Aligning tau's
        # descent direction to it directly addresses wing tilt failures without
        # adding FK to inference.
        tilt_grad_tau_weight = float(_os3.environ.get(
            "R58_TILT_GRAD_TAU_WEIGHT", "0.0"))
        if tilt_grad_tau_weight > 0.0 and tilt_grad is not None:
            target = tilt_grad[:, :self.dim]
            grad_mask = torch.norm(target, dim=1) > 1e-6
            grad_mask = _apply_pair_filter(
                grad_mask,
                "R58_TILT_GRAD_PAIR_TYPES",
                _os3.environ.get("R58_TILT_GRAD_DEFAULT_PAIR_TYPES", ""))
            if grad_mask.any():
                idx = torch.nonzero(grad_mask, as_tuple=False).reshape(-1)
                cap = int(_os3.environ.get("R58_TILT_GRAD_TAU_N", "0"))
                if cap > 0 and idx.numel() > cap:
                    perm = torch.randperm(idx.numel(), device=idx.device)[:cap]
                    idx = idx[perm]
                coords_tilt = points[idx].clone().detach().requires_grad_(True)
                wc = torch.cat([w_cup_2B[:B][idx], w_cup_2B[B:][idx]], dim=0)
                wp = torch.cat([w_pour_2B[:B][idx], w_pour_2B[B:][idx]], dim=0)
                fk_tilt = (
                    fk_features[idx].detach() if fk_features is not None else None)
                tau_tilt, _, _, _, _ = self.network.out(
                    coords_tilt, detach_input=False,
                    w_cup_override=wc.detach(), w_pour_override=wp.detach(),
                    fk_features=fk_tilt)
                g = torch.autograd.grad(
                    tau_tilt.sum(), coords_tilt,
                    create_graph=True, retain_graph=True)[0]
                desc = -g[:, :self.dim]
                desc = desc / torch.norm(
                    desc, dim=1, keepdim=True).clamp(min=1e-8)
                target = target[idx]
                target = target / torch.norm(
                    target, dim=1, keepdim=True).clamp(min=1e-8)
                align_vec = 1.0 - torch.sum(desc * target, dim=1)
                tilt_grad_tau_term = _weighted_mean(
                    align_vec, idx, tilt_grad_tau_weight)

        # --- Corner handover-ratio monotonicity alignment for tau ---
        # handover_grad.npy stores +grad_q h for h=cup_tilt/(cup_tilt+pour_tilt)
        # only at FK-corner configs. If tau descent has positive dot with
        # this vector, MPPI's local steps are biased toward monotonically increasing
        # handover progress instead of snapping the wrist backward.
        handover_grad_weight = float(_os3.environ.get(
            "R58_HANDOVER_GRAD_WEIGHT", "0.0"))
        if handover_grad_weight > 0.0 and handover_grad is not None:
            target = handover_grad[:, :self.dim]
            grad_mask = torch.norm(target, dim=1) > 1e-6
            grad_mask = _apply_pair_filter(
                grad_mask,
                "R58_HANDOVER_GRAD_PAIR_TYPES",
                _os3.environ.get("R58_HANDOVER_GRAD_DEFAULT_PAIR_TYPES", "cross"))
            if grad_mask.any():
                idx = torch.nonzero(grad_mask, as_tuple=False).reshape(-1)
                cap = int(_os3.environ.get("R58_HANDOVER_GRAD_N", "0"))
                if cap > 0 and idx.numel() > cap:
                    perm = torch.randperm(idx.numel(), device=idx.device)[:cap]
                    idx = idx[perm]
                coords_h = points[idx].clone().detach().requires_grad_(True)
                wc = torch.cat([w_cup_2B[:B][idx], w_cup_2B[B:][idx]], dim=0)
                wp = torch.cat([w_pour_2B[:B][idx], w_pour_2B[B:][idx]], dim=0)
                fk_h = (
                    fk_features[idx].detach() if fk_features is not None else None)
                tau_h, _, _, _, _ = self.network.out(
                    coords_h, detach_input=False,
                    w_cup_override=wc.detach(), w_pour_override=wp.detach(),
                    fk_features=fk_h)
                g = torch.autograd.grad(
                    tau_h.sum(), coords_h,
                    create_graph=True, retain_graph=True)[0]
                desc = -g[:, :self.dim]
                desc = desc / torch.norm(
                    desc, dim=1, keepdim=True).clamp(min=1e-8)
                target = target[idx]
                target = target / torch.norm(
                    target, dim=1, keepdim=True).clamp(min=1e-8)
                dot = torch.sum(desc * target, dim=1)
                mode = _os3.environ.get(
                    "R58_HANDOVER_GRAD_MODE", "hinge").strip().lower()
                if mode == "align":
                    h_vec = 1.0 - dot
                else:
                    margin = float(_os3.environ.get(
                        "R58_HANDOVER_GRAD_MARGIN", "0.1"))
                    h_vec = torch.relu(margin - dot)
                handover_grad_term = _weighted_mean(
                    h_vec, idx, handover_grad_weight)

        # --- R58I cross gradient-direction alignment loss ---
        # Diagnosis: for cross pairs, tau VALUES are correct (monotone along the
        # straight start->goal line) but its GRADIENT DIRECTION deflects the
        # planner off the (mostly tilt-valid) straight path. Fix the FIELD so
        # -grad(tau) at the START config points along u = (q_goal - q_start).
        # Active ONLY for cross pairs (pair_type==2) and ONLY when the env var
        # R58_ALIGN_WEIGHT is set > 0. Default 0.0 => byte-identical to before.
        align_dir_weight = float(_os3.environ.get("R58_ALIGN_WEIGHT", "0.0"))
        if (align_dir_weight > 0.0 and pair_type is not None):
            is_cross = (pair_type == 2)
            if is_cross.any():
                idx = torch.nonzero(is_cross, as_tuple=False).reshape(-1)
                # Build a fresh leaf for the cross subset's full [start, goal]
                # coords so autograd.grad can take d_tau/d_coords w.r.t. it,
                # with create_graph=True so the alignment loss is trainable.
                coords_cross = points[idx].clone().detach().requires_grad_(True)
                # Per-point (2*B_cross) routing weights for the cross subset:
                # take the start/goal weights already computed for these rows.
                wc = torch.cat([w_cup_2B[:B][idx], w_cup_2B[B:][idx]], dim=0)
                wp = torch.cat([w_pour_2B[:B][idx], w_pour_2B[B:][idx]], dim=0)
                fk_cross = fk_features[idx].detach() if fk_features is not None else None
                tau_cross, _, _, _, _ = self.network.out(
                    coords_cross, detach_input=False,
                    w_cup_override=wc.detach(), w_pour_override=wp.detach(),
                    fk_features=fk_cross)
                g = torch.autograd.grad(
                    tau_cross.sum(), coords_cross,
                    create_graph=True, retain_graph=True)[0]
                g_start = g[:, :self.dim]                       # d_tau/d_q_start
                q_s = coords_cross[:, :self.dim]
                q_g = coords_cross[:, self.dim:]
                u = q_g - q_s                                   # desired direction
                u = u / (torch.norm(u, dim=1, keepdim=True) + 1e-8)
                desc = -g_start
                desc = desc / (torch.norm(desc, dim=1, keepdim=True) + 1e-8)
                cos = torch.sum(desc * u, dim=1)                # cosine similarity
                align_vec = 1.0 - cos
                if sample_weight is not None:
                    sw = sample_weight[idx].detach().reshape(-1)
                    align_dir_term = align_dir_weight * (
                        align_vec * sw).sum() / sw.sum().clamp(min=1e-8)
                else:
                    align_dir_term = align_dir_weight * torch.mean(align_vec)

        # ---- R62: DIRECT slowness supervision -------------------------------
        # The local_finsler slowness head has no direct target: its only signal is
        # through d(tau)/dq in the Eikonal residual.  Measured 2026-08-31: at init
        # l = speed*||grad tau|| < 1, so that residual pushes slowness UP, the head
        # hits its ceiling within ONE epoch, and the gradient dies there -- every
        # trained checkpoint ended with a constant metric (straight-line geodesics).
        # tau over a short chord is ~ eps * slowness, so regressing tau(q,q+eps*d)/eps
        # onto the label 1/speed(q) anchors the head to data it already has, in value
        # form (no double-backward, far better conditioned than the gradient form).
        slowness_term = tau.new_zeros(())
        _sw = float(os.environ.get('R62_SLOWNESS_WEIGHT', '0.0'))
        _eps_slow = _ep_slowness
        if _sw > 0.0 and _eps_slow is not None:
            # Direct supervision of the local_finsler slowness head at the chord
            # endpoints, where the labels live.  The head is otherwise trained ONLY
            # through d(tau)/dq in the Eikonal residual; measured 2026-08-31 that
            # drives it to its bound within one epoch, after which the map's
            # derivative is ~1e-4 and no loss can recover it -- every checkpoint
            # ended emitting a constant metric (straight-line geodesics).
            _s_start, _s_goal = _eps_slow
            _t_start = (1.0 / Yobs[:, 0].detach().clamp(min=1e-3))
            _t_goal = (1.0 / Yobs[:, 1].detach().clamp(min=1e-3))
            slowness_term = _sw * 0.5 * (
                (_s_start - _t_start).square().mean()
                + (_s_goal - _t_goal).square().mean())
            with torch.no_grad():
                _p = torch.cat([_s_start, _s_goal]).float()
                _t = torch.cat([_t_start, _t_goal]).float()
                _pc = _p - _p.mean(); _tc = _t - _t.mean()
                _den = (_pc.norm() * _tc.norm()).clamp(min=1e-8)
                _yo = torch.cat([Yobs[:, 0], Yobs[:, 1]]).float()
                self._slowness_diag = (
                    float(_p.mean()), float(_p.std()),
                    float(_t.mean()), float(_t.std()),
                    float((_pc * _tc).sum() / _den),
                    float(_yo.min()), float(_yo.max()))

        # ---- R65: interior slowness supervision -------------------------------
        # The endpoint anchor above pins slowness only where training pairs START and
        # END.  Our cross failures happen at MID-PATH waypoints, and the label field
        # there is ambiguous (measured 2026-09-08 at the 101 violating cup-zone
        # waypoints: 44.5% of the 32 nearest endpoints are slow, 37.4% fast), so a
        # smooth fit over endpoint supervision alone averages the penalty down:
        #     true label 26.03  ->  neighbour mean 13.86  ->  model 4.66
        # This term pins the interior points to their own true labels.  It does NOT
        # change the quadrature INPUTS -- that was R63, which cost 38-46 points.
        interior_sup_term = tau.new_zeros(())
        _isw = float(os.environ.get('R65_INTERIOR_SUP_WEIGHT', '0.0'))
        if _isw > 0.0 and getattr(self.network, 'interior_supervision', False):
            _i_pred = _in_slowness
            _i_q = _in_q
            if _i_pred is not None and _i_q is not None:
                # ONE-SIDED, and that is not a softening -- it is what makes this
                # exactly correct.  The pipeline's speed is
                #     clip(min(d_orient, d_obstacle, 0.5*d_self), offset, margin)/margin
                # (verified 2026-09-08: reproduces speed_cupping.npy to 0 error, 100%).
                # _interior_target_slowness knows only d_orient, so it OVERSTATES the
                # true speed wherever obstacle or self-collision binds -- 4.0% of
                # points.  But since min() can only shrink d, orientation-only slowness
                # is a provable LOWER BOUND on the true slowness.  Penalising the model
                # only when it is FASTER than that bound can therefore never ask for a
                # wrong value, while still hitting our actual failure, which is the
                # field predicting 4.66 where the truth is 26.03.
                _i_tgt = self.network._interior_target_slowness(_i_q).detach()
                interior_sup_term = _isw * torch.relu(
                    _i_tgt - _i_pred).square().mean()
                with torch.no_grad():
                    _ip = _i_pred.float(); _it = _i_tgt.float()
                    _ipc = _ip - _ip.mean(); _itc = _it - _it.mean()
                    _iden = (_ipc.norm() * _itc.norm()).clamp(min=1e-8)
                    self._interior_sup_diag = (
                        float(_ip.mean()), float(_ip.std()),
                        float(_it.mean()), float(_it.std()),
                        float((_ipc * _itc).sum() / _iden))

        loss = (
            beta * loss_n + slowness_term + interior_sup_term + path_eik_term + rank_term
            + path_handover_mono_term
            + segment_barrier_term
            + line_preserve_term
            + tilt_grad_tau_term
            + handover_grad_term
            + align_dir_term
        )

        self.last_aux_terms = {
            "path_eik": float(path_eik_term.detach()),
            "path_rank": float(rank_term.detach()),
            "path_handover_mono": float(path_handover_mono_term.detach()),
            "segment_barrier": float(segment_barrier_term.detach()),
            "segment_invalid_rate": float(segment_invalid_rate.detach()),
            "line_preserve": float(line_preserve_term.detach()),
            "line_valid_rate": float(line_preserve_valid_rate.detach()),
            "slowness_sup": float(slowness_term.detach()),
            "interior_sup": float(interior_sup_term.detach()),
            "slowness_pred_mean": getattr(self, "_slowness_diag", (0,)*7)[0],
            "slowness_pred_std": getattr(self, "_slowness_diag", (0,)*7)[1],
            "slowness_tgt_mean": getattr(self, "_slowness_diag", (0,)*7)[2],
            "slowness_corr": getattr(self, "_slowness_diag", (0,)*7)[4],
            "slowness_yobs_min": getattr(self, "_slowness_diag", (0,)*7)[5] if len(getattr(self, "_slowness_diag", ())) > 5 else 0.0,
            "slowness_yobs_max": getattr(self, "_slowness_diag", (0,)*7)[6] if len(getattr(self, "_slowness_diag", ())) > 6 else 0.0,
            "safety_head": float(safety_loss.detach().mean()),
            "speed_supervision": float(
                speed_supervision_loss.detach().mean()),
            "collocation_speed_supervision": float(
                collocation_speed_loss.detach()),
            "collocation_speed_component_supervision": float(
                collocation_speed_component_loss.detach()),
            "collocation_speed_component_mae": float(
                collocation_speed_component_mae.detach()),
            "collocation_speed_component_corr_mean": float(
                collocation_speed_component_corr_mean.detach()),
            "collocation_speed_component_corr_min": float(
                collocation_speed_component_corr_min.detach()),
            "collocation_speed_uncomposed_supervision": float(
                collocation_speed_uncomposed_loss.detach()),
            "collocation_speed_uncomposed_mae": float(
                collocation_speed_uncomposed_mae.detach()),
            "collocation_speed_uncomposed_corr": float(
                collocation_speed_uncomposed_corr.detach()),
            "collocation_speed_consensus_identification": float(
                collocation_speed_consensus_identification_loss.detach()),
            "collocation_speed_consensus_alpha_mae": float(
                collocation_speed_consensus_alpha_mae.detach()),
            "collocation_speed_consensus_alpha_target_mean": float(
                collocation_speed_consensus_alpha_target_mean.detach()),
            "collocation_speed_consensus_alpha_prediction_mean": float(
                collocation_speed_consensus_alpha_prediction_mean.detach()),
            "collocation_speed_consensus_identified_fraction": float(
                collocation_speed_consensus_identified_fraction.detach()),
            "collocation_speed_consensus_release_bce": float(
                collocation_speed_consensus_release_bce.detach()),
            "collocation_speed_consensus_magnitude_mse": float(
                collocation_speed_consensus_magnitude_mse.detach()),
            "collocation_speed_consensus_negative_magnitude_mse": float(
                collocation_speed_consensus_negative_magnitude_mse.detach()),
            "collocation_speed_consensus_specificity_bce": float(
                collocation_speed_consensus_specificity_bce.detach()),
            "collocation_speed_consensus_release_target_mean": float(
                collocation_speed_consensus_release_target_mean.detach()),
            "collocation_speed_consensus_release_prediction_mean": float(
                collocation_speed_consensus_release_prediction_mean.detach()),
            "collocation_speed_consensus_magnitude_target_mean": float(
                collocation_speed_consensus_magnitude_target_mean.detach()),
            "collocation_speed_consensus_magnitude_prediction_mean": float(
                collocation_speed_consensus_magnitude_prediction_mean.detach()),
            "collocation_speed_consensus_negative_fraction": float(
                collocation_speed_consensus_negative_fraction.detach()),
            "collocation_speed_consensus_negative_magnitude_prediction_mean": float(
                collocation_speed_consensus_negative_magnitude_prediction_mean.detach()),
            "collocation_speed_consensus_effective_release_mse": float(
                collocation_speed_consensus_effective_release_mse.detach()),
            "collocation_speed_consensus_effective_release_target_mean": float(
                collocation_speed_consensus_effective_release_target_mean.detach()),
            "collocation_speed_consensus_effective_release_prediction_mean": float(
                collocation_speed_consensus_effective_release_prediction_mean.detach()),
            "collocation_speed_consensus_negative_effective_release_mse": float(
                collocation_speed_consensus_negative_effective_release_mse.detach()),
            "collocation_speed_consensus_negative_effective_release_prediction_mean": float(
                collocation_speed_consensus_negative_effective_release_prediction_mean.detach()),
            "collocation_speed_consensus_sampling_support_pool": float(
                0 if self._consensus_support_index_pool is None
                else self._consensus_support_index_pool.numel()),
            "collocation_speed_consensus_sampling_danger_pool": float(
                0 if self._consensus_danger_index_pool is None
                else self._consensus_danger_index_pool.numel()),
            "collocation_speed_consensus_negative_weight": float(
                self.collocation_speed_consensus_negative_weight),
            "collocation_speed_consensus_head_component_weight": float(
                self.collocation_speed_consensus_head_component_weight),
            "collocation_speed_consensus_specificity_positive_mean": float(
                collocation_speed_consensus_specificity_positive_mean.detach()),
            "collocation_speed_consensus_specificity_negative_mean": float(
                collocation_speed_consensus_specificity_negative_mean.detach()),
            "collocation_speed_repair_identification": float(
                collocation_speed_repair_identification_loss.detach()),
            "collocation_speed_repair_rho_mae": float(
                collocation_speed_repair_rho_mae.detach()),
            "collocation_speed_repair_release_bce": float(
                collocation_speed_repair_release_bce.detach()),
            "collocation_speed_repair_magnitude_mse": float(
                collocation_speed_repair_magnitude_mse.detach()),
            "collocation_speed_repair_support_target_mean": float(
                collocation_speed_repair_support_target_mean.detach()),
            "collocation_speed_repair_release_prediction_mean": float(
                collocation_speed_repair_release_prediction_mean.detach()),
            "collocation_speed_repair_magnitude_target_mean": float(
                collocation_speed_repair_magnitude_target_mean.detach()),
            "collocation_speed_repair_magnitude_prediction_mean": float(
                collocation_speed_repair_magnitude_prediction_mean.detach()),
            "collocation_speed_repair_coordination_beta_mean": float(
                collocation_speed_repair_coordination_beta_mean.detach()),
            "collocation_speed_repair_coordination_zero_fraction": float(
                collocation_speed_repair_coordination_zero_fraction.detach()),
            "coherent_eikonal": float(coherent_eikonal_loss.detach()),
            "coherent_eikonal_residual": float(
                coherent_eikonal_residual.detach()),
            "coherent_first_order_residual": float(
                coherent_first_order_residual.detach()),
            "coherent_viscosity_laplacian_rms": float(
                coherent_viscosity_laplacian_rms.detach()),
            "coherent_subsolution_violation": float(
                coherent_subsolution_violation.detach()),
            "coherent_subsolution_value": float(
                coherent_subsolution_value.detach()),
            "coherent_discount_value": float(
                coherent_discount_value.detach()),
            "coherent_goal_count": float(coherent_goal_count.detach()),
            "monotone_fd": float(monotone_fd_loss.detach()),
            "monotone_fd_residual": float(monotone_fd_residual.detach()),
            "monotone_fd_valid_rate": float(
                monotone_fd_valid_rate.detach()),
            "monotone_fd_norm_mean": float(
                monotone_fd_norm_mean.detach()),
            "trusted_anchor_maximal": float(trusted_anchor_loss.detach()),
            "trusted_anchor_hard_count": float(
                trusted_anchor_stats["hard_count"]),
            "trusted_anchor_direct_hard_count": float(
                trusted_anchor_stats["direct_hard_count"]),
            "trusted_anchor_route_pair_count": float(
                trusted_anchor_stats["route_pair_count"]),
            "trusted_anchor_mean": float(
                trusted_anchor_stats["trusted_anchor_mean"]),
            "trusted_anchor_ratio": float(
                trusted_anchor_stats["ratio_mean"]),
            "trusted_anchor_violation_fraction": float(
                trusted_anchor_stats["violation_fraction"]),
            "trusted_anchor_violation_p95": float(
                trusted_anchor_stats["violation_p95"]),
            "collocation_speed_mae": float(
                collocation_speed_mae.detach()),
            "collocation_speed_rmse": float(
                collocation_speed_rmse.detach()),
            "collocation_speed_corr": float(
                collocation_speed_corr.detach()),
            "collocation_speed_slow_recall": float(
                collocation_speed_slow_recall.detach()),
            "clearance_supervision": float(
                clearance_supervision_loss.detach().mean()),
            "hopf_cole": float(hopf_cole_loss.detach().mean()),
            "hopf_boundary_fraction": float(
                hopf_boundary_fraction.detach()),
            "hopf_characteristic_count": float(
                hopf_characteristic_count.detach()),
            "hopf_state_violation_rate": float(
                hopf_state_violation_rate.detach()),
            "derivative_clip": float(derivative_clip),
            "viscosity_bellman": float(
                viscosity_loss.detach().mean()
                + viscosity_cloud_loss.detach()),
            "viscosity_domain_valid_rate": float(
                viscosity_domain_valid_rate.detach()),
            "viscosity_upwind_rate": float(
                viscosity_upwind_rate.detach()),
            "viscosity_signed_residual": float(
                viscosity_signed_residual.detach()),
            "viscosity_target_above_rate": float(
                viscosity_target_above_rate.detach()),
            "viscosity_local_boundary_rate": float(
                viscosity_local_boundary_rate.detach()),
            "fermat_path": float(fermat_path_loss.detach().mean()),
            "tilt_grad_tau": float(tilt_grad_tau_term.detach()),
            "handover_grad": float(handover_grad_term.detach()),
            "cross_align": float(align_dir_term.detach()),
        }

        return loss, loss_n, diff, path_eik_term.detach(), rank_term.detach()

    def StartSpeed(self, Xp):
        batch_size = 5000
        Ypred_list = []
        for i in range(math.ceil(Xp.shape[0] / batch_size)):
            Xp_tmp = Xp[i * batch_size:min((i + 1) * batch_size, Xp.shape[0]), :]
            tau, w, Xp_tmp, _, _ = self.network.out(Xp_tmp)
            dtau = self.gradient(tau, Xp_tmp)

            D = Xp_tmp[:, self.dim:] - Xp_tmp[:, :self.dim]
            T0 = torch.einsum('ij,ij->i', D, D)
            DT0 = dtau[:, :self.dim]
            T3 = tau[:, 0] ** 2
            T01 = T0 * torch.einsum('ij,ij->i', DT0, DT0)
            T02 = -2 * tau[:, 0] * torch.einsum('ij,ij->i', DT0, D)
            S0 = T01 + T02 + T3
            Ypred = 1 / torch.sqrt(S0)
            Ypred_list.append(Ypred.clone())
            del Xp_tmp, w, tau, dtau, D, DT0, T0, T01, T02, T3, S0, Ypred
        del Xp
        return torch.cat(Ypred_list, dim=0)

    def TravelTimes(self, Xp, **routing_kwargs):
        tau, w, coords, _, _ = self.network.out(Xp, **routing_kwargs)
        TT = tau[:, 0]
        return TT

    def Tau(self, Xp):
        Xp = Xp.to(torch.device(self.device))
        tau, w, coords, _, _ = self.network.out(Xp)
        return (tau - 1) / 100

    def Speed(self, Xp, **routing_kwargs):
        Xp = Xp.to(torch.device(self.device))
        tau, w, Xp, _, _ = self.network.out(Xp, **routing_kwargs)
        dtau = self.gradient(tau, Xp)

        DT0 = dtau[:, :self.dim]
        S = torch.einsum('ij,ij->i', DT0, DT0)
        Ypred = 1 / torch.sqrt(S)

        del Xp, tau, dtau
        return Ypred

    def Gradient(self, Xp, **routing_kwargs):
        tau, w, Xp, _, _ = self.network.out(Xp, **routing_kwargs)
        dtau = self.gradient(tau, Xp)

        Ypred0 = -dtau[:, :self.dim]
        Spred0 = torch.norm(Ypred0, dim=1).view(-1, 1)
        Ypred0 = 1 / Spred0 ** 2 * Ypred0

        Ypred1 = -dtau[:, self.dim:]
        Spred1 = torch.norm(Ypred1, dim=1).view(-1, 1)
        Ypred1 = 1 / Spred1 ** 2 * Ypred1

        return torch.cat((Ypred0, Ypred1), dim=1)

    def plot(self, epoch, total_train_loss, alpha, source):
        # Minimal plot for multi-task -- reuse single-task pattern
        if igl is None:
            return
        limit = 0.5
        xmin = [-limit, -limit]
        xmax = [limit, limit]

        size = 81
        spacing = 2 * limit / (size - 1)
        X, Y = np.meshgrid(np.arange(-limit, limit + 0.1 * spacing, spacing),
                            np.arange(-limit, limit + 0.1 * spacing, spacing))

        scale = np.pi / 0.5
        Xsrc = source

        XP = np.zeros((len(X.flatten()), 2 * self.dim))
        XP[:, :self.dim] = Xsrc
        XP[:, self.dim:] = Xsrc
        XP = XP / scale
        XP[:, 0] = X.flatten()
        XP[:, 1] = Y.flatten()
        XP = Variable(Tensor(XP)).to(self.device)

        tt = self.TravelTimes(XP)
        ss = self.Speed(XP)
        TT = tt.to('cpu').data.numpy().reshape(X.shape)
        S = ss.to('cpu').data.numpy().reshape(X.shape)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        quad1 = ax.pcolormesh(X, Y, S, vmin=0, vmax=1)
        ax.contour(X, Y, TT, np.arange(0, 10, 0.02), cmap='gist_heat', linewidths=0.5)
        plt.colorbar(quad1, ax=ax, pad=0.1, label='Predicted Velocity')
        plt.savefig(self.path + "/plots" + str(epoch) + "_" + str(alpha) +
                    "_" + str(round(total_train_loss, 4)) + "_0.jpg", bbox_inches='tight')
        plt.close(fig)
