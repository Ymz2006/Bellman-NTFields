import matplotlib
import os
import numpy as np
import math
import random
import time

import torch
import torch.nn.functional as F

from torch.nn import Linear
from torch.nn import LayerNorm, InstanceNorm1d
from torch import Tensor
from torch.nn import Conv3d
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable, grad
from torch.cuda.amp import autocast
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import copy

import matplotlib
import matplotlib.pylab as plt

from timeit import default_timer as timer

try:
    from .zone_router import corner_weights_torch
    from .semantic_relation_gate import (
        ANALYTIC_CONTINUOUS_PERMISSION_TAU,
        FULLSEMANTIC_ATTENTION_TEMPERATURES_M,
        FULLSEMANTIC_FREE_DISTANCE_M,
        FULLSEMANTIC_HEAD_COUNT,
        FULLSEMANTIC_ORIENTATION_MARGIN_SCALE,
        FULLSEMANTIC_REPLICA_COUNT,
        FULLSEMANTIC_SELF_RELATION_COUNT,
        FULLSEMANTIC_SELF_SCALE,
        FULLSEMANTIC_SPHERE_COUNT,
        FullSemanticRelationReplica,
        analytic_continuous_orientation_margins,
        analytic_continuous_approval,
        fullsemantic_fk8_from_pose,
        fullsemantic_orientation_relations,
        fullsemantic_relation_features,
        fullsemantic_relation_minima,
    )
except ImportError:
    from zone_router import corner_weights_torch
    from semantic_relation_gate import (
        ANALYTIC_CONTINUOUS_PERMISSION_TAU,
        FULLSEMANTIC_ATTENTION_TEMPERATURES_M,
        FULLSEMANTIC_FREE_DISTANCE_M,
        FULLSEMANTIC_HEAD_COUNT,
        FULLSEMANTIC_ORIENTATION_MARGIN_SCALE,
        FULLSEMANTIC_REPLICA_COUNT,
        FULLSEMANTIC_SELF_RELATION_COUNT,
        FULLSEMANTIC_SELF_SCALE,
        FULLSEMANTIC_SPHERE_COUNT,
        FullSemanticRelationReplica,
        analytic_continuous_orientation_margins,
        analytic_continuous_approval,
        fullsemantic_fk8_from_pose,
        fullsemantic_orientation_relations,
        fullsemantic_relation_features,
        fullsemantic_relation_minima,
    )

torch.backends.cudnn.benchmark = True


ROUTE_CHART_CONSENSUS_MODE = (
    'corridor_hurdle_semantic_veto_min_odds_residual_routechart4_soft0125')
ROUTE_CHART_UNION_CONSENSUS_MODE = (
    'corridor_hurdle_semantic_veto_min_odds_residual_routechart4_unionlme005')
ROUTE_CHART_FOURIER_RISK_CONSENSUS_MODE = (
    'corridor_hurdle_semantic_veto_min_odds_residual_'
    'routechart4_unionlme005_fourier_physicalrisk4')
ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_CONSENSUS_MODE = (
    'corridor_hurdle_semantic_veto_min_odds_residual_'
    'routechart4_unionlme005_partialsharedfourier_physicalrisk4')
ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_MARGIN_RESCUE_CONSENSUS_MODE = (
    'corridor_hurdle_semantic_margin025_rescue_gain1_min_odds_residual_'
    'routechart4_unionlme005_partialsharedfourier_physicalrisk4')
ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_SEMANTIC_ANCHOR_RESCUE_CONSENSUS_MODE = (
    'corridor_hurdle_semantic_margin025_rescue_gain1_anchorapproved_'
    'min_odds_residual_routechart4_unionlme005_'
    'partialsharedfourier_physicalrisk4')
ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_REPLICA_MEDIAN_ANCHOR_RESCUE_CONSENSUS_MODE = (
    'corridor_hurdle_anchorprivate5_replica3_median_margin025_'
    'anchorapproved_min_odds_residual_routechart4_unionlme005_'
    'partialsharedfourier_physicalrisk4')
ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_FULLSEMANTIC_ANCHOR_RESCUE_CONSENSUS_MODE = (
    'corridor_hurdle_fullsemantic4_relationqkv_replica3_median_margin025_'
    'anchorapproved_min_odds_residual_routechart4_unionlme005_'
    'partialsharedfourier_physicalrisk4')
ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_ANALYTIC_CONTINUOUS_ANCHOR_RESCUE_CONSENSUS_MODE = (
    'corridor_hurdle_analyticcontinuous4_exactmargin_min_tau001_'
    'anchorapproved_min_odds_residual_routechart4_unionlme005_'
    'partialsharedfourier_physicalrisk4')
ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_ANALYTIC_SPEED_ENVELOPE_CONSENSUS_MODE = (
    'corridor_hurdle_analyticcontinuous4_exactmargin_min_tau001_'
    'finalspeed_anchor_envelope_min_odds_residual_routechart4_unionlme005_'
    'partialsharedfourier_physicalrisk4')
ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_ANALYTIC_EXACT_SCALAR_CONSENSUS_MODE = (
    'corridor_hurdle_analyticcontinuous4_exactscalar_softmin_tau004_'
    'floor010_min_odds_residual_routechart4_unionlme005_'
    'partialsharedfourier_physicalrisk4')
ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_ANALYTIC_TRUST_CONTINUATION_CONSENSUS_MODE = (
    'corridor_hurdle_analyticcontinuous4_trustc1exp_theta020_floor010_'
    'softmin_tau004_min_odds_residual_routechart4_unionlme005_'
    'partialsharedfourier_physicalrisk4')
ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_ANALYTIC_EXACT_TRUST_ANCHOR_POTENTIAL_CONSENSUS_MODE = (
    'corridor_hurdle_analyticcontinuous4_exacttrust_anchorpotential_c1_'
    'theta020_floor010_softmin_tau004_min_odds_residual_'
    'routechart4_unionlme005_partialsharedfourier_physicalrisk4')
ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_ANALYTIC_EXACT_TRUST_WINNER_TAKE_HEAD_CONSENSUS_MODE = (
    'corridor_hurdle_analyticcontinuous4_exacttrust_winnertakehead_'
    'theta020_floor010_softmin_tau004_min_odds_residual_'
    'routechart4_unionlme005_partialsharedfourier_physicalrisk4')
ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_ANALYTIC_HEADWISE_INTEGRATED_T_CONSENSUS_MODE = (
    'corridor_hurdle_analyticcontinuous4_headwiseintegratedT3_'
    'finalmin_exactphysicalbarrier_theta020_floor010_softmin_tau004_'
    'routechart4_unionlme005_partialsharedfourier_physicalrisk4')
ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_ANALYTIC_HANDOVER_HEADWISE_INTEGRATED_T_CONSENSUS_MODE = (
    'corridor_hurdle_analyticcontinuous4_handoversemantic_headwiseintegratedT4_'
    'finalmin_exactphysicalbarrier_theta020_floor010_softmin_tau004_'
    'routechart4_unionlme005_partialsharedfourier_physicalrisk4')
ANALYTIC_HEADWISE_INTEGRATED_T_MODES = (
    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_ANALYTIC_HEADWISE_INTEGRATED_T_CONSENSUS_MODE,
    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_ANALYTIC_HANDOVER_HEADWISE_INTEGRATED_T_CONSENSUS_MODE,
)
ROUTE_CHART_ANALYTIC_CONTINUOUS_MODES = (
    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_ANALYTIC_CONTINUOUS_ANCHOR_RESCUE_CONSENSUS_MODE,
    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_ANALYTIC_SPEED_ENVELOPE_CONSENSUS_MODE,
    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_ANALYTIC_EXACT_SCALAR_CONSENSUS_MODE,
    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_ANALYTIC_TRUST_CONTINUATION_CONSENSUS_MODE,
    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_ANALYTIC_EXACT_TRUST_ANCHOR_POTENTIAL_CONSENSUS_MODE,
    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_ANALYTIC_EXACT_TRUST_WINNER_TAKE_HEAD_CONSENSUS_MODE,
    *ANALYTIC_HEADWISE_INTEGRATED_T_MODES,
)
ROUTE_CHART_ANCHOR_MODES = (
    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_SEMANTIC_ANCHOR_RESCUE_CONSENSUS_MODE,
    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_REPLICA_MEDIAN_ANCHOR_RESCUE_CONSENSUS_MODE,
    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_FULLSEMANTIC_ANCHOR_RESCUE_CONSENSUS_MODE,
    *ROUTE_CHART_ANALYTIC_CONTINUOUS_MODES,
)
ROUTE_CHART_PARTIAL_SHARED_MODES = (
    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_CONSENSUS_MODE,
    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_MARGIN_RESCUE_CONSENSUS_MODE,
    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_SEMANTIC_ANCHOR_RESCUE_CONSENSUS_MODE,
    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_REPLICA_MEDIAN_ANCHOR_RESCUE_CONSENSUS_MODE,
    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_FULLSEMANTIC_ANCHOR_RESCUE_CONSENSUS_MODE,
    *ROUTE_CHART_ANALYTIC_CONTINUOUS_MODES,
)
ROUTE_CHART_MODES = (
    ROUTE_CHART_CONSENSUS_MODE,
    ROUTE_CHART_UNION_CONSENSUS_MODE,
    ROUTE_CHART_FOURIER_RISK_CONSENSUS_MODE,
    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_CONSENSUS_MODE,
    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_MARGIN_RESCUE_CONSENSUS_MODE,
    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_SEMANTIC_ANCHOR_RESCUE_CONSENSUS_MODE,
    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_REPLICA_MEDIAN_ANCHOR_RESCUE_CONSENSUS_MODE,
    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_FULLSEMANTIC_ANCHOR_RESCUE_CONSENSUS_MODE,
    *ROUTE_CHART_ANALYTIC_CONTINUOUS_MODES,
)
ROUTE_CHART_UNION_MODES = (
    ROUTE_CHART_UNION_CONSENSUS_MODE,
    ROUTE_CHART_FOURIER_RISK_CONSENSUS_MODE,
    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_CONSENSUS_MODE,
    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_MARGIN_RESCUE_CONSENSUS_MODE,
    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_SEMANTIC_ANCHOR_RESCUE_CONSENSUS_MODE,
    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_REPLICA_MEDIAN_ANCHOR_RESCUE_CONSENSUS_MODE,
    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_FULLSEMANTIC_ANCHOR_RESCUE_CONSENSUS_MODE,
    *ROUTE_CHART_ANALYTIC_CONTINUOUS_MODES,
)
ROUTE_CHART_FOURIER_RISK_MODES = (
    ROUTE_CHART_FOURIER_RISK_CONSENSUS_MODE,
    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_CONSENSUS_MODE,
    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_MARGIN_RESCUE_CONSENSUS_MODE,
    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_SEMANTIC_ANCHOR_RESCUE_CONSENSUS_MODE,
    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_REPLICA_MEDIAN_ANCHOR_RESCUE_CONSENSUS_MODE,
    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_FULLSEMANTIC_ANCHOR_RESCUE_CONSENSUS_MODE,
    *ROUTE_CHART_ANALYTIC_CONTINUOUS_MODES,
)
ROUTE_CHART_LIVE_FK_MODES = (
    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_FULLSEMANTIC_ANCHOR_RESCUE_CONSENSUS_MODE,
    *ROUTE_CHART_ANALYTIC_CONTINUOUS_MODES,
)
ROUTE_CHART_COUNT = 4
ROUTE_CHART_TEMPERATURE_SCALE = 0.125
ROUTE_CHART_SEED = 8072065
ROUTE_CHART_UNION_TEMPERATURE = 0.005
ROUTE_CHART_FOURIER_RISK_FREQUENCIES = (1.0, 2.0, 4.0, 8.0)
ROUTE_CHART_FOURIER_RISK_HEAD_COUNT = 4
ROUTE_CHART_MARGIN_RESCUE_CONFIDENCE = 0.25
ROUTE_CHART_MARGIN_RESCUE_GAIN = 1.0
ROUTE_CHART_MARGIN_RESCUE_EVIDENCE_CAP = 4.0
ANALYTIC_TRUST_BOUNDARY = 0.2
ANCHOR_RELIABILITY_REPLICA_COUNT = 3
ANCHOR_RELIABILITY_HEAD_COUNT = 5


class PartialSharedFourierRouteTail(torch.nn.Module):
    """Chart-private hidden/fusion tail fed by one shared Fourier branch."""

    def __init__(self, hidden_dimension):
        super().__init__()
        self.hidden_branch = torch.nn.Sequential(
            Linear(hidden_dimension, 64), torch.nn.SiLU())
        self.private_fusion = torch.nn.Sequential(
            Linear(128, 64), torch.nn.SiLU())
        self.output = Linear(64, 1)
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, hidden, shared_fourier):
        private = self.private_fusion(torch.cat([
            self.hidden_branch(hidden), shared_fourier], dim=1))
        return self.output(private)


class AnchorPrivateReliabilityReplica(torch.nn.Module):
    """One q-only five-head reliability member used by Phase-CR."""

    def __init__(self, anchor_dimension, q_feature_dimension, context_dimension):
        super().__init__()
        self.anchor_branch = torch.nn.Sequential(
            Linear(anchor_dimension, 64), torch.nn.SiLU())
        self.q_branch = torch.nn.Sequential(
            Linear(q_feature_dimension, 128), torch.nn.SiLU(),
            Linear(128, 64), torch.nn.SiLU())
        self.context_branch = torch.nn.Sequential(
            Linear(context_dimension, 64), torch.nn.SiLU())
        self.shared_fusion = torch.nn.Sequential(
            Linear(192, 64), torch.nn.SiLU())
        self.private_tails = torch.nn.ModuleList([
            torch.nn.Sequential(
                Linear(64, 64), torch.nn.SiLU(), Linear(64, 1))
            for _ in range(ANCHOR_RELIABILITY_HEAD_COUNT)
        ])
        for tail in self.private_tails:
            torch.nn.init.zeros_(tail[-1].weight)
            torch.nn.init.zeros_(tail[-1].bias)

    def forward(self, anchor_hidden, q_features, context):
        if not (anchor_hidden.shape[0] == q_features.shape[0]
                == context.shape[0]):
            raise ValueError("anchor reliability inputs must share batch size")
        shared = self.shared_fusion(torch.cat([
            self.anchor_branch(anchor_hidden),
            self.q_branch(q_features),
            self.context_branch(context),
        ], dim=1))
        return torch.cat(
            [tail(shared) for tail in self.private_tails], dim=1)


def sigmoid_out(input):
    return torch.sigmoid(0.1 * input)


def neutral_log_odds_fusion(prior, evidence, eps=1e-7):
    """Fuse two probabilities by adding logits; evidence=0.5 is neutral."""
    prior = prior.clamp(min=eps, max=1.0 - eps)
    evidence = evidence.clamp(min=eps, max=1.0 - eps)
    numerator = prior * evidence
    denominator = numerator + (1.0 - prior) * (1.0 - evidence)
    return numerator / denominator.clamp_min(eps)


def normalized_release_smoothmax(
        probabilities, temperature=ROUTE_CHART_UNION_TEMPERATURE):
    """Smooth OR over releases that exactly preserves identical experts.

    Subtracting ``log(K)`` makes this a normalized log-mean-exp: when every
    expert emits the same release ``r``, the aggregate is ``r``.  Unlike a
    chart-weighted mixture, one confident private expert can therefore retain
    a learned corridor.  The result remains one smooth q-only probability.
    """
    if probabilities.ndim != 2 or probabilities.shape[1] < 1:
        raise ValueError("probabilities must have shape (N,K), K >= 1")
    if temperature <= 0.0:
        raise ValueError("smoothmax temperature must be positive")
    aggregate = temperature * (
        torch.logsumexp(probabilities / temperature, dim=1, keepdim=True)
        - math.log(probabilities.shape[1]))
    return aggregate.clamp(min=0.0, max=1.0)


def analytic_final_speed_envelope(
        base_speed, anchor_speed, approval):
    """Open a final q-only scalar only where exact margins approve it.

    Unlike a release-space anchor, this does not project the formal scalar
    through another model's repaired/direct/magnitude interval.  Zero
    approval exactly preserves the base; full approval returns the pointwise
    maximum of base and anchor.
    """
    if not (base_speed.shape == anchor_speed.shape == approval.shape):
        raise ValueError("base, anchor, and approval must share shape")
    if not (base_speed.is_floating_point()
            and anchor_speed.is_floating_point()
            and approval.is_floating_point()):
        raise TypeError("speed envelope inputs must be floating point")
    return base_speed + approval.clamp(0.0, 1.0) * torch.relu(
        anchor_speed - base_speed)


def analytic_unclamped_scalar_from_components(
        primitive_components, temperature):
    """Compose four primitive heads before the historical final clamp.

    The first two heads are obstacle and scaled self clearance; their hard
    minimum is the canonical safety component.  Safety, active cup, and
    active pour then use the dataset's constant-preserving normalized
    soft-min.
    """
    if primitive_components.ndim != 2 or primitive_components.shape[1] != 4:
        raise ValueError("primitive components must have shape (N,4)")
    if not primitive_components.is_floating_point():
        raise TypeError("primitive components must be floating point")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    safe = primitive_components[:, :2].min(dim=1, keepdim=True).values
    components = torch.cat([safe, primitive_components[:, 2:]], dim=1)
    normalized_log_mean_exp = (
        torch.logsumexp(-components / temperature, dim=1, keepdim=True)
        - math.log(components.shape[1]))
    return -temperature * normalized_log_mean_exp


def analytic_handover_blend_from_fk8(fk8):
    """Return the formal corner handover coordinate from q-derived FK8.

    Zero is the pure cup wing, one is the pure pour wing, and the open
    interval is the corner where the formal cross validator uses trajectory
    monotonicity instead of a pointwise orientation target.
    """
    if fk8.ndim != 2 or fk8.shape[1] != 8:
        raise ValueError("fk8 must have shape (N,8)")
    if not fk8.is_floating_point():
        raise TypeError("fk8 must be floating point")
    theta = torch.atan2(fk8[:, 1], fk8[:, 0])
    midpoint = theta.new_tensor(-3.0 * math.pi / 4.0)
    half_band = theta.new_tensor(math.pi / 4.0)
    wrapped = torch.atan2(
        torch.sin(theta - midpoint), torch.cos(theta - midpoint))
    return ((wrapped + half_band) / (2.0 * half_band)).clamp(0.0, 1.0)


def analytic_handover_components_from_primitives(
        primitive_components, fk8, wing_epsilon=1.0e-6):
    """Replace only orientation terms irrelevant to a cross handover.

    Obstacle and self-collision components are copied exactly.  Cup remains
    active only on the pure cup wing, pour only on the pure pour wing, and
    both are neutral in the corner.  Every input is still a function of q.
    """
    if (primitive_components.ndim != 2
            or primitive_components.shape[1] != 4):
        raise ValueError("primitive components must have shape (N,4)")
    if fk8.shape != (primitive_components.shape[0], 8):
        raise ValueError("fk8 must have shape (N,8) matching components")
    if not primitive_components.is_floating_point():
        raise TypeError("primitive components must be floating point")
    if primitive_components.device != fk8.device:
        raise ValueError("primitive components and fk8 must share a device")
    if primitive_components.dtype != fk8.dtype:
        raise ValueError("primitive components and fk8 must share a dtype")
    wing_epsilon = float(wing_epsilon)
    if not 0.0 <= wing_epsilon < 0.5:
        raise ValueError("wing_epsilon must lie in [0,0.5)")
    blend = analytic_handover_blend_from_fk8(fk8)
    one = torch.ones_like(blend)
    cup = torch.where(
        blend <= wing_epsilon, primitive_components[:, 2], one)
    pour = torch.where(
        blend >= 1.0 - wing_epsilon, primitive_components[:, 3], one)
    return torch.stack([
        primitive_components[:, 0], primitive_components[:, 1], cup, pour,
    ], dim=1)


def analytic_exact_scalar_from_margins(
        primitive_components, temperature, speed_floor):
    """Rebuild the dataset's clipped scalar from four physical heads."""
    if not 0.0 < float(speed_floor) < 1.0:
        raise ValueError("speed floor must lie in (0,1)")
    scalar = analytic_unclamped_scalar_from_components(
        primitive_components, temperature)
    return scalar.clamp(float(speed_floor), 1.0)


def analytic_trust_boundary_continuation(
        raw_scalar, speed_floor,
        trust_boundary=ANALYTIC_TRUST_BOUNDARY):
    """Replace a hard lower clamp by its C1 trust-matched exponential tail.

    The lower branch has the same value and derivative as the identity at the
    trust boundary.  It therefore preserves the complete trusted set while
    retaining a true scalar derivative below that boundary.
    """
    if not raw_scalar.is_floating_point():
        raise TypeError("raw scalar must be floating point")
    speed_floor = float(speed_floor)
    trust_boundary = float(trust_boundary)
    if not 0.0 < speed_floor < trust_boundary < 1.0:
        raise ValueError(
            "require 0 < speed_floor < trust_boundary < 1")
    scale = trust_boundary - speed_floor
    lower = speed_floor + scale * torch.exp(
        (raw_scalar - trust_boundary) / scale)
    return torch.where(
        raw_scalar < trust_boundary,
        lower,
        raw_scalar.clamp(max=1.0))


def analytic_exact_trust_anchor_potential(
        physical_speed, anchor_speed, speed_floor,
        trust_boundary=ANALYTIC_TRUST_BOUNDARY):
    """Use a frozen scalar only to shape the sub-trust optimization basin.

    The squared physical gap makes the anchor correction and its first
    derivatives vanish at the trust boundary.  Its normalized release is in
    [0,1], so the correction cannot cross that boundary.  Above trust the
    physical scalar is returned exactly.
    """
    if physical_speed.shape != anchor_speed.shape:
        raise ValueError("physical_speed and anchor_speed must share shape")
    if not (
            physical_speed.is_floating_point()
            and anchor_speed.is_floating_point()):
        raise TypeError("physical and anchor speeds must be floating point")
    speed_floor = float(speed_floor)
    trust_boundary = float(trust_boundary)
    if not 0.0 < speed_floor < trust_boundary < 1.0:
        raise ValueError(
            "require 0 < speed_floor < trust_boundary < 1")
    scale = trust_boundary - speed_floor
    anchor_release = (
        (anchor_speed - speed_floor) / (1.0 - speed_floor)).clamp(0.0, 1.0)
    gap = (trust_boundary - physical_speed).clamp_min(0.0)
    lower = physical_speed + gap.square() * anchor_release / scale
    boundary = physical_speed.new_tensor(trust_boundary)
    floor = physical_speed.new_tensor(speed_floor)
    strict_subthreshold = torch.nextafter(boundary, floor)
    lower = torch.minimum(lower, strict_subthreshold)
    return torch.where(
        physical_speed < trust_boundary, lower, physical_speed)


def analytic_exact_trust_winner_take_head(
        physical_speed, anchor_speed, speed_floor,
        trust_boundary=ANALYTIC_TRUST_BOUNDARY):
    """Select one physical or sub-trust anchor head without blending them."""
    if physical_speed.shape != anchor_speed.shape:
        raise ValueError("physical_speed and anchor_speed must share shape")
    if not (
            physical_speed.is_floating_point()
            and anchor_speed.is_floating_point()):
        raise TypeError("physical and anchor speeds must be floating point")
    speed_floor = float(speed_floor)
    trust_boundary = float(trust_boundary)
    if not 0.0 < speed_floor < trust_boundary < 1.0:
        raise ValueError(
            "require 0 < speed_floor < trust_boundary < 1")
    anchor_release = (
        (anchor_speed - speed_floor) / (1.0 - speed_floor)).clamp(0.0, 1.0)
    guide = speed_floor + (
        trust_boundary - speed_floor) * anchor_release
    boundary = physical_speed.new_tensor(trust_boundary)
    floor = physical_speed.new_tensor(speed_floor)
    guide = torch.minimum(guide, torch.nextafter(boundary, floor))
    return torch.maximum(physical_speed, guide)


def canonical_symmetric_chord_points(q0, q1, nodes):
    """Evaluate mirrored quadrature nodes from one commutative midpoint.

    The returned order is the usual increasing-node order.  Reversing the
    endpoints therefore returns exactly the reversed point tensor in finite
    precision, rather than merely a mathematically equivalent approximation.
    """
    if q0.ndim != 2 or q0.shape != q1.shape:
        raise ValueError("q0 and q1 must share one rank-2 shape")
    if nodes.ndim != 1 or nodes.numel() < 2:
        raise ValueError("nodes must be a rank-1 tensor with Q >= 2")
    if not (q0.is_floating_point() and q1.is_floating_point()
            and nodes.is_floating_point()):
        raise TypeError("endpoints and nodes must be floating point")
    half = nodes.numel() // 2
    left_nodes = nodes[:half]
    mirrored_right_nodes = torch.flip(nodes[-half:], dims=[0])
    offset = 0.5 * (mirrored_right_nodes - left_nodes)
    midpoint = 0.5 * (q0 + q1)
    delta = q1 - q0
    displacement = offset[None, :, None] * delta[:, None, :]
    left_points = midpoint[:, None, :] - displacement
    mirrored_right_points = midpoint[:, None, :] + displacement
    parts = [left_points]
    if nodes.numel() % 2:
        parts.append(midpoint[:, None, :])
    parts.append(torch.flip(mirrored_right_points, dims=[1]))
    return torch.cat(parts, dim=1)


def reduce_headwise_scalar_chord_times(
        head_speeds, distance, weights, speed_floor):
    """Pair mirrored slownesses, then integrate heads before the final min."""
    if head_speeds.ndim != 3 or head_speeds.shape[2] < 2:
        raise ValueError("head_speeds must have shape (N,Q,H), H >= 2")
    if distance.shape != (head_speeds.shape[0], 1):
        raise ValueError("distance must have shape (N,1)")
    if weights.ndim != 1 or weights.shape[0] != head_speeds.shape[1]:
        raise ValueError("weights must have shape (Q,)")
    if not (
            head_speeds.is_floating_point()
            and distance.is_floating_point()
            and weights.is_floating_point()):
        raise TypeError("head speeds, distance, and weights must be floating")
    if float(speed_floor) <= 0.0:
        raise ValueError("speed_floor must be positive")
    slowness = head_speeds.clamp_min(float(speed_floor)).reciprocal()
    half = head_speeds.shape[1] // 2
    left = slowness[:, :half, :]
    mirrored_right = torch.flip(slowness[:, -half:, :], dims=[1])
    pair_weights = 0.5 * (
        weights[:half] + torch.flip(weights[-half:], dims=[0]))
    pair_terms = (
        left + mirrored_right) * pair_weights[None, :, None]
    mean_slowness = pair_terms.sum(dim=1)
    if head_speeds.shape[1] % 2:
        mean_slowness = mean_slowness + (
            slowness[:, half, :] * weights[half])
    head_times = distance * mean_slowness
    final_time = head_times.min(dim=1, keepdim=True).values
    return final_time, head_times


def monotone_release_veto(proposal_release, evidence, eps=1e-6):
    """Apply only negative log-odds evidence to a scalar release proposal."""
    if proposal_release.shape != evidence.shape:
        raise ValueError("proposal_release and evidence must share shape")
    negative = torch.clamp(evidence, max=0.0)
    safe = proposal_release.clamp(eps, 1.0 - eps)
    vetoed = torch.sigmoid(torch.logit(safe) + negative)
    vetoed = torch.minimum(vetoed, proposal_release)
    return torch.where(evidence >= 0.0, proposal_release, vetoed)


def margin_signed_release_rescue(
        proposal_release, evidence,
        confidence_margin=ROUTE_CHART_MARGIN_RESCUE_CONFIDENCE,
        positive_gain=ROUTE_CHART_MARGIN_RESCUE_GAIN,
        evidence_cap=ROUTE_CHART_MARGIN_RESCUE_EVIDENCE_CAP,
        eps=1e-6):
    """Veto weak evidence and spend only bounded strong positive evidence."""
    if proposal_release.shape != evidence.shape:
        raise ValueError("proposal_release and evidence must share shape")
    if confidence_margin < 0.0:
        raise ValueError("confidence_margin must be nonnegative")
    if positive_gain < 0.0:
        raise ValueError("positive_gain must be nonnegative")
    if evidence_cap <= 0.0:
        raise ValueError("evidence_cap must be positive")
    shifted = evidence - float(confidence_margin)
    vetoed = monotone_release_veto(proposal_release, shifted, eps=eps)
    positive = shifted.clamp(min=0.0, max=float(evidence_cap))
    safe = vetoed.clamp(eps, 1.0 - eps)
    return torch.sigmoid(torch.logit(safe) + float(positive_gain) * positive)


def scalar_speed_to_release(
        speed, repaired, direct, magnitude, eps=1.0e-6):
    """Express a frozen scalar anchor inside the repaired/direct interval."""
    if not (speed.shape == repaired.shape == direct.shape == magnitude.shape):
        raise ValueError("all scalar tensors must share shape")
    if not all(tensor.is_floating_point() for tensor in (
            speed, repaired, direct, magnitude)):
        raise TypeError("all scalar tensors must be floating point")
    gap = torch.relu(direct - repaired)
    safe_gap = torch.where(gap > eps, gap, torch.ones_like(gap))
    interval_fraction = ((speed - repaired) / safe_gap).clamp(0.0, 1.0)
    interval_fraction = torch.where(
        gap > eps, interval_fraction, torch.zeros_like(interval_fraction))
    safe_magnitude = magnitude.clamp_min(eps)
    release = (interval_fraction / safe_magnitude).clamp(0.0, 1.0)
    return torch.where(
        magnitude > eps, release, torch.zeros_like(release))


def monotone_anchor_release_union(
        proposal_release, anchor_release,
        temperature=ROUTE_CHART_UNION_TEMPERATURE):
    """Add the positive normalized-LME anchor increment to a proposal."""
    if proposal_release.shape != anchor_release.shape:
        raise ValueError("proposal and anchor release must share shape")
    if not proposal_release.is_floating_point() or not (
            anchor_release.is_floating_point()):
        raise TypeError("proposal and anchor release must be floating point")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    difference = (
        anchor_release - proposal_release) / float(temperature)
    increment = float(temperature) * (
        F.softplus(difference) - math.log(2.0))
    return (proposal_release + increment.clamp_min(0.0)).clamp(0.0, 1.0)


def semantic_anchor_approval(
        evidence,
        confidence_margin=ROUTE_CHART_MARGIN_RESCUE_CONFIDENCE,
        evidence_cap=ROUTE_CHART_MARGIN_RESCUE_EVIDENCE_CAP):
    """Give the anchor authority only under strong unanimous evidence."""
    if not evidence.is_floating_point():
        raise TypeError("semantic evidence must be floating point")
    if confidence_margin < 0.0:
        raise ValueError("confidence_margin must be nonnegative")
    if evidence_cap <= 0.0:
        raise ValueError("evidence_cap must be positive")
    shifted = evidence - float(confidence_margin)
    return shifted.clamp(
        min=0.0, max=float(evidence_cap)) / float(evidence_cap)


def semantic_approved_anchor_release(
        proposal_release, anchor_release, evidence):
    """Apply Phase-CJ approval to only the anchor's monotone increment."""
    if not (proposal_release.shape == anchor_release.shape == evidence.shape):
        raise ValueError("proposal, anchor, and evidence must share shape")
    unrestricted = monotone_anchor_release_union(
        proposal_release, anchor_release)
    approval = semantic_anchor_approval(evidence)
    return proposal_release + approval * (
        unrestricted - proposal_release)


def replica_median_positive_anchor_release(
        proposal_release, anchor_release, replica_logits):
    """Require two of three five-head replicas before spending anchor lift."""
    if replica_logits.ndim != 3 or replica_logits.shape[0] != (
            ANCHOR_RELIABILITY_REPLICA_COUNT) or replica_logits.shape[2] != (
                ANCHOR_RELIABILITY_HEAD_COUNT):
        raise ValueError("replica logits must have shape (3,N,5)")
    if proposal_release.shape != anchor_release.shape:
        raise ValueError("proposal and anchor release must share shape")
    if proposal_release.shape != (replica_logits.shape[1], 1):
        raise ValueError("scalar releases must have shape (N,1)")
    per_replica = replica_logits.min(dim=2).values
    evidence = per_replica.median(dim=0).values[:, None]
    approval = semantic_anchor_approval(evidence)
    unrestricted = monotone_anchor_release_union(
        proposal_release, anchor_release)
    release = proposal_release + approval * (
        unrestricted - proposal_release)
    return release, approval, evidence


def normalized_logit_softmin_probability(
        probabilities, temperature=1.0, eps=1e-7):
    """Smoothly aggregate semantic probabilities in logit space.

    The normalized log-mean-exp preserves equal inputs exactly.  At small
    temperature it approaches the weakest semantic head; at the registered
    temperature one it lets agreement from all heads contribute without
    introducing a learned travel-time output.
    """
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must have shape (N,H) with H >= 2")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    logits = torch.logit(probabilities.clamp(eps, 1.0 - eps))
    smooth_logit = -temperature * (
        torch.logsumexp(-logits / temperature, dim=1, keepdim=True)
        - math.log(probabilities.shape[1]))
    return torch.sigmoid(smooth_logit)


class Sigmoid_out(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return sigmoid_out(input)


class PairResidualBlock(torch.nn.Module):
    """Zero-initializable refinement block for the endpoint-pair value head."""

    def __init__(self, width, activation='silu'):
        super().__init__()
        self.norm = LayerNorm(width)
        self.fc1 = Linear(width, width)
        self.fc2 = Linear(width, width)
        self.activation = str(activation)
        if self.activation not in ('silu', 'maxmin'):
            raise ValueError(
                f"PairResidualBlock activation must be 'silu' or 'maxmin', "
                f"got {self.activation!r}")

    def forward(self, x):
        hidden = self.fc1(self.norm(x))
        if self.activation == 'silu':
            hidden = F.silu(hidden)
        else:
            hidden = MaxMin()(hidden)
        return x + 0.5 * self.fc2(hidden)


class MaxMin(torch.nn.Module):
    """Pairwise GroupSort activation for a nonsmooth scalar value field.

    A viscosity solution is generally only Lipschitz at a cut locus: when two
    equally short characteristics meet, its gradient switches rather than
    averaging the two directions.  SiLU-only value heads are smooth by
    construction.  Pairwise min/max sorting gives one ordinary MLP enough
    piecewise-smooth capacity to represent that switch without introducing
    experts, candidate paths, or additional outputs.
    """

    def forward(self, x):
        if x.shape[-1] % 2:
            raise ValueError(
                "MaxMin requires an even feature width, got "
                f"{x.shape[-1]}")
        paired = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        lower = paired.min(dim=-1).values
        upper = paired.max(dim=-1).values
        return torch.cat([lower, upper], dim=-1)



_SLOWNESS_FLOOR_BETA = 10.0   # sharpness of the post-residual softplus floor


def _bounded_slowness(logit, s_min=1.0, s_max=12.0):
    """Slowness bounded to the label range via a smooth sigmoid.

    Labels imply slowness = 1/speed in [1, 10] (speed in [0.1, 1.0]).  Bound the
    head to (s_min, s_max) so it can never run away, and so speed <= 1 -- the
    NTFields normalisation -- holds by construction.

    History (2026-08-31): the original ``clamp(softplus(x), max=20)`` has ZERO
    gradient above the cap, and every trained checkpoint ended with 100% of
    inputs above it -- a frozen head emitting a constant metric, whose geodesics
    are straight lines.  Replacing the cap with a leaky ceiling restored the
    gradient but reproduced the "rollback-thrashing collapse" the original
    comment warned about (safety-head loss 4.7e-04 -> 7.7, epochs stalling),
    because an unbounded head really does destabilise the coupled tau.  A
    two-sided smooth bound keeps the gradient alive everywhere inside the range
    the labels actually occupy, without giving the head anywhere to run to.
    """
    # NOTE: a SATURATING map (sigmoid/hard clamp) cannot work here.  Measured
    # 2026-08-31: the head reaches its ceiling within one epoch, and past that
    # the map's derivative is ~1e-4, so no loss -- however strong -- can pull it
    # back.  Both clamp(max=20) and sigmoid(1,12) ended pinned with std 0.000.
    # So keep the map unbounded above with a hard floor at 1 (speed <= 1, the
    # NTFields normalisation) and let the DIRECT slowness supervision
    # (R62_SLOWNESS_WEIGHT) supply the anchor.  An unbounded head with no anchor
    # is what destabilised training earlier; the anchor is what the clamp was
    # standing in for.
    return s_min + F.softplus(logit)

class MultiNN(torch.nn.Module):
    """Dual-head network with constraint-aware routing in overlap zone.

    Architecture:
      - Shared Fourier encoding (B matrix + input_mapping)
      - Shared pe_gate (input modulation -> u, v vectors)
      - Two separate encoder heads (cupping + pouring), each a stack of
        Gated SIREN blocks identical to the single-task encoder
      - Shared encoder_norm (InstanceNorm1d)
      - Zone-aware + constraint-aware routing blends head outputs
      - Shared sym_op (smooth-L2 + grouped LogSumExp -> scalar tau)
    """

    def __init__(self, device, dim, B, h_size=256, nl1=2,
                 y_divide=-0.2, overlap=0.05, route_temp=0.01,
                 separate_pe_gate=False,
                 separate_fourier=False, separate_encoder_norm=False,
                 weight_mode='default',
                 use_projector=False, use_dual_sym=False,
                 blend_mode='feature', detach_routing_weights=False,
                 cup_side=-1, use_orient_head=False,
                 add_orient_to_tau=False, orient_tau_lambda=0.1,
                 use_task_orient_heads=False,
                 use_safety_head=False, add_safety_to_tau=False,
                 use_pair_safety_heads=False,
                 pair_safety_mode='replace',
                 pair_safety_residual_scale=1.0,
                 safety_tau_lambda=0.1,
                 zone_mode='y', corner_zones=None, corner_spatial_temp=0.05,
                 fk_feature_dim=0,
                 use_task_adapters=False, handover_k=0, handover_temp=0.03,
                 value_arch='metric', value_condition='task',
                 finsler_quad_points=2, finsler_task_heads=False,
                 use_cross_slowness_residual=False,
                 cross_residual_scale=0.5,
                 use_local_slowness_residual=False,
                 local_residual_scale=0.2,
                 local_residual_gate='all',
                 endpoint_task_siamese=False,
                 endpoint_dual_value_heads=False,
                 task_embedding_scale=0.25,
                 global_residual_scale=2.0,
                 use_speed_head=False,
                 separate_speed_encoder=False,
                 speed_B=None,
                 speed_residual_layers=0,
                 speed_correction_width=0,
                 speed_correction_scale=1.0,
                 speed_correction_mode='signed_logit',
                 lifted_phase=False,
                 lifted_phase_physical_dim=0,
                 lifted_phase_scale=0.1,
                 lifted_phase_width=128,
                 lifted_phase_logit_scale=8.0,
                 lifted_phase_causal=False,
                 lifted_phase_hazard_floor_scale=0.0,
                 cross_hazard_floor_scale=0.0,
                 use_clearance_head=False,
                 clearance_scale=0.05,
                 clearance_boundary=0.005,
                 clearance_speed_floor=0.001,
                 clearance_residual_layers=0,
                 clearance_fourier_skip=False,
                 couple_speed_latent_to_value=False,
                 partial_shared_embedding_dim=0,
                 pair_slowness_attention_dim=0,
                 speed_constraint_heads=0,
                 speed_constraint_softmin_temperature=0.04,
                 speed_constraint_residual_scale=1.25,
                 speed_constraint_fourier_skip=False,
                 speed_constraint_consensus_width=0,
                 speed_constraint_consensus_mode='single',
                 speed_constraint_repair_width=0,
                 global_residual_positive=False,
                 global_excess_centered_gate=False,
                 global_excess_centered_signed_gate=False,
                 chord_quad_points=3,
                 chord_quadrature='legendre',
                 chord_hazard_floor_scale=1.0,
                 use_raw_coordinate_skip=False,
                 speed_floor=0.1,
                 endpoint_barrier_speed=0.1,
                 endpoint_barrier_scale=1.0,
                 endpoint_barrier_radius=0.05,
                 endpoint_barrier_power=1.0,
                 endpoint_barrier_physical_decode=False,
                 endpoint_barrier_contrast=False,
                 global_pair_residual_layers=0,
                 global_pair_residual_activation='silu',
                 use_viscosity_excess_head=False,
                 viscosity_excess_gate='all',
                 viscosity_residual_mode='positive_time',
                 viscosity_stencil_radius=0.0,
                 viscosity_gate_radius=0.0,
                 viscosity_speed_latent_context=False,
                 viscosity_chord_speed_latent_context=False,
                 viscosity_chord_latent_width=32,
                 viscosity_directed_context=False,
                 viscosity_directed_odd=False,
                 viscosity_directed_odd_scale=1.0,
                 viscosity_handover_phase_context=False,
                 handover_phase_embedding_scale=0.0,
                 viscosity_activation='silu',
                 viscosity_unbounded_logit=False,
                 viscosity_minplus_modes=0,
                 viscosity_minplus_temperature=0.25,
                 viscosity_minplus_scale=1.0,
                 viscosity_minplus_mode='correction',
                 use_viscosity_gain_head=False,
                 viscosity_gain_log_scale=0.6931471805599453,
                 use_causal_handover_head=False,
                 causal_handover_gate='cross_strength',
                 causal_handover_logit_scale=3.0,
                 causal_handover_strength=1.0,
                 causal_handover_phase_context=False,
                 fermat_curve_modes=3,
                 fermat_curve_envelope_detach=False,
                 fermat_curve_quad_points=15,
                 fermat_curve_residual_layers=6,
                 fermat_curve_speed_floor=0.005,
                 fermat_curve_value_weight=1.0,
                 fermat_curve_gate='all'):
        super(MultiNN, self).__init__()
        self.dim = dim
        self.use_task_orient_heads = bool(use_task_orient_heads)
        self.use_orient_head = bool(use_orient_head or self.use_task_orient_heads)
        self.use_safety_head = bool(use_safety_head)
        self.use_pair_safety_heads = bool(use_pair_safety_heads)
        self.pair_safety_mode = str(pair_safety_mode or 'replace')
        if self.pair_safety_mode not in (
                'replace', 'residual', 'constraint_residual'):
            raise ValueError(
                f"pair_safety_mode must be replace, residual, or "
                f"constraint_residual, got "
                f"{self.pair_safety_mode!r}")
        self.pair_safety_residual_scale = float(pair_safety_residual_scale)
        # R58G: optionally bake the orientation cost Phi(start_config) directly
        # into the tau that MPPI descends and training constrains, so the descent
        # basin itself encodes orientation. Default OFF -> byte-identical behavior.
        self.add_orient_to_tau = add_orient_to_tau
        # orient_tau_lambda is read from env so it can be tuned without code edits.
        import os as _os
        self.orient_tau_lambda = float(
            _os.environ.get('R58_ORIENT_TAU_LAMBDA', orient_tau_lambda))
        self.add_safety_to_tau = bool(add_safety_to_tau)
        if self.add_safety_to_tau:
            self.use_safety_head = True
        self.safety_tau_lambda = float(
            _os.environ.get('R58_SAFETY_TAU_LAMBDA', safety_tau_lambda))
        self.safety_tau_lambda_cup = (
            float(_os.environ['R58_SAFETY_TAU_LAMBDA_CUP'])
            if 'R58_SAFETY_TAU_LAMBDA_CUP' in _os.environ else None)
        self.safety_tau_lambda_pour = (
            float(_os.environ['R58_SAFETY_TAU_LAMBDA_POUR'])
            if 'R58_SAFETY_TAU_LAMBDA_POUR' in _os.environ else None)
        self.safety_tau_lambda_cross = (
            float(_os.environ['R58_SAFETY_TAU_LAMBDA_CROSS'])
            if 'R58_SAFETY_TAU_LAMBDA_CROSS' in _os.environ else None)
        self.safety_tau_pair_lambdas = any(v is not None for v in (
            self.safety_tau_lambda_cup,
            self.safety_tau_lambda_pour,
            self.safety_tau_lambda_cross))
        self.safety_tau_query_lambda_cup = (
            float(_os.environ['R58_SAFETY_TAU_QUERY_LAMBDA_CUP'])
            if 'R58_SAFETY_TAU_QUERY_LAMBDA_CUP' in _os.environ else None)
        self.safety_tau_query_lambda_pour = (
            float(_os.environ['R58_SAFETY_TAU_QUERY_LAMBDA_POUR'])
            if 'R58_SAFETY_TAU_QUERY_LAMBDA_POUR' in _os.environ else None)
        self.safety_tau_query_lambda_cross = (
            float(_os.environ['R58_SAFETY_TAU_QUERY_LAMBDA_CROSS'])
            if 'R58_SAFETY_TAU_QUERY_LAMBDA_CROSS' in _os.environ else None)
        self.orient_tau_query_lambda_cup = (
            float(_os.environ['R58_ORIENT_TAU_QUERY_LAMBDA_CUP'])
            if 'R58_ORIENT_TAU_QUERY_LAMBDA_CUP' in _os.environ else None)
        self.orient_tau_query_lambda_pour = (
            float(_os.environ['R58_ORIENT_TAU_QUERY_LAMBDA_POUR'])
            if 'R58_ORIENT_TAU_QUERY_LAMBDA_POUR' in _os.environ else None)
        self.orient_tau_query_lambda_cross = (
            float(_os.environ['R58_ORIENT_TAU_QUERY_LAMBDA_CROSS'])
            if 'R58_ORIENT_TAU_QUERY_LAMBDA_CROSS' in _os.environ else None)
        self.safety_tau_query_lambdas = any(v is not None for v in (
            self.safety_tau_query_lambda_cup,
            self.safety_tau_query_lambda_pour,
            self.safety_tau_query_lambda_cross))
        self.separate_pe_gate = separate_pe_gate
        self.separate_fourier = separate_fourier
        self.separate_encoder_norm = separate_encoder_norm
        self.weight_mode = weight_mode
        self.use_projector = use_projector
        self.use_dual_sym = use_dual_sym
        self.blend_mode = blend_mode
        self.detach_routing_weights = detach_routing_weights
        self.zone_mode = zone_mode
        self.corner_zones = corner_zones
        self.corner_spatial_temp = corner_spatial_temp
        # R62 smooth routing: legacy routing is DISCONTINUOUS in position (hard 1/0
        # outside the overlap band, softmax inside), so tau jumps at the zone edge.
        # These make the weight one continuous sigmoid over position + orientation.
        # Set at construction from the environment and persisted in zone_params, so a
        # checkpoint always evaluates with the routing it was trained under.
        self.smooth_routing = os.environ.get('R62_SMOOTH_ROUTING', '0') == '1'
        self.route_alpha = float(os.environ.get('R62_ROUTE_ALPHA', '0.5'))
        self.route_y_temp = float(os.environ.get('R62_ROUTE_Y_TEMP', '0.3'))
        self.fk_feature_dim = int(fk_feature_dim or 0)
        self.use_task_adapters = bool(use_task_adapters)
        self.handover_k = int(handover_k or 0)
        self.handover_temp = float(handover_temp)
        self.value_arch = value_arch
        self.value_condition = value_condition
        self.finsler_quad_points = int(finsler_quad_points)
        self.finsler_task_heads = bool(finsler_task_heads)
        self.use_cross_slowness_residual = bool(use_cross_slowness_residual)
        self.cross_residual_scale = float(cross_residual_scale)
        self.use_local_slowness_residual = bool(use_local_slowness_residual)
        self.local_residual_scale = float(local_residual_scale)
        self.local_residual_gate = str(local_residual_gate or 'all')
        self.endpoint_task_siamese = bool(endpoint_task_siamese)
        self.endpoint_dual_value_heads = bool(endpoint_dual_value_heads)
        if self.endpoint_dual_value_heads and not self.endpoint_task_siamese:
            raise ValueError(
                "endpoint_dual_value_heads requires endpoint_task_siamese")
        if (self.endpoint_dual_value_heads
                and self.blend_mode not in ('feature', 'slerp')):
            raise ValueError(
                "endpoint_dual_value_heads supports blend_mode=feature or "
                f"slerp, got {self.blend_mode!r}")
        self.task_embedding_scale = float(task_embedding_scale)
        self.global_residual_scale = float(global_residual_scale)
        self.use_speed_head = bool(use_speed_head)
        self.separate_speed_encoder = bool(separate_speed_encoder)
        self.speed_residual_layers = int(speed_residual_layers)
        if self.speed_residual_layers < 0:
            raise ValueError("speed_residual_layers must be non-negative")
        self.speed_correction_width = int(speed_correction_width)
        self.speed_correction_scale = float(speed_correction_scale)
        self.speed_correction_mode = str(speed_correction_mode)
        if self.speed_correction_width < 0:
            raise ValueError("speed_correction_width must be non-negative")
        if self.speed_correction_scale < 0.0:
            raise ValueError("speed_correction_scale must be non-negative")
        if self.speed_correction_mode not in (
                'signed_logit', 'positive_release'):
            raise ValueError(
                "speed_correction_mode must be signed_logit or "
                "positive_release")
        if (self.speed_correction_mode == 'positive_release'
                and self.speed_correction_scale > 1.0):
            raise ValueError(
                "positive_release speed_correction_scale must lie in [0, 1]")
        self.lifted_phase = bool(lifted_phase)
        self.lifted_phase_physical_dim = int(
            lifted_phase_physical_dim or (dim - 1))
        self.lifted_phase_scale = float(lifted_phase_scale)
        self.lifted_phase_width = int(lifted_phase_width)
        self.lifted_phase_logit_scale = float(
            lifted_phase_logit_scale)
        self.lifted_phase_causal = bool(lifted_phase_causal)
        self.lifted_phase_hazard_floor_scale = float(
            lifted_phase_hazard_floor_scale)
        if not 0.0 <= self.lifted_phase_hazard_floor_scale <= 1.0:
            raise ValueError(
                "lifted_phase_hazard_floor_scale must lie in [0, 1]")
        self.cross_hazard_floor_scale = float(cross_hazard_floor_scale)
        if not 0.0 <= self.cross_hazard_floor_scale <= 1.0:
            raise ValueError(
                "cross_hazard_floor_scale must lie in [0, 1]")
        if self.lifted_phase:
            if dim != self.lifted_phase_physical_dim + 1:
                raise ValueError(
                    "lifted_phase requires exactly one coordinate after the "
                    "physical configuration: "
                    f"dim={dim}, physical_dim="
                    f"{self.lifted_phase_physical_dim}")
            if not self.use_speed_head:
                raise ValueError("lifted_phase requires use_speed_head")
            if self.lifted_phase_scale <= 0.0:
                raise ValueError("lifted_phase_scale must be positive")
            if self.lifted_phase_width < 1:
                raise ValueError("lifted_phase_width must be positive")
            if self.lifted_phase_logit_scale <= 0.0:
                raise ValueError(
                    "lifted_phase_logit_scale must be positive")
        elif self.lifted_phase_causal:
            raise ValueError(
                "lifted_phase_causal requires lifted_phase")
        self.use_clearance_head = bool(use_clearance_head)
        self.clearance_scale = float(clearance_scale)
        self.clearance_boundary = float(clearance_boundary)
        self.clearance_speed_floor = float(clearance_speed_floor)
        self.clearance_residual_layers = int(clearance_residual_layers)
        self.clearance_fourier_skip = bool(clearance_fourier_skip)
        if self.use_clearance_head and not (
                self.use_speed_head and self.separate_speed_encoder):
            raise ValueError(
                "use_clearance_head requires use_speed_head and "
                "separate_speed_encoder")
        if self.clearance_scale <= 0.0:
            raise ValueError("clearance_scale must be positive")
        if not 0.0 < self.clearance_boundary < self.clearance_scale:
            raise ValueError(
                "clearance_boundary must lie in (0, clearance_scale)")
        if not 0.0 < self.clearance_speed_floor < 1.0:
            raise ValueError(
                "clearance_speed_floor must lie strictly between 0 and 1")
        if self.clearance_residual_layers < 0:
            raise ValueError("clearance_residual_layers must be non-negative")
        self.couple_speed_latent_to_value = bool(
            couple_speed_latent_to_value)
        self.partial_shared_embedding_dim = int(
            partial_shared_embedding_dim)
        self.pair_slowness_attention_dim = int(
            pair_slowness_attention_dim)
        self.speed_constraint_head_count = int(speed_constraint_heads)
        self.speed_constraint_softmin_temperature = float(
            speed_constraint_softmin_temperature)
        self.speed_constraint_residual_scale = float(
            speed_constraint_residual_scale)
        self.speed_constraint_fourier_skip = bool(
            speed_constraint_fourier_skip)
        self.speed_constraint_consensus_width = int(
            speed_constraint_consensus_width)
        self.speed_constraint_consensus_mode = str(
            speed_constraint_consensus_mode)
        self.speed_constraint_repair_width = int(
            speed_constraint_repair_width)
        if self.partial_shared_embedding_dim < 0:
            raise ValueError(
                "partial_shared_embedding_dim must be non-negative")
        if self.pair_slowness_attention_dim < 0:
            raise ValueError(
                "pair_slowness_attention_dim must be non-negative")
        if (self.speed_constraint_head_count != 0
                and self.speed_constraint_head_count < 2):
            raise ValueError(
                "speed_constraint_heads must be zero or at least two")
        if self.speed_constraint_softmin_temperature <= 0.0:
            raise ValueError(
                "speed_constraint_softmin_temperature must be positive")
        if self.speed_constraint_residual_scale <= 0.0:
            raise ValueError(
                "speed_constraint_residual_scale must be positive")
        if self.speed_constraint_head_count > 0 and not self.use_speed_head:
            raise ValueError(
                "speed_constraint_heads requires use_speed_head")
        if (self.speed_constraint_fourier_skip
                and self.speed_constraint_head_count == 0):
            raise ValueError(
                "speed_constraint_fourier_skip requires constraint heads")
        if self.speed_constraint_consensus_width < 0:
            raise ValueError(
                "speed_constraint_consensus_width must be non-negative")
        if self.speed_constraint_consensus_mode not in (
                'single', 'hurdle', 'corridor_hurdle',
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
            raise ValueError(
                "speed_constraint_consensus_mode must be single, hurdle, "
                "corridor_hurdle, corridor_hurdle_compensated, or "
                "corridor_hurdle_compensated_balanced, or "
                "corridor_hurdle_specificity, or "
                "corridor_hurdle_specificity_odds, or "
                "corridor_hurdle_specificity_orthogonal, or "
                "corridor_hurdle_semantic_veto_min_orthogonal, or "
                "corridor_hurdle_semantic_veto_min_odds, or "
                "corridor_hurdle_semantic_veto_min_odds_residual, or "
                "corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend, or "
                "corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8, or "
                f"one of {ROUTE_CHART_MODES}")
        if (self.speed_constraint_consensus_width > 0
                and self.speed_constraint_head_count == 0):
            raise ValueError(
                "speed constraint consensus requires constraint heads")
        if (self.speed_constraint_consensus_width == 0
                and self.speed_constraint_consensus_mode != 'single'):
            raise ValueError(
                "non-single speed constraint consensus mode requires a "
                "positive consensus width")
        if self.speed_constraint_repair_width < 0:
            raise ValueError(
                "speed_constraint_repair_width must be non-negative")
        if (self.speed_constraint_repair_width > 0
                and self.speed_constraint_head_count == 0):
            raise ValueError(
                "speed constraint repair requires constraint heads")
        if (self.speed_constraint_consensus_mode in (
                'corridor_hurdle', 'corridor_hurdle_compensated',
                'corridor_hurdle_compensated_balanced',
                'corridor_hurdle_specificity',
                'corridor_hurdle_specificity_odds',
                'corridor_hurdle_specificity_orthogonal',
                'corridor_hurdle_semantic_veto_min_orthogonal',
                'corridor_hurdle_semantic_veto_min_odds',
                'corridor_hurdle_semantic_veto_min_odds_residual',
                'corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend',
                'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8',
                *ROUTE_CHART_MODES)
                and self.speed_constraint_repair_width < 1):
            raise ValueError(
                "corridor hurdle consensus requires component repair")
        if (self.speed_constraint_repair_width > 0
                and self.speed_constraint_consensus_width > 0
                and self.speed_constraint_consensus_mode not in (
                    'corridor_hurdle', 'corridor_hurdle_compensated',
                    'corridor_hurdle_compensated_balanced',
                    'corridor_hurdle_specificity',
                    'corridor_hurdle_specificity_odds',
                    'corridor_hurdle_specificity_orthogonal',
                    'corridor_hurdle_semantic_veto_min_orthogonal',
                    'corridor_hurdle_semantic_veto_min_odds',
                    'corridor_hurdle_semantic_veto_min_odds_residual',
                    'corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend',
                    'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8',
                    *ROUTE_CHART_MODES)):
            raise ValueError(
                "speed constraint repair and final consensus are isolated "
                "architectures unless mode is a corridor hurdle")
        if self.separate_speed_encoder and not self.use_speed_head:
            raise ValueError(
                "separate_speed_encoder requires use_speed_head")
        if self.partial_shared_embedding_dim > 0 and not (
                self.endpoint_task_siamese
                and self.use_speed_head
                and self.separate_speed_encoder):
            raise ValueError(
                "partial_shared_embedding_dim requires endpoint_task_siamese, "
                "use_speed_head, and separate_speed_encoder")
        if self.couple_speed_latent_to_value and not (
                self.endpoint_task_siamese
                and self.use_speed_head
                and self.separate_speed_encoder):
            raise ValueError(
                "couple_speed_latent_to_value requires endpoint_task_siamese, "
                "use_speed_head, and separate_speed_encoder")
        if self.pair_slowness_attention_dim > 0 and not (
                self.endpoint_task_siamese
                and self.use_speed_head
                and self.separate_speed_encoder
                and self.value_arch ==
                'bounded_chord_endpoint_eikonal_symmetric'):
            raise ValueError(
                "pair_slowness_attention_dim requires the endpoint-siamese "
                "bounded_chord_endpoint_eikonal_symmetric architecture with "
                "a separate scalar-speed encoder")
        self.global_residual_positive = bool(global_residual_positive)
        # For smooth/global excess architectures, optionally parameterize the
        # one-sided detour gate as a centered softplus.  At the warm-start
        # logit (-6) its physical value is exactly zero, but its derivative is
        # O(1), unlike sigmoid(-6) whose derivative is effectively dead.
        self.global_excess_centered_gate = bool(
            global_excess_centered_gate)
        self.global_excess_centered_signed_gate = bool(
            global_excess_centered_signed_gate)
        self.chord_quad_points = int(chord_quad_points)
        self.chord_quadrature = str(chord_quadrature or 'legendre')
        self.chord_hazard_floor_scale = float(chord_hazard_floor_scale)
        if not 0.0 <= self.chord_hazard_floor_scale <= 1.0:
            raise ValueError(
                "chord_hazard_floor_scale must lie in [0, 1]")
        if self.chord_quadrature not in ('legendre', 'lobatto'):
            raise ValueError(
                "chord_quadrature must be 'legendre' or 'lobatto', got "
                f"{self.chord_quadrature!r}")
        self.use_raw_coordinate_skip = bool(use_raw_coordinate_skip)
        self.speed_floor = float(speed_floor)
        if not 0.0 < self.speed_floor < 1.0:
            raise ValueError("speed_floor must lie strictly between 0 and 1")
        self.endpoint_barrier_speed = float(endpoint_barrier_speed)
        if not self.speed_floor <= self.endpoint_barrier_speed <= 1.0:
            raise ValueError(
                "endpoint_barrier_speed must lie in [speed_floor, 1]")
        self.endpoint_barrier_scale = float(endpoint_barrier_scale)
        if self.endpoint_barrier_scale < 0.0:
            raise ValueError("endpoint_barrier_scale must be non-negative")
        self.endpoint_barrier_radius = float(endpoint_barrier_radius)
        if self.endpoint_barrier_radius <= 0.0:
            raise ValueError("endpoint_barrier_radius must be positive")
        self.endpoint_barrier_power = float(endpoint_barrier_power)
        if self.endpoint_barrier_power < 1.0:
            raise ValueError("endpoint_barrier_power must be at least one")
        self.endpoint_barrier_physical_decode = bool(
            endpoint_barrier_physical_decode)
        self.endpoint_barrier_contrast = bool(endpoint_barrier_contrast)
        if (self.endpoint_barrier_physical_decode
                and self.endpoint_barrier_speed <= self.speed_floor):
            raise ValueError(
                "physical endpoint barrier requires a speed above speed_floor")
        self.global_pair_residual_layers = int(global_pair_residual_layers)
        if self.global_pair_residual_layers < 0:
            raise ValueError("global_pair_residual_layers must be non-negative")
        self.global_pair_residual_activation = str(
            global_pair_residual_activation)
        if self.global_pair_residual_activation not in ('silu', 'maxmin'):
            raise ValueError(
                "global_pair_residual_activation must be 'silu' or "
                f"'maxmin', got {self.global_pair_residual_activation!r}")
        self.use_viscosity_excess_head = bool(
            use_viscosity_excess_head
            or self.value_arch == 'bounded_chord_neural_viscosity_symmetric')
        self.viscosity_excess_gate = str(viscosity_excess_gate)
        if self.viscosity_excess_gate not in (
                'all', 'cross_strength', 'cross_transverse',
                'cross_transverse_saturated', 'cross_chord_evidence',
                'cross_chord_sqrt_evidence',
                'cross_chord_saturated_evidence', 'cross_certain'):
            raise ValueError(
                f"unknown viscosity_excess_gate="
                f"{self.viscosity_excess_gate!r}")
        self.viscosity_residual_mode = str(viscosity_residual_mode)
        if self.viscosity_residual_mode not in (
                'positive_time', 'signed_fraction', 'causal_log_time'):
            raise ValueError(
                f"unknown viscosity_residual_mode="
                f"{self.viscosity_residual_mode!r}")
        self.viscosity_stencil_radius = float(viscosity_stencil_radius)
        if not 0.0 <= self.viscosity_stencil_radius <= 0.25:
            raise ValueError(
                "viscosity_stencil_radius must lie in [0, 0.25]")
        self.viscosity_gate_radius = float(viscosity_gate_radius)
        if not 0.0 <= self.viscosity_gate_radius <= 0.25:
            raise ValueError(
                "viscosity_gate_radius must lie in [0, 0.25]")
        if (self.viscosity_stencil_radius > 0.0
                and self.viscosity_gate_radius > 0.0
                and abs(self.viscosity_stencil_radius
                        - self.viscosity_gate_radius) > 1e-9):
            raise ValueError(
                "viscosity_stencil_radius and viscosity_gate_radius must "
                "match when both are enabled")
        if (self.viscosity_excess_gate in (
                'cross_transverse', 'cross_transverse_saturated')
                and max(self.viscosity_stencil_radius,
                        self.viscosity_gate_radius) <= 0.0):
            raise ValueError(
                "cross_transverse viscosity gating requires a positive "
                "stencil or gate radius")
        self.viscosity_speed_latent_context = bool(
            viscosity_speed_latent_context)
        if self.viscosity_speed_latent_context and not (
                self.use_viscosity_excess_head
                and self.use_speed_head
                and self.separate_speed_encoder):
            raise ValueError(
                "viscosity_speed_latent_context requires the viscosity "
                "branch and a separate scalar-speed encoder")
        self.viscosity_chord_speed_latent_context = bool(
            viscosity_chord_speed_latent_context)
        self.viscosity_chord_latent_width = int(
            viscosity_chord_latent_width)
        if self.viscosity_chord_latent_width < 1:
            raise ValueError(
                "viscosity_chord_latent_width must be positive")
        if self.viscosity_chord_speed_latent_context and not (
                self.use_viscosity_excess_head
                and self.use_speed_head
                and self.separate_speed_encoder):
            raise ValueError(
                "viscosity_chord_speed_latent_context requires the "
                "viscosity branch and a separate scalar-speed encoder")
        self.viscosity_directed_context = bool(
            viscosity_directed_context)
        self.viscosity_directed_odd = bool(viscosity_directed_odd)
        self.viscosity_directed_odd_scale = float(
            viscosity_directed_odd_scale)
        if self.viscosity_directed_odd_scale < 0.0:
            raise ValueError(
                "viscosity_directed_odd_scale must be non-negative")
        if self.viscosity_directed_context and not (
                self.use_viscosity_excess_head
                and self.viscosity_residual_mode in (
                    'signed_fraction', 'causal_log_time')):
            raise ValueError(
                "viscosity_directed_context requires a signed_fraction or "
                "causal_log_time viscosity branch")
        if self.viscosity_directed_odd and not (
                self.viscosity_directed_context
                and self.viscosity_residual_mode == 'signed_fraction'):
            raise ValueError(
                "odd directed viscosity requires directed signed_fraction")
        if self.viscosity_directed_odd and (
                self.viscosity_speed_latent_context
                or self.viscosity_chord_speed_latent_context
                or self.viscosity_stencil_radius > 0.0):
            raise ValueError(
                "odd directed viscosity expects the symmetric scalar-chord "
                "context followed directly by its directed coordinates")
        self.viscosity_handover_phase_context = bool(
            viscosity_handover_phase_context)
        if (self.viscosity_handover_phase_context
                and not self.viscosity_directed_context):
            raise ValueError(
                "viscosity_handover_phase_context requires directed context")
        self.handover_phase_embedding_scale = float(
            handover_phase_embedding_scale)
        if self.handover_phase_embedding_scale < 0.0:
            raise ValueError(
                "handover_phase_embedding_scale must be non-negative")
        self.viscosity_activation = str(viscosity_activation)
        if self.viscosity_activation not in ('silu', 'maxmin'):
            raise ValueError(
                "viscosity_activation must be 'silu' or 'maxmin', got "
                f"{self.viscosity_activation!r}")
        self.viscosity_unbounded_logit = bool(viscosity_unbounded_logit)
        self.viscosity_minplus_modes = int(viscosity_minplus_modes)
        if self.viscosity_minplus_modes not in (0, 2, 3, 4):
            raise ValueError(
                "viscosity_minplus_modes must be 0, 2, 3, or 4")
        self.viscosity_minplus_temperature = float(
            viscosity_minplus_temperature)
        if self.viscosity_minplus_temperature <= 0.0:
            raise ValueError(
                "viscosity_minplus_temperature must be positive")
        self.viscosity_minplus_scale = float(viscosity_minplus_scale)
        self.viscosity_minplus_mode = str(viscosity_minplus_mode)
        if self.viscosity_minplus_mode not in (
                'correction', 'tau_envelope'):
            raise ValueError(
                "viscosity_minplus_mode must be 'correction' or "
                "'tau_envelope'")
        if self.viscosity_minplus_scale < 0.0:
            raise ValueError("viscosity_minplus_scale must be non-negative")
        if self.viscosity_minplus_modes and not (
                self.use_viscosity_excess_head
                and self.viscosity_residual_mode in (
                    'positive_time', 'signed_fraction')):
            raise ValueError(
                "min-plus viscosity modes require the positive-time or "
                "signed-fraction scalar-viscosity branch")
        self.use_viscosity_gain_head = bool(use_viscosity_gain_head)
        self.viscosity_gain_log_scale = float(viscosity_gain_log_scale)
        if self.viscosity_gain_log_scale <= 0.0:
            raise ValueError("viscosity_gain_log_scale must be positive")
        if self.use_viscosity_gain_head and not (
                self.use_viscosity_excess_head
                and self.viscosity_residual_mode == 'signed_fraction'):
            raise ValueError(
                "viscosity gain requires the signed scalar-viscosity branch")
        self.use_causal_handover_head = bool(use_causal_handover_head)
        self.causal_handover_gate = str(causal_handover_gate)
        if self.causal_handover_gate not in (
                'cross_strength', 'cross_certain',
                'cross_chord_evidence', 'cross_chord_sqrt_evidence'):
            raise ValueError(
                f"unknown causal_handover_gate="
                f"{self.causal_handover_gate!r}")
        self.causal_handover_logit_scale = float(
            causal_handover_logit_scale)
        if self.causal_handover_logit_scale <= 0.0:
            raise ValueError("causal_handover_logit_scale must be positive")
        self.causal_handover_strength = float(causal_handover_strength)
        if self.causal_handover_strength < 0.0:
            raise ValueError("causal_handover_strength must be non-negative")
        self.causal_handover_phase_context = bool(
            causal_handover_phase_context)
        if self.use_causal_handover_head and not (
                self.use_viscosity_excess_head
                and self.viscosity_residual_mode == 'signed_fraction'):
            raise ValueError(
                "causal handover composition requires the inherited signed "
                "scalar-viscosity branch")
        if self.use_causal_handover_head and (
                self.viscosity_directed_context
                or self.viscosity_speed_latent_context
                or self.viscosity_chord_speed_latent_context
                or self.viscosity_stencil_radius > 0.0):
            raise ValueError(
                "causal handover composition expects the inherited signed "
                "branch to use only its symmetric scalar-chord context")
        self.fermat_curve_modes = int(fermat_curve_modes)
        if self.fermat_curve_modes < 1:
            raise ValueError("fermat_curve_modes must be positive")
        self.fermat_curve_envelope_detach = bool(
            fermat_curve_envelope_detach)
        self.fermat_curve_quad_points = int(fermat_curve_quad_points)
        if self.fermat_curve_quad_points not in (3, 5, 7, 15):
            raise ValueError(
                "fermat_curve_quad_points must be 3, 5, 7, or 15")
        self.fermat_curve_residual_layers = int(
            fermat_curve_residual_layers)
        if self.fermat_curve_residual_layers < 0:
            raise ValueError(
                "fermat_curve_residual_layers must be non-negative")
        self.fermat_curve_speed_floor = float(fermat_curve_speed_floor)
        if not 0.0 < self.fermat_curve_speed_floor < self.speed_floor:
            raise ValueError(
                "fermat_curve_speed_floor must lie in (0, speed_floor)")
        self.fermat_curve_value_weight = float(fermat_curve_value_weight)
        if not 0.0 <= self.fermat_curve_value_weight <= 1.0:
            raise ValueError(
                "fermat_curve_value_weight must lie in [0, 1]")
        self.fermat_curve_gate = str(fermat_curve_gate)
        if self.fermat_curve_gate not in (
                'all', 'cross_strength', 'cross_certain'):
            raise ValueError(
                "fermat_curve_gate must be all, cross_strength, or "
                "cross_certain")
        if self.use_raw_coordinate_skip:
            # Fourier features are periodic and can alias distinct locations
            # inside finite joint limits.  A single learned raw-q residual
            # preserves absolute configuration identity without introducing a
            # second encoder or any FK/EE input.
            self.raw_coordinate_projection = Linear(dim, h_size)
        if self.local_residual_gate not in ('all', 'cross_strength'):
            raise ValueError(
                f"unknown local_residual_gate={self.local_residual_gate!r}")
        if self.value_arch not in (
                'metric', 'asym_goal', 'local_finsler', 'global_residual',
                'global_residual_symmetric', 'bounded_chord_symmetric',
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
                'bounded_chord_separate_smooth_excess_symmetric',
                'bounded_chord_separate_global_excess_symmetric',
                'bounded_chord_neural_viscosity_symmetric',
                'bounded_chord_signed_clearance_symmetric',
                'bounded_chord_clearance_eikonal_symmetric',
                'bounded_chord_clearance_state_symmetric',
                'endpoint_eikonal_residual_symmetric',
                'local_endpoint_eikonal_residual_symmetric',
                'bounded_chord_profile_symmetric',
                'fermat_waypoint_symmetric', 'fermat_curve_symmetric',
                'fermat_curve_equivariant'):
            raise ValueError(f"unknown value_arch: {self.value_arch}")
        if (self.couple_speed_latent_to_value
                and self.value_arch not in (
                    'bounded_chord_symmetric',
                    'bounded_chord_fermat_curve_symmetric',
                    'bounded_chord_endpoint_fermat_curve_symmetric',
                    'bounded_chord_endpoint_speed_fermat_curve_symmetric',
                    'bounded_chord_log_symmetric',
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
                    'bounded_chord_separate_smooth_excess_symmetric',
                    'bounded_chord_separate_global_excess_symmetric',
                    'bounded_chord_neural_viscosity_symmetric',
                    'bounded_chord_signed_clearance_symmetric',
                    'bounded_chord_clearance_eikonal_symmetric',
                    'bounded_chord_clearance_state_symmetric',
                    'bounded_chord_profile_symmetric')):
            raise ValueError(
                "speed-latent value coupling requires a learned "
                "bounded-chord value architecture")
        if (self.value_arch in (
                'global_residual', 'global_residual_symmetric',
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
                'bounded_chord_separate_smooth_excess_symmetric',
                'bounded_chord_separate_global_excess_symmetric',
                'bounded_chord_neural_viscosity_symmetric',
                'bounded_chord_signed_clearance_symmetric',
                'bounded_chord_clearance_eikonal_symmetric',
                'bounded_chord_clearance_state_symmetric',
                'endpoint_eikonal_residual_symmetric',
                'local_endpoint_eikonal_residual_symmetric',
                'bounded_chord_profile_symmetric',
                'fermat_waypoint_symmetric',
                'fermat_curve_symmetric',
                'fermat_curve_equivariant')
                and not self.endpoint_task_siamese):
            raise ValueError(
                f"{self.value_arch} requires endpoint_task_siamese")
        if self.value_arch in (
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
                'bounded_chord_separate_smooth_excess_symmetric',
                'bounded_chord_separate_global_excess_symmetric',
                'bounded_chord_neural_viscosity_symmetric',
                'bounded_chord_signed_clearance_symmetric',
                'bounded_chord_clearance_eikonal_symmetric',
                'bounded_chord_clearance_state_symmetric',
                'endpoint_eikonal_residual_symmetric',
                'local_endpoint_eikonal_residual_symmetric',
                'bounded_chord_profile_symmetric',
                'fermat_waypoint_symmetric',
                'fermat_curve_symmetric',
                'fermat_curve_equivariant'):
            if not self.use_speed_head:
                raise ValueError(
                    f"{self.value_arch} requires use_speed_head")
            if (self.value_arch in (
                    'bounded_chord_signed_clearance_symmetric',
                    'bounded_chord_clearance_eikonal_symmetric',
                    'bounded_chord_clearance_state_symmetric')
                    and not self.use_clearance_head):
                raise ValueError(
                    f"{self.value_arch} requires "
                    "use_clearance_head")
            if self.value_condition != 'none':
                raise ValueError(
                    f"{self.value_arch} requires value_condition=none")
            if self.chord_quad_points not in (2, 3, 5, 7, 15):
                raise ValueError(
                    "chord_quad_points must be 2, 3, 5, 7, or 15")
            if self.chord_quadrature == 'legendre':
                # Open Gauss--Legendre rule: historical behavior.  It is a
                # high-order integral estimate, but deliberately excludes the
                # two endpoints.
                nodes, weights = np.polynomial.legendre.leggauss(
                    self.chord_quad_points)
            else:
                # Closed Gauss--Lobatto rule.  For the isotropic Eikonal PDE,
                # the derivative of T(q, g) is controlled by the local speed
                # F(q), so an upper-bound chord integral must actually include
                # q (and, by reversibility, g).  The interior nodes are the
                # roots of P'_{n-1}; w_i=2/[n(n-1)P_{n-1}(x_i)^2].
                degree = self.chord_quad_points - 1
                poly = np.polynomial.legendre.Legendre.basis(degree)
                interior = poly.deriv().roots()
                nodes = np.concatenate(([-1.0], interior, [1.0]))
                weights = 2.0 / (
                    self.chord_quad_points * degree * poly(nodes) ** 2)
            self.register_buffer(
                'chord_nodes',
                torch.as_tensor(0.5 * (nodes + 1.0), dtype=torch.float32),
                persistent=False)
            self.register_buffer(
                'chord_weights',
                torch.as_tensor(0.5 * weights, dtype=torch.float32),
                persistent=False)
        if self.value_arch in (
                'bounded_chord_fermat_curve_symmetric',
                'bounded_chord_endpoint_fermat_curve_symmetric',
                'bounded_chord_endpoint_speed_fermat_curve_symmetric'):
            # Keep the proven bounded-chord rule unchanged and use a denser,
            # independent rule only for the one learned curve.  The curve
            # contributes C_curve-C_straight under this same rule, so a zero
            # bend is an exact E260 no-op even when the two rules differ.
            curve_nodes, curve_weights = np.polynomial.legendre.leggauss(
                self.fermat_curve_quad_points)
            self.register_buffer(
                'fermat_curve_nodes',
                torch.as_tensor(
                    0.5 * (curve_nodes + 1.0), dtype=torch.float32),
                persistent=False)
            self.register_buffer(
                'fermat_curve_weights',
                torch.as_tensor(0.5 * curve_weights, dtype=torch.float32),
                persistent=False)
        if self.value_condition not in (
                'none', 'task', 'task_scene', 'task_scene_dist',
                'task_scene_pair', 'task_scene_pair_dist'):
            raise ValueError(f"unknown value_condition: {self.value_condition}")
        if self.finsler_quad_points not in (1, 2, 3, 5, 9):
            raise ValueError(
                f"finsler_quad_points must be 1, 2, 3, 5, or 9, got {self.finsler_quad_points}")

        if separate_fourier:
            # Each subnet gets its own Fourier basis matrix (different random init)
            # B is (B_freqs, dim), stored transposed as (dim, B_freqs).
            self.B = B.T.to(device)  # shared B kept for backward compat
            self.B_cup = B.T.to(device)
            # Generate a DIFFERENT random B for pouring subnet using fixed seed
            # so it's reproducible but distinct from the cupping B
            rng_state = torch.random.get_rng_state()
            torch.manual_seed(42)
            B_pour_raw = B.std().item() * torch.randn_like(B)
            torch.random.set_rng_state(rng_state)  # restore RNG
            self.B_pour = B_pour_raw.T.to(device)
            print(f"Separate Fourier: B_cup {self.B_cup.shape}, B_pour {self.B_pour.shape}")
        else:
            self.B = B.T.to(device)
            print(B.shape)
        input_size = B.shape[0]

        self.scale = 10
        self.act = torch.nn.Softplus(beta=self.scale)
        self.actout = Sigmoid_out()

        self.nl1 = nl1

        # --- encoder_norm: shared or per-head ---
        if separate_encoder_norm:
            self.cup_encoder_norm = InstanceNorm1d(h_size)
            self.pour_encoder_norm = InstanceNorm1d(h_size)
            # Keep shared for backward compat (unused in forward when separate)
            self.encoder_norm = InstanceNorm1d(h_size)
        else:
            self.encoder_norm = InstanceNorm1d(h_size)

        # --- Cupping encoder head ---
        # NOTE: encoder[0] is a dead-weight placeholder (matching single-task structure).
        # The actual forward pass starts at encoder[1]. pe_gate is the real input transform.
        self.cup_encoder = torch.nn.ModuleList()
        self.cup_encoder.append(Linear(h_size, h_size))  # placeholder [0], not used in forward
        for i in range(0, 3 * self.nl1):
            self.cup_encoder.append(Linear(h_size, h_size))
        self.cup_encoder.append(Linear(h_size, h_size))  # final layer

        self.cup_gate = torch.nn.ModuleList()
        for i in range(self.nl1):
            self.cup_gate.append(Linear(1, 1))

        # --- Pouring encoder head ---
        self.pour_encoder = torch.nn.ModuleList()
        self.pour_encoder.append(Linear(h_size, h_size))  # placeholder [0], not used in forward
        for i in range(0, 3 * self.nl1):
            self.pour_encoder.append(Linear(h_size, h_size))
        self.pour_encoder.append(Linear(h_size, h_size))  # final layer

        self.pour_gate = torch.nn.ModuleList()
        for i in range(self.nl1):
            self.pour_gate.append(Linear(1, 1))

        # VNext: one shared geometric trunk plus small task residual adapters.
        # This keeps cup/pour in one latent geometry while allowing task-specific
        # deformation, instead of making cross a direct interpolation of two
        # unrelated fields.
        if self.use_task_adapters:
            self.shared_encoder = torch.nn.ModuleList()
            self.shared_encoder.append(Linear(h_size, h_size))
            for i in range(0, 3 * self.nl1):
                self.shared_encoder.append(Linear(h_size, h_size))
            self.shared_encoder.append(Linear(h_size, h_size))
            self.shared_gate = torch.nn.ModuleList()
            for i in range(self.nl1):
                self.shared_gate.append(Linear(1, 1))
            self.cup_adapter = torch.nn.Sequential(
                Linear(h_size, h_size),
                torch.nn.SiLU(),
                Linear(h_size, h_size),
            )
            self.pour_adapter = torch.nn.Sequential(
                Linear(h_size, h_size),
                torch.nn.SiLU(),
                Linear(h_size, h_size),
            )
            if self.handover_k > 0:
                self.handover_cup = torch.nn.Parameter(
                    0.02 * torch.randn(self.handover_k, h_size, device=device))
                self.handover_pour = torch.nn.Parameter(
                    0.02 * torch.randn(self.handover_k, h_size, device=device))
                self.handover_bridge = torch.nn.Parameter(
                    torch.zeros(self.handover_k, device=device))

        # One shared endpoint encoder.  With value_condition="task" it is
        # conditioned by the local task token.  With value_condition="none"
        # it is a literal single scalar field: no cup/pour token or task
        # parameters are present in the value model.
        if self.endpoint_task_siamese:
            if self.value_condition != 'none':
                self.endpoint_task_embedding = torch.nn.Sequential(
                    Linear(2, h_size),
                    torch.nn.SiLU(),
                    Linear(h_size, h_size),
                )
            self.siamese_encoder = torch.nn.ModuleList()
            self.siamese_encoder.append(Linear(h_size, h_size))
            for _ in range(0, 3 * self.nl1):
                self.siamese_encoder.append(Linear(h_size, h_size))
            self.siamese_encoder.append(Linear(h_size, h_size))
            self.siamese_gate = torch.nn.ModuleList()
            for _ in range(self.nl1):
                self.siamese_gate.append(Linear(1, 1))

            # F(q) is the only supervised scalar in the speed-only
            # formulation.  By default it shares the smooth value encoder.
            # Optionally give it a multiscale internal encoder: obstacle
            # clearance has a thin, high-frequency boundary whereas the
            # viscosity value is globally smoother.  Both fields remain one
            # coupled network and one checkpoint.
            if self.use_speed_head:
                speed_h_size = h_size
                if self.separate_speed_encoder:
                    if speed_B is None:
                        raise ValueError(
                            "separate_speed_encoder requires speed_B")
                    speed_B = torch.as_tensor(
                        speed_B, dtype=B.dtype, device=device)
                    if speed_B.ndim != 2 or speed_B.shape[1] != dim:
                        raise ValueError(
                            "speed_B must have shape (n_freqs, dim), got "
                            f"{tuple(speed_B.shape)}")
                    self.register_buffer(
                        'B_speed', speed_B.T.contiguous())
                    speed_h_size = 2 * speed_B.shape[0]
                    self.speed_encoder_norm = InstanceNorm1d(speed_h_size)
                    self.speed_pe_gate = torch.nn.ModuleList([
                        Linear(speed_h_size, speed_h_size),
                        Linear(speed_h_size, speed_h_size),
                    ])
                    self.speed_encoder = torch.nn.ModuleList()
                    self.speed_encoder.append(
                        Linear(speed_h_size, speed_h_size))
                    for _ in range(0, 3 * self.nl1):
                        self.speed_encoder.append(
                            Linear(speed_h_size, speed_h_size))
                    self.speed_encoder.append(
                        Linear(speed_h_size, speed_h_size))
                    self.speed_gate = torch.nn.ModuleList([
                        Linear(1, 1) for _ in range(self.nl1)
                    ])
                self.speed_feature_dim = speed_h_size
                self.speed_residual_blocks = torch.nn.ModuleList([
                    PairResidualBlock(speed_h_size)
                    for _ in range(self.speed_residual_layers)
                ])
                self.speed_head = torch.nn.Sequential(
                    Linear(speed_h_size, speed_h_size // 2),
                    torch.nn.SiLU(),
                    Linear(speed_h_size // 2, 1),
                )
                if self.speed_constraint_consensus_mode in ROUTE_CHART_ANCHOR_MODES:
                    if not self.separate_speed_encoder:
                        raise ValueError(
                            "semantic anchor rescue requires a separate "
                            "speed encoder")
                    # Phase-CJ deploys the formal leader as a second frozen
                    # q-only scalar proposal.  It shares only the immutable
                    # Fourier basis and affine output convention with the CH
                    # field; all learned encoder and head tensors are private.
                    self.speed_constraint_anchor_pe_gate = copy.deepcopy(
                        self.speed_pe_gate)
                    self.speed_constraint_anchor_encoder = copy.deepcopy(
                        self.speed_encoder)
                    self.speed_constraint_anchor_gate = copy.deepcopy(
                        self.speed_gate)
                    self.speed_constraint_anchor_residual_blocks = (
                        copy.deepcopy(self.speed_residual_blocks))
                    self.speed_constraint_anchor_head = copy.deepcopy(
                        self.speed_head)
                    for module in (
                            self.speed_constraint_anchor_pe_gate,
                            self.speed_constraint_anchor_encoder,
                            self.speed_constraint_anchor_gate,
                            self.speed_constraint_anchor_residual_blocks,
                            self.speed_constraint_anchor_head):
                        module.requires_grad_(False)
                    if self.speed_constraint_consensus_mode == (
                            ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_REPLICA_MEDIAN_ANCHOR_RESCUE_CONSENSUS_MODE):
                        anchor_q_feature_dim = self.dim * (
                            1 + 2 * len(
                                ROUTE_CHART_FOURIER_RISK_FREQUENCIES))
                        self.register_buffer(
                            'speed_constraint_anchor_reliability_q_mean',
                            torch.zeros(self.dim))
                        self.register_buffer(
                            'speed_constraint_anchor_reliability_q_std',
                            torch.ones(self.dim))
                        self.speed_constraint_anchor_reliability_replicas = (
                            torch.nn.ModuleList([
                                AnchorPrivateReliabilityReplica(
                                    speed_h_size, anchor_q_feature_dim, 8)
                                for _ in range(
                                    ANCHOR_RELIABILITY_REPLICA_COUNT)
                            ]))
                        self.speed_constraint_anchor_reliability_replicas.requires_grad_(
                            False)
                    if self.speed_constraint_consensus_mode == (
                            ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_FULLSEMANTIC_ANCHOR_RESCUE_CONSENSUS_MODE):
                        # Phase-DF retains three independently trained
                        # four-head members.  Their deterministic FK/relation
                        # inputs are registered below as immutable buffers;
                        # the CuRobo kinematic evaluator itself is attached at
                        # runtime and is never serialized into the checkpoint.
                        self.speed_constraint_fullsemantic_replicas = (
                            torch.nn.ModuleList([
                                FullSemanticRelationReplica(
                                    speed_h_size, 8)
                                for _ in range(
                                    FULLSEMANTIC_REPLICA_COUNT)
                            ]))
                        self.speed_constraint_fullsemantic_replicas.requires_grad_(
                            False)
                        self.register_buffer(
                            'speed_constraint_fullsemantic_q_mean',
                            torch.zeros(self.dim))
                        self.register_buffer(
                            'speed_constraint_fullsemantic_q_std',
                            torch.ones(self.dim))
                        self.register_buffer(
                            'speed_constraint_fullsemantic_fk8_mean',
                            torch.zeros(8))
                        self.register_buffer(
                            'speed_constraint_fullsemantic_fk8_std',
                            torch.ones(8))
                        self.register_buffer(
                            'speed_constraint_fullsemantic_q_center',
                            torch.zeros(self.dim))
                        self.register_buffer(
                            'speed_constraint_fullsemantic_q_range',
                            torch.ones(self.dim))
                        self.register_buffer(
                            'speed_constraint_fullsemantic_temperatures',
                            torch.as_tensor(
                                FULLSEMANTIC_ATTENTION_TEMPERATURES_M,
                                dtype=B.dtype, device=device))
                        self.register_buffer(
                            'speed_constraint_fullsemantic_relation_scale',
                            torch.as_tensor(
                                0.01, dtype=B.dtype, device=device))
                        self.register_buffer(
                            'speed_constraint_fullsemantic_component_boundary',
                            torch.as_tensor(
                                0.0, dtype=B.dtype, device=device))
                        self.register_buffer(
                            'speed_constraint_fullsemantic_radii',
                            torch.zeros(
                                FULLSEMANTIC_SPHERE_COUNT,
                                dtype=B.dtype, device=device))
                        self.register_buffer(
                            'speed_constraint_fullsemantic_cuboid_centers',
                            torch.zeros(2, 3, dtype=B.dtype, device=device))
                        self.register_buffer(
                            'speed_constraint_fullsemantic_cuboid_half',
                            torch.zeros(2, 3, dtype=B.dtype, device=device))
                        self.register_buffer(
                            'speed_constraint_fullsemantic_self_pairs',
                            torch.zeros(
                                FULLSEMANTIC_SELF_RELATION_COUNT, 2,
                                dtype=torch.long, device=device))
                        object.__setattr__(
                            self,
                            '_speed_constraint_fullsemantic_robot_model',
                            None)
                    if self.speed_constraint_consensus_mode in (
                            ROUTE_CHART_ANALYTIC_CONTINUOUS_MODES):
                        # Phase-DV confirmed a deterministic four-margin gate:
                        # exact obstacle/self minima plus active cup/pour
                        # margins.  Only immutable geometry buffers are
                        # serialized; there are no learned semantic modules.
                        self.register_buffer(
                            'speed_constraint_fullsemantic_q_center',
                            torch.zeros(self.dim))
                        self.register_buffer(
                            'speed_constraint_fullsemantic_q_range',
                            torch.ones(self.dim))
                        self.register_buffer(
                            'speed_constraint_fullsemantic_component_boundary',
                            torch.as_tensor(
                                0.0, dtype=B.dtype, device=device))
                        self.register_buffer(
                            'speed_constraint_analytic_continuous_tau',
                            torch.as_tensor(
                                ANALYTIC_CONTINUOUS_PERMISSION_TAU,
                                dtype=B.dtype, device=device))
                        self.register_buffer(
                            'speed_constraint_fullsemantic_radii',
                            torch.zeros(
                                FULLSEMANTIC_SPHERE_COUNT,
                                dtype=B.dtype, device=device))
                        self.register_buffer(
                            'speed_constraint_fullsemantic_cuboid_centers',
                            torch.zeros(2, 3, dtype=B.dtype, device=device))
                        self.register_buffer(
                            'speed_constraint_fullsemantic_cuboid_half',
                            torch.zeros(2, 3, dtype=B.dtype, device=device))
                        self.register_buffer(
                            'speed_constraint_fullsemantic_self_pairs',
                            torch.zeros(
                                FULLSEMANTIC_SELF_RELATION_COUNT, 2,
                                dtype=torch.long, device=device))
                        object.__setattr__(
                            self,
                            '_speed_constraint_fullsemantic_robot_model',
                            None)
                # Optional semantic factorization of the same q-only scalar
                # coefficient.  The three registered heads predict signed
                # residuals for safety, active-cup, and active-pour clearance.
                # Their analytic normalized soft minimum is still the only
                # deployed F(q).  Zero final layers make the extension exactly
                # idempotent at a warm start: every component equals F_base.
                constraint_input_size = speed_h_size * (
                    2 if self.speed_constraint_fourier_skip else 1)
                self.speed_constraint_heads = torch.nn.ModuleList([
                    torch.nn.Sequential(
                        Linear(constraint_input_size, speed_h_size // 2),
                        torch.nn.SiLU(),
                        Linear(speed_h_size // 2, 1),
                    )
                    for _ in range(self.speed_constraint_head_count)
                ])
                if self.speed_constraint_consensus_width > 0:
                    consensus_input_size = (
                        speed_h_size + self.speed_constraint_head_count + 2)
                    if self.speed_constraint_consensus_mode == 'single':
                        self.speed_constraint_consensus_head = (
                            torch.nn.Sequential(
                                Linear(
                                    consensus_input_size,
                                    self.speed_constraint_consensus_width),
                                torch.nn.SiLU(),
                                Linear(
                                    self.speed_constraint_consensus_width, 1),
                            ))
                    else:
                        # Zero-inflated consensus: a shared q-only embedding
                        # feeds private release and magnitude tails.  Their
                        # product is still one alpha and therefore one F(q).
                        self.speed_constraint_consensus_trunk = (
                            torch.nn.Sequential(
                                Linear(
                                    consensus_input_size,
                                    self.speed_constraint_consensus_width),
                                torch.nn.SiLU(),
                            ))
                        consensus_heads = {
                            'release': Linear(
                                self.speed_constraint_consensus_width, 1),
                        }
                        if self.speed_constraint_consensus_mode in (
                                'corridor_hurdle_specificity',
                                'corridor_hurdle_specificity_odds',
                                'corridor_hurdle_specificity_orthogonal',
                                'corridor_hurdle_semantic_veto_min_orthogonal',
                                'corridor_hurdle_semantic_veto_min_odds',
                                'corridor_hurdle_semantic_veto_min_odds_residual',
                                'corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend',
                                'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8',
                                *ROUTE_CHART_MODES):
                            specificity_outputs = (
                                self.speed_constraint_head_count
                                if self.speed_constraint_consensus_mode
                                in (
                                    'corridor_hurdle_semantic_veto_min_orthogonal',
                                    'corridor_hurdle_semantic_veto_min_odds',
                                    'corridor_hurdle_semantic_veto_min_odds_residual',
                                    'corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend',
                                    'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8',
                                    *ROUTE_CHART_MODES)
                                else 1)
                            consensus_heads['specificity'] = Linear(
                                self.speed_constraint_consensus_width,
                                specificity_outputs)
                        consensus_heads['magnitude'] = Linear(
                            self.speed_constraint_consensus_width, 1)
                        self.speed_constraint_consensus_heads = (
                            torch.nn.ModuleDict(consensus_heads))
                        if self.speed_constraint_consensus_mode in (
                                'corridor_hurdle_semantic_veto_min_odds_residual',
                                'corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend',
                                'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8',
                                *ROUTE_CHART_MODES):
                            # A private nonlinear tail refines only the
                            # conditional magnitude logit.  Its zero-final
                            # residual makes an AQ checkpoint an exact
                            # functional warm start while leaving the shared
                            # corridor representation and every semantic head
                            # untouched.
                            self.speed_constraint_consensus_magnitude_residual = (
                                torch.nn.Sequential(
                                    Linear(
                                        self.speed_constraint_consensus_width,
                                        64),
                                    torch.nn.SiLU(),
                                    Linear(64, 1),
                                ))
                        if (self.speed_constraint_consensus_mode
                                == 'corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend'):
                            # A second private head selects how much evidence
                            # to borrow from the normalized all-head logit
                            # soft-min.  Its zero-final initialization gives
                            # tanh(delta)=0, exactly reproducing Phase AR's
                            # hard weakest-head field before optimization.
                            self.speed_constraint_consensus_evidence_blend_residual = (
                                torch.nn.Sequential(
                                    Linear(
                                        self.speed_constraint_consensus_width,
                                        64),
                                    torch.nn.SiLU(),
                                    Linear(64, 1),
                                ))
                        if (self.speed_constraint_consensus_mode
                                == 'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8'):
                            # One bounded signed logit correction per semantic
                            # permission head.  The shared q-only embedding is
                            # the query; the three semantic permissions retain
                            # their identities as fixed keys/values.  A zero
                            # final layer exactly recovers Phase AR, while the
                            # hard weakest-head fusion still emits one alpha
                            # and one deployed scalar F(q).
                            self.speed_constraint_consensus_head_logit_residual = (
                                torch.nn.Sequential(
                                    Linear(
                                        self.speed_constraint_consensus_width,
                                        64),
                                    torch.nn.SiLU(),
                                    Linear(
                                        64,
                                        self.speed_constraint_head_count),
                                ))
                        if self.speed_constraint_consensus_mode in ROUTE_CHART_MODES:
                            # Four smooth q-space configuration charts share
                            # the frozen corridor embedding while retaining
                            # private bounded release-logit tails.  Their fixed
                            # gate is fitted only from training support states
                            # and is persisted as ordinary checkpoint buffers.
                            if (self.speed_constraint_consensus_mode
                                    in ROUTE_CHART_PARTIAL_SHARED_MODES):
                                route_q_feature_dim = self.dim * (
                                    1 + 2 * len(
                                        ROUTE_CHART_FOURIER_RISK_FREQUENCIES))
                                self.register_buffer(
                                    'speed_constraint_route_fourier_frequencies',
                                    torch.tensor(
                                        ROUTE_CHART_FOURIER_RISK_FREQUENCIES,
                                        dtype=torch.float32))
                                self.speed_constraint_route_fourier_branch = (
                                    torch.nn.Sequential(
                                        Linear(route_q_feature_dim, 128),
                                        torch.nn.SiLU(),
                                        Linear(128, 64),
                                        torch.nn.SiLU(),
                                    ))
                                self.speed_constraint_consensus_route_experts = (
                                    torch.nn.ModuleList([
                                        PartialSharedFourierRouteTail(
                                            self.speed_constraint_consensus_width)
                                        for _ in range(ROUTE_CHART_COUNT)
                                    ]))
                            else:
                                self.speed_constraint_consensus_route_experts = (
                                    torch.nn.ModuleList([
                                        torch.nn.Sequential(
                                            Linear(
                                                self.speed_constraint_consensus_width,
                                                64),
                                            torch.nn.SiLU(),
                                            Linear(64, 1),
                                        )
                                        for _ in range(ROUTE_CHART_COUNT)
                                    ]))
                            self.register_buffer(
                                'speed_constraint_route_chart_mean',
                                torch.zeros(self.dim))
                            self.register_buffer(
                                'speed_constraint_route_chart_std',
                                torch.ones(self.dim))
                            self.register_buffer(
                                'speed_constraint_route_chart_centers',
                                torch.zeros(ROUTE_CHART_COUNT, self.dim))
                            self.register_buffer(
                                'speed_constraint_route_chart_temperature',
                                torch.ones(()))
                            self.register_buffer(
                                'speed_constraint_route_chart_fitted',
                                torch.zeros((), dtype=torch.bool))
                        if (self.speed_constraint_consensus_mode
                                in ROUTE_CHART_FOURIER_RISK_MODES):
                            # Phase-BV frozen-additive scalar veto.  A shared
                            # representation combines the inherited corridor
                            # embedding, its complete scalar context, and a
                            # deterministic multiband standardized-q encoder.
                            # Four private logits represent safe/cup/pour and
                            # actionable risk; their minimum can only lower
                            # the route-union proposal in log-odds space.
                            context_dim = (
                                self.dim + self.speed_constraint_head_count
                                + 10)
                            q_feature_dim = self.dim * (
                                1 + 2 * len(
                                    ROUTE_CHART_FOURIER_RISK_FREQUENCIES))
                            self.register_buffer(
                                'speed_constraint_fourier_risk_frequencies',
                                torch.tensor(
                                    ROUTE_CHART_FOURIER_RISK_FREQUENCIES,
                                    dtype=torch.float32))
                            self.speed_constraint_fourier_risk_hidden_branch = (
                                torch.nn.Sequential(
                                    Linear(
                                        self.speed_constraint_consensus_width,
                                        64),
                                    torch.nn.SiLU(),
                                ))
                            self.speed_constraint_fourier_risk_context_branch = (
                                torch.nn.Sequential(
                                    Linear(context_dim, 64),
                                    torch.nn.SiLU(),
                                ))
                            self.speed_constraint_fourier_risk_q_branch = (
                                torch.nn.Sequential(
                                    Linear(q_feature_dim, 128),
                                    torch.nn.SiLU(),
                                    Linear(128, 64),
                                    torch.nn.SiLU(),
                                ))
                            self.speed_constraint_fourier_risk_fusion = (
                                torch.nn.Sequential(
                                    Linear(64 * 3, 64),
                                    torch.nn.SiLU(),
                                ))
                            self.speed_constraint_fourier_risk_private_tails = (
                                torch.nn.ModuleList([
                                    torch.nn.Sequential(
                                        Linear(64, 64),
                                        torch.nn.SiLU(),
                                        Linear(64, 1),
                                    ) for _ in range(
                                        ROUTE_CHART_FOURIER_RISK_HEAD_COUNT)
                                ]))
                            for tail in (
                                    self.speed_constraint_fourier_risk_private_tails):
                                torch.nn.init.zeros_(tail[-1].weight)
                                torch.nn.init.zeros_(tail[-1].bias)
                if self.speed_constraint_repair_width > 0:
                    # Per-factor zero-inflated reliability repair.  One shared
                    # q-only embedding feeds private release/magnitude outputs
                    # for safety, cup, and pour.  These only raise a component
                    # toward the direct scalar before the one analytic softmin.
                    repair_input_size = (
                        speed_h_size + self.speed_constraint_head_count + 2)
                    self.speed_constraint_repair_trunk = torch.nn.Sequential(
                        Linear(
                            repair_input_size,
                            self.speed_constraint_repair_width),
                        torch.nn.SiLU(),
                    )
                    self.speed_constraint_repair_heads = torch.nn.ModuleDict({
                        'release': Linear(
                            self.speed_constraint_repair_width,
                            self.speed_constraint_head_count),
                        'magnitude': Linear(
                            self.speed_constraint_repair_width,
                            self.speed_constraint_head_count),
                    })
                if self.lifted_phase:
                    # One scalar F(q,p), not a second field.  The inherited
                    # q-only logit is an upper bound and this in-network branch
                    # can only lower it when the continuous handover state p is
                    # inconsistent with the learned physical configuration.
                    # Seven analytic phase coordinates avoid asking the q
                    # encoder to resolve a narrow 0.1-wide dimension through
                    # its frozen Fourier basis.
                    phase_features = 7
                    phase_in = speed_h_size + phase_features
                    self.lifted_phase_speed_net = torch.nn.Sequential(
                        LayerNorm(phase_in),
                        Linear(phase_in, self.lifted_phase_width),
                        torch.nn.SiLU(),
                        Linear(self.lifted_phase_width, 1),
                    )
                if self.speed_correction_width:
                    # A direct scalar correction retains the complete
                    # multiscale Fourier coordinate instead of perturbing the
                    # inherited latent representation everywhere. Its final
                    # layer is initialized to zero below, making this branch
                    # an exact function-preserving extension. In
                    # ``positive_release`` mode it can only reopen free space
                    # that the inherited conservative field made too slow.
                    self.speed_correction_head = torch.nn.Sequential(
                        LayerNorm(speed_h_size),
                        Linear(speed_h_size, self.speed_correction_width),
                        torch.nn.SiLU(),
                        Linear(self.speed_correction_width, 1),
                    )
                if self.use_clearance_head:
                    # A direct signed-distance coordinate retains information
                    # that the historical clipped speed target destroys for
                    # every point at/below the strict boundary.  It shares the
                    # same scalar geometry trunk and remains part of this one
                    # network/checkpoint.
                    clearance_in = speed_h_size * (
                        2 if self.clearance_fourier_skip else 1)
                    self.clearance_head = torch.nn.Sequential(
                        Linear(clearance_in, speed_h_size // 2),
                        torch.nn.SiLU(),
                        Linear(speed_h_size // 2, 1),
                    )
                    # The clipped-speed encoder is preserved exactly. These
                    # zero-initialized residual blocks add clearance-specific
                    # capacity after the shared kinematic representation, so
                    # fitting signed depth cannot alter the proven speed field.
                    self.clearance_residual_blocks = torch.nn.ModuleList([
                        PairResidualBlock(speed_h_size)
                        for _ in range(self.clearance_residual_layers)
                    ])
                if self.partial_shared_embedding_dim > 0:
                    # A low-frequency q-only lane is shared by E and F while
                    # both proven private encoders remain intact.  The two
                    # zero-initialized adapters below make this a strictly
                    # function-preserving expansion of a separated checkpoint.
                    shared_dim = self.partial_shared_embedding_dim
                    self.partial_shared_stem = torch.nn.Sequential(
                        LayerNorm(h_size),
                        Linear(h_size, shared_dim),
                        torch.nn.SiLU(),
                        Linear(shared_dim, shared_dim),
                        torch.nn.SiLU(),
                    )
                    self.partial_shared_value_adapter = Linear(
                        shared_dim, h_size)
                    self.partial_shared_speed_adapter = Linear(
                        shared_dim, speed_h_size)

        if self.value_arch in (
                'global_residual', 'global_residual_symmetric',
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
                'bounded_chord_separate_smooth_excess_symmetric',
                'bounded_chord_separate_global_excess_symmetric',
                'bounded_chord_neural_viscosity_symmetric',
                'bounded_chord_signed_clearance_symmetric',
                'bounded_chord_clearance_eikonal_symmetric',
                'bounded_chord_clearance_state_symmetric',
                'endpoint_eikonal_residual_symmetric',
                'local_endpoint_eikonal_residual_symmetric',
                'bounded_chord_profile_symmetric',
                'fermat_waypoint_symmetric',
                'fermat_curve_symmetric',
                'fermat_curve_equivariant'):
            if self.value_arch in (
                    'global_residual_symmetric', 'bounded_chord_symmetric',
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
                    'bounded_chord_separate_smooth_excess_symmetric',
                    'bounded_chord_separate_global_excess_symmetric',
                    'bounded_chord_neural_viscosity_symmetric',
                    'bounded_chord_signed_clearance_symmetric',
                    'bounded_chord_clearance_eikonal_symmetric',
                    'bounded_chord_clearance_state_symmetric',
                    'endpoint_eikonal_residual_symmetric',
                    'local_endpoint_eikonal_residual_symmetric',
                    'bounded_chord_profile_symmetric',
                    'fermat_waypoint_symmetric', 'fermat_curve_symmetric'):
                pair_in = 3 * h_size + dim + 1
            else:
                pair_in = 4 * h_size + 2 * dim + 1
            if self.use_speed_head:
                pair_in += 2
            if self.couple_speed_latent_to_value:
                # Reversal-invariant local obstacle geometry from the same
                # scalar-speed trunk.  Unlike a scalar endpoint speed, these
                # features retain which kinematic modes created the nearby
                # boundary, so value gradients can inherit that structure.
                pair_in += 3 * self.speed_feature_dim
            if self.pair_slowness_attention_dim > 0:
                # Query: one reversal-invariant endpoint-pair representation.
                # Keys: local F-latent tokens at the scalar chord quadrature
                # points plus their reversal-invariant position and explicit
                # slowness. Values stay physical and interpretable below:
                # [slowness, log-slowness, hazard, symmetric position].
                # This is one in-network Q/K/V reduction, not external
                # retrieval, a waypoint bank, or a travel-time target.
                attention_pair_in = pair_in
                attention_dim = self.pair_slowness_attention_dim
                self.pair_slowness_attention_query = torch.nn.Sequential(
                    LayerNorm(attention_pair_in),
                    Linear(attention_pair_in, attention_dim),
                )
                self.pair_slowness_attention_key = torch.nn.Sequential(
                    LayerNorm(self.speed_feature_dim + 2),
                    Linear(self.speed_feature_dim + 2, attention_dim),
                )
                self.pair_slowness_attention_value_dim = 4
                pair_in += self.pair_slowness_attention_value_dim
            if self.value_arch == 'bounded_chord_profile_symmetric':
                # Reversal-invariant chord slowness profile (K values) plus
                # its quadrature mean.  These are appended so a warm-started
                # bounded-chord first layer can copy its old columns exactly
                # and initialize the new physical evidence at zero.
                pair_in += self.chord_quad_points + 1
            self.global_residual_net = torch.nn.Sequential(
                Linear(pair_in, h_size),
                torch.nn.SiLU(),
                Linear(h_size, h_size),
                torch.nn.SiLU(),
                Linear(
                    h_size,
                    (self.fermat_curve_modes * dim
                     if self.value_arch in (
                         'fermat_curve_symmetric',
                         'fermat_curve_equivariant')
                     else (dim
                           if self.value_arch == 'fermat_waypoint_symmetric'
                           else (2
                                 if self.value_arch in (
                                     'bounded_chord_speed_barrier_symmetric',
                                     'bounded_chord_local_speed_barrier_symmetric',
                                     'bounded_chord_global_excess_symmetric',
                                     'bounded_chord_smooth_excess_symmetric',
                                     'bounded_chord_signed_clearance_symmetric',
                                     'bounded_chord_eikonal_refine_symmetric',
                                     'endpoint_eikonal_residual_symmetric')
                                 else 1)))),
            )
            if self.value_arch in (
                    'bounded_chord_separate_smooth_excess_symmetric',
                    'bounded_chord_separate_global_excess_symmetric'):
                # Keep the inherited endpoint/chord value head at one output
                # and add an independent scalar viscosity branch.  The
                # branch consumes symmetric endpoint features plus the same
                # sampled chord-speed profile; its zero final layer makes
                # this an exact no-op at warm start.
                excess_pair_in = pair_in + self.chord_quad_points + 1
                self.global_excess_net = torch.nn.Sequential(
                    Linear(excess_pair_in, h_size),
                    torch.nn.SiLU(),
                    Linear(h_size, h_size),
                    torch.nn.SiLU(),
                    Linear(h_size, 1),
                )
            if self.use_viscosity_excess_head:
                # The inherited bounded-chord head remains byte-for-byte the
                # proven value branch.  This dedicated in-model branch sees a
                # reversal-invariant endpoint pair and the same scalar
                # slowness samples already used to form the chord bound.  In
                # ``positive_time`` mode it learns the historical non-negative
                # O(D^2) excess.  In ``signed_fraction`` mode it learns an
                # exact-zero signed correction to the bounded-chord mixing
                # logit, allowing the viscosity equation to rotate a wrong
                # characteristic instead of only raising its travel time.  No
                # geometry label, normal, critic, candidate bank, or second
                # inference model is introduced.
                # ``bounded_chord_profile_symmetric`` already appends the
                # reversal-invariant chord profile to ``pair_in`` above.  The
                # other bounded-chord heads only expose that profile to the
                # viscosity branch, so they need the extra columns here;
                # counting them twice makes the profile branch fail at the
                # first forward pass.
                viscosity_pair_in = pair_in + (
                    0 if self.value_arch ==
                    'bounded_chord_profile_symmetric'
                    else self.chord_quad_points + 1)
                if (self.viscosity_speed_latent_context
                        and not self.couple_speed_latent_to_value):
                    # Reuse the already-computed, speed-supervised endpoint
                    # latent only inside the new viscosity branch.  Keeping it
                    # out of the inherited global head preserves that head's
                    # dimensions and floating-point forward exactly.
                    viscosity_pair_in += 3 * self.speed_feature_dim
                if self.viscosity_chord_speed_latent_context:
                    # Compress each already-evaluated chord-point latent and
                    # retain its complete reversal-invariant profile.  The
                    # projection is deliberately narrow: it exposes which
                    # speed-supervised local modes occur along the segment
                    # without materializing K full-width feature vectors in
                    # every MPPI batch.
                    self.viscosity_chord_latent_projection = (
                        torch.nn.Sequential(
                            Linear(
                                self.speed_feature_dim,
                                self.viscosity_chord_latent_width),
                            torch.nn.SiLU(),
                            Linear(
                                self.viscosity_chord_latent_width,
                                self.viscosity_chord_latent_width),
                            torch.nn.SiLU(),
                        ))
                    viscosity_pair_in += (
                        self.chord_quad_points
                        * self.viscosity_chord_latent_width)
                if self.viscosity_directed_context:
                    # A scalar isotropic Eikonal field is reversible.  The
                    # cup->pour handover is not: its causal state constraint
                    # distinguishes q0->q1 from q1->q0.  Expose only ordered
                    # endpoint quantities already present in this one model:
                    # the task-routed latent difference, signed joint delta,
                    # signed endpoint-speed difference, and three routing
                    # phase coordinates.  No second field or path candidate is
                    # introduced.
                    viscosity_pair_in += h_size + dim + 4
                    if self.viscosity_handover_phase_context:
                        viscosity_pair_in += 3
                if self.viscosity_stencil_radius > 0.0:
                    # At the chord midpoint, probe the same scalar field on
                    # both sides of every joint-axis direction after removing
                    # its chord-parallel component.  These 2*d ordered
                    # slownesses plus the center retain which transverse side
                    # is freer while remaining exactly endpoint-reversal
                    # invariant.  This is a fixed PDE stencil, not a bank of
                    # decoded paths or inference candidates.
                    viscosity_pair_in += 2 * dim + 1
                activation = (
                    torch.nn.SiLU
                    if self.viscosity_activation == 'silu'
                    else MaxMin)
                self.viscosity_excess_net = torch.nn.Sequential(
                    Linear(viscosity_pair_in, h_size),
                    activation(),
                    Linear(h_size, h_size),
                    activation(),
                    Linear(h_size, 1),
                )
                if self.viscosity_minplus_modes:
                    # A global Eikonal viscosity solution is the lower
                    # envelope of locally smooth arrival-time branches and is
                    # therefore generally non-smooth at a change of homotopy.
                    # Keep this representation inside one scalar network: a
                    # shared trunk feeds a few internal scalar modes, whose
                    # normalized soft minimum is added to the inherited
                    # signed correction.  No modes are exposed as candidate
                    # paths and inference still returns one scalar value.
                    self.viscosity_minplus_trunk = torch.nn.Sequential(
                        Linear(viscosity_pair_in, h_size),
                        activation(),
                    )
                    self.viscosity_minplus_heads = torch.nn.ModuleList([
                        torch.nn.Sequential(
                            Linear(h_size, h_size),
                            activation(),
                            Linear(h_size, 1),
                        )
                        for _ in range(self.viscosity_minplus_modes)
                    ])
                if self.use_viscosity_gain_head:
                    # Learn how strongly to trust the already trained signed
                    # viscosity correction at each endpoint pair.  The gain
                    # reads only the same scalar-speed/chord representation,
                    # and exp(log(2)*tanh(.)) keeps it positive and within
                    # [1/2, 2].  Its exact-zero last layer starts at gain 1,
                    # preserving the inherited scalar value bit-for-bit.
                    self.viscosity_gain_net = torch.nn.Sequential(
                        Linear(viscosity_pair_in, h_size),
                        activation(),
                        Linear(h_size, h_size),
                        activation(),
                        Linear(h_size, 1),
                    )
            if self.use_causal_handover_head:
                # One scalar viscosity solution can contain a reversible
                # obstacle distance and a directed state-constraint term.
                # The latter reads the same symmetric endpoint/chord evidence
                # plus ordered endpoint differences.  Its zero-initialized
                # last layer makes this complete architecture exactly equal
                # to the inherited signed Eikonal value at warm start.
                causal_pair_in = (
                    pair_in + self.chord_quad_points + 1
                    + h_size + dim + 4)
                if self.causal_handover_phase_context:
                    causal_pair_in += 3
                self.causal_handover_net = torch.nn.Sequential(
                    Linear(causal_pair_in, h_size),
                    torch.nn.SiLU(),
                    Linear(h_size, h_size),
                    torch.nn.SiLU(),
                    Linear(h_size, 1),
                )
            # The legacy pair head has only two nonlinear layers.  These
            # residual refinements add global endpoint-interaction capacity
            # while remaining an exact no-op at initialization, so a
            # speed-field checkpoint can be deepened without changing its
            # initial value function.
            self.global_pair_residual_blocks = torch.nn.ModuleList([
                PairResidualBlock(
                    h_size, activation=self.global_pair_residual_activation)
                for _ in range(self.global_pair_residual_layers)
            ])
            if self.value_arch in (
                    'bounded_chord_fermat_curve_symmetric',
                    'bounded_chord_endpoint_fermat_curve_symmetric',
                    'bounded_chord_endpoint_speed_fermat_curve_symmetric'):
                # One directed decoder is evaluated in both endpoint orders
                # and projected onto the exact sine-mode parity subspaces.
                # It shares the encoder and scalar speed field with the base
                # value; it is not a second model or a bank of candidates.
                speed_profile_arch = (
                    self.value_arch ==
                    'bounded_chord_endpoint_speed_fermat_curve_symmetric')
                if speed_profile_arch:
                    # A shared point encoder reads only q, chord position,
                    # and this same model's scalar speed.  Quadrature pooling
                    # makes its output size independent of the number of
                    # probes, so dense scalar-only training and economical
                    # deployment use exactly the same learned function.
                    self.fermat_speed_profile_net = torch.nn.Sequential(
                        Linear(dim + 2, h_size),
                        torch.nn.SiLU(),
                        Linear(h_size, h_size),
                        torch.nn.SiLU(),
                    )
                directed_pair_in = (
                    4 * h_size + 2 * dim + 3
                    + (h_size if speed_profile_arch else 0))
                self.fermat_curve_net = torch.nn.Sequential(
                    Linear(directed_pair_in, h_size),
                    torch.nn.SiLU(),
                    Linear(h_size, h_size),
                    torch.nn.SiLU(),
                    Linear(h_size, self.fermat_curve_modes * dim),
                )
                self.fermat_curve_residual_blocks = torch.nn.ModuleList([
                    PairResidualBlock(h_size)
                    for _ in range(self.fermat_curve_residual_layers)
                ])

        # --- pe_gate: shared or per-head ---
        if separate_pe_gate:
            # Per-head pe_gate: each subnet gets independent u,v modulation
            # This removes the shared-modulation bottleneck where pe_gate must
            # simultaneously work for cupping (palm up) and pouring (tilt forward).
            self.cup_pe_gate = torch.nn.ModuleList()
            self.cup_pe_gate.append(Linear(h_size, h_size))
            self.cup_pe_gate.append(Linear(h_size, h_size))
            self.pour_pe_gate = torch.nn.ModuleList()
            self.pour_pe_gate.append(Linear(h_size, h_size))
            self.pour_pe_gate.append(Linear(h_size, h_size))
            # Keep shared pe_gate for backward compat (unused in forward)
            self.pe_gate = torch.nn.ModuleList()
            self.pe_gate.append(Linear(h_size, h_size))
            self.pe_gate.append(Linear(h_size, h_size))
        else:
            self.pe_gate = torch.nn.ModuleList()
            self.pe_gate.append(Linear(h_size, h_size))
            self.pe_gate.append(Linear(h_size, h_size))

        # Optional pointwise FK-conditioning. Each start/goal endpoint receives
        # a compact FK feature vector projected into the Fourier hidden space.
        if self.fk_feature_dim > 0:
            self.fk_feature_proj = torch.nn.Sequential(
                Linear(self.fk_feature_dim, h_size),
                torch.nn.SiLU(),
                Linear(h_size, h_size),
            )

        # --- Shared projector (optional) ---
        if use_projector:
            self.projector = Linear(h_size, h_size)

        # --- R58F decoupled additive orientation head (optional) ---
        # Consumes the per-config blended encoder feature and outputs a scalar
        # orientation cost Phi_orient(q) >= 0 (softplus). Because it reads the
        # same per-config feature that already encodes orientation, its gradient
        # is NOT diluted by the position-reaching geometry that dominates tau.
        if self.use_orient_head:
            # R58F fix: feed the head a Fourier encoding of the RAW joint config
            # (config-only, zone-agnostic) instead of the position-optimized
            # blended encoder feature. Wrist tilt is a deterministic function of
            # q, so Fourier(q) gives the head direct access to recover it.
            #
            # The orientation target d_orient(q) is a HIGH-FREQUENCY function of
            # q (it comes through FK orientation). The shared B basis (B_scale
            # ~0.2, tuned for smooth travel-time) is too low-frequency and a tiny
            # 64-unit head cannot fit it (standalone probe: corr~0.16). A
            # DEDICATED high-frequency basis + a wider/deeper head recovers it
            # (standalone probe: B_scale=3, 256 freqs, 256x256 MLP -> corr~0.88).
            # R58S: B_scale is env-configurable. R58F used 3.0 (high-freq) to FIT the
            # tilt VALUE, but that yields an ERRATIC gradient (step-to-step direction
            # consistency -0.69; pure -grad Phi descent INCREASES tilt). A SMOOTH
            # potential (low B_scale ~0.3) gives a consistent tilt-descent gradient --
            # which is what tau_total = tau_reach + lambda*d_orient actually needs.
            import os as _osb
            orient_freqs = int(_osb.environ.get('R58_ORIENT_FREQS', '256'))
            orient_b_scale = float(_osb.environ.get('R58_ORIENT_BSCALE', '3.0'))
            B_orient = orient_b_scale * torch.randn(dim, orient_freqs)
            # Buffer so it is saved/loaded with the checkpoint (deterministic).
            self.register_buffer('B_orient', B_orient.to(device))
            fourier_in = 2 * orient_freqs  # input_mapping -> cat([sin, cos])

            def make_orient_head():
                return torch.nn.Sequential(
                    Linear(fourier_in, 256),
                    torch.nn.ReLU(),
                    Linear(256, 256),
                    torch.nn.ReLU(),
                    Linear(256, 1),
                    torch.nn.Softplus(),
                )

            if self.use_task_orient_heads:
                self.orient_head_cup = make_orient_head()
                self.orient_head_pour = make_orient_head()
                # Compatibility for older diagnostics/loss branches that refer
                # to orient_head directly. The routed heads are used in out().
                self.orient_head = make_orient_head()
            else:
                self.orient_head = make_orient_head()

        # Optional learned collision/self-collision potential. This is a
        # config-only learned field supervised from offline distance labels; no
        # teacher trajectory, teacher direction, planner-side rerank, or online
        # collision oracle is used at inference.
        if self.use_safety_head:
            import os as _oss
            safety_freqs = int(_oss.environ.get('R58_SAFETY_FREQS', '256'))
            safety_b_scale = float(_oss.environ.get('R58_SAFETY_BSCALE', '1.0'))
            B_safety = safety_b_scale * torch.randn(dim, safety_freqs)
            self.register_buffer('B_safety', B_safety.to(device))
            safety_in = 2 * safety_freqs
            def make_safety_head():
                return torch.nn.Sequential(
                    Linear(safety_in, 256),
                    torch.nn.ReLU(),
                    Linear(256, 256),
                    torch.nn.ReLU(),
                    Linear(256, 1),
                    torch.nn.Softplus(),
                )
            self.safety_head = make_safety_head()
            if self.use_pair_safety_heads:
                self.safety_head_cup = make_safety_head()
                self.safety_head_pour = make_safety_head()
                self.safety_head_cross = make_safety_head()

        # --- Cross-attention blend (optional) ---
        if blend_mode == 'cross_attention':
            self.cross_attn_q = Linear(h_size, h_size)
            self.cross_attn_k = Linear(h_size, h_size)
            self.cross_attn_v = Linear(h_size, h_size)
            self.cross_attn_scale = h_size ** -0.5

        # --- FiLM blend (optional): learned per-dimension routing ---
        if blend_mode == 'film':
            # Input: ee_y scalar → (h_size,) gamma and beta
            self.film_net = torch.nn.Sequential(
                Linear(1, h_size),
                torch.nn.ReLU(),
                Linear(h_size, 2 * h_size),  # gamma and beta
            )

        # B-series: asymmetric goal-conditioned value field.
        # tau(q, goal, task) is not a symmetric latent distance. It is a direct
        # ordered value with the boundary condition tau(q, q)=0 enforced by the
        # joint-space distance factor.
        if self.value_arch == 'asym_goal':
            cond_dim = {
                'none': 0,
                'task': 10,
                'task_scene': 14,
                'task_scene_dist': 15,
                'task_scene_pair': 14,
                'task_scene_pair_dist': 15,
            }[self.value_condition]
            self.asym_value_condition_dim = cond_dim
            self.asym_value_net = torch.nn.Sequential(
                Linear(4 * h_size + cond_dim, h_size),
                torch.nn.SiLU(),
                Linear(h_size, h_size),
                torch.nn.SiLU(),
                Linear(h_size, 1),
            )
        elif self.value_arch == 'local_finsler':
            point_cond_dim = {
                'none': 0,
                'task': 5,
                'task_scene': 7,
                'task_scene_dist': 8,
                'task_scene_pair': 21,
                'task_scene_pair_dist': 22,
            }[self.value_condition]
            self.local_finsler_condition_dim = point_cond_dim
            slowness_in = h_size + dim + point_cond_dim

            def make_slowness_net(init_slowness=3.1):
                """Slowness MLP, output layer initialised INSIDE the label range.

                Measured 2026-08-31: with default init the final layer emits
                logits around 8.6, which saturates any bounded slowness map at
                epoch 1 -- the head is dead before training starts and the field
                degenerates to a constant metric (straight-line geodesics).
                Shrink the last layer and set its bias so the head begins at the
                mean label slowness (1/speed ~ 3.1), well inside the (1, 12)
                range where _bounded_slowness has a live gradient.
                """
                net = torch.nn.Sequential(
                    Linear(slowness_in, h_size),
                    torch.nn.SiLU(),
                    Linear(h_size, h_size),
                    torch.nn.SiLU(),
                    Linear(h_size, 1),
                )
                head = net[-1]
                inner = getattr(head, "weight", None)
                with torch.no_grad():
                    for attr in ("weight", "weight_g", "weight_v"):
                        w = getattr(head, attr, None)
                        if w is not None and w.requires_grad is not None:
                            w.mul_(0.01)
                    frac = (float(init_slowness) - 1.0) / 11.0
                    frac = min(max(frac, 1e-3), 1.0 - 1e-3)
                    bias = getattr(head, "bias", None)
                    if bias is not None:
                        bias.fill_(float(np.log(frac / (1.0 - frac))))
                return net

            if self.finsler_task_heads:
                self.local_slowness_cup = make_slowness_net()
                self.local_slowness_pour = make_slowness_net()
            else:
                self.local_slowness_net = make_slowness_net()

            if self.use_cross_slowness_residual:
                self.cross_slowness_residual = make_slowness_net()
            if self.use_local_slowness_residual:
                self.local_slowness_residual = make_slowness_net()

        # --- Zone routing params ---
        self.y_divide = y_divide
        self.overlap = overlap
        self.route_temp = route_temp
        # G1 uses cupping on the lower-y side. UR5e uses cupping on higher-y.
        self.cup_side = 1.0 if cup_side >= 0 else -1.0

    def _cup_distance(self, ee_y, y_divide=None):
        """Signed Y distance from the boundary, positive on the cupping side."""
        y_div = self.y_divide if y_divide is None else y_divide
        return self.cup_side * (ee_y - y_div)

    def init_weights(self, m):
        if type(m) == torch.nn.Linear:
            stdv = np.sqrt(2.0 / (m.weight.size(0) + m.weight.size(1)))
            torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=stdv, a=-2.0 * stdv, b=2.0 * stdv)
            m.bias.data.fill_(0)

        # Initialize gates for both heads
        for i in range(self.nl1):
            self.cup_gate[i].weight.data.fill_(0)
            self.cup_gate[i].bias.data.fill_(0)
            self.pour_gate[i].weight.data.fill_(0)
            self.pour_gate[i].bias.data.fill_(0)
            if hasattr(self, 'shared_gate'):
                self.shared_gate[i].weight.data.fill_(0)
                self.shared_gate[i].bias.data.fill_(0)

    def init_task_adapter_weights(self):
        """Start task adapters as near-identity residuals."""
        if not getattr(self, 'use_task_adapters', False):
            return
        for adapter in (self.cup_adapter, self.pour_adapter):
            last = adapter[-1]
            if isinstance(last, torch.nn.Linear):
                torch.nn.init.zeros_(last.weight)
                torch.nn.init.zeros_(last.bias)

    def init_raw_coordinate_skip_weights(self):
        """Start the optional raw-q path as an exact no-op residual."""
        if not self.use_raw_coordinate_skip:
            return
        torch.nn.init.zeros_(self.raw_coordinate_projection.weight)
        torch.nn.init.zeros_(self.raw_coordinate_projection.bias)

    def init_cross_residual_weights(self):
        """Start the optional cross residual as an exact no-op."""
        for enabled_attr, module_attr in (
                ('use_cross_slowness_residual', 'cross_slowness_residual'),
                ('use_local_slowness_residual', 'local_slowness_residual')):
            if not getattr(self, enabled_attr, False):
                continue
            last = getattr(self, module_attr)[-1]
            if isinstance(last, torch.nn.Linear):
                torch.nn.init.zeros_(last.weight)
                torch.nn.init.zeros_(last.bias)

    def init_global_residual_weights(self):
        """Initialize the global value as exact normalized Euclidean time."""
        if self.value_arch not in (
                'global_residual', 'global_residual_symmetric',
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
                'bounded_chord_separate_smooth_excess_symmetric',
                'bounded_chord_separate_global_excess_symmetric',
                'bounded_chord_neural_viscosity_symmetric',
                'bounded_chord_signed_clearance_symmetric',
                'bounded_chord_clearance_eikonal_symmetric',
                'bounded_chord_clearance_state_symmetric',
                'endpoint_eikonal_residual_symmetric',
                'local_endpoint_eikonal_residual_symmetric',
                'bounded_chord_profile_symmetric',
                'fermat_waypoint_symmetric',
                'fermat_curve_symmetric',
                'fermat_curve_equivariant'):
            return
        last = self.global_residual_net[-1]
        if self.value_arch in (
                'fermat_waypoint_symmetric', 'fermat_curve_symmetric',
                'fermat_curve_equivariant'):
            # A perfectly straight initial path is a stationary point in a
            # symmetric barrier.  A tiny deterministic perturbation lets the
            # one waypoint choose either valid side while remaining virtually
            # identical to the straight chord at initialization.
            torch.nn.init.normal_(last.weight, mean=0.0, std=1e-3)
            torch.nn.init.zeros_(last.bias)
        else:
            torch.nn.init.zeros_(last.weight)
        if self.value_arch == 'endpoint_eikonal_residual_symmetric':
            # Row 0 controls how quickly purely local endpoint slowness fades
            # with separation. Row 1 starts as a tiny positive global excess;
            # both retain useful derivatives while the exact local Eikonal
            # limit is supplied structurally by the endpoint speeds.
            with torch.no_grad():
                last.bias[0] = 5.0
                last.bias[1] = 0.05
        elif self.value_arch == 'local_endpoint_eikonal_residual_symmetric':
            # The endpoint asymptotic is analytically capped to one local
            # radius, so the only neural output is the global D^2 excess.
            # A small nonzero bias avoids the zero derivative of tanh(raw)^2.
            with torch.no_grad():
                last.bias[0] = 0.05
        elif self.value_arch in (
                'bounded_chord_speed_barrier_symmetric',
                'bounded_chord_local_speed_barrier_symmetric',
                'bounded_chord_global_excess_symmetric',
                'bounded_chord_smooth_excess_symmetric',
                'bounded_chord_signed_clearance_symmetric'):
            # Row 0 exactly warm-starts the proven chord fraction. Row 1 is a
            # near-zero positive gate for the selected scalar-speed-derived
            # correction. Its small nonzero derivative lets Eikonal/Bellman
            # supervision decide whether that extra capacity is needed.
            with torch.no_grad():
                last.bias[0] = -2.0
                last.bias[1] = -6.0
        elif self.value_arch == 'bounded_chord_eikonal_refine_symmetric':
            # Row 0 is the proven global chord fraction.  Row 1 is a positive
            # attenuation rate for an exact local Eikonal envelope.  A large
            # initial rate makes the envelope vanish away from the diagonal,
            # so a bounded-chord checkpoint can be inherited without
            # disturbing its already useful global viscosity branch.
            with torch.no_grad():
                last.bias[0] = -2.0
                last.bias[1] = 20.0
        elif self.value_arch == 'factored_source_eikonal_symmetric':
            # R=0 is the exact analytic source-factorization control.  Unlike
            # a bounded-chord sigmoid fraction, this output is a signed log
            # correction and must not inherit the historical -2 fraction
            # bias.
            torch.nn.init.zeros_(last.bias)
        elif self.value_arch in (
                'bounded_chord_symmetric',
                'bounded_chord_fermat_curve_symmetric',
                'bounded_chord_endpoint_fermat_curve_symmetric',
                'bounded_chord_endpoint_speed_fermat_curve_symmetric',
                'bounded_chord_log_symmetric',
                'chord_integral_symmetric',
                'bounded_chord_hazard_symmetric',
                'bounded_chord_hazard_floor_symmetric',
                'bounded_chord_endpoint_eikonal_symmetric',
                'bounded_chord_local_endpoint_eikonal_symmetric',
                'bounded_chord_separate_smooth_excess_symmetric',
                'bounded_chord_separate_global_excess_symmetric',
                'bounded_chord_neural_viscosity_symmetric',
                'bounded_chord_profile_symmetric'):
            # Begin near the Euclidean lower bound while retaining a useful
            # derivative for the learned position between the two bounds.
            torch.nn.init.constant_(last.bias, -2.0)
        elif self.value_arch == 'bounded_chord_eikonal_symmetric':
            # The output is a non-negative attenuation rate kappa in
            # exp(-||q-g|| * kappa).  A moderate positive start keeps long
            # pairs close to the Euclidean lower bound while preserving the
            # exact local Eikonal limit at q -> g.
            torch.nn.init.constant_(last.bias, 5.0)
        elif self.global_residual_positive:
            # tanh(raw)^2 is non-negative.  A small nonzero start avoids the
            # zero derivative of the square while remaining nearly Euclidean.
            torch.nn.init.constant_(last.bias, 0.05)
        else:
            torch.nn.init.zeros_(last.bias)
        if self.use_speed_head:
            speed_last = self.speed_head[-1]
            torch.nn.init.zeros_(speed_last.weight)
        if self.use_clearance_head:
            clearance_last = self.clearance_head[-1]
            torch.nn.init.zeros_(clearance_last.weight)
            torch.nn.init.zeros_(clearance_last.bias)
            torch.nn.init.constant_(speed_last.bias, 4.0)

    def init_speed_constraint_weights(self):
        """Initialize every semantic residual head as an exact scalar no-op."""
        for head in getattr(self, 'speed_constraint_heads', []):
            torch.nn.init.zeros_(head[-1].weight)
            torch.nn.init.zeros_(head[-1].bias)

    def init_speed_constraint_consensus_weights(self):
        """Initialize ordinary consensus neutrally or corridor consensus sparsely."""
        if self.speed_constraint_consensus_width < 1:
            return
        if self.speed_constraint_consensus_mode == 'single':
            head = self.speed_constraint_consensus_head
            torch.nn.init.zeros_(head[-1].weight)
            torch.nn.init.zeros_(head[-1].bias)
            return
        if self.speed_constraint_consensus_mode in (
                'corridor_hurdle', 'corridor_hurdle_compensated',
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
            # Training-domain exact labels put corridor support near one
            # percent and its conditional interpolation magnitude near 0.1.
            # These fixed sparse priors make the new upward-only branch a
            # sub-1e-3 perturbation while retaining ordinary sigmoid BCE/MSE.
            priors = {'release': 0.01, 'magnitude': 0.1}
            if self.speed_constraint_consensus_mode in (
                    'corridor_hurdle_specificity',
                    'corridor_hurdle_specificity_odds',
                    'corridor_hurdle_specificity_orthogonal',
                    'corridor_hurdle_semantic_veto_min_orthogonal',
                    'corridor_hurdle_semantic_veto_min_odds',
                    'corridor_hurdle_semantic_veto_min_odds_residual',
                    'corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend',
                    'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8',
                    *ROUTE_CHART_MODES):
                # Neutral class-balanced initialization for the new private
                # specificity classifier.  Phase-AJ loading overwrites the
                # inherited release and magnitude tensors only.
                priors['specificity'] = 0.5
            for name, head in self.speed_constraint_consensus_heads.items():
                probability = priors[name]
                bias = math.log(probability / (1.0 - probability))
                torch.nn.init.zeros_(head.weight)
                torch.nn.init.constant_(head.bias, bias)
            if self.speed_constraint_consensus_mode in (
                    'corridor_hurdle_semantic_veto_min_odds_residual',
                    'corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend',
                    'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8',
                    *ROUTE_CHART_MODES):
                residual_last = (
                    self.speed_constraint_consensus_magnitude_residual[-1])
                torch.nn.init.zeros_(residual_last.weight)
                torch.nn.init.zeros_(residual_last.bias)
            if (self.speed_constraint_consensus_mode
                    == 'corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend'):
                blend_last = (
                    self.speed_constraint_consensus_evidence_blend_residual[-1])
                torch.nn.init.zeros_(blend_last.weight)
                torch.nn.init.zeros_(blend_last.bias)
            if (self.speed_constraint_consensus_mode
                    == 'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8'):
                headwise_last = (
                    self.speed_constraint_consensus_head_logit_residual[-1])
                torch.nn.init.zeros_(headwise_last.weight)
                torch.nn.init.zeros_(headwise_last.bias)
            if self.speed_constraint_consensus_mode in ROUTE_CHART_MODES:
                for expert in self.speed_constraint_consensus_route_experts:
                    if (self.speed_constraint_consensus_mode
                            in ROUTE_CHART_PARTIAL_SHARED_MODES):
                        torch.nn.init.zeros_(expert.output.weight)
                        torch.nn.init.zeros_(expert.output.bias)
                    else:
                        torch.nn.init.zeros_(expert[-1].weight)
                        torch.nn.init.zeros_(expert[-1].bias)
            if (self.speed_constraint_consensus_mode
                    in ROUTE_CHART_FOURIER_RISK_MODES):
                for tail in self.speed_constraint_fourier_risk_private_tails:
                    torch.nn.init.zeros_(tail[-1].weight)
                    torch.nn.init.zeros_(tail[-1].bias)
            return
        # sigmoid(bias)^2 = 0.5 keeps the hurdle product at the same neutral
        # alpha as the historical single head before either private tail has
        # learned.  The experts coincide in a from-scratch no-op warm start,
        # while Phase-AB initialization remains directly comparable to AE/AF.
        neutral_probability = math.sqrt(0.5)
        neutral_bias = math.log(
            neutral_probability / (1.0 - neutral_probability))
        for head in self.speed_constraint_consensus_heads.values():
            torch.nn.init.zeros_(head.weight)
            torch.nn.init.constant_(head.bias, neutral_bias)

    def init_speed_constraint_repair_weights(self):
        """Initialize sparse repair with a 0.1 release and 0.5 magnitude."""
        if self.speed_constraint_repair_width < 1:
            return
        priors = {'release': 0.1, 'magnitude': 0.5}
        for name, head in self.speed_constraint_repair_heads.items():
            probability = priors[name]
            bias = math.log(probability / (1.0 - probability))
            torch.nn.init.zeros_(head.weight)
            torch.nn.init.constant_(head.bias, bias)

    def init_global_excess_weights(self):
        """Start the separate smooth scalar excess as an exact no-op."""
        branch = getattr(self, 'global_excess_net', None)
        if branch is None:
            return
        last = branch[-1]
        torch.nn.init.zeros_(last.weight)
        # The physical gate subtracts sigmoid(-6), so this bias is exactly
        # neutral while retaining a nonzero learning derivative.
        torch.nn.init.constant_(last.bias, -6.0)

    def init_global_pair_residual_weights(self):
        """Make every optional pair-refinement block an exact identity."""
        for block in getattr(self, 'global_pair_residual_blocks', []):
            torch.nn.init.zeros_(block.fc2.weight)
            torch.nn.init.zeros_(block.fc2.bias)

    def init_partial_shared_embedding_weights(self):
        """Start the shared E/F lane as an exact two-branch no-op."""
        if self.partial_shared_embedding_dim <= 0:
            return
        for adapter in (
                self.partial_shared_value_adapter,
                self.partial_shared_speed_adapter):
            torch.nn.init.zeros_(adapter.weight)
            torch.nn.init.zeros_(adapter.bias)

    def init_fermat_curve_weights(self):
        """Initialize the single curve as an exact straight-line no-op."""
        branch = getattr(self, 'fermat_curve_net', None)
        if branch is None:
            return
        profile = getattr(self, 'fermat_speed_profile_net', None)
        if profile is not None:
            for layer in profile:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)
        for layer in branch[:-1]:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        torch.nn.init.zeros_(branch[-1].weight)
        torch.nn.init.zeros_(branch[-1].bias)
        for block in self.fermat_curve_residual_blocks:
            torch.nn.init.xavier_uniform_(block.fc1.weight)
            torch.nn.init.zeros_(block.fc1.bias)
            torch.nn.init.zeros_(block.fc2.weight)
            torch.nn.init.zeros_(block.fc2.bias)

    def init_viscosity_excess_weights(self):
        """Start either conditional viscosity residual as an exact no-op."""
        branch = getattr(self, 'viscosity_excess_net', None)
        if branch is None:
            return
        last = branch[-1]
        torch.nn.init.zeros_(last.weight)
        if self.viscosity_residual_mode in (
                'signed_fraction', 'causal_log_time'):
            torch.nn.init.zeros_(last.bias)
        else:
            torch.nn.init.constant_(last.bias, -6.0)

    def init_viscosity_minplus_weights(self):
        """Start the internal lower envelope as an exact scalar no-op."""
        heads = getattr(self, 'viscosity_minplus_heads', None)
        if heads is None:
            return
        for head in heads:
            torch.nn.init.zeros_(head[-1].weight)
            torch.nn.init.zeros_(head[-1].bias)

    def init_viscosity_gain_weights(self):
        """Start conditional viscosity trust at the exact identity gain."""
        branch = getattr(self, 'viscosity_gain_net', None)
        if branch is None:
            return
        torch.nn.init.zeros_(branch[-1].weight)
        torch.nn.init.zeros_(branch[-1].bias)

    def init_causal_handover_weights(self):
        """Start the directed factor as an exact identity multiplier."""
        branch = getattr(self, 'causal_handover_net', None)
        if branch is None:
            return
        for layer in branch[:-1]:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        torch.nn.init.zeros_(branch[-1].weight)
        torch.nn.init.zeros_(branch[-1].bias)

    def init_clearance_residual_weights(self):
        """Make optional clearance adapters exact identities at warm start."""
        for block in getattr(self, 'clearance_residual_blocks', []):
            torch.nn.init.zeros_(block.fc2.weight)
            torch.nn.init.zeros_(block.fc2.bias)

    def init_speed_residual_weights(self):
        """Make optional scalar-speed refiners exact identities."""
        for block in getattr(self, 'speed_residual_blocks', []):
            torch.nn.init.xavier_uniform_(block.fc1.weight)
            torch.nn.init.zeros_(block.fc1.bias)
            torch.nn.init.zeros_(block.fc2.weight)
            torch.nn.init.zeros_(block.fc2.bias)

    def init_speed_correction_weights(self):
        """Make the optional direct scalar correction an exact no-op."""
        head = getattr(self, 'speed_correction_head', None)
        if head is None:
            return
        last = head[-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)

    def _apply_speed_correction(self, logit, points):
        """Apply the optional correction while retaining one scalar field.

        ``signed_logit`` is the historical unconstrained residual.
        ``positive_release`` instead maps the residual to a fractional
        reopening of the inherited free-space probability.  It is exactly a
        no-op at zero initialization, stays in [0, 1], and cannot make any
        configuration slower than the inherited field.
        """
        correction = getattr(self, 'speed_correction_head', None)
        if correction is None:
            return logit
        if points is None:
            raise RuntimeError("speed correction requires point coordinates")
        basis = self.B_speed if self.separate_speed_encoder else self.B
        raw_fourier = self.input_mapping(points, basis)
        raw_correction = correction(raw_fourier)
        if self.speed_correction_mode == 'signed_logit':
            return logit + self.speed_correction_scale * raw_correction

        # clamp_min, unlike ReLU, has unit derivative at the zero boundary in
        # PyTorch. Thus the exactly-zero final layer can learn immediately
        # while the deployed correction remains one-sided by construction.
        release = torch.tanh(raw_correction).clamp_min(0.0)
        eps = torch.finfo(logit.dtype).eps
        release = (self.speed_correction_scale * release).clamp_max(
            1.0 - eps)
        # If p=sigmoid(logit), reopening a fraction r of the remaining free
        # probability gives p' = p + (1-p)r and therefore
        # logit(p') = log(exp(logit)+r) - log(1-r).  This form avoids the
        # sigmoid/logit round trip (and its infinities at saturated p=1).
        log_release = torch.log(
            release.clamp_min(torch.finfo(logit.dtype).tiny))
        return (
            torch.logaddexp(logit, log_release)
            - torch.log1p(-release))

    def init_lifted_phase_weights(self):
        """Make the lifted scalar-speed restriction an exact warm-start no-op."""
        branch = getattr(self, 'lifted_phase_speed_net', None)
        if branch is None:
            return
        last = branch[-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.constant_(last.bias, -2.0)

    def _lifted_phase_features(self, points):
        """Continuous p features for the final coordinate of (q, p)."""
        z = points[:, -1:]
        p = (z / self.lifted_phase_scale + 0.5).clamp(0.0, 1.0)
        return torch.cat([
            p,
            1.0 - p,
            p * (1.0 - p),
            torch.sin(math.pi * p),
            torch.cos(math.pi * p),
            torch.sin(2.0 * math.pi * p),
            torch.cos(2.0 * math.pi * p),
        ], dim=1)

    def _apply_lifted_phase_logit(self, points, latent, logit):
        """Intersect the inherited field with one learned phase-valid set."""
        branch = getattr(self, 'lifted_phase_speed_net', None)
        if branch is None:
            return logit
        raw = branch(torch.cat([
            latent, self._lifted_phase_features(points)], dim=1))
        origin = torch.sigmoid(raw.new_tensor(-2.0))
        gate = (torch.sigmoid(raw) - origin).clamp_min(0.0)
        return logit - self.lifted_phase_logit_scale * gate

    def _global_pair_output(self, pair):
        """Run the symmetric pair head plus optional residual reasoning."""
        x = pair
        for layer in self.global_residual_net[:-1]:
            x = layer(x)
        for block in getattr(self, 'global_pair_residual_blocks', []):
            x = block(x)
        return self.global_residual_net[-1](x)

    def _fermat_curve_pair_output(self, pair):
        """Run the one-curve decoder plus its identity refinements."""
        x = pair
        for layer in self.fermat_curve_net[:-1]:
            x = layer(x)
        for block in self.fermat_curve_residual_blocks:
            x = block(x)
        return self.fermat_curve_net[-1](x)

    def input_mapping(self, x, B_matrix=None):
        if B_matrix is None:
            B_matrix = self.B
        w = 2. * np.pi * B_matrix
        x_proj = x @ w
        mapped = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        if self.use_raw_coordinate_skip:
            raw = self.raw_coordinate_projection(x)
            if raw.shape[1] == mapped.shape[1]:
                mapped = mapped + raw
        return mapped

    def _stack_fk_features(self, fk_features, size, device, dtype):
        """Return endpoint-stacked FK features with shape (2*B, F)."""
        if self.fk_feature_dim <= 0:
            return None
        if fk_features is None:
            return torch.zeros((2 * size, self.fk_feature_dim),
                               device=device, dtype=dtype)
        fk = fk_features.to(device=device, dtype=dtype)
        if fk.dim() != 2:
            raise ValueError(f"fk_features must be 2D, got shape {tuple(fk.shape)}")
        if fk.shape[0] == size and fk.shape[1] == 2 * self.fk_feature_dim:
            return torch.cat([fk[:, :self.fk_feature_dim],
                              fk[:, self.fk_feature_dim:]], dim=0)
        if fk.shape[0] == 2 * size and fk.shape[1] == self.fk_feature_dim:
            return fk
        raise ValueError(
            f"fk_features shape {tuple(fk.shape)} is incompatible with "
            f"batch={size}, fk_feature_dim={self.fk_feature_dim}")

    def lip_norm(self, w):
        if getattr(self, 'disable_lip_norm', False):
            return w
        absrowsum = torch.sqrt(torch.sum(w ** 2, dim=1)).detach()
        scale = 1 + 1e-5 - self.act(1 - 1 / absrowsum)
        return w * scale.unsqueeze(1)

    def _run_encoder(self, x, u, v, encoder, gate):
        """Run one encoder head (cupping or pouring) through its gated SIREN blocks.

        Args:
            x: (2*B, h_size) input after pe_gate modulation is NOT applied here;
               x is the Fourier-encoded input projected through encoder[0].
            u: (2*B, h_size) modulation vector from shared pe_gate
            v: (2*B, h_size) modulation vector from shared pe_gate
            encoder: ModuleList of linear layers for this head
            gate: ModuleList of gate layers for this head

        Returns:
            x: (2*B, h_size) encoded features before norm
        """
        # NOTE: encoder[0] is a dead-weight placeholder (not used, matching single-task).
        # The SIREN loop starts at encoder[1]. pe_gate output (u,v) is the real transform.

        for ii in range(0, self.nl1):
            x_tmp = x

            w = encoder[3 * ii + 1].weight
            b = encoder[3 * ii + 1].bias
            w = self.lip_norm(w)
            y = x @ w.T + b
            x = u * torch.sin(y) + v * (1 - torch.sin(y))

            w = encoder[3 * ii + 2].weight
            b = encoder[3 * ii + 2].bias
            w = self.lip_norm(w)
            y = x @ w.T + b
            x = u * torch.sin(y) + v * (1 - torch.sin(y))

            w = encoder[3 * ii + 3].weight
            b = encoder[3 * ii + 3].bias
            w = self.lip_norm(w)
            y = x @ w.T + b

            weight = torch.sigmoid(0.1 * gate[ii].weight)
            x = (1 - weight) * x_tmp + (weight) * torch.sin(y)

        # Final layer
        w = encoder[-1].weight
        b = encoder[-1].bias
        w = self.lip_norm(w)
        x = x @ w.T + b

        return x

    def _run_endpoint_task_siamese(self, x_fourier, w_cup, w_pour):
        """Encode every endpoint with one shared network.

        ``value_condition='none'`` intentionally ignores routing metadata and
        represents one global value field.  This is the appropriate form when
        the dataset supplies one scalar speed field rather than distinct task
        fields.
        """
        if self.value_condition == 'none':
            x = x_fourier
        else:
            task = torch.stack([w_cup, w_pour], dim=1).to(x_fourier)
            task_shift = self.task_embedding_scale * torch.tanh(
                self.endpoint_task_embedding(task))
            x = x_fourier + task_shift

        w = self.lip_norm(self.pe_gate[0].weight)
        u = torch.sin(x @ w.T + self.pe_gate[0].bias)
        w = self.lip_norm(self.pe_gate[1].weight)
        v = torch.sin(x @ w.T + self.pe_gate[1].bias)
        private = self._run_encoder(
            x, u, v, self.siamese_encoder, self.siamese_gate)
        if self.partial_shared_embedding_dim > 0:
            shared = self.partial_shared_stem(x_fourier)
            private = private + self.partial_shared_value_adapter(shared)
        return private

    def _apply_partial_shared_speed(self, latent, points):
        """Inject the same q-only shared lane into F's private latent."""
        if self.partial_shared_embedding_dim <= 0:
            return latent
        if points is None:
            raise RuntimeError(
                "partially shared speed embedding requires point coordinates")
        value_fourier = self.input_mapping(points)
        shared = self.partial_shared_stem(value_fourier)
        return latent + self.partial_shared_speed_adapter(shared)

    def _blend_endpoint_value_heads(
            self, f_cup, f_pour, w_cup, w_pour):
        """Blend the two legacy task charts into one endpoint embedding.

        This is the G1 cup/pour representation transplanted into the modern
        scalar-PDE value path.  The two encoders are internal coordinate
        charts; all value architectures still consume only the single blended
        embedding and emit one scalar travel-time field.
        """
        if self.blend_mode == 'feature':
            return (w_cup.unsqueeze(1) * f_cup
                    + w_pour.unsqueeze(1) * f_pour)

        # Preserve the proven G1 R40/R55 SLERP definition exactly.  Routing is
        # pointwise, so start and goal each obtain their own continuous chart
        # embedding before the shared symmetric value operator is evaluated.
        f_cup_norm = F.normalize(f_cup, dim=-1)
        f_pour_norm = F.normalize(f_pour, dim=-1)
        cos_omega = (f_cup_norm * f_pour_norm).sum(
            dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
        omega = torch.acos(cos_omega)
        sin_omega = torch.sin(omega).clamp(min=1e-6)
        t = w_pour.unsqueeze(1)
        blended = (
            (torch.sin((1 - t) * omega) / sin_omega) * f_cup
            + (torch.sin(t * omega) / sin_omega) * f_pour)
        magnitude = 0.5 * (
            torch.norm(f_cup, dim=-1, keepdim=True)
            + torch.norm(f_pour, dim=-1, keepdim=True))
        return blended * magnitude

    def compute_window_weights(self, ee_y, d_orient_cup, d_orient_pour, ee_xy=None):
        """Compute routing weights for each endpoint.

        Supports multiple weight_mode strategies:
          'default'        - original: hard zones + softmax orient in overlap
          'sharp_boundary' - near-zero overlap (hard switch at y_divide)
          'sigmoid_y'      - smooth sigmoid on Y distance, temp=0.02
          'sigmoid_y_wide' - smooth sigmoid on Y distance, temp=0.05
          'linear_ramp'    - linear interpolation in overlap zone
          'y_plus_orient'  - 70% sigmoid_y + 30% orient softmax
          'shifted_boundary' - same as default but y_divide shifted to -0.175
          'no_overlap'     - hard split at y_divide, no overlap zone
          'orient_only'    - pure orientation routing everywhere
          'y_distance_scaled' - smooth ramp over full Y range (scale=0.15)
          'adaptive_temp'  - sigmoid with distance-dependent temperature

        Args:
            ee_y: (2*B,) EE Y-positions for all endpoints
            d_orient_cup: (2*B,) cupping orientation distance
            d_orient_pour: (2*B,) pouring orientation distance

        Returns:
            w_cup, w_pour: (2*B,) routing weights, sum to 1
        """
        device = ee_y.device
        n = ee_y.shape[0]
        mode = self.weight_mode

        if self.zone_mode == 'corner_table' and ee_xy is not None and self.corner_zones is not None:
            return corner_weights_torch(
                ee_xy, d_orient_cup, d_orient_pour,
                self.corner_zones,
                route_temp=self.route_temp,
                spatial_temp=self.corner_spatial_temp,
                smooth_routing=getattr(self, 'smooth_routing', False),
                route_alpha=getattr(self, 'route_alpha', 0.5),
                route_y_temp=getattr(self, 'route_y_temp', 0.3))

        if mode == 'no_overlap' or mode == 'sharp_boundary':
            # Hard split at y_divide, no overlap zone at all
            w_cup = (self._cup_distance(ee_y) >= 0).float()
            w_pour = 1.0 - w_cup
            return w_cup, w_pour

        if mode == 'sigmoid_y':
            # Smooth sigmoid: w_cup = σ((y_divide - ee_y) / temp)
            temp = 0.02
            w_cup = torch.sigmoid(self._cup_distance(ee_y) / temp)
            w_pour = 1.0 - w_cup
            return w_cup, w_pour

        if mode == 'sigmoid_y_wide':
            # Smooth sigmoid with wider temperature
            temp = 0.05
            w_cup = torch.sigmoid(self._cup_distance(ee_y) / temp)
            w_pour = 1.0 - w_cup
            return w_cup, w_pour

        if mode == 'linear_ramp':
            # Linear interpolation in [lower, upper], clipped
            upper = self.y_divide + self.overlap
            lower = self.y_divide - self.overlap
            width = upper - lower + 1e-8
            w_cup = torch.clamp(self._cup_distance(ee_y) / width + 0.5, 0.0, 1.0)
            w_pour = 1.0 - w_cup
            return w_cup, w_pour

        if mode == 'y_plus_orient':
            # 70% sigmoid_y + 30% orient softmax
            alpha = 0.7
            # Sigmoid component
            temp_y = 0.02
            w_cup_y = torch.sigmoid(self._cup_distance(ee_y) / temp_y)
            # Orient component: softmax on d_orient globally
            logits = torch.stack([d_orient_cup / self.route_temp,
                                  d_orient_pour / self.route_temp], dim=-1)
            weights_orient = torch.softmax(logits, dim=-1)
            w_cup = alpha * w_cup_y + (1 - alpha) * weights_orient[:, 0]
            w_pour = 1.0 - w_cup
            return w_cup, w_pour

        if mode == 'shifted_boundary':
            # Same as default but with y_divide shifted to -0.175
            y_div = -0.175
            w_cup = torch.zeros(n, device=device)
            w_pour = torch.zeros(n, device=device)
            dist = self._cup_distance(ee_y, y_div)
            cup_only = dist > self.overlap
            pour_only = dist < -self.overlap
            w_cup[cup_only] = 1.0
            w_pour[pour_only] = 1.0
            overlap_mask = (~cup_only) & (~pour_only)
            if overlap_mask.any():
                logits = torch.stack([d_orient_cup[overlap_mask] / self.route_temp,
                                      d_orient_pour[overlap_mask] / self.route_temp], dim=-1)
                weights = torch.softmax(logits, dim=-1)
                w_cup[overlap_mask] = weights[:, 0]
                w_pour[overlap_mask] = weights[:, 1]
            return w_cup, w_pour

        if mode == 'orient_only':
            # Pure orientation routing everywhere (no Y gating)
            logits = torch.stack([d_orient_cup / self.route_temp,
                                  d_orient_pour / self.route_temp], dim=-1)
            weights = torch.softmax(logits, dim=-1)
            return weights[:, 0], weights[:, 1]

        if mode == 'y_distance_scaled':
            # Smooth ramp over full Y range, scale=0.15
            scale = 0.15
            w_cup = torch.clamp(self._cup_distance(ee_y) / scale + 0.5, 0.0, 1.0)
            w_pour = 1.0 - w_cup
            return w_cup, w_pour

        # --- Round 2: y_distance_scaled refinements ---
        if mode == 'yds_scale010':
            scale = 0.10
            w_cup = torch.clamp(self._cup_distance(ee_y) / scale + 0.5, 0.0, 1.0)
            return w_cup, 1.0 - w_cup

        if mode == 'yds_scale020':
            scale = 0.20
            w_cup = torch.clamp(self._cup_distance(ee_y) / scale + 0.5, 0.0, 1.0)
            return w_cup, 1.0 - w_cup

        if mode == 'yds_scale025':
            scale = 0.25
            w_cup = torch.clamp(self._cup_distance(ee_y) / scale + 0.5, 0.0, 1.0)
            return w_cup, 1.0 - w_cup

        if mode == 'yds_scale030':
            scale = 0.30
            w_cup = torch.clamp(self._cup_distance(ee_y) / scale + 0.5, 0.0, 1.0)
            return w_cup, 1.0 - w_cup

        if mode == 'yds_ydiv175':
            # y_distance_scaled but y_divide shifted to -0.175
            scale = 0.15
            y_div = -0.175
            w_cup = torch.clamp(self._cup_distance(ee_y, y_div) / scale + 0.5, 0.0, 1.0)
            return w_cup, 1.0 - w_cup

        if mode == 'yds_ydiv225':
            # y_distance_scaled but y_divide shifted to -0.225
            scale = 0.15
            y_div = -0.225
            w_cup = torch.clamp(self._cup_distance(ee_y, y_div) / scale + 0.5, 0.0, 1.0)
            return w_cup, 1.0 - w_cup

        if mode == 'yds_ydiv175_s020':
            # Shifted boundary + wider scale
            scale = 0.20
            y_div = -0.175
            w_cup = torch.clamp(self._cup_distance(ee_y, y_div) / scale + 0.5, 0.0, 1.0)
            return w_cup, 1.0 - w_cup

        if mode == 'yds_orient_boost':
            # y_distance_scaled + orient boost in middle region
            scale = 0.15
            w_cup_y = torch.clamp(self._cup_distance(ee_y) / scale + 0.5, 0.0, 1.0)
            # In middle band (0.3 < w_cup_y < 0.7), boost with orient
            mid_mask = (w_cup_y > 0.3) & (w_cup_y < 0.7)
            w_cup = w_cup_y.clone()
            if mid_mask.any():
                logits = torch.stack([d_orient_cup[mid_mask] / self.route_temp,
                                      d_orient_pour[mid_mask] / self.route_temp], dim=-1)
                orient_w = torch.softmax(logits, dim=-1)[:, 0]
                # Blend: 60% Y + 40% orient in middle band
                w_cup[mid_mask] = 0.6 * w_cup_y[mid_mask] + 0.4 * orient_w
            return w_cup, 1.0 - w_cup

        if mode == 'yds_quadratic':
            # Quadratic (smooth) version of y_distance_scaled
            scale = 0.15
            t = torch.clamp(self._cup_distance(ee_y) / scale + 0.5, 0.0, 1.0)
            # Smoothstep: 3t^2 - 2t^3 (smoother transitions at boundaries)
            w_cup = t * t * (3.0 - 2.0 * t)
            return w_cup, 1.0 - w_cup

        if mode == 'yds_asymmetric':
            # Asymmetric: faster transition on cupping side, gentler on pouring side
            # Cupping dominates when clearly in cupping zone, pouring needs more Y-distance
            cup_scale = 0.10  # Sharp cupping boundary
            pour_scale = 0.20  # Gentle pouring transition
            diff = self._cup_distance(ee_y)
            w_cup = torch.where(
                diff > 0,
                torch.clamp(diff / cup_scale + 0.5, 0.5, 1.0),
                torch.clamp(diff / pour_scale + 0.5, 0.0, 0.5))
            return w_cup, 1.0 - w_cup

        # --- Round 3: asymmetric refinements ---
        if mode == 'asym_c008_p025':
            # Sharper cup (0.08), wider pour (0.25)
            cup_scale, pour_scale = 0.08, 0.25
            diff = self._cup_distance(ee_y)
            w_cup = torch.where(diff > 0,
                torch.clamp(diff / cup_scale + 0.5, 0.5, 1.0),
                torch.clamp(diff / pour_scale + 0.5, 0.0, 0.5))
            return w_cup, 1.0 - w_cup

        if mode == 'asym_c010_p030':
            # Same cup sharpness, even wider pour (match scale030's pour)
            cup_scale, pour_scale = 0.10, 0.30
            diff = self._cup_distance(ee_y)
            w_cup = torch.where(diff > 0,
                torch.clamp(diff / cup_scale + 0.5, 0.5, 1.0),
                torch.clamp(diff / pour_scale + 0.5, 0.0, 0.5))
            return w_cup, 1.0 - w_cup

        if mode == 'asym_c012_p025':
            # Slightly softer cup (0.12), wider pour (0.25)
            cup_scale, pour_scale = 0.12, 0.25
            diff = self._cup_distance(ee_y)
            w_cup = torch.where(diff > 0,
                torch.clamp(diff / cup_scale + 0.5, 0.5, 1.0),
                torch.clamp(diff / pour_scale + 0.5, 0.0, 0.5))
            return w_cup, 1.0 - w_cup

        if mode == 'asym_c015_p030':
            # Original yds scale for cup, wide pour
            cup_scale, pour_scale = 0.15, 0.30
            diff = self._cup_distance(ee_y)
            w_cup = torch.where(diff > 0,
                torch.clamp(diff / cup_scale + 0.5, 0.5, 1.0),
                torch.clamp(diff / pour_scale + 0.5, 0.0, 0.5))
            return w_cup, 1.0 - w_cup

        if mode == 'asym_c008_p020':
            # Very sharp cup (0.08), same pour (0.20) as winner
            cup_scale, pour_scale = 0.08, 0.20
            diff = self._cup_distance(ee_y)
            w_cup = torch.where(diff > 0,
                torch.clamp(diff / cup_scale + 0.5, 0.5, 1.0),
                torch.clamp(diff / pour_scale + 0.5, 0.0, 0.5))
            return w_cup, 1.0 - w_cup

        if mode == 'asym_c010_p015':
            # Sharp cup (0.10), tighter pour (0.15) — more separation
            cup_scale, pour_scale = 0.10, 0.15
            diff = self._cup_distance(ee_y)
            w_cup = torch.where(diff > 0,
                torch.clamp(diff / cup_scale + 0.5, 0.5, 1.0),
                torch.clamp(diff / pour_scale + 0.5, 0.0, 0.5))
            return w_cup, 1.0 - w_cup

        if mode == 'asym_c010_p025':
            # Winner cup (0.10), between winner and scale030 pour (0.25)
            cup_scale, pour_scale = 0.10, 0.25
            diff = self._cup_distance(ee_y)
            w_cup = torch.where(diff > 0,
                torch.clamp(diff / cup_scale + 0.5, 0.5, 1.0),
                torch.clamp(diff / pour_scale + 0.5, 0.0, 0.5))
            return w_cup, 1.0 - w_cup

        if mode == 'asym_c005_p020':
            # Extreme sharp cup (0.05), same pour as winner
            cup_scale, pour_scale = 0.05, 0.20
            diff = self._cup_distance(ee_y)
            w_cup = torch.where(diff > 0,
                torch.clamp(diff / cup_scale + 0.5, 0.5, 1.0),
                torch.clamp(diff / pour_scale + 0.5, 0.0, 0.5))
            return w_cup, 1.0 - w_cup

        if mode == 'asym_c010_p020_ydiv225':
            # Winner formula but y_divide shifted to -0.225 (more cupping space)
            cup_scale, pour_scale = 0.10, 0.20
            y_div = -0.225
            diff = self._cup_distance(ee_y, y_div)
            w_cup = torch.where(diff > 0,
                torch.clamp(diff / cup_scale + 0.5, 0.5, 1.0),
                torch.clamp(diff / pour_scale + 0.5, 0.0, 0.5))
            return w_cup, 1.0 - w_cup

        # --- Round 5: shifted overlap into pour zone ---
        if mode == 'asym_shift_p015':
            # y_divide=-0.15, cup=0.10, pour=0.20
            # Cupping pure zone extends to -0.25 (wider than before)
            # Overlap starts at -0.15, covering entire pouring eval range
            cup_scale, pour_scale = 0.10, 0.20
            y_div = -0.15
            diff = self._cup_distance(ee_y, y_div)
            w_cup = torch.where(diff > 0,
                torch.clamp(diff / cup_scale + 0.5, 0.5, 1.0),
                torch.clamp(diff / pour_scale + 0.5, 0.0, 0.5))
            return w_cup, 1.0 - w_cup

        if mode == 'asym_shift_p0175':
            # y_divide=-0.175, cup=0.10, pour=0.20
            # Moderate shift — cupping pure zone to -0.225
            cup_scale, pour_scale = 0.10, 0.20
            y_div = -0.175
            diff = self._cup_distance(ee_y, y_div)
            w_cup = torch.where(diff > 0,
                torch.clamp(diff / cup_scale + 0.5, 0.5, 1.0),
                torch.clamp(diff / pour_scale + 0.5, 0.0, 0.5))
            return w_cup, 1.0 - w_cup

        if mode == 'asym_shift_p015_wp030':
            # y_divide=-0.15, cup=0.10, pour=0.30 (extra wide pour transition)
            # Even more gradual pouring constraint onset
            cup_scale, pour_scale = 0.10, 0.30
            y_div = -0.15
            diff = self._cup_distance(ee_y, y_div)
            w_cup = torch.where(diff > 0,
                torch.clamp(diff / cup_scale + 0.5, 0.5, 1.0),
                torch.clamp(diff / pour_scale + 0.5, 0.0, 0.5))
            return w_cup, 1.0 - w_cup

        if mode == 'asym_shift_p0175_wp025':
            # y_divide=-0.175, cup=0.10, pour=0.25
            cup_scale, pour_scale = 0.10, 0.25
            y_div = -0.175
            diff = self._cup_distance(ee_y, y_div)
            w_cup = torch.where(diff > 0,
                torch.clamp(diff / cup_scale + 0.5, 0.5, 1.0),
                torch.clamp(diff / pour_scale + 0.5, 0.0, 0.5))
            return w_cup, 1.0 - w_cup

        if mode == 'asym_shift_p015_wp025':
            # y_divide=-0.15, cup=0.10, pour=0.25
            cup_scale, pour_scale = 0.10, 0.25
            y_div = -0.15
            diff = self._cup_distance(ee_y, y_div)
            w_cup = torch.where(diff > 0,
                torch.clamp(diff / cup_scale + 0.5, 0.5, 1.0),
                torch.clamp(diff / pour_scale + 0.5, 0.0, 0.5))
            return w_cup, 1.0 - w_cup

        if mode == 'adaptive_temp':
            # Sigmoid with distance-dependent temperature
            # Sharp near boundary, smoother far away
            dist = torch.abs(ee_y - self.y_divide)
            temp = 0.01 + 0.04 * dist
            w_cup = torch.sigmoid(self._cup_distance(ee_y) / temp)
            w_pour = 1.0 - w_cup
            return w_cup, w_pour

        # mode == 'default': original behavior
        w_cup = torch.zeros(n, device=device)
        w_pour = torch.zeros(n, device=device)

        dist = self._cup_distance(ee_y)

        cup_only = dist > self.overlap
        w_cup[cup_only] = 1.0
        w_pour[cup_only] = 0.0

        pour_only = dist < -self.overlap
        w_cup[pour_only] = 0.0
        w_pour[pour_only] = 1.0

        overlap_mask = (~cup_only) & (~pour_only)
        if overlap_mask.any():
            logits_cup = d_orient_cup[overlap_mask] / self.route_temp
            logits_pour = d_orient_pour[overlap_mask] / self.route_temp
            logits = torch.stack([logits_cup, logits_pour], dim=-1)
            weights = torch.softmax(logits, dim=-1)
            w_cup[overlap_mask] = weights[:, 0]
            w_pour[overlap_mask] = weights[:, 1]

        return w_cup, w_pour

    def _sym_op(self, f, size, encoder_norm=None):
        """Apply encoder_norm + split + smooth-L2 + grouped LogSumExp.

        Args:
            f: (2*B, h_size) features (start stacked with goal)
            size: B (batch size)
            encoder_norm: optional per-head InstanceNorm1d (uses shared if None)

        Returns:
            tau: (B, 1) scalar travel time
        """
        if encoder_norm is None:
            encoder_norm = self.encoder_norm
        y = encoder_norm(f)
        x0_out = y[:size, ...]
        x1_out = y[size:, ...]
        x = torch.sqrt((x0_out - x1_out) ** 2 + 1e-6)
        x = x.view(x.shape[0], -1, 16)
        x = (torch.logsumexp(10 * x, 2) - np.log(16)) / 10
        x = 0.1 * (torch.sum(x, dim=1, keepdim=True))
        return x

    def _zero_boundary_sym_op(self, f, size):
        """Shared latent metric with the boundary ``T(q, q)=0`` exactly.

        The legacy smooth absolute value has a small positive offset at zero.
        Subtracting its analytic offset keeps the same smooth gradients while
        making the Eikonal boundary condition structural rather than learned.
        """
        y = self.encoder_norm(f)
        delta = y[:size] - y[size:]
        eps = 1e-6
        x = torch.sqrt(delta.square() + eps) - math.sqrt(eps)
        x = x.view(x.shape[0], -1, 16)
        x = (torch.logsumexp(10.0 * x, dim=2) - math.log(16.0)) / 10.0
        return 0.1 * x.sum(dim=1, keepdim=True)

    def _global_residual_tau(self, coords, features, size):
        """Goal-conditioned Eikonal value with a stable straight-line prior."""
        normalized = self.encoder_norm(features)
        z0 = normalized[:size]
        z1 = normalized[size:]
        delta = coords[:, self.dim:] - coords[:, :self.dim]
        distance = torch.linalg.vector_norm(
            delta, dim=1, keepdim=True)
        if self.value_arch == 'global_residual_symmetric':
            # A scalar isotropic speed is reversible, so the viscosity
            # solution obeys T(q,g)=T(g,q).  Make that PDE symmetry exact.
            pair_parts = [
                z0 + z1, (z1 - z0).abs(), z0 * z1,
                delta.abs(), distance,
            ]
        else:
            pair_parts = [
                z0, z1, z1 - z0, z0 * z1,
                delta, delta.abs(), distance,
            ]
        if self.use_speed_head:
            endpoint_points = torch.cat([
                coords[:, :self.dim], coords[:, self.dim:]], dim=0)
            speed = self.predict_speed(
                endpoint_points, features=features)
            s0 = speed[:size]
            s1 = speed[size:]
            if self.value_arch == 'global_residual_symmetric':
                pair_parts.extend([s0 + s1, (s1 - s0).abs()])
            else:
                pair_parts.extend([s0, s1])
        pair = torch.cat(pair_parts, dim=1)
        raw_residual = torch.tanh(self._global_pair_output(pair))
        if self.global_residual_positive:
            # F(q)<=1 implies T(q,g)>=||q-g||.  Squaring makes that physical
            # lower bound structural and keeps the detour bounded.
            residual = self.global_residual_scale * raw_residual.square()
        else:
            residual = self.global_residual_scale * raw_residual
        return distance * torch.exp(residual)

    def _endpoint_eikonal_residual_tau(self, coords, features, size):
        """Smooth global viscosity value with an exact local Eikonal limit.

        For an isotropic scalar speed, symmetry and the boundary condition
        imply T(q, q+dq)=||dq||/F(q)+o(||dq||).  The endpoint slowness below
        enforces that first-order term exactly.  Its influence is allowed to
        decay with pair separation, while a non-negative O(distance**2)
        residual learns global detours from the PDE.  Unlike a chord-bound
        value, no privileged straight path or intermediate speed samples are
        inserted into T, so a sharp local speed boundary does not create a
        jagged straight-line texture in the global viscosity solution.
        """
        normalized = self.encoder_norm(features)
        z0 = normalized[:size]
        z1 = normalized[size:]
        q0 = coords[:, :self.dim]
        q1 = coords[:, self.dim:]
        delta = q1 - q0
        distance = torch.linalg.vector_norm(
            delta, dim=1, keepdim=True)

        endpoint_points = torch.cat([q0, q1], dim=0)
        endpoint_speed_latent = None
        if self.couple_speed_latent_to_value:
            endpoint_speed_latent = self.speed_encoder_norm(
                self._encode_speed_points(endpoint_points))
            for block in getattr(self, 'speed_residual_blocks', []):
                endpoint_speed_latent = block(endpoint_speed_latent)
            endpoint_speed_latent = self._apply_partial_shared_speed(
                endpoint_speed_latent, endpoint_points)
            speed_logit = self._speed_logit_from_latent(
                endpoint_speed_latent, endpoint_points)
            speed = self.speed_floor + (
                1.0 - self.speed_floor) * torch.sigmoid(speed_logit)
        else:
            speed = self.predict_speed(endpoint_points, features=features)
        s0 = speed[:size]
        s1 = speed[size:]
        pair_parts = [
            z0 + z1,
            (z1 - z0).abs(),
            z0 * z1,
            delta.abs(),
            distance,
            s0 + s1,
            (s1 - s0).abs(),
        ]
        if endpoint_speed_latent is not None:
            f0 = endpoint_speed_latent[:size]
            f1 = endpoint_speed_latent[size:]
            pair_parts.extend([
                f0 + f1,
                (f1 - f0).abs(),
                f0 * f1,
            ])
        pair = torch.cat(pair_parts, dim=1)
        raw = self._global_pair_output(pair)
        if self.value_arch == 'local_endpoint_eikonal_residual_symmetric':
            # Preserve the exact D/F endpoint asymptotic while preventing the
            # endpoint speed from exerting a radial force over an entire
            # start--goal pair.  The fixed exponential has e-folding radius
            # ``endpoint_barrier_radius`` in normalized C-space.  Away from
            # that local cell, only the learned non-negative O(D^2) viscosity
            # excess can shape the global field.
            attenuation_rate = raw.new_full(
                (raw.shape[0], 1),
                1.0 / max(float(self.endpoint_barrier_radius), 1e-6))
            global_raw = raw[:, :1]
        else:
            attenuation_rate = F.softplus(raw[:, :1])
            global_raw = raw[:, 1:2]
        global_excess = (
            self.global_residual_scale * torch.tanh(global_raw).square())
        endpoint_excess = (
            0.5 * (s0.clamp_min(self.speed_floor).reciprocal()
                   + s1.clamp_min(self.speed_floor).reciprocal())
            - 1.0).clamp_min(0.0)
        effective_slowness = (
            1.0
            + endpoint_excess * torch.exp(-distance * attenuation_rate)
            + distance * global_excess)
        return distance * effective_slowness

    def _factored_source_eikonal_tau(self, coords, features, size):
        """Symmetric source-factorized value with exact local Eikonal slope.

        The analytic source term blends the normalized free-space slowness
        with the reciprocal speeds at both endpoints over one local radius.
        Crucially, these speeds remain attached to the same differentiable
        q-only scalar speed head used by deployment, so autograd includes the
        chain-rule terms that the historical detached-label prototype missed.
        A bounded symmetric residual acts only through ``exp(D * R)``; it is
        therefore exactly neutral on the diagonal and cannot change
        ``T(q, q)=0`` or the first-order ``D/F(q)`` asymptotic.
        """
        normalized = self.encoder_norm(features)
        z0 = normalized[:size]
        z1 = normalized[size:]
        q0 = coords[:, :self.dim]
        q1 = coords[:, self.dim:]
        delta = q1 - q0
        distance = torch.linalg.vector_norm(delta, dim=1, keepdim=True)

        endpoint_points = torch.cat([q0, q1], dim=0)
        endpoint_speed_latent = None
        if self.couple_speed_latent_to_value:
            endpoint_speed_latent = self.speed_encoder_norm(
                self._encode_speed_points(endpoint_points))
            for block in getattr(self, 'speed_residual_blocks', []):
                endpoint_speed_latent = block(endpoint_speed_latent)
            endpoint_speed_latent = self._apply_partial_shared_speed(
                endpoint_speed_latent, endpoint_points)
            speed_logit = self._speed_logit_from_latent(
                endpoint_speed_latent, endpoint_points)
            speed = self.speed_floor + (
                1.0 - self.speed_floor) * torch.sigmoid(speed_logit)
        else:
            speed = self.predict_speed(endpoint_points, features=features)
        s0 = speed[:size].clamp_min(self.speed_floor)
        s1 = speed[size:].clamp_min(self.speed_floor)
        endpoint_slowness = 0.5 * (s0.reciprocal() + s1.reciprocal())
        radius = max(float(self.endpoint_barrier_radius), 1e-6)
        source_time = (
            distance
            + (endpoint_slowness - 1.0)
            * radius
            * (-torch.expm1(-distance / radius)))

        pair_parts = [
            z0 + z1,
            (z1 - z0).abs(),
            z0 * z1,
            delta.abs(),
            distance,
            s0 + s1,
            (s1 - s0).abs(),
        ]
        if endpoint_speed_latent is not None:
            f0 = endpoint_speed_latent[:size]
            f1 = endpoint_speed_latent[size:]
            pair_parts.extend([
                f0 + f1,
                (f1 - f0).abs(),
                f0 * f1,
            ])
        pair = torch.cat(pair_parts, dim=1)
        residual = self.global_residual_scale * torch.tanh(
            self._global_pair_output(pair))
        return source_time * torch.exp(distance * residual)

    def _encode_scalar_points(self, points):
        """Encode arbitrary configurations with the smooth value trunk."""
        x_fourier = self.input_mapping(points)
        dummy = torch.zeros(
            points.shape[0], device=points.device, dtype=points.dtype)
        return self._run_endpoint_task_siamese(x_fourier, dummy, dummy)

    def _encode_speed_points(self, points):
        """Encode configurations with the local multiscale speed trunk."""
        if not self.separate_speed_encoder:
            return self._encode_scalar_points(points)
        x = self.input_mapping(points, self.B_speed)
        w = self.lip_norm(self.speed_pe_gate[0].weight)
        u = torch.sin(x @ w.T + self.speed_pe_gate[0].bias)
        w = self.lip_norm(self.speed_pe_gate[1].weight)
        v = torch.sin(x @ w.T + self.speed_pe_gate[1].bias)
        return self._run_encoder(
            x, u, v, self.speed_encoder, self.speed_gate)

    def _encode_speed_constraint_anchor_points(self, points):
        """Encode q with the frozen, private Phase-CJ anchor trunk."""
        if self.speed_constraint_consensus_mode not in ROUTE_CHART_ANCHOR_MODES:
            raise RuntimeError("semantic-approved scalar anchor is disabled")
        if points is None:
            raise ValueError("semantic-approved scalar anchor requires q")
        x = self.input_mapping(points, self.B_speed)
        w = self.lip_norm(
            self.speed_constraint_anchor_pe_gate[0].weight)
        u = torch.sin(
            x @ w.T + self.speed_constraint_anchor_pe_gate[0].bias)
        w = self.lip_norm(
            self.speed_constraint_anchor_pe_gate[1].weight)
        v = torch.sin(
            x @ w.T + self.speed_constraint_anchor_pe_gate[1].bias)
        return self._run_encoder(
            x, u, v,
            self.speed_constraint_anchor_encoder,
            self.speed_constraint_anchor_gate)

    def _speed_constraint_anchor_hidden(self, points):
        """Return the frozen anchor latent shared only as reliability input."""
        latent = self.speed_encoder_norm(
            self._encode_speed_constraint_anchor_points(points))
        for block in self.speed_constraint_anchor_residual_blocks:
            latent = block(latent)
        return latent

    def _predict_speed_constraint_anchor(self, points):
        """Return the frozen formal-leader scalar used only as a proposal."""
        latent = self._speed_constraint_anchor_hidden(points)
        logit = self.speed_constraint_anchor_head(latent)
        return self.speed_floor + (
            1.0 - self.speed_floor) * torch.sigmoid(logit)

    def _speed_constraint_anchor_reliability_q_features(self, points):
        """Return standardized q plus the fixed four-band Fourier features."""
        if self.speed_constraint_consensus_mode != (
                ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_REPLICA_MEDIAN_ANCHOR_RESCUE_CONSENSUS_MODE):
            raise RuntimeError("anchor replica reliability is disabled")
        if points.ndim != 2 or points.shape[1] != self.dim:
            raise ValueError(
                f"anchor reliability q must have shape (N,{self.dim})")
        standardized = (
            (points - self.speed_constraint_anchor_reliability_q_mean)
            / self.speed_constraint_anchor_reliability_q_std.clamp_min(1e-4))
        frequencies = points.new_tensor(
            ROUTE_CHART_FOURIER_RISK_FREQUENCIES)
        phase = (
            math.pi * standardized[:, :, None]
            * frequencies[None, None, :])
        return torch.cat([
            standardized,
            torch.sin(phase).flatten(start_dim=1),
            torch.cos(phase).flatten(start_dim=1),
        ], dim=1)

    def _speed_constraint_anchor_reliability_logits(
            self, points, anchor_hidden, context):
        """Return three independent five-head q-only reliability tensors."""
        if self.speed_constraint_consensus_mode != (
                ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_REPLICA_MEDIAN_ANCHOR_RESCUE_CONSENSUS_MODE):
            raise RuntimeError("anchor replica reliability is disabled")
        if context.shape != (points.shape[0], 8):
            raise ValueError(
                "anchor reliability context must have shape (N,8)")
        q_features = self._speed_constraint_anchor_reliability_q_features(
            points)
        return torch.stack([
            replica(anchor_hidden, q_features, context)
            for replica in self.speed_constraint_anchor_reliability_replicas
        ])

    def configure_fullsemantic_robot_model(
            self, robot_model, q_center, q_range):
        """Attach the pinned FK evaluator required by the Phase-DF scalar F.

        Only the runtime evaluator is attached here.  All learned tensors,
        normalization, scene primitives, radii, and relation indices remain
        serialized in the checkpoint.  The supplied normalization must match
        those immutable buffers before any prediction is allowed.
        """
        if self.speed_constraint_consensus_mode not in ROUTE_CHART_LIVE_FK_MODES:
            raise RuntimeError("full-semantic relation gate is disabled")
        if robot_model is None:
            raise ValueError("full-semantic relation gate requires a robot model")
        center = torch.as_tensor(
            q_center,
            device=self.speed_constraint_fullsemantic_q_center.device,
            dtype=self.speed_constraint_fullsemantic_q_center.dtype).reshape(-1)
        span = torch.as_tensor(
            q_range,
            device=self.speed_constraint_fullsemantic_q_range.device,
            dtype=self.speed_constraint_fullsemantic_q_range.dtype).reshape(-1)
        if center.shape != (self.dim,) or span.shape != (self.dim,):
            raise ValueError(
                f"full-semantic normalization must have shape ({self.dim},)")
        if (not torch.allclose(
                    center, self.speed_constraint_fullsemantic_q_center,
                    atol=1e-7, rtol=0.0)
                or not torch.allclose(
                    span, self.speed_constraint_fullsemantic_q_range,
                    atol=1e-7, rtol=0.0)):
            raise ValueError("full-semantic joint normalization mismatch")
        object.__setattr__(
            self, '_speed_constraint_fullsemantic_robot_model', robot_model)

    def _speed_constraint_fullsemantic_kinematics(self, points):
        """Return FK8 and ordered 53-sphere centers for normalized q."""
        if self.speed_constraint_consensus_mode not in ROUTE_CHART_LIVE_FK_MODES:
            raise RuntimeError("full-semantic relation gate is disabled")
        robot_model = getattr(
            self, '_speed_constraint_fullsemantic_robot_model', None)
        if robot_model is None:
            raise RuntimeError(
                "full-semantic relation gate is not configured with the "
                "pinned robot model")
        if points is None or points.ndim != 2 or points.shape[1] != self.dim:
            raise ValueError(
                f"full-semantic q must have shape (N,{self.dim})")
        fk_parts = []
        sphere_parts = []
        geometry_chunk = 8192
        for begin in range(0, points.shape[0], geometry_chunk):
            end = min(begin + geometry_chunk, points.shape[0])
            q = points[begin:end]
            q_rad = (
                q * self.speed_constraint_fullsemantic_q_range[None, :]
                + self.speed_constraint_fullsemantic_q_center[None, :])
            state = robot_model.get_state(q_rad)
            spheres = state.link_spheres_tensor
            expected = (
                end - begin, FULLSEMANTIC_SPHERE_COUNT, 4)
            if tuple(spheres.shape) != expected:
                raise ValueError(
                    "full-semantic robot sphere shape mismatch: "
                    f"{tuple(spheres.shape)} != {expected}")
            radius_error = torch.max(torch.abs(
                spheres[..., 3]
                - self.speed_constraint_fullsemantic_radii[None, :]))
            if float(radius_error.detach().cpu()) > 1e-6:
                raise ValueError("full-semantic robot sphere radii mismatch")
            fk_parts.append(fullsemantic_fk8_from_pose(
                state.ee_pose.position, state.ee_pose.quaternion))
            sphere_parts.append(spheres[..., :3].reshape(end - begin, -1))
        return torch.cat(fk_parts, dim=0), torch.cat(sphere_parts, dim=0)

    def _speed_constraint_fullsemantic_logits(
            self, points, anchor_hidden, context):
        """Return the three frozen four-head Phase-DF logit tensors."""
        if anchor_hidden.shape[0] != points.shape[0]:
            raise ValueError("anchor hidden and q batch sizes differ")
        if context.shape != (points.shape[0], 8):
            raise ValueError("full-semantic scalar context must have shape (N,8)")
        fk8, spheres = self._speed_constraint_fullsemantic_kinematics(points)
        obstacle, self_relation = fullsemantic_relation_features(
            spheres,
            self.speed_constraint_fullsemantic_radii,
            self.speed_constraint_fullsemantic_cuboid_centers,
            self.speed_constraint_fullsemantic_cuboid_half,
            self.speed_constraint_fullsemantic_self_pairs,
            self.speed_constraint_fullsemantic_temperatures)
        boundary = self.speed_constraint_fullsemantic_component_boundary
        obstacle_threshold = FULLSEMANTIC_FREE_DISTANCE_M * boundary
        self_threshold = obstacle_threshold / FULLSEMANTIC_SELF_SCALE
        relation_scale = (
            self.speed_constraint_fullsemantic_relation_scale.clamp_min(1e-8))
        obstacle = (obstacle - obstacle_threshold) / relation_scale
        self_relation = (self_relation - self_threshold) / relation_scale
        cup_relation, pour_relation = fullsemantic_orientation_relations(
            fk8, self.corner_zones, boundary)
        q_standardized = (
            (points - self.speed_constraint_fullsemantic_q_mean)
            / self.speed_constraint_fullsemantic_q_std.clamp_min(1e-4))
        fk8_standardized = (
            (fk8 - self.speed_constraint_fullsemantic_fk8_mean)
            / self.speed_constraint_fullsemantic_fk8_std.clamp_min(1e-4))
        return torch.stack([
            replica(
                anchor_hidden, q_standardized, context,
                fk8_standardized, obstacle, self_relation,
                cup_relation, pour_relation)
            for replica in self.speed_constraint_fullsemantic_replicas
        ])

    def _speed_constraint_fullsemantic_evidence(
            self, points, anchor_hidden, context):
        """Weakest head within member, then median across three members."""
        logits = self._speed_constraint_fullsemantic_logits(
            points, anchor_hidden, context)
        if logits.shape != (
                FULLSEMANTIC_REPLICA_COUNT, points.shape[0],
                FULLSEMANTIC_HEAD_COUNT):
            raise RuntimeError("full-semantic replica output shape mismatch")
        per_replica = logits.min(dim=2).values
        evidence = per_replica.median(dim=0).values[:, None]
        return evidence, logits

    def _speed_constraint_analytic_continuous_margins(self, points):
        """Return four exact signed margins in common component units."""
        if self.speed_constraint_consensus_mode not in (
                ROUTE_CHART_ANALYTIC_CONTINUOUS_MODES):
            raise RuntimeError("analytic continuous semantic gate is disabled")
        fk8, spheres = self._speed_constraint_fullsemantic_kinematics(points)
        obstacle, self_clearance = fullsemantic_relation_minima(
            spheres,
            self.speed_constraint_fullsemantic_radii,
            self.speed_constraint_fullsemantic_cuboid_centers,
            self.speed_constraint_fullsemantic_cuboid_half,
            self.speed_constraint_fullsemantic_self_pairs)
        boundary = self.speed_constraint_fullsemantic_component_boundary
        obstacle_component = obstacle.clamp(
            -FULLSEMANTIC_FREE_DISTANCE_M,
            FULLSEMANTIC_FREE_DISTANCE_M) / FULLSEMANTIC_FREE_DISTANCE_M
        self_component = (
            FULLSEMANTIC_SELF_SCALE * self_clearance).clamp(
                -FULLSEMANTIC_FREE_DISTANCE_M,
                FULLSEMANTIC_FREE_DISTANCE_M) / FULLSEMANTIC_FREE_DISTANCE_M
        cup_margin, pour_margin = analytic_continuous_orientation_margins(
            fk8, self.corner_zones, boundary)
        return torch.stack([
            obstacle_component - boundary,
            self_component - boundary,
            cup_margin,
            pour_margin,
        ], dim=1)

    def _bounded_chord_tau(
            self, coords, features, size, cross_strength=None,
            directed_context=None, handover_phase=None):
        """Put travel time between physical lower and upper path bounds.

        With scalar isotropic speed ``0 < F <= 1``, Euclidean distance is a
        lower bound on travel time and the speed integral over the endpoint
        chord is an upper bound.  If the chord is entirely free (``F=1``),
        both bounds coincide and straight-line behavior is exact by
        construction.  All chord speeds come from this same network.
        """
        normalized = self.encoder_norm(features)
        z0 = normalized[:size]
        z1 = normalized[size:]
        q0 = coords[:, :self.dim]
        q1 = coords[:, self.dim:]
        delta = q1 - q0
        distance = torch.linalg.vector_norm(
            delta, dim=1, keepdim=True)
        if self.speed_constraint_consensus_mode in (
                ANALYTIC_HEADWISE_INTEGRATED_T_MODES):
            tau, _ = self.headwise_scalar_chord_travel_times(q0, q1)
            return tau
        endpoint_points = torch.cat([q0, q1], dim=0)
        endpoint_speed_latent = None
        if (self.couple_speed_latent_to_value
                or self.viscosity_speed_latent_context):
            endpoint_speed_latent = self.speed_encoder_norm(
                self._encode_speed_points(endpoint_points))
            # This path must be exactly the same scalar-speed field as
            # ``predict_speed``.  In particular, later speed-only refinements
            # are part of the deployed field; bypassing them when exposing the
            # latent silently changed endpoint speeds as soon as coupling was
            # enabled.  Keep the refined latent for the value context and run
            # the identical scalar output mapping here.
            for block in getattr(self, 'speed_residual_blocks', []):
                endpoint_speed_latent = block(endpoint_speed_latent)
            endpoint_speed_latent = self._apply_partial_shared_speed(
                endpoint_speed_latent, endpoint_points)
            endpoint_speed_logit = self._speed_logit_from_latent(
                endpoint_speed_latent, endpoint_points)
            endpoint_speed = self.speed_floor + (
                1.0 - self.speed_floor) * torch.sigmoid(endpoint_speed_logit)
        else:
            endpoint_speed = self.predict_speed(
                endpoint_points, features=features)
        if (self.value_arch ==
                'bounded_chord_clearance_eikonal_symmetric'):
            # This architecture has one physical propagation field, not an
            # auxiliary clearance penalty.  The historical task speed keeps
            # orientation constraints, while signed clearance restores the
            # obstacle-depth information lost by clipping F at 0.1.
            endpoint_speed = torch.minimum(
                endpoint_speed,
                self.predict_clearance_speed(endpoint_points))
        clearance_state_depth = None
        if (self.value_arch ==
                'bounded_chord_clearance_state_symmetric'):
            endpoint_clearance = self.predict_clearance(endpoint_points)
            clearance_boundary = (
                self.clearance_boundary / self.clearance_scale)
            # State-constraint boundary coordinate. It is exactly zero for a
            # feasible endpoint and increases linearly with signed penetration
            # depth.  Its gradient is learned solely from scalar clearance.
            clearance_state_depth = (
                (clearance_boundary - endpoint_clearance)
                / max(clearance_boundary, 1e-8)).clamp_min(0.0)
        s0 = endpoint_speed[:size]
        s1 = endpoint_speed[size:]
        pair_parts = [
            z0 + z1,
            (z1 - z0).abs(),
            z0 * z1,
            delta.abs(),
            distance,
            s0 + s1,
            (s1 - s0).abs(),
        ]
        viscosity_speed_context = []
        if endpoint_speed_latent is not None:
            f0 = endpoint_speed_latent[:size]
            f1 = endpoint_speed_latent[size:]
            speed_pair_parts = [
                f0 + f1,
                (f1 - f0).abs(),
                f0 * f1,
            ]
            if self.couple_speed_latent_to_value:
                pair_parts.extend(speed_pair_parts)
            elif self.viscosity_speed_latent_context:
                viscosity_speed_context.extend(speed_pair_parts)

        # Gauss-Legendre quadrature on [0, 1].  Symmetric nodes preserve
        # T(q, g) == T(g, q) exactly up to floating-point roundoff.
        nodes = self.chord_nodes.to(dtype=coords.dtype)
        weights = self.chord_weights.to(dtype=coords.dtype)
        chord_points = (
            q0[:, None, :] + nodes[None, :, None] * delta[:, None, :])
        chord_flat = chord_points.reshape(-1, self.dim)
        chord_speed_latent = None
        # Large MPPI batches can contain hundreds of thousands of quadrature
        # states.  A high-frequency independent scalar-speed encoder is still
        # a pure q-only field, but evaluating all of those states in one
        # matrix can exceed the inference GPU memory.  Chunk only the
        # no-grad inference path; training retains the original single call
        # (and therefore identical autograd semantics).  Concatenating the
        # scalar outputs preserves the exact chord quadrature and symmetry.
        eval_chunk = 65536
        chunked_eval = (
            not torch.is_grad_enabled()
            and chord_flat.shape[0] > eval_chunk)
        need_chord_speed_latent = bool(
            self.viscosity_chord_speed_latent_context
            or self.pair_slowness_attention_dim > 0)
        if need_chord_speed_latent:
            if chunked_eval:
                speed_parts = []
                latent_parts = []
                for begin in range(0, chord_flat.shape[0], eval_chunk):
                    end = min(begin + eval_chunk, chord_flat.shape[0])
                    speed_part, latent_part = (
                        self._predict_propagation_speed_with_latent(
                            chord_flat[begin:end]))
                    speed_parts.append(speed_part)
                    latent_parts.append(latent_part)
                chord_speed = torch.cat(speed_parts, dim=0)
                chord_speed_latent = torch.cat(latent_parts, dim=0)
            else:
                chord_speed, chord_speed_latent = (
                    self._predict_propagation_speed_with_latent(chord_flat))
            chord_speed = chord_speed.reshape(
                size, self.chord_quad_points)
            chord_speed_latent = chord_speed_latent.reshape(
                size, self.chord_quad_points, self.speed_feature_dim)
        else:
            if chunked_eval:
                speed_parts = []
                for begin in range(0, chord_flat.shape[0], eval_chunk):
                    end = min(begin + eval_chunk, chord_flat.shape[0])
                    speed_parts.append(
                        self.predict_propagation_speed(
                            chord_flat[begin:end]))
                chord_speed = torch.cat(speed_parts, dim=0)
            else:
                chord_speed = self.predict_propagation_speed(chord_flat)
            chord_speed = chord_speed.reshape(size, self.chord_quad_points)
        chord_slowness = chord_speed.clamp_min(self.speed_floor).reciprocal()
        mean_slowness = (
            chord_slowness * weights[None, :]).sum(dim=1, keepdim=True)
        chord_time = distance * mean_slowness

        if self.pair_slowness_attention_dim > 0:
            if chord_speed_latent is None:
                raise RuntimeError(
                    "pair slowness attention is missing chord F latents")
            pair_query_input = torch.cat(pair_parts, dim=1)
            query = self.pair_slowness_attention_query(
                pair_query_input)[:, None, :]
            # rho(t)=1-|2t-1| is invariant under endpoint reversal. The
            # physical chord tokens then only permute when q0 and q1 swap,
            # so the attention reduction is exactly symmetric in theory.
            symmetric_position = (
                1.0 - (2.0 * nodes - 1.0).abs()).clamp(0.0, 1.0)
            position = symmetric_position[None, :, None].expand(
                size, -1, -1)
            key_input = torch.cat([
                chord_speed_latent,
                chord_slowness[:, :, None],
                position,
            ], dim=2)
            key = self.pair_slowness_attention_key(key_input)
            score = (query * key).sum(dim=2) / math.sqrt(
                float(self.pair_slowness_attention_dim))
            # The quadrature weights form the prior attention distribution.
            # With equal learned compatibility, the summary therefore
            # reduces to the same physical integration measure used by the
            # chord upper bound rather than an arbitrary uniform token mean.
            score = score + torch.log(
                weights.clamp_min(torch.finfo(weights.dtype).tiny))[None, :]
            attention = torch.softmax(score, dim=1)
            hazard = (1.0 - chord_speed).clamp(0.0, 1.0)
            values = torch.stack([
                chord_slowness,
                torch.log(chord_slowness.clamp_min(1.0)),
                hazard,
                position[:, :, 0],
            ], dim=2)
            attention_context = (
                attention[:, :, None] * values).sum(dim=1)
            pair_parts.append(attention_context)
            if getattr(self, 'capture_pair_slowness_attention', False):
                self.captured_pair_slowness_attention = attention
                self.captured_pair_slowness_values = values
        viscosity_pair = None
        viscosity_symmetric_pair = None
        viscosity_reverse_pair = None
        causal_handover_pair = None
        transverse_contrast = None
        if getattr(self, 'use_viscosity_excess_head', False):
            # A reversal-invariant representation of the scalar slowness
            # profile.  Reversing the endpoints swaps quadrature order, so the
            # pairwise means/absolute differences (plus an odd center) retain
            # obstacle structure without choosing an orientation.
            half = self.chord_quad_points // 2
            left = chord_slowness[:, :half]
            right = torch.flip(chord_slowness[:, -half:], dims=[1])
            profile_parts = [
                0.5 * (left + right),
                (left - right).abs(),
            ]
            if self.chord_quad_points % 2:
                profile_parts.append(
                    chord_slowness[:, half:half + 1])
            chord_latent_parts = []
            if self.viscosity_chord_speed_latent_context:
                if chord_speed_latent is None:
                    raise RuntimeError(
                        "viscosity chord context is missing F latents")
                projected = self.viscosity_chord_latent_projection(
                    chord_speed_latent.reshape(
                        -1, self.speed_feature_dim)).reshape(
                            size,
                            self.chord_quad_points,
                            self.viscosity_chord_latent_width)
                latent_left = projected[:, :half, :]
                latent_right = torch.flip(
                    projected[:, -half:, :], dims=[1])
                chord_latent_parts.extend([
                    0.5 * (latent_left + latent_right),
                    (latent_left - latent_right).abs(),
                ])
                if self.chord_quad_points % 2:
                    chord_latent_parts.append(
                        projected[:, half:half + 1, :])
                chord_latent_parts = [
                    torch.cat(chord_latent_parts, dim=1).reshape(size, -1)]
            directed_parts = []
            if self.viscosity_directed_context:
                directed_width = (
                    6 if self.viscosity_handover_phase_context else 3)
                if directed_context is None:
                    directed_context = coords.new_zeros(
                        (size, directed_width))
                directed_context = directed_context.to(
                    device=coords.device, dtype=coords.dtype)
                if directed_context.shape != (size, directed_width):
                    raise ValueError(
                        "directed viscosity context must have shape "
                        f"({size}, {directed_width}), got "
                        f"{tuple(directed_context.shape)}")
                directed_parts.append(torch.cat([
                    z1 - z0,
                    delta,
                    s1 - s0,
                    directed_context,
                ], dim=1))
            transverse_parts = []
            probe_radius = max(
                self.viscosity_stencil_radius, self.viscosity_gate_radius)
            if probe_radius > 0.0:
                midpoint = 0.5 * (q0 + q1)
                unit = delta / distance.clamp_min(1e-6)
                eye = torch.eye(
                    self.dim, device=coords.device, dtype=coords.dtype)
                # Rows of I-u u^T are the fixed joint axes projected onto the
                # hyperplane transverse to the endpoint chord.  Reversing the
                # pair changes u to -u and therefore leaves this stencil
                # exactly unchanged.
                projection = (
                    eye[None, :, :]
                    - unit[:, :, None] * unit[:, None, :])
                projection_norm = torch.linalg.vector_norm(
                    projection, dim=2, keepdim=True)
                fallback = eye[None, :, :].expand(size, -1, -1)
                transverse = torch.where(
                    projection_norm > 1e-5,
                    projection / projection_norm.clamp_min(1e-6),
                    fallback)
                radius = probe_radius
                plus = (midpoint[:, None, :] + radius * transverse).clamp(
                    -0.5, 0.5)
                minus = (midpoint[:, None, :] - radius * transverse).clamp(
                    -0.5, 0.5)
                # A full MPPI batch can contain hundreds of thousands of
                # endpoint pairs.  The transverse probe evaluates 2*d
                # additional scalar states per pair, so materializing that
                # tensor in one call can exceed inference memory even though
                # the underlying field is unchanged.  Match the existing
                # no-grad chord-query chunking: training keeps one autograd
                # graph, while deployment concatenates numerically identical
                # chunks.
                probe_flat = plus.reshape(-1, self.dim)
                probe_minus_flat = minus.reshape(-1, self.dim)
                probe_chunk = 65536
                if (not torch.is_grad_enabled()
                        and probe_flat.shape[0] > probe_chunk):
                    plus_parts = []
                    minus_parts = []
                    for begin in range(0, probe_flat.shape[0], probe_chunk):
                        end = min(begin + probe_chunk, probe_flat.shape[0])
                        plus_parts.append(
                            self.predict_propagation_speed(
                                probe_flat[begin:end]))
                        minus_parts.append(
                            self.predict_propagation_speed(
                                probe_minus_flat[begin:end]))
                    plus_speed = torch.cat(plus_parts, dim=0).reshape(
                        size, self.dim)
                    minus_speed = torch.cat(minus_parts, dim=0).reshape(
                        size, self.dim)
                    center_speed = self.predict_propagation_speed(midpoint)
                else:
                    center_speed = self.predict_propagation_speed(midpoint)
                    plus_speed = self.predict_propagation_speed(
                        probe_flat).reshape(size, self.dim)
                    minus_speed = self.predict_propagation_speed(
                        probe_minus_flat).reshape(size, self.dim)
                # A scalar field contains directional evidence only when its
                # two sides disagree.  This normalized, reversal-invariant
                # contrast suppresses learned detours in locally symmetric
                # regions without introducing labels, normals, or a second
                # inference model.
                transverse_contrast = (
                    (plus_speed - minus_speed).abs().amax(
                        dim=1, keepdim=True)
                    / max(1.0 - self.speed_floor, 1e-6)).clamp(0.0, 1.0)
                if self.viscosity_stencil_radius > 0.0:
                    transverse_parts.append(torch.cat([
                        plus_speed.clamp_min(self.speed_floor).reciprocal(),
                        minus_speed.clamp_min(self.speed_floor).reciprocal(),
                        center_speed.clamp_min(self.speed_floor).reciprocal(),
                    ], dim=1))
            viscosity_pair = torch.cat(
                pair_parts + [torch.cat(profile_parts, dim=1),
                              mean_slowness] + viscosity_speed_context
                + chord_latent_parts + directed_parts + transverse_parts,
                dim=1)
            if self.viscosity_directed_odd:
                if not directed_parts:
                    raise RuntimeError(
                        "odd directed viscosity is missing directed context")
                directed = directed_parts[0]
                base_parts = (
                    pair_parts + [torch.cat(profile_parts, dim=1),
                                  mean_slowness])
                viscosity_symmetric_pair = torch.cat(
                    base_parts + [torch.zeros_like(directed)], dim=1)
                context_width = (
                    6 if self.viscosity_handover_phase_context else 3)
                context = directed[:, -(context_width):]
                reverse_context_parts = [
                    context[:, 1:2], context[:, 0:1],
                    -context[:, 2:3],
                ]
                if self.viscosity_handover_phase_context:
                    reverse_context_parts.extend([
                        context[:, 4:5], context[:, 3:4],
                        -context[:, 5:6],
                    ])
                reverse_context = torch.cat(
                    reverse_context_parts, dim=1)
                reverse_directed = torch.cat([
                    -(z1 - z0),
                    -delta,
                    -(s1 - s0),
                    reverse_context,
                ], dim=1)
                viscosity_reverse_pair = torch.cat(
                    base_parts + [reverse_directed], dim=1)
            if self.use_causal_handover_head:
                directed_width = (
                    6 if self.causal_handover_phase_context else 3)
                if directed_context is None:
                    directed_context = coords.new_zeros(
                        (size, directed_width))
                directed_context = directed_context.to(
                    device=coords.device, dtype=coords.dtype)
                if directed_context.shape != (size, directed_width):
                    raise ValueError(
                        "directed causal handover context must have shape "
                        f"({size}, {directed_width}), got "
                        f"{tuple(directed_context.shape)}")
                causal_directed = torch.cat([
                    z1 - z0,
                    delta,
                    s1 - s0,
                    directed_context,
                ], dim=1)
                causal_handover_pair = torch.cat(
                    pair_parts + [torch.cat(profile_parts, dim=1),
                                  mean_slowness, causal_directed],
                    dim=1)
        signed_clearance_extra = None
        if (self.value_arch ==
                'bounded_chord_signed_clearance_symmetric'):
            clearance_speed = self.predict_clearance_speed(
                chord_flat).reshape(size, self.chord_quad_points)
            # The original scalar field still carries task/orientation
            # constraints.  Signed clearance only sharpens the physical and
            # self-collision component, so taking the minimum cannot erase
            # those existing constraints.
            signed_speed = torch.minimum(chord_speed, clearance_speed)
            signed_slowness = signed_speed.clamp_min(
                self.clearance_speed_floor).reciprocal()
            signed_mean_slowness = (
                signed_slowness * weights[None, :]).sum(
                    dim=1, keepdim=True)
            signed_chord_time = distance * signed_mean_slowness
            signed_clearance_extra = (
                signed_chord_time - chord_time).clamp_min(0.0)

        if self.value_arch == 'chord_integral_symmetric':
            # Minimal Fermat model: integrate the one learned scalar slowness
            # field along the endpoint chord.  No pair/value head, candidate
            # bank, or external geometry query participates in this cost.
            return chord_time

        if self.value_arch == 'bounded_chord_profile_symmetric':
            # The previous head had to infer an obstructed chord from the two
            # endpoints alone even though this forward pass already samples
            # F along the chord.  Expose those same scalar samples without
            # breaking reversibility: paired nodes contribute their mean and
            # absolute difference, while an odd center node is unchanged.
            half = self.chord_quad_points // 2
            left = chord_slowness[:, :half]
            right = torch.flip(chord_slowness[:, -half:], dims=[1])
            profile_parts = [
                0.5 * (left + right),
                (left - right).abs(),
            ]
            if self.chord_quad_points % 2:
                profile_parts.append(chord_slowness[:, half:half + 1])
            pair_parts.extend([
                torch.cat(profile_parts, dim=1),
                mean_slowness,
            ])

        pair = torch.cat(pair_parts, dim=1)
        global_excess_pair = None
        if self.value_arch in (
                'bounded_chord_separate_smooth_excess_symmetric',
                'bounded_chord_separate_global_excess_symmetric'):
            # The independent excess branch receives the same reversible
            # chord evidence as the physical upper bound.
            half = self.chord_quad_points // 2
            left = chord_slowness[:, :half]
            right = torch.flip(chord_slowness[:, -half:], dims=[1])
            excess_profile = [
                0.5 * (left + right),
                (left - right).abs(),
            ]
            if self.chord_quad_points % 2:
                excess_profile.append(
                    chord_slowness[:, half:half + 1])
            global_excess_pair = torch.cat([
                pair,
                torch.cat(excess_profile, dim=1),
                mean_slowness,
            ], dim=1)
        raw_pair = self._global_pair_output(pair)
        separate_global_excess_raw = None
        if global_excess_pair is not None:
            separate_global_excess_raw = self.global_excess_net(
                global_excess_pair)
        viscosity_minplus_tau_modes = None
        viscosity_tau_envelope = None
        if viscosity_symmetric_pair is not None:
            symmetric_raw = self.viscosity_excess_net(
                viscosity_symmetric_pair)
            forward_raw = self.viscosity_excess_net(viscosity_pair)
            reverse_raw = self.viscosity_excess_net(
                viscosity_reverse_pair)
            neural_excess_raw = (
                symmetric_raw
                + 0.5 * self.viscosity_directed_odd_scale
                * (forward_raw - reverse_raw))
        else:
            neural_excess_raw = (
                self.viscosity_excess_net(viscosity_pair)
                if viscosity_pair is not None else None)
        if (viscosity_pair is not None
                and getattr(self, 'use_viscosity_gain_head', False)):
            gain_raw = self.viscosity_gain_net(viscosity_pair)
            viscosity_gain = torch.exp(
                self.viscosity_gain_log_scale * torch.tanh(gain_raw))
            neural_excess_raw = neural_excess_raw * viscosity_gain
        if (viscosity_pair is not None
                and getattr(self, 'viscosity_minplus_modes', 0)):
            minplus_hidden = self.viscosity_minplus_trunk(viscosity_pair)
            minplus_modes = torch.cat([
                head(minplus_hidden)
                for head in self.viscosity_minplus_heads
            ], dim=1)
            temperature = self.viscosity_minplus_temperature
            if self.viscosity_minplus_mode == 'tau_envelope':
                # Each internal mode is a complete scalar-fraction
                # correction.  The lower envelope is taken later in physical
                # travel-time space, rather than in the pre-sigmoid logit.
                # This is the smooth analogue of min_i T_i, with the base
                # correction retained as one additional (K+1)-st mode.
                # Zero-initialized mode heads therefore recover the inherited
                # scalar value exactly at warm start.
                viscosity_minplus_tau_modes = (
                    self.viscosity_minplus_scale * minplus_modes)
            else:
                # Subtract the same logsumexp evaluated at exact zero.  Besides
                # normalizing the soft minimum, this makes all-zero mode outputs
                # a bit-exact zero correction rather than merely approximately
                # zero after subtracting log(K).
                zero_normalizer = torch.logsumexp(
                    torch.zeros_like(minplus_modes), dim=1, keepdim=True)
                minplus_raw = -temperature * (
                    torch.logsumexp(
                        -minplus_modes / temperature, dim=1, keepdim=True)
                    - zero_normalizer)
                minplus_raw = self.viscosity_minplus_scale * minplus_raw
                neural_excess_raw = (
                    minplus_raw if neural_excess_raw is None
                    else neural_excess_raw + minplus_raw)
        causal_handover_raw = (
            self.causal_handover_net(causal_handover_pair)
            if causal_handover_pair is not None else None)
        causal_log_time_raw = None
        endpoint_speed_barrier_raw = None
        local_eikonal_rate_raw = None
        global_excess_raw = None
        signed_clearance_gate_raw = None
        if self.value_arch in (
                'bounded_chord_separate_smooth_excess_symmetric',
                'bounded_chord_separate_global_excess_symmetric'):
            raw_fraction = raw_pair
            global_excess_raw = separate_global_excess_raw
        elif self.value_arch in (
                'bounded_chord_speed_barrier_symmetric',
                'bounded_chord_local_speed_barrier_symmetric'):
            raw_fraction = raw_pair[:, :1]
            endpoint_speed_barrier_raw = raw_pair[:, 1:2]
        elif self.value_arch in (
                'bounded_chord_global_excess_symmetric',
                'bounded_chord_smooth_excess_symmetric'):
            raw_fraction = raw_pair[:, :1]
            global_excess_raw = raw_pair[:, 1:2]
        elif self.value_arch == 'bounded_chord_signed_clearance_symmetric':
            raw_fraction = raw_pair[:, :1]
            signed_clearance_gate_raw = raw_pair[:, 1:2]
        elif self.value_arch == 'bounded_chord_eikonal_refine_symmetric':
            raw_fraction = raw_pair[:, :1]
            local_eikonal_rate_raw = raw_pair[:, 1:2]
        else:
            raw_fraction = raw_pair
        if (neural_excess_raw is not None
                and self.viscosity_residual_mode == 'signed_fraction'):
            # Correct the learned chord fraction itself rather than adding a
            # second positive time potential.  This is the smallest smooth
            # capacity that can change a characteristic's direction in both
            # senses while preserving all structural bounds:
            #
            #   fraction = sigmoid(base_logit + delta_logit),
            #   T in [Euclidean distance, scalar chord time].
            #
            # The final sigmoid, not an intermediate tanh, is the physical
            # constraint: it always keeps T inside
            # [Euclidean distance, scalar chord time].  Letting the learned
            # pre-sigmoid correction be unbounded avoids vanishing gradients
            # at a viscosity cut locus while retaining the exact-zero warm
            # start.  Historical checkpoints keep their capped behavior.
            if getattr(self, 'viscosity_unbounded_logit', False):
                signed_logit = neural_excess_raw
            else:
                viscosity_logit_scale = float(getattr(
                    self, 'viscosity_logit_scale',
                    self.global_residual_scale))
                signed_logit = viscosity_logit_scale * torch.tanh(
                    neural_excess_raw / max(viscosity_logit_scale, 1e-6))
            viscosity_support = torch.ones_like(signed_logit)
            if self.viscosity_excess_gate in (
                    'cross_strength', 'cross_transverse',
                    'cross_transverse_saturated',
                    'cross_chord_evidence', 'cross_chord_sqrt_evidence',
                    'cross_chord_saturated_evidence', 'cross_certain'):
                if cross_strength is None:
                    viscosity_support = torch.zeros_like(signed_logit)
                    signed_logit = torch.zeros_like(signed_logit)
                else:
                    support = cross_strength.to(
                        device=signed_logit.device,
                        dtype=signed_logit.dtype).reshape(-1, 1)
                    if self.viscosity_excess_gate == 'cross_certain':
                        support = (2.0 * support - 1.0).clamp(0.0, 1.0)
                    elif self.viscosity_excess_gate in (
                            'cross_chord_evidence',
                            'cross_chord_sqrt_evidence',
                            'cross_chord_saturated_evidence'):
                        # The graph solution is trustworthy unconditionally
                        # inside its local edge scale.  For longer pairs, let
                        # it override the inherited characteristic only when
                        # the scalar chord itself contains integrated barrier
                        # evidence.  Both supports are smooth and use values
                        # already present in this single Eikonal field.
                        edge_scale = 0.22
                        local = ((1.1 * edge_scale - distance)
                                 / (0.2 * edge_scale)).clamp(0.0, 1.0)
                        local = local.square() * (3.0 - 2.0 * local)
                        barrier_slowness = 1.0 / max(
                            self.endpoint_barrier_speed, 1e-6)
                        chord_evidence = (
                            (mean_slowness - barrier_slowness) / (2.0 / 3.0)
                        ).clamp(0.0, 1.0)
                        chord_evidence = chord_evidence.square() * (
                            3.0 - 2.0 * chord_evidence)
                        evidence = 1.0 - (
                            1.0 - local) * (1.0 - chord_evidence)
                        if (self.viscosity_excess_gate ==
                                'cross_chord_sqrt_evidence'):
                            # Half-strength evidence is the geometric midpoint
                            # between unconditional cross support (factor 1)
                            # and fully evidence-conditioned support (factor e).
                            # It remains one smooth scalar-field gate.
                            evidence = evidence.sqrt()
                        elif (self.viscosity_excess_gate ==
                              'cross_chord_saturated_evidence'):
                            # A second probabilistic-union step sharpens
                            # confidence without a fitted threshold:
                            # e -> 1-(1-e)^2 = e(2-e).  A nearly certain
                            # scalar barrier therefore preserves the learned
                            # viscosity correction, while a nearly clear
                            # chord still suppresses it.  This remains a
                            # smooth function of this model's scalar speed.
                            evidence = evidence * (2.0 - evidence)
                        support = support * evidence
                    elif self.viscosity_excess_gate in (
                            'cross_transverse',
                            'cross_transverse_saturated'):
                        if transverse_contrast is None:
                            support = torch.zeros_like(support)
                        else:
                            contrast = transverse_contrast.to(
                                device=signed_logit.device,
                                dtype=signed_logit.dtype)
                            if (self.viscosity_excess_gate ==
                                    'cross_transverse_saturated'):
                                # A smooth upwind switch: weak transverse
                                # evidence is ignored, while clear scalar-field
                                # asymmetry passes the learned correction at
                                # full strength instead of attenuating it.
                                contrast = ((contrast - 0.2) / 0.6).clamp(
                                    0.0, 1.0)
                                contrast = contrast.square() * (
                                    3.0 - 2.0 * contrast)
                            support = support * contrast
                    viscosity_support = support
                    signed_logit = signed_logit * support
            raw_fraction = raw_fraction + signed_logit
            if viscosity_minplus_tau_modes is not None:
                # Form K internal alternatives around the same base fraction
                # and take one normalized smooth minimum in travel-time
                # units.  The first candidate is the learned base viscosity
                # correction; each remaining candidate is one internal scalar
                # mode.  The common support is inherited from the same scalar
                # field gate used by the base correction.
                if getattr(self, 'viscosity_unbounded_logit', False):
                    mode_signed = viscosity_minplus_tau_modes
                else:
                    viscosity_logit_scale = float(getattr(
                        self, 'viscosity_logit_scale',
                        self.global_residual_scale))
                    mode_signed = viscosity_logit_scale * torch.tanh(
                        viscosity_minplus_tau_modes
                        / max(viscosity_logit_scale, 1e-6))
                mode_signed = mode_signed * viscosity_support
                mode_logits = raw_fraction + mode_signed
                candidate_logits = torch.cat([
                    raw_fraction, mode_logits], dim=1)
                candidate_fraction = torch.sigmoid(candidate_logits)
                candidate_tau = distance + candidate_fraction * (
                    chord_time - distance).clamp_min(0.0)
                envelope_temperature = max(
                    float(self.viscosity_minplus_temperature), 1e-6)
                zero_normalizer = math.log(candidate_tau.shape[1])
                viscosity_tau_envelope = -envelope_temperature * (
                    torch.logsumexp(
                        -candidate_tau / envelope_temperature,
                        dim=1, keepdim=True)
                    - candidate_tau.new_tensor(zero_normalizer))
            # Do not also apply the historical positive-time branch below.
            neural_excess_raw = None
        elif (neural_excess_raw is not None
              and self.viscosity_residual_mode == 'causal_log_time'):
            # A directed state-constrained Eikonal distance need not fit below
            # the straight-chord time: that chord may move backwards through
            # the handover phase and is then not an admissible path.  Learn one
            # ordered log-time correction around the inherited scalar value.
            # Multiplying T (rather than adding a constant) preserves T(q,q)=0.
            causal_log_time_raw = neural_excess_raw
            neural_excess_raw = None
        if self.value_arch == 'bounded_chord_eikonal_symmetric':
            # Local consistency follows directly from the Eikonal equation:
            # as q1 -> q0, the geodesic and the chord coincide, hence
            # T(q0,q1) = ||q1-q0|| / F(q0) + o(||q1-q0||).  Parameterizing a
            # non-negative attenuation *rate* instead of an unconstrained
            # fraction makes fraction(0)=1 structural, while retaining the
            # full [distance, chord_time] range at every nonzero separation.
            attenuation_rate = F.softplus(raw_fraction)
            fraction = torch.exp(-distance * attenuation_rate)
        else:
            fraction = torch.sigmoid(raw_fraction)
        if local_eikonal_rate_raw is not None:
            # Preserve the learned E217 global viscosity branch while making
            # the infinitesimal limit obey F(q)|grad_q T|=1 exactly.  The
            # probabilistic-union form keeps fraction in [0, 1], so tau stays
            # between its Euclidean lower bound and same-field chord upper
            # bound for every pair:
            #
            #   f = 1 - (1 - f_global)(1 - exp(-D*kappa)).
            #
            # At D->0, f->1 and chord_time->D/F(q).  At finite D a learned
            # positive kappa can make the correction arbitrarily local,
            # leaving the inherited global solution unchanged.
            attenuation_rate = F.softplus(local_eikonal_rate_raw)
            local_gate = torch.exp(-distance * attenuation_rate)
            fraction = 1.0 - (1.0 - fraction) * (1.0 - local_gate)
        endpoint_depth = None
        endpoint_depth_contrast = None
        if self.value_arch in (
                'bounded_chord_endpoint_eikonal_symmetric',
                'bounded_chord_endpoint_fermat_curve_symmetric',
                'bounded_chord_endpoint_speed_fermat_curve_symmetric',
                'bounded_chord_local_endpoint_eikonal_symmetric',
                # The global/smooth excess variants can be warm-started from
                # the endpoint-Eikonal field.  Keep its scalar endpoint
                # extension so adding the second (D^2) row is an exact
                # function-preserving expansion rather than silently
                # removing the state-constraint barrier.
                'bounded_chord_global_excess_symmetric',
                'bounded_chord_smooth_excess_symmetric',
                'bounded_chord_separate_smooth_excess_symmetric',
                'bounded_chord_separate_global_excess_symmetric',
                # The profile extension keeps the same scalar endpoint
                # barrier as the bounded endpoint field.  This is important
                # for a function-preserving warm start: exposing chord
                # samples must not silently remove the boundary extension.
                'bounded_chord_profile_symmetric'):
            # The Eikonal solution lives on the feasible domain.  When a
            # planner nevertheless proposes an endpoint inside a near-zero
            # speed set, extend T into that excluded set with a finite signed-
            # depth barrier.  Under the scalar data's exponential interior
            # map F=F_b exp((d-d_b)/d_b), log(F_b/F) is exactly normalized
            # penetration depth.  This is much better conditioned than
            # forcing the entire chord cost (which scales as 1/F) and uses no
            # critic, normal, collision label, or additional model.
            boundary = self.endpoint_barrier_speed
            if self.endpoint_barrier_physical_decode:
                # F_store = floor + (1-floor)*sigmoid(logit) is merely the
                # historical on-disk encoding.  The state-constrained PDE
                # lives on F_physical=sigmoid(logit), whose boundary is zero.
                # Use log-sigmoid directly so the repulsive derivative does
                # not disappear when F_store rounds numerically to its floor.
                log_physical = self.predict_log_physical_speed(
                    endpoint_points, features=features)
                log_boundary = math.log(
                    (boundary - self.speed_floor)
                    / (1.0 - self.speed_floor))
                physical_depth = (
                    log_boundary - log_physical).clamp(0.0, 20.0)
                depth0 = physical_depth[:size]
                depth1 = physical_depth[size:]
            else:
                depth0 = torch.log(
                    boundary / s0.clamp_min(self.speed_floor)).clamp_min(0.0)
                depth1 = torch.log(
                    boundary / s1.clamp_min(self.speed_floor)).clamp_min(0.0)
            if (not self.endpoint_barrier_physical_decode
                    and self.endpoint_barrier_power != 1.0):
                # Keep the penetration depth at the learned field's floor
                # unchanged while making the extension approach the feasible
                # boundary with zero slope for power > 1.  This permits a
                # wider state-constraint support without pushing equally hard
                # on legitimate near-boundary states.  It depends only on the
                # same scalar speed F; no normal, critic, or geometry query is
                # introduced.
                floor_depth = math.log(
                    boundary / self.speed_floor)
                if floor_depth > 0.0:
                    depth0 = floor_depth * (
                        depth0 / floor_depth).clamp(0.0, 1.0).pow(
                            self.endpoint_barrier_power)
                    depth1 = floor_depth * (
                        depth1 / floor_depth).clamp(0.0, 1.0).pow(
                            self.endpoint_barrier_power)
            endpoint_depth = 0.5 * (depth0 + depth1)
            endpoint_depth_contrast = (depth0 - depth1).square()
        if self.value_arch == 'bounded_chord_log_symmetric':
            # Interpolate between the same physical bounds in log time rather
            # than linearly in time:
            #
            #   T = D * (C / D) ** fraction
            #     = D * mean_slowness ** fraction.
            #
            # This remains exactly in [D, C], is symmetric and zero on the
            # diagonal, and is invariant to the dynamic range of C/D.  A
            # near-zero signed-clearance speed can therefore be a strong
            # barrier without forcing the learned fraction to collapse as
            # 1/(C/D).
            log_ratio = torch.log(mean_slowness.clamp_min(1.0))
            return distance * torch.exp(fraction * log_ratio)
        if self.value_arch == 'bounded_chord_hazard_symmetric':
            # F=1 is free space.  Make the learned detour correction vanish
            # to second order there, so tiny speed-fit ripples cannot bend an
            # otherwise safe straight line.  The gate has no tuned constant:
            # it is exactly one minus the chord's harmonic-mean speed.
            hazard = (1.0 - mean_slowness.reciprocal()).clamp(0.0, 1.0)
            fraction = fraction * hazard
        elif self.value_arch == 'bounded_chord_hazard_floor_symmetric':
            # A chord that visibly crosses low speed cannot be assigned an
            # arbitrarily tiny share of its measured travel-time penalty.
            # The parameter-free probabilistic union preserves the learned
            # fraction in free space and lower-bounds it by 1-F_harmonic.
            hazard = (
                self.chord_hazard_floor_scale
                * (1.0 - mean_slowness.reciprocal()).clamp(0.0, 1.0))
            fraction = 1.0 - (1.0 - fraction) * (1.0 - hazard)
        if (self.lifted_phase
                and self.lifted_phase_hazard_floor_scale > 0.0):
            # The inherited pair head was trained in q only and can suppress
            # almost all of a newly visible (q,p) chord hazard by assigning a
            # near-zero mixing fraction.  Require a phase-changing pair to
            # retain a proportional share of the very same scalar-speed
            # evidence already measured on its chord.  The probabilistic
            # union keeps fraction in [0, 1], is reversal invariant, and is
            # exactly the inherited value whenever p0 == p1.  This is part of
            # the single scalar value architecture, not an inference-time
            # penalty, critic, candidate score, or second field.
            phase_delta = delta[:, -1:].abs()
            phase_span = (
                phase_delta / self.lifted_phase_scale).clamp(0.0, 1.0)
            scalar_hazard = (
                1.0 - mean_slowness.reciprocal()).clamp(0.0, 1.0)
            phase_hazard_floor = (
                self.lifted_phase_hazard_floor_scale
                * phase_span * scalar_hazard)
            fraction = (
                1.0 - (1.0 - fraction) * (1.0 - phase_hazard_floor))
        if (self.cross_hazard_floor_scale > 0.0
                and cross_strength is not None):
            # A viscosity solution cannot make a measured low-speed chord
            # arbitrarily cheap merely because the learned mixing logit is
            # small.  Cross-task pairs are exactly where that failure changes
            # the characteristic's homotopy; pairs that do not cross the
            # model's own task-routing boundary remain exactly inherited.
            # This lower bound uses only the same scalar F samples
            # already used by the Eikonal chord quadrature.  It is therefore a
            # structural part of this one value field, not an auxiliary cost,
            # critic, validator, candidate score, or inference-time rescue.
            scalar_hazard = (
                1.0 - mean_slowness.reciprocal()).clamp(0.0, 1.0)
            cross_probability = cross_strength.to(
                device=fraction.device, dtype=fraction.dtype).reshape(-1, 1)
            # Routing weights are soft probabilities.  Their raw XOR is
            # nonzero even when both endpoints weakly belong to the same
            # task, which would leak a cross-only structural correction into
            # ordinary cup/pour motion.  Remove that ambiguity below the
            # decision boundary and map a certain cross pair back to one.
            cross_gate = (2.0 * cross_probability - 1.0).clamp(0.0, 1.0)
            cross_hazard_floor = (
                self.cross_hazard_floor_scale * cross_gate * scalar_hazard)
            fraction = (
                1.0 - (1.0 - fraction) * (1.0 - cross_hazard_floor))
        tau = distance + fraction * (
            chord_time - distance).clamp_min(0.0)
        if viscosity_tau_envelope is not None:
            # The internal soft envelope is already in physical scalar-time
            # units.  Endpoint/state barriers below remain common additive
            # terms, so this preserves one symmetric value field and the
            # exact zero warm start.
            tau = viscosity_tau_envelope
        if causal_log_time_raw is not None:
            if getattr(self, 'viscosity_unbounded_logit', False):
                causal_log_scale = causal_log_time_raw
            else:
                viscosity_logit_scale = float(getattr(
                    self, 'viscosity_logit_scale',
                    self.global_residual_scale))
                causal_log_scale = viscosity_logit_scale * torch.tanh(
                    causal_log_time_raw / max(viscosity_logit_scale, 1e-6))
            if cross_strength is None:
                causal_support = torch.zeros_like(causal_log_scale)
            else:
                causal_support = cross_strength.to(
                    device=tau.device, dtype=tau.dtype).reshape(-1, 1)
                if self.viscosity_excess_gate == 'cross_certain':
                    causal_support = (
                        2.0 * causal_support - 1.0).clamp(0.0, 1.0)
                elif self.viscosity_excess_gate in (
                        'cross_chord_evidence',
                        'cross_chord_sqrt_evidence'):
                    # A directed graph target is locally trustworthy, but a
                    # long endpoint chord may belong to a different
                    # characteristic.  Let the causal correction act on long
                    # pairs only when this same scalar-speed field measures a
                    # barrier along their chord.  This is an internal support
                    # of the single Eikonal value, not a candidate score or
                    # post-planning validator.
                    edge_scale = 0.22
                    local = ((1.1 * edge_scale - distance)
                             / (0.2 * edge_scale)).clamp(0.0, 1.0)
                    local = local.square() * (3.0 - 2.0 * local)
                    barrier_slowness = 1.0 / max(
                        self.endpoint_barrier_speed, 1e-6)
                    chord_evidence = (
                        (mean_slowness - barrier_slowness) / (2.0 / 3.0)
                    ).clamp(0.0, 1.0)
                    chord_evidence = chord_evidence.square() * (
                        3.0 - 2.0 * chord_evidence)
                    evidence = 1.0 - (
                        1.0 - local) * (1.0 - chord_evidence)
                    if (self.viscosity_excess_gate ==
                            'cross_chord_sqrt_evidence'):
                        evidence = evidence.sqrt()
                    causal_support = causal_support * evidence
            tau = tau * torch.exp(causal_support * causal_log_scale)
        if self.value_arch in (
                'bounded_chord_fermat_curve_symmetric',
                'bounded_chord_endpoint_fermat_curve_symmetric',
                'bounded_chord_endpoint_speed_fermat_curve_symmetric'):
            # Replace only the straight-path part of E260's upper bound by
            # the improvement supplied by one amortized Fermat curve:
            #
            #   U = C_chord(base rule)
            #       + w [C_curve(dense rule)-C_chord(dense rule)]
            #   T = D + fraction * (U-D).
            #
            # A zero bend gives C_curve==C_chord and is therefore exactly the
            # inherited E260 field.  No path is selected or reranked: one
            # continuously decoded curve changes one scalar value field.
            if (self.fermat_curve_value_weight > 0.0
                    or getattr(self, 'capture_fermat_training_cost', False)):
                (curve_time, straight_time,
                 physical_curve_time, physical_straight_time) = (
                    self._bounded_fermat_curve_costs(
                        coords, features, size))
                curve_upper = (
                    chord_time
                    + self.fermat_curve_value_weight
                    * (curve_time - straight_time))
                if self.fermat_curve_gate != 'all':
                    if cross_strength is None:
                        curve_support = torch.zeros_like(distance)
                    else:
                        curve_support = cross_strength.to(
                            device=distance.device,
                            dtype=distance.dtype).reshape(-1, 1)
                        if self.fermat_curve_gate == 'cross_certain':
                            curve_support = (
                                2.0 * curve_support - 1.0).clamp(0.0, 1.0)
                    curve_upper = (
                        chord_time + curve_support
                        * (curve_upper - chord_time))
                curve_upper = curve_upper.clamp_min(distance)
                tau = distance + fraction * (curve_upper - distance)
                if getattr(self, 'capture_fermat_training_cost', False):
                    self.captured_fermat_physical_curve_time = (
                        physical_curve_time)
                    self.captured_fermat_physical_straight_time = (
                        physical_straight_time)
        if signed_clearance_gate_raw is not None:
            # Exact no-op warm start, with a learnable one-sided gate.  The
            # added cost is still a scalar chord integral from this same
            # network; no collision query, normal, candidate, or reranker is
            # present at inference.
            gate_origin = torch.sigmoid(
                signed_clearance_gate_raw.new_tensor(-6.0))
            clearance_gate = (
                torch.sigmoid(signed_clearance_gate_raw)
                - gate_origin).clamp_min(0.0)
            tau = (
                tau + fraction * clearance_gate
                * signed_clearance_extra)
        if global_excess_raw is not None:
            # The straight chord is only a sampled upper-bound proxy.  A thin
            # low-speed set or a change of optimal homotopy can require a
            # smooth global viscosity correction that a bounded sigmoid
            # fraction cannot express.  Make that capacity second-order in
            # separation, so T(q,q)=0 and the complete first-order local
            # Eikonal limit are inherited unchanged from the base branch.
            # It is gated by the same scalar chord speed and therefore
            # vanishes exactly on a free chord without normals, labels, or an
            # external critic.
            if self.value_arch in (
                    'bounded_chord_global_excess_symmetric',
                    'bounded_chord_separate_global_excess_symmetric'):
                excess_support = (
                    1.0 - mean_slowness.reciprocal()).clamp(0.0, 1.0)
            else:
                # The viscosity solution is globally smooth away from its cut
                # locus.  Multiplying its correction by a sampled straight-
                # chord hazard re-introduces the jagged local speed boundary
                # that the global branch is meant to propagate around.  Let
                # the learned O(D^2) term carry that topology instead; its
                # diagonal derivative is still exactly zero.
                excess_support = torch.ones_like(distance)
            # Subtract the deterministic initialization exactly: the new
            # branch is a true no-op at warm start.  The centered-softplus
            # option has derivative sigmoid(0)=1/2 at the neutral logit,
            # rather than the nearly-zero derivative of sigmoid(-6), while
            # retaining a one-sided non-negative correction.
            if self.global_excess_centered_signed_gate:
                # A smooth signed correction is useful when the learned
                # straight chord is itself not a certified upper bound (and
                # lets the PDE reduce an overestimated chord as well as raise
                # an infeasible detour).  The centered value is still exactly
                # zero at raw=-6 and has derivative 1/2 there.
                global_gate = (
                    F.softplus(global_excess_raw + 6.0)
                    - F.softplus(global_excess_raw.new_zeros(())))
            elif self.global_excess_centered_gate:
                global_gate = (
                    F.softplus(global_excess_raw + 6.0)
                    - F.softplus(global_excess_raw.new_zeros(())))
                global_gate = global_gate.clamp_min(0.0)
            else:
                gate_origin = torch.sigmoid(
                    global_excess_raw.new_tensor(-6.0))
                global_gate = (
                    torch.sigmoid(global_excess_raw)
                    - gate_origin).clamp_min(0.0)
            tau = (
                tau + self.global_residual_scale
                * distance.square() * excess_support * global_gate)
        if neural_excess_raw is not None:
            # A conditional approximation to the global viscosity correction.
            # D^2 makes both T(q,q)=0 and the complete first-order local
            # Eikonal limit structural.  Subtracting the initialized gate makes
            # a warm start exactly identical to E260, while clamp_min forbids
            # the new branch from shortening its already valid travel time.
            if (self.viscosity_residual_mode == 'positive_time'
                    and self.viscosity_unbounded_logit):
                # In positive-time mode the chord is only a local proxy, not
                # a certified global upper bound: a collision-free geodesic
                # may have to detour and therefore satisfy T > C_chord.  Use
                # the same centered smooth gate as the scalar global-excess
                # branch, but on the dedicated viscosity head.  D^2 keeps
                # T(q,q)=0 and preserves the first-order Eikonal limit; the
                # softplus is unbounded above so this branch can represent a
                # true detour without task labels, paths, or a second field.
                neural_gate = (
                    F.softplus(neural_excess_raw + 6.0)
                    - F.softplus(neural_excess_raw.new_zeros(())))
                neural_gate = neural_gate.clamp_min(0.0)
            else:
                gate_origin = torch.sigmoid(
                    neural_excess_raw.new_tensor(-6.0))
                neural_gate = (
                    torch.sigmoid(neural_excess_raw)
                    - gate_origin).clamp_min(0.0)
            if self.viscosity_excess_gate in (
                    'cross_strength', 'cross_transverse',
                    'cross_transverse_saturated',
                    'cross_chord_evidence', 'cross_chord_sqrt_evidence',
                    'cross_certain'):
                # Preserve the inherited within-task field exactly and place
                # global detour capacity only where the endpoint boundary
                # conditions differ.  The gate is the existing continuous
                # routing probability, not a critic or extra inference model.
                if cross_strength is None:
                    neural_gate = torch.zeros_like(neural_gate)
                else:
                    support = cross_strength.to(
                        device=neural_gate.device,
                        dtype=neural_gate.dtype).reshape(-1, 1)
                    if self.viscosity_excess_gate == 'cross_certain':
                        support = (2.0 * support - 1.0).clamp(0.0, 1.0)
                    elif self.viscosity_excess_gate in (
                            'cross_chord_evidence',
                            'cross_chord_sqrt_evidence'):
                        edge_scale = 0.22
                        local = ((1.1 * edge_scale - distance)
                                 / (0.2 * edge_scale)).clamp(0.0, 1.0)
                        local = local.square() * (3.0 - 2.0 * local)
                        barrier_slowness = 1.0 / max(
                            self.endpoint_barrier_speed, 1e-6)
                        chord_evidence = (
                            (mean_slowness - barrier_slowness) / (2.0 / 3.0)
                        ).clamp(0.0, 1.0)
                        chord_evidence = chord_evidence.square() * (
                            3.0 - 2.0 * chord_evidence)
                        evidence = 1.0 - (
                            1.0 - local) * (1.0 - chord_evidence)
                        if (self.viscosity_excess_gate ==
                                'cross_chord_sqrt_evidence'):
                            evidence = evidence.sqrt()
                        support = support * evidence
                    elif self.viscosity_excess_gate in (
                            'cross_transverse',
                            'cross_transverse_saturated'):
                        if transverse_contrast is None:
                            support = torch.zeros_like(support)
                        else:
                            contrast = transverse_contrast.to(
                                device=neural_gate.device,
                                dtype=neural_gate.dtype)
                            if (self.viscosity_excess_gate ==
                                    'cross_transverse_saturated'):
                                contrast = ((contrast - 0.2) / 0.6).clamp(
                                    0.0, 1.0)
                                contrast = contrast.square() * (
                                    3.0 - 2.0 * contrast)
                            support = support * contrast
                    neural_gate = neural_gate * support
            tau = (
                tau + self.global_residual_scale
                * distance.square() * neural_gate)
        if endpoint_speed_barrier_raw is not None:
            # F=1 contributes exactly zero.  As F falls, -log(F) is a smooth
            # dimensionless depth whose derivative exposes grad F directly to
            # grad_q T.  The learned positive gate is pair-dependent but lives
            # in this same value head; no second model or inference correction
            # is involved.
            endpoint_hazard = -0.5 * (
                torch.log(s0.clamp_min(self.speed_floor))
                + torch.log(s1.clamp_min(self.speed_floor)))
            barrier_gate = torch.sigmoid(endpoint_speed_barrier_raw)
            barrier_length = distance
            if (self.value_arch ==
                    'bounded_chord_local_speed_barrier_symmetric'):
                # The endpoint term is one local Bellman/Eikonal step, not a
                # second path-length estimate.  Scaling it by the complete
                # q--goal distance injects an unrelated radial gradient and
                # lets a local clearance defect distort every long pair.
                # Saturating at the largest supervised local step preserves
                # grad(log F) as a local repulsive direction while removing
                # that spurious long-range distance gradient.  A hard cap is
                # also consistent with the nonsmooth viscosity solution at a
                # change of optimal characteristic.
                barrier_length = distance.clamp(
                    max=self.endpoint_barrier_radius)
            tau = (
                tau + self.endpoint_barrier_scale
                * barrier_length * barrier_gate * endpoint_hazard)
        if endpoint_depth is not None:
            if self.endpoint_barrier_contrast:
                # Symmetric zero-diagonal state potential.  Unlike the
                # historical D * mean(depth) extension, this term has no
                # q--goal radial derivative: its gradient comes only from
                # the same scalar speed boundary at the two endpoints.
                tau = (
                    tau + self.endpoint_barrier_scale
                    * endpoint_depth_contrast)
            else:
                endpoint_length = distance
                if (self.value_arch ==
                        'bounded_chord_local_endpoint_eikonal_symmetric'):
                    # This is a boundary extension over one local Eikonal
                    # step, not a second global path cost.  Capping its length
                    # removes the unrelated q--goal radial force on long
                    # pairs while the derivative of log(F_boundary/F(q))
                    # retains the scalar field's local repulsive direction.
                    endpoint_length = distance.clamp(
                        max=self.endpoint_barrier_radius)
                tau = (
                    tau + self.endpoint_barrier_scale
                    * endpoint_length * endpoint_depth)
        if clearance_state_depth is not None:
            # Extend the feasible-domain viscosity solution into invalid
            # endpoint states by one local Eikonal length.  The capped length
            # avoids a spurious goal-radial force, symmetry is exact, and the
            # diagonal remains zero.  Unlike a learned critic, this boundary
            # condition contains no label beyond signed scalar clearance.
            depth0 = clearance_state_depth[:size]
            depth1 = clearance_state_depth[size:]
            local_length = distance.clamp(
                max=self.endpoint_barrier_radius)
            tau = (
                tau + self.endpoint_barrier_scale * local_length
                * 0.5 * (depth0 + depth1))
        if self.handover_phase_embedding_scale > 0.0:
            phase_start, phase_goal = self._endpoint_values(
                handover_phase, size, coords.device, coords.dtype)
            phase_delta = (phase_goal - phase_start).unsqueeze(1)
            if cross_strength is None:
                phase_support = torch.zeros_like(phase_delta)
            else:
                # Remove ambiguous same-task routing mass.  This keeps the
                # inherited cup/pour value exactly unchanged and makes the
                # product-space coordinate active only for a confidently
                # cross-task boundary condition.
                phase_support = (2.0 * cross_strength.to(
                    device=tau.device, dtype=tau.dtype).reshape(-1, 1)
                    - 1.0).clamp(0.0, 1.0)
            phase_length = (
                self.handover_phase_embedding_scale
                * phase_support * phase_delta)
            # One scalar Eikonal distance in the product embedding.  vector_norm
            # has a zero subgradient at the diagonal, preserving T(q,q)=0.
            tau = torch.linalg.vector_norm(
                torch.cat([tau, phase_length], dim=1),
                dim=1, keepdim=True)
        if causal_handover_raw is not None:
            causal_scale = (
                self.causal_handover_logit_scale * torch.tanh(
                    causal_handover_raw
                    / self.causal_handover_logit_scale))
            if cross_strength is None:
                causal_support = torch.zeros_like(causal_scale)
            else:
                causal_support = cross_strength.to(
                    device=tau.device, dtype=tau.dtype).reshape(-1, 1)
                if self.causal_handover_gate == 'cross_certain':
                    causal_support = (
                        2.0 * causal_support - 1.0).clamp(0.0, 1.0)
                elif self.causal_handover_gate in (
                        'cross_chord_evidence',
                        'cross_chord_sqrt_evidence'):
                    edge_scale = 0.22
                    local = ((1.1 * edge_scale - distance)
                             / (0.2 * edge_scale)).clamp(0.0, 1.0)
                    local = local.square() * (3.0 - 2.0 * local)
                    barrier_slowness = 1.0 / max(
                        self.endpoint_barrier_speed, 1e-6)
                    chord_evidence = (
                        (mean_slowness - barrier_slowness) / (2.0 / 3.0)
                    ).clamp(0.0, 1.0)
                    chord_evidence = chord_evidence.square() * (
                        3.0 - 2.0 * chord_evidence)
                    evidence = 1.0 - (
                        1.0 - local) * (1.0 - chord_evidence)
                    if (self.causal_handover_gate ==
                            'cross_chord_sqrt_evidence'):
                        evidence = evidence.sqrt()
                    causal_support = causal_support * evidence
            tau = tau * torch.exp(
                self.causal_handover_strength
                * causal_support * causal_scale)
        return tau

    def _bounded_fermat_curve_costs(self, coords, features, size):
        """Return stored/physical costs of one reversible learned curve.

        The only environmental quantity used here is this model's scalar
        speed.  The affine storage floor is retained for the outer E260 value
        scale and removed for the inner Fermat objective, where the strict
        invalid-set boundary must approach zero propagation speed.
        """
        normalized = self.encoder_norm(features)
        z0 = normalized[:size]
        z1 = normalized[size:]
        q0 = coords[:, :self.dim]
        q1 = coords[:, self.dim:]
        delta = q1 - q0
        distance = torch.linalg.vector_norm(
            delta, dim=1, keepdim=True)
        endpoint_points = torch.cat([q0, q1], dim=0)
        endpoint_speed = self.predict_speed(
            endpoint_points, features=features)
        s0 = endpoint_speed[:size]
        s1 = endpoint_speed[size:]

        nodes = self.fermat_curve_nodes.to(dtype=coords.dtype)
        weights = self.fermat_curve_weights.to(dtype=coords.dtype)
        base = q0[:, None, :] + nodes[None, :, None] * delta[:, None, :]
        base_flat = base.reshape(-1, self.dim)
        stored_straight_speed = self.predict_propagation_speed(
            base_flat).reshape(size, self.fermat_curve_quad_points)

        profile_forward = None
        profile_reverse = None
        if (self.value_arch ==
                'bounded_chord_endpoint_speed_fermat_curve_symmetric'):
            # The curve decoder is no longer blind between its endpoints.
            # Read a variable-length set of samples from the *same* scalar
            # speed field along the chord and pool them into one directed
            # context.  q and t identify where the slowdown occurs; the
            # normalized log slowdown is the signed-distance-like evidence.
            # No collision query, normal, candidate path, or iterative solve
            # is introduced.
            physical_profile_speed = (
                (stored_straight_speed - self.speed_floor)
                / (1.0 - self.speed_floor)).clamp(0.0, 1.0)
            physical_profile_speed = (
                self.fermat_curve_speed_floor
                + (1.0 - self.fermat_curve_speed_floor)
                * physical_profile_speed)
            log_scale = -math.log(self.fermat_curve_speed_floor)
            slowdown = (
                -torch.log(physical_profile_speed.clamp_min(
                    self.fermat_curve_speed_floor)) / log_scale
            ).clamp(0.0, 1.0)
            centered_t = (2.0 * nodes - 1.0)[None, :, None].expand(
                size, -1, -1)
            forward_samples = torch.cat([
                base, centered_t, slowdown[:, :, None]], dim=2)
            reverse_samples = torch.cat([
                base.flip(1), centered_t,
                slowdown.flip(1)[:, :, None]], dim=2)
            forward_embedding = self.fermat_speed_profile_net(
                forward_samples.reshape(-1, self.dim + 2)).reshape(
                    size, self.fermat_curve_quad_points, -1)
            reverse_embedding = self.fermat_speed_profile_net(
                reverse_samples.reshape(-1, self.dim + 2)).reshape(
                    size, self.fermat_curve_quad_points, -1)
            forward_measure = (
                weights[None, :] * slowdown)[:, :, None]
            reverse_measure = (
                weights[None, :] * slowdown.flip(1))[:, :, None]
            profile_forward = (
                forward_measure * forward_embedding).sum(dim=1)
            profile_reverse = (
                reverse_measure * reverse_embedding).sum(dim=1)

        pair_forward_parts = [
            z0, z1, z1 - z0, z0 * z1,
            delta, delta.abs(), distance, s0, s1,
        ]
        pair_reverse_parts = [
            z1, z0, z0 - z1, z1 * z0,
            -delta, delta.abs(), distance, s1, s0,
        ]
        if profile_forward is not None:
            pair_forward_parts.append(profile_forward)
            pair_reverse_parts.append(profile_reverse)
        pair_forward = torch.cat(pair_forward_parts, dim=1)
        pair_reverse = torch.cat(pair_reverse_parts, dim=1)
        raw_forward = self._fermat_curve_pair_output(
            pair_forward).reshape(
                size, self.fermat_curve_modes, self.dim)
        raw_reverse = self._fermat_curve_pair_output(
            pair_reverse).reshape(
                size, self.fermat_curve_modes, self.dim)

        mode_number = torch.arange(
            1, self.fermat_curve_modes + 1,
            device=coords.device, dtype=coords.dtype)
        parity = torch.where(
            mode_number.remainder(2) == 1,
            torch.ones_like(mode_number),
            -torch.ones_like(mode_number))
        equivariant_raw = 0.5 * (
            raw_forward + parity[None, :, None] * raw_reverse)
        distance_gate = distance / (distance + 0.05)
        coefficients = (
            torch.tanh(equivariant_raw)
            * distance_gate[:, None, :]
            / mode_number[None, :, None])

        harmonic_straight_speed = stored_straight_speed.clamp_min(
            self.speed_floor).reciprocal().mean(
                dim=1, keepdim=True).reciprocal()
        chord_hazard = (1.0 - harmonic_straight_speed).clamp(0.0, 1.0)
        coefficients = coefficients * chord_hazard[:, None, :]
        # At inference the decoder represents the already-amortized inner
        # minimizer.  Fermat's envelope derivative holds that entire curve
        # fixed while differentiating its endpoint cost.
        if not self.training and self.fermat_curve_envelope_detach:
            coefficients = coefficients.detach()

        phase = math.pi * mode_number[:, None] * nodes[None, :]
        basis = torch.sin(phase)
        basis_derivative = (
            math.pi * mode_number[:, None] * torch.cos(phase))
        residual = torch.einsum('bmd,mk->bkd', coefficients, basis)
        residual_derivative = torch.einsum(
            'bmd,mk->bkd', coefficients, basis_derivative)

        # Apply the bend with the exact tanh-addition identity
        #
        #   tanh(atanh(x) + r) = (x + tanh(r)) / (1 + x tanh(r)).
        #
        # Writing it in rational form avoids the ill-conditioned atanh near
        # joint limits.  More importantly, r=0 is now an exact straight-line
        # no-op in both value and endpoint gradient, including x=+/-1.
        base_unit = (2.0 * base).clamp(-1.0, 1.0)
        bend_unit = torch.tanh(residual)
        denominator = (1.0 + base_unit * bend_unit).clamp_min(1e-6)
        squashed = (base_unit + bend_unit) / denominator
        curve_points = 0.5 * squashed
        base_unit_derivative = 2.0 * delta[:, None, :]
        bend_unit_derivative = (
            (1.0 - bend_unit.square()) * residual_derivative)
        numerator_derivative = (
            base_unit_derivative + bend_unit_derivative)
        denominator_derivative = (
            base_unit_derivative * bend_unit
            + base_unit * bend_unit_derivative)
        curve_derivative = 0.5 * (
            numerator_derivative * denominator
            - (base_unit + bend_unit) * denominator_derivative
        ) / denominator.square()
        curve_speed_norm = torch.linalg.vector_norm(
            curve_derivative, dim=2)

        curve_flat = curve_points.reshape(-1, self.dim)
        stored_curve_speed = self.predict_propagation_speed(
            curve_flat).reshape(size, self.fermat_curve_quad_points)
        stored_curve_time = (
            curve_speed_norm
            / stored_curve_speed.clamp_min(self.speed_floor)
            * weights[None, :]).sum(dim=1, keepdim=True)
        stored_straight_time = (
            distance
            * (stored_straight_speed.clamp_min(self.speed_floor).reciprocal()
               * weights[None, :]).sum(dim=1, keepdim=True))

        # Undo the historical affine coding F_store=0.1+0.9*F_physical.
        # This is still the same one-dimensional scalar field, not a second
        # prediction head or any geometric query.
        raw_curve_speed = self.predict_speed(curve_flat).reshape(
            size, self.fermat_curve_quad_points)
        raw_straight_speed = self.predict_speed(base_flat).reshape(
            size, self.fermat_curve_quad_points)
        physical_curve_speed = (
            (raw_curve_speed - self.speed_floor)
            / (1.0 - self.speed_floor)).clamp(0.0, 1.0)
        physical_straight_speed = (
            (raw_straight_speed - self.speed_floor)
            / (1.0 - self.speed_floor)).clamp(0.0, 1.0)
        physical_curve_speed = (
            self.fermat_curve_speed_floor
            + (1.0 - self.fermat_curve_speed_floor)
            * physical_curve_speed)
        physical_straight_speed = (
            self.fermat_curve_speed_floor
            + (1.0 - self.fermat_curve_speed_floor)
            * physical_straight_speed)
        physical_curve_time = (
            curve_speed_norm / physical_curve_speed
            * weights[None, :]).sum(dim=1, keepdim=True)
        physical_straight_time = (
            distance
            * (physical_straight_speed.reciprocal()
               * weights[None, :]).sum(dim=1, keepdim=True))
        if getattr(self, 'capture_fermat_training_cost', False):
            # Scalar-only training may impose the state-constraint boundary
            # F(q)>F_boundary densely along this same single curve.  Exposing
            # the already-computed scalar samples avoids any collision query.
            self.captured_fermat_stored_curve_speed = raw_curve_speed
        if getattr(self, 'capture_fermat_curve', False):
            self.captured_fermat_curve = curve_points.detach()
        return (
            stored_curve_time, stored_straight_time,
            physical_curve_time, physical_straight_time)

    def _fermat_waypoint_tau(self, coords, features, size):
        """Travel time of one learned, symmetric two-segment Fermat path.

        The waypoint is produced by this same network and every path cost is
        integrated from its scalar speed head.  There is no candidate bank or
        inference-time optimizer.  The construction provides, exactly:

        * ``T(q, q) = 0``;
        * ``T(q, g) = T(g, q)``;
        * ``T(q, g) >= ||q-g||`` because the speed head is at most one.

        Unlike the straight-chord upper bound, the path can bend around a
        low-speed set.  This is the smallest path decoder that can express a
        genuine detour while retaining one encoder and one checkpoint.
        """
        normalized = self.encoder_norm(features)
        z0 = normalized[:size]
        z1 = normalized[size:]
        q0 = coords[:, :self.dim]
        q1 = coords[:, self.dim:]
        delta = q1 - q0
        distance = torch.linalg.vector_norm(
            delta, dim=1, keepdim=True)
        endpoint_points = torch.cat([q0, q1], dim=0)
        endpoint_speed = self.predict_speed(
            endpoint_points, features=features)
        s0 = endpoint_speed[:size]
        s1 = endpoint_speed[size:]
        pair = torch.cat([
            z0 + z1,
            (z1 - z0).abs(),
            z0 * z1,
            delta.abs(),
            distance,
            s0 + s1,
            (s1 - s0).abs(),
        ], dim=1)

        # The pair features are invariant to swapping q0/q1, hence the same
        # physical waypoint is predicted in both directions.  Scale the bend
        # by endpoint distance for an exact zero boundary and by the available
        # joint-limit room so quadrature points stay in normalized bounds.
        midpoint = 0.5 * (q0 + q1)
        room = (0.5 - midpoint.abs()).clamp_min(0.0)
        boundary_gate = distance / (distance + 0.05)
        bend = torch.tanh(self._global_pair_output(pair))
        waypoint = midpoint + room * boundary_gate * bend

        # Envelope theorem for a variational path: once the inner waypoint
        # minimization has produced w*(q,g), d min_w C(q,g,w)/dq is the partial
        # derivative of C with w held fixed.  Backpropagating through the
        # waypoint predictor at inference adds an unrelated dw*/dq term and
        # can turn a good path into a bad descent vector.  Keep the full graph
        # in training so the path head remains learnable; use the mathematically
        # correct envelope derivative in eval/MPPI.
        cost_waypoint = waypoint if self.training else waypoint.detach()

        nodes = self.chord_nodes.to(dtype=coords.dtype)
        weights = self.chord_weights.to(dtype=coords.dtype)
        segment_start = torch.stack([q0, cost_waypoint], dim=1)
        segment_end = torch.stack([cost_waypoint, q1], dim=1)
        segment_delta = segment_end - segment_start
        segment_length = torch.linalg.vector_norm(
            segment_delta, dim=2)
        path_points = (
            segment_start[:, :, None, :]
            + nodes[None, None, :, None]
            * segment_delta[:, :, None, :])
        path_flat = path_points.reshape(-1, self.dim)
        path_speed = self.predict_speed(
            path_flat).reshape(size, 2, self.chord_quad_points)
        segment_slowness = (
            path_speed.clamp_min(self.speed_floor).reciprocal()
            * weights[None, None, :]).sum(dim=2)
        return (segment_length * segment_slowness).sum(
            dim=1, keepdim=True)

    def _fermat_curve_tau(self, coords, features, size):
        """Travel time along one smooth reversible curve from the same model.

        Odd sine modes vanish at both endpoints and are invariant under
        ``t -> 1-t``.  Pair coefficients are endpoint-swap invariant, so the
        physical curve is exactly the same when traversed in reverse.  There
        is one curve, no candidate set and no inference-time optimization.
        """
        normalized = self.encoder_norm(features)
        z0 = normalized[:size]
        z1 = normalized[size:]
        q0 = coords[:, :self.dim]
        q1 = coords[:, self.dim:]
        delta = q1 - q0
        distance = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
        endpoint_points = torch.cat([q0, q1], dim=0)
        endpoint_speed = self.predict_speed(
            endpoint_points, features=features)
        s0 = endpoint_speed[:size]
        s1 = endpoint_speed[size:]
        pair = torch.cat([
            z0 + z1,
            (z1 - z0).abs(),
            z0 * z1,
            delta.abs(),
            distance,
            s0 + s1,
            (s1 - s0).abs(),
        ], dim=1)

        raw = torch.tanh(self._global_pair_output(pair)).reshape(
            size, self.fermat_curve_modes, self.dim)
        # A conservative endpoint envelope keeps the complete curve inside
        # normalized joint limits.  The distance gate preserves T(q,q)=0.
        room = (0.5 - torch.maximum(q0.abs(), q1.abs())).clamp_min(0.0)
        distance_gate = distance / (distance + 0.05)
        coefficients = (
            raw * room[:, None, :] * distance_gate[:, None, :]
            / float(self.fermat_curve_modes))
        if not self.training:
            # Envelope theorem: the derivative of the minimized path cost
            # holds the inner curve optimum fixed.
            coefficients = coefficients.detach()

        nodes = self.chord_nodes.to(dtype=coords.dtype)
        weights = self.chord_weights.to(dtype=coords.dtype)
        odd_modes = (
            2 * torch.arange(
                self.fermat_curve_modes,
                device=coords.device, dtype=coords.dtype) + 1)
        phase = (
            math.pi * odd_modes[:, None] * nodes[None, :])
        basis = torch.sin(phase)
        basis_derivative = (
            math.pi * odd_modes[:, None] * torch.cos(phase))

        base = q0[:, None, :] + nodes[None, :, None] * delta[:, None, :]
        residual = torch.einsum('bmd,mk->bkd', coefficients, basis)
        curve_points = base + residual
        curve_derivative = (
            delta[:, None, :]
            + torch.einsum(
                'bmd,mk->bkd', coefficients, basis_derivative))
        curve_flat = curve_points.reshape(-1, self.dim)
        curve_speed = self.predict_speed(
            curve_flat).reshape(size, self.chord_quad_points)
        integrand = (
            torch.linalg.vector_norm(curve_derivative, dim=2)
            / curve_speed.clamp_min(self.speed_floor))
        return (integrand * weights[None, :]).sum(dim=1, keepdim=True)

    def _fermat_curve_equivariant_tau(self, coords, features, size):
        """Optical length of one unrestricted, exactly reversible curve.

        A general endpoint-preserving curve needs every sine mode.  Under an
        endpoint swap, odd-mode coefficients are invariant while even-mode
        coefficients change sign.  One shared directed pair network is run in
        both orders and projected onto those two parity subspaces.  This is an
        algebraic symmetry of one model, not an ensemble or a candidate set.

        The historical scalar target stores physical distance in ``[0.1, 1]``:
        ``0.1`` is the strict invalid-set boundary and ``1`` is free space.
        Removing that affine storage offset recovers the state-constrained PDE
        speed, which tends to zero at the invalid boundary.  No extra label or
        head is involved.
        """
        normalized = self.encoder_norm(features)
        z0 = normalized[:size]
        z1 = normalized[size:]
        q0 = coords[:, :self.dim]
        q1 = coords[:, self.dim:]
        delta = q1 - q0
        distance = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
        endpoint_points = torch.cat([q0, q1], dim=0)
        endpoint_speed = self.predict_speed(
            endpoint_points, features=features)
        s0 = endpoint_speed[:size]
        s1 = endpoint_speed[size:]

        pair_forward = torch.cat([
            z0, z1, z1 - z0, z0 * z1,
            delta, delta.abs(), distance, s0, s1,
        ], dim=1)
        pair_reverse = torch.cat([
            z1, z0, z0 - z1, z1 * z0,
            -delta, delta.abs(), distance, s1, s0,
        ], dim=1)
        raw_forward = self._global_pair_output(pair_forward).reshape(
            size, self.fermat_curve_modes, self.dim)
        raw_reverse = self._global_pair_output(pair_reverse).reshape(
            size, self.fermat_curve_modes, self.dim)

        mode_number = torch.arange(
            1, self.fermat_curve_modes + 1,
            device=coords.device, dtype=coords.dtype)
        # sin(n*pi*(1-t)) = parity_n * sin(n*pi*t).
        parity = torch.where(
            mode_number.remainder(2) == 1,
            torch.ones_like(mode_number),
            -torch.ones_like(mode_number))
        equivariant_raw = 0.5 * (
            raw_forward + parity[None, :, None] * raw_reverse)
        distance_gate = distance / (distance + 0.05)
        # Dividing by mode number keeps every mode's derivative on the same
        # scale while leaving the first global bend fully expressive.
        coefficients = (
            torch.tanh(equivariant_raw)
            * distance_gate[:, None, :]
            / mode_number[None, :, None])
        # The default retains the full derivative because this is an
        # amortized curve, not an exactly solved inner optimum for every pair.
        # The explicit envelope variant tests the complementary Fermat limit:
        # once training has made the inner curve nearly stationary,
        # d(min_curve C)/dq holds its coefficients fixed.  The flag is stored
        # in the checkpoint and changes no forward value or supervision.
        if not self.training and self.fermat_curve_envelope_detach:
            coefficients = coefficients.detach()

        nodes = self.chord_nodes.to(dtype=coords.dtype)
        weights = self.chord_weights.to(dtype=coords.dtype)
        base = q0[:, None, :] + nodes[None, :, None] * delta[:, None, :]
        stored_chord_speed = self.predict_speed(
            base.reshape(-1, self.dim)).reshape(
                size, self.chord_quad_points)
        harmonic_chord_speed = stored_chord_speed.clamp_min(
            self.speed_floor).reciprocal().mean(
                dim=1, keepdim=True).reciprocal()
        # Fermat's principle makes the straight chord exact in a homogeneous
        # free region.  Gate learned bends by obstruction measured from this
        # same scalar field, so approximation error cannot invent a detour on
        # an actually free chord.  This is continuous and label-free.
        chord_hazard = (1.0 - harmonic_chord_speed).clamp(0.0, 1.0)
        coefficients = coefficients * chord_hazard[:, None, :]
        phase = math.pi * mode_number[:, None] * nodes[None, :]
        basis = torch.sin(phase)
        basis_derivative = (
            math.pi * mode_number[:, None] * torch.cos(phase))
        residual = torch.einsum('bmd,mk->bkd', coefficients, basis)
        residual_derivative = torch.einsum(
            'bmd,mk->bkd', coefficients, basis_derivative)

        # Work in an atanh coordinate so every quadrature point remains
        # inside normalized joint limits while the zero-residual curve is
        # exactly the endpoint chord.
        base_unit = (2.0 * base).clamp(-1.0 + 1e-5, 1.0 - 1e-5)
        base_logit = torch.atanh(base_unit)
        latent_curve = base_logit + residual
        squashed = torch.tanh(latent_curve)
        curve_points = 0.5 * squashed
        if getattr(self, "capture_fermat_curve", False):
            # Explicit opt-in for offline architecture audits.  Normal
            # training and inference retain no intermediate path tensor.
            self.captured_fermat_curve = curve_points.detach()
        base_logit_derivative = (
            2.0 * delta[:, None, :]
            / (1.0 - base_unit.square()).clamp_min(1e-5))
        curve_derivative = (
            0.5 * (1.0 - squashed.square())
            * (base_logit_derivative + residual_derivative))

        curve_flat = curve_points.reshape(-1, self.dim)
        stored_speed = self.predict_speed(curve_flat)
        # The learned scalar is affine-coded distance.  Undo only that coding;
        # the small numerical floor keeps quadrature finite at the boundary.
        physical_speed = (
            (stored_speed - self.speed_floor)
            / (1.0 - self.speed_floor)).clamp(0.0, 1.0)
        physical_speed = (
            self.clearance_speed_floor
            + (1.0 - self.clearance_speed_floor) * physical_speed)
        physical_speed = physical_speed.reshape(
            size, self.chord_quad_points)
        integrand = (
            torch.linalg.vector_norm(curve_derivative, dim=2)
            / physical_speed)
        return (integrand * weights[None, :]).sum(dim=1, keepdim=True)

    def _predict_speed_logit(self, points, features=None,
                             apply_lifted_phase=True):
        """Return the one scalar field's logit before its storage mapping.

        ``features`` may reuse the value encoding in the historical shared
        architecture.  The separate architecture deliberately recomputes a
        multiscale local encoding from ``points`` while remaining coupled to
        the same travel-time forward pass.
        """
        if not self.use_speed_head:
            raise RuntimeError("scalar speed head is disabled")
        if self.separate_speed_encoder:
            z = self.speed_encoder_norm(
                self._encode_speed_points(points))
        else:
            if features is None:
                features = self._encode_scalar_points(points)
            z = self.encoder_norm(features)
        for block in getattr(self, 'speed_residual_blocks', []):
            z = block(z)
        z = self._apply_partial_shared_speed(z, points)
        logit = self._speed_logit_from_latent(z, points)
        if self.lifted_phase and apply_lifted_phase:
            if points is None:
                raise RuntimeError(
                    "lifted phase speed requires point coordinates")
            logit = self._apply_lifted_phase_logit(points, z, logit)
        return logit

    def _base_speed_logit_from_latent(self, latent, points):
        """Evaluate the inherited scalar head before semantic composition."""
        logit = self.speed_head(latent)
        return self._apply_speed_correction(logit, points)

    def _speed_constraint_components_from_latent(
            self, latent, points, base_logit=None):
        """Predict signed dimensionless clearance factors from one q latent.

        These factors are training-visible diagnostics, not multiple deployed
        propagation fields.  The base scalar plus zero-initialized residuals
        makes all factors identical at initialization, which exactly preserves
        the inherited scalar under normalized soft-min composition.
        """
        heads = getattr(self, 'speed_constraint_heads', None)
        if heads is None or len(heads) == 0:
            raise RuntimeError("scalar constraint-component heads are disabled")
        if base_logit is None:
            base_logit = self._base_speed_logit_from_latent(latent, points)
        base_speed = self.speed_floor + (
            1.0 - self.speed_floor) * torch.sigmoid(base_logit)
        head_input = latent
        if self.speed_constraint_fourier_skip:
            if points is None:
                raise RuntimeError(
                    "constraint Fourier skip requires point coordinates")
            basis = self.B_speed if self.separate_speed_encoder else self.B
            raw_fourier = self.input_mapping(points, basis)
            if raw_fourier.shape[1] != latent.shape[1]:
                raise RuntimeError(
                    "constraint Fourier/latent widths differ: "
                    f"{raw_fourier.shape[1]} vs {latent.shape[1]}")
            head_input = torch.cat([latent, raw_fourier], dim=1)
        residual = torch.cat([head(head_input) for head in heads], dim=1)
        components = (
            base_speed
            + self.speed_constraint_residual_scale * torch.tanh(residual))
        return components.clamp(-1.0, 1.0)

    def _aggregate_speed_constraint_components(self, components):
        """Analytically reduce component clearances to the one scalar F(q)."""
        if (components.ndim != 2
                or components.shape[1] != self.speed_constraint_head_count):
            raise ValueError(
                "constraint components must have shape (N,H), got "
                f"{tuple(components.shape)}")
        temperature = self.speed_constraint_softmin_temperature
        normalized_log_mean_exp = (
            torch.logsumexp(-components / temperature, dim=1, keepdim=True)
            - math.log(self.speed_constraint_head_count))
        aggregate = -temperature * normalized_log_mean_exp
        return aggregate.clamp(self.speed_floor, 1.0)

    @torch.no_grad()
    def fit_speed_constraint_route_charts(self, support_points, iterations=40):
        """Fit the registered deterministic nearest-centroid q-space gate."""
        if self.speed_constraint_consensus_mode not in ROUTE_CHART_MODES:
            raise RuntimeError("configuration-chart routing is disabled")
        if (support_points.ndim != 2
                or support_points.shape[1] != self.dim
                or support_points.shape[0] < ROUTE_CHART_COUNT):
            raise ValueError(
                "support_points must have shape (N,dim) with N >= "
                f"{ROUTE_CHART_COUNT}, got {tuple(support_points.shape)}")
        values = support_points.detach().cpu().numpy().astype(
            np.float64, copy=False)
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1)
        std = np.maximum(std, 1e-4)
        standardized = (values - mean) / std
        rng = np.random.default_rng(ROUTE_CHART_SEED)
        centers = [standardized[int(rng.integers(len(standardized)))]]
        for _ in range(1, ROUTE_CHART_COUNT):
            stacked = np.stack(centers)
            distance = np.square(
                standardized[:, None, :] - stacked[None, :, :]).sum(axis=2)
            nearest = distance.min(axis=1)
            probability = nearest / max(float(nearest.sum()), 1e-12)
            centers.append(standardized[int(rng.choice(
                len(standardized), p=probability))])
        centers = np.stack(centers)
        assignment = np.full(len(standardized), -1, dtype=np.int64)
        for _ in range(iterations):
            distance = np.square(
                standardized[:, None, :] - centers[None, :, :]).sum(axis=2)
            new_assignment = distance.argmin(axis=1)
            new_centers = np.stack([
                standardized[new_assignment == index].mean(axis=0)
                for index in range(ROUTE_CHART_COUNT)
            ])
            centers = new_centers
            if np.array_equal(new_assignment, assignment):
                assignment = new_assignment
                break
            assignment = new_assignment
        distance = np.square(
            standardized[:, None, :] - centers[None, :, :]).sum(axis=2)
        assigned_distance = distance[
            np.arange(len(standardized)), assignment]
        temperature = max(
            float(np.median(assigned_distance))
            * ROUTE_CHART_TEMPERATURE_SCALE,
            1e-6)
        device = self.speed_constraint_route_chart_mean.device
        dtype = self.speed_constraint_route_chart_mean.dtype
        self.speed_constraint_route_chart_mean.copy_(
            torch.as_tensor(mean, device=device, dtype=dtype))
        self.speed_constraint_route_chart_std.copy_(
            torch.as_tensor(std, device=device, dtype=dtype))
        self.speed_constraint_route_chart_centers.copy_(
            torch.as_tensor(centers, device=device, dtype=dtype))
        self.speed_constraint_route_chart_temperature.copy_(
            torch.as_tensor(temperature, device=device, dtype=dtype))
        self.speed_constraint_route_chart_fitted.fill_(True)
        return {
            "counts": np.bincount(
                assignment, minlength=ROUTE_CHART_COUNT).tolist(),
            "temperature": temperature,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "centers": centers.tolist(),
        }

    def _speed_constraint_route_chart_gate(self, points):
        """Return one smooth q-only simplex weight per route chart."""
        if self.speed_constraint_consensus_mode not in ROUTE_CHART_MODES:
            raise RuntimeError("configuration-chart routing is disabled")
        if points is None:
            raise ValueError("configuration-chart routing requires q points")
        if not bool(self.speed_constraint_route_chart_fitted.item()):
            return points.new_full(
                (points.shape[0], ROUTE_CHART_COUNT),
                1.0 / ROUTE_CHART_COUNT)
        standardized = (
            (points - self.speed_constraint_route_chart_mean)
            / self.speed_constraint_route_chart_std.clamp_min(1e-4))
        distance = torch.cdist(
            standardized,
            self.speed_constraint_route_chart_centers,
            p=2).square()
        temperature = self.speed_constraint_route_chart_temperature.clamp_min(
            1e-6)
        return torch.softmax(-distance / temperature, dim=1)

    def _speed_constraint_fourier_risk_q_features(self, standardized_q):
        """Return q plus the fixed Phase-BV four-band sine/cosine features."""
        if (standardized_q.ndim != 2
                or standardized_q.shape[1] != self.dim):
            raise ValueError(
                f"standardized_q must have shape (N,{self.dim})")
        frequencies = self.speed_constraint_fourier_risk_frequencies.to(
            device=standardized_q.device, dtype=standardized_q.dtype)
        phase = (
            math.pi * standardized_q[:, :, None]
            * frequencies[None, None, :])
        return torch.cat([
            standardized_q,
            torch.sin(phase).flatten(start_dim=1),
            torch.cos(phase).flatten(start_dim=1),
        ], dim=1)

    def _speed_constraint_route_fourier_q_features(self, q_normalized):
        """Return q plus fixed four-band features for Phase-BY route heads."""
        if (q_normalized.ndim != 2
                or q_normalized.shape[1] != self.dim):
            raise ValueError(
                f"q_normalized must have shape (N,{self.dim})")
        frequencies = self.speed_constraint_route_fourier_frequencies.to(
            device=q_normalized.device, dtype=q_normalized.dtype)
        phase = (
            math.pi * q_normalized[:, :, None]
            * frequencies[None, None, :])
        return torch.cat([
            q_normalized,
            torch.sin(phase).flatten(start_dim=1),
            torch.cos(phase).flatten(start_dim=1),
        ], dim=1)

    def _speed_constraint_route_residual(self, hidden, points):
        """Evaluate legacy or partial-shared private route residuals."""
        if (self.speed_constraint_consensus_mode
                in ROUTE_CHART_PARTIAL_SHARED_MODES):
            if points is None:
                raise ValueError("partial-shared Fourier route heads require q")
            q_features = self._speed_constraint_route_fourier_q_features(
                points)
            shared_fourier = self.speed_constraint_route_fourier_branch(
                q_features)
            raw = torch.cat([
                expert(hidden, shared_fourier)
                for expert in self.speed_constraint_consensus_route_experts
            ], dim=1)
        else:
            raw = torch.cat([
                expert(hidden)
                for expert in self.speed_constraint_consensus_route_experts
            ], dim=1)
        return 8.0 * torch.tanh(raw)

    def _speed_constraint_fourier_risk_logits(self, hidden, context):
        """Evaluate the four internal Phase-BV specialists."""
        if (self.speed_constraint_consensus_mode
                not in ROUTE_CHART_FOURIER_RISK_MODES):
            raise RuntimeError("Fourier physical/risk heads are disabled")
        expected_context = self.dim + self.speed_constraint_head_count + 10
        if context.shape != (hidden.shape[0], expected_context):
            raise ValueError(
                "Fourier physical/risk context has unexpected shape: "
                f"{tuple(context.shape)}")
        q_features = self._speed_constraint_fourier_risk_q_features(
            context[:, :self.dim])
        shared = self.speed_constraint_fourier_risk_fusion(torch.cat([
            self.speed_constraint_fourier_risk_hidden_branch(hidden),
            self.speed_constraint_fourier_risk_context_branch(context),
            self.speed_constraint_fourier_risk_q_branch(q_features),
        ], dim=1))
        return torch.cat([
            tail(shared)
            for tail in self.speed_constraint_fourier_risk_private_tails
        ], dim=1)

    def _apply_speed_constraint_consensus(
            self, latent, base_speed, components, aggregate, points=None):
        """Fuse semantic soft-min and direct shared estimates into one F."""
        if self.speed_constraint_consensus_width < 1:
            return aggregate
        if self.speed_constraint_consensus_mode in (
                ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_ANALYTIC_EXACT_SCALAR_CONSENSUS_MODE,
                ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_ANALYTIC_TRUST_CONTINUATION_CONSENSUS_MODE,
                ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_ANALYTIC_EXACT_TRUST_ANCHOR_POTENTIAL_CONSENSUS_MODE,
                ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_ANALYTIC_EXACT_TRUST_WINNER_TAKE_HEAD_CONSENSUS_MODE,
                *ANALYTIC_HEADWISE_INTEGRATED_T_MODES):
            if points is None:
                raise ValueError("analytic scalar composition requires q")
            margins = self._speed_constraint_analytic_continuous_margins(
                points)
            primitive_components = (
                margins
                + self.speed_constraint_fullsemantic_component_boundary)
            if self.speed_constraint_consensus_mode in (
                    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_ANALYTIC_TRUST_CONTINUATION_CONSENSUS_MODE,
                    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_ANALYTIC_EXACT_TRUST_ANCHOR_POTENTIAL_CONSENSUS_MODE,
                    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_ANALYTIC_EXACT_TRUST_WINNER_TAKE_HEAD_CONSENSUS_MODE,
                    *ANALYTIC_HEADWISE_INTEGRATED_T_MODES):
                raw_scalar = analytic_unclamped_scalar_from_components(
                    primitive_components,
                    self.speed_constraint_softmin_temperature)
                physical_speed = analytic_trust_boundary_continuation(
                    raw_scalar, self.speed_floor,
                    ANALYTIC_TRUST_BOUNDARY)
                if self.speed_constraint_consensus_mode == (
                        ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_ANALYTIC_EXACT_TRUST_ANCHOR_POTENTIAL_CONSENSUS_MODE):
                    anchor_speed = self._predict_speed_constraint_anchor(
                        points)
                    return analytic_exact_trust_anchor_potential(
                        physical_speed, anchor_speed, self.speed_floor,
                        ANALYTIC_TRUST_BOUNDARY)
                if self.speed_constraint_consensus_mode == (
                        ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_ANALYTIC_EXACT_TRUST_WINNER_TAKE_HEAD_CONSENSUS_MODE):
                    anchor_speed = self._predict_speed_constraint_anchor(
                        points)
                    return analytic_exact_trust_winner_take_head(
                        physical_speed, anchor_speed, self.speed_floor,
                        ANALYTIC_TRUST_BOUNDARY)
                return physical_speed
            return analytic_exact_scalar_from_margins(
                primitive_components,
                self.speed_constraint_softmin_temperature,
                self.speed_floor)
        inputs = torch.cat(
            [latent, base_speed, components, aggregate], dim=1)
        alpha, _ = self._speed_constraint_consensus_from_inputs(
            inputs, points=points)
        if self.speed_constraint_consensus_mode in (
                'corridor_hurdle', 'corridor_hurdle_compensated',
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
            candidate = aggregate + alpha * torch.relu(
                base_speed - aggregate)
            if self.speed_constraint_consensus_mode == (
                    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_ANALYTIC_SPEED_ENVELOPE_CONSENSUS_MODE):
                if points is None:
                    raise ValueError("analytic final-speed envelope requires q")
                anchor_speed = self._predict_speed_constraint_anchor(points)
                margins = self._speed_constraint_analytic_continuous_margins(
                    points)
                approval = analytic_continuous_approval(
                    margins.min(dim=1, keepdim=True).values,
                    self.speed_constraint_analytic_continuous_tau)
                candidate = analytic_final_speed_envelope(
                    candidate, anchor_speed, approval)
            return candidate.clamp(self.speed_floor, 1.0)
        return torch.lerp(aggregate, base_speed, alpha).clamp(
            self.speed_floor, 1.0)

    def _speed_constraint_consensus_from_inputs(self, inputs, points=None):
        """Return one alpha and its optional zero-inflated private factors."""
        if self.speed_constraint_consensus_width < 1:
            raise RuntimeError("scalar speed consensus is disabled")
        if self.speed_constraint_consensus_mode == 'single':
            alpha = torch.sigmoid(self.speed_constraint_consensus_head(inputs))
            return alpha, alpha
        hidden = self.speed_constraint_consensus_trunk(inputs)
        release = torch.sigmoid(
            self.speed_constraint_consensus_heads['release'](hidden))
        magnitude_logit = (
            self.speed_constraint_consensus_heads['magnitude'](hidden))
        if self.speed_constraint_consensus_mode in (
                'corridor_hurdle_semantic_veto_min_odds_residual',
                'corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend',
                'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8',
                *ROUTE_CHART_MODES):
            magnitude_logit = (
                magnitude_logit
                + self.speed_constraint_consensus_magnitude_residual(hidden))
        magnitude = torch.sigmoid(magnitude_logit)
        if self.speed_constraint_consensus_mode in (
                'corridor_hurdle_specificity',
                'corridor_hurdle_specificity_odds',
                'corridor_hurdle_specificity_orthogonal',
                'corridor_hurdle_semantic_veto_min_orthogonal',
                'corridor_hurdle_semantic_veto_min_odds',
                'corridor_hurdle_semantic_veto_min_odds_residual',
                'corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend',
                'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8',
                *ROUTE_CHART_MODES):
            specificity = torch.sigmoid(
                self.speed_constraint_consensus_heads['specificity'](hidden))
            factors = torch.cat(
                [release, specificity, magnitude], dim=1)
            if self.speed_constraint_consensus_mode in (
                    'corridor_hurdle_semantic_veto_min_orthogonal',
                    'corridor_hurdle_semantic_veto_min_odds',
                    'corridor_hurdle_semantic_veto_min_odds_residual',
                    'corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend',
                    'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8',
                    *ROUTE_CHART_MODES):
                # Each private tail predicts whether one physical component
                # permits an upward aggregate repair.  A single unsafe
                # component must retain veto power, so the deployed scalar
                # uses their exact weakest-link conjunction.  The endpoint
                # model still receives only one q-only alpha and one F(q).
                semantic_permission = specificity.min(
                    dim=1, keepdim=True).values
                if self.speed_constraint_consensus_mode in (
                        'corridor_hurdle_semantic_veto_min_odds',
                        'corridor_hurdle_semantic_veto_min_odds_residual',
                        'corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend',
                        'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8',
                        *ROUTE_CHART_MODES):
                    # A 0.5 weakest-link permission is neutral.  Values above
                    # or below 0.5 add positive or negative evidence to the
                    # inherited coarse release in log-odds space, so the
                    # semantic heads can lift true support without losing
                    # their veto direction.  Deployment is still one scalar
                    # q-only alpha and one final F(q).
                    if (self.speed_constraint_consensus_mode
                            == 'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8'):
                        head_logit_residual = 8.0 * torch.tanh(
                            self.speed_constraint_consensus_head_logit_residual(
                                hidden))
                        corrected_specificity = torch.sigmoid(
                            torch.logit(specificity.clamp(1e-6, 1.0 - 1e-6))
                            + head_logit_residual)
                        semantic_permission = corrected_specificity.min(
                            dim=1, keepdim=True).values
                        factors = torch.cat(
                            [release, specificity, magnitude,
                             head_logit_residual], dim=1)
                    effective_release = neutral_log_odds_fusion(
                        release, semantic_permission)
                    if self.speed_constraint_consensus_mode in ROUTE_CHART_MODES:
                        route_residual = self._speed_constraint_route_residual(
                            hidden, points)
                        route_release = torch.sigmoid(
                            torch.logit(
                                effective_release.clamp(1e-6, 1.0 - 1e-6))
                            + route_residual)
                        route_gate = self._speed_constraint_route_chart_gate(
                            points)
                        if (self.speed_constraint_consensus_mode
                                in ROUTE_CHART_UNION_MODES):
                            effective_release = normalized_release_smoothmax(
                                route_release)
                        else:
                            effective_release = (
                                route_gate * route_release).sum(
                                    dim=1, keepdim=True)
                        factor_parts = [
                            release, specificity, magnitude,
                            route_release, route_gate]
                        if (self.speed_constraint_consensus_mode
                                in ROUTE_CHART_FOURIER_RISK_MODES):
                            if points is None:
                                raise ValueError(
                                    "Fourier physical/risk veto requires q")
                            standardized_q = (
                                (points - self.speed_constraint_route_chart_mean)
                                / self.speed_constraint_route_chart_std.clamp_min(
                                    1e-4))
                            direct_index = -(
                                self.speed_constraint_head_count + 2)
                            direct = inputs[:, direct_index:direct_index + 1]
                            repaired = inputs[:, -1:]
                            context = torch.cat([
                                standardized_q,
                                direct,
                                repaired,
                                magnitude,
                                release,
                                specificity,
                                neutral_log_odds_fusion(
                                    release, semantic_permission),
                                route_release,
                                effective_release,
                            ], dim=1)
                            semantic_logits = (
                                self._speed_constraint_fourier_risk_logits(
                                    hidden, context))
                            evidence = semantic_logits.min(
                                dim=1, keepdim=True).values
                            if self.speed_constraint_consensus_mode in (
                                    ROUTE_CHART_ANALYTIC_CONTINUOUS_MODES):
                                # Phase-DV replaces saturated learned logits
                                # with four exact signed geometry margins.
                                # Their minimum controls only one bounded
                                # anchor increment; the public field remains
                                # one q-only scalar.
                                ch_release = margin_signed_release_rescue(
                                    effective_release, evidence)
                                anchor_hidden = (
                                    self._speed_constraint_anchor_hidden(
                                        points))
                                anchor_logit = (
                                    self.speed_constraint_anchor_head(
                                        anchor_hidden))
                                anchor_speed = self.speed_floor + (
                                    1.0 - self.speed_floor) * torch.sigmoid(
                                        anchor_logit)
                                anchor_release = scalar_speed_to_release(
                                    anchor_speed, repaired, direct, magnitude)
                                margins = (
                                    self._speed_constraint_analytic_continuous_margins(
                                        points))
                                weakest_margin = margins.min(
                                    dim=1, keepdim=True).values
                                approval = analytic_continuous_approval(
                                    weakest_margin,
                                    self.speed_constraint_analytic_continuous_tau)
                                unrestricted = monotone_anchor_release_union(
                                    ch_release, anchor_release)
                                effective_release = (
                                    ch_release + approval
                                    * (unrestricted - ch_release))
                                factor_parts.extend([
                                    semantic_logits,
                                    margins,
                                    weakest_margin,
                                    approval,
                                ])
                                factors = torch.cat(factor_parts, dim=1)
                                return effective_release * magnitude, factors
                            if self.speed_constraint_consensus_mode == (
                                    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_FULLSEMANTIC_ANCHOR_RESCUE_CONSENSUS_MODE):
                                # Phase-DF uses the frozen Phase-CH release as
                                # its baseline.  Four relation-semantic heads
                                # decide whether the formal scalar anchor may
                                # contribute one bounded monotone increment.
                                # No head becomes a propagation field.
                                ch_release = margin_signed_release_rescue(
                                    effective_release, evidence)
                                anchor_hidden = (
                                    self._speed_constraint_anchor_hidden(
                                        points))
                                anchor_logit = (
                                    self.speed_constraint_anchor_head(
                                        anchor_hidden))
                                anchor_speed = self.speed_floor + (
                                    1.0 - self.speed_floor) * torch.sigmoid(
                                        anchor_logit)
                                anchor_release = scalar_speed_to_release(
                                    anchor_speed, repaired, direct, magnitude)
                                gap = torch.relu(direct - repaired)
                                ch_speed = (
                                    repaired + ch_release * magnitude * gap)
                                fullsemantic_context = torch.cat([
                                    anchor_speed,
                                    ch_speed,
                                    direct,
                                    repaired,
                                    magnitude,
                                    ch_release,
                                    anchor_release,
                                    gap,
                                ], dim=1)
                                fullsemantic_evidence, replica_logits = (
                                    self._speed_constraint_fullsemantic_evidence(
                                        points, anchor_hidden,
                                        fullsemantic_context))
                                approval = semantic_anchor_approval(
                                    fullsemantic_evidence)
                                unrestricted = monotone_anchor_release_union(
                                    ch_release, anchor_release)
                                effective_release = (
                                    ch_release + approval
                                    * (unrestricted - ch_release))
                                factor_parts.extend([
                                    semantic_logits,
                                    replica_logits.permute(1, 0, 2).reshape(
                                        points.shape[0], -1),
                                    fullsemantic_evidence,
                                    approval,
                                ])
                                factors = torch.cat(factor_parts, dim=1)
                                return effective_release * magnitude, factors
                            if self.speed_constraint_consensus_mode == (
                                    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_REPLICA_MEDIAN_ANCHOR_RESCUE_CONSENSUS_MODE):
                                # Phase-CR reconstructs the passed Phase-CQ
                                # field exactly: CH first, then one monotone
                                # anchor increment controlled by the median
                                # of three weakest-five-head evidences.
                                ch_release = margin_signed_release_rescue(
                                    effective_release, evidence)
                                anchor_hidden = (
                                    self._speed_constraint_anchor_hidden(
                                        points))
                                anchor_logit = (
                                    self.speed_constraint_anchor_head(
                                        anchor_hidden))
                                anchor_speed = self.speed_floor + (
                                    1.0 - self.speed_floor) * torch.sigmoid(
                                        anchor_logit)
                                anchor_release = scalar_speed_to_release(
                                    anchor_speed, repaired, direct, magnitude)
                                gap = torch.relu(direct - repaired)
                                ch_speed = (
                                    repaired + ch_release * magnitude * gap)
                                reliability_context = torch.cat([
                                    anchor_speed,
                                    ch_speed,
                                    direct,
                                    repaired,
                                    magnitude,
                                    ch_release,
                                    anchor_release,
                                    gap,
                                ], dim=1)
                                replica_logits = (
                                    self._speed_constraint_anchor_reliability_logits(
                                        points, anchor_hidden,
                                        reliability_context))
                                effective_release, _, _ = (
                                    replica_median_positive_anchor_release(
                                        ch_release, anchor_release,
                                        replica_logits))
                                factor_parts.extend([
                                    semantic_logits,
                                    replica_logits.permute(1, 0, 2).reshape(
                                        points.shape[0], -1),
                                ])
                                factors = torch.cat(factor_parts, dim=1)
                                return effective_release * magnitude, factors
                            if (self.speed_constraint_consensus_mode ==
                                    ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_SEMANTIC_ANCHOR_RESCUE_CONSENSUS_MODE):
                                anchor_speed = (
                                    self._predict_speed_constraint_anchor(
                                        points))
                                anchor_release = scalar_speed_to_release(
                                    anchor_speed, repaired, direct, magnitude)
                                effective_release = (
                                    semantic_approved_anchor_release(
                                        effective_release,
                                        anchor_release,
                                        evidence))
                                effective_release = (
                                    margin_signed_release_rescue(
                                        effective_release, evidence))
                            elif (self.speed_constraint_consensus_mode
                                    == ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_MARGIN_RESCUE_CONSENSUS_MODE):
                                effective_release = margin_signed_release_rescue(
                                    effective_release, evidence)
                            else:
                                effective_release = monotone_release_veto(
                                    effective_release, evidence)
                            factor_parts.append(semantic_logits)
                        factors = torch.cat(factor_parts, dim=1)
                        return effective_release * magnitude, factors
                    if (self.speed_constraint_consensus_mode
                            == 'corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend'):
                        smooth_permission = (
                            normalized_logit_softmin_probability(
                                specificity, temperature=1.0))
                        smooth_release = neutral_log_odds_fusion(
                            release, smooth_permission)
                        evidence_blend = torch.tanh(
                            self.speed_constraint_consensus_evidence_blend_residual(
                                hidden))
                        effective_release = torch.clamp(
                            effective_release
                            + evidence_blend
                            * (smooth_release - effective_release),
                            min=0.0, max=1.0)
                        factors = torch.cat(
                            [release, specificity, magnitude,
                             evidence_blend], dim=1)
                    return effective_release * magnitude, factors
                return release * semantic_permission * magnitude, factors
            if (self.speed_constraint_consensus_mode
                    == 'corridor_hurdle_specificity_odds'):
                effective_release = neutral_log_odds_fusion(
                    release, specificity)
                return effective_release * magnitude, factors
            return release * specificity * magnitude, factors
        factors = torch.cat([release, magnitude], dim=1)
        return release * magnitude, factors

    def _speed_constraint_repair_from_inputs(self, inputs):
        """Return per-component repair and release/magnitude factors."""
        if self.speed_constraint_repair_width < 1:
            raise RuntimeError("scalar speed component repair is disabled")
        hidden = self.speed_constraint_repair_trunk(inputs)
        release = torch.sigmoid(
            self.speed_constraint_repair_heads['release'](hidden))
        magnitude = torch.sigmoid(
            self.speed_constraint_repair_heads['magnitude'](hidden))
        factors = torch.stack([release, magnitude], dim=2)
        return release * magnitude, factors

    def _apply_speed_constraint_repair(
            self, latent, base_speed, components, aggregate):
        """Raise each false-low candidate only toward the direct scalar."""
        if self.speed_constraint_repair_width < 1:
            return components
        inputs = torch.cat(
            [latent, base_speed, components, aggregate], dim=1)
        repair, _ = self._speed_constraint_repair_from_inputs(inputs)
        upward_gap = torch.relu(base_speed - components)
        return components + repair * upward_gap

    def _speed_logit_from_latent(self, latent, points):
        """Return the final scalar field logit from a computed speed latent."""
        base_logit = self._base_speed_logit_from_latent(latent, points)
        if self.speed_constraint_head_count == 0:
            return base_logit
        components = self._speed_constraint_components_from_latent(
            latent, points, base_logit=base_logit)
        aggregate = self._aggregate_speed_constraint_components(components)
        base_speed = self.speed_floor + (
            1.0 - self.speed_floor) * torch.sigmoid(base_logit)
        components = self._apply_speed_constraint_repair(
            latent, base_speed, components, aggregate)
        aggregate = self._aggregate_speed_constraint_components(components)
        aggregate = self._apply_speed_constraint_consensus(
            latent, base_speed, components, aggregate, points=points)
        affine_probability = (
            (aggregate - self.speed_floor) / (1.0 - self.speed_floor))
        eps = torch.finfo(aggregate.dtype).eps
        return torch.logit(affine_probability.clamp(eps, 1.0 - eps))

    def predict_speed_components(self, points, features=None):
        """Return training-only signed factors whose soft minimum is F(q)."""
        if self.speed_constraint_head_count == 0:
            raise RuntimeError("scalar constraint-component heads are disabled")
        if self.separate_speed_encoder:
            latent = self.speed_encoder_norm(
                self._encode_speed_points(points))
        else:
            if features is None:
                features = self._encode_scalar_points(points)
            latent = self.encoder_norm(features)
        for block in getattr(self, 'speed_residual_blocks', []):
            latent = block(latent)
        latent = self._apply_partial_shared_speed(latent, points)
        return self._speed_constraint_components_from_latent(latent, points)

    def predict_speed_constraint_repair_factors(
            self, points, features=None):
        """Return per-component release/magnitude values with shape N,H,2."""
        if self.speed_constraint_repair_width < 1:
            raise RuntimeError("scalar speed component repair is disabled")
        if self.separate_speed_encoder:
            latent = self.speed_encoder_norm(
                self._encode_speed_points(points))
        else:
            if features is None:
                features = self._encode_scalar_points(points)
            latent = self.encoder_norm(features)
        for block in getattr(self, 'speed_residual_blocks', []):
            latent = block(latent)
        latent = self._apply_partial_shared_speed(latent, points)
        base_logit = self._base_speed_logit_from_latent(latent, points)
        base_speed = self.speed_floor + (
            1.0 - self.speed_floor) * torch.sigmoid(base_logit)
        components = self._speed_constraint_components_from_latent(
            latent, points, base_logit=base_logit)
        aggregate = self._aggregate_speed_constraint_components(components)
        inputs = torch.cat(
            [latent, base_speed, components, aggregate], dim=1)
        _, factors = self._speed_constraint_repair_from_inputs(inputs)
        return factors

    def predict_repaired_speed_components(self, points, features=None):
        """Return internal components after the one-sided reliability repair."""
        if self.speed_constraint_repair_width < 1:
            raise RuntimeError("scalar speed component repair is disabled")
        if self.separate_speed_encoder:
            latent = self.speed_encoder_norm(
                self._encode_speed_points(points))
        else:
            if features is None:
                features = self._encode_scalar_points(points)
            latent = self.encoder_norm(features)
        for block in getattr(self, 'speed_residual_blocks', []):
            latent = block(latent)
        latent = self._apply_partial_shared_speed(latent, points)
        base_logit = self._base_speed_logit_from_latent(latent, points)
        base_speed = self.speed_floor + (
            1.0 - self.speed_floor) * torch.sigmoid(base_logit)
        components = self._speed_constraint_components_from_latent(
            latent, points, base_logit=base_logit)
        aggregate = self._aggregate_speed_constraint_components(components)
        return self._apply_speed_constraint_repair(
            latent, base_speed, components, aggregate)

    def predict_uncomposed_speed(self, points, features=None):
        """Return the diagnostic shared estimate before semantic soft-min.

        This is the common base used by every constraint residual head.  It is
        not the deployed propagation field; exposing it only lets audits
        measure whether independent head errors create a false-low union.
        """
        if not self.use_speed_head:
            raise RuntimeError("scalar speed head is disabled")
        if self.separate_speed_encoder:
            latent = self.speed_encoder_norm(
                self._encode_speed_points(points))
        else:
            if features is None:
                features = self._encode_scalar_points(points)
            latent = self.encoder_norm(features)
        for block in getattr(self, 'speed_residual_blocks', []):
            latent = block(latent)
        latent = self._apply_partial_shared_speed(latent, points)
        logit = self._base_speed_logit_from_latent(latent, points)
        return self.speed_floor + (
            1.0 - self.speed_floor) * torch.sigmoid(logit)

    def predict_speed_consensus_alpha(self, points, features=None):
        """Return diagnostic q-dependent trust in the direct scalar branch."""
        if self.speed_constraint_consensus_width < 1:
            raise RuntimeError("scalar speed consensus is disabled")
        if self.separate_speed_encoder:
            latent = self.speed_encoder_norm(
                self._encode_speed_points(points))
        else:
            if features is None:
                features = self._encode_scalar_points(points)
            latent = self.encoder_norm(features)
        for block in getattr(self, 'speed_residual_blocks', []):
            latent = block(latent)
        latent = self._apply_partial_shared_speed(latent, points)
        base_logit = self._base_speed_logit_from_latent(latent, points)
        base_speed = self.speed_floor + (
            1.0 - self.speed_floor) * torch.sigmoid(base_logit)
        components = self._speed_constraint_components_from_latent(
            latent, points, base_logit=base_logit)
        aggregate = self._aggregate_speed_constraint_components(components)
        components = self._apply_speed_constraint_repair(
            latent, base_speed, components, aggregate)
        aggregate = self._aggregate_speed_constraint_components(components)
        inputs = torch.cat(
            [latent, base_speed, components, aggregate], dim=1)
        alpha, _ = self._speed_constraint_consensus_from_inputs(
            inputs, points=points)
        return alpha

    def predict_speed_consensus_factors(self, points, features=None):
        """Return alpha, or the hurdle release/magnitude private outputs."""
        if self.speed_constraint_consensus_width < 1:
            raise RuntimeError("scalar speed consensus is disabled")
        if self.separate_speed_encoder:
            latent = self.speed_encoder_norm(
                self._encode_speed_points(points))
        else:
            if features is None:
                features = self._encode_scalar_points(points)
            latent = self.encoder_norm(features)
        for block in getattr(self, 'speed_residual_blocks', []):
            latent = block(latent)
        latent = self._apply_partial_shared_speed(latent, points)
        base_logit = self._base_speed_logit_from_latent(latent, points)
        base_speed = self.speed_floor + (
            1.0 - self.speed_floor) * torch.sigmoid(base_logit)
        components = self._speed_constraint_components_from_latent(
            latent, points, base_logit=base_logit)
        aggregate = self._aggregate_speed_constraint_components(components)
        components = self._apply_speed_constraint_repair(
            latent, base_speed, components, aggregate)
        aggregate = self._aggregate_speed_constraint_components(components)
        inputs = torch.cat(
            [latent, base_speed, components, aggregate], dim=1)
        _, factors = self._speed_constraint_consensus_from_inputs(
            inputs, points=points)
        return factors

    def predict_base_speed(self, points, features=None):
        """Return the inherited q-only scalar before the lifted restriction.

        This is used only to formulate the monotone residual's scalar
        regression target during training.  Deployment still exposes exactly
        one ``predict_speed`` output, namely the lifted field.
        """
        logit = self._predict_speed_logit(
            points, features=features, apply_lifted_phase=False)
        return self.speed_floor + (
            1.0 - self.speed_floor) * torch.sigmoid(logit)

    def predict_speed(self, points, features=None):
        """Predict the single scalar speed F(q) inside this network."""
        logit = self._predict_speed_logit(points, features=features)
        return self.speed_floor + (1.0 - self.speed_floor) * torch.sigmoid(
            logit)

    def predict_headwise_scalar_speeds(self, points, features=None):
        """Return the registered ordered q-only scalar propagation heads.

        These heads remain separate only for scalar integration.  The public
        pair output reduces their complete symmetric travel times to one T.
        """
        if self.speed_constraint_consensus_mode not in (
                ANALYTIC_HEADWISE_INTEGRATED_T_MODES):
            raise RuntimeError("headwise scalar integration mode is disabled")
        if points is None or points.ndim != 2 or points.shape[1] != self.dim:
            raise ValueError(
                f"points must have shape (N,{self.dim})")
        if self.separate_speed_encoder:
            latent = self.speed_encoder_norm(
                self._encode_speed_points(points))
        else:
            if features is None:
                features = self._encode_scalar_points(points)
            latent = self.encoder_norm(features)
        for block in getattr(self, 'speed_residual_blocks', []):
            latent = block(latent)
        latent = self._apply_partial_shared_speed(latent, points)
        base_logit = self._base_speed_logit_from_latent(latent, points)
        base_speed = self.speed_floor + (
            1.0 - self.speed_floor) * torch.sigmoid(base_logit)
        components = self._speed_constraint_components_from_latent(
            latent, points, base_logit=base_logit)
        aggregate = self._aggregate_speed_constraint_components(components)
        components = self._apply_speed_constraint_repair(
            latent, base_speed, components, aggregate)
        aggregate = self._aggregate_speed_constraint_components(components)
        inputs = torch.cat(
            [latent, base_speed, components, aggregate], dim=1)
        alpha, _ = self._speed_constraint_consensus_from_inputs(
            inputs, points=points)
        analytic_component = (
            aggregate + alpha * torch.relu(base_speed - aggregate)).clamp(
                self.speed_floor, 1.0)

        margins = self._speed_constraint_analytic_continuous_margins(points)
        primitive = (
            margins + self.speed_constraint_fullsemantic_component_boundary)
        raw = analytic_unclamped_scalar_from_components(
            primitive, self.speed_constraint_softmin_temperature)
        physical = analytic_trust_boundary_continuation(
            raw, self.speed_floor, ANALYTIC_TRUST_BOUNDARY)
        anchor = self._predict_speed_constraint_anchor(points)
        heads = [physical, anchor, analytic_component]
        if self.speed_constraint_consensus_mode == (
                ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_ANALYTIC_HANDOVER_HEADWISE_INTEGRATED_T_CONSENSUS_MODE):
            # Preserve P/L/A exactly and append the Phase-FS semantic
            # handover field.  The second deterministic FK call intentionally
            # mirrors the frozen diagnostic formula before any optimization.
            fk8, _ = self._speed_constraint_fullsemantic_kinematics(points)
            handover_primitive = (
                analytic_handover_components_from_primitives(primitive, fk8))
            raw_handover = analytic_unclamped_scalar_from_components(
                handover_primitive,
                self.speed_constraint_softmin_temperature)
            handover = analytic_trust_boundary_continuation(
                raw_handover, self.speed_floor, ANALYTIC_TRUST_BOUNDARY)
            heads.append(handover)
        return torch.cat(heads, dim=1)

    def headwise_scalar_chord_travel_times(self, q0, q1):
        """Return final T and all complete headwise chord integrals."""
        if self.speed_constraint_consensus_mode not in (
                ANALYTIC_HEADWISE_INTEGRATED_T_MODES):
            raise RuntimeError("headwise scalar integration mode is disabled")
        if q0.shape != q1.shape or q0.ndim != 2 or q0.shape[1] != self.dim:
            raise ValueError(f"q0 and q1 must share shape (N,{self.dim})")
        delta = q1 - q0
        distance = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
        nodes = self.chord_nodes.to(device=q0.device, dtype=q0.dtype)
        weights = self.chord_weights.to(device=q0.device, dtype=q0.dtype)
        points = canonical_symmetric_chord_points(q0, q1, nodes)
        head_count = (
            4 if self.speed_constraint_consensus_mode == (
                ROUTE_CHART_PARTIAL_SHARED_FOURIER_RISK_ANALYTIC_HANDOVER_HEADWISE_INTEGRATED_T_CONSENSUS_MODE)
            else 3)
        speeds = self.predict_headwise_scalar_speeds(
            points.reshape(-1, self.dim)).reshape(
                q0.shape[0], self.chord_quad_points, head_count)
        return reduce_headwise_scalar_chord_times(
            speeds, distance, weights, self.speed_floor)

    def _predict_speed_with_latent(self, points, features=None):
        """Return the scalar speed and its final speed-supervised latent.

        This is used only when an in-model value branch explicitly requests
        context from chord points that it already has to evaluate.  Keeping
        the scalar output mapping identical to ``predict_speed`` avoids a
        second field or an additional encoder pass.
        """
        if not self.use_speed_head:
            raise RuntimeError("scalar speed head is disabled")
        if self.separate_speed_encoder:
            latent = self.speed_encoder_norm(
                self._encode_speed_points(points))
        else:
            if features is None:
                features = self._encode_scalar_points(points)
            latent = self.encoder_norm(features)
        for block in getattr(self, 'speed_residual_blocks', []):
            latent = block(latent)
        latent = self._apply_partial_shared_speed(latent, points)
        logit = self._speed_logit_from_latent(latent, points)
        if self.lifted_phase:
            if points is None:
                raise RuntimeError(
                    "lifted phase speed requires point coordinates")
            logit = self._apply_lifted_phase_logit(points, latent, logit)
        speed = self.speed_floor + (
            1.0 - self.speed_floor) * torch.sigmoid(logit)
        return speed, latent

    def predict_log_physical_speed(self, points, features=None):
        """Log of the physical speed before its historical affine encoding.

        Computing ``logsigmoid`` from the same scalar head's logit is
        numerically stable in deep invalid regions where subtracting the
        stored 0.1 floor would round to zero.  This is another coordinate of
        the same scalar prediction, not an auxiliary output head.
        """
        return F.logsigmoid(
            self._predict_speed_logit(points, features=features))

    def predict_speed_from_features(self, features):
        """Backward-compatible shared-encoder speed API."""
        if self.separate_speed_encoder:
            raise RuntimeError(
                "separate speed encoder requires point coordinates; "
                "call predict_speed(points) instead")
        return self.predict_speed(None, features=features)

    def predict_clearance(self, points):
        """Predict normalized signed physical clearance from scalar data.

        The returned coordinate is clearance / ``clearance_scale`` clipped by
        tanh to [-1, 1].  It is deliberately not a normal vector and does not
        query FK or collision geometry at inference.
        """
        if not self.use_clearance_head:
            raise RuntimeError("signed-clearance head is disabled")
        latent = self.speed_encoder_norm(
            self._encode_speed_points(points))
        for block in getattr(self, 'speed_residual_blocks', []):
            latent = block(latent)
        latent = self._apply_partial_shared_speed(latent, points)
        for block in getattr(self, 'clearance_residual_blocks', []):
            latent = block(latent)
        if self.clearance_fourier_skip:
            # Preserve the complete shared kinematic Fourier coordinate. The
            # old clipped-speed trunk may intentionally discard distinctions
            # that signed clearance still needs near obstacle boundaries.
            raw_fourier = self.input_mapping(points, self.B_speed)
            latent = torch.cat([latent, raw_fourier], dim=1)
        return torch.tanh(self.clearance_head(latent))

    def clearance_to_speed(self, clearance):
        """Map signed clearance to a positive Eikonal propagation speed.

        Outside the strict boundary this is the historical linear distance
        speed.  Inside, an exponential continuation retains signed-depth
        ordering and a nonzero gradient instead of collapsing every obstacle
        sample to 0.1.  Value and first derivative agree at the boundary.
        """
        boundary = self.clearance_boundary / self.clearance_scale
        outside = clearance.clamp(min=boundary, max=1.0)
        inside = boundary * torch.exp(
            (clearance - boundary) / max(boundary, 1e-8))
        return torch.where(
            clearance < boundary, inside, outside).clamp(
                min=self.clearance_speed_floor, max=1.0)

    def predict_clearance_speed(self, points):
        """Signed-clearance-derived scalar speed for physical feasibility."""
        return self.clearance_to_speed(self.predict_clearance(points))

    def predict_propagation_speed(self, points, features=None):
        """Return the scalar speed used by this architecture's Eikonal PDE.

        Most historical checkpoints propagate with their learned task speed.
        The clearance-Eikonal architecture instead uses the intersection of
        that task-feasible field and the learned signed physical-clearance
        field.  This remains one scalar field inside one checkpoint.
        """
        speed = self.predict_speed(points, features=features)
        if (self.value_arch ==
                'bounded_chord_clearance_eikonal_symmetric'):
            speed = torch.minimum(
                speed, self.predict_clearance_speed(points))
        return speed

    def _predict_propagation_speed_with_latent(
            self, points, features=None):
        """Return the deployed scalar propagation speed and speed latent."""
        speed, latent = self._predict_speed_with_latent(
            points, features=features)
        if (self.value_arch ==
                'bounded_chord_clearance_eikonal_symmetric'):
            speed = torch.minimum(
                speed, self.predict_clearance_speed(points))
        return speed, latent

    def _feature_distance(self, a, b, encoder_norm=None):
        """Same grouped smooth-L2 metric as _sym_op, for arbitrary feature pairs."""
        if encoder_norm is None:
            encoder_norm = self.encoder_norm
        n = a.shape[0]
        y = encoder_norm(torch.cat([a, b], dim=0))
        a_n = y[:n, ...]
        b_n = y[n:, ...]
        x = torch.sqrt((a_n - b_n) ** 2 + 1e-6)
        x = x.view(x.shape[0], -1, 16)
        x = (torch.logsumexp(10 * x, 2) - np.log(16)) / 10
        return 0.1 * torch.sum(x, dim=1, keepdim=True)

    def _pairwise_feature_distance_from_normed(self, a, b):
        """Grouped smooth-L2 distances for all pairs in two normalized sets."""
        x = torch.sqrt((a[:, None, :] - b[None, :, :]) ** 2 + 1e-6)
        x = x.view(x.shape[0], x.shape[1], -1, 16)
        x = (torch.logsumexp(10 * x, 3) - np.log(16)) / 10
        return 0.1 * torch.sum(x, dim=2)

    def _handover_tau(self, f_cup, f_pour, w_cup_2B, w_pour_2B, size):
        """Compose cross-task distance through learned latent handover prototypes."""
        if not (getattr(self, 'use_task_adapters', False) and self.handover_k > 0):
            return None, None
        y = self.encoder_norm(torch.cat([
            f_cup, f_pour, self.handover_cup, self.handover_pour], dim=0))
        f_cup_n = y[:2 * size]
        f_pour_n = y[2 * size:4 * size]
        hcup_n = y[4 * size:4 * size + self.handover_k]
        hpour_n = y[4 * size + self.handover_k:]

        cup_start = f_cup_n[:size]
        cup_goal = f_cup_n[size:]
        pour_start = f_pour_n[:size]
        pour_goal = f_pour_n[size:]

        d_start_cup = self._pairwise_feature_distance_from_normed(cup_start, hcup_n)
        d_goal_pour = self._pairwise_feature_distance_from_normed(pour_goal, hpour_n)
        d_start_pour = self._pairwise_feature_distance_from_normed(pour_start, hpour_n)
        d_goal_cup = self._pairwise_feature_distance_from_normed(cup_goal, hcup_n)

        bridge = (
            F.softplus(self.handover_bridge)
            - float(np.log(2.0))
        ).view(1, self.handover_k)
        tau_cp = -self.handover_temp * torch.logsumexp(
            -(d_start_cup + bridge + d_goal_pour) / self.handover_temp,
            dim=1, keepdim=True)
        tau_pc = -self.handover_temp * torch.logsumexp(
            -(d_start_pour + bridge + d_goal_cup) / self.handover_temp,
            dim=1, keepdim=True)

        cross_cp = (w_cup_2B[:size] * w_pour_2B[size:]).unsqueeze(1)
        cross_pc = (w_pour_2B[:size] * w_cup_2B[size:]).unsqueeze(1)
        cross_strength = torch.clamp(cross_cp + cross_pc, 0.0, 1.0)
        tau_h = (cross_cp * tau_cp + cross_pc * tau_pc) / cross_strength.clamp(min=1e-6)
        return tau_h, cross_strength

    def _endpoint_values(self, values, size, device, dtype):
        if values is None:
            return (torch.zeros(size, device=device, dtype=dtype),
                    torch.zeros(size, device=device, dtype=dtype))
        v = values.to(device=device, dtype=dtype)
        if v.dim() == 2 and v.shape[1] == 1:
            v = v[:, 0]
        if v.shape[0] == 2 * size:
            return v[:size], v[size:]
        if v.shape[0] == size:
            return v, v
        raise ValueError(f"endpoint values shape {tuple(values.shape)} "
                         f"does not match batch size {size}")

    def _endpoint_xy(self, ee_xy, size, device, dtype):
        if ee_xy is None:
            z = torch.zeros((size, 2), device=device, dtype=dtype)
            return z, z
        xy = ee_xy.to(device=device, dtype=dtype)
        if xy.shape[0] == 2 * size and xy.shape[1] == 2:
            return xy[:size], xy[size:]
        if xy.shape[0] == size and xy.shape[1] == 4:
            return xy[:, :2], xy[:, 2:4]
        if xy.shape[0] == size and xy.shape[1] == 2:
            return xy, xy
        raise ValueError(f"ee_xy shape {tuple(ee_xy.shape)} "
                         f"does not match batch size {size}")

    def _stack_endpoint_values(self, values, size, device, dtype):
        if values is None:
            return torch.zeros(2 * size, device=device, dtype=dtype)
        v = values.to(device=device, dtype=dtype)
        if v.dim() == 2 and v.shape[1] == 1:
            v = v[:, 0]
        if v.shape[0] == 2 * size:
            return v
        if v.shape[0] == size:
            return v.repeat(2)
        raise ValueError(f"endpoint values shape {tuple(values.shape)} "
                         f"does not match batch size {size}")

    def _stack_endpoint_xy(self, ee_xy, size, device, dtype):
        if ee_xy is None:
            return torch.zeros((2 * size, 2), device=device, dtype=dtype)
        xy = ee_xy.to(device=device, dtype=dtype)
        if xy.shape[0] == 2 * size and xy.shape[1] == 2:
            return xy
        if xy.shape[0] == size and xy.shape[1] == 4:
            return torch.cat([xy[:, :2], xy[:, 2:4]], dim=0)
        if xy.shape[0] == size and xy.shape[1] == 2:
            return xy.repeat(2, 1)
        raise ValueError(f"ee_xy shape {tuple(ee_xy.shape)} "
                         f"does not match batch size {size}")

    # ---- R63: true (FK-recomputed) routing at interior quadrature points ----
    #
    # _local_finsler_tau samples slowness at alpha in [0,1] along the chord.  The
    # FEATURES at each alpha are honestly recomputed via _per_config_features(q_alpha),
    # but the routing weights and `cond` were LINEARLY INTERPOLATED between the two
    # endpoints.  `cond` carries d_orient_cup / d_orient_pour -- the constraint
    # distances -- so interpolation makes the interior blind to a mid-path violation.
    #
    # Measured 2026-09-07, 4000 real cross training pairs, 9-point rule:
    #   interpolated d_orient_cup  ~ +0.004   (reads compliant)
    #   true         d_orient_cup  ~ -0.040   (violating, near the -margin floor)
    #   mean |error| ~ 0.05 = the ENTIRE margin; d_orient is clipped to +-0.05
    #   52.3% of interior points are reported compliant while the config violates
    #   at alpha=0.5, 90% of cross chords truly violate the cup constraint
    #
    # Attaching a robot model here lets the interior samples carry the truth instead.
    # The recomputed values are DETACHED: at the endpoints `cond` arrives from the
    # dataloader as data, not as a function of q in the graph, so the interior must
    # match that semantics (and it keeps FK out of the backward pass).

    def configure_interior_routing(self, robot_model, q_center, q_range,
                                   margin, cup_tilt_max, pour_tilt_max, tilt_band,
                                   task_axes, corner_zones,
                                   route_temp=0.01, smooth_routing=False,
                                   route_alpha=0.5, route_y_temp=0.3,
                                   offset=0.005,
                                   use_for_inputs=True, use_for_supervision=False):
        """Attach FK at the interior quadrature points.

        TWO INDEPENDENT USES -- keep them separate, they are not the same experiment:

        use_for_inputs (R63)      feed TRUE cond/weights to the quadrature instead of
                                  the interpolated ones.  TESTED AND REJECTED: cost
                                  38-46 points.  The interpolated interior is slack the
                                  network needs to shape tau into a value function.

        use_for_supervision (R65) leave the inputs exactly as they are, and only add a
                                  LOSS term pinning interior slowness to its true label.
                                  Motivation, measured 2026-09-08 at the 101 violating
                                  cup-zone waypoints:
                                      true label at the waypoint      26.03
                                      mean of 32 nearest endpoints    13.86
                                      model predicts                   4.66
                                  The neighbourhood is ambiguous (44.5% slow, 37.4%
                                  fast), so a smooth fit over ENDPOINT supervision alone
                                  averages away ~1.9x of the penalty.  Pinning interior
                                  points targets that gap.  Expected to be PARTIAL: the
                                  remaining ~3x (13.86 -> 4.66) is the model undershooting
                                  its own neighbourhood, which is optimisation pressure,
                                  not missing supervision.
        """
        if self.value_condition != 'task_scene':
            raise ValueError(
                "interior routing is implemented for value_condition='task_scene' "
                f"only (the pair-context variants embed start/goal context that "
                f"cannot be evaluated at an interior point); got "
                f"{self.value_condition!r}")
        if robot_model is None:
            raise ValueError("interior routing requires a robot model for FK")
        dev = next(self.parameters()).device
        object.__setattr__(self, '_interior_robot_model', robot_model)
        object.__setattr__(self, '_interior_q_center',
                           torch.as_tensor(q_center, dtype=torch.float32,
                                           device=dev).reshape(-1))
        object.__setattr__(self, '_interior_q_range',
                           torch.as_tensor(q_range, dtype=torch.float32,
                                           device=dev).reshape(-1))
        self._interior_margin = float(margin)
        self._interior_cup_tilt_max = float(cup_tilt_max)
        self._interior_pour_tilt_max = float(pour_tilt_max)
        self._interior_tilt_band = float(tilt_band)
        self._interior_task_axes = task_axes
        self._interior_corner_zones = corner_zones
        self._interior_route_temp = float(route_temp)
        self._interior_smooth_routing = bool(smooth_routing)
        self._interior_route_alpha = float(route_alpha)
        self._interior_route_y_temp = float(route_y_temp)
        self._interior_offset = float(offset)
        self.interior_routing = bool(use_for_inputs)
        self.interior_supervision = bool(use_for_supervision)

    @torch.no_grad()
    def _interior_target_slowness(self, q_norm):
        """True blended target slowness 1/Yobs at normalized configs q_norm.

        Reproduces exactly what the trainer builds for the ENDPOINT anchor:
        speed_t = clip(d_orient_t, offset, margin)/margin, quartic-transformed, then
        blended by the routing weights.  Any drift from that construction would
        supervise the interior against a different target than the endpoints.
        """
        _, _, d_cup, d_pour, w_cup, w_pour = self._interior_routing_at(q_norm)
        m = self._interior_margin
        off = self._interior_offset
        s_cup = (d_cup.clamp(min=off, max=m) / m)
        s_pour = (d_pour.clamp(min=off, max=m) / m)
        q_cup = s_cup * s_cup * (2.0 - s_cup) * (2.0 - s_cup)
        q_pour = s_pour * s_pour * (2.0 - s_pour) * (2.0 - s_pour)
        yobs = (w_cup * q_cup + w_pour * q_pour).clamp(min=1e-3)
        return 1.0 / yobs

    @torch.no_grad()
    def _interior_routing_at(self, q_norm):
        """(N,dim) normalized configs -> true (ee_y, ee_xy, d_cup, d_pour, w_cup, w_pour)."""
        from ur5e_task_axes import task_axis_arrays, tilt_deg_from_quat
        from models.metric_arm.zone_router import corner_weights_torch

        rm = getattr(self, '_interior_robot_model', None)
        if rm is None:
            raise RuntimeError("interior routing is not configured")
        q_rad = q_norm.detach() * self._interior_q_range + self._interior_q_center
        state = rm.get_state(q_rad)
        ee_pos = state.ee_pose.position
        quat = state.ee_pose.quaternion

        cup_l, cup_w = task_axis_arrays(self._interior_task_axes, 'cup')
        pour_l, pour_w = task_axis_arrays(self._interior_task_axes, 'pour')
        cup_tilt = tilt_deg_from_quat(quat, cup_l, cup_w).reshape(-1)
        pour_tilt = tilt_deg_from_quat(quat, pour_l, pour_w).reshape(-1)

        # SIGNED form -- must match pkl_to_training_data_multi_ur5e._orient_d_from_tilt
        # under --signed_orient, which is what produced d_orient_*.npy for this data.
        m = self._interior_margin
        band = max(self._interior_tilt_band, 1e-6)
        d_cup = (m * ((self._interior_cup_tilt_max - cup_tilt) / band)).clamp(-m, m)
        d_pour = (m * ((self._interior_pour_tilt_max - pour_tilt) / band)).clamp(-m, m)

        w_cup, w_pour = corner_weights_torch(
            ee_pos[:, :2], d_cup, d_pour, self._interior_corner_zones,
            route_temp=self._interior_route_temp,
            smooth_routing=self._interior_smooth_routing,
            route_alpha=self._interior_route_alpha,
            route_y_temp=self._interior_route_y_temp,
            route_margin=m, route_tilt_band=band)
        return (ee_pos[:, 1], ee_pos[:, :2], d_cup, d_pour, w_cup, w_pour)

    def _point_value_condition(self, size, coords, w_cup_2B, w_pour_2B,
                               ee_y, d_orient_cup, d_orient_pour, ee_xy):
        if self.value_condition == 'none':
            return coords.new_zeros((2 * size, 0))

        dtype = coords.dtype
        device = coords.device
        ee_y_2B = self._stack_endpoint_values(ee_y, size, device, dtype)
        d_cup_2B = self._stack_endpoint_values(d_orient_cup, size, device, dtype)
        d_pour_2B = self._stack_endpoint_values(d_orient_pour, size, device, dtype)
        cond = [
            w_cup_2B.unsqueeze(1),
            w_pour_2B.unsqueeze(1),
            d_cup_2B.unsqueeze(1),
            d_pour_2B.unsqueeze(1),
            ee_y_2B.unsqueeze(1),
        ]

        if self.value_condition in (
                'task_scene', 'task_scene_dist',
                'task_scene_pair', 'task_scene_pair_dist'):
            cond.append(self._stack_endpoint_xy(ee_xy, size, device, dtype))

        if self.value_condition in ('task_scene_pair', 'task_scene_pair_dist'):
            ee_y_s, ee_y_g = self._endpoint_values(ee_y, size, device, dtype)
            d_cup_s, d_cup_g = self._endpoint_values(
                d_orient_cup, size, device, dtype)
            d_pour_s, d_pour_g = self._endpoint_values(
                d_orient_pour, size, device, dtype)
            xy_s, xy_g = self._endpoint_xy(ee_xy, size, device, dtype)
            pair_ctx = torch.cat([
                w_cup_2B[:size].unsqueeze(1),
                w_pour_2B[:size].unsqueeze(1),
                w_cup_2B[size:].unsqueeze(1),
                w_pour_2B[size:].unsqueeze(1),
                d_cup_s.unsqueeze(1),
                d_pour_s.unsqueeze(1),
                d_cup_g.unsqueeze(1),
                d_pour_g.unsqueeze(1),
                ee_y_s.unsqueeze(1),
                ee_y_g.unsqueeze(1),
                xy_s,
                xy_g,
            ], dim=1)
            cond.append(torch.cat([pair_ctx, pair_ctx], dim=0))

        if self.value_condition in ('task_scene_dist', 'task_scene_pair_dist'):
            dist = torch.norm(
                coords[:, self.dim:] - coords[:, :self.dim],
                p=2, dim=1, keepdim=True)
            cond.append(dist.repeat(2, 1))

        return torch.cat(cond, dim=1)

    def _asym_value_condition(self, size, coords, w_cup_2B, w_pour_2B,
                              ee_y, d_orient_cup, d_orient_pour, ee_xy):
        if self.value_condition == 'none':
            return coords.new_zeros((size, 0))

        dtype = coords.dtype
        device = coords.device
        ee_y_s, ee_y_g = self._endpoint_values(ee_y, size, device, dtype)
        d_cup_s, d_cup_g = self._endpoint_values(d_orient_cup, size, device, dtype)
        d_pour_s, d_pour_g = self._endpoint_values(d_orient_pour, size, device, dtype)
        cond = [
            w_cup_2B[:size].unsqueeze(1),
            w_pour_2B[:size].unsqueeze(1),
            w_cup_2B[size:].unsqueeze(1),
            w_pour_2B[size:].unsqueeze(1),
            d_cup_s.unsqueeze(1),
            d_pour_s.unsqueeze(1),
            d_cup_g.unsqueeze(1),
            d_pour_g.unsqueeze(1),
            ee_y_s.unsqueeze(1),
            ee_y_g.unsqueeze(1),
        ]

        if self.value_condition in (
                'task_scene', 'task_scene_dist',
                'task_scene_pair', 'task_scene_pair_dist'):
            xy_s, xy_g = self._endpoint_xy(ee_xy, size, device, dtype)
            cond.extend([xy_s, xy_g])
        if self.value_condition in ('task_scene_dist', 'task_scene_pair_dist'):
            dist = torch.norm(
                coords[:, self.dim:] - coords[:, :self.dim],
                p=2, dim=1, keepdim=True)
            cond.append(dist)

        return torch.cat(cond, dim=1)

    def _asym_goal_tau(self, coords, x_fourier, w_cup_2B, w_pour_2B,
                       ee_y, d_orient_cup, d_orient_pour, ee_xy):
        size = coords.shape[0]
        z0 = x_fourier[:size]
        z1 = x_fourier[size:]
        cond = self._asym_value_condition(
            size, coords, w_cup_2B, w_pour_2B,
            ee_y, d_orient_cup, d_orient_pour, ee_xy)
        pair = torch.cat([z0, z1, z1 - z0, z0 * z1, cond], dim=1)
        scale = F.softplus(self.asym_value_net(pair)) + 1e-4
        dist = torch.norm(
            coords[:, self.dim:] - coords[:, :self.dim],
            p=2, dim=1, keepdim=True)
        return dist * scale

    def _local_finsler_quad_rule(self, device, dtype):
        if self.finsler_quad_points == 1:
            alphas = (0.0,)
            weights = (1.0,)
        elif self.finsler_quad_points == 2:
            alphas = (0.0, 1.0)
            weights = (0.5, 0.5)
        elif self.finsler_quad_points == 3:
            alphas = (0.0, 0.5, 1.0)
            weights = (1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0)
        elif self.finsler_quad_points == 5:
            alphas = (0.0, 0.25, 0.5, 0.75, 1.0)
            weights = (1.0 / 12.0, 4.0 / 12.0, 2.0 / 12.0,
                       4.0 / 12.0, 1.0 / 12.0)
        else:
            # R62: 9-node composite Simpson (h=1/8), the consistent refinement of
            # the 5-node rule above (which is composite Simpson at h=1/4).  Motive:
            # cross paths fail the per-waypoint tilt gate by 1-2 deg at waypoints
            # BETWEEN the nodes tau actually samples, so the integral does not see
            # the violation it is being scored on.  Endpoints stay at alpha=0 and 1
            # so the direct slowness supervision still anchors them.
            alphas = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
            weights = (1.0 / 24.0, 4.0 / 24.0, 2.0 / 24.0, 4.0 / 24.0,
                       2.0 / 24.0, 4.0 / 24.0, 2.0 / 24.0, 4.0 / 24.0,
                       1.0 / 24.0)
        return alphas, torch.tensor(weights, device=device, dtype=dtype)

    def _apply_cross_slowness_residual(self, base, f_cup, f_pour, w_cup, w_pour,
                                       direction, cond, cross_strength):
        f_blended = w_cup.unsqueeze(1) * f_cup + w_pour.unsqueeze(1) * f_pour
        f_local = self.encoder_norm(f_blended)
        local_in = torch.cat([f_local, direction, cond], dim=1)
        out = base
        if (getattr(self, 'use_cross_slowness_residual', False)
                and cross_strength is not None):
            raw = self.cross_slowness_residual(local_in)
            delta = self.cross_residual_scale * torch.tanh(raw)
            strength = cross_strength.to(
                device=base.device, dtype=base.dtype).reshape(-1, 1)
            out = out * torch.exp(strength * delta)
        if getattr(self, 'use_local_slowness_residual', False):
            raw = self.local_slowness_residual(local_in)
            delta = self.local_residual_scale * torch.tanh(raw)
            if getattr(self, 'local_residual_gate', 'all') == 'cross_strength':
                if cross_strength is None:
                    gate = torch.zeros_like(base)
                else:
                    gate = cross_strength.to(
                        device=base.device, dtype=base.dtype).reshape(-1, 1)
                delta = gate * delta
            out = out * torch.exp(delta)
        if out is not base:
            # R62: _bounded_slowness floors `base` at 1 (=> speed <= 1, the NTFields
            # normalisation), but the residuals above multiply by exp(+-scale) and can
            # push it back under 1. Measured 2026-09-01 on wall25_gate15: min slowness
            # 0.816, i.e. speed 1.22. Re-impose the floor AFTER the residuals with a
            # smooth (softplus) max so the gradient survives -- a hard clamp here would
            # recreate the dead-gradient trap the bound was rewritten to avoid.
            out = 1.0 + F.softplus((out - 1.0) * _SLOWNESS_FLOOR_BETA) / _SLOWNESS_FLOOR_BETA
        return out

    def _cross_strength_from_weights(self, w_cup_2B, w_pour_2B, size):
        cross_cp = w_cup_2B[:size] * w_pour_2B[size:]
        cross_pc = w_pour_2B[:size] * w_cup_2B[size:]
        return torch.clamp(cross_cp + cross_pc, 0.0, 1.0)

    def _local_finsler_slowness(self, f_cup, f_pour, w_cup, w_pour,
                                direction, cond, cross_strength=None):
        if self.finsler_task_heads:
            f_cup_local = self.encoder_norm(f_cup)
            f_pour_local = self.encoder_norm(f_pour)
            cup_in = torch.cat([f_cup_local, direction, cond], dim=1)
            pour_in = torch.cat([f_pour_local, direction, cond], dim=1)
            # Clamp: labels imply slowness (1/speed) in roughly [1, 1/offset*margin]
            # = [1, ~10] for the default margin/offset. Softplus is unbounded
            # above, so when these heads are trainable (not frozen) an
            # unconstrained blow-up here destabilizes the eikonal loss --
            # observed as a rollback-thrashing collapse during unfrozen/
            # partial-unfrozen training. 20.0 gives ~2x headroom over the
            # label range without capping normal learning.
            s_cup = _bounded_slowness(self.local_slowness_cup(cup_in))
            s_pour = _bounded_slowness(self.local_slowness_pour(pour_in))
            base = w_cup.unsqueeze(1) * s_cup + w_pour.unsqueeze(1) * s_pour
            return self._apply_cross_slowness_residual(
                base, f_cup, f_pour, w_cup, w_pour,
                direction, cond, cross_strength)

        f_blended = w_cup.unsqueeze(1) * f_cup + w_pour.unsqueeze(1) * f_pour
        f_local = self.encoder_norm(f_blended)
        local_in = torch.cat([f_local, direction, cond], dim=1)
        base = _bounded_slowness(self.local_slowness_net(local_in))
        return self._apply_cross_slowness_residual(
            base, f_cup, f_pour, w_cup, w_pour,
            direction, cond, cross_strength)

    def _local_finsler_tau(self, coords, f_cup, f_pour, w_cup_2B, w_pour_2B,
                           ee_y, d_orient_cup, d_orient_pour, ee_xy):
        size = coords.shape[0]
        delta = coords[:, self.dim:] - coords[:, :self.dim]
        dist = torch.norm(delta, p=2, dim=1, keepdim=True)
        direction = delta / dist.clamp(min=1e-8)

        cond = self._point_value_condition(
            size, coords, w_cup_2B, w_pour_2B,
            ee_y, d_orient_cup, d_orient_pour, ee_xy)

        if (not self.finsler_task_heads) and self.finsler_quad_points == 2:
            direction_2B = torch.cat([direction, direction], dim=0)
            cross_strength_pair = self._cross_strength_from_weights(
                w_cup_2B, w_pour_2B, size)
            slowness = self._local_finsler_slowness(
                f_cup=f_cup,
                f_pour=f_pour,
                w_cup=w_cup_2B,
                w_pour=w_pour_2B,
                direction=direction_2B,
                cond=cond,
                cross_strength=cross_strength_pair.repeat(2))
            slowness = 0.5 * (slowness[:size] + slowness[size:])
            return dist * slowness

        alphas, quad_weights = self._local_finsler_quad_rule(coords.device, coords.dtype)
        q0 = coords[:, :self.dim]

        f_cup_s, f_cup_g = f_cup[:size], f_cup[size:]
        f_pour_s, f_pour_g = f_pour[:size], f_pour[size:]
        w_cup_s, w_cup_g = w_cup_2B[:size], w_cup_2B[size:]
        w_pour_s, w_pour_g = w_pour_2B[:size], w_pour_2B[size:]
        cond_s, cond_g = cond[:size], cond[size:]
        cross_strength_pair = self._cross_strength_from_weights(
            w_cup_2B, w_pour_2B, size)

        f_cup_samples = []
        f_pour_samples = []
        w_cup_samples = []
        w_pour_samples = []
        cond_samples = []
        cross_strength_samples = []
        interior_on = bool(getattr(self, 'interior_routing', False))
        for alpha in alphas:
            q_alpha = None
            if alpha == 0.0:
                f_cup_a, f_pour_a = f_cup_s, f_pour_s
            elif alpha == 1.0:
                f_cup_a, f_pour_a = f_cup_g, f_pour_g
            else:
                q_alpha = q0 + alpha * delta
                f_cup_a, f_pour_a = self._per_config_features(q_alpha)

            beta = 1.0 - alpha
            f_cup_samples.append(f_cup_a)
            f_pour_samples.append(f_pour_a)
            if interior_on and q_alpha is not None:
                # R63: recompute the routing weights and cond at the TRUE interior
                # config instead of interpolating the endpoints.  See
                # configure_interior_routing for the measured motivation.
                (ey_a, exy_a, dc_a, dp_a,
                 wc_a, wp_a) = self._interior_routing_at(q_alpha)
                w_cup_samples.append(wc_a)
                w_pour_samples.append(wp_a)
                # Passing (size,) values makes _point_value_condition repeat them
                # across both halves, so the first `size` rows are exactly the
                # per-point cond for this interior sample, in the canonical layout.
                cond_samples.append(self._point_value_condition(
                    size, coords, wc_a.repeat(2), wp_a.repeat(2),
                    ey_a, dc_a, dp_a, exy_a)[:size])
            else:
                w_cup_samples.append(beta * w_cup_s + alpha * w_cup_g)
                w_pour_samples.append(beta * w_pour_s + alpha * w_pour_g)
                cond_samples.append(beta * cond_s + alpha * cond_g)
            cross_strength_samples.append(cross_strength_pair)

        k = len(alphas)
        slowness = self._local_finsler_slowness(
            f_cup=torch.cat(f_cup_samples, dim=0),
            f_pour=torch.cat(f_pour_samples, dim=0),
            w_cup=torch.cat(w_cup_samples, dim=0),
            w_pour=torch.cat(w_pour_samples, dim=0),
            direction=direction.repeat(k, 1),
            cond=torch.cat(cond_samples, dim=0),
            cross_strength=torch.cat(cross_strength_samples, dim=0))
        slowness = slowness.view(k, size, 1)
        # R62: cache the slowness at the chord ENDPOINTS (alpha=0 is the start
        # config, alpha=1 the goal) so the loss can regress them directly against
        # the label 1/speed.  Without this the head's only signal is d(tau)/dq in
        # the Eikonal residual, which drives it to its rail in one epoch.
        if alphas[0] == 0.0 and alphas[-1] == 1.0:
            self._last_endpoint_slowness = (slowness[0, :, 0], slowness[-1, :, 0])
        else:
            self._last_endpoint_slowness = None
        # R65: cache the INTERIOR slowness plus the configs it was evaluated at, so the
        # loss can pin them to their true labels.  Inputs are untouched -- this only
        # exposes what the field already predicts there.  Cached only when interior
        # supervision is on, so the baseline path allocates nothing.
        if getattr(self, 'interior_supervision', False):
            interior = [j for j, a in enumerate(alphas) if 0.0 < a < 1.0]
            if interior:
                self._last_interior_slowness = torch.cat(
                    [slowness[j, :, 0] for j in interior], dim=0)
                self._last_interior_q = torch.cat(
                    [(q0 + float(alphas[j]) * delta).detach() for j in interior], dim=0)
            else:
                self._last_interior_slowness = None
                self._last_interior_q = None
        slowness = (quad_weights.view(k, 1, 1) * slowness).sum(dim=0)
        return dist * slowness

    def _per_config_features(self, x):
        """Encode a batch of single configs (N, dim) -> (f_cup, f_pour) each (N, h_size).

        Mirrors the encoder path of out() but operates on raw single configs
        (no start/goal split), so it can be reused by orient_cost.
        """
        if self.separate_fourier:
            x_fourier_cup = self.input_mapping(x, self.B_cup)
            x_fourier_pour = self.input_mapping(x, self.B_pour)
        else:
            x_fourier_cup = self.input_mapping(x)
            x_fourier_pour = x_fourier_cup

        # The endpoint-Siamese scalar formulation has one active encoder.
        # Legacy cup/pour modules remain in old checkpoint state dicts for
        # compatibility, but must not be evaluated: their outputs are unused
        # and doing so triples encoder compute while falsely suggesting a
        # multi-model inference path.
        if self.endpoint_task_siamese:
            f_cup = None
            f_pour = None
        elif getattr(self, 'use_task_adapters', False):
            w = self.pe_gate[0].weight
            b = self.pe_gate[0].bias
            w = self.lip_norm(w)
            u = torch.sin(x_fourier_cup @ w.T + b)

            w = self.pe_gate[1].weight
            b = self.pe_gate[1].bias
            w = self.lip_norm(w)
            v = torch.sin(x_fourier_cup @ w.T + b)

            f_base = self._run_encoder(
                x_fourier_cup, u, v, self.shared_encoder, self.shared_gate)
            f_cup = f_base + self.cup_adapter(f_base)
            f_pour = f_base + self.pour_adapter(f_base)
        elif self.separate_pe_gate:
            w = self.lip_norm(self.cup_pe_gate[0].weight); b = self.cup_pe_gate[0].bias
            u_cup = torch.sin(x_fourier_cup @ w.T + b)
            w = self.lip_norm(self.cup_pe_gate[1].weight); b = self.cup_pe_gate[1].bias
            v_cup = torch.sin(x_fourier_cup @ w.T + b)
            w = self.lip_norm(self.pour_pe_gate[0].weight); b = self.pour_pe_gate[0].bias
            u_pour = torch.sin(x_fourier_pour @ w.T + b)
            w = self.lip_norm(self.pour_pe_gate[1].weight); b = self.pour_pe_gate[1].bias
            v_pour = torch.sin(x_fourier_pour @ w.T + b)
            f_cup = self._run_encoder(x_fourier_cup, u_cup, v_cup, self.cup_encoder, self.cup_gate)
            f_pour = self._run_encoder(x_fourier_pour, u_pour, v_pour, self.pour_encoder, self.pour_gate)
        else:
            w = self.lip_norm(self.pe_gate[0].weight); b = self.pe_gate[0].bias
            u = torch.sin(x_fourier_cup @ w.T + b)
            w = self.lip_norm(self.pe_gate[1].weight); b = self.pe_gate[1].bias
            v = torch.sin(x_fourier_cup @ w.T + b)
            f_cup = self._run_encoder(x_fourier_cup, u, v, self.cup_encoder, self.cup_gate)
            f_pour = self._run_encoder(x_fourier_pour, u, v, self.pour_encoder, self.pour_gate)

        if self.use_projector and not self.endpoint_task_siamese:
            f_cup = self.projector(f_cup)
            f_pour = self.projector(f_pour)
        return f_cup, f_pour

    def orient_cost(self, coords, ee_y=None, d_orient_cup=None, d_orient_pour=None,
                    ee_xy=None,
                    w_cup_override=None, w_pour_override=None, detach_input=False):
        """R58F decoupled orientation cost Phi_orient for start and goal configs.

        Returns (phi_start, phi_goal), each (B, 1) and >= 0. When the orient head
        is disabled, returns zeros (so callers are unaffected).

        coords: (B, 2*dim). First dim cols = q_start, last dim cols = q_goal.

        R58F fix: the head consumes a Fourier encoding of the raw joint config
        instead of the position-optimized blended encoder feature. With
        use_task_orient_heads, cup/pour scalar heads are blended by routing
        weights, avoiding one scalar having to fit two different task cones.
        """
        size = coords.shape[0]
        if not getattr(self, 'use_orient_head', False):
            z = coords.new_zeros((size, 1))
            return z, z.clone()

        if detach_input:
            coords = coords.clone().detach().requires_grad_(True)
        x0 = coords[:, :self.dim]
        x1 = coords[:, self.dim:]
        x = torch.vstack((x0, x1))  # (2B, dim)

        # Dedicated high-frequency basis for the orientation head (see __init__).
        x_fourier = self.input_mapping(x, self.B_orient)  # (2B, 2*orient_freqs)
        if getattr(self, 'use_task_orient_heads', False):
            phi_cup = self.orient_head_cup(x_fourier)
            phi_pour = self.orient_head_pour(x_fourier)
            if w_cup_override is not None and w_pour_override is not None:
                if w_cup_override.shape[0] == size:
                    w_cup = w_cup_override.repeat(2)
                    w_pour = w_pour_override.repeat(2)
                else:
                    w_cup = w_cup_override
                    w_pour = w_pour_override
                w_cup = w_cup.to(device=coords.device, dtype=phi_cup.dtype).unsqueeze(1)
                w_pour = w_pour.to(device=coords.device, dtype=phi_pour.dtype).unsqueeze(1)
            else:
                w_cup = torch.full_like(phi_cup, 0.5)
                w_pour = torch.full_like(phi_pour, 0.5)
            phi = w_cup * phi_cup + w_pour * phi_pour
        else:
            phi = self.orient_head(x_fourier)  # (2B, 1), >= 0
        phi_start = phi[:size]
        phi_goal = phi[size:]
        return phi_start, phi_goal

    def safety_cost(self, coords, detach_input=False):
        """Learned collision/self-collision potential for start and goal configs."""
        size = coords.shape[0]
        if not getattr(self, 'use_safety_head', False):
            z = coords.new_zeros((size, 1))
            return z, z.clone()

        if detach_input:
            coords = coords.clone().detach().requires_grad_(True)
        x0 = coords[:, :self.dim]
        x1 = coords[:, self.dim:]
        x = torch.vstack((x0, x1))
        x_fourier = self.input_mapping(x, self.B_safety)
        phi = self.safety_head(x_fourier)
        return phi[:size], phi[size:]

    def pair_safety_phi(self, x_fourier, pair_type_context=None,
                        shared_phi=None):
        """Return the pair-aware safety potential.

        ``replace`` preserves the original independent-head experiment.
        ``residual`` factorizes the field into a frozen/common geometry term
        plus a task-conditioned correction.  Residual heads use the existing
        Softplus architecture and are zero-centred at Softplus(0)=log(2), so a
        zero-initialized final layer is exactly identical to the shared model.
        """
        if shared_phi is None:
            shared_phi = self.safety_head(x_fourier)
        if (not getattr(self, 'use_pair_safety_heads', False)
                or pair_type_context is None
                or not hasattr(self, 'safety_head_cup')
                or not hasattr(self, 'safety_head_pour')
                or not hasattr(self, 'safety_head_cross')):
            return shared_phi

        pt = pair_type_context.to(
            device=shared_phi.device, dtype=torch.long).reshape(-1)
        if pt.shape[0] != shared_phi.shape[0]:
            return shared_phi

        phi_cup = self.safety_head_cup(x_fourier)
        phi_pour = self.safety_head_pour(x_fourier)
        phi_cross = self.safety_head_cross(x_fourier)
        mode = getattr(self, 'pair_safety_mode', 'replace')
        if mode in ('residual', 'constraint_residual'):
            residual = self.pair_safety_raw_residual(
                x_fourier, pair_type_context,
                pair_outputs=(phi_cup, phi_pour, phi_cross))
            if mode == 'constraint_residual':
                # Constraint factorization: a task head may add a barrier but
                # can never cancel the common collision/reach potential.
                residual = torch.relu(residual)
            return torch.clamp_min(
                shared_phi + self.pair_safety_residual_scale * residual, 0.0)

        selected = shared_phi
        selected = torch.where((pt == 0).unsqueeze(1), phi_cup, selected)
        selected = torch.where((pt == 1).unsqueeze(1), phi_pour, selected)
        selected = torch.where((pt == 2).unsqueeze(1), phi_cross, selected)
        return selected

    def pair_safety_raw_residual(self, x_fourier, pair_type_context,
                                 pair_outputs=None):
        """Zero-centred task correction before the inference-time barrier."""
        zero = math.log(2.0)
        selected = x_fourier.new_full((x_fourier.shape[0], 1), zero)
        if pair_type_context is None:
            return selected - zero
        pt = pair_type_context.to(
            device=x_fourier.device, dtype=torch.long).reshape(-1)
        if pt.shape[0] != x_fourier.shape[0]:
            return selected - zero
        if pair_outputs is None:
            pair_outputs = (
                self.safety_head_cup(x_fourier),
                self.safety_head_pour(x_fourier),
                self.safety_head_cross(x_fourier),
            )
        phi_cup, phi_pour, phi_cross = pair_outputs
        selected = torch.where((pt == 0).unsqueeze(1), phi_cup, selected)
        selected = torch.where((pt == 1).unsqueeze(1), phi_pour, selected)
        selected = torch.where((pt == 2).unsqueeze(1), phi_cross, selected)
        return selected - zero

    def _safety_tau_lambda_for_pairs(self, phi_start, w_cup_start, w_pour_start,
                                     w_cup_goal=None, w_pour_goal=None,
                                     pair_type_context=None):
        if (getattr(self, 'safety_tau_query_lambdas', False)
                and pair_type_context is not None):
            pt = pair_type_context.to(
                device=phi_start.device, dtype=torch.long).reshape(-1)
            if pt.shape[0] == phi_start.shape[0]:
                lam = torch.full(
                    (pt.shape[0],), self.safety_tau_lambda,
                    device=phi_start.device, dtype=phi_start.dtype)
                if self.safety_tau_query_lambda_cup is not None:
                    lam = torch.where(
                        pt == 0, torch.full_like(lam, self.safety_tau_query_lambda_cup), lam)
                if self.safety_tau_query_lambda_pour is not None:
                    lam = torch.where(
                        pt == 1, torch.full_like(lam, self.safety_tau_query_lambda_pour), lam)
                if self.safety_tau_query_lambda_cross is not None:
                    lam = torch.where(
                        pt == 2, torch.full_like(lam, self.safety_tau_query_lambda_cross), lam)
                return lam.unsqueeze(1)

        if not getattr(self, 'safety_tau_pair_lambdas', False):
            return self.safety_tau_lambda
        if (w_cup_start is None or w_pour_start is None or
                w_cup_goal is None or w_pour_goal is None):
            return self.safety_tau_lambda

        dtype = phi_start.dtype
        device = phi_start.device
        w_cup_s = w_cup_start.to(device=device, dtype=dtype).reshape(-1)
        w_pour_s = w_pour_start.to(device=device, dtype=dtype).reshape(-1)
        w_cup_g = w_cup_goal.to(device=device, dtype=dtype).reshape(-1)
        w_pour_g = w_pour_goal.to(device=device, dtype=dtype).reshape(-1)
        if not (w_cup_s.shape == w_pour_s.shape == w_cup_g.shape == w_pour_g.shape):
            return self.safety_tau_lambda

        same_cup = w_cup_s * w_cup_g
        same_pour = w_pour_s * w_pour_g
        cross = w_cup_s * w_pour_g + w_pour_s * w_cup_g
        denom = (same_cup + same_pour + cross).clamp_min(1e-6)

        lam_cup = self.safety_tau_lambda if self.safety_tau_lambda_cup is None else self.safety_tau_lambda_cup
        lam_pour = self.safety_tau_lambda if self.safety_tau_lambda_pour is None else self.safety_tau_lambda_pour
        lam_cross = self.safety_tau_lambda if self.safety_tau_lambda_cross is None else self.safety_tau_lambda_cross
        lam = (lam_cup * same_cup + lam_pour * same_pour + lam_cross * cross) / denom
        return lam.unsqueeze(1)

    def _orient_tau_lambda_for_pairs(self, phi_start, pair_type_context=None):
        has_query_lam = (
            self.orient_tau_query_lambda_cup is not None
            or self.orient_tau_query_lambda_pour is not None
            or self.orient_tau_query_lambda_cross is not None)
        if has_query_lam and pair_type_context is not None:
            pt = pair_type_context.to(
                device=phi_start.device, dtype=torch.long).reshape(-1)
            if pt.shape[0] == phi_start.shape[0]:
                lam = torch.full(
                    (pt.shape[0],), self.orient_tau_lambda,
                    device=phi_start.device, dtype=phi_start.dtype)
                if self.orient_tau_query_lambda_cup is not None:
                    lam = torch.where(
                        pt == 0,
                        torch.full_like(lam, self.orient_tau_query_lambda_cup),
                        lam)
                if self.orient_tau_query_lambda_pour is not None:
                    lam = torch.where(
                        pt == 1,
                        torch.full_like(lam, self.orient_tau_query_lambda_pour),
                        lam)
                if self.orient_tau_query_lambda_cross is not None:
                    lam = torch.where(
                        pt == 2,
                        torch.full_like(lam, self.orient_tau_query_lambda_cross),
                        lam)
                return lam.unsqueeze(1)
        return self.orient_tau_lambda

    def _add_start_potentials_to_tau(self, tau, coords,
                                     w_cup_start=None, w_pour_start=None,
                                     w_cup_goal=None, w_pour_goal=None,
                                     pair_type_context=None):
        """Bake enabled config-only potentials into tau at the candidate start."""
        if getattr(self, 'add_orient_to_tau', False) and getattr(self, 'use_orient_head', False):
            x_start = coords[:, :self.dim]
            x_start_fourier = self.input_mapping(x_start, self.B_orient)
            if getattr(self, 'use_task_orient_heads', False):
                phi_cup = self.orient_head_cup(x_start_fourier)
                phi_pour = self.orient_head_pour(x_start_fourier)
                if w_cup_start is None or w_pour_start is None:
                    w_cup = torch.full_like(phi_cup, 0.5)
                    w_pour = torch.full_like(phi_pour, 0.5)
                else:
                    w_cup = w_cup_start.to(
                        device=coords.device, dtype=phi_cup.dtype).unsqueeze(1)
                    w_pour = w_pour_start.to(
                        device=coords.device, dtype=phi_pour.dtype).unsqueeze(1)
                phi_start = w_cup * phi_cup + w_pour * phi_pour
            else:
                phi_start = self.orient_head(x_start_fourier)
            lam = self._orient_tau_lambda_for_pairs(
                phi_start, pair_type_context)
            tau = tau + lam * phi_start
        if getattr(self, 'add_safety_to_tau', False) and getattr(self, 'use_safety_head', False):
            x_start = coords[:, :self.dim]
            x_start_fourier = self.input_mapping(x_start, self.B_safety)
            phi_start = self.pair_safety_phi(
                x_start_fourier, pair_type_context=pair_type_context)
            lam = self._safety_tau_lambda_for_pairs(
                phi_start, w_cup_start, w_pour_start, w_cup_goal, w_pour_goal,
                pair_type_context)
            tau = tau + lam * phi_start
        return tau

    def out(self, coords, ee_y=None, d_orient_cup=None, d_orient_pour=None,
            ee_xy=None,
            w_cup_override=None, w_pour_override=None, detach_input=True,
            fk_features=None, pair_type_context=None,
            handover_phase=None):
        """Forward pass with feature-level blending (FB-NTFields style).

        Each encoder head outputs latent features. Features are blended
        using window weights, then a SINGLE sym_op computes tau from
        the blended features. This ensures smooth gradient flow across
        zone boundaries.

        Args:
            detach_input: if True (default, unchanged behavior), the input is
                cloned/detached into a fresh leaf with requires_grad=True, so
                gradients w.r.t. tau are taken against that internal leaf. If
                False, `coords` is used as-is (the caller is responsible for
                setting requires_grad) so that tau stays differentiable w.r.t.
                the caller's coords. The path-loss self-rollout needs this so
                autograd.grad(tau, external_q) connects through the graph.

        Returns: tau, f_blended, coords, f_cup, f_pour
        """
        if detach_input:
            coords = coords.clone().detach().requires_grad_(True)
        # when detach_input=False: use coords as-is. Downstream only slices
        # coords into x0/x1 (no leaf requirement), and the returned 3rd value
        # stays connected to the caller's tensor.
        size = coords.shape[0]
        x0 = coords[:, :self.dim]
        x1 = coords[:, self.dim:]

        x = torch.vstack((x0, x1))  # (2*B, dim)

        # Fourier encoding: shared or per-head
        if self.separate_fourier:
            x_fourier_cup = self.input_mapping(x, self.B_cup)
            x_fourier_pour = self.input_mapping(x, self.B_pour)
        else:
            x_fourier_cup = self.input_mapping(x)
            x_fourier_pour = x_fourier_cup

        fk_2B = self._stack_fk_features(
            fk_features, size, coords.device, x_fourier_cup.dtype)
        if fk_2B is not None:
            fk_emb = self.fk_feature_proj(fk_2B)
            x_fourier_cup = x_fourier_cup + fk_emb
            x_fourier_pour = x_fourier_pour + fk_emb if self.separate_fourier else x_fourier_cup

        if getattr(self, 'use_task_adapters', False):
            w = self.pe_gate[0].weight
            b = self.pe_gate[0].bias
            w = self.lip_norm(w)
            u = torch.sin(x_fourier_cup @ w.T + b)

            w = self.pe_gate[1].weight
            b = self.pe_gate[1].bias
            w = self.lip_norm(w)
            v = torch.sin(x_fourier_cup @ w.T + b)

            f_base = self._run_encoder(
                x_fourier_cup, u, v, self.shared_encoder, self.shared_gate)
            f_cup = f_base + self.cup_adapter(f_base)
            f_pour = f_base + self.pour_adapter(f_base)
        elif self.separate_pe_gate:
            w = self.cup_pe_gate[0].weight
            b = self.cup_pe_gate[0].bias
            w = self.lip_norm(w)
            u_cup = torch.sin(x_fourier_cup @ w.T + b)
            w = self.cup_pe_gate[1].weight
            b = self.cup_pe_gate[1].bias
            w = self.lip_norm(w)
            v_cup = torch.sin(x_fourier_cup @ w.T + b)

            w = self.pour_pe_gate[0].weight
            b = self.pour_pe_gate[0].bias
            w = self.lip_norm(w)
            u_pour = torch.sin(x_fourier_pour @ w.T + b)
            w = self.pour_pe_gate[1].weight
            b = self.pour_pe_gate[1].bias
            w = self.lip_norm(w)
            v_pour = torch.sin(x_fourier_pour @ w.T + b)

            f_cup = self._run_encoder(x_fourier_cup, u_cup, v_cup, self.cup_encoder, self.cup_gate)
            f_pour = self._run_encoder(x_fourier_pour, u_pour, v_pour, self.pour_encoder, self.pour_gate)
        else:
            w = self.pe_gate[0].weight
            b = self.pe_gate[0].bias
            w = self.lip_norm(w)
            u = torch.sin(x_fourier_cup @ w.T + b)

            w = self.pe_gate[1].weight
            b = self.pe_gate[1].bias
            w = self.lip_norm(w)
            v = torch.sin(x_fourier_cup @ w.T + b)

            f_cup = self._run_encoder(x_fourier_cup, u, v, self.cup_encoder, self.cup_gate)
            f_pour = self._run_encoder(x_fourier_pour, u, v, self.pour_encoder, self.pour_gate)

        # --- Optional shared projector (before blending) ---
        if self.use_projector:
            f_cup = self.projector(f_cup)
            f_pour = self.projector(f_pour)

        # --- Determine per-point routing weights ---
        # Each of the 2B points (B starts + B goals) gets its OWN window weight.
        # In cross-zone queries, start may be in cupping zone while goal is in pouring zone.
        if w_cup_override is not None and w_pour_override is not None:
            # Override already has shape (B,) — expand to (2B,) for start+goal
            if w_cup_override.shape[0] == size:
                # Caller provided per-start weights only; for feature blending
                # we need per-point weights. Use same for goal (legacy compat).
                w_cup_2B = w_cup_override.repeat(2)
                w_pour_2B = w_pour_override.repeat(2)
            else:
                w_cup_2B = w_cup_override
                w_pour_2B = w_pour_override
        elif ee_y is not None:
            if ee_y.shape[0] == 2 * size:
                # ee_y for all 2B points — compute weights per point
                d_cup_all = d_orient_cup if d_orient_cup.shape[0] == 2 * size else d_orient_cup.repeat(2)
                d_pour_all = d_orient_pour if d_orient_pour.shape[0] == 2 * size else d_orient_pour.repeat(2)
                ee_xy_all = None
                if ee_xy is not None:
                    ee_xy_all = ee_xy if ee_xy.shape[0] == 2 * size else ee_xy.repeat(2, 1)
                w_cup_2B, w_pour_2B = self.compute_window_weights(
                    ee_y, d_cup_all, d_pour_all, ee_xy=ee_xy_all)
            elif ee_y.shape[0] == size:
                # Only start ee_y — replicate for goal (backward compat)
                w_cup, w_pour = self.compute_window_weights(
                    ee_y, d_orient_cup, d_orient_pour, ee_xy=ee_xy)
                w_cup_2B = w_cup.repeat(2)
                w_pour_2B = w_pour.repeat(2)
            else:
                raise ValueError(f"ee_y shape {ee_y.shape} doesn't match batch size {size}")
        else:
            w_cup_2B = torch.full((2 * size,), 0.5, device=coords.device)
            w_pour_2B = torch.full((2 * size,), 0.5, device=coords.device)

        # Pair labels are sufficient to pin the endpoint task for the two
        # within-task cases. Cross-task endpoints retain their spatially
        # inferred mixture so the handover can change continuously with q.
        if self.endpoint_task_siamese and pair_type_context is not None:
            pair = pair_type_context.to(
                device=coords.device, dtype=torch.long).reshape(-1)
            if pair.shape[0] == size:
                w_cup_2B = w_cup_2B.clone()
                w_pour_2B = w_pour_2B.clone()
                cup = pair == 0
                pour = pair == 1
                w_cup_2B[:size] = torch.where(
                    cup, torch.ones_like(w_cup_2B[:size]),
                    torch.where(pour, torch.zeros_like(w_cup_2B[:size]),
                                w_cup_2B[:size]))
                w_pour_2B[:size] = 1.0 - w_cup_2B[:size]
                w_cup_2B[size:] = torch.where(
                    cup, torch.ones_like(w_cup_2B[size:]),
                    torch.where(pour, torch.zeros_like(w_cup_2B[size:]),
                                w_cup_2B[size:]))
                w_pour_2B[size:] = 1.0 - w_cup_2B[size:]

        # --- Detach routing weights to prevent boundary-seeking gradients ---
        if hasattr(self, 'detach_routing_weights') and self.detach_routing_weights:
            w_cup_2B = w_cup_2B.detach()
            w_pour_2B = w_pour_2B.detach()

        if self.endpoint_task_siamese:
            if self.separate_fourier and not self.endpoint_dual_value_heads:
                raise ValueError(
                    "endpoint_task_siamese requires one shared Fourier basis")
            if self.endpoint_dual_value_heads:
                f_siamese = self._blend_endpoint_value_heads(
                    f_cup, f_pour, w_cup_2B, w_pour_2B)
            else:
                f_siamese = self._run_endpoint_task_siamese(
                    x_fourier_cup, w_cup_2B, w_pour_2B)
            if self.value_arch in (
                    'endpoint_eikonal_residual_symmetric',
                    'local_endpoint_eikonal_residual_symmetric'):
                tau = self._endpoint_eikonal_residual_tau(
                    coords, f_siamese, size)
            elif self.value_arch == 'factored_source_eikonal_symmetric':
                tau = self._factored_source_eikonal_tau(
                    coords, f_siamese, size)
            elif self.value_arch in (
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
                    'bounded_chord_local_endpoint_eikonal_symmetric',
                    'bounded_chord_speed_barrier_symmetric',
                    'bounded_chord_local_speed_barrier_symmetric',
                    'bounded_chord_global_excess_symmetric',
                    'bounded_chord_smooth_excess_symmetric',
                    'bounded_chord_separate_smooth_excess_symmetric',
                    'bounded_chord_separate_global_excess_symmetric',
                    'bounded_chord_neural_viscosity_symmetric',
                    'bounded_chord_signed_clearance_symmetric',
                    'bounded_chord_clearance_eikonal_symmetric',
                    'bounded_chord_clearance_state_symmetric',
                    'bounded_chord_profile_symmetric'):
                cross_strength = self._cross_strength_from_weights(
                    w_cup_2B, w_pour_2B, size)
                directed_context = None
                need_directed_context = bool(
                    getattr(self, 'viscosity_directed_context', False)
                    or getattr(self, 'use_causal_handover_head', False))
                if need_directed_context:
                    pour_start = w_pour_2B[:size]
                    pour_goal = w_pour_2B[size:]
                    directed_context = torch.stack([
                        pour_start,
                        pour_goal,
                        pour_goal - pour_start,
                    ], dim=1)
                    need_phase_context = bool(getattr(
                        self, 'viscosity_handover_phase_context', False))
                    need_phase_context = (
                        need_phase_context or bool(getattr(
                            self, 'causal_handover_phase_context', False)))
                    if need_phase_context:
                        phase_start, phase_goal = self._endpoint_values(
                            handover_phase, size, coords.device, coords.dtype)
                        phase_context = torch.stack([
                            phase_start,
                            phase_goal,
                            phase_goal - phase_start,
                        ], dim=1)
                        directed_context = torch.cat([
                            directed_context, phase_context], dim=1)
                tau = self._bounded_chord_tau(
                    coords, f_siamese, size,
                    cross_strength=cross_strength,
                    directed_context=directed_context,
                    handover_phase=handover_phase)
            elif self.value_arch == 'fermat_waypoint_symmetric':
                tau = self._fermat_waypoint_tau(coords, f_siamese, size)
            elif self.value_arch == 'fermat_curve_symmetric':
                tau = self._fermat_curve_tau(coords, f_siamese, size)
            elif self.value_arch == 'fermat_curve_equivariant':
                tau = self._fermat_curve_equivariant_tau(
                    coords, f_siamese, size)
            elif self.value_arch in (
                    'global_residual', 'global_residual_symmetric'):
                tau = self._global_residual_tau(coords, f_siamese, size)
            else:
                tau = self._zero_boundary_sym_op(f_siamese, size)
            if self.speed_constraint_consensus_mode not in (
                    ANALYTIC_HEADWISE_INTEGRATED_T_MODES):
                tau = self._add_start_potentials_to_tau(
                    tau, coords, w_cup_2B[:size], w_pour_2B[:size],
                    w_cup_2B[size:], w_pour_2B[size:], pair_type_context)
            return tau, f_siamese, coords, f_siamese, f_siamese

        if getattr(self, 'value_arch', 'metric') == 'asym_goal':
            f_blended = w_cup_2B.unsqueeze(1) * f_cup + w_pour_2B.unsqueeze(1) * f_pour
            tau = self._asym_goal_tau(
                coords, x_fourier_cup, w_cup_2B, w_pour_2B,
                ee_y, d_orient_cup, d_orient_pour, ee_xy)

            tau = self._add_start_potentials_to_tau(
                tau, coords, w_cup_2B[:size], w_pour_2B[:size],
                w_cup_2B[size:], w_pour_2B[size:], pair_type_context)

            return tau, f_blended, coords, f_cup, f_pour

        if getattr(self, 'value_arch', 'metric') == 'local_finsler':
            f_blended = w_cup_2B.unsqueeze(1) * f_cup + w_pour_2B.unsqueeze(1) * f_pour
            tau = self._local_finsler_tau(
                coords, f_cup, f_pour, w_cup_2B, w_pour_2B,
                ee_y, d_orient_cup, d_orient_pour, ee_xy)

            tau = self._add_start_potentials_to_tau(
                tau, coords, w_cup_2B[:size], w_pour_2B[:size],
                w_cup_2B[size:], w_pour_2B[size:], pair_type_context)

            return tau, f_blended, coords, f_cup, f_pour

        # --- Blending mode ---
        if hasattr(self, 'blend_mode') and self.blend_mode == 'tau_blend':
            # Tau-level blending: each head computes τ independently, then blend
            tau_cup = self._sym_op(f_cup, size)    # (B, 1)
            tau_pour = self._sym_op(f_pour, size)  # (B, 1)
            # Per-pair weights: average of start and goal w_cup
            w_cup_pair = 0.5 * (w_cup_2B[:size] + w_cup_2B[size:])  # (B,)
            w_pour_pair = 1.0 - w_cup_pair
            tau = w_cup_pair.unsqueeze(1) * tau_cup + w_pour_pair.unsqueeze(1) * tau_pour
            f_blended = w_cup_2B.unsqueeze(1) * f_cup + w_pour_2B.unsqueeze(1) * f_pour
        elif hasattr(self, 'blend_mode') and self.blend_mode == 'tau_blend_perpoint':
            # Tau-level blending with per-point (not per-pair) weights
            tau_cup = self._sym_op(f_cup, size)
            tau_pour = self._sym_op(f_pour, size)
            # Use start weight for tau blending (start determines the gradient direction)
            w_cup_s = w_cup_2B[:size]
            w_pour_s = w_pour_2B[:size]
            tau = w_cup_s.unsqueeze(1) * tau_cup + w_pour_s.unsqueeze(1) * tau_pour
            f_blended = w_cup_2B.unsqueeze(1) * f_cup + w_pour_2B.unsqueeze(1) * f_pour
        elif hasattr(self, 'blend_mode') and self.blend_mode == 'tau_softmax':
            # Softmax over tau_cup, tau_pour (smooth min, not linear blend)
            tau_cup = self._sym_op(f_cup, size)
            tau_pour = self._sym_op(f_pour, size)
            # LogSumExp approximation to soft-min: -log(exp(-a*t1) + exp(-a*t2))/a
            alpha_sm = 10.0
            tau_stack = torch.cat([tau_cup, tau_pour], dim=1)  # (B, 2)
            tau = (-torch.logsumexp(-alpha_sm * tau_stack, dim=1, keepdim=True)) / alpha_sm
            f_blended = w_cup_2B.unsqueeze(1) * f_cup + w_pour_2B.unsqueeze(1) * f_pour
        elif hasattr(self, 'blend_mode') and self.blend_mode == 'cross_attention':
            # Cross-attention: each head attends to the other
            q_cup = self.cross_attn_q(f_cup)   # (2B, h)
            k_pour = self.cross_attn_k(f_pour)  # (2B, h)
            v_pour = self.cross_attn_v(f_pour)  # (2B, h)
            # Per-point attention score (dot product, not sequence attention)
            attn = (q_cup * k_pour).sum(-1, keepdim=True) * self.cross_attn_scale  # (2B, 1)
            attn = torch.sigmoid(attn)  # soft gate [0, 1]
            # Blend: cup features enriched by pouring info
            f_blended = (1 - attn) * f_cup + attn * v_pour
            # Weight by routing: in cup zone, mostly f_cup; in pour zone, mostly attended
            f_blended = w_cup_2B.unsqueeze(1) * f_cup + w_pour_2B.unsqueeze(1) * f_blended

            tau = self._sym_op(f_blended, size)

        elif hasattr(self, 'blend_mode') and self.blend_mode == 'film':
            # FiLM: learned per-dimension modulation based on ee_y
            if ee_y is not None:
                ee_y_input = ee_y.unsqueeze(1) if ee_y.dim() == 1 else ee_y[:, :1]
                film_out = self.film_net(ee_y_input)  # (2B, 2*h)
                gamma = film_out[:, :f_cup.shape[1]]  # (2B, h)
                beta = film_out[:, f_cup.shape[1]:]   # (2B, h)
                # gamma controls per-dim blend, beta is offset
                gamma_sig = torch.sigmoid(gamma)  # [0,1] per dimension
                f_blended = gamma_sig * f_cup + (1 - gamma_sig) * f_pour + beta
            else:
                f_blended = 0.5 * f_cup + 0.5 * f_pour

            tau = self._sym_op(f_blended, size)

        elif hasattr(self, 'blend_mode') and self.blend_mode == 'slerp':
            # Match the G1 R40A/R55 slerp architecture: interpolate task
            # features on the sphere before the shared sym_op.
            f_cup_norm = F.normalize(f_cup, dim=-1)
            f_pour_norm = F.normalize(f_pour, dim=-1)
            cos_omega = (f_cup_norm * f_pour_norm).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
            omega = torch.acos(cos_omega)
            sin_omega = torch.sin(omega).clamp(min=1e-6)
            t = w_pour_2B.unsqueeze(1)

            f_blended = (torch.sin((1 - t) * omega) / sin_omega) * f_cup + \
                        (torch.sin(t * omega) / sin_omega) * f_pour
            mag = 0.5 * (torch.norm(f_cup, dim=-1, keepdim=True) + torch.norm(f_pour, dim=-1, keepdim=True))
            f_blended = f_blended * mag

            tau = self._sym_op(f_blended, size)

        else:
            # Default: Feature-level blending (FB-NTFields style)
            f_blended = w_cup_2B.unsqueeze(1) * f_cup + w_pour_2B.unsqueeze(1) * f_pour

            tau = self._sym_op(f_blended, size)

            if self.use_dual_sym:
                tau_cup = self._sym_op(f_cup, size)
                tau_pour = self._sym_op(f_pour, size)
                tau = torch.max(tau, torch.max(tau_cup, tau_pour))

        tau_h, cross_strength = self._handover_tau(
            f_cup, f_pour, w_cup_2B, w_pour_2B, size)
        if tau_h is not None:
            tau = (1.0 - cross_strength) * tau + cross_strength * tau_h

        tau = self._add_start_potentials_to_tau(
            tau, coords, w_cup_2B[:size], w_pour_2B[:size],
            w_cup_2B[size:], w_pour_2B[size:], pair_type_context)

        return tau, f_blended, coords, f_cup, f_pour

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        output, _, coords, _, _ = self.out(coords)
        return output, coords
