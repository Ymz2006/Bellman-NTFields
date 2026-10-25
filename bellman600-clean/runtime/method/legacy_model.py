"""Current NoBias52k: shared encoder, task adapters, late travel-time fusion.

Inputs are normalized configuration pairs. FK position supplies smoothstep
weights; speed labels are training targets only. No fusion MLP is evaluated.
"""

from norm_readouts import SCREEN_NORMS, screen_distance
from pathlib import Path
import numpy as np
import torch
from checkpoint_compat import CheckpointBackbone
from fk import PositionFK

ROOT = Path(__file__).resolve().parents[1]
# Historical checkpoint metadata, not a selectable architecture.
ARCHITECTURE = "two_encoder_overlap_mlp_cotraining_v1"


class CurrentField(CheckpointBackbone):
    def __init__(self, device, dim, B, **kwargs):
        self.joint_weight = kwargs.pop("joint_weight", 1.0)
        self.latent_scale = kwargs.pop("latent_scale", 0.1)
        self.latent_norm = kwargs.pop("latent_norm", "smooth_l2")
        if self.joint_weight not in [0.0, 1.0] or self.latent_scale not in [0.1, 0.5, 1.0, 2.0] or self.latent_norm not in ["smooth_l2", "l1", "linf", "logsumexp01", "lp15n", "lp3n", "lp4n", "mix_l1_25", "mix_l1_50", "mix_l4_50"]:
            raise ValueError("Unsupported latent readout configuration")
        self.fusion_stage = kwargs.pop("fusion_stage", "scalar_late")
        if self.fusion_stage not in ["scalar_late", "delta_before_norm"]:
            raise ValueError("Unknown fusion stage")
        self.prediction_routing = kwargs.pop("prediction_routing", "smoothstep")
        if self.prediction_routing not in ["smoothstep", "overlap_half"]:
            raise ValueError("Unknown prediction routing")
        self.routing_shape = kwargs.pop("routing_shape", "linear")
        self.shared_trunk = kwargs.pop("shared_trunk", False)
        if self.routing_shape not in ["linear", "smoothstep"]:
            raise ValueError("Unknown routing shape")
        self.readout = kwargs.pop("readout", "metric")
        if (
            self.readout != "anchored_metric"
            or self.routing_shape != "smoothstep"
            or not self.shared_trunk
        ):
            raise ValueError("Expected the current shared smoothstep SameHead model")
        super().__init__(device, dim, B, **kwargs)
        h = kwargs.get("h_size", 256)
        # Serialized legacy tensors only: retain RNG/Adam/checkpoint compatibility.
        # fields() never calls this module.
        self.fusion_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * h + 2, 128), torch.nn.SiLU(), torch.nn.Linear(128, h)
        ).to(device)
        torch.nn.init.zeros_(self.fusion_mlp[-1].weight)
        torch.nn.init.zeros_(self.fusion_mlp[-1].bias)
        if self.shared_trunk:
            if self.separate_fourier or self.separate_pe_gate or self.use_projector:
                raise ValueError("Shared trunk requires shared PE and no projector")
            self.use_task_adapters = True
            self.shared_encoder = self.cup_encoder
            self.shared_gate = self.cup_gate
            del self.cup_encoder, self.cup_gate, self.pour_encoder, self.pour_gate
            self.cup_adapter = torch.nn.Sequential(
                torch.nn.Linear(h, 64), torch.nn.SiLU(), torch.nn.Linear(64, h)
            ).to(device)
            self.pour_adapter = torch.nn.Sequential(
                torch.nn.Linear(h, 64), torch.nn.SiLU(), torch.nn.Linear(64, h)
            ).to(device)
            for adapter in [self.cup_adapter, self.pour_adapter]:
                torch.nn.init.zeros_(adapter[-1].weight)
                torch.nn.init.zeros_(adapter[-1].bias)
        self.position_fk = PositionFK().to(device)
        norm = np.load(ROOT / "configs/joint_normalization.npz")
        self.register_buffer(
            "fk_center",
            torch.tensor(norm["q_center"], dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "fk_range",
            torch.tensor(norm["q_range"], dtype=torch.float32, device=device),
        )
        self.register_buffer("_cotraining_version", torch.tensor(1, device=device))
        self.to(device)
        self.cup_adapter[-1].register_parameter("bias", None)
        self.pour_adapter[-1].register_parameter("bias", None)

    def compute_window_weights(
        self, ee_y, d_orient_cup=None, d_orient_pour=None, ee_xy=None
    ):
        y = ee_xy[:, 1] if ee_xy is not None else ee_y
        if y is None:
            raise ValueError("Position is required for routing")
        cup = ((y + 0.1) / 0.2).clamp(0, 1)
        if self.routing_shape == "smoothstep":
            cup = cup.square() * (3 - 2 * cup)
        weights = torch.stack([cup, 1 - cup], dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return weights[:, 0], weights[:, 1]

    def prediction_weights(self, y, smooth_cup):
        if self.prediction_routing == "smoothstep":
            return smooth_cup
        # Only the predicted field changes; fields() still returns smooth labels.
        overlap = (y > -0.1) & (y < 0.1)
        return torch.where(overlap, torch.full_like(smooth_cup, 0.5), smooth_cup)

    def scalar_time(self, f, coords):
        n = len(coords)
        delta = coords[:, :6] - coords[:, 6:]
        distance = (delta.square().sum(dim=1, keepdim=True) + 1e-12).sqrt() - 1e-6
        latent = (
            (f[:n] - f[n:]).square().sum(dim=1, keepdim=True) + 1e-12
        ).sqrt() - 1e-6
        return distance + 0.1 * latent

    def fields(self, coords, positions=None, all_heads=True):
        n = len(coords)
        x = torch.cat([coords[:, :6], coords[:, 6:]], 0)
        if positions is None:
            positions = self.position_fk(x * self.fk_range + self.fk_center)
        wc, wp = self.compute_window_weights(positions[:, 1])
        c, p = self._per_config_features(x)
        tc = self.scalar_time(c, coords)
        tp = self.scalar_time(p, coords)
        prediction_wc = self.prediction_weights(positions[:, 1], wc)
        w = 0.5 * (prediction_wc[:n] + prediction_wc[n:])
        if self.fusion_stage == "scalar_late":
            tf = w[:, None] * tc + (1 - w[:, None]) * tp
        else:
            delta_c = c[:n] - c[n:]
            delta_p = p[:n] - p[n:]
            delta = w[:, None] * delta_c + (1 - w[:, None]) * delta_p
            dq = coords[:, :6] - coords[:, 6:]
            distance = (dq.square().sum(dim=1, keepdim=True) + 1e-12).sqrt() - 1e-6
            if self.latent_norm == "smooth_l2":
                latent = (delta.square().sum(dim=1, keepdim=True) + 1e-12).sqrt() - 1e-6
            elif self.latent_norm == "l1":
                latent = delta.abs().sum(dim=1, keepdim=True)
            elif self.latent_norm in SCREEN_NORMS:
                latent = screen_distance(delta, self.latent_norm)
            elif self.latent_norm == "logsumexp01":
                # Fixed tau=0.1, normalized to give zero on the diagonal.
                values = delta.abs() / 0.1
                latent = 0.1 * (torch.logsumexp(values, dim=1, keepdim=True) - torch.logsumexp(torch.zeros_like(values), dim=1, keepdim=True))
            else:
                latent = delta.abs().amax(dim=1, keepdim=True)
            tf = self.latent_scale * latent
            if self.joint_weight == 1.0:
                tf = distance + tf
        # Compatibility feature return only, not used by the time readout; no fusion MLP invocation.
        z = wc[:, None] * c + wp[:, None] * p
        return (
            [tc if all_heads else None, tp if all_heads else None, tf],
            z,
            c,
            p,
            wc,
            wp,
        )

    def out(
        self,
        coords,
        ee_y=None,
        d_orient_cup=None,
        d_orient_pour=None,
        ee_xy=None,
        w_cup_override=None,
        w_pour_override=None,
        detach_input=True,
        fk_features=None,
        pair_type_context=None,
        handover_phase=None,
    ):
        if any(
            x is not None
            for x in [
                w_cup_override,
                w_pour_override,
                fk_features,
                pair_type_context,
                handover_phase,
            ]
        ):
            raise ValueError("No speed, task ID or routing override inputs")
        if detach_input:
            coords = coords.clone().detach().requires_grad_(True)
        # Training derives position differentiably. MPPI supplies its existing FK position.
        positions = None
        if ee_xy is not None:
            positions = torch.cat([ee_xy, torch.zeros_like(ee_xy[:, :1])], dim=1)
        elif ee_y is not None:
            positions = torch.stack([ee_y * 0, ee_y, ee_y * 0], dim=1)
        if positions is None and not self.training:
            raise ValueError("Inference requires position-only FK routing")
        v, z, c, p, _, _ = self.fields(coords, positions, all_heads=False)
        return v[2], z, coords, c, p


def compute_position_routing_info(
    q_rad_flat,
    robot_model,
    margin=None,
    cup_tilt_max=None,
    cup_tilt_soft=None,
    pour_tilt_max=None,
    pour_tilt_soft=None,
    pour_rzy_safe=None,
    pour_rzy_soft=None,
    return_xy=False,
    return_fk_features=False,
    task_axes=None,
):
    """Legacy planner ABI, reading only FK position; zeros are unused ABI slots."""
    if return_fk_features:
        raise ValueError("Position-only architecture cannot request FK features")
    position = robot_model.get_state(q_rad_flat).ee_pose.position
    y = position[:, 1]
    unused = torch.zeros_like(y)
    if return_xy:
        return y, position[:, :2], unused, unused
    return y, unused, unused


def load_checkpoint(checkpoint, device="cuda:0"):
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    if ck.get("repair_variant") != "samehead_no_bias":
        raise ValueError("Expected NoBias52k-compatible weights")
    net = CurrentField(device, 6, ck["B_state_dict"], **ck["kwargs"])
    net.load_state_dict(ck["model_state_dict"], strict=True)
    net.eval()
    # Preserve the historical loader's RNG consumption before seeded MPPI.
    q = torch.randn(4, 12, device=device) * 0.1
    with torch.no_grad():
        xyz = net.position_fk(
            torch.cat([q[:, :6], q[:, 6:]]) * net.fk_range + net.fk_center
        )
        cost = net.out(q, ee_xy=xyz[:, :2], detach_input=False)[0]
        if not torch.isfinite(cost).all():
            raise ValueError("Nonfinite loaded cost")
    return net
