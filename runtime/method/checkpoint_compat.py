"""Frozen constructor compatibility for the selected historical checkpoint.

This retains upstream parameter registration and RNG consumption so the saved
weights, Adam indices and planning seeds remain reproducible. Research changes
belong in model.py/loss.py. The legacy fusion_mlp is serialized but never called.
"""

from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/ntrl"))
from models.metric_arm import model_network_multi as upstream

ARCHITECTURE = "position_routing_metric_multispeed_v2"
POSITION_SCALE_M = 0.1 * 0.3 / 0.95


class CheckpointBackbone(upstream.MultiNN):
    def __init__(self, device, dim, B, **kwargs):
        required = dict(
            value_arch="metric",
            value_condition="none",
            use_orient_head=False,
            add_orient_to_tau=False,
            use_task_orient_heads=False,
            use_safety_head=False,
            add_safety_to_tau=False,
            use_pair_safety_heads=False,
            use_speed_head=False,
            use_cross_slowness_residual=False,
            use_local_slowness_residual=False,
            handover_k=0,
            use_dual_sym=False,
            fk_feature_dim=0,
            blend_mode="feature",
            endpoint_task_siamese=False,
        )
        for key, value in required.items():
            if key in kwargs and kwargs[key] != value:
                raise ValueError(f"{ARCHITECTURE} requires {key}={value!r}")
            kwargs[key] = value
        super().__init__(device, dim, B, **kwargs)
        self.register_buffer(
            "_position_routing_version", torch.tensor(2, device=device)
        )
        self.register_buffer(
            "_position_scale_m", torch.tensor(POSITION_SCALE_M, device=device)
        )
        self.smooth_routing = True
        self.route_alpha = 1.0
        self.route_y_temp = POSITION_SCALE_M / self.overlap
        self.interior_routing = False
        self.interior_supervision = False
