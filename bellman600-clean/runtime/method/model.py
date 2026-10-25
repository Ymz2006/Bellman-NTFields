"""Compact NoBias52k: unchanged active field, without dormant parameters.

The legacy constructor preserves frozen initialization RNG consumption. Pruning
happens before optimization/inference; no dormant tensors remain registered.
"""
import torch
from legacy_model import (
    CurrentField as LegacyField,
    ARCHITECTURE,
    compute_position_routing_info,
)
from parameter_cleanup import prune, removed_names


class CurrentField(LegacyField):
    def __init__(self, device, dim, B, **kwargs):
        super().__init__(device, dim, B, **kwargs)
        prune(self)


def compact_state(ck):
    state = ck["model_state_dict"]
    version = ck.get("parameter_cleanup_version")
    if version is None:
        if not removed_names() <= state.keys():
            raise ValueError("Incomplete legacy checkpoint")
        return {k: v for k, v in state.items() if k not in removed_names()}
    if version != 1 or removed_names() & state.keys():
        raise ValueError("Invalid compact checkpoint format")
    return state


def load_checkpoint(checkpoint, device="cuda:0"):
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    if ck.get("repair_variant") != "samehead_no_bias":
        raise ValueError("Expected NoBias52k-compatible weights")
    net = CurrentField(device, 6, ck["B_state_dict"], **ck["kwargs"])
    net.load_state_dict(compact_state(ck), strict=True)
    if ck.get("parameter_cleanup_version") == 1:
        if list(dict(net.named_parameters())) != ck["parameter_order"]:
            raise ValueError("Compact parameter order mismatch")
    net.eval()
    # Same historical loader probe and random-number consumption.
    q = torch.randn(4, 12, device=device) * 0.1
    with torch.no_grad():
        xyz = net.position_fk(
            torch.cat([q[:, :6], q[:, 6:]]) * net.fk_range + net.fk_center
        )
        cost = net.out(q, ee_xy=xyz[:, :2], detach_input=False)[0]
        if not torch.isfinite(cost).all():
            raise ValueError("Nonfinite loaded cost")
    return net
