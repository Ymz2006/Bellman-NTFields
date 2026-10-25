"""Shared routing helpers for multi-task cup/pour zones.

The legacy multi-task setup separates tasks with a single EE-Y boundary.  The
corner-table setup uses two rectangular task regions plus a small transition
box, so routing must look at EE-X and EE-Y together.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ZONE_CUPPING = 0
ZONE_POURING = 1


def _as_box(region: dict[str, Any] | None) -> tuple[float, float, float, float] | None:
    if not region:
        return None
    x0, x1 = region["x"]
    y0, y1 = region["y"]
    return (float(x0), float(x1), float(y0), float(y1))


def normalize_corner_zones(zones: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a compact JSON/checkpoint-safe corner-table zone spec."""
    if not zones or zones.get("type") != "corner_table":
        return None
    cup = _as_box(zones.get("cup_region"))
    pour = _as_box(zones.get("pour_region"))
    if cup is None or pour is None:
        raise ValueError("corner_table zones require cup_region and pour_region")
    overlap = _as_box(zones.get("overlap_region"))
    return {
        "type": "corner_table",
        "cup_region": cup,
        "pour_region": pour,
        "overlap_region": overlap,
    }


def load_zone_config(scene_path: str | Path | None, requested_mode: str = "auto") -> tuple[str, dict[str, Any] | None]:
    """Load zone mode and corner regions from a scene yaml.

    requested_mode:
      - "auto": use scene zones.type when present, else legacy "y"
      - "y": force legacy Y split
      - "corner_table": require scene corner-table regions
    """
    if requested_mode not in {"auto", "y", "corner_table"}:
        raise ValueError(f"unknown zone mode: {requested_mode}")
    if requested_mode == "y":
        return "y", None
    if scene_path is None:
        if requested_mode == "corner_table":
            raise ValueError("--zone_mode corner_table requires a scene yaml")
        return "y", None

    with open(scene_path) as f:
        data = yaml.safe_load(f) or {}
    zones_raw = data.get("zones", {}) or {}
    if zones_raw.get("type") == "corner_table":
        return "corner_table", normalize_corner_zones(zones_raw)
    if requested_mode == "corner_table":
        raise ValueError(f"{scene_path} does not define zones.type: corner_table")
    return "y", None


def load_dataset_zone_config(data_dir: str | Path) -> tuple[str, dict[str, Any] | None]:
    """Read zone routing metadata saved by data generation."""
    data_dir = Path(data_dir)
    routing_path = data_dir / "routing_config.json"
    if routing_path.exists():
        cfg = json.loads(routing_path.read_text())
        mode = cfg.get("zone_mode", "y")
        return mode, cfg.get("corner_zones")

    cfg_path = data_dir / "pipeline_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        mode = cfg.get("zone_mode", "y")
        return mode, cfg.get("corner_zones")

    return "y", None


def _box_contains_np(x: np.ndarray, y: np.ndarray, box) -> np.ndarray:
    x0, x1, y0, y1 = box
    return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)


def _box_distance_np(x: np.ndarray, y: np.ndarray, box) -> np.ndarray:
    x0, x1, y0, y1 = box
    dx = np.maximum(np.maximum(x0 - x, x - x1), 0.0)
    dy = np.maximum(np.maximum(y0 - y, y - y1), 0.0)
    return np.sqrt(dx * dx + dy * dy)


def assign_corner_masks_np(ee_xy: np.ndarray, corner_zones: dict[str, Any],
                           include_overlap: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cup/pour/overlap masks for EE XY positions.

    Points outside all configured boxes are left out of both task masks. Runtime
    routing has a nearest-region fallback; data generation should explicitly
    sample only configured regions.
    """
    xy = np.asarray(ee_xy, dtype=np.float32)
    x = xy[:, 0]
    y = xy[:, 1]
    cup_box = corner_zones["cup_region"]
    pour_box = corner_zones["pour_region"]
    overlap_box = corner_zones.get("overlap_region")

    cup_only = _box_contains_np(x, y, cup_box)
    pour_only = _box_contains_np(x, y, pour_box)
    overlap_mask = cup_only & pour_only
    if overlap_box is not None:
        overlap_mask = overlap_mask | _box_contains_np(x, y, overlap_box)

    if include_overlap:
        cup_mask = cup_only | overlap_mask
        pour_mask = pour_only | overlap_mask
    else:
        cup_mask = cup_only & ~overlap_mask
        pour_mask = pour_only & ~overlap_mask
    return cup_mask, pour_mask, overlap_mask


def assign_corner_zone_ids_np(ee_xy: np.ndarray, corner_zones: dict[str, Any],
                              source_id: np.ndarray | None = None) -> np.ndarray:
    """Assign a 0/1 zone id for labels, using source_id to break overlap ties."""
    xy = np.asarray(ee_xy, dtype=np.float32)
    x = xy[:, 0]
    y = xy[:, 1]
    cup_mask, pour_mask, overlap_mask = assign_corner_masks_np(xy, corner_zones, include_overlap=True)
    cup_box = corner_zones["cup_region"]
    pour_box = corner_zones["pour_region"]
    d_cup = _box_distance_np(x, y, cup_box)
    d_pour = _box_distance_np(x, y, pour_box)

    zone = np.where(d_cup <= d_pour, ZONE_CUPPING, ZONE_POURING).astype(np.int64)
    zone[cup_mask & ~pour_mask] = ZONE_CUPPING
    zone[pour_mask & ~cup_mask] = ZONE_POURING
    if source_id is not None:
        src = np.asarray(source_id)
        zone[overlap_mask & (src == 0)] = ZONE_CUPPING
        zone[overlap_mask & (src == 1)] = ZONE_POURING
    return zone


def corner_transition_mask_np(ee_xy: np.ndarray, corner_zones: dict[str, Any], band: float) -> np.ndarray:
    """Configs near the transition/inside-corner box for local cross pairing."""
    xy = np.asarray(ee_xy, dtype=np.float32)
    x = xy[:, 0]
    y = xy[:, 1]
    overlap_box = corner_zones.get("overlap_region")
    if overlap_box is not None:
        return _box_distance_np(x, y, overlap_box) <= float(band)
    d_cup = _box_distance_np(x, y, corner_zones["cup_region"])
    d_pour = _box_distance_np(x, y, corner_zones["pour_region"])
    return np.minimum(d_cup, d_pour) <= float(band)


def _box_contains_torch(x, y, box):
    x0, x1, y0, y1 = [float(v) for v in box]
    return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)


def _box_distance_torch(x, y, box):
    import torch

    x0, x1, y0, y1 = [float(v) for v in box]
    dx = torch.clamp(torch.maximum(x.new_tensor(x0) - x, x - x.new_tensor(x1)), min=0.0)
    dy = torch.clamp(torch.maximum(y.new_tensor(y0) - y, y - y.new_tensor(y1)), min=0.0)
    return torch.sqrt(dx * dx + dy * dy + 1e-12)


def corner_masks_torch(ee_xy, corner_zones: dict[str, Any], valid_mask=None,
                       assign_outside: bool = True):
    """Torch cup/pour/overlap masks for runtime checks."""
    x = ee_xy[..., 0]
    y = ee_xy[..., 1]
    cup_box = corner_zones["cup_region"]
    pour_box = corner_zones["pour_region"]
    overlap_box = corner_zones.get("overlap_region")

    cup_box_mask = _box_contains_torch(x, y, cup_box)
    pour_box_mask = _box_contains_torch(x, y, pour_box)
    overlap_mask = cup_box_mask & pour_box_mask
    if overlap_box is not None:
        overlap_mask = overlap_mask | _box_contains_torch(x, y, overlap_box)

    cup_mask = cup_box_mask | overlap_mask
    pour_mask = pour_box_mask | overlap_mask

    if assign_outside:
        outside = ~(cup_mask | pour_mask)
        if outside.any():
            d_cup = _box_distance_torch(x, y, cup_box)
            d_pour = _box_distance_torch(x, y, pour_box)
            nearest_cup = d_cup <= d_pour
            cup_mask = cup_mask | (outside & nearest_cup)
            pour_mask = pour_mask | (outside & ~nearest_cup)

    if valid_mask is not None:
        cup_mask = cup_mask & valid_mask
        pour_mask = pour_mask & valid_mask
        overlap_mask = overlap_mask & valid_mask
    return cup_mask, pour_mask, overlap_mask


def combine_overlap_tilt_pass_torch(
        cup_ok, pour_ok, cup_pass, pour_pass, overlap_mask, logic: str):
    """Apply the requested OR/BOTH rule only inside the overlap region."""
    import torch

    if logic not in ("or", "both"):
        raise ValueError(f"overlap logic must be 'or' or 'both', got {logic!r}")
    overlap_pass = (
        cup_pass & pour_pass if logic == "both" else cup_pass | pour_pass
    )
    return torch.where(overlap_mask, overlap_pass, cup_ok & pour_ok)



def resolve_route_scales(corner_zones, route_temp=0.01, route_alpha=0.5,
                         route_y_temp=0.3, y_scale=None, tilt_scale=None,
                         margin=0.05, tilt_band=15.0):
    """Return (y_scale in metres, tilt_scale in degrees) for smooth routing.

    Direct spelling wins.  Otherwise derive the equivalent scales from the legacy
    (alpha, route_y_temp, route_temp) triple so old checkpoints keep their exact
    behaviour:

        A = alpha / (half_band * route_y_temp)          -> y_scale    = 1 / A
        B = (1 - alpha) / route_temp * margin/tilt_band  -> tilt_scale = 1 / B
    """
    if y_scale is not None and tilt_scale is not None:
        return float(y_scale), float(tilt_scale)
    overlap_box = corner_zones.get("overlap_region") if corner_zones else None
    if overlap_box is not None:
        half_band = 0.5 * (float(overlap_box[3]) - float(overlap_box[2]))
    else:
        half_band = 0.5 * (float(corner_zones["cup_region"][2])
                           - float(corner_zones["pour_region"][3]))
    half_band = max(abs(half_band), 1e-6)
    a = float(route_alpha)
    A = a / (half_band * max(float(route_y_temp), 1e-9))
    B = (1.0 - a) / max(float(route_temp), 1e-12) * float(margin) / float(tilt_band)
    return (1.0 / A if A > 0 else float("inf"),
            1.0 / B if B > 0 else float("inf"))


def corner_weights_torch(ee_xy, d_orient_cup, d_orient_pour,
                         corner_zones: dict[str, Any], route_temp: float = 0.01,
                         spatial_temp: float = 0.05,
                         smooth_routing: bool = False,
                         route_alpha: float = 0.5,
                         route_y_temp: float = 0.3,
                         y_scale: float | None = None,
                         tilt_scale: float | None = None,
                         route_margin: float = 0.05,
                         route_tilt_band: float = 15.0):
    """Region-aware routing weights for corner-table EE XY positions.

    Legacy (smooth_routing=False): hard 1/0 outside the overlap band, softmax over
    orientation distance inside it.  Measured 2026-09-06: this is DISCONTINUOUS in
    position -- at y = +0.100 every config has w_cup = 1.000 (std 0.000), and 5 mm
    inside the band the mean drops to 0.453 with individual configs flipping to
    w_cup ~ 0.  Since tau = w_cup*s_cup + w_pour*s_pour, tau itself jumps at the zone
    boundary, and MPPI rollouts straddle that boundary at step_size 0.010.

    Smooth (smooth_routing=True): one continuous sigmoid over a blend of a spatial
    logit and the orientation logit,

        s        = ee_y / half_band                       (unclipped, so it saturates
                                                           deep inside either zone)
        logit    = alpha * s / route_y_temp
                 + (1 - alpha) * (d_cup - d_pour) / route_temp
        w_cup    = sigmoid(logit)

    Position dominates far from the band (recovering hard 1/0 asymptotically) and
    orientation decides inside it, with no jump anywhere.
    """
    import torch

    x = ee_xy[:, 0]
    y = ee_xy[:, 1]
    cup_box = corner_zones["cup_region"]
    pour_box = corner_zones["pour_region"]
    overlap_box = corner_zones.get("overlap_region")

    if smooth_routing:
        # Two coefficients fully determine this routing:
        #
        #   logit = ee_y / y_scale  +  (pour_tilt - cup_tilt) / tilt_scale
        #
        # y_scale is the lateral distance worth one logit unit (metres) and
        # tilt_scale the tilt disagreement worth one logit unit (degrees).  The
        # legacy (alpha, route_y_temp, route_temp) triple is three knobs for these
        # two numbers, with alpha additionally pinned to [0, 1] as a convex weight --
        # so route_y_temp existed only to widen the range alpha could reach.  Both
        # spellings are accepted; y_scale/tilt_scale take precedence when given.
        y_scale, tilt_scale = resolve_route_scales(
            corner_zones, route_temp=route_temp, route_alpha=route_alpha,
            route_y_temp=route_y_temp, y_scale=y_scale, tilt_scale=tilt_scale,
            margin=route_margin, tilt_band=route_tilt_band)
        # d_orient = margin * (tilt_max - tilt) / tilt_band, so
        # (d_cup - d_pour) = margin/tilt_band * (pour_tilt - cup_tilt); express the
        # orientation term directly in degrees via that constant.
        deg_per_d = float(route_tilt_band) / max(float(route_margin), 1e-12)
        logit = (y / y_scale
                 + (d_orient_cup - d_orient_pour) * deg_per_d / tilt_scale)
        w_cup = torch.sigmoid(logit)
        return w_cup, 1.0 - w_cup

    in_cup_box = _box_contains_torch(x, y, cup_box)
    in_pour_box = _box_contains_torch(x, y, pour_box)
    overlap_mask = in_cup_box & in_pour_box
    if overlap_box is not None:
        overlap_mask = overlap_mask | _box_contains_torch(x, y, overlap_box)

    cup_only = in_cup_box & ~overlap_mask
    pour_only = in_pour_box & ~overlap_mask
    outside = ~(cup_only | pour_only | overlap_mask)

    w_cup = torch.zeros_like(x)
    w_pour = torch.zeros_like(x)
    w_cup[cup_only] = 1.0
    w_pour[pour_only] = 1.0

    if overlap_mask.any():
        logits = torch.stack([
            d_orient_cup[overlap_mask] / float(route_temp),
            d_orient_pour[overlap_mask] / float(route_temp),
        ], dim=-1)
        weights = torch.softmax(logits, dim=-1)
        w_cup[overlap_mask] = weights[:, 0]
        w_pour[overlap_mask] = weights[:, 1]

    if outside.any():
        import os as _os
        # The inner-corner transition region is outside both boxes. The default
        # spatial softmax gives ~50/50 there (equidistant to both arms), smearing
        # cup+pour head outputs and producing the measured ~28deg mid-cross tilt
        # peak. Three routing modes for the outside/corner region:
        if _os.environ.get('R58_CORNER_ANGULAR_ROUTE', '0') == '1':
            # ANGULAR ROUTE (xy-driven, consistent with the orientation morph):
            # weight cup-vs-pour by the SAME angular progress s used by the
            # 2D-corner orientation morph (s=0 cup wing -X, s=1 pour wing -Y).
            # The tau-blend weights then sweep cup->pour through the corner in
            # lockstep with the orientation target, instead of flattening to
            # 50/50 at the diagonal. Band matches the morph's overlap_deg.
            from morph_orient import corner_blend_param
            ov = float(_os.environ.get('R58_CORNER_ROUTE_OVERLAP_DEG', '25'))
            s = corner_blend_param(x[outside], y[outside], overlap_deg=ov)
            w_pour[outside] = s
            w_cup[outside] = 1.0 - s
            return w_cup, w_pour
        elif _os.environ.get('R58_CORNER_OUTSIDE_ORIENT', '0') == '1':
            # DIAGNOSTIC: route by orientation distance (commit to whichever
            # head's pose constraint is more satisfied). Ruled out a routing-only
            # fix (made cross worse) -- kept for reproducibility.
            logits = torch.stack([
                d_orient_cup[outside] / float(route_temp),
                d_orient_pour[outside] / float(route_temp),
            ], dim=-1)
        else:
            d_cup = _box_distance_torch(x[outside], y[outside], cup_box)
            d_pour = _box_distance_torch(x[outside], y[outside], pour_box)
            logits = torch.stack([
                -d_cup / max(float(spatial_temp), 1e-6),
                -d_pour / max(float(spatial_temp), 1e-6),
            ], dim=-1)
        weights = torch.softmax(logits, dim=-1)
        w_cup[outside] = weights[:, 0]
        w_pour[outside] = weights[:, 1]

    return w_cup, w_pour
