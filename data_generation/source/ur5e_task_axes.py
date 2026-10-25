"""UR5e task-axis configuration shared by data generation, eval, and GUI.

The old UR5e pipeline hard-coded:
  cup:  grasp_frame +X aligned with world +Z
  pour: grasp_frame +Z aligned with world +X

Corner-table scenes make this assumption too easy to hide.  This module gives
scene YAML a single explicit source of truth.  DEFAULT_TASK_AXES preserves the
legacy convention for old artifacts; the corner-table scene overrides pour
world_axis to [0, -1, 0].

task_axes:
  cup:
    local_axis: [1, 0, 0]
    world_axis: [0, 0, 1]
  pour:
    local_axis: [0, 0, 1]
    world_axis: [0, -1, 0]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


DEFAULT_TASK_AXES = {
    "cup": {
        "local_axis": [1.0, 0.0, 0.0],
        "world_axis": [0.0, 0.0, 1.0],
    },
    "pour": {
        "local_axis": [0.0, 0.0, 1.0],
        "world_axis": [1.0, 0.0, 0.0],
    },
}


def _jsonable_axes(axes: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, list[float]]]:
    return {
        task: {
            "local_axis": [float(x) for x in spec["local_axis"]],
            "world_axis": [float(x) for x in spec["world_axis"]],
        }
        for task, spec in axes.items()
    }


def _unit_np(v: Any) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    if arr.shape != (3,):
        raise ValueError(f"axis must have shape (3,), got {arr.shape}")
    n = float(np.linalg.norm(arr))
    if n < 1e-8:
        raise ValueError("axis vector must be nonzero")
    return arr / n


def load_task_axes(scene_path: str | Path | None = None) -> dict[str, dict[str, list[float]]]:
    axes = {
        task: {
            "local_axis": list(spec["local_axis"]),
            "world_axis": list(spec["world_axis"]),
        }
        for task, spec in DEFAULT_TASK_AXES.items()
    }
    if scene_path is not None and Path(scene_path).exists():
        with open(scene_path) as f:
            data = yaml.safe_load(f) or {}
        raw = data.get("task_axes", {}) or {}
        for task in ("cup", "pour"):
            if task not in raw:
                continue
            for key in ("local_axis", "world_axis"):
                if key in raw[task]:
                    axes[task][key] = _unit_np(raw[task][key]).astype(float).tolist()
    for task in ("cup", "pour"):
        axes[task]["local_axis"] = _unit_np(axes[task]["local_axis"]).astype(float).tolist()
        axes[task]["world_axis"] = _unit_np(axes[task]["world_axis"]).astype(float).tolist()
    return axes


def canonical_task_axes(axes: Any = None) -> dict[str, dict[str, list[float]]]:
    """Normalize a task-axis dict to JSON-safe unit-vector lists."""
    if axes is None:
        axes = DEFAULT_TASK_AXES
    out = load_task_axes(None)
    for task in ("cup", "pour"):
        raw_task = axes.get(task, {}) if isinstance(axes, dict) else {}
        for key in ("local_axis", "world_axis"):
            if key in raw_task:
                out[task][key] = _unit_np(raw_task[key]).astype(float).tolist()
    return _jsonable_axes(out)


def task_axes_from_jsonish(value: Any) -> dict[str, dict[str, list[float]]] | None:
    """Parse task axes saved as JSON, scalar npz array, or a raw dict."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        elif value.size == 1:
            value = value.reshape(()).item()
        else:
            raise ValueError(f"task_axes_json array has unsupported shape {value.shape}")
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        if not value:
            return None
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError(f"unsupported task axes payload: {type(value).__name__}")
    return canonical_task_axes(value)


def task_axes_from_npz(data) -> tuple[dict[str, dict[str, list[float]]], str]:
    """Return task axes from an npz-like object and whether they were explicit.

    Older UR5e caches predate task_axes_json.  Those are treated as the legacy
    default convention, but callers can use the returned source string to mark
    the cache as legacy rather than fully provenance-clean.
    """
    if hasattr(data, "files") and "task_axes_json" in data.files:
        parsed = task_axes_from_jsonish(data["task_axes_json"])
        if parsed is not None:
            return parsed, "explicit"
    return canonical_task_axes(DEFAULT_TASK_AXES), "legacy_default"


def task_axes_equal(a: Any, b: Any, atol: float = 1e-5) -> bool:
    a_axes = canonical_task_axes(a)
    b_axes = canonical_task_axes(b)
    for task in ("cup", "pour"):
        for key in ("local_axis", "world_axis"):
            if not np.allclose(a_axes[task][key], b_axes[task][key], atol=atol, rtol=0.0):
                return False
    return True


def task_axes_json(axes: Any) -> str:
    return json.dumps(canonical_task_axes(axes), sort_keys=True)


def task_axis_arrays(axes: dict[str, dict[str, list[float]]], task: str):
    spec = axes[task]
    return _unit_np(spec["local_axis"]), _unit_np(spec["world_axis"])


def _rotmat_from_quat_wxyz_torch(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    R = torch.empty((quat.shape[0], 3, 3), dtype=quat.dtype, device=quat.device)
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


def axis_alignment_from_quat(
    quat: torch.Tensor,
    local_axis,
    world_axis,
) -> torch.Tensor:
    """Return dot((R(q) @ local_axis), world_axis) for a batch of quats."""
    local = torch.as_tensor(local_axis, dtype=quat.dtype, device=quat.device)
    world = torch.as_tensor(world_axis, dtype=quat.dtype, device=quat.device)
    local = local / local.norm().clamp(min=1e-8)
    world = world / world.norm().clamp(min=1e-8)
    R = _rotmat_from_quat_wxyz_torch(quat)
    axis_world = torch.einsum("bij,j->bi", R, local)
    return torch.einsum("bi,i->b", axis_world, world)


def tilt_deg_from_quat(quat: torch.Tensor, local_axis, world_axis) -> torch.Tensor:
    dot = axis_alignment_from_quat(quat, local_axis, world_axis)
    return torch.acos(torch.clamp(dot, -1.0, 1.0)) * (180.0 / 3.14159265)


def transform_axis_dot(T: np.ndarray, local_axis, world_axis) -> float:
    local = _unit_np(local_axis)
    world = _unit_np(world_axis)
    R = np.asarray(T[:3, :3], dtype=np.float32)
    return float((R @ local).dot(world))


def transform_tilt_deg(T: np.ndarray, local_axis, world_axis) -> float:
    dot = transform_axis_dot(T, local_axis, world_axis)
    return float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))


def axis_label(axis) -> str:
    arr = _unit_np(axis)
    names = ["X", "Y", "Z"]
    i = int(np.argmax(np.abs(arr)))
    sign = "+" if arr[i] >= 0 else "-"
    if np.allclose(np.abs(arr[i]), 1.0, atol=1e-5) and np.allclose(np.delete(arr, i), 0.0, atol=1e-5):
        return f"{sign}{names[i]}"
    return f"({arr[0]:+.2f},{arr[1]:+.2f},{arr[2]:+.2f})"
