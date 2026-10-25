"""Shared, backward-compatible labels for tabletop evaluation strata."""

from __future__ import annotations

from pathlib import Path

import numpy as np


PAIR_NAMES = {0: "cupping", 1: "pouring", 2: "cross"}


def pair_labels(pair_type) -> np.ndarray:
    values = np.asarray(pair_type)
    if values.dtype.kind in "iu":
        return np.asarray([
            PAIR_NAMES.get(int(value), f"type{int(value)}") for value in values
        ])
    return values.astype(str)


def derive_query_groups(pair_type, direct_line_strict_feasible=None) -> np.ndarray:
    """Derive useful GUI groups when a suite has no explicit group labels.

    A non-direct cupping witness is the wall-detour cup stratum, while a
    non-direct cross witness is the obstacle-cross stratum.  Old suites with
    no direct-line annotation retain their ordinary pair labels.
    """
    labels = pair_labels(pair_type)
    if direct_line_strict_feasible is None:
        return labels.copy()
    direct = np.asarray(direct_line_strict_feasible, dtype=bool)
    if direct.shape != labels.shape:
        raise ValueError("direct-line labels must align with pair_type")
    groups = labels.copy().astype("<U32")
    groups[(labels == "cupping") & direct] = "cupping/ordinary"
    groups[(labels == "cupping") & ~direct] = "cupping/wall-detour"
    groups[(labels == "cross") & direct] = "cross/direct"
    groups[(labels == "cross") & ~direct] = "cross/obstacle"
    return groups


def groups_from_npz(data, n=None) -> np.ndarray:
    for key in ("query_group_label", "scenario_label"):
        if key in data.files:
            groups = np.asarray(data[key]).astype(str)
            if n is not None and len(groups) != n:
                raise ValueError(f"{key} does not align with trajectories")
            return groups
    if "pair_type_label" in data.files:
        pairs = data["pair_type_label"]
    elif "pair_type" in data.files:
        pairs = data["pair_type"]
    else:
        return np.full(int(n or 0), "unknown", dtype=str)
    direct = (
        data["direct_line_strict_feasible"]
        if "direct_line_strict_feasible" in data.files else None)
    return derive_query_groups(pairs, direct)


def eval_result_matches_suite(result_path, suite_path) -> bool:
    """Match trajectory results to one fixed suite without path-root coupling."""
    expected = Path(suite_path)
    try:
        data = np.load(result_path, allow_pickle=True)
    except (OSError, ValueError):
        return False
    if "eval_set_path" not in data.files:
        return False
    recorded = Path(str(np.asarray(data["eval_set_path"]).reshape(()).item()))
    try:
        if recorded.exists() and expected.exists():
            if recorded.resolve() == expected.resolve():
                return True
    except OSError:
        pass
    # Docker and host paths commonly have different roots; the immutable suite
    # filename remains identical in both environments.
    return recorded.name == expected.name


def result_model_label(result_path) -> str:
    """Return an explicit model label when present, else the artifact filename."""
    path = Path(result_path)
    try:
        data = np.load(path, allow_pickle=True)
        if "model_label" in data.files:
            label = str(np.asarray(data["model_label"]).reshape(()).item()).strip()
            if label:
                return label
    except (OSError, ValueError):
        pass
    return path.name
