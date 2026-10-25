"""Same frozen MPPI; time raw CPU radian configurations -> final CPU trajectory.
All input normalization, concatenation, H2D, planning, denormalization and D2H
are inside the timed call. Startup/warmup/posthoc scoring are separate.
"""

import sys, json, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
from model import load_checkpoint, compute_position_routing_info

sys.path.insert(0, "/workspace")
import ur5e_plan_eval_multi as planner


def argument(name):
    return sys.argv[sys.argv.index(name) + 1]


suite = np.load(argument("--eval_set"), allow_pickle=True)
norm = np.load(argument("--norm_file"))
qcenter = np.asarray(norm["q_center"], dtype=np.float32)
qrange = np.asarray(norm["q_range"], dtype=np.float32)
starts = np.asarray(suite["starts_rad"], dtype=np.float32)
goals = np.asarray(suite["goals_rad"], dtype=np.float32)
# Audit equality outside timing; the timed invocation below really recomputes it.
assert np.array_equal(
    (starts - qcenter) / qrange, np.asarray(suite["starts_norm"], dtype=np.float32)
)
assert np.array_equal(
    (goals - qcenter) / qrange, np.asarray(suite["goals_norm"], dtype=np.float32)
)


def load(model, path):
    model.network = load_checkpoint(path, model.Params["Device"])
    model.B = model.network.B


planner.MultiModel.load = load
planner.compute_routing_info = compute_position_routing_info
original = planner.MPPI_batch_multi
rows = []
warmups = []
cpu_paths = []


def raw_plan(
    model, start_rad, goal_rad, robot_model, q_center, q_range, device, args, kwargs
):
    torch.cuda.synchronize()
    begin = time.perf_counter()
    pair = np.concatenate(
        [(start_rad - qcenter) / qrange, (goal_rad - qcenter) / qrange]
    )[None, :]
    xp = torch.from_numpy(pair).to(device)
    result = original(model, xp, robot_model, q_center, q_range, *args, **kwargs)
    paths = [
        (torch.cat(p) * q_range + q_center).detach().cpu().numpy() for p in result[0]
    ]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - begin
    return result, paths, elapsed


def timed(model, XP_batch, robot_model, q_center, q_range, *args, **kwargs):
    assert len(XP_batch) == 1 and XP_batch.shape[1] == 12
    if kwargs.get("steps") != 400:
        assert kwargs.get("steps") == 12
        begin = time.perf_counter()
        r = original(model, XP_batch, robot_model, q_center, q_range, *args, **kwargs)
        torch.cuda.synchronize()
        warmups.append(time.perf_counter() - begin)
        return r
    i = len(rows)
    result, paths, elapsed = raw_plan(
        model,
        starts[i],
        goals[i],
        robot_model,
        q_center,
        q_range,
        XP_batch.device,
        args,
        kwargs,
    )
    assert len(paths) == 1
    cpu_paths.append(paths[0])
    rows.append(dict(query_id=i, seconds=elapsed, waypoints=len(paths[0])))
    return result


planner.MPPI_batch_multi = timed
planner.main()
p = Path(argument("--save_trajectories"))
saved = np.load(p, allow_pickle=True)
assert len(rows) == len(saved["trajectories"])
assert all(np.array_equal(a, b) for a, b in zip(cpu_paths, saved["trajectories"]))
p.with_suffix(".raw_timing.json").write_text(
    json.dumps(
        dict(
            boundary="raw CPU start/goal radians -> complete CPU trajectory radians",
            includes=[
                "CPU normalization",
                "pair concatenation",
                "H2D",
                "MPPI",
                "denormalization",
                "D2H",
            ],
            excludes=[
                "checkpoint loading",
                "startup",
                "warmup",
                "native or independent scoring",
                "serialization",
            ],
            query_count=len(rows),
            all_failures_included=True,
            raw_normalization_bitwise_matches_frozen_suite=True,
            returned_paths_bitwise_match_saved=True,
            warmup_seconds=warmups,
            rows=rows,
        ),
        indent=2,
    )
)
