"""Where do the runs that never reach the goal stop?  -> report/stalls.json (one record per registered run, cached; copy of
jeffrey/octave_detune/stalls.py for this study, plus the exact-octave reference runs of report/summary.json).

Uses each run's planning-seed-0 paths (paths_seed0.npy: start, the MPPI positions, then the goal that plan.mppi appends)
and its rows.npz flags.  For every run that never arrived (not converged) the last MPPI position is classified:
  stopped against a wall   moved < 0.1 cell over its last 20 steps, signed clearance < 0.15 cell (the 0.5 x 0.2 cell robot
                           touches or nearly touches a wall; negative = inside one)
  stopped in the open      moved < 0.1 cell, clearance >= 0.15 cell (a free corridor is 0.9 cell wide, so the robot is
                           well clear of both walls): MPPI sits in a local minimum of the learned T away from any wall
  still moving             moved >= 0.1 cell: ran out of steps
A stop also counts as on the true route when it is within 0.4 cell of the oracle path of the query (the prepared
oracle_paths_seed0.npy, MPPI on the true travel time): the field steered the right way and stalled on the way.
CPU only, no GPU.  ~/.venvs/bellman-ntfield/bin/python jeffrey/sharp_features/stalls.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
import maze as MZ  # noqa: E402

PREP = {'13': ROOT / 'results/registered_20261101T192504Z-13cf1d04', '22': ROOT / 'results/registered_20261101T193403Z-cea05796'}
OUT = HERE / 'report/stalls.json'
MOVED, NEAR, ROUTE = 0.1, 0.15, 0.4          # cells


def densify(p, res=0.01):
    out = [p[0]]
    for a, b in zip(p[:-1], p[1:]):
        k = max(1, int(np.ceil(np.abs(b - a).max() / res)))
        out += [a + (b - a) * t for t in np.linspace(0, 1, k + 1)[1:]]
    return np.array(out)


def classify(m, paths, rows, oracle):
    ok, conv, col = rows['success'][0], rows['converged'][0], rows['collision'][0]
    rec = dict(runs=int(ok.size), success=int(ok.sum()), through_wall=int((~ok & conv & col).sum()), other=int((~ok & conv & ~col).sum()),
               stop_wall=0, stop_open=0, moving=0, stop_wall_route=0, stop_open_route=0, clear=[])
    for i in np.nonzero(~ok & ~conv)[0]:
        p = np.asarray(paths[i])
        tail = p[max(0, len(p) - 22):len(p) - 1]                       # the last 20 MPPI steps (the goal is appended)
        moved = float(np.linalg.norm(tail[-1, :2] - tail[0, :2]) * m.E)
        c = float(m.clearance(torch.tensor(p[-2][None], dtype=torch.float32))[0])
        rec['clear'].append(round(c, 4))
        if moved >= MOVED:
            rec['moving'] += 1
            continue
        kind = 'stop_wall' if c < NEAR else 'stop_open'
        rec[kind] += 1
        rec[kind + '_route'] += int(float(np.min(np.linalg.norm(oracle[i] - p[-2][:2], axis=1)) * m.E) < ROUTE)
    return rec


def main():
    done = dict(json.load(open(HERE / 'queue/state.json'))['done'])
    summ = json.loads((HERE / 'report/summary.json').read_text())
    for n, v in summ['sizes'].items():
        for s, ref in v['reference'].items():
            for i, rid in enumerate(ref['runs']):
                done[f'ref|{n}|{s}|octave-ref{i}'] = rid
    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    old = {k: v for k, v in old.items() if 'stop_open_route' in v}
    mazes, oracle = {}, {}
    for n, prep in PREP.items():
        m = MZ.Maze(int(n), 'cpu')
        assert np.array_equal(m.V.astype(int), np.array(json.load(open(prep / 'maze.json'))['V'])), 'maze differs'
        mazes[n] = m
        oracle[n] = [densify(np.asarray(o)[:, :2]) for o in np.load(prep / 'oracle_paths_seed0.npy', allow_pickle=True)]
    new = 0
    for key, rid in sorted(done.items()):
        if rid in old:
            continue
        _, n, seed, spec = key.split('|')
        d = ROOT / f'results/registered_{rid}'
        if not (d / 'paths_seed0.npy').exists():
            continue
        rec = classify(mazes[n], np.load(d / 'paths_seed0.npy', allow_pickle=True), np.load(d / 'rows.npz'), oracle[n])
        old[rid] = dict(n=n, seed=seed, spec=spec, **rec)
        new += 1
    OUT.write_text(json.dumps(old))
    print(f'{OUT}: {len(old)} runs ({new} new)')


if __name__ == '__main__':
    main()
