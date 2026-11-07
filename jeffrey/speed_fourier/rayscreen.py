"""Pre-training screen for part 4: ray rows (rays.py) and the places an L1 or L-infinity norm can go on the integral rows.

    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= ~/.venvs/bellman-ntfield/bin/python jeffrey/speed_fourier/rayscreen.py   -> report/rayscreen.json, .md

No training. Every candidate is a set of 16 rows on the node grid. Scores, each under the L1, L2 and L-infinity norm over the rows:

  sep      mean row difference between two adjacent cell centres across a wall / across an open passage (screen.py's score, there with L2).
           1 = the rows do not tell the two sides of a wall apart
  rank     Spearman correlation over the prepared planning queries between the row distance of start and goal and the true travel time T*:
           is the norm of the row difference already a travel time?
  rough    median over 4,000 free points of (row distance to a point 0.1 cells away) / (row distance to a point 1 cell away) * 10: 1 for rows
           that change evenly with distance (any linear ramp), above 1 when nearby points already differ as much as far ones (steps, shadow lines)

Candidates: hit counts and depths (rays.py), the exact integral of 1 / speed (part 2), the integral divided by its L1 or L-infinity norm over the
directions at each point, landmark rows (L1 or L-infinity distance between the integral rows of the point and of 16 landmark cell centres chosen
without the maze graph; the same with hit counts), plain ramps with no walls (what rank and rough are for any position code), and the
corridor-distance rows of part 1 (geo) as the reference.
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
sys.path.insert(0, str(ROOT / 'jeffrey/grid_sync'))
import linefit as LF  # noqa: E402
import rays as RY  # noqa: E402
import speedfourier as SF  # noqa: E402
import maze as MZ  # noqa: E402
import embed as EM  # noqa: E402
import screen as SC  # noqa: E402

PREPARED = {13: '20261101T192504Z-13cf1d04', 22: '20261101T193403Z-cea05796'}
NORMS = dict(L1=lambda d: np.abs(d).sum(-1), L2=lambda d: np.sqrt((d ** 2).sum(-1)), Linf=lambda d: np.abs(d).max(-1))


def read(table, xy):
    """Rows at normalized points (N, 2): linear interpolation on the node grid."""
    R = table.shape[0]
    c = [(xy[:, 0] + 0.5) * (R - 1), (xy[:, 1] + 0.5) * (R - 1)]
    return np.stack([ndimage.map_coordinates(table[..., k], c, order=1, mode='nearest') for k in range(table.shape[-1])], -1)


def scores(table, maze, wall, open_, q, free, rng):
    t = EM.standardize(np.asarray(table, np.float32)).astype(np.float64)
    at = lambda c: t[8 + 16 * c[0], 8 + 16 * c[1]]
    dw = np.array([at(a) - at(b) for a, b in wall])
    do = np.array([at(a) - at(b) for a, b in open_])
    ds = read(t, q['start'][:, :2]) - read(t, q['goal'][:, :2])
    ang = rng.uniform(0, 2 * np.pi, len(free))
    u = np.stack([np.cos(ang), np.sin(ang)], 1) * maze.pitch
    base = read(t, free)
    near, far = read(t, free + 0.1 * u) - base, read(t, free + u) - base
    out = {}
    for name, f in NORMS.items():
        out[name] = dict(sep=float(f(dw).mean() / f(do).mean()), rank=float(spearmanr(f(ds), q['t_star']).statistic),
                         rough=float(np.median(10 * f(near) / np.maximum(f(far), 1e-12))))
    return out


def main():
    res = {}
    for n in (13, 22):
        maze = MZ.Maze(n, 'cpu')
        wall, open_ = SC.pairs(maze)
        q = np.load(ROOT / f'results/registered_{PREPARED[n]}/queries.npz')
        s = LF.speed_map(maze)
        z = 1 / np.maximum(s, SF.SPEED_FLOOR)
        rng = np.random.default_rng(0)
        R = s.shape[0]
        ok = np.argwhere(s[16:-16, 16:-16] > 0.95) + 16                       # free nodes at least a cell from the edge
        free = (ok[rng.choice(len(ok), 4000, replace=False)] / (R - 1)) - 0.5
        geo = LF.landmark_table(maze, 16)[0]
        I16 = RY.integral_rows(maze, z, 16)
        hits = RY.table(maze, 'hits', 16)[0].astype(np.float64)
        cand = {'hits to the edge': hits}
        for D in (1.0, 2.0, 4.0):
            cand[f'hits within {D:g} cells'] = RY.table(maze, 'hits', 16, D)[0]
        for D in (1.0, 2.0, 4.0):
            cand[f'depth, at most {D:g} cells'] = RY.table(maze, 'depth', 16, D)[0]
        cand['integral of 1 / speed (part 2)'] = I16
        cand['unit ramps (length along the line, no walls)'] = RY.integral_rows(maze, np.ones_like(z), 16)
        for norm, name in (('l1', 'L1'), ('linf', 'Linf')):
            cand[f'integral / its {name} norm over directions'] = RY.rows(maze, 'intnorm', 16, None, norm, z)[0]
        for norm, name in (('l1', 'L1'), ('linf', 'Linf')):
            cand[f'landmark rows: {name} of integral differences (64 directions)'] = RY.rows(maze, 'markint', 16, None, norm, z)[0]
            cand[f'landmark rows: {name} of hit-count differences (64 rays)'] = RY.rows(maze, 'markhits', 16, None, norm)[0]
        cand['corridor distance to 16 landmarks (geo, part 1)'] = geo
        out = {}
        for k, t in cand.items():
            out[k] = scores(t, maze, wall, open_, q, free, rng)
            print(n, f'{k:62s}', ' | '.join(f'{nm} sep {v["sep"]:.2f} rank {v["rank"]:+.2f} rough {v["rough"]:.1f}' for nm, v in out[k].items()), flush=True)
        res[str(n)] = out
    (HERE / 'report/rayscreen.json').write_text(json.dumps(res, indent=1) + '\n')
    lines = ['# Part 4 screen (rayscreen.py). sep: row difference across a wall / across an open passage; rank: Spearman of row distance and T* over the queries; rough: see the script.',
             '', '| rows | maze | sep L1 | sep L2 | sep Linf | rank L1 | rank L2 | rank Linf | rough L2 |', '|---|---|---|---|---|---|---|---|---|']
    for k in res['22']:
        for n in ('22', '13'):
            v = res[n][k]
            lines.append(f'| {k} | {n}×{n} | ' + ' | '.join(f'{v[m]["sep"]:.2f}' for m in NORMS) + ' | ' + ' | '.join(f'{v[m]["rank"]:.2f}' for m in NORMS) + f' | {v["L2"]["rough"]:.1f} |')
    (HERE / 'report/rayscreen.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
