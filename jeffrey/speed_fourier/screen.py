"""Pre-training screen of row designs: how much the rows differ between two cell centres across a wall against across an open passage.

    OMP_NUM_THREADS=4 ~/.venvs/bellman-ntfield/bin/python jeffrey/speed_fourier/screen.py     -> report/screen.json, report/screen.md

Inference only, no training. For every pair of edge-adjacent cells the rows (standardized as the network sees them) are read at the two cell
centres and the Euclidean difference is taken; the ratio reported is mean(wall pairs) / mean(open pairs). 1 = the rows say nothing about
which side of a wall a point is on beyond what they say about any step of one cell; the corridor-distance rows (geo) of the first part of
the study, the control that planned best on 22 x 22, are the reference. Row designs:

  line          the speed rows of the first part (sum of fitted sin / cos, o = 1 - speed) and the same for the slowness 1 / speed
  integral      the integral of the fitted slowness along the line from where it enters the box (the travel time along the line); M = 0 exact
  window D      travel time for the next D cells along the line, forward and backward (F(s + D) - F(s)), from the same integral
  reach B       distance along the line reachable within travel time B, forward and backward: the integral inverted
  share         the integral divided by the line's total: the fraction of the line's travel time spent so far
  geo           corridor distance to 16 landmark cells (linefit.landmark_table)
"""
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
sys.path.insert(0, str(ROOT / 'jeffrey/grid_sync'))
import linefit as LF  # noqa: E402
import speedfourier as SF  # noqa: E402
import maze as MZ  # noqa: E402
import embed as EM  # noqa: E402


def pairs(maze):
    n, V, H = maze.n, np.asarray(maze.V), np.asarray(maze.H)
    wall, open_ = [], []
    for i in range(n):
        for j in range(n):
            if i + 1 < n:
                (wall if V[i + 1, j] else open_).append(((i, j), (i + 1, j)))
            if j + 1 < n:
                (wall if H[i, j + 1] else open_).append(((i, j), (i, j + 1)))
    return wall, open_


def separation(table, wall, open_):
    def at(c):
        return table[8 + 16 * c[0], 8 + 16 * c[1]]
    dw = np.array([np.linalg.norm(at(a) - at(b)) for a, b in wall])
    do = np.array([np.linalg.norm(at(a) - at(b)) for a, b in open_])
    return dict(wall=float(dw.mean()), open=float(do.mean()), ratio=float(dw.mean() / do.mean()), ratio_median=float(np.median(dw) / np.median(do)))


def windowed(F, D, h):
    k = int(round(D / h))
    ahead = np.concatenate([F[:, k:], np.repeat(F[:, -1:], k, 1)], 1)
    behind = np.concatenate([np.repeat(F[:, :1], k, 1), F[:, :-k]], 1)
    return ahead - F, F - behind


def reach(F, B, u_s):
    G = np.maximum.accumulate(F, 1)
    fwd, bwd = np.zeros_like(G), np.zeros_like(G)
    for i in range(G.shape[0]):
        fwd[i] = u_s[np.searchsorted(G[i], G[i] + B, side='left').clip(0, len(u_s) - 1)] - u_s
        bwd[i] = u_s - u_s[np.searchsorted(G[i], G[i] - B, side='right').clip(0, len(u_s) - 1)]
    return fwd, bwd


def main():
    res = {}
    for n in (13, 22):
        maze = MZ.Maze(n, 'cpu')
        wall, open_ = pairs(maze)
        s = LF.speed_map(maze)
        R, h, ax = s.shape[0], 1.0 / (16 * maze.E), np.linspace(-0.5, 0.5, s.shape[0])
        z, o = 1 / np.maximum(s, SF.SPEED_FLOOR), 1 - s
        out = {}
        for name, sig in (('obstacle', o), ('slow', z)):
            for scope, M in (('line', 30), ('integral', 0), ('integral', 30), ('integral', 5), ('integral', 120)):
                if scope == 'line' or name == 'slow':
                    out[f'{name} {scope} M{M}'] = separation(EM.standardize(LF.fit(sig, maze.E, 16, M, scope)[0]), wall, open_)
        out['slow both M30'] = separation(EM.standardize(LF.fit(z, maze.E, 16, 30, 'both', o_line=o)[0]), wall, open_)
        out['slow iterms M31 (128)'] = separation(EM.standardize(LF.fit(z, maze.E, 128, 31, 'iterms')[0]), wall, open_)
        for M in (0, 30):
            fits = [LF.fit_integral(z, th, M, h) for th in LF.angles(8)]
            for kind, params in (('window', (0.5, 1.0, 2.0, 4.0)), ('reach', (1, 2, 4, 8, 16))):
                for prm in params:
                    tab = np.zeros((R, R, 16), np.float32)
                    for k, (th, (ramp, terms, total, exact, u_s, u_t, f)) in enumerate(zip(LF.angles(8), fits)):
                        fwd, bwd = windowed(total, prm / maze.E, h) if kind == 'window' else reach(total, prm / maze.E, u_s)
                        tab[..., k], tab[..., 8 + k] = LF.to_table(fwd, th, u_s, u_t, ax), LF.to_table(bwd, th, u_s, u_t, ax)
                    out[f'slow {kind} {prm:g} M{M}'] = separation(EM.standardize(tab), wall, open_)
            tab = np.zeros((R, R, 16), np.float32)
            for k, (th, (ramp, terms, total, exact, u_s, u_t, f)) in enumerate(zip(LF.angles(16), [LF.fit_integral(z, th, M, h) for th in LF.angles(16)])):
                tab[..., k] = LF.to_table(total / np.maximum(np.maximum.accumulate(total, 1).max(1, keepdims=True), 1e-9), th, u_s, u_t, ax)
            out[f'slow share M{M}'] = separation(EM.standardize(tab), wall, open_)
        out['geo'] = separation(EM.standardize(LF.landmark_table(maze, 16)[0]), wall, open_)
        res[str(n)] = dict(wall_pairs=len(wall), open_pairs=len(open_), rows=out)
        for k, v in out.items():
            print(n, f'{k:28s} ratio {v["ratio"]:.2f} (median {v["ratio_median"]:.2f}) wall {v["wall"]:.3f} open {v["open"]:.3f}', flush=True)
    (HERE / 'report/screen.json').write_text(json.dumps(res, indent=1) + '\n')
    names = list(res['13']['rows'])
    lines = ['# Row separation screen (screen.py): mean row difference across a wall / across an open passage, adjacent cell centres', '', '| rows | 22×22 | 13×13 |', '|---|---|---|']
    lines += [f'| {k} | {res["22"]["rows"][k]["ratio"]:.2f} | {res["13"]["rows"][k]["ratio"]:.2f} |' for k in names]
    (HERE / 'report/screen.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
