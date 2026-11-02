"""U-turn queries around wall ends (one maze): certified start/goal pairs on the two sides of a free wall end.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/maze_scale/uturn.py --prepared <stage-1 dir> --seed 0 --out <dir>

A free wall end is an interior lattice vertex (1 <= i, j <= n - 1) with exactly one wall segment attached (V: vertical
walls at x = i over y in [j, j + 1], H: horizontal walls at y = j over x in [i, i + 1]; maze.py). Its last segment
separates two cells; the query joins their centres: both 0.5 cells from the wall and 0.5 cells back from the end, one
cell apart in a straight line through the wall, so the robot has to go around the end (a u-turn). The three other
directions at the vertex are open, so the way around is free. --count queries: a random subset of the ends (seed
--seed); a maze with fewer ends reuses them in turn with new draws; start side random, headings uniform in
[-pi/2 + 0.1, pi/2 - 0.1].
Truth: truth.Grid on the full maze, one Bellman-Ford solve per goal (as stage 1), T*, certified truth path (descend +
exact clearance). Controls: straight segment, and the oracle (ntrl-demo's MPPI on T*, planning seeds 0/1/2), with the
stage-2 planner settings and budget rule (max(min_steps, 3 x truth C-space path length / step)).
Outputs: uturn.npz (start, goal, end vertex/side, t_star, path_cells, path_clen, euclid_cells), maze.json (copy),
truth_paths.npy, results.json.
"""
import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import maze as MZ  # noqa: E402
import plan as P  # noqa: E402
import truth as TR  # noqa: E402
from common import PLAN_SEEDS, Logger, dump, save_paths  # noqa: E402

QUERY_CLEAR = 0.1


def free_ends(V, H):
    """[(i, j, direction)] for interior vertices with exactly one attached wall; direction of that wall from the vertex."""
    n = H.shape[0]
    out = []
    for i in range(1, n):
        for j in range(1, n):
            att = dict(up=bool(V[i, j]), down=bool(V[i, j - 1]), right=bool(H[i, j]), left=bool(H[i - 1, j]))
            on = [k for k, v in att.items() if v]
            if len(on) == 1:
                out.append((i, j, on[0]))
    return out


def side_cells(i, j, d):
    """Centres (cells) of the two cells beside the last segment of a wall leaving vertex (i, j) in direction d."""
    if d == 'up':
        return (i - 0.5, j + 0.5), (i + 0.5, j + 0.5)
    if d == 'down':
        return (i - 0.5, j - 0.5), (i + 0.5, j - 0.5)
    if d == 'right':
        return (i + 0.5, j - 0.5), (i + 0.5, j + 0.5)
    return (i - 0.5, j - 0.5), (i - 0.5, j + 0.5)


def certify(maze, grid, fields, gi, start, goal):
    paths = grid.descend(fields, gi, start, goal)
    ok, plen, clen = [], [], []
    for p in paths:
        dense = P.densify(p, 0.005 * maze.pitch)
        ok.append(bool((maze.clearance(dense) >= 0).all()))
        seg = p[1:] - p[:-1]
        plen.append(float(torch.linalg.vector_norm(seg[:, :2], dim=1).sum() / maze.pitch))
        clen.append(float(torch.linalg.vector_norm(seg, dim=1).sum()))
    return paths, np.array(ok), np.array(plen), np.array(clen)


def budget_of(maze, path_clen):
    st = P.settings(maze)
    return st, np.maximum(st['min_steps'], np.ceil(st['budget_factor'] * path_clen / st['step']))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--count', type=int, default=40)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    log = Logger(a.out / 'log.txt')
    dev = 'cuda:0'
    t0 = time.time()
    info = json.load(open(a.prepared / 'maze.json'))
    maze = MZ.Maze(info['n'], dev)
    assert np.array_equal(maze.V.astype(int), np.array(info['V'])) and np.array_equal(maze.H.astype(int), np.array(info['H'])), 'prepared maze differs'
    shutil.copyfile(a.prepared / 'maze.json', a.out / 'maze.json')
    if a.smoke:
        a.count = 4
    rng = np.random.default_rng(a.seed)
    ends = free_ends(maze.V, maze.H)
    order = rng.permutation(len(ends)).tolist()
    pick = (order * (a.count // len(ends) + 1))[:a.count]   # small mazes: ends reused with new headings and start side
    S, G, meta = [], [], []
    for k in pick:
        i, j, d = ends[k]
        c0, c1 = side_cells(i, j, d)
        if rng.random() < 0.5:
            c0, c1 = c1, c0
        th = rng.uniform(-math.pi / 2 + 0.1, math.pi / 2 - 0.1, 2)
        S.append((c0[0], c0[1], th[0]))
        G.append((c1[0], c1[1], th[1]))
        meta.append((i, j, ['up', 'down', 'right', 'left'].index(d)))
    S = torch.tensor(S, device=dev, dtype=maze.dtype)
    G = torch.tensor(G, device=dev, dtype=maze.dtype)
    S, G = maze.from_cells(S[:, 0], S[:, 1], S[:, 2]), maze.from_cells(G[:, 0], G[:, 1], G[:, 2])
    cs, cg = maze.clearance(S), maze.clearance(G)
    assert float(cs.min()) >= QUERY_CLEAR and float(cg.min()) >= QUERY_CLEAR, (float(cs.min()), float(cg.min()))
    res = dict(stage='uturn', maze=maze.describe(), seed=a.seed, smoke=a.smoke, gpu=torch.cuda.get_device_name(0), free_ends=len(ends), distinct_ends=len(set(pick)), queries=len(pick),
               clearance_cells_min=float(torch.minimum(cs, cg).min()))
    log(f'maze n={maze.n}: {len(ends)} free wall ends, {len(pick)} u-turn queries (start/goal clearance >= {res["clearance_cells_min"]:.3f} cells)')

    # truth: one solve per goal
    grid = TR.Grid(maze)
    tg = time.time()
    fields, sweeps = [], []
    for i in range(0, len(G), 8):
        f, it = grid.solve(G[i:i + 8])
        fields.append(f)
        sweeps.append(it)
        log(f'truth goals {i}-{i + len(f) - 1}: {it} sweeps, {time.time() - tg:.0f}s')
        del f
        torch.cuda.empty_cache()
    fields = torch.cat(fields)
    gi = torch.arange(len(G), device=dev)
    t_star = grid.query_time(fields, gi, S)
    tpaths, ok, plen, clen = certify(maze, grid, fields, gi, S, G)
    euclid = torch.linalg.vector_norm(S[:, :2] - G[:, :2], dim=1) / maze.pitch
    res['truth'] = dict(sweeps=sweeps, seconds=time.time() - tg, certified=int(ok.sum()), t_star_cells_median=float((t_star / maze.pitch).median()),
                        t_star_cells_max=float((t_star / maze.pitch).max()), path_cells_median=float(np.median(plen)), path_cells_max=float(np.max(plen)))
    log(f'truth: {res["truth"]}')
    assert ok.all(), f'{int((~ok).sum())} u-turn queries not certified by their truth path'
    np.savez(a.out / 'uturn.npz', start=S.cpu().numpy(), goal=G.cpu().numpy(), end=np.array(meta), t_star=t_star.cpu().numpy(),
             d_cells=(t_star / maze.pitch).cpu().numpy(), path_cells=plen, path_clen=clen, euclid_cells=euclid.cpu().numpy())
    save_paths(a.out / 'truth_paths.npy', tpaths)

    # controls
    st, budget = budget_of(maze, clen)
    budget = torch.tensor(budget, device=dev)
    rows = P.score(maze, P.straight(S, G), [True] * len(S))
    res['straight'] = dict(success=int(sum(r['success'] for r in rows)), total=len(S))

    def oracle(pairs, ids):
        g = gi[ids].repeat_interleave(len(pairs) // len(ids))
        return grid.oracle_cost(fields, g, pairs[:, :3], G)

    res['oracle'] = {}
    for ps in PLAN_SEEDS:
        paths, stop, conv = P.mppi(oracle, S, G, ps, budget, maze.lo, maze.hi, **st)
        rows = P.score(maze, [paths[i] for i in range(len(S))], conv)
        res['oracle'][str(ps)] = dict(success=int(sum(r['success'] for r in rows)), converged=int(sum(r['converged'] for r in rows)),
                                      collision=int(sum(r['collision'] for r in rows)), total=len(S))
        log(f'oracle MPPI seed {ps}: {res["oracle"][str(ps)]}')
    res['planner'] = dict(st, budget_median=float(budget.median()), budget_max=float(budget.max()))
    res['elapsed_seconds'] = time.time() - t0
    res['complete'] = True
    dump(a.out / 'results.json', res)
    log(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
