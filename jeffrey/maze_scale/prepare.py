"""Stage 1 of the maze-scale test (one maze): maze + .obj, ntrl-demo training pairs, ground truth, distance-binned
certified queries, planner controls and the configuration pool for segment DP.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/maze_scale/prepare.py --n 13 --out <dir>

1. maze      maze.py layout (seed 1000 + n), maze.obj (cell units) checked against ntrl-demo's to_off normalization,
             maze.json (geometry, wall arrays).
2. data      ntrl-demo point-recipe pairs in the robot's C-space (data.py), data seed 0 -> data/; 50,000 held-out pairs
             (seed 1) -> heldout/ for the speed-fit diagnostic.
3. truth     --goals goal poses (clearance >= 0.1 cells, uniform heading) and T*(., goal) grids (truth.py), stored as
             float16 truth_fields.npy (G, nx, ny, nt).
4. queries   --candidates start poses (clearance >= 0.1 cells); every (start, goal) pair gets D = T* / pitch, the travel
             distance in cells at full speed; per bin of common.BINS up to --quota pairs (spread over goals, at most
             ceil(quota / 12) per goal and bin before relaxing), each certified by its truth path (truth.Grid.descend,
             collision-free under the exact scorer). queries.npz: start, goal, goal_index, t_star, d_cells, bin,
             euclid_cells, path_cells (xy length of the truth path), path_clen (its C-space length, the planner budget),
             true_dir (unit -grad T* at the start).
5. controls  straight segment; oracle = the same MPPI driven by T* (truth.Grid.oracle_cost: trilinear on the goal's grid,
             exact straight-segment cost within two nodes of the goal), planning seeds 0/1/2.
6. pool      --pool collision-free configurations (uniform in the C-space box) for segment DP -> pool.npy.
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import maze as MZ  # noqa: E402
import data as D  # noqa: E402
import truth as TR  # noqa: E402
import plan as P  # noqa: E402
from common import BINS, PLAN_SEEDS, Logger, bin_labels, by_bin, dump, save_paths  # noqa: E402

QUERY_CLEAR = 0.1


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--n-data', type=int, default=500_000)
    p.add_argument('--goals', type=int, default=48)
    p.add_argument('--quota', type=int, default=40)
    p.add_argument('--candidates', type=int, default=200_000)
    p.add_argument('--pool', type=int, default=200_000)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    log = Logger(out / 'log.txt')
    dev = 'cuda:0'
    t0 = time.time()
    if a.smoke:
        a.n_data, a.goals, a.quota, a.candidates, a.pool = 20_000, 8, 6, 20_000, 20_000
    maze = MZ.Maze(a.n, dev)
    res = dict(stage='prepare', maze=maze.describe(), smoke=a.smoke, gpu=torch.cuda.get_device_name(0), bins=BINS, query_clearance_cells=QUERY_CLEAR)

    # 1. maze
    MZ.write_obj(out / 'maze.obj', maze.boxes)
    centre, extent = MZ.to_off_transform(MZ.read_obj_vertices(out / 'maze.obj'))
    assert abs(extent - maze.E) < 1e-5 and abs(centre[0] - a.n / 2) < 1e-5 and abs(centre[1] - a.n / 2) < 1e-5, (centre, extent)
    dump(out / 'maze.json', dict(maze.describe(), V=maze.V.astype(int).tolist(), H=maze.H.astype(int).tolist(), boxes_cells=maze.boxes.tolist(),
                                 to_off=dict(centre=centre.tolist(), extent=extent)))
    log(f'maze n={a.n}: {len(maze.boxes)} walls, E={maze.E:g} cells, pitch {maze.pitch:.5f}, phi_max {maze.phi_max:.5f}, planner scale {maze.planner_scale():.4f}')

    # 2. data
    x, speed, normal, d, stats = D.sample(maze, a.n_data, 0)
    (out / 'data').mkdir(exist_ok=True)
    np.save(out / 'data/sampled_points.npy', x)
    np.save(out / 'data/speed.npy', speed)
    np.save(out / 'data/normal.npy', normal)
    res['data'] = stats
    hx, hspeed, _, hd, hstats = D.sample(maze, 5_000 if a.smoke else 50_000, 1)
    (out / 'heldout').mkdir(exist_ok=True)
    np.save(out / 'heldout/sampled_points.npy', hx)
    np.save(out / 'heldout/clearance_cells.npy', hd.astype(np.float32))
    log(f'data: {stats}')
    del x, speed, normal, d
    torch.cuda.empty_cache()

    # 3. truth
    gen = torch.Generator(dev).manual_seed(7)
    goals = maze.free(a.goals, gen, QUERY_CLEAR)
    tg = time.time()
    grid = TR.Grid(maze)
    fields, sweeps = [], []
    for i in range(0, len(goals), 8):
        f, it = grid.solve(goals[i:i + 8])
        fields.append(f)
        sweeps.append(it)
        log(f'truth goals {i}-{i + len(f) - 1}: {it} sweeps, {time.time() - tg:.0f}s')
    fields = torch.cat(fields)
    np.save(out / 'truth_fields.npy', fields.half().cpu().numpy())
    res['truth'] = dict(grid_shape=list(grid.shape), hx=grid.hx, ht=grid.ht, h_cells=TR.H_CELLS, n_theta=TR.N_THETA, sweeps=sweeps, seconds=time.time() - tg,
                        unreachable_fraction=float((~torch.isfinite(fields)).float().mean()))
    np.save(out / 'goals.npy', goals.cpu().numpy())

    # 4. queries
    starts = maze.free(a.candidates, gen, QUERY_CLEAR)
    G = len(goals)
    T = torch.empty(G, len(starts), device=dev)
    for g in range(G):
        for i in range(0, len(starts), 50_000):
            s = starts[i:i + 50_000]
            T[g, i:i + len(s)] = grid.query_time(fields, torch.full((len(s),), g, device=dev, dtype=torch.long), s)
    Dc = (T / maze.pitch).cpu().numpy()
    rng = np.random.default_rng(3)
    picked = []                                                         # (goal, start index, bin)
    available = {}
    for b in range(len(BINS) - 1):
        gi, si = np.nonzero((Dc >= BINS[b]) & (Dc < BINS[b + 1]))
        available[bin_labels()[b]] = int(len(gi))
        if not len(gi):
            continue
        order = rng.permutation(len(gi))[:200_000]
        cap = math.ceil(a.quota / 12)
        want = min(int(1.5 * a.quota) + 4, len(order))
        chosen, used, per_goal = [], set(), {}
        while len(chosen) < want:
            for k in order:
                if len(chosen) >= want:
                    break
                g = int(gi[k])
                if k not in used and per_goal.get(g, 0) < cap:
                    chosen.append(int(k))
                    used.add(int(k))
                    per_goal[g] = per_goal.get(g, 0) + 1
            cap *= 2
        picked += [(int(gi[k]), int(si[k]), b) for k in chosen]
    picked = np.array(picked)
    gi = torch.tensor(picked[:, 0], device=dev)
    S, Gq = starts[torch.tensor(picked[:, 1], device=dev)], goals[gi]
    tpaths, ok, plen, clen = [], [], [], []
    for i in range(0, len(picked), 256):
        pt, o, pl, cl = certify(maze, grid, fields, gi[i:i + 256], S[i:i + 256], Gq[i:i + 256])
        tpaths += pt
        ok.append(o)
        plen.append(pl)
        clen.append(cl)
    ok, plen, clen = np.concatenate(ok), np.concatenate(plen), np.concatenate(clen)
    keep = []
    for b in range(len(BINS) - 1):
        idx = [i for i in np.nonzero(picked[:, 2] == b)[0] if ok[i]][:a.quota]
        keep += idx
    keep = np.array(sorted(keep, key=lambda i: (picked[i, 2], i)))
    res['queries'] = dict(available_pairs=available, picked=int(len(picked)), certified=int(ok.sum()), uncertified=int((~ok).sum()), kept=int(len(keep)),
                          per_bin={bin_labels()[b]: int((picked[keep, 2] == b).sum()) for b in range(len(BINS) - 1)})
    kq = torch.tensor(keep, device=dev)
    S, Gq, gi = S[kq], Gq[kq], gi[kq]
    t_star = T[gi, torch.tensor(picked[keep, 1], device=dev)]
    true_dir = torch.nn.functional.normalize(-grid.gradient(fields, gi, S), dim=1)
    euclid = torch.linalg.vector_norm(S[:, :2] - Gq[:, :2], dim=1) / maze.pitch
    np.savez(out / 'queries.npz', start=S.cpu().numpy(), goal=Gq.cpu().numpy(), goal_index=gi.cpu().numpy(), t_star=t_star.cpu().numpy(),
             d_cells=(t_star / maze.pitch).cpu().numpy(), bin=picked[keep, 2], euclid_cells=euclid.cpu().numpy(), path_cells=plen[keep],
             path_clen=clen[keep], true_dir=true_dir.cpu().numpy())
    save_paths(out / 'truth_paths.npy', [tpaths[i] for i in keep])
    log(f'queries: {res["queries"]}')
    bins = picked[keep, 2]

    # 5. controls
    st = P.settings(maze)
    budget = torch.tensor(np.maximum(st['min_steps'], np.ceil(st['budget_factor'] * clen[keep] / st['step'])), device=dev)
    res['planner'] = dict(st, budget_median=float(budget.median()), budget_max=float(budget.max()))
    rows = P.score(maze, P.straight(S, Gq), [True] * len(S))
    res['straight'] = dict(success=int(sum(r['success'] for r in rows)), by_bin=by_bin(rows, bins))

    def oracle(pairs, ids):
        g = gi[ids].repeat_interleave(len(pairs) // len(ids))
        return grid.oracle_cost(fields, g, pairs[:, :3], goals)

    res['oracle'] = {}
    for ps in PLAN_SEEDS:
        tp = time.time()
        paths, stop, conv = P.mppi(oracle, S, Gq, ps, budget, maze.lo, maze.hi, **st)
        rows = P.score(maze, [paths[i] for i in range(len(S))], conv)
        res['oracle'][str(ps)] = dict(success=int(sum(r['success'] for r in rows)), converged=int(sum(r['converged'] for r in rows)),
                                      collision=int(sum(r['collision'] for r in rows)), by_bin=by_bin(rows, bins),
                                      time_ratio_median=float(np.median([r['time'] / float(t_star[i]) for i, r in enumerate(rows) if r['success']])) if any(r['success'] for r in rows) else None,
                                      seconds=time.time() - tp, success_ids=[i for i, r in enumerate(rows) if r['success']])
        if ps == 0:
            save_paths(out / 'oracle_paths_seed0.npy', [paths[i] for i in range(len(S))])
        log(f'oracle MPPI seed {ps}: {res["oracle"][str(ps)]["success"]}/{len(S)} by bin {res["oracle"][str(ps)]["by_bin"]} ({time.time() - tp:.0f}s)')
    log(f'straight: {res["straight"]}')

    # 6. pool
    pool = maze.free(a.pool, torch.Generator(dev).manual_seed(11), 0.0)
    np.save(out / 'pool.npy', pool.cpu().numpy())
    res['pool'] = dict(n=len(pool))
    res['elapsed_seconds'] = time.time() - t0
    res['complete'] = True
    dump(out / 'results.json', res)
    log(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
