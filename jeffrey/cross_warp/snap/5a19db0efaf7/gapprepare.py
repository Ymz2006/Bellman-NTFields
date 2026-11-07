"""Prepare a gap maze (gap.py: the 22 x 22 cross maze's four 8 x 8 quadrants pushed apart by a band of 6 k cells):
maze, ntrl-demo training pairs, ground truth and queries stratified by where their ends lie.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/cross_warp/gapprepare.py --scale 4 --out <dir>
    ... --scale 1 --source <maze_scale plus-prepare run>     (k = 1: reuse that run's maze, data and held-out pairs)

1. maze     gap.install(k); Maze(16 + 6 k); maze.obj / maze.json as maze_scale/prepare.py (variant 'gap').
2. data     ntrl-demo's point recipe (maze_scale/data.py, seed 0, 500,000 pairs) and 50,000 held-out pairs (seed 1),
            unless --source supplies them (symlinked).
3. goals    --goals poses with clearance >= 0.1 cells, stratified by region: goals / 8 in each quadrant, the rest in the
            open band (generator seed 7).
4. truth    T*(., goal) on maze_scale's C-space grid (1/15 cell, 13 headings) by truthfast.FastGrid (Dijkstra in C,
            the fixed point of truth.Grid's Jacobi sweeps).
5. queries  --candidates starts (clearance >= 0.1, seed 11); every (start, goal) pair gets a region-pair category
            (gap.CATEGORIES: same quadrant, adjacent quadrants, diagonal quadrants, quadrant - open, open - open) and its
            true travel distance; per category --quota pairs spread over the goals (at most ceil(quota / 8) per goal before
            relaxing), each certified by its truth path (collision-free under the exact scorer, prepare.certify).
            queries.npz: prepare.py's fields plus category, region_start, region_goal.
6. controls straight segment; oracle = the same MPPI on T* (planning seed 0), by category.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = next((q for q in HERE.parents if (q / 'baselines/ntrl-demo').is_dir()), Path(os.environ.get('NTFIELD_ROOT', '/home/jeffrey/bellman-ntfield')))
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
sys.path.insert(0, str(HERE))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scale', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source', type=Path, default=None)
    p.add_argument('--n-data', type=int, default=500_000)
    p.add_argument('--goals', type=int, default=48)
    p.add_argument('--quota', type=int, default=60)
    p.add_argument('--candidates', type=int, default=200_000)
    p.add_argument('--threads', type=int, default=12)
    p.add_argument('--save-fields-gb', type=float, default=1.0, help='store truth_fields.npy (float16) only below this size')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from common import BINS, Logger, bin_labels, bin_of, dump, save_paths
    log = Logger(a.out / 'log.txt')
    import gap
    gap.install(a.scale)
    import maze as MZ
    import data as D
    import plan as P
    import prepare as PR
    import truthfast as TF
    dev = 'cuda:0'
    t0 = time.time()
    if a.smoke:
        a.n_data, a.goals, a.quota, a.candidates = 20_000, 16, 4, 20_000
    n = gap.BLOCK * 2 + gap.BASE_WIDTH * a.scale
    maze = MZ.Maze(n, dev)
    res = dict(stage='gapprepare', maze=maze.describe(), scale=a.scale, smoke=a.smoke, gpu=torch.cuda.get_device_name(0), source=str(a.source) if a.source else None,
               categories=gap.CATEGORIES, quota=a.quota, query_clearance_cells=PR.QUERY_CLEAR)

    # 1-2. maze and data
    if a.source is not None:
        info = json.load(open(a.source / 'maze.json'))
        assert np.array_equal(maze.V.astype(int), np.array(info['V'])) and np.array_equal(maze.H.astype(int), np.array(info['H'])), 'source maze differs'
        for name in ('data', 'heldout'):
            if not (a.out / name).exists():
                (a.out / name).symlink_to((a.source / name).resolve())
        res['data'] = dict(source=str(a.source))
    else:
        MZ.write_obj(a.out / 'maze.obj', maze.boxes)
        centre, extent = MZ.to_off_transform(MZ.read_obj_vertices(a.out / 'maze.obj'))
        assert abs(extent - maze.E) < 1e-5 and abs(centre[0] - n / 2) < 1e-5 and abs(centre[1] - n / 2) < 1e-5, (centre, extent)
        x, speed, normal, d, stats = D.sample(maze, a.n_data, 0)
        (a.out / 'data').mkdir(exist_ok=True)
        np.save(a.out / 'data/sampled_points.npy', x)
        np.save(a.out / 'data/speed.npy', speed)
        np.save(a.out / 'data/normal.npy', normal)
        res['data'] = stats
        hx, _, _, hd, _ = D.sample(maze, 5_000 if a.smoke else 50_000, 1)
        (a.out / 'heldout').mkdir(exist_ok=True)
        np.save(a.out / 'heldout/sampled_points.npy', hx)
        np.save(a.out / 'heldout/clearance_cells.npy', hd.astype(np.float32))
        log(f'data: {stats}')
        del x, speed, normal, d
    dump(a.out / 'maze.json', dict(maze.describe(), V=maze.V.astype(int).tolist(), H=maze.H.astype(int).tolist(), boxes_cells=maze.boxes.tolist()))
    log(f'gap maze k={a.scale}: n={n}, {len(maze.boxes)} walls, E={maze.E:g} cells, pitch {maze.pitch:.5f}')

    def regions(q):
        x, y, _ = maze.to_cells(q)
        return gap.region_of(n, x.cpu().numpy(), y.cpu().numpy())

    # 3. goals, stratified by region
    gen = torch.Generator(dev).manual_seed(7)
    per_quad = a.goals // 8
    want = [per_quad] * 4 + [a.goals - 4 * per_quad]
    have = [[] for _ in range(5)]
    while any(len(h) < w for h, w in zip(have, want)):
        cand = maze.free(65_536, gen, PR.QUERY_CLEAR)
        rg = regions(cand)
        for r in range(5):
            need = want[r] - len(have[r])
            if need > 0:
                have[r] += list(cand[torch.tensor(rg == r, device=dev)][:need])
    goals = torch.stack([g for h in have for g in h])
    np.save(a.out / 'goals.npy', goals.cpu().numpy())

    # 4. truth
    tg = time.time()
    fg = TF.FastGrid(maze, a.out / 'build', threads=a.threads)
    fields = []
    for i in range(0, len(goals), 4 * a.threads):
        f, _ = fg.solve(goals[i:i + 4 * a.threads])
        fields.append(f)
        log(f'truth goals {i}-{i + len(f) - 1}: Dijkstra {time.time() - tg:.0f}s')
    fields = torch.cat(fields)
    size_gb = fields.numel() * 2 / 1e9
    if size_gb < a.save_fields_gb:
        np.save(a.out / 'truth_fields.npy', fields.half().cpu().numpy())
    res['truth'] = dict(grid_shape=list(fg.shape), hx=fg.hx, ht=fg.ht, solver='dijkstra_full.c', seconds=time.time() - tg,
                        unreachable_fraction=float((~torch.isfinite(fields)).float().mean()), saved=size_gb < a.save_fields_gb)

    # 5. queries by region-pair category
    starts = maze.free(a.candidates, torch.Generator(dev).manual_seed(11), PR.QUERY_CLEAR)
    G = len(goals)
    T = torch.empty(G, len(starts), device=dev)
    for g in range(G):
        for i in range(0, len(starts), 50_000):
            s = starts[i:i + 50_000]
            T[g, i:i + len(s)] = fg.query_time(fields, torch.full((len(s),), g, device=dev, dtype=torch.long), s)
    rs, rgl = regions(starts), regions(goals)
    cat = gap.category(rs[None, :].repeat(G, 0), rgl[:, None].repeat(len(starts), 1))
    Tn = T.cpu().numpy()
    valid = np.isfinite(Tn) & (Tn / maze.pitch >= 0.5)
    rng = np.random.default_rng(3)
    picked, available = [], {}
    for c, name in enumerate(gap.CATEGORIES):
        gi, si = np.nonzero(valid & (cat == c))
        available[name] = int(len(gi))
        if not len(gi):
            continue
        order = rng.permutation(len(gi))[:400_000]
        cap = math.ceil(a.quota / 8)
        want_n = min(int(1.5 * a.quota) + 4, len(order))
        chosen, used, per_goal = [], set(), {}
        while len(chosen) < want_n:
            for k in order:
                if len(chosen) >= want_n:
                    break
                g = int(gi[k])
                if k not in used and per_goal.get(g, 0) < cap:
                    chosen.append(int(k))
                    used.add(int(k))
                    per_goal[g] = per_goal.get(g, 0) + 1
            cap *= 2
        picked += [(int(gi[k]), int(si[k]), c) for k in chosen]
    picked = np.array(picked)
    gi_t = torch.tensor(picked[:, 0], device=dev)
    S, Gq = starts[torch.tensor(picked[:, 1], device=dev)], goals[gi_t]
    tpaths, ok, plen, clen = [], [], [], []
    for i in range(0, len(picked), 128):
        pt, o, pl, cl = PR.certify(maze, fg, fields, gi_t[i:i + 128], S[i:i + 128], Gq[i:i + 128])
        tpaths += pt
        ok.append(o)
        plen.append(pl)
        clen.append(cl)
    ok, plen, clen = np.concatenate(ok), np.concatenate(plen), np.concatenate(clen)
    keep = []
    for c in range(len(gap.CATEGORIES)):
        keep += [i for i in np.nonzero(picked[:, 2] == c)[0] if ok[i]][:a.quota]
    keep = np.array(sorted(keep, key=lambda i: (picked[i, 2], i)))
    kq = torch.tensor(keep, device=dev)
    S, Gq, gi_t = S[kq], Gq[kq], gi_t[kq]
    t_star = T[gi_t, torch.tensor(picked[keep, 1], device=dev)]
    d_cells = (t_star / maze.pitch).cpu().numpy()
    true_dir = torch.nn.functional.normalize(-fg.gradient(fields, gi_t, S), dim=1)
    euclid = torch.linalg.vector_norm(S[:, :2] - Gq[:, :2], dim=1) / maze.pitch
    category = picked[keep, 2]
    np.savez(a.out / 'queries.npz', start=S.cpu().numpy(), goal=Gq.cpu().numpy(), goal_index=gi_t.cpu().numpy(), t_star=t_star.cpu().numpy(),
             d_cells=d_cells, bin=bin_of(d_cells), category=category, region_start=regions(S), region_goal=regions(Gq),
             euclid_cells=euclid.cpu().numpy(), path_cells=plen[keep], path_clen=clen[keep], true_dir=true_dir.cpu().numpy())
    save_paths(a.out / 'truth_paths.npy', [tpaths[i] for i in keep])
    res['queries'] = dict(available_pairs=available, picked=int(len(picked)), certified=int(ok.sum()), kept=int(len(keep)),
                          per_category={name: int((category == c).sum()) for c, name in enumerate(gap.CATEGORIES)},
                          d_cells_by_category={name: [float(np.quantile(d_cells[category == c], q)) for q in (0.1, 0.5, 0.9)]
                                               for c, name in enumerate(gap.CATEGORIES) if (category == c).any()},
                          per_bin={l: int((bin_of(d_cells) == b).sum()) for b, l in enumerate(bin_labels()) if (bin_of(d_cells) == b).any()})
    log(f'queries: {res["queries"]}')

    # 6. controls
    st = P.settings(maze)
    budget = torch.tensor(np.maximum(st['min_steps'], np.ceil(st['budget_factor'] * clen[keep] / st['step'])), device=dev)
    res['planner'] = dict(st, budget_median=float(budget.median()), budget_max=float(budget.max()))

    def by_cat(ok_rows):
        ok_rows = np.asarray(ok_rows)
        return {name: [int(ok_rows[category == c].sum()), int((category == c).sum())] for c, name in enumerate(gap.CATEGORIES)}

    rows = P.score(maze, P.straight(S, Gq), [True] * len(S))
    res['straight'] = dict(success=int(sum(r['success'] for r in rows)), by_category=by_cat([r['success'] for r in rows]))

    def oracle(pairs, ids):
        g = gi_t[ids].repeat_interleave(len(pairs) // len(ids))
        return fg.oracle_cost(fields, g, pairs[:, :3], goals)

    tp = time.time()
    paths, stop, conv = P.mppi(oracle, S, Gq, 0, budget, maze.lo, maze.hi, **st)
    rows = P.score(maze, [paths[i] for i in range(len(S))], conv)
    res['oracle'] = dict(success=int(sum(r['success'] for r in rows)), by_category=by_cat([r['success'] for r in rows]), seconds=time.time() - tp)
    log(f'straight: {res["straight"]}; oracle: {res["oracle"]}')
    res['elapsed_seconds'] = time.time() - t0
    res['complete'] = True
    dump(a.out / 'results.json', res)
    log(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
