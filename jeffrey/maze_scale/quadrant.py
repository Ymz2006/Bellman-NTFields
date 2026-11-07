"""Quadrant queries for the plus test: start and goal in the same quadrant (the four maze corners outside the +), with
the same certified truth path in the original 22 x 22 maze and its plus variant (plus.py), binned by distance.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/maze_scale/quadrant.py --n 22 --plus-width 6 --out <dir>

Question: the plus maze's ntrl-demo plans far better than the original's. Is that because the field has fewer walls to
hold? Inside a quadrant the walls are identical in both mazes, so a query whose start, goal and whole truth path lie in
one quadrant is the same planning problem in both. Only the rest of the maze (and so the number of walls the network
must draw) differs. If ntrl-demo solves these queries more often on the plus maze, the wall count limits the field.

1. mazes     original maze.Maze(n) and the plus variant (plus.install patches the generator in this process).
             A quadrant is the set of cells with column and row both outside the + band [c0, c0 + width).
2. goals     --goals-per-quadrant goal poses per quadrant (clearance >= 0.1 cells in both mazes, uniform heading).
3. truth     truth.Grid T* per goal in each maze.
4. queries   --candidates start poses (clearance >= 0.1 cells in both mazes); a pair is eligible when start and goal are
             in the same quadrant and T*_plus and T*_orig agree within --t-tol (relative). Per distance bin (common.BINS)
             up to --quota pairs, spread over quadrants and goals, each certified by its truth path in BOTH mazes
             (collision-free under the exact scorer) with the path's centre staying inside the quadrant.
5. controls  straight segment and the oracle (MPPI on T* of that maze), planning seeds 0/1/2, in each maze.
Outputs: orig/ and plus/ each hold maze.json, uturn.npz (start, goal, quadrant, bin, t_star, d_cells, path_cells,
path_clen = the original maze's truth path, so both evaluations use the same budget), truth_paths.npy and results.json,
in the format uturneval.py reads; results.json at the top level summarises both.
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
sys.path.insert(0, str(HERE))
import maze as MZ  # noqa: E402
import plan as P  # noqa: E402
import truth as TR  # noqa: E402
import plus as PL  # noqa: E402
from uturn import certify, budget_of  # noqa: E402
from common import BINS, PLAN_SEEDS, Logger, bin_labels, by_bin, dump, save_paths  # noqa: E402

QUERY_CLEAR = 0.1


def quadrant_of(maze, q, c0, width):
    """Quadrant index 0..3 of each pose's centre cell (-1 inside the + band)."""
    x, y, _ = maze.to_cells(q)
    qx = torch.where(x < c0, 0, torch.where(x >= c0 + width, 1, -1))
    qy = torch.where(y < c0, 0, torch.where(y >= c0 + width, 1, -1))
    return torch.where((qx >= 0) & (qy >= 0), qx + 2 * qy, -1)


def path_in_quadrant(maze, path, quad, c0, width):
    return bool((quadrant_of(maze, path, c0, width) == quad).all())


def maze_json(maze):
    return dict(maze.describe(), V=maze.V.astype(int).tolist(), H=maze.H.astype(int).tolist(), boxes_cells=maze.boxes.tolist())


def solve_all(grid, goals, log, tag):
    fields = []
    for i in range(0, len(goals), 8):
        f, it = grid.solve(goals[i:i + 8])
        fields.append(f)
        log(f'{tag} truth goals {i}-{i + len(f) - 1}: {it} sweeps')
        torch.cuda.empty_cache()
    return torch.cat(fields)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, required=True)
    p.add_argument('--plus-width', type=int, default=6)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--goals-per-quadrant', type=int, default=12)
    p.add_argument('--quota', type=int, default=40)
    p.add_argument('--candidates', type=int, default=200_000)
    p.add_argument('--t-tol', type=float, default=0.005)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    out = a.out
    for sub in ('orig', 'plus'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    log = Logger(out / 'log.txt')
    dev = 'cuda:0'
    t0 = time.time()
    if a.smoke:
        a.goals_per_quadrant, a.quota, a.candidates = 2, 4, 20_000
    width = a.plus_width
    c0 = (a.n - width) // 2
    orig = MZ.Maze(a.n, dev)
    PL.install(width)
    plus = MZ.Maze(a.n, dev)
    assert plus.describe().get('variant') == 'plus'
    mazes = dict(orig=orig, plus=plus)
    quad_cells = int(((np.arange(a.n) < c0) | (np.arange(a.n) >= c0 + width)).sum()) // 2
    res = dict(stage='quadrant', n=a.n, plus_width=width, band=[c0, c0 + width], quadrant_cells=quad_cells, smoke=a.smoke, bins=BINS, t_tol=a.t_tol,
               gpu=torch.cuda.get_device_name(0), mazes={k: m.describe() for k, m in mazes.items()})
    V0, H0 = PL.open_plus(orig.V, orig.H, width)
    assert np.array_equal(V0, plus.V) and np.array_equal(H0, plus.H), 'plus maze is not the original with the + opened'
    # walls whose segment touches a quadrant cell's interior side: identical by construction (only + interior walls go)
    log(f'n={a.n}, + band [{c0}, {c0 + width}), quadrants {quad_cells}x{quad_cells} cells; walls orig {len(orig.boxes)}, plus {len(plus.boxes)}')

    def free_both(count, gen):
        out_, have = [], 0
        while have < count:
            q = orig.uniform(262144, gen)
            ok = (orig.clearance(q) >= QUERY_CLEAR) & (plus.clearance(q) >= QUERY_CLEAR) & (quadrant_of(orig, q, c0, width) >= 0)
            out_.append(q[ok])
            have += int(ok.sum())
        return torch.cat(out_)[:count]

    # 2. goals: per quadrant
    gen = torch.Generator(dev).manual_seed(11)
    pool = free_both(40 * a.goals_per_quadrant * 4, gen)
    pq = quadrant_of(orig, pool, c0, width)
    goals = torch.cat([pool[pq == k][:a.goals_per_quadrant] for k in range(4)])
    gq = quadrant_of(orig, goals, c0, width)
    assert len(goals) == 4 * a.goals_per_quadrant

    # 3. truth in both mazes
    grids = {k: TR.Grid(m) for k, m in mazes.items()}
    fields = {k: solve_all(grids[k], goals, log, k) for k in mazes}

    # 4. queries
    starts = free_both(a.candidates, gen)
    sq = quadrant_of(orig, starts, c0, width)
    T = {}
    for k in mazes:
        T[k] = torch.empty(len(goals), len(starts), device=dev)
        for g in range(len(goals)):
            for i in range(0, len(starts), 50_000):
                s = starts[i:i + 50_000]
                T[k][g, i:i + len(s)] = grids[k].query_time(fields[k], torch.full((len(s),), g, device=dev, dtype=torch.long), s)
    same_quad = gq[:, None] == sq[None, :]
    agree = (T['plus'] - T['orig']).abs() <= a.t_tol * T['orig']
    Dc = (T['orig'] / orig.pitch)
    elig = same_quad & agree & torch.isfinite(T['orig'])
    res['eligibility'] = dict(same_quadrant=int(same_quad.sum()), same_quadrant_and_agree=int(elig.sum()),
                              plus_shorter=int((same_quad & (T['plus'] < T['orig'] * (1 - a.t_tol))).sum()))
    Dc, elig = Dc.cpu().numpy(), elig.cpu().numpy()
    gqn = gq.cpu().numpy()
    rng = np.random.default_rng(5)
    picked, available = [], {}
    for b in range(len(BINS) - 1):
        gi, si = np.nonzero(elig & (Dc >= BINS[b]) & (Dc < BINS[b + 1]))
        available[bin_labels()[b]] = [int((gqn[gi] == k).sum()) for k in range(4)]
        if not len(gi):
            continue
        order = rng.permutation(len(gi))[:200_000]
        want = min(2 * a.quota + 8, len(order))
        capq, capg = math.ceil(want / 4), math.ceil(want / (4 * a.goals_per_quadrant)) + 1
        chosen, used, perq, perg = [], set(), {}, {}
        while len(chosen) < want:
            for k in order:
                if len(chosen) >= want:
                    break
                g, qd = int(gi[k]), int(gqn[gi[k]])
                if k not in used and perq.get(qd, 0) < capq and perg.get(g, 0) < capg:
                    chosen.append(int(k))
                    used.add(int(k))
                    perq[qd] = perq.get(qd, 0) + 1
                    perg[g] = perg.get(g, 0) + 1
            capq *= 2
            capg *= 2
        picked += [(int(gi[k]), int(si[k]), b) for k in chosen]
    picked = np.array(picked)
    gi = torch.tensor(picked[:, 0], device=dev)
    S, Gq = starts[torch.tensor(picked[:, 1], device=dev)], goals[gi]
    quad = gqn[picked[:, 0]]
    cert = {}
    for k, m in mazes.items():
        tp, ok, pl, cl = [], [], [], []
        for i in range(0, len(picked), 256):
            a_, b_, c_, d_ = certify(m, grids[k], fields[k], gi[i:i + 256], S[i:i + 256], Gq[i:i + 256])
            tp += a_
            ok.append(b_)
            pl.append(c_)
            cl.append(d_)
        inq = np.array([path_in_quadrant(m, tp[i], int(quad[i]), c0, width) for i in range(len(tp))])
        cert[k] = dict(paths=tp, ok=np.concatenate(ok), inq=inq, plen=np.concatenate(pl), clen=np.concatenate(cl))
    good = cert['orig']['ok'] & cert['plus']['ok'] & cert['orig']['inq'] & cert['plus']['inq']
    # keep up to quota per bin, round-robin over quadrants
    keep = []
    for b in range(len(BINS) - 1):
        byq = [[i for i in np.nonzero((picked[:, 2] == b) & (quad == k))[0] if good[i]] for k in range(4)]
        sel = []
        while len(sel) < a.quota and any(byq):
            for k in range(4):
                if byq[k] and len(sel) < a.quota:
                    sel.append(byq[k].pop(0))
        keep += sel
    keep = np.array(sorted(keep, key=lambda i: (picked[i, 2], i)))
    bins = picked[keep, 2]
    res['queries'] = dict(available_pairs_by_quadrant=available, picked=int(len(picked)), certified_both=int((cert['orig']['ok'] & cert['plus']['ok']).sum()),
                          path_in_quadrant_both=int(good.sum()), kept=int(len(keep)),
                          per_bin={bin_labels()[b]: int((bins == b).sum()) for b in range(len(BINS) - 1) if (bins == b).any()},
                          per_bin_quadrant={bin_labels()[b]: [int(((bins == b) & (quad[keep] == k)).sum()) for k in range(4)] for b in range(len(BINS) - 1) if (bins == b).any()})
    log(f'queries: {res["queries"]}')
    kq = torch.tensor(keep, device=dev)
    S, Gq, gk = S[kq], Gq[kq], gi[kq]
    si = torch.tensor(picked[keep, 1], device=dev)
    euclid = torch.linalg.vector_norm(S[:, :2] - Gq[:, :2], dim=1) / orig.pitch
    clen = cert['orig']['clen'][keep]
    res['truth_agreement'] = dict(t_rel_diff_max=float(((T['plus'][gk, si] - T['orig'][gk, si]).abs() / T['orig'][gk, si]).max()),
                                  path_clen_rel_diff_median=float(np.median(np.abs(cert['plus']['clen'][keep] - clen) / clen)))

    # 5. per maze: files + controls
    res['controls'] = {}
    for k, m in mazes.items():
        o = out / k
        dump(o / 'maze.json', maze_json(m))
        t_star = T[k][gk, si]
        np.savez(o / 'uturn.npz', start=S.cpu().numpy(), goal=Gq.cpu().numpy(), quadrant=quad[keep], bin=bins, t_star=t_star.cpu().numpy(),
                 d_cells=(t_star / m.pitch).cpu().numpy(), path_cells=cert['orig']['plen'][keep], path_clen=clen, euclid_cells=euclid.cpu().numpy())
        save_paths(o / 'truth_paths.npy', [cert[k]['paths'][i] for i in keep])
        st, budget = budget_of(m, clen)
        budget = torch.tensor(budget, device=dev)
        rows = P.score(m, P.straight(S, Gq), [True] * len(S))
        c = dict(straight=dict(success=int(sum(r['success'] for r in rows)), by_bin=by_bin(rows, bins)), oracle={})

        def oracle(pairs, ids, k=k):
            g = gk[ids].repeat_interleave(len(pairs) // len(ids))
            return grids[k].oracle_cost(fields[k], g, pairs[:, :3], goals)

        for ps in PLAN_SEEDS[:1] if a.smoke else PLAN_SEEDS:
            paths, stop, conv = P.mppi(oracle, S, Gq, ps, budget, m.lo, m.hi, **st)
            rows = P.score(m, [paths[i] for i in range(len(S))], conv)
            c['oracle'][str(ps)] = dict(success=int(sum(r['success'] for r in rows)), by_bin=by_bin(rows, bins), total=len(S))
            log(f'{k} oracle seed {ps}: {c["oracle"][str(ps)]}')
        log(f'{k} straight: {c["straight"]}')
        res['controls'][k] = c
        sub = dict(res, stage='quadrant', variant=k, maze=m.describe(), queries_kept=int(len(keep)), straight=c['straight'], oracle=c['oracle'], complete=True)
        dump(o / 'results.json', sub)
    res['elapsed_seconds'] = time.time() - t0
    res['complete'] = True
    dump(out / 'results.json', res)
    log(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
