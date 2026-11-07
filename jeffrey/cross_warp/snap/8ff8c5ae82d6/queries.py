"""A second, larger query set on a prepared jeffrey/maze_scale maze (for the plus maze: plus.install first), so that each
trained field's success is measured on more queries than the prepare run's 40 per distance bin.

    CUDA_VISIBLE_DEVICES=0 python jeffrey/cross_warp/queries.py --prepared <maze_scale (plus-)prepare run> --out <dir>

Same machinery as maze_scale/prepare.py steps 3-5 (truth.Grid fields, D = T* / pitch bins, truth-path certification
with prepare.certify, straight-line and oracle controls), with new goals and starts: --goals goal poses (generator seed
--goal-seed, clearance >= 0.1 cells), --candidates starts (seed --start-seed), and per bin up to --quota certified pairs
spread over the goals (at most ceil(quota / 24) per goal and bin before relaxing). The output folder holds
queries.npz / truth_paths.npy / goals.npy in prepare.py's layout plus links to the source run's maze.json, data/ and
heldout/, so maze_scale/evaluate.py runs on it unchanged.
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--plus-width', type=int, default=6)
    p.add_argument('--goals', type=int, default=96)
    p.add_argument('--candidates', type=int, default=200_000)
    p.add_argument('--quota', type=int, default=200)
    p.add_argument('--goal-seed', type=int, default=17)
    p.add_argument('--start-seed', type=int, default=19)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from common import BINS, PLAN_SEEDS, Logger, bin_labels, by_bin, dump, save_paths
    log = Logger(a.out / 'log.txt')
    if a.plus_width:
        import plus as PL
        PL.install(a.plus_width)
    import maze as MZ
    import truth as TR
    import plan as P
    import prepare as PR
    dev = 'cuda:0'
    t0 = time.time()
    if a.smoke:
        a.goals, a.candidates, a.quota = 8, 20_000, 6
    info = json.load(open(a.prepared / 'maze.json'))
    maze = MZ.Maze(info['n'], dev)
    assert np.array_equal(maze.V.astype(int), np.array(info['V'])) and np.array_equal(maze.H.astype(int), np.array(info['H'])), 'prepared maze differs'
    for name in ('maze.json', 'data', 'heldout'):
        link = a.out / name
        if not link.exists():
            link.symlink_to((a.prepared / name).resolve())
    res = dict(stage='queries', source=str(a.prepared), maze=maze.describe(), goals=a.goals, candidates=a.candidates, quota=a.quota,
               goal_seed=a.goal_seed, start_seed=a.start_seed, smoke=a.smoke, gpu=torch.cuda.get_device_name(0))
    goals = maze.free(a.goals, torch.Generator(dev).manual_seed(a.goal_seed), PR.QUERY_CLEAR)
    grid = TR.Grid(maze)
    fields = []
    for i in range(0, len(goals), 8):
        f, it = grid.solve(goals[i:i + 8])
        fields.append(f)
        log(f'truth goals {i}-{i + len(f) - 1}: {it} sweeps, {time.time() - t0:.0f}s')
    fields = torch.cat(fields)
    np.save(a.out / 'goals.npy', goals.cpu().numpy())
    starts = maze.free(a.candidates, torch.Generator(dev).manual_seed(a.start_seed), PR.QUERY_CLEAR)
    G = len(goals)
    T = torch.empty(G, len(starts), device=dev)
    for g in range(G):
        for i in range(0, len(starts), 50_000):
            s = starts[i:i + 50_000]
            T[g, i:i + len(s)] = grid.query_time(fields, torch.full((len(s),), g, device=dev, dtype=torch.long), s)
    Dc = (T / maze.pitch).cpu().numpy()
    rng = np.random.default_rng(23)
    picked, available = [], {}
    for b in range(len(BINS) - 1):
        gi, si = np.nonzero((Dc >= BINS[b]) & (Dc < BINS[b + 1]))
        available[bin_labels()[b]] = int(len(gi))
        if not len(gi):
            continue
        order = rng.permutation(len(gi))[:400_000]
        cap = math.ceil(a.quota / 24)
        want = min(int(1.3 * a.quota) + 4, len(order))
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
        pt, o, pl, cl = PR.certify(maze, grid, fields, gi[i:i + 256], S[i:i + 256], Gq[i:i + 256])
        tpaths += pt
        ok.append(o)
        plen.append(pl)
        clen.append(cl)
    ok, plen, clen = np.concatenate(ok), np.concatenate(plen), np.concatenate(clen)
    keep = []
    for b in range(len(BINS) - 1):
        keep += [i for i in np.nonzero(picked[:, 2] == b)[0] if ok[i]][:a.quota]
    keep = np.array(sorted(keep, key=lambda i: (picked[i, 2], i)))
    res['queries'] = dict(available_pairs=available, picked=int(len(picked)), certified=int(ok.sum()), kept=int(len(keep)),
                          per_bin={bin_labels()[b]: int((picked[keep, 2] == b).sum()) for b in range(len(BINS) - 1)},
                          goals_used=int(len(set(picked[keep, 0].tolist()))))
    kq = torch.tensor(keep, device=dev)
    S, Gq, gi = S[kq], Gq[kq], gi[kq]
    t_star = T[gi, torch.tensor(picked[keep, 1], device=dev)]
    true_dir = torch.nn.functional.normalize(-grid.gradient(fields, gi, S), dim=1)
    euclid = torch.linalg.vector_norm(S[:, :2] - Gq[:, :2], dim=1) / maze.pitch
    np.savez(a.out / 'queries.npz', start=S.cpu().numpy(), goal=Gq.cpu().numpy(), goal_index=gi.cpu().numpy(), t_star=t_star.cpu().numpy(),
             d_cells=(t_star / maze.pitch).cpu().numpy(), bin=picked[keep, 2], euclid_cells=euclid.cpu().numpy(), path_cells=plen[keep],
             path_clen=clen[keep], true_dir=true_dir.cpu().numpy())
    save_paths(a.out / 'truth_paths.npy', [tpaths[i] for i in keep])
    log(f'queries: {res["queries"]}')
    bins = picked[keep, 2]
    st = P.settings(maze)
    budget = torch.tensor(np.maximum(st['min_steps'], np.ceil(st['budget_factor'] * clen[keep] / st['step'])), device=dev)
    rows = P.score(maze, P.straight(S, Gq), [True] * len(S))
    res['straight'] = dict(success=int(sum(r['success'] for r in rows)), by_bin=by_bin(rows, bins))

    def oracle(pairs, ids):
        g = gi[ids].repeat_interleave(len(pairs) // len(ids))
        return grid.oracle_cost(fields, g, pairs[:, :3], goals)

    res['oracle'] = {}
    for ps in PLAN_SEEDS[:1]:
        paths, stop, conv = P.mppi(oracle, S, Gq, ps, budget, maze.lo, maze.hi, **st)
        rows = P.score(maze, [paths[i] for i in range(len(S))], conv)
        res['oracle'][str(ps)] = dict(success=int(sum(r['success'] for r in rows)), by_bin=by_bin(rows, bins))
        log(f'oracle MPPI seed {ps}: {res["oracle"][str(ps)]}')
    log(f'straight: {res["straight"]}')
    res['elapsed_seconds'] = time.time() - t0
    res['complete'] = True
    dump(a.out / 'results.json', res)
    log(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
