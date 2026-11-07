"""Plan a query set with trained fields twice: maze_scale's MPPI in C-space, and the same MPPI in the warp's stretched
space (warpplan.mppi_stretched). Inference only; paired by query and planning seed.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/cross_warp/stretchq.py --queries <gapprepare run> --runs <run dir>,... --out <dir>

Per run: the field (warp.load_field for cross-warp runs; plan.load_field with the identity warp for maze_scale
ntrl-demo runs), the warp's median quadrant stretch m_q, step_z = step * m_q, budgets from the true paths' length in
z; both planners on planning seeds --plan-seeds; the exact scorer (plan.score). results.json: per run and planner,
success total and by region-pair category, collisions, stalls, median steps; <out>/<run id>/rows.npz.
"""
import argparse
import json
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
    p.add_argument('--queries', type=Path, required=True)
    p.add_argument('--runs', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--plus-width', type=int, default=6)
    p.add_argument('--gap-scale', type=int, default=0)
    p.add_argument('--plan-seeds', default='0')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from common import Logger, dump
    log = Logger(a.out / 'log.txt')
    if a.gap_scale:
        import gap as GP
        GP.install(a.gap_scale)
    elif a.plus_width:
        import plus as PL
        PL.install(a.plus_width)
    import gap as GP
    import maze as MZ
    import plan as P
    import warp as W
    import warpplan as WP
    dev = 'cuda:0'
    t0 = time.time()
    info = json.load(open(a.queries / 'maze.json'))
    maze = MZ.Maze(info['n'], dev)
    assert np.array_equal(maze.V.astype(int), np.array(info['V'])) and np.array_equal(maze.H.astype(int), np.array(info['H'])), 'maze differs'
    q = np.load(a.queries / 'queries.npz')
    tp = np.load(a.queries / 'truth_paths.npy', allow_pickle=True)
    S, G = torch.tensor(q['start'], device=dev), torch.tensor(q['goal'], device=dev)
    cat = q['category'] if 'category' in q.files else np.zeros(len(S), int)
    if a.smoke:
        S, G, cat, tp, keep = S[:12], G[:12], cat[:12], tp[:12], slice(0, 12)
    else:
        keep = slice(None)
    seeds = [int(s) for s in a.plan_seeds.split(',')]
    st = P.settings(maze)
    base_budget = torch.tensor(np.maximum(st['min_steps'], np.ceil(st['budget_factor'] * q['path_clen'][keep] / st['step'])), device=dev)
    runs = [Path(r) for r in a.runs.split(',') if r][: 1 if a.smoke else None]
    res = dict(stage='stretchq', queries=str(a.queries), plan_seeds=seeds, gpu=torch.cuda.get_device_name(0), planner=st, runs={}, smoke=a.smoke)

    def by_cat(ok):
        return {name: [int(ok[:, cat == c].sum()), int(ok[:, cat == c].size)] for c, name in enumerate(GP.CATEGORIES) if (cat == c).any()}

    for r in runs:
        d = json.load(open(r / 'results.json'))
        rid = r.name.replace('registered_', '')
        ck = r / 'training/checkpoint.pt'
        if d.get('stage') == 'cross-warp':
            net = W.load_field(ck, dev)
            warp = net.warp
        else:
            net = P.load_field(ck)
            warp = WP.Identity()
        cost = P.field_cost(net)
        m_q = WP.quadrant_stretch(warp, maze)
        step_z = st['step'] * m_q
        Lz = WP.path_length_z(warp, maze, list(tp))
        budget_z = torch.tensor(np.maximum(st['min_steps'], np.ceil(st['budget_factor'] * Lz / step_z)), device=dev)
        entry = dict(stage=d.get('stage'), seed=d.get('seed'), m_q=m_q, step_z=step_z, budget_median=dict(original=float(base_budget.float().median()), stretched=float(budget_z.float().median())))
        out = a.out / rid
        out.mkdir(exist_ok=True)
        rows_all = {}
        for planner in ('original', 'stretched'):
            ok = np.zeros((len(seeds), len(S)), bool)
            coll, conv_all, steps = np.zeros_like(ok), np.zeros_like(ok), []
            tpl = time.time()
            for i, ps in enumerate(seeds):
                if planner == 'original':
                    paths, stop, conv = P.mppi(cost, S, G, ps, base_budget, maze.lo, maze.hi, **st)
                else:
                    paths, stop, conv = WP.mppi_stretched(cost, warp, S, G, ps, budget_z, maze.lo, maze.hi, step_z=step_z, **st)
                rows = P.score(maze, [paths[j] for j in range(len(S))], conv)
                ok[i] = [x['success'] for x in rows]
                coll[i] = [x['collision'] for x in rows]
                conv_all[i] = [x['converged'] for x in rows]
                steps.append(float(stop.float().median()))
            rows_all[planner] = dict(success=ok, collision=coll, converged=conv_all)
            entry[planner] = dict(success_total=[int(ok.sum()), int(ok.size)], by_category=by_cat(ok), collision=int((coll & ~ok).sum()),
                                  stalled=int((~conv_all & ~coll).sum()), steps_median=float(np.median(steps)), seconds=time.time() - tpl)
            log(f'{rid} ({d.get("stage")}, seed {d.get("seed")}) {planner}: {ok.sum()}/{ok.size} by category {entry[planner]["by_category"]} '
                f'(walls {entry[planner]["collision"]}, stalls {entry[planner]["stalled"]}, median steps {entry[planner]["steps_median"]:.0f}, m_q {m_q:.2f}, {time.time() - tpl:.0f}s)')
        np.savez(out / 'rows.npz', **{f'{pl}_{k}': v for pl, rr in rows_all.items() for k, v in rr.items()}, category=cat)
        res['runs'][rid] = entry
        dump(a.out / 'results.json', res)
        del net
        torch.cuda.empty_cache()
    res['elapsed_seconds'] = time.time() - t0
    res['complete'] = True
    dump(a.out / 'results.json', res)
    log(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
