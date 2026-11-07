"""Plan a query set with trained fields at several MPPI step sizes. Inference only; paired by query and planning seed.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/cross_warp/stepq.py --queries <gapprepare run> --runs <run dir>,... --out <dir> \
        --step-scales 0.5,1,2,4,8,16 --cap-scales 1,2,4,8,16

The planner is maze_scale's (plan.mppi with plan.settings: 50 samples, 5-step horizon, step 0.0615 cells, the same
share of a cell on every maze) with its step multiplied by m and its temperature divided by m. That is plan.settings'
own rule when the cell changes size: the softmax stays as selective per step length. The tolerance is unchanged
(0.041 cells). The 5-step look-ahead grows with the step (0.31 m cells). Budget: plan's rule
max(min_steps, ceil(budget_factor * L / step)), with L the certified true path's length, so every m may travel 3 L.

Two planners:
  mppi      plan.mppi itself, at every m in --step-scales.
  mppi+cap  the same with each step capped at the distance to the goal (warpplan.mppi_stretched with the identity
            warp), at every m in --cap-scales. A step much longer than the tolerance could otherwise circle the
            goal without landing within it; at m = 1 the two should agree (the stretchq controls).
results.json: per run, planner and m: success total and by region-pair category, walls, stalls (stretchq's
definitions), the stalls ending within two steps of the goal, the median final distance of the stalls (cells), the
median steps. <out>/<run id>/rows.npz has the per-query flags and final distances.
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


def floats(s):
    return [float(x) for x in s.split(',') if x]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--queries', type=Path, required=True)
    p.add_argument('--runs', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--plus-width', type=int, default=6)
    p.add_argument('--gap-scale', type=int, default=0)
    p.add_argument('--plan-seeds', default='0')
    p.add_argument('--step-scales', default='0.5,1,2,4,8,16')
    p.add_argument('--cap-scales', default='1,2,4,8,16')
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
    S, G = torch.tensor(q['start'], device=dev), torch.tensor(q['goal'], device=dev)
    cat = q['category'] if 'category' in q.files else np.zeros(len(S), int)
    clen = q['path_clen']
    if a.smoke:
        S, G, cat, clen = S[:12], G[:12], cat[:12], clen[:12]
    seeds = [int(s) for s in a.plan_seeds.split(',')]
    st = P.settings(maze)
    plan_list = [('mppi', m) for m in floats(a.step_scales)] + [('mppi+cap', m) for m in floats(a.cap_scales)]
    if a.smoke:
        plan_list = [x for x in plan_list if x[1] in (1.0, 8.0)]
    runs = [Path(r) for r in a.runs.split(',') if r][: 1 if a.smoke else None]
    res = dict(stage='stepq', queries=str(a.queries), plan_seeds=seeds, gpu=torch.cuda.get_device_name(0), planner=st,
               step_cells=st['step'] / maze.pitch, tol_cells=st['tol'] / maze.pitch, plans=[f'{pl}@{m:g}' for pl, m in plan_list], runs={}, smoke=a.smoke)

    def by_cat(ok):
        return {name: [int(ok[:, cat == c].sum()), int(ok[:, cat == c].size)] for c, name in enumerate(GP.CATEGORIES) if (cat == c).any()}

    for r in runs:
        d = json.load(open(r / 'results.json'))
        rid = r.name.replace('registered_', '')
        ck = r / 'training/checkpoint.pt'
        net = W.load_field(ck, dev) if d.get('stage') == 'cross-warp' else P.load_field(ck)
        cost = P.field_cost(net)
        entry = dict(stage=d.get('stage'), seed=d.get('seed'), plans={})
        out = a.out / rid
        out.mkdir(exist_ok=True)
        arrays = {}
        for planner, m in plan_list:
            stm = dict(st, step=st['step'] * m, temperature=st['temperature'] / m)
            budget = torch.tensor(np.maximum(st['min_steps'], np.ceil(st['budget_factor'] * clen / stm['step'])), device=dev)
            ok = np.zeros((len(seeds), len(S)), bool)
            coll, conv_all, final = np.zeros_like(ok), np.zeros_like(ok), np.zeros(ok.shape)
            steps = []
            tpl = time.time()
            for i, ps in enumerate(seeds):
                if planner == 'mppi':
                    paths, stop, conv = P.mppi(cost, S, G, ps, budget, maze.lo, maze.hi, **stm)
                else:
                    paths, stop, conv = WP.mppi_stretched(cost, WP.Identity(), S, G, ps, budget, maze.lo, maze.hi, step_z=stm['step'], **stm)
                rows = P.score(maze, [paths[j] for j in range(len(S))], conv)
                ok[i] = [x['success'] for x in rows]
                coll[i] = [x['collision'] for x in rows]
                conv_all[i] = [x['converged'] for x in rows]
                final[i] = [float(torch.linalg.vector_norm(paths[j][-2] - G[j])) / maze.pitch for j in range(len(S))]
                steps.append(float(stop.float().median()))
            stall = ~conv_all & ~coll
            near = ~conv_all & (final < 2 * stm['step'] / maze.pitch)
            key = f'{planner}@{m:g}'
            arrays.update({f'{key}_success': ok, f'{key}_collision': coll, f'{key}_converged': conv_all, f'{key}_final_cells': final})
            entry['plans'][key] = dict(planner=planner, m=m, step_cells=stm['step'] / maze.pitch, lookahead_cells=stm['horizon'] * stm['step'] / maze.pitch,
                                       temperature=stm['temperature'], budget_median=float(budget.float().median()),
                                       success_total=[int(ok.sum()), int(ok.size)], by_category=by_cat(ok), collision=int((coll & ~ok).sum()),
                                       stalled=int(stall.sum()), unconverged_near_goal=int(near.sum()),
                                       stall_final_cells_median=float(np.median(final[stall])) if stall.any() else None,
                                       steps_median=float(np.median(steps)), seconds=time.time() - tpl)
            e = entry['plans'][key]
            log(f'{rid} ({d.get("stage")}, seed {d.get("seed")}) {key} (step {e["step_cells"]:.3f} cells): {ok.sum()}/{ok.size} by category {e["by_category"]} '
                f'(walls {e["collision"]}, stalls {e["stalled"]}, unconverged within 2 steps of the goal {e["unconverged_near_goal"]}, '
                f'stall final distance {e["stall_final_cells_median"]}, median steps {e["steps_median"]:.0f}, {e["seconds"]:.0f}s)')
            res['runs'][rid] = entry
            dump(a.out / 'results.json', res)
        np.savez(out / 'rows.npz', category=cat, **arrays)
        del net
        torch.cuda.empty_cache()
    res['elapsed_seconds'] = time.time() - t0
    res['complete'] = True
    dump(a.out / 'results.json', res)
    log(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
