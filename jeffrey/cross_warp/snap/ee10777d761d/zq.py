"""Plan a gap query set with trained fields in x and in the warp's space z (zplan.py). Inference only; paired by query
and planning seed.

    CUDA_VISIBLE_DEVICES=0 python jeffrey/cross_warp/zq.py --queries <gapprepare run> --runs <run dir>,... --out <dir> \
        --gap-scale 8 --plan-seeds 0,1,2

Per run: x  = plan.mppi in x (the gap study's planner, plan.settings(maze)), on the field's cost divided by t_scale;
         z  = plan.mppi in z = phi(x) (zplan.plan_z: step and tolerance x the quadrant stretch m, budget from the true
              path's length in z), the path mapped back to x exactly and scored in x.
t_scale = 1 for fields trained in x and m for fields trained in z (trainer.py z_space), the conversion between their
time units inside the quadrants (zplan module docstring). With no warp (ntrl-demo) or a uniform stretch both planners
are plan.mppi itself: the controls.
results.json: per run and planner, success total and by region-pair category, walls, stalls (stretchq's definitions),
median steps, the settings, and the quality of the successful paths: median travel time over the true optimum T*
(maze_scale's scorer: the path's time at the training speed) and median xy length over the certified true path's; also the inverse map's largest round-trip error |phi(phi^-1(z)) - z| over every z path.
<out>/<run id>/rows.npz: per planner the per-query success / collision / converged flags (plan seeds x queries).
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
    p.add_argument('--plan-seeds', default='0,1,2')
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
    import zplan as ZP
    dev = 'cuda:0'
    t0 = time.time()
    info = json.load(open(a.queries / 'maze.json'))
    maze = MZ.Maze(info['n'], dev)
    assert np.array_equal(maze.V.astype(int), np.array(info['V'])) and np.array_equal(maze.H.astype(int), np.array(info['H'])), 'maze differs'
    q = np.load(a.queries / 'queries.npz')
    S, G = torch.tensor(q['start'], device=dev), torch.tensor(q['goal'], device=dev)
    cat = q['category'] if 'category' in q.files else np.zeros(len(S), int)
    clen = q['path_clen']
    t_star, path_cells = q['t_star'], q['path_cells']
    truth_paths = np.load(a.queries / 'truth_paths.npy', allow_pickle=True)
    if a.smoke:
        S, G, cat, clen, truth_paths, t_star, path_cells = S[:12], G[:12], cat[:12], clen[:12], truth_paths[:12], t_star[:12], path_cells[:12]
    seeds = [int(s) for s in a.plan_seeds.split(',')][: 1 if a.smoke else None]
    st = P.settings(maze)
    runs = [Path(r) for r in a.runs.split(',') if r][: 2 if a.smoke else None]
    res = dict(stage='zq', queries=str(a.queries), plan_seeds=seeds, gpu=torch.cuda.get_device_name(0), planner=st, runs={}, smoke=a.smoke)

    def by_cat(ok):
        return {name: [int(ok[:, cat == c].sum()), int(ok[:, cat == c].size)] for c, name in enumerate(GP.CATEGORIES) if (cat == c).any()}

    def summary(ok, coll, conv, steps, tr, lr):
        stall = ~conv & ~coll
        med = lambda a, m: float(np.nanmedian(a[m])) if np.isfinite(a[m]).any() else None
        return dict(success_total=[int(ok.sum()), int(ok.size)], by_category=by_cat(ok), collision=int((coll & ~ok).sum()), stalled=int(stall.sum()),
                    steps_median=float(np.median(steps)), time_ratio_median=med(tr, ok), length_ratio_median=med(lr, ok),
                    time_ratio_by_category={name: med(tr, ok & (cat == c)[None]) for c, name in enumerate(GP.CATEGORIES) if (cat == c).any()})

    for r in runs:
        d = json.load(open(r / 'results.json'))
        rid = r.name.replace('registered_', '')
        ck = r / 'training/checkpoint.pt'
        net = W.load_field(ck, dev) if d.get('stage') == 'cross-warp' else P.load_field(ck)
        warp = ZP.field_warp(net)
        m = WP.quadrant_stretch(warp, maze)
        zt = ZP.z_trained(net)
        t_scale = m if zt else 1.0
        Lz = WP.path_length_z(warp, maze, list(truth_paths))
        entry = dict(stage=d.get('stage'), seed=d.get('seed'), z_trained=zt, quadrant_stretch=m, t_scale=t_scale, plans={})
        out = a.out / rid
        out.mkdir(exist_ok=True)
        arrays = {}
        worst = 0.0
        for planner in ('x', 'z'):
            ok = np.zeros((len(seeds), len(S)), bool)
            coll, conv_all = np.zeros_like(ok), np.zeros_like(ok)
            tr, lr = np.full(ok.shape, np.nan), np.full(ok.shape, np.nan)       # path time / T*, xy length / true path length
            steps = []
            tpl = time.time()
            for i, ps in enumerate(seeds):
                if planner == 'x':
                    paths, stop, conv, used = ZP.plan_x(net, maze, S, G, ps, clen, t_scale, st)
                else:
                    paths, stop, conv, used, pz = ZP.plan_z(net, maze, S, G, ps, Lz, t_scale, st)
                    for j in range(len(S)):                 # inverse map round trip on every z waypoint
                        worst = max(worst, float((warp(paths[j][1:-1]) - pz[j][1:-1]).abs().max()) if len(pz[j]) > 2 else 0.0)
                rows = P.score(maze, paths, conv)
                ok[i] = [x['success'] for x in rows]
                coll[i] = [x['collision'] for x in rows]
                conv_all[i] = [x['converged'] for x in rows]
                tr[i] = [x['time'] / max(float(t_star[j]), 1e-9) for j, x in enumerate(rows)]
                lr[i] = [x['length_cells'] / max(float(path_cells[j]), 1e-9) for j, x in enumerate(rows)]
                steps.append(float(stop.float().median()))
            arrays.update({f'{planner}_success': ok, f'{planner}_collision': coll, f'{planner}_converged': conv_all, f'{planner}_time_ratio': tr, f'{planner}_length_ratio': lr})
            e = summary(ok, coll, conv_all, steps, tr, lr)
            e.update(settings={k: (float(v) if isinstance(v, (int, float)) else v) for k, v in used.items()}, seconds=time.time() - tpl)
            entry['plans'][planner] = e
            log(f'{rid} ({d.get("stage")}, seed {d.get("seed")}, {"z" if zt else "x"}-trained, m {m:.3f}) MPPI in {planner}: {ok.sum()}/{ok.size} '
                f'by category {e["by_category"]} (walls {e["collision"]}, stalls {e["stalled"]}, median steps {e["steps_median"]:.0f}, {e["seconds"]:.0f}s)')
        entry['inverse_roundtrip_max'] = worst
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
