"""Plan a query set (queries.py output, or any prepared run) with trained fields of registered runs (inference only).

    CUDA_VISIBLE_DEVICES=0 python jeffrey/cross_warp/evalq.py --queries <queries run> --runs <run dir>,<run dir>,... --out <dir>

Each run is a cross-warp run (warp.load_field) or a maze_scale ntrl / plus-ntrl / fourier run (plan.load_field);
maze_scale/evaluate.evaluate plans it on the query set (ntrl-demo's MPPI per cell, planning seeds --plan-seeds) and
scores it exactly, as in the training runs. Per run: <out>/<run id>/rows.npz and log lines; results.json collects
success totals and by bin, and by where the query's ends lie relative to the + (plus maze).
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
        band = GP.BASE_WIDTH * a.gap_scale
    else:
        band = a.plus_width
        if a.plus_width:
            import plus as PL
            PL.install(a.plus_width)
    import maze as MZ
    import plan as P
    import evaluate as EV
    import warp as W
    import diag as DG
    dev = 'cuda:0'
    t0 = time.time()
    info = json.load(open(a.queries / 'maze.json'))
    maze = MZ.Maze(info['n'], dev)
    assert np.array_equal(maze.V.astype(int), np.array(info['V'])) and np.array_equal(maze.H.astype(int), np.array(info['H'])), 'maze differs'
    q = np.load(a.queries / 'queries.npz')
    where = None
    if band:
        where = DG.where_plus(maze, q['start'], band).astype(int) + DG.where_plus(maze, q['goal'], band).astype(int)
    seeds = [int(s) for s in a.plan_seeds.split(',')]
    runs = [Path(r) for r in a.runs.split(',') if r]
    if a.smoke:
        runs = runs[:1]
    res = dict(stage='evalq', queries=str(a.queries), plan_seeds=seeds, gpu=torch.cuda.get_device_name(0), runs={}, smoke=a.smoke)
    for r in runs:
        d = json.load(open(r / 'results.json'))
        rid = r.name.replace('registered_', '')
        ck = r / 'training/checkpoint.pt'
        net = W.load_field(ck, dev) if d.get('stage') == 'cross-warp' else P.load_field(ck)
        cost, grad = P.field_cost(net), lambda pairs: P.field_grad(net, pairs)
        out = a.out / rid
        out.mkdir(exist_ok=True)
        e = EV.evaluate(maze, a.queries, cost, grad, out, log, label=f'{rid} ({d.get("stage")}, seed {d.get("seed")})', plan_seeds=seeds)
        rows = np.load(out / 'rows.npz')
        entry = dict(stage=d.get('stage'), seed=d.get('seed'), success_total=e['success_total'], success_by_bin=e['success_by_bin'],
                     collision=int(rows['collision'].sum()), stalled=int((~rows['converged']).sum()), field=e['field'])
        if where is not None:
            entry['by_where'] = {name: [int(rows['success'][:, where == k].sum()), int(rows['success'][:, where == k].size)]
                                 for k, name in ((2, 'both'), (1, 'one'), (0, 'neither'))}
        if 'category' in q.files:
            import gap as GP
            entry['by_category'] = {name: [int(rows['success'][:, q['category'] == c].sum()), int(rows['success'][:, q['category'] == c].size)]
                                    for c, name in enumerate(GP.CATEGORIES) if (q['category'] == c).any()}
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
