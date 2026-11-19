"""Plan a query set (a cross_warp 'queries' run or any prepare run) with the fields of registered embed-warp runs: maze_scale's
evaluate.evaluate per run (MPPI seeds 0/1/2, exact scoring, success by bin), failures by arrival, results.json keyed by run id.

    python jeffrey/embed_warp/evalq.py --queries <query-set run> --runs <run dir>,<run dir>,... --out <dir>

The scene is the one the fields were trained on (the query set must be on the same scene: checked through maze.json).
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
sys.path.insert(0, str(HERE))
import features as FE  # noqa: E402
import diagnostics as DG  # noqa: E402
import scene as SC  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--queries', type=Path, required=True)
    p.add_argument('--runs', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from common import Logger, dump
    import plan as P
    import evaluate as EV
    log = Logger(a.out / 'log.txt')
    dev = 'cuda:0'
    t0 = time.time()
    runs = [Path(r) for r in a.runs.split(',') if r]
    first = json.load(open(runs[0] / 'results.json'))
    scene, info = SC.load_scene(first['prepared'], dev)
    qinfo = json.load(open(a.queries / 'maze.json'))
    assert qinfo['n'] == info['n'] and qinfo.get('variant', 'plain') == info.get('variant', 'plain'), 'query set is on another scene'
    res = dict(stage='embed-warp-evalq', queries=str(a.queries), scene=scene.describe(), runs={}, smoke=a.smoke)
    for r in runs:
        rr = json.load(open(r / 'results.json'))
        assert rr['prepared'] == first['prepared'], 'runs on different prepares'
        net = FE.load_field(r / 'training/checkpoint.pt', dev)
        cost, grad = P.field_cost(net), lambda pairs: P.field_grad(net, pairs)
        sub = a.out / r.name
        sub.mkdir(exist_ok=True)
        plan_seeds = [0] if a.smoke else [0, 1, 2]
        e = EV.evaluate(scene, a.queries, cost, grad, sub, log, label=r.name, plan_seeds=plan_seeds)
        rows = np.load(sub / 'rows.npz')
        e['failures'] = DG.failures_by_arrival(rows)
        e['label'] = rr.get('args', {}).get('gate'), rr['embed']['octaves']
        e['embed'] = rr['embed']
        e['seed'] = rr['seed']
        res['runs'][r.name] = e
        log(f'{r.name}: {e["success_total"]} failures {e["failures"]}')
        del net
        torch.cuda.empty_cache()
    res['elapsed_seconds'] = time.time() - t0
    res['complete'] = True
    dump(a.out / 'results.json', res)


if __name__ == '__main__':
    main()
