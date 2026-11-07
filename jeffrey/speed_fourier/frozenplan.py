"""Does a trained field still plan when the added rows are switched off? Inference only: report/frozenplan.json.

    OMP_NUM_THREADS=4 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/speed_fourier/frozenplan.py --ids <run id> ... [--check]

Per run of form top / geo: the trained network with the 16 added rows of both endpoints pinned to a constant (ablate.freeze: their median over cell
centres and headings), scored with the run's own planning evaluation (maze_scale/evaluate.evaluate: the prepared queries, MPPI seeds 0 / 1 / 2).
If planning survives, the rows shaped what the network learned from the octaves; if it falls, the network reads the rows when it plans.
--check also repeats the intact evaluation and compares it with the run's stored successes.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
sys.path.insert(0, str(ROOT / 'jeffrey/grid_sync'))
import ablate as AB  # noqa: E402
import embed as EM  # noqa: E402
import mechanism as M  # noqa: E402
import evaluate as EV  # noqa: E402
import plan as P  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ids', nargs='+', required=True)
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    out = HERE / 'report/frozenplan.json'
    res = json.loads(out.read_text()) if out.exists() else {}
    for rid in a.ids:
        if rid in res:
            continue
        maze, prepared, net, d = M.load(ROOT / f'results/registered_{rid}')
        e = d.get('embedding') or {}
        assert e.get('form') in ('top', 'geo', 'oracle'), rid
        cost, grad = P.field_cost(net), lambda pairs: P.field_grad(net, pairs)
        r = dict(n=maze.n, seed=d['seed'], form=e['form'], scope=e.get('scope'), walls=e.get('walls'), stored=d['success_total'])
        with tempfile.TemporaryDirectory() as tmp:
            quiet = lambda m: None
            if a.check:
                r['intact'] = EV.evaluate(maze, prepared, cost, grad, Path(tmp), quiet)['success_total']
            phi = net.input_mapping.__func__.__closure__[0].cell_contents
            EM.install(net, AB.freeze(maze, phi))
            ev = EV.evaluate(maze, prepared, cost, grad, Path(tmp), quiet)
        r['frozen'] = ev['success_total']
        r['frozen_t_ratio'], r['frozen_spearman'] = ev['field']['t_over_tstar_median'], ev['field']['spearman']
        r['frozen_collisions'] = sum(v['collision'] for v in ev['planning'].values())
        res[rid] = r
        out.write_text(json.dumps(res, indent=1) + '\n')
        print(rid, r, flush=True)
        del net
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
