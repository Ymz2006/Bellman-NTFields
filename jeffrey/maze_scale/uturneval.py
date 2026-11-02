"""Plan the u-turn queries (uturn.py) of one maze with registered learned fields.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/maze_scale/uturneval.py --prepared <uturn dir> --runs <run dir>,<run dir> --out <dir>

Fields by the run's stage: ntrl / fourier (ntrl-demo checkpoint), zoom (zoomfield.load_zoomed), segdp (fields.load_chain;
a chain on a zoomed base (segdp2, config.zoom) is rebuilt with the zoomed base and Scaled levels). Planner: ntrl-demo's
MPPI with the stage-2 settings and budget rule, planning seeds 0/1/2, exact scorer (plan.score). Per run: success,
collisions and convergence per planning seed, per-query success counts, median T(start, goal) / T*. Paths of planning seed 0 are saved
per run (paths/<run id>.npy). results.json: runs[<run id>] = {stage, label, seed, success, total, ...}.
"""
import argparse
import copy
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
import maze as MZ  # noqa: E402
import plan as P  # noqa: E402
from fields import ChainField, load_chain  # noqa: E402
from zoomfield import Scaled, load_zoomed  # noqa: E402
from uturn import budget_of  # noqa: E402
from common import PLAN_SEEDS, Logger, dump, save_paths  # noqa: E402


def load_run_field(run_dir, dev):
    """(cost, label info) for a registered maze-scale run directory."""
    d = json.load(open(run_dir / 'results.json'))
    stage = d['stage']
    info = dict(stage=stage, seed=d['seed'], n=d['maze']['n'])
    if stage in ('ntrl', 'fourier'):
        net = P.load_field(run_dir / 'training/checkpoint.pt', 'metric', dev)
        info['b_scale'] = d.get('b_scale', 1.0)
        return P.field_cost(net), info
    if stage == 'zoom':
        info['zoom'] = d['zoom']
        return P.field_cost(load_zoomed(run_dir / 'training/checkpoint.pt', d['zoom'], dev)), info
    assert stage == 'segdp', stage
    info.update(config=d.get('config', {}), normal_term=d.get('normal_term', False), levels=d['levels'])
    payload = torch.load(run_dir / 'chain/field.pt', map_location=dev, weights_only=False)
    if 'zoom' in payload:
        base = load_zoomed(payload['base_checkpoint'], payload['zoom'], dev)
        models = [base]
        for sd in payload['level_state_dicts']:
            net = copy.deepcopy(base)
            net.load_state_dict(sd, strict=True)
            models.append(net)
        chain = ChainField(models, payload['edges'], payload['blend'])
    else:
        chain, _ = load_chain(run_dir / 'chain/field.pt', dev)
    return chain.cost, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True, help='uturn.py output')
    p.add_argument('--runs', required=True, help='comma-separated registered run directories')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    (a.out / 'paths').mkdir(parents=True, exist_ok=True)
    log = Logger(a.out / 'log.txt')
    dev = 'cuda:0'
    t0 = time.time()
    info = json.load(open(a.prepared / 'maze.json'))
    maze = MZ.Maze(info['n'], dev)
    assert np.array_equal(maze.V.astype(int), np.array(info['V'])), 'prepared maze differs'
    z = np.load(a.prepared / 'uturn.npz')
    S, G = torch.tensor(z['start'], device=dev), torch.tensor(z['goal'], device=dev)
    t_star = z['t_star']
    st, budget = budget_of(maze, z['path_clen'])
    budget = torch.tensor(budget, device=dev)
    seeds = PLAN_SEEDS[:1] if a.smoke else PLAN_SEEDS
    res = dict(stage='uturneval', maze=maze.describe(), uturn=str(a.prepared), queries=len(S), smoke=a.smoke, gpu=torch.cuda.get_device_name(0), runs={})
    for rd in [Path(r) for r in a.runs.split(',') if r]:
        cost, rinfo = load_run_field(rd, dev)
        assert rinfo['n'] == maze.n, (rd, rinfo['n'], maze.n)
        rid = rd.name.replace('registered_', '')
        t_pred = cost(torch.cat([S, G], 1)).cpu().numpy()
        r = dict(rinfo, success=[], collision=[], converged=[], per_query=np.zeros(len(S), int), t_over_tstar_median=float(np.median(t_pred / t_star)))
        for ps in seeds:
            paths, stop, conv = P.mppi(lambda pairs, ids=None: cost(pairs), S, G, ps, budget, maze.lo, maze.hi, **st)
            rows = P.score(maze, [paths[i] for i in range(len(S))], conv)
            ok = np.array([q['success'] for q in rows])
            r['success'].append(int(ok.sum()))
            r['collision'].append(int(sum(q['collision'] for q in rows)))
            r['converged'].append(int(sum(q['converged'] for q in rows)))
            r['per_query'] += ok
            if ps == 0:
                save_paths(a.out / 'paths' / f'{rid}.npy', [paths[i] for i in range(len(S))])
        r['per_query'] = r['per_query'].tolist()
        r['total'] = [int(sum(r['success'])), len(seeds) * len(S)]
        res['runs'][rid] = r
        log(f'{rid} {rinfo["stage"]} seed {rinfo["seed"]}: {r["total"][0]}/{r["total"][1]} (collisions {r["collision"]}, converged {r["converged"]}), T/T* {r["t_over_tstar_median"]:.2f}')
        del cost
        torch.cuda.empty_cache()
    res['elapsed_seconds'] = time.time() - t0
    res['complete'] = True
    dump(a.out / 'results.json', res)
    log(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
