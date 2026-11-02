"""Long-gap planning diagnostic on the cabinet (never a benchmark, never selects a checkpoint).

The frozen cabinet suites only hold short queries (start-goal 0.06-0.57 normalized). This builds 400 longer queries
from the suites' own validated endpoints: all 1,200 starts and goals are pooled, random pairs (seed 20261027) are
kept when their normalized distance is in [0.8, 1.6]. The queries are NOT certified (no witness path), so success
rates are only comparable between fields on this same set. Every listed checkpoint runs the unchanged native RK4
(experiments/transition_wall/evaluate.py) and, with --mppi-seeds, the release MPPI with its original settings
(as experiments/cabinet/evaluate.py), both scored by the release scorer in the cabinet scene with the tilt gate off.
"""
import sys
sys.path.insert(0, '/clean/method')
import argparse, importlib.util, json, runpy
from pathlib import Path
import numpy as np
import torch
from common import RELEASE, sha, write
_spec = importlib.util.spec_from_file_location('transition_wall_evaluate', '/clean/experiments/transition_wall/evaluate.py')
SHARED = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(SHARED)
sys.path.insert(0, '/workspace')
NORM = RELEASE/'configs/joint_normalization.npz'


def build(tests, count, low, high):
    manifest = json.loads((tests/'suite_manifest.json').read_text()); suites = [np.load(tests/'suites'/i['file'], allow_pickle=True) for i in manifest['suites']]
    rad = np.concatenate([np.concatenate([s['starts_rad'], s['goals_rad']]) for s in suites]); norm = np.concatenate([np.concatenate([s['starts_norm'], s['goals_norm']]) for s in suites])
    rng = np.random.default_rng(20261027); pairs = []
    while len(pairs) < count:
        i, j = rng.integers(len(norm), size=2); d = float(np.linalg.norm(norm[i]-norm[j]))
        if low <= d <= high: pairs.append((i, j))
    i, j = np.array(pairs).T
    return dict(starts_rad=rad[i], goals_rad=rad[j], starts_norm=norm[i], goals_norm=norm[j], corner_zones_json=suites[0]['corner_zones_json'],
                gap=np.linalg.norm(norm[i]-norm[j], axis=1), start_index=i, goal_index=j)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--checkpoints', type=Path, nargs='+', required=True); p.add_argument('--labels', nargs='+', required=True)
    p.add_argument('--scene', type=Path, required=True); p.add_argument('--tests', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=400); p.add_argument('--low', type=float, default=.8); p.add_argument('--high', type=float, default=1.6)
    p.add_argument('--mppi-seeds', type=int, nargs='*', default=[]); a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False); suite = build(a.tests, a.count, a.low, a.high); path = a.output/'long_queries.npz'; np.savez(path, **suite)
    report = dict(queries=a.count, gap_range=[a.low, a.high], gap_quantiles=np.quantile(suite['gap'], [0, .25, .5, .75, 1]).tolist(), suite_sha256=sha(path), fields={})

    def invoke(script, args):
        sys.argv = [str(script), *map(str, args)]
        runpy.run_path(str(script), run_name='__main__')

    for label, checkpoint in zip(a.labels, a.checkpoints):
        net = SHARED.load_scalar(checkpoint, 'cuda:0'); trajectories, reached = SHARED.native(net, np.load(path, allow_pickle=True), a.count)
        out = a.output/f'rk4_{label}.npz'; np.savez(out, trajectories=trajectories, query_ids=np.arange(a.count), converged=reached)
        invoke('/workspace/score_cross_suite.py', ['--suite', path, '--scene', a.scene, '--norm_file', NORM, '--trajectories', out, '--labels', label,
                                                   '--json_out', a.output/f'rk4_{label}.score.json', '--tilt_deg', '180'])
        row = json.loads((a.output/f'rk4_{label}.score.json').read_text())[label]
        reasons = {}
        for r in row['per_query'].values():
            for x in r['reasons']: reasons[x] = reasons.get(x, 0)+1
        report['fields'][label] = dict(checkpoint=str(checkpoint), sha256=sha(checkpoint), success=row['success'], converged=int(reached.sum()), failures=reasons,
                                       success_ids=[int(i) for i in row['solved_ids']])
        print('LONG_RK4', label, row['success'], reasons, flush=True); write(a.output/'long.json', report)
        del net; torch.cuda.empty_cache()
    if a.mppi_seeds:
        import ur5e_plan_eval_multi as planner
        import model as release_model
        release_model.load_checkpoint = lambda path, device='cuda:0': SHARED.PlannerAdapter(SHARED.load_scalar(path, device))
        base = (planner.MPPI_batch_multi, planner.MultiModel.load, planner.compute_routing_info)
        for label, checkpoint in zip(a.labels, a.checkpoints):
            for seed in a.mppi_seeds:
                out = a.output/f'mppi_{label}_seed{seed}.npz'
                planner.MPPI_batch_multi, planner.MultiModel.load, planner.compute_routing_info = base; planner._GRAPH_COST_CACHE.clear()
                invoke(RELEASE/'method/evaluate.py', ['--checkpoint', checkpoint, '--scene', a.scene, '--norm_file', NORM, '--eval_set', path, '--n_queries', a.count, '--batch_size', '1',
                    '--sample_num', '1536', '--horizon', '7', '--mppi_cost_beta', '12.5', '--mppi_steps', '400', '--step_size', '.010', '--seed', seed, '--goal_cost_weight', '4',
                    '--overlap', '.10', '--y_divide', '0', '--save_trajectories', out])
                planner.MPPI_batch_multi, planner.MultiModel.load, planner.compute_routing_info = base; planner._GRAPH_COST_CACHE.clear(); torch.cuda.empty_cache()
                key = f'{label}_mppi{seed}'
                invoke('/workspace/score_cross_suite.py', ['--suite', path, '--scene', a.scene, '--norm_file', NORM, '--trajectories', out, '--labels', key,
                                                           '--json_out', a.output/f'mppi_{label}_seed{seed}.score.json', '--tilt_deg', '180'])
                row = json.loads((a.output/f'mppi_{label}_seed{seed}.score.json').read_text())[key]
                reasons = {}
                for r in row['per_query'].values():
                    for x in r['reasons']: reasons[x] = reasons.get(x, 0)+1
                report['fields'][label][f'mppi_seed{seed}'] = dict(success=row['success'], failures=reasons, success_ids=[int(i) for i in row['solved_ids']])
                print('LONG_MPPI', label, seed, row['success'], reasons, flush=True); write(a.output/'long.json', report)
    report['complete'] = True; write(a.output/'long.json', report)


if __name__ == '__main__': main()
