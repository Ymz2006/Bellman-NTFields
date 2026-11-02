"""Close-range planning with the true local travel time (segment-DP task 8b, diagnostic only).

Question: if segments 1-2 (gap <= 0.16) were the actual truth instead of a learned field, how well would they plan?
The "truth" field is local_truth.py's path-optimized T*(q, g) (polyline from the straight line, Adam, the cheaper of
that path and the straight line) with its envelope-theorem gradient dT*/dq, wrapped as a differentiable field so the
unchanged native RK4 of experiments/transition_wall/evaluate.py descends it exactly as it descends a network. Every
RK4 stage re-optimizes the path from the current configuration.

Two query sets, both gap <= 0.16:
  pool     --pairs per segment of --segments, drawn by sampler.PairSampler with blend 0 (the specialists' pair
           distribution restricted to the segment); collision-checked pool poses, NOT certified (no witness path)
  frozen   every query of the 6 certified frozen cabinet suites with gap <= 0.16
Fields: truth (the oracle), straight (T = |q - g|: RK4 walks the straight line) and the listed checkpoints. Scored by the
release scorer in the cabinet scene, tilt gate off, as long.py. Per converged trajectory also its travel time
(integral of ds / S along the executed path) over a reference T* (local_truth with --ref-waypoints / --ref-steps).
"""
import sys
sys.path.insert(0, '/clean/method'); sys.path.insert(0, '/clean/jeffrey/segment_dp')
import argparse, importlib.util, json, runpy, time
from pathlib import Path
import numpy as np
import torch
from common import RELEASE, sha, write
from sampler import PairSampler, load_pool
from local_truth import local_truth
from truth import segment_cost
from fields import load_field
from eikonal import EDGES
_spec = importlib.util.spec_from_file_location('transition_wall_evaluate', '/clean/experiments/transition_wall/evaluate.py')
SHARED = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(SHARED)
sys.path.insert(0, '/workspace')
NORM = RELEASE/'configs/joint_normalization.npz'


class _Oracle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, g, field):
        with torch.enable_grad():
            T, _, gq, gg = local_truth(field.speed, x.detach(), g.detach(), waypoints=field.waypoints, steps=field.steps)
        field.calls += 1; field.evaluations += len(x)
        ctx.save_for_backward(gq, gg); return T.detach()

    @staticmethod
    def backward(ctx, grad):
        gq, gg = ctx.saved_tensors
        return grad[:, None]*gq, grad[:, None]*gg, None


class TruthField(torch.nn.Module):
    def __init__(self, speed, waypoints, steps):
        super().__init__(); self.speed, self.waypoints, self.steps, self.calls, self.evaluations = speed, waypoints, steps, 0, 0

    def forward(self, x, g):
        return _Oracle.apply(x, g, self)


class StraightField(torch.nn.Module):
    def forward(self, x, g):
        return torch.linalg.vector_norm(x-g, dim=-1)


def frozen_close(tests, high):
    manifest = json.loads((tests/'suite_manifest.json').read_text()); suites = [np.load(tests/'suites'/i['file'], allow_pickle=True) for i in manifest['suites']]
    keys = ['starts_rad', 'goals_rad', 'starts_norm', 'goals_norm']
    cat = {k: np.concatenate([s[k] for s in suites]) for k in keys}
    origin = np.concatenate([[(i['task'], n) for n in range(len(s['starts_norm']))] for i, s in zip(manifest['suites'], suites)])
    gap = np.linalg.norm(cat['starts_norm']-cat['goals_norm'], axis=1); keep = np.flatnonzero(gap <= high)
    return dict(**{k: v[keep] for k, v in cat.items()}, corner_zones_json=suites[0]['corner_zones_json'], gap=gap[keep], origin=origin[keep].astype(str))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--prepared', type=Path, required=True); p.add_argument('--scene', type=Path, required=True); p.add_argument('--tests', type=Path, required=True)
    p.add_argument('--checkpoints', type=Path, nargs='*', default=[]); p.add_argument('--labels', nargs='*', default=[]); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pairs', type=int, default=100, help='per segment'); p.add_argument('--segments', default='1,2'); p.add_argument('--seed', type=int, default=20261101)
    p.add_argument('--waypoints', type=int, default=8); p.add_argument('--steps', type=int, default=100); p.add_argument('--ref-waypoints', type=int, default=16); p.add_argument('--ref-steps', type=int, default=250)
    p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    assert len(a.checkpoints) == len(a.labels)
    a.output.mkdir(parents=True, exist_ok=True); started = time.time()
    pool, _, speed, info = load_pool(a.prepared, 'cuda:0'); speed.requires_grad_(False)
    norm = np.load(NORM); to_rad = lambda x: (x*norm['q_range']+norm['q_center']).astype(np.float32)
    edges = [float(x) for x in EDGES.split(',')] if isinstance(EDGES, str) else list(EDGES); segs = [int(x)-1 for x in a.segments.split(',')]
    high = edges[max(segs)+1]
    # pool pairs: the specialists' sampler restricted to the segments (blend 0)
    generator = torch.Generator(device='cuda:0').manual_seed(a.seed); sampler = PairSampler(pool, edges, 0.0, generator)
    n = 4 if a.smoke else a.pairs; seg = torch.tensor(segs, device='cuda:0').repeat_interleave(n)
    qi, gi, gap, _ = sampler.sample(seg); qn, gn = pool[qi].cpu().numpy(), pool[gi].cpu().numpy()
    corner = np.load(a.tests/'suites'/json.loads((a.tests/'suite_manifest.json').read_text())['suites'][0]['file'], allow_pickle=True)['corner_zones_json']
    sets = dict(pool=dict(starts_norm=qn, goals_norm=gn, starts_rad=to_rad(qn), goals_rad=to_rad(gn), corner_zones_json=corner, gap=gap.cpu().numpy(),
                          segment=seg.cpu().numpy()+1, source_index=qi.cpu().numpy(), goal_index=gi.cpu().numpy()),
                frozen=frozen_close(a.tests, high))
    if a.smoke: sets['frozen'] = {k: (v[:4] if k != 'corner_zones_json' else v) for k, v in sets['frozen'].items()}
    report = dict(gap_max=high, segments=a.segments, seed=a.seed, truth_settings=dict(waypoints=a.waypoints, steps=a.steps), reference_settings=dict(waypoints=a.ref_waypoints, steps=a.ref_steps),
                  smoke=a.smoke, **info, sets={}, fields={})
    paths = {}
    for name, suite in sets.items():
        path = a.output/f'close_{name}.npz'; np.savez(path, **suite); paths[name] = path
        q = torch.as_tensor(suite['starts_norm'], device='cuda:0'); g = torch.as_tensor(suite['goals_norm'], device='cuda:0')
        ref, line, _, _ = local_truth(speed, q, g, waypoints=a.ref_waypoints, steps=a.ref_steps)
        suite['T_ref'] = ref.cpu().numpy(); suite['T_line'] = line.cpu().numpy()
        report['sets'][name] = dict(queries=len(q), sha256=sha(path), gap_quantiles=np.quantile(suite['gap'], [0, .25, .5, .75, 1]).tolist(),
                                    straight_over_ref_median=float(np.median(suite['T_line']/suite['T_ref'])), straight_over_ref_p90=float(np.quantile(suite['T_line']/suite['T_ref'], .9)))
        print('SET', name, json.dumps(report['sets'][name]), flush=True)

    def invoke(script, args):
        sys.argv = [str(script), *map(str, args)]
        runpy.run_path(str(script), run_name='__main__')

    fields = [('truth', None), ('straight', None)] + list(zip(a.labels, a.checkpoints))
    for label, checkpoint in fields:
        if label == 'truth': net = TruthField(speed, a.waypoints, a.steps)
        elif label == 'straight': net = StraightField()
        else: net = load_field(checkpoint, 'cuda:0')
        row = dict(checkpoint=None if checkpoint is None else str(checkpoint), sha256=None if checkpoint is None else sha(checkpoint))
        for name, suite in sets.items():
            t0 = time.time(); count = len(suite['starts_norm'])
            trajectories, reached = SHARED.native(net, suite, count)
            out = a.output/f'rk4_{name}_{label}.npz'; np.savez(out, trajectories=trajectories, query_ids=np.arange(count), converged=reached)
            invoke('/workspace/score_cross_suite.py', ['--suite', paths[name], '--scene', a.scene, '--norm_file', NORM, '--trajectories', out, '--labels', label,
                                                       '--json_out', a.output/f'rk4_{name}_{label}.score.json', '--tilt_deg', '180'])
            score = json.loads((a.output/f'rk4_{name}_{label}.score.json').read_text())[label]
            reasons = {}
            for r in score['per_query'].values():
                for x in r['reasons']: reasons[x] = reasons.get(x, 0)+1
            ok = np.zeros(count, bool); ok[[int(i) for i in score['solved_ids']]] = True
            cost = np.full(count, np.nan); steps = np.array([len(t) for t in trajectories])
            with torch.no_grad():
                for i in np.flatnonzero(reached):
                    x = torch.as_tensor((trajectories[i]-norm['q_center'])/norm['q_range'], device='cuda:0', dtype=torch.float32)
                    cost[i] = float(segment_cost(speed, x[:-1], x[1:], 5).sum())
            ratio = cost/suite['T_ref']; med = lambda m: None if not m.any() else float(np.median(ratio[m]))
            row[name] = dict(success=int(ok.sum()), converged=int(reached.sum()), queries=count, failures=reasons, success_ids=np.flatnonzero(ok).tolist(),
                             cost_over_ref_converged=med(reached), cost_over_ref_success=med(ok), cost_over_ref_p90=None if not reached.any() else float(np.nanquantile(ratio, .9)),
                             steps_median=float(np.median(steps)), seconds=time.time()-t0,
                             by_segment={int(s): dict(queries=int((suite['gap'] > edges[s-1]).__and__(suite['gap'] <= edges[s]).sum()),
                                                      success=int((ok & (suite['gap'] > edges[s-1]) & (suite['gap'] <= edges[s])).sum())) for s in (1, 2)})
            print('CLOSE', name, label, json.dumps({k: v for k, v in row[name].items() if k != 'success_ids'}), flush=True)
        if label == 'truth': row.update(oracle_calls=net.calls, oracle_evaluations=net.evaluations)
        report['fields'][label] = row; write(a.output/'close.json', report)
        del net; torch.cuda.empty_cache()
    report.update(complete=True, seconds=time.time()-started); write(a.output/'close.json', report)


if __name__ == '__main__': main()
