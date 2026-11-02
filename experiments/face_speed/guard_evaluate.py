"""Evaluate a trained travel-time field on the frozen cabinet suites with the per-piece step filter (guard.py).

Planners: the native RK4 of experiments/transition_wall/evaluate.py with every RK4 step projected by the filter,
and the ntrl-demo MPPI (metric/field.py settings) for either field with its update projected the same way; the
release MPPI planner is not changed (frozen runtime). Scoring is the unchanged release scorer, tilt gate off,
exactly as experiments/cabinet/evaluate.py. --guard none runs the same code without the filter (control).

    guard_evaluate.py --run /clean/results/registered_<id> --scene ... --tests ... --output ... --cover faces|spheres --rows pieces|body --alpha 0.5 --floor 0.15
"""
import sys
sys.path.insert(0, '/clean/method'); sys.path.insert(0, '/clean'); sys.path.insert(0, '/workspace')
import argparse, gc, importlib.util, json, runpy, time
from pathlib import Path
from common import RELEASE, sha, write
import numpy as np
import torch
from experiments.face_speed.clearance import PieceClearance
from experiments.face_speed.guard import StepGuard
_spec = importlib.util.spec_from_file_location('transition_wall_evaluate', '/clean/experiments/transition_wall/evaluate.py')
SHARED = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(SHARED); load_scalar = SHARED.load_scalar
NORM = RELEASE/'configs/joint_normalization.npz'; TILT_OFF = '180'


def guarded_native(net, suite, count, guard):
    """experiments/transition_wall/evaluate.py native() with the step filter after each RK4 increment."""
    q = torch.tensor(suite['starts_norm'][:count], device='cuda:0'); goal = torch.tensor(suite['goals_norm'][:count], device='cuda:0')
    paths = [[x.copy()] for x in q.cpu().numpy()]; reached = np.zeros(count, bool)

    def direction(x, g):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True); grad = torch.autograd.grad(net(x, g).sum(), x)[0]
        return -grad/torch.sqrt(grad.square().sum(-1, keepdim=True)+1e-12)

    for step in range(400):
        dist = (goal-q).norm(dim=1); reached |= (dist <= .005).cpu().numpy(); ids = np.flatnonzero(~reached)
        if not len(ids): break
        qt, gt = q[ids], goal[ids]; h = dist[ids].clamp_max(.01)[:, None]
        k1 = direction(qt, gt); k2 = direction((qt+h*.5*k1).clamp(-.5, .5), gt); k3 = direction((qt+h*.5*k2).clamp(-.5, .5), gt); k4 = direction((qt+h*k3).clamp(-.5, .5), gt)
        dq = h/6*(k1+2*k2+2*k3+k4)
        if guard is not None: dq = guard.filter(qt, dq)
        q[ids] = (qt+dq).clamp(-.5, .5).detach()
        for i, x in zip(ids, q[ids].cpu().numpy()): paths[i].append(x.copy())
    reached |= ((goal-q).norm(dim=1) <= .005).cpu().numpy()
    norm = np.load(NORM); result = np.empty(count, dtype=object)
    for i, path in enumerate(paths):
        if reached[i]: path.append(suite['goals_norm'][i].copy())
        rad = np.array(path)*norm['q_range']+norm['q_center']; rad[0] = suite['starts_rad'][i]
        if reached[i]: rad[-1] = suite['goals_rad'][i]
        result[i] = rad.astype(np.float32)
    return result, reached


def pair_cost(net, method):
    """Travel time of (q, goal) rows for either field: ntrl-demo's NN.out for metric, TravelCost for Bellman."""
    if method == 'metric': return lambda coords: net.net.out(coords)[0][:, 0]
    return lambda coords: net(coords[:, :6], coords[:, 6:]).reshape(-1)


@torch.no_grad()
def guarded_ntrl_mppi(cost_of, suite, count, seed, guard, steps=200, sample_num=50, horizon=5, step=.015, temperature=50., device='cuda:0'):
    """metric/field.py ntrl_mppi (ntrl-demo settings: 50 samples, horizon 5, 200 steps, step 0.015, temperature 50)
    for either field, with the update projected by the filter."""
    torch.manual_seed(seed)
    xp = torch.cat([torch.tensor(suite['starts_norm'][:count]), torch.tensor(suite['goals_norm'][:count])], 1).to(device)
    prior = torch.zeros(count, 6, device=device); active = torch.ones(count, dtype=torch.bool, device=device)
    paths = [[x.copy()] for x in xp[:, :6].cpu().numpy()]; converged = np.zeros(count, bool)
    for _ in range(steps):
        ids = torch.nonzero(active).squeeze(1)
        if not len(ids): break
        cur = xp[ids]; tmp = cur[:, None, None, :].repeat(1, sample_num, horizon, 1)
        dP = step*torch.randn(len(ids), sample_num, 1, 6, device=device)+step*torch.randn(len(ids), sample_num, horizon, 6, device=device)
        dP = dP+2*prior[ids][:, None, None, :]; dP = dP/(torch.linalg.vector_norm(dP, dim=3, keepdim=True).clamp(min=step)/step)
        tmp[..., :6] = tmp[..., :6]+torch.cumsum(dP, dim=2)
        cost = cost_of(tmp[:, :, [0, -1], :].reshape(-1, 12)).reshape(len(ids), sample_num, 2); cost = 10*cost[..., 0]+cost[..., 1]
        weight = torch.softmax(-temperature*cost, dim=1); update = torch.einsum('bs,bsd->bd', weight, dP[:, :, 0, :])
        if guard is not None: update = guard.filter(cur[:, :6], update)
        prior[ids] = update; xp[ids, :6] = cur[:, :6]+update
        new = xp[ids, :6].cpu().numpy(); dist = torch.linalg.vector_norm(xp[ids, 6:]-xp[ids, :6], dim=1)
        for k, i in enumerate(ids.tolist()): paths[i].append(new[k].copy())
        done = dist < .01; active[ids[done]] = False; converged[ids[done].cpu().numpy()] = True
    norm = np.load(NORM); result = np.empty(count, dtype=object)
    for i, path in enumerate(paths):
        path.append(suite['goals_norm'][i].copy()); rad = np.array(path)*norm['q_range']+norm['q_center']; rad[0] = suite['starts_rad'][i]; rad[-1] = suite['goals_rad'][i]
        result[i] = rad.astype(np.float32)
    return result, converged


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True); p.add_argument('--scene', type=Path, required=True); p.add_argument('--tests', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--cover', choices=['faces', 'spheres', 'links'], default='faces'); p.add_argument('--rows', choices=['pieces', 'body', 'none'], default='pieces')
    p.add_argument('--alpha', type=float, default=.5); p.add_argument('--floor', type=float, default=.15); p.add_argument('--sweeps', type=int, default=30); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    status = json.loads((a.run/'status.json').read_text()); progress = json.loads((a.run/'training/progress.json').read_text())
    assert progress['complete'] and (a.smoke or not status['smoke']), f'{a.run.name}: training is not complete (a smoke checkpoint cannot be fully evaluated)'
    method = 'bellman' if status['method'] == 'bellman' else 'metric'; checkpoint = a.run/'training/checkpoint.pt'
    a.output.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    manifest = json.loads((a.tests/'suite_manifest.json').read_text()); assert manifest['scene_sha256'] == sha(a.scene) and (a.smoke or not manifest['smoke'])
    for item in manifest['suites']: assert sha(a.tests/'suites'/item['file']) == item['sha256'], item['file']
    suites = manifest['suites'][:1] if a.smoke else manifest['suites']; count = min(2 if a.smoke else 100, *[item['n'] for item in suites]); seeds = [0] if a.smoke else [0, 1, 2]
    if method == 'metric':
        spec = importlib.util.spec_from_file_location('cabinet_metric_field', '/clean/experiments/cabinet/metric/field.py'); FIELD = importlib.util.module_from_spec(spec); spec.loader.exec_module(FIELD); load = FIELD.load_metric
    else: load = load_scalar
    guard = None if a.rows == 'none' else StepGuard(PieceClearance(a.scene, a.cover, 'cuda:0'), a.alpha, a.floor, a.sweeps, a.rows)
    tag = 'none' if a.rows == 'none' else f'{a.cover}_{a.rows}'
    report = dict(complete=False, smoke=a.smoke, method=method, run=a.run.name, field_cover=status.get('cover'), checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), scene=str(a.scene), scene_sha256=sha(a.scene),
                  tests_manifest_sha256=sha(a.tests/'suite_manifest.json'), scorer_tilt_deg=float(TILT_OFF), guard=dict(cover=a.cover, rows=a.rows, alpha=a.alpha, floor=a.floor, sweeps=a.sweeps),
                  experiment='Cabinet task: travel-time field with the per-piece step filter on the frozen cabinet suites', scores=[], guard_reports={})

    def invoke(script, args):
        sys.argv = [str(script), *map(str, args)]; runpy.run_path(str(script), run_name='__main__')

    def score(path, suite, label, task, seed, planner, converged):
        result = a.output/(label+'.score.json')
        invoke('/workspace/score_cross_suite.py', ['--suite', suite, '--scene', a.scene, '--norm_file', NORM, '--trajectories', path, '--labels', label, '--json_out', result, '--tilt_deg', TILT_OFF])
        row = json.loads(result.read_text())[label]; assert row['n'] == count; failures = {}
        for r in row['per_query'].values():
            for reason in r['reasons']: failures[reason] = failures.get(reason, 0)+1
        report['scores'].append(dict(method=planner, task=task, planning_seed=seed, n=count, success=row['success'], label=label, failures=failures, converged=int(np.sum(converged)), success_ids=[int(i) for i in row['solved_ids']]))
        write(a.output/'evaluation.json', report); print('GUARD_SCORE', json.dumps({k: v for k, v in report['scores'][-1].items() if k != 'success_ids'}), flush=True)

    net = load(checkpoint, 'cuda:0'); methods = ['guarded_rk4']
    for item in suites:
        task = item['task']; suite = a.tests/'suites'/item['file']
        if guard: guard.summary = dict(steps=0, active_steps=0, rows_active=0, relative_change=0., retries=0, stalled=0)
        paths, reached = guarded_native(net, np.load(suite, allow_pickle=True), count, guard)
        label = f'{method}_guard_{tag}_rk4_{task}'; path = a.output/(label+'.npz'); np.savez(path, trajectories=paths, query_ids=np.arange(count), converged=reached)
        if guard: report['guard_reports'][label] = guard.report()
        score(path, suite, label, task, None, 'guarded_rk4', reached)
    methods.append('guarded_ntrl_mppi'); cost_of = pair_cost(net, method)
    for seed in seeds:
            for item in suites:
                task = item['task']; suite = a.tests/'suites'/item['file']
                if guard: guard.summary = dict(steps=0, active_steps=0, rows_active=0, relative_change=0., retries=0, stalled=0)
                paths, converged = guarded_ntrl_mppi(cost_of, np.load(suite, allow_pickle=True), count, seed, guard)
                label = f'{method}_guard_{tag}_ntrl_mppi_{task}_seed{seed}'; path = a.output/(label+'.npz'); np.savez(path, trajectories=paths, query_ids=np.arange(count), converged=converged)
                if guard: report['guard_reports'][label] = guard.report()
                score(path, suite, label, task, seed, 'guarded_ntrl_mppi', converged)
    n = count*len(suites); report['totals'] = {}
    for m in methods:
        rows = [r for r in report['scores'] if r['method'] == m]; runs = 1 if m == 'guarded_rk4' else len(seeds)
        report['totals'][m] = dict(success=sum(r['success'] for r in rows), n=n*runs, converged=sum(r['converged'] for r in rows), per_seed=None if m == 'guarded_rk4' else [sum(r['success'] for r in rows if r['planning_seed'] == s) for s in seeds])
    report['native_success'] = report['totals']['guarded_rk4']['success']; report['complete'] = True; report['elapsed_seconds'] = time.monotonic()-started
    write(a.output/'evaluation.json', report)
    lines = [f'# Cabinet task: {method} field of {a.run.name} with the step filter ({tag})', '', 'SMOKE ONLY.' if a.smoke else 'Certified queries; release scorer in the cabinet scene, tilt gate off.', '',
             f"Filter: cover {a.cover}, rows {a.rows}, alpha {a.alpha}, floor {a.floor}, {a.sweeps} sweeps." if guard else 'No filter (control).', '']
    for m in methods:
        t = report['totals'][m]; lines.append(f"{m}: {t['success']}/{t['n']} ({100*t['success']/t['n']:.2f}%), converged {t['converged']}/{t['n']}"+('' if t['per_seed'] is None else f", per seed {t['per_seed']}")+'.')
    lines += ['', '|Suite|'+'|'.join(methods)+'|', '|---|'+'---:|'*len(methods)]
    for item in suites:
        lines.append(f"|{item['task']}|"+'|'.join(f"{sum(r['success'] for r in report['scores'] if r['method'] == m and r['task'] == item['task'])}/{count*len([r for r in report['scores'] if r['method'] == m and r['task'] == item['task']])}" for m in methods)+'|')
    if guard:
        g = list(report['guard_reports'].values()); lines += ['', f"Filter activity (RK4 and MPPI steps): active on {100*np.mean([x['filter_active_frac'] for x in g]):.1f}% of steps, mean relative step change {np.mean([x['mean_relative_change'] for x in g]):.3f}, "
                                                              f"post-step halvings {sum(x['retries'] for x in g)}, stalled steps {sum(x['stalled_steps'] for x in g)}."]
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__': main()
