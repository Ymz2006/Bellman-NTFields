"""Baseline field planners with one guard: the learned distance model's predicted obstacle clearance (distance_guard.py).

Planners are the protocol's own: the native RK4 descent of experiments/transition_wall/evaluate.py (400 steps,
step min(|g - q|, 0.01)) and the ntrl-demo MPPI of experiments/face_speed/guard_evaluate.py (50 samples, horizon 5,
200 steps, step 0.015, temperature 50), batched over all suites. The guard replaces each step by its guarded step,
and a query only counts as converged when the final jump to the goal also passes the guard. --guard none is the
same code without a guard (control). Scoring is the unchanged release scorer, tilt gate off.

Every returned path is also audited with the exact geometry (not used by the planner) to split failures into
obstacle and self-collision.

    guard_only.py --output DIR --scene S.yml --tests DIR --checkpoint C --guard none|reject|project
                  [--distance distance.pt | --exact] [--eps E] [--planners rk4 ntrl_mppi]
"""
import sys
sys.path.insert(0, '/clean/method'); sys.path.insert(0, '/clean'); sys.path.insert(0, '/workspace')
import argparse, json, math, runpy, time
from pathlib import Path
import numpy as np
import torch
from common import RELEASE, sha, write
from jeffrey.fields import Field
from jeffrey.geometry import ArmClearance, load_distance_model, OBSTACLE_MARGIN, SELF_MARGIN
from jeffrey.distance_guard import DistanceGuard

NORM = RELEASE/'configs/joint_normalization.npz'; TILT_OFF = '180'


def rk4(field, S, G, guard, steps=400):
    q, goal = S.clone(), G.clone(); n = len(q); paths = [[x] for x in q.cpu().numpy()]; reached = torch.zeros(n, dtype=torch.bool, device=q.device)

    def direction(x, g):
        grad, _ = field.gradient(x, g); return -grad/torch.sqrt(grad.square().sum(-1, keepdim=True)+1e-12)

    def close(ids):
        done = (goal[ids]-q[ids]).norm(dim=1) <= .005
        if guard is not None and bool(done.any()): done[done.clone()] = guard.segment_ok(q[ids[done]], goal[ids[done]], guard.floor[ids[done]])
        reached[ids[done]] = True

    for _ in range(steps):
        ids = torch.nonzero(~reached).squeeze(1); close(ids); ids = torch.nonzero(~reached).squeeze(1)
        if not len(ids): break
        qt, gt = q[ids], goal[ids]; h = (gt-qt).norm(dim=1).clamp_max(.01)[:, None]
        k1 = direction(qt, gt); k2 = direction((qt+h*.5*k1).clamp(-.5, .5), gt); k3 = direction((qt+h*.5*k2).clamp(-.5, .5), gt); k4 = direction((qt+h*k3).clamp(-.5, .5), gt)
        dq = h/6*(k1+2*k2+2*k3+k4)
        if guard is not None: dq = guard.filter(qt, dq, ids)
        q[ids] = (qt+dq).clamp(-.5, .5)
        for i, x in zip(ids.tolist(), q[ids].cpu().numpy()): paths[i].append(x)
    close(torch.nonzero(~reached).squeeze(1))
    reached = reached.cpu().numpy()
    return [np.array(p+[g] if r else p) for p, r, g in zip(paths, reached, G.cpu().numpy())], reached


@torch.no_grad()
def ntrl_mppi(field, S, G, guard, seed, steps=200, sample_num=50, horizon=5, step=.015, temperature=50., rollout_guard=False, penalty=1e3):
    torch.manual_seed(seed); dev = S.device; n = len(S)
    q, goal = S.clone(), G.clone(); prior = torch.zeros(n, 6, device=dev); active = torch.ones(n, dtype=torch.bool, device=dev)
    paths = [[x] for x in q.cpu().numpy()]; converged = torch.zeros(n, dtype=torch.bool, device=dev)
    for _ in range(steps):
        ids = torch.nonzero(active).squeeze(1)
        if not len(ids): break
        cur = q[ids]; m = len(ids)
        dP = step*torch.randn(m, sample_num, 1, 6, device=dev)+step*torch.randn(m, sample_num, horizon, 6, device=dev)
        dP = dP+2*prior[ids][:, None, None, :]; dP = dP/(torch.linalg.vector_norm(dP, dim=3, keepdim=True).clamp(min=step)/step)
        roll = cur[:, None, None, :]+torch.cumsum(dP, dim=2)
        gg = goal[ids][:, None, :].expand(m, sample_num, 6)
        cost = torch.stack([field.value(roll[:, :, k].reshape(-1, 6), gg.reshape(-1, 6)).reshape(m, sample_num) for k in (0, -1)], -1)
        cost = 10*cost[..., 0]+cost[..., 1]
        if rollout_guard:                                                                         # rollouts the model predicts to come too close
            e = guard.env(roll.reshape(-1, 6)).reshape(m, sample_num, horizon, -1)
            cost = cost+penalty*(e < guard.floor[ids][:, None, None, :]).any(3).any(2)
        weight = torch.softmax(-temperature*cost, dim=1); update = torch.einsum('bs,bsd->bd', weight, dP[:, :, 0, :])
        if guard is not None: update = guard.filter(cur, update, ids)
        prior[ids] = update; q[ids] = cur+update
        for k, i in enumerate(ids.tolist()): paths[i].append(q[i].cpu().numpy())
        done = torch.linalg.vector_norm(goal[ids]-q[ids], dim=1) < .01
        if guard is not None and bool(done.any()): done[done.clone()] = guard.segment_ok(q[ids[done]], goal[ids[done]], guard.floor[ids[done]])
        active[ids[done]] = False; converged[ids[done]] = True
    return [np.array(p+[g]) for p, g in zip(paths, G.cpu().numpy())], converged.cpu().numpy()


@torch.no_grad()
def audit(exact, path, q_range, step_rad=1e-3, chunk=65536):
    """Exact minimum obstacle and self clearance (m) over the scorer's densified points of a normalized path."""
    path = torch.as_tensor(path, device=q_range.device, dtype=torch.float32); rad = path*q_range; pts = [path[:1]]
    for k in range(1, len(path)):
        m = max(1, int(math.ceil(float((rad[k]-rad[k-1]).abs().max())/step_rad)))
        pts.append(path[k-1]+(path[k]-path[k-1])*(torch.arange(1, m+1, device=path.device, dtype=path.dtype)[:, None]/m))
    pts = torch.cat(pts); env, slf = float('inf'), float('inf')
    for s in range(0, len(pts), chunk):
        e, d = exact.components(pts[s:s+chunk]); env, slf = min(env, float(e.min())), min(slf, float(d.min()))
    return env, slf


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); p.add_argument('--scene', type=Path, required=True); p.add_argument('--tests', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True); p.add_argument('--field', default='bellman'); p.add_argument('--guard', choices=['none', 'reject', 'project'], required=True)
    p.add_argument('--distance', type=Path); p.add_argument('--exact', action='store_true'); p.add_argument('--eps', type=float); p.add_argument('--alpha', type=float, default=.5)
    p.add_argument('--activate', type=float, default=.03); p.add_argument('--retries', type=int, default=4); p.add_argument('--buffer', type=float, default=0.); p.add_argument('--rk4-steps', type=int, default=400); p.add_argument('--mppi-steps', type=int, default=200); p.add_argument('--rollout-guard', action='store_true')
    p.add_argument('--planners', nargs='+', default=['rk4', 'ntrl_mppi']); p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2]); p.add_argument('--count', type=int, default=100)
    p.add_argument('--suites', nargs='*'); p.add_argument('--label', default='distance_guard'); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False); started = time.monotonic(); dev = 'cuda:0'
    manifest = json.loads((a.tests/'suite_manifest.json').read_text()); assert manifest['scene_sha256'] == sha(a.scene) and (a.smoke or not manifest['smoke'])
    for item in manifest['suites']: assert sha(a.tests/'suites'/item['file']) == item['sha256'], item['file']
    suites = [s for s in manifest['suites'] if not a.suites or s['task'] in a.suites]
    if a.smoke: suites, a.seeds, a.count = suites[:1], a.seeds[:1], 2
    count = min(a.count, *[s['n'] for s in suites])
    guard, dist_info = None, dict(mode='none')
    assert not (a.rollout_guard and a.guard == 'none'), 'the rollout guard needs a distance model'
    if a.guard != 'none':
        assert a.exact != (a.distance is not None), 'choose --distance or --exact'
        if a.exact: distance, eps, dist_info = None, 0. if a.eps is None else a.eps, dict(mode='exact')
        else:
            distance, payload = load_distance_model(a.distance, dev); eps = payload['eps'] if a.eps is None else a.eps
            dist_info = dict(mode='learned', path=str(a.distance), sha256=sha(a.distance), calibrated_eps=payload['eps'])
        guard = DistanceGuard(ArmClearance(a.scene, distance, eps, dev), a.guard, a.alpha, a.activate, a.retries, buffer=a.buffer)
    exact = ArmClearance(a.scene, None, 0., dev); q_range = exact.robot.q_range
    field = Field(a.field, a.checkpoint, dev); norm = np.load(NORM)
    report = dict(complete=False, smoke=a.smoke, label=a.label, field=a.field, checkpoint=str(a.checkpoint), checkpoint_sha256=sha(a.checkpoint), guard=a.guard, distance=dist_info,
                  eps=guard.clear.eps if guard else None, alpha=a.alpha, activate=a.activate, retries=a.retries, buffer=a.buffer, rk4_steps=a.rk4_steps, mppi_steps=a.mppi_steps, rollout_guard=a.rollout_guard,
                  scene=str(a.scene), scene_sha256=sha(a.scene), tests_manifest_sha256=sha(a.tests/'suite_manifest.json'), scorer_tilt_deg=float(TILT_OFF),
                  experiment='Baseline field planners with only the predicted-distance guard (jeffrey/distance_guard.py)', scores=[], guard_reports={})
    loaded = {item['task']: np.load(a.tests/'suites'/item['file'], allow_pickle=True) for item in suites}
    S = torch.tensor(np.concatenate([loaded[i['task']]['starts_norm'][:count] for i in suites]), device=dev)
    G = torch.tensor(np.concatenate([loaded[i['task']]['goals_norm'][:count] for i in suites]), device=dev)
    if guard: guard.set_endpoints(S, G)

    def invoke(script, args):
        sys.argv = [str(script), *map(str, args)]; runpy.run_path(str(script), run_name='__main__')

    runs = [(pl, None) for pl in a.planners if pl == 'rk4']+[(pl, s) for pl in a.planners if pl == 'ntrl_mppi' for s in a.seeds]
    for planner, seed in runs:
        if guard: guard.reset()
        begin = time.monotonic()
        paths, conv = rk4(field, S, G, guard, a.rk4_steps) if planner == 'rk4' else ntrl_mppi(field, S, G, guard, seed, a.mppi_steps, rollout_guard=a.rollout_guard)
        seconds = time.monotonic()-begin; tag = planner if seed is None else f'{planner}_seed{seed}'
        if guard: report['guard_reports'][tag] = guard.report()
        for k, item in enumerate(suites):
            task = item['task']; suite_path = a.tests/'suites'/item['file']; suite = loaded[task]; traj = np.empty(count, dtype=object); clear = []
            for i in range(count):
                path = paths[k*count+i]; rad = path*norm['q_range']+norm['q_center']; rad[0] = suite['starts_rad'][i]
                if conv[k*count+i] or planner == 'ntrl_mppi': rad[-1] = suite['goals_rad'][i]
                traj[i] = rad.astype(np.float32); clear.append(audit(exact, path, q_range))
            label = f'{a.label}_{tag}_{task}'; npz = a.output/(label+'.npz'); np.savez(npz, trajectories=traj, query_ids=np.arange(count), converged=conv[k*count:(k+1)*count])
            result = a.output/(label+'.score.json')
            invoke('/workspace/score_cross_suite.py', ['--suite', suite_path, '--scene', a.scene, '--norm_file', NORM, '--trajectories', npz, '--labels', label, '--json_out', result, '--tilt_deg', TILT_OFF])
            row = json.loads(result.read_text())[label]; assert row['n'] == count; failures = {}
            c = conv[k*count:(k+1)*count]; env = np.array([x[0] for x in clear]); slf = np.array([x[1] for x in clear])
            ok = np.zeros(count, bool); ok[[int(i) for i in row['solved_ids']]] = True
            for r in row['per_query'].values():
                for reason in r['reasons']: failures[reason] = failures.get(reason, 0)+1
            breakdown = dict(converged=int(c.sum()), not_converged=int((~c).sum()),
                             converged_failed=int((c & ~ok).sum()), converged_obstacle=int((c & (env < OBSTACLE_MARGIN)).sum()), converged_self=int((c & (slf < SELF_MARGIN)).sum()),
                             converged_obstacle_only=int((c & (env < OBSTACLE_MARGIN) & (slf >= SELF_MARGIN)).sum()), converged_self_only=int((c & (slf < SELF_MARGIN) & (env >= OBSTACLE_MARGIN)).sum()))
            report['scores'].append(dict(planner=planner, task=task, planning_seed=seed, n=count, success=row['success'], failures=failures, **breakdown,
                                         min_exact_obstacle_mm_converged=float(env[c].min()*1000) if c.any() else None, seconds=seconds/len(suites),
                                         success_ids=[int(i) for i in row['solved_ids']], cs_len_mean=row['cs_len_mean'], ee_len_mean=row['ee_len_mean']))
            write(a.output/'evaluation.json', report)
            print('GUARD_ONLY_SCORE', json.dumps({k: v for k, v in report['scores'][-1].items() if k != 'success_ids'}), flush=True)
    report['totals'] = {}
    keys = ['success', 'n', 'converged', 'not_converged', 'converged_failed', 'converged_obstacle', 'converged_self', 'converged_obstacle_only', 'converged_self_only', 'seconds']
    for pl in a.planners:
        rows = [r for r in report['scores'] if r['planner'] == pl]
        report['totals'][pl] = {k: sum(r[k] for r in rows) for k in keys}
        report['totals'][pl]['per_seed'] = None if pl == 'rk4' else [sum(r['success'] for r in rows if r['planning_seed'] == s) for s in a.seeds]
        report['totals'][pl]['per_suite'] = {s['task']: sum(r['success'] for r in rows if r['task'] == s['task']) for s in suites}
    report['complete'] = True; report['elapsed_seconds'] = time.monotonic()-started; write(a.output/'evaluation.json', report)
    lines = [f'# Predicted-distance guard only: {a.label}', '', 'SMOKE ONLY.' if a.smoke else 'Frozen certified queries; release scorer in the scene, tilt gate off.', '',
             f"Field {a.field} ({a.checkpoint}); guard {a.guard}; distance {dist_info['mode']}" + (f", eps {guard.clear.eps*1000:.1f} mm" if guard else '') + '.', '',
             '|Planner|Success|Converged|Converged but failed (obstacle / self)|Not converged|', '|---|---:|---:|---|---:|']
    for pl, t in report['totals'].items():
        lines.append(f"|{pl}|{t['success']}/{t['n']} ({100*t['success']/t['n']:.1f}%)|{t['converged']}|{t['converged_failed']} ({t['converged_obstacle']} / {t['converged_self']})|{t['not_converged']}|")
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__': main()
