"""Score the guarded field planner on a scene's frozen cabinet suites with the unchanged release scorer.

Protocol of experiments/cabinet/evaluate.py: 100 certified queries per suite, planning seeds 0/1/2 (the planner's
own random stream), release scorer in the scene with its tilt gate off (--tilt_deg 180). The planner's success
counts nothing; the scorer decides. Run inside the launch.py mounts (/clean, /workspace).

    evaluate.py --output DIR --scene S.yml --tests DIR --field bellman|pnt|euclid [--checkpoint C]
                [--distance distance.pt | --exact] [--eps E] [planner options]
"""
import sys
sys.path.insert(0, '/clean/method'); sys.path.insert(0, '/clean'); sys.path.insert(0, '/workspace')
import argparse, json, runpy, time
from pathlib import Path
import numpy as np
import torch
from common import RELEASE, sha, write
from jeffrey.fields import Field
from jeffrey.geometry import ArmClearance, load_distance_model
from jeffrey.planner import GuardedPlanner

NORM = RELEASE/'configs/joint_normalization.npz'; TILT_OFF = '180'


def planner_args(p):
    g = p.add_argument_group('planner')
    g.add_argument('--particles', type=int, default=4); g.add_argument('--iters', type=int, default=300); g.add_argument('--step', type=float, default=.01)
    g.add_argument('--candidates', type=int, default=8); g.add_argument('--soft', type=float, default=.01); g.add_argument('--lam', type=float, default=2.)
    g.add_argument('--unidirectional', action='store_true'); g.add_argument('--connect-every', type=int, default=2); g.add_argument('--connect-radius', type=float, default=.25)
    g.add_argument('--connect-nodes', type=int, default=6); g.add_argument('--stall', type=int, default=12); g.add_argument('--escape', type=int, default=10)
    g.add_argument('--no-escape', action='store_true'); g.add_argument('--shortcut', action='store_true'); g.add_argument('--kappa', type=float, default=.1); g.add_argument('--sigmas', type=float, nargs='+', default=[.3, .6, 1., 1.5])


def build_planner(a, field, clear):
    return GuardedPlanner(field, clear, particles=a.particles, iters=a.iters, step=a.step, candidates=a.candidates, sigmas=tuple(a.sigmas), soft=a.soft, lam=a.lam,
                          bidirectional=not a.unidirectional, connect_every=a.connect_every, connect_radius=a.connect_radius, connect_nodes=a.connect_nodes,
                          stall=a.iters+1 if a.no_escape else a.stall, escape=a.escape, depart_kappa=a.kappa)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); p.add_argument('--scene', type=Path, required=True); p.add_argument('--tests', type=Path, required=True)
    p.add_argument('--field', choices=['bellman', 'pnt', 'euclid'], required=True); p.add_argument('--checkpoint', type=Path)
    p.add_argument('--distance', type=Path); p.add_argument('--exact', action='store_true'); p.add_argument('--eps', type=float, help='override the calibrated eps (m)')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2]); p.add_argument('--suites', nargs='*'); p.add_argument('--count', type=int, default=100)
    p.add_argument('--label', default='guarded'); p.add_argument('--smoke', action='store_true'); planner_args(p); a = p.parse_args()
    assert a.exact != (a.distance is not None), 'choose --distance or --exact'
    a.output.mkdir(parents=True, exist_ok=False); started = time.monotonic(); dev = 'cuda:0'
    manifest = json.loads((a.tests/'suite_manifest.json').read_text())
    assert manifest['scene_sha256'] == sha(a.scene) and (a.smoke or not manifest['smoke'])
    for item in manifest['suites']: assert sha(a.tests/'suites'/item['file']) == item['sha256'], item['file']
    suites = [s for s in manifest['suites'] if not a.suites or s['task'] in a.suites]
    if a.smoke: suites, a.seeds, a.count = suites[:1], a.seeds[:1], 2
    count = min(a.count, *[s['n'] for s in suites])
    if a.exact: distance, eps, dist_info = None, 0. if a.eps is None else a.eps, dict(mode='exact')
    else:
        distance, payload = load_distance_model(a.distance, dev); eps = payload['eps'] if a.eps is None else a.eps
        dist_info = dict(mode='learned', path=str(a.distance), sha256=sha(a.distance), calibrated_eps=payload['eps'])
    clear = ArmClearance(a.scene, distance, eps, dev)
    field = Field(a.field, a.checkpoint, dev); planner = build_planner(a, field, clear)
    report = dict(complete=False, smoke=a.smoke, label=a.label, field=a.field, checkpoint=str(a.checkpoint) if a.checkpoint else None,
                  checkpoint_sha256=sha(a.checkpoint) if a.checkpoint else None, distance=dist_info, eps=eps, scene=str(a.scene), scene_sha256=sha(a.scene),
                  tests_manifest_sha256=sha(a.tests/'suite_manifest.json'), scorer_tilt_deg=float(TILT_OFF),
                  planner={k: v for k, v in vars(a).items() if k in ('particles', 'iters', 'step', 'candidates', 'soft', 'lam', 'unidirectional', 'connect_every', 'connect_radius',
                                                                    'connect_nodes', 'stall', 'escape', 'no_escape', 'kappa', 'sigmas', 'shortcut')},
                  experiment='Guarded field planner (jeffrey/planner.py) on the frozen cabinet suites', scores=[])
    norm = np.load(NORM)

    def invoke(script, args):
        sys.argv = [str(script), *map(str, args)]; runpy.run_path(str(script), run_name='__main__')

    loaded = {item['task']: np.load(a.tests/'suites'/item['file'], allow_pickle=True) for item in suites}
    for seed in a.seeds:
        # One planner batch for all suites of a seed (the loop is latency-bound); scored per suite.
        S = np.concatenate([loaded[i['task']]['starts_norm'][:count] for i in suites]); G = np.concatenate([loaded[i['task']]['goals_norm'][:count] for i in suites])
        all_paths, all_info, all_stats = planner.plan(S, G, seed=seed)
        if a.shortcut:
            begin = time.monotonic(); ids = [i for i, r in enumerate(all_info) if r['connected']]
            for i, p in zip(ids, planner.shortcut([all_paths[i] for i in ids])): all_paths[i] = p
            all_stats['shortcut_seconds'] = time.monotonic()-begin; all_stats['seconds'] += all_stats['shortcut_seconds']
        for k, item in enumerate(suites):
            task = item['task']; suite_path = a.tests/'suites'/item['file']; suite = loaded[task]
            paths, info = all_paths[k*count:(k+1)*count], all_info[k*count:(k+1)*count]
            stats = dict(all_stats, seconds=all_stats['seconds']/len(suites), batch_seconds=all_stats['seconds'], batch_queries=len(S))
            traj = np.empty(count, dtype=object); audit = []
            for i, path in enumerate(paths):
                rad = path*norm['q_range']+norm['q_center']; rad[0] = suite['starts_rad'][i]; rad[-1] = suite['goals_rad'][i]; traj[i] = rad.astype(np.float32)
                audit.append(clear.dense_margin(torch.as_tensor(path, device=dev)) if info[i]['connected'] else None)
            label = f'{a.label}_{task}_seed{seed}'; npz = a.output/(label+'.npz')
            np.savez(npz, trajectories=traj, query_ids=np.arange(count), connected=np.array([r['connected'] for r in info]))
            result = a.output/(label+'.score.json')
            invoke('/workspace/score_cross_suite.py', ['--suite', suite_path, '--scene', a.scene, '--norm_file', NORM, '--trajectories', npz, '--labels', label, '--json_out', result, '--tilt_deg', TILT_OFF])
            row = json.loads(result.read_text())[label]; assert row['n'] == count
            failures, conn_fail = {}, {}
            for i, r in row['per_query'].items():
                for reason in r['reasons']:
                    failures[reason] = failures.get(reason, 0)+1
                    if info[int(i)]['connected']: conn_fail[reason] = conn_fail.get(reason, 0)+1
            conn = [r['connected'] for r in info]
            report['scores'].append(dict(task=task, planning_seed=seed, n=count, success=row['success'], connected=int(sum(conn)),
                                         unconnected_departure_failed=int(sum(1 for r in info if not r['connected'] and 3 in r['depart'])), failures=failures, connected_failures=conn_fail,
                                         min_learned_dense_margin_mm=float(min([m for m in audit if m is not None], default=float('nan'))*1000),
                                         seconds=stats['seconds'], stats={k: v for k, v in stats.items() if k != 'seconds'}, success_ids=[int(i) for i in row['solved_ids']],
                                         cs_len_mean=row['cs_len_mean'], ee_len_mean=row['ee_len_mean']))
            write(a.output/'evaluation.json', report)
            print('GUARDED_SCORE', json.dumps({k: v for k, v in report['scores'][-1].items() if k not in ('success_ids', 'stats')}), flush=True)
    rows = report['scores']; n = sum(r['n'] for r in rows)
    report['totals'] = dict(success=sum(r['success'] for r in rows), n=n, connected=sum(r['connected'] for r in rows), seconds=sum(r['seconds'] for r in rows),
                            per_seed=[sum(r['success'] for r in rows if r['planning_seed'] == s) for s in a.seeds],
                            per_suite={s['task']: sum(r['success'] for r in rows if r['task'] == s['task']) for s in suites})
    report['complete'] = True; report['elapsed_seconds'] = time.monotonic()-started; write(a.output/'evaluation.json', report)
    t = report['totals']
    lines = [f'# Guarded field planner: {a.label}', '', 'SMOKE ONLY.' if a.smoke else 'Frozen certified queries; release scorer in the scene, tilt gate off.', '',
             f"Field {a.field} ({a.checkpoint}); distance {dist_info['mode']}, eps {eps*1000:.1f} mm; planner {json.dumps(report['planner'])}.", '',
             f"Success {t['success']}/{t['n']} ({100*t['success']/t['n']:.2f}%), connected {t['connected']}, per seed {t['per_seed']}, planning time {t['seconds']:.0f} s.", '',
             '|Suite|Success|Connected|Failures (connected paths)|', '|---|---:|---:|---|']
    for s in suites:
        rs = [r for r in rows if r['task'] == s['task']]; cf = {}
        for r in rs:
            for k, v in r['connected_failures'].items(): cf[k] = cf.get(k, 0)+v
        lines.append(f"|{s['task']}|{sum(r['success'] for r in rs)}/{sum(r['n'] for r in rs)}|{sum(r['connected'] for r in rs)}|{cf}|")
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__': main()
