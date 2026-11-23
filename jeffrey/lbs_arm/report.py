"""Collect the LBS arm study's runs and the registered baselines into report/runs.json (and a short table on stdout).

    ~/.venvs/bellman-ntfield/bin/python jeffrey/lbs_arm/report.py

Own runs: queue/state.json (+ queue/manual.json), each {id: field run, eval: official evaluation run}. Baselines: the registered
ntrl-demo and P-NTFields cabinet runs (their evaluation.json), same suites, same scorer. Queries are split by the suites' own flag
"the straight line passes the scorer" into line-feasible and detour queries.
"""
import json
import os
import re
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = ROOT / 'results'
TESTS = ROOT / 'experiments/cabinet/tests'
REG = Path(os.environ.get('NTFIELDS_REGISTRY', Path.home() / '.local/share/ntfields-experiments'))
BASE = [('ntrl_best', 'ntrl-demo, best earlier setting', '20261026T041234Z-664b5d50'), ('ntrl_base', 'ntrl-demo, baseline recipe', '20261025T065945Z-3c53b360'),
        ('pnt', 'P-NTFields', '20261026T080627Z-53c1dd59')]
TASKS = ['free_to_free', 'free_to_compartment', 'compartment_to_free', 'compartment_to_compartment', 'same_compartment', 'anywhere']


def suites(tests=TESTS):
    man = json.loads((Path(tests) / 'suite_manifest.json').read_text())
    return {i['task']: np.load(Path(tests) / 'suites' / i['file'], allow_pickle=True)['direct_line_strict_feasible'].astype(bool) for i in man['suites']}


LINE = suites()
_LINES = {}


def scene_of(prepared):
    """(scene name, straight-line flags of its frozen suites) of a prepared run."""
    info = json.loads((Path(prepared) / 'scene.json').read_text())
    name = info['tests_scene']
    if name not in _LINES:
        _LINES[name] = LINE if name == 'cabinet' else suites(info['tests'])
    return name, _LINES[name]


def official(run, LINE=LINE):
    """Per planner: total, n, per task, detour / line split, failure reasons."""
    ev = json.loads((RES / f'registered_{run}' / 'evaluation/evaluation.json').read_text())
    out = {}
    for m in sorted(set(s['method'] for s in ev['scores'])):
        rows = [s for s in ev['scores'] if s['method'] == m]
        per = {t: sum(s['success'] for s in rows if s['task'] == t) for t in TASKS}
        det = sum(int((~LINE[s['task']][np.array(s['success_ids'], int)]).sum()) if s['success_ids'] else 0 for s in rows)
        runs = len(rows) // len(TASKS)
        fails = {}
        for s in rows:
            for k, v in s['failures'].items():
                fails[k] = fails.get(k, 0) + v
        tot = sum(per.values())
        n_det = sum(int((~LINE[t]).sum()) for t in TASKS) * runs
        out[m] = dict(success=tot, n=600 * runs, per_task=per, per_task_n=100 * runs, detour=[det, n_det], line=[tot - det, 600 * runs - n_det], failures=fails,
                      per_seed=ev.get('totals', {}).get(m, {}).get('per_seed'))
    return out, ev.get('ntrl_mppi_settings')


EUCLID = {'real_world_12dof': '20261121T211502Z-0611c599'}


def progress(job):
    log = HERE / 'queue/logs' / (''.join(c if c.isalnum() or c in '-_.' else '_' for c in job)[:180] + '.log')
    return None


def main():
    out = dict(tasks=TASKS, line_feasible={t: int(LINE[t].sum()) for t in TASKS}, baselines=[], runs=[], pending=[])
    for key, name, rid in BASE:
        o, st = official(rid)
        out['baselines'].append(dict(key=key, name=name, id=rid, official=o, planner=st))
    state = json.loads((HERE / 'queue/state.json').read_text()) if (HERE / 'queue/state.json').exists() else dict(done={}, failed={})
    done = dict(json.loads((HERE / 'queue/manual.json').read_text()))
    done.update({k: v for k, v in state['done'].items() if isinstance(v, dict)})
    for job, ids in done.items():
        label, _, seed, extra = job.split('|')[:4]
        fr = json.loads((RES / f'registered_{ids["id"]}' / 'results.json').read_text())
        sname, sline = scene_of(fr['prepared'])
        row = dict(label=label, scene=sname, prepared=Path(fr['prepared']).name[-8:], seed=int(seed), extra=extra, id=ids['id'], eval=ids.get('eval'), arm=fr['arm'], lift=fr['lift'], freq=fr['config']['freq'], pair_cup=fr['config'].get('pair_cup', 0.0), min_clear=fr['config']['min_clear'],
                   data_cup=json.loads((Path(fr['prepared']) / 'results.json').read_text())['data'].get('cup_share_arg', 0.0),
                   own=dict(success=fr['success'], by_task=fr['success_by_task'], detour=fr['detour'], plan=fr['plan']),
                   course=[dict(epoch=v['epoch'], success=v['success'], runs=v['runs'], detour=v['detour'], by_task=v['by_task'], arrived_contact=v['arrived_contact'], not_arrived=v['not_arrived'],
                                minutes=v['train_seconds'] / 60) for v in fr['validate']],
                   refresh=[dict(epoch=r['epoch'], reached=r['reached'], collided=r['collided'], stalled=r['stalled'], traversals=r['traversals']) for r in fr['refresh']],
                   train_min=fr['train_only_seconds'] / 60, minutes=fr['elapsed_seconds'] / 60, gpu=fr['gpu'], planner=fr['planner'], field=fr.get('field'))
        if ids.get('eval'):
            row['official'], _ = official(ids['eval'], sline)
        out['runs'].append(row)
    jobs = [l.strip() for l in (HERE / 'queue/jobs.txt').read_text().splitlines() if l.strip() and not l.startswith('#')]
    for job in jobs:
        if job in done:
            continue
        label, _, seed, extra = job.split('|')[:4]
        ep = None
        for d in sorted(RES.glob('registered_2026112*'), reverse=True)[:12]:
            lg = d / 'log.txt'
            spec = REG / 'runs' / d.name[len('registered_'):] / 'spec.json'
            if lg.exists() and not (d / 'results.json').exists() and spec.exists():
                cfg = json.loads(spec.read_text()).get('config', {})
                if cfg.get('action') == 'lbsarm-field' and cfg.get('label') == label and not cfg.get('smoke'):
                    m = re.findall(r'epoch (\d+): ntrl loss', lg.read_text())
                    ep = int(m[-1]) if m else 0
                    break
        out['pending'].append(dict(label=label, seed=int(seed), extra=extra, failed=job in state.get('failed', {}), epoch=ep))
    for d in sorted(HERE.glob('tests_*')):
        if d.name[len('tests_'):] not in _LINES and (d / 'suite_manifest.json').exists():
            _LINES[d.name[len('tests_'):]] = suites(d)
    out['scenes'] = {k: dict(line_feasible={t: int(v[t].sum()) for t in TASKS}) for k, v in _LINES.items()}
    out['close'] = {}       # closepairs.py runs (any from 2026-11-22 on): scene -> field name -> rows by step length; a later run replaces a field of the same name
    for d in sorted(RES.glob('registered_2026112[2-9]*')) + sorted(RES.glob('registered_202612*')):
        fr = d / 'results.json'
        if not fr.exists():
            continue
        e = json.loads(fr.read_text())
        if e.get('stage') == 'lbsarm-belldiag' and not e.get('smoke'):
            out.setdefault('belldiag', {})[scene_of(e['prepared'])[0]] = dict(run=d.name[-8:], radius_cells=e['radius_cells'], fields=e['fields'])
            continue
        if e.get('stage') != 'lbsarm-closepairs' or e.get('smoke'):
            continue
        sn = scene_of(e['prepared'])[0]
        for row in e['lengths']:
            for name, v in row.get('fields', {}).items():
                out['close'].setdefault(sn, {}).setdefault(name, {'run': d.name[-8:], 'field_run': Path(e['fields'][name]).name[-8:], 'rows': {}})
                out['close'][sn][name]['run'] = d.name[-8:]
                out['close'][sn][name]['rows'][str(row['cells'])] = dict(n=row['pairs'], length=row['length'], ratio=v['all']['ratio_quantiles'], within=v['all']['within_20pct'],
                                                                         groups={k: dict(n=v[k]['n'], ratio=v[k]['ratio_quantiles'][1]) for k in ('full_speed', 'slowed', 'slow') if k in v})
    out['euclid'] = {}      # the planner on T = |x - g| (euclid.py): first setting = the field runs' planner settings
    for k, rid in EUCLID.items():
        fe = RES / f'registered_{rid}' / 'results.json'
        if fe.exists():
            e = json.loads(fe.read_text())
            r0 = e['runs'][0]
            pl = r0['plan']
            out['euclid'][k] = dict(id=rid, planner=r0['planner'], success=r0['success'], by_task={t: sum(v['by_task'][t] for v in pl.values()) for t in TASKS},
                                    detour=[sum(v['detour'][0] for v in pl.values()), sum(v['detour'][1] for v in pl.values())], line=[sum(v['line_feasible'][0] for v in pl.values()), sum(v['line_feasible'][1] for v in pl.values())],
                                    not_arrived=sum(v['not_arrived'] for v in pl.values()), arrived_contact=sum(v['arrived_contact'] for v in pl.values()),
                                    settings=[dict(steps=r['planner']['steps'], samples=r['planner']['sample_num'], success=r['success']) for r in e['runs']])
    (HERE / 'report').mkdir(exist_ok=True)
    (HERE / 'report/runs.json').write_text(json.dumps(out, indent=1, default=float))
    pct = lambda a, b: f'{a}/{b} ({100 * a / b:.1f} %)'  # noqa: E731
    print('straight line', sum(out['line_feasible'].values()), '/ 600')
    for b in out['baselines']:
        o = b['official']['ntrl_mppi']
        print(f"{b['name']:34s} {pct(o['success'], o['n'])} detour {pct(*o['detour'])} c2c {o['per_task']['compartment_to_compartment']} rk4 {b['official']['native_rk4']['success']}")
    for r in out['runs']:
        if 'official' in r:
            o = r['official']['ntrl_mppi']
            print(f"{r['label']:10s} {'' if r['scene'] == 'cabinet' else r['scene'] + ' '}gain {r['lift']['gain']} freq {r['freq'][0]:g}.. {pct(o['success'], o['n'])} detour {pct(*o['detour'])} per task {list(o['per_task'].values())} rk4 {r['official']['native_rk4']['success']} "
                  f"fail {o['failures']} own {r['own']['success'][0]} train {r['train_min']:.0f} min")
        else:
            print(r['label'], 'own scorer', r['own']['success'], 'no official evaluation')
    print('pending', [(p['label'], p['epoch'], p['failed']) for p in out['pending']])


if __name__ == '__main__':
    main()
