"""Collect the registered runs of the arm-placement study into report/summary.json and report/tables.md.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/arm_placement/report.py

Reads queue/jobs.jsonl + queue/state.json (job -> registered run) and each run's own files. A placement row holds its reach, its suite
generation, its prepared data, the Euclidean-field control, the straight line, and every field run (label, training seed).
"""
import json
import re
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'report'
TASKS = ['free_to_free', 'free_to_compartment', 'compartment_to_free', 'compartment_to_compartment', 'same_compartment', 'anywhere']
HARD = ['compartment_to_compartment', 'same_compartment']
BASE = dict(dx=0.25, s=0.30, h=0.60)


def run_dir(rid):
    return ROOT / f'results/registered_{rid}'


def load(rid, name='results.json'):
    f = run_dir(rid) / name
    return json.loads(f.read_text()) if f.exists() else None


def placement_of(pl):
    if pl == 'base':
        return dict(BASE)
    m = re.match(r'p_d(\d+)_s(\d+)_h(\d+)', pl)
    return dict(dx=int(m.group(1)) / 100, s=int(m.group(2)) / 100, h=int(m.group(3)) / 100)


def field_row(rid, label, seed):
    r = load(rid)
    if r is None:
        return None
    n = r['success'][1]
    per = n // len(r['plan']) // len(TASKS)
    plan = r['plan']
    tot = lambda k: int(sum(p[k] for p in plan.values()))  # noqa: E731
    return dict(run=rid, label=label, seed=seed, success=r['success'], solved=100 * r['success'][0] / n, by_task=r['success_by_task'], per_task_runs=per * len(plan),
                hard=[int(sum(r['success_by_task'][t] for t in HARD)), per * len(plan) * len(HARD)], detour=r['detour'], line=[tot_pair(plan, 'line_feasible', 0), tot_pair(plan, 'line_feasible', 1)],
                not_arrived=tot('not_arrived'), arrived_contact=tot('arrived_contact'), contact_self=tot('contact_self'), contact_obstacle=tot('contact_obstacle'),
                per_seed=[p['success'] for p in plan.values()], by_ends={k: [int(sum(p['by_ends_in_cupboard'][k][0] for p in plan.values())), int(sum(p['by_ends_in_cupboard'][k][1] for p in plan.values()))] for k in ('0', '1', '2')},
                validate=[[v['epoch'], v['success'], v['runs'], sum(v['by_task'][t] for t in HARD)] for v in r.get('validate', [])], train_min=r.get('train_only_seconds', 0) / 60, total_min=r['elapsed_seconds'] / 60,
                final_loss=r.get('final_loss'), args=r['args'], T_query_median=r['field']['T_test_queries_median'])


def tot_pair(plan, key, i):
    return int(sum(p[key][i] for p in plan.values()))


def main():
    OUT.mkdir(exist_ok=True)
    state = json.loads((HERE / 'queue/state.json').read_text())
    jobs = [json.loads(line) for line in (HERE / 'queue/jobs.jsonl').read_text().splitlines() if line.strip() and not line.startswith('#')]
    reach = {}
    for f in sorted(OUT.glob('reach*.json')):
        for row in json.loads(f.read_text())['placements']:
            reach[(round(row['dx'], 3), round(row['s'], 3), round(row['h'], 3))] = row
    P, pending, extra, plans, hard = {}, [], [], [], {}
    manual = json.loads((HERE / 'queue/manual.json').read_text()) if (HERE / 'queue/manual.json').exists() else []
    for m in manual:            # runs registered by hand (not through the queue)
        state[m['id']] = dict(status='done', run=m['run'])
    for j in jobs + [dict(id=m['id'], mode=m['mode']) for m in manual]:
        s = state.get(j['id'], dict(status='pending'))
        if s['status'] != 'done':
            pending.append(dict(id=j['id'], status=s['status']))
            continue
        rid = s['run']
        mode = j['mode']
        if j['id'].startswith(('tests_hard_', 'prep_hard_')):
            g = load(rid, 'tests/generation.json') if mode == 'tests' else load(rid)
            hard.setdefault(j['id'].split('_hard_')[1], {})[mode] = dict(run=rid, **(dict(strata={t: dict(candidates=v['candidates'], certified=v.get('certified', 0), direct=v['certified_direct']) for t, v in g['strata'].items()}, minutes=g['timing']['total'] / 60) if mode == 'tests'
                                                                                else dict(queries=g['queries']['n'], straight_line=g['queries']['straight_line_here'], witness_sources=g['queries']['witness_sources'], dist_quantiles=g['queries']['dist_quantiles'])))
            continue
        if mode == 'evalfield':
            r = load(rid)
            if r is None:
                continue
            m = re.match(r'(.+?)_(lift|ntrl|[a-z0-9]+)_(base|p_d\d+_s\d+_h\d+)(?:_s(\d+))?$', j['id'])
            fl = np.load(run_dir(rid) / 'flags.npz') if (run_dir(rid) / 'flags.npz').exists() else None
            for e in r['runs']:
                key = f"{e['field']}|{e['planner'].get('kind', 'forward')}|{e['planner']['steps']}|{e['planner']['sample_num']}|{e['planner']['horizon']}|"
                ok = np.stack([fl[key + s_] for s_ in e['plan']]) if fl is not None and key + '0' in fl else None
                plans.append(dict(job=j['id'], suite='hard' if j['id'].startswith('hard_') else 'placement', run=rid, placement=m.group(3) if m else None, field=e['field'], seed=int(m.group(4)) if m and m.group(4) else None, field_run=Path(e['run']).name.replace('registered_', ''),
                                  kind=e['planner'].get('kind', 'forward'), steps=e['planner']['steps'], samples=e['planner']['sample_num'], horizon=e['planner']['horizon'],
                                  success=e['success'], solved=100 * e['success'][0] / e['success'][1], by_task=e['success_by_task'], not_arrived=e.get('not_arrived'), arrived_contact=e.get('arrived_contact'),
                                  per_seed=[p_['success'] for p_ in e['plan'].values()], queries=None if ok is None else int(ok.shape[1]),
                                  any_seed=None if ok is None else int(ok.any(0).sum()), all_seeds=None if ok is None else int(ok.all(0).sum())))
            continue
        if mode in ('reach', 'demo'):
            extra.append(dict(id=j['id'], mode=mode, run=rid))
            if mode == 'reach':
                for row in load(rid, 'reach.json')['placements']:
                    reach[(round(row['dx'], 3), round(row['s'], 3), round(row['h'], 3))] = dict(row, run=rid)
            continue
        if mode in ('tests', 'prepare', 'euclid'):
            pl = j['id'].split('_', 1)[1]
        else:
            m = re.match(r'(.+?)_(base|p_d\d+_s\d+_h\d+)_s(\d+)$', j['id'])
            label, pl, seed = m.group(1), m.group(2), int(m.group(3))
        row = P.setdefault(pl, dict(name=pl, **placement_of(pl), suite='frozen 600 (lbs_arm, 10,000 RRT iterations)' if pl == 'base' else 'own', fields=[]))
        if mode == 'tests':
            g = load(rid, 'tests/generation.json')
            st = g['strata']
            row['tests'] = dict(run=rid, minutes=g['timing']['total'] / 60, zones_reached=g.get('zones_reached'), endpoint_accept=g['endpoint_accept'], per_stratum=g['parameters']['per_stratum'],
                                rrt_iterations=g['parameters']['rrt_iterations'], rounds=len(g['rounds']),
                                strata={t: dict(candidates=v['candidates'], certified=v.get('certified', 0), direct=v['certified_direct'], one_arm=v['certified_one_arm_at_a_time'],
                                                share=v.get('certified', 0) / v['candidates'], selected_direct=v['selected_direct'], distance_median=v['selected_distance_rad_quantiles'][2],
                                                witness_length_median=v['selected_witness_length_rad_median']) for t, v in st.items()},
                                certified_share=sum(v.get('certified', 0) for v in st.values()) / sum(v['candidates'] for v in st.values()),
                                first_round={t: None for t in st})
            # certification rate of the first round only (every stratum's candidates are an unbiased draw there)
            fr = {}
            for t in st:
                z = np.load(run_dir(rid) / f'tests/work/candidates_{t}.npz')
                m0 = z['round'] == 0
                fr[t] = [int((z['status'][m0] == 'certified').sum()), int(m0.sum()), int(z['direct'][m0].sum())]
            row['tests']['first_round'] = fr
        elif mode == 'prepare':
            r = load(rid)
            q = r['queries']
            row['prepare'] = dict(run=rid, minutes=r['seconds'] / 60, E=r['E_cells'], valid_of_in_box=r['occupancy']['valid_of_in_box'], in_box_of_uniform=r['occupancy']['in_box_of_uniform'],
                                  full_speed_of_valid=r['occupancy']['full_speed_of_valid'], queries=q['n'], straight_line=q['straight_line_here'], straight_line_by_task=q['straight_line_by_task'],
                                  witness_sources=q['witness_sources'], dist_quantiles=q['dist_quantiles'], ends_in_cupboard=q['ends_in_cupboard'], x1_full_speed=r['data']['x1_full_speed'],
                                  x0_all_arms_in_cupboard=r['data']['x0_all_arms_in_cupboard'], candidates_drawn=r['data']['candidates_drawn'], planner=r['planner'])
        elif mode == 'euclid':
            r = load(rid)
            e = r['runs'][0]
            names = list(e['plan']['0']['by_task'])
            row['euclid'] = dict(run=rid, success=e['success'], solved=100 * e['success'][0] / e['success'][1], by_task={t: int(sum(p['by_task'][t] for p in e['plan'].values())) for t in names},
                                 not_arrived=int(sum(p['not_arrived'] for p in e['plan'].values())), detour=[tot_pair(e['plan'], 'detour', 0), tot_pair(e['plan'], 'detour', 1)])
        elif mode == 'field':
            fr = field_row(rid, label, seed)
            if fr:
                row['fields'].append(fr)
    by_run = {}
    for q in plans:
        if q['suite'] == 'placement':
            by_run.setdefault(q['field_run'], []).append(q)
    for row in P.values():
        for f in row['fields']:
            f['planners'] = sorted(by_run.get(f['run'], []), key=lambda q: (q['kind'], q['steps'], q['samples']))
    for pl, row in P.items():
        row['reach'] = reach.get((round(row['dx'], 3), round(row['s'], 3), round(row['h'], 3)))
    rows = sorted(P.values(), key=lambda r: (r['name'] != 'base', r['h'], r['s'], r['dx']))
    summary = dict(placements=rows, pending=pending, extra=extra, plans=plans, hard_sets=hard, reach_grid=list(reach.values()), tasks=TASKS, hard=HARD)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, default=float))
    L = ['| placement | dx | s | h | reach union / both | straight line | Euclid | ' + ' | '.join(sorted({f["label"] for r in rows for f in r["fields"]})) + ' |', '|---|---|---|---|---|---|---|' + '---|' * len({f["label"] for r in rows for f in r["fields"]})]
    labels = sorted({f['label'] for r in rows for f in r['fields']})
    for r in rows:
        rc = r.get('reach')
        cell = lambda lab: '; '.join(f'{f["success"][0]}/{f["success"][1]} = {f["solved"]:.1f}% (hard {f["hard"][0]}/{f["hard"][1]}) `{f["run"][-8:]}`' for f in r['fields'] if f['label'] == lab) or '-'  # noqa: E731
        pr = r.get('prepare')
        L.append(f'| {r["name"]} | {r["dx"]} | {r["s"]} | {r["h"]} | ' + (f'{100 * rc["union_mean"]:.0f} / {100 * rc["both_mean"]:.0f} %' if rc else '-') + ' | '
                 + (f'{pr["straight_line"]}/{pr["queries"]}' if pr else '-') + ' | ' + (f'{r["euclid"]["success"][0]}/{r["euclid"]["success"][1]}' if 'euclid' in r else '-') + ' | ' + ' | '.join(cell(lab) for lab in labels) + ' |')
    L += ['', '| placement | field | seed | planner | steps | samples | solved | f2f | f2c | c2f | c2c | same | any | run |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        for f in r['fields']:
            L.append(f'| {r["name"]} | {f["label"]} | {f["seed"]} | forward (field run) | 600 | 50 | {f["success"][0]}/{f["success"][1]} = {f["solved"]:.1f}% | ' + ' | '.join(str(f['by_task'][t]) for t in TASKS) + f' | `{f["run"][-8:]}` |')
            for q in f.get('planners', []):
                L.append(f'| {r["name"]} | {f["label"]} | {f["seed"]} | {q["kind"]} | {q["steps"]} | {q["samples"]} | {q["success"][0]}/{q["success"][1]} = {q["solved"]:.1f}% | ' + ' | '.join(str(q['by_task'][t]) for t in TASKS) + f' | `{q["run"][-8:]}` |')
    (OUT / 'tables.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'{len(rows)} placements, {sum(len(r["fields"]) for r in rows)} field runs, {len(pending)} jobs not done')


if __name__ == '__main__':
    main()
