"""Collect the LBS final benchmark from the queue's registered runs -> report/summary.json and report/tables.md.

Per scene instance: bench.json of bench:<scene>:sampling (RRT-Connect, LazyPRM) and bench:<scene>:learned (ntrl-demo, LBS).
RRT-Connect / LazyPRM: the 30 s pass (queue/state30.json, bench:<scene>:<rrtc|lazyprm>30) replaces the 10 s pass of an instance as
soon as it is in; 'limits' says which limit each cell's instances used, 'pct_within10' is the share of the 30 s runs solved within 10 s.
Per table cell (scene size x planner): the instances that are in, pooled: success % of all their queries, the per-instance success %,
time and path length mean / std / median over the successful queries; also over the queries every planner solved ('common').
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
Q = HERE / 'queue'
METHODS = ['ntrl', 'lbs', 'rrtc', 'lazyprm']
NAME = dict(ntrl='ntrl-demo', lbs='LBS', rrtc='RRT-Connect', lazyprm='LazyPRM')
HOUSES = ['Collierville', 'Benevolence', 'Onaga', 'Coffeen', 'Forkland', 'Allensville', 'Mifflinburg', 'Noxapater', 'Markleeville', 'Lakeville']
# more houses, LBS only (queue/runner_more.py, 2026-11-24): every house with a finished fast LBS bench, ntrl-demo only where LBS > 80 %
_SM = Q / 'statemore.json'
MORE_DONE = json.load(open(_SM))['done'] if _SM.exists() else {}
MORE = [k[len('fast:'):-2] for k in MORE_DONE if k.startswith('fast:')]
HOUSES = HOUSES + MORE
TABLES = [dict(key='maze', title='Maze, 2-D', cols=[(f'maze{n}', f'{n} × {n}', [f'maze{n}_{i}' for i in (1, 2, 3)]) for n in (10, 14, 18, 22)]),
          dict(key='gap', title='Gap maze, 2-D (10 × 10 base)', cols=[(f'gap{k}', f'gap ×{k} ({10 + 6 * k} cells)', [f'gap{k}_{i}' for i in (1, 2, 3)]) for k in (1, 2, 4, 8)]),
          dict(key='gibson', title='Gibson houses, 3-D', cols=[(h, h, [f'{h}3d']) for h in HOUSES])]


def stats(v):
    v = np.asarray(v, float)
    return dict(mean=float(v.mean()), std=float(v.std()), median=float(np.median(v)), n=int(len(v))) if len(v) else None


def load():
    state = json.load(open(Q / 'state.json'))
    done = state['done']
    s30 = Q / 'state30.json'
    done30 = json.load(open(s30))['done'] if s30.exists() else {}
    sf = Q / 'statefast.json'
    donefast = json.load(open(sf))['done'] if sf.exists() else {}
    inst = {}
    for t in TABLES:
        for _, _, keys in t['cols']:
            for k in keys:
                d = dict(rows={}, runs={}, limit={}, prep=done.get(f'prep:{k}'))
                for g in ('sampling', 'learned'):
                    rid = done.get(f'bench:{k}:{g}')
                    if not rid:
                        continue
                    b = json.load(open(ROOT / 'results' / f'registered_{rid}' / 'bench.json'))
                    d['truth'] = b['truth']
                    d['scene_name'] = b['scene_name']
                    for m, v in b['methods'].items():
                        d['rows'][m] = v['rows']
                        d['runs'][m] = dict(bench=rid, field=v.get('field', '').split('registered_')[-1] or None)
                        d['limit'][m] = b.get('time_limit')
                for m in ('rrtc', 'lazyprm'):
                    rid = done30.get(f'bench:{k}:{m}30')
                    if not rid:
                        continue
                    b = json.load(open(ROOT / 'results' / f'registered_{rid}' / 'bench.json'))
                    d['truth'] = b['truth']
                    d['scene_name'] = b['scene_name']
                    d['rows'][m] = b['methods'][m]['rows']
                    d['runs'][m] = dict(bench=rid, field=None)
                    d['limit'][m] = b.get('time_limit')
                if f'fast:{k}' in MORE_DONE:                              # a house of runner_more.py: LBS (fast) and, if LBS > 80 %, ntrl-demo
                    d['prep'] = MORE_DONE.get(f'prep:{k}')
                    rid = MORE_DONE[f'fast:{k}']
                    b = json.load(open(ROOT / 'results' / f'registered_{rid}' / 'bench.json'))
                    d['truth'], d['scene_name'], d['more'] = b['truth'], b['scene_name'], True
                    d['rows']['lbs'] = fast_rows(rid, True)
                    d['runs']['lbs'] = dict(bench=rid, field=MORE_DONE.get(f'lbs:{k}'), planner='lbs_fast c=ro,k=8,n=bulk, RTX 5070')
                    nb = MORE_DONE.get(f'nbench:{k}')
                    if nb:
                        b = json.load(open(ROOT / 'results' / f'registered_{nb}' / 'bench.json'))
                        d['rows']['ntrl'] = b['methods']['ntrl']['rows']
                        d['runs']['ntrl'] = dict(bench=nb, field=MORE_DONE.get(f'ntrl:{k}'))
                    inst[k] = d
                    continue
                rid = donefast.get(f'fast:{k}')
                if rid:                                            # LBS with jeffrey/lbs_fast's planner on the RTX 5070 replaces the CPU run
                    d['rows']['lbs'] = fast_rows(rid, k.endswith('3d'))
                    d['runs']['lbs'] = dict(bench=rid, field=d['runs'].get('lbs', {}).get('field'), planner='lbs_fast c=ro,k=8,n=bulk, RTX 5070')
                inst[k] = d
    return inst, state


def fast_rows(rid, house):
    """bench.json of a jeffrey/lbs_fast bench run -> rows in this study's format. time = wall time of Planner.plan (start / goal on the GPU
    to the path on the CPU). plan.score's length_cells is the x-y length; for the 3-D houses this study reports the x-y-z length, recomputed
    from the saved paths with the cells-per-unit factor E = score length / x-y length of the same path."""
    run = ROOT / 'results' / f'registered_{rid}'
    b = json.load(open(run / 'bench.json'))
    rows = []
    paths = np.load(run / 'paths_fast.npy', allow_pickle=True) if house else None
    for i, r in enumerate(b['variants']['fast']['rows']):
        L = r['length_cells']
        if house:
            x = np.asarray(paths[i], float)
            xy = np.linalg.norm(np.diff(x[:, :2], axis=0), axis=1).sum()
            if xy > 1e-9:
                L = float(np.linalg.norm(np.diff(x[:, :3], axis=0), axis=1).sum() * L / xy)
        ok = r['success'] and r['steps'] <= r['budget']                # block of 8: an arrival past the budget does not count
        rows.append(dict(success=ok, converged=r['converged'], collision=r['collision'], time=r['wall'], steps=r['steps'], budget=r['budget'],
                         length_cells=L))
    return rows


def cell(inst, keys, m):
    have = [k for k in keys if m in inst[k]['rows']]
    if not have:
        return None
    rows = [r for k in have for r in inst[k]['rows'][m]]
    ok = [r for r in rows if r['success']]
    out = dict(instances=len(have), of=len(keys), n=len(rows), success=len(ok), pct=100 * len(ok) / len(rows),
               per_instance=[100 * np.mean([r['success'] for r in inst[k]['rows'][m]]) for k in have],
               time=stats([r['time'] for r in ok]), length=stats([r['length_cells'] for r in ok]))
    ratio = []
    for k in have:
        tp = inst[k]['truth']['path_cells']
        ratio += [r['length_cells'] / tp[i] for i, r in enumerate(inst[k]['rows'][m]) if r['success'] and tp[i] > 0]
    out['length_ratio'] = stats(ratio)
    if m in ('rrtc', 'lazyprm'):
        out['raw_length'] = stats([r['raw_length_cells'] for r in ok])
        out['solve_time'] = stats([r['solve_time'] for r in ok])
        out['simplify_time'] = stats([r['simplify_time'] for r in ok])
        out['timeouts'] = int(sum(not r['found'] for r in rows))
        out['limits'] = sorted({float(inst[k]['limit'].get(m) or 0) for k in have})
        if out['limits'] == [30.0]:
            out['pct_within10'] = 100 * sum(r['success'] and r['solve_time'] <= 10 for r in rows) / len(rows)
    else:
        out['through_wall'] = int(sum(r['converged'] and r['collision'] for r in rows))
        out['not_arrived'] = int(sum(not r['converged'] for r in rows))
    common = [k for k in keys if all(mm in inst[k]['rows'] for mm in METHODS)]
    if common:
        L, T = [], []
        for k in common:
            R = inst[k]['rows']
            both = [i for i in range(len(R[m])) if all(R[mm][i]['success'] for mm in METHODS)]
            L += [R[m][i]['length_cells'] for i in both]
            T += [R[m][i]['time'] for i in both]
        out['common'] = dict(instances=len(common), length=stats(L), time=stats(T))
    return out


def main():
    inst, state = load()
    out = dict(tables=[], methods=METHODS, names=NAME, instances={}, queue=dict(done=len(state['done']), failed=state['failed']))
    for k, d in inst.items():
        out['instances'][k] = dict(prep=d['prep'], scene_name=d.get('scene_name'), runs=d['runs'], more=d.get('more', False),
                                   cells={m: cell(inst, [k], m) for m in METHODS})
    for t in TABLES:
        out['tables'].append(dict(key=t['key'], title=t['title'], cols=[dict(key=c, label=lab, instances=keys, cells={m: cell(inst, keys, m) for m in METHODS}) for c, lab, keys in t['cols']]))
    (HERE / 'report').mkdir(exist_ok=True)
    json.dump(out, open(HERE / 'report/summary.json', 'w'), indent=1)
    lines = []
    for t in out['tables']:
        lines += [f'## {t["title"]}', '', '| planner | ' + ' | '.join(c['label'] for c in t['cols']) + ' |', '|---|' + '---|' * len(t['cols'])]
        for m in METHODS:
            cs = []
            for c in t['cols']:
                x = c['cells'][m]
                cs.append('' if x is None else f'{x["pct"]:.1f} % ({x["instances"]}/{x["of"]}); t {x["time"]["mean"]:.2f}±{x["time"]["std"]:.2f} s; L {x["length"]["mean"]:.1f}±{x["length"]["std"]:.1f}'
                          if x['time'] else f'{x["pct"]:.1f} %')
            lines.append(f'| {NAME[m]} | ' + ' | '.join(cs) + ' |')
        lines.append('')
    (HERE / 'report/tables.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
