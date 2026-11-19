"""Tables of the mobile-arm study from its registered runs (queue/state.json + queue/houses.json) -> report/runs.json, report/tables.md.

    python3 jeffrey/mobile_arm/report9.py
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VARIANTS = ['ntrl', 'ctl', 'mid']
NAME = dict(ntrl='ntrl-demo (baseline)', ctl='control: resumed at 2,000 without the term', mid='ntrl-demo + midpoint loss (resumed at 2,000)')
HOUSES = ['allensville', 'lakeville', 'klickitat']
# the bare SE(2) rectangle (3 DOF), ntrl-demo, same houses and distance bins (jeffrey/lift_scale registered runs; its own queries)
SE2 = {'allensville': {'87358': '20261117T121605Z-9c558da3', '1': '20261117T143525Z-fdfc256c'},
       'lakeville': {'87358': '20261117T094420Z-31c51802', '1': '20261117T140830Z-1e68d6a5', '2': '20261117T153034Z-153e533a'},
       'klickitat': {'87358': '20261117T104445Z-0c50416c', '1': '20261117T134051Z-0075f71c', '2': '20261117T153019Z-18f6316c'}}


def load(rid):
    d = ROOT / 'results' / f'registered_{rid}'
    r = json.loads((d / 'results.json').read_text()) if (d / 'results.json').exists() else None
    return d, r


def summarize(rid):
    d, r = load(rid)
    if r is None or not r.get('complete'):
        return None
    rows = np.load(d / 'rows.npz', allow_pickle=True)
    succ, conv, coll = rows['success'], rows['converged'], rows['collision']
    n = succ.size
    out = dict(run=rid, solved=100 * succ.mean(), solved_n=[int(succ.sum()), int(n)], per_seed=[int(x) for x in succ.sum(1)],
               through=100 * (~succ & conv & coll).sum() / n, stall=100 * (~conv).sum() / n, parts=r['arrived_through_parts'],
               time_over_ref=r.get('time_over_ref_median'), time_over_ref_by_bin=r.get('time_over_ref_by_bin'),
               by_bin={k: 100 * v[0] / v[1] for k, v in r['success_by_bin'].items()}, by_bin_n=r['success_by_bin'],
               field={k: r['field'][k] for k in ('t_over_ref_median', 't_over_se2_median', 'below_lower_bound', 'above_upper_bound', 'spearman_ref')},
               field_by_bin=r['field']['by_bin'], speed_fit=r['field']['speed_fit'],
               validate=[[v['epoch'], 100 * v['success'] / v['runs'], v['arrived_through'], v['not_arrived']] for v in r.get('validate', [])],
               train_minutes=r['train_seconds'] / 60, minutes=r['elapsed_seconds'] / 60, rollbacks=r['rollbacks'], final_loss=r['final_loss'], gpu=r['gpu'])
    ref = r.get('refresh') or []
    if ref:
        out['refresh'] = [[s['epoch'], 100 * s['reached'] / s['traversals'], 100 * s['stalled'] / s['traversals'], 100 * s['collided'] / s['traversals'],
                           s['nodes'], s['log_net_over_target_median'], s['seconds']] for s in ref]
        out['refresh_seconds_mean'] = float(np.mean([s['seconds'] for s in ref]))
    return out


def main():
    q = HERE / 'queue'
    houses = json.loads((q / 'houses.json').read_text())
    state = json.loads((q / 'state.json').read_text())
    res = dict(houses={}, runs={}, names=NAME, variants=VARIANTS)
    for key in HOUSES:
        prep = ROOT / houses[key]
        pr = json.loads((prep / 'results.json').read_text())
        mz = json.loads((prep / 'maze.json').read_text())
        z = np.load(prep / 'queries.npz')
        res['houses'][key] = dict(name=mz['scene']['house']['name'], E=mz['E_cells'], prepare=prep.name.replace('registered_', ''), house_prepare=Path(mz['house_prepare']).name.replace('registered_', ''),
                                  queries=int(len(z['bin'])), per_bin=pr['queries']['per_bin'], straight=100 * pr['straight']['success'] / len(z['bin']),
                                  straight_by_bin={k: 100 * v[0] / v[1] for k, v in pr['straight']['by_bin'].items()}, straight_parts=pr['straight']['contact_parts'],
                                  reference=pr['reference'], data=pr['data'], tuck_orders=pr['queries']['tuck_orders'], untuck_orders=pr['queries']['untuck_orders'])
        res['runs'][key] = {}
        for job, rid in state['done'].items():
            label, hk, seed, extra = job.split('|')
            if hk != key or label not in VARIANTS:
                continue
            s = summarize(rid)
            if s is None:
                continue
            res['runs'][key].setdefault(label, {})[seed] = s
    # pooled over seeds
    pooled = {}
    for key, by in res['runs'].items():
        pooled[key] = {}
        # compare on paired seeds only: the seeds whose baseline, control and midpoint runs are all finished
        paired = set.intersection(*[set(by.get(v, {})) for v in VARIANTS]) if all(v in by for v in VARIANTS) else set()
        res.setdefault('paired_seeds', {})[key] = sorted(paired)
        for v, seeds in by.items():
            S = [seeds[sd] for sd in sorted(seeds) if sd in paired] if paired else list(seeds.values())
            n = sum(s['solved_n'][1] for s in S)
            pooled[key][v] = dict(seeds=sorted(sd for sd in seeds if not paired or sd in paired), solved=100 * sum(s['solved_n'][0] for s in S) / n, solved_seed=[s['solved'] for s in S],
                                  through=float(np.mean([s['through'] for s in S])), stall=float(np.mean([s['stall'] for s in S])),
                                  by_bin={b: float(np.mean([s['by_bin'][b] for s in S])) for b in S[0]['by_bin']},
                                  t_over_ref=float(np.mean([s['field']['t_over_ref_median'] for s in S])), t_over_se2=float(np.mean([s['field']['t_over_se2_median'] for s in S])),
                                  below_lower=float(np.mean([s['field']['below_lower_bound'] for s in S])), spearman=float(np.mean([s['field']['spearman_ref'] for s in S])),
                                  time_over_ref=float(np.mean([s['time_over_ref'] for s in S if s['time_over_ref'] is not None])) if any(s['time_over_ref'] is not None for s in S) else None)
    res['pooled'] = pooled
    se2 = {}
    for key, runs in SE2.items():
        vals = {}
        for seed, rid in runs.items():
            r = json.loads((ROOT / 'results' / f'registered_{rid}' / 'results.json').read_text())
            vals[seed] = dict(run=rid, solved=100 * r['success_total'][0] / r['success_total'][1])
        se2[key] = dict(seeds=vals, solved=float(np.mean([v['solved'] for v in vals.values()])))
    res['se2'] = se2
    (HERE / 'report').mkdir(exist_ok=True)
    (HERE / 'report/runs.json').write_text(json.dumps(res, indent=1, default=float))
    # tables.md
    L = ['# Mobile arm (9 DOF) in Gibson houses: tables', '', 'Generated by report9.py from the registered runs in queue/state.json.', '']
    L += ['## Houses and queries', '', '| house | E (cells) | prepare run | house prepare (SE(2) truth) | queries | straight segment solved % | T_ref / T*_SE2 (median) | arm share of T_ref (median) |',
          '|---|---|---|---|---|---|---|---|']
    for key, h in res['houses'].items():
        L.append(f"| {h['name']} | {h['E']:g} | {h['prepare']} | {h['house_prepare']} | {h['queries']} | {h['straight']:.1f} | {h['reference']['t_ref_over_t_se2_median']:.2f} | {h['reference']['tuck_share_of_t_ref_median']:.2f} |")
    L += ['', '## Planning (MPPI planning seeds 0/1/2, exact scoring)', '',
          '| house | variant | seed | solved % | per planning seed | arrived through % | did not arrive % | first contact (base / arm-wall / floor / base box / self) | path time / T_ref | T/T_ref | T/T*_SE2 | T below T*_SE2 | Spearman(T, T_ref) | log-speed corr | run |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for key in HOUSES:
        for v in VARIANTS:
            for seed, s in sorted(res['runs'][key].get(v, {}).items()):
                p = s['parts']
                L.append(f"| {res['houses'][key]['name']} | {v} | {seed} | {s['solved']:.1f} | {'/'.join(map(str, s['per_seed']))} of {s['solved_n'][1] // 3} | {s['through']:.1f} | {s['stall']:.1f} | "
                         f"{p['rect']} / {p['arm']} / {p['floor']} / {p['base']} / {p['self']} | {s['time_over_ref'] if s['time_over_ref'] is None else round(s['time_over_ref'], 3)} | "
                         f"{s['field']['t_over_ref_median']:.3f} | {s['field']['t_over_se2_median']:.3f} | {s['field']['below_lower_bound']:.3f} | {s['field']['spearman_ref']:.3f} | "
                         f"{s['speed_fit']['log_speed_corr']:.3f} | {s['run']} |")
    L += ['', '## Pooled over seeds', '', '| house | variant | seeds | solved % | per seed | arrived through % | did not arrive % |', '|---|---|---|---|---|---|---|']
    for key in HOUSES:
        for v in VARIANTS:
            if v in pooled.get(key, {}):
                p = pooled[key][v]
                L.append(f"| {res['houses'][key]['name']} | {v} | {len(p['seeds'])} | {p['solved']:.1f} | {', '.join(f'{x:.1f}' for x in p['solved_seed'])} | {p['through']:.1f} | {p['stall']:.1f} |")
    L += ['', '## Context: the bare SE(2) rectangle (3 DOF), ntrl-demo, same houses (jeffrey/lift_scale runs, their own queries)', '',
          '| house | solved % (mean over seeds) | per seed (run) |', '|---|---|---|']
    for key in HOUSES:
        x = se2[key]
        L.append(f"| {res['houses'][key]['name']} | {x['solved']:.1f} | " + ', '.join(f"{sd}: {v['solved']:.1f} ({v['run']})" for sd, v in x['seeds'].items()) + ' |')
    L += ['', '## Solved % by base distance bin (cells), pooled over seeds', '']
    for key in HOUSES:
        if not pooled.get(key):
            continue
        bins = list(next(iter(pooled[key].values()))['by_bin'])
        L += [f"### {res['houses'][key]['name']}", '', '| variant | ' + ' | '.join(bins) + ' |', '|---|' + '---|' * len(bins)]
        L.append('| straight segment | ' + ' | '.join(f"{res['houses'][key]['straight_by_bin'].get(b, float('nan')):.0f}" for b in bins) + ' |')
        for v in VARIANTS:
            if v in pooled[key]:
                L.append(f'| {v} | ' + ' | '.join(f"{pooled[key][v]['by_bin'][b]:.0f}" for b in bins) + ' |')
        L.append('')
    L += ['## Speed fit on held-out pairs by the constraint that sets the label', '', '| house | variant | seed | base vs walls | arm vs walls | internal (floor / base / self) |', '|---|---|---|---|---|---|']
    for key in HOUSES:
        for v in VARIANTS:
            for seed, s in sorted(res['runs'][key].get(v, {}).items()):
                bc = s['speed_fit']['by_constraint']
                cell = lambda k: f"corr {bc[k]['log_speed_corr']:.2f}, pred/true {bc[k]['pred_over_true_median']:.2f}" if k in bc else '–'
                L.append(f"| {res['houses'][key]['name']} | {v} | {seed} | {cell('base_rect')} | {cell('arm_walls')} | {cell('internal')} |")
    L += ['', '## During training (planning seed 0) and midpoint refreshes', '']
    for key in HOUSES:
        for v in VARIANTS:
            for seed, s in sorted(res['runs'][key].get(v, {}).items()):
                val = ', '.join(f'{e}: {p:.1f}' for e, p, _, _ in s['validate'])
                L.append(f"- {res['houses'][key]['name']} {v} seed {seed}: validate {val}; training {s['train_minutes']:.1f} min, total {s['minutes']:.1f} min, rollbacks {s['rollbacks']}, {s['gpu']}")
                if s.get('refresh'):
                    rr = s['refresh']
                    L.append(f"  - refreshes: {len(rr)}, mean {s['refresh_seconds_mean']:.0f} s; drives reaching their goal (epoch: %) " +
                             ', '.join(f'{e}: {a:.0f}' for e, a, *_ in rr[::5]) + '; median log(T/target) ' + ', '.join(f'{e}: {x:.2f}' for e, _, _, _, _, x, _ in rr[::5] if x is not None))
    (HERE / 'report/tables.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
