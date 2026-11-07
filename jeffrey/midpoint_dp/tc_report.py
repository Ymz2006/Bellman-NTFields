"""Tables for the close-range network test (Tc replaces the exact speed in the midpoint loss; closenet.py), from
registered runs only (report.collect()).

    ~/.venvs/bellman-ntfield/bin/python jeffrey/midpoint_dp/tc_report.py [--wall]

Writes report/tc_tables.md and report/tc.json. Reference rows: control (same resumed loop, no term), the literal
recursion (network leaves) and the main recipe with exact speeds (MAIN). --wall adds the near-wall descent share
(report.wall_cache, GPU for runs not cached yet).
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import report as R  # noqa: E402

LITERAL = 'plain, net leaves, tree'
REFS = ['control', LITERAL, R.MAIN]


def tc_rows(runs, n):
    return sorted([v for (m, v) in runs if m == n and v.startswith('Tc ')])


def loop_queries(r):
    ref = r.get('refresh', [])
    if not ref or 'clearance_queries' not in ref[0]:
        return None
    return sum(x['clearance_queries'] + x.get('clearance_queries_ends', 0) for x in ref)


def diag(r):
    d = [x['tc_diag'] for x in r.get('refresh', []) if x.get('tc_diag')]
    if not d:
        return None
    med = lambda k: float(np.median([x[k] for x in d if x.get(k) is not None])) if any(x.get(k) is not None for x in d) else None
    last = d[-1]
    return dict(leaf_along_abs_dlog=med('leaf_along_abs_dlog_median'), leaf_along_near_wall=med('leaf_along_dlog_near_wall_median'),
                target_dlog_last=last['target_dlog_median'], target_abs_dlog_last=last['target_abs_dlog_median'],
                cap_tc=med('cap_binds_tc'), cap_truth=med('cap_binds_truth'),
                collided_truth=last['collided_truth'], collided_used=last['collided_used'], collided_both=last['collided_both'])


def fmt(x, nd=2):
    return '' if x is None else f'{x:.{nd}f}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wall', action='store_true')
    a = p.parse_args()
    runs = R.collect()
    tcv = sorted({v for (m, v) in runs if v.startswith('Tc ')})
    wall = R.grad_cache(runs, a.wall, extra=tcv)
    labels = R.bin_labels()
    md, data = [], {}
    for n in R.SIZES:
        rows = [v for v in REFS if (n, v) in runs] + tc_rows(runs, n)
        if not any(v.startswith('Tc ') for v in rows):
            continue
        # 1. success per training seed
        md += [f'## {n}×{n}: success', '', 'MPPI success, planning seeds 0/1/2 pooled, per training seed.', '',
               '| field | ' + ' | '.join(f'seed {s}' for s in R.SEEDS) + ' | all | % |', '|---|' + '---|' * (len(R.SEEDS) + 2)]
        for v in rows:
            rs = runs[(n, v)]
            cells = [f'{rs[s]["success_total"][0]}/{rs[s]["success_total"][1]}' if s in rs else '' for s in R.SEEDS]
            A = sum(r['success_total'][0] for r in rs.values())
            B = sum(r['success_total'][1] for r in rs.values())
            md.append(f'| {v} | ' + ' | '.join(cells) + f' | {A}/{B} | {R.pct(A, B)} |')
            data.setdefault(str(n), {})[v] = dict(success={str(s): rs[s]['success_total'] for s in rs}, ids={str(s): rs[s]['_id'] for s in rs})
        md.append('')
        # 2. field and Tc diagnostics, training seed 87358
        md += [f'## {n}×{n}: field, Tc and ground-truth use (training seed 87358)', '',
               'T/T*: median learned over true travel time per distance bin. Tc: pairs it was trained on, exact-clearance queries spent on them '
               '(sampling filter + labels) and its held-out |log error|. Loop queries: exact-clearance queries during the 30 refreshes '
               '(traversal endpoints, wall-avoiding MPPI moves, collision cut); not counted in runs before this test. '
               'Leaf error: median |log(Tc time / exact time)| over the leaves of the diagnostic refreshes (the exact time is computed '
               'for this column only); target bias: median log(target / target with exact times) at the last diagnostic refresh.', '',
               '| field | success | T/T* ' + ' / '.join(labels[:5]) + ' | Tc pairs | Tc queries (M) | Tc held-out err | loop queries (M) | leaf err | target bias | train min |',
               '|---|---|---|---|---|---|---|---|---|---|']
        for v in rows:
            r = runs[(n, v)].get(87358)
            if r is None:
                continue
            tt = ' / '.join(fmt(r['field']['by_bin'][k]['t_over_tstar_median']) if k in r['field']['by_bin'] else '' for k in labels[:5])
            tc = r.get('tc', {})
            lq = loop_queries(r)
            dg = diag(r) or {}
            md.append(f'| {v} | {r["success_total"][0]}/{r["success_total"][1]} | {tt} | {tc.get("pairs", "")} | '
                      f'{fmt(tc["clearance_queries"] / 1e6, 1) if tc else ""} | {fmt(tc.get("heldout_abs_dlog_mean"), 3) if tc else ""} | '
                      f'{fmt(lq / 1e6, 0) if lq is not None else ""} | {fmt(dg.get("leaf_along_abs_dlog"), 3)} | {fmt(dg.get("target_dlog_last"), 3)} | '
                      f'{r["train_seconds"] / 60:.0f} |')
            data[str(n)][v].update(field=r['field']['by_bin'], tc=tc, loop_queries=lq, diag=dg, train_seconds=r['train_seconds'],
                                   refresh_seconds=float(np.mean([x['seconds'] for x in r['refresh']])) if r.get('refresh') else None,
                                   wall=wall.get(r['_id']))
        md.append('')
        if wall:
            md += [f'## {n}×{n}: near-wall descent (training seed 87358)', '',
                   'Share of near-wall poses whose descent direction points into the wall, by the goal\'s true distance (%).', '',
                   '| field | 0.5-2 | 2-8 | >8 |', '|---|---|---|---|']
            for v in rows:
                r = runs[(n, v)].get(87358)
                w = wall.get(r['_id']) if r else None
                if w and 'ranges' in w:
                    md.append(f'| {v} | ' + ' | '.join(fmt(100 * w['ranges'][k]['into_wall'], 1) if k in w['ranges'] else '' for k in ('0.5-2', '2-8', '>8')) + ' |')
            md.append('')
    R.OUT.mkdir(exist_ok=True)
    (R.OUT / 'tc_tables.md').write_text('\n'.join(md) + '\n')
    (R.OUT / 'tc.json').write_text(json.dumps(data, indent=1))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
