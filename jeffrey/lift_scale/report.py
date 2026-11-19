"""Collect every finished lift-scale run of the queue (queue/state.json: job -> registered id) into report/runs.json and report/tables.md.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/lift_scale/report.py
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'report'
SCENES = ['plus22', 'plus30', 'plus40', 'apt32', 'apt64', 'Allensville', 'Lakeville', 'Klickitat', 'plus22in', 'plus40in', 'apt64in']
SCENE_TITLE = dict(plus22='plus maze 22 x 22', plus30='plus maze 30 x 30', plus40='plus maze 40 x 40', apt32='apartment, 32 cells', apt64='apartment, 64 cells',
                   Allensville='house Allensville, 43 cells', Lakeville='house Lakeville, 83 cells', Klickitat='house Klickitat, 96 cells',
                   plus22in='plus maze 22 x 22, 20 % of the pairs with x1 inside an obstacle', plus40in='plus maze 40 x 40, 20 % inside pairs',
                   apt64in='apartment, 64 cells, 20 % inside pairs')


def load():
    state = json.load(open(HERE / 'queue/state.json'))
    scenes = json.load(open(HERE / 'queue/scenes.json'))
    runs, blocked = [], {}
    for job, rid in state['done'].items():
        parts = job.split('|')
        d = ROOT / f'results/registered_{rid}'
        if not (d / 'results.json').exists():
            continue
        r = json.load(open(d / 'results.json'))
        if parts[0] == 'blocked':
            blocked[parts[1]] = dict(id=rid, **{k: r[k] for k in ('blocked_share', 'blocked48_share', 'missed_by_48', 'by_gap_cells', 'truth', 'nu', 'profile', 'gap_cells_quantiles')}, E=r['scene']['E_cells'])
            continue
        if parts[0] != 'field':
            continue
        _, label, key, seed, extra = parts
        acc, wf, st = r['accuracy'], r['wall_fit'], r['stalls']
        tot = r['success_total']
        row = dict(id=rid, scene=key, label=label, seed=int(seed), extra=extra, E=r['scene']['E_cells'], solved=100 * tot[0] / tot[1], success=tot,
                   by_bin={k: 100 * v[0] / v[1] for k, v in r['success_by_bin'].items()}, by_bin_n=r['success_by_bin'],
                   wall=100 * r['failures']['arrived_wall'] / r['failures']['runs'], stall=100 * r['failures']['not_arrived'] / r['failures']['runs'],
                   within10=100 * acc['within10'], ratio=acc['ratio_median'], short_share=100 * acc['short_share'], long_share=100 * acc['long_share'],
                   los_within10=100 * acc.get('los', {}).get('within10', np.nan), los_ratio=acc.get('los', {}).get('ratio_median'),
                   rho=acc.get('detour', {}).get('rho_median'), seen_through=100 * acc.get('detour', {}).get('seen_through', np.nan),
                   detour_by_length=acc.get('detour', {}).get('by_length', {}), by_distance=acc.get('by_distance', {}),
                   wall_innermost=wf['innermost'], wall_by_speed={k: v['slope'] for k, v in wf['by_speed'].items()}, overshoot=100 * wf['overshoot_share'],
                   stalls=st, descent=r.get('descent_detour'), bumps=r['bumps']['bumps_per_path'], course=r.get('course', {}), by_where=r.get('by_where'),
                   t_max=acc['t_max'], t_star_max=acc['t_star_max'], rollbacks=r['rollbacks'], train_min=r['train_seconds'] / 60, push=r['push'], field=r['field'],
                   final_loss=r['final_loss'], nu=r['nu'])
        runs.append(row)
    return runs, blocked, scenes, state


def pooled(runs, scene, label):
    rs = [r for r in runs if r['scene'] == scene and r['label'] == label]
    if not rs:
        return None
    s = sum(r['success'][0] for r in rs)
    n = sum(r['success'][1] for r in rs)

    def mean(k):
        v = [r[k] for r in rs if r[k] is not None and not (isinstance(r[k], float) and np.isnan(r[k]))]
        return float(np.mean(v)) if v else None
    return dict(seeds=len(rs), solved=100 * s / n, solved_each=[round(r['solved'], 1) for r in sorted(rs, key=lambda r: r['seed'])], within10=mean('within10'), ratio=mean('ratio'),
                los_within10=mean('los_within10'), rho=mean('rho'), seen_through=mean('seen_through'), wall_innermost=mean('wall_innermost'), wall=mean('wall'), stall=mean('stall'),
                by_bin={k: 100 * sum(r['by_bin_n'][k][0] for r in rs) / sum(r['by_bin_n'][k][1] for r in rs) for k in rs[0]['by_bin_n']})


def fmt(v, d=1):
    return '–' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:.{d}f}'


def main():
    OUT.mkdir(exist_ok=True)
    runs, blocked, scenes, state = load()
    json.dump(dict(runs=runs, blocked=blocked, scenes=scenes, failed=state['failed']), open(OUT / 'runs.json', 'w'), indent=1, default=float)
    labels = []
    for r in runs:
        if r['label'] not in labels:
            labels.append(r['label'])
    lines = ['# lift_scale tables (report.py; every number from registered runs)\n']
    lines.append('## Scenes\n')
    lines.append('| scene | E (cells) | blocked training pairs % | missed by 48 points % | largest T* (box lengths) | nu median / q90 | prepare | blocked |')
    lines.append('|---|---:|---:|---:|---:|---:|---|---|')
    for s in SCENES:
        if s in blocked:
            b = blocked[s]
            lines.append(f"| {SCENE_TITLE[s]} | {b['E']:.1f} | {100 * b['blocked_share']:.1f} | {100 * b['missed_by_48']:.1f} | {b['truth']['t_star_max']:.2f} | {b['nu']['median']:.2f} / {b['nu']['q90']:.2f} | "
                         f"{scenes[s]['prepared'].replace('results/registered_', '')} | {b['id']} |")
    lines.append('\n## Pooled over training seeds\n')
    lines.append('| scene | arm | seeds | solved % (each seed) | wall % | stall % | T within 10 % | median T/T* | line-of-sight within 10 % | detour index | seen through % | S x slope innermost |')
    lines.append('|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for s in SCENES:
        for lab in labels:
            p = pooled(runs, s, lab)
            if p:
                lines.append(f"| {SCENE_TITLE[s]} | {lab} | {p['seeds']} | {p['solved']:.1f} ({' / '.join(map(str, p['solved_each']))}) | {fmt(p['wall'])} | {fmt(p['stall'])} | {fmt(p['within10'])} | "
                             f"{fmt(p['ratio'], 3)} | {fmt(p['los_within10'])} | {fmt(p['rho'], 2)} | {fmt(p['seen_through'])} | {fmt(p['wall_innermost'], 2)} |")
    lines.append('\n## Solved % by true travel distance (cells), pooled\n')
    for s in SCENES:
        rows = [(lab, pooled(runs, s, lab)) for lab in labels]
        rows = [(lab, p) for lab, p in rows if p]
        if not rows:
            continue
        bins = list(rows[0][1]['by_bin'])
        lines.append(f'\n**{SCENE_TITLE[s]}**\n')
        lines.append('| arm | ' + ' | '.join(bins) + ' |')
        lines.append('|---|' + '---:|' * len(bins))
        for lab, p in rows:
            lines.append(f'| {lab} | ' + ' | '.join(fmt(p['by_bin'].get(b), 0) for b in bins) + ' |')
    lines.append('\n## Every run\n')
    lines.append('| scene | arm | seed | id | solved % | wall % | stall % | T within 10 % | T/T* | LOS within 10 % | detour index | S x slope innermost | rollbacks | train min | extra |')
    lines.append('|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for r in sorted(runs, key=lambda r: (SCENES.index(r['scene']), labels.index(r['label']), r['seed'])):
        lines.append(f"| {r['scene']} | {r['label']} | {r['seed']} | {r['id']} | {r['solved']:.1f} | {r['wall']:.1f} | {r['stall']:.1f} | {r['within10']:.1f} | {r['ratio']:.3f} | {fmt(r['los_within10'])} | "
                     f"{fmt(r['rho'], 2)} | {fmt(r['wall_innermost'], 2)} | {r['rollbacks']} | {r['train_min']:.1f} | `{r['extra']}` |")
    if state['failed']:
        lines.append('\n## Failed jobs\n')
        for j, rid in state['failed'].items():
            lines.append(f'* `{j}` -> {rid}')
    (OUT / 'tables.md').write_text('\n'.join(lines) + '\n')
    print(f'{len(runs)} field runs, {len(blocked)} scenes -> {OUT}/tables.md')


if __name__ == '__main__':
    main()
