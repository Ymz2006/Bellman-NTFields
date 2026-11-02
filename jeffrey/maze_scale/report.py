"""Collect the registered maze-scale runs into tables (report/tables.md, report/summary.json).

    ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/report.py

Runs are found under results/registered_*/results.json by stage (prepare / ntrl / segdp) and maze size; superseded
runs (registry notes, listed in SUPERSEDED) are skipped. Success = MPPI successes over planning seeds 0/1/2, pooled
per training seed; '-' = no queries in that bin for that maze.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'report'
SUPERSEDED = {'20261101T192118Z-57dea88a', '20261101T192127Z-848660ef', '20261101T192138Z-d3554001', '20261101T192118Z-645cf7ff'}


# Dropped from the report on request (2026-11-01): the 40 x 40 maze (replaced by 10 x 10) and the zoomed-input control
# (zoom runs and the segment-DP chain on a zoomed base). Their registered runs stay on disk; README lists them.
DROPPED_N = {40}
DROPPED_STAGES = {'zoom'}


def dropped(d):
    if d.get('maze', {}).get('variant') == 'plus':      # the plus maze (plus.py) is reported on its own, not as 22 x 22
        return d.get('stage') not in ('plus-prepare', 'plus-ntrl', 'plus-uturn', 'uturneval')
    return d.get('maze', {}).get('n') in DROPPED_N or d.get('stage') in DROPPED_STAGES or 'zoom' in d.get('config', {})


def runs(stage, include_dropped=False):
    out = []
    for r in sorted(ROOT.glob('results/registered_*/results.json')):
        rid = r.parent.name.replace('registered_', '')
        if rid in SUPERSEDED:
            continue
        try:
            d = json.loads(r.read_text())
        except Exception:
            continue
        if not include_dropped and dropped(d):
            continue
        if d.get('stage') == stage and d.get('complete') and not d.get('smoke'):
            d['_id'], d['_dir'] = rid, r.parent
            out.append(d)
    return out


def base_scale(d):
    """Fourier scale of a segment-DP run's base network (1 for a plain ntrl-demo base, b_scale for a fourier.py base)."""
    f = Path(d.get('base', '')).parents[1] / 'results.json' if d.get('base') else None
    if f is None or not f.exists():
        return 1.0
    return float(json.loads(f.read_text()).get('b_scale', 1.0))


def short_variant(d):
    """Display name of a segment-DP run's variant."""
    c = d.get('config', {})
    if 'zoom' in c:
        return 'segment DP, zoomed base'
    if d.get('normal_term'):
        w = c.get('normal_weight', 1.0)
        r = c.get('ratio', 1.35)
        k = c.get('candidates', 128)
        name = 'segment DP + wall term' + ('' if w == 1.0 else f' (weight {w:g})')
        if r != 1.35:
            name += f', {d["levels"]} levels (ratio {r:.3f})'
        if k != 128:
            name += f', {k} split candidates'
        b = base_scale(d)
        if b != 1.0:
            name += f', B × {b:g} base'
        return name
    if c.get('check_pieces'):
        return 'segment DP, checked pieces'
    return 'segment DP'


variant_of = short_variant


def collect():
    prep = {d['maze']['n']: d for d in runs('prepare')}
    ntrl, seg = {}, {}
    for d in runs('ntrl'):
        ntrl.setdefault(d['maze']['n'], {})[d['seed']] = d
    for d in runs('segdp'):
        seg.setdefault(variant_of(d), {}).setdefault(d['maze']['n'], {})[d['seed']] = d
    return prep, ntrl, seg


def pct(a, b):
    return f'{100 * a / b:.0f}' if b else '-'


def table_success(prep, groups, labels, title):
    """groups: {name: {n: {seed: results}}} -> markdown table of success % by bin (mean over seeds, per-seed list)."""
    lines = [f'### {title}', '', '| maze | field | ' + ' | '.join(labels) + ' | all |', '|' + '---|' * (len(labels) + 3)]
    for n in sorted(prep):
        o = prep[n]['oracle']
        cells = []
        for lab in labels:
            vals = [o[s]['by_bin'].get(lab) for s in o]
            cells.append(pct(sum(v[0] for v in vals), sum(v[1] for v in vals)) if all(vals) else '-')
        tot = sum(o[s]['success'] for s in o), sum(sum(v[1] for v in o[s]['by_bin'].values()) for s in o)
        lines.append(f'| {n} | oracle (MPPI on T*) | ' + ' | '.join(cells) + f' | {pct(*tot)} |')
        st = prep[n]['straight']['by_bin']
        lines.append(f'| {n} | straight line | ' + ' | '.join(pct(*st[lab]) if lab in st else '-' for lab in labels) + f' | {pct(prep[n]["straight"]["success"], sum(v[1] for v in st.values()))} |')
        for name, by_n in groups.items():
            seeds = by_n.get(n, {})
            if not seeds:
                continue
            cells = []
            for lab in labels:
                per = [seeds[s]['success_by_bin'].get(lab) for s in sorted(seeds)]
                if not all(per):
                    cells.append('-')
                    continue
                cells.append(f'{pct(sum(p[0] for p in per), sum(p[1] for p in per))} ({"/".join(pct(*p) for p in per)})')
            tot = [seeds[s]['success_total'] for s in sorted(seeds)]
            lines.append(f'| {n} | {name} [{len(seeds)} seeds] | ' + ' | '.join(cells) + f' | {pct(sum(t[0] for t in tot), sum(t[1] for t in tot))} |')
    return '\n'.join(lines) + '\n'


def table_field(prep, groups, labels, key, title, fmt='{:.2f}'):
    lines = [f'### {title}', '', '| maze | field | ' + ' | '.join(labels) + ' | all |', '|' + '---|' * (len(labels) + 3)]
    for n in sorted(prep):
        for name, by_n in groups.items():
            seeds = by_n.get(n, {})
            if not seeds:
                continue
            cells = []
            for lab in labels:
                vals = [seeds[s]['field']['by_bin'].get(lab, {}).get(key) for s in sorted(seeds)]
                vals = [v for v in vals if v is not None]
                cells.append(fmt.format(np.mean(vals)) if vals else '-')
            allk = {'t_over_tstar_median': 't_over_tstar_median', 'descent_cos_median': 'descent_cos_median', 'spearman': 'spearman'}.get(key)
            tot = [seeds[s]['field'].get(allk) for s in sorted(seeds)] if allk else []
            tot = [v for v in tot if v is not None]
            lines.append(f'| {n} | {name} | ' + ' | '.join(cells) + f' | {fmt.format(np.mean(tot)) if tot else "-"} |')
    return '\n'.join(lines) + '\n'


def main():
    OUT.mkdir(exist_ok=True)
    prep, ntrl, seg = collect()
    labels = []
    for n in sorted(prep):
        for lab in prep[n]['queries']['per_bin']:
            if prep[n]['queries']['per_bin'][lab] and lab not in labels:
                labels.append(lab)
    groups = {'ntrl-demo': ntrl}
    zoom = {}
    for d in runs('zoom'):
        zoom.setdefault(d['maze']['n'], {})[d['seed']] = d
    groups['ntrl-demo, zoomed input'] = zoom
    groups.update(seg)
    md = ['# Maze-scale tables (generated by report.py)', '']
    md.append(table_success(prep, groups, labels, 'MPPI success % by true travel distance (cells); mean over training seeds (per seed)'))
    md.append(table_field(prep, groups, labels, 't_over_tstar_median', 'Field T(start, goal) / T* (median per bin, mean over seeds)'))
    md.append(table_field(prep, groups, labels, 'descent_cos_median', 'Cosine of the field\'s descent direction at the start with the truth\'s (median per bin)'))
    md.append(table_field(prep, groups, labels, 'spearman', 'Spearman rank correlation of T with T* within each bin'))
    (OUT / 'tables.md').write_text('\n'.join(md))
    summary = dict(prepare={n: dict(id=d['_id'], queries=d['queries'], oracle={s: d['oracle'][s]['success'] for s in d['oracle']}, straight=d['straight']['success'])
                            for n, d in prep.items()},
                   ntrl={n: {s: dict(id=d['_id'], success=d['success_total'], by_bin=d['success_by_bin'], field=d['field'], speed_fit=d.get('speed_fit'))
                             for s, d in by.items()} for n, by in ntrl.items()},
                   segdp={v: {n: {s: dict(id=d['_id'], success=d['success_total'], by_bin=d['success_by_bin'], field=d['field'], levels=d['levels'])
                                  for s, d in by.items()} for n, by in byv.items()} for v, byv in seg.items()})
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
