"""Helpers of the lift-scale findings page: numbers from report/runs.json, chart specs for page_assets/charts.js, HTML pieces."""
import html
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'report'
SCENES = ['plus22', 'plus30', 'plus40', 'apt32', 'Allensville', 'apt64', 'Lakeville', 'Klickitat']
TITLE = dict(plus22='plus maze 22 × 22', plus30='plus maze 30 × 30', plus40='plus maze 40 × 40', apt32='apartment, 32 cells', apt64='apartment, 64 cells',
             Allensville='house Allensville, 43 cells', Lakeville='house Lakeville, 83 cells', Klickitat='house Klickitat, 96 cells')
SHORT = dict(plus22='plus 22', plus30='plus 30', plus40='plus 40', apt32='apartment 32', apt64='apartment 64', Allensville='Allensville 43', Lakeville='Lakeville 83', Klickitat='Klickitat 96')
ARM = dict(ntrl='ntrl-demo (control)', lift='lift', liftpush='lift + push', pushonly='push without the lift (control)', lift_c10='lift + constant strong push',
           vec='vector lift', vecpush='vector lift + push', lift_r6='lift + push relaxed to 6e-3', liftpush_oct8='lift + push + octaves ×2, ×4, ×8',
           lift_push='lift + push, one-sided term not faded', push='push without the lift, one-sided term not faded', lift_nohead='lift, heading-blind network',
           vec_c10='vector lift + constant strong push', lift_x='lift + constant strong push + exact slope cap', lift_x3='lift + constant weak push + exact slope cap',
           liftpush_cell='lift + push scaled with the scene', liftpush_b='lift + push, Fourier frequencies scaled with the scene', ntrl_b='ntrl-demo, Fourier frequencies scaled with the scene',
           lift_in='lift, inside pairs', lift_in_long='lift, inside pairs, 12,500 epochs', liftx_in='lift + strong push + exact cap, inside pairs',
           liftpush_in='lift + push, inside pairs', liftq_in='lift + strong push + stiff cap, inside pairs', liftq3_in='lift + 3x strong push + stiff cap, inside pairs',
           liftq='lift + strong push + stiff cap', vec_in='vector lift, inside pairs', vecpush_in='vector lift + push, inside pairs',
           lift_c6_oct8='lift + octaves + constant push 6e-3', lift_r6_oct8='lift + octaves + push relaxed to 6e-3', ntrl_oct8='ntrl-demo + octaves (control)',
           lift_oct8='lift + octaves', liftpush_oct='lift + push + octaves', ntrl_oct='ntrl-demo + octaves (control)',
           lift_c10_oct8='lift + octaves + constant strong push', lift_c10_oct='lift + octaves + constant strong push', liftpush_oct2='lift + push + octaves one octave higher')
SLOT = dict(ntrl=1, lift=2, liftpush=3, pushonly=4, ntrl_oct=5, liftpush_oct=6, lift_c10_oct=2, lift_c10=5, vecpush=6, vec=5, lift_r6=5, liftpush_oct8=6, vec_c10=2, lift_x=6, lift_x3=5)


class Data:
    def __init__(self):
        d = json.load(open(OUT / 'runs.json'))
        self.runs, self.blocked, self.scenes = d['runs'], d['blocked'], d['scenes']
        self.jumps = json.load(open(OUT / 'jumps.json')) if (OUT / 'jumps.json').exists() else []
        self.heading = json.load(open(OUT / 'heading.json')) if (OUT / 'heading.json').exists() else []

    ALIAS = {('plus22', 'liftpush_oct'): 'liftpush_oct8', ('plus22', 'ntrl_oct'): 'ntrl_oct8', ('plus22', 'lift_c10_oct'): 'lift_c10_oct8'}

    def sel(self, scene, label, seed=None):
        labels = {label, self.ALIAS.get((scene, label), label)}   # on the 22-cell maze the per-cell octaves are x2, x4, x8 themselves
        return [r for r in self.runs if r['scene'] == scene and r['label'] in labels and (seed is None or r['seed'] == seed)]

    def one(self, scene, label, seed=87358):
        rs = self.sel(scene, label, seed)
        return rs[0] if rs else None

    def pool(self, scene, label):
        rs = self.sel(scene, label)
        if not rs:
            return None
        s = sum(r['success'][0] for r in rs)
        n = sum(r['success'][1] for r in rs)

        def mean(k):
            v = [r[k] for r in rs if r.get(k) is not None and not (isinstance(r[k], float) and np.isnan(r[k]))]
            return float(np.mean(v)) if v else None
        out = dict(seeds=len(rs), solved=100 * s / n, each=[r['solved'] for r in sorted(rs, key=lambda r: r['seed'])],
                   by_bin={k: 100 * sum(r['by_bin_n'][k][0] for r in rs) / sum(r['by_bin_n'][k][1] for r in rs) for k in rs[0]['by_bin_n']})
        for k in ('within10', 'ratio', 'los_within10', 'los_ratio', 'rho', 'seen_through', 'wall_innermost', 'wall', 'stall', 'short_share', 'long_share', 'overshoot'):
            out[k] = mean(k)
        return out

    def has(self, scene, label):
        return bool(self.sel(scene, label))

    def scenes_with(self, *labels):
        return [s for s in SCENES if all(self.has(s, l) for l in labels)]


def f(v, d=1):
    return '–' if v is None or (isinstance(v, float) and not np.isfinite(v)) else f'{v:.{d}f}'


def esc(s):
    return html.escape(str(s))


def hbar(title, sub, groups, xmax=100, ticks=(0, 25, 50, 75, 100), digits=1, suffix=' %', value_name='value', legend=None, xmin=0, unit=False):
    """groups: [(name, [(label, value, slot, note), ...]), ...]"""
    spec = dict(type='hbar', title=title, sub=sub, xmax=xmax, xmin=xmin, ticks=list(ticks), digits=digits, unit_suffix=suffix, value_name=value_name,
                groups=[dict(name=n, rows=[dict(label=l, value=v, slot=s, **({'note': note} if note else {})) for l, v, s, note in rows]) for n, rows in groups])
    if unit:
        spec['unit'] = True
    if legend:
        spec['legend'] = [dict(name=n, slot=s) for n, s in legend]
    return spec


def line(title, sub, series, xlabel='', ylabel='', xticks=None, yticks=None, xticklabels=None, refs=None, ymin=None, ymax=None, xmin=None, xmax=None, logx=False, digits=3, xname='x', height=300):
    """series: [(name, slot, [(x, y), ...], dash)]"""
    spec = dict(type='line', title=title, sub=sub, xlabel=xlabel, ylabel=ylabel, digits=digits, xname=xname, height=height,
                series=[dict(name=n, slot=s, points=[[float(x), float(y)] for x, y in pts if y is not None and np.isfinite(y)], **({'dash': dash} if dash else {})) for n, s, pts, dash in series],
                legend=[dict(name=n, slot=s) for n, s, _, _ in series])
    for k, v in (('xticks', xticks), ('yticks', yticks), ('xticklabels', xticklabels), ('ymin', ymin), ('ymax', ymax), ('xmin', xmin), ('xmax', xmax)):
        if v is not None:
            spec[k] = v
    if refs:
        spec['refs'] = [dict(y=y, label=l) for y, l in refs]
        spec['legend'] = spec['legend'] + [dict(name=l, ref=True) for _, l in refs if l]
    if logx:
        spec['logx'] = True
    return spec


def table(head, rows, left=1):
    out = ['<table><thead><tr>' + ''.join(f'<th{" class=l" if i < left else ""}>{esc(h)}</th>' for i, h in enumerate(head)) + '</tr></thead><tbody>']
    for r in rows:
        out.append('<tr>' + ''.join(f'<td{" class=l" if i < left else ""}>{c}</td>' for i, c in enumerate(r)) + '</tr>')
    out.append('</tbody></table>')
    return '\n'.join(out)


def details(summary, body):
    return f'<details><summary>{esc(summary)}</summary><div class="tbl">{body}</div></details>'


def chart(key):
    return f'<div class="chart" data-chart="{key}" role="figure"></div>'


def figure(src, caption, alt=''):
    return f'<figure><img src="{src}" alt="{esc(alt or caption[:120])}"><figcaption>{caption}</figcaption></figure>'


def course(D, items, key, scale=1.0):
    """Series of a course measure: items = [(scene, label, name, slot), ...] (seed 87358 runs)."""
    out = []
    for scene, label, name, slot in items:
        r = D.one(scene, label)
        if not r or not r.get('course'):
            continue
        pts = [(int(e), (c.get(key) * scale if c.get(key) is not None else None)) for e, c in sorted(r['course'].items(), key=lambda t: int(t[0]))]
        out.append((name, slot, pts, ''))
    return out


def course_solved(D, items):
    """Planning seed 0 at the saved epochs plus the final field (mean of the planning seeds)."""
    out = []
    for scene, label, name, slot in items:
        r = D.one(scene, label)
        if not r or not r.get('course'):
            continue
        pts = [(int(e), 100 * c['success'][0] / c['success'][1]) for e, c in sorted(r['course'].items(), key=lambda t: int(t[0])) if 'success' in c]
        pts.append((max(int(e) for e in r['course']), r['solved']))
        out.append((name, slot, pts, ''))
    return out
