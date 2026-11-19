"""Findings page of the sharp-feature study: report/sharp_features.html (inline SVG charts, theme-aware, hover tips).

    ~/.venvs/bellman-ntfield/bin/python jeffrey/sharp_features/report.py      # report/summary.json first
    ~/.venvs/bellman-ntfield/bin/python jeffrey/sharp_features/page.py [--status 'text']

Numbers come from report/summary.json (report.py) and report/screen.json (screen.py); the words from page_text.py.
"""
import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import svgcharts as C  # noqa: E402
import page_text as TX  # noqa: E402

OUT = HERE / 'report'
SIZES = ['13', '22']
CLIP = 30.0
esc = C.esc
POWERS = [2, 4, 8, 16]

LABEL = {
    'Rpowmix@half': 'half the rows → sin^p, cos^p, p = 2…16 mixed', 'Rpowmix@all': 'every row → p = 2…16 mixed',
    'Rpowmix@quarter': 'a quarter of the rows → p mixed',
    'Rpow4.partner@half': 'half the directions, each with (sin, cos, sin⁴, cos⁴)',
    'Rspow4@half': 'half the rows → signed |sin|⁴ (no fold)', 'Rspow16@half': 'half the rows → signed |sin|¹⁶ (no fold)',
    'Rpow4@top': 'top octave → sin⁴, cos⁴', 'Rpow16@top': 'top octave → sin¹⁶, cos¹⁶',
    'Apowmix@all': '+ sin^p, cos^p of every row, p mixed', 'Apowcat@all': '+ all four powers of every row (1,280 inputs)',
    'Apow16z@all': '+ sin¹⁶, cos¹⁶ standardized',
    'Asc@new128': 'control: + 128 new octave rows (more sin/cos)', 'Adup@all': 'control: + a copy of the 128 rows',
    'Aharm2@all': 'control: + every row at 2× frequency', 'Asc@new512': 'control: + 512 new octave rows (1,280 inputs)',
    'Rtri@all': 'every row → (8/π) arcsin(sin, cos) (as written)', 'Rtri1@all': 'every row → unit triangle',
    'Rsc3.27@all': 'gain control: sin/cos × 3.27 (triangle RMS)', 'Rtri@half': 'half the rows → triangle (amplitude 4)',
    'Atri@all': '+ triangle pair of every row (amplitude 4)', 'Atri1@all': '+ unit triangle pair of every row',
    'Agtri@allx2': '+ gate × triangle, 2 random h per row (as written)', 'Agtri1@allx2': '+ gate × unit triangle, 2 h per row',
    'Agsc@all': '+ gate × (sin, cos), 1 h per row', 'Agate@allx2': '+ gate alone, 2 h per row',
    'Agtri@topx8': '+ gate × triangle, top octave, 8 h per row', 'Agtri.grid@allx2': '+ gate × triangle, h at grid vertices',
    'Agtri.a3@allx2': '+ gate × triangle, slope a = 3', 'Agtri.a0.3@allx2': '+ gate × triangle, slope a = 0.3',
    'Agtrip@all': '+ gate × triangle pair, 1 h per row', 'Rtri@all+Agtri@allx2': 'triangle rows + gate × triangle',
    'Asc@axis': 'control: + plain rows on the grid axes', 'Apow16@axis': '+ sin¹⁶, cos¹⁶ on the grid axes',
    'Apow4@axis': '+ sin⁴, cos⁴ on the grid axes', 'Atri@axis': '+ triangle on the grid axes',
    'Agtri.grid@axis': '+ gate × triangle, an edge on every wall line', 'Agate.grid@axis': '+ gate alone, an edge on every wall line',
    'ctx:shift2': 'context: every row at 2× frequency (replaces)', 'ctx:shift0.5': 'context: every row at ½ frequency',
    'ntrl-demo': 'context: ntrl-demo (no octaves)',
}
for p in POWERS:
    sup = {2: '²', 4: '⁴', 8: '⁸', 16: '¹⁶'}[p]
    LABEL[f'Rpow{p}@half'] = f'half the rows → sin{sup}, cos{sup}'
    LABEL[f'Rpow{p}@all'] = f'every row → sin{sup}, cos{sup}'
    LABEL[f'Apow{p}@all'] = f'+ sin{sup}, cos{sup} of every row'

GROUPS = [
    ('Idea 1, way 1: replace (256 inputs)', ['Rpowmix@half', 'Rpow2@half', 'Rpow4@half', 'Rpow8@half', 'Rpow16@half', 'Rpowmix@quarter',
                                              'Rpowmix@all', 'Rpow2@all', 'Rpow4@all', 'Rpow8@all', 'Rpow16@all', 'Rpow4@top', 'Rpow16@top',
                                              'Rpow4.partner@half', 'Rspow4@half', 'Rspow16@half']),
    ('Idea 1, way 2: add (512 inputs) and its baselines', ['Apowmix@all', 'Apow2@all', 'Apow4@all', 'Apow8@all', 'Apow16@all', 'Apow16z@all',
                                                           'Apowcat@all', 'Asc@new128', 'Adup@all', 'Aharm2@all', 'Asc@new512']),
    ('Idea 2: triangle', ['Rtri@all', 'Rtri1@all', 'Rsc3.27@all', 'Rtri@half', 'Atri@all', 'Atri1@all']),
    ('Idea 3: tanh gate × triangle', ['Agtri@allx2', 'Agtri1@allx2', 'Agtrip@all', 'Agsc@all', 'Agate@allx2', 'Agtri@topx8',
                                      'Agtri.grid@allx2', 'Agtri.a3@allx2', 'Agtri.a0.3@allx2', 'Rtri@all+Agtri@allx2']),
    ('Grid-axis rows (crests and gate edges on the wall lines)', ['Asc@axis', 'Apow4@axis', 'Apow16@axis', 'Atri@axis', 'Agate.grid@axis', 'Agtri.grid@axis']),
    ('Context (earlier runs)', ['ctx:shift2', 'ctx:shift0.5', 'ntrl-demo']),
]


def role(key):
    """Mark colour by idea (validated all-pairs, light and dark): powers orange, triangle aqua, gates blue; plain controls ink-2;
    context muted."""
    if key.startswith(('ctx:', 'ntrl')):
        return 'r-ctx'
    if key in ('Asc@new128', 'Adup@all', 'Aharm2@all', 'Asc@new512', 'Asc@axis', 'Rsc3.27@all', 'Agate@allx2', 'Agate.grid@axis'):
        return 'r-ctl'
    if key.startswith(('Ag', 'Rtri@all+Ag')):
        return 'r-gate'
    if 'tri' in key:
        return 'r-tri'
    return 'r-pow'


def spec(S, n, key):
    return S['sizes'].get(n, {}).get('specs', {}).get(key)


def pm(S, n, key, m='success_rate'):
    e = spec(S, n, key)
    return (e or {}).get('paired', {}).get(m)


def sd_of(S, n, m='success_rate'):
    return ((S['noise'].get(n) or {}).get(m) or {}).get('sd')


def pct(v, p=1):
    return '–' if v is None else f'{100 * v:.{p}f} %'


# ---------------------------------------------------------------- feature shapes
def chart_shapes():
    zs = np.linspace(-2 * math.pi, 2 * math.pi, 801)
    tri = lambda z: 1 - (2 / math.pi) * np.abs(np.remainder(z + math.pi, 2 * math.pi) - math.pi)  # noqa: E731
    panels = []
    # powers
    f = C.Frame(-2, 2, -1.1, 1.1, w=320, h=210, left=34, right=10, top=12, bottom=38)
    body = f.axes([-2, -1, 0, 1, 2], [-1, 0, 1], xlabel='phase z (units of π)', xfmt=lambda v: f'{v:g}π' if v not in (0,) else '0',
                  yfmt=lambda v: f'{v:g}')
    body.append(C.polyline(f, list(zip(zs / math.pi, np.sin(zs))), 'r-ink', width=1.6))
    for i, p in enumerate(POWERS):
        body.append(C.polyline(f, list(zip(zs / math.pi, np.sin(zs) ** p)), f'r-pow op{i}', width=1.8))
    body.append(f'<text class="lbl small" x="{f.X(-1.5) + 8:.1f}" y="{f.Y(-0.55):.1f}">sin z</text>')
    panels.append(('sin<sup>p</sup> z, p = 2, 4, 8, 16 (darker = higher p)', C.svg(f, body, 'sin z and its even powers')))
    # triangle
    f = C.Frame(-2, 2, -4.4, 4.4, w=320, h=210, left=34, right=10, top=12, bottom=38)
    body = f.axes([-2, -1, 0, 1, 2], [-4, -2, 0, 2, 4], xlabel='phase z (units of π)', xfmt=lambda v: f'{v:g}π' if v not in (0,) else '0',
                  yfmt=lambda v: f'{v:g}')
    body.append(C.polyline(f, list(zip(zs / math.pi, np.cos(zs))), 'r-ink', width=1.6))
    body.append(C.polyline(f, list(zip(zs / math.pi, 4 * tri(zs))), 'r-tri', width=2))
    body.append(C.polyline(f, list(zip(zs / math.pi, tri(zs))), 'r-tri', width=1.6, dash='4 3'))
    body.append(f'<text class="lbl small" x="{f.X(0) + 6:.1f}" y="{f.Y(4) + 4:.1f}">(8/π) arcsin(cos z)</text>')
    body.append(f'<text class="lbl small" x="{f.X(1.05):.1f}" y="{f.Y(-1.2) + 12:.1f}">cos z, unit triangle</text>')
    panels.append(('the triangle as written (amplitude 4) and at amplitude 1', C.svg(f, body, 'triangle wave against cos z')))
    # gates
    f = C.Frame(-2, 2, -4.4, 4.4, w=320, h=210, left=34, right=10, top=12, bottom=38)
    body = f.axes([-2, -1, 0, 1, 2], [-4, -2, 0, 2, 4], xlabel='phase z (units of π)', xfmt=lambda v: f'{v:g}π' if v not in (0,) else '0',
                  yfmt=lambda v: f'{v:g}')
    body.append(C.polyline(f, list(zip(zs / math.pi, 4 * tri(zs))), 'r-tri', width=1.2, dash='2 3'))
    for i, h in enumerate([-math.pi, 0.0, 0.5 * math.pi]):
        g = np.maximum(0, np.tanh(zs - h)) * 4 * tri(zs)
        body.append(C.polyline(f, list(zip(zs / math.pi, g)), f'r-gate op{i + 1}', width=2))
        body.append(f'<line class="base" x1="{f.X(h / math.pi):.1f}" x2="{f.X(h / math.pi):.1f}" y1="{f.Y(0) - 4:.1f}" y2="{f.Y(0) + 4:.1f}"/>')
    panels.append(('gate × triangle, h = −π, 0, π/2', C.svg(f, body, 'tanh-gated triangle for three offsets')))
    return panels


# ---------------------------------------------------------------- paired dot chart
def chart_rows(S, n, keys, m='success_rate', xlabel='success minus the exact octave of the same seed (points)', scale=100.0, unit='pts', w=520):
    rows = [k for k in keys if pm(S, n, k, m)]
    if not rows:
        return '<p class="empty">No runs yet.</p>'
    sd = sd_of(S, n, m)
    band = 2 * sd * math.sqrt(4 / 3) * scale if sd else None
    vals = [0.0]
    for k in rows:
        d = pm(S, n, k, m)
        vals += [scale * v for v in d['deltas'].values()]
        if d.get('se'):
            vals += [scale * (d['mean'] - 2 * d['se']), scale * (d['mean'] + 2 * d['se'])]
    clip = CLIP if scale == 100.0 else max(abs(v) for v in vals) * 1.1 + 1e-9
    lo = max(min(min(vals), -(band or 1)) * 1.08, -clip)
    hi = min(max(max(vals), band or 1) * 1.08, clip)
    if hi - lo < 1e-9:
        lo, hi = -1, 1
    cx = lambda v: min(max(v, lo), hi)  # noqa: E731
    rowh = 22
    f = C.Frame(lo, hi, 0, 1, w=w, h=16 + rowh * len(rows) + 40, left=250, right=14, top=12, bottom=40)
    body = []
    if band:
        x0, x1 = f.X(max(-band, lo)), f.X(min(band, hi))
        body.append(f'<rect class="band" x="{x0:.1f}" y="{f.t}" width="{x1 - x0:.1f}" height="{f.h - f.t - f.b}"/>')
    for t in C.nice_ticks(lo, hi, 6):
        x = f.X(t)
        body.append(f'<line class="{"zero" if abs(t) < 1e-12 else "grid"}" x1="{x:.1f}" x2="{x:.1f}" y1="{f.t}" y2="{f.h - f.b}"/>')
        body.append(f'<text class="ax" x="{x:.1f}" y="{f.h - f.b + 16}" text-anchor="middle">{"0" if abs(t) < 1e-12 else f"{t:+g}"}</text>')
    body.append(f'<text class="ax axt" x="{(f.l + f.w - f.r) / 2:.1f}" y="{f.h - 4}" text-anchor="middle">{esc(xlabel)}</text>')
    for i, k in enumerate(rows):
        d = pm(S, n, k, m)
        y = f.t + i * rowh + rowh / 2
        cls = role(k)
        body.append(f'<text class="ax rowlab" x="{f.l - 8}" y="{y + 4:.1f}" text-anchor="end">{esc(LABEL.get(k, k))}</text>')
        for s, v in d['deltas'].items():
            body.append(f'<circle class="{cls} seed" cx="{f.X(cx(scale * v)):.1f}" cy="{y:.1f}" r="3"/>')
        mean, se = scale * d['mean'], (scale * d['se'] if d.get('se') else None)
        if se:
            body.append(f'<line class="{cls} whisk" x1="{f.X(cx(mean - 2 * se)):.1f}" x2="{f.X(cx(mean + 2 * se)):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>')
        if mean < lo or mean > hi:
            left = mean < lo
            tx = f.X(lo) + 8 if left else f.X(hi) - 8
            body.append(f'<text class="lbl small" x="{tx:.1f}" y="{y - 6:.1f}" text-anchor="{"start" if left else "end"}">'
                        f'{"← " if left else ""}{mean:+.1f}{"" if left else " →"}</text>')
        e = spec(S, n, k)
        gl = e.get('gained_lost', {})
        tip = (f'{LABEL.get(k, k)} [{k}]: {mean:+.1f} {unit}' + (f' ± {se:.1f} (z {d["z"]:+.1f})' if se else '') + f' over {d["k"]} seed(s) ('
               + ', '.join(f'{s}: {scale * v:+.1f}' for s, v in d['deltas'].items()) + ')')
        if m == 'success_rate':
            tip += f'; success {pct(e["pooled"]["success"] / e["pooled"]["runs"])}'
            if gl:
                tip += '; runs gained / lost vs the exact octave run: ' + ', '.join(f'{v["gained"]}/{v["lost"]}' for v in gl.values())
        xm = f.X(cx(mean))
        body.append(f'<g class="pt" data-tip="{esc(tip)}" tabindex="0"><rect class="hit" x="{f.l}" y="{y - rowh / 2:.1f}" width="{f.w - f.l - f.r}" height="{rowh}"/>'
                    f'<line class="{cls} meanbar" x1="{xm:.1f}" x2="{xm:.1f}" y1="{y - 8:.1f}" y2="{y + 8:.1f}"/></g>')
    return C.svg(f, body, f'Paired change of {m} against the exact octave, {n}x{n}')


def chart_rows2(S, keys, m='success_rate', xlabel='success minus the exact octave of the same seed (points)', scale=100.0, unit='pts'):
    """Rows = specs, two plot columns (13x13 | 22x22) with their own x scale; seed dots, mean bar, +-2 SE whisker, and the
    +-2 x one-run noise band of each maze."""
    rows = [k for k in keys if any(pm(S, n, k, m) for n in SIZES)]
    if not rows:
        return '<p class="empty">No runs yet.</p>'
    rowh, top, bottom, lab = 22, 30, 42, 300
    colw, gap, W = 330, 24, 300 + 330 * 2 + 24 + 10
    H = top + rowh * len(rows) + bottom
    body = []
    for ci, n in enumerate(SIZES):
        x0 = lab + ci * (colw + gap)
        sd = sd_of(S, n, m)
        band = 2 * sd * math.sqrt(4 / 3) * scale if sd else None
        vals = [0.0] + ([-band, band] if band else [])
        for k in rows:
            d = pm(S, n, k, m)
            if d:
                vals += [scale * v for v in d['deltas'].values()]
                if d.get('se'):
                    vals += [scale * (d['mean'] - 2 * d['se']), scale * (d['mean'] + 2 * d['se'])]
        lo, hi = max(min(vals) * 1.06, -CLIP), min(max(vals) * 1.06, CLIP)
        if hi - lo < 1e-9:
            lo, hi = -1, 1
        X = lambda v, lo=lo, hi=hi, x0=x0: x0 + (min(max(v, lo), hi) - lo) / (hi - lo) * colw  # noqa: E731
        body.append(f'<text class="axt ax" x="{x0 + colw / 2:.1f}" y="14" text-anchor="middle">{n}×{n} maze</text>')
        if band:
            body.append(f'<rect class="band" x="{X(-band):.1f}" y="{top - 6}" width="{X(band) - X(-band):.1f}" height="{H - top - bottom + 12}"/>')
        for t in C.nice_ticks(lo, hi, 6):
            x = X(t)
            body.append(f'<line class="{"zero" if abs(t) < 1e-12 else "grid"}" x1="{x:.1f}" x2="{x:.1f}" y1="{top - 6}" y2="{H - bottom + 6}"/>')
            body.append(f'<text class="ax" x="{x:.1f}" y="{H - bottom + 20}" text-anchor="middle">{"0" if abs(t) < 1e-12 else f"{t:+g}"}</text>')
        for i, k in enumerate(rows):
            d = pm(S, n, k, m)
            y = top + i * rowh + rowh / 2
            if not d:
                body.append(f'<text class="ax small" x="{x0 + colw / 2:.1f}" y="{y + 4:.1f}" text-anchor="middle">pending</text>')
                continue
            cls = role(k)
            for v in d['deltas'].values():
                body.append(f'<circle class="{cls} seed" cx="{X(scale * v):.1f}" cy="{y:.1f}" r="3"/>')
            mean, se = scale * d['mean'], (scale * d['se'] if d.get('se') else None)
            if se:
                body.append(f'<line class="{cls} whisk" x1="{X(mean - 2 * se):.1f}" x2="{X(mean + 2 * se):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>')
            if mean < lo or mean > hi:
                left = mean < lo
                body.append(f'<text class="lbl small" x="{X(lo) + 6 if left else X(hi) - 6:.1f}" y="{y - 6:.1f}" text-anchor="{"start" if left else "end"}">'
                            f'{"← " if left else ""}{mean:+.1f}{"" if left else " →"}</text>')
            e = spec(S, n, k)
            gl = e.get('gained_lost', {})
            tip = (f'{n}×{n}, {LABEL.get(k, k)} [{k}]: {mean:+.1f} {unit}' + (f' ± {se:.1f} (z {d["z"]:+.1f})' if se else '') + f' over {d["k"]} seed(s) ('
                   + ', '.join(f'{s_}: {scale * v:+.1f}' for s_, v in d['deltas'].items()) + ')')
            if m == 'success_rate':
                tip += f'; success {pct(e["pooled"]["success"] / e["pooled"]["runs"])}'
                if gl:
                    tip += '; runs gained / lost vs the exact octave run: ' + ', '.join(f'{v["gained"]}/{v["lost"]}' for v in gl.values())
            xm = X(mean)
            body.append(f'<g class="pt" data-tip="{esc(tip)}" tabindex="0"><rect class="hit" x="{x0}" y="{y - rowh / 2:.1f}" width="{colw}" height="{rowh}"/>'
                        f'<line class="{cls} meanbar" x1="{xm:.1f}" x2="{xm:.1f}" y1="{y - 8:.1f}" y2="{y + 8:.1f}"/></g>')
    for i, k in enumerate(rows):
        y = top + i * rowh + rowh / 2
        body.append(f'<text class="ax rowlab" x="{lab - 12}" y="{y + 4:.1f}" text-anchor="end">{esc(LABEL.get(k, k))}</text>')
    body.append(f'<text class="ax axt" x="{lab + colw + gap / 2:.1f}" y="{H - 4}" text-anchor="middle">{esc(xlabel)}</text>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(xlabel)}" preserveAspectRatio="xMidYMid meet">' + ''.join(body) + '</svg>'


def chart_fail2(S, keys):
    """Rows = specs, two columns (13x13 | 22x22): change in runs through a wall (filled) and never arrived (open)."""
    rows = [k for k in keys if any(pm(S, n, k, 'wall_rate') for n in SIZES)]
    if not rows:
        return '<p class="empty">No runs yet.</p>'
    rowh, top, bottom, lab = 22, 30, 42, 300
    colw, gap = 330, 24
    W = lab + 2 * colw + gap + 10
    H = top + rowh * len(rows) + bottom
    body = []
    for ci, n in enumerate(SIZES):
        x0 = lab + ci * (colw + gap)
        vals = [0.0]
        for k in rows:
            for m in ('wall_rate', 'never_arrived_rate'):
                d = pm(S, n, k, m)
                if d:
                    vals.append(100 * d['mean'])
        lo, hi = max(min(vals) - 2, -45), min(max(vals) + 2, 45)
        X = lambda v, lo=lo, hi=hi, x0=x0: x0 + (min(max(v, lo), hi) - lo) / (hi - lo) * colw  # noqa: E731
        body.append(f'<text class="axt ax" x="{x0 + colw / 2:.1f}" y="14" text-anchor="middle">{n}×{n} maze</text>')
        for t in C.nice_ticks(lo, hi, 6):
            x = X(t)
            body.append(f'<line class="{"zero" if abs(t) < 1e-12 else "grid"}" x1="{x:.1f}" x2="{x:.1f}" y1="{top - 6}" y2="{H - bottom + 6}"/>')
            body.append(f'<text class="ax" x="{x:.1f}" y="{H - bottom + 20}" text-anchor="middle">{"0" if abs(t) < 1e-12 else f"{t:+g}"}</text>')
        for i, k in enumerate(rows):
            y = top + i * rowh + rowh / 2
            dw, dn = pm(S, n, k, 'wall_rate'), pm(S, n, k, 'never_arrived_rate')
            if not (dw and dn):
                body.append(f'<text class="ax small" x="{x0 + colw / 2:.1f}" y="{y + 4:.1f}" text-anchor="middle">pending</text>')
                continue
            cls = role(k)
            xw, xn = X(100 * dw['mean']), X(100 * dn['mean'])
            body.append(f'<line class="base" x1="{min(xw, xn):.1f}" x2="{max(xw, xn):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>')
            tip = (f'{n}×{n}, {LABEL.get(k, k)} [{k}]: reached the goal only through a wall {100 * dw["mean"]:+.1f} points, never reached it '
                   f'{100 * dn["mean"]:+.1f} points (over {dw["k"]} seed(s), against the exact octave of the same seed)')
            body.append(f'<g class="pt" data-tip="{esc(tip)}" tabindex="0"><rect class="hit" x="{x0}" y="{y - rowh / 2:.1f}" width="{colw}" height="{rowh}"/>'
                        f'<circle class="{cls} dot ring" cx="{xw:.1f}" cy="{y:.1f}" r="4.5"/>'
                        f'<circle class="{cls} dot hollow" cx="{xn:.1f}" cy="{y:.1f}" r="4.5"/></g>')
    for i, k in enumerate(rows):
        y = top + i * rowh + rowh / 2
        body.append(f'<text class="ax rowlab" x="{lab - 12}" y="{y + 4:.1f}" text-anchor="end">{esc(LABEL.get(k, k))}</text>')
    body.append(f'<text class="ax axt" x="{lab + colw + gap / 2:.1f}" y="{H - 4}" text-anchor="middle">change in the share of runs (points), against the exact octave of the same seed</text>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="failure split" preserveAspectRatio="xMidYMid meet">' + ''.join(body) + '</svg>'


# ---------------------------------------------------------------- power sweep
POWER_SERIES = [  # (label, key pattern, mix key, dash, marker class)
    ('half the rows replaced', 'Rpow{p}@half', 'Rpowmix@half', None, ''),
    ('every row replaced', 'Rpow{p}@all', 'Rpowmix@all', '6 4', ' hollow'),
    ('added to every row', 'Apow{p}@all', 'Apowmix@all', '1.5 3', ' sq'),
]
POWER_REFS = [('Asc@new128', '+ 128 new rows'), ('Adup@all', '+ copy of the rows'), ('Aharm2@all', '+ rows at 2× frequency')]


def chart_power(S, n):
    xs = {2: 1, 4: 2, 8: 3, 16: 4, 'mix': 5.2}
    pts_all = []
    for lab, pat, mix, dash, mk in POWER_SERIES:
        pts = []
        for p in POWERS:
            d = pm(S, n, pat.format(p=p))
            if d:
                pts.append((xs[p], 100 * d['mean'], 100 * (d.get('se') or 0), pat.format(p=p), d))
        d = pm(S, n, mix)
        if d:
            pts.append((xs['mix'], 100 * d['mean'], 100 * (d.get('se') or 0), mix, d))
        pts_all.append((lab, dash, mk, pts))
    if not any(p for *_, p in pts_all):
        return '<p class="empty">No power runs yet.</p>'
    vals = [0.0] + [y + s for *_, pts in pts_all for _, y, s, _, _ in pts] + [y - s for *_, pts in pts_all for _, y, s, _, _ in pts]
    refs = [(k, lab, pm(S, n, k)) for k, lab in POWER_REFS if pm(S, n, k)]
    vals += [100 * d['mean'] for _, _, d in refs]
    lo, hi = min(vals) - 2, max(vals) + 2
    f = C.Frame(0.6, 6.6, lo, hi, w=460, h=300, left=46, right=96, top=14, bottom=42)
    body = f.axes([1, 2, 3, 4, 5.2], C.nice_ticks(lo, hi, 6), xlabel='power p', xfmt=lambda v: {1: '2', 2: '4', 3: '8', 4: '16', 5.2: 'mixed'}[v],
                  yfmt=lambda v: '0' if abs(v) < 1e-9 else f'{v:+g}')
    body.append(f'<line class="zero" x1="{f.l}" x2="{f.w - f.r}" y1="{f.Y(0):.1f}" y2="{f.Y(0):.1f}"/>')
    ends = [(f.w - f.r + 4, f.Y(0) + 4, 'exact octave')]
    for k, lab, d in refs:
        y = 100 * d['mean']
        body.append(f'<line class="refdot" x1="{f.l}" x2="{f.w - f.r}" y1="{f.Y(y):.1f}" y2="{f.Y(y):.1f}"/>')
        tip = f'{lab} [{k}] (plain control, 512 inputs): {y:+.1f} points over {d["k"]} seed(s)'
        body.append(f'<rect class="hit" data-tip="{esc(tip)}" tabindex="0" x="{f.l}" y="{f.Y(y) - 5:.1f}" width="{f.w - f.l - f.r}" height="10"/>')
        ends.append((f.w - f.r + 4, f.Y(y) + 4, lab))
    for lab, dash, mk, pts in pts_all:
        line = [(x, y) for x, y, _, k, _ in pts if k != 'mix' and x < 5]
        body.append(C.polyline(f, line, 'r-pow', width=2, dash=dash))
        for x, y, s, k, d in pts:
            if s:
                body.append(f'<line class="r-pow whisk" x1="{f.X(x):.1f}" x2="{f.X(x):.1f}" y1="{f.Y(y - 2 * s):.1f}" y2="{f.Y(y + 2 * s):.1f}"/>')
            tip = f'{LABEL.get(k, k)} [{k}]: {y:+.1f}' + (f' ± {s:.1f}' if s else '') + f' points over {d["k"]} seed(s)'
            body.append(C.dot(f, x, y, 'r-pow' + mk, r=4.5, tip=tip))
        if line:
            ends.append((f.X(line[-1][0]) + 8, f.Y(line[-1][1]) + 4, lab))
    for lx, ly, t in C_spread(ends):
        body.append(f'<text class="lbl small" x="{lx:.1f}" y="{ly:.1f}">{esc(t)}</text>')
    return C.svg(f, body, f'Paired success change against the power p, {n}x{n}')


def C_spread(items, gap=11.0, char=5.6):
    out = []
    for x, y, t in sorted(items, key=lambda i: i[1]):
        for _ in range(30):
            hit = [oy for ox, oy, ot in out if x < ox + char * len(ot) + 4 and ox < x + char * len(t) + 4 and abs(oy - y) < gap]
            if not hit:
                break
            y = max(hit) + gap
        out.append((x, y, t))
    return out


# ---------------------------------------------------------------- failure split
def chart_fail(S, n, keys, w=520):
    rows = [k for k in keys if pm(S, n, k, 'wall_rate') and pm(S, n, k, 'never_arrived_rate')]
    if not rows:
        return '<p class="empty">No runs yet.</p>'
    vals = [0.0]
    for k in rows:
        for m in ('wall_rate', 'never_arrived_rate'):
            vals.append(100 * pm(S, n, k, m)['mean'])
    lo, hi = max(min(vals) - 2, -CLIP), min(max(vals) + 2, CLIP)
    cx = lambda v: min(max(v, lo), hi)  # noqa: E731
    rowh = 22
    f = C.Frame(lo, hi, 0, 1, w=w, h=16 + rowh * len(rows) + 40, left=250, right=14, top=12, bottom=40)
    body = []
    for t in C.nice_ticks(lo, hi, 6):
        x = f.X(t)
        body.append(f'<line class="{"zero" if abs(t) < 1e-12 else "grid"}" x1="{x:.1f}" x2="{x:.1f}" y1="{f.t}" y2="{f.h - f.b}"/>')
        body.append(f'<text class="ax" x="{x:.1f}" y="{f.h - f.b + 16}" text-anchor="middle">{"0" if abs(t) < 1e-12 else f"{t:+g}"}</text>')
    body.append(f'<text class="ax axt" x="{(f.l + f.w - f.r) / 2:.1f}" y="{f.h - 4}" text-anchor="middle">change in share of runs (points)</text>')
    for i, k in enumerate(rows):
        y = f.t + i * rowh + rowh / 2
        cls = role(k)
        dw, dn = pm(S, n, k, 'wall_rate'), pm(S, n, k, 'never_arrived_rate')
        xw, xn = f.X(cx(100 * dw['mean'])), f.X(cx(100 * dn['mean']))
        body.append(f'<text class="ax rowlab" x="{f.l - 8}" y="{y + 4:.1f}" text-anchor="end">{esc(LABEL.get(k, k))}</text>')
        body.append(f'<line class="base" x1="{min(xw, xn):.1f}" x2="{max(xw, xn):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>')
        tip = (f'{LABEL.get(k, k)} [{k}]: reached the goal only through a wall {100 * dw["mean"]:+.1f} points, never reached it '
               f'{100 * dn["mean"]:+.1f} points (over {dw["k"]} seed(s), against the exact octave of the same seed)')
        body.append(f'<g class="pt" data-tip="{esc(tip)}" tabindex="0"><rect class="hit" x="{f.l}" y="{y - rowh / 2:.1f}" width="{f.w - f.l - f.r}" height="{rowh}"/>'
                    f'<circle class="{cls} dot ring" cx="{xw:.1f}" cy="{y:.1f}" r="4.5"/>'
                    f'<circle class="{cls} dot hollow" cx="{xn:.1f}" cy="{y:.1f}" r="4.5"/></g>')
    return C.svg(f, body, f'Change in wall crossings and never-arrived runs, {n}x{n}')


# ---------------------------------------------------------------- near-wall slope vs local minima
def chart_tradeoff(S, n, keys):
    pts = []
    for k in keys:
        a, b = pm(S, n, k, 'slope_2_8'), pm(S, n, k, 'bumps')
        if a and b:
            pts.append((a['mean'], b['mean'], k))
    if len(pts) < 3:
        return '<p class="empty">Not enough runs yet.</p>'
    xs = [p[0] for p in pts] + [0]
    ys = [p[1] for p in pts] + [0]
    xr = max(xs) - min(xs) or 0.1
    yr = max(ys) - min(ys) or 0.1
    f = C.Frame(min(xs) - 0.08 * xr, max(xs) + 0.08 * xr, min(ys) - 0.1 * yr, max(ys) + 0.1 * yr, w=460, h=320, left=50, right=16, top=14, bottom=44)
    body = f.axes(C.nice_ticks(f.x0, f.x1, 5), C.nice_ticks(f.y0, f.y1, 5), xlabel='change in near-wall slope S |∇T| (target 1)',
                  ylabel='change in local minima per long path', xfmt=lambda v: '0' if abs(v) < 1e-9 else f'{v:+.2g}',
                  yfmt=lambda v: '0' if abs(v) < 1e-9 else f'{v:+.2g}')
    body.append(f'<line class="zero" x1="{f.X(0):.1f}" x2="{f.X(0):.1f}" y1="{f.t}" y2="{f.h - f.b}"/>')
    body.append(f'<line class="zero" x1="{f.l}" x2="{f.w - f.r}" y1="{f.Y(0):.1f}" y2="{f.Y(0):.1f}"/>')
    for x, y, k in pts:
        s = pm(S, n, k)
        tip = (f'{LABEL.get(k, k)} [{k}]: near-wall slope {x:+.3f}, local minima {y:+.2f} per path'
               + (f', success {100 * s["mean"]:+.1f} points' if s else ''))
        body.append(C.dot(f, x, y, role(k), r=4.5, tip=tip))
    return C.svg(f, body, f'Near-wall slope change against local-minimum change, {n}x{n}')


def chart_mech(S, n, xm='eik_res', ym='uphill', xlabel='eikonal residual along true paths, median |S |∇T| − 1|',
               ylabel='uphill share of the path', ylog=True):
    """Every run of the maze: the octave reference trainings (ink) and every spec x seed (colour by idea)."""
    pts = []
    for s_, ref in (S['sizes'].get(n) or {}).get('reference', {}).items():
        for r in ref.get('records', []):
            if r.get(xm) is not None and r.get(ym) is not None:
                pts.append((r[xm], r[ym], 'r-ink', f'exact octave, seed {s_} [{r["id"]}]', r.get('success_rate')))
    for key, e in (S['sizes'].get(n) or {}).get('specs', {}).items():
        for s_, r in e['per_seed'].items():
            if r.get(xm) is not None and r.get(ym) is not None:
                pts.append((r[xm], r[ym], role(key), f'{LABEL.get(key, key)} [{key}], seed {s_}', r.get('success_rate')))
    if len(pts) < 3:
        return '<p class="empty">Not enough runs yet.</p>'
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    xr = (max(xs) - min(xs)) or 0.1
    if ylog:
        y0, y1 = max(min(ys) / 1.6, 1e-3), max(ys) * 1.6
    else:
        yr = (max(ys) - min(ys)) or 0.1
        y0, y1 = min(ys) - 0.08 * yr, max(ys) + 0.08 * yr
    f = C.Frame(max(min(xs) - 0.08 * xr, 0), max(xs) + 0.08 * xr, y0, y1, w=460, h=320, left=54, right=16, top=14, bottom=44, ylog=ylog)
    if ylog:
        yt = [v for v in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5) if y0 <= v <= y1]
        yfmt = lambda v: f'{100 * v:g} %'  # noqa: E731
    else:
        yt = C.nice_ticks(y0, y1, 5)
        yfmt = lambda v: f'{100 * v:.0f} %'  # noqa: E731
    body = f.axes(C.nice_ticks(f.x0, f.x1, 5), yt, xlabel=xlabel, ylabel=ylabel, xfmt=lambda v: f'{v:.2g}', yfmt=yfmt)
    for x, y, cls, name, sr in sorted(pts, key=lambda p: p[2] == 'r-ink'):
        tip = f'{name}: {xm} {x:.3f}, {ym} {y:.3f}' + (f', success {100 * sr:.1f} %' if sr is not None else '')
        body.append(C.dot(f, x, y, cls, r=4.5, tip=tip))
    return C.svg(f, body, f'{ylabel} against {xlabel}, every run, {n}x{n}')


MID_ORDER = ['octave', 'Asc@new128', 'Apowmix@all', 'Rpowmix@half', 'Rtri@all', 'Agtri@allx2', 'Apow16@axis', 'Agtri@topx8']


def chart_midpoint(MS, n='22', measure='rate', xlabel='success (planning seeds 0/1/2)', scale=100.0, unit='%'):
    """Rows = base fields; open dot = the base run (no midpoint loss), filled dot = forked at epoch 2,000 and trained with the midpoint
    loss; means over the seeds that have the midpoint run; vertical rules = the exact octave without / with the loss on the same seeds."""
    specs = (MS.get('specs') or {}).get(n) or {}
    rows = [k for k in MID_ORDER if k in specs] + [k for k in specs if k not in MID_ORDER]
    if not rows:
        return '<p class="empty">No midpoint runs yet.</p>'
    bkey = 'base_rate' if measure == 'rate' else 'base_' + measure

    def mean(e, key):
        v = [x[key] for x in e['seeds'].values() if x.get(key) is not None]
        return float(np.mean(v)) if v else None
    vals = []
    for k in rows:
        for key in (measure, bkey):
            v = mean(specs[k], key)
            if v is not None:
                vals.append(scale * v)
    lo, hi = min(vals), max(vals)
    pad = max((hi - lo) * 0.08, 1.0)
    lo, hi = max(lo - pad, 0), hi + pad
    rowh = 26
    f = C.Frame(lo, hi, 0, 1, w=640, h=20 + rowh * len(rows) + 44, left=250, right=18, top=16, bottom=42)
    body = []
    for t in C.nice_ticks(lo, hi, 6):
        x = f.X(t)
        body.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{f.t}" y2="{f.h - f.b}"/>')
        body.append(f'<text class="ax" x="{x:.1f}" y="{f.h - f.b + 16}" text-anchor="middle">{t:g}{unit if unit == "%" else ""}</text>')
    o = specs.get('octave')
    if o:
        for key, cls, lab in ((bkey, 'refdot', 'octave'), (measure, 'zero', 'octave + midpoint')):
            v = mean(o, key)
            if v is not None:
                x = f.X(scale * v)
                body.append(f'<line class="{cls}" x1="{x:.1f}" x2="{x:.1f}" y1="{f.t - 4}" y2="{f.h - f.b}"/>')
    body.append(f'<text class="ax axt" x="{(f.l + f.w - f.r) / 2:.1f}" y="{f.h - 4}" text-anchor="middle">{esc(xlabel)}</text>')
    for i, k in enumerate(rows):
        e = specs[k]
        y = f.t + i * rowh + rowh / 2
        cls = role(k) if k != 'octave' else 'r-ink'
        b0, b1 = mean(e, bkey), mean(e, measure)
        body.append(f'<text class="ax rowlab" x="{f.l - 10}" y="{y + 4:.1f}" text-anchor="end">{esc("exact octave" if k == "octave" else LABEL.get(k, k))}</text>')
        if b0 is not None and b1 is not None:
            body.append(f'<line class="base" x1="{f.X(scale * b0):.1f}" x2="{f.X(scale * b1):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>')
        seeds = ', '.join(f'{s}: {scale * (x.get(bkey) or 0):.1f} → {scale * (x.get(measure) or 0):.1f}' for s, x in sorted(e['seeds'].items()))
        tip = f'{"exact octave" if k == "octave" else LABEL.get(k, k)} [{k}], {xlabel}: without the midpoint loss → with it, by seed: {seeds}'
        if measure == 'rate' and e.get('vs_octave_mid_mean') is not None:
            tip += f'; with the loss vs the octave with the loss {100 * e["vs_octave_mid_mean"]:+.1f} points'
        parts = []
        if b0 is not None:
            parts.append(f'<circle class="{cls} dot hollow" cx="{f.X(scale * b0):.1f}" cy="{y:.1f}" r="5"/>')
        if b1 is not None:
            parts.append(f'<circle class="{cls} dot ring" cx="{f.X(scale * b1):.1f}" cy="{y:.1f}" r="5"/>')
        body.append(f'<g class="pt" data-tip="{esc(tip)}" tabindex="0"><rect class="hit" x="{f.l}" y="{y - rowh / 2:.1f}" width="{f.w - f.l - f.r}" height="{rowh}"/>{"".join(parts)}</g>')
    return C.svg(f, body, f'{xlabel} without and with the midpoint loss, {n}x{n}')


def table_md(md_path):
    out, rows, in_table = [], [], False

    def flush():
        if not rows:
            return
        head, body = rows[0], [r for r in rows[2:]]
        out.append('<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{esc(h)}</th>' for h in head) + '</tr></thead><tbody>'
                   + ''.join('<tr>' + ''.join(f'<td>{esc(c.strip("`"))}</td>' for c in r) + '</tr>' for r in body) + '</tbody></table></div>')
        rows.clear()

    for line in md_path.read_text().splitlines():
        if line.startswith('|'):
            rows.append([c.strip() for c in line.strip().strip('|').split('|')])
            in_table = True
            continue
        if in_table:
            flush()
            in_table = False
        if line.startswith('## '):
            out.append(f'<h4>{esc(line[3:])}</h4>')
        elif line.strip() and not line.startswith('# '):
            out.append(f'<p>{esc(line)}</p>')
    flush()
    return ''.join(out)


def screen_table():
    p = OUT / 'screen.json'
    if not p.exists():
        return ''
    sc = json.loads(p.read_text())
    head = ['spec', 'maze', 'arc near wall / open', 'arc share of sharp columns', 'fold across grid lines', 'linear speed readout R²', 'effective rank', 'RMS']
    rows = []
    for key, r in sorted(sc.items(), key=lambda kv: (kv[0].split('|')[2], int(kv[0].split('|')[0]))):
        n, _, s = key.split('|')
        rows.append([s, f'{n}×{n}', f'{r["arc_ratio"]:.3f}', '–' if 'arc_sharp_share' not in r else f'{r["arc_sharp_share"]:.2f}', f'{r["fold_ratio"]:.3f}',
                     f'{r["speed_probe_r2"]:.3f}', f'{r["rank_effective"]:.0f}', f'{r["rms"]:.2f}'])
    return ('<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{esc(h)}</th>' for h in head) + '</tr></thead><tbody>'
            + ''.join('<tr>' + ''.join(f'<td>{esc(c)}</td>' for c in r) + '</tr>' for r in rows) + '</tbody></table></div>')


def usage_table():
    p = OUT / 'usage.json'
    if not p.exists():
        return ''
    us = json.loads(p.read_text())
    head = ['maze', 'seed', 'spec', 'weight share of the sharp columns', 'their share of the input', 'octave reference on the same columns']
    rows = []
    for key, r in sorted(us.items(), key=lambda kv: (kv[1]['n'], kv[1]['seed'], kv[1]['spec'])):
        if '|plan' in key:
            continue
        w, rw = r['weight'], r.get('ref_weight')
        rows.append([f'{r["n"]}×{r["n"]}', str(r['seed']), r['spec'], f'{w["mean"]:.3f}', f'{w["uniform"]:.3f}', '–' if not rw else f'{rw["mean"]:.3f}'])
    return ('<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{esc(h)}</th>' for h in head) + '</tr></thead><tbody>'
            + ''.join('<tr>' + ''.join(f'<td>{esc(c)}</td>' for c in r) + '</tr>' for r in rows) + '</tbody></table></div>')


def run_table(S):
    head = ['maze', 'spec', 'seed', 'run id', 'success']
    rows = []
    for n in SIZES:
        for key, e in sorted((S['sizes'].get(n) or {}).get('specs', {}).items()):
            if key.startswith(('ctx:', 'ntrl')):
                continue
            for s, r in sorted(e['per_seed'].items()):
                rows.append([f'{n}×{n}', key, s, r['id'], f'{r["success"]}/{r["runs"]}'])
    return ('<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{esc(h)}</th>' for h in head) + '</tr></thead><tbody>'
            + ''.join('<tr>' + ''.join(f'<td><code>{esc(c)}</code></td>' if j == 3 else f'<td>{esc(c)}</td>' for j, c in enumerate(r)) + '</tr>' for r in rows)
            + '</tbody></table></div>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--status', default='')
    a = ap.parse_args()
    S = json.loads((OUT / 'summary.json').read_text())
    charts = dict(shapes=chart_shapes())
    for n in SIZES:
        charts[f'power{n}'] = chart_power(S, n)
        allkeys = [k for _, keys in GROUPS for k in keys]
        charts[f'trade{n}'] = chart_tradeoff(S, n, allkeys)
        charts[f'mech{n}'] = chart_mech(S, n)
        charts[f'mechs{n}'] = chart_mech(S, n, xm='uphill', ym='success_rate', xlabel='uphill share of the true path',
                                         ylabel='success', ylog=False)
    for gi, (title, keys) in enumerate(GROUPS):
        charts[f'g{gi}'] = chart_rows2(S, keys)
    charts['fail'] = chart_fail2(S, [k for _, keys in GROUPS[:5] for k in keys])
    charts['stall'] = chart_rows2(S, [k for _, keys in GROUPS[:5] for k in keys], m='stop_open',
                                  xlabel='change in runs stopped in an open corridor, no wall within 0.15 cell (points, planning seed 0)')
    MS = json.loads((HERE.parent / 'sharp_midpoint/report/summary.json').read_text()) if (HERE.parent / 'sharp_midpoint/report/summary.json').exists() else {}
    charts['mid_rate'] = chart_midpoint(MS)
    charts['mid_uphill'] = chart_midpoint(MS, measure='uphill', xlabel='share of the true paths running uphill (%)')
    legends = dict(
        mid=C.legend([('r-ink hollow', 'without the midpoint loss (open)', 'dot'), ('r-ink', 'with it, forked at epoch 2,000 (filled)', 'dot'),
                      ('refdot', 'exact octave', 'rule'), ('zero', 'exact octave + midpoint loss', 'rule')]),
        mech=C.legend([('r-ink', 'exact octave (reference trainings)', 'dot'), ('r-pow', 'powers', 'dot'), ('r-tri', 'triangle', 'dot'),
                       ('r-gate', 'gates', 'dot'), ('r-ctl', 'plain controls', 'dot')]),
        roles=C.legend([('r-pow', 'idea 1: powers', 'dot'), ('r-tri', 'idea 2: triangle', 'dot'), ('r-gate', 'idea 3: gates', 'dot'),
                        ('r-ctl', 'plain controls', 'dot'), ('r-ctx', 'earlier runs', 'dot'),
                        ('band', 'one seed: ±2 × training noise', 'band')]),
        power=C.legend([('r-pow', 'half the rows replaced', 'line'), ('r-pow', 'every row replaced', 'dash'), ('r-pow', 'added (512 inputs)', 'dot-line'),
                        ('refdot', 'plain controls of the added width (labelled)', 'rule')]),
        fail=C.legend([('r-ctl', 'reached the goal only through a wall', 'dot'), ('r-ctl hollow', 'never reached the goal', 'dot')]),
    )
    tables = table_md(OUT / 'tables.md') if (OUT / 'tables.md').exists() else ''
    page = TX.render(S=S, charts=charts, legends=legends, tables=tables, screen=screen_table(), usage=usage_table(), runs=run_table(S), status=a.status,
                     groups=GROUPS, pm=lambda n, k, m='success_rate': pm(S, n, k, m), spec=lambda n, k: spec(S, n, k), label=LABEL)
    (OUT / 'sharp_features.html').write_text(page)
    print('wrote', OUT / 'sharp_features.html', len(page) // 1024, 'KB')


if __name__ == '__main__':
    main()
