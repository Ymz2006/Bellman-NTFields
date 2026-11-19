"""Findings page of the mobile-arm study: report/runs.json (report9.py) + pagetext9.py -> report/mobile_arm.html (published as an artifact).
Charts are SVG drawn here, colored by CSS tokens (both themes), hover text in <title>.

    python3 jeffrey/mobile_arm/report9.py && python3 jeffrey/mobile_arm/page9.py
"""
import base64
import html
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pagetext9 as PT  # noqa: E402

HOUSES = ['allensville', 'lakeville', 'klickitat']
VAR = ['ntrl', 'ctl', 'mid']
VNAME = dict(ntrl='ntrl-demo (baseline)', ctl='control (resumed, no term)', mid='ntrl-demo + midpoint loss', straight='straight segment')
VCOL = dict(ntrl='--s1', ctl='--s4', mid='--s3', straight='--s5')

CSS = r"""
/* Layout: one 72ch reading column inside a 1120px frame; charts take the frame width; tables hide in <details>. Tokens as the earlier study pages. */
:root {
  --bg: #f4f5f2; --surface: #fcfcfb; --ink: #121513; --ink2: #4b524e; --muted: #7b827e; --rule: #dcdfdb; --grid: #e7e9e5; --base: #c3c7c2;
  --accent: #17654d; --wash: #e6efe9;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #8d928e; --s5: #c9a227;
  --ok: #3b7d5c; --stall: #e9b44c; --wall: #d03b3b; --inner: #8e5cc4;
  --f-display: "Archivo", "Helvetica Neue", Arial, sans-serif;
  --f-body: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  --f-mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #111412; --surface: #1a1b1a; --ink: #edf0ed; --ink2: #b7beb9; --muted: #8b928e; --rule: #2d312e; --grid: #262a27; --base: #3a3f3b;
    --accent: #6fd3ad; --wash: #1c2a24;
    --s1: #3987e5; --s2: #d95926; --s3: #22b07f; --s4: #9aa09c; --s5: #d4b03d;
    --ok: #4f9d76; --stall: #c99a33; --wall: #e66767; --inner: #a77de0;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #111412; --surface: #1a1b1a; --ink: #edf0ed; --ink2: #b7beb9; --muted: #8b928e; --rule: #2d312e; --grid: #262a27; --base: #3a3f3b;
  --accent: #6fd3ad; --wash: #1c2a24;
  --s1: #3987e5; --s2: #d95926; --s3: #22b07f; --s4: #9aa09c; --s5: #d4b03d;
  --ok: #4f9d76; --stall: #c99a33; --wall: #e66767; --inner: #a77de0;
  color-scheme: dark;
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font: 400 15.5px/1.62 var(--f-body); margin: 0; }
.page { max-width: 1120px; margin: 0 auto; padding-block: 36px 72px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); display: grid; gap: 40px; }
.col { max-width: 72ch; min-width: 0; }
header { display: grid; gap: 12px; }
.eyebrow { font: 500 12px/1.4 var(--f-mono); letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }
h1 { font: 700 clamp(28px, 4.6vw, 42px)/1.08 var(--f-display); letter-spacing: -0.012em; margin: 0; text-wrap: balance; }
h2 { font: 650 23px/1.2 var(--f-display); margin: 0 0 8px; text-wrap: balance; }
h3 { font: 650 16.5px/1.3 var(--f-display); margin: 14px 0 4px; }
p { margin: 0 0 12px; }
.lede { font-size: 17px; line-height: 1.55; color: var(--ink); }
section { display: grid; gap: 12px; min-width: 0; }
code { font: 13px var(--f-mono); background: var(--wash); padding: 1px 4px; border-radius: 4px; }
a { color: var(--accent); }
.answers { display: grid; border: 1px solid var(--rule); border-radius: 6px; background: var(--surface); }
.answers > div { display: grid; grid-template-columns: minmax(0, 16em) minmax(0, 1fr); gap: 16px; padding: 12px 16px; border-top: 1px solid var(--rule); }
.answers > div:first-child { border-top: none; }
.answers .q { font-weight: 600; font-size: 14.5px; }
.answers .a { color: var(--ink2); font-size: 14.5px; min-width: 0; }
.answers .a b { color: var(--ink); }
@media (max-width: 640px) { .answers > div { grid-template-columns: minmax(0, 1fr); gap: 4px; } }
.chart { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 14px 14px 8px; min-width: 0; }
.chart .ctitle { font-weight: 600; font-size: 14.5px; margin: 0 0 2px; }
.chart .csub { font-size: 13px; color: var(--ink2); margin: 0 0 8px; max-width: 100ch; }
.chart svg { display: block; width: 100%; height: auto; overflow: visible; }
.chart svg text { font-family: var(--f-body); }
.ax { fill: var(--muted); font-size: 10.5px; font-family: var(--f-mono) !important; }
.lab { fill: var(--ink2); font-size: 12.5px; }
.small { fill: var(--ink2); font-size: 11px; }
.val { fill: var(--ink); font-size: 11px; font-family: var(--f-mono) !important; }
.grid { stroke: var(--grid); stroke-width: 1; }
.base { stroke: var(--base); stroke-width: 1; }
.ln { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.dash { stroke-dasharray: 4 3; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 12.5px; color: var(--ink2); margin: 2px 0 8px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { display: inline-block; width: 12px; height: 12px; border-radius: 2px; }
.multi { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 320px), 1fr)); gap: 10px; }
.panel { border: 1px solid var(--rule); border-radius: 5px; padding: 8px 8px 2px; min-width: 0; background: var(--surface); }
.panel h4 { font: 600 13px var(--f-display); margin: 0 0 2px 2px; }
figure { margin: 0; display: grid; gap: 6px; }
figure img { width: 100%; height: auto; border-radius: 6px; border: 1px solid var(--rule); background: #ffffff; }
figcaption { font-size: 13px; color: var(--ink2); }
.figs { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 330px), 1fr)); gap: 12px; }
details { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; }
details > summary { cursor: pointer; padding: 10px 14px; font-weight: 600; font-size: 14.5px; }
details > summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.tbl { overflow-x: auto; padding: 0 14px 14px; }
table { border-collapse: collapse; font-size: 12.5px; font-variant-numeric: tabular-nums; min-width: 760px; }
th, td { text-align: right; padding: 4px 8px; border-bottom: 1px solid var(--rule); white-space: nowrap; }
th:first-child, td:first-child, th.l, td.l { text-align: left; }
th { font: 600 11.5px var(--f-mono); color: var(--muted); }
td.num { font-family: var(--f-mono); }
ul.tight, ol.tight { margin: 0 0 12px; padding-left: 20px; display: grid; gap: 6px; }
.notep { font-size: 13.5px; color: var(--ink2); }
.math { font: 13px/1.55 var(--f-mono); background: var(--wash); border-radius: 5px; padding: 10px 14px; overflow-x: auto; white-space: pre; max-width: 100%; }
footer { color: var(--muted); font-size: 12.5px; border-top: 1px solid var(--rule); padding-top: 12px; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""


def esc(s):
    return html.escape(str(s))


def f(v, k=1):
    if v is None:
        return '–'
    out = f'{v:.{k}f}'
    return out[1:] if out.startswith('-') and float(out) == 0 else out


def legend(keys):
    return '<div class="legend">' + ''.join(f'<span><i style="background: var({VCOL[k]})"></i>{esc(VNAME[k])}</span>' for k in keys) + '</div>'


def seeds_of(R, h, v):
    """Run summaries of field v in house h on the paired seeds (all three fields finished); all seeds while no seed is paired."""
    runs = R['runs'].get(h, {}).get(v, {})
    paired = set(R.get('paired_seeds', {}).get(h, []))
    return [runs[sd] for sd in sorted(runs) if not paired or sd in paired]


# ---- charts ------------------------------------------------------------------------------------------------------------------------------
def chart_solved(R):
    """Horizontal bars per house: straight, ntrl, ctl, mid (pooled), dots = single seeds."""
    rows = []
    for h in HOUSES:
        if h not in R['pooled'] or not R['pooled'][h]:
            continue
        hh = R['houses'][h]
        rows.append(('head', f'{hh["name"]} ({hh["E"]:g} cells, {hh["queries"]} queries)'))
        rows.append(('straight', h, hh['straight'], []))
        for v in VAR:
            p = R['pooled'][h].get(v)
            if p:
                rows.append((v, h, p['solved'], p['solved_seed']))
    W, left, right, rh = 760, 210, 70, 22
    H = 16 + sum(26 if r[0] == 'head' else rh for r in rows) + 26
    x = lambda v: left + (W - left - right) * v / 100
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Queries solved per house and field">']
    y = 10
    for t in range(0, 101, 20):
        out.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="8" y2="{H - 22}"/><text class="ax" x="{x(t):.1f}" y="{H - 8}" text-anchor="middle">{t} %</text>')
    for r in rows:
        if r[0] == 'head':
            y += 8
            out.append(f'<text class="lab" x="0" y="{y + 10}" style="font-weight: 600">{esc(r[1])}</text>')
            y += 18
            continue
        v, h, val, seeds = r
        out.append(f'<text class="small" x="{left - 8}" y="{y + 14}" text-anchor="end">{esc(VNAME[v])}</text>')
        out.append(f'<rect x="{left}" y="{y + 4}" width="{x(val) - left:.1f}" height="{rh - 8}" rx="2" style="fill: var({VCOL[v]}); opacity: 0.85"><title>{esc(VNAME[v])}: {val:.1f} % solved'
                   + (f' (seeds {", ".join(f"{s:.1f}" for s in seeds)})' if seeds else '') + '</title></rect>')
        for s in seeds:
            out.append(f'<circle cx="{x(s):.1f}" cy="{y + rh / 2:.1f}" r="3.2" style="fill: var(--surface); stroke: var(--ink); stroke-width: 1.2"><title>one training seed: {s:.1f} %</title></circle>')
        out.append(f'<text class="val" x="{max([x(val)] + [x(sv) for sv in seeds]) + 8:.1f}" y="{y + 15}">{val:.1f}</text>')
        y += rh
    out.append('</svg>')
    return '\n'.join(out)


def line_panel(title, xs, series, ymin, ymax, yfmt, ref=None, w=340, h=210, xlab=None):
    """series: list of (key, label, values (None allowed), dashed)."""
    left, right, top, bot = 40, 10, 10, 34
    X = lambda i: left + (w - left - right) * (i / max(len(xs) - 1, 1))
    Y = lambda v: top + (h - top - bot) * (1 - (v - ymin) / (ymax - ymin))
    out = [f'<div class="panel"><h4>{esc(title)}</h4><svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}">']
    ticks = [ymin + (ymax - ymin) * k / 4 for k in range(5)]
    for t in ticks:
        out.append(f'<line class="grid" x1="{left}" x2="{w - right}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/><text class="ax" x="{left - 4}" y="{Y(t) + 3:.1f}" text-anchor="end">{yfmt(t)}</text>')
    if ref is not None:
        out.append(f'<line class="base" x1="{left}" x2="{w - right}" y1="{Y(ref):.1f}" y2="{Y(ref):.1f}" style="stroke: var(--ink2); stroke-dasharray: 2 2"/>')
    for i, lab in enumerate(xs):
        out.append(f'<text class="ax" x="{X(i):.1f}" y="{h - bot + 14}" text-anchor="middle">{esc(lab)}</text>')
    if xlab:
        out.append(f'<text class="ax" x="{(left + w - right) / 2:.1f}" y="{h - 4}" text-anchor="middle">{esc(xlab)}</text>')
    for key, label, vals, dashed in series:
        pts = [(X(i), Y(v), v, xs[i]) for i, v in enumerate(vals) if v is not None and ymin <= v <= ymax]
        if not pts:
            continue
        d = ' '.join(f'{"M" if k == 0 else "L"}{a:.1f} {b:.1f}' for k, (a, b, _, _) in enumerate(pts))
        out.append(f'<path class="ln{" dash" if dashed else ""}" d="{d}" style="stroke: var({VCOL[key]})"/>')
        for a, b, v, xl in pts:
            out.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="3.4" style="fill: var({VCOL[key]}); stroke: var(--surface); stroke-width: 1.5"><title>{esc(label)}, {esc(xl)}: {yfmt(v, True)}</title></circle>')
    out.append('</svg></div>')
    return '\n'.join(out)


def chart_bins(R):
    panels = []
    for h in HOUSES:
        P = R['pooled'].get(h)
        if not P:
            continue
        bins = list(next(iter(P.values()))['by_bin'])
        hh = R['houses'][h]
        series = [('straight', VNAME['straight'], [hh['straight_by_bin'].get(b) for b in bins], True)]
        series += [(v, VNAME[v], [P[v]['by_bin'].get(b) for b in bins], False) for v in VAR if v in P]
        panels.append(line_panel(hh['name'], [b.split('-')[0] + '–' + b.split('-')[1] for b in bins], series, 0, 100, lambda t, full=False: f'{t:.0f}' + (' %' if full else ''),
                                 xlab='base distance T*_SE2 (cells)'))
    return '<div class="multi">' + '\n'.join(panels) + '</div>'


def chart_outcomes(R):
    """Stacked bars: solved / arrived through a wall / arrived through itself, floor or base / did not arrive."""
    rows = []
    for h in HOUSES:
        for v in VAR:
            S = seeds_of(R, h, v)
            if not S:
                continue
            n = sum(s['solved_n'][1] for s in S)
            solved = sum(s['solved_n'][0] for s in S)
            wall = sum(s['parts']['rect'] + s['parts']['arm'] for s in S)
            inner = sum(s['parts']['floor'] + s['parts']['base'] + s['parts']['self'] for s in S)
            stall = sum(round(s['stall'] * s['solved_n'][1] / 100) for s in S)
            arm = sum(s['parts']['arm'] for s in S)
            rows.append((R['houses'][h]['name'], v, n, solved, wall, inner, stall, arm))
    W, left, right, rh = 760, 250, 20, 24
    H = 8 + rh * len(rows) + 30
    x = lambda v: left + (W - left - right) * v / 100
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Outcome of every planning run">']
    for t in range(0, 101, 20):
        out.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="4" y2="{H - 24}"/><text class="ax" x="{x(t):.1f}" y="{H - 8}" text-anchor="middle">{t} %</text>')
    y = 6
    prev = None
    for name, v, n, solved, wall, inner, stall, arm in rows:
        lab = f'{name if name != prev else ""}  {VNAME[v]}'
        prev = name
        out.append(f'<text class="small" x="{left - 8}" y="{y + 15}" text-anchor="end">{esc(lab.strip())}</text>')
        acc = 0.0
        for val, col, what in ((solved, '--ok', 'solved'), (wall, '--wall', f'arrived through a wall (arm first {arm} of {wall})'), (inner, '--inner', 'arrived through its own base, the floor or itself'),
                               (stall, '--stall', 'did not arrive')):
            pc = 100 * val / n
            if pc <= 0:
                continue
            out.append(f'<rect x="{x(acc):.1f}" y="{y + 4}" width="{x(acc + pc) - x(acc):.1f}" height="{rh - 8}" style="fill: var({col})"><title>{esc(name)}, {esc(VNAME[v])}: {esc(what)} {pc:.1f} % ({val} of {n} runs)</title></rect>')
            acc += pc
        y += rh
    out.append('</svg>')
    lg = ('<div class="legend"><span><i style="background: var(--ok)"></i>solved</span><span><i style="background: var(--wall)"></i>arrived through a wall (base or arm)</span>'
          '<span><i style="background: var(--inner)"></i>arrived through its own base, the floor or itself</span><span><i style="background: var(--stall)"></i>did not arrive</span></div>')
    return lg + '\n'.join(out)


def chart_field(R):
    panels = []
    for h in HOUSES:
        P = R['runs'].get(h)
        if not P:
            continue
        hh = R['houses'][h]
        bins = list(hh['reference']['t_ref_over_t_se2_by_bin'])
        series = [('straight', 'reference path T_ref (upper bound)', [hh['reference']['t_ref_over_t_se2_by_bin'].get(b) for b in bins], True)]
        for v in VAR:
            if v in P:
                S = seeds_of(R, h, v)
                series.append((v, VNAME[v], [sum(s['field_by_bin'][b]['t_over_se2'] for s in S) / len(S) if all(b in s['field_by_bin'] for s in S) else None for b in bins], False))
        panels.append(line_panel(hh['name'], [b.split('-')[0] + '–' + b.split('-')[1] for b in bins], series, 0.6, 1.8, lambda t, full=False: f'{t:.2f}' if full else f'{t:.1f}',
                                 ref=1.0, xlab='base distance (cells); y = T / T*_SE2'))
    return '<div class="multi">' + '\n'.join(panels) + '</div>'


def chart_course(R):
    panels = []
    for h in HOUSES:
        P = R['runs'].get(h)
        if not P:
            continue
        epochs = [1000, 2000, 3000, 4000, 5000]
        series = []
        for v in VAR:
            if v not in P:
                continue
            S = seeds_of(R, h, v)
            vals = []
            for e in epochs:
                xs = []
                for s in S:
                    if e == 5000:
                        xs.append(100 * s['per_seed'][0] / (s['solved_n'][1] // 3))
                    else:
                        m = [p for ep, p, _, _ in s['validate'] if ep == e]
                        if m:
                            xs.append(m[0])
                        elif v != 'ntrl' and e <= 2000:
                            mm = [p for bs in seeds_of(R, h, 'ntrl') for ep, p, _, _ in bs['validate'] if ep == e]
                            xs += mm
                vals.append(sum(xs) / len(xs) if xs else None)
            series.append((v, VNAME[v], vals, False))
        lo = min(min(x for x in s[2] if x is not None) for s in series)
        hi = max(max(x for x in s[2] if x is not None) for s in series)
        panels.append(line_panel(R['houses'][h]['name'], [str(e) for e in epochs], series, max(0, 10 * int(lo / 10) - 10), min(100, 10 * int(hi / 10) + 10),
                                 lambda t, full=False: f'{t:.0f}' + (' %' if full else ''), xlab='epoch (planning seed 0)'))
    return '<div class="multi">' + '\n'.join(panels) + '</div>'


def chart_refresh(R):
    panels = []
    for h in HOUSES:
        P = R['runs'].get(h, {}).get('mid')
        if not P:
            continue
        S = [s for s in P.values() if s.get('refresh')]
        if not S:
            continue
        ep = [r[0] for r in S[0]['refresh']]
        reach = [sum(s['refresh'][k][1] for s in S if k < len(s['refresh'])) / len(S) for k in range(len(ep))]
        stall = [sum(s['refresh'][k][2] for s in S if k < len(s['refresh'])) / len(S) for k in range(len(ep))]
        idx = list(range(0, len(ep), max(1, len(ep) // 6)))
        panels.append(line_panel(R['houses'][h]['name'], [str(ep[i]) for i in idx], [('mid', 'drives reaching their goal', [reach[i] for i in idx], False),
                                                                                    ('ctl', 'drives that stall', [stall[i] for i in idx], True)],
                                 0, 100, lambda t, full=False: f'{t:.0f}' + (' %' if full else ''), xlab='refresh epoch'))
    lg = ('<div class="legend"><span><i style="background: var(--s3)"></i>drives that reach their goal</span>'
          '<span><i style="background: var(--s4)"></i>drives that stall (dashed)</span></div>')
    return lg + '<div class="multi">' + '\n'.join(panels) + '</div>'


def posthoc_rows(R):
    """Pooled forensics (planning seed 0 paths of every training seed) per house and field, from report/posthoc_runs.json."""
    ph = json.loads((HERE / 'report/posthoc_runs.json').read_text()) if (HERE / 'report/posthoc_runs.json').exists() else {}
    out = {}
    for h in HOUSES:
        for v in VAR:
            runs = [s['run'] for s in seeds_of(R, h, v)]
            rs = [ph[r] for r in runs if r in ph]
            if not rs:
                continue
            n = sum(R['houses'][h]['queries'] for _ in rs)
            num = lambda k: sum(x[k] for x in rs)
            med = lambda k: sum(x[k] for x in rs if x[k] is not None) / max(1, sum(1 for x in rs if x[k] is not None))
            out[(h, v)] = dict(n=n, seeds=len(rs), contacts=num('contacts'), arm_first=num('arm_first'), base_first=num('base_first'), internal=num('internal'),
                               reach_near=med('reach_near_wall_median'), reach_open=med('reach_open_median'), tucked_near=med('tucked_share_near_wall'),
                               tucked_open=med('tucked_share_open'), arm_base_gap=med('arm_first_base_clearance_median'), arm_reach=med('arm_first_reach_median'))
    return out


def chart_contacts(R):
    """Stacked bars of the first contact of runs that arrived through something: arm against a wall (base clear), base against a wall, internal."""
    rows = posthoc_rows(R)
    if not rows:
        return '<p class="notep">Forensics pending.</p>'
    W, left, right, rh = 760, 250, 20, 24
    H = 8 + rh * len(rows) + 30
    vmax = max(100 * (r['contacts']) / r['n'] for r in rows.values())
    top = max(10, 5 * int(vmax / 5) + 5)
    x = lambda v: left + (W - left - right) * v / top
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="First contact of runs that arrived through something">']
    step = 5 if top <= 30 else 10
    for t in range(0, top + 1, step):
        out.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="4" y2="{H - 24}"/><text class="ax" x="{x(t):.1f}" y="{H - 8}" text-anchor="middle">{t} %</text>')
    y, prev = 6, None
    for (h, v), r in rows.items():
        name = R['houses'][h]['name']
        lab = f'{name if name != prev else ""}  {VNAME[v]}'
        prev = name
        out.append(f'<text class="small" x="{left - 8}" y="{y + 15}" text-anchor="end">{esc(lab.strip())}</text>')
        acc = 0.0
        for val, col, what in ((r['arm_first'], '--s5', 'the arm hit a wall first, the base itself clear'), (r['base_first'], '--wall', 'the base hit a wall first'),
                               (r['internal'], '--inner', 'the arm hit the floor, the base or itself')):
            pc = 100 * val / r['n']
            if pc <= 0:
                continue
            out.append(f'<rect x="{x(acc):.1f}" y="{y + 4}" width="{x(acc + pc) - x(acc):.1f}" height="{rh - 8}" style="fill: var({col})"><title>{esc(name)}, {esc(VNAME[v])}: {esc(what)} {pc:.1f} % ({val} of {r["n"]} runs)</title></rect>')
            acc += pc
        y += rh
    out.append('</svg>')
    lg = ('<div class="legend"><span><i style="background: var(--s5)"></i>arm hit a wall first (base clear of it)</span><span><i style="background: var(--wall)"></i>base hit a wall first</span>'
          '<span><i style="background: var(--inner)"></i>arm hit the floor, the base or itself</span></div>')
    return lg + '\n'.join(out)


def table_posthoc(R):
    rows = posthoc_rows(R)
    head = ['house', 'field', 'training seeds', 'arrived through %', 'arm first %', 'base first %', 'internal %', 'base gap when the arm hits (cells)', 'arm reach when it hits (cells)',
            'arm reach near walls (cells)', 'arm reach in the open (cells)', 'tucked share near walls', 'tucked share in the open']
    out = ['<table><thead><tr>' + ''.join(f'<th class="{"l" if i < 2 else ""}">{esc(x)}</th>' for i, x in enumerate(head)) + '</tr></thead><tbody>']
    for (h, v), r in rows.items():
        pc = lambda k: f(100 * r[k] / r['n'])
        cells = [R['houses'][h]['name'], VNAME[v], r['seeds'], pc('contacts'), pc('arm_first'), pc('base_first'), pc('internal'), f(r['arm_base_gap'], 2), f(r['arm_reach'], 2),
                 f(r['reach_near'], 2), f(r['reach_open'], 2), f(r['tucked_near'], 2), f(r['tucked_open'], 2)]
        out.append('<tr>' + ''.join(f'<td class="{"l" if i < 2 else "num"}">{esc(c)}</td>' for i, c in enumerate(cells)) + '</tr>')
    out.append('</tbody></table>')
    return '\n'.join(out)


# ---- tables ------------------------------------------------------------------------------------------------------------------------------
def table_runs(R):
    head = ['house', 'field', 'seed', 'solved %', 'per planning seed', 'through a wall %', 'through itself / floor / base %', 'did not arrive %', 'path time / T_ref',
            'T / T_ref', 'T / T*_SE2', 'T below T*_SE2 %', 'Spearman', 'log-speed corr', 'run']
    out = ['<table><thead><tr>' + ''.join(f'<th class="{"l" if i in (0, 1, 14) else ""}">{esc(x)}</th>' for i, x in enumerate(head)) + '</tr></thead><tbody>']
    for h in HOUSES:
        for v in VAR:
            for seed, s in sorted(R['runs'].get(h, {}).get(v, {}).items()):
                n = s['solved_n'][1]
                wall = 100 * (s['parts']['rect'] + s['parts']['arm']) / n
                inner = 100 * (s['parts']['floor'] + s['parts']['base'] + s['parts']['self']) / n
                cells = [R['houses'][h]['name'], VNAME[v], seed, f(s['solved']), '/'.join(map(str, s['per_seed'])) + f' of {n // 3}', f(wall), f(inner), f(s['stall']),
                         f(s['time_over_ref'], 3), f(s['field']['t_over_ref_median'], 3), f(s['field']['t_over_se2_median'], 3), f(100 * s['field']['below_lower_bound']),
                         f(s['field']['spearman_ref'], 3), f(s['speed_fit']['log_speed_corr'], 3), s['run']]
                out.append('<tr>' + ''.join(f'<td class="{"l" if i in (0, 1) else ("l num" if i == 14 else "num")}">{esc(c)}</td>' for i, c in enumerate(cells)) + '</tr>')
    out.append('</tbody></table>')
    return '\n'.join(out)


def table_bins(R):
    out = []
    for h in HOUSES:
        P = R['pooled'].get(h)
        if not P:
            continue
        hh = R['houses'][h]
        bins = list(next(iter(P.values()))['by_bin'])
        out.append(f'<h3>{esc(hh["name"])}</h3><table style="min-width: 560px"><thead><tr><th class="l">field</th>' + ''.join(f'<th>{esc(b)} cells</th>' for b in bins) + '</tr></thead><tbody>')
        out.append('<tr><td class="l">straight segment</td>' + ''.join(f'<td class="num">{f(hh["straight_by_bin"].get(b), 0)}</td>' for b in bins) + '</tr>')
        for v in VAR:
            if v in P:
                out.append(f'<tr><td class="l">{esc(VNAME[v])}</td>' + ''.join(f'<td class="num">{f(P[v]["by_bin"].get(b), 0)}</td>' for b in bins) + '</tr>')
        out.append('</tbody></table>')
    return '\n'.join(out)


def table_speed(R):
    head = ['house', 'field', 'seed', 'base vs walls: corr', 'pred / true', 'arm vs walls: corr', 'pred / true', 'internal: corr', 'pred / true']
    out = ['<table><thead><tr>' + ''.join(f'<th class="{"l" if i < 2 else ""}">{esc(x)}</th>' for i, x in enumerate(head)) + '</tr></thead><tbody>']
    for h in HOUSES:
        for v in VAR:
            for seed, s in sorted(R['runs'].get(h, {}).get(v, {}).items()):
                bc = s['speed_fit']['by_constraint']
                cells = [R['houses'][h]['name'], VNAME[v], seed]
                for k in ('base_rect', 'arm_walls', 'internal'):
                    cells += [f(bc[k]['log_speed_corr'], 2), f(bc[k]['pred_over_true_median'], 2)] if k in bc else ['–', '–']
                out.append('<tr>' + ''.join(f'<td class="{"l" if i < 2 else "num"}">{esc(c)}</td>' for i, c in enumerate(cells)) + '</tr>')
    out.append('</tbody></table>')
    return '\n'.join(out)


def table_houses(R):
    head = ['house', 'cells across', 'queries', 'straight segment solved %', 'T_ref / T*_SE2 (median)', 'arm share of T_ref', 'x0 label set by base / arm / internal %', 'prepare run', 'SE(2) truth run']
    out = ['<table><thead><tr>' + ''.join(f'<th class="{"l" if i in (0, 7, 8) else ""}">{esc(x)}</th>' for i, x in enumerate(head)) + '</tr></thead><tbody>']
    for h in HOUSES:
        hh = R['houses'][h]
        a = hh['data']['x0_active']
        cells = [hh['name'], f(hh['E'], 0), hh['queries'], f(hh['straight']), f(hh['reference']['t_ref_over_t_se2_median'], 2), f(hh['reference']['tuck_share_of_t_ref_median'], 2),
                 f'{100 * a["base_rect"]:.0f} / {100 * a["arm_walls"]:.0f} / {100 * a["internal"]:.0f}', hh['prepare'], hh['house_prepare']]
        out.append('<tr>' + ''.join(f'<td class="{"l num" if i in (7, 8) else ("l" if i == 0 else "num")}">{esc(c)}</td>' for i, c in enumerate(cells)) + '</tr>')
    out.append('</tbody></table>')
    return '\n'.join(out)


def img(path):
    return 'data:image/png;base64,' + base64.b64encode(Path(path).read_bytes()).decode()


def main():
    R = json.load(open(HERE / 'report/runs.json'))
    T = PT.text(R)
    house_imgs = ''.join(f'<figure><img src="{img(HERE / "report" / ("house_" + R["houses"][h]["name"] + ".png"))}" alt="{esc(R["houses"][h]["name"])} floor plan with queries">'
                         f'<figcaption>{esc(R["houses"][h]["name"])}: one query per distance bin, start and goal configurations (base and arm spheres) and the reference path.</figcaption></figure>'
                         for h in HOUSES if (HERE / 'report' / ('house_' + R['houses'][h]['name'] + '.png')).exists())
    body = f"""<title>{T['title']}</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;650;700&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}</style>
<div class="page">
<header class="col">
  <div class="eyebrow">jeffrey/mobile_arm · 9-DOF wheeled base + arm · Gibson houses · every number from registered runs</div>
  <h1>{T['h1']}</h1>
  <p class="lede">{T['lede']}</p>
  {T.get('status', '')}
</header>
<section><h2>Answers</h2><div class="answers">{T['answers']}</div></section>
<section>
  <div class="col"><h2>The robot</h2>{T['robot']}</div>
  <figure><img src="{img(HERE / 'report/robot.png')}" alt="The mobile manipulator in three poses, plan and side views"><figcaption>{T['robot_cap']}</figcaption></figure>
</section>
<section>
  <div class="col"><h2>Queries solved</h2>{T['solved_intro']}</div>
  <div class="chart"><div class="ctitle">Queries solved, % of runs (queries × planning seeds 0, 1, 2), pooled over training seeds</div><div class="csub">{T['solved_sub']}</div>{chart_solved(R)}</div>
</section>
<section>
  <div class="col"><h2>By distance</h2>{T['bins_intro']}</div>
  <div class="chart"><div class="ctitle">Queries solved by the base's true travel distance, % of runs</div><div class="csub">{T['bins_sub']}</div>{legend(['straight'] + VAR)}{chart_bins(R)}</div>
</section>
<section>
  <div class="col"><h2>How the runs end</h2>{T['fails_intro']}</div>
  <div class="chart"><div class="ctitle">Outcome of every planning run, % of runs</div><div class="csub">{T['fails_sub']}</div>{chart_outcomes(R)}</div>
</section>
<section>
  <div class="col"><h2>Which part touches first</h2>{T['contacts_intro']}</div>
  <div class="chart"><div class="ctitle">Runs that arrived through something, by the part that touched first, % of runs (planning seed 0)</div><div class="csub">{T['contacts_sub']}</div>{chart_contacts(R)}</div>
  <div class="col">{T['contacts_after']}</div>
</section>
<section>
  <div class="col"><h2>What the loss does to the field</h2>{T['field_intro']}</div>
  <div class="chart"><div class="ctitle">Learned travel time T(start, goal) over the base's lower bound T*_SE2 (median by distance bin)</div><div class="csub">{T['field_sub']}</div><div class="legend"><span><i style="background: var(--s5)"></i>reference path, tuck + SE(2) optimum + untuck (upper bound)</span>{''.join(f'<span><i style="background: var({VCOL[k]})"></i>{esc(VNAME[k])}</span>' for k in VAR)}</div>{chart_field(R)}</div>
  <div class="col">{T['field_after']}</div>
</section>
<section>
  <div class="col"><h2>During training</h2>{T['course_intro']}</div>
  <div class="chart"><div class="ctitle">Queries solved with planning seed 0 at saved epochs (mean over training seeds)</div><div class="csub">{T['course_sub']}</div>{legend(VAR)}{chart_course(R)}</div>
  <div class="chart"><div class="ctitle">The midpoint loss's training drives (8,192 per refresh, mean over seeds)</div><div class="csub">{T['refresh_sub']}</div>{chart_refresh(R)}</div>
</section>
<section>
  <div class="col"><h2>The houses</h2>{T['houses_intro']}</div>
  <div class="figs">{house_imgs}</div>
</section>
<section class="col"><h2>How it was run</h2>{T['setup']}</section>
<section class="col"><h2>Caveats</h2>{T['caveats']}</section>
<details><summary>Every run: planning, failures, field accuracy, run ids</summary><div class="tbl">{table_runs(R)}</div></details>
<details><summary>Success by distance bin</summary><div class="tbl">{table_bins(R)}</div></details>
<details><summary>Speed fit on held-out pairs, by the constraint that sets the label</summary><div class="tbl">{table_speed(R)}</div></details>
<details><summary>First contacts and arm posture (planning seed 0 paths)</summary><div class="tbl">{table_posthoc(R)}</div></details>
<details><summary>Houses, queries and data</summary><div class="tbl">{table_houses(R)}</div></details>
<footer>{T['footer']}</footer>
</div>
"""
    (HERE / 'report/mobile_arm.html').write_text(body)
    print('wrote', HERE / 'report/mobile_arm.html', len(body), 'bytes')


if __name__ == '__main__':
    main()
