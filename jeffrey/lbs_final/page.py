"""report/summary.json (report.py) -> report/lbs_final.html, the artifact "2026-11-22 LBS Final Results"."""
import html
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
TITLE = '2026-11-22 LBS Final Results'
COLOR = dict(ntrl='var(--s1)', lbs='var(--s2)', rrtc='var(--s3)', lazyprm='var(--s4)')

CSS = '''
:root {
  /* Layout: one reading column inside a 1120px frame; charts take the frame width; big tables live in <details>. */
  --bg: #f5f6f8; --surface: #fcfcfb; --ink: #15181d; --ink-2: #4a505e; --ink-3: #777e8e; --rule: #dde1e7; --grid: #e7e9ed;
  --accent: #1f5fb4; --code-bg: #eceff3;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100;
  --font-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-head: "IBM Plex Sans Condensed", "IBM Plex Sans", system-ui, sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #121418; --surface: #1a1a19; --ink: #eceef2; --ink-2: #b6bbc7; --ink-3: #8a90a0; --rule: #2d3139; --grid: #2a2d33;
  --accent: #7fb0f0; --code-bg: #20232a; --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #121418; --surface: #1a1a19; --ink: #eceef2; --ink-2: #b6bbc7; --ink-3: #8a90a0; --rule: #2d3139; --grid: #2a2d33;
  --accent: #7fb0f0; --code-bg: #20232a; --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; color-scheme: dark; }
body { background: var(--bg); color: var(--ink); font-family: var(--font-body); font-size: 15.5px; line-height: 1.6; }
.wrap { max-width: 1120px; margin: 0 auto; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 28px 64px; }
header.top { display: grid; gap: 10px; padding-block: 8px 20px; border-bottom: 1px solid var(--rule); margin-bottom: 22px; }
.eyebrow { font-family: var(--font-mono); font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--ink-3); }
h1 { font-family: var(--font-head); font-weight: 600; font-size: clamp(28px, 4.2vw, 40px); line-height: 1.12; margin: 0; text-wrap: balance; }
h2 { font-family: var(--font-head); font-weight: 600; font-size: 25px; line-height: 1.2; margin: 44px 0 10px; padding-top: 14px; border-top: 1px solid var(--rule); text-wrap: balance; }
h3 { font-family: var(--font-head); font-weight: 600; font-size: 18px; margin: 22px 0 6px; }
p, li { max-width: 76ch; }
p { margin: 0 0 12px; }
.lede { font-size: 16.5px; color: var(--ink-2); max-width: 82ch; }
code { font-family: var(--font-mono); font-size: 0.88em; background: var(--code-bg); padding: 0 4px; border-radius: 3px; }
a { color: var(--accent); }
.status { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 10px 14px; margin: 8px 0; font-size: 14.5px; }
.answers { display: grid; border: 1px solid var(--rule); border-radius: 6px; background: var(--surface); margin: 6px 0 10px; }
.answers > div { display: grid; grid-template-columns: minmax(0, 13em) minmax(0, 1fr); gap: 16px; padding: 10px 16px; border-top: 1px solid var(--rule); font-size: 14.5px; }
.answers > div:first-child { border-top: none; }
.answers .q { font-weight: 600; }
.answers .a { color: var(--ink-2); min-width: 0; }
@media (max-width: 640px) { .answers > div { grid-template-columns: minmax(0, 1fr); gap: 2px; } }
.sum { overflow-x: auto; margin: 8px 0 14px; }
.sum table, details table { border-collapse: collapse; font-variant-numeric: tabular-nums; }
.sum table { font-size: 14.5px; }
th, td { text-align: right; padding: 5px 10px; border-bottom: 1px solid var(--rule); white-space: nowrap; }
th:first-child, td:first-child, .l { text-align: left; }
th { font-weight: 600; color: var(--ink-2); font-size: 13px; }
td.best { font-weight: 600; }
td.pend { color: var(--ink-3); }
.key { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 7px; vertical-align: 0; }
.chart { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 12px 14px 6px; min-width: 0; }
.chart .ct { font-weight: 600; font-size: 14.5px; margin: 0; }
.chart .cs { font-size: 13px; color: var(--ink-2); margin: 0 0 6px; }
.chart svg { display: block; width: 100%; height: auto; overflow: visible; }
.chart svg text { font-family: var(--font-body); fill: var(--ink-2); font-size: 12px; }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 480px), 1fr)); gap: 14px; margin: 10px 0; }
.charts.wide { grid-template-columns: minmax(0, 1fr); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13px; color: var(--ink-2); margin: 4px 0 8px; }
details { border: 1px solid var(--rule); border-radius: 6px; background: var(--surface); margin: 10px 0; }
details > summary { cursor: pointer; padding: 9px 14px; font-weight: 600; font-size: 14.5px; }
details > summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
details .tbl { overflow-x: auto; padding: 0 14px 12px; }
details table { font-size: 12.5px; }
.note { font-size: 13.5px; color: var(--ink-2); border-left: 2px solid var(--rule); padding-left: 12px; margin: 8px 0 12px; max-width: 92ch; }
.tip { position: fixed; pointer-events: none; z-index: 20; background: var(--surface); color: var(--ink); border: 1px solid var(--rule); border-radius: 5px; padding: 6px 9px; font-size: 12.5px; box-shadow: 0 4px 14px rgba(0,0,0,0.14); max-width: 320px; }
footer { margin-top: 44px; padding-top: 12px; border-top: 1px solid var(--rule); color: var(--ink-3); font-size: 12.5px; }
.hit { fill: transparent; }
.mk:hover .vis, .mk:focus .vis { opacity: 0.75; }
'''

JS = '''
(function(){
  var tip = document.createElement('div'); tip.className = 'tip'; tip.hidden = true; document.body.appendChild(tip);
  function show(e, el){ tip.innerHTML = el.getAttribute('data-tip'); tip.hidden = false;
    var x = (e.clientX || el.getBoundingClientRect().left) + 14, y = (e.clientY || el.getBoundingClientRect().top) + 12;
    var w = tip.offsetWidth, h = tip.offsetHeight;
    if (x + w > window.innerWidth - 8) x = window.innerWidth - w - 8; if (y + h > window.innerHeight - 8) y = y - h - 24;
    tip.style.left = x + 'px'; tip.style.top = y + 'px'; }
  document.querySelectorAll('[data-tip]').forEach(function(el){
    el.addEventListener('mousemove', function(e){ show(e, el); });
    el.addEventListener('mouseleave', function(){ tip.hidden = true; });
    el.addEventListener('focus', function(e){ show(e, el); });
    el.addEventListener('blur', function(){ tip.hidden = true; });
  });
})();
'''


def esc(s):
    return html.escape(str(s), quote=True)


def legend(D):
    return '<div class="legend">' + ''.join(f'<span><i class="key" style="background:{COLOR[m]}"></i>{D["names"][m]}</span>' for m in D['methods']) + '</div>'


def fmt_t(x):
    return f'{x:.3f}' if x < 0.1 else f'{x:.2f}' if x < 10 else f'{x:.1f}'


def chart(D, t, kind):
    """One chart over the table's columns, four planners side by side: kind 'success' (columns), 'time' (log), 'length' (linear)."""
    cols, ms = t['cols'], D['methods']
    W, H, left, right, top, bottom = 1000, (230 if kind == 'success' else 250), 56, 12, 12, 46
    pw, ph = W - left - right, H - top - bottom
    gw = pw / len(cols)
    bw = min(22, gw * 0.8 / len(ms) - 2)
    vals = []
    for c in cols:
        for m in ms:
            x = c['cells'][m]
            if x is None:
                continue
            if kind == 'success':
                vals.append(x['pct'])
            elif x[kind]:
                vals += [x[kind]['mean'], x[kind]['mean'] + x[kind]['std']]
    if kind == 'success':
        lo, hi, ticks = 0, 100, [0, 25, 50, 75, 100]
    elif kind == 'time':
        pos = [v for v in vals if v > 0] or [0.1, 1]
        lo = 10 ** math.floor(math.log10(min(pos) * 0.8))
        hi = 10 ** math.ceil(math.log10(max(pos)))
        ticks = [10 ** k for k in range(int(round(math.log10(lo))), int(round(math.log10(hi))) + 1)]
    else:
        top_v = max(vals) if vals else 10
        step = 10 ** math.floor(math.log10(top_v / 4)) if top_v > 0 else 1
        step = next(s * step for s in (1, 2, 2.5, 5, 10) if top_v / (s * step) <= 5)
        lo, hi = 0, step * math.ceil(top_v / step)
        ticks = [lo + i * step for i in range(int(round(hi / step)) + 1)]

    def y(v):
        if kind == 'time':
            v = max(v, lo)
            return top + ph - ph * (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        return top + ph - ph * (v - lo) / (hi - lo)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(t["title"])}: {kind}">']
    for v in ticks:
        lab = f'{v:g} %' if kind == 'success' else (f'{v:g} s' if kind == 'time' else f'{v:g}')
        out.append(f'<line x1="{left}" x2="{W - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="var(--grid)" stroke-width="1"/>'
                   f'<text x="{left - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{lab}</text>')
    for gi, c in enumerate(cols):
        cx = left + gi * gw + gw / 2
        out.append(f'<text x="{cx:.1f}" y="{H - bottom + 18}" text-anchor="middle">{esc(c["label"].split(" (")[0])}</text>')
        n_in = max((c['cells'][m] or {}).get('instances', 0) for m in ms)
        if n_in < len(c['instances']):
            out.append(f'<text x="{cx:.1f}" y="{H - bottom + 34}" text-anchor="middle" style="fill:var(--ink-3);font-size:11px">{n_in} of {len(c["instances"])} in</text>')
        for mi, m in enumerate(ms):
            x = c['cells'][m]
            bx = cx + (mi - (len(ms) - 1) / 2) * (bw + 2)
            if x is None:
                continue
            name = D['names'][m]
            if kind == 'success':
                v = x['pct']
                y0, y1 = y(0), y(v)
                h = max(y0 - y1, 0.5)
                r = min(4, h / 2, bw / 2)
                d = (f'M{bx - bw / 2:.1f},{y0:.1f} V{y1 + r:.1f} Q{bx - bw / 2:.1f},{y1:.1f} {bx - bw / 2 + r:.1f},{y1:.1f} H{bx + bw / 2 - r:.1f} '
                     f'Q{bx + bw / 2:.1f},{y1:.1f} {bx + bw / 2:.1f},{y1 + r:.1f} V{y0:.1f} Z')
                tipt = f'<b>{esc(name)}</b> · {esc(c["label"])}<br>{v:.1f} % solved ({x["success"]} of {x["n"]} queries, {x["instances"]} of {x["of"]} instances)'
                out.append(f'<g class="mk" tabindex="0" data-tip="{esc(tipt)}"><rect class="hit" x="{bx - bw / 2 - 1:.1f}" y="{top}" width="{bw + 2:.1f}" height="{ph}"/>'
                           f'<path class="vis" d="{d}" fill="{COLOR[m]}"/></g>')
            else:
                s = x[kind]
                if not s:
                    continue
                m_, sd = s['mean'], s['std']
                ya, yb, ym = y(max(m_ - sd, lo if kind == 'time' else 0)), y(m_ + sd), y(m_)
                unit = ' s' if kind == 'time' else ' cells'
                val = (fmt_t(m_) + ' ± ' + fmt_t(sd)) if kind == 'time' else f'{m_:.1f} ± {sd:.1f}'
                tipt = (f'<b>{esc(name)}</b> · {esc(c["label"])}<br>mean ± std {val}{unit} (median {fmt_t(s["median"]) if kind == "time" else round(s["median"], 1)}{unit})'
                        f'<br>over the {s["n"]} solved queries')
                out.append(f'<g class="mk" tabindex="0" data-tip="{esc(tipt)}"><rect class="hit" x="{bx - bw / 2 - 1:.1f}" y="{top}" width="{bw + 2:.1f}" height="{ph}"/>'
                           f'<line x1="{bx:.1f}" x2="{bx:.1f}" y1="{ya:.1f}" y2="{yb:.1f}" stroke="{COLOR[m]}" stroke-width="2" stroke-linecap="round"/>'
                           f'<circle class="vis" cx="{bx:.1f}" cy="{ym:.1f}" r="5" fill="{COLOR[m]}" stroke="var(--surface)" stroke-width="2"/></g>')
    out.append(f'<line x1="{left}" x2="{W - right}" y1="{top + ph:.1f}" y2="{top + ph:.1f}" stroke="var(--ink-3)" stroke-width="1"/>')
    out.append('</svg>')
    return ''.join(out)


def summary_table(D, t, metric='pct'):
    head = '<tr><th>planner</th>' + ''.join(f'<th>{esc(c["label"])}</th>' for c in t['cols']) + '<th>average</th></tr>'
    best = {}
    for c in t['cols']:
        v = [c['cells'][m]['pct'] for m in D['methods'] if c['cells'][m]]
        best[c['key']] = max(v) if v else None
    rows = []
    for m in D['methods']:
        cells, vals = [], []
        for c in t['cols']:
            x = c['cells'][m]
            if x is None:
                cells.append('<td class="pend">pending</td>')
                continue
            part = '' if x['instances'] == x['of'] else f' <span style="color:var(--ink-3)">({x["instances"]}/{x["of"]})</span>'
            cls = ' class="best"' if best[c['key']] is not None and abs(x['pct'] - best[c['key']]) < 1e-9 else ''
            cells.append(f'<td{cls}>{x["pct"]:.1f} %{part}</td>')
            if x['instances'] == x['of']:
                vals.append(x['pct'])
        avg = f'{sum(vals) / len(vals):.1f} %' if len(vals) == len(t['cols']) else (f'<span style="color:var(--ink-3)">{sum(vals) / len(vals):.1f} % of {len(vals)}</span>' if vals else '')
        rows.append(f'<tr><td class="l"><i class="key" style="background:{COLOR[m]}"></i>{D["names"][m]}</td>{"".join(cells)}<td>{avg}</td></tr>')
    return f'<div class="sum"><table><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>'


def full_table(D, t):
    head = ('<tr><th>scene</th><th>planner</th><th>instances</th><th>solved</th><th>per instance</th><th>time mean ± std (s)</th><th>time median</th>'
            '<th>length mean ± std (cells)</th><th>length / shortest</th><th>failures</th><th>on queries all four solve: time</th><th>: length</th></tr>')
    rows = []
    for c in t['cols']:
        for m in D['methods']:
            x = c['cells'][m]
            if x is None:
                rows.append(f'<tr><td class="l">{esc(c["label"])}</td><td class="l">{D["names"][m]}</td><td colspan="10" class="l pend">pending</td></tr>')
                continue
            tm, ln, lr = x['time'], x['length'], x['length_ratio']
            fail = (f'{x["timeouts"]} time-outs' if 'timeouts' in x else f'{x["through_wall"]} through a wall, {x["not_arrived"]} not arrived')
            cm = x.get('common')
            rows.append(f'<tr><td class="l">{esc(c["label"])}</td><td class="l">{D["names"][m]}</td><td>{x["instances"]}/{x["of"]}</td><td>{x["pct"]:.1f} % ({x["success"]}/{x["n"]})</td>'
                        f'<td>{" / ".join(f"{v:.1f}" for v in x["per_instance"])}</td>'
                        f'<td>{fmt_t(tm["mean"]) + " ± " + fmt_t(tm["std"]) if tm else ""}</td><td>{fmt_t(tm["median"]) if tm else ""}</td>'
                        f'<td>{pm(ln)}</td><td>{pm(lr, 3).split(" ±")[0]}</td><td class="l">{fail}</td>'
                        f'<td>{fmt_t(cm["time"]["mean"]) + " ± " + fmt_t(cm["time"]["std"]) if cm and cm["time"] else ""}</td>'
                        f'<td>{pm(cm["length"]).split(" ±")[0] if cm else ""}</td></tr>')
    return f'<div class="tbl"><table><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>'


def instance_table(D, t):
    head = '<tr><th>scene</th><th>planner</th><th>solved</th><th>time mean ± std (s)</th><th>length mean ± std (cells)</th><th>registered runs (bench / field)</th></tr>'
    rows = []
    for c in t['cols']:
        for k in c['instances']:
            I = D['instances'].get(k, {})
            for m in D['methods']:
                x = (I.get('cells') or {}).get(m)
                run = (I.get('runs') or {}).get(m, {})
                if not x:
                    rows.append(f'<tr><td class="l">{esc(k)}</td><td class="l">{D["names"][m]}</td><td colspan="4" class="l pend">pending</td></tr>')
                    continue
                tm, ln = x['time'], x['length']
                rows.append(f'<tr><td class="l">{esc(k)}</td><td class="l">{D["names"][m]}</td><td>{x["pct"]:.1f} %</td>'
                            f'<td>{fmt_t(tm["mean"]) + " ± " + fmt_t(tm["std"]) if tm else ""}</td><td>{pm(ln)}</td>'
                            f'<td class="l"><code>{esc(run.get("bench", ""))}</code>{" / <code>" + esc(run["field"]) + "</code>" if run.get("field") else ""}</td></tr>')
    return f'<div class="tbl"><table><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>'


def pm(s, d=2):
    return f'{s["mean"]:.{d}f} ± {s["std"]:.{d}f}' if s else ''


def averages(D, t, m):
    v = [c['cells'][m]['pct'] for c in t['cols'] if c['cells'][m] and c['cells'][m]['instances'] == c['cells'][m]['of']]
    return (sum(v) / len(v), len(v)) if v else (None, 0)


def lede(D):
    parts = []
    for t in D['tables']:
        bits = []
        for m in D['methods']:
            a, n = averages(D, t, m)
            if a is not None:
                bits.append(f'{D["names"][m]} {a:.1f} %' + ('' if n == len(t['cols']) else f' ({n} of {len(t["cols"])} sizes)'))
        if bits:
            parts.append(f'<b>{esc(t["title"])}</b>: ' + ', '.join(bits))
    if not parts:
        return 'Results land here as the runs finish.'
    return 'Queries solved, average over the sizes that are complete. ' + '; '.join(parts) + '.'


def body(D):
    n_bench = sum(1 for I in D['instances'].values() for m in D['methods'] if I['cells'].get(m))
    total = sum(len(c['instances']) for t in D['tables'] for c in t['cols']) * len(D['methods'])
    h = [f'<title>{TITLE}</title>', '<meta name="claude:viewport-fit-sides" content="cover">',
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">',
         f'<style>{CSS}</style>', '<div class="wrap">',
         '<header class="top"><div class="eyebrow">jeffrey/lbs_final · 500 random queries per scene · all planners on the CPU, one query at a time · every number from registered runs</div>',
         '<h1>LBS final results: LBS against ntrl-demo, RRT-Connect and LazyPRM</h1>',
         f'<p class="lede">{lede(D)}</p></header>',
         f'<div class="status"><b>{n_bench} of {total}</b> scene × planner results are in. Cells marked (k/3) pool the instances finished so far.</div>',
         '<h2 id="setup">What was run</h2><div class="answers">',
         '<div><div class="q">Scenes</div><div class="a"><b>Maze 2-D</b>: 10, 14, 18, 22 cells a side, three mazes each (maze seeds 1000 i + n). <b>Gap maze 2-D</b>: the 10 × 10 maze cut into four blocks pushed apart by an open band of 6 k cells, k = 1, 2, 4, 8 (16, 22, 34, 58 cells a side), three base mazes each. The SE(2) rectangle robot. <b>Gibson 3-D</b>: a point robot flying in a 2 m slab of the main floor of ten houses: the five of the earlier 3-D test and the five houses with the smallest main floors (Collierville, Benevolence, Onaga, Coffeen, Forkland).</div></div>',
         '<div><div class="q">Queries</div><div class="a">500 random (start, goal) pairs per scene: starts uniform over free space at least 0.1 cells from obstacles, goals drawn from the scene’s 48 random goal poses, every pair certified solvable by its ground-truth path. Gap mazes keep their goal mix (half of the goals inside the maze blocks, half in the open band). A table cell pools the three instances: 1,500 queries.</div></div>',
         '<div><div class="q">Planners</div><div class="a"><b>ntrl-demo</b> and <b>LBS</b>: ntrl-demo’s MPPI on the trained travel-time field (training seed 87358, epoch 5,000), planning seed 0, step budget 3 × the true path. <b>RRT-Connect</b>: OMPL 2.0.1, default range. <b>LazyPRM</b>: a Python port of OMPL’s LazyPRM (k = 5 neighbours, lazy vertex and edge checks, a fresh roadmap per query). Both sampling planners get 10 s per query, then OMPL’s path simplifier.</div></div>',
         '<div><div class="q">Success</div><div class="a">The path reaches the goal and every configuration on it, at 0.005 cells per coordinate, is collision-free under the exact geometry. The sampling planners check motions at that same resolution, so their paths never fail the scorer.</div></div>',
         '<div><div class="q">Time and length</div><div class="a">Wall time to the finished path (MPPI until it reaches the goal; sampling planners solve plus simplification), on two CPU threads, measured while the GPUs trained other runs. Path length in cells: x-y length for the 2-D robot, x-y-z length in the houses (one cell is 0.25 m). Mean ± standard deviation over each planner’s solved queries.</div></div>',
         '</div>']
    for t in D['tables']:
        h.append(f'<h2 id="{t["key"]}">{esc(t["title"])}</h2>')
        h.append(summary_table(D, t))
        wide = ' wide' if len(t['cols']) > 5 else ''
        h.append(legend(D))
        h.append(f'<div class="charts wide"><div class="chart"><p class="ct">Queries solved</p><p class="cs">Share of the {500 * len(t["cols"][0]["instances"])} queries per column.</p>{chart(D, t, "success")}</div></div>')
        h.append(f'<div class="charts wide"><div class="chart"><p class="ct">Planning time</p><p class="cs">Seconds per solved query, mean (dot) ± one standard deviation (line), log scale.</p>{chart(D, t, "time")}</div>'
                 f'<div class="chart"><p class="ct">Path length</p><p class="cs">Cells per solved query, mean ± one standard deviation.</p>{chart(D, t, "length")}</div></div>')
        h.append(f'<details><summary>Full table: {esc(t["title"])} (time, length, failures, the queries all four solve)</summary>{full_table(D, t)}</details>')
        h.append(f'<details><summary>Per scene instance: {esc(t["title"])}</summary>{instance_table(D, t)}</details>')
    h.append('<footer>jeffrey/lbs_final: report.py → report/summary.json → page.py. Fields of instance 1 of maze 10 and 18, gap ×1, ×2, ×4 and of the five earlier 3-D houses are the jeffrey/lbs runs on the same scenes.</footer>')
    h.append('</div>')
    h.append(f'<script>{JS}</script>')
    return '\n'.join(h)


def main():
    D = json.load(open(HERE / 'report/summary.json'))
    (HERE / 'report/lbs_final.html').write_text(body(D))
    print(HERE / 'report/lbs_final.html')


if __name__ == '__main__':
    main()
