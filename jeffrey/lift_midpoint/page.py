"""Findings page of the lift + midpoint study: report/runs.json (report.py) + pagetext.py -> report/lift_midpoint.html (published as an artifact).

    python3 jeffrey/lift_midpoint/report.py && python3 jeffrey/lift_midpoint/page.py
"""
import html
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pagetext as PT  # noqa: E402

SHORT = dict(maze10='maze 10', maze13='maze 13', maze22='maze 22', plus22='plus 22', plus30='plus 30', plus40='plus 40', Allensville='Allensville', Lakeville='Lakeville',
             Klickitat='Klickitat')

CSS = r"""
/* Layout: one 72ch reading column inside a 1120px frame; charts take the frame width; tables hide in <details>. */
:root {
  --bg: #f4f5f2; --surface: #fcfcfb; --ink: #121513; --ink2: #4b524e; --muted: #7b827e; --rule: #dcdfdb; --grid: #e7e9e5; --base: #c3c7c2;
  --accent: #17654d; --wash: #e6efe9;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #8d928e; --s5: #e87ba4; --ref: #121513;
  --ok: #3b7d5c; --stall: #e9b44c; --wall: #d03b3b;
  --f-display: "Archivo", "Helvetica Neue", Arial, sans-serif;
  --f-body: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  --f-mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #111412; --surface: #1a1b1a; --ink: #edf0ed; --ink2: #b7beb9; --muted: #8b928e; --rule: #2d312e; --grid: #262a27; --base: #3a3f3b;
    --accent: #6fd3ad; --wash: #1c2a24;
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #8f9591; --s5: #d55181; --ref: #edf0ed;
    --ok: #4f9d76; --stall: #c99a33; --wall: #e66767;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #111412; --surface: #1a1b1a; --ink: #edf0ed; --ink2: #b7beb9; --muted: #8b928e; --rule: #2d312e; --grid: #262a27; --base: #3a3f3b;
  --accent: #6fd3ad; --wash: #1c2a24;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #8f9591; --s5: #d55181; --ref: #edf0ed;
  --ok: #4f9d76; --stall: #c99a33; --wall: #e66767;
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
.verdict { display: inline-block; font: 500 11px var(--f-mono); letter-spacing: 0.05em; text-transform: uppercase; padding: 1px 7px; border-radius: 3px; margin-right: 6px; vertical-align: 1px; background: var(--wash); box-shadow: inset 0 0 0 1px var(--rule); color: var(--ink); }
.chart { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 14px 14px 8px; min-width: 0; }
.chart .ctitle { font-weight: 600; font-size: 14.5px; margin: 0 0 2px; }
.chart .csub { font-size: 13px; color: var(--ink2); margin: 0 0 8px; max-width: 100ch; }
.chart svg { display: block; width: 100%; height: auto; overflow: visible; }
.chart svg text { font-family: var(--f-body); }
.ax { fill: var(--muted); font-size: 10.5px; font-family: var(--f-mono) !important; }
.lab { fill: var(--ink2); font-size: 12.5px; }
.small { fill: var(--ink2); font-size: 11px; }
.val { fill: var(--ink); font-size: 11px; font-family: var(--f-mono) !important; }
.note { fill: var(--muted); font-size: 10px; }
.grid { stroke: var(--grid); stroke-width: 1; }
.base { stroke: var(--base); stroke-width: 1; }
.row { stroke: var(--grid); stroke-width: 1; }
.link { stroke: var(--base); stroke-width: 2; }
.refline { stroke: var(--ref); stroke-width: 1.5; stroke-dasharray: 2 2; }
.flat { stroke-width: 1.5; stroke-dasharray: 4 3; fill: none; }
.ln { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.hit { fill: transparent; cursor: default; }
.hit:focus-visible, [tabindex]:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.seg.ok { fill: var(--ok); } .seg.stall { fill: var(--stall); } .seg.wall { fill: var(--wall); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 12.5px; color: var(--ink2); margin: 2px 0 8px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.chart .legend svg { display: inline-block; width: 16px; height: 16px; flex: none; }
.legend i.sw { display: inline-block; width: 12px; height: 12px; border-radius: 2px; }
.legend i.sw.ok { background: var(--ok); } .legend i.sw.stall { background: var(--stall); } .legend i.sw.wall { background: var(--wall); }
.legend i.flatkey { display: inline-block; width: 18px; height: 0; border-top: 2px dashed var(--s1); }
.multi { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 250px), 1fr)); gap: 10px; }
.panel { border: 1px solid var(--rule); border-radius: 5px; padding: 8px 8px 2px; min-width: 0; background: var(--surface); }
.panel h4 { font: 600 13px var(--f-display); margin: 0 0 2px 2px; }
.tip { position: fixed; z-index: 20; pointer-events: none; background: var(--surface); color: var(--ink); border: 1px solid var(--rule); border-radius: 5px; padding: 7px 10px; font-size: 12.5px; box-shadow: 0 4px 16px rgba(0,0,0,0.14); max-width: 320px; }
.tip .h { font: 500 11.5px var(--f-mono); color: var(--muted); margin-bottom: 3px; }
.tip .n { color: var(--ink2); font-size: 12px; }
.tip b { font-family: var(--f-mono); font-weight: 500; }
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


def build_data(runs):
    order = [s for s in runs['order'] if s in runs['scenes']]
    rows, bins = {}, {}
    refresh_epochs = set()
    for s in order:
        rows[s] = {}
        for arm, r in runs['scenes'][s].items():
            keep = {k: r.get(k) for k in ('solved', 'stall', 'wall', 'success', 'los_ratio', 'los_within10', 'rho', 'seen_through', 'within10', 'ratio', 'by_bin', 'seed0', 'curve', 'reached', 'detour_drives')}
            rows[s][arm] = keep
            for e, _ in r.get('reached') or []:
                refresh_epochs.add(e)
        bins[s] = sorted({b for r in rows[s].values() for b in r['by_bin']}, key=lambda t: float(t.split('-')[0]))
    return dict(order=order, title={s: runs['title'][s] for s in order}, short={s: SHORT[s] for s in order}, arms=['ntrl', 'lift', 'liftmid', 'mid', 'drive'],
                armName=dict(runs['arm_name'], drive='lift + midpoint, ntrl-demo drives'), rows=rows, bins=bins, refreshEpochs=sorted(refresh_epochs))


def table_all(runs):
    head = ['scene', 'arm', 'solved %', 'stalled %', 'through a wall %', 'T within 10 %', 'median T/T*', 'line of sight within 10 %', 'line of sight T/T*', 'detour index', 'seen through %', 'S × slope, innermost band', 'registered run']
    out = ['<table><thead><tr>' + ''.join(f'<th class="{"l" if i in (0, 1, 12) else ""}">{esc(h)}</th>' for i, h in enumerate(head)) + '</tr></thead><tbody>']
    for s in runs['order']:
        arms = runs['scenes'].get(s, {})
        for arm in runs['arms']:
            if arm not in arms:
                continue
            r = arms[arm]
            pc = lambda v: None if v is None else 100 * v
            cells = [runs['title'][s], runs['arm_name'][arm], f(r['solved']), f(r['stall']), f(r['wall']), f(pc(r['within10'])), f(r['ratio'], 3), f(pc(r['los_within10'])),
                     f(r['los_ratio'], 3), f(r['rho'], 2), f(pc(r['seen_through'])), f(r['wall_innermost'], 2), r['run']]
            out.append('<tr>' + ''.join(f'<td class="{"l" if i in (0, 1) else ("l num" if i == 12 else "num")}">{esc(c)}</td>' for i, c in enumerate(cells)) + '</tr>')
    out.append('</tbody></table>')
    return '\n'.join(out)


def table_bins(runs):
    out = []
    for s in runs['order']:
        arms = runs['scenes'].get(s, {})
        if not arms:
            continue
        bins = sorted({b for r in arms.values() for b in r['by_bin']}, key=lambda t: float(t.split('-')[0]))
        out.append(f'<h3>{esc(runs["title"][s])}</h3><table style="min-width: 520px"><thead><tr><th class="l">arm</th>' + ''.join(f'<th>{esc(b)} cells</th>' for b in bins) + '</tr></thead><tbody>')
        for arm in runs['arms']:
            if arm in arms:
                out.append(f'<tr><td class="l">{esc(runs["arm_name"][arm])}</td>' + ''.join(f'<td class="num">{f(arms[arm]["by_bin"].get(b), 0)}</td>' for b in bins) + '</tr>')
        out.append('</tbody></table>')
    return '\n'.join(out)


def table_course(runs):
    out = ['<table style="min-width: 640px"><thead><tr><th class="l">scene</th><th class="l">arm</th><th class="l">queries solved, planning seed 0 (epoch: %)</th><th class="l">training drives reaching their goal (refresh epoch: %)</th></tr></thead><tbody>']
    for s in runs['order']:
        for arm in ('lift', 'liftmid', 'mid'):
            r = runs['scenes'].get(s, {}).get(arm)
            if not r:
                continue
            curve = ', '.join(f'{e}: {v:.1f}' for e, v in r.get('curve') or [])
            reach = ', '.join(f'{e}: {v:.0f}' for e, v in (r.get('reached') or [])[::5])
            out.append(f'<tr><td class="l">{esc(runs["title"][s])}</td><td class="l">{esc(runs["arm_name"][arm])}</td><td class="l num">{esc(curve)}</td><td class="l num">{esc(reach)}</td></tr>')
    out.append('</tbody></table>')
    return '\n'.join(out)


def main():
    runs = json.load(open(HERE / 'report/runs.json'))
    D = build_data(runs)
    T = PT.text(runs)
    js = (HERE / 'page_assets/charts.js').read_text()
    body = f"""<title>2026-11-17 Lift + Midpoint</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;650;700&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}</style>
<div class="page">
<header class="col">
  <div class="eyebrow">jeffrey/lift_midpoint · SE(2) rectangle robot in 2-D scenes · training seed 87358 · every number from registered runs</div>
  <h1>{T['h1']}</h1>
  <p class="lede">{T['lede']}</p>
  {T['status']}
</header>
<section><h2>Answers</h2><div class="answers">{T['answers']}</div></section>
<section>
  <div class="col"><h2>Queries solved</h2>{T['solved_intro']}</div>
  <div class="chart" data-chart="solved"><div class="ctitle">Queries solved, % of runs (queries × planning seeds 0, 1, 2)</div><div class="csub">{T['solved_sub']}</div></div>
</section>
<section>
  <div class="col"><h2>What the midpoint loss changes in the field</h2>{T['field_intro']}</div>
  <div class="chart" data-chart="los"><div class="ctitle">Travel time along free lines of sight, learned over true (median)</div><div class="csub">{T['los_sub']}</div></div>
  <div class="chart" data-chart="rho"><div class="ctitle">Detour index on pairs whose true path is at least 30 % longer than the straight line (median)</div><div class="csub">{T['rho_sub']}</div></div>
  <div class="col">{T['field_after']}</div>
</section>
<section>
  <div class="col"><h2>Why: the loss learns only from drives the current field can make</h2>{T['why_intro']}</div>
  <div class="chart" data-chart="detours"><div class="ctitle">Training drives that reach their goal along a detour, % of the 8,192 drives of each refresh</div><div class="csub">{T['reached_sub']}</div></div>
  <div class="col">{T['why_after']}</div>
</section>
<section>
  <div class="col"><h2>How the runs end</h2>{T['fails_intro']}</div>
  <div class="chart" data-chart="fails"><div class="ctitle">Outcome of every planning run, % of runs</div><div class="csub">{T['fails_sub']}</div></div>
</section>
<section>
  <div class="col"><h2>During training</h2>{T['course_intro']}</div>
  <div class="chart" data-chart="course"><div class="ctitle">Queries solved with planning seed 0 at saved epochs</div><div class="csub">{T['course_sub']}</div></div>
</section>
<section>
  <div class="col"><h2>By distance</h2>{T['bins_intro']}</div>
  <div class="chart" data-chart="bins"><div class="ctitle">Queries solved by true travel distance (cells at full speed), % of runs</div><div class="csub">40 queries per bin × 3 planning seeds.</div></div>
</section>
<section class="col"><h2>How it was run</h2>{T['setup']}</section>
<section class="col"><h2>Caveats</h2>{T['caveats']}</section>
<details><summary>Every arm: planning, travel times, detours, wall layer, run ids</summary><div class="tbl">{table_all(runs)}</div></details>
<details><summary>Success by distance bin</summary><div class="tbl">{table_bins(runs)}</div></details>
<details><summary>During training: planning seed 0 and training drives</summary><div class="tbl">{table_course(runs)}</div></details>
<footer>Study folder jeffrey/lift_midpoint (README: request, setup, registered run ids; report/tables.md). Parent pages: <a href="https://claude.ai/artifact/HBHUi6dRwWctNp1eDQMs9d">2026-11-11 Travel-Time Lift</a>, <a href="https://claude.ai/artifact/L7QtsFtB8Kk6nGv3mJ8d4U">Travel-Time Midpoint Loss</a>, <a href="https://claude.ai/artifact/3NQ9XSVrVgRCMBjaugePSM">2026-11-17 Lift Scale Test</a>.</footer>
</div>
<script>window.LM_DATA = {json.dumps(D, separators=(',', ':'))};</script>
<script>{js}</script>
"""
    (HERE / 'report/lift_midpoint.html').write_text(body)
    print('wrote', HERE / 'report/lift_midpoint.html', len(body), 'bytes')


if __name__ == '__main__':
    main()
