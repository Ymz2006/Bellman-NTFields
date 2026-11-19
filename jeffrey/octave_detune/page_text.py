"""Words and page shell of the octave-detune findings page (page.py fills in the charts and numbers)."""
import html

esc = lambda s: html.escape(str(s), quote=True)  # noqa: E731

CSS = r"""
:root {
  /* Layout: one reading column; every chart is a pair of small multiples (13x13 | 22x22) on the same scale family. */
  --bg: #f5f6f8; --surface: #fcfcfd; --ink: #111418; --ink-2: #4b525b; --muted: #858c96; --grid: #e3e6ea; --base: #c5cad1;
  --line: rgba(17, 20, 24, 0.10); --band: rgba(42, 120, 214, 0.10); --pos: 27, 135, 82; --neg: 200, 62, 62; --tile: rgba(17, 20, 24, 0.04);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --s6: #008300; --s7: #6250d6; --s8: #e34948;
  --q1: #b3a9ee; --q2: #7b6be0; --q3: #3f2f9e;
  --c-low: #eb6834; --c-high: #2a78d6; --c-jit: #1baf7a;
  --display: "Space Grotesk", "Segoe UI", system-ui, sans-serif; --text: "Inter", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0e1114; --surface: #161a1e; --ink: #eef1f4; --ink-2: #b8bfc8; --muted: #8a919b; --grid: #262b31; --base: #3a4149;
    --line: rgba(255, 255, 255, 0.10); --band: rgba(57, 135, 229, 0.16); --pos: 64, 190, 120; --neg: 236, 105, 105; --tile: rgba(255, 255, 255, 0.05);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #0ca30c; --s7: #9085e9; --s8: #e66767;
    --q1: #574ab8; --q2: #8d82e8; --q3: #d0c9f9;
    --c-low: #d95926; --c-high: #3987e5; --c-jit: #199e70;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #0e1114; --surface: #161a1e; --ink: #eef1f4; --ink-2: #b8bfc8; --muted: #8a919b; --grid: #262b31; --base: #3a4149;
  --line: rgba(255, 255, 255, 0.10); --band: rgba(57, 135, 229, 0.16); --pos: 64, 190, 120; --neg: 236, 105, 105; --tile: rgba(255, 255, 255, 0.05);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #0ca30c; --s7: #9085e9; --s8: #e66767;
  --q1: #574ab8; --q2: #8d82e8; --q3: #d0c9f9;
  --c-low: #d95926; --c-high: #3987e5; --c-jit: #199e70;
  color-scheme: dark;
}
body { background: var(--bg); color: var(--ink); font-family: var(--text); font-size: 15px; line-height: 1.55; }
.wrap { max-width: 1000px; margin: 0 auto; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 28px 64px; }
h1, h2, h3 { font-family: var(--display); text-wrap: balance; line-height: 1.2; margin: 0; }
h1 { font-size: 32px; font-weight: 600; letter-spacing: -0.01em; }
h2 { font-size: 22px; font-weight: 600; margin-top: 44px; }
h3 { font-size: 16px; font-weight: 600; }
h4 { font-family: var(--display); font-size: 15px; margin: 22px 0 6px; }
p { margin: 0; max-width: 68ch; }
.eyebrow { font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
.stack { display: flex; flex-direction: column; gap: 12px; }
header.stack { gap: 10px; }
.lede { font-size: 17px; color: var(--ink-2); }
.answer { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 18px 20px; display: flex; flex-direction: column; gap: 10px; margin-top: 18px; }
.answer h2 { margin: 0; font-size: 18px; }
.answer ul { margin: 0; padding-left: 20px; display: flex; flex-direction: column; gap: 6px; max-width: 76ch; }
.pair { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(420px, 100%), 1fr)); gap: 18px; }
.panel { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 14px 14px 8px; min-width: 0; display: flex; flex-direction: column; gap: 6px; }
.panel h3 { font-size: 14px; color: var(--ink-2); font-weight: 600; }
.panel svg { width: 100%; height: auto; display: block; overflow: visible; }
.wide svg { width: 100%; height: auto; display: block; }
.note { font-size: 13px; color: var(--ink-2); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 13px; color: var(--ink-2); }
.legend .key { display: inline-flex; align-items: center; gap: 6px; }
svg text { font-family: var(--text); }
.ax { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.axt { fill: var(--ink-2); font-size: 11.5px; }
.lbl { fill: var(--ink-2); font-size: 11px; }
.small { font-size: 10.5px; }
.grid { stroke: var(--grid); stroke-width: 1; }
.zero { stroke: var(--base); stroke-width: 1.5; }
.base { stroke: var(--base); stroke-width: 1; }
.lock { stroke: var(--base); stroke-width: 1; }
.ref { stroke: var(--muted); stroke-width: 1; }
.band { fill: var(--band); }
.hit { fill: transparent; }
.ln.s1 { stroke: var(--s1); } .ln.s2 { stroke: var(--s2); } .ln.s3 { stroke: var(--s3); } .ln.s4 { stroke: var(--s4); }
.ln.s5 { stroke: var(--s5); } .ln.s6 { stroke: var(--s6); } .ln.s7 { stroke: var(--s7); } .ln.s8 { stroke: var(--s8); }
.dot.s1, .seed.s1 { fill: var(--s1); } .dot.s2, .seed.s2 { fill: var(--s2); } .dot.s3, .seed.s3 { fill: var(--s3); } .dot.s4, .seed.s4 { fill: var(--s4); }
.dot.s5, .seed.s5 { fill: var(--s5); } .dot.s6, .seed.s6 { fill: var(--s6); } .dot.s7, .seed.s7 { fill: var(--s7); } .dot.s8, .seed.s8 { fill: var(--s8); }
.ln.q1 { stroke: var(--q1); } .ln.q2 { stroke: var(--q2); } .ln.q3 { stroke: var(--q3); }
.dot.q1 { fill: var(--q1); } .dot.q2 { fill: var(--q2); } .dot.q3 { fill: var(--q3); }
.ln.ref { stroke: var(--muted); stroke-width: 1.6; }
.ln.k-exact { stroke: var(--ink); } .ln.k-low { stroke: var(--c-low); } .ln.k-high { stroke: var(--c-high); } .ln.k-jit { stroke: var(--c-jit); }
.ln.k-ref { stroke: var(--ink-2); stroke-width: 1.8; }
.dot.k-exact, .seed.k-exact { fill: var(--ink); } .dot.k-low, .seed.k-low { fill: var(--c-low); } .dot.k-high, .seed.k-high { fill: var(--c-high); }
.dot.k-jit, .seed.k-jit { fill: var(--c-jit); } .dot.k-ref, .seed.k-ref { fill: var(--ink-2); }
.dot.k-part, .seed.k-part { fill: var(--surface); stroke: var(--c-low); stroke-width: 2; }
.dot.hollow.k-exact { fill: var(--surface); stroke: var(--ink); stroke-width: 2; } .dot.hollow.k-low { fill: var(--surface); stroke: var(--c-low); stroke-width: 2; }
.dot.hollow.k-high { fill: var(--surface); stroke: var(--c-high); stroke-width: 2; }
.tick.k-exact { stroke: var(--ink); } .tick.k-low { stroke: var(--c-low); } .tick.k-high { stroke: var(--c-high); } .tick.k-jit { stroke: var(--c-jit); }
.tick.k-part { stroke: var(--c-low); stroke-dasharray: 2 2; }
.meanbar.k-exact, .whisk.k-exact { stroke: var(--ink); } .meanbar.k-low, .whisk.k-low, .meanbar.k-part, .whisk.k-part { stroke: var(--c-low); }
.meanbar.k-high, .whisk.k-high { stroke: var(--c-high); } .meanbar.k-jit, .whisk.k-jit { stroke: var(--c-jit); } .meanbar.k-ref, .whisk.k-ref { stroke: var(--ink-2); }
.whisk.k-part { stroke-dasharray: 3 2; }
.ring { stroke: var(--surface); stroke-width: 2; }
.seed { opacity: 0.45; }
.tick { stroke-width: 1.4; opacity: 0.55; }
.tick.s1 { stroke: var(--s1); } .tick.s2 { stroke: var(--s2); } .tick.s6 { stroke: var(--s6); } .tick.s7 { stroke: var(--s7); } .tick.s8 { stroke: var(--s8); }
.meanbar { stroke-width: 3; stroke-linecap: round; }
.meanbar.s2 { stroke: var(--s2); } .meanbar.s3 { stroke: var(--s3); } .meanbar.s4 { stroke: var(--s4); } .meanbar.s6 { stroke: var(--s6); }
.meanbar.s7 { stroke: var(--s7); } .meanbar.s8 { stroke: var(--s8); }
.whisk { stroke-width: 1.5; stroke-linecap: round; }
.whisk.s2 { stroke: var(--s2); } .whisk.s3 { stroke: var(--s3); } .whisk.s4 { stroke: var(--s4); } .whisk.s6 { stroke: var(--s6); }
.whisk.s7 { stroke: var(--s7); } .whisk.s8 { stroke: var(--s8); }
.pt:hover .dot, .pt:focus .dot { stroke: var(--ink); stroke-width: 2; }
.pt:focus, .hit:focus { outline: none; }
.pt:focus-visible .dot, .pt:focus-visible .meanbar { stroke: var(--ink); stroke-width: 2.5; }
#tip { position: fixed; pointer-events: none; background: var(--surface); color: var(--ink); border: 1px solid var(--line); border-radius: 8px;
  padding: 7px 10px; font-size: 12.5px; max-width: 320px; box-shadow: 0 4px 18px rgba(0,0,0,0.12); z-index: 20; font-variant-numeric: tabular-nums; }
.scroll { overflow-x: auto; max-width: 100%; }
table { border-collapse: collapse; font-size: 12.5px; font-variant-numeric: tabular-nums; }
th, td { padding: 5px 8px; border-bottom: 1px solid var(--grid); text-align: right; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
thead th { color: var(--ink-2); font-weight: 600; border-bottom: 1px solid var(--base); }
.cmap { border-collapse: separate; border-spacing: 3px; font-size: 12px; }
.cmap th, .cmap td { border-bottom: none; padding: 5px 6px; }
.cmap thead th { white-space: normal; vertical-align: bottom; min-width: 50px; line-height: 1.25; font-size: 11.5px; }
.cmap tbody td { background: var(--tile); border-radius: 4px; }
.cmap td.pos { background: rgba(var(--pos), calc(0.14 + 0.48 * var(--z))); }
.cmap td.neg { background: rgba(var(--neg), calc(0.14 + 0.48 * var(--z))); }
.cmap .refrow td, .cmap .refrow th { color: var(--ink-2); font-style: italic; background: transparent; }
details { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 10px 14px; }
details + details { margin-top: 10px; }
summary { cursor: pointer; font-weight: 600; font-family: var(--display); }
summary:focus-visible { outline: 2px solid var(--s1); outline-offset: 3px; border-radius: 4px; }
code { font-family: var(--mono); font-size: 12.5px; }
.empty { color: var(--muted); font-size: 13px; }
.li { padding-left: 14px; text-indent: -10px; }
.li::before { content: "– "; color: var(--muted); }
@media (prefers-reduced-motion: no-preference) { #tip { transition: opacity 80ms; } }
"""

SCRIPT = r"""
(function () {
  const tip = document.getElementById('tip');
  function show(el, x, y) {
    const t = el.getAttribute('data-tip');
    if (!t) return;
    tip.textContent = t;
    tip.hidden = false;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let left = x + 14, top = y + 14;
    if (left + w > window.innerWidth - 8) left = x - w - 14;
    if (top + h > window.innerHeight - 8) top = y - h - 14;
    tip.style.left = Math.max(8, left) + 'px';
    tip.style.top = Math.max(8, top) + 'px';
  }
  function hide() { tip.hidden = true; }
  document.querySelectorAll('[data-tip]').forEach(function (el) {
    el.addEventListener('pointermove', function (e) { show(el, e.clientX, e.clientY); });
    el.addEventListener('pointerleave', hide);
    el.addEventListener('focus', function () { const r = el.getBoundingClientRect(); show(el, r.right, r.top); });
    el.addEventListener('blur', hide);
  });
})();
"""


def pct(v, p=1):
    return '–' if v is None else f'{100 * v:.{p}f} %'


def render(S, charts, legends, tables, field):
    o13, o22 = field('octave', '13'), field('octave', '22')

    def rate(k, n):
        v = field(k, n)
        return pct(v['pooled']['rate']) if v else '–'

    def diff(k, n):
        v = field(k, n)
        return '–' if not v or v.get('diff_mean_points') is None else f'{v["diff_mean_points"]:+.1f}'

    answer = ANSWER(S, field, rate, diff)
    return f"""<title>2026-11-17 Octave Detuning</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600&family=JetBrains+Mono:wght@400&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header class="stack">
  <span class="eyebrow">jeffrey/octave_detune · grid-sync octaves · 13×13 and 22×22 mazes · ntrl-demo</span>
  <h1>2026-11-17 Octave Detuning</h1>
  <p class="lede">Is the octave method good because its rows sit exactly on the octave frequencies f_grid / 2<sup>k</sup>, locked to
  the grid, or because it reaches higher frequencies than ntrl-demo? The test keeps everything of the octave matrix and moves
  the frequencies a little off the octave values, coherently (every row by the same factor) or row by row, and measures every
  outcome of the trained field, including how fast it converges.</p>
</header>
{answer}

<h2>What was changed</h2>
<div class="stack">
<p>Each row of the matrix is one plane wave. Its dominant frequency, shown as the half period along the grid axis, sits on one of the
octave positions (hairlines) in the exact method; the top octave has a half period of one cell. A shift moves every row by the same
factor (orange lower, blue higher; ×0.5 and ×2 land on the octave positions again); a jitter (teal) moves each row by its own random
factor, keeping each octave's geometric mean exact; the dashed rows move only part of the band. Rows, directions, heading column and network
initialization are the same in every variant of a training seed. Hover a row for its lock error: the phase mismatch at the outermost grid
line against an exactly locked row.</p>
<div class="panel wide"><h3>Dominant frequency of the 128 rows, 22×22 maze, training seed 87358</h3>{charts['freq']}</div>
</div>

<h2>Success against the frequency factor</h2>
<div class="stack">
{SCAN_TEXT(S, field, rate, diff)}
{legends['scan']}
<div class="pair">
<div class="panel"><h3>13×13 · 600 runs per training seed</h3>{charts['scan13']}</div>
<div class="panel"><h3>22×22 · 720 runs per training seed</h3>{charts['scan22']}</div>
</div>
</div>

<h2>What fails along the scan</h2>
<div class="stack">
{FAIL_TEXT(S, field, rate, diff)}
{legends['fail']}
<div class="pair">
<div class="panel"><h3>Change in failures, 13×13</h3>{charts['fail13']}</div>
<div class="panel"><h3>Change in failures, 22×22</h3>{charts['fail22']}</div>
</div>
{STALL_TEXT(S, field, rate, diff)}
{legends['stall']}
<div class="pair">
<div class="panel"><h3>Where the runs that never arrive stop, 13×13</h3>{charts['stall13']}</div>
<div class="panel"><h3>Where the runs that never arrive stop, 22×22</h3>{charts['stall22']}</div>
</div>
</div>

<h2>Paired differences against the exact octave</h2>
<div class="stack">
{PAIRED_TEXT(S, field, rate, diff)}
{legends['paired']}
<div class="pair">
<div class="panel"><h3>13×13</h3>{charts['paired13']}</div>
<div class="panel"><h3>22×22</h3>{charts['paired22']}</div>
</div>
</div>

<h2>Does the loss follow the lock error?</h2>
<div class="stack">
{LOCK_TEXT(S, field, rate, diff)}
{legends['lock']}
<div class="pair">
<div class="panel"><h3>13×13</h3>{charts['lock13']}</div>
<div class="panel"><h3>22×22</h3>{charts['lock22']}</div>
</div>
</div>

<h2>Which part of the band carries the ×0.9 loss?</h2>
<div class="stack">
{BAND_TEXT(S, field, rate, diff)}
<div class="pair">
<div class="panel"><h3>Success, 13×13</h3>{charts['band_success']}</div>
<div class="panel"><h3>Runs that reach the goal through a wall, 13×13</h3>{charts['band_wall']}</div>
<div class="panel"><h3>Final training loss, 13×13</h3>{charts['band_loss']}</div>
<div class="panel"><h3>Held-out speed correlation, 13×13</h3>{charts['band_corr']}</div>
</div>
</div>

<h2>Convergence</h2>
<div class="stack">
{CONV_TEXT(S, field, rate, diff)}
{legends['conv']}
<div class="pair">
<div class="panel"><h3>Success during training, 13×13</h3>{charts['conv13']}</div>
<div class="panel"><h3>Success during training, 22×22</h3>{charts['conv22']}</div>
<div class="panel"><h3>Training loss, 13×13</h3>{charts['loss13']}</div>
<div class="panel"><h3>Training loss, 22×22</h3>{charts['loss22']}</div>
<div class="panel"><h3>Travel-time error during training, 13×13</h3>{charts['err13']}</div>
<div class="panel"><h3>Travel-time error during training, 22×22</h3>{charts['err22']}</div>
</div>
{EPOCHS_TEXT(S, field, rate, diff)}
{legends['epochs']}
<div class="pair">
<div class="panel"><h3>Success at three epochs against the shift, 13×13</h3>{charts['epochs13']}</div>
<div class="panel"><h3>Success at three epochs against the shift, 22×22</h3>{charts['epochs22']}</div>
</div>
{LOSSF_TEXT(S, field, rate, diff)}
{legends['metric']}
<div class="pair">
<div class="panel"><h3>Change in final training loss, 13×13</h3>{charts['lossf13']}</div>
<div class="panel"><h3>Change in final training loss, 22×22</h3>{charts['lossf22']}</div>
<div class="panel"><h3>Change in loss settling epoch, 13×13</h3>{charts['settle13']}</div>
<div class="panel"><h3>Change in loss settling epoch, 22×22</h3>{charts['settle22']}</div>
</div>
</div>

<h2>Every measured change</h2>
<div class="stack">
{CMAP_TEXT(S, field, rate, diff)}
{charts['cmap13']}
{charts['cmap22']}
</div>

<h2>Where in the maze the change happens</h2>
<div class="stack">
{SPATIAL_TEXT(S, field, rate, diff)}
{legends['spatial']}
<div class="pair">
<div class="panel"><h3>Change in near-wall slope by ring, 13×13</h3>{charts['spatial13']}</div>
<div class="panel"><h3>Change in near-wall slope by ring, 22×22</h3>{charts['spatial22']}</div>
</div>
</div>

<h2>Details</h2>
<div class="stack">
<details><summary>All tables (generated by report.py)</summary>{tables}</details>
<details><summary>Method, controls and caveats</summary><div class="stack" style="margin-top:10px">{METHOD}</div></details>
</div>
</div>
<div id="tip" role="tooltip" hidden></div>
<script>{SCRIPT}</script>
"""


# ---------------------------------------------------------------------------------------------------- words
SCAN = 'the wide scan (×0.5 … ×2)'
PLANNED = [('shift0.9', ('13', '22'), 3, '±10 % shifts'), ('shift1.1', ('13', '22'), 3, '±10 % shifts'),
           ('jitter0.05', ('13', '22'), 3, 'jitter ±5 %'), ('jitter0.2', ('13', '22'), 3, 'jitter ±20 %'),
           ('jitter0.4142', ('13', '22'), 3, 'the log-uniform fill (same band, no discrete frequencies)'),
           ('shift1.189', ('13', '22'), 3, SCAN), ('shift0.841', ('13', '22'), 3, SCAN), ('shift1.414', ('13', '22'), 3, SCAN),
           ('shift0.707', ('13', '22'), 3, SCAN), ('shift2', ('13', '22'), 3, SCAN), ('shift0.5', ('13', '22'), 3, SCAN),
           ('top0.9', ('13',), 3, 'the part-of-band test (top octave ×0.9 / the rest ×0.9)'),
           ('rest0.9', ('13',), 3, 'the part-of-band test (top octave ×0.9 / the rest ×0.9)'),
           ('octave', ('22',), 5, 'extra 22×22 seeds 3 and 4'), ('shift0.95', ('22',), 5, 'extra 22×22 seeds 3 and 4'),
           ('shift1.414', ('22',), 5, 'extra 22×22 seeds 3 and 4'), ('shift2', ('22',), 5, 'extra 22×22 seeds 3 and 4'),
           ('shift1.05', ('22',), 5, 'extra 22×22 seeds 3 and 4'), ('jitter0.1', ('22',), 5, 'extra 22×22 seeds 3 and 4'),
           ('shift0.9999', ('22',), 5, 'extra 22×22 seeds 3 and 4')]


def minus(t):
    return t.replace('-', '−')


def dse(field, k, n):
    v = field(k, n)
    if not v or v.get('diff_mean_points') is None:
        return '–'
    se = v.get('diff_se_points')
    return minus(f'{v["diff_mean_points"]:+.1f}') + (f' ± {se:.1f}' if se else '')


def pm(field, k, n, m, scale=1.0, p=3):
    v = field(k, n)
    d = (v or {}).get('paired_metrics', {}).get(m)
    return '–' if not d else minus(f'{scale * d["mean"]:+.{p}f}')


def loss_pct(field, k, n):
    v = field(k, n)
    d = (v or {}).get('paired_metrics', {}).get('loss_final_mean100')
    import math
    return '–' if not d else minus(f'{100 * (math.exp(d["mean"]) - 1):+.0f} %')


def lockerr(field, k, n):
    v = field(k, n)
    return '–' if not v or v.get('edge_phase_error_top') is None else f'{v["edge_phase_error_top"]:.1f}'


def wall(field, k, n):
    v = field(k, n)
    return '–' if not v else pct(v['wall_rate'])


def wl(field, k, n):
    v = field(k, n)
    if not v or not v.get('paired'):
        return ''
    seeds = len(v['paired'])
    if v['losses'] == seeds:
        return 'every seed lost' if seeds > 1 else 'one seed so far'
    return f'{v["losses"]} of {seeds} seeds lost'


def gse(S, g, n, m='success_rate', scale=100.0, p=1):
    """Pooled change of a group of variants (report.GROUPS) with its standard error."""
    d = ((S.get('groups') or {}).get(g) or {}).get(n, {}).get(m)
    if not d:
        return '–'
    return minus(f'{scale * d["mean"]:+.{p}f} ± {scale * d["se"]:.{p}f}')


def gloss(S, g, n):
    import math
    d = ((S.get('groups') or {}).get(g) or {}).get(n, {}).get('loss_final_mean100')
    return '–' if not d else minus(f'{100 * (math.exp(d["mean"]) - 1):+.0f} %')


def stall_pct(S, n, k, key):
    """Share of a variant's planning-seed-0 runs in one stall class (report/stalls.json; 'octave' = the exact pool)."""
    g = ((S.get('stalls') or {}).get(n) or {}).get(k)
    return '–' if not g or not g.get('runs') else f'{100 * g[key] / g["runs"]:.1f} %'


def only2(field):
    """'only ' when x2 settles later than the octave on 22x22 but earlier than x1.41 (the contrast with the unlocked higher shifts)."""
    st = lambda k: ((field(k, '22') or {}).get('paired_metrics') or {}).get('loss_settle_epoch', {}).get('mean')
    a, b = st('shift2'), st('shift1.414')
    return 'only ' if a is not None and b is not None and 0 < a < b else ''


def settle(field, k, n, signed=False):
    """Change in the loss settling epoch against the exact-frequency trainings: signed epochs, or 'N epochs later/earlier'."""
    d = ((field(k, n) or {}).get('paired_metrics') or {}).get('loss_settle_epoch')
    if not d:
        return '–'
    if signed:
        return minus(f'{d["mean"]:+,.0f}')
    return f'{abs(d["mean"]):,.0f} epochs ' + ('later' if d['mean'] > 0 else 'earlier')


def absv(field, k, n, m, p=2):
    v = (field(k, n) or {}).get(m)
    return '–' if v is None else f'{v:.{p}f}'


def settle_abs(field, k, n):
    v = (field(k, n) or {}).get('loss_settle_epoch')
    return '–' if v is None else f'{100 * round(v / 100):,.0f}'


def span(field, keys, n):
    """Lowest and highest paired success change over a set of variants, as text."""
    ds = [field(k, n)['diff_mean_points'] for k in keys if field(k, n) and field(k, n).get('diff_mean_points') is not None]
    return ('–', '–') if not ds else (minus(f'{min(ds):+.1f}'), minus(f'{max(ds):+.1f}'))


def pending(S, field):
    left = []
    for k, sizes, want, name in PLANNED:
        if any(not field(k, n) or len(field(k, n)['seeds']) < want for n in sizes) and name not in left:
            left.append(name)
    return left


def ANSWER(S, field, rate, diff):
    sd13 = (S['noise'].get('13') or {}).get('sd_run_points')
    sd22 = (S['noise'].get('22') or {}).get('sd_run_points')
    left = pending(S, field)
    status = ('Interim. Still running: ' + '; '.join(left) + '. The page is regenerated as runs finish.') if left else 'All planned runs are in.'
    return f"""<section class="answer" aria-label="Answer">
<h2>{"Answer so far" if left else "Answer"}: the frequency level matters, the exact octave values matter little</h2>
<p class="note">{status}</p>
<ul>
<li><strong>Sitting exactly on the octaves matters little for success.</strong> No single slight detuning that keeps the band where it
is or moves it a little higher can be told apart from training noise. Each row moved by its own random ±5, ±10 or ±20 % (jitter): {dse(field, 'jitter0.05', '13')},
{dse(field, 'jitter0.1', '13')} and {dse(field, 'jitter0.2', '13')} points on 13×13, {dse(field, 'jitter0.05', '22')}, {dse(field, 'jitter0.1', '22')}
and {dse(field, 'jitter0.2', '22')} on 22×22. Every row 5 % or 10 % higher: {dse(field, 'shift1.05', '13')} and {dse(field, 'shift1.1', '13')} on
13×13, {dse(field, 'shift1.05', '22')} and {dse(field, 'shift1.1', '22')} on 22×22. Pooled over these five and the fill below:
{gse(S, 'not_lower', '13')} points on 13×13 ({rate('octave', '13')} for the exact octave), no measurable change, and {gse(S, 'not_lower', '22')}
on 22×22 ({rate('octave', '22')}), a small loss at about two standard errors that comes with slower convergence (below). For scale, two trainings with the same frequencies to 1e-4 differ by {sd13:.1f} points (SD of one run) on 13×13
and {sd22:.1f} on 22×22.</li>
<li><strong>No discrete frequency is needed at all.</strong> The log-uniform fill spreads every octave evenly over ±half an octave, so the
128 rows cover the band continuously: no two rows share a frequency and none is locked to the grid. It solves {rate('jitter0.4142', '13')} of
the 13×13 runs ({dse(field, 'jitter0.4142', '13')} points) and {rate('jitter0.4142', '22')} of the 22×22 runs ({dse(field, 'jitter0.4142', '22')}).
What the octave method needs is rows across this band, not the octave values themselves.</li>
<li><strong>Moving every row lower does matter, and the lock does not rescue it.</strong> 10 % lower: {dse(field, 'shift0.9', '13')} points on 13×13 ({wl(field, 'shift0.9', '13')})
and {dse(field, 'shift0.9', '22')} on 22×22 ({wl(field, 'shift0.9', '22')}); 10 % higher: {dse(field, 'shift1.1', '13')} and
{dse(field, 'shift1.1', '22')}. The two shifts break the lock by exactly the same amount (the top octave ends {lockerr(field, 'shift0.9', '13')} rad
out of phase at the 13×13 border, {lockerr(field, 'shift0.9', '22')} rad at the 22×22 border), so the loss comes from the lower frequencies,
not from the broken lock. 5 % lower sits in between: {dse(field, 'shift0.95', '13')} on 13×13 and {dse(field, 'shift0.95', '22')} on 22×22.
Further down, success keeps falling, and ×0.5, back on the octave positions exactly one octave lower and locked to the grid, is the
worst variant of all: {dse(field, 'shift0.5', '13')} points on 13×13 and {dse(field, 'shift0.5', '22')} on 22×22.</li>
<li><strong>What the lower frequencies lose is the walls, and the top octave carries it.</strong> With every row 10 % lower,
{wall(field, 'shift0.9', '13')} of the 13×13 runs reach the goal through a wall, against {wall(field, 'octave', '13')} for the exact octave
({wall(field, 'shift0.9', '22')} against {wall(field, 'octave', '22')} on 22×22), and the training loss ends {loss_pct(field, 'shift0.9', '13')}
and {loss_pct(field, 'shift0.9', '22')} higher. The wall barrier weakens as much near the maze centre as near the border, where a broken lock
would show most (change in near-wall slope, outer minus inner third of the maze, for ×0.9 and ×0.95: {gse(S, 'ring_down', '13', 'slope_outer_minus_inner', 1, 2)}
on 13×13, {gse(S, 'ring_down', '22', 'slope_outer_minus_inner', 1, 2)} on 22×22). {band_sentence(field)}</li>
<li><strong>Going higher than the octave gains no success, and far higher trades wall crossings for local minima.</strong> Every row
×1.19, ×1.41 and ×2 (locked again one octave up, a half period of half a cell): {dse(field, 'shift1.189', '13')},
{dse(field, 'shift1.414', '13')} and {dse(field, 'shift2', '13')} points on 13×13, {dse(field, 'shift1.189', '22')},
{dse(field, 'shift1.414', '22')} and {dse(field, 'shift2', '22')} on 22×22. At ×2 the walls hold: {wall(field, 'shift2', '13')} and
{wall(field, 'shift2', '22')} of the runs reach the goal through a wall, against {wall(field, 'octave', '13')} and {wall(field, 'octave', '22')}
for the exact octave, and the training loss ends {loss_pct(field, 'shift2', '13')} and {loss_pct(field, 'shift2', '22')}. But the learned
travel time gains local minima along the true paths ({pm(field, 'shift2', '13', 'bumps', 1, 2)} and {pm(field, 'shift2', '22', 'bumps', 1, 2)}
per path, where the exact octave has {absv(field, 'octave', '13', 'bumps')} and {absv(field, 'octave', '22', 'bumps')}), and the planner stalls in them: the runs that never arrive change by {pm(field, 'shift2', '13', 'never_arrived_rate', 100, 1)} and
{pm(field, 'shift2', '22', 'never_arrived_rate', 100, 1)} points, mostly stopped in open corridor, many of them on the true route. On 13×13,
where the octave's runs rarely cross a wall, the stalls cost success; on 22×22, where {wall(field, 'octave', '22')} of them do, the two
roughly cancel.</li>
<li><strong>What the exact values do buy: faster convergence on the larger maze and a slightly better fit.</strong> On 22×22, success at
epoch 2,000 changes by {gse(S, 'not_lower', '22', 'succ_at_2000')} points pooled over the six detunings that do not lower the band (the fill
{pm(field, 'jitter0.4142', '22', 'succ_at_2000', 100, 1)}, ×0.9 {pm(field, 'shift0.9', '22', 'succ_at_2000', 100, 1)}); by epoch 5,000 the gap
has narrowed to {gse(S, 'not_lower', '22')}, the small final loss above. On 13×13 the lag is small ({gse(S, 'not_lower', '13', 'succ_at_1000')} at epoch 1,000,
{gse(S, 'not_lower', '13', 'succ_at_2000')} at 2,000). At epoch 5,000 the same six end with a higher training loss ({gloss(S, 'not_lower', '13')}
on 13×13, {gloss(S, 'not_lower', '22')} on 22×22), a lower held-out speed correlation ({gse(S, 'not_lower', '13', 'speed_corr', 1, 3)} and
{gse(S, 'not_lower', '22', 'speed_corr', 1, 3)}) and a weaker near-wall slope ({gse(S, 'not_lower', '13', 'slope_2_8', 1, 2)} and
{gse(S, 'not_lower', '22', 'slope_2_8', 1, 2)}, target 1). These changes are consistent across variants but small: they cost no measurable success on 13×13 and about 2 points on 22×22.
The scan agrees: every shift between the octave and ×2 (×1.05 to ×1.41) ends with a slightly higher training loss than the octave
({loss_pct(field, 'shift1.414', '22')} at ×1.41 on 22×22), and on 22×22 its loss settles later the higher it sits (×1.41: {settle(field, 'shift1.414', '22')} than the octave), while ×2,
locked again, ends lower and settles {only2(field)}{settle(field, 'shift2', '22')}.</li>
<li><strong>The gain over ntrl-demo survives.</strong> Against the exact octave, ntrl-demo changes success by {dse(field, 'ntrl-demo', '13')}
points on 13×13 and {dse(field, 'ntrl-demo', '22')} on 22×22; the detunings that do not lower the band keep nearly all of that gain.</li>
</ul>
</section>"""


HIGHER = ('shift1.05', 'shift1.1', 'shift1.189', 'shift1.414', 'shift2')


def SCAN_TEXT(S, field, rate, diff):
    lo22, hi22 = span(field, HIGHER, '22')
    return f"""<p>Every row multiplied by the same factor. The exact octave sits at ×1; ×0.5 and ×2 are locked again, one octave lower and
higher. Big dots are the pooled success over training seeds, small dots single seeds; the rules are the earlier fields on the same maze
and queries.</p>
<p>Below the octave, success falls as the band moves down: from {dse(field, 'shift0.95', '13')} points at ×0.95 to
{dse(field, 'shift0.5', '13')} at ×0.5 on 13×13, and from {dse(field, 'shift0.95', '22')} to {dse(field, 'shift0.5', '22')} on 22×22. Above it,
success changes little: every factor from ×1.05 to ×2 lies between {lo22} and {hi22} points on 22×22, and on 13×13 it slips slowly
to {dse(field, 'shift2', '13')} at ×2. The two locked bands, ×0.5 and ×2, lie on the trend of their unlocked neighbours: being back on
the octave positions adds nothing to success.</p>"""


def FAIL_TEXT(S, field, rate, diff):
    def w(k, n):
        return pm(field, k, n, 'wall_rate', 100, 1)

    def nv(k, n):
        return pm(field, k, n, 'never_arrived_rate', 100, 1)
    return f"""<p>A run fails in one of two ways: it reaches the goal only after crossing a wall, or it never reaches the goal (the planner
stalls, usually in a local minimum of the field). Each point is the change in one kind against the same seed's exact-frequency trainings,
mean over training seeds, with ±2 standard errors from training noise as the whisker; the exact octave is the reference at zero.</p>
<p>The two kinds move in opposite directions. Lowering the band turns stalls into wall crossings: on 22×22, ×0.5 changes the wall
crossings by {w('shift0.5', '22')} points and the runs that never arrive by {nv('shift0.5', '22')}. Raising it turns crossings into
stalls: ×2 changes them by {w('shift2', '22')} and {nv('shift2', '22')}. On 13×13 the octave's runs rarely fail either way
({wall(field, 'octave', '13')} cross a wall); there ×0.5 adds {w('shift0.5', '13')} points of wall crossings and ×2 adds
{nv('shift2', '13')} points of runs that never arrive.</p>"""


def STALL_TEXT(S, field, rate, diff):
    def s(n, k, key):
        return stall_pct(S, n, k, key)
    return f"""<p>Where a run that never arrives stops tells the two causes apart. The planner's last position (planning seed 0, the only
seed whose paths are kept) either presses against a wall, clearance under 0.15 cell (the field leads into the wall and the wall holds),
or sits in the open corridor, clearance 0.15 cell or more of a 0.9-cell corridor: a local minimum of the learned travel time away from
any wall. Rows run from the lowest factor at the top to the highest at the bottom; hover a dot for how many of the stops lie on the
true route.</p>
<p>On 22×22 the exact octave's stalled runs mostly press against a wall ({s('22', 'octave', 'stop_wall')} of all runs, against
{s('22', 'octave', 'stop_open')} in the open). The lower bands stop against a wall less often ({s('22', 'shift0.5', 'stop_wall')} at ×0.5):
their softer walls let the planner through, into the wall crossings above. ×2 turns this around: {s('22', 'shift2', 'stop_open')} of its runs
stop in the open corridor, and {s('22', 'shift2', 'stop_open_route')} of all its runs stop there on the true route, against
{s('22', 'octave', 'stop_open_route')} for the exact octave. Those runs were heading the right way and stalled in a ripple of the learned
travel time. 13×13 shows the same on a smaller scale: {s('13', 'shift2', 'stop_open')} of the ×2 runs stop in the open, against
{s('13', 'octave', 'stop_open')}.</p>"""


def PAIRED_TEXT(S, field, rate, diff):
    return """<p>Each field against the trainings of the same seed that have the exact octave frequencies: the exact octave, the two nulls
(×1.0001 and ×0.9999: the same frequencies, a different floating-point path) and the grid-sync run where it ran on the other GPU.
Same seed means the same draw, initialization and data order. Small dots are seeds, the bar is their mean with ±2 standard errors.
The band is how far a single seed lands from that reference by training noise alone (±2 SD). The grid-sync controls are the
earlier lock tests: magnitudes shuffled within an octave, and constant magnitude per octave. Changes beyond ±16 points are drawn at
the edge with their value written out.</p>"""


def band_sentence(field):
    """The part-of-band result for the answer (13x13), or the pending note."""
    if not (field('top0.9', '13') and field('rest0.9', '13')):
        return ('The likely reason is resolution: the exact top octave has a half period of one cell, the finest scale of the maze, and '
                '10 % lower it spans 1.1 cells. The part-of-band runs in the queue test this directly.')
    return (f"The top octave is where the exact octave meets the finest scale of the maze, a half period of one cell. Lowering only "
            f"the top octave by 10 % (the 13×13 part-of-band runs) reproduces the whole fit loss of the ×0.9 shift: training loss "
            f"{loss_pct(field, 'top0.9', '13')} against {loss_pct(field, 'shift0.9', '13')}, held-out speed correlation "
            f"{pm(field, 'top0.9', '13', 'speed_corr')} against {pm(field, 'shift0.9', '13', 'speed_corr')}, and "
            f"{pm(field, 'top0.9', '13', 'wall_rate', 100, 1)} against {pm(field, 'shift0.9', '13', 'wall_rate', 100, 1)} points of wall "
            f"crossings. Lowering every other octave by 10 % with the top exact changes nothing measurable (loss "
            f"{loss_pct(field, 'rest0.9', '13')}, wall crossings {pm(field, 'rest0.9', '13', 'wall_rate', 100, 1)} points, success "
            f"{dse(field, 'rest0.9', '13')}). Raising the top octave above the cell scale (×1.1: half period 0.9 cell) costs nothing, so "
            f"what matters is that the band reaches the cell scale, not that it sits on it.")


def BAND_TEXT(S, field, rate, diff):
    top, rest = field('top0.9', '13'), field('rest0.9', '13')
    intro = ("""<p>The ×0.9 shift lowers every octave. Two runs per 13×13 seed split it: only the top octave ×0.9 (its 32 rows have a
half period of one cell; every other row stays exactly on its octave) and every row except the top ×0.9 (the top octave exact).
Together the two make the ×0.9 shift. Each panel shows the change against the same seed's exact-frequency trainings: seeds as small
dots, the mean as a bar, ±2 standard errors from training noise as the whisker.""")
    if not (top and rest):
        return intro + '</p>'
    return intro + f""" Lowering only the top octave reproduces the fit loss of the whole shift (final loss
{loss_pct(field, 'top0.9', '13')} against {loss_pct(field, 'shift0.9', '13')}, speed correlation {pm(field, 'top0.9', '13', 'speed_corr')} against
{pm(field, 'shift0.9', '13', 'speed_corr')}) and about half of its extra wall crossings; lowering the rest changes nothing measurable. On success
alone the split is within noise: {dse(field, 'top0.9', '13')} and {dse(field, 'rest0.9', '13')} points, against {dse(field, 'shift0.9', '13')} for the
whole shift.</p>"""


def LOCK_TEXT(S, field, rate, diff):
    return """<p>Each variant's mean paired difference against how far its top-octave rows are from the lock: the phase mismatch with an
exactly locked row at the outermost grid line (median over the top octave's rows). The grid-sync controls break the lock by
reassigning magnitudes instead of scaling them. If the lock is what matters, the points fall along one downward trend. They do not:
×0.9 and ×1.1 have the same lock error, and only the lower one loses. ×0.5 and ×2 are locked (zero error), one octave lower and higher.
The wide-scan shifts (×0.71 to ×1.41) are left out here: they move the whole band by a quarter octave or more, and their lock error
(several radians) is shown by the scan above.</p>"""


def CONV_TEXT(S, field, rate, diff):
    return """<p>Success with planning seed 0 at the saved checkpoints (every 500 epochs here, every 1,000 for the earlier fields),
all evaluated on the same GPU, mean over training seeds 87358, 1 and 2 for every line (the extra 22×22 seeds 3 and 4 are left out here so
that every line has the same seeds); the loss is ntrl-demo's own training loss from the console log, in 50-epoch means. Jitter is the mean of
±5, ±10 and ±20 % where complete; best B × s is ntrl-demo's random matrix scaled ×3 (13×13) or ×6 (22×22).</p>"""


def EPOCHS_TEXT(S, field, rate, diff):
    return f"""<p>Convergence along the scan: success at epochs 1,000, 2,000 and 5,000 for every coherent shift with all its training seeds in
(planning seed 0, mean over training seeds 87358, 1 and 2). A shift that only converges more slowly sits lower at 1,000 and 2,000 but meets the others at 5,000; one that converges
to a worse field stays lower at 5,000.</p>
<p>The lower shifts converge to a worse field. The higher shifts up to ×1.41 lag on 22×22 at epoch 2,000 (×1.41
{pm(field, 'shift1.414', '22', 'succ_at_2000', 100, 1)} points) and are level with the octave within noise by 5,000 (between
{span(field, HIGHER[:4], '22')[0]} and {span(field, HIGHER[:4], '22')[1]} points). ×2 is the exception both ways: on 22×22 it leads early
({pm(field, 'shift2', '22', 'succ_at_1000', 100, 1)} points at epoch 1,000, {pm(field, 'shift2', '22', 'succ_at_2000', 100, 1)} at 2,000) and ends
level ({dse(field, 'shift2', '22')}); on 13×13 it trails throughout ({pm(field, 'shift2', '13', 'succ_at_1000', 100, 1)} at 1,000,
{dse(field, 'shift2', '13')} at 5,000).</p>"""


def LOSSF_TEXT(S, field, rate, diff):
    hs = [settle(field, k, '22', signed=True) for k in ('shift1.05', 'shift1.1', 'shift1.189', 'shift1.414')]
    h = ', '.join(hs[:-1]) + ' and ' + hs[-1]
    return f"""<p>The training loss along the scan, against the same seed's exact-frequency trainings (whiskers ±2 standard errors). The final
loss is the mean of the last 100 epochs. The settling epoch is the first epoch whose trailing 100-epoch mean is within 10 % of that run's
final loss: the exact octave settles at about {settle_abs(field, 'octave', '13')} epochs on 13×13 and {settle_abs(field, 'octave', '22')} on 22×22.</p>
<p>The final loss is lowest at the two locked bands that reach the cell scale, ×1 and ×2. Every unlocked shift between them ends a few percent
higher than the octave although its band is higher, so being locked to the grid helps the fit a little; the scan cannot tell how much of
the ×2 gain is the lock and how much the half-cell octave it adds. Lower bands settle earlier because they stop improving sooner. On 22×22 the unlocked higher
bands keep improving longer the higher they sit (settling epoch at ×1.05, ×1.1, ×1.19 and ×1.41: {h} epochs against the octave),
and ×2, locked, settles {only2(field)}{settle(field, 'shift2', '22')} than the octave. On 13×13 every shift settles within noise of the octave's epoch or earlier.</p>"""


def CMAP_TEXT(S, field, rate, diff):
    return """<p>Change of every measure against the exact-frequency trainings of the same seed, mean over seeds, at epoch 5,000 (success
at 1,000 and 2,000 epochs with planning seed 0). Green is better, red worse. The shade is the change over its standard error from training
noise (the scatter of the exact-frequency trainings themselves); full shade at three standard errors, no shade below one. Success,
never-arrived and wall-contact shares in points, final loss in percent. Hover a cell for its standard error.</p>"""


def SPATIAL_TEXT(S, field, rate, diff):
    def g(grp, n, m):
        d = ((S.get('groups') or {}).get(grp) or {}).get(n, {}).get(m)
        if not d:
            return '–'
        v = f'{d["mean"]:+.2f}'
        return minus(('0.00' if v in ('+0.00', '-0.00') else v) + f' ± {d["se"]:.2f}')

    def oi(grp, n):
        return g(grp, n, 'slope_outer_minus_inner')

    def ring(grp, n, r):
        d = ((S.get('groups') or {}).get(grp) or {}).get(n, {}).get(f'slope_ring_{r}')
        return '–' if not d else minus(f'{d["mean"]:+.2f}')
    return f"""<p>A detuned row agrees with the locked row at the maze centre and drifts away from it toward the border (×1.1 ends
{lockerr(field, 'shift1.1', '22')} rad out of phase at the 22×22 border), so a lock effect would show as a wall barrier (near-wall slope, target 1)
lost in the outer third of the maze and kept in the inner third. Each point is the change against the same seed's exact-frequency
trainings, mean over the group's runs, with ±2 standard errors from training noise as the whisker; the zero line is the exact octave.</p>
<p>No detuning follows that pattern: each group loses about the same slope in every ring. Outer minus inner third, ×0.9 / ×0.95:
{oi('ring_down', '13')} on 13×13 and {oi('ring_down', '22')} on 22×22; ×1.05 / ×1.1: {oi('ring_up', '13')} and {oi('ring_up', '22')};
jitter: {oi('ring_jitter', '13')} and {oi('ring_jitter', '22')}. The one difference beyond a standard error, the lower shifts on 13×13, goes the
other way: they lose more slope at the centre ({ring('ring_down', '13', 'inner')}) than at the border ({ring('ring_down', '13', 'outer')}). What the
detunings lose is spread over the whole maze, as a weaker fit would be, not gathered where the phase has drifted.</p>"""


METHOD = """<p>Mazes, data, network (656,900 parameters), losses, optimizer, 5,000 epochs, MPPI planner and exact scorer are those of
<code>jeffrey/maze_scale</code> / <code>jeffrey/grid_sync</code>; only the Fourier matrix changes. 13×13: 200 queries × planning seeds 0/1/2;
22×22: 240 queries. Training seeds 87358 / 1 / 2 on 13×13 and 22×22, plus 3 / 4 on 22×22. Each (maze, seed) group runs on one GPU;
the exact octave rerun on the 5070 is bit-identical to the grid-sync run.</p>
<p>Part of the band: <code>top0.9</code> multiplies only the top octave's rows (the quarter of the rows on
f_grid, 13×13) by 0.9 and leaves every other row exactly locked; <code>rest0.9</code> multiplies every other row by 0.9 and keeps the top
octave exact. Together they make the ×0.9 shift.</p>
<p>Failures are split by arrival (the planner appends the goal to every path, so the collision flag alone counts stalls as wall contacts).
Near-wall slope: median S·|∇T| at held-out points with trained speed below 0.3 (target 1); minima per path: local minima of T along the
true paths of queries of 8 cells or more.</p>"""
