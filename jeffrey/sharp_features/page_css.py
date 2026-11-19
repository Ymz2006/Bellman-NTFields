"""CSS and tooltip script of the sharp-feature findings page (from jeffrey/octave_detune/page_text.py, role colours added)."""

CSS = r"""
:root {
  /* Layout: one reading column; each result is a pair of small multiples (13x13 | 22x22); colour = which idea. */
  --bg: #f5f6f8; --surface: #fcfcfd; --ink: #111418; --ink-2: #4b525b; --muted: #858c96; --grid: #e3e6ea; --base: #c5cad1;
  --line: rgba(17, 20, 24, 0.10); --band: rgba(42, 120, 214, 0.10); --pos: 27, 135, 82; --neg: 200, 62, 62; --tile: rgba(17, 20, 24, 0.04);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --s6: #008300; --s7: #6250d6; --s8: #e34948;
  --q1: #b3a9ee; --q2: #7b6be0; --q3: #3f2f9e;
  --c-pow: #eb6834; --c-gate: #2a78d6; --c-tri: #1baf7a;
  --display: "Space Grotesk", "Segoe UI", system-ui, sans-serif; --text: "Inter", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0e1114; --surface: #161a1e; --ink: #eef1f4; --ink-2: #b8bfc8; --muted: #8a919b; --grid: #262b31; --base: #3a4149;
    --line: rgba(255, 255, 255, 0.10); --band: rgba(57, 135, 229, 0.16); --pos: 64, 190, 120; --neg: 236, 105, 105; --tile: rgba(255, 255, 255, 0.05);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #0ca30c; --s7: #9085e9; --s8: #e66767;
    --q1: #574ab8; --q2: #8d82e8; --q3: #d0c9f9;
    --c-pow: #d95926; --c-gate: #3987e5; --c-tri: #199e70;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #0e1114; --surface: #161a1e; --ink: #eef1f4; --ink-2: #b8bfc8; --muted: #8a919b; --grid: #262b31; --base: #3a4149;
  --line: rgba(255, 255, 255, 0.10); --band: rgba(57, 135, 229, 0.16); --pos: 64, 190, 120; --neg: 236, 105, 105; --tile: rgba(255, 255, 255, 0.05);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #0ca30c; --s7: #9085e9; --s8: #e66767;
  --q1: #574ab8; --q2: #8d82e8; --q3: #d0c9f9;
  --c-pow: #d95926; --c-gate: #3987e5; --c-tri: #199e70;
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
.ln.r-pow { stroke: var(--c-pow); } .ln.r-tri { stroke: var(--c-tri); } .ln.r-gate { stroke: var(--c-gate); } .ln.r-ctl { stroke: var(--ink-2); }
.ln.r-ctx { stroke: var(--muted); } .ln.r-ink { stroke: var(--ink); }
.ln.op0 { opacity: 0.38; } .ln.op1 { opacity: 0.58; } .ln.op2 { opacity: 0.78; } .ln.op3 { opacity: 1; }
.dot.r-pow, .seed.r-pow { fill: var(--c-pow); } .dot.r-tri, .seed.r-tri { fill: var(--c-tri); } .dot.r-gate, .seed.r-gate { fill: var(--c-gate); }
.dot.r-ctl, .seed.r-ctl { fill: var(--ink-2); } .dot.r-ctx, .seed.r-ctx { fill: var(--muted); } .dot.r-ink { fill: var(--ink); }
.dot.hollow.r-ink { fill: var(--surface); stroke: var(--ink); stroke-width: 2; }
.dot.hollow.r-pow { fill: var(--surface); stroke: var(--c-pow); stroke-width: 2; } .dot.hollow.r-tri { fill: var(--surface); stroke: var(--c-tri); stroke-width: 2; }
.dot.hollow.r-gate { fill: var(--surface); stroke: var(--c-gate); stroke-width: 2; } .dot.hollow.r-ctl { fill: var(--surface); stroke: var(--ink-2); stroke-width: 2; }
.dot.hollow.r-ctx { fill: var(--surface); stroke: var(--muted); stroke-width: 2; }
.dot.sq.r-pow { fill: var(--c-pow); stroke: var(--ink); stroke-width: 1.5; }
.meanbar.r-pow, .whisk.r-pow { stroke: var(--c-pow); } .meanbar.r-tri, .whisk.r-tri { stroke: var(--c-tri); } .meanbar.r-gate, .whisk.r-gate { stroke: var(--c-gate); }
.meanbar.r-ctl, .whisk.r-ctl { stroke: var(--ink-2); } .meanbar.r-ctx, .whisk.r-ctx { stroke: var(--muted); }
.rowlab { fill: var(--ink-2); font-size: 11.5px; }
.refdot { stroke: var(--muted); stroke-width: 1.4; stroke-dasharray: 2 3; }
.full svg { width: 100%; height: auto; display: block; }
.status { font-size: 13px; color: var(--ink-2); background: var(--tile); border-radius: 8px; padding: 8px 12px; max-width: 76ch; }
.shapes { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(280px, 100%), 1fr)); gap: 14px; }
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


