"""Page style and tooltip script shared with jeffrey/lbs_final/page.py (the same design tokens and validated chart palette)."""
CSS = '''
:root {
  /* Layout: one reading column inside a 1120px frame; charts take the frame width; big tables live in <details>. */
  --bg: #f5f6f8; --surface: #fcfcfb; --ink: #15181d; --ink-2: #4a505e; --ink-3: #777e8e; --rule: #dde1e7; --grid: #e7e9ed;
  --accent: #1f5fb4; --code-bg: #eceff3;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4;
  --font-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-head: "IBM Plex Sans Condensed", "IBM Plex Sans", system-ui, sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #121418; --surface: #1a1a19; --ink: #eceef2; --ink-2: #b6bbc7; --ink-3: #8a90a0; --rule: #2d3139; --grid: #2a2d33;
  --accent: #7fb0f0; --code-bg: #20232a; --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #121418; --surface: #1a1a19; --ink: #eceef2; --ink-2: #b6bbc7; --ink-3: #8a90a0; --rule: #2d3139; --grid: #2a2d33;
  --accent: #7fb0f0; --code-bg: #20232a; --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; color-scheme: dark; }
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


