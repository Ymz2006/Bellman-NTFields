"""Findings page: report/embed_warp.html from report/runs.json (report.py), the figures (figs.py) and the mechanism panels.

    python jeffrey/embed_warp/page.py        # then publish report/embed_warp.html with files report/*.png

Charts are inline SVG drawn here from the pooled numbers (one scale per chart, theme tokens for every color); large tables sit under
<details>. Prose numbers are read from the data, never typed.
"""
import html
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE / 'report'
BINS = ['0.5-2', '2-4', '4-8', '8-16', '16-32', '32-64', '64-128']
SCENES = [('plus', 'plus maze (22 x 22, 6-cell open cross)'), ('disks', 'disks (22-cell box, 40 disks)'), ('gibson', 'Gibson apartment (64 cells across)'), ('gibson32', 'Gibson apartment (32 cells across)')]
# categorical slots (reference palette, fixed order): ntrl, oct, geo, geo8, learn, other
SLOT = dict(ntrl='--s1', oct='--s2', geo='--s3', geo8='--s4', learn='--s5', learn_in='--s6', geo0='--s3', geob='--s3', oct8='--s2')
NAMES = dict(ntrl='ntrl-demo', oct='octaves always on', geo='clearance gate (x2..x16)', geo8='clearance gate (x2..x8)', learn='learned gate', learn_in='learned gate + inside pairs',
             geo0='clearance gate, mu_min 0', geob='clearance gate, base x1.41', oct8='octaves x2..x8 always on', kde='band-density gate', diff='diffusion gate', curv='curvature gate',
             curvr='eikonal-misfit gate', geot='tight band', geow='wide band', geo24='24 rows per octave', geo32='x2..x32', geo64='x4..x64', oct32='x2..x32 always on', geom3='mu_min 0.3',
             learnb='learned gate, budget', learnx='learned gate on x', geo4='x2,x4', geo8_0='x2..x8, mu_min 0', geo8b='x2..x8, base x1.41', learn8='learned gate (x2..x8)', learn8_in='learned (x2..x8) + inside pairs',
             ntrl_in='ntrl-demo + inside pairs', oct_in='always on + inside pairs', geo_in='clearance gate + inside pairs', ntrl_td='ntrl-demo, TD step 0.05 cells', geo_td='clearance gate, TD 0.05',
             geo0_td='clearance gate mu_min 0, TD 0.05')


def esc(s):
    return html.escape(str(s))


def load():
    d = json.load(open(REPORT / 'runs.json'))
    pool = {(p['scene'], p['label']): p for p in d['pooled']}
    poolq = {(p['scene'], p['label']): p for p in d.get('evalq_pooled', [])}
    return d['rows'], d['pooled'], pool, poolq


def pct(pool, scene, label, nd=1):
    p = pool.get((scene, label))
    return f'{p["pct"]:.{nd}f}' if p else '–'


def seeds_txt(p):
    return ' / '.join(f'{v:.1f}' for v in p['per_seed'].values())


# --- SVG charts --------------------------------------------------------------------------------------------------------
def bar_chart(items, title, ymax=100, width=640):
    """items: [(label, pct, per_seed list, css var)] horizontal bars with per-seed dots."""
    n = len(items)
    rowh, L, R, T, B = 26, 230, 60, 28, 26
    H = T + B + rowh * n
    W = width
    X = lambda v: L + (W - L - R) * v / ymax
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}" class="chart"><title>{esc(title)}</title>']
    for t in range(0, ymax + 1, 25):
        out.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{T - 6}" y2="{H - B}" class="grid"/><text x="{X(t):.1f}" y="{H - B + 14}" text-anchor="middle" class="tick">{t}</text>')
    for i, (lab, v, per, css) in enumerate(items):
        y = T + rowh * i
        out.append(f'<text x="{L - 8}" y="{y + rowh / 2 + 4}" text-anchor="end" class="lab">{esc(lab)}</text>')
        out.append(f'<rect x="{L}" y="{y + 5}" width="{max(X(v) - L, 0):.1f}" height="{rowh - 10}" rx="3" fill="var({css})"><title>{esc(lab)}: {v:.1f} %</title></rect>')
        for s in per:
            out.append(f'<circle cx="{X(s):.1f}" cy="{y + rowh / 2:.1f}" r="3.5" fill="var(--surface)" stroke="var(--fg)" stroke-width="1.2"><title>seed: {s:.1f} %</title></circle>')
        out.append(f'<text x="{X(v) + 6:.1f}" y="{y + rowh / 2 + 4}" class="val">{v:.1f}</text>')
    out.append(f'<text x="{W - R}" y="{H - 4}" text-anchor="end" class="tick">success, % of query runs</text></svg>')
    return '\n'.join(out)


def line_chart(series, bins, title, width=640):
    """series: [(label, [pct or None per bin], css)]."""
    L, R, T, B = 44, 16, 24, 40
    W, H = width, 260
    X = lambda i: L + (W - L - R) * i / max(len(bins) - 1, 1)
    Y = lambda v: T + (H - T - B) * (1 - v / 100)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}" class="chart"><title>{esc(title)}</title>']
    for t in (0, 25, 50, 75, 100):
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/><text x="{L - 6}" y="{Y(t) + 4:.1f}" text-anchor="end" class="tick">{t}</text>')
    for i, b in enumerate(bins):
        out.append(f'<text x="{X(i):.1f}" y="{H - B + 16}" text-anchor="middle" class="tick">{esc(b)}</text>')
    out.append(f'<text x="{(L + W - R) / 2:.1f}" y="{H - 6}" text-anchor="middle" class="tick">true travel distance, cells</text>')
    for lab, vals, css in series:
        pts = [(X(i), Y(v)) for i, v in enumerate(vals) if v is not None]
        if not pts:
            continue
        d = 'M' + ' L'.join(f'{x:.1f},{y:.1f}' for x, y in pts)
        out.append(f'<path d="{d}" fill="none" stroke="var({css})" stroke-width="2" stroke-linejoin="round"/>')
        for (x, y), v in zip(pts, [v for v in vals if v is not None]):
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="var({css})" stroke="var(--surface)" stroke-width="1.5"><title>{esc(lab)}: {v:.0f} %</title></circle>')
        x, y = pts[-1]
        out.append(f'<text x="{x + 6:.1f}" y="{y + 4:.1f}" class="lab" fill="var(--fg)">{esc(lab)}</text>')
    out.append('</svg>')
    return '\n'.join(out)


def stacked_fail(items, title, width=640):
    """items: [(label, wall, stall, solved, total, css)]: horizontal 100 % bars solved / arrived+wall / not arrived."""
    n = len(items)
    rowh, L, R, T, B = 26, 230, 16, 28, 26
    H = T + B + rowh * n
    W = width
    X = lambda v: L + (W - L - R) * v
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}" class="chart"><title>{esc(title)}</title>']
    for i, (lab, wall, stall, ok, tot, css) in enumerate(items):
        y = T + rowh * i
        out.append(f'<text x="{L - 8}" y="{y + rowh / 2 + 4}" text-anchor="end" class="lab">{esc(lab)}</text>')
        a, b, c = ok / tot, wall / tot, stall / tot
        out.append(f'<rect x="{X(0):.1f}" y="{y + 5}" width="{X(a) - X(0) - 1:.1f}" height="{rowh - 10}" rx="2" fill="var(--ok)"><title>solved {ok}/{tot}</title></rect>')
        out.append(f'<rect x="{X(a) + 1:.1f}" y="{y + 5}" width="{max(X(a + b) - X(a) - 2, 0):.1f}" height="{rowh - 10}" rx="2" fill="var(--wall)"><title>arrived through a wall {wall}</title></rect>')
        out.append(f'<rect x="{X(a + b) + 1:.1f}" y="{y + 5}" width="{max(X(1) - X(a + b) - 1, 0):.1f}" height="{rowh - 10}" rx="2" fill="var(--stall)"><title>never arrived (stall) {stall}</title></rect>')
        out.append(f'<text x="{X(a) - 4:.1f}" y="{y + rowh / 2 + 4}" text-anchor="end" class="val" fill="var(--surface)">{100 * a:.0f}</text>')
    out.append(f'<text x="{L}" y="{H - 4}" class="tick">solved · arrived through a wall · never arrived</text></svg>')
    return '\n'.join(out)


def scatter(points, title, xlabel, width=640):
    """points: [(x, y, label, css)]."""
    L, R, T, B = 44, 16, 20, 40
    W, H = width, 240
    xs = [p[0] for p in points]
    xmax = max(xs) * 1.1 if xs else 1
    X = lambda v: L + (W - L - R) * v / xmax
    Y = lambda v: T + (H - T - B) * (1 - v / 100)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}" class="chart"><title>{esc(title)}</title>']
    for t in (50, 60, 70, 80, 90, 100):
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/><text x="{L - 6}" y="{Y(t) + 4:.1f}" text-anchor="end" class="tick">{t}</text>')
    for t in range(0, int(xmax) + 1, 2):
        out.append(f'<text x="{X(t):.1f}" y="{H - B + 16}" text-anchor="middle" class="tick">{t}</text>')
    out.append(f'<text x="{(L + W - R) / 2:.1f}" y="{H - 6}" text-anchor="middle" class="tick">{esc(xlabel)}</text>')
    for x, y, lab, css in points:
        out.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="6" fill="var({css})" stroke="var(--surface)" stroke-width="1.5"><title>{esc(lab)}: stretch x{x:.1f}, {y:.1f} %</title></circle>')
        out.append(f'<text x="{X(x) + 8:.1f}" y="{Y(y) + 4:.1f}" class="lab">{esc(lab)}</text>')
    out.append('</svg>')
    return '\n'.join(out)


# --- page ---------------------------------------------------------------------------------------------------------------
def method_bars(pool, scene, labels):
    items = []
    for lab in labels:
        p = pool.get((scene, lab))
        if p:
            items.append((NAMES.get(lab, lab), p['pct'], list(p['per_seed'].values()), SLOT.get(lab, '--s6')))
    return items


def bin_series(pool, scene, labels):
    bins = [b for b in BINS if any(b in pool[(scene, l)]['by_bin'] for l in labels if (scene, l) in pool)]
    series = []
    for lab in labels:
        p = pool.get((scene, lab))
        if p:
            series.append((NAMES.get(lab, lab), [100 * p['by_bin'][b][0] / p['by_bin'][b][1] if b in p['by_bin'] and p['by_bin'][b][1] else None for b in bins], SLOT.get(lab, '--s6')))
    return series, bins


def fail_items(pool, scene, labels):
    items = []
    for lab in labels:
        p = pool.get((scene, lab))
        if p:
            items.append((NAMES.get(lab, lab), p['arrived_wall'], p['not_arrived'], p['success'], p['runs'], SLOT.get(lab, '--s6')))
    return items


def full_table(pooled):
    rows = ['<table><thead><tr><th>scene</th><th>method</th><th>seeds</th><th>success</th><th>%</th><th>per seed</th><th>through wall</th><th>never arrived</th><th>speed corr</th><th>wall slope</th><th>wall cos</th><th>attract</th><th>bumps/path</th><th>stretch band / open</th><th>aniso band</th><th>run ids</th></tr></thead><tbody>']
    for p in pooled:
        ib, io = p.get('iso_band') or {}, p.get('iso_open') or {}
        rows.append(f'<tr><td>{esc(p["scene"])}</td><td>{esc(NAMES.get(p["label"], p["label"]))} <code>{esc(p["label"])}</code></td><td>{len(p["seeds"])}</td><td>{p["success"]}/{p["runs"]}</td><td>{p["pct"]:.1f}</td><td>{esc(seeds_txt(p))}</td>'
                    f'<td>{p["arrived_wall"]}</td><td>{p["not_arrived"]}</td><td>{p["speed_corr"]:.2f}</td><td>{p["slope"]:.2f}</td><td>{p["cos"]:.2f}</td><td>{p["attract"]:.3f}</td><td>{p["bumps"]:.2f}</td>'
                    f'<td>{ib.get("stretch", float("nan")):.1f} / {io.get("stretch", float("nan")):.1f}</td><td>{ib.get("aniso", float("nan")):.2f}</td><td class="ids">{" ".join(esc(i) for i in p["ids"])}</td></tr>')
    rows.append('</tbody></table>')
    return '\n'.join(rows)


def where_table(pool):
    labs = [l for l in ('ntrl', 'oct', 'geo', 'geo0', 'geob', 'geo8', 'learn', 'learn_in') if ('plus', l) in pool and pool[('plus', l)].get('by_where')]
    rows = ['<table><thead><tr><th>method</th><th>both ends in the +</th><th>one end</th><th>neither (maze to maze)</th></tr></thead><tbody>']
    for l in labs:
        w = pool[('plus', l)]['by_where']
        rows.append(f'<tr><td>{esc(NAMES.get(l, l))}</td>' + ''.join(f'<td>{w[k][0]}/{w[k][1]} ({100 * w[k][0] / max(w[k][1], 1):.0f} %)</td>' for k in ('both', 'one', 'neither')) + '</tr>')
    rows.append('</tbody></table>')
    return '\n'.join(rows)


def fig(name, caption):
    p = REPORT / name
    if not p.exists():
        return ''
    return f'<figure><img src="{name}" alt="{esc(caption)}" loading="lazy"><figcaption>{caption}</figcaption></figure>'


def main():
    rows, pooled, pool, poolq = load()
    have = lambda s, l: (s, l) in pool
    main_labels = ['ntrl', 'oct', 'oct8', 'geo', 'geo0', 'geob', 'geo8', 'learn', 'learn_in']
    P = lambda l, nd=1: pct(pool, 'plus', l, nd)
    D = lambda l, nd=1: pct(pool, 'disks', l, nd)
    G = lambda l, nd=1: pct(pool, 'gibson', l, nd)
    g8 = pool.get(('plus', 'geo8'))
    nt = pool.get(('plus', 'ntrl'))
    stretch_pts = []
    for l in ('geo8', 'geo', 'geo24', 'geo32', 'geo64', 'geo4'):
        p = pool.get(('plus', l))
        if p and p.get('iso_band'):
            stretch_pts.append((p['iso_band']['stretch'], p['pct'], NAMES.get(l, l), SLOT.get(l, '--s6')))
    css = """
<style>
/* layout: one reading column, charts full width of it, figures edge to edge inside it */
:root{--bg:#f6f5f1;--surface:#ffffff;--fg:#1d1f24;--muted:#5d6068;--rule:#dcd9d0;--accent:#1c5cab;--chip:#e9e6dd;
 --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#6250d6;--ok:#1baf7a;--wall:#e34948;--stall:#b9b4a6;
 --display:"Source Serif 4",Georgia,"Times New Roman",serif;--body:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif;--mono:"IBM Plex Mono",Menlo,Consolas,monospace}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#15171b;--surface:#1e2127;--fg:#ecebe6;--muted:#a9adb6;--rule:#343842;--accent:#86b6ef;--chip:#2a2e36;
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#9085e9;--ok:#199e70;--wall:#e66767;--stall:#6b6f78;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#15171b;--surface:#1e2127;--fg:#ecebe6;--muted:#a9adb6;--rule:#343842;--accent:#86b6ef;--chip:#2a2e36;
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#9085e9;--ok:#199e70;--wall:#e66767;--stall:#6b6f78;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:var(--body);line-height:1.5;font-size:16px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:0 48px}
main{max-width:860px;margin:0 auto}
h1,h2,h3{font-family:var(--display);text-wrap:balance;line-height:1.15}
h1{font-size:2.1rem;margin:40px 0 6px;font-weight:600}
h2{font-size:1.45rem;margin:44px 0 10px;font-weight:600}
h3{font-size:1.1rem;margin:26px 0 6px;font-weight:600}
p,li{max-width:68ch}
.eyebrow{font-family:var(--mono);font-size:.78rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-top:28px}
.lede{font-size:1.12rem;max-width:66ch}
.verdict{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:22px 0 8px}
.tile{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:12px 14px}
.tile .n{font-family:var(--display);font-size:1.9rem;font-weight:600;font-variant-numeric:tabular-nums;line-height:1.1}
.tile .l{color:var(--muted);font-size:.86rem;margin-top:4px}
.chart{width:100%;height:auto;display:block;margin:10px 0 4px}
.chart .grid{stroke:var(--rule);stroke-width:1}
.chart .tick{fill:var(--muted);font-size:11px;font-family:var(--mono)}
.chart .lab{fill:var(--fg);font-size:12px}
.chart .val{fill:var(--fg);font-size:11px;font-family:var(--mono)}
figure{margin:18px 0}figure img{display:block;max-width:100%;border-radius:6px;border:1px solid var(--rule)}
figcaption{color:var(--muted);font-size:.88rem;margin-top:6px;max-width:72ch}
.cap{color:var(--muted);font-size:.88rem;max-width:72ch;margin:0 0 14px}
code{font-family:var(--mono);font-size:.86em;background:var(--chip);padding:1px 5px;border-radius:4px}
pre{font-family:var(--mono);font-size:.84rem;background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:12px 14px;overflow-x:auto}
details{margin:12px 0;border:1px solid var(--rule);border-radius:8px;background:var(--surface)}
summary{cursor:pointer;padding:10px 14px;font-weight:600}
details>div{padding:0 14px 14px;overflow-x:auto}
table{border-collapse:collapse;font-size:.84rem;font-variant-numeric:tabular-nums;min-width:600px}
th,td{padding:5px 8px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
td.ids{font-family:var(--mono);font-size:.72rem;white-space:normal;max-width:260px}
.legend{display:flex;flex-wrap:wrap;gap:10px 18px;font-size:.86rem;color:var(--muted);margin:4px 0 10px}
.legend span::before{content:"";display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px;background:var(--c)}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
.two>div{min-width:0}
a{color:var(--accent)}
.note{border-left:3px solid var(--accent);padding:4px 14px;margin:14px 0;color:var(--fg)}
@media (prefers-reduced-motion: reduce){*{scroll-behavior:auto}}
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono&display=swap">
"""
    leg = lambda labs: '<div class="legend">' + ''.join(f'<span style="--c:var({SLOT.get(l, "--s6")})">{esc(NAMES.get(l, l))}</span>' for l in labs if any(have(s, l) for s, _ in SCENES)) + '</div>'
    h = []
    h.append('<title>2026-11-12 Embedding-space warp</title>\n<meta name="claude:viewport-fit-sides" content="cover">' + css + '<main>')
    h.append('<p class="eyebrow">jeffrey/embed_warp · ntrl-demo metric fields · 2026-11-12</p>')
    h.append('<h1>Warping the embedding instead of the maze</h1>')
    h.append('<p class="lede">Every earlier warp moved the C-space coordinates before ntrl-demo\'s Fourier encoder, and a map of a 3-D space can only stretch a region by stretching it more along some direction than others. '
             'Here the coordinates stay put and a diffeomorphism acts on the <em>embedding</em>: ntrl-demo\'s 128 Fourier rows are split into base rows and octaves at 2, 4, 8 (16) times their frequency, and a coupling layer '
             'Ψ(u₀, u₁, …) = (u₀, μ₁(u₀)u₁, …) opens the octaves where travel time should be high and closes them in the open. The induced metric is an isotropic stretch λ(x) plus one small rank-one term; measured anisotropy at the walls is 1.00.</p>')
    h.append('<p class="lede"><b>What came out.</b> The isotropic multi-scale embedding is the part that works. Whitened octaves at ×2, ×4, ×8 with no gate at all (<code>oct8</code>) reach 79.3 % on the held-out 1,200 plus-maze queries over three training seeds against ntrl-demo\'s 68.4 %, level with the hand-constructed warp of the earlier study (80.1 %) without any warp, and 93–97 % on the disks against 84 %. Every gated variant also beats the control there (clearance gate ×2..×16 74.3 %, learned gate 72.8 %, clearance gate ×2..×8 73.6 %, three seeds each), but none beats the ungated ×2..×8 set: the gate earns its keep only when the octaves go too high (at ×16 the always-on control falls to 68.8 % and the gates hold 73–74 %). The learned gate rediscovers the geometry by itself from the travel-time loss. Every octave field has far sharper walls (runs that arrived through a wall 299 → 96–197 of 2,160 on the 240-set) and trades them for stalls at thin walls whose far-side value the loss never constrains. On the Gibson apartment (64 and 32 cells, three seeds at 32) nothing changes: its failures are far-field sinks in rooms.</p>')
    # verdict tiles
    tiles = []
    q8, qn = poolq.get(('plus', 'oct8')), poolq.get(('plus', 'ntrl'))
    if q8 and qn:
        tiles.append((f'{q8["pct"]:.1f} %', f'plus maze, held-out 1,200 queries, octaves x2,x4,x8 without a gate (ntrl-demo {qn["pct"]:.1f} %), {len(q8["seeds"])} seeds'))
    elif g8 and nt:
        tiles.append((f'{g8["pct"]:.1f} %', f'plus maze, clearance gate x2..x8 (ntrl-demo {nt["pct"]:.1f} %), {len(g8["seeds"])} seed{"s" if len(g8["seeds"]) > 1 else ""}'))
    if have('disks', 'learn') and have('disks', 'ntrl'):
        tiles.append((f'{D("learn")} %', f'disks, learned gate (ntrl-demo {D("ntrl")} %)'))
    if have('plus', 'geo') and have('plus', 'ntrl'):
        tiles.append((f'{pool[("plus", "ntrl")]["arrived_wall"]} → {pool[("plus", "geo")]["arrived_wall"]}', 'plus-maze runs that arrived through a wall, ntrl-demo → clearance gate'))
    if have('gibson', 'geo') and have('gibson', 'ntrl'):
        tiles.append((f'{G("geo")} %', f'Gibson apartment at 64 cells, clearance gate (ntrl-demo {G("ntrl")} %)'))
    h.append('<div class="verdict">' + ''.join(f'<div class="tile"><div class="n">{esc(a)}</div><div class="l">{esc(b)}</div></div>' for a, b in tiles) + '</div>')
    h.append('<p class="cap">Success = MPPI plan (ntrl-demo\'s planner, planning seeds 0/1/2) that arrives within tolerance and keeps the exact robot clearance non-negative along the whole path. '
             'Per-seed values are shown as dots on the bars; one training of the same recipe moves by about 3–5 points, so single-seed differences below that are noise.</p>')

    # --- the map
    h.append('<h2>1. The map</h2>')
    h.append('<p>ntrl-demo\'s travel time is T(x₀, x₁) = metric(enc(γ(x₀)), enc(γ(x₁))) with γ(x) = [sin 2πBx, cos 2πBx] ∈ R²⁵⁶. The encoder is Lipschitz in γ (row-normalized weights, sine activations), so the rate at which T can change along a direction of the C-space is set by the arc length the embedding has there: the Gram matrix JᵀJ of J = dγ/dx, which for random rows is (2π)²·128·σ²·I, the same in every direction and everywhere.</p>')
    h.append('<p>The new embedding keeps the architecture (656,900 parameters) and reassigns rows: 128 − K·m base rows at ntrl-demo\'s frequency, K octaves of m rows (whitened, so each octave\'s Gram is exactly a multiple of I) at scales 2, 4, 8 (16). The map of the embedding space is a coupling layer, triangular with a positive diagonal, hence a diffeomorphism for any smooth gate μ:</p>')
    h.append('<pre>Γ(x) = (γ₀(x), γ₁(x), …, γ_K(x))            Ψ(u₀, u₁, …, u_K) = (u₀, μ₁(u₀)·u₁, …, μ_K(u₀)·u_K),   μ_k ∈ [μ_min, 1]\n\n'
             'JᵀJ = g₀ + Σ_k μ_k² g_k + Σ_k m |∇μ_k|² n nᵀ        g_k = (2π)² B_kᵀB_k = c_k·I  (whitened);  the cross terms J_kᵀγ_k vanish exactly\n\n'
             'stretch λ(x) = sqrt(1 + Σ_k μ_k(x)² m σ_k² / (128 σ₀²))   isotropic;  the rank-one term along ∇μ is small when the gate changes over more than an octave wavelength</pre>')
    h.append('<p>Where the gate is open (near obstacles) the network sees λ times the arc length per unit of C-space in every direction; where it is closed the octaves collapse to μ_min and the network sees ntrl-demo\'s own low frequencies, so nothing aliases in the open, which is what sank the uniform magnifications (B × s) and the compressing warps of the earlier studies.</p>')
    iso_rows = []
    for s, _ in SCENES:
        for l in ('ntrl', 'oct', 'geo', 'geo8', 'learn'):
            p = pool.get((s, l))
            if p and p.get('iso_band'):
                iso_rows.append(f'<tr><td>{esc(s)}</td><td>{esc(NAMES.get(l, l))}</td><td>{p["iso_band"]["stretch"]:.2f}</td><td>{p["iso_band"]["aniso"]:.3f}</td><td>{(p["iso_open"] or {}).get("stretch", float("nan")):.2f}</td><td>{(p["iso_open"] or {}).get("aniso", float("nan")):.3f}</td></tr>')
    h.append('<details><summary>Measured metric the network sees (central differences of the full input mapping, 4,096 poses each)</summary><div><p class="cap">Stretch = sqrt(trace JᵀJ / trace J₀ᵀJ₀) relative to ntrl-demo\'s features; anisotropy = sqrt(λ_max / λ_min) of JᵀJ. Band = held-out near-wall poses, open = clearance ≥ 2 cells. For comparison the constructed C-space warp of the gap study had anisotropy 6 in the band arms.</p>'
             '<table><thead><tr><th>scene</th><th>method</th><th>band stretch</th><th>band aniso</th><th>open stretch</th><th>open aniso</th></tr></thead><tbody>' + ''.join(iso_rows) + '</tbody></table></div></details>')
    h.append('<h3>Gates</h3><ul>'
             '<li><b>clearance gate</b> (<code>geo</code>): from the collision oracle, open (1) up to 0.5 cells of clearance and inside obstacles, closed (μ_min = 0.02) from 1.5 cells on, smoothstep between. The band sampling\'s own quantity.</li>'
             '<li><b>band-density gate</b> (<code>kde</code>): from the training pairs\' near-wall x₀ alone, blurred density normalized to its median over occupied cells.</li>'
             '<li><b>diffusion gate</b> (<code>diff</code>): the obstacle boundary held at 1, obstacles insulating, heat spread through free space by masked Jacobi sweeps (the physics analogue the request asked for).</li>'
             '<li><b>curvature gate</b> (<code>curv</code>): Hessian norm of a trained ntrl-demo field at every grid node, median over 8 goals, normalized; trained separately, frozen.</li>'
             '<li><b>learned gate</b> (<code>learn</code>): μ = sigmoid(MLP(u₀)) on ntrl-demo\'s own features, trained by the travel-time loss with the network (own AdamW group, no budget unless noted).</li>'
             '<li><b>always on</b> (<code>oct</code>): μ ≡ 1, the uniform multi-scale control; <b>ntrl-demo</b>: no octaves (bit for bit the package\'s training, checked against the registered identity run).</li></ul>')
    h.append(fig('gates_plus.png', 'Gate μ at heading 0 over the plus maze for each gate kind (seed 87358; obstacles black; the title gives that run\'s success). The learned gate, started at 0.5 everywhere and trained by nothing but the travel-time loss, opens over the quadrant mazes and shuts over the open cross by itself.'))

    # --- results
    h.append('<h2>2. Results by scene</h2>')
    h.append(fig('scenes.png', 'The three scenes as the rectangle robot (0.5 × 0.2 cells) sees them at heading 0: the 22 × 22 plus maze with its 6-cell open cross, a 22-cell box with 40 disks (radii 0.5–1.4 cells, gaps ≥ 0.7 cells), and the P-NTFields Gibson apartment at 64 cells across (its scan walls a quarter of a cell thick).'))
    qlabs = [l for l in ('ntrl', 'oct', 'oct8', 'geo', 'geo0', 'geob', 'geo8', 'learn', 'learn_in') if ('plus', l) in poolq]
    for s, title in SCENES:
        if not any(have(s, l) for l in main_labels):
            continue
        h.append(f'<h3>{esc(title)}</h3>')
        h.append(bar_chart(method_bars(pool, s, main_labels), f'{s}: success by method'))
        series, bins = bin_series(pool, s, [l for l in ('ntrl', 'oct8', 'geo', 'geo8', 'learn', 'learn_in') if have(s, l)])
        if series:
            h.append(line_chart(series, bins, f'{s}: success by true travel distance'))
        h.append(stacked_fail(fail_items(pool, s, [l for l in main_labels if have(s, l)]), f'{s}: how runs end'))
        if s == 'plus' and qlabs:
            h.append('<h3>Plus maze, held-out 1,200-query set</h3>')
            h.append('<p>The 240 prepared queries are the selection set; the 1,200-query set of the cross-warp study (same maze, never used for any choice here) is the held-out test: 3,600 planner runs per field, per-seed dots.</p>')
            h.append(bar_chart([(NAMES.get(l, l), poolq[('plus', l)]['pct'], list(poolq[('plus', l)]['per_seed'].values()), SLOT.get(l, '--s6')) for l in qlabs], 'plus maze, 1,200-query set: success by method'))
            bins_q = [b for b in BINS if any(b in poolq[('plus', l)]['by_bin'] for l in qlabs)]
            h.append(line_chart([(NAMES.get(l, l), [100 * poolq[('plus', l)]['by_bin'][b][0] / poolq[('plus', l)]['by_bin'][b][1] if b in poolq[('plus', l)]['by_bin'] else None for b in bins_q], SLOT.get(l, '--s6')) for l in qlabs if l in ('ntrl', 'oct8', 'geo8', 'geo0', 'learn')], bins_q, 'plus maze, 1,200-query set: success by true travel distance'))
            h.append(stacked_fail([(NAMES.get(l, l), poolq[('plus', l)]['arrived_wall'], poolq[('plus', l)]['not_arrived'], poolq[('plus', l)]['success'], poolq[('plus', l)]['runs'], SLOT.get(l, '--s6')) for l in qlabs], 'plus maze, 1,200-query set: how runs end'))
    h.append('<p>Two scene-dependent readings. On the <b>plus maze</b> the ×2..×16 octaves always on lose a few points to ntrl-demo (the open cross aliases) and the gates recover them; with the top octave at ×8 the always-on octaves are the best field of the study (held-out 79.3 % over three seeds, section below) and the first to solve the long maze-to-maze queries (32–64 cells, "neither end in the +"); the gated ×2..×16 fields follow at 73–74 % held-out, the gated ×2..×8 at 73.6 %. On the <b>disks</b> there is no open space wider than two cells, so the gate hardly matters and the octaves alone lift success by about nine points with zero wall contacts. On the <b>Gibson apartment</b> every method fails the same way at 64 cells (section 3), and at 32 cells, over three seeds, ntrl-demo 81.9 %, learned gate 80.3 %, clearance gate ×2..×8 78.9 %: no gain, the fixed gate a little worse.</p>')
    if have('plus', 'ntrl'):
        h.append('<details><summary>Plus maze: success by where the query ends lie</summary><div>' + where_table(pool) + '</div></details>')

    # --- mechanism
    h.append('<h2>3. Mechanism: what the stretch fixes and what it cannot</h2>')
    if nt and have('plus', 'geo'):
        pg = pool[('plus', 'geo')]
        h.append(f'<p>Every local measure of the field improves with the gated octaves: speed fit (log-correlation with the label on held-out pairs) {nt["speed_corr"]:.2f} → {pg["speed_corr"]:.2f}, near-wall eikonal slope S|∇T| {nt["slope"]:.2f} → {pg["slope"]:.2f} (target 1), direction into the wall {nt["cos"]:.2f} → {pg["cos"]:.2f}, share of near-wall poses whose descent points into the wall {nt["attract"]:.3f} → {pg["attract"]:.3f}, local minima per true path {nt["bumps"]:.2f} → {pg["bumps"]:.2f}. '
                 f'Runs that arrived through a wall fall from {nt["arrived_wall"]} to {pg["arrived_wall"]}. Yet the ×2..×16 field plans no better than ntrl-demo in total, because the same queries fail in a different way: they stall at the wall.</p>')
    h.append('<p>Where the stalls end: pressed against a thin wall one to three cells from the goal, with the field\'s descent direction pointing at the goal <em>through</em> the wall and T at 0.6 of the truth (a quarter of the stall points of the clearance-gated field are inside the wall). There is no barrier along the true path from the stall point (T decreases immediately), so these are not local minima; the field simply says the goal is right there. ntrl-demo says the same and walks through (its 125 wall contacts); the gated field has the barrier and stops.</p>')
    h.append(fig('stalls_plus_geo.png', 'Clearance-gated field (×2..×16), plus maze, eight of its stalled plans (planning seed 0): T(·, goal) at the stall heading around the stall point (red dot, last 80 MPPI positions in red), the goal as a star. The descent direction at the dot points at the goal; the wall between them carries a steep barrier but the value behind it is wrong.'))
    h.append('<p>Per query, the picture is sharp: the 44 queries whose true path is 1.3–1.6 times the straight line (a wall between start and goal) are solved by no field; ntrl-demo fails 66 of their 132 runs through the wall and 45 by stalling, the gated fields 6 through the wall and 105–111 by stalling. The solved sets overlap on ~153 of ~165 queries. The remaining error is the <em>value</em> of T behind a thin wall, which no local term of ntrl-demo\'s loss constrains: the sampler never places a pair endpoint inside an obstacle, so the eikonal never integrates the floor speed across the wall.</p>')
    h.append('<details><summary>Success by detour factor (plus maze, seed 87358, 3 planning seeds)</summary><div><p class="cap">Detour factor = true travel time / straight-line time. Counts are runs out of 3 × queries in the bin.</p><pre>'
             'detour      queries   ntrl-demo         clearance gate    clearance gate mu_min 0   wide band        learned\n'
             '1-1.1        147      93 % (25 w, 6 s)   92 % (12 w, 25 s)  92 % (0 w, 36 s)         91 % (6 w, 36 s)  92 % (1 w, 36 s)\n'
             '1.1-1.3       28      64 % (3 w, 27 s)   62 % (6 w, 26 s)   64 % (7 w, 23 s)         61 % (9 w, 24 s)  60 % (4 w, 30 s)\n'
             '1.3-1.6       44      16 % (66 w, 45 s)  11 % (6 w, 111 s)  16 % (6 w, 105 s)        14 % (6 w, 107 s) 12 % (74 w, 42 s)\n'
             '1.6-2.2       19      35 % (28 w, 9 s)   32 % (24 w, 15 s)  37 % (6 w, 30 s)         63 % (6 w, 15 s)  47 % (7 w, 23 s)\n'
             '(w = arrived through a wall, s = never arrived)</pre></div></details>')
    h.append('<h3>Three loss-side tests of the leak</h3>')
    tests = []
    for l, txt in (('ntrl_in', 'ntrl-demo + 20 % pairs with x₁ inside an obstacle'), ('geo_in', 'clearance gate + inside pairs'), ('learn_in', 'learned gate + inside pairs'), ('oct_in', 'always on + inside pairs'),
                   ('ntrl_td', 'ntrl-demo, TD step 0.05 cells'), ('geo_td', 'clearance gate, TD step 0.05 cells'), ('geo0_td', 'clearance gate μ_min 0, TD step 0.05 cells')):
        p = pool.get(('plus', l))
        if p:
            tests.append(f'<tr><td>{esc(txt)}</td><td>{p["pct"]:.1f}</td><td>{p["arrived_wall"]}</td><td>{p["not_arrived"]}</td><td>{p["speed_corr"]:.2f}</td><td>{p["attract"]:.3f}</td><td>{p["bumps"]:.2f}</td></tr>')
    h.append('<p><b>Inside-obstacle pairs</b> (a data variant: 20 % of pairs with x₁ inside a wall, label = floor speed, normal from the oracle): the demanded gradient inside a wall is 83 times the free one; ntrl-demo\'s features cannot carry it and collapse (wall contacts double, attraction into walls triples), the ×2..×16 clearance gate is unchanged, and the learned gate gains three points. '
             '<b>A TD step shorter than the wall</b> (0.05 cells instead of ntrl-demo\'s 0.66): on ntrl-demo it halves the wall contacts without changing success; on the gated fields it removes the smoothing the high octaves relied on and the field ripples (3.6–4 local minima per true path, success 57 %). The long TD step is a second source of walk-throughs and, at the same time, the regularizer of the octaves.</p>')
    h.append('<details><summary>Loss-side tests, plus maze, seed 87358</summary><div><table><thead><tr><th>variant</th><th>%</th><th>through wall</th><th>never arrived</th><th>speed corr</th><th>attract</th><th>bumps/path</th></tr></thead><tbody>' + ''.join(tests) + '</tbody></table></div></details>')
    h.append('<h3>How much stretch</h3>')
    if stretch_pts:
        h.append(scatter(stretch_pts, 'plus maze: success vs band stretch', 'stretch at the walls relative to ntrl-demo (sqrt of the trace ratio)'))
    h.append('<p>With the same clearance gate, success on the plus maze falls as the top octave rises: ×2..×8 (stretch 3.8) is best, ×2..×16 (7.4) and 24 rows per octave (9.0) lose it again, and the field\'s local minima per true path grow with the stretch. Without any gate the same sweep reads (seed 87358) ×2,×4 69.6 %, ×3,×6 74.6, ×2,×4,×8 68.9 (but 82.8 and 84.2 on seeds 1 and 2), ×2,×4,×8,×12 76.7, ×2..×16 66.0; 24 or 32 rows per octave and unwhitened rows stay within that spread. The wall barrier is complete at ×8; the ×16 octave adds ripple that the planner falls into at long range, and that is the one setting where the gate pays.</p>')
    h.append('<h3>Gibson: a different failure</h3>')
    h.append(fig('stalls_gibson_ntrl.png', 'ntrl-demo on the Gibson apartment at 64 cells, eight stalled plans: the stall points sit in bowl-shaped local minima in the middle of rooms where the field reads about one cell of travel to a goal 4–25 cells away (T/T* 0.56 at the stall points, |∇T| 0.1). These are the metric network\'s far-field sinks (the "wormholes" of the gap study), not a wall-resolution problem.'))
    h.append('<p>On the apartment every method stalls in the same places (ntrl-demo 237, always on 243, learned 250, clearance gate 231 runs never arrive) and the gate only takes the wall contacts from 14 to 3–6. The scene is 64 cells across with goals up to 128 cells away; what fails is the long-range value of T, which the embedding\'s local resolution does not touch.</p>')

    # --- what did not help
    h.append('<h2>4. What did not help, and open items</h2><ul>')
    for l, txt in (('kde', 'the band-density gate: it leaves parts of the corridors half open (0.81 inside the band) and loses ten points'), ('diff', 'the diffusion gate: its slow decay leaves the open cross at μ 0.13–0.17 with spatial structure, which ripples the band field (0.89 minima per path)'),
                   ('learnb', 'a budget on the learned gate'), ('geot', 'a tighter band'), ('geo24', 'more rows per octave'), ('curv', 'the curvature gate from a trained field'), ('learnx', 'the learned gate on x instead of u₀')):
        p = pool.get(('plus', l))
        if p:
            h.append(f'<li>{esc(txt)}: {p["pct"]:.1f} % (plus maze, seed 87358).</li>')
    h.append('<li>Open: the far-field value behind thin walls (plus) and the room sinks (Gibson) are global errors that need a non-local term (the midpoint / segment-DP losses of the earlier studies) on top of this embedding; the ×2..×8 recipe is one point in a sweep that should be run per scene scale.</li></ul>')

    # --- tables
    h.append('<h2>5. Every run</h2>')
    h.append('<details><summary>All registered runs, pooled over training seeds</summary><div>' + full_table(pooled) + '</div></details>')
    h.append('<p class="cap">Code: <code>jeffrey/embed_warp/</code> (features.py the map and gates, gates.py the fixed gate tables, trainer.py ntrl-demo\'s loop, scene.py the disks and Gibson scenes with maze_scale\'s interface, pipeline.py / prepare.py / evalq.py the stages, run.py the registry). Every run is registered; results in <code>results/registered_&lt;id&gt;/</code>. Modified-environment results (host venv, RTX 3090 / 5070).</p>')
    h.append('</main>')
    (REPORT / 'embed_warp.html').write_text('\n'.join(h))
    print(REPORT / 'embed_warp.html', len('\n'.join(h)) // 1024, 'KB')


if __name__ == '__main__':
    main()
