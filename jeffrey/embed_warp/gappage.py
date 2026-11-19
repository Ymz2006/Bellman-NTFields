"""Gap mazes (10 x 10 base, quadrants pushed apart by x1..x12): report/gap_runs.json, report/gap_tables.md and the findings page
report/embed_gap.html, from the registered embed-warp runs on the five maze10 gap prepares.

    python jeffrey/embed_warp/gappage.py        # then gapfigs.py, then publish report/embed_gap.html with report/gap_*.png

Two octave families: fixed x2, x4, x8 (the plus maze's best) and per cell, (x2, x4, x8) times E / 22.1 so that a quadrant cell sees the
plus maze's spectrum (equal to the fixed family at x1 and x2, where the factor is <= 1 and is clamped to 1). Each with the clearance gate,
the diffusion gate, the learned gate and the always-on control; ntrl-demo on the same seeds.
"""
import html
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPORT = HERE / 'report'
sys.path.insert(0, str(HERE))
import report as RP  # noqa: E402
import page as PG    # noqa: E402

GAPS = [1, 2, 4, 8, 12]
PREPARES = {1: '20261105T070523Z-5877176f', 2: '20261105T070535Z-01c9f536', 4: '20261105T070549Z-25e73d68', 8: '20261105T070611Z-cf3b0b02', 12: '20261105T070402Z-98f3cab7'}
ZP = {4: '20261105T070709Z-0edca351', 12: '20261105T070657Z-4ef430ba'}          # same mazes and queries, pairs drawn on the warped maze (cross_warp --z-draw)
CATS = ['same quadrant', 'adjacent quadrants', 'diagonal quadrants', 'quadrant - open', 'open - open']
FIXED = ['oct8', 'geo8', 'diff8', 'learn8']
CELL = ['octc', 'geoc', 'diffc', 'learnc']
ALIAS = dict(octc='oct8', geoc='geo8', diffc='diff8', learnc='learn8')        # per-cell = fixed where E / 22.1 <= 1 (x1, x2)
NAMES = dict(ntrl='ntrl-demo', oct8='always on, x2..x8', geo8='clearance gate, x2..x8', diff8='diffusion gate, x2..x8', learn8='learned gate, x2..x8',
             octc='always on, per cell', geoc='clearance gate, per cell', diffc='diffusion gate, per cell', learnc='learned gate, per cell')
SLOT = dict(ntrl='--s1', oct8='--s2', octc='--s2', geo8='--s3', geoc='--s3', diff8='--s6', diffc='--s6', learn8='--s5', learnc='--s5')
# references from earlier studies on the same five mazes and queries (training seed 87358 unless pooled; jeffrey/cross_warp report/zdraw10.md,
# jeffrey/line_warp README): constructed C-space warp trained on the real maze with ntrl-demo's own pairs (cross_warp 'pm', planned on the maze)
REF_PM = {1: 86.3, 2: 68.6, 4: 28.6, 8: 16.7, 12: 17.9}
REF_ZDRAW = {1: 92.3, 2: 86.6, 4: 84.8, 8: 81.2, 12: 85.1}       # constructed warp, trained AND planned on the warped maze, pairs drawn there
REF_LINE = {1: 89.7, 2: 91.0, 4: 93.6, 8: 89.9, 12: 93.3}        # learned line warp, same warped-maze setup


def esc(s):
    return html.escape(str(s))


def load():
    rows, zp = [], []
    for r in RP.load_runs():
        if not r['scene'].startswith('gap'):
            continue
        pid = Path(r['prepared']).name.replace('registered_', '')
        r['gap'] = int(r['scene'][3:])
        if pid in PREPARES.values():
            rows.append(r)
        elif pid in ZP.values():
            zp.append(r)
    return rows, zp


def pool(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r['gap'], r['label'])].append(r)
    for k in GAPS:                                      # per-cell family at x1, x2 = the fixed runs
        for c, f in ALIAS.items():
            if (k, c) not in g and (k, f) in g and k <= 2:
                g[(k, c)] = g[(k, f)]
    out = {}
    for (k, lab), rs in g.items():
        seen, rs2 = set(), []
        for r in rs:
            if r['seed'] not in seen:
                seen.add(r['seed'])
                rs2.append(r)
        s, n = sum(r['success'] for r in rs2), sum(r['runs'] for r in rs2)
        cats = {c: [sum(r['by_category'][c]['success'] for r in rs2 if r.get('by_category') and c in r['by_category']),
                    sum(r['by_category'][c]['runs'] for r in rs2 if r.get('by_category') and c in r['by_category'])] for c in CATS}
        out[(k, lab)] = dict(gap=k, label=lab, seeds=sorted(r['seed'] for r in rs2), success=s, runs=n, pct=100.0 * s / n,
                             per_seed={str(r['seed']): round(r['pct'], 1) for r in sorted(rs2, key=lambda r: r['seed'])}, cats=cats,
                             arrived_wall=sum(r['failures']['arrived_wall'] for r in rs2), not_arrived=sum(r['failures']['not_arrived'] for r in rs2),
                             speed_corr=float(np.mean([r['speed_corr'] for r in rs2])), slope=float(np.mean([r['near_wall']['slope'] for r in rs2])),
                             attract=float(np.mean([r['near_wall']['attract'] for r in rs2])), bumps=float(np.mean([r['bumps']['bumps_per_path'] or 0 for r in rs2])),
                             t_ratio=float(np.mean([r['t_ratio'] for r in rs2])), octaves=rs2[0]['octaves'],
                             iso_band=rs2[0]['iso'].get('band'), iso_open=rs2[0]['iso'].get('open'),
                             gate_mu=rs2[0].get('gate_mu'), gate_by_clearance=rs2[0].get('gate_by_clearance'), ids=[r['id'] for r in rs2])
    return out


def cpct(p, c):
    v = p['cats'][c]
    return 100.0 * v[0] / v[1] if v[1] else None


def line_gap(series, title, refs=(), width=640):
    """series: [(label, {gap: pct}, css)]; refs: [(label, {gap: pct})] drawn dashed in muted ink. x = gap (categorical)."""
    L, R, T, B = 44, 168, 18, 42
    W, H = width, 280
    X = lambda i: L + (W - L - R) * i / (len(GAPS) - 1)
    Y = lambda v: T + (H - T - B) * (1 - v / 100)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}" class="chart"><title>{esc(title)}</title>']
    for t in (0, 25, 50, 75, 100):
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/><text x="{L - 6}" y="{Y(t) + 4:.1f}" text-anchor="end" class="tick">{t}</text>')
    for i, k in enumerate(GAPS):
        out.append(f'<text x="{X(i):.1f}" y="{H - B + 16}" text-anchor="middle" class="tick">×{k}</text>')
    out.append(f'<text x="{(L + W - R) / 2:.1f}" y="{H - 6}" text-anchor="middle" class="tick">gap (band 6k cells; maze 16 → 82 cells)</text>')
    labels = []
    for lab, vals in refs:
        pts = [(X(i), Y(vals[k])) for i, k in enumerate(GAPS) if vals.get(k) is not None]
        d = 'M' + ' L'.join(f'{x:.1f},{y:.1f}' for x, y in pts)
        out.append(f'<path d="{d}" fill="none" stroke="var(--muted)" stroke-width="1.5" stroke-dasharray="4 4"/>')
        labels.append((pts[-1][1], lab, 'var(--muted)'))
    for lab, vals, css in series:
        pts = [(X(i), Y(vals[k]), vals[k]) for i, k in enumerate(GAPS) if vals.get(k) is not None]
        if not pts:
            continue
        d = 'M' + ' L'.join(f'{x:.1f},{y:.1f}' for x, y, _ in pts)
        out.append(f'<path d="{d}" fill="none" stroke="var({css})" stroke-width="2" stroke-linejoin="round"/>')
        for x, y, v in pts:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="var({css})" stroke="var(--surface)" stroke-width="1.5"><title>{esc(lab)}: {v:.1f} %</title></circle>')
        labels.append((pts[-1][1], lab, f'var({css})'))
    labels.sort()
    last = -1e9
    for y, lab, col in labels:                          # end labels, nudged apart
        y = max(y, last + 13)
        last = y
        out.append(f'<text x="{W - R + 8}" y="{y + 4:.1f}" class="lab" fill="{col}" style="fill:{col}">{esc(lab)}</text>')
    out.append('</svg>')
    return '\n'.join(out)


def cat_bars(P, k, labels, title):
    """Grouped horizontal bars: per query kind, one bar per method (gap k)."""
    rowh, gh, L, R, T, B = 13, 10, 150, 50, 10, 24
    n = len(labels)
    H = T + B + len(CATS) * (n * rowh + gh)
    W = 640
    X = lambda v: L + (W - L - R) * v / 100
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}" class="chart"><title>{esc(title)}</title>']
    for t in range(0, 101, 25):
        out.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{T}" y2="{H - B}" class="grid"/><text x="{X(t):.1f}" y="{H - B + 14}" text-anchor="middle" class="tick">{t}</text>')
    y = T
    for c in CATS:
        out.append(f'<text x="{L - 8}" y="{y + n * rowh / 2 + 4:.1f}" text-anchor="end" class="lab">{esc(c)}</text>')
        for lab in labels:
            p = P.get((k, lab))
            v = cpct(p, c) if p else None
            if v is not None:
                out.append(f'<rect x="{L}" y="{y + 1}" width="{max(X(v) - L, 0.5):.1f}" height="{rowh - 2}" rx="2" fill="var({SLOT[lab]})"><title>{esc(NAMES[lab])}, {esc(c)}: {v:.0f} %</title></rect>')
                out.append(f'<text x="{X(v) + 4:.1f}" y="{y + rowh - 3}" class="val">{v:.0f}</text>')
            y += rowh
        y += gh
    out.append('</svg>')
    return '\n'.join(out)


def tables_md(P):
    lines = ['# Gap mazes, 10 x 10 base: embedding-warp gates (generated by gappage.py)\n',
             'Success % of 900 planner runs per training seed (300 queries x planning seeds 0/1/2); pooled over training seeds; per-seed in brackets.\n',
             '| method | ' + ' | '.join(f'x{k}' for k in GAPS) + ' |', '|---|' + '---:|' * len(GAPS)]
    for lab in ['ntrl'] + FIXED + CELL:
        cells = []
        for k in GAPS:
            p = P.get((k, lab))
            cells.append(f'{p["pct"]:.1f} ({" / ".join(f"{v:.1f}" for v in p["per_seed"].values())})' if p else '')
        lines.append(f'| {NAMES[lab]} (`{lab}`) | ' + ' | '.join(cells) + ' |')
    for k in GAPS:
        lines += [f'\n## x{k}: by query kind (%), failures\n', '| method | seeds | ' + ' | '.join(CATS) + ' | through wall | never arrived | speed corr | wall slope | attract | bumps/path | T/T* | octaves | run ids |',
                  '|---|---|' + '---:|' * (len(CATS) + 7) + '---|---|']
        for lab in ['ntrl'] + FIXED + CELL:
            p = P.get((k, lab))
            if not p:
                continue
            lines.append(f'| {NAMES[lab]} | {",".join(map(str, p["seeds"]))} | ' + ' | '.join(f'{cpct(p, c):.0f}' if cpct(p, c) is not None else '' for c in CATS) +
                         f' | {p["arrived_wall"]} | {p["not_arrived"]} | {p["speed_corr"]:.2f} | {p["slope"]:.2f} | {p["attract"]:.3f} | {p["bumps"]:.2f} | {p["t_ratio"]:.2f} | {",".join(f"{o:g}" for o in p["octaves"])} | {" ".join(p["ids"])} |')
    return '\n'.join(lines) + '\n'


ZPL = ['ntrl', 'octc', 'geoc', 'diffc', 'learnc']


def zp_rows(P, Z):
    out = []
    for k in sorted(ZP):
        for l in ZPL:
            a, b = P.get((k, l)), Z.get((k, l + '_zp'))
            if a and b:
                out.append((k, l, a, b))
    return out


def zp_md(P, Z):
    lines = ['\n## Pairs drawn on the warped maze (training seed 87358; own pairs at the same seed)\n', '| gap | method | own pairs % | warped-maze pairs % | same quadrant | adjacent | diagonal | quadrant - open | through wall | never arrived | run id |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for k, l, a, b in zp_rows(P, Z):
        own = a['per_seed'].get('87358')
        lines.append(f'| x{k} | {NAMES[l]} | {own} | {b["pct"]:.1f} | ' + ' | '.join(f'{cpct(b, c):.0f}' for c in CATS[:4]) + f' | {b["arrived_wall"]} | {b["not_arrived"]} | {" ".join(b["ids"])} |')
    return '\n'.join(lines) + '\n'


def main():
    rows, zp = load()
    P = pool(rows)
    Z = pool(zp)
    REPORT.mkdir(exist_ok=True)
    json.dump(dict(rows=rows, pooled=[dict(v) for v in P.values()], zp_rows=zp, zp_pooled=[dict(v) for v in Z.values()]), open(REPORT / 'gap_runs.json', 'w'), indent=1, default=float)
    md = tables_md(P) + zp_md(P, Z)
    (REPORT / 'gap_tables.md').write_text(md)
    print(md)
    print(len(rows), 'runs')
    page(P, Z)


def series(P, labels):
    return [(NAMES[l], {k: P[(k, l)]['pct'] for k in GAPS if (k, l) in P}, SLOT[l]) for l in labels]


def page(P, Z):
    from gaptext import TEXT       # prose, written from the numbers after the runs (gaptext.py)
    css = open(HERE / 'page.py').read().split('css = """', 1)[1].split('"""', 1)[0]
    h = ['<title>2026-11-13 Embedding warp on gap mazes</title>\n<meta name="claude:viewport-fit-sides" content="cover">' + css + '<main>']
    h.append('<p class="eyebrow">jeffrey/embed_warp · gap mazes on the 10 × 10 base · 2026-11-12/13</p>')
    h.append('<h1>Embedding-space gates on the widening gap</h1>')
    for t in TEXT['lede']:
        h.append(f'<p class="lede">{t}</p>')
    tiles = TEXT['tiles'](P)
    h.append('<div class="verdict">' + ''.join(f'<div class="tile"><div class="n">{esc(a)}</div><div class="l">{esc(b)}</div></div>' for a, b in tiles) + '</div>')
    h.append(PG.fig('gap_mazes10.png', 'The five mazes: the 10 × 10 maze cut into four 5 × 5 quadrants (every wall kept) and pushed apart by an open band of 6k cells, k = 1, 2, 4, 8, 12 (16 to 82 cells a side); one true path each. Same 300 queries per maze as the cross-warp study.'))
    h.append('<h2>1. Success as the gap grows</h2>')
    for t in TEXT['results']:
        h.append(f'<p>{t}</p>')
    refs = [('C-space warp', REF_PM)]
    h.append('<h3>Per-cell octaves: (×2, ×4, ×8) × E / 22.1</h3>')
    h.append(line_gap(series(P, ['ntrl'] + CELL), 'success by gap, per-cell octaves', refs))
    h.append('<h3>Fixed octaves: ×2, ×4, ×8 at every gap</h3>')
    h.append(line_gap(series(P, ['ntrl'] + FIXED), 'success by gap, fixed octaves', refs))
    h.append('<p class="cap">Dashed: the constructed C-space warp of the cross-warp study trained on the real maze with the same pairs (training seed 87358), the fair warp reference. ' + TEXT['refs_note'] + '</p>')
    h.append('<details><summary>Success per gap, every method, per training seed</summary><div><table><thead><tr><th>method</th>' + ''.join(f'<th>×{k}</th>' for k in GAPS) + '</tr></thead><tbody>' +
             ''.join(f'<tr><td>{esc(NAMES[l])}</td>' + ''.join(f'<td>{P[(k, l)]["pct"]:.1f}<br><small>{" / ".join(f"{v:.0f}" for v in P[(k, l)]["per_seed"].values())}</small></td>' if (k, l) in P else '<td>–</td>' for k in GAPS) + '</tr>' for l in ['ntrl'] + FIXED + CELL) +
             '</tbody></table></div></details>')
    h.append('<h2>2. Which queries</h2>')
    for t in TEXT['cats']:
        h.append(f'<p>{t}</p>')
    for k in TEXT['cat_gaps']:
        h.append(f'<h3>×{k}</h3>')
        h.append(cat_bars(P, k, ['ntrl'] + TEXT['cat_labels'], f'gap x{k}: success by query kind'))
    h.append('<div class="legend">' + ''.join(f'<span style="--c:var({SLOT[l]})">{esc(NAMES[l])}</span>' for l in ['ntrl'] + TEXT['cat_labels']) + '</div>')
    h.append('<h3>How runs end</h3>')
    for k in TEXT['fail_gaps']:
        items = [(f'×{k} ' + NAMES[l], P[(k, l)]['arrived_wall'], P[(k, l)]['not_arrived'], P[(k, l)]['success'], P[(k, l)]['runs'], SLOT[l]) for l in ['ntrl'] + TEXT['cat_labels'] if (k, l) in P]
        h.append(PG.stacked_fail(items, f'gap x{k}: how runs end'))
    h.append('<h2>3. The gates</h2>')
    for t in TEXT['gates']:
        h.append(f'<p>{t}</p>')
    h.append(PG.fig('gap_gates.png', TEXT['gates_caption']))
    gt = []
    for k in GAPS:
        for l in ('geoc', 'diffc', 'learnc', 'geo8', 'diff8', 'learn8'):
            p = P.get((k, l))
            if p and p.get('gate_mu'):
                gm = p['gate_mu']
                gt.append(f'<tr><td>×{k}</td><td>{esc(NAMES[l])}</td><td>{gm["band"]["mean"]:.2f}</td><td>{gm["open"]["mean"]:.2f}</td><td>{gm["uniform"]["mean"]:.2f}</td>'
                          f'<td>{(p["iso_band"] or {}).get("stretch", float("nan")):.1f}</td><td>{(p["iso_open"] or {}).get("stretch", float("nan")):.2f}</td><td>{(p["iso_band"] or {}).get("aniso", float("nan")):.3f}</td></tr>')
    h.append('<details><summary>Gate values and the metric the network sees (training seed 87358)</summary><div><p class="cap">μ mean over near-wall poses (band), poses with clearance ≥ 2 cells (open) and uniform poses; stretch = sqrt(trace JᵀJ / trace J₀ᵀJ₀) against ntrl-demo\'s features; anisotropy = sqrt(λmax / λmin) at the band.</p>'
             '<table><thead><tr><th>gap</th><th>method</th><th>μ band</th><th>μ open</th><th>μ uniform</th><th>stretch band</th><th>stretch open</th><th>aniso band</th></tr></thead><tbody>' + ''.join(gt) + '</tbody></table></div></details>')
    h.append('<h2>4. Mechanism</h2>')
    for t in TEXT['mechanism']:
        h.append(f'<p>{t}</p>')
    h.append(PG.fig('gap_far.png', TEXT['far_caption']))
    for t in TEXT.get('mechanism2', []):
        h.append(f'<p>{t}</p>')
    zr = zp_rows(P, Z)
    if zr:
        h.append('<h3>Test: nine times more pairs inside the quadrants</h3>')
        for t in TEXT['zp']:
            h.append(f'<p>{t}</p>')
        items = []
        for k, l, a, b in zr:
            short = dict(ntrl='ntrl-demo', octc='always on', geoc='clearance', diffc='diffusion', learnc='learned')[l]
            items.append((f'×{k} {short}, own pairs', a['per_seed'].get('87358', a['pct']), [], SLOT[l]))
            items.append((f'×{k} {short}, warped pairs', b['pct'], [], SLOT[l]))
        h.append(PG.bar_chart(items, 'own pairs vs pairs drawn on the warped maze, training seed 87358'))
        h.append('<details><summary>Warped-maze pairs by query kind</summary><div><table><thead><tr><th>gap</th><th>method</th><th>own pairs %</th><th>warped-maze pairs %</th>' + ''.join(f'<th>{esc(c)}</th>' for c in CATS[:4]) + '<th>through wall</th><th>never arrived</th><th>run id</th></tr></thead><tbody>' +
                 ''.join(f'<tr><td>×{k}</td><td>{esc(NAMES[l])}</td><td>{a["per_seed"].get("87358")}</td><td>{b["pct"]:.1f}</td>' + ''.join(f'<td>{cpct(b, c):.0f}</td>' for c in CATS[:4]) + f'<td>{b["arrived_wall"]}</td><td>{b["not_arrived"]}</td><td class="ids">{" ".join(b["ids"])}</td></tr>' for k, l, a, b in zr) +
                 '</tbody></table></div></details>')
    h.append('<h2>5. Verdict</h2>')
    for t in TEXT['verdict']:
        h.append(f'<p>{t}</p>')
    full = ['<table><thead><tr><th>gap</th><th>method</th><th>seeds</th><th>%</th><th>per seed</th>' + ''.join(f'<th>{esc(c)}</th>' for c in CATS) +
            '<th>through wall</th><th>never arrived</th><th>speed corr</th><th>wall slope</th><th>attract</th><th>bumps/path</th><th>octaves</th><th>run ids</th></tr></thead><tbody>']
    for k in GAPS:
        for l in ['ntrl'] + FIXED + CELL:
            p = P.get((k, l))
            if not p or (k <= 2 and l in CELL):
                continue
            full.append(f'<tr><td>×{k}</td><td>{esc(NAMES[l])}</td><td>{len(p["seeds"])}</td><td>{p["pct"]:.1f}</td><td>{" / ".join(f"{v:.1f}" for v in p["per_seed"].values())}</td>' +
                        ''.join(f'<td>{cpct(p, c):.0f}</td>' if cpct(p, c) is not None else '<td></td>' for c in CATS) +
                        f'<td>{p["arrived_wall"]}</td><td>{p["not_arrived"]}</td><td>{p["speed_corr"]:.2f}</td><td>{p["slope"]:.2f}</td><td>{p["attract"]:.3f}</td><td>{p["bumps"]:.2f}</td>'
                        f'<td>{",".join(f"{o:g}" for o in p["octaves"])}</td><td class="ids">{" ".join(esc(i) for i in p["ids"])}</td></tr>')
    full.append('</tbody></table>')
    h.append('<details><summary>Every run, pooled over training seeds</summary><div>' + '\n'.join(full) + '</div></details>')
    h.append('<p class="cap">Code: <code>jeffrey/embed_warp/</code> (gappage.py, gapfigs.py, gaptext.py for this page; the map, gates and trainer as in the main study). Prepares: the cross-warp study\'s maze10 gap prepares '
             + ', '.join(f'×{k} <code>{v}</code>' for k, v in PREPARES.items()) + '. Every run is registered; modified-environment results (host venv, RTX 3090 / 5070). Main study page: <a href="https://claude.ai/artifact/FqBhMFaPPqFAYP9fCJJRXS">2026-11-12 Embedding-space warp</a>.</p>')
    h.append('</main>')
    (REPORT / 'embed_gap.html').write_text('\n'.join(h))
    print(REPORT / 'embed_gap.html', len('\n'.join(h)) // 1024, 'KB')


if __name__ == '__main__':
    main()
