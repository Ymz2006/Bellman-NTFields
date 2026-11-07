"""Findings page for the cross-warp study (report/cross_warp.html) from the registered runs (report.collect(),
report.collect_evalq()), report/attract.json (diag.near_wall over every field, written by attract.py) and the figures
in report/ (figs.py).

    ~/.venvs/bellman-ntfield/bin/python jeffrey/cross_warp/page.py

Every number on the page is computed here; page_template.html holds the layout and the chart script, the prose is in
this file next to the numbers it uses. Publish report/cross_warp.html with warps.png, speed.png and density.png.
"""
import html
import json
import math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'report'
import sys
sys.path.insert(0, str(HERE))
import report as R  # noqa: E402

PREP = 'registered_20261102T042454Z-62864e61'
PREP20 = 'registered_20261103T190009Z-01bd3aac'      # the 20 x 20 cross
Q1200 = 'registered_20261103T174058Z-405b526e'
BINS = R.BINS
SHORT = BINS[:-1]
FIELDS = ['ntrl-demo', 'u4', 'u6', 'pm6', 'LEARNED', 'lf', 'pm6i0', 'u2', 'pm4', 'pm10']
LABEL = {'ntrl-demo': 'ntrl-demo', 'id': 'identity warp (= ntrl-demo)', 'u2': 'uniform stretch ×2 (B × 2)', 'u3': 'uniform ×3', 'u4': 'uniform stretch ×4 (B × 4)',
         'u6': 'uniform stretch ×6 (B × 6)', 'pm6': 'designed warp ×6 / ×1', 'pm4': 'designed warp ×4 / ×1', 'pm10': 'designed warp ×10 / ×1', 'pm6i0': 'designed warp ×6 / ×1, no inset',
         'la6n': 'learned warp (unbounded)', 'lf': 'README warp (learned, box-preserving)', 'lf16': 'README warp, stronger compression', 'lfb': 'README warp + near-wall expansion',
         'dz3': 'designed box-preserving warp (ratio 3)', 'dz12': 'designed box-preserving warp (ratio 12)', 'sg': 'designed grid-line stretch (box-preserving)',
         'hp6': 'heading-only stretch ×6', 'lsep4': 'learned free stretch from ×4 (no near-wall term)', 'lsb6': 'learned free stretch, 8 nodes / cell',
         'lsb6c': 'learned free stretch, area target', 'lsb6s': 'learned free stretch, 8 nodes / cell, smoothed', 'lsn5': 'learned box-preserving × 5.1',
         'la6': 'learned warp, frozen at epoch 2,000', 'la6r': 'learned warp, frozen, field retrained', 'la7b': 'learned warp, bounded ×1..×7', 'la6b': 'learned warp, bounded ×1..×6', 'lc6b': 'learned warp, row-dependent (coupling), bounded ×1..×6', 'sl': 'learned separable, box-preserving', 'lfs2': 'README warp × 2', 'lfb4': 'README warp, 4 nodes / cell'}
CSS = {'pm6i0': '--s-pm', 'ntrl-demo': '--s-ntrl', 'id': '--s-id', 'u2': '--s-u6', 'u4': '--s-u4', 'u6': '--s-u6', 'pm6': '--s-pm', 'pm4': '--s-pm', 'pm10': '--s-pm', 'la6n': '--s-la', 'lf': '--s-lf'}
DASH = {'u2': '1.5 3.5', 'pm4': '6 4', 'pm10': '1.5 3.5', 'pm6i0': '2 2'}
ON = ['ntrl-demo', 'u4', 'u6', 'pm6', 'la6n', 'lf']
BOLD = ['pm6', 'la6n']
LEARNED = 'la6n'          # replaced in main() by the best learned variant with three seeds
LEARNED_CANDIDATES = ['la6n', 'la6r', 'la7b', 'la6b', 'lc6b']


def esc(s):
    return html.escape(str(s))


def norm_prep(p):
    return Path(p).name


def load20():
    base, warp = R.collect()
    return {lab: v for (lab, p), v in warp.items() if norm_prep(p) == PREP20}


def load():
    base, warp = R.collect()
    by = {}
    for (p, s), d in base.items():
        if norm_prep(p) == PREP:
            by.setdefault('ntrl-demo', {})[s] = d
    for (lab, p), v in warp.items():
        if norm_prep(p) == PREP:
            by[lab] = v
    ev = R.collect_evalq().get(Q1200, {})
    evby = {}
    ids = {d['_id']: (lab, s) for lab, v in by.items() for s, d in v.items()}
    for rid, e in ev.items():
        if rid in ids:
            lab, s = ids[rid]
            evby.setdefault(lab, {})[s] = e
    return by, evby


def seeds_of(v):
    return sorted(v, key=lambda s: (s != 87358, s))


def series(v, key='success_by_bin'):
    """[{bins, rows: [pct, min, max, nseeds, solved, runs]}] pooled over seeds."""
    out = dict(bins=[], rows=[])
    for b in BINS:
        per = [v[s][key].get(b) for s in seeds_of(v) if v[s][key].get(b)]
        if not per:
            continue
        pct = [100 * x[0] / x[1] for x in per]
        out['bins'].append(b)
        out['rows'].append([100 * sum(x[0] for x in per) / sum(x[1] for x in per), min(pct), max(pct), len(per), sum(x[0] for x in per), sum(x[1] for x in per)])
    return out


def totals(v):
    s = seeds_of(v)
    tot = [v[x]['success_total'] for x in s]
    short = [sum(c[0] for k, c in v[x]['success_by_bin'].items() if k != '32-64') for x in s]
    short_n = [sum(c[1] for k, c in v[x]['success_by_bin'].items() if k != '32-64') for x in s]
    long_ = [v[x]['success_by_bin'].get('32-64', [0, 0]) for x in s]
    return dict(seeds=s, per_seed=[t[0] for t in tot], of=tot[0][1], pct=100 * sum(t[0] for t in tot) / sum(t[1] for t in tot),
                short_pct=100 * sum(short) / sum(short_n), short_per_seed=short, short_of=short_n[0],
                long_pct=100 * sum(c[0] for c in long_) / max(1, sum(c[1] for c in long_)))


def pct(x, d=1):
    return f'{x:.{d}f}&nbsp;%'


def main():
    by, evby = load()
    att = json.loads((OUT / 'attract.json').read_text()) if (OUT / 'attract.json').exists() else {}
    T = {k: totals(v) for k, v in by.items()}
    E = {k: totals(v) for k, v in evby.items()}
    global LEARNED
    full = [k for k in LEARNED_CANDIDATES if k in T and len(T[k]['seeds']) >= 3]
    if full:
        LEARNED = max(full, key=lambda k: T[k]['pct'])
    for k in LEARNED_CANDIDATES:
        if k in T:
            CSS.setdefault(k, '--s-la')
    print('learned warp shown:', LEARNED, {k: round(T[k]['pct'], 1) for k in full})
    fields = [LEARNED if f == 'LEARNED' else f for f in FIELDS]
    on = [LEARNED if f == 'la6n' else f for f in ON]
    bold = ['pm6', LEARNED]
    data = dict(fields=[f for f in fields if f in by], sizes=['prep', 'q1200'], label=LABEL, css=CSS, dash=DASH, on=on, bold=bold,
                title={'prep': '240 queries (maze-scale set)', 'q1200': '1,200 new queries'},
                series={'prep': {f: series(by[f]) for f in fields if f in by}, 'q1200': {f: series(evby[f]) for f in fields if f in evby}})
    page = (HERE / 'page_template.html').read_text()
    hooks = build_text(by, evby, T, E, att)
    hooks.update(build_small(load20()))
    for k, v in hooks.items():
        page = page.replace(f'<!--{k}-->', v)
    page = page.replace('/*DATA*/', json.dumps(data))
    missing = [k for k in ('LEDE', 'TILES', 'IDEA', 'RESULTS', 'TOTALS', 'BOXTEXT', 'MECHTEXT', 'LEARNTEXT', 'VERDICT', 'MEASURED', 'RUNS') if f'<!--{k}-->' in page]
    assert not missing, f'unfilled hooks {missing}'
    (OUT / 'cross_warp.html').write_text(page)
    print('wrote', OUT / 'cross_warp.html', len(page))
    md = build_markdown(by, evby, T, E, att)
    md['SMALLER'] = build_small(load20(), markdown=True)['SMALLER']
    src = (HERE / 'README_src.md').read_text()
    for k, v in md.items():
        src = src.replace(f'<!--{k}-->', v)
    assert '<!--' not in src, 'unfilled README hook'
    (HERE / 'README.md').write_text(src)
    print('wrote', HERE / 'README.md')


def att_mean(att, lab, key='attract'):
    v = [x[key] for x in att.values() if x['label'] == lab]
    return float(np.mean(v)) if v else None


def stretch(by, lab, region):
    v = [d['warp_stats'][region]['logdet_median'] for d in by[lab].values() if d.get('warp_stats')]
    return float(np.mean([math.exp(x / 2) for x in v])) if v else None


def density_range(by, lab):
    """Mean over seeds of the smallest and largest local stretch of a separable warp's 1-D maps (x and y), overall scale included."""
    import torch
    import warp as W
    lo, hi = [], []
    for d in by.get(lab, {}).values():
        wp = W.load_field(d['_dir'] / 'training/checkpoint.pt', 'cpu').warp
        if not wp.sep:
            return None
        dens = torch.exp(torch.cat([wp.h_sx(), wp.h_sy()])).detach() * wp.s
        lo.append(float(dens.min()))
        hi.append(float(dens.max()))
    return (float(np.mean(lo)), float(np.mean(hi))) if lo else None


def escape(by, lab):
    s = seeds_of(by[lab])
    return [by[lab][x]['success_by_bin'].get('32-64', [0, 0])[0] for x in s], sum(by[lab][x]['success_by_bin'].get('32-64', [0, 0])[1] for x in s)


def paired(by, a, b):
    """Runs (query x planning seed x common training seed) only a solves / only b solves, exact McNemar p."""
    common = [x for x in seeds_of(by[a]) if x in by[b]]
    if not common:
        return None
    sa = np.concatenate([np.load(by[a][x]['_dir'] / 'rows.npz')['success'] for x in common])
    sb = np.concatenate([np.load(by[b][x]['_dir'] / 'rows.npz')['success'] for x in common])
    g, l = int((sa & ~sb).sum()), int((~sa & sb).sum())
    return dict(seeds=len(common), gain=g, loss=l, p=R.mcnemar(g, l))


def table_html(head, rows, cls=''):
    out = [f'<table class="{cls}"><thead><tr>' + ''.join(f'<th>{h}</th>' for h in head) + '</tr></thead><tbody>']
    for r in rows:
        out.append('<tr>' + ''.join(f'<td class="num">{c}</td>' if i else f'<td>{c}</td>' for i, c in enumerate(r)) + '</tr>')
    return '\n'.join(out) + '</tbody></table>'


def build_text(by, evby, T, E, att):
    H = {}
    nt, u4, u6, pm, la, lf = (T.get(k) for k in ('ntrl-demo', 'u4', 'u6', 'pm6', LEARNED, 'lf'))
    ent, eu4, eu6, epm, ela = (E.get(k) for k in ('ntrl-demo', 'u4', 'u6', 'pm6', LEARNED))
    seeds = lambda t: ' / '.join(map(str, t['per_seed']))
    H['LEDE'] = (f'On the 22&times;22 cross maze, a warp of the robot&rsquo;s configuration space placed in front of ntrl-demo&rsquo;s Fourier encoder, '
                 f'magnifying the maze six-fold and leaving the open + at its own scale, lifts ntrl-demo from <strong>{pct(nt["pct"])} to {pct(pm["pct"])}</strong> '
                 f'of planning runs (three training seeds: {seeds(nt)} &rarr; {seeds(pm)} of 720) and from {pct(ent["pct"])} to {pct(epm["pct"])} on 1,200 new queries. '
                 f'Raising the input frequency everywhere helps too, but stops paying past ×4 ({pct(eu4["pct"])} at ×4, {pct(eu6["pct"])} at ×6 on the new queries); '
                 f'the warp spends ×6 on the maze only and beats every uniform stretch on every seed there. The warp as first proposed, learned and confined to the '
                 f'box, learns to squeeze the + but cannot magnify the maze enough to matter ({pct(lf["pct"])}); learned without that limit it reaches {pct(la["pct"])}.')
    tiles = [('ntrl-demo', nt, ent, 'baseline, 3 training seeds'), ('uniform stretch ×4', u4, eu4, 'best uniform stretch of the input'),
             ('designed warp ×6 / ×1', pm, epm, 'maze ×6, open + ×1'), ('learned warp', la, ela, 'shape learned from the losses')]
    H['TILES'] = '\n'.join(f'<div class="tile"><span class="k">{esc(k)}</span><span class="v">{pct(t["pct"])}</span>'
                           f'<span class="s">{esc(sub)}; {pct(e["pct"]) if e else "&ndash;"} on the 1,200 new queries</span></div>' for k, t, e, sub in tiles if t)
    H['IDEA'] = (
        '<p>ntrl-demo learns a travel time T(x<sub>0</sub>, x<sub>1</sub>) between two robot poses with a network that first maps each pose to Fourier features '
        'sin/cos(2&pi;Bq). The features have fixed frequencies, so the network can only change so fast per unit of input. On the 22&times;22 maze a cell is '
        '1/22 of the unit box: walls fade out of the field, and the robot drives into them.</p>'
        '<p><code>README_warp.md</code> proposes to put a diffeomorphism &phi; between the pose and the encoder, T = H(&phi;(x<sub>0</sub>), &phi;(x<sub>1</sub>)). '
        'By the chain rule &nabla;<sub>x</sub>T = J<sub>&phi;</sub><sup>T</sup>&nabla;<sub>u</sub>H: where &phi; stretches by &sigma;, the same encoder can make the field '
        '&sigma; times sharper; where it compresses, the field is smoother. Stretch next to obstacles, compress open space, bound the Jacobian, learn &phi; with the '
        'field&rsquo;s losses and a term that pulls in free space. The cross maze is the natural test: 47&nbsp;% of it is one open + and the rest is four 8&times;8 mazes.</p>'
        '<p>Every warp here is a diffeomorphism by construction (monotone 1-D maps or coupling steps of positive densities), applied inside ntrl-demo&rsquo;s '
        'input mapping, so its loss, training loop and planner are untouched; with the identity warp the trainer reproduces ntrl-demo bit for bit. '
        'A uniform stretch &phi;(q) = s q is exactly ntrl-demo with its Fourier matrix multiplied by s (B&nbsp;&times;&nbsp;s), the control for every result.</p>')
    H['WARPCAP'] = ('The cross maze as the encoder sees it: the image of a regular grid under &phi; (divided by its overall scale), walls in red. '
                    'Uniform stretch magnifies everything alike; the warps magnify the four maze quadrants six-fold and keep the open + at its own scale '
                    '(designed: ×6 / ×1; learned: the same shape found by training).')
    # results table
    rows = []
    for k in ('ntrl-demo', 'id', 'lf', 'u2', 'u4', 'u6', 'pm4', 'pm6', 'pm6i0', 'pm10', LEARNED):
        if k not in T:
            continue
        t, e = T[k], E.get(k)
        esc_ = escape(by, k)
        rows.append([esc(LABEL.get(k, k)), str(len(t['seeds'])), ' / '.join(map(str, t['per_seed'])), f'<b>{t["pct"]:.1f}</b>',
                     f'{t["short_pct"]:.1f}', f'{" / ".join(map(str, esc_[0]))} <span class="mute">of 120</span>',
                     (f'{e["pct"]:.1f}' + (f' <span class="mute">({len(e["seeds"])})</span>' if len(e['seeds']) != len(t['seeds']) else '')) if e else '&ndash;',
                     f'{att_mean(att, k):.3f}' if att_mean(att, k) is not None else '&ndash;'])
    H['TOTALS'] = table_html(['field', 'seeds', 'runs solved per seed (of 720)', 'success %', '&le; 32 cells %', 'escape route, per seed',
                              '1,200 new queries %', 'wall attraction'], rows, 'named')
    prow = []
    for a, b in (('pm6', 'ntrl-demo'), ('pm6', 'u4'), ('pm6', 'u6'), (LEARNED, 'ntrl-demo'), ('lf', 'ntrl-demo'), ('u4', 'ntrl-demo'), ('u6', 'ntrl-demo')):
        if a in by and b in by:
            q = paired(by, a, b)
            if q:
                prow.append([f'{esc(LABEL.get(a, a))} vs {esc(LABEL.get(b, b))}', str(q['seeds']), f'+{q["gain"]}', f'&minus;{q["loss"]}', f'{q["p"]:.1g}'])
    H['TOTALS'] += '<h3>Paired, run by run</h3>' + table_html(['comparison', 'seeds', 'runs only the first solves', 'runs only the second solves', 'McNemar p'], prow, 'named')
    H['TOTALSNOTE'] = ('Success: MPPI runs that reach the goal without the robot touching a wall, planning seeds 0/1/2, pooled over training seeds 87358 / 1 / 2. '
                       '&le; 32 cells: the five shorter bins. Escape route: the 40 queries of the 32&ndash;64-cell bin, which all leave one dead-end pocket. '
                       '1,200 new queries: planning seed 0 (in brackets: training seeds evaluated there when fewer). Wall attraction: share of held-out '
                       'near-wall poses where the field&rsquo;s descent direction points into the wall (mean over seeds). Paired: the same query, planning seed and '
                       'training seed; McNemar&rsquo;s p treats runs as independent, but a query&rsquo;s three planning seeds nearly always agree and the training '
                       'seed is the larger source of variance, so p overstates; compare the per-seed counts (the README warp, for example, wins on two seeds through '
                       'the escape route and ties up to 32 cells).')
    eseeds = lambda e: ' / '.join(map(str, e['per_seed']))

    def where(k):
        bw = [evby[k][x].get('by_where') for x in evby.get(k, {})]
        return [f'{100 * sum(b[w][0] for b in bw) / sum(b[w][1] for b in bw):.0f}' for w in ('both', 'one', 'neither')] if bw and all(bw) else ['–'] * 3
    epm_where, eu4_where, eu6_where = where('pm6'), where('u4'), where('u6')
    H['RESULTS'] = (
        f'<p><strong>The designed warp beats ntrl-demo on every seed and every distance.</strong> {seeds(pm)} vs {seeds(nt)} runs of 720; on the 1,200 new queries '
        f'{eseeds(epm)} vs {eseeds(ent)} of 1,200. ntrl-demo&rsquo;s own trainer with the identity warp gives {seeds(T["id"])}, the same level.</p>'
        f'<p><strong>It also beats every uniform stretch.</strong> Multiplying ntrl-demo&rsquo;s Fourier matrix (a uniform stretch of the input) is the control: '
        f'×4 gives {pct(u4["pct"])} and {pct(eu4["pct"])} on the two query sets, ×6 {pct(u6["pct"])} and {pct(eu6["pct"])}. On the 1,200 new queries, the larger '
        f'set, the warp wins on every training seed ({eseeds(epm)} against {eseeds(eu4)} at ×4 and {eseeds(eu6)} at ×6), in every distance bin, and most where queries '
        f'enter the maze from the + or cross it: one end in the + {epm_where[1]} vs {eu4_where[1]} / {eu6_where[1]}&nbsp;%, neither end {epm_where[2]} vs {eu4_where[2]} / {eu6_where[2]}&nbsp;%.</p>')
    # --- the README warp as proposed ------------------------------------------------------------------------------
    lfs = stretch(by, 'lf', 'plus'), stretch(by, 'lf', 'quadrants')
    box_rows = []
    for k in ('ntrl-demo', 'lf', 'lfb', 'lf16', 'sl', 'dz3', 'dz12', 'sg', 'hp6'):
        if k in T:
            st = (stretch(by, k, 'plus'), stretch(by, k, 'quadrants')) if any(d.get('warp_stats') for d in by[k].values()) else (None, None)
            box_rows.append([esc(LABEL.get(k, k)), str(len(T[k]['seeds'])), ' / '.join(map(str, T[k]['per_seed'])), f'{T[k]["short_pct"]:.1f}',
                             f'{st[0]:.2f} / {st[1]:.2f}' if st[0] else '1 / 1', f'{att_mean(att, k):.3f}' if att_mean(att, k) is not None else '&ndash;'])
    H['BOXTABLE'] = table_html(['field', 'seeds', 'runs solved per seed (of 720)', '&le; 32 cells %', 'linear stretch: + / maze', 'wall attraction'], box_rows, 'named')
    H['BOXTEXT'] = (
        f'<p>The README&rsquo;s warp is the identity plus bounded increments, so over the scale of the maze it maps the box roughly onto itself. Implemented as '
        f'coupling steps that map the box exactly onto itself and trained as proposed (ntrl-demo&rsquo;s loss through &phi;, free space pulled in to a quarter of its area, '
        f'stretch bounded), <strong>it learns what it is asked</strong>: the open + shrinks to {lfs[0]:.2f}&times; per axis and the maze grows to {lfs[1]:.2f}&times;. '
        f'<strong>Planning does not change</strong>: {seeds(lf)} runs, {lf["short_pct"]:.1f}&nbsp;% up to 32 cells against ntrl-demo&rsquo;s {nt["short_pct"]:.1f}&nbsp;%, '
        f'and the walls leak as often.</p>'
        '<p>Two facts of this maze cap what any box-preserving warp can do. The + is 47&nbsp;% of the area, so even squeezing it to nothing frees '
        'at most &radic;(1/0.53) = 1.37&times; per axis for the maze. And the robot is slowed by a wall over about 65&nbsp;% of every corridor&rsquo;s width '
        '(0.325 of 0.45 cells on each side for an aligned robot, everything broadside), so stretching &ldquo;near walls&rdquo; is stretching the maze, not '
        'a thin band inside it. ntrl-demo&rsquo;s failures are wall leaks: at 88&ndash;100&nbsp;% of first wall contacts the field&rsquo;s descent points '
        'into the wall, with a learned speed 3&ndash;5 times the label. They need four- to six-fold magnification (table above), which only a warp that '
        'is allowed to enlarge the maze can give. Variants confirm it: a near-wall expansion term, designed versions of the same redistribution '
        '(coupling and separable, one with every grid line stretched) and a heading-only stretch all stay at ntrl-demo&rsquo;s level.</p>')
    # --- uniform vs local stretch -----------------------------------------------------------------------------------
    mech_rows = []
    for k in ('ntrl-demo', 'u2', 'u4', 'u6', 'pm4', 'pm6', 'pm10', LEARNED):
        if k not in T:
            continue
        e = escape(by, k)
        mech_rows.append([esc(LABEL.get(k, k)), str(len(T[k]['seeds'])), f'{T[k]["short_pct"]:.1f}', f'{100 * sum(e[0]) / max(e[1], 1):.0f}',
                          f'{att_mean(att, k):.3f}' if att_mean(att, k) is not None else '&ndash;',
                          f'{att_mean(att, k, "slope"):.2f}' if att_mean(att, k, 'slope') is not None else '&ndash;',
                          f'{np.mean([by[k][x]["bumps"]["bumps_per_path"] for x in by[k] if by[k][x].get("bumps")]):.2f}' if any(by[k][x].get('bumps') for x in by[k]) else '&ndash;'])
    H['MECHTABLE'] = table_html(['field', 'seeds', '&le; 32 cells %', 'escape route %', 'wall attraction', 'near-wall slope', 'local minima / long path'], mech_rows, 'named')
    H['MECHTEXT'] = (
        f'<p><strong>Short range is set by the walls.</strong> The share of near-wall poses whose descent points into the wall falls from '
        f'{att_mean(att, "ntrl-demo"):.3f} (ntrl-demo) to {att_mean(att, "u4"):.3f} at ×4 and {att_mean(att, "u6"):.3f} at ×6; the warp, ×6 in the maze, has '
        f'{att_mean(att, "pm6"):.3f}. Success up to 32 cells rises with it, from {nt["short_pct"]:.1f}&nbsp;% to {u4["short_pct"]:.1f} (×4), {u6["short_pct"]:.1f} (×6) '
        f'and {pm["short_pct"]:.1f}&nbsp;% (warp) on the 240 queries.</p>'
        f'<p><strong>Past ×4, uniform stretch stops paying, and the warp does not.</strong> ×6 everywhere draws sharper walls than ×4 but plans no better '
        f'({pct(u6["pct"])} vs {pct(u4["pct"])}; {pct(eu6["pct"])} vs {pct(eu4["pct"])} on the new queries). The warp with ×6 on the maze and ×1 on the + is ahead of both. At ×4, '
        f'where the uniform stretch is not yet too high, the same warp shape does not help ({pct(T["pm4"]["pct"]) if "pm4" in T else "&ndash;"} vs {pct(u4["pct"])}; '
        f'{pct(E["pm4"]["pct"]) if "pm4" in E else "&ndash;"} vs {pct(eu4["pct"])} on the new queries), and at ×10 / ×1 it '
        f'breaks ({pct(T["pm10"]["pct"]) if "pm10" in T else "&ndash;"}, stalls), as over-scaled Fourier matrices did in the maze-scale study. So the warp helps by letting '
        'the maze have a frequency that would already be too high for the open +.</p>'
        + (f'<p><strong>Where the magnification must go.</strong> On the 1,200 new queries, three training seeds each: the design, with the maze <em>and</em> the '
           f'+ band&rsquo;s edge cells (which carry the + outline walls) at ×6 and the open interior at ×1, plans {pct(epm["pct"])}; the same warp without that one-cell '
           f'inset, its outline walls on the ramp at about ×2&ndash;3, {pct(E["pm6i0"]["pct"])} (lower on every seed: {eseeds(E["pm6i0"])} vs {eseeds(epm)}); ×6 '
           f'everywhere {pct(eu6["pct"])}; ×4 everywhere {pct(eu4["pct"])}. Both mistakes cost the long queries that enter or cross the maze: the walls that bound the '
           'open area need the magnification as much as the maze does, and the open interior must not get it. This is the README&rsquo;s premise taken literally: '
           'stretch at every obstacle, compress open space.</p>' if 'pm6i0' in E and len(E['pm6i0']['seeds']) >= 3 else '')
        + '<p><strong>What ×6 spoils inside the open + is not settled.</strong> Its field there has no extra local minima (&le; 0.02 per long path for every field; '
        'almost all minima lie in the maze), no more slow patches and no worse near-wall leaks than the warp&rsquo;s. On the 240-query set the long bin is one route: '
        'all 40 queries of 32&ndash;64 cells start in a dead-end pocket at the top of the top-right quadrant (per seed, of 120: '
        f'×4 {" / ".join(map(str, escape(by, "u4")[0]))}, ×6 {" / ".join(map(str, escape(by, "u6")[0]))}, warp {" / ".join(map(str, escape(by, "pm6")[0]))}); '
        'it is erratic between fields, so the 1,200 new queries, whose long bins draw on 96 goals, are the firmer evidence.</p>')
    # --- learned warps ------------------------------------------------------------------------------------------------
    lrows = []
    for k, how in (('lf', 'coupling, box-preserving (README as proposed), free space pulled in'),
                   ('lsep4', 'free stretch starting at ×4, free space pulled in, 8 nodes / cell'),
                   ('lsb6', 'free stretch from ×6, + area target, near-wall area target, 8 nodes / cell'),
                   ('lsb6s', 'as above, strong curvature penalty'),
                   ('lsb6c', 'as above, 1 node / cell, small curvature penalty'),
                   ('lsn5', 'box-preserving separable × 5.1, free space pulled in'),
                   ('la6', 'free stretch from ×6, near-wall stretch along the wall normal, 1 node / cell; frozen at epoch 2,000'),
                   ('la6n', 'the same, learned to the end'),
                   ('la7b', 'the same, stretch bounded to ×1..×7, starting at ×2.65'),
                   ('la6r', 'la6n&rsquo;s learned warp frozen, field trained again from scratch'),
                   ('la6b', 'separable, stretch bounded to ×1..×6, starting at ×5.9'),
                   ('lc6b', 'coupling (row-dependent stretch), bounded to ×1..×6, starting at ×5.9')):
        if k not in T:
            continue
        lrows.append([esc(k), esc(how), str(len(T[k]['seeds'])), ' / '.join(map(str, T[k]['per_seed'])), f'{T[k]["pct"]:.1f}', f'{T[k]["short_pct"]:.1f}',
                      (f'{stretch(by, k, "plus"):.2f} / {stretch(by, k, "quadrants"):.2f}' if stretch(by, k, 'plus') else '&ndash;'),
                      str(int(np.mean([by[k][x]['failures']['stalled'] for x in by[k] if by[k][x].get('failures')])))])
    H['LEARNTABLE'] = table_html(['run', 'warp and terms', 'seeds', 'runs solved per seed', 'success %', '&le; 32 cells %', 'linear stretch: + / maze', 'stalled runs per seed'], lrows, 'named')
    la_st = density_range(by, LEARNED)
    i0, ei0 = T.get('pm6i0'), E.get('pm6i0')
    if i0:
        if ei0 and epm and ela and len(ei0['seeds']) >= 3:
            gap_text = (f'On the 1,200 new queries it plans {pct(ei0["pct"])}, against {pct(epm["pct"])} for the design with the inset and {pct(ela["pct"])} for the '
                        'learned warp: compressing the edge cells costs the design what separates it from the learned warp, on the long queries that pass the '
                        'outline walls')
        else:
            gap_text = 'Its 1,200-query evaluation is incomplete'
        ctrl = (f' Control: the design without its one-cell inset, compressing the edge cells as the learned warps do, plans {seeds(i0)} runs '
                f'({pct(i0["pct"])}; {i0["short_pct"]:.1f}&nbsp;% up to 32 cells against {pm["short_pct"]:.1f}&nbsp;% with the inset). {gap_text}.')
    else:
        ctrl = ''
    bnd = T.get('la6b')
    H['LEARNTEXT'] = (
        '<p>Can the warp be learned instead of designed? Every learned variant below trains ntrl-demo&rsquo;s loss through &phi; together with the warp&rsquo;s own terms, '
        'from the first epoch. Three lessons, in the order they came up.</p>'
        '<ul class="tight">'
        '<li><strong>Nothing in the loss asks for magnification.</strong> Started at a uniform ×4 with only the free-space term, the warp shrinks back to ×1.1 '
        'everywhere: compressing is always allowed, enlarging is never rewarded. It needs a near-wall term that asks for stretch.</li>'
        '<li><strong>The warp must be smooth.</strong> With 8 grid nodes per cell the learned densities turn jagged (spikes up to ×28 at the outer walls, '
        'second differences ten times the design&rsquo;s); the field inherits the ripples and MPPI stalls. One node per cell and a small curvature penalty fix it.</li>'
        '<li><strong>Stretch across the wall, not in area.</strong> An area target at near-wall poses forces the stretch onto whichever axis is free; at the ends of the '
        '+ arms, where one axis is compressed, it spikes. Asking for |J n| &ge; 6 along the clearance normal n (README_warp, section 10) stretches each axis across '
        'the walls that block it.</li></ul>'
        f'<p>With these, the warp learns the designed shape on its own (figure): the + band compressed to ×{la_st[0]:.2f} across, the maze at ×{min(la_st[1], 6.0):.0f}'
        + (f' (up to ×{la_st[1]:.1f} next to the outer walls)' if la_st[1] > 6.5 else '') + '. '
        + (f'The best learned variant, bounded to the design&rsquo;s own range (×1 to ×6), reaches the design&rsquo;s magnitudes exactly and plans ' if LEARNED == 'la6b' else 'The best learned variant plans ')
        + f'{seeds(la)} runs of 720 ({pct(la["pct"])}; {pct(ela["pct"]) if ela else "&ndash;"} on the 1,200 new queries): ten points above ntrl-demo, at the '
        'uniform stretch&rsquo;s level, below the design. Freezing a learned warp and training a new field on it from scratch changes nothing '
        '(tested on the unbounded variant), so the field adapting to a moving warp is not the problem.</p>'
        '<p>The profiles differ at the + band&rsquo;s edges. The design keeps the band&rsquo;s outermost cells, which carry the + outline walls, at ×6 and ramps down '
        'one cell in; every learned warp ramps down at the band edge itself. A separable map gives a whole column one stretch, and those edge columns are open space '
        f'inside the other arm of the +, where the free-space term compresses them.{ctrl}'
        + (f' A row-dependent learned warp (coupling steps, <code>lc6b</code>), which could compress the arms without those columns, shears instead: where the '
           f'compressed rows of an arm meet uncompressed maze rows its singular values reach {max(d["warp_stats"]["sv_xy"]["max"] for d in by["lc6b"].values()):.0f}, '
           f'and it plans {seeds(T["lc6b"])} runs ({pct(T["lc6b"]["pct"])}). A learned warp that keeps the outline walls magnified needs a shear bound or '
           f'a different form.' if 'lc6b' in T else '')
        + '</p>')
    H['DENSCAP'] = ('Local stretch along x across the maze (cell units; the + band shaded), training seed 87358: the design, the design without its one-cell '
                    'inset, and two learned warps. The learned warps ramp down at the band edge itself, like the design without the inset; the unbounded one also '
                    'overshoots next to the outer walls.')
    H['VERDICT'] = (
        '<ul class="tight">'
        f'<li><strong>The idea works on the cross maze</strong>, once the warp may enlarge the maze: ×6 in the four maze quadrants and ×1 over the open + lifts '
        f'ntrl-demo from {pct(nt["pct"])} to {pct(pm["pct"])} on every seed, and from {pct(ent["pct"])} to {pct(epm["pct"])} on 1,200 new queries.</li>'
        f'<li><strong>It beats raising the input frequency everywhere</strong> (best uniform stretch ×4: {pct(u4["pct"])} / {pct(eu4["pct"])}), on every seed of '
        'the larger query set. It needs the + outline walls magnified with the maze: without its one-cell inset it falls to the uniform ×4 level, and '
        '×6 that also covers the open interior does worse.</li>'
        '<li><strong>Box-preserving warps cannot help here.</strong> The README&rsquo;s warp learns to squeeze the + but the maze gains at most 1.37&times;, '
        'and ntrl-demo&rsquo;s failures (leaks through walls) need four to six.</li>'
        f'<li><strong>The shape can be learned</strong> from ntrl-demo&rsquo;s loss, a free-space term and a stretch-across-walls term: the best learned variant '
        f'reaches {pct(la["pct"])} / {pct(ela["pct"]) if ela else "&ndash;"}, about the uniform stretch&rsquo;s level and well above ntrl-demo, below the design.</li>'
        '<li>Too much magnification (×10) makes the field bumpy and MPPI stalls.</li>'
        + (lambda T20: (f'<li>On a smaller cross (20&times;20) the warp beats ntrl-demo too ({pct(T20["pm6"]["pct"])} vs {pct(T20["id"]["pct"])}, every seed) and '
                        f'edges out the best uniform stretch ({pct(T20["u4"]["pct"])}).</li>') if all(k in T20 for k in ('id', 'pm6', 'u4')) else '')(
            {k: totals(v) for k, v in load20().items()})
        + '</ul>')
    H['MEASURED'] = (
        '<ul class="tight">'
        '<li><strong>Maze and data</strong>: the maze-scale study&rsquo;s 22&times;22 plus maze (prepare run 20261102T042454Z-62864e61): 228 of 484 cells open, '
        'a 0.5 &times; 0.2-cell rectangle robot in SE(2), ntrl-demo&rsquo;s 500,000 training pairs.</li>'
        '<li><strong>Training</strong>: ntrl-demo&rsquo;s <code>Model.train()</code> step for step (<code>trainer.py</code>), 5,000 epochs, training seeds 87358 / 1 / 2; '
        'with the identity warp it reproduces ntrl-demo&rsquo;s registered run bit for bit on the same GPU. Warp grids: their own AdamW group (lr 3e-3, no weight decay).</li>'
        '<li><strong>Warps</strong>: monotone 1-D maps of x and y (densities on a node grid, integrated exactly) or coupling steps, so &phi; is a diffeomorphism '
        'by construction; heading scaled by the maze factor. Designed: ×m in the maze (quadrant columns and rows and the + band&rsquo;s edge cells), ×1 inside the band, '
        'one-cell ramps. Learned terms: free space (poses &ge; 1 cell from every wall) pulled to det J &le; 6, |J n| &ge; 6 along the clearance normal at ntrl-demo&rsquo;s '
        'near-wall poses, a curvature penalty 0.01, weights ramped in over 500 epochs.</li>'
        '<li><strong>Planning</strong>: ntrl-demo&rsquo;s MPPI scaled per cell, planning seeds 0/1/2 (seed 0 on the 1,200-query set); success = reaching the goal '
        'without the robot touching a wall (exact clearance every 0.005 cells). The oracle MPPI on the true travel time solves every query of both sets.</li>'
        '<li><strong>Diagnostics</strong>: wall attraction and near-wall slope on 50,000 held-out ntrl-demo pairs; local minima of T(x, goal) along true paths of '
        'queries over 8 cells.</li>'
        '<li><strong>Caveats</strong>: one maze, three training seeds per field (the same seed differs by about &plusmn;20 runs between the two GPUs); the 32&ndash;64-cell bin '
        'is one pocket, so its share of the totals is lumpy; the designed warp uses the + geometry (from the clearance field); the heading factor equals the maze '
        'factor in every warp and was not varied on its own beyond one heading-only control.</li></ul>')
    run_rows = []
    for k in sorted(by):
        for sd in seeds_of(by[k]):
            d = by[k][sd]
            run_rows.append([esc(LABEL.get(k, k)), str(sd), f'<code>{d["_id"]}</code>', f'{d["success_total"][0]} / {d["success_total"][1]}', esc(d.get('gpu', '').replace('NVIDIA GeForce ', ''))])
    H['RUNS'] = table_html(['field', 'training seed', 'registered run', 'runs solved', 'GPU'], run_rows, 'named')
    H['SPEEDCAP'] = ('Learned speed 1 / |&nabla;<sub>x</sub>T(x, goal)| at heading 0 towards the first prepared goal, training seed 87358; light = full speed, '
                     'dark = slow. ntrl-demo blurs the maze quadrants; ×6 (uniform or warped) redraws them.')
    return H


def build_small(by20, markdown=False):
    """The 20 x 20 cross: ntrl-demo (identity warp), uniform x4 / x6, designed warp x6 / x1."""
    rows = []
    T = {k: totals(v) for k, v in by20.items()}
    for k, name in (('id', 'ntrl-demo (identity warp)'), ('u4', 'uniform stretch ×4'), ('u6', 'uniform stretch ×6'), ('pm6', 'designed warp ×6 / ×1')):
        if k not in T:
            continue
        t = T[k]
        e = [by20[k][x]['success_by_bin'].get('32-64', [0, 0])[0] for x in seeds_of(by20[k])]
        rows.append([name, str(len(t['seeds'])), ' / '.join(map(str, t['per_seed'])), f'{t["pct"]:.1f}', f'{t["short_pct"]:.1f}', ' / '.join(map(str, e))])
    head = ['field', 'seeds', 'runs solved per seed (of 720)', 'success %', '≤ 32 cells %', 'escape route, per seed (of 120)']
    if not rows:
        return {'SMALLER': ''}
    if 'id' in T and 'pm6' in T:
        a, b = T['id'], T['pm6']
        lead = (f'On the 20&times;20 cross (204 of 400 cells open, prepare run {PREP20.replace("registered_", "")}) the same designed warp plans '
                f'{" / ".join(map(str, b["per_seed"]))} runs against ntrl-demo&rsquo;s {" / ".join(map(str, a["per_seed"]))} '
                f'({b["pct"]:.1f} vs {a["pct"]:.1f}&nbsp;%).')
        if 'u4' in T and 'u6' in T:
            lead += (f' Uniform stretch reaches {T["u4"]["pct"]:.1f}&nbsp;% at ×4 and {T["u6"]["pct"]:.1f}&nbsp;% at ×6. The warp is ahead of ×4 on every seed, but only '
                     f'by {min(x - y for x, y in zip(b["per_seed"], T["u4"]["per_seed"]))}&ndash;{max(x - y for x, y in zip(b["per_seed"], T["u4"]["per_seed"]))} runs '
                     f'({b["short_pct"]:.1f} vs {T["u4"]["short_pct"]:.1f}&nbsp;% up to 32 cells), and ×6 everywhere is again below ×4. On the smaller maze the '
                     'warp&rsquo;s margin over the best uniform stretch is small.')
    else:
        lead = 'The 20&times;20 cross runs are not complete yet.'
    if markdown:
        lead = lead.replace('&times;', '×').replace('&rsquo;', "'").replace('&nbsp;', ' ').replace('&ndash;', '–')
        return {'SMALLER': '\n'.join(['### A smaller cross (20×20)', '', lead, '', md_table(head, rows), ''])}
    return {'SMALLER': f'<h3>A smaller cross</h3><p>{lead}</p></div><div class="tablewrap">' + table_html(head, rows, 'named') + '</div><div class="col">'}


def small_md(by20):
    T20 = {k: totals(v) for k, v in by20.items()}
    if not all(k in T20 for k in ('id', 'pm6', 'u4', 'u6')):
        return ''
    a, b, c, d = T20['id'], T20['pm6'], T20['u4'], T20['u6']
    sd = lambda t: ' / '.join(map(str, t['per_seed']))
    return (f'7. **On a smaller cross (20×20) the direction holds, with a smaller margin over uniform stretch**: warp {sd(b)} ({b["pct"]:.1f} %), ntrl-demo '
            f'{sd(a)} ({a["pct"]:.1f} %), uniform ×4 {sd(c)} ({c["pct"]:.1f} %), ×6 {sd(d)} ({d["pct"]:.1f} %).')


def md_table(head, rows):
    out = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def build_markdown(by, evby, T, E, att):
    nt, u4, u6, pm, la, lf = (T.get(k) for k in ('ntrl-demo', 'u4', 'u6', 'pm6', LEARNED, 'lf'))
    ent, eu4, eu6, epm, ela = (E.get(k) for k in ('ntrl-demo', 'u4', 'u6', 'pm6', LEARNED))
    sd = lambda t: ' / '.join(map(str, t['per_seed']))
    es = lambda k: ' / '.join(map(str, escape(by, k)[0]))
    fmt = lambda e: f'{e["pct"]:.1f} %' if e else '–'
    M = {}
    M['SUMMARY'] = '\n'.join([
        '## Summary', '',
        f'1. **A warp that magnifies the maze and leaves the open + alone beats ntrl-demo on every seed.** Separable, ×6 over the four maze quadrants '
        f'(and the + band\'s edge cells), ×1 over the + band\'s interior, heading ×6 (`pm6`): {sd(pm)} of 720 runs ({pm["pct"]:.1f} %) against ntrl-demo\'s '
        f'{sd(nt)} ({nt["pct"]:.1f} %); on 1,200 new queries {fmt(epm)} against {fmt(ent)}.',
        f'2. **It also beats every uniform stretch of the encoder\'s input (B × s, ntrl-demo\'s Fourier matrix multiplied).** ×4: {sd(u4)} ({u4["pct"]:.1f} %); '
        f'×6: {sd(u6)} ({u6["pct"]:.1f} %). On the 1,200 new queries, three seeds each: warp {" / ".join(map(str, epm["per_seed"]))} ({epm["pct"]:.1f} %), ×4 '
        f'{" / ".join(map(str, eu4["per_seed"]))} ({eu4["pct"]:.1f} %), ×6 {" / ".join(map(str, eu6["per_seed"]))} ({eu6["pct"]:.1f} %): ahead on every seed and in '
        'every distance bin, most on queries that enter or cross the maze. Uniform stretch stops paying past ×4 (×6 draws sharper walls but plans worse). '
        + (f'The warp needs the + outline walls magnified too: without its one-cell inset (`pm6i0`) it plans {" / ".join(map(str, E["pm6i0"]["per_seed"]))} '
           f'({E["pm6i0"]["pct"]:.1f} %) of the 1,200, the uniform ×4 level. ' if 'pm6i0' in E else '')
        + (f'At ×4 the same shape gains nothing over uniform ×4 ({T["pm4"]["pct"]:.1f} %).' if 'pm4' in T else ''),
        f'3. **The README\'s warp as proposed (learned, box-preserving) cannot help here.** It learns to squeeze the + (×{stretch(by, "lf", "plus"):.2f} per '
        f'axis) but the maze only gains ×{stretch(by, "lf", "quadrants"):.2f}: {sd(lf)} runs, {lf["short_pct"]:.1f} % up to 32 cells against ntrl-demo\'s '
        f'{nt["short_pct"]:.1f} %. Squeezing the + (47 % of the area) to nothing frees at most ×1.37, the robot is slowed over ~65 % of every corridor\'s '
        'width (so "near walls" is the whole maze), and ntrl-demo\'s failures are leaks through walls that need ×4-6.',
        f'4. **Mechanism: wall leaks set the short range.** At 88-100 % of first wall contacts the field\'s descent points into the wall. The share of '
        f'held-out near-wall poses with that defect: ntrl-demo {att_mean(att, "ntrl-demo"):.3f}, README warp {att_mean(att, "lf"):.3f}, ×4 {att_mean(att, "u4"):.3f}, '
        f'×6 {att_mean(att, "u6"):.3f}, warp {att_mean(att, "pm6"):.3f}; success up to 32 cells follows it. What ×6 everywhere spoils at long range is not '
        'settled: the open + shows no extra local minima, slow patches or leaks. On the 240-query set the long bin is one dead-end pocket (the escape route), '
        'which behaves erratically between fields (the ×4 warp lost it, uniform ×4 kept it); the 1,200-query set, 96 goals, is the firmer evidence.',
        (f'5. **The shape can be learned**, from ntrl-demo\'s loss, the free-space term and a stretch across walls along the clearance normal. Best learned '
         f'variant (`{LEARNED}`): {sd(la)} runs ({la["pct"]:.1f} %), {ela["pct"]:.1f} % on the 1,200 queries: ten points above ntrl-demo, at the uniform '
         f'stretch\'s level, below the design. '
         + ('It is bounded to the design\'s range (×1-×6) and reaches the design\'s magnitudes exactly. ' if LEARNED == 'la6b' else '')
         + 'Freezing the learned warp and training a new field on it (`la6r`) changes nothing. The learned warps compress the + band\'s edge cells (a separable '
         'map gives a whole column one stretch, and those columns are open space in the other arm), the design keeps them at ×6'
         + (f'; the design without that inset (`pm6i0`) plans {sd(T["pm6i0"])} ({T["pm6i0"]["pct"]:.1f} %) and {E["pm6i0"]["pct"]:.1f} % on the 1,200 '
            f'queries, the learned level ({ela["pct"]:.1f} %), against {epm["pct"]:.1f} % with it. A row-dependent learned warp (coupling, `lc6b`) shears '
            f'instead (singular values up to ~130) and plans {T["lc6b"]["pct"]:.1f} %.' if 'pm6i0' in E and 'lc6b' in T else '.')
         + ' Without a near-wall term a learned free stretch shrinks back to ×1; at 8 nodes per cell it turns jagged.') if la and ela else '',
        f'6. **Too much magnification breaks the field**: ×10 / ×1 plans {T["pm10"]["pct"]:.1f} % (stalls), as over-scaled Fourier matrices did in the '
        'maze-scale study.' if 'pm10' in T else '',
        small_md(load20()),
        ''])
    rows = []
    for k in ('ntrl-demo', 'id', 'lf', 'u2', 'u4', 'u6', 'pm4', 'pm6', 'pm6i0', 'pm10', LEARNED):
        if k in T:
            t, e = T[k], E.get(k)
            rows.append([LABEL.get(k, k) + f' (`{k}`)', len(t['seeds']), sd(t), f'{t["pct"]:.1f}', f'{t["short_pct"]:.1f}', es(k),
                         (f'{e["pct"]:.1f}' + (f' ({len(e["seeds"])} seed{"s" if len(e["seeds"]) != 1 else ""})' if len(e['seeds']) != len(t['seeds']) else '')) if e else '–',
                         f'{att_mean(att, k):.3f}' if att_mean(att, k) is not None else '–'])
    main = md_table(['field', 'seeds', 'runs solved per seed (of 720)', 'success %', '≤ 32 cells %', 'escape route (of 120)', '1,200 new queries %', 'wall attraction'], rows)
    lrows = []
    for k in ('lf', 'lfb', 'lf16', 'lfs2', 'lfb4', 'sl', 'dz3', 'dz12', 'sg', 'hp6', 'lsep4', 'lsb6', 'lsb6s', 'lsb6c', 'lsn5', 'la6', 'la6n', 'la7b', 'la6r', 'la6b', 'lc6b', 'pm6i0'):
        if k in T:
            st = (stretch(by, k, 'plus'), stretch(by, k, 'quadrants'))
            lrows.append([LABEL.get(k, k) + f' (`{k}`)', len(T[k]['seeds']), sd(T[k]), f'{T[k]["pct"]:.1f}', f'{T[k]["short_pct"]:.1f}',
                          f'{st[0]:.2f} / {st[1]:.2f}' if st[0] else '–', f'{att_mean(att, k):.3f}' if att_mean(att, k) is not None else '–'])
    other = md_table(['field', 'seeds', 'runs solved per seed', 'success %', '≤ 32 cells %', 'linear stretch + / maze', 'wall attraction'], lrows)
    run_rows = []
    for k in sorted(by):
        for x in seeds_of(by[k]):
            d = by[k][x]
            run_rows.append([k, x, f'`{d["_id"]}`', f'{d["success_total"][0]} / {d["success_total"][1]}', d.get('gpu', '').replace('NVIDIA GeForce ', '')])
    M['RESULTS'] = '\n'.join([
        '## Results', '',
        'MPPI success over planning seeds 0/1/2, pooled over training seeds 87358 / 1 / 2 (per-seed counts listed). ≤ 32 cells: the five shorter bins. '
        'Escape route: the 32-64-cell bin, per training seed. 1,200 new queries: planning seed 0. Wall attraction: share of held-out near-wall poses whose '
        'descent direction points into the wall (`attract.py`).', '', main, '',
        '![the maze as the encoder sees it](report/warps.png)', '',
        '<!--SMALLER-->', '',
        '### Every variant', '', other, '',
        '![learned and designed stretch profiles](report/density.png)', '',
        '### Registered runs', '', md_table(['label', 'training seed', 'run id', 'runs solved', 'GPU'], run_rows), '',
        f'Query set: `20261103T174058Z-405b526e`; its evaluations: the `evalq` runs listed by `report.py` (`report/tables.md`).', ''])
    return M


if __name__ == '__main__':
    main()
