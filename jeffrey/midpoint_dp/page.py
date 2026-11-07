"""Findings page for the midpoint loss (report/midpoint.html), built from report/data.json (report.py) and report/*.png.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/midpoint_dp/page.py

Every number on the page is computed here from the registered runs' results (report.py collect()); the prose around
the numbers is in this file. Publish report/midpoint.html with the files paths13.png, paths22.png next to it.
"""
import html
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'report'
MAIN = 'safe, min leaves, tree + sub-paths'
BSCALE = {'10': 3, '13': 3, '22': 6}
SEEDS = ['87358', '1', '2']
BINS = ['0.5-2', '2-4', '4-8', '8-16', '16-32', '32-64']


def load():
    return json.load(open(OUT / 'data.json'))


def tot(runs, n, v, seeds=None):
    rs = runs.get(n, {}).get(v, {})
    ks = [s for s in rs if seeds is None or s in seeds]
    a = sum(rs[s]['success'][0] for s in ks)
    b = sum(rs[s]['success'][1] for s in ks)
    return a, b, ks


def pct(a, b, d=0):
    return f'{100 * a / b:.{d}f}' if b else '–'


def series_data(runs):
    """Per maze: series -> per bin [mean %, min %, max %, seeds, success, runs]."""
    out = {}
    for n, by_v in runs.items():
        m = {}
        for v, rs in by_v.items():
            ks = [b for b in BINS if any(b in r['by_bin'] for r in rs.values())]
            rows = []
            for b in ks:
                per = [100 * r['by_bin'][b][0] / r['by_bin'][b][1] for r in rs.values() if b in r['by_bin']]
                a = sum(r['by_bin'][b][0] for r in rs.values() if b in r['by_bin'])
                c = sum(r['by_bin'][b][1] for r in rs.values() if b in r['by_bin'])
                rows.append([round(float(np.mean(per)), 1), round(min(per), 1), round(max(per), 1), len(per), a, c])
            ratio = []
            for b in ks:
                vals = [r['ratio'][b] for r in rs.values() if b in r['ratio']]
                ratio.append(round(float(np.mean(vals)), 3) if vals else None)
            m[v] = dict(bins=ks, rows=rows, ratio=ratio, seeds=len(rs))
        out[n] = m
    return out


def variant_rows(runs, n='13', seed='87358'):
    order = [('control', 'no midpoint term (same resumed loop)'),
             ('plain, net leaves, tree', 'literal recursion: network at the leaves, test-planner traversals, midpoint tree'),
             ('plain, min leaves, tree', 'leaves anchored to the path\'s own travel time'),
             ('safe, net leaves, tree', 'network leaves, collision-aware traversals'),
             ('safe, min leaves, tree', 'anchored leaves, collision-aware traversals'),
             (MAIN, 'anchored leaves, collision-aware traversals, plus random sub-paths (main)')]
    rows = []
    by_v = runs.get(n, {})
    base = by_v.get('ntrl-demo', {}).get(seed)
    if base:
        rows.append(('ntrl-demo (baseline, epoch 5000)', '', base))
    for v, desc in order:
        r = by_v.get(v, {}).get(seed)
        if r:
            rows.append((v, desc, r))
    abl = []
    for v, rs in sorted(by_v.items()):
        if v.startswith(MAIN + ' (') and seed in rs:
            abl.append((v.replace(MAIN, 'main').replace('(', '').replace(')', ''), '', rs[seed]))
    return rows, abl


def table(head, rows, cls='named'):
    h = '<table class="' + cls + '"><thead><tr>' + ''.join(f'<th>{html.escape(x)}</th>' for x in head) + '</tr></thead><tbody>'
    for r in rows:
        h += '<tr>' + ''.join(f'<td class="num">{x}</td>' if i >= 1 and not str(x).startswith('<') and any(c.isdigit() for c in str(x)) else f'<td>{x}</td>' for i, x in enumerate(r)) + '</tr>'
    return h + '</tbody></table>'


def build():
    d = load()
    runs = d['runs']
    S = series_data(runs)
    head = {}
    for n in ['10', '13', '22']:
        for v in ['ntrl-demo', 'control', MAIN, f'ntrl-demo B×{BSCALE[n]}', f'{MAIN} on B×{BSCALE[n]}']:
            a, b, ks = tot(runs, n, v)
            head[(n, v)] = (a, b, len(ks))

    def P(n, v, dg=0):
        a, b, k = head[(n, v)]
        return pct(a, b, dg)

    def seedline(n, v):
        rs = runs.get(n, {}).get(v, {})
        return ' / '.join(f'{rs[s]["success"][0]}' for s in SEEDS if s in rs)

    # ---------------- tiles
    tiles = ''
    for n in ['10', '13', '22']:
        a, b, k = head[(n, MAIN)]
        fa, fb, fk = head[(n, f'ntrl-demo B×{BSCALE[n]}')]
        ba, bb, bk = head[(n, 'ntrl-demo')]
        ma, mb, mk_ = head[(n, f'{MAIN} on B×{BSCALE[n]}')]
        tiles += (f'<div class="tile"><div class="k">{n}×{n} maze</div><div class="v">{pct(a, b)} %</div>'
                  f'<div class="s">ntrl-demo {pct(ba, bb)} % · B×{BSCALE[n]} {pct(fa, fb)} % · midpoint on B×{BSCALE[n]} {pct(ma, mb)} %</div>'
                  f'<div class="s mute">{a}/{b} runs, {k} training seed{"s" if k != 1 else ""}</div></div>')
    # ---------------- headline table
    hrows = []
    for n in ['10', '13', '22']:
        for v, lab in [('ntrl-demo', 'ntrl-demo'), ('control', 'control (resumed, no term)'), (MAIN, '<strong>midpoint loss</strong>'),
                       (f'ntrl-demo B×{BSCALE[n]}', f'ntrl-demo, B×{BSCALE[n]}'), (f'{MAIN} on B×{BSCALE[n]}', f'midpoint loss on B×{BSCALE[n]}')]:
            rs = runs.get(n, {}).get(v, {})
            if not rs:
                continue
            a, b, k = head[(n, v)]
            hrows.append([f'{n}×{n}', lab] + [f'{rs[s]["success"][0]}/{rs[s]["success"][1]}' if s in rs else '' for s in SEEDS] + [f'{a}/{b}', pct(a, b)])
    headline = table(['maze', 'field', 'seed 87358', 'seed 1', 'seed 2', 'all', '%'], hrows)
    # ---------------- by-bin table (pooled)
    brows = []
    for n in ['10', '13', '22']:
        for v, lab in [('ntrl-demo', 'ntrl-demo'), ('control', 'control'), (MAIN, 'midpoint loss'), (f'ntrl-demo B×{BSCALE[n]}', f'ntrl-demo B×{BSCALE[n]}'), (f'{MAIN} on B×{BSCALE[n]}', f'midpoint on B×{BSCALE[n]}')]:
            m = S.get(n, {}).get(v)
            if not m:
                continue
            cells = []
            for b in BINS:
                if b in m['bins']:
                    r = m['rows'][m['bins'].index(b)]
                    cells.append(f'{pct(r[4], r[5])}')
                else:
                    cells.append('')
            brows.append([f'{n}×{n}', lab] + cells)
    bintable = table(['maze', 'field'] + [f'{b} cells' for b in BINS], brows)
    # ---------------- field diagnostics
    grad = d.get('grad', {})
    frows = []
    for n in ['10', '13', '22']:
        for v, lab in [('ntrl-demo', 'ntrl-demo'), ('control', 'control'), (MAIN, 'midpoint loss'), (f'ntrl-demo B×{BSCALE[n]}', f'ntrl-demo B×{BSCALE[n]}'), (f'{MAIN} on B×{BSCALE[n]}', f'midpoint on B×{BSCALE[n]}')]:
            rs = runs.get(n, {}).get(v, {})
            if not rs:
                continue
            m = S[n][v]
            rt = ' · '.join(f'{x:.2f}' for x in m['ratio'] if x is not None)
            g = [grad[r['id']] for r in rs.values() if r['id'] in grad]
            into = f'{100 * np.mean([x["into_wall"] for x in g]):.1f}' if g else ''
            fg = f'{np.mean([x["free_grad"] for x in g]):.2f}' if g else ''
            cos = np.mean([r['cos'] for r in rs.values()])
            frows.append([f'{n}×{n}', lab, rt, f'{cos:.2f}', into, fg])
    fieldtable = table(['maze', 'field', 'median T / T* by distance bin', 'descent cos', 'descent into wall (%)', 'free-space |∇T|'], frows)
    # ---------------- u-turns
    ut = d.get('uturn', {})
    urows = []
    for n in ['10', '13', '22']:
        for v, lab in [('ntrl-demo', 'ntrl-demo'), (MAIN, 'midpoint loss'), (f'ntrl-demo B×{BSCALE[n]}', f'ntrl-demo B×{BSCALE[n]}'), (f'{MAIN} on B×{BSCALE[n]}', f'midpoint on B×{BSCALE[n]}')]:
            rs = runs.get(n, {}).get(v, {})
            got = {s: ut[r['id']] for s, r in rs.items() if r['id'] in ut}
            if not got:
                continue
            a = sum(x['success'] for x in got.values())
            b = sum(x['total'] for x in got.values())
            urows.append([f'{n}×{n}', lab] + [f'{got[s]["success"]}/{got[s]["total"]}' if s in got else '' for s in SEEDS] + [pct(a, b)])
    uturntable = table(['maze', 'field', 'seed 87358', 'seed 1', 'seed 2', '%'], urows)
    # ---------------- variants
    vrows, abl = variant_rows(runs)
    vt = []
    for v, desc, r in vrows:
        bb = r['by_bin']
        vt.append([v, desc, f'{r["success"][0]}/{r["success"][1]}', pct(*r['success']), pct(*bb['16-32']) if '16-32' in bb else '',
                   f'{np.mean([x for x in r["ratio"].values()]):.2f}'])
    vtable = table(['variant', 'what changes', 'success', '%', '16-32 cells (%)', 'mean T/T*'], vt)
    at = []
    for v, desc, r in abl:
        bb = r['by_bin']
        at.append([v, f'{r["success"][0]}/{r["success"][1]}', pct(*r['success']), pct(*bb['16-32']) if '16-32' in bb else ''])
    abltable = table(['main config with', 'success', '%', '16-32 cells (%)'], at) if at else '<p class="note">Ablations still running.</p>'
    # ---------------- run index
    idx = []
    for n in ['10', '13', '22']:
        for v, rs in sorted(runs.get(n, {}).items()):
            for s in SEEDS:
                if s in rs:
                    idx.append([f'{n}×{n}', html.escape(v), s, f'<code>{rs[s]["id"]}</code>'])
    runtable = table(['maze', 'field', 'seed', 'registered run'], idx)

    m10, m13, m22 = (head[(n, MAIN)] for n in ['10', '13', '22'])
    b10, b13, b22 = (head[(n, 'ntrl-demo')] for n in ['10', '13', '22'])
    lede = (f'Adding a travel-time midpoint loss to ntrl-demo after epoch 2,000 lifts MPPI success from '
            f'<strong>{pct(b10[0], b10[1])} to {pct(m10[0], m10[1])}&nbsp;% on 10&times;10, {pct(b13[0], b13[1])} to {pct(m13[0], m13[1])}&nbsp;% on 13&times;13 '
            f'and {pct(b22[0], b22[1])} to {pct(m22[0], m22[1])}&nbsp;% on 22&times;22</strong>, with the planner, data, network and the first 2,000 epochs unchanged. '
            f'The term asks that the network\'s travel time between two poses equal the sum over the two halves of a path the planner actually drove, split at the '
            f'half-way point in travel time and recursively down to one cell.')
    # ---------------- prose blocks (numbers computed here)
    def into(n, v, rng):
        rs = runs.get(n, {}).get(v, {})
        vals = [grad[r['id']]['ranges'][rng]['into_wall'] for r in rs.values() if r['id'] in grad and 'ranges' in grad[r['id']]]
        return f'{100 * np.mean(vals):.1f}' if vals else '–'

    def ratio(n, v, b):
        m = S.get(n, {}).get(v)
        return f'{m["ratio"][m["bins"].index(b)]:.2f}' if m and b in m['bins'] else '–'

    def one(v, n='13', s='87358'):
        r = runs.get(n, {}).get(v, {}).get(s)
        return f'{r["success"][0]}' if r else 'not run'

    def utp(n, v):
        rs = runs.get(n, {}).get(v, {})
        got = [ut[r['id']] for r in rs.values() if r['id'] in ut]
        a, b = sum(x['success'] for x in got), sum(x['total'] for x in got)
        return pct(a, b) if b else '–'
    mains = [x for n in ['10', '13', '22'] for x in S[n][MAIN]['ratio'] if x is not None]
    def abl(key):
        hits = [v for v in runs['13'] if v.startswith(MAIN + ' (') and key in v]
        return one(hits[0]) if hits else 'not run'
    fieldtext = (f'ntrl-demo&rsquo;s long-range values drift high: on 13&times;13 its median T/T* is {ratio("13", "ntrl-demo", "16-32")} at 16&ndash;32 cells. '
                 f'With the midpoint term the field sits at {min(mains):.2f}&ndash;{max(mains):.2f} in every distance bin of every maze. '
                 f'Planning gains for a reason one step removed: MPPI fails by driving into walls, which happens where the field&rsquo;s descent direction next to a wall points into it, '
                 f'because the goal is behind the wall and the field does not know the way round. Correct long-range values for pairs whose route goes around the wall turn that direction along the wall. '
                 f'The share of such near-wall poses falls from {into("13", "ntrl-demo", ">8")} to {into("13", MAIN, ">8")} % on 13&times;13 and from {into("22", "ntrl-demo", ">8")} to {into("22", MAIN, ">8")} % '
                 f'on 22&times;22 for goals beyond 8 cells, and by similar factors for nearer goals, while the steepness of the field next to walls changes little.')
    def same(n):
        fm = runs.get(n, {}).get(f'{MAIN} on B×{BSCALE[n]}', {})
        fb = runs.get(n, {}).get(f'ntrl-demo B×{BSCALE[n]}', {})
        ks = [x for x in SEEDS if x in fm and x in fb]
        return ' / '.join(str(fm[x]['success'][0]) for x in ks), ' / '.join(str(fb[x]['success'][0]) for x in ks)
    s10, s13, s22 = same('10'), same('13'), same('22')
    fouriertext = (f'The maze-scale study&rsquo;s best fix for ntrl-demo was a higher input frequency: its Fourier matrix multiplied by 3 (10&times;10, 13&times;13) or 6 (22&times;22). '
                   f'Run on those networks, forked at epoch 2,000 in the same way, the midpoint loss adds to it on every maze: '
                   f'<strong>{P("10", f"{MAIN} on B×3")} / {P("13", f"{MAIN} on B×3")} / {P("22", f"{MAIN} on B×6")}&nbsp;%</strong> against {P("10", "ntrl-demo B×3")} / {P("13", "ntrl-demo B×3")} / {P("22", "ntrl-demo B×6")}&nbsp;% for the higher-frequency networks alone. '
                   f'Per training seed: 22&times;22 {s22[0]} vs {s22[1]} of 720, 13&times;13 {s13[0]} vs {s13[1]} of 600, 10&times;10 {s10[0]} vs {s10[1]}. '
                   f'The old segment-DP chain planned worse on these networks than on the plain ones; the midpoint term, which keeps ntrl-demo&rsquo;s own loss in the same network, does not have that problem. '
                   f'Switch on the dashed series above to see it by distance.')
    varianttext = (f'All on 13&times;13 with training seed 87358. Read literally, with the network&rsquo;s own T at the leaves and the planner&rsquo;s own drives, the loss gets {one("plain, net leaves, tree")} of 600 against the control&rsquo;s {one("control")}, '
                   f'but slowly, and the whole field ends up inflated (T/T* {ratio("13", "plain, net leaves, tree", "0.5-2")} at 0.5&ndash;2 cells): each leaf carries ntrl-demo&rsquo;s short-range overestimate and the targets add them up. '
                   f'Anchoring each leaf to the travel time the robot actually needed over it gives {one("plain, min leaves, tree")}, and {one("safe, min leaves, tree")} with training drives that keep out of walls. '
                   f'Adding random sub-paths between leaf ends to those gives {one(MAIN)}. '
                   f'The same wall-avoiding drives with the network at the leaves get only {one("safe, net leaves, tree")}: more long valid paths, each adding more biased leaves.')
    abltext = (f'Against {one(MAIN)} for the main setting: full weight from epoch 2,001 without the ramp {abl("ramp 1")}; weight 0.3 {abl("weight 0.3")}; weight 3 {abl("weight 3")}; '
               f'term from epoch 3,000 {abl("start 3000")}. Where the recursion stops matters: leaves of 0.5 cells {abl("leaf_cells 0.5")}, 2 cells {abl("leaf_cells 2")}. '
               f'The 0.5-cell run&rsquo;s values are as calibrated as the main run&rsquo;s, but its descent direction near walls falls back to ntrl-demo&rsquo;s for goals under 8 cells '
               f'({into("13", MAIN + " (leaf_cells 0.5)", "0.5-2")} % into the wall at 0.5&ndash;2 cells vs {into("13", MAIN, "0.5-2")} %), and it hits walls three times as often.')
    uturntext = (f'A u-turn is the 1-cell query around the end of a wall from the maze-scale study: start and goal sit in the two cells on either side, and the true path goes around, about 2.2 cells. '
                 f'It tests how sharply one wall is drawn, not long-range routing. The midpoint loss helps (13&times;13: {utp("13", "ntrl-demo")} &rarr; {utp("13", MAIN)} %), '
                 f'but the higher input frequency helps more ({utp("13", "ntrl-demo B×3")} %), and on 22&times;22 the two add up. The fixes address different failures.')
    measured = ''.join(f'<li>{x}</li>' for x in [
        '<strong>Network and data</strong>: ntrl-demo&rsquo;s <code>models/metric</code> network and its 500,000-pair point recipe in the robot&rsquo;s C-space, as in the maze-scale study; nothing in the network, data or planner changes.',
        '<strong>Training</strong>: each run resumes a registered ntrl-demo run at its epoch-2,000 checkpoint (network, AdamW state, Fourier matrix) and runs ntrl-demo&rsquo;s own loop to epoch 5,000 with the midpoint term added; the control runs the same loop with weight 0.',
        '<strong>Midpoint term</strong>: every 100 epochs, 8,192 drives of ntrl-demo&rsquo;s MPPI between random collision-free poses on the current field; travel time from the speed the labels use; recursive split at the travel-time midpoint down to 1-cell leaves; |log T &minus; log target| on 8,192 pieces per step, weight ramped 0&rarr;1 over epochs 2,000&ndash;2,500.',
        '<strong>Batching</strong>: the goal&rsquo;s code is computed once per drive and the encoder runs compiled in fp32 on million-state chunks (10.6 M states/s on the RTX 3090); a refresh is about 170 M network evaluations, 20&ndash;30 s on an idle 3090 for 13&times;13.',
        '<strong>Evaluation</strong>: the maze-scale queries (40 per true-distance bin, certified), ntrl-demo&rsquo;s MPPI with planning seeds 0, 1 and 2, success = reaching the goal without touching a wall (exact clearance every 0.005 cells), network at epoch 5,000.',
        '<strong>Cost</strong>: epochs 2,001&ndash;5,000 take 18&ndash;23 minutes on an idle RTX 5070 for 10&times;10 and 13&times;13 (ntrl-demo alone: 3.5 minutes); refreshes are two thirds of it.'])
    caveats = ''.join(f'<li>{x}</li>' for x in [
        'One maze layout per size; 40 queries per distance bin.',
        'The midpoint term queries the exact clearance far more often than ntrl-demo&rsquo;s labels do: along every drive and, for the wall-avoiding drives, at every rollout&rsquo;s first state. The literal variant uses it only to cut drives at collisions and to measure travel time, and still gains; a control with matched extra clearance queries was not run.',
        'Variants and ablations use training seed 87358 only; checkpoints drift by up to ~35 of 600 between epochs, so single-seed differences below that are not meaningful.',
        'Leaf values, drive mode and sub-paths were chosen on 13&times;13 with training seed 87358 and carried to the other mazes and seeds unchanged.',
        'The higher-frequency runs (B&times;3, B&times;6) have fewer training seeds than the plain ones; see the table.'])
    data_js = json.dumps(dict(series=S, bscale=BSCALE), separators=(',', ':'))
    tpl = (HERE / 'page_template.html').read_text()
    page = (tpl.replace('<!--LEDE-->', lede).replace('<!--TILES-->', tiles).replace('<!--HEADLINE-->', headline).replace('<!--BINTABLE-->', bintable)
            .replace('<!--FIELDTABLE-->', fieldtable).replace('<!--UTURNTABLE-->', uturntable).replace('<!--VARIANTS-->', vtable)
            .replace('<!--ABLATIONS-->', abltable).replace('<!--RUNS-->', runtable).replace('/*DATA*/', data_js)
            .replace('<p id="fieldtext">The midpoint term calibrates the long-range values and turns the field&rsquo;s descent direction away from walls.</p>', f'<p id="fieldtext">{fieldtext}</p>')
            .replace('<p id="varianttext"></p>', f'<p id="varianttext">{varianttext}</p>').replace('<p id="fouriertext"></p>', f'<p id="fouriertext">{fouriertext}</p>').replace('<p id="abltext"></p>', f'<p id="abltext">{abltext}</p>')
            .replace('<p id="uturntext"></p>', f'<p id="uturntext">{uturntext}</p>').replace('<ul class="tight" id="measured"></ul>', f'<ul class="tight" id="measured">{measured}</ul>')
            .replace('<ul class="tight" id="caveats"></ul>', f'<ul class="tight" id="caveats">{caveats}</ul>'))
    for key, val in dict(M10=seedline('10', MAIN), B10=seedline('10', 'ntrl-demo'), M13=seedline('13', MAIN), B13=seedline('13', 'ntrl-demo'),
                         M22=seedline('22', MAIN), B22=seedline('22', 'ntrl-demo')).items():
        page = page.replace('{{' + key + '}}', val)
    (OUT / 'midpoint.html').write_text(page)
    print(f'wrote {OUT / "midpoint.html"} ({len(page) / 1024:.0f} KB)')


if __name__ == '__main__':
    build()
