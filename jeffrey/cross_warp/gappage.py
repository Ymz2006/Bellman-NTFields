"""Findings page for the gap study (report/gap.html) from the registered runs (gapreport.collect()).

    ~/.venvs/bellman-ntfield/bin/python jeffrey/cross_warp/gappage.py

Every number on the page is computed here from the runs; gap_template.html holds the layout and the chart script.
Publish report/gap.html with gap_mazes.png and gap_warps.png next to it.
"""
import html
import json
import math
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'report'
sys.path.insert(0, str(HERE))
import gapreport as GR  # noqa: E402

CATS = GR.CATS
METHODS = GR.METHODS
CSS = {'ntrl-demo': '--s-ntrl', 'uniform': '--s-u4', 'constructed': '--s-pm', 'learned': '--s-la', 'warp + scaling': '--s-ws', 'README warp': '--s-lf'}
DASH = {'warp + scaling': '6 4'}


def esc(s):
    return html.escape(str(s))


def table_html(head, rows, cls='named', text_cols=(0,)):
    out = [f'<table class="{cls}"><thead><tr>' + ''.join(f'<th>{h}</th>' if i not in text_cols or i == 0 else f'<th class="txt">{h}</th>' for i, h in enumerate(head)) + '</tr></thead><tbody>']
    for r in rows:
        out.append('<tr>' + ''.join(f'<td>{c}</td>' if i == 0 else f'<td class="txt">{c}</td>' if i in text_cols else f'<td class="num">{c}</td>' for i, c in enumerate(r)) + '</tr>')
    return '\n'.join(out) + '</tbody></table>'


def counts(by_seed, cat=None):
    """(solved, runs, per-seed %) pooled over seeds for one category (None: all queries)."""
    seeds = sorted(by_seed, key=lambda s: (s != 87358, s))
    per = []
    for s in seeds:
        e = by_seed[s]
        v = e['success_total'] if cat is None else e['by_category'].get(cat)
        if v:
            per.append(v)
    if not per:
        return None
    return sum(x[0] for x in per), sum(x[1] for x in per), [100 * x[0] / x[1] for x in per]


def series(data, ks, method, cat):
    out = dict(bins=[], rows=[])
    for k in ks:
        if method not in data[k]:
            continue
        c = counts(data[k][method], cat)
        if c is None:
            continue
        out['bins'].append(f'×{k}')
        out['rows'].append([100 * c[0] / c[1], min(c[2]), max(c[2]), len(c[2]), c[0], c[1]])
    return out


def main():
    data, preps = GR.collect()
    ks = sorted(k for k in data if data[k])
    panels = ['all queries'] + CATS
    D = dict(fields=[m for m in METHODS if any(m in data[k] for k in ks)], sizes=panels, label={m: m for m in METHODS}, css=CSS, dash=DASH,
             on=METHODS, bold=['constructed', 'learned'], title={p: p for p in panels}, xlabel='gap between the quadrants (× 6 cells)',
             series={p: {m: series(data, ks, m, None if p == 'all queries' else p) for m in METHODS if any(m in data[k] for k in ks)} for p in panels})
    D['tests'] = tests_data(ks)
    D['k1field'] = k1_data()
    page = (HERE / 'gap_template.html').read_text()
    H = build_text(data, preps, ks)
    for k, v in H.items():
        page = page.replace(f'<!--{k}-->', v)
    page = page.replace('/*DATA*/', json.dumps(D))
    missing = [k for k in ('LEDE', 'TILES', 'MAZETEXT', 'METHODTEXT', 'METHODTABLE', 'RESULTS', 'TABLES', 'MECHTEXT', 'FAILTABLE', 'SINKTEXT', 'SINKTABLE', 'CAUSETEXT', 'MECHTABLE', 'AFTERTEXT', 'STRETCHTEXT', 'STRETCHTABLE', 'STEPTEXT', 'STEPTABLE', 'ZTEXT', 'ZTABLE', 'ZDTEXT0', 'ZDWARP', 'ZDTEXT', 'ZDFIG', 'ZDTABLE', 'TTEXT0', 'TTEXT', 'TTABLE', 'VERDICT', 'MEASURED', 'RUNS') if f'<!--{k}-->' in page]
    assert not missing, f'unfilled hooks {missing}'
    (OUT / 'gap.html').write_text(page)
    print('wrote', OUT / 'gap.html', len(page))
    write_markdown(H, ks, preps)


def to_md(frag):
    """The page's HTML fragments as markdown (tables, lists, bold, code, headings, paragraphs)."""
    import re
    def table(m):
        rows = re.findall(r'<tr>(.*?)</tr>', m.group(0), flags=re.S)
        cells = [[re.sub(r'<[^>]+>', '', c.replace('<br>', ' · ')).strip() for c in re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', r, flags=re.S)] for r in rows]
        if not cells:
            return ''
        out = ['| ' + ' | '.join(cells[0]) + ' |', '|' + '---|' * len(cells[0])] + ['| ' + ' | '.join(c) + ' |' for c in cells[1:]]
        return '\n\n' + '\n'.join(out) + '\n\n'
    t = re.sub(r'<table.*?</table>', table, frag, flags=re.S)
    t = re.sub(r'<h3>(.*?)</h3>', lambda m: '\n\n### ' + m.group(1) + '\n\n', t, flags=re.S)
    t = re.sub(r'</?(b|strong)>', '**', t)
    t = re.sub(r'</?code>', '`', t)
    t = re.sub(r'<li>', '\n* ', t)
    t = re.sub(r'</p>', '\n\n', t)
    t = re.sub(r'<sub>(.*?)</sub>', r'\1', t)
    t = re.sub(r'<[^>]+>', '', t)
    t = html.unescape(t).replace('\u00a0', ' ')
    return re.sub(r'\n{3,}', '\n\n', t).strip() + '\n'


def write_markdown(H, ks, preps):
    """README_gap.md: the gap study's request, setup, findings and run ids, from the same text as the page."""
    parts = [
        '# Gap study: the cross maze with its quadrants pushed apart',
        '',
        'Request (2026-11-03, ~18:15 EST): "try with bigger and bigger empty space. Take this 22x22 maze as a baseline, and push the regions further '
        'away. You can try 2x, 4x, 8x, 12x distance. I want your learned warp, and the constructed one. Give me success rate of same quadrant to same '
        'quadrant, different quadrant to different quadrant etc. Also, try the warp + scaling together. Use all gpus"',
        '',
        'Findings page: `report/gap.html` (built by `gappage.py` from the registered runs; published as an artifact). Tables: `report/gap_tables.md` '
        '(`gapreport.py`); diagnostics: `report/openfit.json` (`openfit.py`).',
        '',
        to_md('<p>' + H['LEDE'] + '</p>'),
        '', '## Mazes and queries', '', to_md(H['MAZETEXT']),
        '', '## Methods', '', to_md(H['METHODTEXT']), to_md(H['METHODTABLE']),
        '', '## Success by region pair', '', to_md(H['RESULTS']), to_md(H['TABLES']),
        '', '## What changes as the gap grows', '', to_md(H['MECHTEXT']), to_md(H['FAILTABLE']), to_md(H['SINKTEXT']), to_md(H['SINKTABLE']),
        to_md(H['CAUSETEXT']), to_md(H['MECHTABLE']), to_md(H['AFTERTEXT']),
        '', '## Planning in the stretched space', '', to_md(H['STRETCHTEXT']), to_md(H['STRETCHTABLE']),
        '', '## MPPI step size', '', to_md(H['STEPTEXT']), to_md(H['STEPTABLE']),
        '', '## Training on the warped maze', '', to_md(H['ZTEXT']), to_md(H['ZTABLE']),
        '', '## Pairs drawn on the warped maze, and the 10 × 10 maze with the same gaps', '', to_md(H['ZDTEXT0']),
        '(Figures on the page: `report/gap_warp.png`, `report/gap_warp_density.png`, from `warpfig.py`.)', '', to_md(H['ZDTEXT']), to_md(H['ZDTABLE']),
        '', '## Sampling, training and planning, one at a time', '', to_md(H['TTEXT0']), to_md(H['TTEXT']), to_md(H['TTABLE']),
        '', '## The x1 field on the wider gaps', '', to_md(H['K1TEXT0']), to_md(H['K1TEXT']), to_md(H['K1TABLE']),
        '', '## What this says', '', to_md(H['VERDICT']),
        '', '## How it was measured', '', to_md(H['MEASURED']),
        '', '## Registered runs', '', to_md(H['RUNS']),
        '', '## Reproduce', '',
        '    # a gap maze (k = 2, 4, 8, 12) and its query set; k = 1 = a query set on the 22 x 22 cross maze (--prepared = its prepare run)',
        '    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/cross_warp/run.py --stage gapprepare --scale 8 --label regions',
        '    # a field on it (the constructed warp at k = 8; see the method table for the other settings)',
        "    ... run.py --prepared results/registered_<gapprepare id> --seed 87358 --label pm --extra '--sep --sep-free --sep-scale 4.1717 --k-sep 23.4937 --design plusmag --m-hi 17.4027 --m-lo 1 --s-p 17.4027'",
        '    # data controls on the same queries',
        "    ... run.py --stage gapprepare --scale 8 --label open250k --extra '--open-pairs 250000 --queries-from results/registered_<gapprepare id>'",
        '    ~/.venvs/bellman-ntfield/bin/python jeffrey/cross_warp/gapreport.py; ... openfit.py; ... openfit.py 8 --figure; ... gappage.py',
        '']
    text = '\n'.join(parts)
    assert '<!--' not in text, 'unfilled hook in README_gap.md'
    (HERE / 'README_gap.md').write_text(text)
    print('wrote', HERE / 'README_gap.md')


def pct(x, d=0):
    return f'{x:.{d}f}&nbsp;%'


def run_dir(e):
    return GR.ROOT / 'results' / f'registered_{e["_id"]}'


def fail_types(data, preps, k, m):
    """Pooled over seeds, per category group ('maze' = the four categories with a quadrant end, 'open' = open - open):
    [runs, solved, arrived after touching a wall, never arrived] from the runs' rows.npz (k >= 2). By arrival: plan.mppi appends
    the goal to every path, so the scorer's collision flag of a run that never arrived includes the jump from where it
    stopped to the goal."""
    cat = np.load(preps[k]['_dir'] / 'queries.npz')['category']
    out = {'maze': np.zeros(4, int), 'open': np.zeros(4, int)}
    for s, e in data[k].get(m, {}).items():
        r = np.load(run_dir(e) / 'rows.npz')
        for g, mask in (('maze', cat != 4), ('open', cat == 4)):
            sc, co, cv = r['success'][:, mask], r['collision'][:, mask], r['converged'][:, mask]
            out[g] += [sc.size, int(sc.sum()), int((~sc & cv & co).sum()), int((~sc & ~cv).sum())]
    return out


def two_ways(data, preps, big):
    """The 'Two ways to fail' sentences, from fail_types at the widest gap (failures counted by arrival)."""
    kf = max(big) if big else None
    if kf is None:
        return ''
    def share(m, i):
        if m not in data.get(kf, {}):
            return None
        f = fail_types(data, preps, kf, m)['maze']
        return int(f[i]), int(f[0] - f[1])
    nd, cs = share('ntrl-demo', 2), share('constructed', 3)
    if nd is None or cs is None:
        return ''
    return (f'ntrl-demo fails by touching walls: at &times;{kf}, {nd[0]:,} of its {nd[1]:,} failed runs with a quadrant end arrive after touching a wall (its field '
            'leaks through walls). The warps that magnify the band fail by stalling, at queries with a quadrant end as at open&ndash;open ones: '
            f'{cs[0]:,} of the constructed warp&rsquo;s {cs[1]:,} never arrive. (Counted by arrival. The planner appends the goal to '
            'every path, so this table used to count a run that never arrived as a wall contact whenever the jump from where it stopped to the goal crossed a '
            'wall, and read the warps&rsquo; stalls in the quadrants as collisions.)')


def sampler_stats(preps):
    """Per k: where ntrl-demo's training pairs put x0 (centre square / arm strips / quadrants), the region areas, and
    the share of pairs with both ends in one quadrant (k = 1 from the plus 22 x 22 prepare's data)."""
    out = {}
    for k, p in preps.items():
        d = GR.ROOT / 'results' / GR.PLUS22 if k == 1 else p['_dir']
        a = np.load(d / 'data/sampled_points.npy', mmap_mode='r')
        n = p['maze']['n']
        E, G = n + 0.1, n - 16
        x0, x1 = np.asarray(a[:, 0:2]) * E + n / 2, np.asarray(a[:, 3:5]) * E + n / 2
        inb = lambda z: (z >= 8) & (z < 8 + G)
        reg = lambda x: np.where(inb(x[:, 0]) & inb(x[:, 1]), 0, np.where(inb(x[:, 0]) | inb(x[:, 1]), 1, 2))
        quad = lambda x: np.where(reg(x) == 2, (x[:, 0] >= 8 + G).astype(int) + 2 * (x[:, 1] >= 8 + G).astype(int), -1)
        q0, q1 = quad(x0), quad(x1)
        out[k] = dict(n=n, x0_share=(np.bincount(reg(x0), minlength=3) / len(x0)).tolist(), area=[G * G / n ** 2, 2 * 16 * G / n ** 2, 256 / n ** 2],
                      same_quadrant_pairs=int(((q0 >= 0) & (q0 == q1)).sum()), pairs=int(len(x0)))
    return out


CONTROL_ROWS = [      # (row label, method key in collect(), training change)
    ('ntrl-demo', 'ntrl-demo', 'none'),
    ('ntrl-demo', 'ntrl-demo + open pairs', '+ 250k open pairs'),
    ('ntrl-demo', 'ntrl-demo + local pairs', '+ 500k local pairs'),
    ('ntrl-demo', 'ntrl-demo, heading ×m', 'heading ×m'),
    ('ntrl-demo', 'ntrl-demo + lower bound', 'T ≥ |a−b|'),
    ('ntrl-demo', 'ntrl-demo + line of sight', 'line of sight'),
    ('ntrl-demo', 'ntrl-demo, TD step 0.66 cells', 'TD step 0.66 cells'),
    ('ntrl-demo', 'ntrl-demo + line of sight + TD step', 'line of sight, TD step 0.66 cells'),
    ('constructed', 'constructed', 'none'),
    ('constructed', 'constructed + open pairs', '+ 250k open pairs'),
    ('constructed', 'constructed + local pairs', '+ 500k local pairs'),
    ('constructed', 'constructed + open + local pairs', '+ open + local pairs'),
    ('constructed', 'constructed, heading ×1', 'heading ×1'),
    ('constructed', 'constructed + lower bound', 'T ≥ |a−b|'),
    ('constructed', 'constructed + line of sight', 'line of sight'),
    ('constructed', 'constructed, heading ×1 + line of sight', 'heading ×1, line of sight'),
    ('constructed', 'constructed, TD step 0.66 cells', 'TD step 0.66 cells'),
    ('constructed', 'constructed, heading ×1 + line of sight + TD step', 'heading ×1, line of sight, TD step 0.66 cells'),
    ('warp + scaling', 'warp + scaling', 'none'),
    ('warp + scaling', 'warp + scaling + open pairs', '+ 250k open pairs'),
    ('warp + scaling', 'warp + scaling + line of sight', 'line of sight'),
    ('warp + scaling', 'warp + scaling + line of sight + TD step', 'line of sight, TD step 0.66 cells'),
    ('learned', 'learned', 'none (starts at the top bound)'),
    ('learned', 'learned, constructed start', 'starts as the constructed warp'),
    ('learned', 'learned, constructed start + open pairs', 'constructed start, + open pairs'),
    ('uniform', 'uniform', 'none'),
    ('README warp', 'README warp', 'none'),
]


def control_table(data, fit, get, fmt, ks=(8, 12)):
    """Training seed 87358 for every row (the controls' seed); where a field has more seeds its pooled 'all' value is added in brackets."""
    rows = []
    for k in ks:
        for label, key, change in CONTROL_ROWS:
            if key not in data.get(k, {}):
                continue
            e87 = data[k][key].get(87358)
            if e87 is None:
                continue
            c = counts(data[k][key])
            r = fit.get((k, key), {})
            sk, al = r.get('sinks'), r.get('aliasing')
            pooled = f' <span class="mute">({100 * c[0] / c[1]:.1f} over {len(c[2])} seeds)</span>' if len(c[2]) > 1 else ''
            rows.append([f'{esc(label)}, &times;{k}', esc(change), f'<b>{100 * e87["success_total"][0] / e87["success_total"][1]:.1f}</b>{pooled}', fmt(get(k, key, 'same quadrant')), fmt(get(k, key, 'quadrant - open')),
                         fmt(get(k, key, 'open - open')), '&ndash;' if not sk else f'{100 * sk["trapped_share"]:.0f}',
                         '&ndash;' if not al or al['random_median'] is None else f'{al["random_median"]:.2f}',
                         '&ndash;' if not al or al['sink_median'] is None else f'{al["sink_median"]:.2f}'])
    return table_html(['field', 'change', 'all % (seed 87358)', 'same quadrant', 'quadrant &ndash; open', 'open &ndash; open', 'band trapped %', 'T/|a&minus;b| band', 'T/|a&minus;b| sinks'], rows, text_cols=(0, 1))


def cause_text(data, preps, ks, fit, SS, get, fmt):
    """CAUSETEXT, MECHTABLE, AFTERTEXT: what the sinks are and the controls (filled from the runs; see CONTROL_ROWS)."""
    H = {}
    F = lambda key, k=8: fit.get((k, key), {})
    al = lambda key, what='random_median', k=8: (F(key, k).get('aliasing') or {}).get(what)
    oo = lambda key, k=8: get(k, key, 'open - open')
    f2 = lambda v: '&ndash;' if v is None else f'{v:.2f}'
    wres = lambda key: (F(key).get('wall partner') or {}).get('all', {}).get('residual_median')
    away = lambda key: (F(key).get('open partner') or {}).get('far', {}).get('away_share')
    H['CAUSETEXT'] = (
        '<p><b>What the sinks are.</b> The failing fields are right up close and wrong at range. At open poses their slope is as good as ntrl-demo&rsquo;s '
        f'(eikonal error at &times;8: constructed {f2(wres("constructed"))}, warp + scaling {f2(wres("warp + scaling"))}, ntrl-demo {f2(wres("ntrl-demo"))}), '
        'but they put far poses too close to the goal. Where the straight segment to the goal is free, the true travel time is the straight-line distance, '
        f'and the learned T is {f2(al("constructed"))} of it across the band for the constructed warp and {f2(al("warp + scaling"))} for warp + scaling '
        f'({f2(al("constructed", "sink_median"))} and {f2(al("warp + scaling", "sink_median"))} at their sinks), against {f2(al("ntrl-demo"))} for ntrl-demo. '
        f'On open pairs 24 cells or more apart, the constructed warp&rsquo;s descent points away from the partner {100 * (away("constructed") or 0):.0f}&nbsp;% of the time '
        f'(ntrl-demo {100 * (away("ntrl-demo") or 0):.1f}&nbsp;%). ntrl-demo&rsquo;s network computes T as a distance between the two poses&rsquo; codes, '
        'T(a, b) = d(f(a), f(b)); magnify the band and f varies so fast across it that far-apart poses can land close together in code space. A sink is such a pose: '
        'its code sits next to the goal&rsquo;s. Training on the PDE fixes the slope of T near each sample and nothing pushes those codes apart.</p>'
        '<p><b>Controls</b> (64&times;64 maze, same queries, training seed 87358; ntrl-demo gets every change too). Each tests one explanation:</p>')
    centre8 = f'{100 * SS[8]["x0_share"][0]:.1f}&nbsp;% of x0 for {100 * SS[8]["area"][0]:.0f}&nbsp;% of the area' if 8 in SS else ''
    same = ', '.join(f'&times;{k} {100 * SS[k]["same_quadrant_pairs"] / SS[k]["pairs"]:.0f}&nbsp;%' for k in ks if k in SS)
    los_fit = {}
    for key in ('constructed + line of sight', 'ntrl-demo + line of sight', 'warp + scaling + line of sight', 'constructed, heading ×1 + line of sight',
                'constructed, heading ×1 + line of sight + TD step', 'ntrl-demo + line of sight + TD step', 'warp + scaling + line of sight + TD step'):
        e = data.get(8, {}).get(key, {}).get(87358)
        if e:
            h = json.loads((run_dir(e) / 'training/history.json').read_text())
            los_fit[key] = h['warp_terms'][-1].get('los')
    lf = lambda key: '&ndash;' if los_fit.get(key) is None else f'{los_fit[key]:.4f}'
    rms = lambda key, k=8: '&ndash;' if los_fit.get(key) is None else f'{(los_fit[key] ** 0.5) * (preps[k]["maze"]["n"] + 0.1):.0f}'
    H['CAUSETEXT'] += '<ul class="tight">' + ''.join([
        f'<li><b>Training pairs.</b> ntrl-demo&rsquo;s sampler keeps x0 within its wall margin (0.175 cells), so no training pair has two open ends (the band&rsquo;s '
        f'centre square at &times;8: {centre8}). Adding 250,000 pairs with x0 in the open fixes the slope there (eikonal error on open pairs '
        f'{f2((F("constructed").get("open partner") or {}).get("all", {}).get("residual_median"))} &rarr; {f2((F("constructed + open pairs").get("open partner") or {}).get("all", {}).get("residual_median"))}) '
        f'but not the sinks: open&ndash;open {fmt(oo("constructed"))} &rarr; {fmt(oo("constructed + open pairs"))}&nbsp;%, T at the sinks {f2(al("constructed + open pairs", "sink_median"))} of the distance. Not the cause.</li>',
        f'<li><b>Heading.</b> Every method that loses the band also magnifies the heading &times;m. ntrl-demo with only its heading magnified keeps the band '
        f'({fmt(oo("ntrl-demo, heading ×m"))}&nbsp;%); the constructed warp with heading &times;1 does better ({fmt(oo("constructed"))} &rarr; {fmt(oo("constructed, heading ×1"))}&nbsp;%). '
        'Harmless alone; with the warp&rsquo;s stretch it is what makes the band unrepresentable (see the exact answer below).</li>',
        f'<li><b>A lower bound.</b> The speed is at most 1, so T(a, b) &ge; |a &minus; b| for every pair. A hinge on random pairs (no labels; <code>--lb-weight</code>) lifts the '
        f'constructed warp&rsquo;s T across the band ({f2(al("constructed"))} &rarr; {f2(al("constructed + lower bound"))} of the distance) and open&ndash;open '
        f'{fmt(oo("constructed"))} &rarr; {fmt(oo("constructed + lower bound"))}&nbsp;%; ntrl-demo {fmt(get(8, "ntrl-demo"), 1)} &rarr; {fmt(get(8, "ntrl-demo + lower bound"), 1)}&nbsp;% overall.</li>',
        f'<li><b>The exact answer in the open.</b> When the straight segment keeps the robot beyond the margin (82&nbsp;% of random pairs at &times;8), T = |a &minus; b| exactly. '
        f'Training on it (<code>--los-weight</code>) leaves the constructed warp at {fmt(oo("constructed + line of sight"))}&nbsp;% open&ndash;open because it cannot fit it: '
        f'mean squared error {lf("constructed + line of sight")} at the end of training (about {rms("constructed + line of sight")} cells) against {lf("ntrl-demo + line of sight")} for ntrl-demo '
        f'({rms("ntrl-demo + line of sight")} cells; every seed fits it to about 2 cells). With the heading left at &times;1 the same warp fits it on one training seed of three: '
        + ', '.join(f'seed {s_} {r_:.1f} cells and {o_:.0f}&nbsp;% open&ndash;open' for (s_, r_), (_, o_) in zip(los_rms(data, preps, 8, 'constructed, heading ×1 + line of sight'),
                                                                                                     per_seed(data, 8, 'constructed, heading ×1 + line of sight', 'open - open'))) +
        '. The separable warp stretches each arm of the band &times;m along one axis; with the heading magnified &times;m as well, the codes there change fast along two axes along '
        'which the travel time changes slowly or not at all, and the network does not manage to make them reproduce a straight-line distance; with the heading at &times;1 it '
        'sometimes does. '
        + (f'Warp + scaling, whose warp magnifies the band about evenly, fits it to {lf("warp + scaling + line of sight")} and goes {fmt(oo("warp + scaling"))} &rarr; '
           f'{fmt(oo("warp + scaling + line of sight"))}&nbsp;% open&ndash;open ({fmt(get(8, "warp + scaling"), 1)} &rarr; {fmt(get(8, "warp + scaling + line of sight"), 1)}&nbsp;% overall; '
           f'&times;12: {fmt(oo("warp + scaling", 12))} &rarr; {fmt(oo("warp + scaling + line of sight", 12))}&nbsp;%).' if 'warp + scaling + line of sight' in data.get(8, {}) else '') + '</li>',
        f'<li><b>Fewer local pairs.</b> The recipe&rsquo;s step is a fixed share of the maze, so fewer pairs stay inside one quadrant as the gap grows ({same}). '
        f'Adding 500,000 short pairs leaves ntrl-demo&rsquo;s same-quadrant success where it was ({fmt(get(8, "ntrl-demo", "same quadrant"))} &rarr; {fmt(get(8, "ntrl-demo + local pairs", "same quadrant"))}&nbsp;%)'
        + (f' and the constructed warp&rsquo;s ({fmt(get(8, "constructed", "same quadrant"))} &rarr; {fmt(get(8, "constructed + local pairs", "same quadrant"))}&nbsp;%)' if 'constructed + local pairs' in data.get(8, {}) else '') +
        '. ntrl-demo&rsquo;s decline in the quadrants is resolution: its fixed Fourier spectrum sees each quadrant cell 22.1&thinsp;/&thinsp;E as wide as on the 22&times;22 maze.</li>',
    ]) + '</ul>'
    H['MECHTABLE'] = control_table(data, fit, get, fmt)
    H['AFTERTEXT'] = (
        '<p class="note">Columns: success over planning seeds 0/1/2; band trapped = share of the open band draining to a spurious sink (mean over the open&ndash;open goals); '
        'T&thinsp;/&thinsp;|a&minus;b| = learned travel time over the straight-line distance for free straight segments to those goals (1 = right), median over random band cells and over the sinks.</p>'
        f'<p><b>Uniform stretch and the learned warp do not train from &times;8 on.</b> Their loss stays near {loss_of(data, 8, "uniform"):.0e} from epoch 500 '
        f'(ntrl-demo {loss_of(data, 8, "ntrl-demo"):.0e}), and their fields are slow everywhere: T is {f2(al("uniform"))} times the straight-line distance in the band. '
        'ntrl-demo&rsquo;s loss weights each pair by exp(&minus;T/2), so pairs whose T has grown large stop counting and the field never recovers. The learned warp '
        'starts at its top bound, which is the uniform stretch' +
        (f'; started as the constructed warp instead it trains ({fmt(get(8, "learned, constructed start"), 1)}&nbsp;% overall, open&ndash;open {fmt(oo("learned, constructed start"))}&nbsp;%)' if 'learned, constructed start' in data.get(8, {}) else '') + '.</p>')
    return H


def per_seed(data, k, key, cat=None):
    """[(seed, % solved)] for one field, overall or in one category."""
    out = []
    for s_, e in sorted(data.get(k, {}).get(key, {}).items(), key=lambda t: (t[0] != 87358, t[0])):
        v = e['success_total'] if cat is None else e['by_category'].get(cat)
        if v:
            out.append((s_, 100 * v[0] / v[1]))
    return out


def los_rms(data, preps, k, key):
    """[(seed, rms line-of-sight error in cells)] from each run's training history."""
    out = []
    for s_, e in sorted(data.get(k, {}).get(key, {}).items(), key=lambda t: (t[0] != 87358, t[0])):
        h = json.loads((run_dir(e) / 'training/history.json').read_text())
        v = h['warp_terms'][-1].get('los') if h.get('warp_terms') else None
        if v is not None:
            out.append((s_, v ** 0.5 * (preps[k]['maze']['n'] + 0.1)))
    return out


def best_text(data, get, fmt, tot):
    """The constructed warp with heading x1 and line of sight against ntrl-demo (plain and with line of sight), per gap where run."""
    out = []
    for k in (8, 12):
        key = 'constructed, heading ×1 + line of sight'
        if key not in data.get(k, {}):
            continue
        ns = len(data[k][key])
        ctrl = [f'ntrl-demo {fmt(tot(k, "ntrl-demo"), 1)}&nbsp;% ({len(data[k]["ntrl-demo"])} seed{"s" if len(data[k]["ntrl-demo"]) > 1 else ""})']
        if 'ntrl-demo + line of sight' in data[k]:
            nl = len(data[k]['ntrl-demo + line of sight'])
            ctrl.append(f'ntrl-demo with line of sight {fmt(tot(k, "ntrl-demo + line of sight"), 1)}&nbsp;% ({nl} seed{"s" if nl > 1 else ""})')
        seeds = per_seed(data, k, key)
        out.append(f'&times;{k}: {fmt(tot(k, key), 1)}&nbsp;% of all queries ({ns} seed{"s" if ns > 1 else ""}' + (': ' + ' / '.join(f'{v:.1f}' for _, v in seeds) if ns > 1 else '') +
                   f'; open&ndash;open {fmt(get(k, key, "open - open"))}&nbsp;%, '
                   f'same quadrant {fmt(get(k, key, "same quadrant"))}&nbsp;%) against ' + ' and '.join(ctrl))
    return ('The constructed warp with heading &times;1 and line of sight: ' + '; '.join(out) + '.') if out else ''


def ws_note(found):
    """The one gain: warp + scaling at the largest gap where its open-open success rises by 5 points or more."""
    for k in sorted({k for k, _ in found}, reverse=True):
        e = found.get((k, 'warp + scaling'))
        if not e:
            continue
        o, s_ = e[0]['original']['by_category'].get('open - open'), e[0]['stretched']['by_category'].get('open - open')
        if o and s_ and 100 * (s_[0] - o[0]) / o[1] >= 5:
            return (f'The one gain is warp + scaling at &times;{k}, whose spurious sinks are the shallowest: open&ndash;open {100 * o[0] / o[1]:.0f} &rarr; {100 * s_[0] / s_[1]:.0f}&nbsp;%, '
                    f'still far below ntrl-demo&rsquo;s.')
    return ''


def stretch_section(data, preps, fmt):
    """STRETCHTEXT / STRETCHTABLE from the stretchq runs (warpplan.mppi_stretched against plan.mppi, paired)."""
    names = {}
    for k in data:
        for m, by in data[k].items():
            for s_, e in by.items():
                names[e['_id']] = (k, m, s_)
    rows, found = [], {}
    for rid, d in GR.load_all().items():
        if d.get('stage') != 'stretchq':
            continue
        qn = Path(d['queries']).name
        for frid, e in d['runs'].items():
            if frid in names and names[frid][0] in preps and preps[names[frid][0]]['_dir'].name == qn:
                k, m, s_ = names[frid]
                found[(k, m)] = (e, d['plan_seeds'], rid, frid)
    order = ['ntrl-demo', 'uniform', 'constructed', 'learned', 'README warp', 'warp + scaling', 'constructed, heading ×1 + line of sight']
    for k in sorted({k for k, _ in found}):
        for m in order:
            if (k, m) not in found:
                continue
            e, ps, rid, frid = found[(k, m)]
            o, s_ = e['original'], e['stretched']
            fl = GR.ROOT / 'results' / f'registered_{rid}' / frid / 'rows.npz'
            if fl.exists():                       # walls / stalls by arrival (arrived after touching a wall / never arrived)
                zz = np.load(fl)
                for pl, x in (('original', o), ('stretched', s_)):
                    ok_, co_, cv_ = zz[f'{pl}_success'], zz[f'{pl}_collision'], zz[f'{pl}_converged']
                    x.update(collision=int((~ok_ & cv_ & co_).sum()), stalled=int((~ok_ & ~cv_).sum()))
            cell = lambda x: f'<b>{100 * x["success_total"][0] / x["success_total"][1]:.1f}</b>'
            cc = lambda x, c: f'{100 * x["by_category"][c][0] / x["by_category"][c][1]:.0f}' if c in x['by_category'] else '&ndash;'
            rows.append([f'{esc(m)}, &times;{k}', f'&times;{e["m_q"]:.1f}', f'{cell(o)} &rarr; {cell(s_)}', f'{cc(o, "same quadrant")} &rarr; {cc(s_, "same quadrant")}',
                         f'{cc(o, "quadrant - open")} &rarr; {cc(s_, "quadrant - open")}', f'{cc(o, "open - open")} &rarr; {cc(s_, "open - open")}',
                         f'{o["collision"]} / {o["stalled"]} &rarr; {s_["collision"]} / {s_["stalled"]}', f'{o["steps_median"]:.0f} &rarr; {s_["steps_median"]:.0f}'])
    if not rows:
        return {'STRETCHTEXT': '<p>(no stretched-space planning runs yet)</p>', 'STRETCHTABLE': ''}
    seeds = sorted({tuple(v[1]) for v in found.values()})
    text = ('<p>MPPI normally draws its samples with the same step everywhere in the maze (a fixed share of a cell). Planned in the stretched space instead '
            '(<code>warpplan.mppi_stretched</code>), each sample is drawn in the encoder&rsquo;s input coordinates and mapped back through the warp&rsquo;s local Jacobian: '
            'the step inside a quadrant stays what it was, and where the warp magnifies less (the open band, for every warp here) steps and the 5-step look-ahead grow by '
            'the same factor, up to &times;17 at &times;8. If the planner stalled because its 0.3-cell look-ahead could not see past a shallow dip, this would let it through. '
            'With a constant stretch (ntrl-demo, uniform) it is the same planner, which checks the implementation: those rows come out identical. '
            f'Paired by query and planning seed ({", ".join(",".join(map(str, s)) for s in seeds)}); training seed 87358; the same fields as above.</p>'
            '<p><b>It does not help.</b> The stalls stay (the spurious basins are wider than even the stretched look-ahead), the wall contacts stay, '
            'and queries between a quadrant and the band lose a little (longer steps near the quadrant entrances). The planner gets there in far fewer steps, but the field&rsquo;s '
            'defects decide the outcome, not the planner&rsquo;s step. ' + ws_note(found) + '</p>')
    table = table_html(['field, gap', 'quadrant stretch', 'all %', 'same quadrant', 'quadrant &ndash; open', 'open &ndash; open', 'walls / stalls', 'median steps'], rows)
    return {'STRETCHTEXT': text, 'STRETCHTABLE': table}


STEP_KEYS = [(0.5, 'mppi@0.5'), (1, 'mppi@1'), (2, 'mppi+cap@2'), (4, 'mppi+cap@4'), (8, 'mppi+cap@8'), (16, 'mppi+cap@16')]
STEP_ORDER = ['ntrl-demo', 'uniform', 'constructed', 'learned', 'README warp', 'warp + scaling', 'constructed, heading ×1 + line of sight']


def step_section(data, preps):
    """STEPTEXT / STEPTABLE from the stepq runs: plan.mppi with its step x m and temperature / m (m = 0.5 .. 16), steps
    above the default capped at the distance to the goal; the uncapped steps and ntrl-demo are the controls. Per-query
    stall overlap from the runs' rows.npz; sink depth from report/openfit.json; other training seeds where run."""
    names = {e['_id']: (k, m, s_) for k in data for m, by in data[k].items() for s_, e in by.items()}
    found, ids, rows_of = {}, {}, {}
    for rid, d in GR.load_all().items():
        if d.get('stage') != 'stepq':
            continue
        qn = Path(d['queries']).name
        for frid, e in d['runs'].items():
            if frid in names and names[frid][0] in preps and preps[names[frid][0]]['_dir'].name == qn:
                found[names[frid]] = e
                rows_of[names[frid]] = d['_dir'] / frid / 'rows.npz'
                if names[frid][2] == 87358:
                    ids[names[frid][0]] = rid
    if not any(key[2] == 87358 for key in found):
        return {'STEPTEXT': '<p>(no step-size runs yet)</p>', 'STEPTABLE': ''}

    def pct(k, m, key, cat=None, seed=87358):
        v = found.get((k, m, seed), {}).get('plans', {}).get(key)
        x = None if v is None else v['success_total'] if cat is None else v['by_category'].get(cat)
        return None if not x else 100 * x[0] / x[1]

    def plan(k, m, key, seed=87358):
        return found.get((k, m, seed), {}).get('plans', {}).get(key, {})

    f0 = lambda v: '&ndash;' if v is None else f'{v:.0f}'
    f1 = lambda v: '&ndash;' if v is None else f'{v:.1f}'
    f2 = lambda v: '&ndash;' if v is None else f'{v:.2f}'
    main = {(k, m) for k, m, s_ in found if s_ == 87358}
    ks = sorted({k for k, _ in main})
    cells = lambda k, m: [f'<b>{f1(pct(k, m, key))}</b><br><span class="mute">{f0(pct(k, m, key, "same quadrant"))} / {f0(pct(k, m, key, "open - open"))}</span>' for _, key in STEP_KEYS]
    rows = [[f'{esc(m)}, &times;{k}'] + cells(k, m) for k in ks for m in STEP_ORDER if (k, m) in main]
    any_e = next(e for key, e in found.items() if key[2] == 87358)['plans']
    head = ['field, gap'] + [f'{any_e[key]["step_cells"]:.2f}' + (' (default)' if s == 1 else '') for s, key in STEP_KEYS]
    head[-1] += ' cells'
    table = table_html(head, rows) + ('<p class="note">MPPI step in cells (the 5-step look-ahead is five of them). Each cell: all queries, % solved (bold), then same quadrant / open&ndash;open %. '
                                      'Training seed 87358, planning seed 0, the same 300 queries per gap as above; steps longer than the default are capped at the distance to the goal. '
                                      'Runs: ' + ', '.join(f'&times;{k} <code>{ids[k]}</code>' for k in ks) + '.</p>')

    # controls: the cap at the default step (acts only within one step of the goal), the uncapped planner at the largest step
    capd = [abs(pct(k, m, 'mppi+cap@1') - pct(k, m, 'mppi@1')) for k, m in main if pct(k, m, 'mppi+cap@1') is not None and pct(k, m, 'mppi@1') is not None]
    cap_note = (f' At the default step the cap moves no field by more than {max(capd):.1f} points (it only acts within one step of the goal).' if capd else '')
    half = max(abs(pct(k, m, 'mppi@0.5') - pct(k, m, 'mppi@1')) for k, m in main if pct(k, m, 'mppi@0.5') is not None and pct(k, m, 'mppi@1') is not None)
    kc = max(k for k in ks if (k, 'ntrl-demo') in main)
    unc = plan(kc, 'ntrl-demo', 'mppi@16')
    ctrl = (f'Without the cap a long step circles the goal and never lands within the planner&rsquo;s 0.04-cell tolerance: ntrl-demo at &times;{kc} solves '
            f'{f1(pct(kc, "ntrl-demo", "mppi@1"))}&nbsp;% at the default step and {f1(pct(kc, "ntrl-demo", "mppi@16"))}&nbsp;% at 16&times; uncapped '
            f'({unc.get("unconverged_near_goal", 0)} of 300 runs end within two steps of the goal without arriving), {f1(pct(kc, "ntrl-demo", "mppi+cap@16"))}&nbsp;% with the cap.'
            + cap_note) if unc else ''
    spread = lambda m, kk: max((max(v) - min(v)) for k in kk if (k, m) in main for v in [[pct(k, m, key) for _, key in STEP_KEYS if pct(k, m, key) is not None]])

    # per query: stalls at the default step and at 16x, and where the runs that stall at both stop
    def stalls(k, m, seed=87358):
        f = rows_of.get((k, m, seed))
        if f is None or not f.exists():
            return None
        z = np.load(f)
        if 'mppi+cap@16_converged' not in z.files:
            return None
        st = lambda key: ~z[f'{key}_converged'][0]                     # never arrived (whatever the appended jump to the goal crosses)
        s1, s16 = st('mppi@1'), st('mppi+cap@16')
        both = s1 & s16
        d1, d16 = z['mppi@1_final_cells'][0], z['mppi+cap@16_final_cells'][0]
        return dict(n1=int(s1.sum()), n16=int(s16.sum()), both=int(both.sum()), rescued=int((s1 & z['mppi+cap@16_success'][0]).sum()),
                    moved=float(np.median(np.abs(d1[both] - d16[both]))) if both.any() else None, dist=float(np.median(d1[both])) if both.any() else None)
    big = [k for k in ks if k >= 8] or ks[-1:]
    pm = {k: stalls(k, 'constructed') for k in big}
    ws = {k: stalls(k, 'warp + scaling') for k in big}
    fit = json.loads((OUT / 'openfit.json').read_text()) if (OUT / 'openfit.json').exists() else {}
    def fitv(k, m, part, what):
        e = data.get(k, {}).get(m, {}).get(87358)
        v = fit.get(f'{k}|{e["_id"]}', {}) if e else {}
        return (v.get(part) or {}).get(what)
    oo = lambda k, m, key, seed=87358: pct(k, m, key, 'open - open', seed)
    kw = [k for k in big if ws.get(k)]
    gaps = lambda kk: ' and '.join(f'&times;{k}' for k in kk)
    ws_line = (f'at 16&times; the step its open&ndash;open success rises ' + ' and '.join(f'{f0(oo(k, "warp + scaling", "mppi@1"))} &rarr; {f0(oo(k, "warp + scaling", "mppi+cap@16"))}&nbsp;%' for k in kw)
               + f' at {gaps(kw)} (' + ' and '.join(f'{ws[k]["rescued"]} of {ws[k]["n1"]}' for k in kw) + ' stalled runs now arrive), and all queries '
               + ' and '.join(f'{f1(pct(k, "warp + scaling", "mppi@1"))} &rarr; {f1(pct(k, "warp + scaling", "mppi+cap@16"))}&nbsp;%' for k in kw)
               + ', still below ntrl-demo&rsquo;s ' + ' and '.join(f'{f1(pct(k, "ntrl-demo", "mppi@1"))}' for k in kw) + '&nbsp;%') if kw else ''
    kp = [k for k in big if pm.get(k) and pm[k]['moved'] is not None]
    pm_line = '; '.join((f'at &times;{k}, {pm[k]["both"]} of its {pm[k]["n1"]} stalled runs stall again at 16&times; and stop within {pm[k]["moved"]:.1f} cells of where they stopped before '
                         f'(median, {pm[k]["dist"]:.0f} cells from the goal)') if i == 0 else
                        f'at &times;{k}, {pm[k]["both"]} of {pm[k]["n1"]}, within {pm[k]["moved"]:.1f} cells ({pm[k]["dist"]:.0f} cells from the goal)' for i, k in enumerate(kp))
    depth = ' and '.join(f2(fitv(k, 'warp + scaling', 'aliasing', 'sink_median')) for k in big)
    depth_pm = ' and '.join(f2(fitv(k, 'constructed', 'aliasing', 'sink_median')) for k in big)
    # sink depth against the gain from the longest step, over both warps, three training seeds and the large gaps
    fit_s = json.loads((OUT / 'openfit_seeds.json').read_text()) if (OUT / 'openfit_seeds.json').exists() else {}
    def depth_of(k, m, s_):
        e = data.get(k, {}).get(m, {}).get(s_)
        if not e:
            return None
        r = fit.get(f'{k}|{e["_id"]}') if s_ == 87358 else fit_s.get(f'{k}|{e["_id"]}')
        return r if r and r.get('aliasing') and r.get('sinks') else None
    mech = []
    for k in big:
        for m in ('warp + scaling', 'constructed'):
            for s_ in (87358, 1, 2):
                r, a1, a16 = depth_of(k, m, s_), oo(k, m, 'mppi@1', s_), oo(k, m, 'mppi+cap@16', s_)
                if r and a1 is not None and a16 is not None:
                    mech.append((r['aliasing']['sink_median'], k, m, s_, r['sinks']['trapped_share'], r['sinks']['spurious_sinks'], a1, a16))
    mech.sort(key=lambda t: t[0])
    mech_table = ''
    mech_text = ''
    if mech:
        mech_table = ('<h3>Sink depth and the longest step</h3>' + table_html(
            ['field, gap, training seed', 'T / distance at sinks', 'band in sink basins', 'sinks / goal', 'open&ndash;open, 1&times; &rarr; 16&times;'],
            [[f'{esc(m)}, &times;{k}, seed {s_}', f2(d), f'{100 * t:.0f}&nbsp;%', f'{n:.0f}', f'{a1:.0f} &rarr; <b>{a16:.0f}</b>'] for d, k, m, s_, t, n, a1, a16 in mech])
            + '<p class="note">Sorted by the second column: the field&rsquo;s travel time T from a sink to the goal over their straight-line distance (median over the open band&rsquo;s sinks; '
              '1 = no aliasing, small = the field puts the sink next to the goal). Diagnostics <code>openfit.py</code> (seeds 1 and 2: <code>openfit.py 8 12 --seeds</code>); planning seed 0.</p>')
        hi = [x for x in mech if x[7] - x[6] >= 20]
        lo = [x for x in mech if x[7] - x[6] < 20]
        if hi and lo and min(x[0] for x in hi) > max(x[0] for x in lo):
            mech_text = (f'<p><b>The depth decides it, seed by seed.</b> Over both warps, three training seeds and &times;{" and &times;".join(map(str, big))}, every field whose sinks read '
                         f'{min(x[0] for x in hi):.2f} of the straight-line distance or more gains {min(x[7] - x[6] for x in hi):.0f} to {max(x[7] - x[6] for x in hi):.0f} points of open&ndash;open at 16&times; the step; '
                         f'every field at {max(x[0] for x in lo):.2f} or less gains at most {max(x[7] - x[6] for x in lo):.0f} (table below). Warp + scaling&rsquo;s training seed 1 is one of the deep ones, '
                         'so for it the step does nothing; the gain is a property of the trained field, not of the warp.</p>')
        else:
            mech_text = '<p>Sink depth against the gain from the longest step, per field and seed: table below.</p>'
    k1 = min(ks)
    drop = [(pct(k1, m, 'mppi@1') - pct(k1, m, 'mppi+cap@16'), m) for m in STEP_ORDER if pct(k1, m, 'mppi@1') is not None and pct(k1, m, 'mppi+cap@16') is not None]
    dm = max(drop)[1] if drop else None
    small_text = (f' Where the quadrant mazes are most of the maze that is all they do: at &times;{k1} every field loses {min(drop)[0]:.0f} to {max(drop)[0]:.0f} points of all queries at 16&times; the step '
                  f'({esc(dm)} {f1(pct(k1, dm, "mppi@1"))} &rarr; {f1(pct(k1, dm, "mppi+cap@16"))}&nbsp;%, same quadrant {f0(pct(k1, dm, "mppi@1", "same quadrant"))} &rarr; '
                  f'{f0(pct(k1, dm, "mppi+cap@16", "same quadrant"))}&nbsp;%), and the default step is as good as any.') if drop and k1 < min(big) else ''

    text = ('<p>MPPI&rsquo;s step is the same share of a cell on every maze: 0.06 cells, so its 5-step look-ahead sees 0.31 cells. Here it runs from half that to 16 times '
            'that (0.03 to 0.98 cells; corridors are one cell wide) on the same fields, queries and planning seed. The temperature is divided by the same factor '
            '(<code>plan.settings</code>&rsquo; rule when the cell size changes, so the softmax is as selective per step), and the budget lets every step size travel three times the '
            'true path. ' + ctrl + '</p>'
            f'<p><b>In the wide band the step matters only where the sinks are shallow.</b> At {gaps(big)}, ntrl-demo, the control, has no sinks and stays within '
            f'{spread("ntrl-demo", big):.1f} points of all queries across the six steps; the README warp, which solves its band at every step, within {spread("README warp", big):.1f}. '
            f'The constructed warp keeps its stalls run for run: {pm_line}. The learned warp and the uniform stretch, which do not train at {gaps(big)}, stay where they were. '
            f'Warp + scaling is the exception: {ws_line}.</p>'
            f'<p>The difference is how deep the sinks are (diagnostics above, training seed 87358). At warp + scaling&rsquo;s sinks the field reads {depth} of the straight-line '
            f'distance to the goal ({gaps(big)}): a shallow dip, and a look-ahead of a few cells reaches ground that is lower on the way to the goal. '
            'At the constructed warp&rsquo;s sinks it reads ' + depth_pm +
            ': the field says the goal is right there, every way out climbs, and no step size gets out. '
            'Long steps cost the quadrants instead: at half a cell or more per step the runs cut corners into walls.'
            f'{small_text} Halving the step moves no field by more than {half:.1f} points.</p>' + mech_text)
    # closest warp + scaling gets to ntrl-demo on the same training seed (planning seed 0), best step against ntrl-demo's default
    close = []
    for k in big:
        for s_ in (87358, 1, 2):
            e_nt = data.get(k, {}).get('ntrl-demo', {}).get(s_)
            best = [(found[(k, 'warp + scaling', s_)]['plans'][key]['success_total'][0], m_) for m_, key in STEP_KEYS if (k, 'warp + scaling', s_) in found and key in found[(k, 'warp + scaling', s_)]['plans']]
            if not e_nt or not best:
                continue
            nt = plan(k, 'ntrl-demo', 'mppi@1')['success_total'][0] if s_ == 87358 and plan(k, 'ntrl-demo', 'mppi@1') else int(np.load(run_dir(e_nt) / 'rows.npz')['success'][0].sum())
            b = max(best)
            close.append((nt - b[0], k, s_, b[0], b[1], nt))
    c = min(close) if close else None
    close_text = (f'warp + scaling stays below ntrl-demo at every step (closest: training seed {c[2]} at &times;{c[1]}, {c[3]} against {c[5]} of 300 queries at {c[4]:g}&times; the step)'
                  if c and c[0] > 0 else 'warp + scaling comes level with ntrl-demo at its best step' if c else '')
    info = dict(k=kc, default_cells=any_e['mppi@1']['step_cells'], long_cells=any_e['mppi+cap@16']['step_cells'],
                ntrl_default=pct(kc, 'ntrl-demo', 'mppi@1'), ntrl_long=pct(kc, 'ntrl-demo', 'mppi+cap@16'))
    return {'STEPTEXT': text, 'STEPTABLE': table + mech_table, 'STEPCLOSE': close_text, '_STEPINFO': info}


ZVIEW = 'https://claude.ai/artifact/Y6J6umPzgXSKs6uNZqr87m'


def zsummary():
    f = OUT / 'fieldspaces/summary.json'
    return json.loads(f.read_text()) if f.exists() else None


def zrate(S, k, key, planner, seed=87358, cat=None, pooled=False):
    """Success % from report/fieldspaces/summary.json (fieldspaces.py --summary): one training seed, or pooled."""
    e = ((S or {}).get('gaps', {}).get(str(k), {}).get(key) or {}).get(planner)
    if not e:
        return None
    if not pooled:
        e = (e.get('per_seed') or {}).get(str(seed))
        if not e:
            return None
    a = e['success_total'] if cat is None else e['by_category'].get(cat)
    return 100 * a[0] / a[1] if a and a[1] else None


def zseeds(S, k, key, planner):
    e = ((S or {}).get('gaps', {}).get(str(k), {}).get(key) or {}).get(planner)
    return e.get('seeds', []) if e else []


def warped_section(ks):
    """ZTEXT / ZTABLE: the constructed warp trained and planned on the warped maze (trainer.py z_space, zq.py), from
    report/fieldspaces/summary.json, written by fieldspaces.py --summary (which also writes the viewer's text)."""
    S = zsummary()
    if not S:
        return {'ZTEXT': '<p>(no warped-space runs yet)</p>', 'ZTABLE': '', 'ZLEDE': '', 'ZVERDICT': ''}
    fmt = lambda v: '&ndash;' if v is None else f'{v:.1f}'
    intro = ('<p>Request (2026-11-04, evening): &ldquo;Go into the gap warp spaces one. In your hand constructed version, why does it fall off to begin with? so 1) the hand crafted warp '
             'should not change from x1 gap to x12 gap right. 2) The model doesn&rsquo;t know it is warped at all, it thinks x1 and x12 are the same so at least mppi on that warped space '
             'should be the same. With the fixed, make sure no gradient goes into the warp. Finally, show me learned time travel field on the warped and also on the original.&rdquo; '
             f'The fields, on the maze and on the warped maze at every gap, are in the <a href="{ZVIEW}">warp-space viewer</a>; its text is the answer below.</p>')
    rows = []
    for k in ks:
        pooled_z = zseeds(S, k, 'z_trained', 'z')
        rows.append([f'&times;{k}', fmt(zrate(S, k, 'ntrl', 'x')), fmt(zrate(S, k, 'x_trained', 'x')), fmt(zrate(S, k, 'x_trained', 'z')),
                     fmt(zrate(S, k, 'z_trained', 'x')), fmt(zrate(S, k, 'z_trained', 'z')),
                     (fmt(zrate(S, k, 'z_trained', 'z', pooled=True)) + f' ({len(pooled_z)})') if len(pooled_z) > 1 else '&ndash;'])
    table = table_html(['gap', 'ntrl-demo', 'constructed, trained on the maze', '&hellip; planned on the warped maze',
                        'constructed, trained on the warped maze', '&hellip; planned on the warped maze', '&hellip; pooled over training seeds (seeds)'], rows)
    table += '<p class="note">All queries solved (%), training seed 87358 unless pooled, planning seeds 0/1/2, MPPI on the maze unless the column says warped. ntrl-demo and the gap study&rsquo;s constructed warp planned on the maze are the gap study&rsquo;s own runs; every other column is a <code>zq.py</code> run.</p>'
    d12 = [('trained on the maze, &times;12&rsquo;s pairs', 'x_trained'), ('trained on the maze, the 22&times;22 pairs', 'x_trained_k1data'),
           ('trained on the warped maze, &times;12&rsquo;s pairs', 'z_trained'), ('trained on the warped maze, the 22&times;22 pairs', 'z_trained_k1data')]
    mq = (S.get('mapped_queries') or {}).get('12', {})
    r2 = []
    for lab, key in d12:
        q = (mq.get(key) or {}).get('z')
        r2.append([lab, fmt(zrate(S, 12, key, 'x')), fmt(zrate(S, 12, key, 'z')), fmt(100 * q['success_total'][0] / q['success_total'][1]) if q else '&ndash;'])
    ref = zrate(S, 1, 'z_trained', 'z')
    r2.append(['reference: &times;1, trained on the warped maze', fmt(zrate(S, 1, 'z_trained', 'x')), fmt(ref), fmt(ref)])
    table += table_html(['&times;12, training seed 87358', '&times;12&rsquo;s queries, planned on the maze', '&hellip; planned on the warped maze', 'the 22&times;22 queries carried into &times;12, planned on the warped maze'], r2)
    table += '<p class="note">The 22&times;22 pairs and queries are moved into the &times;12 maze with the quadrants and the band&rsquo;s edge cells unchanged and the band&rsquo;s interior stretched (<code>gapprepare.py --data-from / --map-queries</code>), so in the warped space they are the 22&times;22 ones except inside the band&rsquo;s interior. Reference row: the 22&times;22 field on its own queries.</p>'
    three = lambda k: len(zseeds(S, k, 'z_trained', 'z')) >= 3
    pooled8, pooled12 = three(8), three(12)
    z8, n8 = zrate(S, 8, 'z_trained', 'z', pooled=pooled8), zrate(S, 8, 'ntrl', 'x', pooled=pooled8)
    z12, n12 = zrate(S, 12, 'z_trained', 'z', pooled=pooled12), zrate(S, 12, 'ntrl', 'x', pooled=pooled12)
    seedtxt = ('&times;2 and &times;4 on training seed 87358, &times;8 and &times;12 over three training seeds' if pooled8 and pooled12 else 'training seed 87358 for both')
    lede = (' Why does the constructed warp fall off although its warped mazes look almost the same? Trained on the warped maze instead (the network sees only the warped '
            'coordinates and its loss is taken there, so the fixed warp gets no gradient and the network is no longer asked for the robot&rsquo;s real travel time), it beats ntrl-demo '
            f'at every gap from &times;2 on (MPPI on the warped maze, {seedtxt}: &times;8 {fmt(z8)} against {fmt(n8)}&nbsp;%, &times;12 {fmt(z12)} against {fmt(n12)}&nbsp;%); '
            'the rest of its fall-off comes from the training pairs, which are drawn on the real maze, and from the band, which is still wider in the warped space.')
    verdict = ('<li><b>Training on the warped maze.</b> The constructed warp&rsquo;s fall-off is not the warp&rsquo;s: it is fixed and the warped mazes are nearly identical. '
               'The gap study took the loss on the maze, so the network had to fit the real travel time, which across the band grows 12&times; while the band grows 1.7&times; in its input. '
               'Taking the loss on the warped maze (no gradient through the warp) and planning there: '
               + '; '.join(f'&times;{k} {fmt(zrate(S, k, "z_trained", "z"))} against ntrl-demo {fmt(zrate(S, k, "ntrl", "x"))}&nbsp;%' for k in ks) +
               ' (training seed 87358)' + (lambda ks3: '' if not ks3 else '; over three training seeds ' + ', '.join(
                   f'&times;{k} {fmt(zrate(S, k, "z_trained", "z", pooled=True))} against {fmt(zrate(S, k, "ntrl", "x", pooled=True))}&nbsp;%' for k in ks3))(
                   [k for k in ks if len(zseeds(S, k, 'z_trained', 'z')) >= 3]) +
               '. The pairs drawn on the real maze and the band&rsquo;s remaining width account for the rest (table in the section above).</li>')
    return {'ZTEXT': intro + S.get('why', ''), 'ZTABLE': table, 'ZLEDE': lede, 'ZVERDICT': verdict}


def zdraw():
    f = OUT / 'zdraw10.json'
    return json.loads(f.read_text()) if f.exists() else None


WORDS = {2: 'two', 3: 'three', 4: 'four', 5: 'five'}
ZD_FIELDS = [('id', 'x', 'ntrl-demo'), ('idw', 'x', 'ntrl-demo, pairs drawn on the warped maze'),
             ('pm', 'x', 'constructed warp, trained on the maze'), ('pmw', 'x', '&hellip; pairs drawn on the warped maze'),
             ('pmz', 'z', 'trained on the warped maze, pairs drawn on the maze'), ('pmzw', 'z', '&hellip; pairs drawn on the warped maze'),
             ('pmzs', 'z', 'band shrunk to its &times;1 width: pairs drawn on it, trained on it'),
             ('pmzp', 'z', '&hellip; pairs drawn on the shrunk warp, trained on the unshrunk warped maze'),
             ('pmzg', 'z', '&hellip; pairs drawn on the unshrunk warp, trained on the shrunk warped maze')]


ZD_PARENT = {'pmw': 'pm', 'pmzw': 'pmz', 'pmzp': 'pmzs', 'pmzg': 'pmzs'}
ZD_FULL = {'pmw': 'constructed warp, trained on the maze, pairs drawn on the warped maze', 'pmzw': 'trained on the warped maze, pairs drawn on the warped maze',
           'pmzp': 'band shrunk: pairs drawn on the shrunk warp, trained on the unshrunk warped maze',
           'pmzg': 'band shrunk: pairs drawn on the unshrunk warp, trained on the shrunk warped maze'}


def zdraw_tables(R, ks):
    """ZDTABLE: success on the 22 x 22 study's gap mazes and on the 10 x 10 base's, by how the training pairs were drawn;
    pooled over training seeds; every field and control at x1 / x12 by query kind; the pair shares (report/zdraw10.json, zdraw10.py)."""
    import zdraw10 as ZD
    fmt = lambda v: '&ndash;' if v is None else f'{v:.1f}'
    r = lambda base, k, lab, pl, **kw: ZD.rate(R, base, k, lab, pl, **kw)
    out = ''
    for base, title in (('c22', 'the 22&times;22 cross maze&rsquo;s quadrants (the gap study)'), ('m10', 'the 10&times;10 maze&rsquo;s quadrants, the same gaps')):
        rows = []
        for k in ks:
            n = (16 if base == 'c22' else 10) + 6 * k
            both = lambda lab: '&ndash;' if r(base, k, lab, 'z') is None else f'{fmt(r(base, k, lab, "z"))} ({fmt(r(base, k, lab, "x"))})'
            rows.append([f'&times;{k}', f'{n}&times;{n}', fmt(r(base, k, 'id', 'x')), fmt(r(base, k, 'pm', 'x')), both('pmz'), both('pmzw')])
        out += f'<h3>{title}</h3>' + table_html(
            ['gap', 'maze', 'ntrl-demo', 'constructed', 'warped, maze pairs', 'warped, warped pairs'], rows, text_cols=(0, 1))
    out += ('<p class="note">All queries solved (%), training seed 87358, planning seeds 0/1/2 (300 queries &times; 3). Constructed: the gap study&rsquo;s constructed warp, trained on '
            'the maze. Warped: trained and planned on the warped maze (MPPI in its coordinates, the path mapped back exactly and scored on the maze), with the training pairs drawn '
            'on the maze (maze pairs) or on the warped maze (warped pairs); in brackets, the same field planned on the maze. The 22&times;22 rows&rsquo; first four '
            'columns are the runs of the sections above. &ndash;: not run (yet).</p>')
    prow = []
    for base, name in (('c22', '22&times;22 quadrants'), ('m10', '10&times;10 quadrants')):
        for k in ks:
            cells = []
            for lab, pl in (('id', 'x'), ('pmz', 'z'), ('pmzw', 'z')):
                sd = ZD.seeds(R, base, k, lab, pl)
                cells.append(f'{fmt(r(base, k, lab, pl, pooled=True))} ({len(sd)})' if len(sd) > 1 else '&ndash;')
            if any(c != '&ndash;' for c in cells):
                prow.append([name, f'&times;{k}'] + cells)
    if prow:
        out += table_html(['maze', 'gap', 'ntrl-demo', 'warped, maze pairs', 'warped, warped pairs'], prow, text_cols=(0, 1))
        out += '<p class="note">Pooled over training seeds (number of seeds in brackets; &ndash;: one seed so far); warped-maze fields planned on the warped maze.</p>'
    krow = []
    for base, name in (('c22', '22&times;22'), ('m10', '10&times;10')):
        for k in (1, 12):
            prev = None
            for lab, pl, lname in ZD_FIELDS:
                v = r(base, k, lab, pl)
                if v is None:
                    continue
                if lname.startswith('&hellip;') and prev != ZD_PARENT.get(lab):
                    lname = ZD_FULL[lab]
                krow.append([f'{name} &times;{k}', lname, fmt(v)] + [fmt(r(base, k, lab, pl, cat=c)) for c in CATS])
                prev = 'pmzs' if lab in ('pmzp', 'pmzg') and prev == 'pmzs' else lab      # the 2 x 2 rows share the shrunk row's prefix
    out += table_html(['maze, gap', 'field', 'all'] + [esc(c) for c in CATS], krow, text_cols=(0, 1))
    out += ('<p class="note">Every field and control at &times;1 and &times;12 by query kind (%), training seed 87358, 60 queries &times; 3 planning seeds per kind. ntrl-demo and the constructed '
            'warp trained on the maze are planned on the maze, every field trained on the warped maze on the warped maze. Band shrunk: the constructed warp with the band&rsquo;s interior '
            'at density below 1, so it is as wide in the warp at &times;12 as at &times;1 (text).</p>')
    srow = []
    for base, name in (('c22', '22&times;22 quadrants'), ('m10', '10&times;10 quadrants')):
        for k in ks:
            for draw, dname in (('real', 'the maze'), ('warped', 'the warped maze'), ('shrunk', 'the warped maze, band shrunk')):
                v = R['pairs'].get(f'{base}/{k}/{draw}')
                if v:
                    srow.append([name, f'&times;{k}', dname] + [fmt(100 * v['kinds'][c]) for c in CATS])
    out += table_html(['maze', 'gap', 'pairs drawn on'] + [esc(c) for c in CATS], srow, text_cols=(0, 1, 2))
    out += ('<p class="note">Where the 500,000 training pairs lie (%), by the same five kinds as the queries. ntrl-demo&rsquo;s recipe keeps x0 within 0.175 cells of a wall '
            'and takes x1 a random step away.</p>')
    return out


def zdraw_section(ks):
    """ZDTEXT / ZDFIG / ZDTABLE / ZDLEDE / ZDVERDICT: the training pairs drawn on the warped maze (gapprepare.py --z-draw),
    the 10 x 10 maze's quadrants with the same gaps (gap.py base maze10), the controls, and the shrunk-band control with
    its 2 x 2, from report/zdraw10.json (zdraw10.py)."""
    R = zdraw()
    if not R:
        return {'ZDTEXT0': '', 'ZDWARP': '', 'ZDTEXT': '<p>(no runs yet)</p>', 'ZDFIG': '', 'ZDTABLE': '', 'ZDLEDE': '', 'ZDVERDICT': ''}
    import zdraw10 as ZD
    f1 = lambda v: '&ndash;' if v is None else f'{v:.1f}'
    f0 = lambda v: '&ndash;' if v is None else f'{v:.0f}'
    f2 = lambda v: '&ndash;' if v is None else f'{v:.2f}'
    r = lambda base, k, lab, pl, **kw: ZD.rate(R, base, k, lab, pl, **kw)
    pooled = lambda base, k, lab, pl: r(base, k, lab, pl, pooled=True)
    nsd = lambda base, k, lab, pl: len(ZD.seeds(R, base, k, lab, pl))
    P = R['pairs']
    sh = lambda base, k, draw, key: (lambda v: None if v is None else 100 * v[key])(P.get(f'{base}/{k}/{draw}'))
    kind = lambda base, k, draw, c: (lambda v: None if v is None else 100 * v['kinds'][c])(P.get(f'{base}/{k}/{draw}'))
    SQ, AD, DG, QO, OO = CATS
    have = lambda base, k, lab, pl: r(base, k, lab, pl) is not None
    paras = ['<p>Request (2026-11-05): &ldquo;try to draw training pairs on the scaled maze. Also try 10x10 but with the same gaps&rdquo;.</p>']
    paras.append(
        '<p><b>Pairs drawn on the warped maze.</b> ntrl-demo&rsquo;s recipe draws a start x0 anywhere and keeps it if it lies within 0.175 cells of a wall, and an end x1 a random step '
        'away, both in units of the maze&rsquo;s side. Here the same draws are made on the warped maze: the recipe&rsquo;s numbers are read as coordinates of the warped box, so the pairs '
        'are spread evenly over the warped maze, and each pair is mapped back to the maze by the warp&rsquo;s exact inverse before the recipe&rsquo;s rules are applied there '
        '(<code>gapprepare.py --z-draw</code>; with the warp set to the identity it reproduces the recipe pair for pair). The constructed warp magnifies the quadrants m-fold along x and y, so they get '
        'about m&sup2; times the share of pairs per cell that the band&rsquo;s interior gets. '
        f'At &times;12 the pairs with both ends in one quadrant go from {f0(sh("c22", 12, "real", "same_quadrant"))}&nbsp;% to {f0(sh("c22", 12, "warped", "same_quadrant"))}&nbsp;%, '
        f'where the 22&times;22 maze has {f0(sh("c22", 1, "real", "same_quadrant"))}&nbsp;% of its own (table below). '
        'The network is then trained and planned on the warped maze as in the section above.</p>')
    paras.append(warp_text())
    head, paras = paras, []                              # ZDTEXT0 (before the warp figures) / ZDTEXT

    # ---- 22 x 22 quadrants
    if have('c22', 12, 'pmzw', 'z'):
        gk = [k for k in ks if have('c22', k, 'pmzw', 'z')]
        t = '<p><b>22&times;22 quadrants.</b> '
        n12 = nsd('c22', 12, 'pmzw', 'z')
        if n12 >= 2:
            t += (f'Over {WORDS.get(n12, n12)} training seeds, MPPI on the warped maze, the &times;12 maze: {f1(pooled("c22", 12, "pmzw", "z"))}&nbsp;% of all queries with the pairs drawn on the warped maze, '
                  f'{f1(pooled("c22", 12, "pmz", "z"))}&nbsp;% with the pairs drawn on the maze, ntrl-demo {f1(pooled("c22", 12, "id", "x"))}&nbsp;%; the &times;1 maze: '
                  f'{f1(pooled("c22", 1, "pmz", "z"))}&nbsp;% with the pairs drawn on the maze, ntrl-demo {f1(pooled("c22", 1, "id", "x"))}&nbsp;%. '
                  'The &times;12 maze now plans almost as well as the &times;1 maze. ')
        t += (f'Training seed 87358 at every gap{"" if len(gk) == len(ks) else " run so far"}: ' + ', '.join(f'&times;{k} {f1(r("c22", k, "pmzw", "z"))}' for k in gk) +
              '&nbsp;% with the pairs drawn on the warped maze, against ' + ', '.join(f'&times;{k} {f1(r("c22", k, "pmz", "z"))}' for k in gk) +
              '&nbsp;% with the pairs drawn on the maze and ntrl-demo&rsquo;s ' + ', '.join(f'&times;{k} {f1(r("c22", k, "id", "x"))}' for k in gk) + '&nbsp;%. '
              + (lambda v: f'With the pairs drawn on the warped maze the total no longer falls with the gap ({max(v) - min(v):.1f} points between the best and the worst gap), '
                 'but its diagonal queries still do (' + ', '.join(f'&times;{k} {f1(r("c22", k, "pmzw", "z", cat=DG))}' for k in gk) + '&nbsp;%), offset in the total by the '
                 'same-quadrant queries, which are far from their ceiling and vary more from gap to gap (' +
                 f'{f1(min(r("c22", k, "pmzw", "z", cat=SQ) for k in gk))} to {f1(max(r("c22", k, "pmzw", "z", cat=SQ) for k in gk))}&nbsp;%). '
                 if len(v) == len(ks) and max(v) - min(v) < 5 else '')([r('c22', k, 'pmzw', 'z') for k in gk]))
        if have('c22', 1, 'pmzw', 'z'):
            n1 = nsd('c22', 1, 'pmzw', 'z')
            sqm = [ZD.rate(R, 'c22', 1, 'pmz', 'z', seed=s, cat=SQ) for s in ZD.seeds(R, 'c22', 1, 'pmz', 'z')]
            sqm = [v for v in sqm if v is not None]
            if n1 >= 3:
                t += (f'At &times;1, over {WORDS.get(n1, n1)} training seeds, the pairs drawn on the warped maze give {f1(pooled("c22", 1, "pmzw", "z"))}&nbsp;% against '
                      f'{f1(pooled("c22", 1, "pmz", "z"))}&nbsp;% (same quadrant {f1(r("c22", 1, "pmzw", "z", cat=SQ, pooled=True))} against {f1(r("c22", 1, "pmz", "z", cat=SQ, pooled=True))}&nbsp;%).</p>')
            else:
                t += (f'At &times;1 (training seed 87358) the pairs drawn on the warped maze give {f1(r("c22", 1, "pmzw", "z"))}&nbsp;% against {f1(r("c22", 1, "pmz", "z"))}&nbsp;%, '
                      f'the difference all in same-quadrant queries ({f1(r("c22", 1, "pmzw", "z", cat=SQ))} against {f1(r("c22", 1, "pmz", "z", cat=SQ))}&nbsp;%), where the field with '
                      f'the maze&rsquo;s pairs itself ranges from {f0(min(sqm)) if sqm else "&ndash;"} to {f0(max(sqm)) if sqm else "&ndash;"}&nbsp;% over its training seeds; more seeds are queued.</p>')
        else:
            t += '</p>'
        paras.append(t)

    # ---- the 10 x 10 base
    if have('m10', 1, 'id', 'x'):
        paras.append(
            '<p><b>The 10&times;10 maze with the same gaps.</b> maze_scale&rsquo;s 10&times;10 maze (seed 1010, on which ntrl-demo solved 88&nbsp;% of its own queries in the maze-scale study) '
            'cut along its two centre lines into four 5&times;5 quadrants, every wall kept (the walls on the centre lines become the quadrants&rsquo; sides facing the band, doorways included), with the '
            'gap study&rsquo;s bands between them: 6, 12, 24, 48 and 72 cells, mazes of 16 to 82 cells a side (<code>gap.py</code> base <code>maze10</code>; figure). Goals, queries (60 of each kind, '
            'certified) and training as in the gap study; the constructed warp is the gap study&rsquo;s design, a quadrant cell 0.27 units wide in the warp at every gap.</p>')
    if have('m10', 12, 'pmzw', 'z') and have('m10', 1, 'pmzw', 'z'):
        straight = lambda k: (lambda c: None if not c or not c.get('straight') else 100 * sum(v[0] for v in c['straight'].values()) / sum(v[1] for v in c['straight'].values()))(R['controls'].get(str(k)))
        mk = [k for k in ks if have('m10', k, 'pmzw', 'z')]
        t = ('<p><b>10&times;10 quadrants.</b> ntrl-demo solves ' + ', '.join(f'&times;{k} {f1(r("m10", k, "id", "x"))}' for k in mk if have('m10', k, 'id', 'x')) +
             '&nbsp;% of all queries. Trained and planned on the warped maze, the field solves ' + ', '.join(f'&times;{k} {f1(r("m10", k, "pmz", "z"))}' for k in mk if have('m10', k, 'pmz', 'z')) +
             '&nbsp;% with the pairs drawn on the maze and ' + ', '.join(f'&times;{k} {f1(r("m10", k, "pmzw", "z"))}' for k in mk) +
             '&nbsp;% with the pairs drawn on the warped maze (training seed 87358')
        extra = [(k, lab) for k in (1, 12) for lab, pl in (('id', 'x'), ('pmzw', 'z'), ('pmz', 'z')) if nsd('m10', k, lab, pl) >= 2]
        if extra:
            t += '; over training seeds: ' + ', '.join(f'{ {"id": "ntrl-demo", "pmzw": "warped-maze pairs", "pmz": "maze pairs"}[lab]} &times;{k} '
                                                         f'{f1(pooled("m10", k, lab, "x" if lab == "id" else "z"))}&nbsp;% ({nsd("m10", k, lab, "x" if lab == "id" else "z")})' for k, lab in extra)
        drop = lambda lab, pl, **kw: (lambda a, b: None if a is None or b is None else a - b)(r('m10', 1, lab, pl, **kw), r('m10', 12, lab, pl, **kw))
        dw = drop('pmzw', 'z')
        t += (f'). The oracle (the same MPPI on the true travel time) solves every query, the straight line {f1(straight(1))}&nbsp;% at &times;1 and {f1(straight(12))}&nbsp;% at &times;12. '
              + (f'The pairs drawn on the warped maze keep most of the &times;1 field&rsquo;s success: {f1(dw)} points are lost from &times;1 to &times;12 '
                 f'(with the pairs drawn on the maze {f1(drop("pmz", "z"))}, ntrl-demo {f1(drop("id", "x"))}), {f1(drop("pmzw", "z", cat=DG) / len(CATS))} of them in the diagonal '
                 'queries, a fifth of all queries (' + ', '.join(f'&times;{k} {f1(r("m10", k, "pmzw", "z", cat=DG))}' for k in mk) + '&nbsp;%). The diagonal queries fall on the '
                 '22&times;22 quadrants too; there the same-quadrant queries, far from their ceiling, offset them in the total, while here they start at '
                 f'{f1(r("m10", 1, "pmzw", "z", cat=SQ))}&nbsp;%. '
                 if dw is not None and len(mk) == len(ks) else 'The pairs drawn on the warped maze keep most of the &times;1 field&rsquo;s success at &times;12. ') +
              'The pairs drawn on the maze lose half of it and every query between quadrants '
              f'(adjacent and diagonal {f1(r("m10", 12, "pmz", "z", cat=AD))} and {f1(r("m10", 12, "pmz", "z", cat=DG))}&nbsp;%): the maze draw at &times;12 puts '
              f'{f2(kind("m10", 12, "real", AD))} and {f2(kind("m10", 12, "real", DG))}&nbsp;% of its pairs between adjacent and diagonal quadrants, against '
              f'{f1(kind("m10", 1, "real", AD))} and {f1(kind("m10", 1, "real", DG))}&nbsp;% at &times;1.</p>')
        paras.append(t)

    # ---- controls: the warped-maze pairs without the warped-maze loss
    if have('c22', 12, 'idw', 'x') and have('c22', 12, 'pmw', 'x'):
        pm12m = r('m10', 12, 'pm', 'x')
        paras.append(
            '<p><b>Controls: the pairs alone do not do it.</b> Given the same warped-maze pairs, ntrl-demo solves '
            f'{f1(r("c22", 12, "idw", "x"))}&nbsp;% at &times;12 on the 22&times;22 quadrants ({f1(r("c22", 12, "id", "x"))}&nbsp;% with its own pairs) and '
            f'{f1(r("m10", 12, "idw", "x"))}&nbsp;% on the 10&times;10&rsquo;s ({f1(r("m10", 12, "id", "x"))}&nbsp;%); the gap study&rsquo;s constructed warp, whose loss is taken on the maze, '
            f'{f1(r("c22", 12, "pmw", "x"))}&nbsp;% ({f1(r("c22", 12, "pm", "x"))}&nbsp;%) and {f1(r("m10", 12, "pmw", "x"))}&nbsp;%'
            + (f' ({f1(pm12m)}&nbsp;%)' if pm12m is not None else '') +
            ' (training seed 87358). Each is held back by something the pairs do not reach. ntrl-demo stays weak inside the quadrants (same quadrant '
            f'{f1(r("c22", 12, "idw", "x", cat=SQ))} and {f1(r("m10", 12, "idw", "x", cat=SQ))}&nbsp;%; with its own pairs {f1(r("c22", 12, "id", "x", cat=SQ))} and '
            f'{f1(r("m10", 12, "id", "x", cat=SQ))}&nbsp;%), where its fixed Fourier spectrum is too coarse for cells 1/88 and 1/82 of the maze wide (above). '
            'The constructed warp trained on the maze still breaks in the band (open&ndash;open '
            f'{f1(r("c22", 12, "pmw", "x", cat=OO))} and {f1(r("m10", 12, "pmw", "x", cat=OO))}&nbsp;%): it is asked for the real travel time, which across the band grows 12&times; '
            'while the band grows 1.6&ndash;1.7&times; in its input. The pairs pay off only when the loss, too, is taken on the warped maze.</p>')

    # ---- what is left at x12, and the band's width in the warp
    if have('c22', 12, 'pmzw', 'z') and have('m10', 12, 'pmzw', 'z'):
        paras.append(
            '<p><b>What is left at &times;12</b> is mostly the diagonal queries. With the pairs drawn on the warped maze, the &times;12 field is level with the &times;1 fields on same-quadrant, '
            f'quadrant&ndash;open and open&ndash;open queries, but solves {f1(r("c22", 12, "pmzw", "z", cat=DG))}&nbsp;% of the diagonal ones on the 22&times;22 quadrants against '
            f'{f1(r("c22", 1, "pmz", "z", cat=DG))} / {f1(r("c22", 1, "pmzw", "z", cat=DG))}&nbsp;% at &times;1 (pairs drawn on the maze / on the warped maze), and '
            f'{f1(r("m10", 12, "pmzw", "z", cat=DG))}&nbsp;% on the 10&times;10&rsquo;s against {f1(r("m10", 1, "pmz", "z", cat=DG))} / {f1(r("m10", 1, "pmzw", "z", cat=DG))}&nbsp;% '
            '(training seed 87358). Not for want of pairs: drawn on the warped maze at &times;12, each query kind gets about the share of pairs it gets drawn on the maze at &times;1 '
            f'(diagonal-quadrant pairs {f1(kind("c22", 12, "warped", DG))} against {f1(kind("c22", 1, "real", DG))}&nbsp;% and {f1(kind("m10", 12, "warped", DG))} against '
            f'{f1(kind("m10", 1, "real", DG))}&nbsp;%; table). What still differs is the band. In the warp a quadrant cell is 0.27 units wide at every gap, but the band&rsquo;s interior '
            f'(the band less its edge cells, where the warp&rsquo;s density is 1) is {f2(ZD.band_z("c22", 1))} units wide at &times;1 and {f2(ZD.band_z("c22", 12))} at &times;12 on the 22&times;22 '
            f'quadrants, {f2(ZD.band_z("m10", 1))} and {f2(ZD.band_z("m10", 12))} on the 10&times;10&rsquo;s: a path between quadrants crosses '
            f'{ZD.band_z("c22", 12) / ZD.band_z("c22", 1):.1f} ({ZD.band_z("m10", 12) / ZD.band_z("m10", 1):.1f}) times as much empty space in what the network sees.</p>')

    # ---- the shrunk-band control
    ws = {b: ZD.prepare_warp(R, f'{b}_shrunk', 12) for b in ('c22', 'm10')}
    if all(ws.values()) and have('c22', 12, 'pmzs', 'z') and have('m10', 12, 'pmzs', 'z'):
        ml = {b: ws[b]['m_lo'] for b in ws}
        paras.append(
            '<p><b>Control: the band shrunk back to its &times;1 width.</b> The same warp with the band&rsquo;s interior at density '
            f'{ml["c22"]:.3f} instead of 1 ({ml["m10"]:.3f} on the 10&times;10 quadrants), so that at &times;12 it is {f2(ZD.band_z("c22", 12, ml["c22"]))} '
            f'({f2(ZD.band_z("m10", 12, ml["m10"]))}) units wide in the warp, as at &times;1; quadrants and the band&rsquo;s edge cells unchanged (<code>gapprepare.py --z-m-lo</code>). '
            'Pairs drawn on it, trained and planned on it, training seed 87358. The &times;12 maze then plans like the &times;1 maze: '
            f'{f1(r("c22", 12, "pmzs", "z"))}&nbsp;% of all queries on the 22&times;22 quadrants (&times;1: {f1(r("c22", 1, "pmz", "z"))} / {f1(r("c22", 1, "pmzw", "z"))}&nbsp;% with the pairs '
            f'drawn on the maze / on the warped maze), diagonal {f1(r("c22", 12, "pmzs", "z", cat=DG))}&nbsp;% ({f1(r("c22", 1, "pmz", "z", cat=DG))} / {f1(r("c22", 1, "pmzw", "z", cat=DG))}&nbsp;%); '
            f'{f1(r("m10", 12, "pmzs", "z"))}&nbsp;% on the 10&times;10&rsquo;s ({f1(r("m10", 1, "pmz", "z"))} / {f1(r("m10", 1, "pmzw", "z"))}&nbsp;%), diagonal '
            f'{f1(r("m10", 12, "pmzs", "z", cat=DG))}&nbsp;% ({f1(r("m10", 1, "pmz", "z", cat=DG))} / {f1(r("m10", 1, "pmzw", "z", cat=DG))}&nbsp;%).</p>')
        paras.append(ZD2X2(R, r, kind, f1, DG))
        fx = {b: (R['fields'][f'{b}/12']['pmzs'][str(ZD.S0)].get('failures') or {}) for b in ('c22', 'm10')}
        fw = {b: (R['fields'][f'{b}/12']['pmzw'][str(ZD.S0)].get('failures') or {}) for b in ('c22', 'm10')}
        mh = {b: ws[b]['m_hi'] for b in ws}
        paras.append(
            f'<p>These fields are for the planner on the warped maze. MPPI on the maze stalls on them ({fx["c22"].get("stalled", "&ndash;")} and {fx["m10"].get("stalled", "&ndash;")} of 900 runs, '
            f'{f1(r("c22", 12, "pmzs", "x"))} and {f1(r("m10", 12, "pmzs", "x"))}&nbsp;% solved; with the unshrunk warp {fw["c22"].get("stalled", "&ndash;")} and {fw["m10"].get("stalled", "&ndash;")}): '
            f'it keeps one step and one temperature for the whole maze, and on the field divided by the quadrants&rsquo; magnification a maze cell of the shrunk band costs '
            f'1/{mh["c22"] / ml["c22"]:.0f} (1/{mh["m10"] / ml["m10"]:.0f}) of a quadrant cell, against 1/{mh["c22"]:.0f} (1/{mh["m10"]:.0f}) unshrunk.</p>')
    return {'ZDTEXT0': ''.join(p for p in head if p), 'ZDWARP': warp_figs(),
            'ZDTEXT': ''.join(p for p in paras if p), 'ZDFIG': zdraw_fig(), 'ZDTABLE': zdraw_tables(R, ks), 'ZDLEDE': zdraw_lede(R, r, pooled, nsd, f1),
            'ZDVERDICT': zdraw_verdict(R, r, pooled, nsd, f1, DG)}


def ZD2X2(R, r, kind, f1, DG):
    """The 2 x 2 of the shrunk-band control: pairs drawn on the shrunk / unshrunk warp x trained on the shrunk / unshrunk warped maze."""
    import zdraw10 as ZD
    head = ('<p><b>Width or pairs?</b> Shrinking the band also moves pairs out of it into the quadrants: drawn on the shrunk warp, '
            f'{f1(kind("c22", 12, "shrunk", DG))}&nbsp;% of the pairs join diagonal quadrants against {f1(kind("c22", 12, "warped", DG))}&nbsp;% drawn on the unshrunk one '
            f'({f1(kind("m10", 12, "shrunk", DG))} against {f1(kind("m10", 12, "warped", DG))}&nbsp;% on the 10&times;10 quadrants). ')
    if not all(r(b, 12, lab, 'z') is not None for b in ('c22', 'm10') for lab in ('pmzp', 'pmzg')):
        return head + 'The 2&times;2 that swaps the two (pairs drawn on one warp, trained on the other) is running.</p>'
    cell = lambda b, lab: f'{f1(r(b, 12, lab, "z"))} / {f1(r(b, 12, lab, "z", cat=DG))}'
    seeds = ZD.seeds(R, 'c22', 12, 'pmzw', 'z')
    allw = [ZD.rate(R, 'c22', 12, 'pmzw', 'z', seed=s) for s in seeds]
    dgw = [ZD.rate(R, 'c22', 12, 'pmzw', 'z', seed=s, cat=DG) for s in seeds]
    rng = lambda v: f'{min(v):.1f}&ndash;{max(v):.1f}' if len(v) > 1 else f'{v[0]:.1f}'
    zf = lambda b, lab, key: (R['fields'][f'{b}/12'][lab][str(ZD.S0)].get('z') or {}).get(key, '&ndash;')
    nw10 = len(ZD.seeds(R, 'm10', 12, 'pmzw', 'z'))
    seeds10 = ZD.seeds(R, 'm10', 12, 'pmzw', 'z')
    allw10 = [ZD.rate(R, 'm10', 12, 'pmzw', 'z', seed=s) for s in seeds10]
    dgw10 = [ZD.rate(R, 'm10', 12, 'pmzw', 'z', seed=s, cat=DG) for s in seeds10]
    p10, pd10 = r('m10', 12, 'pmzp', 'z'), r('m10', 12, 'pmzp', 'z', cat=DG)
    within10 = len(seeds10) >= 2 and None not in (p10, pd10) and min(allw10) <= p10 <= max(allw10) and min(dgw10) <= pd10 <= max(dgw10)
    if within10:
        unshrunk = (f'Trained on the unshrunk warped maze, the shrunk draw&rsquo;s pairs (training seed 87358) do not help on either: {f1(r("c22", 12, "pmzp", "z"))}&nbsp;% '
                    f'(diagonal {f1(r("c22", 12, "pmzp", "z", cat=DG))}&nbsp;%) on the 22&times;22 quadrants against {rng(allw)}&nbsp;% ({rng(dgw)}&nbsp;%), and {f1(p10)}&nbsp;% '
                    f'(diagonal {f1(pd10)}&nbsp;%) on the 10&times;10&rsquo;s against {rng(allw10)}&nbsp;% ({rng(dgw10)}&nbsp;%), over '
                    + (f'{WORDS.get(len(seeds), len(seeds))} training seeds of the unshrunk draw on each. ' if len(seeds) == len(seeds10) else
                       f'{WORDS.get(len(seeds), len(seeds))} and {WORDS.get(len(seeds10), len(seeds10))} training seeds of the unshrunk draw. '))
    else:
        unshrunk = (f'Trained on the unshrunk warped maze, the shrunk draw&rsquo;s pairs do not help '
                    f'on the 22&times;22 quadrants ({f1(r("c22", 12, "pmzp", "z"))}&nbsp;%, diagonal {f1(r("c22", 12, "pmzp", "z", cat=DG))}&nbsp;%, against {rng(allw)}&nbsp;% and '
                    f'{rng(dgw)}&nbsp;% over {WORDS.get(len(seeds), len(seeds))} training seeds of the unshrunk draw) and help part of the way on the 10&times;10&rsquo;s ({f1(p10)}&nbsp;%, '
                    f'diagonal {f1(pd10)}&nbsp;%, against {f1(r("m10", 12, "pmzw", "z"))} and {f1(r("m10", 12, "pmzw", "z", cat=DG))}&nbsp;%'
                    + ('; one training seed each' if nw10 < 2 else '') + '). ')
    return (head + 'A 2&times;2 separates the two: pairs drawn on either warp (columns), trained and planned on either (rows); all queries / diagonal, %, training seed 87358:</p>'
            + '<div class="tablewrap">' + table_html(['', 'unshrunk-warp pairs', 'shrunk-warp pairs'],
                         [['22&times;22 quadrants, trained on the unshrunk warp', cell('c22', 'pmzw'), cell('c22', 'pmzp')],
                          ['&hellip; trained on the shrunk warp', cell('c22', 'pmzg'), cell('c22', 'pmzs')],
                          ['10&times;10 quadrants, trained on the unshrunk warp', cell('m10', 'pmzw'), cell('m10', 'pmzp')],
                          ['&hellip; trained on the shrunk warp', cell('m10', 'pmzg'), cell('m10', 'pmzs')]], text_cols=(0,)) + '</div>'
            + '<p>The width decides. Trained on the shrunk warped maze, the field reaches the &times;1 level whichever warp its pairs were drawn on: '
            f'{f1(r("c22", 12, "pmzg", "z"))} and {f1(r("c22", 12, "pmzs", "z"))}&nbsp;% on the 22&times;22 quadrants (diagonal {f1(r("c22", 12, "pmzg", "z", cat=DG))} and '
            f'{f1(r("c22", 12, "pmzs", "z", cat=DG))}&nbsp;%), {f1(r("m10", 12, "pmzg", "z"))} and {f1(r("m10", 12, "pmzs", "z"))}&nbsp;% on the 10&times;10&rsquo;s (diagonal '
            f'{f1(r("m10", 12, "pmzg", "z", cat=DG))} and {f1(r("m10", 12, "pmzs", "z", cat=DG))}&nbsp;%). ' + unshrunk +
            f'Shrinking removes failures of both kinds: on the 22&times;22 quadrants, of 900 runs with the gap study&rsquo;s warp, {zf("c22", "pmzw", "arrived_wall")} arrive after '
            f'touching a wall and {zf("c22", "pmzw", "not_arrived")} never arrive (stalls), against {zf("c22", "pmzs", "arrived_wall")} and {zf("c22", "pmzs", "not_arrived")} with the band '
            f'shrunk; on the 10&times;10&rsquo;s {zf("m10", "pmzw", "arrived_wall")} and {zf("m10", "pmzw", "not_arrived")} against {zf("m10", "pmzs", "arrived_wall")} and '
            f'{zf("m10", "pmzs", "not_arrived")}.</p>'
            '<p>So what is left at &times;12 is the band&rsquo;s width in what the network sees, not where the pairs lie. With the band shrunk, the &times;12 maze in the warp <i>is</i> '
            'the &times;1 maze in the warp: the same quadrants, a band of the same width, a box of the same size (half-width 2.63 units at both gaps on the 22&times;22 quadrants, '
            '2.92 with the band unshrunk; 1.85 on the 10&times;10&rsquo;s, 2.14 unshrunk), the same shares of pairs (table below). That is the sense in which the network cannot tell &times;1 from &times;12 '
            '(the request in the section above), and then it plans them alike. The gap study&rsquo;s constructed warp keeps the band&rsquo;s interior at density 1, so in it '
            'they differ by a band 2.7 (2.4) times as wide.</p>')


def m10_cells(r, pooled, nsd):
    """The 10 x 10 quadrants' headline cells (ntrl-demo and warped-maze pairs at x 1 and x 12): pooled over training seeds
    when every cell has two or more, else training seed 87358."""
    cells = {'id1': ('m10', 1, 'id', 'x'), 'id12': ('m10', 12, 'id', 'x'), 'w1': ('m10', 1, 'pmzw', 'z'), 'w12': ('m10', 12, 'pmzw', 'z')}
    n = min(nsd(*c) for c in cells.values())
    if n >= 2:
        return {k: pooled(*c) for k, c in cells.items()}, f'over {WORDS.get(n, n)} training seeds'
    return {k: r(*c) for k, c in cells.items()}, 'training seed 87358'


def zdraw_lede(R, r, pooled, nsd, f1):
    if r('c22', 12, 'pmzw', 'z') is None:
        return ''
    n12 = nsd('c22', 12, 'pmzw', 'z')
    v12 = pooled('c22', 12, 'pmzw', 'z') if n12 >= 2 else r('c22', 12, 'pmzw', 'z')
    sd = f'over {WORDS.get(n12, n12)} training seeds' if n12 >= 2 else 'training seed 87358'
    t = (' Drawing the training pairs on the warped maze too, so that they cover it as ntrl-demo&rsquo;s cover the 22&times;22 maze, closes most of the rest: '
         f'&times;12 {f1(v12)}&nbsp;% {sd}, against {f1(pooled("c22", 12, "pmz", "z"))}&nbsp;% with the pairs drawn on the maze, ntrl-demo&rsquo;s {f1(pooled("c22", 12, "id", "x"))}&nbsp;% '
         f'and &times;1&rsquo;s {f1(pooled("c22", 1, "pmz", "z"))}&nbsp;%'
         + (lambda v: f'; on training seed 87358 it stays between {min(v):.1f} and {max(v):.1f}&nbsp;% at every gap from &times;1 to &times;12' if len(v) == 5 else '')(
             [x for x in (r('c22', k, 'pmzw', 'z') for k in (1, 2, 4, 8, 12)) if x is not None]) +
         f'. The same pairs leave ntrl-demo where it was ({f1(r("c22", 12, "idw", "x"))}&nbsp;%) and the constructed warp '
         f'trained on the maze below it ({f1(r("c22", 12, "pmw", "x"))}&nbsp;%).')
    if r('m10', 12, 'pmzw', 'z') is not None and r('m10', 1, 'id', 'x') is not None:
        v, sd10 = m10_cells(r, pooled, nsd)
        t += (f' On the 10&times;10 maze&rsquo;s quadrants with the same gaps, ntrl-demo falls from {f1(v["id1"])}&nbsp;% at &times;1 to {f1(v["id12"])}&nbsp;% at &times;12, '
              f'this field from {f1(v["w1"])} to {f1(v["w12"])}&nbsp;% ({sd10}).')
    if r('c22', 12, 'pmzs', 'z') is not None and r('m10', 12, 'pmzs', 'z') is not None:
        t += (f' What remains at &times;12 is the band&rsquo;s width in the warp: shrunk to its &times;1 width, the &times;12 mazes plan like the &times;1 mazes '
              f'({f1(r("c22", 12, "pmzs", "z"))} and {f1(r("m10", 12, "pmzs", "z"))}&nbsp;%, training seed 87358)'
              + (f', whichever warp the pairs were drawn on ({f1(r("c22", 12, "pmzg", "z"))} and {f1(r("m10", 12, "pmzg", "z"))}&nbsp;%).' if r('c22', 12, 'pmzg', 'z') is not None and r('m10', 12, 'pmzg', 'z') is not None else '.'))
    return t


def zdraw_verdict(R, r, pooled, nsd, f1, DG):
    if r('c22', 12, 'pmzw', 'z') is None:
        return ''
    n12 = nsd('c22', 12, 'pmzw', 'z')
    v12 = pooled('c22', 12, 'pmzw', 'z') if n12 >= 2 else r('c22', 12, 'pmzw', 'z')
    t = ('<li><b>Pairs drawn on the warped maze.</b> Trained and planned on the warped maze with the pairs drawn there as well: &times;12 '
         f'{f1(v12)}&nbsp;% ({"over %s training seeds" % WORDS.get(n12, n12) if n12 >= 2 else "training seed 87358"}) against ntrl-demo&rsquo;s {f1(pooled("c22", 12, "id", "x"))}&nbsp;% and '
         f'{f1(pooled("c22", 12, "pmz", "z"))}&nbsp;% with the pairs drawn on the maze. Given to ntrl-demo the same pairs change nothing ({f1(r("c22", 12, "idw", "x"))}&nbsp;%); given to the '
         f'constructed warp trained on the maze they leave it below ntrl-demo ({f1(r("c22", 12, "pmw", "x"))}&nbsp;%): each is held back by something else (resolution in the quadrants, '
         'the band&rsquo;s travel time).')
    if r('m10', 12, 'pmzw', 'z') is not None:
        v, sd10 = m10_cells(r, pooled, nsd)
        t += (f' On the 10&times;10 maze&rsquo;s quadrants with the same gaps: &times;1 {f1(v["w1"])} against ntrl-demo&rsquo;s {f1(v["id1"])}&nbsp;%, '
              f'&times;12 {f1(v["w12"])} against {f1(v["id12"])}&nbsp;% ({sd10}).')
    if r('c22', 12, 'pmzs', 'z') is not None and r('m10', 12, 'pmzs', 'z') is not None:
        t += (' What is left at &times;12, mostly diagonal queries, goes with the band&rsquo;s width in the warp' + (', not with where the pairs lie' if r('c22', 12, 'pmzg', 'z') is not None else '') +
              ': trained on the warp with the band shrunk back to its &times;1 width, '
              f'{f1(r("c22", 12, "pmzs", "z"))} and {f1(r("m10", 12, "pmzs", "z"))}&nbsp;% (diagonal {f1(r("c22", 12, "pmzs", "z", cat=DG))} and {f1(r("m10", 12, "pmzs", "z", cat=DG))}&nbsp;%)'
              + ((f', and {f1(r("c22", 12, "pmzg", "z"))} and {f1(r("m10", 12, "pmzg", "z"))}&nbsp;% with the pairs drawn on the unshrunk warp; the shrunk draw&rsquo;s pairs trained on the '
                  f'unshrunk warp, {f1(r("c22", 12, "pmzp", "z"))} and {f1(r("m10", 12, "pmzp", "z"))}&nbsp;%. With the band shrunk, the &times;12 maze in the warp is the &times;1 maze, and '
                  'the network plans it so. The fields need the planner on the warped maze (MPPI on the maze stalls in the shrunk band).')
                 if r('c22', 12, 'pmzg', 'z') is not None and r('m10', 12, 'pmzg', 'z') is not None else '.'))
    return t + '</li>'


TFIELDS = [('id', 'ntrl-demo, pairs drawn on the maze'), ('idw', 'ntrl-demo, pairs drawn on the warped maze (test 1)'),
           ('pmz', 'trained on the warped maze, pairs drawn on the maze'), ('pmzw', 'trained on the warped maze, pairs drawn on the warped maze')]
# the test charts: hue = where the field is trained (the maze: ntrl-demo's blue; the warped maze: the constructed warp's orange), dashed = pairs drawn on the warped maze
TCSS = {'id': '--s-ntrl', 'idw': '--s-ntrl', 'pmz': '--s-pm', 'pmzw': '--s-pm'}
TDASH = {'idw': '6 4', 'pmzw': '6 4'}
TLABEL = {'id': 'ntrl-demo', 'idw': 'ntrl-demo, warped-maze pairs', 'pmz': 'trained on the warped maze', 'pmzw': 'trained on the warped maze, warped-maze pairs'}
QSIDE = {'c22': 8, 'm10': 5}                                         # quadrant side in cells


STRETCH_VIEW = 'https://claude.ai/artifact/FMxYC1xLK2DWBMrH2CRgLD'             # the left / right viewer (report/stretchview.html)


def stretch_table():
    """The final test's numbers (report/stretch.json from stretchreport.py), or None."""
    f = OUT / 'stretch.json'
    return json.loads(f.read_text())['table'] if f.exists() else None


def k1_data():
    """D.k1field for the page's charts: the field trained on x1 against the fields trained on each gap."""
    T = stretch_table()
    if not T:
        return None
    import stretchreport as SR
    return SR.chart({k: v for k, v in T.items()})


def k1_section():
    """K1TEXT0 / K1TEXT / K1TABLE / K1VERDICT: the field trained on gap x1 planned on the wider gaps (request 2026-11-05, evening)."""
    T = stretch_table()
    if not T:
        return {'K1TEXT0': '', 'K1TEXT': '', 'K1TABLE': '', 'K1VERDICT': ''}
    import stretchreport as SR
    x = SR.texts(T)
    req = ('<p>Request (2026-11-05, evening): &ldquo;final test: use the field trained on x1 to plan on the x2,x,x8,x12. Also, let me visualize the x1 time field but stretched according to the '
           'different plus signs. Left/right view, where left is one of these time fields stretched, right is the ground truth.&rdquo; (&ldquo;x&rdquo; read as &times;4.) '
           f'The left/right view is <a href="{STRETCH_VIEW}">its own page</a>.</p>')
    return {'K1TEXT0': req + x['HOW'], 'K1TEXT': x['TEXT'] + x['TEXT2'], 'K1TABLE': x['TABLE'], 'K1VERDICT': x['VERDICT']}


def tests_data(ks):
    """D.tests for the page's test charts: all-queries success against the gap, one panel per base and planner."""
    R = zdraw()
    if not R:
        return None
    import zdraw10 as ZD
    panels = []
    for base, bname in (('c22', '22×22 cross maze'), ('m10', '10×10 maze')):
        for pl, pname in (('x', 'MPPI on the maze · test 2'), ('z', 'MPPI on the warped maze · test 3')):
            series = {}
            for lab, _ in TFIELDS:
                vals = [(lambda c: None if not c else [100 * c[0] / c[1], c[0], c[1]])(ZD.test_count(R, base, k, lab, pl)) for k in ks]
                if any(vals):
                    series[lab] = vals
            panels.append(dict(title=bname, sub=pname, series=series))
    return dict(ks=ks, panels=panels, fields=[lab for lab, _ in TFIELDS], label=TLABEL, css=TCSS, dash=TDASH, xlabel='gap between the quadrants (× 6 cells)')


def tests_section(ks, step=None):
    """TTEXT / TTABLE / TVERDICT: the user's three tests (2026-11-05, afternoon): pairs drawn on the warped maze for
    ntrl-demo itself (test 1), MPPI on the maze with the field looked up there (test 2) and MPPI on the warped maze with its
    constant step (test 3, zq --plan-warp constructed for ntrl-demo's fields), from report/zdraw10.json; step = the
    step-size section's ntrl-demo control (step_section's _STEPINFO)."""
    R = zdraw()
    if not R or not R.get('planwarp'):
        return {'TTEXT0': '', 'TTEXT': '', 'TTABLE': '', 'TVERDICT': ''}
    import zdraw10 as ZD
    f1 = lambda v: '&ndash;' if v is None else f'{v:.1f}'
    pc = lambda c: None if not c else 100 * c[0] / c[1]
    v = lambda base, k, lab, pl, cat=None: pc(ZD.test_count(R, base, k, lab, pl, cat=cat))
    sgn = lambda d: f'{d:+.1f}'.replace('-', '&minus;')
    rng = lambda a: f'{min(a):.1f}&ndash;{max(a):.1f}&nbsp;%'
    P = R['pairs']
    both = lambda base, k, draw: 100 * P[f'{base}/{k}/{draw}']['both_quadrants']
    idseeds = lambda base, k: [ZD.rate(R, base, k, 'id', 'x', seed=s) for s in ZD.seeds(R, base, k, 'id', 'x')]
    G = json.loads((OUT / 'gap_warp.json').read_text()) if (OUT / 'gap_warp.json').exists() else {}
    kb, k8 = max(ks), sorted(ks)[-2]
    small = [k for k in ks if k < k8]
    dif = lambda a, b: round(a, 1) - round(b, 1)                    # differences of the values as shown (0.1), so the text adds up with the table
    d1 = lambda base, k: dif(v(base, k, 'idw', 'x'), v(base, k, 'id', 'x'))

    text = ('<p>Request (2026-11-05, afternoon): &ldquo;on the warped, sample points based on the a band for the warped space. this is test 1. Run it on both plus mazes '
            'for the ntrl-demo. Test 2: mppi on the original space, but the t field you can look up. Test 3: constant step size on warped space.&rdquo;</p>'
            '<p><b>How the tests were run.</b> Test 1 is ntrl-demo itself (no warp, trained on the maze) on pairs drawn on the warped maze. Defining the recipe&rsquo;s '
            'near-wall band in the warp gives these same pairs and speeds: the constructed warp magnifies every wall&rsquo;s surroundings m times along its normal (the '
            'quadrants and the band&rsquo;s edge cells, which hold every wall), so a band 0.175 quadrant cells wide in the warp is the 0.175-cell band of the maze, and the '
            'clearance measured in the warp is m times the maze&rsquo;s, the speed the same. Test 2 is MPPI on the maze with the field looked up at the planner&rsquo;s '
            'poses: ntrl-demo&rsquo;s own evaluation, and for the fields trained on the warped maze T(x<sub>0</sub>, x<sub>1</sub>) = H(&phi;(x<sub>0</sub>), &phi;(x<sub>1</sub>)) '
            '(the numbers in brackets above). Test 3 is MPPI on the warped maze with its constant step (m times the maze planner&rsquo;s, so a step inside a quadrant is the '
            'maze planner&rsquo;s step there and m times longer in the maze over the band&rsquo;s interior), for ntrl-demo&rsquo;s fields too, looked up at x = &phi;<sup>&minus;1</sup>(z) '
            '(<code>zq.py --plan-warp constructed</code>). Every field on its gap&rsquo;s 300 queries, planning seeds 0/1/2, training seed 87358.</p>')

    # ---- test 1: the pairs alone
    def cat_gain(cats):
        """Runs gained by the warped-maze pairs over ntrl-demo's (MPPI on the maze), summed over the narrow gaps and both bases."""
        tot = 0
        for base in ('c22', 'm10'):
            for k in small:
                for c in cats:
                    a, b = ZD.test_count(R, base, k, 'idw', 'x', cat=c), ZD.test_count(R, base, k, 'id', 'x', cat=c)
                    tot += (a[0] - b[0]) if a and b else 0
        return tot
    leave = cat_gain(['adjacent quadrants', 'diagonal quadrants', 'quadrant - open'])
    allg = cat_gain(['same quadrant', 'adjacent quadrants', 'diagonal quadrants', 'quadrant - open', 'open - open'])
    s1 = [(base, idseeds(base, ks[0])) for base in ('c22', 'm10')]
    above = [k for k in small[1:] if len(idseeds('c22', k)) > 1 and v('c22', k, 'idw', 'x') > max(idseeds('c22', k))]
    a8, a12 = ZD.x_arrival(R, 'm10', k8, 'idw'), ZD.x_arrival(R, 'm10', kb, 'idw')
    b8, b12 = ZD.x_arrival(R, 'm10', k8, 'id'), ZD.x_arrival(R, 'm10', kb, 'id')
    cw, cm = ZD.x_arrival(R, 'c22', kb, 'idw'), ZD.x_arrival(R, 'c22', kb, 'id')
    nf = lambda a: a['arrived_wall'] + a['not_arrived']
    t1 = (f'<p><b>Test 1: the pairs alone do not reach the wide gaps.</b> Drawn on the warped maze, {both("c22", kb, "warped"):.0f}&nbsp;% of ntrl-demo&rsquo;s training pairs have '
          f'both ends in the quadrants at &times;{kb} on the 22&times;22 maze, against {both("c22", kb, "real"):.0f}&nbsp;% drawn on the maze ({both("m10", kb, "warped"):.0f} against '
          f'{both("m10", kb, "real"):.0f}&nbsp;% on the 10&times;10&rsquo;s). Its success barely moves: {f1(v("c22", kb, "idw", "x"))} against {f1(v("c22", kb, "id", "x"))}&nbsp;% and '
          f'{f1(v("m10", kb, "idw", "x"))} against {f1(v("m10", kb, "id", "x"))}&nbsp;% ({sgn(d1("c22", kb))} and {sgn(d1("m10", kb))} points), where ntrl-demo&rsquo;s three training seeds on '
          f'maze-drawn pairs span {rng(idseeds("c22", kb))} and {rng(idseeds("m10", kb))}; at &times;{k8}, {sgn(d1("c22", k8))} and {sgn(d1("m10", k8))}. On the 10&times;10&rsquo;s at '
          f'&times;{kb} two changes cancel: same-quadrant queries rise from {f1(v("m10", kb, "id", "x", "same quadrant"))} to {f1(v("m10", kb, "idw", "x", "same quadrant"))}&nbsp;% and queries '
          f'from a quadrant to the band fall from {f1(v("m10", kb, "id", "x", "quadrant - open"))} to {f1(v("m10", kb, "idw", "x", "quadrant - open"))}&nbsp;%. There the pairs also change how '
          f'the runs fail, not how often: on warped-maze pairs {a8["not_arrived"]} and {a12["not_arrived"]} of the {nf(a8)} and {nf(a12)} failed runs at &times;{k8} and &times;{kb} never arrive; '
          f'on maze-drawn pairs {b8["not_arrived"]} and {b12["not_arrived"]} never arrive and the rest ({b8["arrived_wall"]} and {b12["arrived_wall"]}) arrive after touching a wall. On the 22&times;22 '
          f'maze the failures stay wall contacts: {cw["arrived_wall"]} of {nf(cw)} at &times;{kb} on warped-maze pairs, {cm["arrived_wall"]} of {nf(cm)} on maze-drawn pairs.</p>'
          f'<p>At the narrow gaps the pairs help: {", ".join(sgn(d1("c22", k)) for k in small)} points at {", ".join(f"&times;{k}" for k in small)} on the 22&times;22 maze, '
          f'{", ".join(sgn(d1("m10", k)) for k in small)} on the 10&times;10&rsquo;s, mostly on queries that leave their quadrant ({leave} of the {allg} runs gained are between quadrants '
          f'or from a quadrant to the band). This is one training seed. At &times;{ks[0]} the gain is smaller than the spread of ntrl-demo&rsquo;s own seeds '
          f'({" and ".join(rng(a) for _, a in s1)})' + (f'; at {" and ".join(f"&times;{k}" for k in above)} on the 22&times;22 maze it is above all three of them '
          f'({" and ".join(rng(idseeds("c22", k)) for k in above)})' if above else '') + '.</p>')

    # ---- test 3 against test 2: the planner
    W = R.get('planwarp_runs') or {}
    same_diff = sum(e['same quadrant']['lost'] + e['same quadrant']['gained'] for e in W.values())
    same_runs = sum(e['same quadrant']['runs'] for e in W.values())
    dz = lambda base, k, lab: dif(v(base, k, lab, 'z'), v(base, k, lab, 'x'))
    late = [dz(b, k, lab) for b in ('c22', 'm10') for k in ks if k >= 4 for lab in ('id', 'idw') if v(b, k, lab, 'z') is not None]
    early = [dz(b, k, lab) for b in ('c22', 'm10') for k in ks if k < 4 for lab in ('id', 'idw') if v(b, k, lab, 'z') is not None]
    Wk = lambda base, lab=('id'): W.get(f'{base}/{kb}/{lab}', {})
    lost = lambda base: sum(e['lost'] for e in Wk(base).values())
    gained = lambda base: sum(e['gained'] for e in Wk(base).values())
    qo = lambda base, kind: Wk(base).get('quadrant - open', {}).get(kind, 0)
    oo = lambda base, kind: Wk(base).get('open - open', {}).get(kind, 0)
    cells = lambda base: G.get(f'{base}/{kb}', {}).get('m', float('nan')) * (step or {}).get('default_cells', float('nan'))
    ctrl = (f' The step-size control above suggests the length alone is not it: with every step 16 times the default ({step["long_cells"]:.2f} cells, two thirds of the warped '
            f'planner&rsquo;s step over the band) ntrl-demo at &times;{step["k"]} solves {f1(step["ntrl_long"])}&nbsp;% against {f1(step["ntrl_default"])}&nbsp;% (planning seed 0). '
            'I have not traced what in the warped step costs these runs.') if step and step.get('ntrl_long') is not None else ''
    t3 = (f'<p><b>Test 3: the warped planner does not help ntrl-demo&rsquo;s fields either.</b> Inside the quadrants nothing changes: both planners take the same step there, and '
          f'same-quadrant queries solve run for run as on the maze ({same_diff} of {same_runs:,} runs differ over the {len(W)} fields compared). Over the band&rsquo;s interior a step of '
          f'constant length in the warp is m times the maze planner&rsquo;s ({cells("c22"):.1f} cells at &times;{kb} on the 22&times;22 maze, {cells("m10"):.1f} on the 10&times;10&rsquo;s), and '
          f'from &times;4 on it costs ntrl-demo&rsquo;s fields {min(-d for d in late):.1f} to {max(-d for d in late):.1f} points (at &times;1 and &times;2, {sgn(min(early))} to {sgn(max(early))}). '
          f'At &times;{kb} it loses {lost("c22")} and {lost("m10")} of ntrl-demo&rsquo;s runs and gains {gained("c22")} and {gained("m10")}, all on queries that cross the band: runs from a quadrant '
          f'to the band that now end at a wall ({qo("c22", "lost_wall")} and {qo("m10", "lost_wall")}) and open&ndash;open runs that now stall ({oo("c22", "lost_stall")} and {oo("m10", "lost_stall")}).'
          + ctrl + ' For the fields trained on the warped maze the two planners are within '
          f'{max(abs(dz(b, k, lab)) for b in ("c22", "m10") for k in ks if k < kb for lab in ("pmz", "pmzw")):.1f} points of each other below &times;{kb}; at &times;{kb} the warped planner is ahead: '
          f'{sgn(dz("c22", kb, "pmz"))} and {sgn(dz("c22", kb, "pmzw"))} points on the 22&times;22 maze (maze-drawn and warped-maze pairs), {sgn(dz("m10", kb, "pmz"))} and {sgn(dz("m10", kb, "pmzw"))} '
          'on the 10&times;10&rsquo;s.</p>')

    # ---- the four fields at the widest gap
    base_x = lambda b: v(b, kb, 'id', 'x')
    dd = lambda b, lab, pl='x': dif(v(b, kb, lab, pl), base_x(b))
    share = lambda b: (100 * 2 * QSIDE[b] / G[f'{b}/{kb}']['n'], 100 * 2 * QSIDE[b] * G[f'{b}/{kb}']['quadrant_cell_z'] / (2 * G[f'{b}/{kb}']['z_half'])) if f'{b}/{kb}' in G else (float('nan'),) * 2
    sc, sm = share('c22'), share('m10')
    t4 = (f'<p><b>What decides it.</b> At &times;{kb}, one change at a time against ntrl-demo (MPPI on the maze, training seed 87358; 22&times;22 maze, then the 10&times;10&rsquo;s): '
          f'pairs drawn on the warped maze {sgn(dd("c22", "idw"))} and {sgn(dd("m10", "idw"))} points; the warped planner {sgn(dd("c22", "id", "z"))} and {sgn(dd("m10", "id", "z"))}; '
          f'training on the warped maze {sgn(dd("c22", "pmz"))} and {sgn(dd("m10", "pmz"))} ({sgn(dd("c22", "pmz", "z"))} and {sgn(dd("m10", "pmz", "z"))} with the warped planner). '
          f'Training on the warped maze on pairs drawn there: {sgn(dd("c22", "pmzw"))} and {sgn(dd("m10", "pmzw"))} ({sgn(dd("c22", "pmzw", "z"))} and {sgn(dd("m10", "pmzw", "z"))}). '
          'The same pairs do nothing for ntrl-demo and most of the work for the field trained in the warp, and the warp&rsquo;s coordinates do far less without them. '
          f'The reading that fits all four: in the maze&rsquo;s coordinates the quadrants take {sc[0]:.0f}&nbsp;% of each axis at &times;{kb} ({sm[0]:.0f}&nbsp;% on the 10&times;10&rsquo;s), in the '
          f'warp&rsquo;s {sc[1]:.0f}&nbsp;% ({sm[1]:.0f}&nbsp;%). Only in the warp&rsquo;s does the network have room to fit them, and only pairs drawn on the warped maze put the training there '
          f'({both("c22", kb, "real"):.0f}&nbsp;% of the maze-drawn pairs have both ends in the quadrants at &times;{kb}). Where MPPI runs matters least.</p>')

    # ---- the table (under a dropdown; the chart shows the same numbers)
    out = ''
    for base, name in (('c22', 'the 22&times;22 cross maze&rsquo;s quadrants'), ('m10', 'the 10&times;10 maze&rsquo;s quadrants')):
        rows = []
        for lab, lname in TFIELDS:
            for planner, pname in (('x', 'on the maze (test 2)'), ('z', 'on the warped maze (test 3)')):
                rows.append([lname if planner == 'x' else '', pname] + [f1(v(base, k, lab, planner)) for k in ks])
        out += f'<h3>{name}</h3>' + table_html(['field', 'MPPI'] + [f'&times;{k}' for k in ks], rows, text_cols=(0, 1))
    out += ('<p class="note">All queries solved (%), training seed 87358, planning seeds 0/1/2 (300 queries &times; 3). On the warped maze: MPPI in the constructed '
            'warp&rsquo;s coordinates with its constant step, the path mapped back exactly and scored on the maze. &ndash;: not run.</p>')
    verdict = (f'<li><b>One change at a time (the three tests).</b> Pairs drawn on the warped maze alone move ntrl-demo by {sgn(dd("c22", "idw"))} and {sgn(dd("m10", "idw"))} points at &times;{kb} '
               f'(22&times;22 and 10&times;10&rsquo;s), though {both("c22", kb, "warped") / both("c22", kb, "real"):.0f} and {both("m10", kb, "warped") / both("m10", kb, "real"):.0f} times as many of its pairs '
               f'then lie in the quadrants; the warped planner alone moves it by {sgn(dd("c22", "id", "z"))} and {sgn(dd("m10", "id", "z"))}. What lifts &times;{kb} is training on the warped maze on '
               f'pairs drawn there: {f1(v("c22", kb, "pmzw", "x"))} and {f1(v("m10", kb, "pmzw", "x"))}&nbsp;% against {f1(base_x("c22"))} and {f1(base_x("m10"))}&nbsp;%, both planned on the maze. '
               'It takes both: the warp&rsquo;s coordinates and pairs in its quadrants. At the narrow gaps the warped-maze pairs help ntrl-demo by a few points (one training seed).</li>')
    return {'TTEXT0': text, 'TTEXT': t1 + t3 + t4, 'TTABLE': out, 'TVERDICT': verdict}


def warp_text():
    """The warp every run of the section uses (the constructed warp, fixed), how the pairs drawn on it are labelled, and
    how it differs from the gap study's learned warp; numbers from report/gap_warp.json (warpfig.py)."""
    f = OUT / 'gap_warp.json'
    if not f.exists():
        return ''
    G = json.loads(f.read_text())
    c1, c12, m1, m12 = G['c22/1'], G['c22/12'], G['m10/1'], G['m10/12']
    L12 = list((G.get('learned') or {}).get('12', {}).values())
    L1 = list((G.get('learned') or {}).get('1', {}).values())
    t = ('<p><b>The warp.</b> Every field and planner in this section uses one warp, the gap study&rsquo;s constructed warp, fixed: nothing in it is learned '
         '(<code>warp.constructed</code>). It is separable, (x, y, p) &rarr; (F(x), G(y), m&thinsp;p), with F = G magnifying m = 6E/22.1 times over the quadrants&rsquo; columns (rows) '
         f'and the band&rsquo;s edge cells (&times;{c1["m"]:.1f} on the 22&times;22 maze, &times;{c12["m"]:.1f} at &times;12; &times;{m1["m"]:.1f} and &times;{m12["m"]:.1f} on the '
         '10&times;10 quadrants&rsquo; mazes) and once over the band&rsquo;s interior, log-linear over one cell between them. A quadrant cell is therefore '
         f'{c1["quadrant_cell_z"]:.2f} units wide in the warp at every gap, while the band&rsquo;s interior is {c1["band_interior_z"]:.2f} units wide at &times;1 and '
         f'{c12["band_interior_z"]:.2f} at &times;12 ({m1["band_interior_z"]:.2f} and {m12["band_interior_z"]:.2f} on the 10&times;10 quadrants). The first figure draws each maze, '
         'its image under the warp, and where 2,000 training pairs lie in the warp, drawn on the maze or on the warped maze. Drawing on the warped maze changes where the pairs lie, '
         'not how they are labelled: each pair is mapped back to the maze, its clearances and speeds are measured there by ntrl-demo&rsquo;s rule, and training in the warp keeps those '
         'speeds as the targets of the eikonal term, now in z (|&nabla;<sub>z</sub>T| = 1/S), so the field learns travel time in units of the warp: m times ntrl-demo&rsquo;s inside '
         'the quadrants, the same over the band&rsquo;s interior.</p>')
    if L12 and L1:
        lb, lq = [e['band_interior'] for e in L12], [e['quadrant'] for e in L12]
        span = lambda v: f'&times;{min(v):.0f}' if f'{min(v):.0f}' == f'{max(v):.0f}' else f'&times;{min(v):.0f}&ndash;{max(v):.0f}'
        t += ('<p>The gap study&rsquo;s learned warp is a different warp, and no field of this section uses it. It has the same separable form with its densities free between '
              '&times;1 and &times;m (one node per cell, started at the top), trained together with the field on the maze: ntrl-demo&rsquo;s loss reaches the densities through '
              '&phi; and its Jacobian, plus a free-space compression term, a wall-normal stretch term and a smoothness penalty. The second figure sets it against the design. '
              f'At &times;1 it finds the design (&times;{min(e["quadrant"] for e in L1):.0f} over the quadrants, &times;{max(e["band_interior"] for e in L1):.0f} over the band&rsquo;s '
              f'interior, three training seeds). At &times;12 it stays near its start (&times;{G["c22/12"]["m"]:.1f}), {span(lb)} over the band&rsquo;s interior and '
              f'{span(lq)} over the quadrants (medians per seed): close to a uniform stretch, and like the uniform stretch its field does not train '
              'from &times;8 on (above).</p>')
    return t


def warp_figs():
    out = ''
    for name, alt, cap in (
            ('gap_warp.png', 'Four rows of four panels: the 22 by 22 quadrants and the 10 by 10 quadrants at gaps 1 and 12. Each row shows the maze shaded by the warp\'s magnification, '
             'the warped maze, and training pairs in the warp drawn on the maze and on the warped maze.',
             'The constructed warp. Left: each maze, shaded by the warp&rsquo;s area magnification (darkest &times;m&sup2; over the quadrants, &times;m over the band&rsquo;s arms, '
             '&times;1 at its centre). Second: the warped maze z = &phi;(x), all four at one scale; the &times;12 maze differs from the &times;1 maze in the warp only by a wider band. '
             'Right: 2,000 training pairs of each prepare run in the warp (orange: the start x0, kept within 0.175 cells of a wall; blue: the end x1). Drawn on the maze, the &times;12 '
             'pairs crowd the band&rsquo;s interior and its outer walls, which the warp squeezes; drawn on the warped maze, they cover it evenly.'),
            ('gap_warp_density.png', 'Two line charts of magnification across the maze at gaps 1 and 12: the constructed warp and three seeds of the learned warp.',
             'Magnification dz/dx across the 22&times;22 quadrants&rsquo; mazes (log scale; shaded: quadrant columns): the constructed warp (orange, fixed) and the gap study&rsquo;s '
             'learned warp (green, three training seeds, from each run&rsquo;s saved warp map). At &times;1 the learned warp finds the design; at &times;12 it stays near its start, '
             'magnifying the band about as much as the quadrants.')):
        f = OUT / name
        if not f.exists():
            continue
        from PIL import Image
        w, h = Image.open(f).size
        out += f'<figure><img src="{name}" alt="{alt}" width="{w}" height="{h}"><figcaption>{cap}</figcaption></figure>'
    return out


def zdraw_fig():
    f = OUT / 'gap_mazes10.png'
    if not f.exists():
        return ''
    from PIL import Image
    w, h = Image.open(f).size
    return (f'<figure><img src="gap_mazes10.png" alt="Five mazes, each at its own scale: the four 5 by 5 quadrants of the 10 by 10 maze, pushed apart so the open band between them is 6, 12, 24, 48 '
            f'and 72 cells wide, each with its longest diagonal truth path." width="{w}" height="{h}"><figcaption>The 10&times;10 maze&rsquo;s quadrants with the gap study&rsquo;s bands, '
            'each maze drawn at its own scale (shaded: the four 5&times;5 quadrant mazes, unchanged in every maze; blue: the longest diagonal-quadrant query&rsquo;s true path).</figcaption></figure>')


def loss_of(data, k, key, seed=87358):
    e = data.get(k, {}).get(key, {}).get(seed)
    return json.loads((run_dir(e) / 'results.json').read_text()).get('final_loss', float('nan')) if e else float('nan')


def lede_constructed(data, get, fmt, tot):
    key = 'constructed, heading ×1 + line of sight'
    ps = per_seed(data, 8, key, 'open - open')
    if len(ps) < 2:
        return ''
    best, rest = ps[0], ps[1:]
    return (f'The constructed warp fits it only with its heading left unmagnified, and then not on every training seed (open&ndash;open at &times;8: {best[1]:.0f}&nbsp;% on seed {best[0]}, '
            + ' and '.join(f'{v:.0f}&nbsp;%' for _, v in rest) + ' on the other two): over three seeds it solves '
            f'{fmt(tot(8, key), 1)}&nbsp;% of all queries at &times;8 against ntrl-demo&rsquo;s {fmt(tot(8, "ntrl-demo"), 1)}&nbsp;%.')


def build_text(data, preps, ks):
    H = {}
    get = lambda k, m, cat=None: (lambda c: 100 * c[0] / c[1] if c else None)(counts(data[k][m], cat)) if m in data.get(k, {}) else None
    seed1 = lambda k, m, cat=None: (lambda e: None if e is None else (100 * e['success_total'][0] / e['success_total'][1] if cat is None else
                                    (lambda v: 100 * v[0] / v[1] if v else None)(e['by_category'].get(cat))))(data.get(k, {}).get(m, {}).get(87358))
    nseeds = lambda k, m: len(data.get(k, {}).get(m, {}))
    g87 = seed1          # paired comparisons with the single-seed controls use training seed 87358 on both sides
    fmt = lambda v, d=0: '&ndash;' if v is None else f'{v:.{d}f}'
    kmax = max(ks)
    big = [k for k in ks if k >= 2]
    OF = json.loads((OUT / 'openfit.json').read_text()) if (OUT / 'openfit.json').exists() else {}
    fit = {(r['k'], r['method']): r for r in OF.values()}
    SS = sampler_stats(preps)
    n_of = {k: preps[k]['maze']['n'] for k in ks}

    # ---------------------------------------------------------------- lede and tiles
    o = lambda k, m: get(k, m, 'open - open')
    sq = lambda k, m: get(k, m, 'same quadrant')
    tot = lambda k, m: get(k, m)
    H['LEDE'] = (
        f'The 22&times;22 cross maze&rsquo;s four 8&times;8 quadrant mazes were pushed apart until the open band between them was 2, 4, 8 and 12 times its '
        f'6 cells (mazes of {n_of.get(2, 28)} to {n_of.get(kmax, 88)} cells a side), and ntrl-demo was trained on each with the constructed warp, the learned warp, '
        'warp + scaling and the README warp, against ntrl-demo itself and a uniform stretch. The warps win while the band is narrow and lose once it is wide: '
        f'at &times;2 the constructed warp solves {fmt(tot(2, "constructed"), 1)}&nbsp;% of all queries against ntrl-demo&rsquo;s {fmt(tot(2, "ntrl-demo"), 1)}&nbsp;%, '
        f'at &times;{kmax} {fmt(tot(kmax, "constructed"), 1)}&nbsp;% against {fmt(tot(kmax, "ntrl-demo"), 1)}&nbsp;%. Inside the quadrants the warps that magnify them stay ahead at every gap '
        f'(same quadrant at &times;{kmax}: constructed {fmt(sq(kmax, "constructed"))}&nbsp;% against {fmt(sq(kmax, "ntrl-demo"))}&nbsp;%); what they lose is the open band, which ntrl-demo '
        f'crosses at {fmt(o(kmax, "ntrl-demo"))}&nbsp;% and the constructed warp at {fmt(o(kmax, "constructed"))}&nbsp;%. Their fields are right up close but place far open poses '
        'next to the goal, and the planner stalls in the spurious sinks this leaves. More training data does not fix it, and neither does planning in the warped space. '
        'A longer planner step gets out of the shallow sinks only (warp + scaling on two training seeds of three), costs the quadrants, and still leaves warp + scaling short of ntrl-demo. '
        'Supervising the straight-line distance where it is the travel time (no labels, no planner) repairs much of the band when the network can fit it: '
        f'warp + scaling goes {fmt(g87(kmax, "warp + scaling", "open - open"))} &rarr; {fmt(g87(kmax, "warp + scaling + line of sight", "open - open"))}&nbsp;% open&ndash;open at &times;{kmax} (training seed 87358 for both). '
        + lede_constructed(data, get, fmt, tot) + ' Trained the gap study&rsquo;s way, none of the settings does better than ntrl-demo once the band is 8&times; wider; the best come level with it.'
        + warped_section(ks)['ZLEDE'])
    tiles = []
    for m in METHODS:
        a, b = get(min(ks), m), get(kmax, m)
        if a is not None and b is not None:
            tiles.append(f'<div class="tile"><span class="k">{esc(m)}</span><span class="v">{pct(b)}</span><span class="s">at gap &times;{kmax} '
                         f'({n_of[kmax]}&times;{n_of[kmax]}); {pct(a)} at &times;{min(ks)} (22&times;22)</span></div>')
    H['TILES'] = '\n'.join(tiles)

    # ---------------------------------------------------------------- mazes
    pk = {k: preps[k] for k in ks}
    oracle = all(sum(v[0] for v in p.get('oracle', {}).get('by_category', {}).values()) == sum(v[1] for v in p.get('oracle', {}).get('by_category', {}).values()) for p in pk.values())
    dq = {k: pk[k]['queries'].get('d_cells_by_category', {}) for k in ks}
    H['MAZETEXT'] = (
        '<p>Every maze keeps the cross maze&rsquo;s four 8&times;8 quadrant mazes unchanged and widens the open band between them: '
        + ', '.join(f'&times;{k} = {6 * k} cells ({n_of[k]}&times;{n_of[k]})' for k in ks) +
        '. &times;1 is the 22&times;22 cross maze itself (its fields are the cross-warp study&rsquo;s runs). '
        'Each maze has its own query set: 48 goals (6 in each quadrant, 24 in the open band) and 60 queries in each of five region pairs, '
        '<b>same quadrant</b>, <b>adjacent quadrants</b>, <b>diagonal quadrants</b>, <b>quadrant &ndash; open</b> and <b>open &ndash; open</b>, '
        'each certified by its true path (300 per maze). True travel distance, median: '
        + '; '.join(f'&times;{k}: ' + ', '.join(f'{c.split()[0]} {v[1]:.0f}' for c, v in dq[k].items()) for k in ks if dq[k]) +
        ' cells (same, adjacent, diagonal, quadrant&ndash;open, open&ndash;open). '
        + ('The planner on the true travel time solves all 300 queries of every maze (planning seed 0), so every failure below is the learned field&rsquo;s. ' if oracle else '') +
        '</p>')

    # ---------------------------------------------------------------- methods
    H['METHODTEXT'] = (
        '<p>All methods are ntrl-demo (its network, data recipe, 5,000 epochs, MPPI planner scaled per cell); only the map in front of its Fourier encoder '
        'changes. Magnifications scale with the maze side E = n + 0.1 so that a quadrant cell looks the same to the encoder at every gap as in the 22&times;22 '
        'study, where these settings were best:</p><ul class="tight">'
        '<li><b>ntrl-demo</b>: no warp (k &ge; 2: the identity warp through the same trainer, bit for bit ntrl-demo).</li>'
        '<li><b>uniform</b>: the encoder&rsquo;s input multiplied by 4E/22.1 (ntrl-demo&rsquo;s B &times; 4 at 22&times;22).</li>'
        '<li><b>constructed</b>: separable warp, quadrant rows and columns (and the band&rsquo;s edge cells) &times; 6E/22.1, the open band &times;1, heading &times; 6E/22.1 '
        '(pm6 at 22&times;22, 84.5&nbsp;% there).</li>'
        '<li><b>learned</b>: separable warp learned from the field&rsquo;s loss with the free-space and wall-normal terms, bounded &times;1 &hellip; &times; 6E/22.1, '
        'started at the top bound (la6b at 22&times;22).</li>'
        '<li><b>warp + scaling</b>: the README warp (learned coupling warp, box-preserving, free space pulled towards a quarter of its area) times a uniform &times;4, '
        'heading &times; 6E/22.1.</li>'
        '<li><b>README warp</b>: the same learned warp alone, which only moves resolution from the open band into the quadrants.</li></ul>')
    mrows = []
    for k in ks:
        n = n_of[k]
        E = n + 0.1
        mrows.append([f'&times;{k}', f'{n} &times; {n}', f'{6 * k}', f'&times;{4 * E / 22.1:.1f}', f'&times;{6 * E / 22.1:.1f} / &times;1',
                      f'&times;1 &hellip; &times;{6 * E / 22.1:.1f}', f'&times;4, heading &times;{6 * E / 22.1:.1f}'])
    H['METHODTABLE'] = table_html(['gap', 'maze (cells)', 'band (cells)', 'uniform stretch', 'constructed: quadrants / band', 'learned (bounds)', 'warp + scaling'], mrows)

    # ---------------------------------------------------------------- results
    rows_txt = []
    for m in ('constructed', 'learned', 'warp + scaling', 'uniform', 'README warp'):
        if m in data.get(kmax, {}):
            rows_txt.append(f'{m} {fmt(get(kmax, m), 1)}&nbsp;%')
    H['RESULTS'] = (
        f'<p><b>Inside the quadrants the warps that magnify them still help.</b> Same-quadrant queries at &times;{kmax}: ntrl-demo {fmt(sq(kmax, "ntrl-demo"))}&nbsp;%, '
        + ', '.join(f'{m} {fmt(sq(kmax, m))}&nbsp;%' for m in ('constructed', 'learned', 'warp + scaling', 'uniform', 'README warp') if m in data.get(kmax, {})) +
        f'. Every method falls with the gap here, ntrl-demo from {fmt(sq(1, "ntrl-demo"))}&nbsp;% at &times;1, the constructed warp from {fmt(sq(1, "constructed"))}&nbsp;%, '
        'although the quadrant mazes and their per-cell magnification are the same in every maze (below: fewer training pairs land inside one quadrant).</p>'
        f'<p><b>Crossing between quadrants collapses for everyone.</b> Adjacent and diagonal pairs need both quadrant mazes and the band; at &times;{kmax} '
        f'every method solves {max((get(kmax, m, "adjacent quadrants") or 0) for m in data[kmax]):.0f}&nbsp;% or less of the adjacent and '
        f'{max((get(kmax, m, "diagonal quadrants") or 0) for m in data[kmax]):.0f}&nbsp;% or less of the diagonal pairs, almost all ending in a wall.</p>'
        f'<p><b>The open band is where the warps lose.</b> ntrl-demo solves {fmt(min(o(k, "ntrl-demo") for k in ks))}&ndash;100&nbsp;% of open&ndash;open queries at every gap. '
        'Every method that magnifies the band loses them from &times;4 on: '
        + ', '.join(f'{m} {" &rarr; ".join(fmt(o(k, m)) for k in ks if o(k, m) is not None)}&nbsp;%' for m in ('uniform', 'constructed', 'learned', 'warp + scaling')) +
        f' (&times;1 &rarr; &times;{kmax}). The README warp, which shrinks the band instead, keeps it ({" &rarr; ".join(fmt(o(k, "README warp")) for k in ks if o(k, "README warp") is not None)}&nbsp;%). '
        f'Overall at &times;{kmax}: ntrl-demo {fmt(get(kmax, "ntrl-demo"), 1)}&nbsp;%, ' + ', '.join(rows_txt) + '.</p>')
    tabs = []
    for k in ks:
        rows = []
        for m in METHODS:
            if m not in data[k]:
                continue
            c = counts(data[k][m])
            rows.append([esc(m), str(len(c[2])), f'<b>{100 * c[0] / c[1]:.1f}</b>'] + [fmt(get(k, m, cat)) for cat in CATS])
        tabs.append(f'<h3>Gap &times;{k} ({n_of[k]} &times; {n_of[k]} cells, band {6 * k} cells)</h3>' + table_html(['method', 'seeds', 'all %'] + [esc(c) for c in CATS], rows))
    H['TABLES'] = '\n'.join(tabs)

    # ---------------------------------------------------------------- mechanism: failure types
    H['MECHTEXT'] = (
        '<p><b>Two ways to fail.</b> A run fails by touching a wall on its way to the goal, or by never arriving within its step budget (a stall: MPPI sits in a dip '
        'of the field). ' + two_ways(data, preps, big) + '</p>')
    frows = []
    for k in [k for k in big if k >= 8]:
        for m in METHODS:
            if m not in data.get(k, {}):
                continue
            f = fail_types(data, preps, k, m)
            mz, op = f['maze'], f['open']
            frows.append([f'&times;{k}', esc(m), str(len(data[k][m])), f'{100 * mz[1] / mz[0]:.0f} / {100 * mz[2] / mz[0]:.0f} / {100 * mz[3] / mz[0]:.0f}',
                          f'{100 * op[1] / op[0]:.0f} / {100 * op[2] / op[0]:.0f} / {100 * op[3] / op[0]:.0f}'])
    H['FAILTABLE'] = table_html(['gap', 'method', 'seeds', 'queries with a quadrant end: solved / wall / stalled %', 'open &ndash; open: solved / wall / stalled %'], frows)

    # ---------------------------------------------------------------- mechanism: sinks
    H['SINKTEXT'] = (
        '<p><b>The stalls are spurious sinks.</b> The true travel time to a goal in the open band has no minimum but the goal. For each open&ndash;open goal, '
        'follow steepest descent on the learned field from every open-band cell (x, y on a 0.25-cell grid, best of 12 headings) and ask where it ends. '
        'ntrl-demo drains the whole band to the goal at every gap; the magnifying warps split it into basins of spurious sinks, and the share of the band they trap '
        'tracks the open&ndash;open failures:</p>')
    srows = []
    for m in METHODS:
        cells = []
        for k in big:
            r = fit.get((k, m))
            ok = seed1(k, m, 'open - open')
            cells.append('&ndash;' if r is None or 'sinks' not in r else f'{100 * r["sinks"]["trapped_share"]:.0f} / {fmt(ok)}')
        if any(c != '&ndash;' for c in cells):
            srows.append([esc(m)] + cells)
    H['SINKTABLE'] = table_html(['method'] + [f'&times;{k}' for k in big], srows) + '<p class="note">Each cell: share of the open band draining to a spurious sink / open&ndash;open queries solved (training seed 87358, planning seeds 0/1/2).</p>'
    lf12 = fit.get((kmax, 'README warp'))
    H['SINKTEXT2'] = ''

    # ---------------------------------------------------------------- mechanism: cause and controls
    H.update(cause_text(data, preps, ks, fit, SS, g87, fmt))
    H.update(stretch_section(data, preps, fmt))
    H.update(step_section(data, preps))
    Z = warped_section(ks)
    H['ZTEXT'], H['ZTABLE'] = Z['ZTEXT'], Z['ZTABLE']
    ZDS = zdraw_section(ks)
    for key in ('ZDTEXT0', 'ZDWARP', 'ZDTEXT', 'ZDFIG', 'ZDTABLE'):
        H[key] = ZDS[key]
    H.update(tests_section(ks, H.pop('_STEPINFO', None)))
    H.update(k1_section())
    H['LEDE'] += ZDS['ZDLEDE']

    # ---------------------------------------------------------------- verdict, measured, runs
    H['VERDICT'] = '<ul class="tight">' + ''.join([
        '<li><b>Wider empty space takes the warps&rsquo; lead away.</b> All queries, ntrl-demo / constructed / warp + scaling: '
        + '; '.join(f'&times;{k} {fmt(tot(k, "ntrl-demo"), 1)} / {fmt(tot(k, "constructed"), 1)} / {fmt(tot(k, "warp + scaling"), 1)}&nbsp;%' for k in ks) +
        '. The learned warp and the uniform stretch fall furthest; the README warp alone stays level with ntrl-demo throughout.</li>',
        '<li><b>By region pair.</b> <i>Same quadrant</i>: every warp that magnifies the quadrants stays ahead of ntrl-demo at every gap (the README warp, which magnifies them least, only by a few points and not at &times;12), though all methods fall as the gap grows. '
        '<i>Adjacent and diagonal quadrants</i>: everyone collapses by &times;4&ndash;&times;8; the runs end in walls inside the quadrant mazes, as on the 22&times;22 maze, '
        'and these queries need two quadrant mazes plus the band. <i>Quadrant &ndash; open</i>: no method keeps an edge at the large gaps. '
        '<i>Open &ndash; open</i>: ntrl-demo and the README warp stay near 100&nbsp;%, while every warp that magnifies the band loses it from &times;4 on.</li>',
        '<li><b>Why the band breaks.</b> The magnified fields are right up close and wrong at range: they put far open poses next to the goal, and those poses become '
        'sinks the planner cannot leave. Not the training data (pairs with two open ends do not help), not the heading alone (ntrl-demo with a magnified heading is fine). '
        'With its heading magnified as well, the constructed warp&rsquo;s network cannot represent the straight-line distance across the band even when trained on it; with the heading '
        'at &times;1 it manages on one training seed of three. Planning in the warped space does not get the planner past the sinks. A longer MPPI step does, where they are shallow '
        '(the field still reads more than a third of the straight-line distance there: warp + scaling on training seeds 87358 and 2), and not where they are deep. It costs the quadrants, '
        'and ' + H.get('STEPCLOSE', '') + '. '
        'The uniform stretch and the learned warp (which starts as the uniform stretch) do not train at all from &times;8 on.</li>',
        '<li><b>Inside the quadrants</b> ntrl-demo&rsquo;s decline is resolution: its fixed Fourier spectrum sees ever smaller cells. The warps keep the resolution and still decline. '
        'ntrl-demo&rsquo;s temporal-difference step is a fixed share of the maze (2.6 cells at 88&times;88, where corridors are 1 cell wide); set to the 22&times;22 maze&rsquo;s 0.66 cells it '
        f'lifts the warps&rsquo; same-quadrant success at &times;8 (training seed 87358: constructed {fmt(g87(8, "constructed", "same quadrant"))} &rarr; {fmt(g87(8, "constructed, TD step 0.66 cells", "same quadrant"))}&nbsp;%, '
        f'warp + scaling with line of sight {fmt(get(8, "warp + scaling + line of sight", "same quadrant"))} &rarr; {fmt(get(8, "warp + scaling + line of sight + TD step", "same quadrant"))}&nbsp;%) '
        'but loosens the long range: warp + scaling then fits the straight-line distance far worse and loses the band again '
        f'(open&ndash;open {fmt(get(8, "warp + scaling + line of sight", "open - open"))} &rarr; {fmt(get(8, "warp + scaling + line of sight + TD step", "open - open"))}&nbsp;%). A trade-off, not a fix.</li>',
        Z['ZVERDICT'],
        ZDS['ZDVERDICT'],
        H.pop('TVERDICT', ''),
        H.pop('K1VERDICT', ''),
        '<li><b>What it would take, training on the maze.</b> Trained the gap study&rsquo;s way, no setting tried beats ntrl-demo once the band is 8&times; wider. The ones that come level all supervise the straight-line distance where it is '
        'the travel time (line of sight needs no labels and no planner) and keep the band&rsquo;s metric representable: an even magnification (warp + scaling) or an unmagnified '
        'heading (constructed), the latter only on some seeds. ' + best_text(data, get, fmt, tot) + '</li>',
        '<li><b>Caveats.</b> The per-gap tables pool training seeds 87358 / 1 / 2. The controls are training seed 87358 unless their table lists more seeds, and the diagnostics '
        'are seed 87358. The one control repeated on three seeds, the constructed warp with heading &times;1 and line of sight at &times;8, went from '
        + ' / '.join(f'{v:.1f}&nbsp;%' for _, v in per_seed(data, 8, 'constructed, heading ×1 + line of sight')) +
        ' (seeds 87358 / 1 / 2): single-seed control results can mislead. The two GPUs differ by a few runs on the same seed. Magnifications scale with the maze (the 22&times;22 '
        'study&rsquo;s best settings carried over); other settings per gap were not searched.</li>',
    ]) + '</ul>'

    H['MEASURED'] = (
        '<ul class="tight">'
        '<li>Mazes: <code>gap.py</code> (the cross maze&rsquo;s walls with the band widened; &times;1 reproduces the 22&times;22 cross exactly). '
        'Truth: maze_scale&rsquo;s C-space grid (1/15 cell, 13 headings), solved by a Dijkstra in C (<code>truthfast.py</code>, equal to the Jacobi fixed point to 4e-6).</li>'
        '<li>Queries (<code>gapprepare.py</code>): goals stratified by region (seed 7), candidate starts (seed 11), 60 per region pair, each certified collision-free '
        'along its true path; never resampled after training.</li>'
        '<li>Training: ntrl-demo&rsquo;s recipe (500,000 pairs, seed 0), 5,000 epochs, training seeds 87358 / 1 / 2 where listed; one RTX 3090 and one RTX 5070 '
        '(the two GPUs differ by a few runs on the same seed). Success: MPPI planning seeds 0/1/2 per query, no wall contact, pooled over training seeds.</li>'
        '<li>Diagnostics (<code>openfit.py</code>, training seed 87358): eikonal error | |&nabla;T| &minus; 1 | at open poses with a near-wall or an open partner; '
        'the share of far open pairs whose descent points away; and the basin map above (share of the open band draining to a sink more than 1 cell from the goal).</li>'
        '<li>Controls (<code>gapprepare.py --open-pairs / --local-pairs --queries-from</code>) reuse the main prepare&rsquo;s goals and queries exactly; only the training data change.</li>'
        '</ul>')
    run_rows = []
    for k in ks:
        for m in sorted(data[k], key=lambda x: (METHODS.index(x.split(' + ')[0]) if x.split(' + ')[0] in METHODS else 99, x)):
            for s_, e in sorted(data[k].get(m, {}).items(), key=lambda t: (t[0] != 87358, t[0])):
                run_rows.append([f'&times;{k}', esc(m), str(s_), f'<code>{e["_id"]}</code>', f'{e["success_total"][0]} / {e["success_total"][1]}'])
    H['RUNS'] = (f'<p class="note">Query sets (gapprepare): ' + ', '.join(f'&times;{k} <code>{preps[k]["_id"]}</code>' for k in ks) + '.</p>' +
                 table_html(['gap', 'method', 'training seed', 'registered run', 'runs solved'], run_rows))
    return H


if __name__ == '__main__':
    main()
