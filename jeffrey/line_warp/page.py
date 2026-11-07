"""The findings page (report/line_warp.html) from report/summary.json (report.py) and page_template.html.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/line_warp/report.py && ~/.venvs/bellman-ntfield/bin/python jeffrey/line_warp/page.py

Charts are drawn by the page's own script from the DATA blob; every number in the text is read from the summary. text.py holds the prose.
"""
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'report'
GAPS = [1, 2, 4, 8, 12]
CATS = ['same quadrant', 'adjacent quadrants', 'diagonal quadrants', 'quadrant - open', 'open - open']
CAT_SHORT = ['same quadrant', 'adjacent quadrants', 'diagonal quadrants', 'quadrant – open', 'open – open']
BASE = {'c22': '22×22 quadrants', 'm10': '10×10 quadrants'}


def table(head, rows, txt=()):
    """HTML table; txt: indices of text columns."""
    h = ''.join(f'<th class="{"txt" if i in txt else ""}">{c}</th>' for i, c in enumerate(head))
    b = ''
    for r in rows:
        b += '<tr>' + ''.join(f'<td class="{"txt" if i in txt else "num"}">{c}</td>' for i, c in enumerate(r)) + '</tr>'
    return f'<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def f1(v):
    return '–' if v is None else f'{v:.1f}'


def q1200(S, name):
    v = S['q1200'].get(name)
    if not v:
        return None
    t = list(v.values())
    return dict(pct=100 * sum(x[0] for x in t) / sum(x[1] for x in t), per_seed=[100 * x[0] / x[1] for x in t], raw=[x[0] for x in t], of=t[0][1])


def plus_panels(S):
    def rows(items):
        out = []
        for name, css, pct, seeds, of in items:
            if pct is None:
                continue
            per = [100 * s / of for s in seeds]
            out.append(dict(name=name, css=css, lo=min(per), mid=pct, hi=max(per), text=[['pooled', f'{pct:.1f} %'], ['per training seed', ' / '.join(f'{x:.1f}' for x in per)]]))
        return out
    C = S['plus_controls']
    R = S['plus'].get('recipe')
    D = S['plus'].get('dil')

    def c240(n):
        return C[n]['pct'], C[n]['per_seed'], C[n]['runs_per_seed']

    a = [('ntrl-demo', '--s-id', *c240('ntrl-demo')), ('uniform B × 4', '--s-uni', *c240('uniform B x 4')), ('earlier learned warp', '--s-con2', *c240('earlier learned warp (la6b)')),
         ('hand-built × 6', '--s-con', *c240('constructed x 6 (pm6)'))]
    if R:
        a.insert(3, ('line loss, no dilation', '--s-learn', R['pct'], R['per_seed'], R['runs_per_seed']))
    if D:
        a.insert(4, ('line loss + dilation', '--s-learn', D['pct'], D['per_seed'], D['runs_per_seed']))
    T = S['plus'].get('sep_h070')
    if T:
        a.insert(3, ('bound 0.70, no TV', '--s-con2', T['pct'], T['per_seed'], T['runs_per_seed']))
    b = []
    for label, css, key in (('ntrl-demo', '--s-id', 'ntrl-demo'), ('uniform B × 4', '--s-uni', 'uniform B x 4'), ('earlier learned warp', '--s-con2', 'earlier learned warp (la6b)'), ('bound 0.70, no TV', '--s-con2', 'bound 0.70, no TV'),
                            ('line loss, no dilation', '--s-learn', 'recipe without dilation'), ('line loss + dilation', '--s-learn', 'the recipe'), ('hand-built × 6', '--s-con', 'constructed x 6 (pm6)')):
        q = q1200(S, key)
        if q:
            b.append((label, css, q['pct'], q['raw'], q['of']))
    base = dict(xlabel='runs solved (%)', min=40, max=100, log=False, ref=100, ticks=[40, 60, 80, 100], css='--s-learn')
    return [dict(base, title='240 queries', sub='used to choose settings', rows=rows(a)), dict(base, title='1,200 queries', sub='held out', rows=rows(b))]


def split_chart(S, keys, xs, xlabel, xname, title, sub):
    """Solved, never arrived and arrived through a wall against a setting of the plus maze's fit (each point pools its seeds)."""
    have = [(k, x) for k, x in zip(keys, xs) if k in S['plus']]
    series = {'succ': [], 'stall': [], 'wall': []}
    for k, _ in have:
        r = S['plus'][k]['rows']
        n = sum(x['runs'] for x in r)
        for f, v in (('succ', sum(x['success'] for x in r)), ('stall', sum(x['not_arrived'] for x in r)), ('wall', sum(x['arrived_wall'] for x in r))):
            series[f].append([100 * v / n, None, None, f'{100 * v / n:.1f} %'])
    return dict(fields=['succ', 'stall', 'wall'], label=dict(succ='solved', stall='never arrived', wall='arrived through a wall'), css=dict(succ='--s-learn', stall='--s-con', wall='--s-id'), dash={}, xs=[x for _, x in have],
                xlabel=xlabel, xname=xname, ylabel='share of runs (%)', ymax=100, panels=[dict(title=title, sub=sub, series=series)])


def tv_chart(S):
    return split_chart(S, ['tv02', 'tv05', 'tv10', 'tv20', 'tv40'], ['0.02', '0.05', '0.10', '0.20', '0.40'], 'total variation weight', 'TV', 'Plus maze, total variation', 'one seed')


def bound_chart(S):
    return split_chart(S, ['sep_h070', 'sep_h100', 'sep_h125', 'sep_h160', 'sep_h200'], ['0.70', '1.00', '1.25', '1.60', '2.00'], 'stretch bound, no TV', 'bound', 'Plus maze, bound only', '0.70, 2.00: 3 seeds')


def gap_chart(S):
    F = ['learned', 'pmzs', 'pmzw', 'pmz', 'id', 'pm']
    label = dict(learned='learned warp (line loss)', pmzw='hand-built warp, warped pairs', pmzs='hand-built, band also shrunk by hand', pmz='hand-built warp, real pairs', id='ntrl-demo', pm='hand-built, trained on the maze')
    css = dict(learned='--s-learn', pmzw='--s-con', pmzs='--s-con2', pmz='--s-con', id='--s-id', pm='--s-id')
    dash = dict(pmz='4 3', pm='4 3')
    panels = []
    for base in ('c22', 'm10'):
        s = {f: [] for f in F}
        for k in GAPS:
            e = S['gap'][f'{base}/{k}']
            if 'pct' in e:
                lo, hi = min(e['per_seed']), max(e['per_seed'])
                s['learned'].append([e['pct'], lo, hi, f"{e['pct']:.1f} % ({len(e['per_seed'])} seed{'s' if len(e['per_seed']) > 1 else ''})"])
            else:
                s['learned'].append(None)
            for f in F[1:]:
                c = e['controls'][f]
                s[f].append(None if c['pct'] is None else [c['pct'], None, None, f"{c['pct']:.1f} % ({c['seeds']} seed{'s' if c['seeds'] > 1 else ''})"])
        panels.append(dict(title=BASE[base], sub='runs solved (%)', series=s))
    return dict(fields=F, label=label, css=css, dash=dash, off=['pmz', 'pm'], xs=[f'×{k}' for k in GAPS], xlabel='gap', xname='gap', ylabel='runs solved (%)', ymax=100, panels=panels)


def cat_chart(S):
    F = ['learned', 'pmzw', 'id']
    label = dict(learned='learned warp', pmzw='hand-built warp, warped pairs', id='ntrl-demo')
    css = dict(learned='--s-learn', pmzw='--s-con', id='--s-id')
    panels = []
    for base in ('c22', 'm10'):
        e = S['gap'][f'{base}/12']
        s = {f: [] for f in F}
        for c in CATS:
            v = e.get('by_category', {}).get(c)
            s['learned'].append(None if v is None else [v, None, None, f'{v:.1f} %'])
            for f in F[1:]:
                x = e['controls'][f]['by_category'].get(c)
                s[f].append(None if x is None else [x, None, None, f'{x:.1f} %'])
        panels.append(dict(title=f'{BASE[base]}, gap ×12', sub='runs solved (%)', series=s))
    return dict(fields=F, label=label, css=css, dash={}, xs=['same', 'adjacent', 'diagonal', 'quad–open', 'open–open'], xlabel='where the query’s ends lie', xname='kind', ylabel='runs solved (%)', ymax=100, panels=panels)


GROUPS = [['free ResNet (stripes)', '--s-uni'], ['bounded ResNet', '--s-con2'], ['separable or triangular', '--s-learn'], ['controls', '--s-id']]


def group(g):
    return 0 if g.startswith('free') else 1 if g.startswith('bounded') else 3 if g == 'control' else 2


NAMES = {'ntrl-demo': 'ntrl-demo', 'uniform B x 4': 'uniform B × 4', 'constructed x 6 (pm6)': 'hand-built × 6', 'earlier learned warp (la6b)': 'earlier learned warp'}


def scatter_charts(S):
    L = S['ablation']['links']
    base = dict(ymin=20, ymax=100, yticks=[20, 40, 60, 80, 100], groups=GROUPS)
    loss = [dict(x=q['ratio'], y=q['pct'], g=group(q['group']), name=q['name'], text=[['runs solved', f"{q['pct']:.1f} %"], ['largest / smallest run', f"{q['ratio']:.1f}"]]) for q in L['loss']]
    field = [dict(x=q['bumps'], y=q['pct'], g=group(q['group']), name=NAMES.get(q['name'], q['name']), text=[['runs solved', f"{q['pct']:.1f} %"], ['local minima per long path', f"{q['bumps']:.2f}"], ['speed correlation', f"{q['speed_corr']:.2f}"]]) for q in L['fields']]
    for q in loss:
        if q['name'].startswith('free ResNet'):
            q.update(label='stripes', dx=8, dy=3.5)
        if q['name'].startswith('the recipe:'):
            q.update(label='the recipe', dx=8, dy=3.5)
    for q in field:
        if q['name'].startswith('free ResNet, axis lines, final'):
            q.update(label='stripes', dx=-10, dy=-7, anchor='end')
        if q['name'] == 'ntrl-demo':
            q.update(label='ntrl-demo', dx=-6, dy=17, anchor='end')
    return [dict(base, title=f"The loss: {len(loss)} fits", sub=f"r = {L['r_ratio']:+.2f}".replace('-', '−'), xlabel='largest / smallest run after the fit (the loss, lower is better)', xmin=6, xmax=26, xticks=[10, 15, 20, 25], points=loss),
            dict(base, title=f"The field: {len(field)} fields", sub=f"r = {L['r_bumps']:+.2f}".replace('-', '−'), xlabel='local minima of T along a long true path', logx=True, xmin=0.03, xmax=30, xticks=[0.1, 1, 10], points=field)]


ROWS = (('TV + dilation', 'both', '--s-learn'), ('TV only', 'tv', '--s-learn'), ('dilation only', 'dil', '--s-learn'), ('neither', 'none', '--s-learn'), ('bound at TV’s range', 'range', '--s-con2'))


def seed_note(n):
    return 'one seed' if n == 1 else ''


def loo_chart(S):
    G = S['ablation']['grid']

    def row(name, css, pct, per, note=''):
        t = [['pooled', f'{pct:.1f} %']]
        if len(per) > 1:
            t.append(['per training seed', ' / '.join(f'{x:.1f}' for x in per)])
        if note:
            t.append([note, ''])
        return dict(name=name, css=css, lo=min(per), mid=pct, hi=max(per), text=t)
    base = dict(xlabel='runs solved (%)', min=0, max=100, log=False, ref=100, ticks=[0, 25, 50, 75, 100], css='--s-learn')
    panels = []
    for b, title in (('m10', '10×10 quadrants, ×12'), ('c22', '22×22 quadrants, ×12')):
        g = G[b]
        c = S['gap'][f'{b}/12']['controls']
        r = [row(name, css, g[key]['pct'], g[key]['per_seed'], seed_note(g[key]['seeds'])) for name, key, css in ROWS if key in g]
        r.append(row('hand-built', '--s-con', c['pmzw']['pct'], list(c['pmzw']['by_seed'].values())))
        r.append(row('ntrl-demo', '--s-id', c['id']['pct'], list(c['id']['by_seed'].values())))
        panels.append(dict(base, title=title, sub='field trained in z', rows=r))
    r = []
    for name, css, key in (('TV + dilation', '--s-learn', 'the recipe'), ('TV only', '--s-learn', 'recipe without dilation'), ('dilation only', '--s-learn', 'recipe without TV'), ('neither', '--s-learn', 'bound 2, no TV'), ('bound at TV’s range', '--s-con2', 'bound 0.70, no TV'),
                           ('hand-built', '--s-con', 'constructed x 6 (pm6)'), ('ntrl-demo', '--s-id', 'ntrl-demo')):
        q = q1200(S, key)
        if q:
            r.append(row(name, css, q['pct'], q['per_seed']))
    panels.append(dict(base, title='Plus maze, 1,200 queries', sub='field trained in x', rows=r))
    return panels


def window_chart(S):
    A = S['ablation']
    keys = ['tight', 'b125', 'range', 'b250', 'dil']
    xs = ['0.70', '1.25', 'TV’s range', '2.50', '3.00']
    series = {}
    for b in ('m10', 'c22'):
        g = A['grid'][b]
        pt = lambda c: None if c not in g else [g[c]['pct'], min(g[c]['per_seed']), max(g[c]['per_seed']), f"{g[c]['pct']:.1f} % ({g[c]['seeds']} seed{'s' if g[c]['seeds'] > 1 else ''})"]
        series[b] = [pt(c) for c in keys]
        series[b + 'tv'] = [[g['both']['pct'], None, None, f"{g['both']['pct']:.1f} % (3 seeds)"] for _ in keys]
    label = dict(m10='10×10, bound', c22='22×22, bound', m10tv='10×10, TV + dilation', c22tv='22×22, TV + dilation')
    return dict(fields=['m10', 'c22', 'm10tv', 'c22tv'], label=label, css=dict(m10='--s-learn', c22='--s-con', m10tv='--s-learn', c22tv='--s-con'), dash=dict(m10tv='4 3', c22tv='4 3'), xs=xs, xlabel='stretch bound (all with dilation, no TV)', xname='bound',
                ylabel='runs solved (%)', ymax=100, nodots=['m10tv', 'c22tv'], panels=[dict(title='Gap ×12: how wide the window is', sub='runs solved (%)', series=series)])


def fail_chart(S):
    G = S['ablation']['grid']
    panels = []
    for b, title in (('m10', '10×10 quadrants, ×12'), ('c22', '22×22 quadrants, ×12')):
        rows = []
        for name, key in (('TV + dilation', 'both'), ('bound at TV’s range', 'range'), ('TV only', 'tv'), ('dilation only', 'dil'), ('neither', 'none'), ('bound 0.70', 'tight')):
            g = G[b].get(key)
            if g:
                rows.append(dict(name=name, parts=[g['pct'], g['wall'], g['stall']], note=f"{g['seeds']} seed{'s' if g['seeds'] > 1 else ''}"))
        panels.append(dict(title=title, sub='share of runs (%)', rows=rows))
    return dict(parts=[['solved', '--s-learn'], ['walked into a wall', '--s-id'], ['never arrived', '--s-con']], panels=panels)


def cell_table(S):
    G = S['ablation']['grid']
    names = dict(both='TV + dilation', tv='TV only', dil='dilation only (bound 3)', none='neither (bound 3)', tight='bound 0.70 + dilation', b125='bound 1.25 + dilation', range='bound at TV’s range + dilation', b250='bound 2.5 + dilation')
    rows = []
    for b in ('m10', 'c22'):
        for key, name in names.items():
            g = G[b].get(key)
            if not g:
                continue
            p = lambda v, k=1: '–' if v is None else f'{100 * v:.{k}f}'
            n = lambda v, k=2: '–' if v is None else f'{v:.{k}f}'
            rows.append([f'{BASE[b]}, {name}', ' / '.join(f'{x:.1f}' for x in g['per_seed']), ' / '.join(map(str, g['wall_runs'])), ' / '.join(map(str, g['stall_runs'])), n(g['range']), n(g['tilt']), n(g['band_q'], 3), n(g['s']), p(g['two_in']), p(g['two_out']), n(g['corr_z']), p(g['into'])])
    return table(['fit', 'solved (%) per seed', 'walked into a wall (of 900)', 'never arrived (of 900)', 'log-density range', 'tilt in a quadrant', 'band / quadrant', 'box in z', 'slow zone under 2 steps, quadrant (%)', 'same, outside (%)', 'speed correlation in z', 'descent into the wall (%)'], rows, txt=(0,))


def plus_table(S):
    rows = []
    for name, v in S['plus_controls'].items():
        r = v['rows']
        rows.append([f'control: {name}', ' / '.join(str(x) for x in v['per_seed']), f1(v['pct']), sum(x['arrived_wall'] for x in r), sum(x['not_arrived'] for x in r), f"{sum(x['bumps'] for x in r) / len(r):.2f}"])
    for v in S['plus'].values():
        r = v['rows']
        rows.append([v['label'], ' / '.join(str(x) for x in v['per_seed']), f1(v['pct']), sum(x['arrived_wall'] for x in r), sum(x['not_arrived'] for x in r), f"{sum(x['bumps'] for x in r) / len(r):.2f}"])
    return table(['field', 'runs solved per seed (of 720)', '%', 'arrived through a wall', 'never arrived', 'local minima per long path'], rows, txt=(0,))


def gap_table(S):
    rows = []
    for base in ('c22', 'm10'):
        for k in GAPS:
            e = S['gap'][f'{base}/{k}']
            c = e['controls']
            seeds = ' / '.join(f'{x:.1f}' for x in e.get('per_seed', [])) or '–'
            rows.append([f'{BASE[base]}, ×{k}', f1(e.get('pct')), seeds, f1(c['id']['pct']), f1(c['pm']['pct']), f1(c['pmz']['pct']), f1(c['pmzw']['pct']), f1(c['pmzs']['pct'])])
    return table(['maze, gap', 'learned', 'per seed', 'ntrl-demo', 'hand-built, on the maze', 'hand-built, real pairs', 'hand-built, warped pairs', 'band shrunk by hand'], rows, txt=(0,))


def fit_table(S):
    rows = []
    for f in S['fits']:
        rows.append([f['label'], f"{100 * f['det_positive']:.2f}", 'yes' if f['outline'] else 'no', f"{f['sv'][0]:.3f} – {f['sv'][1]:.1f}", f"{f['aniso_median']:.1f} / {f['aniso_q99']:.0f}", f"{f['rotation']:.1f}", f"{f['ratio']:.1f}", f"{f['log_std']:.2f}"])
    return table(['warp', 'det J > 0 (%)', 'outline simple', 'singular values', 'anisotropy median / q99', 'rotation (°)', 'largest / smallest run', 'spread of log run'], rows, txt=(0,))


def run_table(B):
    rows = [[k, v] for k, v in sorted(B.items())]
    return table(['job', 'registered run'], rows, txt=(0, 1))


def main():
    import text as T
    S = json.load(open(OUT / 'summary.json'))
    B = json.load(open(OUT / 'runs.json'))
    tpl = (HERE / 'page_template.html').read_text()
    data = dict(tv=tv_chart(S), bound=bound_chart(S), plus=plus_panels(S), gap=gap_chart(S), cats=cat_chart(S), scatter=scatter_charts(S), loo=loo_chart(S), window=window_chart(S), fail=fail_chart(S))
    blocks = T.blocks(S)
    blocks.update(PLUSTABLE=plus_table(S), GAPTABLE=gap_table(S), FITTABLE=fit_table(S), RUNTABLE=run_table(B), CELLTABLE=cell_table(S))
    for k, v in blocks.items():
        tpl = tpl.replace(f'<!--{k}-->', v)
    tpl = tpl.replace('/*DATA*/', json.dumps(data, ensure_ascii=False))
    missing = [w for w in tpl.split('<!--')[1:] if w.split('-->')[0].isupper()]
    assert not missing, [m[:20] for m in missing]
    (OUT / 'line_warp.html').write_text(tpl)
    print(f"wrote {OUT / 'line_warp.html'} ({len(tpl) / 1024:.0f} kB)")


if __name__ == '__main__':
    main()
