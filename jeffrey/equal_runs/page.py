"""The findings page, from report/summary.json: python page.py -> report/equal_runs.html (images next to it)."""
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'report'
S = json.load(open(OUT / 'summary.json'))
T10 = {r['name']: r for r in S['p10']}
T12 = {r['name']: r for r in S['g12']}
SC = {r['key']: r for r in S['scenes']}
TITLE = '2026-11-07 Equal-Run Warp'
C_ID, C_LINE, C_FIX, C_DENS, C_REF = '--s-id', '--s-uni', '--s-con', '--s-learn', '--s-con2'
GROUP_CSS = dict(none=C_ID, literal=C_LINE, fixed=C_LINE, sep=C_ID, any=C_FIX, pair=C_FIX, old=C_REF, limit=C_REF, density=C_DENS)


def n(v, d=2):
    return '–' if v is None else f'{v:.{d}f}'


def row(T, name):
    return T.get(name)


def dot(r, key, label, css, d=3):
    return dict(x=r[key], css=css, label=label, value=n(r[key], d))


def r2_rows(T, names, key='wall_r2', label='R² near walls'):
    """Rows of a learnability dot plot: one dot per method."""
    out = []
    for name, short in names:
        r = row(T, name)
        if r and r.get(key) is not None:
            out.append(dict(name=short, key=r['group'] == 'density', dots=[dict(x=max(r[key], 0.0), css=GROUP_CSS[r['group']], label=label, value=n(r[key], 3))]))
    return out


def table(head, rows, txt=()):
    h = ''.join(f'<th class="{"txt" if i in txt else ""}">{c}</th>' for i, c in enumerate(head))
    b = ''.join('<tr>' + ''.join(f'<td class="{"txt" if i in txt else "num" if i else ""}">{c}</td>' for i, c in enumerate(r)) + '</tr>' for r in rows)
    return f'<div class="tablewrap"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def mark(v):
    return {True: '<span class="yes">pass</span>', False: '<span class="no">fail</span>', None: '<span class="na">–</span>'}[v]


def rule_table():
    names = ['No warp', 'Every run of a line equal, as asked (coupling)', 'Every run of a line equal, as asked (flow)', 'Fixed target 0.10 (coupling)', 'Fixed target 0.10 (flow)', 'Fixed target 0.10, every angle (coupling)', 'Old loss with shear (coupling)',
             'Old loss, update size capped', 'Old loss, stiff near walls', 'Equal runs, worst run of a line weighted (power 2)', 'Equal runs, stiff near walls', 'Equal runs, every angle, stiff near walls', 'Long runs pushed, stiff near walls, stretch bound 8', 'Long runs pushed, stiff near walls, stretch bound 8, every angle', 'Cartogram, h 2, ε 0.3', 'Learned flow, h 2, ε 0.3, transport 1']
    rows = []
    for k in names[1:]:
        r = row(T10, k)
        if r:
            t = r['tests']
            rows.append([html.escape(k), mark(t['runs']), mark(t['stripes']), mark(t['valid']), mark(t['speed']), mark(t['walls']), n(r.get('wall_r2'), 3), n(r['r2'], 3), n(r['spacing_cv']), n(r['aniso_q50'], 1)])
    return table(['Method', 'Runs more equal', 'No stripes', 'Valid map', 'Speed as learnable', 'Walls as learnable', 'R² near walls', 'R² over the box', 'Peak spacing spread', 'Anisotropy'], rows)


def all_rows(T):
    rows = []
    for r in T:
        seeds = f" ({r['seeds']} seeds)" if r.get('seeds', 1) > 1 else ''
        rows.append([html.escape(r['name']) + seeds, n(r.get('wall_r2'), 3), n(r.get('wall_r2_quadrants'), 3), n(r.get('wall_r2_band'), 3), n(r.get('r2'), 3), n(r.get('zone_q05')), n(r.get('zone_q50')), n(r.get('zone_log_std')), n(r.get('cv_line')), n(r.get('spacing_cv')), n(r.get('long_share')), n(r.get('any_long_share')), n(r.get('any_stripe')),
                     n(r.get('crossing_cv')), n(r.get('runs_per_line'), 1), n(r.get('sv_q01')), n(r.get('sv_q99'), 1), n(r.get('aniso_q50'), 1), n(r.get('det_positive'), 3), r['run']])
    return table(['Method', 'R² near walls', '… quadrant walls', '… band walls', 'R² over the box', 'Wall zone q05', 'Wall zone median', 'Wall zone spread', 'Run spread in a line', 'Peak spacing spread', 'Long-run share', 'Long-run share, every angle', 'Anisotropy of runs, every angle', 'Crossing spread', 'Runs per line', 'Stretch q01', 'Stretch q99', 'Anisotropy', 'det J > 0', 'Run'], rows, txt=(19,))


def scene_rows():
    rows = []
    for r in S['scenes']:
        i, c, f, s, l = r.get('identity', {}), r.get('carto'), r.get('flow'), r.get('sep'), r.get('lines')
        w = lambda q: n(q.get('wall_r2'), 3) if q else '–'
        rows.append([html.escape(r['name']), w(i), w(c), w(f), w(s), w(l), n(i.get('r2'), 3), n(c and c['r2'], 3), n(c and c['zone_q05']), n(c and c['zone_q50']), n(s and s['zone_q50']), n(i.get('crossing_cv')), n(c and c['crossing_cv']), n(i.get('spacing_cv')), n(c and c['spacing_cv']), n(c and c['aniso_q50'], 1), (c or f)['run']])
    return table(['Scene', 'Near walls: no warp', 'cartogram', 'flow', 'separable flow', 'line loss, stiff', 'Over the box: no warp', 'cartogram', 'Wall zone q05', 'Wall zone median', 'Median, separable', 'Crossing spread, no warp', 'Crossing spread', 'Peak spacing, no warp', 'Peak spacing', 'Anisotropy', 'Run'], rows, txt=(16,))


def cal_rows():
    rows = []
    for b, name in (('plus', 'plus maze'), ('m10', '10×10 quadrants ×12'), ('c22', '22×22 quadrants ×12')):
        for r in S['calibration'][b]['rows']:
            rows.append([name, html.escape(str(r['name'])), n(r['plan'], 1), n(r.get('wall'), 3), n(r['r2'], 3), n(r['zone_q05']), n(r['zone_q50']), r['run'] or '–'])
    return table(['Maze', 'Map', 'Planning %', 'R² near walls', 'R² over the box', 'Wall zone q05', 'Wall zone median', 'Fit run'], rows, txt=(1, 7))


def scatter(b, title):
    c = S['calibration'][b]
    pts = [dict(x=max(r['wall'], -0.05), y=r['plan'], css=C_REF if r['key'] != 'identity' else C_ID, name=str(r['name']), text=[['R² near walls', n(r['wall'], 3)], ['planning', f"{r['plan']:.1f} %"], ['R² over the box', n(r['r2'], 3)]]) for r in c['rows'] if r.get('wall') is not None]
    return dict(title=title, sub=f"rank correlation {c['wall']['rank']:+.2f}, {c['wall']['n']} maps", xmin=-0.1, xmax=1.0, ymin=0, ymax=100, yticks=[0, 25, 50, 75, 100], xticks=[0, 0.25, 0.5, 0.75, 1.0], xlabel='R² near walls (no-warp maps at or below 0 sit at the left edge)', points=pts)


def data():
    d = {}
    # along the axes against every angle
    rows = []
    for name, short in (('No warp', 'No warp'), ('Every run of a line equal, as asked (coupling)', 'Every run of a line equal'), ('Fixed target 0.10 (coupling)', 'Fixed target, axis lines'), ('Fixed target 0.10, every angle (coupling)', 'Fixed target, lines at every angle'), ('Cartogram, h 2, ε 0.3', 'Density form')):
        r = row(T10, name)
        if r and r.get('any_log_std_len') is not None:
            rows.append(dict(name=short, dots=[dict(x=r['log_std_len'], css=C_LINE, label='along the axes', value=n(r['log_std_len'])), dict(x=r['any_log_std_len'], css=C_DENS, label='at every angle', value=n(r['any_log_std_len']))]))
    d['axes'] = [dict(title='Spread of log run length, 10×10 maze', sub='lower is more equal', xmin=0.3, xmax=1.2, ticks=[0.3, 0.6, 0.9, 1.2], xlabel='length-weighted', legend=[['along the axes', C_LINE], ['at every angle', C_DENS]], groups=[dict(rows=rows)])]
    # learnability on the 10 x 10
    names = [('No warp', 'No warp'), ('Every run of a line equal, as asked (coupling)', 'Every run of a line equal (coupling)'), ('Every run of a line equal, as asked (flow)', 'Every run of a line equal (flow)'), ('All runs of a batch equal, target follows the runs (coupling)', 'Target follows the runs'), ('Same on a log scale (coupling)', 'Same, log scale'),
             ('Fixed target 0.10 (coupling)', 'Fixed target, axes'), ('Fixed target 0.10, stretch bound 4 (coupling)', 'Fixed target, stretch bound 4'), ('Fixed target 0.10, no shear (separable)', 'Fixed target, no shear'), ('Fixed target 0.10, every angle (coupling)', 'Fixed target, every angle'), ('Pair form, shift 0.10', 'Pair form'),
             ('Old loss with shear (coupling)', 'Old loss, shear'), ('Old loss, update size capped', 'Old loss, update capped'), ('Equal runs, worst run of a line weighted (power 2)', 'Worst run weighted'), ('Old loss, stiff near walls', 'Old loss, stiff near walls'), ('Equal runs, stiff near walls', 'Equal runs, stiff near walls'), ('Equal runs, every angle, stiff near walls', 'Same, every angle'), ('Long runs pushed, stiff near walls, stretch bound 8', 'Line loss with limits'), ('Long runs pushed, stiff near walls, stretch bound 8, every angle', 'Same, every angle'),
             ('Cartogram, h 2, ε 0.3', 'Density form'), ('Learned flow, h 2, ε 0.3, transport 1', 'Learned density form')]
    ident = row(T10, 'No warp')
    d['r210'] = [dict(title='Speed learned near walls, 10×10 maze', sub='R² on fixed near-wall points', xmin=0.4, xmax=1.0, ticks=[0.4, 0.6, 0.8, 1.0], xlabel='R² near walls', refs=[dict(x=ident['identity'].get('wall_r2'), label=f"no warp {ident['identity'].get('wall_r2') or 0:.3f}")] if ident['identity'].get('wall_r2') is not None else [],
                     legend=[['line losses as asked', C_LINE], ['every angle, pair form', C_FIX], ['old loss and limits', C_REF], ['density form', C_DENS]], groups=[dict(rows=r2_rows(T10, names[1:]))])]
    # x12: R2 and wall zone
    names = [('Fixed target 0.10, along axes (coupling)', 'Fixed target, axes'), ('Fixed target 0.10, every angle (coupling)', 'Fixed target, every angle'), ('Fixed target 0.10, no shear (separable)', 'Fixed target, no shear'), ('Pair form, shift 0.10', 'Pair form'), ('Old loss with shear', 'Old loss, shear'), ('Old loss, others held still within an update', 'Old loss, others held still'), ('Old loss, update size capped', 'Old loss, update capped'), ('Old loss, stiff near walls', 'Old loss, stiff near walls'),
             ('Equal runs, worst run of a line weighted (power 2)', 'Worst run weighted'), ('Equal runs, only the worst 5 % of runs pushed', 'Worst 5 % of runs only'), ('Equal runs, stiff near walls', 'Equal runs, stiff near walls'), ('Only long runs pushed, target from run count, stiff near walls', 'Only long runs, stiff near walls'), ('Long runs pushed, stiff near walls, stretch bound 8', 'Line loss with limits'), ('Long runs pushed, stiff near walls, stretch bound 8, every angle', 'Same, every angle'), ('Cartogram, h 2, ε 0.3', 'Density form'), ('Learned flow, transport 1 (small network, 3 seeds)', 'Learned density form')]
    rec = S['recipe'].get('both', {})
    both_r2 = S['noise']['g12']['recipe']['mean'] if 'g12' in S.get('noise', {}) else rec.get('r2')
    legend = [['line losses as asked', C_LINE], ['every angle, pair form', C_FIX], ['old loss and limits', C_REF], ['density form', C_DENS]]
    recw = S['regions'].get('recipe', {}).get('wall_mean', 0.97)
    d['g12a'] = [dict(title='Speed learned near walls, ×12 maze', sub='R² on fixed near-wall points', xmin=0, xmax=1.0, ticks=[0, 0.25, 0.5, 0.75, 1.0], xlabel='R² near walls', refs=[dict(x=recw, label=f'last study\'s recipe {recw:.3f}')], legend=legend, groups=[dict(rows=r2_rows(T12, names))])]
    rows = []
    for name, short in names:
        r = row(T12, name)
        if r and r.get('zone_q05') is not None:
            rows.append(dict(name=short, key=r['group'] == 'density', dots=[dict(x=r['zone_q05'], css=GROUP_CSS[r['group']], label='wall zone q05', value=n(r['zone_q05']))]))
    d['g12b'] = [(dict(title='Thinnest wall zones, ×12 maze', sub='5th percentile stretch, higher is better', xmin=0, xmax=6, ticks=[0, 2, 4, 6], xlabel='stretch of the slow zone at walls', refs=[dict(x=rec.get('zone_q05', 4.74), label=f"last study's recipe {rec.get('zone_q05', 4.74):.2f}")], legend=legend, groups=[dict(rows=rows)]))]
    # scenes
    rows = []
    for r in S['scenes']:
        dots = []
        for tag, css, label in (('identity', C_ID, 'no warp'), ('sep', C_FIX, 'separable flow'), ('lines', C_REF, 'line loss with limits'), ('flow', C_LINE, 'learned density form'), ('carto', C_DENS, 'density form')):
            q = r.get(tag) or {}
            if q.get('wall_r2') is not None:
                dots.append(dict(x=max(q['wall_r2'], 0.0), css=css, label=label, value=n(q['wall_r2'], 3)))
        if dots:
            rows.append(dict(name=r['name'], dots=dots))
    d['scenes'] = [dict(title='Speed learned near walls, by scene', sub='R²; values at or below 0 sit at 0', xmin=0, xmax=1.0, ticks=[0, 0.25, 0.5, 0.75, 1.0], xlabel='R² near walls', legend=[['no warp', C_ID], ['separable flow', C_FIX], ['line loss with limits', C_REF], ['learned density form', C_LINE], ['density form', C_DENS]], groups=[dict(rows=rows)])]
    # arm
    a = S['arm']
    if a:
        rows = [dict(name=r['name'], dots=[dict(x=r['r2'], css=C_DENS, label='R²', value=n(r['r2'], 3))]) for r in a]
        d['arm'] = [dict(title='6-D arm: speed learnability against blur', sub='R² at s = 1', xmin=0.2, xmax=0.7, ticks=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7], xlabel='R²', refs=[dict(x=a[0]['r2_identity'], label=f"no warp {a[0]['r2_identity']:.2f}")], groups=[dict(rows=rows)])]
    lines = {}
    gp = []
    for b, name in (('m10', '10×10 quadrants'), ('c22', '22×22 quadrants')):
        g = {r['k']: r for r in S['gaps'].get(b, [])}
        ks = [1, 2, 4, 8, 12]
        rz = {int(k): v for k, v in S['recipe_zone'].get(b, {}).items()}
        gp.append(dict(title=name, sub='wall-zone stretch, median', xs=[f'×{k}' for k in ks], xlabel='gap', ymax=8, yticks=[0, 2, 4, 6, 8], series=[dict(name="last study's recipe", css=C_REF, values=[rz.get(k) for k in ks]), dict(name='density form', css=C_DENS, values=[g[k]['zone_q50'] if k in g else None for k in ks])]))
    lines['gaps'] = gp
    sc = {'calib': [scatter('plus', 'Plus maze'), scatter('m10', '10×10 quadrants, ×12'), scatter('c22', '22×22 quadrants, ×12')]}
    return dict(dots=d, scatter=sc, lines=lines)


BODY = (HERE / 'page_text.html').read_text() if (HERE / 'page_text.html').exists() else ''


def fill(text):
    """{name} placeholders of page_text.html from the tables built here."""
    import page_values as PV
    v = PV.values(S, T10, T12, SC)
    v.update(RULE=rule_table(), ALL10=all_rows(S['p10']), ALL12=all_rows(S['g12']), SCENES=scene_rows(), CAL=cal_rows())
    for k, x in v.items():
        text = text.replace('{{' + k + '}}', str(x))
    return text


def main():
    t = (HERE / 'page_template.html').read_text()
    t = t.replace('<!--TITLE-->', TITLE).replace('<!--CSS-->', (HERE / 'page_css.txt').read_text()).replace('<!--BODY-->', fill(BODY)).replace('/*DATA*/', json.dumps(data()))
    (OUT / 'equal_runs.html').write_text(t)
    print('wrote', OUT / 'equal_runs.html', len(t))


if __name__ == '__main__':
    main()
