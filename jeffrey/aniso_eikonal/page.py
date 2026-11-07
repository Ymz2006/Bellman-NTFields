"""The findings page (report/aniso_eikonal.html) from report/summary.json (report.py), the runs' rows.npz and text_blocks.py.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/aniso_eikonal/report.py && ~/.venvs/bellman-ntfield/bin/python jeffrey/aniso_eikonal/page.py

Charts are drawn by the page's own script from the DATA blob; every number in the text is read from the summary.
"""
import html
import json
import math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'report'
CATS = ['same quadrant', 'adjacent quadrants', 'diagonal quadrants', 'quadrant - open', 'open - open']
CAT_SHORT = ['same quadrant', 'adjacent quadrants', 'diagonal quadrants', 'quadrant – open', 'open – open']
GAPS = [1, 2, 4, 8, 12]
BASE_NAME = {'c22': '22×22 quadrants', 'm10': '10×10 quadrants'}
REF, PLAIN = 'ziso-open', 'zaniso-Uquad'
# the full recipe at each gap: the first of these variants that has runs (multi-scale head; its coarse factor is the only thing set per gap)
RECIPE_ORDER = ['zaniso-Uquad-open-gs4x1+4x2+4x4+4x8-pwtauz-lb10', 'zaniso-Uquad-open-gs8x1+8x4-pwtauz-lb10', 'zaniso-Uquad-open-gs8x1+8x6-pwtauz-lb10',
                'zaniso-Uquad-open-gs8x1+8x12-pwtauz-lb10', 'zaniso-Uquad-open-pwtauz-lb10', 'zaniso-Uquad-open-lb10']
RECIPE_AT = {}        # (base, gap) -> variant, to pin a choice; filled by recipe()


def recipe(base, k):
    if (base, k) not in RECIPE_AT:
        have = [v for v in RECIPE_PIN.get(k, []) if (base, k, v) in POOL]
        RECIPE_AT[(base, k)] = have[0] if have else None
    return RECIPE_AT[(base, k)]


# gap -> the recipe's variant: inner product, open-start pairs, lower bound 10, pair weight tau x rho, head 8 fine + 8 coarse groups
# (coarse factor 4 up to gap x4, 6 beyond)
RECIPE_PIN = {1: ['zaniso-Uquad-open-gs8x1+8x4-pwtauz-lb10'], 2: ['zaniso-Uquad-open-gs8x1+8x4-pwtauz-lb10'], 4: ['zaniso-Uquad-open-gs8x1+8x4-pwtauz-lb10'],
              8: ['zaniso-Uquad-open-gs8x1+8x6-pwtauz-lb10'], 12: ['zaniso-Uquad-open-gs8x1+8x6-pwtauz-lb10']}
# (key, label, css token, dash, planner shown in the charts): hue = the field
RES = 'zres-Uquad-open-far-lb10'      # the head trained with the plain length + the far term trained with the inner product on the sum
FIELDS = [(REF, 'trained on the stretched space (plain length), same pairs', '--s-iso', '', 'z'), ('RECIPE', 'inner product, one field, full recipe', '--s-split', '', 'z'),
          ('RES', 'two-part field: stretched-space head + learned correction', '--s-post', '', 'z'), ('RESH', '… its head alone (the reference inside the same runs)', '--s-post', '4 3', 'zh'),
          (PLAIN, 'inner product alone', '--s-an', '', 'z')]
PLANNER = {f[0]: f[4] for f in FIELDS}
SCATTER = [REF, PLAIN, 'RECIPE', 'RES']


def resolve(base, k, key):
    return recipe(base, k) if key == 'RECIPE' else RES if key in ('RES', 'RESH') else key


UNITS = [('zaniso-Ubox', 'maze side'), ('zaniso-U44.2', '44'), ('zaniso-U22.1', '22'), ('zaniso-U11.05', '11'), ('zaniso-U7.37', '7.4'), ('zaniso-Uquad', '3.7')]

S = json.load(open(OUT / 'summary.json'))
POOL = {(p['base'], p['k'], p['variant']): p for p in S['pooled']}
RUNS = S['runs']
VER = S['verify']


def P(base, k, variant):
    return POOL.get((base, k, variant))


def plan(base, k, variant, planner=None):
    """The pooled plan entry: the z planner when it was run (fields on the warp), else x."""
    p = P(base, k, variant)
    if not p:
        return None
    for pl in ([planner] if planner else ['z', 'x']):
        if pl in p['plans']:
            return p['plans'][pl]
    return None


def pc(base, k, variant, planner=None, cat=None):
    e = plan(base, k, variant, planner)
    if not e:
        return None
    return e['by_category'].get(cat) if cat else e['pct']


def f1(v, d=1):
    return '–' if v is None or (isinstance(v, float) and not math.isfinite(v)) else f'{v:.{d}f}'


def run_of(base, k, variant, seed=87358):
    for r in RUNS:
        if (r['base'], r['k'], r['variant'], r['seed']) == (base, k, variant, seed):
            return r
    return None


def rows_of(run):
    return np.load(ROOT / 'results' / f'registered_{run["id"]}' / 'rows.npz')


# --- chart data ---------------------------------------------------------------------------------------------------------
def line_point(base, k, key, cat=None):
    variant = resolve(base, k, key)
    if variant is None:
        return None
    e = plan(base, k, variant, PLANNER.get(key)) or (None if key == 'RESH' else plan(base, k, variant))
    if not e:
        return None
    p = P(base, k, variant)
    v = e['by_category'].get(cat) if cat else e['pct']
    if v is None:
        return None
    if cat:
        return [v, None, None, f'{v:.0f} %']
    lo, hi = min(e['per_seed']), max(e['per_seed'])
    n = len(p['seeds'])
    return [v, lo if n > 1 else None, hi if n > 1 else None, f'{v:.1f} %' + (f'  ({lo:.0f}–{hi:.0f}, {n} seeds)' if n > 1 else '  (1 seed)')]


def main_data():
    fields = [f for f in FIELDS if any(line_point(b, k, f[0]) for b in BASE_NAME for k in GAPS)]
    def td(panels, off=()):
        return dict(fields=[f[0] for f in fields], label={f[0]: f[1] for f in fields}, css={f[0]: f[2] for f in fields}, dash={f[0]: f[3] for f in fields},
                    xs=[f'×{k}' for k in GAPS], xname='gap', xlabel='gap between the quadrants (× the 6-cell band)', ylabel='queries solved (%)', ymax=100, panels=panels, off=list(off))
    main = td([dict(title=BASE_NAME[b], sub='all queries solved (%)', series={f[0]: [line_point(b, k, f[0]) for k in GAPS] for f in fields}) for b in BASE_NAME])
    cats = td([dict(title=CAT_SHORT[i], sub=BASE_NAME[b], series={f[0]: [line_point(b, k, f[0], c) for k in GAPS] for f in fields}) for b in BASE_NAME for i, c in enumerate(CATS)],
              )

    def acc_point(b, k, key):
        if key == 'RESH':                 # the head alone is a stretched-space field: its accuracy is the reference's
            return None
        v = resolve(b, k, key)
        p = P(b, k, v) if v else None
        if not p or not p.get('time_accuracy'):
            return None
        w = 100 * p['time_accuracy']['within_10pct']
        return [w, None, None, f'{w:.0f} %  (median ratio {p["time_accuracy"]["ratio_median"]:.2f})']
    acc = td([dict(title=BASE_NAME[b], sub='travel time within 10 % of the truth (% of queries)', series={f[0]: [acc_point(b, k, f[0]) for k in GAPS] for f in fields}) for b in BASE_NAME])
    acc['ylabel'] = 'queries within 10 % (%)'
    return main, cats, acc


MARGIN_BINS = ['0.02-0.06', '0.06-0.12', '0.12-0.175']
FAR_GOALS = ['adjacent quadrant', 'diagonal quadrant']


def feel(base, k, variant, goals=FAR_GOALS, seed=87358):
    """wallfeel.py: share (%) of near-wall poses inside the quadrants where the field's travel time rises into the wall,
    for goals of the given classes; None if the run was not analysed."""
    r = run_of(base, k, variant, seed) if variant else None
    f = OUT / 'wallfeel' / f'{r["id"]}.json' if r else None
    if not f or not f.exists():
        return None
    w = json.load(open(f))['feel']
    n = sum(w[g][b]['poses'] for g in goals for b in MARGIN_BINS)
    return 100 * sum(w[g][b]['rises'] * w[g][b]['poses'] for g in goals for b in MARGIN_BINS) / n


def feel_raw(base, k, variant, seed=87358):
    r = run_of(base, k, variant, seed) if variant else None
    f = OUT / 'wallfeel' / f'{r["id"]}.json' if r else None
    return json.load(open(f)) if f and f.exists() else None


def arrival(base, k, variant, planner='z', seed=87358):
    """wallfeel.py: open -> quadrant plans, median distance (cells) from the true path's entrance where the plan first reaches the goal quadrant's outline."""
    w = feel_raw(base, k, variant, seed)
    return ((w or {}).get('arrival') or {}).get(planner, {}).get('median_cells')


def stalled_open(base, k, variant, cats=('quadrant - open', 'open - open'), seed=87358):
    """wallfeel.py: plans (planning seed 0) in these categories that never arrive and end in the open band without touching a wall."""
    w = feel_raw(base, k, variant, seed)
    return None if not w else sum(w['fail_where'][c]['stalled']['band'] for c in cats)


def fail_where(base, k, variant, cats=('adjacent quadrants', 'diagonal quadrants'), seed=87358):
    """wallfeel.py: of the failed plans (planning seed 0) in these categories, (failed, first contact inside the start quadrant)."""
    r = run_of(base, k, variant, seed) if variant else None
    f = OUT / 'wallfeel' / f'{r["id"]}.json' if r else None
    if not f or not f.exists():
        return None
    w = json.load(open(f))['fail_where']
    failed = sum(w[c]['queries'] - w[c]['solved'] for c in cats)
    return failed, sum(w[c]['first_contact']['start quadrant'] for c in cats)


def feel_data():
    fields = [f for f in FIELDS if f[0] != 'RESH' and any(feel(b, k, resolve(b, k, f[0])) is not None for b in BASE_NAME for k in GAPS)]
    if not fields:
        return None
    def pt(b, k, key, goals):
        v = feel(b, k, resolve(b, k, key), goals)
        return None if v is None else [v, None, None, f'{v:.0f} %']
    panels = [dict(title=BASE_NAME[b], sub=sub, series={f[0]: [pt(b, k, f[0], goals) for k in GAPS] for f in fields})
              for b in BASE_NAME for goals, sub in ((['same quadrant'], 'goal in the same quadrant'), (FAR_GOALS, 'goal in another quadrant'))]
    return dict(fields=[f[0] for f in fields], label={f[0]: f[1] for f in fields}, css={f[0]: f[2] for f in fields}, dash={f[0]: f[3] for f in fields},
                xs=[f'×{k}' for k in GAPS], xname='gap', xlabel='gap between the quadrants (× the 6-cell band)', ylabel='near-wall poses where the travel time rises into the wall (%)', ymax=100, panels=panels, ref=50, reflabel='50 %: no better than a coin flip')


RESFULL = 'zres-Uquad-open-far-lb10-full1s'    # the correction trained alone on the gap study's full-loss fields (frozen), target floored at the head's own slope
RESFULL2 = 'zres-Uquad-open-far-lb10-full'     # ... with the plain two-sided term (the control)


def full_data():
    """The correction on existing fields: the gap study's full-loss stretched-space fields with and without it."""
    kinds = [('zh', 'the full-loss field itself (stretched space)', '--s-iso', '4 3'), ('z', 'the same field + learned correction', '--s-post', ''),
             ('z2', '… with the plain two-sided term (control)', '--s-an', '2 3')]
    if not any(P(b, k, RESFULL) for b in BASE_NAME for k in GAPS):
        return None
    var = {'zh': (RESFULL, 'zh'), 'z': (RESFULL, 'z'), 'z2': (RESFULL2, 'z')}

    def pt(b, k, key):
        v_, pl = var[key]
        v = pc(b, k, v_, pl) if P(b, k, v_) else None
        return None if v is None else [v, None, None, f'{v:.1f} %']

    def acc(b, k, key):
        v_, pl = var[key]
        p_ = P(b, k, v_)
        if not p_ or key == 'zh' or not p_.get('time_accuracy'):
            return None
        w = 100 * p_['time_accuracy']['within_10pct']
        return [w, None, None, f'{w:.0f} %']
    panels = [dict(title=BASE_NAME[b], sub='queries solved (%)', series={key: [pt(b, k, key) for k in GAPS] for key, _, _, _ in kinds}) for b in BASE_NAME]
    panels += [dict(title=BASE_NAME[b], sub='travel time within 10 % of the truth (% of queries)', series={key: [acc(b, k, key) for k in GAPS] for key, _, _, _ in kinds}) for b in BASE_NAME]
    return dict(fields=[x[0] for x in kinds], label={x[0]: x[1] for x in kinds}, css={x[0]: x[2] for x in kinds}, dash={x[0]: x[3] for x in kinds},
                xs=[f'×{k}' for k in GAPS], xname='gap', xlabel='gap between the quadrants (× the 6-cell band)', ylabel='queries solved (%)', ymax=100, panels=panels)


def unit_data(k=12):
    kinds = [('all', 'all queries', '--ink2', ''), ('same quadrant', 'same quadrant', '--s-ref', ''), ('quadrant - open', 'quadrant – open', '--s-post', ''), ('open - open', 'open – open', '--s-lb', '')]
    panels = []
    for b in BASE_NAME:
        series = {}
        for key, _, _, _ in kinds:
            pts = []
            for v, _ in UNITS:
                e = plan(b, k, v)
                if not e:
                    pts.append(None)
                    continue
                val = e['pct'] if key == 'all' else e['by_category'].get(key)
                p = P(b, k, v)
                stopped = any((run_of(b, k, v, s) or {}).get('stopped') for s in p['seeds'])
                pts.append([val, None, None, f'{val:.0f} %' + (' (training broke down)' if stopped else '')])
            series[key] = pts
        panels.append(dict(title=BASE_NAME[b], sub=f'gap ×{k}, queries solved (%)', series=series))
    return dict(fields=[x[0] for x in kinds], label={x[0]: x[1] for x in kinds}, css={x[0]: x[2] for x in kinds}, dash={x[0]: x[3] for x in kinds},
                xs=[u[1] for u in UNITS], xname='unit', xlabel='cells of free travel per unit of the network’s time', ylabel='queries solved (%)', ymax=100, panels=panels)


def scatter_data(k=12):
    panels, mx = [], 0.0
    for b in BASE_NAME:
        for key, label, css, _, _ in [f for f in FIELDS if f[0] in SCATTER]:
            v = resolve(b, k, key)
            r = run_of(b, k, v) if v else None
            if not r:
                continue
            z = rows_of(r)
            pts = [[round(float(t), 4), round(float(tp), 4), int(c)] for t, tp, c in zip(z['t_star'], z['T_pred'], z['category'])]
            mx = max(mx, float(z['t_star'].max()))
            panels.append(dict(title=label.replace(', same pairs', '').replace(' (plain length)', '').replace(': stretched-space head + learned correction', '').replace(', one field, full recipe', ', one field'), sub=f'{BASE_NAME[b]}, gap ×{k}', css=css, pts=pts))
    M = math.ceil(mx * 10 + 1) / 10
    return dict(kinds=CAT_SHORT, start=2, max=M, ticks=[round(t, 1) for t in np.arange(0, M + 1e-9, 0.5)], panels=panels,
                xlabel='true travel time T* (maze sides at full speed)', ylabel='the field’s travel time')


def price_data(key='c22_k12'):
    v = VER.get(key)
    if not v:
        return None
    by = v['price']['by_category']
    a = dict(title='The warped maze’s travel time', sub='over the maze’s, per query', xlabel='T_w / m over T* (log scale)', min=0.03, max=1.5, log=True, ref=1.0,
             ticks=[0.05, 0.1, 0.2, 0.5, 1], css='--s-iso', rows=[])
    b = dict(title='Its shortest path, timed on the maze', sub='over the maze’s shortest', xlabel='path travel time / optimum', min=0.98, max=1.12, log=False,
             ref=1.0, ticks=[1.0, 1.05, 1.1], css='--s-iso', rows=[])
    for c, name in zip(CATS, CAT_SHORT):
        if c not in by:
            continue
        e = by[c]
        t = e['warped_time_over_true']
        a['rows'].append(dict(name=name, lo=t['p10'], mid=t['median'], hi=t['p90'], text=[['median', f'{t["median"]:.2f}'], ['10th–90th percentile', f'{t["p10"]:.2f}–{t["p90"]:.2f}']]))
        pp = e['warped_path_over_true_path']
        b['rows'].append(dict(name=name, lo=pp['p10'], mid=pp['median'], hi=pp['p90'], text=[['median', f'+{100 * (pp["median"] - 1):.1f} %'], ['90th percentile', f'+{100 * (pp["p90"] - 1):.1f} %'],
                                                                                           ['worst query', f'+{100 * (e["worst"] - 1):.0f} %'], ['more than 5 % slower', f'{100 * e["share_over_5pct"]:.0f} % of queries']]))
    return [a, b]


def split_range_data(keys=('c22_k12', 'm10_k12')):
    out = []
    for key in keys:
        v = VER.get(key)
        if not v or 'split_time_over_true' not in v['price']['all']:
            continue
        P_ = dict(title=f'Warped-maze time + deficit, {BASE_NAME[v["base"]]}', sub=f'over the maze’s travel time, gap ×{v["k"]}, exact fields', xlabel='(T_w / m + |Δu|) / T*', min=0.9, max=1.1, log=False, ref=1.0,
                  ticks=[0.9, 0.95, 1.0, 1.05, 1.1], css='--s-split', rows=[])
        for c, name in zip(CATS, CAT_SHORT):
            e = v['price']['by_category'].get(c)
            if not e:
                continue
            t = e['split_time_over_true']
            P_['rows'].append(dict(name=name, lo=t['p10'], mid=t['median'], hi=t['p90'], text=[['median', f'{t["median"]:.3f}'], ['10th–90th percentile', f'{t["p10"]:.3f}–{t["p90"]:.3f}'],
                                                                                            ['largest error', f'{100 * e["split_worst"]:.1f} %'],
                                                                                            ['without the deficit', f'{e["warped_time_over_true"]["median"]:.2f}']]))
        out.append(P_)
    return out or None


# --- tables -------------------------------------------------------------------------------------------------------------
def table(head, rows, cls=''):
    th = ''.join(f'<th{" class=txt" if i == 0 else ""}>{h}</th>' for i, h in enumerate(head))
    body = ''.join('<tr>' + ''.join(f'<td class="{"txt" if i == 0 else "num"}">{c}</td>' for i, c in enumerate(r)) + '</tr>' for r in rows)
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


def eq_table(key='c22_k12'):
    v = VER.get(key)
    if not v:
        return ''
    rows = []
    for name, e in v['equations']['by_region'].items():
        rows.append([name, f'{e["maze_equation_on_x"]["median"]:.3f}', f'{e["inner_product_on_z"]["median"]:.3f}',
                     f'{e["plain_length_on_z"]["median"]:.2f} <span class="mute">({e["plain_length_on_z"]["p10"]:.1f}–{e["plain_length_on_z"]["p90"]:.1f})</span>',
                     f'{e["warped_truth_inner_product"]["median"]:.3f}',
                     f'{e["steepest_z_vs_fastest_deg_free"]["median"]:.0f}°' if e.get('steepest_z_vs_fastest_deg_free') else '–'])
    return table(['region', 'T*, on the maze', 'T*, inner product on the warp', 'T*, plain length on the warp', 'T<sub>w</sub>, inner product', 'steepest descent on the warp vs fastest'], rows)


def speed_table(k=12):
    regs = ['quadrants', 'band edge cells', 'arms', 'centre']
    rows = []
    for b in BASE_NAME:
        for key, label, _, _, _ in FIELDS:
            v = resolve(b, k, key)
            p = P(b, k, v) if v else None
            if not p or not p.get('speed_on_maze'):
                continue
            rows.append([f'{BASE_NAME[b]} · {label}'] + [f'{p["speed_on_maze"][r]:.2f}' if r in p['speed_on_maze'] else '–' for r in regs])
    return table(['field (gap ×12)'] + [f'{r}' for r in regs], rows)


def res_table():
    rows = []
    for b in BASE_NAME:
        for k in GAPS:
            p = P(b, k, RES)
            if not p:
                continue
            ta, rc, rf = p.get('time_accuracy') or {}, P(b, k, recipe(b, k)) if recipe(b, k) else None, P(b, k, REF)
            rows.append([f'{BASE_NAME[b]} ×{k}', len(p['seeds']), f1(pc(b, k, RES, 'z')), f1(pc(b, k, RES, 'zh')), f1(pc(b, k, recipe(b, k), 'z')) if rc else '–', f1(pc(b, k, REF, 'z')) if rf else '–',
                         f1(100 * ta['within_10pct'], 0) if ta else '–', f1(100 * rc['time_accuracy']['within_10pct'], 0) if rc and rc.get('time_accuracy') else '–',
                         f1(100 * rf['time_accuracy']['within_10pct'], 0) if rf and rf.get('time_accuracy') else '–'])
    return table(['maze, gap', 'seeds (two-part)', 'two-part field (%)', 'its head alone (%)', 'single field, recipe (%)', 'reference run (%)', 'T within 10 %: two-part', '… single field', '… reference'], rows) if rows else ''


def full_table():
    rows = []
    for b in BASE_NAME:
        for k in GAPS:
            p = P(b, k, RESFULL)
            if p:
                ta = p.get('time_accuracy') or {}
                rows.append([f'{BASE_NAME[b]} ×{k}', f1(pc(b, k, RESFULL, 'z')), f1(pc(b, k, RESFULL, 'zh')), f1(100 * ta['within_10pct'], 0) if ta else '–', f1(pc(b, k, RESFULL2, 'z')) if P(b, k, RESFULL2) else '–'])
    return table(['maze, gap', 'full-loss field + learned correction (%)', 'the full-loss field itself (%)', 'T within 10 % with the correction (%)', 'control: plain two-sided term (%)'], rows) if rows else ''


def all_table():
    rows = []
    for (b, k, v), p in sorted(POOL.items(), key=lambda t: (t[0][0], t[0][1], t[0][2])):
        z, x = p['plans'].get('z'), p['plans'].get('x')
        e = z or x
        ta = p.get('time_accuracy') or {}
        rows.append([f'{BASE_NAME[b]} ×{k} · <code>{v}</code>', len(p['seeds']), f1(z and z['pct']), f1(x and x['pct'])] + [f1(e['by_category'].get(c), 0) for c in CATS]
                    + [f'{e["not_arrived"]} / {e["arrived_wall"]}', f1(ta.get('ratio_median'), 2), f1(100 * ta['within_10pct'], 0) if ta else '–', f1(e.get('time_ratio_median'), 3)])
    return table(['maze, gap, variant', 'seeds', 'planner on the warp', 'planner on the maze'] + CAT_SHORT + ['never arrive / arrive through a wall', 'T / T* (median)', 'within 10 % (%)', 'path time / T*'], rows)


def run_table():
    rows = [[f'<code>{r["id"]}</code>', f'{BASE_NAME[r["base"]]} ×{r["k"]}', f'<code>{r["variant"]}</code>', r['seed'],
             f1(r['plans']['z']['success'] / r['plans']['z']['runs'] * 100) if 'z' in r['plans'] else '–', f1(r['plans']['x']['success'] / r['plans']['x']['runs'] * 100) if 'x' in r['plans'] else '–']
            for r in sorted(RUNS, key=lambda r: (r['base'], r['k'], r['variant'], r['seed']))]
    rows += [[f'<code>{v["id"]}</code>', f'{BASE_NAME[v["base"]]} ×{v["k"]}', 'ground-truth check' + (' (shrunk band)' if key.endswith('_sh') else ''), '–', '–', '–'] for key, v in sorted(VER.items())]
    return table(['registered run', 'maze', 'variant', 'training seed', 'planner on the warp (%)', 'planner on the maze (%)'], rows)


def build():
    import text_blocks as TB
    main, cats, acc = main_data()
    data = dict(main=main, cats=cats, acc=acc, unit=unit_data(), scatter=scatter_data(), price=price_data(), feel=feel_data(), full=full_data())
    t = (HERE / 'page_template.html').read_text()
    blocks = TB.blocks(dict(P=P, plan=plan, pc=pc, f1=f1, run_of=run_of, rows_of=rows_of, VER=VER, POOL=POOL, RUNS=RUNS, BASE_NAME=BASE_NAME, CATS=CATS, GAPS=GAPS, table=table, recipe=recipe, feel=feel, fail_where=fail_where, feel_raw=feel_raw, arrival=arrival, stalled_open=stalled_open,
                            have_paths=(OUT / 'paths.png').exists(), have_fields=(OUT / 'fields.png').exists(), have_deficit=(OUT / 'deficit.png').exists(),
                            have_wallsfar=(OUT / 'wallsfar.png').exists(), figs=json.load(open(OUT / 'figs.json')) if (OUT / 'figs.json').exists() else {}))
    blocks.update(EQTABLE=eq_table(), ALLTABLE=all_table(), RUNTABLE=run_table(), RESTABLE=res_table())
    for key, val in blocks.items():
        tag = f'<!--{key}-->'
        assert tag in t, f'template lacks {tag}'
        t = t.replace(tag, val)
    t = t.replace('/*DATA*/', json.dumps(data, separators=(',', ':'), default=float))
    import re
    left = re.findall(r'<!--[A-Z0-9]+-->', t)
    assert not left, f'unfilled blocks: {left}'
    (OUT / 'aniso_eikonal.html').write_text(t)
    print(f'wrote {OUT / "aniso_eikonal.html"} ({len(t) / 1024:.0f} KiB)')


if __name__ == '__main__':
    build()
