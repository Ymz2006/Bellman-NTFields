"""Tables of the line-warp study from registered runs.

    python report.py [--state <queue state.json>]     --state refreshes report/runs.json (job id -> run dir) from the queue

report/runs.json is the study's record of its runs (the queue itself lives in a scratchpad and does not outlast the session). Writes
report/summary.json. Controls come from cross_warp's registered runs (CONTROLS); measurements on the runs from report/mechanism.json (mechanism.py, wallhits.py).
"""
import argparse
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'report'
PLUS = '20261102T042454Z-62864e61'
CONTROLS = {       # cross_warp's fields on the plus maze, 3 training seeds each (87358, 1, 2)
    'ntrl-demo': ['20261103T163539Z-48369bc1', '20261103T171116Z-e9eb6c1d', '20261103T170442Z-1832e0eb'],
    'uniform B x 4': ['20261103T172457Z-78235cd0', '20261103T180931Z-9b9b0029', '20261103T175150Z-253aed03'],
    'constructed x 6 (pm6)': ['20261103T171757Z-79906972', '20261103T173406Z-5a9640e4', '20261103T173057Z-c4de6c30'],
    'earlier learned warp (la6b)': ['20261103T191129Z-0b3ade71', '20261103T190455Z-09087650', '20261103T193254Z-f38c0a35'],
}


def run_dir(rid):
    return ROOT / 'results' / (rid if rid.startswith('registered_') else f'registered_{rid}')


def field_row(r):
    """One field run's summary (its own 240-query set)."""
    d = json.load(open(r / 'results.json'))
    rows = np.load(r / 'rows.npz')
    ok, conv, coll = rows['success'], rows['converged'], rows['collision']
    t = d['success_total']
    nw = d.get('near_wall', {})
    allw = nw.get('all') or {}
    return dict(run=r.name.replace('registered_', ''), seed=d.get('seed'), gpu=(d.get('gpu') or '')[-8:], success=t[0], runs=t[1], pct=100 * t[0] / t[1], arrived_wall=int((~ok & conv & coll).sum()),
                not_arrived=int((~ok & ~conv).sum()), by_bin={k: 100 * v[0] / v[1] for k, v in d['success_by_bin'].items()}, by_where={k: 100 * v['success'] / v['runs'] for k, v in d.get('by_where', {}).items()},
                speed_corr=d['speed_fit']['log_speed_corr'], t_ratio=d['field']['t_over_tstar_median'], bumps=d['bumps']['bumps_per_path'], attract=allw.get('attract'), s=d.get('s'), s_p=d.get('s_p'),
                speed_free=d['speed_fit']['speed_pred_median_free'], speed_wall=d['speed_fit']['speed_pred_median_at_floor'], rollbacks=d.get('rollbacks'), descent_cos=d['field']['descent_cos_median'])


def pooled(rows):
    n = sum(r['runs'] for r in rows)
    k = sum(r['success'] for r in rows)
    return dict(pct=100 * k / n, seeds=len(rows), per_seed=[r['success'] for r in rows], runs_per_seed=rows[0]['runs'])


CATS = ['same quadrant', 'adjacent quadrants', 'diagonal quadrants', 'quadrant - open', 'open - open']
GAPS = [1, 2, 4, 8, 12]
SEEDS = [87358, 1, 2]
# Stage 2 on the plus maze: (key, label, group, job ids in the queue's record)
PLUS_ROWS = [
    ('stripes_axis', 'free ResNet, axis lines, s 6', 'free ResNet (stripes)', ['A_s6']),
    ('stripes_axis4', 'free ResNet, axis lines, s 4', 'free ResNet (stripes)', ['A_s4']),
    ('stripes_final', 'free ResNet, axis lines, final code, quadrant stretch 6', 'free ResNet (stripes)', ['A2_m6']),
    ('stripes_any', 'free ResNet, any-angle lines, s 6', 'free ResNet (stripes)', ['B_s6']),
    ('stripes_any4', 'free ResNet, any-angle lines, s 4', 'free ResNet (stripes)', ['B_s4']),
    ('res_a4', 'ResNet, anisotropy limit 4', 'bounded ResNet', ['rsA4_m6']),
    ('res_a12', 'ResNet, anisotropy limit 12', 'bounded ResNet', ['rsA12_m6']),
    ('res_tv', 'ResNet, limit 12 + TV 0.05', 'bounded ResNet', ['resTV_m6']),
    ('sep_h070', 'separable, stretch bound 0.70, no TV', 'separable, no TV', ['sep070_m6', 'sep070_m6_1', 'sep070_m6_2']),
    ('sep_h100', 'separable, stretch bound 1.00, no TV', 'separable, no TV', ['sep100_m6']),
    ('sep_h125', 'separable, stretch bound 1.25, no TV', 'separable, no TV', ['sep125_m6']),
    ('sep_h160', 'separable, stretch bound 1.60, no TV', 'separable, no TV', ['sep160_m6']),
    ('sep_h200', 'separable, stretch bound 2.00, no TV', 'separable, no TV', ['sep200_m6', 'sep200_m6_1', 'sep200_m6_2']),
    ('sepL125', 'separable, low-pass, bound 1.25', 'separable, low-pass', ['sepL125_m6']),
    ('sepL200', 'separable, low-pass, bound 2.00', 'separable, low-pass', ['sepL200_m6']),
    ('tv02', 'separable, TV 0.02', 'separable + TV', ['tv02_m6']),
    ('tv05', 'separable, TV 0.05', 'separable + TV', ['tv05_m6']),
    ('tv10', 'separable, TV 0.10', 'separable + TV', ['tv10_m6']),
    ('tv20', 'separable, TV 0.20', 'separable + TV', ['tv20_m6']),
    ('tv40', 'separable, TV 0.40', 'separable + TV', ['tv40_m6']),
    ('tri_tv', 'triangular, TV 0.05', 'triangular + TV', ['triTV_m6']),
    ('any', 'recipe with any-angle lines', 'recipe variants', ['anyTV_m6']),
    ('walls', 'recipe with workspace walls as the obstacle', 'recipe variants', ['wallsTV_m6']),
    ('ntdil', 'the recipe without TV (bound 2, dilation)', 'recipe variants', ['ntdil_m6_87358', 'ntdil_m6_1', 'ntdil_m6_2']),
    ('recipe', 'the recipe without dilation (separable, TV 0.10, lines through the boundary)', 'the recipe', ['bt10_m6_87358', 'bt10_m6_1', 'bt10_m6_2']),
    ('dil', 'the recipe: separable, TV 0.10, lines through the boundary, 1-cell dilation', 'the recipe', ['dil_m6_87358', 'dil_m6_1', 'dil_m6_2']),
]
FITS = [('free ResNet, axis lines', 'fitA'), ('free ResNet, axis lines, final code', 'fitA2'), ('free ResNet, any-angle lines', 'fitB'), ('free ResNet, shear penalty 30', 'shear30'), ('free ResNet, runs also compared across lines', 'gmix'), ('free triangular map', 'triFree'),
        ('ResNet, anisotropy limit 12', 'rsA12'), ('ResNet, anisotropy limit 4', 'rsA4'), ('ResNet, limit 12 + TV', 'resTV'), ('ResNet, gap x4', 'c22k4res'), ('ResNet, gap x12, frequencies scaled', 'resK12a2'), ('ResNet, gap x12, low frequencies', 'resK12b2'),
        ('separable, bound 0.70, no TV', 'sep070'), ('separable, bound 1.25, no TV', 'sep125'), ('separable, low-pass', 'sepL125'), ('separable, TV 0.05', 'tv05'), ('separable, TV 0.10', 'bt10'), ('separable, TV 0.10, workspace walls', 'wallsTV'), ('the recipe (TV 0.10 + dilation)', 'dil'), ('the recipe without TV', 'ntdil'),
        ('separable, third step level', 'inset1'), ('triangular + TV', 'triTV')]


def book():
    return json.load(open(OUT / 'runs.json'))


def gap_jobs(base, k):
    """The recipe's z planner runs at one gap: seed -> job id."""
    t = 'c22k12dil' if (base, k) == ('c22', 12) else f'{base}k{k}d'
    return {87358: f'{t}_q87358', 1: f'{t}_q1', 2: f'{t}_q2'}


def zq_entry(rid):
    d = json.load(open(run_dir(rid) / 'results.json'))
    (fid, e), = d['runs'].items()
    return dict(field=fid, seed=e['seed'], m=e['quadrant_stretch'], inverse=e['inverse_roundtrip_max'], **{k: e['plans']['z'][k] for k in ('success_total', 'by_category', 'arrived_wall', 'not_arrived', 'time_ratio_median', 'length_ratio_median')})


def control_rate(R, base, k, lab, planner, cat=None):
    """Pooled success (%) of a gap study field from cross_warp/report/zdraw10.json (zdraw10.rate, pooled)."""
    if base == 'c22' and lab in ('id', 'pm', 'pmz'):
        by = R['c22_study'].get(str(k), {}).get(lab, {})
    else:
        by = R['fields'].get(f'{base}/{k}', {}).get(lab, {})
    ents = [v.get(planner) for v in by.values() if v.get(planner)]
    a = [e['success_total'] if cat is None else e['by_category'].get(cat) for e in ents]
    a = [x for x in a if x]
    return (100 * sum(x[0] for x in a) / sum(x[1] for x in a), len(a)) if a else (None, 0)


def control_seeds(R, base, k, lab, planner):
    """Training seed -> success (%) of a gap study field."""
    by = (R['c22_study'].get(str(k), {}) if base == 'c22' and lab in ('id', 'pm', 'pmz') else R['fields'].get(f'{base}/{k}', {})).get(lab, {})
    return {s: 100 * v[planner]['success_total'][0] / v[planner]['success_total'][1] for s, v in by.items() if v.get(planner)}


def summary():
    B = book()
    out = dict(plus={}, plus_controls={}, fits=[], gap={}, q1200={})
    for name, ids in CONTROLS.items():
        rows = [field_row(run_dir(i)) for i in ids]
        out['plus_controls'][name] = dict(rows=rows, **pooled(rows))
    for key, label, group, jobs in PLUS_ROWS:
        rows = [field_row(run_dir(B[j])) for j in jobs if j in B]
        if rows:
            out['plus'][key] = dict(label=label, group=group, rows=rows, **pooled(rows))
    R = json.load(open(OUT / 'mechanism.json')).get('ratios', {})
    for label, job in FITS:
        if job in B:
            d = json.load(open(run_dir(B[job]) / 'results.json'))
            w = d['warp']
            same = R.get(job)        # plus-maze fits: measured again with one procedure (mechanism.py ratios)
            out['fits'].append(dict(label=label, run=B[job], det_positive=w['det_positive'], outline=w['outline_simple'], sv=w['sv_min_max'], aniso_median=w['anisotropy_median'], aniso_q99=w['anisotropy_q99'],
                                    rotation=w['rotation_abs_median_deg'], ratio=same['ratio'] if same else w['runs']['ratio_median'], log_std=same['log_std'] if same else w['runs']['log_std'], runs_per_line=w['runs']['runs_per_line'], area=w.get('area_stretch'),
                                    identity_ratio=R['identity']['ratio'] if same else d['identity_runs']['ratio_median'], identity_log_std=R['identity']['log_std'] if same else d['identity_runs']['log_std']))
    Z = json.load(open(ROOT / 'jeffrey/cross_warp/report/zdraw10.json'))
    for base in ('c22', 'm10'):
        for k in GAPS:
            ent = dict(learned=[], controls={})
            for seed, job in gap_jobs(base, k).items():
                if job in B:
                    ent['learned'].append(zq_entry(B[job]))
            for lab, planner in (('id', 'x'), ('pm', 'x'), ('pmz', 'z'), ('pmzw', 'z'), ('pmzs', 'z')):
                r, n = control_rate(Z, base, k, lab, planner)
                ent['controls'][lab] = dict(pct=r, seeds=n, by_category={c: control_rate(Z, base, k, lab, planner, c)[0] for c in CATS}, by_seed=control_seeds(Z, base, k, lab, planner))
            if ent['learned']:
                a = [e['success_total'] for e in ent['learned']]
                ent['pct'] = 100 * sum(x[0] for x in a) / sum(x[1] for x in a)
                ent['per_seed'] = [100 * x[0] / x[1] for x in a]
                ent['by_category'] = {c: 100 * sum(e['by_category'][c][0] for e in ent['learned']) / sum(e['by_category'][c][1] for e in ent['learned']) for c in CATS if all(c in e['by_category'] for e in ent['learned'])}
            out['gap'][f'{base}/{k}'] = ent
    for job, name in (('bt10_e1200', 'recipe without dilation'), ('dil_e1200', 'the recipe'), ('ntdil_e1200', 'recipe without TV'), ('sep070_e1200', 'bound 0.70, no TV'), ('sep200_e1200', 'bound 2, no TV')):
        if job in B:
            e = json.load(open(run_dir(B[job]) / 'results.json'))
            out['q1200'][name] = {r: v['success_total'] for r, v in e['runs'].items()}
    for rid, labels in Q1200_CONTROLS.items():
        e = json.load(open(run_dir(rid) / 'results.json'))
        for r, v in e['runs'].items():
            for name, ids in CONTROLS.items():
                if r in ids:
                    out['q1200'].setdefault(name, {})[r] = v['success_total']
    out['ablation'] = ablation(B, out)
    out['cat_text'], out['arch_text'], out['fitseeds'], out['gap_text'] = category_text(out), arch_text(B, out), fit_seed_text(B), gap_text(out)
    json.dump(out, open(OUT / 'summary.json', 'w'), indent=1)
    return out


def zq_pct(B, job):
    e = zq_entry(B[job])
    return 100 * e['success_total'][0] / e['success_total'][1], e


def zq_x(B, job):
    d = json.load(open(run_dir(B[job]) / 'results.json'))
    (_, e), = d['runs'].items()
    t = e['plans']['x']['success_total']
    return 100 * t[0] / t[1]


# TV x dilation at gap x12, z-trained: cell -> job per training seed (m10k12t10_q87358 predates the quadrant-stretch fix; its rerun is _qB).
# tight, b125, range, b250 and dil are a sweep of the stretch bound with dilation and no TV; range is the bound at the range TV reached by itself on that maze.
GRID = {'c22': dict(both=['c22k12dil_q87358', 'c22k12dil_q1', 'c22k12dil_q2'], tv=['c22k12t10_q87358', 'c22k12t10_q1', 'c22k12t10_q2'], dil=['c22k12nt_q87358', 'c22k12nt_q1', 'c22k12nt_q2'],
                    none=['c22k12n0_q87358', 'c22k12n0_q1', 'c22k12n0_q2'], tight=['c22k12h07_q87358'], b125=['c22k12b125_q87358'], range=['c22k12r_q87358', 'c22k12r_q1', 'c22k12r_q2'], b250=['c22k12b250_q87358']),
        'm10': dict(both=['m10k12d_q87358', 'm10k12d_q1', 'm10k12d_q2'], tv=['m10k12t10_qB', 'm10k12t10_q1', 'm10k12t10_q2'], dil=['m10k12nt_q87358', 'm10k12nt_q1', 'm10k12nt_q2'],
                    none=['m10k12n0_q87358', 'm10k12n0_q1', 'm10k12n0_q2'], tight=['m10k12h07_q87358'], b125=['m10k12b125_q87358'], range=['m10k12r_q87358', 'm10k12r_q1', 'm10k12r_q2'], b250=['m10k12b250_q87358'])}
BOUNDS = {'c22': [('tight', 0.70), ('b125', 1.25), ('range', 1.875), ('b250', 2.5), ('dil', 3.0)], 'm10': [('tight', 0.70), ('b125', 1.25), ('range', 1.775), ('b250', 2.5), ('dil', 3.0)]}


def grid_cells(B):
    """Per cell: outcome over its seeds (wall = arrived through a wall, stall = never arrived, as % of runs), and from mechanism.json what the fit did and what the field learned in z (wallhits.py field;
    the speed_fit of a z-trained field's own run is in x units and scales with the map's local stretch, so it is not used here)."""
    M = json.load(open(OUT / 'mechanism.json'))
    out = {}
    for base, cells in GRID.items():
        out[base] = {}
        for cell, jobs in cells.items():
            have = [j for j in jobs if j in B]
            if not have:
                continue
            e = [zq_entry(B[j]) for j in have]
            t = [x['success_total'] for x in e]
            n = sum(x[1] for x in t)
            f = [field_row(run_dir(B[j.replace('_q', '_zw')])) for j in have]
            fz, z = M.get('fieldz', {}).get(base, {}).get(cell, {}), M.get('zones12', {}).get(base, {}).get(cell, {})
            sh = M.get('shape', {}).get(z.get('job'), {})
            out[base][cell] = dict(pct=100 * sum(x[0] for x in t) / n, per_seed=[100 * x[0] / x[1] for x in t], wall=100 * sum(x['arrived_wall'] for x in e) / n, stall=100 * sum(x['not_arrived'] for x in e) / n,
                                   wall_runs=[x['arrived_wall'] for x in e], stall_runs=[x['not_arrived'] for x in e], of=t[0][1], seeds=len(e), bumps=float(np.mean([r['bumps'] for r in f])), rollbacks=float(np.mean([r['rollbacks'] for r in f])),
                                   corr_z=fz.get('corr'), into=fz.get('into'), two_in=z.get('two_in_quadrant'), two_out=z.get('two_outside'), s=z.get('s'), range=sh.get('range'), tilt=sh.get('tilt'), edges=sh.get('edges'), band_q=sh.get('band_over_quadrant'))
    return out


# plus-maze fit -> its row of PLUS_ROWS, for the loss against planning
FIT_ROW = dict(fitA2='stripes_final', fitB='stripes_any', rsA12='res_a12', rsA4='res_a4', resTV='res_tv', sep125='sep_h125', sep070='sep_h070', sepL125='sepL125', tv05='tv05', bt10='recipe', dil='dil', ntdil='ntdil', triTV='tri_tv', tv40='tv40')


def links(out, a):
    """How planning relates to what the field learned and to the loss (correlations over fields; a field is the mean of its seeds)."""
    rows = []
    groups = [(k, v['group'], v) for k, v in out['plus'].items()] + [(k, 'control', v) for k, v in out['plus_controls'].items()]
    for key, g, v in groups:
        m = lambda k: float(np.mean([r[k] for r in v['rows']]))
        rows.append(dict(key=key, name=v.get('label', key), group=g, pct=v['pct'], bumps=m('bumps'), speed_corr=m('speed_corr'), speed_wall=m('speed_wall'), descent_cos=m('descent_cos')))
    r = lambda x, k: float(np.corrcoef([q[k] for q in x], [q['pct'] for q in x])[0, 1])
    sep = [x for x in rows if x['group'].startswith(('separable', 'recipe', 'the recipe'))]
    M = json.load(open(OUT / 'mechanism.json'))
    R = M.get('ratios', {})
    loss = [dict(name=out['plus'][k]['label'], group=out['plus'][k]['group'], ratio=R[j]['ratio'], pct=out['plus'][k]['pct']) for j, k in FIT_ROW.items() if j in R and k in out['plus']]
    res = dict(fields=rows, loss=loss, separable=len(sep), r_bumps=r(rows, 'bumps'), r_bumps_sep=r(sep, 'bumps'), r_speed=r(rows, 'speed_corr'), r_wall=r(rows, 'speed_wall'), r_cos=r(rows, 'descent_cos'),
               r_ratio=float(np.corrcoef([x['ratio'] for x in loss], [x['pct'] for x in loss])[0, 1]), identity_ratio=R['identity']['ratio'])
    fz = M.get('fieldz', {})
    if 'plus' in fz:
        pct = dict(a['z1'], recipe=out['gap']['c22/1']['pct'], tv=a['z1']['tv_nodil'])
        z1 = [(fz['plus'][k]['corr'], pct[k]) for k in fz['plus'] if k in pct]
        res['z1_points'] = [dict(name=k, corr=fz['plus'][k]['corr'], into=fz['plus'][k]['into'], pct=pct[k]) for k in fz['plus'] if k in pct]
        res['r_z1'] = float(np.corrcoef(*zip(*z1))[0, 1])
    for b in ('c22', 'm10'):
        c = [(g['corr_z'], g['pct']) for g in a['grid'][b].values() if g.get('corr_z') is not None]
        if len(c) > 2:
            res[f'r_z12_{b}'] = float(np.corrcoef(*zip(*c))[0, 1])
    return res


def ablation(B, out):
    """Single-run comparisons quoted in the text (training seed 87358, gap x12)."""
    M = json.load(open(OUT / 'mechanism.json'))
    z = M['slow_zone']
    a = dict(stalls=M.get('stalls'), step=z['step'], zone_nodil=z['no dilation']['band_side'], zone_dil=z['dilation']['band_side'], zone_nodil_share=100 * z['no dilation']['band_side'] / z['no dilation']['quadrant'],
             zone_dil_share=100 * z['dilation']['band_side'] / z['dilation']['quadrant'])
    a['nodil_c22'], e0 = zq_pct(B, 'c22k12t10_q87358')
    a['dil_c22'], e1 = zq_pct(B, 'c22k12dil_q87358')
    a['nodil_m10'], _ = zq_pct(B, 'm10k12t10_qB')
    a['dil_m10'], _ = zq_pct(B, 'm10k12d_q87358')
    qo = lambda e: 100 * e['by_category']['quadrant - open'][0] / e['by_category']['quadrant - open'][1]
    a['qo_nodil'], a['qo_dil'] = qo(e0), qo(e1)
    a['width'] = [a['nodil_c22'], zq_pct(B, 'c22k12dil05_q87358')[0], a['dil_c22'], zq_pct(B, 'c22k12dil20_q87358')[0]]
    a['stuck'] = zq_pct(B, 'm12tv_q')[0]
    a['z1'] = {k: zq_pct(B, j)[0] for k, j in (('recipe', 'c22k1d_q87358'), ('tv_nodil', 'bt10_q'), ('wiggly', 'sep125_q'), ('resnet', 'resA4_q'), ('stripes', 'stripeA_q'), ('notv', 'c22k1nt_q87358'), ('tight', 'sep070_q')) if j in B}
    a['z1_fail'] = {k: {f: zq_entry(B[j])[f] for f in ('arrived_wall', 'not_arrived', 'success_total')} for k, j in (('recipe', 'c22k1d_q87358'), ('tv_nodil', 'bt10_q'), ('wiggly', 'sep125_q'), ('resnet', 'resA4_q'), ('stripes', 'stripeA_q'), ('notv', 'c22k1nt_q87358'), ('tight', 'sep070_q')) if j in B}
    for k, j in (('notv_c22', 'c22k12nt_q87358'), ('notv_m10', 'm10k12nt_q87358'), ('tight_c22', 'c22k12h07_q87358'), ('tight_m10', 'm10k12h07_q87358'), ('xtrained_c22', 'c22k12dil_xq')):
        if j in B:
            a[k] = zq_x(B, j) if k.startswith('xtrained') else zq_pct(B, j)[0]
    a['grid'] = grid_cells(B)
    a['notv_k12'] = zq_pct(B, 'g12q')[0]
    a['res_k4'], a['sep_k4'] = zq_pct(B, 'c22k4res_q87358')[0], zq_pct(B, 'c22k4d_q87358')[0]
    e = zq_entry(B['c22k4res_q87358'])['by_category']
    a['res_k4_cross'] = [100 * e[c][0] / e[c][1] for c in ('adjacent quadrants', 'diagonal quadrants')]
    a['zones'], a['paths'], a['variation'], a['shape'], a['zones12'], a['hits'] = M.get('zones'), M.get('paths'), M.get('variation'), M.get('shape'), M.get('zones12'), M.get('hits')
    a['bounds'] = {b: [dict(bound=h, **a['grid'][b][c]) for c, h in v if c in a['grid'][b]] for b, v in BOUNDS.items()}
    a['fieldz'], a['steps'] = M.get('fieldz'), M.get('steps')
    a['links'] = links(out, a)
    if 'dil_region' in B:
        d = json.load(open(run_dir(B['dil_region']) / 'results.json'))
        t = [e['plans']['x']['success_total'] for e in d['runs'].values()]
        a['region_x'] = 100 * sum(x[0] for x in t) / sum(x[1] for x in t)
    w = out['plus'].get('walls')
    a['walls_text'] = f"A field through that warp solves {w['pct']:.1f} % of the plus maze on one seed." if w else ''
    return a


def fit_seed_text(B):
    """Band / quadrant area stretch of the recipe's fit under three fit seeds."""
    v = []
    for job in ('bt10', 'seed1', 'seed2'):
        if job in B:
            a = json.load(open(run_dir(B[job]) / 'results.json'))['warp']['area_stretch']
            v.append(a['arms'] / a['quadrants'])
    return 'band / quadrant stretch ' + ', '.join(f'{x:.2f}' for x in v) + f' over {len(v)} fit seeds'


def gap_text(out):
    """The paragraphs under the gap charts, from the numbers: where the learned warp stands against the hand-built one, with the spread between seeds."""
    lines = []
    for base, name in (('m10', '10×10 quadrants'), ('c22', '22×22 quadrants')):
        rows = [(k, out['gap'][f'{base}/{k}']) for k in GAPS]
        rows = [(k, e) for k, e in rows if 'pct' in e]
        learned = ' / '.join(f"{e['pct']:.1f}" for _, e in rows)
        hand = ' / '.join(f1(e['controls']['pmzw']['pct']) for _, e in rows)
        ntrl = ' / '.join(f1(e['controls']['id']['pct']) for _, e in rows)
        ks = ' / '.join(f'×{k}' for k, _ in rows)
        lines.append(f"<strong>{name}</strong> at {ks}: learned {learned} %, hand-built {hand} %, ntrl-demo {ntrl} %.")
    t = '<p>' + ' '.join(lines) + '</p>'

    def diff(base, k):
        e = out['gap'][f'{base}/{k}']
        return e['pct'] - e['controls']['pmzw']['pct']

    def spread(base):
        return max(max(out['gap'][f'{base}/{k}']['per_seed']) - min(out['gap'][f'{base}/{k}']['per_seed']) for k in GAPS)
    sign = lambda v: f'{v:+.1f}'.replace('-', '−')
    wide = lambda base: ' / '.join(sign(diff(base, k)) for k in GAPS[1:])
    pairs = []
    for base in ('m10', 'c22'):
        e = out['gap'][f'{base}/12']
        c = e['controls']['pmzw']['by_seed']
        pairs.append(min(l['success_total'][0] / l['success_total'][1] * 100 - c[str(l['seed'])] for l in e['learned']))
    t += (f"<p>ntrl-demo falls with the gap. The learned warp does not. Its warped mazes look the same at every gap (above), so the field faces nearly the same problem, which is what the gap study’s band-width test said was needed. "
          f"From ×2 to ×12 it is {wide('m10')} points from the hand-built warp on the 10×10 quadrants and {wide('c22')} on the 22×22 quadrants. Seeds of one setting differ by up to {spread('m10'):.0f} and {spread('c22'):.0f} points, so on the 22×22 quadrants only ×4 and ×12 stand out. "
          f"At ×12, the widest gap, the learned warp is ahead on each of the three seeds, by at least {pairs[0]:.1f} and {pairs[1]:.1f} points. At ×1 it is {sign(diff('m10', 1))} and {sign(diff('c22', 1))}: with little band to compress, the hand-built warp keeps a small lead.</p>")
    return t


def f1(v):
    return '–' if v is None else f'{v:.1f}'


def category_text(out):
    parts = []
    for base, name in (('m10', '10×10 quadrants'), ('c22', '22×22 quadrants')):
        e = out['gap'][f'{base}/12']
        if 'by_category' not in e:
            continue
        c = e['controls']
        d = 'diagonal quadrants'
        parts.append(f"Diagonal queries on the {name}: {e['by_category'][d]:.0f} %, against {c['pmzw']['by_category'][d]:.0f} % for the hand-built warp, {c['pmzs']['by_category'][d]:.0f} % with the band also shrunk by hand, and {c['id']['by_category'][d]:.0f} % for ntrl-demo.")
    return ' '.join(parts)


def arch_text(B, out):
    P = out['plus']
    fits = {f['label']: f for f in out['fits']}
    t = '<p>You asked me to start from a residual network, so here is how each form did. All three use Fourier features.</p><ul class="tight">'
    if 'res_a4' in P and 'res_a12' in P:
        t += (f"<li><strong>ResNet.</strong> Free, it makes stripes. With its anisotropy limited it is a valid map on the plus maze, but its stretch varies everywhere, and it plans worse the more it distorts: limit 4 {P['res_a4']['pct']:.1f} %, limit 12 {P['res_a12']['pct']:.1f} %, free {P['stripes_any4']['pct']:.0f} % at best (one seed each). ")
        if 'res_tv' in P:
            t += f"Total variation on its stretch gives {P['res_tv']['pct']:.1f} %. "
        res4 = gap_res(B)
        a = out['ablation']
        t += (f"At gap ×4 it finds a valid star-shaped map (below){res4}, and it almost never crosses between quadrants (adjacent {a['res_k4_cross'][0]:.0f} %, diagonal {a['res_k4_cross'][1]:.0f} %). At ×12 two attempts folded: det J is positive on only {100 * fits['ResNet, gap x12, frequencies scaled']['det_positive']:.0f} and {100 * fits['ResNet, gap x12, low frequencies']['det_positive']:.0f} % of the grid, and the outline crosses itself. "
              f"One likely reason: most of the box is empty, so the loss has no boundary points there to hold the map while it carries the quadrants across. I did not test that.</li>")
    if 'tri_tv' in P:
        t += f"<li><strong>Triangular map</strong> u = F(x | y), v = G(y | u). Diffeomorphic by construction, and each quadrant gets its own adjustment. With total variation it gives {P['tri_tv']['pct']:.1f} % on the plus maze (one seed): the extra freedom buys nothing here.</li>"
    t += ("<li><strong>Separable map</strong> (F(x), G(y)). The simplest of the three, and the best on these mazes. It cannot shear, so stripes are out, and it matches an axis-aligned band exactly. Its log-density is a Fourier-feature network and its line loss is unchanged.</li></ul>")
    return t


def gap_res(B):
    if 'c22k4res_q87358' not in B:
        return ''
    p, _ = zq_pct(B, 'c22k4res_q87358')
    return f', which solves {p:.1f} % against the separable map\'s {zq_pct(B, "c22k4d_q87358")[0]:.1f} % (one seed)'


Q1200_CONTROLS = {'20261103T175033Z-30674627': 1, '20261103T192052Z-60802151': 2, '20261103T200417Z-698857d8': 3}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--state', type=Path, default=None)
    a = p.parse_args()
    OUT.mkdir(exist_ok=True)
    path = OUT / 'runs.json'
    if a.state:
        st = json.load(open(a.state))
        old = json.load(open(path)) if path.exists() else {}
        old.update({k: v.replace('results/registered_', '') for k, v in st.items() if k != '_running' and isinstance(v, str) and not v.startswith('FAILED')})
        json.dump(old, open(path, 'w'), indent=0, sort_keys=True)
    out = summary()
    print(f'{len(book())} runs recorded')
    for name, v in out['plus_controls'].items():
        print(f"  {name:32s} {v['pct']:5.1f} % {v['per_seed']}")
    for k, v in out['plus'].items():
        print(f"  {v['label'][:48]:48s} {v['pct']:5.1f} % {v['per_seed']}")
    for name, v in out['q1200'].items():
        t = list(v.values())
        print(f"  1200-set {name:32s} {100 * sum(x[0] for x in t) / sum(x[1] for x in t):5.1f} % {[x[0] for x in t]}")
    for key, e in out['gap'].items():
        if 'pct' in e:
            print(f"  {key:8s} learned {e['pct']:5.1f} ({len(e['learned'])} seeds {[round(x, 1) for x in e['per_seed']]}) | " + ', '.join(f"{l} {c['pct']:.1f}" for l, c in e['controls'].items() if c['pct'] is not None))


if __name__ == '__main__':
    main()
