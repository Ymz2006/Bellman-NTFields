"""Tables of the equal-runs study from registered runs.

    python report.py [--state <queue state.json>]     --state refreshes report/runs.json (label -> run directory) from the queue

report/runs.json is the study's record of its runs. Measures come from the remeasure runs (re_*) where those exist, so a table never mixes versions of a measure; learnability scores from each run's own results.json. Writes report/summary.json.
"""
import argparse
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'report'
LW = json.load(open(ROOT / 'jeffrey/line_warp/report/summary.json'))

P10 = [
    ('No warp', 'none', ['p10_ident2']),
    ('Every run of a line equal, as asked (coupling)', 'literal', ['p10_coupling_cvline', 'p10_coupling_cvline_seed1', 'p10_coupling_cvline_seed2']),
    ('Every run of a line equal, as asked (flow)', 'literal', ['p10_flow_cvline']),
    ('Same on a log scale (coupling)', 'literal', ['p10_coupling_lvline']),
    ('Same on a log scale (flow)', 'literal', ['p10_flow_lvline']),
    ('All runs of a batch equal, target follows the runs (coupling)', 'literal', ['p10_coupling_lvadapt']),
    ('All runs of a batch equal, target follows the runs (flow)', 'literal', ['p10_flow_lvadapt']),
    ('Fixed target 0.05 (coupling)', 'fixed', ['p10_coupling_cv_t05_k16']),
    ('Fixed target 0.07 (coupling)', 'fixed', ['p10_coupling_cv_t07_k16']),
    ('Fixed target 0.10 (coupling)', 'fixed', ['p10_coupling_cv_t10_k16']),
    ('Fixed target 0.10 (flow)', 'fixed', ['p10_flow_cv_t10_k16']),
    ('Fixed target 0.10 (ResNet)', 'fixed', ['p10_res_cv_t10_k16']),
    ('Fixed target 0.07, log scale (coupling)', 'fixed', ['p10_coupling_lv_t07_k16']),
    ('Fixed target 0.05, stretch bound 4 (coupling)', 'fixed', ['p10_coupling_cv_t05_k4']),
    ('Fixed target 0.07, stretch bound 4 (coupling)', 'fixed', ['p10_coupling_cv_t07_k4']),
    ('Fixed target 0.10, stretch bound 4 (coupling)', 'fixed', ['p10_coupling_cv_t10_k4']),
    ('Fixed target 0.10, stretch bound 4 (flow)', 'fixed', ['p10_flow_cv_t10_k4']),
    ('Fixed target 0.07, no shear (separable)', 'sep', ['p10_sep_cv_t07']),
    ('Fixed target 0.10, no shear (separable)', 'sep', ['p10_sep_cv_t10']),
    ('Fixed target 0.10, every angle (coupling)', 'any', ['p10_any_coupling_cv_t10', 'p10_any_coupling_cv_t10_seed1', 'p10_any_coupling_cv_t10_seed2']),
    ('Fixed target 0.07, every angle (coupling)', 'any', ['p10_any_coupling_cv_t07']),
    ('Fixed target 0.10, every angle (flow)', 'any', ['p10_any_flow_cv_t10']),
    ('Fixed target 0.07, every angle (flow)', 'any', ['p10_any_flow_cv_t07']),
    ('Pair form, shift 0.05', 'pair', ['p10e_pair_t05']),
    ('Pair form, shift 0.10', 'pair', ['p10e_pair_t10']),
    ('Old loss with shear (coupling)', 'old', ['p10c_hard']),
    ('Old loss with shear, high frequencies (coupling)', 'old', ['p10_coupling_hard']),
    ('Old loss with shear (flow)', 'old', ['p10_flow_hard']),
    ('Old loss, others held still within an update', 'limit', ['p10c_hard_anchor']),
    ('Old loss, update size capped', 'limit', ['p10c_hard_trust']),
    ('Old loss, stiff near walls', 'limit', ['p10c_hard_near']),
    ('Equal runs, worst run of a line weighted (power 2)', 'limit', ['p10c_pow2']),
    ('Equal runs, worst run of a line weighted (power 8)', 'limit', ['p10c_pow8']),
    ('Equal runs, only the worst 5 % of runs pushed', 'limit', ['p10c_top05']),
    ('Equal runs, stiff near walls', 'limit', ['p10g_eq_near']),
    ('Equal runs, every angle, stiff near walls', 'limit', ['p10g_eq_any_near']),
    ('Long runs pushed, stiff near walls, stretch bound 8', 'limit', ['p10k_k8']),
    ('Long runs pushed, stiff near walls, stretch bound 8, every angle', 'limit', ['p10k_k8_any']),
    ('Only long runs pushed, target from run count, stiff near walls', 'limit', ['p10i_cap_auto_near']),
    ('Only long runs pushed, target from run count, every angle, stiff near walls', 'limit', ['p10i_cap_auto_any_near']),
    ('Only runs far too long pushed, target from run count, axes', 'limit', ['p10f_axis_auto1']),
    ('Only runs far too long pushed, target from run count, every angle', 'limit', ['p10f_any_auto1']),
    ('Same, target 1.5 × higher, axes', 'limit', ['p10f_axis_auto15']),
    ('Same, target 1.5 × higher, every angle', 'limit', ['p10f_any_auto15']),
    ('Cartogram, h 2, ε 0.3', 'density', ['carto_p10_h2_e0.3']),
    ('Learned flow, h 2, ε 0.3, transport 1', 'density', ['nf4_p10_w1']),
    ('Learned flow, no transport', 'density', ['nf4_p10_w0']),
]
G12 = [
    ('No warp', 'none', ['g12_ident']),
    ('Fixed target 0.10, along axes (coupling)', 'fixed', ['g12b_coupling_axis_cv']),
    ('Fixed target 0.10, along axes, high frequencies (coupling)', 'fixed', ['g12_coupling_cv_t10']),
    ('Fixed target 0.10, high frequencies (flow)', 'fixed', ['g12_flow_cv_t10']),
    ('Fixed target 0.10, no shear (separable)', 'sep', ['g12_sep_cv_t10']),
    ('Fixed target 0.10, every angle (coupling)', 'any', ['g12b_coupling_any_cv']),
    ('Fixed target 0.10, every angle, high frequencies (coupling)', 'any', ['g12_any_coupling_cv_t10']),
    ('Pair form, shift 0.10', 'pair', ['g12e_pair_t10']),
    ('Old loss with shear', 'old', ['g12c_hard']),
    ('Old loss, others held still within an update', 'limit', ['g12c_hard_anchor']),
    ('Old loss, update size capped', 'limit', ['g12c_hard_trust']),
    ('Old loss, stiff near walls', 'limit', ['g12c_hard_near', 'g12g_hard_near_seed1', 'g12g_hard_near_seed2']),
    ('Old loss, stiff near walls, every angle', 'limit', ['g12g_hard_any_near']),
    ('Old loss, stiffness: shape 0.1', 'limit', ['g12g_hard_near_s1']),
    ('Old loss, stiffness: shape 1', 'limit', ['g12g_hard_near_s10']),
    ('Old loss, stiffness: variation 0.05', 'limit', ['g12g_hard_near_t5']),
    ('Equal runs, stiff near walls', 'limit', ['g12g_eq_near', 'g12h_eq_near_seed1', 'g12h_eq_near_seed2']),
    ('Equal runs, every angle, stiff near walls', 'limit', ['g12g_eq_any_near']),
    ('Equal runs, target 0.07, stiff near walls', 'limit', ['g12h_eq_near_t07']),
    ('Equal runs, target 0.15, stiff near walls', 'limit', ['g12h_eq_near_t15']),
    ('Long runs pushed, stiff near walls, stretch bound 8', 'limit', ['g12k_k8', 'g12k_k8_seed1', 'g12k_k8_seed2']),
    ('Long runs pushed, stiff near walls, stretch bound 8, every angle', 'limit', ['g12k_k8_any']),
    ('Long runs pushed, stiff near walls, stretch bound 6', 'limit', ['g12k_k6']),
    ('Long runs pushed, stiff near walls, stretch bound 12', 'limit', ['g12k_k12']),
    ('Only long runs pushed, target from run count, stiff near walls', 'limit', ['g12j_auto_near']),
    ('Only long runs pushed, target from run count, every angle, stiff near walls', 'limit', ['g12j_auto_any_near']),
    ('Only long runs pushed, target 0.15, stiff near walls', 'limit', ['g12j_15_near']),
    ('Only long runs pushed, target 0.15, every angle, stiff near walls', 'limit', ['g12j_15_any_near']),
    ('Equal runs, worst run of a line weighted (power 2)', 'limit', ['g12c_pow2']),
    ('Equal runs, worst run of a line weighted (power 8)', 'limit', ['g12c_pow8']),
    ('Equal runs, only the worst 5 % of runs pushed', 'limit', ['g12c_top05']),
    ('Only long runs pushed, along axes', 'limit', ['g12b_coupling_axis_cap', 'g12b_coupling_axis_cap_seed1', 'g12b_coupling_axis_cap_seed2']),
    ('Only long runs pushed, every angle', 'limit', ['g12b_coupling_any_cap']),
    ('Longest runs first, along axes', 'limit', ['g12b_coupling_axis_capfirst']),
    ('Longest runs first, every angle', 'limit', ['g12b_coupling_any_capfirst', 'g12b_coupling_any_capfirst_seed1', 'g12b_coupling_any_capfirst_seed2']),
    ('Long runs, shape kept near walls 0.03, axes', 'limit', ['g12d_coupling_axis_shape03']),
    ('Long runs, shape kept near walls 0.03, every angle', 'limit', ['g12d_coupling_any_shape03']),
    ('Long runs, shape kept near walls 0.1, every angle', 'limit', ['g12d_coupling_any_shape1']),
    ('Only runs far too long pushed, target from run count, axes', 'limit', ['g12f_axis_auto1']),
    ('Only runs far too long pushed, target from run count, every angle', 'limit', ['g12f_any_auto1']),
    ('Same, target 1.5 × higher, axes', 'limit', ['g12f_axis_auto15']),
    ('Same, target 1.5 × higher, every angle', 'limit', ['g12f_any_auto15']),
    ('Only long runs pushed, along axes (flow)', 'limit', ['g12b_flow_axis_cap']),
    ('Only long runs pushed, every angle (flow)', 'limit', ['g12b_flow_any_cap']),
    ('Cartogram, h 2, ε 0.3', 'density', ['carto_g12_h2_e0.3']),
    ('Cartogram with the frame left in', 'density', ['carto_g12_h2_e0.3_noframe']),
    ('Learned flow, transport 1 (small network, 3 seeds)', 'density', ['nf4_g12_w1', 'nf5_g12_s1', 'nf5_g12_s2']),
    ('Learned flow, no transport', 'density', ['nf4_g12_w0']),
    ('Learned flow, transport 10', 'density', ['nf4_g12_w10']),
    ('Learned flow, 256 wide', 'density', ['nf5_g12_big']),
    ('Learned flow, 10 layers, 256 wide', 'density', ['nf5_g12_deep']),
    ('Learned time-integrated flow, h 1, ε 1, no transport', 'density', ['nf_g12_flow']),
]
SCENES = [   # key, name, cartogram label, flow label, separable flow label
    ('m10', '10×10 quadrants, ×12', 'carto_g12_h2_e0.3', 'nf4_g12_w1', None),
    ('c22', '22×22 quadrants, ×12', 'carto_c22g12_h2_e0.3', None, None),
    ('plus', 'plus maze (22×22)', 'carto_plus_h2_e0.3', None, None),
    ('r30g12', '10×10 quadrants, ×12, turned 30°', 'carto_r30g12_h2_e0.3', 'nf2_r30g12_coupling', 'nf2_r30g12_sep'),
    ('r30g4', '10×10 quadrants, ×4, turned 30°', 'carto_r30g4_h2_e0.3', None, None),
    ('r30p10', '10×10 maze, turned 30°', 'carto_r30p10_h2_e0.3', 'nf2_r30p10_coupling', 'nf2_r30p10_sep'),
    ('p10', '10×10 maze', 'carto_p10_h2_e0.3', 'nf4_p10_w1', None),
    ('gib0', 'Scanned floor (2-D)', 'carto_gib0_h2_e0.3', None, None),
    ('gib1', 'Scanned apartment (2-D)', 'carto_gib1_h2_e0.3', 'nf2_gib1_coupling', None),
    ('gib13', 'Scanned apartment (3-D)', 'carto_gib13_h2_e0.3', 'nf2_gib13_coupling', None),
    ('rooms', 'Room of slabs at random angles (3-D)', 'carto_rooms_h2_e0.3', 'nf2_rooms_coupling', None),
    ('se2p10', '10×10 maze, whole C-space (x, y, θ)', 'carto_se2p10_h2_e0.3', 'nf2_se2p10_coupling', None),
    ('se2g4', '10×10 quadrants ×4, whole C-space (x, y, θ)', 'carto_se2g4_h2_e0.3', 'nf2_se2g4_coupling', None),
    ('se2g12', '10×10 quadrants ×12, whole C-space (x, y, θ)', 'carto_se2g12_h2_e0.3', None, None),
]
ARM = [('h 1, ε 1', 'nf2_arm_h1e1'), ('h 2, ε 0.3', 'nf2_arm_h05e03'), ('h 0.25, ε 0.3', 'nf5_arm_a'), ('h 0.25, ε 0.1', 'nf5_arm_c'), ('h 0.1, ε 0.3', 'nf5_arm_b'), ('h 0.1, ε 0.1', 'nf5_arm_d'), ('h 0.05, ε 0.1', 'nf5_arm_e')]
WALL = ['wall_r2', 'wall_r2_quadrants', 'wall_r2_band', 'wall_z_r2']
KEYS = WALL + ['cv_len', 'cv_line', 'spacing_cv', 'spacing_cv_line', 'long_share', 'log_std_len', 'stripe_index', 'axis_gap', 'any_log_std_len', 'any_cv_line', 'any_long_share', 'any_stripe', 'runs_per_line', 'crossing_cv', 'block_speed_cv', 'peak_share', 'free_share',
        'det_positive', 'sv_q01', 'sv_q99', 'aniso_q50', 'aniso_q99', 'zone_q05', 'zone_q50', 'zone_q95', 'zone_log_std']


def run_dir(R, label):
    return ROOT / R[label]


def remeasured(R):
    """label -> measures, from the remeasure runs."""
    out = {}
    for k, v in R.items():
        if k.startswith('re_') and not str(v).startswith('FAILED'):
            d = json.load(open(ROOT / v / 'results.json'))
            for a, b in d['rows'].items():
                out[a] = dict(b['warp'], **{'wall_' + k: v for k, v in (b.get('wall') or {}).items()})
            out['@' + d['scene']] = dict(d['identity'], **{'wall_' + k: v for k, v in (d.get('identity_wall') or {}).items()})
    return out


def one(R, RM, label):
    """Everything about one run, or None if it did not finish."""
    v = R.get(label)
    if v is None or str(v).startswith('FAILED') or not (ROOT / v / 'results.json').exists():
        return None
    d = json.load(open(ROOT / v / 'results.json'))
    w = dict(d['warp'])
    w.update(RM.get(label, {}))
    d['identity'] = dict(d['identity'], **RM.get('@' + str(d.get('scene')), {}))
    sp = d.get('speed') or {}
    return dict(run=v.split('registered_')[1], **{k: w.get(k) for k in KEYS}, r2=(sp.get('4') or {}).get('r2'), r2_s1=(sp.get('1') or {}).get('r2'), corr=(sp.get('4') or {}).get('log_corr'), minutes=d.get('elapsed_seconds', 0) / 60, identity={k: d['identity'].get(k) for k in KEYS},
                r2_identity=((d.get('speed_identity') or {}).get('4') or {}).get('r2'), r2_s1_identity=((d.get('speed_identity') or {}).get('1') or {}).get('r2'))


def pooled(rows):
    """Mean of a list of runs of the same setting, with the range of R2."""
    rows = [r for r in rows if r]
    if not rows:
        return None
    out = dict(rows[0])
    for k in KEYS + ['r2', 'r2_s1', 'corr', 'minutes']:
        v = [r[k] for r in rows if r.get(k) is not None]
        out[k] = float(np.mean(v)) if v else None
    out['seeds'] = len(rows)
    out['runs'] = [r['run'] for r in rows]
    r2 = [r['r2'] for r in rows if r.get('r2') is not None]
    out['r2_range'] = [min(r2), max(r2)] if r2 else None
    for k in ('wall_r2', 'wall_r2_quadrants', 'wall_r2_band'):
        v = [r[k] for r in rows if r.get(k) is not None]
        out[k + '_range'] = [min(v), max(v)] if v else None
    return out


def tests(r, ident, r2_tol=0.01):
    """The decision rule of README.md against the identity. None where a measure is missing."""
    def lower(k):
        return None if r.get(k) is None or ident.get(k) is None else r[k] <= 0.7 * ident[k]
    t1 = [lower('cv_len'), lower('cv_line'), lower('spacing_cv')]
    need = ('stripe_index', 'any_stripe', 'any_long_share')
    t2 = None if any(r.get(k) is None or ident.get(k) is None for k in need) else bool(all(r[k] <= ident[k] for k in need))
    t3 = None if r.get('det_positive') is None else bool(r['det_positive'] >= 0.9995)
    t4 = None if r.get('r2') is None or r.get('r2_identity') is None else bool(r['r2'] >= r['r2_identity'] - 0.005)
    t5 = None if r.get('wall_r2') is None or ident.get('wall_r2') is None else bool(r['wall_r2'] >= ident['wall_r2'] - r2_tol)
    return dict(runs=None if None in t1 else all(t1), stripes=t2, valid=t3, speed=t4, walls=t5)


def table(R, RM, spec):
    out = []
    for name, group, labels in spec:
        p = pooled([one(R, RM, l) for l in labels])
        if p:
            p.update(name=name, group=group, labels=labels)
            p['tests'] = tests(p, p['identity'])
            p['passes'] = all(v is True for v in p['tests'].values())
            out.append(p)
    return out


def corr(x, y):
    x, y = np.array(x, float), np.array(y, float)
    ok = ~np.isnan(x) & ~np.isnan(y)
    rk = lambda v: v.argsort().argsort()
    return dict(pearson=float(np.corrcoef(x[ok], y[ok])[0, 1]), rank=float(np.corrcoef(rk(x[ok]), rk(y[ok]))[0, 1]), n=int(ok.sum()))


def calibration(R):
    FIT_ROW = dict(fitA2='stripes_final', fitB='stripes_any', rsA12='res_a12', rsA4='res_a4', resTV='res_tv', sep125='sep_h125', sep070='sep_h070', sepL125='sepL125', tv05='tv05', bt10='recipe', dil='dil', ntdil='ntdil', triTV='tri_tv', tv40='tv40')
    plan = {'plus': {k: LW['plus'][v]['pct'] for k, v in FIT_ROW.items() if v in LW['plus']}, 'm10': {k: v['pct'] for k, v in LW['ablation']['grid']['m10'].items()}, 'c22': {k: v['pct'] for k, v in LW['ablation']['grid']['c22'].items()}}
    plan['plus']['identity'] = LW['plus_controls']['ntrl-demo']['pct']
    plan['m10']['identity'], plan['c22']['identity'] = 26.7, 24.2
    names = {k: LW['plus'][v]['label'] for k, v in FIT_ROW.items() if v in LW['plus']}
    out = {}
    WS = dict(plus='ss_plusw', m10='ss_g12w', c22='ss_c22w')
    for base in ('plus', 'm10', 'c22'):
        s = json.load(open(ROOT / R['cal_' + base] / 'results.json'))['rows']
        z = json.load(open(ROOT / R['cal_zone_' + base] / 'results.json'))['rows']
        w = json.load(open(ROOT / R[WS[base]] / 'results.json'))['rows']
        alias = dict(both='recipe') if base == 'm10' else {}
        rows = []
        for k in s:
            if k in plan[base]:
                ww = w.get(alias.get(k, k)) or {}
                rows.append(dict(key=k, name=names.get(k, k), plan=plan[base][k], r2=s[k]['speed']['4']['r2'], corr=s[k]['speed']['4']['log_corr'], zone_q05=z[k]['zone']['zone_q05'], zone_q50=z[k]['zone']['zone_q50'], valid=s[k]['valid_share'], run=s[k]['run'], wall=ww.get('wall_mean'), wall0=(ww.get('wall_r2') or [None])[0]))
        rows.sort(key=lambda r: r['plan'])
        have = [r for r in rows if r['wall'] is not None]
        new = {k: dict(wall=v['wall_mean'], wall_sd=v['wall_sd'], r2=v['mean']) for k, v in w.items() if k in ('carto', 'flow', 'equal_stiff', 'old_stiff')}
        out[base] = dict(rows=rows, r2=corr([r['r2'] for r in rows], [r['plan'] for r in rows]), zone=corr([r['zone_q05'] for r in rows], [r['plan'] for r in rows]), wall=corr([r['wall'] for r in have], [r['plan'] for r in have]), new=new)
    return out


def recipe_rows(R):
    """line_warp's maps on this study's measures, with their learnability from the calibration."""
    if 'cal_all_m10' not in R:
        return {}
    a = json.load(open(ROOT / R['cal_all_m10'] / 'results.json'))['rows']
    s = json.load(open(ROOT / R['cal_m10'] / 'results.json'))['rows']
    return {k: dict(a[k]['warp'], r2=s[k]['speed']['4']['r2'], r2_s1=s[k]['speed']['1']['r2']) for k in a if 'warp' in a[k]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--state')
    a = p.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.state:
        s = json.load(open(a.state))
        (OUT / 'runs.json').write_text(json.dumps({k: v for k, v in s.items() if not k.startswith('_') and not str(v).startswith('FAILED')}, indent=0))
    R = json.load(open(OUT / 'runs.json'))
    RM = remeasured(R)
    out = dict(p10=table(R, RM, P10), g12=table(R, RM, G12), calibration=calibration(R), recipe=recipe_rows(R))
    out['scenes'] = []
    for key, name, c, f, s in SCENES:
        row = dict(key=key, name=name)
        for tag, l in (('carto', c), ('flow', f), ('sep', s)):
            if l:
                row[tag] = one(R, RM, l)
        extra = dict(r30g12='r30k_k8', c22='c22k_k8', plus='plusk_k8', gib1='gib1k_k8', se2g4='se2k_k8', m10='g12k_k8', p10='p10k_k8').get(key)
        if extra:
            row['lines'] = one(R, RM, extra)
        if key == 'r30g12':
            row['lines_any'] = one(R, RM, 'r30k_k8_any')
        ref = row.get('carto') or row.get('flow')
        if ref:
            row['identity'] = dict(ref['identity'], r2=ref['r2_identity'], r2_s1=ref['r2_s1_identity'])   # wall_* of the identity are in ref['identity'] once remeasured
            out['scenes'].append(row)
    arm = []
    for name, l in ARM:
        v = R.get(l)
        if v and (ROOT / v / 'results.json').exists():
            d = json.load(open(ROOT / v / 'results.json'))
            sp, si = d['speed'], d['speed_identity']
            k = '1'
            arm.append(dict(name=name, run=v.split('registered_')[1], r2=sp[k]['r2'], r2_identity=si[k]['r2'], **{m: d['warp'].get(m) for m in ('zone_q05', 'zone_q50', 'aniso_q50', 'crossing_cv', 'long_share')}, identity={m: d['identity'].get(m) for m in ('crossing_cv', 'long_share')}))
    out['arm'] = arm
    gaps = {}
    for base, tag, ref in (('m10', 'm10', 'm10k'), ('c22', 'c22', 'c22k')):
        g = []
        for k in (1, 2, 4, 8, 12):
            l = f'carto_{tag}k{k}_h2_e0.3' if k < 12 else ('carto_g12_h2_e0.3' if base == 'm10' else 'carto_c22g12_h2_e0.3')
            r = one(R, RM, l)
            if r:
                g.append(dict(k=k, zone_q50=r['zone_q50'], zone_q05=r['zone_q05'], r2=r['r2'], r2_identity=r['r2_identity'], spacing=r['spacing_cv'], spacing_identity=r['identity']['spacing_cv'], crossing=r['crossing_cv'], crossing_identity=r['identity'].get('crossing_cv'), run=r['run']))
        gaps[base] = g
    out['gaps'] = gaps
    sweep = {}
    for tag in ('g12', 'p10'):
        rows = []
        for h in ('0.5', '1', '2', '3'):
            for e in ('0.3', '1', '3'):
                r = one(R, RM, f'carto_{tag}_h{h}_e{e}')
                if r:
                    rows.append(dict(h=float(h), eps=float(e), r2=r['r2'], zone_q05=r['zone_q05'], zone_q50=r['zone_q50'], zone_log_std=r['zone_log_std'], aniso=r['aniso_q50'], spacing=r['spacing_cv'], long=r['long_share'], run=r['run']))
        sweep[tag] = rows
    out['sweep'] = sweep
    fresh = []
    for l in [f'carto_p10s{s}_h2_e0.3' for s in (1, 2, 3)] + [f'carto_g13s{s}_h2_e0.3' for s in (1, 2, 3)]:
        r = one(R, RM, l)
        if r:
            fresh.append(dict(label=l, r2=r['r2'], r2_identity=r['r2_identity'], aniso=r['aniso_q50'], sv=[r['sv_q01'], r['sv_q99']], run=r['run']))
    out['fresh'] = fresh
    out['noise'] = {k: json.load(open(ROOT / R[f'ss_{k}'] / 'results.json'))['rows'] for k in ('g12', 'p10') if f'ss_{k}' in R}
    out['regions'] = json.load(open(ROOT / R['ss_g12r'] / 'results.json'))['rows'] if 'ss_g12r' in R else {}
    out['walls10'] = json.load(open(ROOT / R['ss_p10w'] / 'results.json'))['rows'] if 'ss_p10w' in R else {}
    rz = {}
    for base in ('m10', 'c22'):
        rz[base] = {}
        for k in (1, 2, 4, 8):
            l = f'cal_zone_{base}k{k}'
            if l in R:
                rz[base][k] = json.load(open(ROOT / R[l] / 'results.json'))['rows']['both']['zone']['zone_q50']
        rz[base][12] = next(r['zone_q50'] for r in out['calibration'][base]['rows'] if r['key'] == 'both')
    out['recipe_zone'] = rz
    (OUT / 'summary.json').write_text(json.dumps(out, indent=1))
    print('p10 rows', len(out['p10']), 'g12 rows', len(out['g12']), 'scenes', len(out['scenes']), 'arm', len(arm))
    for t in ('p10', 'g12'):
        print(t)
        for r in out[t]:
            print(f"  {r['name'][:56]:56s} r2 {r['r2'] if r['r2'] is None else round(r['r2'], 3)} tests {[{True: 'y', False: 'n', None: '-'}[v] for v in r['tests'].values()]} seeds {r['seeds']}")


if __name__ == '__main__':
    main()
