"""Tables and summary of the octave-detune study from the registered runs: report/summary.json and report/tables.md.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/octave_detune/report.py

Fields: this study's runs (stage octdetune: exact grid-sync octaves and their detuned versions), and for context the
earlier runs on the same mazes: jeffrey/grid_sync octave / octaveperm / octavecircle, jeffrey/maze_scale ntrl-demo and
B x s. Success = MPPI successes over planning seeds 0/1/2 (results.json); the paired difference of a detuned field is
taken against the exact octave of the same training seed run in the same (maze, seed) group on the same GPU (this
study's rerun; on the 5070 it is bit-identical to the grid-sync run). Field measures at epoch 5,000: results.json
(octdetune) or the convergence record (earlier fields, report/conv). Convergence: report/conv/<id>.json (planning
seed 0 at every evaluated checkpoint, per-epoch loss and epoch rollbacks from the registry console log).
"""
import json
import math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'report'
CONV = OUT / 'conv'
RINGS = ('inner', 'middle', 'outer')


def spatial_record(rid):
    """Near-wall slope of a run by ring of the maze (inner / middle / outer third, spatial.py), and outer minus inner."""
    p = OUT / 'spatial.json'
    if not hasattr(spatial_record, 'cache'):
        spatial_record.cache = json.loads(p.read_text()) if p.exists() else {}
    ring = (spatial_record.cache.get(rid) or {}).get('field') or {}
    sl = {r: (ring.get(r) or {}).get('slope') for r in RINGS}
    out = {f'slope_ring_{r}': sl[r] for r in RINGS}
    out['slope_outer_minus_inner'] = sl['outer'] - sl['inner'] if sl['outer'] is not None and sl['inner'] is not None else None
    return out
PREPARED = {13: '20261101T192504Z-13cf1d04', 22: '20261101T193403Z-cea05796'}
SIZES = [13, 22]
BINS = ['0.5-2', '2-4', '4-8', '8-16', '16-32', '32-64']
NULL = 'shift1.0001'
NULL2 = 'shift0.9999'
SHIFTS = ['shift0.5', 'shift0.707', 'shift0.841', 'shift0.9', 'shift0.95', 'shift0.9999', 'octave', 'shift1.0001', 'shift1.05', 'shift1.1', 'shift1.189', 'shift1.414', 'shift2']
JITTERS = ['octave', 'jitter0.05', 'jitter0.1', 'jitter0.2', 'jitter0.4142']
BAND = ['top0.9', 'rest0.9']
LEVELS = ['levels0.05', 'levels0.1']
CONTROLS = ['octaveperm', 'octavecircle', 'B x 3', 'B x 6', 'ntrl-demo']


def label(key):
    if key == 'octave':
        return 'octave (exact)'
    if key == NULL:
        return 'null: x1.0001'
    if key == NULL2:
        return 'null: x0.9999'
    if key.startswith('shift'):
        s = float(key[5:])
        return f'shift x{s:g} (2^{math.log2(s):+.2f})'
    if key == 'jitter0.4142':
        return 'log-uniform fill (jitter ±half an octave)'
    if key.startswith('jitter'):
        return f'jitter ±{100 * float(key[6:]):g} %'
    if key.startswith('levels'):
        return f'levels ±{100 * float(key[6:]):g} % alternating'
    if key.startswith('top'):
        return f'top octave only x{float(key[3:]):g}'
    if key.startswith('rest'):
        return f'all but the top octave x{float(key[4:]):g}'
    return dict(octaveperm='octaveperm (magnitudes shuffled, grid_sync)', octavecircle='octavecircle (constant |f|, grid_sync)').get(key, key)


def log2_shift(key):
    if key == 'octave':
        return 0.0
    if key.startswith('shift'):
        return math.log2(float(key[5:]))
    return None


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def runs():
    want = {str((ROOT / f'results/registered_{v}').resolve()): n for n, v in PREPARED.items()}
    out = []
    for r in sorted(ROOT.glob('results/registered_*/results.json')):
        d = load(r)
        if not d or not d.get('complete') or d.get('smoke') or d.get('epochs') != 5000 or d.get('variant', 'metric') != 'metric':
            continue
        n = want.get(str(Path(d.get('prepared', '')).resolve()))
        if n is None:
            continue
        st = d.get('stage')
        if st == 'octdetune':
            key = d['spec']
        elif st == 'gridsync' and d.get('mode') in ('octave', 'octaveperm', 'octavecircle'):
            key = 'gs:' + d['mode']
        elif st == 'ntrl':
            key = 'ntrl-demo'
        elif st == 'fourier' and d.get('b_scale') in (3.0, 6.0):
            key = f'B x {d["b_scale"]:g}'
        else:
            continue
        d['_id'], d['_dir'], d['_n'], d['_key'] = r.parent.name.replace('registered_', ''), r.parent, n, key
        out.append(d)
    return out


def console_stats(rec):
    if not rec or not rec.get('loss'):
        return {}
    loss = np.array([np.nan if v is None else v for v in rec['loss']], float)
    fin = float(np.nanmean(loss[-100:]))
    out = dict(loss_final_mean100=fin, rollbacks=int(sum(rec.get('repeats', {}).values())))
    for e in (100, 500, 1000, 1500, 2000, 3000):
        out[f'loss_mean100_at_{e}'] = float(np.nanmean(loss[max(0, e - 100):e]))
    # settling epoch: first epoch whose trailing 100-epoch mean is within 10 % of the final 100-epoch mean
    trail = np.convolve(np.nan_to_num(loss, nan=np.nanmax(loss)), np.ones(100) / 100, mode='valid')
    hit = np.nonzero(trail <= 1.1 * fin)[0]
    out['loss_settle_epoch'] = int(hit[0] + 100) if len(hit) else None
    return out


def rows_stats(path, prepared):
    r = np.load(path)
    ok, conv, col = r['success'], r['converged'], r['collision']
    q = np.load(Path(prepared) / 'queries.npz')
    err = np.abs(np.log(np.maximum(r['t_pred'], 1e-9) / q['t_star']))
    return dict(ok=ok, never_arrived=int((~ok & ~conv).sum()), arrived_after_wall=int((~ok & conv & col).sum()),
                log_err=float(np.median(err)), log_err_by_bin={BINS[b]: float(np.median(err[q['bin'] == b])) for b in sorted(set(q['bin'].tolist()))},
                length_ratio=float(np.nanmedian(r['length_ratio'])) if np.isfinite(r['length_ratio']).any() else None,
                time_ratio=float(np.nanmedian(r['time_ratio'])) if np.isfinite(r['time_ratio']).any() else None)


def field_record(d):
    """Every final-checkpoint measure of a run in one flat dict."""
    rs = rows_stats(d['_dir'] / 'rows.npz', d['prepared'])
    conv = load(CONV / f'{d["_id"]}.json')
    src = d if d['stage'] == 'octdetune' else ((conv or {}).get('epochs', {}).get('5000') or {})
    w = src.get('wall', {})
    pr = src.get('profiles', {})
    bu = src.get('bumps', {})
    rec = dict(id=d['_id'], key=d['_key'], n=d['_n'], seed=d['seed'], gpu=d.get('gpu'), success=d['success_total'][0], runs=d['success_total'][1],
               by_bin={k: v[0] / v[1] for k, v in d['success_by_bin'].items()},
               never_arrived=rs['never_arrived'], arrived_after_wall=rs['arrived_after_wall'],
               log_err=rs['log_err'], log_err_by_bin=rs['log_err_by_bin'], length_ratio=rs['length_ratio'], time_ratio=rs['time_ratio'],
               t_over_tstar=d['field']['t_over_tstar_median'], spearman=d['field']['spearman'], descent_cos=d['field']['descent_cos_median'],
               speed_corr=d['speed_fit']['log_speed_corr'], speed_mae=d['speed_fit']['log_speed_mae'],
               slope_2_8=w.get('2-8', {}).get('slope'), cos_2_8=w.get('2-8', {}).get('cos'),
               slope_05_2=w.get('0.5-2', {}).get('slope'), cos_05_2=w.get('0.5-2', {}).get('cos'),
               bumps=bu.get('bumps_per_path'), doorway_dip=pr.get('doorway_dip_median'), phantom=pr.get('doorway_phantom_share'),
               wall_S035=pr.get('wall_S_at_035'), steps_median=float(np.median([v['steps_median'] for v in d['planning'].values()])),
               features=d.get('features'))
    rec.update(console_stats(conv))
    rec.update(spatial_record(d['_id']))
    rec['_ok'] = rs['ok']
    if conv:
        rec['curve'] = {int(e): dict(success=v['success_total'][0] / v['success_total'][1], log_err=v['accuracy']['log_err_median'],
                                     speed_corr=v['speed_fit']['log_speed_corr'], slope=v['wall'].get('2-8', {}).get('slope'),
                                     bumps=v['bumps']['bumps_per_path'], never=v['failures']['never_arrived'] / v['failures']['runs'],
                                     wall=v['failures']['arrived_after_wall'] / v['failures']['runs'], loss=v.get('checkpoint_loss'))
                        for e, v in conv['epochs'].items()}
        if conv.get('loss'):
            lo = np.array([np.nan if v is None else v for v in conv['loss']], float)
            rec['loss_curve'] = [float(np.nanmean(lo[i:i + 50])) for i in range(0, 5000, 50)]
    return rec


def collect():
    table = {}
    for d in runs():
        try:
            rec = field_record(d)
        except Exception as e:  # noqa: BLE001
            print('skip', d['_id'], d['_key'], e)
            continue
        key = d['_key']
        # earlier grid-sync runs: kept as 'gs:<mode>'; their octave is the reference only where this study has none
        table.setdefault(d['_n'], {}).setdefault(key, {})[d['seed']] = rec
    for n in table:
        for mode in ('octaveperm', 'octavecircle'):
            if 'gs:' + mode in table[n]:
                table[n][mode] = table[n].pop('gs:' + mode)
    return table


REF_SET = ('octave', NULL, NULL2)


def ref_runs(table, n, seed, exclude=None):
    """Every training of this seed with the exact octave frequencies (to 1e-4): this study's exact octave and both nulls,
    plus the grid-sync run where it ran on another GPU (on the same GPU it is bit-identical to the rerun)."""
    T = table.get(n, {})
    out = [(k, T[k][seed]) for k in REF_SET if seed in T.get(k, {}) and k != exclude]
    g, o = T.get('gs:octave', {}).get(seed), T.get('octave', {}).get(seed)
    if g is not None and (o is None or g['gpu'] != o['gpu']) and exclude != 'gs:octave':
        out.append(('gs:octave', g))
    return [r for _, r in out]


def paired(table, n, key):
    """Per seed: success(key) minus the mean success of that seed's exact-frequency trainings (a null is compared with the
    others); gained / lost runs (query x planning seed) against this study's exact octave run of the seed."""
    out = {}
    for s, r in table.get(n, {}).get(key, {}).items():
        refs = ref_runs(table, n, s, exclude=key if key in REF_SET else None)
        if not refs:
            continue
        ref = float(np.mean([x['success'] / x['runs'] for x in refs]))
        o = table.get(n, {}).get('octave', {}).get(s) or table.get(n, {}).get('gs:octave', {}).get(s)
        a, b = r['_ok'], o['_ok']
        out[s] = dict(diff=r['success'] / r['runs'] - ref, m=len(refs), gained=int((a & ~b).sum()), lost=int((~a & b).sum()), runs=r['runs'])
    return out


PAIRED_METRICS = ['success_rate', 'log_err', 'speed_corr', 'slope_2_8', 'cos_2_8', 'bumps', 'never_arrived_rate', 'wall_rate',
                  'time_ratio', 'loss_final_mean100', 'loss_settle_epoch', 'rollbacks', 'succ_at_1000', 'succ_at_2000',
                  'slope_ring_inner', 'slope_ring_middle', 'slope_ring_outer', 'slope_outer_minus_inner']


def metric(rec, m):
    if m == 'success_rate':
        return rec['success'] / rec['runs']
    if m == 'never_arrived_rate':
        return rec['never_arrived'] / rec['runs']
    if m == 'wall_rate':
        return rec['arrived_after_wall'] / rec['runs']
    if m in ('succ_at_1000', 'succ_at_2000'):
        c = (rec.get('curve') or {}).get(int(m.split('_')[-1]))
        return c['success'] if c else None
    if m == 'loss_final_mean100':
        v = rec.get(m)
        return None if v is None else math.log(v)          # relative change: difference of logs
    return rec.get(m)


def noise_metrics(table, n):
    """Training noise of one run, per measure: the pooled within-seed SD of the exact-frequency trainings (ref_runs)."""
    out = {}
    for m in PAIRED_METRICS:
        ss, dof, samples = 0.0, 0, 0
        for s in table.get(n, {}).get('octave', {}):
            xs = [metric(r, m) for r in ref_runs(table, n, s)]
            xs = [x for x in xs if x is not None]
            if len(xs) >= 2:
                ss += float(np.sum((np.array(xs) - np.mean(xs)) ** 2))
                dof += len(xs) - 1
                samples += len(xs)
        out[m] = dict(sd=math.sqrt(ss / dof) if dof else None, dof=dof, runs=samples)
    return out


def paired_metrics(table, n, key, sd):
    """Mean over seeds of measure(key) minus the mean of that seed's exact-frequency trainings, its standard error from the
    training noise (sd: noise_metrics) and z = mean / se."""
    out = {}
    for m in PAIRED_METRICS:
        deltas, var = {}, 0.0
        for s, r in table.get(n, {}).get(key, {}).items():
            refs = [metric(x, m) for x in ref_runs(table, n, s, exclude=key if key in REF_SET else None)]
            refs = [x for x in refs if x is not None]
            a = metric(r, m)
            if a is None or not refs:
                continue
            deltas[str(s)] = a - float(np.mean(refs))
            var += 1 + 1 / len(refs)
        if deltas:
            k = len(deltas)
            se = sd[m]['sd'] * math.sqrt(var) / k if sd.get(m, {}).get('sd') else None
            mean = float(np.mean(list(deltas.values())))
            out[m] = dict(deltas=deltas, mean=mean, k=k, se=se, z=mean / se if se else None)
    return out


GROUPS = {
    # every detuning that keeps the band where it is or moves it up: the exact values (and the lock) gone, the level kept
    'not_lower': ('shift1.05', 'shift1.1', 'jitter0.05', 'jitter0.1', 'jitter0.2', 'jitter0.4142'),
    # the same band and level, each octave's geometric mean exact, the rows spread off the octave values (no discrete frequency
    # at all in the fill)
    'same_band': ('jitter0.05', 'jitter0.1', 'jitter0.2', 'jitter0.4142'),
    # the near-wall slope by ring of the maze: a lock effect would lose slope in the outer third and keep it in the inner third
    'ring_down': ('shift0.95', 'shift0.9'),
    'ring_up': ('shift1.05', 'shift1.1'),
    'ring_jitter': ('jitter0.05', 'jitter0.1', 'jitter0.2'),
}


def group_metrics(table, n, keys, sd):
    """Pooled paired change over several variants: D = mean over (variant, seed) of measure minus the seed's exact-frequency
    mean. The variants of a seed share that reference, so Var D = sd^2 / N^2 (N + sum_s c_s^2 / m_s), N pairs, c_s variants
    of seed s, m_s reference runs of seed s (one variant alone: the paired_metrics formula)."""
    out = {}
    for m in PAIRED_METRICS:
        deltas, per_seed, refs_n = [], {}, {}
        for key in keys:
            for s, r in table.get(n, {}).get(key, {}).items():
                refs = [metric(x, m) for x in ref_runs(table, n, s)]
                refs = [x for x in refs if x is not None]
                a = metric(r, m)
                if a is None or not refs:
                    continue
                deltas.append(a - float(np.mean(refs)))
                per_seed[s] = per_seed.get(s, 0) + 1
                refs_n[s] = len(refs)
        if not deltas or not sd.get(m, {}).get('sd'):
            continue
        N = len(deltas)
        se = sd[m]['sd'] * math.sqrt(N + sum(c * c / refs_n[s] for s, c in per_seed.items())) / N
        mean = float(np.mean(deltas))
        out[m] = dict(mean=mean, se=se, z=mean / se, pairs=N, variants=[k for k in keys if table.get(n, {}).get(k)])
    return out


def pooled(recs):
    if not recs:
        return None
    k = sum(r['success'] for r in recs.values())
    m = sum(r['runs'] for r in recs.values())
    return dict(success=k, runs=m, rate=k / m, per_seed={str(s): r['success'] for s, r in sorted(recs.items(), key=lambda kv: (kv[0] != 87358, kv[0]))})


def noise(table):
    """Training noise: the exact-frequency trainings of each seed (exact octave, nulls x1.0001 / x0.9999, grid-sync run on
    the other GPU) and the pooled within-seed SD of their success (points); same-GPU reruns of the exact octave (runs)."""
    out = {}
    for n in SIZES:
        T = table.get(n, {})
        per_seed = {}
        for s in sorted(T.get('octave', {}), key=lambda s: (s != 87358, s)):
            per_seed[str(s)] = [round(100 * r['success'] / r['runs'], 2) for r in ref_runs(table, n, s)]
        same = [T['octave'][s]['success'] - g['success'] for s, g in T.get('gs:octave', {}).items() if s in T.get('octave', {}) and g['gpu'] == T['octave'][s]['gpu']]
        ms = noise_metrics(table, n)
        sd = ms['success_rate']['sd']
        out[n] = dict(per_seed_points=per_seed, same_gpu_rerun_diffs=same, sd_run_points=100 * sd if sd else None,
                      dof=ms['success_rate']['dof'], metric_sd=ms)
    return out


def summarize(table):
    keys = []
    for n in SIZES:
        keys += [k for k in table.get(n, {}) if k not in keys]
    order = SHIFTS + [k for k in JITTERS if k != 'octave'] + BAND + LEVELS + CONTROLS + ['gs:octave']
    keys = [k for k in order if k in keys] + sorted(k for k in keys if k not in order)
    S = dict(noise=noise(table), fields={})
    for key in keys:
        S['fields'][key] = {}
        for n in SIZES:
            recs = table.get(n, {}).get(key)
            if not recs:
                continue
            # every field is compared, seed by seed, with the mean of that seed's exact-frequency trainings (ref_runs)
            pr = paired(table, n, key) if key not in ('octave', 'gs:octave') else {}
            diffs = [100 * v['diff'] for v in pr.values()]
            agg = {}
            for m in ('log_err', 't_over_tstar', 'spearman', 'descent_cos', 'speed_corr', 'speed_mae', 'slope_2_8', 'cos_2_8', 'slope_05_2', 'cos_05_2',
                      'bumps', 'doorway_dip', 'phantom', 'wall_S035', 'length_ratio', 'time_ratio', 'steps_median', 'loss_final_mean100', 'rollbacks',
                      'loss_settle_epoch', 'loss_mean100_at_1000', 'loss_mean100_at_2000'):
                vals = [r.get(m) for r in recs.values() if r.get(m) is not None]
                agg[m] = float(np.mean(vals)) if vals else None
            runs_ = sum(r['runs'] for r in recs.values())
            agg['never_arrived_rate'] = sum(r['never_arrived'] for r in recs.values()) / runs_
            agg['wall_rate'] = sum(r['arrived_after_wall'] for r in recs.values()) / runs_
            agg['by_bin'] = {b: float(np.mean([r['by_bin'][b] for r in recs.values() if b in r['by_bin']])) for b in BINS if any(b in r['by_bin'] for r in recs.values())}
            curves = [r['curve'] for r in recs.values() if r.get('curve')]
            common = sorted(set.intersection(*[set(c) for c in curves])) if curves else []
            agg['curve'] = {str(e): {m: float(np.mean([c[e][m] for c in curves if c[e].get(m) is not None])) if any(c[e].get(m) is not None for c in curves) else None
                                     for m in ('success', 'log_err', 'speed_corr', 'slope', 'bumps', 'never', 'wall', 'loss')} for e in common}
            agg['curve_seeds'] = len(curves)
            # per-seed curves, so the page can average every variant over the same seeds (22x22 octave, nulls, x1.41 and x2 add seeds 3, 4)
            agg['curve_by_seed'] = {str(s): {str(e): c for e, c in r['curve'].items()} for s, r in recs.items() if r.get('curve')}
            agg['loss_curve_by_seed'] = {str(s): r['loss_curve'] for s, r in recs.items() if r.get('loss_curve')}
            for e in (1000, 2000):
                c = agg['curve'].get(str(e))
                agg[f'succ_at_{e}'] = c['success'] if c else None
            if agg['curve']:
                es = [e for e in agg['curve'] if int(e) >= 1000]
                agg['succ_curve_mean'] = float(np.mean([agg['curve'][e]['success'] for e in es])) if es else None
            lcs = [r['loss_curve'] for r in recs.values() if r.get('loss_curve')]
            agg['loss_curve_median'] = np.median(np.array(lcs), 0).tolist() if lcs else None
            f = next(iter(recs.values())).get('features') or {}
            pm = paired_metrics(table, n, key, S['noise'][n]['metric_sd']) if key not in ('octave', 'gs:octave') else {}
            sr = pm.get('success_rate', {})
            S['fields'][key][str(n)] = dict(label=label(key), pooled=pooled(recs), paired=pr, paired_metrics=pm,
                                            diff_mean_points=float(np.mean(diffs)) if diffs else None,
                                            diff_sd_points=float(np.std(diffs, ddof=1)) if len(diffs) > 1 else None,
                                            diff_se_points=100 * sr['se'] if sr.get('se') else None, diff_z=sr.get('z'),
                                            wins=sum(d > 0 for d in diffs), losses=sum(d < 0 for d in diffs),
                                            gained=sum(v['gained'] for v in pr.values()), lost=sum(v['lost'] for v in pr.values()),
                                            edge_phase_error_median=f.get('edge_phase_error_median'), edge_phase_error_top=f.get('edge_phase_error_top_octave_median'),
                                            locked_rows=f.get('locked_rows'), log2_shift=log2_shift(key), seeds=sorted(recs), **agg)
    S['groups'] = {g: {str(n): group_metrics(table, n, keys_, S['noise'][n]['metric_sd']) for n in SIZES} for g, keys_ in GROUPS.items()}
    return S


def fmt(v, p=2):
    return '–' if v is None else (f'{v:.{p}f}' if isinstance(v, float) else str(v))


def tables(S):
    md = ['# Octave-detune tables (generated by report.py)', '']
    nz = S['noise']
    md += ['## Training noise', '',
           'Success (%) of every training of a seed with the exact octave frequencies (to 1e-4): the exact octave, the nulls × 1.0001 / × 0.9999, '
           'and the grid-sync run where it ran on the other GPU; pooled within-seed SD of one run; reruns on the same GPU (difference in runs, 0 = bit-identical).', '']
    for n in SIZES:
        v = nz.get(n, nz.get(str(n)))
        if v:
            md.append(f'- {n}×{n}: ' + '; '.join(f'seed {s}: {", ".join(f"{x:.1f}" for x in xs)}' for s, xs in v['per_seed_points'].items())
                      + f'; SD of one run {fmt(v["sd_run_points"], 2)} points ({v["dof"]} degrees of freedom); same-GPU reruns {v["same_gpu_rerun_diffs"]}')
    md += ['', '## Success (planning seeds 0/1/2 pooled over training seeds) and the paired difference to the exact-frequency trainings', '',
           'Δ: success minus the mean of the same seed\'s exact-frequency trainings, mean over seeds ± its standard error from the training noise (z); '
           'gained / lost: runs (query × planning seed) against the exact octave run of the seed.', '',
           '| field | ' + ' | '.join(f'{n}×{n} success [per seed] | Δ points ± se (z; wins/losses) | gained / lost runs' for n in SIZES) + ' |',
           '|---|' + '---|---|---|' * len(SIZES)]
    for key, by_n in S['fields'].items():
        row = [label(key)]
        for n in SIZES:
            v = by_n.get(str(n))
            if v is None:
                row += ['–', '–', '–']
                continue
            p = v['pooled']
            row.append(f'{100 * p["rate"]:.1f} % ({p["success"]}/{p["runs"]}) [{" / ".join(map(str, p["per_seed"].values()))}]')
            z = '–' if v['diff_z'] is None else f'{v["diff_z"]:+.1f}'
            row.append('–' if v['diff_mean_points'] is None else f'{v["diff_mean_points"]:+.1f} ± {fmt(v["diff_se_points"], 1)} ({z}; {v["wins"]}/{v["losses"]})')
            row.append('–' if not v['paired'] else f'{v["gained"]} / {v["lost"]}')
        md.append('| ' + ' | '.join(row) + ' |')
    metrics = [('log_err', 'median |log T/T*|', 3), ('speed_corr', 'speed fit corr', 3), ('slope_2_8', 'near-wall slope (gap 2-8)', 2),
               ('cos_2_8', 'near-wall cos (2-8)', 2), ('bumps', 'local minima per long path', 2), ('phantom', 'phantom doorways', 3),
               ('never_arrived_rate', 'never arrived', 3), ('wall_rate', 'arrived after a wall', 3), ('length_ratio', 'path length / truth', 3),
               ('time_ratio', 'travel time / truth', 3), ('loss_final_mean100', 'final loss (mean of last 100 epochs)', 6),
               ('loss_settle_epoch', 'loss settling epoch', 0), ('rollbacks', 'epoch rollbacks', 1), ('edge_phase_error_top', 'top-octave edge phase error (rad)', 2)]
    for n in SIZES:
        md += ['', f'## Field measures at epoch 5,000, {n}×{n} (mean over training seeds)', '',
               '| field | ' + ' | '.join(m[1] for m in metrics) + ' |', '|---|' + '---|' * len(metrics)]
        for key, by_n in S['fields'].items():
            v = by_n.get(str(n))
            if v is None:
                continue
            md.append(f'| {label(key)} | ' + ' | '.join(fmt(v.get(m), p) if p else fmt(None if v.get(m) is None else int(round(v[m]))) for m, _, p in metrics) + ' |')
    for n in SIZES:
        md += ['', f'## Success by distance bin, {n}×{n} (mean over training seeds)', '', '| field | ' + ' | '.join(BINS) + ' |', '|---|' + '---|' * len(BINS)]
        for key, by_n in S['fields'].items():
            v = by_n.get(str(n))
            if v is None:
                continue
            md.append(f'| {label(key)} | ' + ' | '.join(f'{100 * v["by_bin"][b]:.0f} %' if b in v['by_bin'] else '–' for b in BINS) + ' |')
    for n in SIZES:
        md += ['', f'## Convergence, {n}×{n}: success with planning seed 0 at the evaluated checkpoints (mean over training seeds)', '']
        epochs = sorted({int(e) for v in S['fields'].values() if str(n) in v for e in v[str(n)]['curve']})
        md += ['| field | ' + ' | '.join(str(e) for e in epochs) + ' |', '|---|' + '---|' * len(epochs)]
        for key, by_n in S['fields'].items():
            v = by_n.get(str(n))
            if v is None:
                continue
            md.append(f'| {label(key)} ({v["curve_seeds"]} seeds) | ' + ' | '.join(
                f'{100 * v["curve"][str(e)]["success"]:.0f} %' if str(e) in v['curve'] else '–' for e in epochs) + ' |')
    return '\n'.join(md) + '\n'


def main():
    OUT.mkdir(exist_ok=True)
    table = collect()
    S = summarize(table)
    (OUT / 'summary.json').write_text(json.dumps(S, indent=1, default=float))
    (OUT / 'tables.md').write_text(tables(S))
    print(tables(S))


if __name__ == '__main__':
    main()
