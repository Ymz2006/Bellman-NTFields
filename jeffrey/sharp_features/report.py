"""Tables and summary of the sharp-feature study from the registered runs: report/summary.json and report/tables.md.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/sharp_features/report.py

Fields: this study's runs (stage 'sharp', key = spec) and, as the reference, every earlier training of the same maze and
seed with the exact octave embedding: jeffrey/octave_detune's 'octave' and its nulls x1.0001 / x0.9999 (the same
frequencies to 1e-4, a different floating-point trajectory) and jeffrey/grid_sync's octave run where it ran on the
other GPU (on the same GPU it is bit-identical to octave_detune's rerun, counted once). Context: ntrl-demo
(jeffrey/maze_scale) and octave_detune's x2 shift (every row at twice its frequency: sin / cos at the frequency of a
squared row). Success = MPPI successes over planning seeds 0/1/2.

Paired change of a spec = mean over training seeds of (its success - the mean success of that seed's exact-octave
trainings); its SE from the training noise: Var = sd^2 / k^2 sum_s (1 + 1 / m_s) (k seeds, m_s reference runs of seed
s, sd = pooled within-seed SD of the reference trainings). Against a control spec of the same width (add specs): the
difference of the two runs of each seed, Var = 2 sd^2 / k.
"""
import json
import math
import re
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'report'
PREPARED = {13: '20261101T192504Z-13cf1d04', 22: '20261101T193403Z-cea05796'}
SIZES = [13, 22]
BINS = ['0.5-2', '2-4', '4-8', '8-16', '16-32', '32-64']
REF_KEYS = ('ref:octave', 'ref:null+', 'ref:null-')
CONTROLS = ('Asc@new128', 'Adup@all')            # same width (512) as every 2-per-row add spec
METRICS = ['success_rate', 'never_arrived_rate', 'wall_rate', 'log_err', 'speed_corr', 'slope_2_8', 'cos_2_8', 'bumps',
           'phantom', 'wall_S035', 'loss_final', 'loss_final100', 'loss_at_1000', 'loss_at_2000', 'rollbacks', 'time_ratio', 'length_ratio',
           'uphill', 'eik_res', 'minima', 'depth_q90', 'stop_open', 'stop_open_route', 'stop_wall']


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def key_of(d):
    st = d.get('stage')
    if st == 'sharp':
        return d['spec']
    if st == 'octdetune':
        return {'octave': 'ref:octave', 'shift1.0001': 'ref:null+', 'shift0.9999': 'ref:null-', 'shift2': 'ctx:shift2',
                'shift0.5': 'ctx:shift0.5'}.get(d['spec'])
    if st == 'gridsync' and d.get('mode') == 'octave':
        return 'gs:octave'
    if st == 'ntrl':
        return 'ntrl-demo'
    return None


def runs():
    want = {str((ROOT / f'results/registered_{v}').resolve()): n for n, v in PREPARED.items()}
    out = []
    for r in sorted(ROOT.glob('results/registered_*/results.json')):
        d = load(r)
        if not d or not d.get('complete') or d.get('smoke') or d.get('epochs') != 5000 or d.get('variant', 'metric') != 'metric':
            continue
        n = want.get(str(Path(d.get('prepared', '')).resolve()))
        key = key_of(d) if n else None
        if key is None:
            continue
        d['_id'], d['_dir'], d['_n'], d['_key'] = r.parent.name.replace('registered_', ''), r.parent, n, key
        out.append(d)
    return out


def rows_stats(path, prepared):
    r = np.load(path)
    ok, conv, col = r['success'], r['converged'], r['collision']
    q = np.load(Path(prepared) / 'queries.npz')
    err = np.abs(np.log(np.maximum(r['t_pred'], 1e-9) / q['t_star']))
    return dict(ok=ok, never_arrived=int((~ok & ~conv).sum()), arrived_after_wall=int((~ok & conv & col).sum()),
                log_err=float(np.median(err)),
                length_ratio=float(np.nanmedian(r['length_ratio'])) if np.isfinite(r['length_ratio']).any() else None,
                time_ratio=float(np.nanmedian(r['time_ratio'])) if np.isfinite(r['time_ratio']).any() else None)


CONSOLE = Path.home() / '.local/share/ntfields-experiments/runs'


def console_stats(rid):
    """Per-epoch training loss from the registry console log (ntrl-demo prints it): mean of the last 100 epochs, the mean at
    epochs 900-1000 and 1900-2000, and the number of epoch rollbacks. Cached in report/console.json."""
    cache = OUT / 'console.json'
    if not hasattr(console_stats, 'cache'):
        console_stats.cache = json.loads(cache.read_text()) if cache.exists() else {}
    if rid in console_stats.cache:
        return console_stats.cache[rid]
    path = CONSOLE / rid / 'console.log'
    if not path.exists():
        return {}
    loss, repeats = {}, 0
    for line in path.read_text(errors='replace').splitlines():
        if line.startswith('Epoch = '):
            m = re.match(r'Epoch = (\d+) -- Loss = ([0-9.e+-]+)', line)
            if m:
                loss[int(m.group(1))] = float(m.group(2))
        elif line.startswith('RepeatEpoch'):
            repeats += 1
    if len(loss) < 5000:
        return {}
    arr = np.array([loss[e] for e in sorted(loss)])
    out = dict(loss_final100=float(arr[-100:].mean()), loss_at_1000=float(arr[900:1000].mean()), loss_at_2000=float(arr[1900:2000].mean()),
               rollbacks=repeats, loss_curve=[float(arr[i:i + 100].mean()) for i in range(0, 5000, 100)])
    console_stats.cache[rid] = out
    OUT.mkdir(exist_ok=True)
    cache.write_text(json.dumps(console_stats.cache))
    return out


def stall_of(rid):
    """Where the never-arrived runs of planning seed 0 stop (stalls.py)."""
    if not hasattr(stall_of, 'cache'):
        p = OUT / 'stalls.json'
        stall_of.cache = json.loads(p.read_text()) if p.exists() else {}
    return stall_of.cache.get(rid) or {}


def ripple_of(rid):
    """Travel time along the true paths (ripple.py): uphill share, eikonal residual, local minima, their depth."""
    if not hasattr(ripple_of, 'cache'):
        p = OUT / 'ripple.json'
        ripple_of.cache = json.loads(p.read_text()) if p.exists() else {}
    return ripple_of.cache.get(rid) or {}


def conv_of(rid):
    """Earlier reference runs keep their field measures in octave_detune's convergence record."""
    return load(ROOT / 'jeffrey/octave_detune/report/conv' / f'{rid}.json')


def record(d):
    rs = rows_stats(d['_dir'] / 'rows.npz', d['prepared'])
    if d['stage'] in ('sharp', 'octdetune'):
        src = d
    else:
        src = ((conv_of(d['_id']) or {}).get('epochs', {}).get('5000') or {})
    w = src.get('wall', {})
    pr = src.get('profiles', {})
    bu = src.get('bumps', {})
    sf = src.get('speed_fit', d.get('speed_fit', {})) or {}
    rec = dict(id=d['_id'], key=d['_key'], n=d['_n'], seed=d['seed'], gpu=d.get('gpu'), success=d['success_total'][0], runs=d['success_total'][1],
               by_bin={k: v[0] / v[1] for k, v in d['success_by_bin'].items()},
               never_arrived=rs['never_arrived'], arrived_after_wall=rs['arrived_after_wall'], log_err=rs['log_err'],
               length_ratio=rs['length_ratio'], time_ratio=rs['time_ratio'],
               speed_corr=sf.get('log_speed_corr'), slope_2_8=w.get('2-8', {}).get('slope'), cos_2_8=w.get('2-8', {}).get('cos'),
               slope_05_2=w.get('0.5-2', {}).get('slope'), bumps=bu.get('bumps_per_path'), phantom=pr.get('doorway_phantom_share'),
               wall_S035=pr.get('wall_S_at_035'), loss_final=d.get('final_loss'), parameters=d.get('parameters'),
               input_width=(d.get('plan') or {}).get('input_width', 256), feature_metrics=d.get('feature_metrics'))
    st = stall_of(d['_id'])
    if st:
        rec.update(stop_open=st['stop_open'] / st['runs'], stop_open_route=st['stop_open_route'] / st['runs'], stop_wall=st['stop_wall'] / st['runs'])
    rp = ripple_of(d['_id'])
    rec.update(uphill=rp.get('uphill_share'), eik_res=rp.get('eikonal_residual_median'), minima=rp.get('minima_per_path'),
               depth_q90=rp.get('depth_q90'))
    cs = console_stats(d['_id'])
    rec.update({k: cs.get(k) for k in ('loss_final100', 'loss_at_1000', 'loss_at_2000', 'rollbacks')})
    rec['loss_curve'] = cs.get('loss_curve')
    rec['_ok'] = rs['ok']
    return rec


def metric(rec, m):
    if m == 'success_rate':
        return rec['success'] / rec['runs']
    if m == 'never_arrived_rate':
        return rec['never_arrived'] / rec['runs']
    if m == 'wall_rate':
        return rec['arrived_after_wall'] / rec['runs']
    if m in ('loss_final', 'loss_final100', 'loss_at_1000', 'loss_at_2000'):
        v = rec.get(m)
        return None if v is None else math.log(v)      # relative change: difference of logs
    return rec.get(m)


def collect():
    table = {}
    for d in runs():
        try:
            rec = record(d)
        except Exception as e:  # noqa: BLE001
            print('skip', d['_id'], d['_key'], e)
            continue
        table.setdefault(d['_n'], {}).setdefault(d['_key'], {})[d['seed']] = rec
    return table


def ref_runs(table, n, seed):
    T = table.get(n, {})
    out = [T[k][seed] for k in REF_KEYS if seed in T.get(k, {})]
    g, o = T.get('gs:octave', {}).get(seed), T.get('ref:octave', {}).get(seed)
    if g is not None and (o is None or g['gpu'] != o['gpu']):
        out.append(g)
    return out


def noise(table, n):
    """Pooled within-seed SD of every measure over the reference trainings."""
    out = {}
    seeds = sorted(set(table.get(n, {}).get('ref:octave', {})) | set(table.get(n, {}).get('gs:octave', {})))
    for m in METRICS:
        ss, dof = 0.0, 0
        for s in seeds:
            xs = [metric(r, m) for r in ref_runs(table, n, s)]
            xs = [x for x in xs if x is not None]
            if len(xs) >= 2:
                ss += float(np.sum((np.array(xs) - np.mean(xs)) ** 2))
                dof += len(xs) - 1
        out[m] = dict(sd=math.sqrt(ss / dof) if dof else None, dof=dof)
    return out


def paired(table, n, key, sd):
    """Per measure: per-seed change against the seed's reference mean, mean, SE, z."""
    out = {}
    for m in METRICS:
        deltas, var = {}, 0.0
        for s, r in table.get(n, {}).get(key, {}).items():
            refs = [metric(x, m) for x in ref_runs(table, n, s)]
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


def versus(table, n, key, ctrl, sd):
    """Change against a control spec, paired by seed (both single runs)."""
    out = {}
    A, C = table.get(n, {}).get(key, {}), table.get(n, {}).get(ctrl, {})
    for m in METRICS:
        deltas = {}
        for s in sorted(set(A) & set(C)):
            a, c = metric(A[s], m), metric(C[s], m)
            if a is not None and c is not None:
                deltas[str(s)] = a - c
        if deltas:
            k = len(deltas)
            se = sd[m]['sd'] * math.sqrt(2.0 / k) if sd.get(m, {}).get('sd') else None
            mean = float(np.mean(list(deltas.values())))
            out[m] = dict(deltas=deltas, mean=mean, k=k, se=se, z=mean / se if se else None)
    return out


def gained_lost(table, n, key):
    """Per seed: query x planning-seed runs gained / lost against octave_detune's exact octave run of the seed."""
    out = {}
    for s, r in table.get(n, {}).get(key, {}).items():
        o = table.get(n, {}).get('ref:octave', {}).get(s) or table.get(n, {}).get('gs:octave', {}).get(s)
        if o is None:
            continue
        a, b = r['_ok'], o['_ok']
        out[str(s)] = dict(gained=int((a & ~b).sum()), lost=int((~a & b).sum()))
    return out


IDEA = [('1 powers, replace', re.compile(r'^R(s?pow|powmix)')), ('1 powers, add', re.compile(r'^A(pow|powmix|powcat)')),
        ('add controls', re.compile(r'^A(sc|dup|harm)')), ('2 triangle', re.compile(r'^(Rtri|Atri|Rsc)')),
        ('3 gate', re.compile(r'^(Ag|Rtri@all\+Ag)'))]


def idea_of(key):
    for name, rx in IDEA:
        if rx.search(key):
            return name
    return 'other'


def summarize(table):
    out = dict(noise={}, sizes={})
    for n in SIZES:
        T = table.get(n, {})
        sd = noise(table, n)
        out['noise'][str(n)] = sd
        ref = {}
        for s in sorted(set(T.get('ref:octave', {})) | set(T.get('gs:octave', {}))):
            rr = ref_runs(table, n, s)
            ref[str(s)] = dict(runs=[r['id'] for r in rr], success=[r['success'] for r in rr], mean_rate=float(np.mean([r['success'] / r['runs'] for r in rr])),
                               records=[{m: metric(r, m) for m in METRICS} | dict(id=r['id']) for r in rr])
        specs = {}
        for key, recs in T.items():
            if key.startswith('ref:') or key == 'gs:octave':
                continue
            e = dict(idea=idea_of(key), seeds=sorted(recs), per_seed={str(s): dict(id=r['id'], success=r['success'], runs=r['runs'],
                     **{m: metric(r, m) for m in METRICS}) for s, r in recs.items()},
                     pooled=dict(success=sum(r['success'] for r in recs.values()), runs=sum(r['runs'] for r in recs.values())),
                     paired=paired(table, n, key, sd), gained_lost=gained_lost(table, n, key),
                     input_width=next(iter(recs.values()))['input_width'], parameters=next(iter(recs.values()))['parameters'],
                     feature_metrics=next(iter(recs.values()))['feature_metrics'],
                     by_bin={b: float(np.mean([r['by_bin'][b] for r in recs.values() if b in r['by_bin']])) for b in BINS
                             if any(b in r['by_bin'] for r in recs.values())})
            if key.startswith('A') and key not in CONTROLS:
                e['versus'] = {c: versus(table, n, key, c, sd) for c in CONTROLS if T.get(c)}
            specs[key] = e
        out['sizes'][str(n)] = dict(reference=ref, specs=specs)
    return out


def fmt(x, nd=1, sign=True):
    if x is None:
        return '–'
    return f'{x:+.{nd}f}' if sign else f'{x:.{nd}f}'


def tables(S):
    lines = ['# Sharp features: paired change in success against the exact octave (points, mean over seeds ± SE)', '']
    for n in map(str, SIZES):
        if n not in S['sizes']:
            continue
        sd = S['noise'][n]['success_rate']['sd']
        lines += [f'## {n}×{n} (SD of one run {100 * sd:.2f} points)' if sd else f'## {n}×{n}', '',
                  '| spec | idea | seeds | success | change | z | vs more sin/cos | vs duplicate | wall | never | slope | bumps | speed corr |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        sp = S['sizes'][n]['specs']
        order = sorted(sp, key=lambda k: (k.startswith('ntrl'), k.startswith('ctx'), sp[k]['idea'], k))
        for key in order:
            e = sp[key]
            p = e['paired'].get('success_rate', {})
            vs = e.get('versus', {})
            v1 = vs.get('Asc@new128', {}).get('success_rate', {})
            v2 = vs.get('Adup@all', {}).get('success_rate', {})
            pw = e['paired'].get('wall_rate', {})
            pn = e['paired'].get('never_arrived_rate', {})
            ps = e['paired'].get('slope_2_8', {})
            pb = e['paired'].get('bumps', {})
            pc = e['paired'].get('speed_corr', {})
            rate = e['pooled']['success'] / e['pooled']['runs']
            lines.append(f'| `{key}` | {e["idea"]} | {len(e["seeds"])} | {100 * rate:.1f} % | {fmt(100 * p["mean"]) if p else "–"} ± {fmt(100 * p["se"], 1, False) if p.get("se") else "–"} '
                         f'| {fmt(p.get("z"), 1) if p else "–"} | {fmt(100 * v1["mean"]) if v1 else "–"} | {fmt(100 * v2["mean"]) if v2 else "–"} '
                         f'| {fmt(100 * pw["mean"]) if pw else "–"} | {fmt(100 * pn["mean"]) if pn else "–"} | {fmt(ps.get("mean"), 2) if ps else "–"} '
                         f'| {fmt(pb.get("mean"), 2) if pb else "–"} | {fmt(pc.get("mean"), 3) if pc else "–"} |')
        lines.append('')
    return '\n'.join(lines)


def main():
    table = collect()
    S = summarize(table)
    OUT.mkdir(exist_ok=True)
    (OUT / 'summary.json').write_text(json.dumps(S, indent=1))
    (OUT / 'tables.md').write_text(tables(S))
    print(tables(S))


if __name__ == '__main__':
    main()
