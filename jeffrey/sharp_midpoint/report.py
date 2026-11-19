"""Summary of the sharp-feature + midpoint-loss runs: report/summary.json and report/tables.md.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/sharp_midpoint/report.py

Per (maze, seed, spec): success of the base run (jeffrey/sharp_features at epoch 5,000, or octave_detune's exact octave), success with
the midpoint loss (forked at epoch 2,000 of that base run), and the field measures of both (uphill share of the true paths, local minima,
eikonal residual, runs through a wall / never arrived). Paired comparisons by seed: spec + midpoint against octave + midpoint (does a
sharp feature help once the loss can see folds?) and spec + midpoint against the spec's own base run (what the loss does to it).
SE from the training noise of one run of the exact octave (jeffrey/sharp_features/report/summary.json noise): sd sqrt(2 / k).
"""
import json
import math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'report'
SF = ROOT / 'jeffrey/sharp_features/report/summary.json'


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def rec_of(d):
    rp = d.get('ripple') or {}
    f = d.get('failures') or {}
    w = (d.get('wall') or {}).get('2-8', {})
    k, N = d['success_total']
    return dict(success=k, runs=N, rate=k / N, uphill=rp.get('uphill_share'), minima=rp.get('minima_per_path'), eik=rp.get('eikonal_residual_median'),
                never=f.get('never_arrived', 0) / N if f else None, wall=f.get('arrived_after_wall', 0) / N if f else None, slope=w.get('slope'),
                speed=(d.get('speed_fit') or {}).get('log_speed_corr'), log_err=(d.get('accuracy') or {}).get('log_err_median'))


def base_record(base_dir):
    d = load(Path(base_dir) / 'results.json')
    if not d:
        return None
    r = rec_of(d)
    rp = (load(ROOT / 'jeffrey/sharp_features/report/ripple.json') or {}).get(Path(base_dir).name.replace('registered_', '')) or {}
    if r['uphill'] is None:
        r.update(uphill=rp.get('uphill_share'), minima=rp.get('minima_per_path'), eik=rp.get('eikonal_residual_median'))
    if r['never'] is None and (Path(base_dir) / 'rows.npz').exists():
        z = np.load(Path(base_dir) / 'rows.npz')
        ok, conv, col = z['success'], z['converged'], z['collision']
        r.update(never=float((~ok & ~conv).sum()) / ok.size, wall=float((~ok & conv & col).sum()) / ok.size)
    return r


def collect():
    runs = {}
    for p in sorted(ROOT.glob('results/registered_*/results.json')):
        d = load(p)
        if not d or d.get('stage') != 'sharpmid' or not d.get('complete') or d.get('smoke'):
            continue
        label = 'ctl' if d['config']['weight'] == 0 else 'main'
        key = (str(d['maze']['n']), str(d['seed']), d['spec'], label)
        r = rec_of(d)
        r.update(id=p.parent.name.replace('registered_', ''), base=base_record(Path(d['base_run'])), validate=d.get('validate'))
        runs[key] = r
    return runs


def summarize(runs):
    sf = load(SF) or {}
    sd = {n: ((sf.get('noise') or {}).get(n) or {}).get('success_rate', {}).get('sd') for n in ('13', '22')}
    out = dict(noise_sd=sd, runs={'|'.join(k): v for k, v in runs.items()}, specs={})
    for (n, seed, spec, label), r in runs.items():
        if label != 'main':
            continue
        e = out['specs'].setdefault(n, {}).setdefault(spec, dict(seeds={}, vs_octave_mid={}, vs_base={}))
        e['seeds'][seed] = dict(id=r['id'], rate=r['rate'], base_rate=r['base']['rate'] if r['base'] else None, uphill=r['uphill'],
                                base_uphill=(r['base'] or {}).get('uphill'), minima=r['minima'], base_minima=(r['base'] or {}).get('minima'),
                                never=r['never'], wall=r['wall'], base_never=(r['base'] or {}).get('never'), base_wall=(r['base'] or {}).get('wall'))
        o = runs.get((n, seed, 'octave', 'main'))
        if o is not None:
            e['vs_octave_mid'][seed] = r['rate'] - o['rate']
        if r['base']:
            e['vs_base'][seed] = r['rate'] - r['base']['rate']
    for n, specs in out['specs'].items():
        for spec, e in specs.items():
            for k in ('vs_octave_mid', 'vs_base'):
                v = list(e[k].values())
                if v:
                    se = sd.get(n) * math.sqrt(2 / len(v)) if sd.get(n) else None
                    e[k + '_mean'] = float(np.mean(v))
                    e[k + '_se'] = se
    return out


def tables(S):
    lines = ['# Sharp features + midpoint loss', '']
    for n, specs in sorted(S['specs'].items()):
        lines += [f'## {n}×{n}', '', '| spec | seeds | success without → with midpoint | with midpoint vs octave + midpoint | midpoint gain | uphill without → with | minima without → with |',
                  '|---|---|---|---|---|---|---|']
        for spec, e in sorted(specs.items(), key=lambda kv: (kv[0] != 'octave', kv[0])):
            seeds = sorted(e['seeds'])
            r = [e['seeds'][s] for s in seeds]
            def m(key):
                v = [x[key] for x in r if x.get(key) is not None]
                return float(np.mean(v)) if v else None
            f = lambda v, sc=100, nd=1: '–' if v is None else f'{sc * v:.{nd}f}'  # noqa: E731
            vo = e.get('vs_octave_mid_mean')
            vb = e.get('vs_base_mean')
            vo_s = '–' if vo is None else f'{100 * vo:+.1f}'
            vb_s = '–' if vb is None else f'{100 * vb:+.1f}'
            lines.append(f'| `{spec}` | {len(seeds)} | {f(m("base_rate"))} → {f(m("rate"))} % | {vo_s} | {vb_s} '
                         f'| {f(m("base_uphill"))} → {f(m("uphill"))} % | {f(m("base_minima"), 1, 2)} → {f(m("minima"), 1, 2)} |')
        lines.append('')
    return '\n'.join(lines)


def main():
    S = summarize(collect())
    OUT.mkdir(exist_ok=True)
    (OUT / 'summary.json').write_text(json.dumps(S, indent=1))
    (OUT / 'tables.md').write_text(tables(S))
    print(tables(S))


if __name__ == '__main__':
    main()
