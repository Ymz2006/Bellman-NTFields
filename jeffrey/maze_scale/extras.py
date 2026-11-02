"""Follow-up experiments for the findings page and README: fewer DP levels, Fourier scale, u-turns around wall ends.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/extras.py

collect() -> dict(levels=..., fourier=..., uturn=...) from the registered runs; also writes report/extras.json and
report/extras.md (tables). Success = MPPI successes over planning seeds 0/1/2, pooled over training seeds.
"""
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import report as R  # noqa: E402
from common import bin_labels  # noqa: E402

WALL03 = 'segment DP + wall term (weight 0.3)'


def seed_order(seeds):
    return sorted(seeds, key=lambda s: (s != 87358, s))


def levels(prep, ntrl, seg, n=13):
    labs = [l for l in bin_labels() if prep[n]['queries']['per_bin'].get(l)]
    rows = [('ntrl-demo', 1, None, ntrl[n])]
    for v, by_n in seg.items():
        if v.startswith(WALL03) and n in by_n:
            d0 = next(iter(by_n[n].values()))
            if d0['config'].get('candidates', 128) != 128 or R.base_scale(d0) != 1.0:
                continue
            rows.append((v, d0['levels'] + 1, d0['config'].get('ratio', 1.35), by_n[n]))
    out = []
    for name, nets, ratio, by in sorted(rows, key=lambda r: r[1]):
        seeds = seed_order(by)
        per_bin = []
        for l in labs:
            p = [by[s]['success_by_bin'][l] for s in seeds if l in by[s]['success_by_bin']]
            per_bin.append([sum(x[0] for x in p), sum(x[1] for x in p)])
        tt = [by[s]['field']['by_bin'].get('16-32', {}).get('t_over_tstar_median') for s in seeds]
        tt = [x for x in tt if x is not None]
        tot = [by[s]['success_total'] for s in seeds]
        out.append(dict(name=name, networks=nets, ratio=ratio, edges=by[seeds[0]].get('edges_cells'), seeds=seeds,
                        per_seed=[t[0] for t in tot], per_seed_of=tot[0][1], total=[sum(t[0] for t in tot), sum(t[1] for t in tot)],
                        per_bin=per_bin, t_over_tstar_16_32=float(np.mean(tt)) if tt else None,
                        minutes=[by[s]['elapsed_seconds'] / 60 for s in seeds] if nets > 1 else None, ids=[by[s]['_id'] for s in seeds]))
    return dict(maze=n, bins=labs, rows=out)


def candidates(prep, ntrl, seg, n=13):
    """Split candidates per DP pair (segdp --candidates), wall term weight 0.3, ratio 1.35, 13 x 13."""
    labs = [l for l in bin_labels() if prep[n]['queries']['per_bin'].get(l)]
    rows = []
    for v, by_n in seg.items():
        if not v.startswith(WALL03) or n not in by_n:
            continue
        d0 = next(iter(by_n[n].values()))
        if d0['config'].get('ratio', 1.35) != 1.35 or R.base_scale(d0) != 1.0:
            continue
        rows.append((d0['config'].get('candidates', 128), by_n[n]))
    out = []
    for k, by in sorted(rows):
        seeds = seed_order(by)
        per_bin = []
        for l in labs:
            p = [by[s]['success_by_bin'][l] for s in seeds if l in by[s]['success_by_bin']]
            per_bin.append([sum(x[0] for x in p), sum(x[1] for x in p)])
        tot = [by[s]['success_total'] for s in seeds]
        st = [by[s]['level_stats'] for s in seeds]
        top = [x[-1]['probe']['target_over_truth']['median'] for x in st]
        out.append(dict(candidates=k, seeds=seeds, per_seed=[t[0] for t in tot], per_seed_of=tot[0][1], total=[sum(t[0] for t in tot), sum(t[1] for t in tot)],
                        per_bin=per_bin, top_target_over_truth=float(np.mean(top)),
                        mid_target_over_truth=float(np.mean([x[len(x) // 2]['probe']['target_over_truth']['median'] for x in st])),
                        target_seconds=float(np.mean([sum(l['target_seconds'] for l in x) for x in st])),
                        minutes=[by[s]['elapsed_seconds'] / 60 for s in seeds], ids=[by[s]['_id'] for s in seeds]))
    return dict(maze=n, bins=labs, rows=out)


def scaled_dp(prep, ntrl, seg):
    """Segment DP + wall term (weight 0.3) on a B-scaled ntrl-demo base, against the same field families on B x 1."""
    four = {}
    for d in R.runs('fourier'):
        four.setdefault((d['maze']['n'], float(d['b_scale'])), {})[d['seed']] = d
    out = []
    for v, by_n in seg.items():
        if not (v.startswith(WALL03) and 'B ×' in v):
            continue
        for n, by in sorted(by_n.items()):
            b = R.base_scale(next(iter(by.values())))
            labs = [l for l in bin_labels() if prep[n]['queries']['per_bin'].get(l)]
            rows = []
            for name, runs in (('ntrl-demo', ntrl.get(n, {})), (f'ntrl-demo, B × {b:g}', four.get((n, b), {})),
                               ('segment DP + wall term (weight 0.3)', seg.get(WALL03, {}).get(n, {})),
                               (f'segment DP + wall term (weight 0.3) on the B × {b:g} base', by)):
                if not runs:
                    continue
                seeds = seed_order(runs)
                per_bin = []
                for l in labs:
                    q = [runs[k]['success_by_bin'][l] for k in seeds if l in runs[k]['success_by_bin']]
                    per_bin.append([sum(x[0] for x in q), sum(x[1] for x in q)])
                tot = [runs[k]['success_total'] for k in seeds]
                rows.append(dict(name=name, seeds=seeds, per_seed=[t[0] for t in tot], per_seed_of=tot[0][1],
                                 total=[sum(t[0] for t in tot), sum(t[1] for t in tot)], per_bin=per_bin, ids=[runs[k]['_id'] for k in seeds]))
            out.append(dict(n=n, scale=b, bins=labs, rows=rows))
    return out


def fourier(prep, ntrl):
    by = {}
    for d in R.runs('fourier'):
        by.setdefault(d['maze']['n'], {}).setdefault(float(d['b_scale']), {})[d['seed']] = d
    zoom = {d['maze']['n']: d for d in R.runs('zoom')}
    mazes = []
    for n in sorted(prep):
        if n not in by:
            continue
        labs = [l for l in bin_labels() if prep[n]['queries']['per_bin'].get(l)]
        scales = {1.0: ntrl.get(n, {})}
        scales.update(by[n])
        cols = []
        for s in sorted(scales):
            runs = scales[s]
            seeds = seed_order(runs)
            tot = [runs[k]['success_total'] for k in seeds]
            per_bin = []
            for l in labs:
                p = [runs[k]['success_by_bin'][l] for k in seeds if l in runs[k]['success_by_bin']]
                per_bin.append([sum(x[0] for x in p), sum(x[1] for x in p)] if p else None)
            sp = [runs[k].get('speed_fit', {}).get('log_speed_corr') for k in seeds]
            cols.append(dict(scale=s, seeds=seeds, per_seed=[t[0] for t in tot], per_seed_of=tot[0][1],
                             total=[sum(t[0] for t in tot), sum(t[1] for t in tot)], per_bin=per_bin,
                             speed_fit=float(np.mean([x for x in sp if x is not None])) if any(x is not None for x in sp) else None,
                             ids=[runs[k]['_id'] for k in seeds]))
        z = zoom.get(n)
        mazes.append(dict(n=n, bins=labs, cols=cols, zoom=dict(scale=z['zoom'], total=z['success_total'], seed=z['seed']) if z else None))
    return mazes


def uturn(ntrl, seg):
    prepared = {d['maze']['n']: d for d in R.runs('uturn')}
    evals = {}
    for d in R.runs('uturneval'):
        for rid, r in d['runs'].items():
            evals[rid] = dict(r, _eval=d['_id'])
    names = {}                                                   # run id -> field name, for the runs the main report uses
    for n, by in ntrl.items():
        for s, d in by.items():
            names[d['_id']] = 'ntrl-demo'
    for d in R.runs('zoom'):
        names[d['_id']] = 'ntrl-demo, zoomed input'
    for v, by_n in seg.items():
        for n, by in by_n.items():
            for s, d in by.items():
                names[d['_id']] = v
    for d in R.runs('fourier'):
        names[d['_id']] = f'ntrl-demo, B × {d["b_scale"]:g}'
    mazes = []
    for n in sorted(prepared):
        p = prepared[n]
        o = p['oracle']
        m = dict(n=n, id=p['_id'], queries=p['queries'], free_ends=p['free_ends'], distinct_ends=p.get('distinct_ends'),
                 t_star_cells_median=p['truth']['t_star_cells_median'], path_cells_median=p['truth']['path_cells_median'],
                 oracle=[sum(o[k]['success'] for k in o), sum(o[k]['total'] for k in o)], straight=[p['straight']['success'], p['straight']['total']], fields={})
        for rid, r in evals.items():
            if r['n'] != n or rid not in names:
                continue
            f = m['fields'].setdefault(names[rid], dict(per_seed={}, ids={}))
            f['per_seed'][str(r['seed'])] = r['total']
            f['ids'][str(r['seed'])] = rid
            f.setdefault('collisions', 0)
            f['collisions'] += int(sum(r['collision']))
            f.setdefault('t_over_tstar', []).append(r['t_over_tstar_median'])
        for f in m['fields'].values():
            ks = [str(s) for s in seed_order([int(k) for k in f['per_seed']])]
            f['seeds'] = ks
            f['total'] = [sum(f['per_seed'][k][0] for k in ks), sum(f['per_seed'][k][1] for k in ks)]
            f['t_over_tstar'] = float(np.median(f['t_over_tstar']))
        mazes.append(m)
    return mazes


def collect():
    prep, ntrl, seg = R.collect()
    return dict(levels=levels(prep, ntrl, seg) if 13 in prep else None, candidates=candidates(prep, ntrl, seg) if 13 in prep else None,
                fourier=fourier(prep, ntrl), scaled_dp=scaled_dp(prep, ntrl, seg), uturn=uturn(ntrl, seg))


def markdown(x):
    md = []
    lv = x['levels']
    if lv:
        md += ['**Fewer networks (13 × 13, segment DP + wall term weight 0.3, same ntrl-demo base per seed)**', '',
               '| networks | edge ratio | runs solved per seed (87358 / 1 / 2, of 600) | all | ' + ' | '.join(f'{b} cells' for b in lv['bins']) + ' | T/T* 16–32 | minutes per seed |',
               '|' + '---|' * (6 + len(lv['bins']))]
        for r in lv['rows']:
            md.append(f'| {r["networks"]} | {"–" if r["ratio"] is None else format(r["ratio"], ".3f")} | {" / ".join(map(str, r["per_seed"]))} | '
                      f'{100 * r["total"][0] / r["total"][1]:.0f} % ({r["total"][0]}/{r["total"][1]}) | '
                      + ' | '.join(f'{100 * a / b:.0f}' for a, b in r['per_bin'])
                      + f' | {r["t_over_tstar_16_32"]:.2f} | ' + (' / '.join(f'{m:.0f}' for m in r['minutes']) if r['minutes'] else '–') + ' |')
        md.append('')
    if x['fourier']:
        md += ['**Fourier scale (ntrl-demo with B × s, same seed draw; s = 1 is plain ntrl-demo)**', '',
               '| maze | B scale | runs solved per seed (87358 / 1 / 2) | all | speed fit |', '|---|---|---|---|---|']
        for m in x['fourier']:
            for k, c in enumerate(m['cols']):
                md.append(f'| {m["n"]}×{m["n"]} | {c["scale"]:g} | {" / ".join(map(str, c["per_seed"]))} (of {c["per_seed_of"]}) | '
                          f'{100 * c["total"][0] / c["total"][1]:.0f} % ({c["total"][0]}/{c["total"][1]}) | {"–" if c["speed_fit"] is None else format(c["speed_fit"], ".2f")} |')
        md.append('')
    if x['uturn']:
        md += ['**U-turns around a wall end (40 queries per maze × 3 planning seeds per training seed)**', '',
               '| maze | field | runs solved per seed (87358 / 1 / 2, of 120) | all |', '|---|---|---|---|']
        for m in x['uturn']:
            md.append(f'| {m["n"]}×{m["n"]} | oracle (MPPI on T*) | – | {100 * m["oracle"][0] / m["oracle"][1]:.0f} % ({m["oracle"][0]}/{m["oracle"][1]}) |')
            md.append(f'| | straight line | – | {100 * m["straight"][0] / m["straight"][1]:.0f} % ({m["straight"][0]}/{m["straight"][1]}) |')
            for name, f in m['fields'].items():
                md.append(f'| | {name} | {" / ".join(str(f["per_seed"][k][0]) for k in f["seeds"])} | {100 * f["total"][0] / f["total"][1]:.0f} % ({f["total"][0]}/{f["total"][1]}) |')
        md.append('')
    return '\n'.join(md)


def main():
    x = collect()
    (R.OUT / 'extras.json').write_text(json.dumps(x, indent=1, default=float))
    md = markdown(x)
    (R.OUT / 'extras.md').write_text(md + '\n')
    print(md)


if __name__ == '__main__':
    main()
