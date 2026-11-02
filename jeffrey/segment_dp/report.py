"""Tables of the segment-DP experiment from registered runs (markdown on stdout).

    python3 jeffrey/segment_dp/report.py --index runs.json [--fit cabinet=registered_<id> nowall=registered_<id>]
                                         [--long registered_<id>] [--fields ES EG DP ...]
--index  JSON {"<scene>/<label>": "<run id>"} (labels: w<B>-<E>-<D>[s<seed>] loss ablations, ES/EG/DP[-off],
         B5[-off], ...). Runs of one label with several training seeds: label, labels+"s1", "s2", ...
"""
import argparse, json, re
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[2]


def status(run):
    return json.loads((HERE/'results'/f'registered_{run}'/'status.json').read_text())


def seeds_of(index, scene, label):
    out = []
    for key, run in index.items():
        sc, lab = key.split('/', 1)
        if sc == scene and re.fullmatch(re.escape(label)+r'(s\d+)?', lab): out.append((lab, run))
    return sorted(out)


def planning(index, scene, labels):
    lines = ['|Field|Training seeds|RK4 /600 (each seed)|Mean RK4|MPPI /1800 (each seed)|Mean MPPI|', '|---|---:|---|---:|---|---:|']
    for label in labels:
        runs = seeds_of(index, scene, label)
        if not runs: continue
        t = [status(r)['totals'] for _, r in runs]; rk = [x['native_rk4']['success'] for x in t]; mp = [x['matched_mppi']['success'] for x in t]
        lines.append(f"|{label}|{len(runs)}|{', '.join(map(str, rk))}|{np.mean(rk):.1f}|{', '.join(map(str, mp))}|{np.mean(mp):.1f}|")
    return '\n'.join(lines)


def fit_table(fit, metric, fields, digits=3, group=None):
    edges = fit['edges']; head = '|Gap|Pairs|T*/gap|'+'|'.join(fields)+'|'; lines = [head, '|---|---:|---:|'+'---:|'*len(fields)]
    for s in range(len(edges)-1):
        if group and s not in group: continue
        cells = []
        for f in fields:
            v = fit['fields'].get(f, {}).get(metric, [None]*len(edges))[s]
            cells.append('—' if v is None else f'{v:.{digits}f}')
        tg = fit['truth_over_gap'][s]
        lines.append(f"|{edges[s]:.2f}–{edges[s+1]:.2f}|{fit['pairs'][s]}|{'—' if tg is None else f'{tg:.2f}'}|"+'|'.join(cells)+'|')
    return '\n'.join(lines)


def fit_summary(fit, fields, ranges):
    """Mean of each metric over gap ranges (segment means weighted equally)."""
    edges = np.array(fit['edges']); lines = ['|Field|'+'|'.join(f'{m} {a:g}–{b:g}' for m in ('rel. error', 'Spearman', 'cosine') for a, b in ranges)+'|',
                                            '|---|'+'---:|'*(3*len(ranges))]
    for f in fields:
        if f not in fit['fields']: continue
        row = fit['fields'][f]; cells = []
        for metric in ('rel_error', 'spearman', 'cosine'):
            for a, b in ranges:
                sel = [s for s in range(len(edges)-1) if edges[s] >= a-1e-9 and edges[s+1] <= b+1e-9 and row[metric][s] is not None]
                cells.append(f"{np.mean([row[metric][s] for s in sel]):.3f}" if sel else '—')
        lines.append(f'|{f}|'+'|'.join(cells)+'|')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--index', type=Path, required=True); p.add_argument('--fit', nargs='*', default=[])
    p.add_argument('--long'); p.add_argument('--fields', nargs='*', default=['B5', 'w0-2-0', 'EG', 'ES', 'DP', 'B5-off', 'EG-off', 'ES-off', 'DP-off']); a = p.parse_args()
    index = json.loads(a.index.read_text())
    for scene in ('cabinet', 'nowall'):
        print(f'\n## {scene}\n\n### Loss ablations\n')
        print(planning(index, scene, ['w6-2-2', 'w6-0-2', 'w0-2-2', 'w0-2-0', 'w6-0-0', 'w6-2-0']))
        print('\n### Segment fields\n'); print(planning(index, scene, [f for f in a.fields if not f.startswith('w')]))
    for spec in a.fit:
        scene, run = spec.split('='); fit = json.loads((HERE/'results'/run/'fit.json').read_text())
        fields = ['straight', 'euclid']+[f for f in a.fields if f in fit['fields']]
        print(f'\n## Fit to ground truth: {scene}\n')
        print(fit_summary(fit, fields, [(0, .24), (.24, .72), (.72, 1.6)]))
        for metric in ('rel_error', 'bias', 'spearman', 'cosine'):
            print(f'\n### {metric}\n'); print(fit_table(fit, metric, fields))
    if a.long:
        long = json.loads((HERE/'results'/a.long/'long/long.json').read_text())
        print('\n## Long cabinet queries (gap 0.8–1.6, 400, uncertified)\n\n|Field|RK4|MPPI seed 0|\n|---|---:|---:|')
        for label, row in long['fields'].items():
            print(f"|{label}|{row['success']}|{row.get('mppi_seed0', {}).get('success', '—')}|")


if __name__ == '__main__': main()
