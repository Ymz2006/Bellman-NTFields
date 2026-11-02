"""Markdown tables from the results.json of one or more Eikonal planning runs.

    python jeffrey/gibson_embed/eikonal/report.py results/registered_<id>[=label] ...
"""
import json
import sys
from pathlib import Path


def fmt(x, nd=3):
    return '–' if x is None else f'{x:.{nd}f}'


def main():
    runs = []
    for arg in sys.argv[1:]:
        path, _, label = arg.partition('=')
        d = json.load(open(Path(path) / 'results.json'))
        runs.append((label or Path(path).name, d))
    kinds = ['uniform', 'wall']
    print('| run | model | γ | ' + ' | '.join(f'{k} success | {k} collision | {k} len/geo med | {k} len/geo p90' for k in kinds) + ' | final residual | min residual | rollbacks | train h |')
    print('|' + '---|' * (3 + 4 * len(kinds) + 4))
    for label, d in runs:
        for name, v in d.items():
            pl = v['planning']
            cells = []
            for k in kinds:
                r = pl.get(k)
                if r is None:
                    cells += ['–'] * 4
                    continue
                cells += [fmt(r['success']), fmt(r['collision_among_met']), fmt(r['length_ratio_median']), fmt(r['length_ratio_p90'])]
            hist = v.get('history')
            if hist:
                res = [h['residual'] for h in hist]
                tail = [fmt(res[-1]), fmt(min(res), 4), str(hist[-1]['repeats']), fmt(v['train_seconds'] / 3600, 1)]
                gamma = f"{v.get('gamma', 0):g}"
            else:
                tail = ['–'] * 4
                gamma = 'pretrained'
            print(f'| {label} | {name} | {gamma} | ' + ' | '.join(cells) + ' | ' + ' | '.join(tail) + ' |')


if __name__ == '__main__':
    main()
