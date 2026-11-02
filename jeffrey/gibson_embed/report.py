"""Markdown report for one or more pipeline output directories (probe, per-scale probe, downstream nets).

    python jeffrey/gibson_embed/report.py <dir> [<dir> ...] [--full]
"""
import argparse
import json
from pathlib import Path


def fmt(x, nd=3):
    return '-' if x is None else f'{x:.{nd}f}'


def report(out, full=False):
    out = Path(out)
    lines = []
    grid = json.loads((out / 'grid.json').read_text())
    thick = grid.get('wall_thickness_cells', {})
    leak = grid.get('leak_check', {})
    lines.append(f"### {out.name}: {Path(grid['mesh']).parts[-2]}, {grid['dim']}-D, {grid['res']}^{grid['dim']}")
    lines.append(f"free cells {grid['free_cells']:,}, occupied {100 * grid['occupied_fraction']:.1f}%, {grid['components']} components "
                 f"(largest {100 * grid['largest_component_fraction']:.1f}%), wall thickness median {thick.get('median')} cells, "
                 f"leak check {leak.get('leaks')} / {leak.get('pairs')} face-adjacent free pairs cut by the mesh")
    lines.append('')
    pr = json.loads((out / 'probe.json').read_text())
    pairs = {k: v for k, v in pr.items() if k != 'features'}
    lines.append(f"**Embedding probe** (no training): {pairs}")
    lines.append('')
    lines.append('| feature | AUC wall vs corridor | wall/corridor feature distance | Spearman vs geodesic |')
    lines.append('|---|---|---|---|')
    for k, v in pr['features'].items():
        lines.append(f"| {k} | {fmt(v['auc_stratified'])} | {fmt(v['wall_over_corridor_median'], 2)} | {fmt(v['spearman_geodesic'])} |")
    lines.append('')
    for sc in sorted(out.glob('scales_*.json')):
        s = json.loads(sc.read_text())
        lines.append(f"**Per-scale probe** of `{sc.stem[7:]}`:")
        lines.append('')
        lines.append('| scale | AUC | wall/corridor feature distance | Spearman vs geodesic |')
        lines.append('|---|---|---|---|')
        for k, v in s['scales'].items():
            lines.append(f"| {k} | {fmt(v['auc'])} | {fmt(v['wall_over_corridor'], 2)} | {fmt(v['spearman_geo'])} |")
        lines.append('')
    rp = out / 'results.json'
    if rp.exists():
        res = json.loads(rp.read_text())
        counts = sorted({v['train_sources_count'] for v in res.values()}, reverse=True)
        variants = []
        for k in res:
            v = k.split('@')[0]
            if v not in variants:
                variants.append(v)
        lines.append('**Downstream pair network** (held-out sources; mean |log(pred/true)| over all pairs / wall pairs (local, geodesic >= 2.5x '
                     'Euclid) / corridor pairs (<= 1.15x), and recall (false-alarm rate) of wall pairs at predicted ratio >= 2):')
        lines.append('')
        head = '| input | params |' + ''.join(f' {n} sources: all / wall / corridor / recall (FA) |' for n in counts)
        lines.append(head)
        lines.append('|---|---|' + '---|' * len(counts))
        for v in variants:
            cells = []
            for n in counts:
                key = v if n == counts[0] else f'{v}@{n}'
                if key not in res:
                    cells.append(' - |')
                    continue
                t = res[key]['test_sources']
                cells.append(f" {t['all']['mean_abs_log']:.4f} / {t.get('wall', {}).get('mean_abs_log', float('nan')):.3f} / "
                             f"{t.get('corridor', {}).get('mean_abs_log', float('nan')):.4f} / {t.get('detect_wall', {}).get('recall_at_log2', float('nan')):.2f} ({t.get('detect_wall', {}).get('false_alarm_at_log2', float('nan')):.3f}) |")
            lines.append(f"| {v} | {res[v]['parameters'] / 1e6:.2f}M |" + ''.join(cells))
        lines.append('')
        if full:
            lines.append('Training-source fit (same metrics on the training sources):')
            lines.append('')
            lines.append('| input |' + ''.join(f' {n}: all / wall |' for n in counts))
            lines.append('|---|' + '---|' * len(counts))
            for v in variants:
                cells = []
                for n in counts:
                    key = v if n == counts[0] else f'{v}@{n}'
                    if key not in res:
                        cells.append(' - |')
                        continue
                    t = res[key]['train_sources']
                    cells.append(f" {t['all']['mean_abs_log']:.4f} / {t.get('wall', {}).get('mean_abs_log', float('nan')):.4f} |")
                lines.append(f'| {v} |' + ''.join(cells))
            lines.append('')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dirs', nargs='+')
    p.add_argument('--full', action='store_true')
    a = p.parse_args()
    for d in a.dirs:
        print(report(d, a.full))
        print()


if __name__ == '__main__':
    main()
