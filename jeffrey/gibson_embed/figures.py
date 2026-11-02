"""Figures and a markdown table for one pipeline output directory (2-D): embedding channels, fields, probe, results."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

SEQ = LinearSegmentedColormap.from_list('blue', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
DIV = LinearSegmentedColormap.from_list('bluered', ['#0d366b', '#3987e5', '#f0efec', '#e5533a', '#7a1f12'])
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#7c5cd6', '#8c8c8c']
WALL = ListedColormap(['#00000000', '#1a1a19'])


def wall_overlay(ax, occ):
    ax.imshow(occ.T, cmap=WALL, origin='lower', interpolation='nearest', vmin=0, vmax=1)


def fig_embedding(out, occ):
    f = np.load(out / 'embedding_diff-random.npz')
    feat, snaps = f['feat'].astype(np.float32), f['snapshots']
    per = feat.shape[0] // len(snaps)
    fig, axes = plt.subplots(2, len(snaps), figsize=(3.2 * len(snaps), 6.6))
    for j, k in enumerate(snaps):
        for i in range(2):
            ax = axes[i, j]
            ch = feat[j * per + i]
            ch = np.where(occ, np.nan, ch)
            ax.imshow(ch.T, cmap=DIV, vmin=-2.5, vmax=2.5, origin='lower', interpolation='nearest')
            wall_overlay(ax, occ)
            ax.set_title(f'K = {k} iterations, channel {i}', fontsize=9)
            ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle('Free-space ball average of random features (r = 1 cell): standardized channels, walls in black', fontsize=11)
    fig.tight_layout()
    fig.savefig(out / 'embedding.png', dpi=130)
    plt.close(fig)


def fig_fields(out, occ, variants):
    r = np.load(out / 'renders.npz')
    tr = np.load(out / 'truth.npz')
    dist, sources = tr['dist'].astype(np.float32), tr['sources']
    ids = r['sources']
    variants = [v for v in variants if v in r]
    for n, sid in enumerate(ids):
        truth = np.where(occ, np.nan, dist[sid])
        truth[~np.isfinite(truth)] = np.nan
        cols = 1 + len(variants)
        fig, axes = plt.subplots(2, cols, figsize=(3.1 * cols, 6.4))
        vmax = np.nanpercentile(truth, 99)
        axes[0, 0].imshow(truth.T, cmap=SEQ, vmin=0, vmax=vmax, origin='lower', interpolation='nearest')
        axes[0, 0].set_title('geodesic truth', fontsize=9)
        axes[1, 0].axis('off')
        for j, v in enumerate(variants, 1):
            pred = r[v][n]
            axes[0, j].imshow(pred.T, cmap=SEQ, vmin=0, vmax=vmax, origin='lower', interpolation='nearest')
            axes[0, j].set_title(v, fontsize=9)
            err = np.log(pred / truth)
            axes[1, j].imshow(err.T, cmap=DIV, vmin=-1, vmax=1, origin='lower', interpolation='nearest')
            axes[1, j].set_title(f'log(pred/truth), mean |.| {np.nanmean(np.abs(err)):.3f}', fontsize=9)
        for ax in axes.ravel():
            if ax.axison:
                wall_overlay(ax, occ)
                ax.plot(sources[sid][0], sources[sid][1], marker='o', color='#eb6834', markersize=5)
                ax.set_xticks([]), ax.set_yticks([])
        fig.suptitle(f'Held-out source {sid}: predicted distance field (top, blue = far) and log error (bottom, red = too far, blue = too close)', fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f'field_{n}.png', dpi=120)
        plt.close(fig)


def fig_probe(out):
    pr = json.loads((out / 'probe.json').read_text())
    names = list(pr['features'])
    vals = [pr['features'][n]['auc_stratified'] for n in names]
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(names) + 1.2))
    y = np.arange(len(names))
    ax.barh(y, vals, color='#2a78d6', height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=8, color='#1a1a19')
    ax.axvline(0.5, color='#8c8c8c', lw=1, ls='--')
    ax.set_yticks(y, names, fontsize=8)
    ax.set_xlim(0.4, 1.05)
    ax.invert_yaxis()
    ax.set_xlabel('AUC: wall-straddling vs corridor pairs from ||phi(x) - phi(y)|| (stratified by Euclidean distance)', fontsize=8)
    ax.set_title(f"Embedding probe ({pr['wall_pairs']} wall pairs, {pr['corridor_pairs']} corridor pairs, radius {pr['radius_cells']} cells)", fontsize=9)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / 'probe.png', dpi=130)
    plt.close(fig)


def table(out):
    res = json.loads((out / 'results.json').read_text())
    pr = json.loads((out / 'probe.json').read_text())
    rows = ['| variant | params | probe AUC | test all | test wall | test corridor | test far | wall detect AUC | recall@2 | false alarm@2 | train all |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for v, r in res.items():
        t = r['test_sources']
        auc = pr['features'].get(v, {}).get('auc_stratified', float('nan'))
        d = t.get('detect_wall', {})
        rows.append(f"| {v} | {r['parameters'] / 1e6:.2f}M | {auc:.3f} | {t['all']['mean_abs_log']:.4f} | {t['wall']['mean_abs_log']:.4f} | {t['corridor']['mean_abs_log']:.4f} | "
                    f"{t['far']['mean_abs_log']:.4f} | {d.get('auc', float('nan')):.3f} | {d.get('recall_at_log2', float('nan')):.3f} | {d.get('false_alarm_at_log2', float('nan')):.3f} | {r['train_sources']['all']['mean_abs_log']:.4f} |")
    text = '\n'.join(rows)
    (out / 'table.md').write_text(text + '\n')
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out')
    p.add_argument('--variants', default='fourier,diff-random,blur-random,learned-grid,fourier+diff')
    a = p.parse_args()
    out = Path(a.out)
    occ = np.load(out / 'grid.npz')['occupancy']
    if (out / 'embedding_diff-random.npz').exists() and occ.ndim == 2:
        fig_embedding(out, occ)
    if (out / 'renders.npz').exists():
        fig_fields(out, occ, a.variants.split(','))
    fig_probe(out)
    print(table(out))


if __name__ == '__main__':
    main()
