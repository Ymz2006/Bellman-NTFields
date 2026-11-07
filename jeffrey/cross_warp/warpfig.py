"""Figures of the warp used in gappage.py's section 'Pairs drawn on the warped maze, and the 10 x 10 maze'.

report/gap_warp.png: the constructed warp (warp.constructed, the fixed design every run of that section uses: the
gap study's plusmag, m = 6 E / 22.1 over the quadrants' columns / rows and the band's edge cells, x 1 over the band's
interior, log-linear over one cell). Rows: the 22 x 22 cross maze's quadrants and the 10 x 10 maze's, at x 1 and x 12.
Columns: the maze, shaded by the warp's area magnification F'(x) G'(y) (log scale: m^2 in the quadrants, m on the
band's arms, 1 at its centre); the warped maze z = phi(x) (walls are axis-aligned segments, so their images under the
separable warp are exact), every z panel at one scale; 2,000 training pairs (x0, x1) of the prepare run, in z, drawn
on the maze (gapprepare.py) and on the warped maze (gapprepare.py --z-draw).

report/gap_warp_density.png: dz/dx along x across the 22 x 22 quadrants' mazes at x 1 and x 12: the constructed warp
(the gap study's pm runs) and the gap study's learned warp (three training seeds), from each run's warp_map.npz.

report/gap_warp.json: the numbers the page quotes (box and band widths in z, the learned warp's magnifications).
Run: ~/.venvs/bellman-ntfield/bin/python jeffrey/cross_warp/warpfig.py   (CPU, reads registered runs only)."""
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gapreport as GR  # noqa: E402
import warp as W  # noqa: E402

OUT = HERE / 'report'
RESULTS = GR.ROOT / 'results'
INK, SHADE, X0C, X1C = '#3b3a37', '#eb6834', '#c2410c', '#2a78d6'
COL_C, COL_L = '#eb6834', '#1baf7a'          # gapreport's method colours: constructed, learned
N_PAIRS = 2000


def m_hi(n):
    return 6 * (n + 0.1) / 22.1


def constructed(n, k):
    """The design as a fixed module (cpu)."""
    return W.constructed(SimpleNamespace(n=n, E=n + 0.1, device='cpu'), 6 * k, m_hi(n), 1.0, device='cpu')


def phi_xy(w, xy):
    """(N, 2) normalized x, y -> (N, 2) z."""
    with torch.no_grad():
        q = torch.zeros(len(xy), 3)
        q[:, :2] = torch.as_tensor(np.asarray(xy), dtype=torch.float32)
        return w(q)[:, :2].numpy()


def walls(info):
    """Wall segments (S, 2, 2) in cell units: V[i, j] from (i, j) to (i, j + 1), H[i, j] from (i, j) to (i + 1, j)."""
    V, H = np.array(info['V']), np.array(info['H'])
    segs = [((i, j), (i, j + 1)) for i, j in zip(*np.nonzero(V))] + [((i, j), (i + 1, j)) for i, j in zip(*np.nonzero(H))]
    return np.array(segs, dtype=float)


def pair_file(d):
    """The prepare run's pairs, or its source run's (the 22 x 22 study's x 1 prepare trains on the cross maze's data)."""
    f = d / 'data/sampled_points.npy'
    if f.exists():
        return f
    src = (json.loads((d / 'results.json').read_text()).get('data') or {}).get('source')
    return Path(src) / 'data/sampled_points.npy'


def density_profile(run):
    """(cells, dz/dx) along x at a y row inside the bottom quadrants, from a run's warp_map.npz."""
    z = np.load(RESULTS / f'registered_{run}' / 'warp_map.npz')
    xs = z['xs'].astype(np.float64)
    zx = z['mapped'][:, len(xs) // 7, 0].astype(np.float64)
    return xs, np.gradient(zx, xs)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    P = json.loads((OUT / 'zdraw10.json').read_text())['prepares']
    rows = [('c22', 1, '22×22 quadrants'), ('c22', 12, '22×22 quadrants'), ('m10', 1, '10×10 quadrants'), ('m10', 12, '10×10 quadrants')]
    shade = LinearSegmentedColormap.from_list('mag', ['#ffffff', '#f2b79a'])
    fig, axes = plt.subplots(len(rows), 4, figsize=(11.2, 11.6))
    numbers = {}
    for r, (base, k, name) in enumerate(rows):
        dmain = RESULTS / P[f'{base}_main'][str(k)]
        dz = RESULTS / P[f'{base}_zdraw'][str(k)]
        info = json.loads((dmain / 'maze.json').read_text())
        n = info['n']
        E = n + 0.1
        m = m_hi(n)
        w = constructed(n, k)
        S = walls(info)
        Sn = (S - n / 2) / E                                    # normalized
        Sz = phi_xy(w, Sn.reshape(-1, 2)).reshape(-1, 2, 2)
        # magnification along x (= along y): dz/dx on a fine grid
        xs = np.linspace(-0.5, 0.5, 4001)
        zx = phi_xy(w, np.stack([xs, np.zeros_like(xs)], 1))[:, 0]
        dens = np.gradient(zx, xs)
        c0 = (n - 6 * k) // 2
        cells = xs * E + n / 2
        inner = (cells > c0 + 1) & (cells < c0 + 6 * k - 1)
        zlo, zhi = float(zx[0]), float(zx[-1])
        band_inner = float(np.interp(c0 + 6 * k - 1, cells, zx) - np.interp(c0 + 1, cells, zx))
        numbers[f'{base}/{k}'] = dict(n=n, m=m, z_half=zhi, band_interior_z=band_inner, quadrant_cell_z=m / E,
                                      density_band=float(np.median(dens[inner])), density_quadrant=float(np.median(dens[(cells > 0.5) & (cells < c0 - 0.5)])))
        # column 0: the maze, shaded by area magnification
        ax = axes[r, 0]
        F = np.interp(np.linspace(0, n, 401), cells, dens)
        A = np.log(np.outer(F, F)) / math.log(m * m)           # rows y, columns x (symmetric)
        ax.imshow(A, extent=(0, n, 0, n), origin='lower', cmap=shade, vmin=0, vmax=1, interpolation='nearest')
        ax.add_collection(LineCollection(S, colors=INK, linewidths=1.0 if n <= 22 else 0.6, capstyle='projecting'))
        ax.set_xlim(-0.6, n + 0.6)
        ax.set_ylim(-0.6, n + 0.6)
        ax.set_ylabel(f'{name}, gap ×{k}\n{n}×{n} cells, m = ×{m:.1f}', fontsize=9)
        # column 1: the warped maze
        lim = 3.0
        ax = axes[r, 1]
        ax.add_collection(LineCollection(Sz, colors=INK, linewidths=0.8, capstyle='projecting'))
        ax.set_xlabel(f'box {2 * zhi:.2f} units wide, band interior {band_inner:.2f}', fontsize=8, color='#57534e', labelpad=2)
        # columns 2, 3: pairs in z
        for c, d in ((2, dmain), (3, dz)):
            ax = axes[r, c]
            pts = np.load(pair_file(d), mmap_mode='r')[:N_PAIRS]
            z0 = phi_xy(w, np.asarray(pts[:, :2]))
            z1 = phi_xy(w, np.asarray(pts[:, 3:5]))
            ax.add_collection(LineCollection(Sz, colors='#a8a29e', linewidths=0.5, capstyle='projecting'))
            ax.scatter(z1[:, 0], z1[:, 1], s=1.6, color=X1C, lw=0, alpha=0.7, rasterized=True)
            ax.scatter(z0[:, 0], z0[:, 1], s=1.6, color=X0C, lw=0, alpha=0.8, rasterized=True)
        for c in (1, 2, 3):
            axes[r, c].set_xlim(-lim, lim)
            axes[r, c].set_ylim(-lim, lim)
        for c in range(4):
            ax = axes[r, c]
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(c > 0)
                sp.set_color('#d6d3d1')
    titles = ['the maze (shade: the warp’s area\nmagnification, ×m² / ×m / ×1)', 'the warped maze z = φ(x)\n(all z panels at one scale)',
              'training pairs drawn on the maze,\nseen in z', 'training pairs drawn on the warped\nmaze, seen in z']
    for c, t in enumerate(titles):
        axes[0, c].set_title(t, fontsize=9)
    fig.legend(handles=[Line2D([], [], ls='', marker='o', ms=4, color=X0C, label='x0 (start, within 0.175 cells of a wall)'),
                        Line2D([], [], ls='', marker='o', ms=4, color=X1C, label='x1 (end, a random step away)')],
               loc='lower center', ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.62, 0.0))
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(OUT / 'gap_warp.png', dpi=100)
    plt.close(fig)

    # ---- magnification profiles: constructed vs learned (the gap study, 22 x 22 quadrants)
    out, _ = GR.collect()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.3), sharey=True)
    learned = {}
    for ax, k in zip(axes, (1, 12)):
        n = 16 + 6 * k
        E = n + 0.1
        c0 = (n - 6 * k) // 2
        for x0 in (0, c0 + 6 * k):
            ax.axvspan(x0, x0 + c0, color=SHADE, alpha=0.10, lw=0)
        pm = out[k]['constructed'].get(87358)
        xs, d = density_profile(pm['_id'])
        ax.plot(xs * E + n / 2, d, color=COL_C, lw=2.0, label='constructed (fixed)', zorder=3)
        for i, (s, e) in enumerate(sorted(out[k]['learned'].items(), key=lambda t: str(t[0]))):
            xs, d = density_profile(e['_id'])
            cells = xs * E + n / 2
            ax.plot(cells, d, color=COL_L, lw=1.2, alpha=0.9, label='learned (3 training seeds)' if i == 0 else None, zorder=2)
            q = (cells > 0.5) & (cells < c0 - 0.5)
            b = (cells > c0 + 1.5) & (cells < c0 + 6 * k - 1.5)
            learned.setdefault(str(k), {})[str(s)] = dict(run=e['_id'], quadrant=float(np.median(d[q])), band_interior=float(np.median(d[b])),
                                                          z_half=float(np.interp(0.5, xs, np.cumsum(np.r_[0, 0.5 * (d[1:] + d[:-1]) * np.diff(xs)]) - 0)))
        ax.set_yscale('log')
        ax.set_ylim(0.8, 32)
        ax.set_yticks([1, 2, 4, 8, 16, 24])
        ax.set_yticklabels(['×1', '×2', '×4', '×8', '×16', '×24'])
        ax.set_xlim(0, n)
        ax.set_xlabel(f'x across the maze (cells), gap ×{k}: {n}×{n}, bound ×{m_hi(n):.1f}', fontsize=9)
        ax.tick_params(labelsize=8, colors='#57534e')
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.grid(axis='y', color='#e7e5e4', lw=0.6)
    axes[0].set_ylabel('magnification dz/dx', fontsize=9)
    axes[1].legend(fontsize=8, frameon=False, loc='center')
    fig.tight_layout()
    fig.savefig(OUT / 'gap_warp_density.png', dpi=100)
    plt.close(fig)
    numbers['learned'] = learned
    (OUT / 'gap_warp.json').write_text(json.dumps(numbers, indent=1))
    print('wrote', OUT / 'gap_warp.png', OUT / 'gap_warp_density.png', OUT / 'gap_warp.json')
    for key, v in numbers.items():
        print(key, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()} if key != 'learned' else
              {k: {s: (round(e['quadrant'], 2), round(e['band_interior'], 2)) for s, e in d.items()} for k, d in v.items()})


if __name__ == '__main__':
    main()
