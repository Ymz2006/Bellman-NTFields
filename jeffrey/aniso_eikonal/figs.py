"""Figures of the anisotropic-eikonal study (report/*.png), from registered runs (report/summary.json).

    CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/aniso_eikonal/figs.py [--only indicatrix,fields,paths]

indicatrix.png  what one unit of time buys: equal circles on the maze (the speed is the same in every direction) and their
                images on the warp (ellipses: the same step is longer or shorter in z depending on where and which way).
fields.png      T(., goal) over the gap x12 maze for one goal inside a quadrant: the truth, the warped-maze field read as the
                maze's time, the inner product alone, the split head.
paths.png       the warped maze's shortest path against the maze's, for the queries where it costs the most (verify run).
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
sys.path.insert(0, str(ROOT / 'jeffrey/cross_warp'))
sys.path.insert(0, str(HERE))
OUT = HERE / 'report'
INK, MUTED, WALL = '#15181d', '#737b86', '#2f343b'
C_TRUE, C_WARP, C_ANISO, C_SPLIT = '#2a78d6', '#8b929b', '#eb6834', '#1baf7a'


def setup(k, base):
    import gap as GP
    GP.install(k, base)
    import maze as MZ
    n = 2 * GP.BLOCK + GP.BASE_WIDTH * k
    return GP, MZ.Maze(n, 'cuda:0')


def walls(ax, boxes, fn=None, color=WALL):
    for x0, y0, x1, y1 in boxes:
        if fn is not None:
            (x0, x1), (y0, y1) = fn(np.array([x0, x1])), fn(np.array([y0, y1]))
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=color, lw=0))


def indicatrix(k=1, base='cross22'):
    """Two panels at gap x k: the maze with equal circles, the warped maze with their images."""
    import warp as W
    GP, maze = setup(k, base)
    n, E = maze.n, maze.E
    m_hi = 6 * E / 22.1
    w = W.constructed(maze, GP.BASE_WIDTH * k, m_hi).double()
    t = torch.linspace(-0.5, 0.5, 20001, dtype=torch.float64, device=maze.device)
    with torch.no_grad():
        z = w(torch.stack([t, t, torch.zeros_like(t)], 1))[:, 0].cpu().numpy()
    cells = t.cpu().numpy() * E + n / 2
    F = lambda c: np.interp(c, cells, z)                      # cells -> z
    dF = lambda c: np.interp(c, cells, np.gradient(z, cells))  # dz / dcell
    B = GP.BLOCK
    r = 0.36
    pts = []
    for cx in list(np.arange(0.5, B, 1.0))[1::3] + list(np.linspace(B + 1.6, n - B - 1.6, 2)) + list(np.arange(n - B + 0.5, n, 1.0))[1::3]:
        for cy in list(np.arange(0.5, B, 1.0))[1::3] + list(np.linspace(B + 1.6, n - B - 1.6, 2)) + list(np.arange(n - B + 0.5, n, 1.0))[1::3]:
            pts.append((cx, cy))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.6), dpi=150)
    ax = axes[0]
    walls(ax, maze.boxes)
    for cx, cy in pts:
        ax.add_patch(Ellipse((cx, cy), 2 * r, 2 * r, fc=C_TRUE, ec='none', alpha=0.85))
    ax.set_xlim(-0.2, n + 0.2), ax.set_ylim(-0.2, n + 0.2), ax.set_aspect('equal')
    ax.set_title(f'the maze (gap ×{k}): one unit of time, every direction alike', fontsize=10, color=INK, loc='left')
    ax.set_xlabel('cells', fontsize=9, color=MUTED)
    ax = axes[1]
    walls(ax, maze.boxes, F)
    for cx, cy in pts:
        quad = (cx < B or cx >= n - B) and (cy < B or cy >= n - B)
        ax.add_patch(Ellipse((float(F(cx)), float(F(cy))), 2 * r * float(dF(cx)), 2 * r * float(dF(cy)), fc=C_TRUE if quad else C_ANISO, ec='none', alpha=0.9))
    lo, hi = float(F(0.0)), float(F(float(n)))
    ax.set_xlim(lo - 0.03, hi + 0.03), ax.set_ylim(lo - 0.03, hi + 0.03), ax.set_aspect('equal')
    ax.set_title('the same circles on the warp: small where it compresses, flat in the arms', fontsize=10, color=INK, loc='left')
    ax.set_xlabel('warp units', fontsize=9, color=MUTED)
    for ax in axes:
        ax.tick_params(labelsize=8, colors=MUTED)
        for s in ax.spines.values():
            s.set_color('#c9ced6')
    fig.tight_layout()
    fig.savefig(OUT / 'indicatrix.png', facecolor='white', bbox_inches='tight', pad_inches=0.12)
    plt.close(fig)
    return dict(k=k, base=base, m_hi=m_hi, circle_radius_cells=r)


def deficit(k=12, base='cross22'):
    """Along one axis of the gap x k maze: the warp's coordinate scaled back, phi(x) / m, and the deficit u = x - phi(x) / m."""
    import warp as W
    GP, maze = setup(k, base)
    n, E = maze.n, maze.E
    m_hi = 6 * E / 22.1
    w = W.constructed(maze, GP.BASE_WIDTH * k, m_hi).double()
    t = torch.linspace(-0.5, 0.5, 8001, dtype=torch.float64, device=maze.device)
    with torch.no_grad():
        z = w(torch.stack([t, t, torch.zeros_like(t)], 1))[:, 0].cpu().numpy()
    x = t.cpu().numpy() * E + n / 2                       # cells
    zs = (z - z[0]) / m_hi * E                            # phi(x) / m in cells, from the left wall
    u = x - x[0] - zs
    B = GP.BLOCK
    fig, ax = plt.subplots(figsize=(8.6, 3.5), dpi=150)
    for a, b in ((0, B), (n - B, n)):
        ax.axvspan(a, b, color='#eef1f5', lw=0)
    ax.plot(x, x - x[0], color=MUTED, lw=1.2, ls=(0, (4, 3)), label='x (the maze)')
    ax.plot(x, zs, color=C_TRUE, lw=2.2, label='φ(x) / m  (what the warp keeps)')
    ax.plot(x, u, color=C_SPLIT, lw=2.2, label='u(x) = x − φ(x) / m  (the deficit)')
    ax.set_xlim(0, n), ax.set_ylim(0, n * 1.13)
    ax.set_xlabel('position along one axis of the maze (cells)', fontsize=9, color=MUTED)
    ax.set_ylabel('cells', fontsize=9, color=MUTED)
    ax.text(B / 2, n * 1.05, 'quadrant', ha='center', fontsize=8.5, color=MUTED)
    ax.text(n - B / 2, n * 1.05, 'quadrant', ha='center', fontsize=8.5, color=MUTED)
    ax.text(n / 2, n * 1.05, 'open band', ha='center', fontsize=8.5, color=MUTED)
    ax.legend(fontsize=8.5, frameon=False, loc='center left', bbox_to_anchor=(0.12, 0.62))
    ax.tick_params(labelsize=8, colors=MUTED)
    for sp in ax.spines.values():
        sp.set_color('#c9ced6')
    fig.tight_layout()
    fig.savefig(OUT / 'deficit.png', facecolor='white', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    return dict(k=k, base=base, m_hi=m_hi, u_total_cells=float(u[-1]), kept_cells=float(zs[-1]))


def pick(summary, base, k, variant, seed=87358):
    for r in summary['runs']:
        if r['base'] == base and r['k'] == k and r['variant'] == variant and r['seed'] == seed:
            return r
    return None


def fields(k=4, base='c22', goal_index=30, res=3, variants=(('ziso-open', 'trained on the stretched space\n(its time read as the maze\'s)'), ('zaniso-Uquad', 'inner product alone'),
                                                               (('zaniso-Uquad-open-gs8x1+8x4-pwtauz-lb10', 'zaniso-Uquad-open-pwtauz-lb10', 'zaniso-Uquad-open-lb10'), 'inner product, full recipe'),
                                                               ('zres-Uquad-open-far-lb10', 'two-part field'))):
    """T(x, goal) on a grid of the maze (heading 0), every field in ntrl-demo's normalized time of the maze."""
    import aniso as AN
    import plan as P
    import truthfast as TF
    summary = json.load(open(OUT / 'summary.json'))
    GP, maze = setup(k, {'c22': 'cross22', 'm10': 'maze10'}[base])
    def first(v):
        for name in ([v] if isinstance(v, str) else v):
            r = pick(summary, base, k, name)
            if r is not None:
                return r
        return None
    runs = [(first(v), label) for v, label in variants]
    runs = [(r, label) for r, label in runs if r is not None]
    assert runs, 'no runs to draw'
    reg = json.load(open(ROOT / 'results' / f'registered_{runs[0][0]["id"]}' / 'results.json'))
    prepared = Path(reg['prepared'])
    goals = torch.tensor(np.load(prepared / 'goals.npy'), device='cuda:0')
    g = goals[goal_index:goal_index + 1]
    n, E = maze.n, maze.E
    c = torch.arange(1 / (2 * res), n, 1 / res, device='cuda:0')
    X, Y = torch.meshgrid(c, c, indexing='ij')
    q = torch.stack([(X.reshape(-1) - n / 2) / E, (Y.reshape(-1) - n / 2) / E, torch.zeros(X.numel(), device='cuda:0')], 1)
    free = (maze.clearance(q) > 0).reshape(X.shape).cpu().numpy()
    build = Path(os.environ.get('TMPDIR', '/tmp')) / 'aniso_figs_build'
    fg = TF.FastGrid(maze, build, threads=4)
    Tf, _ = fg.solve(g)
    panels = [('true travel time T*', fg.interp(Tf, torch.zeros(len(q), dtype=torch.long, device='cuda:0'), q).reshape(X.shape).cpu().numpy())]
    del Tf
    pairs = torch.cat([q, g.expand(len(q), 3)], 1)
    for r, label in runs:
        d = json.load(open(ROOT / 'results' / f'registered_{r["id"]}' / 'results.json'))
        ck = Path(d.get('checkpoint_from') or ROOT / 'results' / f'registered_{r["id"]}' / 'training/checkpoint.pt')
        sp = d.get('split')
        net = AN.load_field(ck, 'cuda:0', split=(sp['coeff'], sp['m_ref']) if sp else None)
        panels.append((label, (P.field_cost(net)(pairs) / d['t_scale']).reshape(X.shape).cpu().numpy()))
        del net
    vmax = float(np.nanmax(np.where(free & np.isfinite(panels[0][1]), panels[0][1], np.nan)))
    fig, axes = plt.subplots(1, len(panels), figsize=((3.55 if len(panels) <= 4 else 3.0) * len(panels), 4.1 if len(panels) <= 4 else 3.6), dpi=150)
    for ax, (label, T) in zip(np.atleast_1d(axes), panels):
        im = ax.imshow(np.where(free, T, np.nan).T, origin='lower', extent=(0, n, 0, n), cmap='viridis', vmin=0, vmax=vmax, interpolation='nearest')
        ax.contour(np.linspace(0, n, T.shape[0]), np.linspace(0, n, T.shape[1]), np.where(free, T, np.nan).T, levels=np.linspace(0, vmax, 15)[1:], colors='white', linewidths=0.4, alpha=0.7)
        gx, gy = float(g[0, 0] * E + n / 2), float(g[0, 1] * E + n / 2)
        ax.plot([gx], [gy], marker='*', ms=11, mfc='white', mec=INK, mew=0.8)
        ax.set_title(label, fontsize=9.5, color=INK, loc='left')
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color('#c9ced6')
    cb = fig.colorbar(im, ax=np.atleast_1d(axes).tolist(), fraction=0.025, pad=0.015)
    cb.set_label('travel time to the goal (maze sides at full speed)', fontsize=8.5, color=MUTED)
    cb.ax.tick_params(labelsize=8, colors=MUTED)
    fig.savefig(OUT / 'fields.png', facecolor='white', bbox_inches='tight')
    plt.close(fig)
    return dict(k=k, base=base, goal_index=goal_index, runs=[r['id'] for r, _ in runs], vmax=vmax)


def wallsfar(k=12, base='m10', res=24, variants=(('ziso-open', 'stretched space (reference)'),
                                                 (('zaniso-Uquad-open-gs8x1+8x6-pwtauz-lb10', 'zaniso-Uquad-open-gs8x1+8x4-pwtauz-lb10'), 'inner product, one field (recipe)'),
                                                 ('zres-Uquad-open-far-lb10', 'two-part field'))):
    """The travel time inside one quadrant for a goal in the next quadrant, minus its smallest value there: what MPPI sees of
    the start quadrant's walls when the goal is far. Heading 0; every field in the maze's normalized time."""
    import aniso as AN
    import plan as P
    import truthfast as TF
    summary = json.load(open(OUT / 'summary.json'))
    GP, maze = setup(k, {'c22': 'cross22', 'm10': 'maze10'}[base])

    def first(v):
        for name in ([v] if isinstance(v, str) else v):
            r = pick(summary, base, k, name)
            if r is not None:
                return r
        return None
    runs = [(first(v), label) for v, label in variants]
    runs = [(r, label) for r, label in runs if r is not None]
    assert runs, 'no runs to draw'
    n, E, B = maze.n, maze.E, GP.BLOCK
    # the goal: a free pose in the middle of the bottom-right quadrant (seeded draw; the start quadrant is the bottom-left)
    gen = torch.Generator('cuda:0').manual_seed(11)
    cand = maze.free(200000, gen, 0.3)
    cx, cy = (cand[:, 0] * E + n / 2).cpu().numpy(), (cand[:, 1] * E + n / 2).cpu().numpy()
    j = int(np.argmin((cx - (n - B / 2)) ** 2 + (cy - B / 2) ** 2))
    g = cand[j:j + 1]
    c = torch.arange(1 / (2 * res), B, 1 / res, device='cuda:0')
    X, Y = torch.meshgrid(c, c, indexing='ij')
    q = torch.stack([(X.reshape(-1) - n / 2) / E, (Y.reshape(-1) - n / 2) / E, torch.zeros(X.numel(), device='cuda:0')], 1)
    free = (maze.clearance(q) > 0).reshape(X.shape).cpu().numpy()
    build = Path(os.environ.get('TMPDIR', '/tmp')) / 'aniso_figs_build'
    fg = TF.FastGrid(maze, build, threads=4)
    Tf, _ = fg.solve(g)
    panels = [('true travel time', fg.interp(Tf, torch.zeros(len(q), dtype=torch.long, device='cuda:0'), q).reshape(X.shape).cpu().numpy())]
    del Tf
    pairs = torch.cat([q, g.expand(len(q), 3)], 1)
    for r, label in runs:
        d = json.load(open(ROOT / 'results' / f'registered_{r["id"]}' / 'results.json'))
        net = AN.load_field(ROOT / 'results' / f'registered_{r["id"]}' / 'training/checkpoint.pt', 'cuda:0')
        panels.append((label, (P.field_cost(net)(pairs) / d['t_scale']).reshape(X.shape).cpu().numpy()))
        del net
    # corridor cross-sections: along y at fixed x, a free run 0.5 to 0.8 cells long with a wall at both ends
    runs_y = []
    for i in range(free.shape[0]):
        col = free[i]
        j0 = None
        for jj in range(len(col) + 1):
            if jj < len(col) and col[jj]:
                j0 = jj if j0 is None else j0
            elif j0 is not None:
                if 0.5 * res <= jj - j0 <= 0.8 * res and j0 > 0 and jj < len(col):
                    runs_y.append((i, j0, jj))
                j0 = None
    rng = np.random.default_rng(4)
    # spread the sections over the quadrant: one per third of x, two rows
    chosen = []
    for a_, b_ in ((0, B / 3), (B / 3, 2 * B / 3), (2 * B / 3, B)):
        for lo, hi in ((0, B / 2), (B / 2, B)):
            cand_ = [r_ for r_ in runs_y if a_ <= (r_[0] + 0.5) / res < b_ and lo <= (r_[1] + r_[2]) / (2 * res) < hi]
            if cand_:
                chosen.append(cand_[rng.integers(len(cand_))])
    colors = [INK, C_TRUE, C_SPLIT, '#b8418e']
    fig, axes = plt.subplots(1, len(chosen), figsize=(2.35 * len(chosen), 2.7), dpi=150, sharey=True)
    for ax, (i, j0, j1) in zip(np.atleast_1d(axes), chosen):
        ys = (np.arange(j0, j1) + 0.5) / res
        for (label, T), col_ in zip(panels, colors):
            v = T[i, j0:j1]
            ax.plot(ys - ys[0], (v - np.nanmin(v)) * E, color=col_, lw=1.8 if label.startswith('true') else 1.5, label=label)
        ax.set_title(f'x = {(i + 0.5) / res:.1f}, y from {ys[0]:.1f}', fontsize=8, color=MUTED, loc='left')
        ax.tick_params(labelsize=7.5, colors=MUTED)
        ax.set_xlabel('across the corridor (cells)', fontsize=8, color=MUTED)
        for sp_ in ('top', 'right'):
            ax.spines[sp_].set_visible(False)
        for sp_ in ('left', 'bottom'):
            ax.spines[sp_].set_color('#c9ced6')
    np.atleast_1d(axes)[0].set_ylabel('travel time above its lowest\nvalue on the section (cells)', fontsize=8, color=MUTED)
    np.atleast_1d(axes)[0].legend(fontsize=7.5, frameon=False, loc='upper center', bbox_to_anchor=(len(chosen) / 2, 1.34), ncol=len(panels))
    fig.savefig(OUT / 'wallsfar.png', facecolor='white', bbox_inches='tight')
    plt.close(fig)
    # ends of the cuts at which the travel time rises towards the wall (end value above the value two samples, 0.08 cells, inside)
    rising = {label: int(sum(int(T[i, j0] > T[i, j0 + 2]) + int(T[i, j1 - 1] > T[i, j1 - 3]) for i, j0, j1 in chosen)) for label, T in panels}
    return dict(k=k, base=base, goal_cells=[float(cx[j]), float(cy[j])], runs=[r['id'] for r, _ in runs], cuts=len(chosen), ends=2 * len(chosen), rising_ends=rising)
    return dict(k=k, base=base, goal_cells=[float(cx[j]), float(cy[j])], runs=[r['id'] for r, _ in runs], vmax=vmax)


def paths(key='c22_k12', top=4):
    """The verify run's worst queries: the warped maze's shortest path and the maze's."""
    summary = json.load(open(OUT / 'summary.json'))
    v = summary['verify'][key]
    GP, maze = setup(v['k'], {'c22': 'cross22', 'm10': 'maze10'}[v['base']])
    ex = np.load(ROOT / 'results' / f'registered_{v["id"]}' / 'example_paths.npy', allow_pickle=True)[:top]
    n, E = maze.n, maze.E
    fig, axes = plt.subplots(1, len(ex), figsize=(3.4 * len(ex), 3.7), dpi=150)
    for ax, e in zip(np.atleast_1d(axes), ex):
        walls(ax, maze.boxes)
        for arr, col, lab in ((e['true'], C_TRUE, 'maze\'s shortest'), (e['warped'], C_ANISO, 'warped maze\'s shortest')):
            ax.plot(arr[:, 0] * E + n / 2, arr[:, 1] * E + n / 2, color=col, lw=1.6, label=lab)
        ax.set_xlim(-0.5, n + 0.5), ax.set_ylim(-0.5, n + 0.5), ax.set_aspect('equal')
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f'+{100 * (e["ratio"] - 1):.0f} % travel time', fontsize=9.5, color=INK, loc='left')
        for s in ax.spines.values():
            s.set_color('#c9ced6')
    np.atleast_1d(axes)[0].legend(fontsize=7.5, frameon=False, loc='lower left', bbox_to_anchor=(0.0, -0.17), ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / 'paths.png', facecolor='white', bbox_inches='tight')
    plt.close(fig)
    return dict(key=key, queries=[int(e['query']) for e in ex], ratios=[float(e['ratio']) for e in ex])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--only', default='indicatrix,fields,paths')
    a = p.parse_args()
    OUT.mkdir(exist_ok=True)
    out = {}
    f = OUT / 'figs.json'
    if f.exists():
        out = json.load(open(f))
    for name in a.only.split(','):
        # each figure installs its own gap maze: one figure per process (gap.install patches the maze module once)
        out[name] = {'indicatrix': indicatrix, 'fields': fields, 'paths': paths, 'deficit': deficit, 'wallsfar': wallsfar}[name]()
        json.dump(out, open(f, 'w'), indent=1)
        print(name, out[name])


if __name__ == '__main__':
    main()
