"""Figures for the cross-warp study (inference only, from registered runs): report/*.png.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ~/.venvs/bellman-ntfield/bin/python jeffrey/cross_warp/figs.py \
        --warps lf,dz12 --speed id,u3,lf --seed 87358

warps.png   per variant: the warp's area stretch det J_psi (log2, heading 0) over the maze, and the image of a regular
            grid under phi / s (how the encoder sees the maze)
speed.png   true speed and the learned speed 1 / |grad_x T(x, goal 0)| at heading 0 for ntrl-demo and the variants
success.png success by distance bin, pooled over seeds, for the variants listed with --success
"""
import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
sys.path.insert(0, str(HERE))          # first: this study's report.py, not maze_scale's
import report as R  # noqa: E402
import warp as W  # noqa: E402

OUT = HERE / 'report'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e6e5e0'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#6250d6', '#e34948', '#eda100', '#8a8984']


def maze_from_json(info, device='cuda:0'):
    import maze as MZ
    m = MZ.Maze(info['n'], device)
    m.V, m.H = np.array(info['V'], bool), np.array(info['H'], bool)
    m.boxes = MZ.wall_boxes(m.V, m.H)
    m.Vt, m.Ht = torch.tensor(m.V, device=device), torch.tensor(m.H, device=device)
    return m


def draw_walls(ax, info, color='#3b3a37', lw=0):
    n = info['n']
    E = n + 0.1
    for x0, y0, x1, y1 in info['boxes_cells']:
        ax.add_patch(plt.Rectangle(((x0 - n / 2) / E, (y0 - n / 2) / E), (x1 - x0) / E, (y1 - y0) / E, color=color, lw=lw))


def clean(ax):
    ax.set_xlim(-0.505, 0.505)
    ax.set_ylim(-0.505, 0.505)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def run_of(warp, label, seed, prep):
    by = warp.get((label, prep), {})
    return by.get(seed)


def field(d, device='cuda:0'):
    if d.get('stage') == 'cross-warp':
        return W.load_field(d['_dir'] / 'training/checkpoint.pt', device)
    import plan as P
    return P.load_field(d['_dir'] / 'training/checkpoint.pt')


def speed_map(maze, net, goal, res=401):
    import plan as P
    xs = torch.linspace(-0.5, 0.5, res, device=maze.device)
    X, Y = torch.meshgrid(xs, xs, indexing='ij')
    q = torch.stack([X.reshape(-1), Y.reshape(-1), torch.zeros(res * res, device=maze.device)], 1)
    _, g = P.field_grad(net, torch.cat([q, goal[None].expand(len(q), 3)], 1))
    learned = (1 / torch.linalg.vector_norm(g, dim=1).clamp(min=1e-6)).reshape(res, res)
    true = maze.train_speed(maze.clearance(q)).reshape(res, res)
    return learned.cpu().numpy(), true.cpu().numpy()


def fig_warps(warp, labels, seed, prep, names):
    """Top: linear magnification sqrt(det J_xy) of phi including its overall scale (log colour, x1 .. x10), walls in black.
    Bottom: the image of a regular grid under phi, each panel in its own frame (shape only), walls in red."""
    labels = [l for l in labels if run_of(warp, l, seed, prep)]
    if not labels:
        return
    info = json.load(open(ROOT / prep / 'maze.json'))
    maze = maze_from_json(info)
    fig, axes = plt.subplots(2, len(labels), figsize=(3.4 * len(labels), 7.0), squeeze=False)
    xw = torch.linspace(-0.5, 0.5, 1201, device='cuda:0')
    XX, YY = torch.meshgrid(xw, xw, indexing='ij')
    qq = torch.stack([XX.reshape(-1), YY.reshape(-1), torch.zeros(XX.numel(), device='cuda:0')], 1)
    x, y, _ = maze.to_cells(qq)
    wall = W.wall_distance_cells(maze, torch.stack([x, y], 1)) <= 0.0
    im = None
    for k, lab in enumerate(labels):
        d = run_of(warp, lab, seed, prep)
        wp = W.load_field(d['_dir'] / 'training/checkpoint.pt').warp
        res = 301
        xs = torch.linspace(-0.5, 0.5, res, device='cuda:0')
        X, Y = torch.meshgrid(xs, xs, indexing='ij')
        q = torch.stack([X.reshape(-1), Y.reshape(-1), torch.zeros(res * res, device='cuda:0')], 1)
        with torch.no_grad():
            ld, _ = wp.logdet(q)
            mag = torch.exp(ld / 2) * wp.s
            m = wp(q)
            mw = wp(qq[wall])
        ax = axes[0, k]
        im = ax.imshow(np.log10(mag.reshape(res, res).T.cpu().numpy()), origin='lower', extent=(-0.5, 0.5, -0.5, 0.5), cmap='magma', vmin=-0.30103, vmax=1)
        draw_walls(ax, info, color='#dfe6ee')
        clean(ax)
        ax.set_title(names.get(lab, lab), fontsize=9, color=INK)
        ax = axes[1, k]
        M = m.reshape(res, res, 3).cpu().numpy()
        for i in range(0, res, 6):
            ax.plot(M[i, :, 0], M[i, :, 1], color='#9aa3ad', lw=0.3)
            ax.plot(M[:, i, 0], M[:, i, 1], color='#9aa3ad', lw=0.3)
        mw = mw.cpu().numpy()
        ax.scatter(mw[:, 0], mw[:, 1], s=0.05, color='#b4382f', lw=0)
        lo, hi = M[..., :2].min((0, 1)), M[..., :2].max((0, 1))
        pad = 0.01 * (hi - lo).max()
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=axes[0, :].tolist(), shrink=0.75, ticks=[np.log10(0.5), 0, np.log10(2), np.log10(4), np.log10(6), 1])
    cb.ax.set_yticklabels(['×½', '×1', '×2', '×4', '×6', '×10'])
    cb.set_label('magnification seen by the encoder', fontsize=8)
    fig.savefig(OUT / 'warps.png', dpi=110, bbox_inches='tight')
    plt.close(fig)


def fig_speed(base, warp, labels, seed, prep, names):
    info = json.load(open(ROOT / prep / 'maze.json'))
    maze = maze_from_json(info)
    goal = torch.tensor(np.load(ROOT / prep / 'goals.npy')[0], device='cuda:0')
    cols = []
    if (prep, seed) in base:
        cols.append(('ntrl-demo', base[(prep, seed)]))
    cols += [(names.get(l, l), run_of(warp, l, seed, prep)) for l in labels if run_of(warp, l, seed, prep)]
    fig, axes = plt.subplots(1, len(cols) + 1, figsize=(3.2 * (len(cols) + 1), 3.5))
    true = None
    for k, (name, d) in enumerate(cols):
        learned, true = speed_map(maze, field(d), goal)
        ax = axes[k + 1]
        ax.imshow(np.clip(learned, 0, 1.05).T, origin='lower', extent=(-0.5, 0.5, -0.5, 0.5), cmap='Blues_r', vmin=0, vmax=1.05)
        ax.set_title(f'{name}\nspeed fit {d["speed_fit"]["log_speed_corr"]:.2f}', fontsize=9, color=INK)
        clean(ax)
    axes[0].imshow(np.clip(true, 0, 1.05).T, origin='lower', extent=(-0.5, 0.5, -0.5, 0.5), cmap='Blues_r', vmin=0, vmax=1.05)
    axes[0].set_title('true speed', fontsize=9, color=INK)
    clean(axes[0])
    fig.tight_layout()
    fig.savefig(OUT / 'speed.png', dpi=110, bbox_inches='tight')
    plt.close(fig)


def fig_density(warp, labels, seed, prep, names):
    """Separable warps: local stretch along x (and y) against the cell coordinate, with the + band shaded."""
    info = json.load(open(ROOT / prep / 'maze.json'))
    n, w = info['n'], info.get('plus_width', 0)
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    if w:
        c0 = (n - w) // 2
        ax.axvspan(c0, c0 + w, color='#2a78d6', alpha=0.08, lw=0)
        ax.text(c0 + w / 2, 0.97, 'the + band', ha='center', va='top', fontsize=8, color=INK2, transform=ax.get_xaxis_transform())
    for k, lab in enumerate(labels):
        d = run_of(warp, lab, seed, prep)
        if d is None:
            continue
        net = W.load_field(d['_dir'] / 'training/checkpoint.pt')
        wp = net.warp
        if not wp.sep:
            continue
        hx = wp.h_sx().detach().cpu().numpy()
        z = np.linspace(-0.5, 0.5, len(hx)) * (n + 0.1) + n / 2
        dens = np.exp(hx)
        if wp.sep_norm:
            dens = dens / np.trapz(dens, np.linspace(-0.5, 0.5, len(hx)))
        ax.plot(z, dens * wp.s, color=SERIES[k % len(SERIES)], lw=1.6, label=names.get(lab, lab))
    ax.set_yscale('log')
    ax.set_xlim(0, n)
    ax.set_xlabel('x (cells)', fontsize=9, color=INK2)
    ax.set_ylabel('local stretch dx\'/dx', fontsize=9, color=INK2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / 'density.png', dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', default='results/registered_20261102T042454Z-62864e61')
    p.add_argument('--warps', default='')
    p.add_argument('--speed', default='')
    p.add_argument('--density', default='')
    p.add_argument('--seed', type=int, default=87358)
    p.add_argument('--names', default='{}', help='json {label: display name}')
    a = p.parse_args()
    OUT.mkdir(exist_ok=True)
    base, warp = R.collect()
    names = json.loads(a.names)
    prep = str(Path(a.prepared))
    base = {(Path(k[0]).relative_to(ROOT).as_posix() if Path(k[0]).is_absolute() else k[0], k[1]): v for k, v in base.items()}
    warp = {(k[0], Path(k[1]).relative_to(ROOT).as_posix() if Path(k[1]).is_absolute() else k[1]): v for k, v in warp.items()}
    if a.warps:
        fig_warps(warp, a.warps.split(','), a.seed, prep, names)
    if a.speed:
        fig_speed(base, warp, a.speed.split(','), a.seed, prep, names)
    if a.density:
        fig_density(warp, a.density.split(','), a.seed, prep, names)
    print('figures in', OUT)


if __name__ == '__main__':
    main()
