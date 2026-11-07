"""Figures for the grid-sync write-up (report/*.png). Inference only; nothing is trained or registered here.

    CUDA_VISIBLE_DEVICES=0 ~/.venvs/bellman-ntfield/bin/python jeffrey/grid_sync/figs.py [--only freq,success,speed,profile]

freq.png      the xy frequencies of the Fourier rows (training seed 87358's draw) in grid units (f / f_grid, f_grid = half
              period one cell) for ntrl-demo and each mode, 13 x 13 maze; dashed: the sync lines |f_x| = 1, |f_y| = 1
success.png   MPPI success against maze size: ntrl-demo (3 seeds), the earlier B x s sweep's best per maze, ring, circle,
              literal, scaled, snap (seed 87358 unless pooled seeds exist)
speed.png     learned speed 1 / |grad_x T(x, goal 0)| at heading 0, 13 x 13 and 22 x 22: truth, ntrl-demo, ring, circle
profile.png   learned speed across interior grid lines (mechanism.py profiles): doorways (no wall; truth 1) and walls
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import features as FT  # noqa: E402
_spec = importlib.util.spec_from_file_location('grid_sync_report', HERE / 'report.py')
GR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GR)

OUT = HERE / 'report'
# reference categorical slots (dataviz palette, light mode), fixed per entity
COL = {'ntrl-demo': '#2a78d6', 'ring': '#eb6834', 'circle': '#1baf7a', 'literal': '#eda100', 'scaled': '#e87ba4', 'snap': '#008300',
       'B × s': '#6250d6', 'ringhalf': '#eb6834', 'circlehalf': '#1baf7a', 'octave': '#eb6834', 'octavecircle': '#1baf7a', 'octaveperm': '#1baf7a'}
NAME = {'ntrl-demo': 'ntrl-demo', 'ring': 'grid-synced ring', 'circle': 'circle (control)', 'literal': 'literal (f_grid cos)', 'scaled': 'B × s_eff',
        'snap': 'B × s_eff snapped', 'B × s': 'best B × s (earlier sweep)', 'ringhalf': 'half ntrl-demo + half ring', 'circlehalf': 'half ntrl-demo + half circle',
        'octave': 'grid-synced octaves', 'octavecircle': 'octaves, constant magnitude (control)', 'octaveperm': 'octaves, shuffled magnitudes (control)'}
INK, INK2, GRID, AXIS = '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7'
# line charts: hue = family, line style = variant (solid single scale, dashed octaves, dotted half rows / shuffled)
LINE = {'ntrl-demo': (COL['ntrl-demo'], '-'), 'B × s': (COL['B × s'], '-'), 'ring': (COL['ring'], '-'), 'octave': (COL['ring'], '--'),
        'ringhalf': (COL['ring'], ':'), 'circle': (COL['circle'], '-'), 'octavecircle': (COL['circle'], '--'), 'octaveperm': (COL['circle'], ':'),
        'circlehalf': (COL['circle'], '-.'), 'literal': (COL['literal'], '-'), 'scaled': (COL['scaled'], '-'), 'snap': (COL['snap'], '-')}


def style(ax, grid_axis='y'):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig_freq(n=13, seed=87358):
    E = n + 0.1
    B = FT.draw(seed)
    grid = E / 2
    panels = [('ntrl-demo', B)] + [(m, FT.transform(B, E, m)) for m in ['ring', 'circle', 'literal', 'octave', 'octavecircle', 'scaled', 'snap']]
    fig, axes = plt.subplots(2, 4, figsize=(10.4, 5.6))
    for ax, (name, M) in zip(axes.ravel(), panels):
        f = M[:, :2].numpy() / grid
        lim = 2.9 if name in ('scaled', 'snap') else 1.6
        for v in (-2, -1, -0.5, 0.5, 1, 2):
            if abs(v) <= lim:
                ax.axvline(v, color=AXIS, lw=0.8, ls='--' if abs(v) >= 1 else ':', zorder=0)
                ax.axhline(v, color=AXIS, lw=0.8, ls='--' if abs(v) >= 1 else ':', zorder=0)
        # each row is a plane wave; its frequency vector and the opposite one are the same feature pair
        ax.scatter(np.r_[f[:, 0], -f[:, 0]], np.r_[f[:, 1], -f[:, 1]], s=8, color=COL.get(name, COL['ring']), lw=0, zorder=3)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        style(ax, None)
        ax.set_title(NAME[name], fontsize=9, color=INK)
        ax.tick_params(labelsize=7)
    for ax in axes[1]:
        ax.set_xlabel('f_x / f_grid', fontsize=8, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel('f_y / f_grid', fontsize=8, color=INK2)
    fig.suptitle(f'Fourier rows of ntrl-demo\'s draw (seed {seed}) on the {n} × {n} maze, in grid units (f_grid: half period one cell). '
                 'Dashed: in step with every grid line; dotted: with every other one', fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / 'freq.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


def fig_success():
    t = GR.collect()
    t = {n: {f: {s_: r for s_, r in by_s.items() if s_ in GR.MAIN_SEEDS} for f, by_s in by.items()} for n, by in t.items()}
    sizes = GR.SIZES
    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    series = ['ntrl-demo', 'B × s', 'ring', 'circle', 'literal', 'octave', 'octaveperm', 'octavecircle']
    for name in series:
        xs, ys, lo, hi = [], [], [], []
        for k, n in enumerate(sizes):
            by = t.get(n, {})
            if name == 'B × s':
                cand = [(GR.pooled(v), f) for f, v in by.items() if f.startswith('B ×') and GR.pooled(v)]
                cand = [c for c in cand if len(c[0][2]) == 3]
                if not cand:
                    continue
                (kk, mm, per), _ = max(cand, key=lambda c: c[0][0] / c[0][1])
            else:
                p = GR.pooled(by.get(name, {}))
                if p is None:
                    continue
                kk, mm, per = p
            m = mm / len(per)
            xs.append(k)
            ys.append(100 * kk / mm)
            lo.append(100 * min(per) / m)
            hi.append(100 * max(per) / m)
        if not xs:
            continue
        c, ls = LINE[name]
        ax.plot(xs, ys, color=c, ls=ls, lw=2, marker='o', ms=5, label=NAME[name] + (f' ({len(per)} seeds)' if len(per) > 1 else ''), zorder=3)
        ax.vlines(xs, lo, hi, color=c, lw=1, alpha=0.6)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f'{n} × {n}' for n in sizes])
    ax.set_ylim(0, 102)
    ax.set_ylabel('MPPI success (%)', fontsize=9, color=INK2)
    style(ax)
    ax.legend(fontsize=8, frameon=False, loc='center left', bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(OUT / 'success.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


def fig_speed(sizes=(13, 22), seed=87358, fields=('ntrl-demo', 'ring', 'octave', 'octavecircle')):
    import torch
    sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
    import maze as MZ
    import plan as P
    t = {}
    for d in GR.runs():
        if d['seed'] == seed:
            t[(d['maze']['n'], GR.field_key(d))] = d
    fig, axes = plt.subplots(len(sizes), 1 + len(fields), figsize=(2.9 * (1 + len(fields)), 3.25 * len(sizes)))
    axes = np.array(axes).reshape(len(sizes), -1)
    for r, n in enumerate(sizes):
        maze = MZ.Maze(n)
        goal = torch.tensor(np.load(ROOT / f'results/registered_{GR.PREPARED[n]}/goals.npy')[0], device=maze.device)
        res = 8 * n + 1
        xs = torch.linspace(-0.5, 0.5, res, device=maze.device)
        X, Y = torch.meshgrid(xs, xs, indexing='ij')
        q = torch.stack([X.reshape(-1), Y.reshape(-1), torch.zeros(res * res, device=maze.device)], 1)
        true = maze.train_speed(maze.clearance(q)).reshape(res, res).cpu().numpy()
        imgs = [(true, f'{n} × {n}: true speed')]
        for f in fields:
            d = t.get((n, f))
            if d is None:
                imgs.append((None, f'{NAME[f]}: no run'))
                continue
            net = P.load_field(d['_dir'] / 'training/checkpoint.pt')
            _, g = P.field_grad(net, torch.cat([q, goal[None].expand(len(q), 3)], 1))
            img = (1 / torch.linalg.vector_norm(g, dim=1).clamp(min=1e-6)).reshape(res, res).cpu().numpy()
            imgs.append((img, f'{NAME[f]} (fit {d["speed_fit"]["log_speed_corr"]:.2f})'))
            del net
            torch.cuda.empty_cache()
        for c, (img, title) in enumerate(imgs):
            ax = axes[r, c]
            if img is not None:
                ax.imshow(np.clip(img, 0, 1.05).T, origin='lower', extent=(-0.5, 0.5, -0.5, 0.5), cmap='Blues_r', vmin=0, vmax=1.05)
            ax.set_title(title, fontsize=8, color=INK)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout(h_pad=1.6)
    fig.savefig(OUT / 'speed.png', dpi=120)
    plt.close(fig)


def fig_profile(sizes=(13, 22), seed=87358, fields=('ntrl-demo', 'octave', 'octaveperm', 'octavecircle')):
    mech = json.loads((OUT / 'mechanism.json').read_text())
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mechanism import OFFSETS
    fig, axes = plt.subplots(2, len(sizes), figsize=(3.3 * len(sizes), 5.2), sharey='row')
    axes = np.array(axes).reshape(2, -1)
    for c, n in enumerate(sizes):
        for f in fields:
            r = next((v for v in mech.values() if v['n'] == n and v['field'] == f and v['seed'] == seed), None)
            if r is None:
                continue
            pr = r['profiles']
            col, ls = LINE[f]
            axes[0, c].plot(OFFSETS, pr['profile_doorway_median'], color=col, ls=ls, lw=2, label=NAME[f])
            axes[1, c].plot(OFFSETS, pr['profile_wall_median'], color=col, ls=ls, lw=2, label=NAME[f])
            lab = pr['profile_wall_label']
        axes[0, c].axhline(1, color=INK2, lw=1, ls=':', label='truth')
        if 'lab' in locals():
            axes[1, c].plot(OFFSETS, lab, color=INK2, lw=1, ls=':', label='truth (trained label)')
        axes[0, c].set_title(f'{n} × {n}: across a doorway', fontsize=9, color=INK)
        axes[1, c].set_title(f'{n} × {n}: across a wall', fontsize=9, color=INK)
        for rr in range(2):
            style(axes[rr, c])
            axes[rr, c].set_xlabel('offset from the grid line (cells)', fontsize=8, color=INK2)
    axes[0, 0].set_ylabel('learned speed (median)', fontsize=8, color=INK2)
    axes[1, 0].set_ylabel('learned speed (median)', fontsize=8, color=INK2)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / 'profile.png', dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--only', default='freq,success,speed,profile')
    a = p.parse_args()
    OUT.mkdir(exist_ok=True)
    only = set(a.only.split(','))
    if 'freq' in only:
        fig_freq()
    if 'success' in only:
        fig_success()
    if 'speed' in only:
        fig_speed()
    if 'profile' in only and (OUT / 'mechanism.json').exists():
        fig_profile()


if __name__ == '__main__':
    main()
