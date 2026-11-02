"""Figures for the maze-scale write-up (report/*.png), from the registered runs (report.collect()).

    CUDA_VISIBLE_DEVICES=0 ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/figs.py

mazes.png        the five mazes at the same size with the longest certified query's truth path
success.png      MPPI success by true travel distance, one panel per maze: ntrl-demo (training-seed mean, seed range),
                 segment DP, oracle and straight line
speedmaps.png    learned speed 1 / |grad_x T(x, goal)| at heading 0 against the true speed (n = 7 / 10 / 13 / 22) for
                 ntrl-demo and ntrl-demo with B x 3 (fourier.py)
paths13.png      13-cell maze, one query per bin 4-8 / 8-16 / 16-32 where the fields' outcomes differ
uturn13.png      13-cell maze, u-turns around wall ends (uturn.py): truth and planning seed 0 of ntrl-demo, segment DP
                 + wall term (weight 0.3) and B x 3, close-up around the wall end
Inference only (checkpoints of the registered runs); nothing is trained or registered here.
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import maze as MZ  # noqa: E402
import plan as P  # noqa: E402
import report as R  # noqa: E402
from common import BINS, bin_labels  # noqa: E402

OUT = HERE / 'report'
# validated categorical slots (dataviz reference palette, light mode) and a sequential blue ramp for maze size
C = dict(ntrl='#2a78d6', dp='#eb6834', zoom='#1baf7a', dpchk='#6250d6', oracle='#8a8984', straight='#c3c2b7')
RAMP = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b']
VCOL = {'segment DP': C['dp'], 'segment DP + wall term': '#e34948', 'segment DP + wall term (weight 0.3)': C['dpchk'], 'segment DP, checked pieces': '#e34948', 'segment DP, zoomed base': '#eda100'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e6e5e0'


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def draw_maze(ax, info, color='#3b3a37'):
    n = info['n']
    E = n + MZ.T_WALL
    for x0, y0, x1, y1 in info['boxes_cells']:
        ax.add_patch(plt.Rectangle(((x0 - n / 2) / E, (y0 - n / 2) / E), (x1 - x0) / E, (y1 - y0) / E, color=color, lw=0))
    ax.set_xlim(-0.505, 0.505)
    ax.set_ylim(-0.505, 0.505)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def fig_mazes(prep):
    fig, axes = plt.subplots(1, len(prep), figsize=(3.1 * len(prep), 3.3))
    for ax, n in zip(np.atleast_1d(axes), sorted(prep)):
        d = prep[n]['_dir']
        info = json.load(open(d / 'maze.json'))
        draw_maze(ax, info)
        q = np.load(d / 'queries.npz')
        paths = np.load(d / 'truth_paths.npy', allow_pickle=True)
        k = int(np.argmax(q['d_cells']))
        p = paths[k]
        ax.plot(p[:, 0], p[:, 1], color=C['dp'], lw=1.6)
        ax.plot(*p[0, :2], 'o', color=C['ntrl'], ms=6, mec='white', mew=1)
        ax.plot(*p[-1, :2], 's', color=C['ntrl'], ms=6, mec='white', mew=1)
        ax.set_title(f'{n} x {n} cells  (longest query {q["d_cells"][k]:.0f} cells)', fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / 'mazes.png', dpi=130)
    plt.close(fig)


def pooled(by_seed, lab):
    vals = [d['success_by_bin'].get(lab) for d in by_seed.values()]
    vals = [v for v in vals if v]
    if not vals:
        return None, None, None
    rates = [100 * v[0] / v[1] for v in vals]
    return 100 * sum(v[0] for v in vals) / sum(v[1] for v in vals), min(rates), max(rates)


def fig_success(prep, ntrl, seg):
    labels = bin_labels()
    ns = sorted(prep)
    fig, axes = plt.subplots(1, len(ns), figsize=(3.2 * len(ns), 3.2), sharey=True)
    for ax, n in zip(np.atleast_1d(axes), ns):
        labs = [l for l in labels if prep[n]['queries']['per_bin'].get(l)]
        x = np.arange(len(labs))
        o = prep[n]['oracle']
        orc = [100 * sum(o[s]['by_bin'][l][0] for s in o) / sum(o[s]['by_bin'][l][1] for s in o) for l in labs]
        ax.plot(x, orc, color=C['oracle'], lw=1.2, ls='--', label='oracle (MPPI on T*)')
        st = prep[n]['straight']['by_bin']
        ax.plot(x, [100 * st[l][0] / st[l][1] for l in labs], color=C['straight'], lw=1.2, ls=':', label='straight line')
        series = [('ntrl-demo', ntrl.get(n, {}), C['ntrl'])]
        for v in ('segment DP', 'segment DP + wall term (weight 0.3)'):
            if seg.get(v, {}).get(n):
                series.append((v, seg[v][n], VCOL[v]))
        for name, by_seed, col in series:
            if not by_seed:
                continue
            m = [pooled(by_seed, l) for l in labs]
            y = np.array([v[0] if v[0] is not None else np.nan for v in m])
            ax.plot(x, y, color=col, lw=2, marker='o', ms=4, label=name)
            if len(by_seed) > 1:
                lo = np.array([v[1] if v[1] is not None else np.nan for v in m])
                hi = np.array([v[2] if v[2] is not None else np.nan for v in m])
                ax.vlines(x, lo, hi, color=col, lw=1, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labs, rotation=45, fontsize=7)
        ax.set_title(f'{n} x {n}', fontsize=10, color=INK)
        ax.set_ylim(-3, 103)
        style(ax)
        ax.set_xlabel('true travel distance (cells)', fontsize=8, color=INK2)
    np.atleast_1d(axes)[0].set_ylabel('MPPI success (%)', fontsize=9, color=INK2)
    h, l = np.atleast_1d(axes)[-1].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=min(6, len(l)), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / 'success.png', dpi=130)
    plt.close(fig)


def speed_map(maze, net, goal, res=401):
    xs = torch.linspace(-0.5, 0.5, res, device=maze.device)
    X, Y = torch.meshgrid(xs, xs, indexing='ij')
    q = torch.stack([X.reshape(-1), Y.reshape(-1), torch.zeros(res * res, device=maze.device)], 1)
    _, g = P.field_grad(net, torch.cat([q, goal[None].expand(len(q), 3)], 1))
    learned = (1 / torch.linalg.vector_norm(g, dim=1).clamp(min=1e-6)).reshape(res, res)
    true = maze.train_speed(maze.clearance(q)).reshape(res, res)
    return learned.cpu().numpy(), true.cpu().numpy()


def fig_speedmaps(prep, ntrl, fourier, sizes=(7, 10, 13, 22), seed=87358, scale=3.0):
    """Rows: true speed, ntrl-demo's, and ntrl-demo's with B x scale (fourier.py), at heading 0 towards goal 0."""
    sizes = [n for n in sizes if n in prep and seed in ntrl.get(n, {})]
    if not sizes:
        return
    rows = 3
    fig, axes = plt.subplots(rows, len(sizes), figsize=(3.0 * len(sizes), 3.1 * rows))
    axes = np.array(axes).reshape(rows, -1)
    for k, n in enumerate(sizes):
        maze = MZ.Maze(n)
        goal = torch.tensor(np.load(prep[n]['_dir'] / 'goals.npy')[0], device=maze.device)
        learned, true = speed_map(maze, P.load_field(ntrl[n][seed]['_dir'] / 'training/checkpoint.pt'), goal)
        panels = [(true, f'{n} x {n}: true speed'), (learned, f'ntrl-demo (speed fit {ntrl[n][seed]["speed_fit"]["log_speed_corr"]:.2f})')]
        z = fourier.get((n, scale, seed))
        if z is not None:
            lz, _ = speed_map(maze, P.load_field(z['_dir'] / 'training/checkpoint.pt'), goal)
            panels.append((lz, f'B x {scale:g} (speed fit {z["speed_fit"]["log_speed_corr"]:.2f})'))
        else:
            panels.append((None, f'no B x {scale:g} run yet'))
        for r, (img, title) in enumerate(panels):
            ax = axes[r, k]
            if img is not None:
                ax.imshow(np.clip(img, 0, 1.05).T, origin='lower', extent=(-0.5, 0.5, -0.5, 0.5), cmap='Blues_r', vmin=0, vmax=1.05)
            ax.set_title(title, fontsize=8, color=INK)
            ax.set_xticks([])
            ax.set_yticks([])
            if img is None:
                for sp in ax.spines.values():
                    sp.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / 'speedmaps.png', dpi=120)
    plt.close(fig)


def fig_paths(prep, ntrl, seg, n=40, seed=87358, count=4, name=None, pick_bins=None):
    """Planned paths (planning seed 0) for a few queries: truth, ntrl-demo and the DP chains of the same training seed.
    pick_bins: one query per listed bin label, preferring queries where the fields' outcomes differ."""
    if n not in prep or seed not in ntrl.get(n, {}):
        return
    d = prep[n]['_dir']
    info = json.load(open(d / 'maze.json'))
    q = np.load(d / 'queries.npz')
    truth = np.load(d / 'truth_paths.npy', allow_pickle=True)
    ours = [('ntrl-demo', ntrl[n][seed]['_dir'], C['ntrl'])]
    for v in ('segment DP', 'segment DP + wall term (weight 0.3)'):
        if seed in seg.get(v, {}).get(n, {}):
            ours.append((v, seg[v][n][seed]['_dir'], VCOL[v]))
    ok = np.stack([np.load(rd / 'rows.npz')['success'][0] for _, rd, _ in ours])          # (fields, queries), planning seed 0
    labels = bin_labels()
    if pick_bins:
        picks = []
        for lab in pick_bins:
            b = labels.index(lab)
            idx = np.nonzero(q['bin'] == b)[0]
            if not len(idx):
                continue
            differ = [i for i in idx if ok[:, i].any() and not ok[:, i].all()]
            picks.append(int((differ or list(idx))[0]))
    else:
        picks = [int(i) for i in np.argsort(-q['d_cells'])[:count * 3:3]]
    fig, axes = plt.subplots(1, len(picks), figsize=(3.4 * len(picks), 3.7))
    for ax, i in zip(np.atleast_1d(axes), picks):
        draw_maze(ax, info, color='#9c9b97')
        ax.plot(truth[i][:, 0], truth[i][:, 1], color=INK, lw=1.2, ls='--', label='truth')
        for k, (nm, rd, col) in enumerate(ours):
            p = np.load(rd / 'paths_seed0.npy', allow_pickle=True)[i]
            rows = np.load(rd / 'rows.npz')
            conv, col_hit = bool(rows['converged'][0, i]), bool(rows['collision'][0, i])
            if not conv:
                p = p[:-1]                      # the scorer appends the goal to every path; a stalled run never got there
            if ok[k, i]:
                how = 'reaches goal'
            elif conv:
                how = 'hits a wall'
            else:                               # stalled: did it touch a wall on the way?
                from posthoc import first_contact
                how = 'stalls' if first_contact(MZ.Maze(n), p) is None else 'hits a wall, stalls'
            ax.plot(p[:, 0], p[:, 1], color=col, lw=1.4, label=f'{nm} ({how})')
            if not ok[k, i]:
                ax.plot(*p[-1, :2], 'x', color=col, ms=7, mew=2)
        ax.plot(*truth[i][0, :2], 'o', color=INK, ms=5)
        ax.plot(*truth[i][-1, :2], 's', color=INK, ms=5)
        ax.set_title(f'{n} x {n}, query {i}: {q["d_cells"][i]:.1f} cells', fontsize=9, color=INK)
        ax.legend(fontsize=6, loc='lower left', frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / (name or f'paths{n}.png'), dpi=130)
    plt.close(fig)


def fig_uturn(ntrl, seg, fourier, n=13, seed=87358, count=4, half=1.4):
    """Close-ups of u-turn queries (uturn.py) on the n-cell maze: truth path and planning seed 0 of three fields."""
    ut = [d for d in R.runs('uturn') if d['maze']['n'] == n]
    if not ut:
        return
    ud = ut[-1]['_dir']
    z = np.load(ud / 'uturn.npz')
    truth = np.load(ud / 'truth_paths.npy', allow_pickle=True)
    info = json.load(open(ud / 'maze.json'))
    want = [('ntrl-demo', ntrl.get(n, {}).get(seed), C['ntrl'])]
    w = seg.get('segment DP + wall term (weight 0.3)', {}).get(n, {}).get(seed)
    want.append(('segment DP + wall term (weight 0.3)', w, VCOL['segment DP + wall term (weight 0.3)']))
    want.append(('ntrl-demo, B x 3', fourier.get((n, 3.0, seed)), '#1baf7a'))
    evals = {}
    for d in R.runs('uturneval'):
        for rid in d['runs']:
            evals[rid] = d['_dir']
    fields = []
    for name, d, col in want:
        if d is None or d['_id'] not in evals:
            continue
        paths = np.load(evals[d['_id']] / 'paths' / f'{d["_id"]}.npy', allow_pickle=True)
        fields.append((name, paths, col))
    if not fields:
        return
    maze = MZ.Maze(n)
    outcome = []                                                # per field, per query: 'ok' / 'wall' / 'stall'
    for name, paths, col in fields:
        res = []
        tol = P.settings(maze)['tol']
        for p in paths:
            pt = torch.tensor(p[:-1], device=maze.device)           # the planner appends the goal; judge the driven part
            hit = bool((maze.clearance(P.densify(pt, 0.005 * maze.pitch)) < 0).any()) if len(pt) > 1 else False
            near = float(torch.linalg.vector_norm(pt[-1] - torch.tensor(z['goal'][len(res)], device=maze.device))) <= 1.01 * tol
            res.append('wall' if hit else ('ok' if near else 'stall'))
        outcome.append(res)
    outcome = np.array(outcome)
    differ = [i for i in range(len(truth)) if len(set(outcome[:, i])) > 1]
    picks = (differ + [i for i in range(len(truth)) if i not in differ])[:count]
    fig, axes = plt.subplots(1, len(picks), figsize=(3.2 * len(picks), 3.6))
    E = n + MZ.T_WALL
    for ax, i in zip(np.atleast_1d(axes), picks):
        draw_maze(ax, info, color='#9c9b97')
        vi, vj, _ = z['end'][i]
        cx, cy = (vi - n / 2) / E, (vj - n / 2) / E
        ax.set_xlim(cx - half / E, cx + half / E)
        ax.set_ylim(cy - half / E, cy + half / E)
        ax.plot(truth[i][:, 0], truth[i][:, 1], color=INK, lw=1.2, ls='--', label='truth')
        for k, (name, paths, col) in enumerate(fields):
            p = paths[i] if outcome[k, i] == 'ok' else paths[i][:-1]
            how = dict(ok='reaches goal', wall='hits the wall', stall='stalls')[outcome[k, i]]
            ax.plot(p[:, 0], p[:, 1], color=col, lw=1.5, label=f'{name} ({how})')
            if outcome[k, i] != 'ok':
                ax.plot(*p[-1, :2], 'x', color=col, ms=7, mew=2)
        ax.plot(*truth[i][0, :2], 'o', color=INK, ms=5)
        ax.plot(*truth[i][-1, :2], 's', color=INK, ms=5)
        ax.set_title(f'{n} x {n}, wall end at ({vi}, {vj})', fontsize=9, color=INK)
        ax.legend(fontsize=6, loc='lower left', frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / f'uturn{n}.png', dpi=130)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    prep, ntrl, seg = R.collect()
    fourier = {(d['maze']['n'], float(d['b_scale']), d['seed']): d for d in R.runs('fourier')}
    fig_mazes(prep)
    fig_success(prep, ntrl, seg)
    fig_speedmaps(prep, ntrl, fourier)
    fig_paths(prep, ntrl, seg, n=13, pick_bins=['4-8', '8-16', '16-32'], name='paths13.png')
    fig_uturn(ntrl, seg, fourier)
    print('figures in', OUT)


if __name__ == '__main__':
    main()
