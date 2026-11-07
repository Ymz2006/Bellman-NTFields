"""Figures of the line-warp study (report/*.png). Reads registered fit runs; needs a GPU only for the map evaluations.

    python figs.py   (uses report/runs.json written by report.py)
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / 'jeffrey/maze_scale'), str(ROOT / 'jeffrey/cross_warp')]
import fit as FT  # noqa: E402
import lineloss as LL  # noqa: E402
import netwarp as NW  # noqa: E402
import obstacles as OB  # noqa: E402
import view as V  # noqa: E402

INK, MUTED, GRID, WALL, FREE, ACC, ACC2 = '#1f2a33', '#6b7a86', '#d9dfe4', '#20313f', '#ffffff', '#c8553d', '#2a7f9e'
BAND = '#e9edf1'


def style():
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK, 'axes.spines.top': False,
                         'axes.spines.right': False, 'font.family': 'DejaVu Sans'})


def load_warp(fit_dir, maze, dev='cuda:0'):
    state, cfg, _ = NW.load_fit(Path(fit_dir), dev)
    w = NW.NetWarp(dict(net=cfg, s=1.0)).to(dev)
    w.set_state(state, maze)
    return w


@torch.no_grad()
def steps_along(plane, maze, value, axis, theta, count=8001):
    """The step function along the z-line z_perp = value at heading theta: t (count,), g(t) in {0, 1}, for a plane map (uses its boundary image)."""
    b = OB.planes(maze, 'cspace', 1, 0.05)[0][0] if theta is None else None
    import maze as MZ
    free = OB.free_space(maze.boxes, maze.n, MZ.T_WALL, theta, MZ.ROBOT_L, MZ.ROBOT_W)
    h = MZ.T_WALL / 2
    outline = np.array([[-h, -h], [maze.n + h, -h], [maze.n + h, maze.n + h], [-h, maze.n + h], [-h, -h]])
    b = OB.Boundary(OB.rings_of(free) + [outline], maze.n, maze.E, 0.05)
    Z = plane(torch.tensor(b.points, device=maze.device))
    S = torch.tensor(b.segments, device=maze.device)
    line, t = LL.crossings(Z, S, torch.tensor([value], device=maze.device, dtype=Z.dtype), axis)
    return t.cpu().numpy()


def step_plot(ax, t, color, label):
    """Obstacle (1) / free (0) graph from sorted crossings t: obstacle, free, obstacle, ... , obstacle."""
    xs, ys = [t[0]], [1]
    for i in range(len(t) - 1):
        v = 1 if i % 2 == 0 else 0
        xs += [t[i], t[i + 1]]
        ys += [v, v]
    span = t[-1] - t[0]
    ax.plot((np.array(xs) - t[0]) / span, ys, color=color, lw=1.1, drawstyle='default')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['free', 'obstacle'])
    ax.set_xlim(0, 1)
    ax.set_title(label, loc='left', fontsize=10)


def run_lengths(t):
    return np.diff(t)


def fig_steps(fit_dir, maze, path, theta=0.0, rows=(0.3, 4.5, 10.5)):
    """The graph of the proposal along horizontal lines, in the maze and in the warped space."""
    w = load_warp(fit_dir, maze)
    fig, axes = plt.subplots(len(rows), 2, figsize=(11, 1.5 * len(rows) + 0.6), sharey=True)
    ident = lambda x: x
    with torch.no_grad():
        for r, cell_y in enumerate(rows):
            y = (cell_y - maze.n / 2) / maze.E
            t0 = steps_along(ident, maze, y, 0, theta)
            # the same row in the warped space: the z-line through the row's image at the centre of the maze
            zc = w.plane(torch.tensor([[0.0, y]], device=maze.device))[0, 1].item()
            t1 = steps_along(w.plane, maze, zc, 0, theta)
            step_plot(axes[r, 0], t0, ACC2, f'maze, row at {cell_y:g} cells: largest / smallest run {max(np.diff(t0)) / min(np.diff(t0)):.0f}')
            step_plot(axes[r, 1], t1, ACC, f'warped: largest / smallest run {max(np.diff(t1)) / min(np.diff(t1)):.0f}')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_gallery(items, maze, path, cols=3, theta=0.0):
    """items: (title, fit dir or None); one warped maze each."""
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis('off')
    for ax, (title, fit) in zip(axes, items):
        plane = (lambda x: x) if fit is None else load_warp(fit, maze).plane
        V.draw_maze(ax, plane, maze, theta)
        ax.set_title(title, fontsize=10, loc='left')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


@torch.no_grad()
def collect_runs(plane, maze, lines=512, headings=9, seed=0):
    """All closed runs along axis lines through the boundary (cell units of the maze for the identity, median-run units otherwise): free, obstacle."""
    dev = maze.device
    gen = torch.Generator(dev).manual_seed(seed)
    free, obst = [], []
    for b, _ in OB.planes(maze, 'cspace', headings, 0.1):
        Z = plane(torch.tensor(b.points, device=dev))
        S = torch.tensor(b.segments, device=dev)
        for axis in (0, 1):
            p = 1 - axis
            lo, hi = Z[:, p].min(), Z[:, p].max()
            i = torch.randint(len(Z), (lines,), generator=gen, device=dev)
            c = (Z[i, p] + (hi - lo) * (torch.rand(lines, generator=gen, device=dev) - 0.5) / maze.n).clamp(lo, hi)
            line, t = LL.crossings(Z, S, c, axis)
            ell, mask, _ = LL.runs(line, t, lines)
            kind = torch.arange(ell.shape[1], device=dev)[None] % 2 == LL.FREE_PARITY
            free.append(ell[mask & kind])
            obst.append(ell[mask & ~kind])
    return torch.cat(free).cpu().numpy(), torch.cat(obst).cpu().numpy()


def constructed_density(maze, band, m_hi=6.0, m_lo=1.0):
    """The hand-built pm6 warp's density along x (cross_warp/warp.py design_plusmag), for comparison."""
    import warp as W
    w = W.constructed(maze, band, m_hi, m_lo)
    x = torch.linspace(-0.5, 0.5, 4001, device=maze.device)
    q = torch.zeros(len(x), 3, device=maze.device)
    q[:, 0] = x
    with torch.no_grad():
        z = w(q)[:, 0]
    return (x * maze.E + maze.n / 2).cpu().numpy(), np.gradient(z.cpu().numpy(), x.cpu().numpy())


def fig_tilt(groups, path):
    """Stretch across the first quadrant and the start of the band at gap x12, as a multiple of the quadrant's median stretch, so a flat line at 1 is an even quadrant. groups: [(title, block, n, E, [(label, run dir, color, style)])]."""
    fig, axes = plt.subplots(1, len(groups), figsize=(9.6, 3.4), sharey=True)
    for ax, (title, block, n, E, items) in zip(axes, groups):
        reach = block + 3
        ax.axvspan(block, reach, color=BAND, zorder=0)
        ax.axhline(1, color=GRID, lw=0.8, zorder=1)
        ax.axhline(0.5, color=GRID, lw=0.8, ls=':', zorder=1)
        for label, r, color, style in items:
            H = torch.load(r / 'phi.pt', map_location='cpu', weights_only=False)['state']['Hfrozen'][0, 0, :].numpy()
            c = np.linspace(-0.5, 0.5, len(H)) * E + n / 2
            quad = (c >= 0) & (c < block)
            d = np.exp(H - np.median(H[quad]))
            keep = (c >= 0) & (c <= reach)
            ax.plot(c[keep], d[keep], color=color, ls=style, lw=1.9, label=label, zorder=3)
        ax.set_yscale('log')
        ax.set_yticks([0.1, 0.5, 1, 2, 4])
        ax.set_yticklabels(['0.1', '0.5', '1', '2', '4'])
        ax.minorticks_off()
        ax.set_ylim(0.04, 7)
        ax.set_xlim(0, reach)
        ax.set_title(title, loc='left', fontsize=10.5, color=INK)
        ax.set_xlabel('x (cells); shaded: the band', fontsize=9.5, color=INK)
        ax.tick_params(colors=MUTED, labelsize=9)
        for k in ('top', 'right'):
            ax.spines[k].set_visible(False)
    axes[0].set_ylabel('stretch / quadrant median', fontsize=9.5, color=INK)
    axes[0].legend(frameon=False, fontsize=9, loc='lower left')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_density(items, maze, path):
    """Stretch along x (solid lines) of fitted warps, each scaled to the constructed warp's quadrant stretch."""
    dev = maze.device
    fig, axes = plt.subplots(len(items), 1, figsize=(10, 1.9 * len(items) + 0.5), sharex=True, sharey=True)
    x = torch.linspace(-0.5, 0.5, 4001, device=dev)
    c = (x * maze.E + maze.n / 2).cpu().numpy()
    cc, dc = constructed_density(maze, 6)
    for ax, (title, fit) in zip(np.atleast_1d(axes), items):
        w = load_warp(fit, maze)
        q = torch.zeros(len(x), 2, device=dev)
        q[:, 0] = x
        with torch.no_grad():
            z = w.plane(q)[:, 0]
        d = np.gradient(z.cpu().numpy(), x.cpu().numpy())
        quad = (c < 8) | (c > 14)
        d = d * 6.0 / np.median(d[quad])
        ax.axvspan(8, 14, color='#e9edf1', lw=0)
        ax.plot(cc, dc, color=MUTED, lw=1.2, ls='--', label='hand-built warp (x 6)')
        ax.plot(c, d, color=ACC, lw=1.4, label='learned')
        ax.set_yscale('log')
        ax.set_yticks([1, 2, 6, 20])
        ax.set_yticklabels(['1', '2', '6', '20'])
        ax.minorticks_off()
        ax.set_ylim(0.8, 40)
        ax.set_title(title, loc='left', fontsize=10)
        ax.grid(axis='y', color=GRID, lw=0.6)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=9, loc='upper center', ncol=2)
    np.atleast_1d(axes)[-1].set_xlabel('x (cells); the shaded span is the +')
    fig.supylabel('stretch dz/dx', fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_runs(panels, path):
    """Histograms of run length: panels = [(title, maze, fit dir or None)], lengths in units of each map's median run."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 3.1), sharey=True)
    bins = np.logspace(-1.5, 3, 56)
    for ax, (title, maze, fit, tag) in zip(np.atleast_1d(axes), panels):
        ident = lambda x: x
        for plane, color, label in ((ident, ACC2, 'maze'), (load_warp(fit, maze).plane, ACC, 'warped')):
            f, o = collect_runs(plane, maze)
            r = np.concatenate([f, o])
            r = r[r > 0] / np.median(r[r > 0])
            ax.hist(r, bins=bins, density=False, weights=np.full(len(r), 1 / len(r)), histtype='step', color=color, lw=1.5, label=f'{label}: largest / median {np.quantile(r, 0.999):.0f}')
        ax.set_xscale('log')
        ax.set_title(title, loc='left', fontsize=10)
        ax.set_xlabel('run length / median run')
        ax.legend(frameon=False, fontsize=8.5)
    np.atleast_1d(axes)[0].set_ylabel('share of runs')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def tilt(out, run):
    items = lambda a, b, c: [('TV + dilation (the recipe)', run(a), ACC, '-'), ('bound at TV’s range', run(b), ACC2, '--'), ('bound 3, no TV', run(c), INK, '-')]
    fig_tilt([('10×10 quadrants, gap ×12', 5, 82, 82.1, items('m10k12d', 'm10k12r', 'm10k12nt')), ('22×22 quadrants, gap ×12', 8, 88, 88.1, items('c22k12dil', 'c22k12r', 'c22k12nt'))], out / 'tilt.png')


def main():
    style()
    out = HERE / 'report'
    B = json.load(open(out / 'runs.json'))
    run = lambda k: ROOT / 'results' / f'registered_{B[k]}'
    tilt(out, run)
    if sys.argv[1:] == ['tilt']:
        return
    PLUS = ROOT / 'results/registered_20261102T042454Z-62864e61'
    maze, _ = FT.load_maze(PLUS, 'cuda:0')
    fig_steps(run('dil'), maze, out / 'steps.png', rows=(3.5, 4.5, 6.5))
    fig_gallery([('free ResNet, axis lines: stripes', run('fitA')), ('free ResNet, any-angle lines: stripes', run('fitB')), ('ResNet, anisotropy limit 12', run('rsA12')),
                 ('separable, no TV', run('sep125')), ('the recipe: separable + TV + dilation', run('dil')), ('triangular + TV', run('triTV'))], maze, out / 'gallery.png')
    fig_density([('no regularizer (bound 1.25)', run('sep125')), ('low-pass Fourier features', run('sepL125')), ('total variation 0.10', run('bt10')), ('total variation 0.10 + one-cell dilation: the recipe', run('dil'))], maze, out / 'density.png')


if __name__ == '__main__':
    main()
