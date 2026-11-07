"""Figures and fit curves that need no training: report/rows.png, report/fit.json.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/speed_fourier/figs.py

fit.json: per maze and number of terms M, the median R^2 of a row against the obstacle signal o and the effective rank of 32 rows (exp of the entropy of the
normalized squared singular values), for line and ridge rows.
rows.png: the 13 x 13 obstacle signal with one line, the signal on that line with the sums for M = 5 and 30, and the row maps of two directions at M = 5 and 30.
"""
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
import embed as EM  # noqa: E402
import linefit as LF  # noqa: E402
import maze as MZ  # noqa: E402

TERMS = [2, 5, 10, 15, 20, 30, 60, 120, 240]
INK, MUTED, BLUE, ORANGE = '#15181d', '#737b86', '#2a78d6', '#eb6834'


def effective_rank(X):
    X = X - X.mean(0)
    s = np.linalg.svd(X, compute_uv=False)
    p = s ** 2 / (s ** 2).sum()
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def curves():
    out = {}
    for n in (13, 22):
        m = MZ.Maze(n, 'cpu')
        o = 1 - LF.speed_map(m)
        out[n] = dict(o_mean=float(o.mean()))
        for scope in ('line', 'ridge'):
            rows = []
            for M in TERMS:
                t, d = LF.fit(o, m.E, 32, M, scope)
                rows.append(dict(M=M, r2=d['r2_median'], rank=effective_rank(EM.standardize(t).reshape(-1, 32).astype(np.float64)), f_top=d['f_over_grid_quantiles'][-1]))
                print(n, scope, rows[-1], flush=True)
            out[n][scope] = rows
    (HERE / 'report/fit.json').write_text(json.dumps(out, indent=1) + '\n')


def rows_figure():
    n = 13
    m = MZ.Maze(n, 'cpu')
    o = 1 - LF.speed_map(m)
    h = 1.0 / (LF.NODES_PER_CELL * m.E)
    ax0 = np.linspace(-0.5, 0.5, o.shape[0])
    theta = math.radians(30)
    t0 = 0.06
    fig = plt.figure(figsize=(11.2, 6.4), dpi=170)
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.33, wspace=0.14, left=0.04, right=0.985, top=0.93, bottom=0.08)
    a = fig.add_subplot(gs[:, 0:2])
    a.imshow(o.T, origin='lower', extent=[-0.5, 0.5, -0.5, 0.5], cmap='Blues', vmin=0, vmax=1)
    u = np.linspace(-0.75, 0.75, 400)
    x, y = u * math.cos(theta) - t0 * math.sin(theta), u * math.sin(theta) + t0 * math.cos(theta)
    keep = (np.abs(x) <= 0.5) & (np.abs(y) <= 0.5)
    a.plot(x[keep], y[keep], color=ORANGE, lw=2)
    a.set_title('Obstacle signal o = 1 − speed (13×13), with one line', fontsize=10.5, color=INK, loc='left')
    a.set_xticks([]); a.set_yticks([])
    fits = {}
    for M in (5, 30):
        fitted, u_s, u_t, f, signal = LF.fit_direction(o, theta, M, h, 'line')
        fits[M] = (fitted[np.argmin(np.abs(u_t - t0))], u_s, signal[np.argmin(np.abs(u_t - t0))])
    b = fig.add_subplot(gs[0, 2:4])
    fit5, u_s, sig = fits[5]
    inside = (np.abs(u_s * math.cos(theta) - t0 * math.sin(theta)) <= 0.5) & (np.abs(u_s * math.sin(theta) + t0 * math.cos(theta)) <= 0.5)
    b.fill_between(u_s[inside], 0, sig[inside], color='#c9d6e8', lw=0, label='signal on the line')
    b.plot(u_s[inside], fit5[inside], color=MUTED, lw=1.3, label='sum of 5 sin / cos pairs')
    b.plot(u_s[inside], fits[30][0][inside], color=ORANGE, lw=1.5, label='sum of 30 pairs')
    b.set_title('The line as a signal: position s, value o', fontsize=10.5, color=INK, loc='left')
    b.legend(frameon=False, fontsize=8.5, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.1))
    b.tick_params(labelsize=8, colors=MUTED)
    for s in ('top', 'right'):
        b.spines[s].set_visible(False)
    t30 = {}
    for deg in (30, 120):
        for M in (30,):
            th = math.radians(deg)
            fitted, us, ut, f, _ = LF.fit_direction(o, th, M, h, 'line')
            t30[deg] = LF.to_table(fitted, th, us, ut, ax0)
    for j, deg in enumerate((30, 120)):
        c = fig.add_subplot(gs[1, 2 + j])
        c.imshow(t30[deg].T, origin='lower', extent=[-0.5, 0.5, -0.5, 0.5], cmap='Blues', vmin=0, vmax=1)
        c.set_title(f'Row at {deg}°, 30 pairs', fontsize=10.5, color=INK, loc='left')
        c.set_xticks([]); c.set_yticks([])
    fig.savefig(HERE / 'report/rows.png', facecolor='white')


if __name__ == '__main__':
    torch.set_num_threads(4)
    if '--rows-only' not in sys.argv:
        curves()
    rows_figure()
