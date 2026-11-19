"""Figures of the findings page: the same scene through each method, drawn from the saved maps by the oracle alone.

    python figs.py strip10 | strip12 | scenes        (reads report/runs.json: label -> run directory)
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import maps as M
import remeasure as RM
import scene as SC
import view as V

HERE = Path(__file__).resolve().parent
OUT = HERE / 'report'
RUNS = json.load(open(OUT / 'runs.json'))
dev = 'cuda:0'
INK, SLOW = '#20313f', '#c2703d'


def rid(label):
    return RUNS[label].split('registered_')[1]


def panel(ax, psi, scene, title, k=None, res=560, cut=0.0):
    """Free space white, obstacle dark; the slow zone next to walls tinted, so a crushed zone shows."""
    k = scene.planes // 2 if k is None else k
    s = V.raster(psi, scene, k, res, cut)
    rgb = np.ones((*s.shape, 3))
    slow = (s > 0) & (s < scene.margin)
    rgb[slow] = (0.96, 0.82, 0.70)
    rgb[s <= 0] = (0.125, 0.19, 0.25)
    ax.imshow(rgb, origin='lower', extent=(-.5, .5, -.5, .5), interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color('#c9ced6')
    ax.set_title(title, fontsize=9.5, color='#15181d', pad=4)


def strip(items, scene, path, cols=None, size=2.5, titles_only=False):
    cols = cols or len(items)
    rows = (len(items) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(size * cols, (size + 0.32) * rows), squeeze=False)
    for ax in axs.ravel():
        ax.axis('off')
    for ax, (title, psi) in zip(axs.ravel(), items):
        ax.axis('on')
        panel(ax, psi, scene, title)
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def strip10():
    sc = SC.build('plain:10', dev, 9, 32)
    L = [('no warp', M.Identity()), ('every run equal, per line', RM.load(rid('p10_coupling_cvline'), dev)), ('fixed target, along axes', RM.load(rid('p10_coupling_cv_t10_k16'), dev)), ('fixed target, every angle', RM.load(rid('p10_any_coupling_cv_t10'), dev)),
         ('old loss with shear', RM.load(rid('p10c_hard'), dev)), ('line loss with limits', RM.load(rid('p10k_k8'), dev)), ('density form', RM.load(rid('carto_p10_h2_e0.3'), dev)), ('learned density form', RM.load(rid('nf4_p10_w1'), dev))]
    strip(L, sc, OUT / 'strip10.png', cols=4)


def strip12():
    sc = SC.build('gap:maze10:12', dev, 9, 16)
    sys.path.insert(0, str(SC.ROOT / 'jeffrey/line_warp'))
    import netwarp as NW
    import speedcal as CAL
    state, cfg, _ = NW.load_fit(SC.ROOT / 'results' / f"registered_{json.load(open(SC.ROOT / 'jeffrey/line_warp/report/runs.json'))['m10k12d']}", dev)
    w = NW.NetWarp(dict(net=cfg, s=1.0)).to(dev)
    w.set_state(state, sc.maze)
    L = [('no warp', M.Identity()), ('every run equal, fixed target', RM.load(rid('g12b_coupling_axis_cv'), dev)), ('old loss with shear', RM.load(rid('g12c_hard'), dev)), ('equal runs, stiff near walls', RM.load(rid('g12g_eq_near'), dev)),
         ("last study's recipe (separable)", CAL.Inverse(w)), ('line loss with limits', RM.load(rid('g12k_k8'), dev)), ('density form', RM.load(rid('carto_g12_h2_e0.3'), dev)), ('learned density form', RM.load(rid('nf4_g12_w1'), dev))]
    strip(L, sc, OUT / 'strip12.png', cols=4)


def scenes():
    """Scene and its warp, side by side, for every kind of scene: one row per scene in a 2 x 2 grid of pairs."""
    jobs = [('rot30/gap:maze10:12', 16, 'turned maze, ×12', 'carto_r30g12_h2_e0.3'), ('gibson:1:2', 32, 'scanned apartment', 'carto_gib1_h2_e0.3'), ('gibson:0:2', 32, 'scanned floor', 'carto_gib0_h2_e0.3'), ('se2/gap:maze10:12', 6, 'C-space (x, y, θ), ×12, slice θ = 0', 'carto_se2g12_h2_e0.3')]
    fig, axs = plt.subplots(2, 4, figsize=(11, 6.1))
    for (spec, res, name, label), ax in zip(jobs, axs.reshape(-1, 2)):
        sc = SC.build(spec, dev, 9, res)
        panel(ax[0], M.Identity(), sc, name + ': x', res=480)
        panel(ax[1], RM.load(rid(label), dev), sc, 'z', res=480)
    fig.tight_layout(pad=0.5, h_pad=1.8)
    fig.savefig(OUT / 'scenes.png', dpi=130)
    plt.close(fig)


def density():
    """The idea in three panels: the scene, how its boundary is spread over the box, and the map that spreads it evenly."""
    import carto as C
    sc = SC.build('gap:maze10:12', dev, 9, 16)
    H = C.boundary_density(sc, 512)
    D = C.Diffusion(H)
    hb = 2.0 / sc.E
    B = D.blurred(0.5 * hb * hb).clamp(min=0)
    rho = (0.3 + B / B.mean()).cpu().numpy()
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.4))
    panel(ax[0], M.Identity(), sc, 'the scene: ×12 maze', res=640)
    im = ax[1].imshow(np.log10(rho).T, origin='lower', extent=(-.5, .5, -.5, .5), cmap='Oranges', interpolation='nearest')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title('blurred boundary density (log scale)', fontsize=9.5, color='#15181d', pad=4)
    panel(ax[2], RM.load(rid('carto_g12_h2_e0.3'), dev), sc, 'the cartogram: density spread evenly', res=640)
    fig.tight_layout(pad=0.5)
    fig.savefig(OUT / 'density.png', dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    for name in sys.argv[1:]:
        globals()[name]()
