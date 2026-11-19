"""Figures of the lift-scale page from report/runs.json (report.py) and the registered run folders (read-only).

    CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/lift_scale/figs.py [scenes|fields|stalls ...]

scene_<scene>.png     the scene (robot-centre obstacle map at heading 0), cells on the axes
fields_<scene>.png    travel time to goal 0 at the goal's heading: truth and the arms in FIELD_ARMS (seed 87358), contours every `step` cells
stalls_<scene>.png    planning seed 0: where the runs that did not arrive stopped, per arm, with the query's goal joined by a line
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
sys.path.insert(0, str(HERE))
from report import SCENES, SCENE_TITLE  # noqa: E402  (before scenes: it puts maze_scale, which has its own report.py, first on the path)
import scenes as SN  # noqa: E402
import diag as DG  # noqa: E402

OUT = HERE / 'report'
ARM_TITLE = dict(ntrl='ntrl-demo', lift='lift', liftpush='lift + push', pushonly='push, no lift', lift_c10='lift + constant strong push', vec='vector lift',
                 vecpush='vector lift + push', lift_r6='lift + push relaxed to 6e-3', liftpush_oct8='lift + push + octaves', liftpush_oct='lift + push + octaves', lift_push='lift + push (abrupt)', push='push (abrupt)')
FIELD_ARMS = ['ntrl', 'lift', 'liftpush', 'liftpush_oct']
_scene_cache = {}


def load():
    return json.load(open(OUT / 'runs.json'))


def scene_of(prep):
    if prep not in _scene_cache:
        _scene_cache[prep] = SN.load_scene(ROOT / prep)
    return _scene_cache[prep]


def obstacle_map(scene, phi=0.0, res=None):
    res = res or (8 * scene.n + 1)
    g = torch.linspace(-0.5, 0.5, res, device=scene.device)
    X, Y = torch.meshgrid(g, g, indexing='ij')
    q = torch.stack([X.reshape(-1), Y.reshape(-1), torch.full((res * res,), float(phi), device=scene.device)], 1)
    return scene.clearance(q).reshape(res, res).cpu().numpy(), q


def pick(runs, scene, label, seed=87358):
    label = {('plus22', 'liftpush_oct'): 'liftpush_oct8'}.get((scene, label), label)
    for r in runs:
        if r['scene'] == scene and r['label'] == label and r['seed'] == seed:
            return r
    return None


def scene_fig(data, s):
    scene, info = scene_of(data['scenes'][s]['prepared'])
    d, _ = obstacle_map(scene, 0.0, 700)
    fig, ax = plt.subplots(figsize=(4.2, 4.4))
    ax.imshow((d.T < 0), origin='lower', cmap='Greys', vmin=0, vmax=1.6, extent=[0, scene.E, 0, scene.E], interpolation='nearest')
    ax.set_title(SCENE_TITLE[s], fontsize=12)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f'scene_{s}.png', dpi=70)
    plt.close(fig)


def fields_fig(data, s, arms=FIELD_ARMS, step=None):
    runs = [(a, pick(data['runs'], s, a)) for a in arms]
    runs = [(a, r) for a, r in runs if r]
    if not runs:
        return
    scene, info = scene_of(data['scenes'][s]['prepared'])
    prep = ROOT / data['scenes'][s]['prepared']
    truth = DG.Truth(scene, prep)
    goal = truth.goals[0]
    d, q = obstacle_map(scene, float(goal[2]))
    r = d.shape[0]
    ts = truth.time(torch.zeros(len(q), dtype=torch.long, device=scene.device), q).reshape(r, r).cpu().numpy() * scene.E
    ts[(d < 0) | ~np.isfinite(ts)] = np.nan
    vmax = float(np.nanmax(ts))
    step = step or (2.0 if vmax < 45 else 4.0 if vmax < 90 else 8.0)
    lev = np.arange(0, vmax * 1.6, step)
    fig, axs = plt.subplots(1, len(runs) + 1, figsize=(4.6 * (len(runs) + 1), 5.0))
    gx, gy, _ = scene.to_cells(goal[None])

    def show(ax, f, title):
        f = np.array(f, dtype=np.float32)
        f[d < 0] = np.nan
        ax.imshow(f.T, origin='lower', extent=[0, scene.E, 0, scene.E], cmap='viridis', vmin=0, vmax=vmax)
        ax.contour(f.T, levels=lev, colors='w', linewidths=0.45, extent=[0, scene.E, 0, scene.E])
        ax.contourf((d.T < 0).astype(float), levels=[0.5, 1.5], colors=['#9a9a9a'], extent=[0, scene.E, 0, scene.E])
        ax.plot(float(gx), float(gy), marker='*', color='#ff3b30', ms=13, mec='w', mew=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()
    show(axs[0], ts, 'truth')
    for ax, (a, rr) in zip(axs[1:], runs):
        f = np.load(ROOT / f'results/registered_{rr["id"]}/slice_goal0.npy').astype(np.float32) * scene.E
        show(ax, f, f"{ARM_TITLE.get(a, a)}: {rr['solved']:.0f} % solved")
    fig.tight_layout()
    fig.savefig(OUT / f'fields_{s}.png', dpi=70)
    plt.close(fig)


def stalls_fig(data, s, arms=('ntrl', 'lift', 'liftpush', 'liftpush_oct')):
    runs = [(a, pick(data['runs'], s, a)) for a in arms]
    runs = [(a, r) for a, r in runs if r]
    if not runs:
        return
    scene, info = scene_of(data['scenes'][s]['prepared'])
    prep = ROOT / data['scenes'][s]['prepared']
    Q = np.load(prep / 'queries.npz')
    d, _ = obstacle_map(scene, 0.0, 900)
    fig, axs = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 5.5))
    axs = np.atleast_1d(axs)
    for ax, (a, rr) in zip(axs, runs):
        ax.imshow((d.T < 0), origin='lower', cmap='Greys', vmin=0, vmax=2.2, extent=[0, scene.E, 0, scene.E], interpolation='nearest')
        f = ROOT / f'results/registered_{rr["id"]}/stalls.npz'
        n = 0
        if f.exists():
            z = np.load(f)
            end = z['end']
            ex, ey = end[:, 0] * scene.E + scene.n / 2, end[:, 1] * scene.E + scene.n / 2
            g = Q['goal'][z['query']]
            gx, gy = g[:, 0] * scene.E + scene.n / 2, g[:, 1] * scene.E + scene.n / 2
            for i in range(len(ex)):
                ax.plot([ex[i], gx[i]], [ey[i], gy[i]], color='#2a78d6', lw=0.5, alpha=0.5)
            ax.scatter(gx, gy, s=10, color='#2a78d6', zorder=3)
            ax.scatter(ex, ey, s=16, color='#d03b3b', zorder=4)
            n = len(ex)
        ax.set_title(f"{ARM_TITLE.get(a, a)}: {n} of {len(Q['bin'])} runs stopped short", fontsize=11)
        ax.set_xlim(0, scene.E)
        ax.set_ylim(0, scene.E)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(OUT / f'stalls_{s}.png', dpi=70)
    plt.close(fig)


def main():
    """No argument: one subprocess per scene (the plus-maze variant can be installed once per process). --scene <key> [what ...]: that scene."""
    import subprocess
    data = load()
    if '--scene' in sys.argv:
        k = sys.argv.index('--scene')
        s = sys.argv[k + 1]
        what = sys.argv[k + 2:] or ['scene', 'fields', 'stalls']
        if 'scene' in what:
            scene_fig(data, s)
        if 'fields' in what:
            fields_fig(data, s)
        if 'stalls' in what:
            stalls_fig(data, s)
        return
    for s in SCENES:
        if s in data['scenes'] and 'prepared' in data['scenes'][s]:
            subprocess.run([sys.executable, __file__, '--scene', s] + sys.argv[1:], check=False)


if __name__ == '__main__':
    main()
