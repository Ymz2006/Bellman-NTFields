"""Field pictures of the pair-lift page (read-only on registered runs): travel time to goal 0 at the goal's heading, the truth next to the
fields of panels(), for the prepare run's goal deepest inside a maze quadrant; contours every 2 cells (1 cell in the zoom).   CUDA_VISIBLE_DEVICES=0 ~/.venvs/bellman-ntfield/bin/python jeffrey/pair_lift/figs.py

Writes report/fields_plus22.png and report/fields_plus22_zoom.png (the same fields around the goal's neighbourhood).
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
sys.path.insert(0, str(HERE))
import pairnet as PN  # noqa: E402
import scenes as SN  # noqa: E402
import diag as DG  # noqa: E402
ROOT = PN.ROOT
OUT = HERE / 'report'
PREP = ROOT / 'results/registered_20261102T042454Z-62864e61'


def panels():
    runs = json.load(open(OUT / 'runs.json'))['runs']

    def pick(label, kind=None):
        for r in runs:
            if r['label'] == label and r['scene'] == 'plus22' and r['seed'] == 87358 and (kind is None or r['kind'] == kind):
                return r
    out = [('lift + push', pick('lift + push')), ('lift + blocked bit + push + cap', pick('bit_push_tri')), ('lift + Bellman backup over free steps', pick('bxt1'))]
    return [(t, r) for t, r in out if r]


def main():
    scene, info = SN.load_scene(PREP)
    truth = DG.Truth(scene, PREP)
    gc = torch.stack(scene.to_cells(truth.goals)[:2], 1)
    # the goal deepest inside a maze quadrant (farthest from the open plus in the middle)
    gi = int(torch.argmax((gc - scene.n / 2).abs().amin(1)))
    goal = truth.goals[gi]
    import plan as PL
    res = 8 * scene.n + 1
    g = torch.linspace(-0.5, 0.5, res, device=scene.device)
    X, Y = torch.meshgrid(g, g, indexing='ij')
    q = torch.stack([X.reshape(-1), Y.reshape(-1), torch.full((res * res,), float(goal[2]), device=scene.device)], 1)
    d = scene.clearance(q).reshape(res, res).cpu().numpy()
    ts = truth.time(torch.full((len(q),), gi, dtype=torch.long, device=scene.device), q).reshape(res, res).cpu().numpy() * scene.E
    ts[(d < 0) | ~np.isfinite(ts)] = np.nan
    vmax = float(np.nanmax(ts))
    lev = np.arange(0, vmax * 1.6, 2.0)
    gx, gy, _ = scene.to_cells(goal[None])
    gx, gy = float(gx), float(gy)
    P = panels()
    for name, window in (('fields_plus22', None), ('fields_plus22_zoom', 4.0)):
        fig, axs = plt.subplots(1, len(P) + 1, figsize=(4.4 * (len(P) + 1), 4.9))

        def show(ax, f, title):
            f = np.array(f, dtype=np.float32)
            f[d < 0] = np.nan
            ax.imshow(f.T, origin='lower', extent=[0, scene.E, 0, scene.E], cmap='viridis', vmin=0, vmax=vmax if window is None else 16)
            ax.contour(f.T, levels=lev if window is None else np.arange(0, 60, 1.0), colors='w', linewidths=0.45, extent=[0, scene.E, 0, scene.E])
            ax.contourf((d.T < 0).astype(float), levels=[0.5, 1.5], colors=['#9a9a9a'], extent=[0, scene.E, 0, scene.E])
            ax.plot(gx, gy, marker='*', color='#ff3b30', ms=13, mec='w', mew=0.6)
            ax.set_title(title, fontsize=11)
            ax.set_axis_off()
            if window is not None:
                ax.set_xlim(max(gx - window, 0), min(gx + window, scene.E))
                ax.set_ylim(max(gy - window, 0), min(gy + window, scene.E))
        show(axs[0], ts, 'truth')
        for ax, (title, r) in zip(axs[1:], P):
            net = PN.load_field(ROOT / f'results/registered_{r["id"]}/training/checkpoint.pt', scene)
            f = PL.field_cost(net)(torch.cat([q, goal[None].expand(len(q), 3)], 1)).reshape(res, res).cpu().numpy() * scene.E
            show(ax, f, f"{title}: {r['solved']:.1f} % solved")
        fig.tight_layout()
        fig.savefig(OUT / f'{name}.png', dpi=70)
        plt.close(fig)
        print(OUT / f'{name}.png')


if __name__ == '__main__':
    main()
