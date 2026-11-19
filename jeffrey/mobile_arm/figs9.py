"""Figures of the mobile-arm study: the robot (top and side view, tucked and reaching poses) and a house with query configurations.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/mobile_arm/figs9.py --prepared results/registered_<prepare> --out jeffrey/mobile_arm/report
"""
import argparse
import math
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scene9 as S9  # noqa: E402
import robot as RB  # noqa: E402

LINK_COL = ['#4c6ef5', '#12b886', '#fab005', '#e8590c', '#be4bdb', '#868e96']


def robot_figure(out, dev='cuda:0'):
    arm = RB.Arm(dev, torch.float64)
    poses = [('tucked (TUCK)', S9.TUCK), ('reaching sideways', (0.3, -0.35, 0.6, -1.2, 1.4, 0.0)), ('reaching low and far', (2.692, -0.017, 0.329, -1.924, -0.982, 2.796))]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6))
    for k, (name, q) in enumerate(poses):
        qt = torch.tensor([q], device=dev, dtype=torch.float64)
        p = arm.spheres(qt)[0].cpu().numpy()
        r = arm.radius.cpu().numpy()
        link = arm.link.cpu().numpy()
        d, fl, ba, se = (float(x) for x in arm.internal(qt, parts=True))
        for row, (i, j, lab) in enumerate(((0, 1, 'y'), (0, 2, 'z'))):
            ax = axes[row, k]
            if row == 0:
                ax.add_patch(Polygon([[-RB.BASE_L / 2, -RB.BASE_W / 2], [RB.BASE_L / 2, -RB.BASE_W / 2], [RB.BASE_L / 2, RB.BASE_W / 2], [-RB.BASE_L / 2, RB.BASE_W / 2]],
                                     closed=True, fc='#dee2e6', ec='#495057', lw=1.2))
            else:
                ax.add_patch(Polygon([[-RB.BASE_L / 2, 0], [RB.BASE_L / 2, 0], [RB.BASE_L / 2, RB.BASE_H], [-RB.BASE_L / 2, RB.BASE_H]], closed=True, fc='#dee2e6', ec='#495057', lw=1.2))
                ax.axhline(0, color='#495057', lw=0.8)
            order = np.argsort(p[:, 2] if row == 0 else p[:, 1])
            for s in order:
                ax.add_patch(Circle((p[s, i], p[s, j]), r[s], fc=LINK_COL[link[s]], ec='k', lw=0.3, alpha=0.75))
            ax.set_aspect('equal')
            ax.set_xlim(-0.9, 0.9)
            ax.set_ylim(-0.9, 0.9) if row == 0 else ax.set_ylim(-0.05, 1.1)
            ax.set_xlabel('x (cells, along the heading)')
            ax.set_ylabel(f'{lab} (cells)')
            ax.grid(alpha=0.25)
            if row == 0:
                ax.set_title(f'{name}\ninternal clearance {d:.3f} cells', fontsize=10)
    handles = [Circle((0, 0), 1, fc=c, ec='k', lw=0.3) for c in LINK_COL]
    fig.legend(handles, RB.LINKS, loc='lower center', ncol=6, fontsize=8, frameon=False)
    fig.suptitle('9-DOF mobile manipulator: 0.5 x 0.2-cell base (12.5 x 5 cm), UR5e at 1/5 size (0.8 cells per UR5e metre); top: plan view, bottom: side view', fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out / 'robot.png', dpi=130)
    plt.close(fig)


def house_figure(prepared, out, dev='cuda:0', count=6):
    scene, info = S9.load_scene(prepared, dev)
    z = np.load(prepared / 'queries.npz')
    house = scene.house
    res = 1024
    t = torch.linspace(-0.5, 0.5, res, device=dev)
    X, Y = torch.meshgrid(t, t, indexing='ij')
    occ = (house.point_sdf_cells(torch.stack([X.reshape(-1), Y.reshape(-1)], 1)).reshape(res, res) < 0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 7))
    E, n = scene.E, scene.n
    ax.imshow(occ.T, origin='lower', cmap='Greys', extent=[0, E, 0, E], alpha=0.85)
    bins = z['bin']
    pick = [int(np.nonzero(bins == b)[0][0]) for b in sorted(set(bins.tolist()))][:count]
    ref = np.load(prepared / 'ref_paths.npy', allow_pickle=True)
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(pick)))
    for c, i in zip(cols, pick):
        for key in ('start', 'goal'):
            q = torch.tensor(z[key][i:i + 1], device=dev)
            x, y, th, qj = scene.to_cells(q)
            sp = scene.world_spheres(q)[0].cpu().numpy()
            rr = scene.arm.radius.cpu().numpy()
            cth, sth = math.cos(float(th)), math.sin(float(th))
            corners = [(float(x) + a * RB.BASE_L / 2 * cth - b * RB.BASE_W / 2 * sth, float(y) + a * RB.BASE_L / 2 * sth + b * RB.BASE_W / 2 * cth)
                       for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
            ax.add_patch(Polygon(corners, closed=True, fc=c, ec='k', lw=0.4))
            for s in range(len(rr)):
                ax.add_patch(Circle((sp[s, 0], sp[s, 1]), rr[s], fc=c, ec='none', alpha=0.35))
        w = ref[i]
        ax.plot(w[:, 0] * E + n / 2, w[:, 1] * E + n / 2, color=c, lw=1.2, label=f'{z["d_cells"][i]:.1f} cells')
    ax.set_xlim(0, E)
    ax.set_ylim(0, E)
    ax.set_aspect('equal')
    ax.legend(title='reference path (base distance)', fontsize=8, loc='upper right')
    ax.set_title(f'{info["scene"]["house"]["name"]}: one query per distance bin (start and goal configurations, reference path)', fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f'house_{info["scene"]["house"]["name"]}.png', dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, nargs='*', default=[])
    p.add_argument('--out', type=Path, default=HERE / 'report')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    robot_figure(a.out)
    for prep in a.prepared:
        house_figure(prep, a.out)


if __name__ == '__main__':
    main()
