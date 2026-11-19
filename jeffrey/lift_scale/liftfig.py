"""A picture of the lift on a prepared scene: report/lift_visual_<scene key>.png.

    CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/lift_scale/liftfig.py Klickitat

Panels: the speed profile S(d) and the height G(d) it gives; the whole floor plan with the height h at heading 0 and the zoom window;
the zoom window's speed, its height, the surface (x, y, h) the network takes as input; and a cut across one thin obstacle showing the
clearance, the speed and h on both sides for two headings of the robot.
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from report import SCENE_TITLE  # noqa: E402
import lift as LF  # noqa: E402
import scenes as SN  # noqa: E402
from maze import MARGIN_CELLS, OFFSET_CELLS  # noqa: E402


def grid(scene, x0, x1, y0, y1, n, theta=0.0):
    """Clearance (cells) on an n x n grid over [x0, x1] x [y0, y1] (cells) at heading theta (radians)."""
    xs = torch.linspace(x0, x1, n, device=scene.device)
    ys = torch.linspace(y0, y1, n, device=scene.device)
    X, Y = torch.meshgrid(xs, ys, indexing='ij')
    q = scene.from_cells(X.reshape(-1), Y.reshape(-1), torch.full((n * n,), float(theta), device=scene.device))
    return scene.clearance(q).reshape(n, n)


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else 'Klickitat'
    win = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    scenes = json.load(open(HERE / 'queue/scenes.json'))
    scene, info = SN.load_scene(ROOT / scenes[key]['prepared'])
    prof = LF.Profile('time', device=scene.device)
    E = scene.E

    # whole scene at heading 0
    n_all = int(min(12 * E, 1400))
    d_all = grid(scene, 0, E, 0, E, n_all)
    h_all = prof(d_all.reshape(-1))[0].reshape(n_all, n_all)

    # zoom window: the one with the most band area among windows that are 12-45 % obstacle
    c = max(int(round(n_all * win / E)), 8)
    band = ((d_all > 0) & (d_all < MARGIN_CELLS)).float()[None, None]
    obst = (d_all < 0).float()[None, None]
    kb = torch.nn.functional.avg_pool2d(band, c, stride=max(c // 6, 1))[0, 0]
    ko = torch.nn.functional.avg_pool2d(obst, c, stride=max(c // 6, 1))[0, 0]
    score = torch.where((ko > 0.15) & (ko < 0.4), kb, torch.zeros_like(kb))
    idx = int(score.argmax())
    i, j = idx // score.shape[1], idx % score.shape[1]
    st = max(c // 6, 1)
    wx0, wy0 = i * st * E / n_all, j * st * E / n_all
    wx1, wy1 = wx0 + win, wy0 + win
    n = 900
    d = grid(scene, wx0, wx1, wy0, wy1, n)
    S = scene.train_speed(d)
    h = prof(d.reshape(-1))[0].reshape(n, n)

    # a cut along x across one thin obstacle inside the window: free on both sides
    cut = None
    dn = d.cpu().numpy()
    xs = np.linspace(wx0, wx1, n)
    per = n / win
    for jj in np.linspace(0.15 * n, 0.85 * n, 60).astype(int):
        row = dn[:, jj]
        inside = row < 0
        k = 0
        while k < n:
            if inside[k]:
                e = k
                while e < n and inside[e]:
                    e += 1
                width = (e - k) / per
                a, b = k - int(0.45 * per), e + int(0.45 * per)
                if 0.25 <= width <= 1.2 and a >= 0 and b < n and row[a:k].max() > 0.3 and row[e:b].max() > 0.3 and (row[a:k] > 0).all() and (row[e:b] > 0).all():
                    cut = (jj, a, b)
                    break
                k = e
            else:
                k += 1
        if cut:
            break

    fig = plt.figure(figsize=(17.5, 11.2))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.26)

    # 1. the profile
    ax = fig.add_subplot(gs[0, 0])
    dd = np.linspace(-0.03, 0.22, 600)
    Sd = LF.train_speed_np(dd)
    Gd = prof(torch.tensor(dd, device=scene.device, dtype=torch.float32))[0].cpu().numpy()
    ax.plot(dd, Sd, color='#2a78d6', lw=2.2, label='speed S(d)')
    ax.set_xlabel('clearance d (cells)')
    ax.set_ylabel('speed', color='#2a78d6')
    ax.set_ylim(0, 1.05)
    ax2 = ax.twinx()
    ax2.plot(dd, Gd, color='#eb6834', lw=2.2, label='height G(d)')
    ax2.set_ylabel('height G(d) (cells)', color='#eb6834')
    ax2.set_ylim(0, 3.9)
    for xv, lab in ((MARGIN_CELLS, 'margin 0.175'), (OFFSET_CELLS, 'offset 0.0175'), (0.0, 'surface')):
        ax.axvline(xv, color='#888', lw=0.8, ls='--')
        ax.text(xv + 0.002, 0.5, lab, rotation=90, va='center', ha='left', fontsize=8.5, color='#555', transform=ax.get_xaxis_transform())
    ax.set_title('1. From the speed to the height\nG(d) = travel time from clearance d out of the band', fontsize=10.5)
    ax2.annotate('0.685 cells at the offset', (OFFSET_CELLS, 0.685), xytext=(0.075, 1.25), arrowprops=dict(arrowstyle='->', color='#eb6834'), fontsize=9, color='#eb6834')
    ax2.annotate('3.60 cells from −0.0175 down', (-OFFSET_CELLS, 3.6), xytext=(0.03, 3.3), arrowprops=dict(arrowstyle='->', color='#eb6834'), fontsize=9, color='#eb6834')

    # 2. the whole plan
    ax = fig.add_subplot(gs[0, 1])
    show = np.where(d_all.cpu().numpy() < 0, np.nan, h_all.cpu().numpy())
    ax.imshow((d_all.cpu().numpy().T < 0), origin='lower', cmap='Greys', vmin=0, vmax=2.2, extent=[0, E, 0, E], interpolation='nearest')
    im = ax.imshow(np.ma.masked_invalid(np.where(show > 1e-4, show, np.nan)).T, origin='lower', cmap='inferno', vmin=0, vmax=0.7, extent=[0, E, 0, E], interpolation='nearest')
    ax.add_patch(Rectangle((wx0, wy0), win, win, fill=False, ec='#2a78d6', lw=2))
    ax.set_title(f'2. {SCENE_TITLE.get(key, key)}\nh = 0 on the floor (white); it is nonzero only within 0.175 cells\nof an obstacle (grey), too thin to see here. Blue: zoom window', fontsize=10.5)
    ax.set_xlabel('cells')

    # 3. zoom: speed
    ext = [wx0, wx1, wy0, wy1]
    ax = fig.add_subplot(gs[0, 2])
    Sn = S.cpu().numpy()
    im = ax.imshow(np.where(dn < 0, np.nan, Sn).T, origin='lower', cmap='viridis', vmin=0, vmax=1, extent=ext)
    ax.contourf(xs, np.linspace(wy0, wy1, n), (dn.T < 0).astype(float), levels=[0.5, 1.5], colors=['#9a9a9a'])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label='speed')
    ax.set_title(f'3. Zoom, {win:g} × {win:g} cells: the speed\n(1 on the floor, 0.012 at the obstacle surface)', fontsize=10.5)
    if cut:
        ax.plot([xs[cut[1]], xs[cut[2]]], [np.linspace(wy0, wy1, n)[cut[0]]] * 2, color='#d03b3b', lw=2)

    # 4. zoom: height
    ax = fig.add_subplot(gs[1, 0])
    hn = h.cpu().numpy()
    from matplotlib.colors import PowerNorm
    im = ax.imshow(hn.T, origin='lower', cmap='inferno', norm=PowerNorm(0.4, vmin=0, vmax=3.6), extent=ext)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label='h (cells)', ticks=[0, 0.1, 0.3, 0.685, 1.5, 2.5, 3.6])
    ax.contour(xs, np.linspace(wy0, wy1, n), dn.T, levels=[0.0], colors='w', linewidths=0.6)
    ax.set_title('4. Same window: the height h\n(white line: obstacle surface; inside, h stays at 3.60)', fontsize=10.5)
    ax.set_xlabel('cells')
    if cut:
        ax.plot([xs[cut[1]], xs[cut[2]]], [np.linspace(wy0, wy1, n)[cut[0]]] * 2, color='#7fd0ff', lw=2)

    # 5. the surface
    ax = fig.add_subplot(gs[1, 1], projection='3d')
    stp = 2
    Xg, Yg = np.meshgrid(xs[::stp], np.linspace(wy0, wy1, n)[::stp], indexing='ij')
    Z = hn[::stp, ::stp]
    colors = plt.cm.inferno(np.clip(Z / 3.6, 0, 1))
    ax.plot_surface(Xg, Yg, Z, cmap='inferno', norm=PowerNorm(0.4, vmin=0, vmax=3.6), rstride=1, cstride=1, linewidth=0, antialiased=False)
    ax.set_zlim(0, 3.7)
    ax.set_box_aspect((win, win, 3.6))
    ax.view_init(elev=34, azim=-62)
    ax.set_zlabel('h (cells)')
    ax.set_xlabel('x (cells)')
    ax.set_ylabel('y (cells)')
    ax.set_title('5. The surface (x, y, h) the network takes as input, to scale\nflat floor; each obstacle edge is a cliff as long as\nthe travel time it takes to cross the band', fontsize=10.5)

    # 6. the cut
    ax = fig.add_subplot(gs[1, 2])
    if cut:
        jj, a, b = cut
        yc = float(np.linspace(wy0, wy1, n)[jj])
        xc = torch.linspace(float(xs[a]), float(xs[b]), 1200, device=scene.device)
        for th, col, lab in ((0.0, '#eb6834', 'robot along the cut'), (np.pi / 2, '#1baf7a', 'robot across the cut')):
            q = scene.from_cells(xc, torch.full_like(xc, yc), torch.full_like(xc, float(th)))
            dc = scene.clearance(q)
            hc = prof(dc)[0].cpu().numpy()
            ax.plot(xc.cpu().numpy(), hc, color=col, lw=2, label=f'h, {lab}')
            if th == 0.0:
                ax.fill_between(xc.cpu().numpy(), 0, 3.9, where=(dc.cpu().numpy() < 0), color='#bbb', alpha=0.5, lw=0, label='inside the obstacle (heading 0)')
        ax.set_ylim(0, 3.9)
        ax.set_xlabel('x along the red / blue line of panels 3 and 4 (cells)')
        ax.set_ylabel('h (cells)')
        ax.legend(fontsize=8.5, loc='center', bbox_to_anchor=(0.5, 0.55))
        ax.set_title('6. A cut across one thin obstacle (line in panels 3, 4)\nthe same height on both sides; h depends on the\nclearance alone, and on the heading through it', fontsize=10.5)
    out = HERE / 'report' / f'lift_visual_{key}.png'
    fig.savefig(out, dpi=80, bbox_inches='tight')
    print(out, 'window', (round(wx0, 1), round(wy0, 1)), 'cut', cut)


if __name__ == '__main__':
    main()
