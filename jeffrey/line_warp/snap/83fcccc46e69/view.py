"""Pictures of a fitted warp: the maze as the warp sees it, and the step functions along a few lines before and after."""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import obstacles as OB


@torch.no_grad()
def image_polys(phi, maze, theta, spacing=0.05):
    """Images of the obstacle polygons at heading theta (None: walls), as lists of (M, 2) arrays."""
    import maze as MZ
    u = OB.obstacle_union(maze.boxes, theta, MZ.ROBOT_L, MZ.ROBOT_W)
    polys = list(u.geoms) if hasattr(u, 'geoms') else [u]
    out = []
    for p in polys:
        for r in [p.exterior] + list(p.interiors):
            d = OB.densify(np.asarray(r.coords), spacing)
            x = torch.tensor(((d - maze.n / 2) / maze.E).astype(np.float32), device=maze.device)
            out.append(phi(x).cpu().numpy())
    return out


@torch.no_grad()
def warped_grid(phi, maze, step=1, res=8):
    """Images of the cell grid lines (a list of (M, 2) arrays)."""
    t = torch.linspace(-0.5, 0.5, maze.n * res + 1, device=maze.device)
    out = []
    for c in np.arange(0, maze.n + 1, step):
        v = (c - maze.n / 2) / maze.E
        a = torch.stack([t, torch.full_like(t, v)], 1)
        b = torch.stack([torch.full_like(t, v), t], 1)
        out += [phi(a).cpu().numpy(), phi(b).cpu().numpy()]
    return out


def draw_maze(ax, phi, maze, theta=None, grid=True):
    from shapely.geometry import Polygon
    import maze as MZ
    u = OB.obstacle_union(maze.boxes, theta, MZ.ROBOT_L, MZ.ROBOT_W)
    polys = list(u.geoms) if hasattr(u, 'geoms') else [u]
    # obstacles are drawn as filled outer rings, holes painted back in white
    with torch.no_grad():
        for p in polys:
            d = OB.densify(np.asarray(p.exterior.coords), 0.05)
            z = phi(torch.tensor(((d - maze.n / 2) / maze.E).astype(np.float32), device=maze.device)).cpu().numpy()
            ax.fill(z[:, 0], z[:, 1], color='#20313f', lw=0)
            for r in p.interiors:
                d = OB.densify(np.asarray(r.coords), 0.05)
                z = phi(torch.tensor(((d - maze.n / 2) / maze.E).astype(np.float32), device=maze.device)).cpu().numpy()
                ax.fill(z[:, 0], z[:, 1], color='white', lw=0)
    if grid:
        for g in warped_grid(phi, maze):
            ax.plot(g[:, 0], g[:, 1], lw=0.2, color='#7fa3c0', alpha=0.6)
    ax.set_aspect('equal')
    ax.axis('off')


def overview(phi, maze, path, theta=0.0, title=''):
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    ident = lambda x: x
    draw_maze(ax[0], ident, maze, theta)
    ax[0].set_title('maze (x)')
    draw_maze(ax[1], phi, maze, theta)
    ax[1].set_title(f'warped (z) {title}')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
