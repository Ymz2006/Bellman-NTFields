"""Pictures of a map, drawn from the oracle alone: the scene in z (a raster of z-points read through psi), the step functions along a few lines."""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@torch.no_grad()
def raster(psi, scene, k=4, res=640, cut=0.0):
    """(res, res) clearance of the slice k in z. A 3-D scene shows its z_3 = cut slice."""
    dev = scene.table.v.device
    t = torch.linspace(-0.5, 0.5, res, device=dev)
    X, Y = torch.meshgrid(t, t, indexing='ij')
    z = [X.reshape(-1), Y.reshape(-1)] + [torch.full_like(X.reshape(-1), cut)] * (scene.dim - 2)
    z = torch.stack(z, 1)
    x = torch.cat([psi(z[i:i + 262144]) for i in range(0, len(z), 262144)])
    s = scene.sdf(x, torch.full((len(x),), k if scene.planes > 1 else 0, device=dev, dtype=torch.long))
    return s.view(res, res).T.cpu().numpy()


@torch.no_grad()
def steps_of(psi, scene, count=6, k=4, axis=0, samples=1024, seed=5):
    dev = scene.table.v.device
    gen = torch.Generator(dev).manual_seed(seed)
    base = torch.rand(count, scene.dim, generator=gen, device=dev) - 0.5
    t = torch.linspace(-0.5, 0.5, samples, device=dev)
    import scan as S
    z = S.line_points(base, torch.full((count,), axis, device=dev, dtype=torch.long), t)
    x = psi(z.reshape(-1, scene.dim))
    g = (scene.sdf(x, torch.full((len(x),), k if scene.planes > 1 else 0, device=dev, dtype=torch.long)) <= 0).float().view(count, samples)
    return t.cpu().numpy(), g.cpu().numpy(), base.cpu().numpy()


def overview(psi, scene, path, title='', k=None, ident=lambda z: z):
    k = scene.planes // 2 if k is None else k
    fig = plt.figure(figsize=(15, 7.4))
    ax0 = fig.add_axes([0.01, 0.03, 0.31, 0.9])
    ax1 = fig.add_axes([0.33, 0.03, 0.31, 0.9])
    for ax, f, name in ((ax0, ident, 'scene (x)'), (ax1, psi, 'warped (z)')):
        ax.imshow(raster(f, scene, k) > 0, origin='lower', extent=(-.5, .5, -.5, .5), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    for j, a in enumerate(range(2)):
        ax = fig.add_axes([0.665, 0.53 - 0.47 * j, 0.325, 0.4])
        t, g, _ = steps_of(psi, scene, 6, k, a)
        t0, g0, _ = steps_of(ident, scene, 6, k, a)
        for i in range(len(g)):
            ax.fill_between(t, 2.2 * i, 2.2 * i + g[i], step='mid', color='#20313f', lw=0)
            ax.fill_between(t0, 2.2 * i + 1.05, 2.2 * i + 1.05 + 0.5 * g0[i], step='mid', color='#b5c4d0', lw=0)
        ax.set_yticks([])
        ax.set_xlim(-.5, .5)
        ax.set_title(f'steps along z{a + 1}: dark after the warp, light the same line before', fontsize=9)
    fig.suptitle(title, fontsize=11, x=0.02, ha='left')
    fig.savefig(path, dpi=110)
    plt.close(fig)
