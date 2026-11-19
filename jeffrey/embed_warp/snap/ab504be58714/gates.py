"""Fixed gates w(x) in [0, 1] on a C-space grid (features.GridGate tables), 'trained separately' from the travel-time network.

    geo    from the collision oracle: w = 1 where the robot's clearance is at most r0 cells (and inside obstacles), 0 from r1 cells on,
           a smoothstep between. The band sampling's own quantity (the labels are a function of the same clearance).
    kde    from the band samples alone: the training pairs' x0 are ntrl-demo's band (clearance between offset and margin). Their density on
           the grid, Gaussian-blurred by sigma cells, normalized by its q-quantile over the band's own cells and clipped to [0, 1]. Needs
           no geometry beyond the sampled points.
    diff   a physical analogue: the obstacle boundary is a heat source held at 1, obstacles are insulators, and heat spreads through free
           space for `steps` masked Jacobi sweeps of the 3 x 3 x 3 box average (jeffrey/gibson_embed/diffuse.py's operator) on the grid;
           w = clip(gain x temperature, 0, 1). It decays with the free-space distance from the walls (not the straight-line distance), so
           a pose that is close to a wall only through another wall is not heated.
    curv   from a trained field: at every grid node the Hessian of T(q, g) with respect to q (autograd, Frobenius norm times the cell
           pitch; 'hess') or the eikonal misfit |S |grad T| - 1| ('resid'), median over goals, blurred, divided by its q-quantile over
           free nodes and clipped to [0, 1]: resolution goes where the previous field bends or fails.
Each octave gets the same table unless per_octave tightens the radius for finer octaves (geo only).
"""
import math
import numpy as np
import torch
import torch.nn.functional as Fn


def axes(scene, res_cells=4, n_theta=13):
    nx = int(round(scene.E * res_cells)) + 1
    xs = torch.linspace(-0.5, 0.5, nx, device=scene.device)
    ts = torch.linspace(-scene.phi_max, scene.phi_max, n_theta, device=scene.device)
    return xs, ts


def grid_points(scene, xs, ts):
    q = torch.stack(torch.meshgrid(xs, xs, ts, indexing='ij'), -1).reshape(-1, 3)
    return q, (len(xs), len(xs), len(ts))


def smoothstep(t):
    t = t.clamp(0, 1)
    return t * t * (3 - 2 * t)


def blur(vol, sigma_cells, res_cells, sigma_theta=1.0):
    """Gaussian blur of a (nx, ny, nt) volume: sigma_cells cells in x, y (res_cells nodes per cell), sigma_theta nodes in theta; replicate pad."""
    v = vol[None, None]
    for dim, sig in ((2, sigma_cells * res_cells), (3, sigma_cells * res_cells), (4, sigma_theta)):
        if sig <= 0:
            continue
        r = max(1, int(math.ceil(3 * sig)))
        k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=vol.device, dtype=vol.dtype) / sig) ** 2)
        k = k / k.sum()
        shape = [1, 1, 1, 1, 1]
        shape[dim] = len(k)
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (4 - dim)] = pad[2 * (4 - dim) + 1] = r
        v = Fn.conv3d(Fn.pad(v, pad, mode='replicate'), k.view(shape))
    return v[0, 0]


def geo_table(scene, q, shape, r0, r1, K, per_octave=False):
    d = scene.clearance(q).reshape(shape)
    tabs = []
    for k in range(K):
        f = 0.5 ** k if per_octave else 1.0
        tabs.append(smoothstep((r1 * f - d) / max(r1 * f - r0 * f, 1e-6)))
    return torch.stack(tabs), dict(band_share=float((d <= r0).float().mean()), free_share=float((d >= 0).float().mean()))


def kde_table(scene, band, shape, xs, ts, sigma_cells, res_cells, q_norm, K):
    """band: (N, 3) near-wall configurations (the training x0)."""
    nx, ny, nt = shape
    c = torch.stack([(band[:, 0] + 0.5) * (nx - 1), (band[:, 1] + 0.5) * (ny - 1), (band[:, 2] + scene.phi_max) / (2 * scene.phi_max) * (nt - 1)], 1)
    i = c.round().long()
    i = torch.stack([i[:, 0].clamp(0, nx - 1), i[:, 1].clamp(0, ny - 1), i[:, 2].clamp(0, nt - 1)], 1)
    hist = torch.zeros(nx * ny * nt, device=band.device)
    hist.index_add_(0, (i[:, 0] * ny + i[:, 1]) * nt + i[:, 2], torch.ones(len(i), device=band.device))
    hist = hist.view(nx, ny, nt)
    dens = blur(hist, sigma_cells, res_cells)
    occupied = hist > 0
    ref = float(dens[occupied].quantile(q_norm)) if bool(occupied.any()) else float(dens.max())
    w = (dens / max(ref, 1e-12)).clamp(0, 1)
    return w[None].expand(K, -1, -1, -1).contiguous(), dict(points=int(len(band)), ref=ref, open_share=float((w < 0.05).float().mean()))


def diff_table(scene, q, shape, r_src, steps, gain, K):
    d = scene.clearance(q).reshape(shape)
    free = (d >= 0)
    src = (d <= r_src)                                  # source: band and obstacle interior (held at 1)
    w = src.float()
    m = (free | src).float()                              # cells that conduct: free space and the source
    k = torch.ones(1, 1, 3, 3, 3, device=q.device)
    for _ in range(steps):
        num = Fn.conv3d(Fn.pad((w * m)[None, None], [1] * 6), k)[0, 0]
        den = Fn.conv3d(Fn.pad(m[None, None], [1] * 6), k)[0, 0].clamp(min=1)
        w = torch.where(src, torch.ones_like(w), (num / den) * m)
    w = (gain * w).clamp(0, 1)
    return w[None].expand(K, -1, -1, -1).contiguous(), dict(steps=steps, mean_free=float(w[free].mean()), open_share=float((w[free] < 0.05).float().mean()))


def curv_table(net, scene, q, shape, goals, kind, q_norm, blur_cells, res_cells, K):
    meas = []
    for g in goals:
        rows = []
        for i in range(0, len(q), 8192):
            x = q[i:i + 8192]
            with torch.enable_grad():
                tau, _, Xp = net.out(torch.cat([x, g[None].expand(len(x), 3)], 1))      # Xp: the leaf copy of the pairs (as Function.Loss uses it)
                gr, = torch.autograd.grad(tau[:, 0].sum(), Xp, create_graph=True)
                if kind == 'hess':
                    H = torch.stack([torch.autograd.grad(gr[:, j].sum(), Xp, retain_graph=True)[0][:, :3] for j in range(3)], 1)
                    rows.append(torch.linalg.matrix_norm(H.detach()) * scene.pitch)
                else:
                    S = scene.train_speed(scene.clearance(x))
                    rows.append((S * torch.linalg.vector_norm(gr[:, :3].detach(), dim=1) - 1).abs())
        meas.append(torch.cat(rows))
    c = torch.stack(meas).median(0).values.reshape(shape)
    c = blur(c, blur_cells, res_cells) if blur_cells > 0 else c
    d = scene.clearance(q).reshape(shape)
    free = d >= 0
    ref = float(c[free].quantile(q_norm))
    w = (c / max(ref, 1e-12)).clamp(0, 1)
    return w[None].expand(K, -1, -1, -1).contiguous(), dict(kind=kind, ref=ref, mean_free=float(w[free].mean()), open_share=float((w[free] < 0.05).float().mean()),
                                                         corr_with_band=float(torch.corrcoef(torch.stack([w[free], (d[free] < 0.5).float()]))[0, 1]))


@torch.no_grad()
def table_stats(scene, table, q, shape):
    """Mean gate value by clearance band (cells) for the first octave's table."""
    d = scene.clearance(q).reshape(shape)
    w = table[0]
    out = {}
    for lo, hi in [(-1e9, 0), (0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 1e9)]:
        m = (d >= lo) & (d < hi)
        if int(m.sum()):
            out[f'{lo:g}-{hi:g}' if abs(lo) < 1e8 and hi < 1e8 else ('<0' if lo < -1e8 else f'>{lo:g}')] = float(w[m].mean())
    return out
