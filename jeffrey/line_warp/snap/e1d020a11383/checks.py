"""Measurements of a fitted warp: is it a diffeomorphism, how far did it equalize the runs, what did it do to each region.

* folds: det J on a dense grid of the box (sign and smallest value), and the image of the box outline must be a simple curve;
  together with det J > 0 this makes the map injective on the box (a local diffeomorphism whose boundary curve does not cross
  itself covers its image once).
* runs: the run lengths along lines, at the identity and after the warp, on all planes: max / min per line, and the free runs and
  obstacle runs in the warped space.
* regions: singular values and rotation inside the quadrants, the open band, and near / far from walls.
"""
import math
import numpy as np
import torch
from shapely.geometry import LinearRing

import lineloss as LL
import obstacles as OB
import warpnet as WN


@torch.no_grad()
def grid_jacobian(plane, maze, res=None, eps=None):
    """J of a plane map on a res x res grid of the box: positions (N, 2) and J (N, 2, 2)."""
    res = res or 16 * maze.n
    eps = eps or 0.02 / maze.E
    t = torch.linspace(-0.5 + 1e-3, 0.5 - 1e-3, res, device=maze.device)
    X, Y = torch.meshgrid(t, t, indexing='ij')
    x = torch.stack([X.reshape(-1), Y.reshape(-1)], 1)
    J = torch.cat([WN.jacobian_fd(plane, x[i:i + 65536], eps) for i in range(0, len(x), 65536)])
    return x, J


@torch.no_grad()
def outline_simple(plane, maze):
    t = torch.linspace(-0.5, 0.5, 64 * maze.n + 1, device=maze.device)
    o = lambda v: torch.full_like(t, v)
    edge = torch.cat([torch.stack([t, o(-0.5)], 1), torch.stack([o(0.5), t], 1), torch.stack([t.flip(0), o(0.5)], 1), torch.stack([o(-0.5), t.flip(0)], 1)])
    return bool(LinearRing(plane(edge).cpu().numpy().astype(np.float64)).is_simple)


@torch.no_grad()
def run_stats(plane, maze, mode='cspace', headings=9, lines=256, floor=0.1, seed=0, spacing=0.1):
    """Line statistics along the axes on every plane: per-line max / min of the runs, and quantiles of all free and obstacle runs in
    units of the median run."""
    dev = maze.device
    gen = torch.Generator(dev).manual_seed(seed)
    ratio, count, free, obst = [], [], [], []
    for b, _ in OB.planes(maze, mode, headings, spacing):
        Z = plane(torch.tensor(b.points, device=dev))
        S = torch.tensor(b.segments, device=dev)
        for axis in (0, 1):
            p = 1 - axis
            c = Z[:, p].min() + (Z[:, p].max() - Z[:, p].min()) * torch.rand(lines, generator=gen, device=dev)
            line, t = LL.crossings(Z, S, c, axis)
            ell, mask, n = LL.runs(line, t, lines)
            _, st = LL.per_line(ell, mask, 'hard', floor)
            ratio.append(st['ratio'])
            count.append(st['runs'])
            kind = torch.arange(ell.shape[1], device=dev)[None] % 2 == 0          # runs alternate free, obstacle, free, ...
            free.append(ell[mask & kind])
            obst.append(ell[mask & ~kind])
    ratio, free, obst = torch.cat(ratio), torch.cat(free), torch.cat(obst)
    med = torch.cat([free, obst]).median()
    q = lambda v, a: float(v.quantile(a))
    return dict(ratio_median=q(ratio, 0.5), ratio_q90=q(ratio, 0.9), runs_per_line=float(torch.cat(count).mean()), free_q10_50_90_99=[q(free / med, a) for a in (0.1, 0.5, 0.9, 0.99)],
                obstacle_q10_50_90_99=[q(obst / med, a) for a in (0.1, 0.5, 0.9, 0.99)], log_std=float(torch.log(torch.cat([free, obst]).clamp(min=1e-9)).std()))


def rotation_deg(J):
    return torch.rad2deg(torch.atan2(J[:, 1, 0] - J[:, 0, 1], J[:, 0, 0] + J[:, 1, 1]))


@torch.no_grad()
def measure(warp, maze, block=None):
    """Everything above for a NetWarp (its plane map, with s and the box normalization); in the warped units of s = 1."""
    plane = lambda x: warp.plane(x) / warp.s
    x, J = grid_jacobian(plane, maze)
    s1, s2, det = WN.singular_values(J)
    sign = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    c = x * maze.E + maze.n / 2
    out = dict(det_positive=float((sign > 0).float().mean()), det_min=float(sign.min()), outline_simple=outline_simple(plane, maze), sv_q01_q99=[float(s2.quantile(0.01)), float(s1.quantile(0.99))],
               sv_min_max=[float(s2.min()), float(s1.max())], anisotropy_median=float((s1 / s2.clamp(min=1e-9)).median()), anisotropy_q99=float((s1 / s2.clamp(min=1e-9)).quantile(0.99)),
               rotation_abs_median_deg=float(rotation_deg(J).abs().median()), runs=run_stats(plane, maze))
    width = maze.describe().get('plus_width', 0)
    if width:
        blk = (maze.n - width) / 2
        inq = lambda v: (v < blk) | (v >= maze.n - blk)
        quad = inq(c[:, 0]) & inq(c[:, 1])
        cen = ~inq(c[:, 0]) & ~inq(c[:, 1])
        arm = ~quad & ~cen
        area = (s1 * s2)
        out['area_stretch'] = dict(quadrants=float(area[quad].median()), arms=float(area[arm].median()), centre=float(area[cen].median()))
        out['stretch_sqrt_area_quadrants'] = float(area[quad].median().sqrt())
    return out
