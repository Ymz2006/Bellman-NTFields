"""ntrl-demo training pairs in the 9-D C-space: jeffrey/maze_scale/data.py's recipe (ntrl-demo dataprocessing/speed_sampling_gpu_kdtree_normal.py
in the robot's C-space) with the box of scene9.MobileArm.

ntrl-demo's recipe assumes the unit cube [-0.5, 0.5]^dim; it is applied in the unit cube the C-space box maps to (every coordinate scaled
to unit width about the box centre, as maze_scale does for phi): v0 uniform in the cube, v1 = v0 + unit(rand - 0.5) * U(0, r_max) with
r_max = sqrt(dim) = 3 (ntrl-demo's), both inside the cube; x0, x1 = v0, v1 mapped back. Keep offset < d(x0) < margin and d(x1) > -offset
(d = scene9's clearance, cells); speed = clip(d / margin, offset / margin, 1); normal = unit C-space gradient of d. Files:
sampled_points.npy (N, 18), speed.npy (N, 2), normal.npy (N, 18) (ntrl-demo models/data_mlp.py layout).
"""
import math
import numpy as np
import torch
from maze import MARGIN_CELLS, OFFSET_CELLS
from scene9 import K_INT


def sample(scene, n, seed, r_max=None, batch=1_000_000):
    dev = scene.device
    D = len(scene.lo)
    r_max = math.sqrt(D) if r_max is None else r_max
    gen = torch.Generator(dev).manual_seed(seed)
    width = scene.hi - scene.lo
    centre = (scene.hi + scene.lo) / 2
    xs, ys, ns, which = [], [], [], []
    drawn, total = 0, 0
    while total < n:
        v0 = torch.rand(batch, D, generator=gen, device=dev) - 0.5
        dp = torch.rand(batch, D, generator=gen, device=dev) - 0.5
        rl = torch.rand(batch, 1, generator=gen, device=dev) * r_max
        v1 = v0 + torch.nn.functional.normalize(dp, dim=1) * rl
        drawn += batch
        keep = (v1.abs() <= 0.5).all(1)
        p, q = centre + v0[keep] * width, centre + v1[keep] * width
        d0 = scene.clearance(p)
        keep = (d0 < MARGIN_CELLS) & (d0 > OFFSET_CELLS)
        p, q = p[keep], q[keep]
        d0, n0 = scene.clearance_normal(p)
        d1, n1 = scene.clearance_normal(q)
        keep = (d1 > -OFFSET_CELLS) & ~n0.isnan().any(1) & ~n1.isnan().any(1)
        xs.append(torch.cat([p, q], 1)[keep].cpu())
        ys.append(torch.stack([d0, d1], 1)[keep].cpu())
        ns.append(torch.cat([n0, n1], 1)[keep].cpu())
        pt = scene.parts(p[keep])
        which.append(torch.stack([pt['rect'], pt['arm'], K_INT * pt['int']], 1).argmin(1).cpu())
        total += int(keep.sum())
    x = torch.cat(xs)[:n].numpy().astype(np.float32)
    d = torch.cat(ys)[:n].numpy()
    normal = torch.cat(ns)[:n].numpy().astype(np.float32)
    act = torch.cat(which)[:n].numpy()
    speed = np.clip(d / MARGIN_CELLS, OFFSET_CELLS / MARGIN_CELLS, 1).astype(np.float32)
    gap_cells = np.linalg.norm(x[:, :2] - x[:, D:D + 2], axis=1) / scene.pitch
    gap_c = np.linalg.norm(x[:, :D] - x[:, D:], axis=1) / scene.pitch
    stats = dict(n=len(x), drawn=drawn, r_max=r_max, x1_blocked=float((d[:, 1] <= OFFSET_CELLS).mean()), x1_free=float((d[:, 1] >= MARGIN_CELLS).mean()),
                 gap_xy_cells_quantiles=np.quantile(gap_cells, [0.1, 0.5, 0.9]).tolist(), gap_cspace_cells_quantiles=np.quantile(gap_c, [0.1, 0.5, 0.9]).tolist(),
                 x0_active=dict(base_rect=float((act == 0).mean()), arm_walls=float((act == 1).mean()), internal=float((act == 2).mean())))
    return x, speed, normal, d, stats
