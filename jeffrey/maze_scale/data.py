"""ntrl-demo training pairs for a maze (its point recipe, dataprocessing/speed_sampling_gpu_kdtree_normal.py, in the
robot's C-space; as jeffrey/parking/data.py).

ntrl-demo's recipe assumes the unit cube [-0.5, 0.5]^dim; it is applied in the unit cube the C-space box maps to by
scaling phi by 0.5 / phi_max (x, y unchanged): v0 uniform in the cube, v1 = v0 + unit(rand - 0.5) * U(0, r_max)
(ntrl-demo: r_max = sqrt(dim)), both inside the cube; x0, x1 = v0, v1 mapped back. (Applied directly to the thin phi
slab, the isotropic step would leave the box unless it were nearly flat in phi: median pair gap 0.3 cells at every
maze size, no long pairs.) The distance is the robot's exact signed clearance; keep offset < d(x0) < margin and
d(x1) > -offset; speed = clip(d / margin, offset / margin, 1); normal = unit C-space
gradient of d. margin and offset are 0.175 and 0.0175 cells (maze.py). Files: sampled_points.npy (N, 6),
speed.npy (N, 2), normal.npy (N, 6), the ntrl-demo models/data_mlp.py layout.
"""
import math
import numpy as np
import torch
from maze import MARGIN_CELLS, OFFSET_CELLS, RHO


def sample(maze, n, seed, r_max=math.sqrt(3), batch=2_000_000):
    dev = maze.device
    gen = torch.Generator(dev).manual_seed(seed)
    scale = torch.tensor([1.0, 1.0, 2 * maze.phi_max], device=dev)
    xs, ys, ns = [], [], []
    drawn, total = 0, 0
    while total < n:
        v0 = torch.rand(batch, 3, generator=gen, device=dev) - 0.5
        dp = torch.rand(batch, 3, generator=gen, device=dev) - 0.5
        rl = torch.rand(batch, 1, generator=gen, device=dev) * r_max
        v1 = v0 + torch.nn.functional.normalize(dp, dim=1) * rl
        drawn += batch
        keep = (v1.abs() <= 0.5).all(1)
        p, q = v0[keep] * scale, v1[keep] * scale
        d0, n0 = maze.clearance_normal(p)
        keep = (d0 < MARGIN_CELLS) & (d0 > OFFSET_CELLS)
        p, q, d0, n0 = p[keep], q[keep], d0[keep], n0[keep]
        d1, n1 = maze.clearance_normal(q)
        keep = (d1 > -OFFSET_CELLS) & ~n0.isnan().any(1) & ~n1.isnan().any(1)
        xs.append(torch.cat([p, q], 1)[keep].cpu())
        ys.append(torch.stack([d0, d1], 1)[keep].cpu())
        ns.append(torch.cat([n0, n1], 1)[keep].cpu())
        total += int(keep.sum())
    x = torch.cat(xs)[:n].numpy().astype(np.float32)
    d = torch.cat(ys)[:n].numpy()
    normal = torch.cat(ns)[:n].numpy().astype(np.float32)
    speed = np.clip(d / MARGIN_CELLS, OFFSET_CELLS / MARGIN_CELLS, 1).astype(np.float32)
    gap_cells = np.linalg.norm(x[:, :2] - x[:, 3:5], axis=1) / maze.pitch
    stats = dict(n=len(x), drawn=drawn, r_max=r_max, x1_blocked=float((d[:, 1] <= OFFSET_CELLS).mean()), x1_free=float((d[:, 1] >= MARGIN_CELLS).mean()),
                 gap_cells_quantiles=np.quantile(gap_cells, [0.1, 0.5, 0.9]).tolist(),
                 dtheta_deg_median=float(np.degrees(np.median(np.abs(x[:, 2] - x[:, 5]) * maze.E / RHO))))
    return x, speed, normal, d, stats
