"""ntrl-demo training pairs for the parking scene (its point recipe, speed_sampling_gpu_kdtree_normal.py, in C-space).

Recipe, unchanged except that the distance is the car's exact signed clearance in configuration space:
x0 uniform in the box [-0.5, 0.5]^3, x1 = x0 + unit(rand - 0.5) * U(0, sqrt(dim)), both inside the box;
keep offset < d(x0) < margin and d(x1) > -offset; speed = clip(d / margin, offset / margin, 1); normal = unit
C-space gradient of d (the arm recipe's J^T n, normalized). Files: sampled_points.npy (N, 6), speed.npy (N, 2),
normal.npy (N, 6), the ntrl-demo models/data_mlp.py layout.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch

from scene import Scene, MARGIN, OFFSET


def sample(scene, n, seed, batch=2_000_000):
    dev = scene.device
    gen = torch.Generator(dev).manual_seed(seed)
    xs, ys, ns, drawn = [], [], [], 0
    total = 0
    while total < n:
        p = torch.rand(batch, 3, generator=gen, device=dev) - 0.5
        dp = torch.rand(batch, 3, generator=gen, device=dev) - 0.5
        rl = torch.rand(batch, 1, generator=gen, device=dev) * math.sqrt(3)
        q = p + torch.nn.functional.normalize(dp, dim=1) * rl
        drawn += batch
        keep = (q.abs() <= 0.5).all(1)
        p, q = p[keep], q[keep]
        d0, n0 = scene.clearance_normal(p)
        keep = (d0 < MARGIN) & (d0 > OFFSET)
        p, q, d0, n0 = p[keep], q[keep], d0[keep], n0[keep]
        d1, n1 = scene.clearance_normal(q)
        keep = (d1 > -OFFSET) & ~n0.isnan().any(1) & ~n1.isnan().any(1)
        xs.append(torch.cat([p, q], 1)[keep].cpu())
        ys.append(torch.stack([d0, d1], 1)[keep].cpu())
        ns.append(torch.cat([n0, n1], 1)[keep].cpu())
        total += int(keep.sum())
    x = torch.cat(xs)[:n].numpy().astype(np.float32)
    d = torch.cat(ys)[:n].numpy()
    normal = torch.cat(ns)[:n].numpy().astype(np.float32)
    speed = np.clip(d / MARGIN, OFFSET / MARGIN, 1).astype(np.float32)
    return x, speed, normal, d, drawn


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wall', type=float, required=True)
    p.add_argument('--n', type=int, default=500_000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    x, speed, normal, d, drawn = sample(Scene(a.wall), a.n, a.seed)
    np.save(a.out / 'sampled_points.npy', x)
    np.save(a.out / 'speed.npy', speed)
    np.save(a.out / 'normal.npy', normal)
    json.dump(dict(wall=a.wall, n=len(x), seed=a.seed, drawn=drawn, margin=MARGIN, offset=OFFSET,
                   x1_blocked=float((d[:, 1] <= OFFSET).mean()), x1_free=float((d[:, 1] >= MARGIN).mean())),
              open(a.out / 'data.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
