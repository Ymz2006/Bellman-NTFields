"""ntrl-demo's training pairs drawn on the warped maze of a fitted warp (cross_warp/gapprepare.py sample_z for any invertible warp).

    python zdraw.py --prepared <prepare or gapprepare run> --fit <fit run> --out <dir>

The recipe's draws v0, v1 (the same generator calls as maze_scale/data.sample) are read as normalized coordinates of the bounding box
of the warped maze for x, y (the heading map is linear), and mapped back by phi^-1 (netwarp.inverse_plane). The recipe's rules
(offset < d(x0) < margin, d(x1) > -offset, finite normals) then apply on the real maze. So the pairs are uniform over the warped maze
instead of the real one. Draws outside the image of the box are rejected (a triangular map's image is the box: none).
The run also copies the prepared run's maze.json, goals, queries and truth paths: it is a prepared run with other pairs.
"""
import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = next((q for q in HERE.parents if (q / 'baselines/ntrl-demo').is_dir()), Path(os.environ.get('NTFIELD_ROOT', '/home/jeffrey/bellman-ntfield')))
sys.path[:0] = [str(ROOT / 'jeffrey/maze_scale'), str(ROOT / 'jeffrey/cross_warp'), str(HERE)]
import fit as FT  # noqa: E402
import netwarp as NW  # noqa: E402


def sample(maze, warp, n, seed, r_max=math.sqrt(3), batch=1_000_000, chunk=131072):
    from maze import MARGIN_CELLS, OFFSET_CELLS
    dev = maze.device
    t = torch.linspace(-0.5, 0.5, 2001, device=dev)
    with torch.no_grad():
        edge = torch.cat([torch.stack([t, torch.full_like(t, v)], 1) for v in (-0.5, 0.5)] + [torch.stack([torch.full_like(t, v), t], 1) for v in (-0.5, 0.5)])
        z = warp.plane(edge)
    zlo, zhi = z.amin(0), z.amax(0)
    span = zhi - zlo

    def to_x(v):
        out, res = [], []
        for i in range(0, len(v), chunk):
            vi = v[i:i + chunk]
            zz = zlo + (vi[:, :2] + 0.5) * span
            x = warp.inverse_plane(zz, iters=8)
            res.append((warp.plane(x) - zz).norm(dim=1))
            out.append(torch.cat([x, vi[:, 2:3]], 1))
        return torch.cat(out), torch.cat(res)

    gen = torch.Generator(dev).manual_seed(seed)
    scale = torch.tensor([1.0, 1.0, 2 * maze.phi_max], device=dev)
    xs, ys, ns = [], [], []
    drawn, total, worst, outside = 0, 0, 0.0, 0
    while total < n:
        v0 = torch.rand(batch, 3, generator=gen, device=dev) - 0.5
        dp = torch.rand(batch, 3, generator=gen, device=dev) - 0.5
        rl = torch.rand(batch, 1, generator=gen, device=dev) * r_max
        v1 = v0 + torch.nn.functional.normalize(dp, dim=1) * rl
        drawn += batch
        keep = (v1.abs() <= 0.5).all(1)
        a, ra = to_x(v0[keep])
        b, rb = to_x(v1[keep])
        inside = (a[:, :2].abs() <= 0.5).all(1) & (b[:, :2].abs() <= 0.5).all(1)
        outside += int((~inside).sum())
        worst = max(worst, float(ra[inside].max()), float(rb[inside].max()))
        p, q = a[inside] * scale, b[inside] * scale
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
    stats = dict(n=len(x), drawn=drawn, r_max=r_max, seed=seed, drawn_on='warped maze (z = phi(x))', outside_image=outside, inverse_residual_max=worst, x1_blocked=float((d[:, 1] <= OFFSET_CELLS).mean()),
                 x1_free=float((d[:, 1] >= MARGIN_CELLS).mean()), gap_cells_quantiles=np.quantile(gap_cells, [0.1, 0.5, 0.9]).tolist())
    return x, speed, normal, d, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--fit', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--n-data', type=int, default=500_000)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from common import Logger, dump
    log = Logger(a.out / 'log.txt')
    dev = 'cuda:0'
    t0 = time.time()
    maze, info = FT.load_maze(a.prepared, dev)
    state, netcfg, _ = NW.load_fit(a.fit, dev)
    w = NW.NetWarp(dict(net=netcfg, s=1.0)).to(dev)
    w.set_state(state, maze)
    n = 20_000 if a.smoke else a.n_data
    x, speed, normal, d, stats = sample(maze, w, n, 0)
    log(f'data: {stats}')
    (a.out / 'data').mkdir(exist_ok=True)
    np.save(a.out / 'data/sampled_points.npy', x)
    np.save(a.out / 'data/speed.npy', speed)
    np.save(a.out / 'data/normal.npy', normal)
    hx, _, _, hd, _ = sample(maze, w, 5_000 if a.smoke else 50_000, 1)
    (a.out / 'heldout').mkdir(exist_ok=True)
    np.save(a.out / 'heldout/sampled_points.npy', hx)
    np.save(a.out / 'heldout/clearance_cells.npy', hd.astype(np.float32))
    for f in ('maze.json', 'goals.npy', 'queries.npz', 'truth_paths.npy'):
        shutil.copyfile(a.prepared / f, a.out / f)
    src = json.load(open(a.prepared / 'results.json'))
    res = {k: src[k] for k in ('stage', 'scale', 'maze', 'base', 'truth', 'queries', 'planner', 'straight', 'oracle', 'categories') if k in src}
    res.update(data=stats, fit=str(a.fit), queries_from=str(a.prepared), line_warp_zdraw=True, smoke=a.smoke, complete=True, elapsed_seconds=time.time() - t0)
    dump(a.out / 'results.json', res)
    log(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
