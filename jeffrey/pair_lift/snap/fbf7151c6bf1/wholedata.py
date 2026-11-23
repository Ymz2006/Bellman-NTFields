"""Whole-interior training pairs: the data variant of the "2026-11-18 Lift Variants" page (its Part 1), rebuilt for the local SE(2) scenes.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/pair_lift/wholedata.py --from <prepare run> --out <dir>

ntrl-demo's recipe (jeffrey/maze_scale/data.py: v0 uniform in the unit cube, v1 = v0 + unit(rand - 0.5) * U(0, sqrt 3), both in the cube)
keeps offset < d(x0) < margin and d(x1) > -offset: the first end in the speed band next to an obstacle, the second anywhere outside
obstacles. Here the two lower bounds are dropped: d(x0) < margin, d(x1) anything. Both ends may lie anywhere inside an obstacle, where
the speed label is the floor (clip(d / margin, offset / margin, 1), 0.012 after ntrl-demo's transform) and the normal is the clearance
gradient. Everything else of the prepare run (scene, goals, truth, queries, controls, held-out points) is copied.
"""
import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pairnet as PN  # noqa: E402,F401  (puts lift_scale, embed_warp and maze_scale on the path)
import scenes as SN  # noqa: E402
from maze import MARGIN_CELLS, OFFSET_CELLS, RHO  # noqa: E402


def sample_whole(scene, n, seed, r_max=math.sqrt(3), batch=2_000_000, x0_inside=None):
    dev = scene.device
    gen = torch.Generator(dev).manual_seed(seed)
    scale = torch.tensor([1.0, 1.0, 2 * scene.phi_max], device=dev)
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
        d0, n0 = scene.clearance_normal(p)
        keep = d0 < MARGIN_CELLS                                         # ntrl-demo: and d0 > offset
        if x0_inside is not None:
            # thin the first ends inside obstacles to the share x0_inside of the kept pairs (keep_p from the share the recipe gives in this batch)
            ins = (d0 < -OFFSET_CELLS) & keep
            f = float(ins.sum()) / max(float(keep.sum()), 1.0)
            keep_p = min(1.0, x0_inside * (1 - f) / max(f * (1 - x0_inside), 1e-9))
            keep = keep & (~ins | (torch.rand(len(d0), generator=gen, device=dev) < keep_p))
        p, q, d0, n0 = p[keep], q[keep], d0[keep], n0[keep]
        d1, n1 = scene.clearance_normal(q)
        keep = ~n0.isnan().any(1) & ~n1.isnan().any(1)                   # ntrl-demo: and d1 > -offset
        xs.append(torch.cat([p, q], 1)[keep].cpu())
        ys.append(torch.stack([d0, d1], 1)[keep].cpu())
        ns.append(torch.cat([n0, n1], 1)[keep].cpu())
        total += int(keep.sum())
    x = torch.cat(xs)[:n].numpy().astype(np.float32)
    d = torch.cat(ys)[:n].numpy()
    normal = torch.cat(ns)[:n].numpy().astype(np.float32)
    speed = np.clip(d / MARGIN_CELLS, OFFSET_CELLS / MARGIN_CELLS, 1).astype(np.float32)
    gap_cells = np.linalg.norm(x[:, :2] - x[:, 3:5], axis=1) / scene.pitch
    inside = d < -OFFSET_CELLS
    stats = dict(n=len(x), drawn=drawn, r_max=r_max, x0_inside=float(inside[:, 0].mean()), x1_inside=float(inside[:, 1].mean()), either_inside=float(inside.any(1).mean()),
                 both_inside=float(inside.all(1).mean()), x0_skin=float(((d[:, 0] >= -OFFSET_CELLS) & (d[:, 0] <= OFFSET_CELLS)).mean()), x1_free=float((d[:, 1] >= MARGIN_CELLS).mean()),
                 depth_cells_quantiles=np.quantile(-d[inside], [0.1, 0.5, 0.9, 1.0]).tolist(), gap_cells_quantiles=np.quantile(gap_cells, [0.1, 0.5, 0.9]).tolist(),
                 dtheta_deg_median=float(np.degrees(np.median(np.abs(x[:, 2] - x[:, 5]) * scene.E / RHO))))
    return x, speed, normal, d, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--from', dest='from_run', type=Path, required=True, help='the prepare run whose scene, truth and queries are kept')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--n-data', type=int, default=500_000)
    p.add_argument('--x0-inside', type=float, default=None, help='share of first ends inside an obstacle (default: what the recipe gives)')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    src, out = a.from_run, a.out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    scene, info = SN.load_scene(src, 'cuda:0')
    for f in ('maze.json', 'goals.npy', 'queries.npz', 'truth_paths.npy', 'truth_fields.npy', 'oracle_paths_seed0.npy', 'scene.png', 'clearance_heading0.npy'):
        if (src / f).exists():
            shutil.copyfile(src / f, out / f)
    shutil.copytree(src / 'heldout', out / 'heldout', dirs_exist_ok=True)
    res = json.loads((src / 'results.json').read_text())
    x, speed, normal, d, stats = sample_whole(scene, 20_000 if a.smoke else a.n_data, 0, x0_inside=a.x0_inside)
    (out / 'data').mkdir(exist_ok=True)
    np.save(out / 'data/sampled_points.npy', x)
    np.save(out / 'data/speed.npy', speed)
    np.save(out / 'data/normal.npy', normal)
    res.update(stage='prepare', data=dict(stats, whole_interior=True, x0_inside_target=a.x0_inside), variant_of=str(src), smoke=a.smoke, elapsed_seconds=time.time() - t0, complete=True)
    (out / 'results.json').write_text(json.dumps(res, indent=1, default=float) + '\n')
    msg = f'whole-interior data variant of {src}: {stats}; done in {(time.time() - t0) / 60:.1f} min'
    (out / 'log.txt').write_text(msg + '\n')
    print(msg)


if __name__ == '__main__':
    main()
