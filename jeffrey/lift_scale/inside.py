"""Read-only forensics: the field inside obstacles. For configurations x inside an obstacle (clearance in [-0.28, -0.02] cells) paired with random
band configurations a of the training set: S_floor |grad_x T(a, x)| by depth (1 = the eikonal equation of the closed problem holds), and
the time across a wall: for the mirror pairs of jumps.py, T(a, b) between the two sides against the closed problem's crossing time.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/lift_scale/inside.py --out jeffrey/lift_scale/report/inside.json <run dir>[@epoch] ...
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lift as LF  # noqa: E402
import scenes as SN  # noqa: E402
import jumps as JM  # noqa: E402

DEPTHS = [(-0.02, 0.0), (-0.06, -0.02), (-0.12, -0.06), (-0.2, -0.12), (-0.3, -0.2)]


def analyse(run, epoch=None, n=40000):
    res = json.load(open(run / 'results.json'))
    prep = Path(res['prepared'])
    scene, info = SN.load_scene(prep)
    ck = run / 'training/checkpoint.pt' if epoch is None else run / f'training/checkpoints/epoch_{epoch:05d}.pt'
    net = LF.load_field(ck, scene)
    gen = torch.Generator(scene.device).manual_seed(21)
    q = scene.uniform(40 * n, gen)
    d = scene.clearance(q)
    keep = (d < 0) & (d > -0.3)
    x, dx = q[keep][:n], d[keep][:n]
    a = torch.tensor(np.load(prep / 'heldout/sampled_points.npy')[:len(x), :3], device=scene.device)
    pairs = torch.cat([a, x], 1)
    tau, _, xp = net.out(pairs)
    g, = torch.autograd.grad(tau.sum(), xp)
    floor = float(scene.train_speed(torch.tensor([-1.0], device=scene.device)))
    slope = floor * torch.linalg.vector_norm(g[:, 3:], dim=1)
    out = dict(run=run.name.replace('registered_', ''), epoch=epoch, scene=res['scene_name'], label=None, lift=res['args']['lift'], push=res['args']['push'], depth=res['args'].get('depth'), by_depth={})
    for lo, hi in DEPTHS:
        m = (dx >= lo) & (dx < hi)
        if int(m.sum()) > 30:
            out['by_depth'][f'{-hi:g}-{-lo:g}'] = dict(n=int(m.sum()), slope=float(slope[m].median()), q10=float(slope[m].quantile(0.1)), q90=float(slope[m].quantile(0.9)))
    W = JM.wall_pairs(scene, prep)
    with torch.no_grad():
        tab = net.out(torch.cat([W['A'], W['B']], 1))[0][:, 0] / scene.pitch
        # the closed problem's time straight across: integral of 1 / S along the segment (cells)
        k = 400
        sub = torch.arange(len(W['A']), device=scene.device)[:4000]
        t = (torch.arange(k, device=scene.device, dtype=torch.float32) + 0.5) / k
        pts = W['A'][sub][:, None] + t[None, :, None] * (W['B'][sub] - W['A'][sub])[:, None]
        inv = 1 / scene.train_speed(scene.clearance(pts.reshape(-1, 3))).reshape(len(sub), k)
        straight = inv.mean(1) * torch.linalg.vector_norm(W['B'][sub] - W['A'][sub], dim=1) / scene.pitch
    out['across'] = dict(n=int(len(tab)), median=float(tab.median()), q10=float(tab.quantile(0.1)), q90=float(tab.quantile(0.9)), planar_gap=float(W['planar_gap'].median()),
                         straight_median=float(straight.median()), straight_q10=float(straight.quantile(0.1)), step_true_q90=float(W['step_true'].quantile(0.9)), step_true_max=float(W['step_true'].max()))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+')
    p.add_argument('--out', type=Path, default=None)
    a = p.parse_args()
    rows = []
    for spec in a.runs:
        path, _, ep = spec.partition('@')
        r = analyse(Path(path), int(ep) if ep else None)
        rows.append(r)
        print(f"{r['scene']} | lift {r['lift']} depth {r['depth']} push {r['push']} | epoch {r['epoch']}: S_floor |grad T| by depth (cells) " +
              ', '.join(f"{k}: {v['slope']:.2f}" for k, v in r['by_depth'].items()) + f"; T between mirror configurations across a wall: median {r['across']['median']:.1f} cells (q10 {r['across']['q10']:.1f}, q90 {r['across']['q90']:.1f}); straight crossing in the closed problem {r['across']['straight_median']:.0f} cells (q10 {r['across']['straight_q10']:.0f}); true step q90 {r['across']['step_true_q90']:.1f}, max {r['across']['step_true_max']:.1f}")
    if a.out:
        prev = json.load(open(a.out)) if a.out.exists() else []
        keys = {(r['run'], r['epoch']) for r in rows}
        json.dump([r for r in prev if (r['run'], r['epoch']) not in keys] + rows, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
