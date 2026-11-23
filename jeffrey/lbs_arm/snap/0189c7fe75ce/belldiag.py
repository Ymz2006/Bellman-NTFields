"""The Bellman backup's signed residual on trained fields. No training.

For training pairs (a, g) of a prepared scene and field runs of it: T(a, g) and the backup y = min over the sampled steps of
(true step cost + T(step end, g)), exactly as the Bellman update computes it (core.bellman_term; radius --bell-r-cells, K steps,
cone, long steps), with and without the straight step towards the goal among the candidates. Reported: mean and root mean square
of T - y, in units of the step radius. On a field whose close-pair times are right (lift only), a mean well below 0 says the
backup itself asks for larger times.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/lbs_arm/belldiag.py --prepared <prepare run> --fields lift=<field run> lbs=<field run> --out <dir>
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--fields', nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--pairs', type=int, default=8192)
    p.add_argument('--bell-r-cells', type=float, default=0.663)
    p.add_argument('--bell-k', type=int, default=16)
    p.add_argument('--bell-cone', type=int, default=6)
    p.add_argument('--bell-long', type=float, default=8.0)
    p.add_argument('--seed', type=int, default=20261122)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    info = json.loads((a.prepared / 'scene.json').read_text())
    if info['describe']['dim'] > 6:
        import arm12
        sys.modules['arm'] = arm12
    import arm as A
    import core as C
    import trainer6 as TRN
    from common import Logger, dump
    a.out.mkdir(parents=True, exist_ok=True)
    log = Logger(a.out / 'log.txt')
    dev, t0 = 'cuda:0', time.time()
    scene, info = A.load_scene(a.prepared, dev)
    D = len(scene.lo)
    clear = C.Clearance(scene, compile=False)
    mn = TRN.ntrl_modules()[2]
    if a.smoke:
        a.pairs = 512
    X = np.load(a.prepared / 'data/sampled_points.npy', mmap_mode='r')
    pick = np.sort(np.random.default_rng(a.seed).choice(len(X), a.pairs, replace=False))
    points = torch.tensor(np.asarray(X[pick]), device=dev)
    radius = a.bell_r_cells / scene.E
    gap_len = torch.linalg.vector_norm(points[:, :D] - points[:, D:], dim=1)
    res = dict(stage='lbsarm-belldiag', scene=scene.describe(), prepared=str(a.prepared), smoke=a.smoke, pairs=a.pairs, radius=radius, radius_cells=a.bell_r_cells, K=a.bell_k, cone=a.bell_cone, long=a.bell_long,
               pair_length_cells_quantiles=[float(v) * scene.E for v in torch.quantile(gap_len, torch.tensor([.1, .5, .9], device=dev))], fields={})
    for item in a.fields:
        name, run = item.split('=', 1)
        net = C.load_field(mn, Path(run) / 'training/checkpoint.pt', scene, dev)
        out = {}
        for goal in (False, True):
            tot_sq, tot_gap, tot_used, tot_T, n = 0.0, 0.0, 0.0, 0.0, 0
            for i in range(0, len(points), 1024):
                pts = points[i:i + 1024]
                with torch.enable_grad():
                    tau, _, Xp = net.out(pts)
                    g = torch.autograd.grad(tau.sum(), Xp)[0]
                T = tau.detach()[:, 0]
                gen = torch.Generator(dev).manual_seed(a.seed + i)
                sq, used, gap = C.bellman_term(net, scene, pts, T, (g[:, :D].detach(), g[:, D:].detach()), radius, a.bell_k, gen, clear, cone=a.bell_cone, long=a.bell_long, goal=goal)
                tot_sq += float(sq) * len(pts)
                tot_gap += float(gap) * len(pts)
                tot_used += float(used) * len(pts)
                tot_T += float(T.sum())
                n += len(pts)
            # bellman_term: sq = sum over the two ends of the mean squared residual; gap = mean over the two ends of the mean residual
            out['goal step' if goal else 'as trained'] = dict(mean_residual_radii=tot_gap / n / radius, rms_residual_radii=float(np.sqrt(tot_sq / n / 2)) / radius, used=tot_used / n, T_mean=tot_T / n)
            o = out['goal step' if goal else 'as trained']
            log(f'{name}, candidates {"with the goal step" if goal else "as trained"}: mean T - y {o["mean_residual_radii"]:+.3f} radii, rms {o["rms_residual_radii"]:.3f} radii, pair ends used {o["used"]:.2f}, mean T {o["T_mean"]:.3f}')
        res['fields'][name] = dict(run=run, **out)
    res['elapsed_seconds'] = time.time() - t0
    dump(a.out / 'results.json', res)
    log(f'done in {res["elapsed_seconds"] / 60:.1f} min')


if __name__ == '__main__':
    main()
