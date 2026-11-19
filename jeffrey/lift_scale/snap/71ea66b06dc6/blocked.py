"""Stage 'blocked': for every training pair of a prepare run, whether its straight C-space segment crosses an obstacle (lift.segment_blocked),
by a sample every 0.1 cells (blocked.npy) and by the lift page's 48 points per segment (blocked48.npy), plus scene measures the report
uses: the largest true travel time in box lengths, the share of blocked pairs by pair length, nu = |grad_q d| / E on the band.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/lift_scale/blocked.py --prepared <prepare run> --out <dir>
"""
import argparse
import os
import sys
import time
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lift as LF  # noqa: E402
import scenes as SN  # noqa: E402
import diag as DG  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from common import Logger, dump
    log = Logger(a.out / 'log.txt')
    dev = 'cuda:0'
    t0 = time.time()
    scene, info = SN.load_scene(a.prepared, dev)
    x = np.load(a.prepared / 'data/sampled_points.npy', mmap_mode='r')
    n = len(x)
    dense = np.zeros(n, bool)
    p48 = np.zeros(n, bool)
    step = 20_000 if a.smoke else 100_000
    for i in range(0, n, step):
        pr = torch.tensor(np.asarray(x[i:i + step]), device=dev)
        dense[i:i + step] = LF.segment_blocked(scene, pr).cpu().numpy()
        p48[i:i + step] = LF.segment_blocked(scene, pr, points=48).cpu().numpy()
    np.save(a.out / 'blocked.npy', dense)
    np.save(a.out / 'blocked48.npy', p48)
    gap = np.linalg.norm(np.asarray(x[:, :2]) - np.asarray(x[:, 3:5]), axis=1) * scene.E
    by_gap = {}
    for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 1e9)):
        m = (gap >= lo) & (gap < hi)
        if m.sum():
            by_gap[f'{lo:g}-{hi:g}' if hi < 1e8 else f'>{lo:g}'] = dict(share_of_pairs=float(m.mean()), blocked=float(dense[m].mean()), blocked48=float(p48[m].mean()),
                                                                         missed_by_48=float((dense[m] & ~p48[m]).sum() / max(dense[m].sum(), 1)))
    truth = DG.Truth(scene, a.prepared)
    tmax, unreachable, total = 0.0, 0, 0
    for g in range(truth.fields.shape[0]):
        f = np.asarray(truth.fields[g], dtype=np.float32)
        fin = np.isfinite(f)
        clear = None
        tmax = max(tmax, float(f[fin].max()))
        total += f.size
        unreachable += int((~fin).sum())
    # true travel times of random free pairs: quantiles in box lengths
    pairs, gi, t_star, D = DG.truth_pairs(scene, truth, 4000 if a.smoke else 40000)
    res = dict(stage='lift-scale-blocked', prepared=str(a.prepared), scene=scene.describe(), scene_name=SN.scene_name(info), pairs=int(n),
               blocked_share=float(dense.mean()), blocked48_share=float(p48.mean()), missed_by_48=float((dense & ~p48).sum() / max(dense.sum(), 1)),
               extra_in_48=float((p48 & ~dense).sum() / max(p48.sum(), 1)), by_gap_cells=by_gap, gap_cells_quantiles=np.quantile(gap, [0.1, 0.5, 0.9, 0.99]).tolist(),
               truth=dict(t_star_max=tmax, t_star_max_cells=tmax * scene.E, unreachable_node_share=unreachable / total,
                          pair_t_star_quantiles=torch.quantile(t_star, torch.tensor([0.5, 0.9, 0.99, 1.0], device=dev)).tolist(),
                          detour_pair_share=float((t_star >= 1.3 * D).float().mean()), los_pair_share=float((t_star <= 1.05 * D).float().mean())),
               nu=DG.nu_stats(scene, a.prepared), profile=dict(G_offset=LF.Profile(device=dev).G_offset, G_max=LF.Profile(device=dev).G_max,
                                                               band_length=LF.Profile(device=dev).band_length, floor_slope=LF.Profile(device=dev).floor_slope),
               smoke=a.smoke, elapsed_seconds=time.time() - t0, complete=True)
    dump(a.out / 'results.json', res)
    log(f'{res["scene_name"]}: {n} pairs, blocked {res["blocked_share"]:.3f} (48 points: {res["blocked48_share"]:.3f}, misses {res["missed_by_48"]:.3f} of the blocked), '
        f'largest T* {tmax:.2f} box lengths, nu {res["nu"]}, {time.time() - t0:.0f}s')
    log(f'by pair length (cells): {by_gap}')


if __name__ == '__main__':
    main()
