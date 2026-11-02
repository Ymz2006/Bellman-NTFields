"""Why a DP chain's bias rises or falls with level: target / T* = (teacher bias at the chosen split) x (true via-split
overhead), measured on the truth probe pairs (segment-DP task 8, diagnostic only).

For a chain (its field_segXX.pt snapshots) and a level m (1-based segment): the teacher is the chain's composite of
models 1..m-1 (as dp3.py builds it); the split candidates are dp.targets' (384: half nearest the straight line, half
uniform over the lens |i-k|, |k-j| <= e_(m-1) + blend), drawn with the probe pairs of segment m. Per pair:
  k*      the teacher's argmin split (the DP target's split), and k_line the lens pose nearest the straight line
          (smallest |i-k| + |k-j|: the unselected reference)
  exact   path-optimized travel times of both pieces of k* and of k_line (local_truth: straight-line start,
          --waypoints / --steps; accurate in the median at these lengths, with a tail where the straight start misses
          a homotopy, so medians are reported)
Reports, as medians over pairs: target / T*; selected bias = teacher(k*) / exact(k*); reference bias = teacher(k_line)
/ exact(k_line); overhead = exact(k*) / T* and exact(k_line) / T*; selection = selected / reference bias.
"""
import sys
sys.path.insert(0, '/clean/method'); sys.path.insert(0, '/clean/jeffrey/segment_dp')
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from common import sha, write
from fields import SegmentField, load_field
from sampler import load_pool
from local_truth import local_truth
from dp_exact import candidates


def main():
    p = argparse.ArgumentParser(); p.add_argument('--prepared', type=Path, required=True); p.add_argument('--truth', type=Path, required=True)
    p.add_argument('--chains', type=Path, nargs='+', required=True, help='training/ directories of DP chains'); p.add_argument('--labels', nargs='+', required=True)
    p.add_argument('--levels', default='5,7,9,11'); p.add_argument('--pairs', type=int, default=256); p.add_argument('--candidates', type=int, default=384)
    p.add_argument('--blend', type=float, default=0.016); p.add_argument('--waypoints', type=int, default=16); p.add_argument('--steps', type=int, default=250)
    p.add_argument('--chunk', type=int, default=2048); p.add_argument('--seed', type=int, default=11); p.add_argument('--output', type=Path, required=True); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    started = time.time(); pool, speeds, speed, info = load_pool(a.prepared, 'cuda:0'); speed.requires_grad_(False); square = pool.square().sum(1)
    z = np.load(a.truth); edges = z['edges'].tolist(); report = dict(truth=str(a.truth), truth_sha256=sha(a.truth), waypoints=a.waypoints, steps=a.steps, chains={}, **info)
    exact = lambda x, y: torch.cat([local_truth(speed, pool[x[s:s+a.chunk]], pool[y[s:s+a.chunk]], waypoints=a.waypoints, steps=a.steps)[0] for s in range(0, len(x), a.chunk)])
    for label, chain in zip(a.labels, a.chains):
        rows = []
        for m in map(int, a.levels.split(',')):
            snap = chain/f'field_seg{m-1:02d}.pt'; full = load_field(snap, 'cuda:0'); assert len(full.models) == m-1, (snap, len(full.models))
            teacher = SegmentField(list(full.models), edges[:m], a.blend).eval(); radius = edges[m-1]+a.blend
            sel = np.flatnonzero(z['segment'] == m-1)[:8 if a.smoke else a.pairs]; gen = torch.Generator(device='cuda:0').manual_seed(a.seed+m)
            ti = torch.as_tensor(z['source_index'][sel], device='cuda:0'); tj = torch.as_tensor(z['goal_index'][sel], device='cuda:0'); ts = torch.as_tensor(z['truth'][sel], device='cuda:0')
            index, valid, da, db = candidates(pool, square, ti, tj, radius, a.candidates, gen); n, c = index.shape
            with torch.no_grad():
                k = index.reshape(-1); A = ti.repeat_interleave(c); B = tj.repeat_interleave(c)
                tv = torch.cat([teacher(pool[A[s:s+65536]], pool[k[s:s+65536]])[:, 0]+teacher(pool[k[s:s+65536]], pool[B[s:s+65536]])[:, 0] for s in range(0, len(k), 65536)]).view(n, c)
            tv = tv.masked_fill(~valid, torch.inf); best, which = tv.min(1); ok = torch.isfinite(best)
            line = (da+db).masked_fill(~valid, torch.inf).argmin(1); r = torch.arange(n, device='cuda:0')
            ks, kl = index[r, which], index[r, line]
            es = exact(ti, ks)+exact(ks, tj); el = exact(ti, kl)+exact(kl, tj)
            med = lambda x: float(x[ok].median())
            row = dict(level=m, pairs=int(ok.sum()), target_over_truth=med(best/ts), selected_bias=med(best/es), reference_bias=med(tv[r, line]/el),
                       overhead_selected=med(es/ts), overhead_line=med(el/ts), same_split=float((ks == kl)[ok].float().mean()))
            row['selection'] = row['selected_bias']/row['reference_bias']; rows.append(row); print(label, json.dumps({k2: round(v, 4) if isinstance(v, float) else v for k2, v in row.items()}), flush=True)
            report['chains'][label] = dict(path=str(chain), levels=rows); write(a.output, report)
            del full, teacher; torch.cuda.empty_cache()
    report.update(complete=True, seconds=time.time()-started); write(a.output, report)


if __name__ == '__main__': main()
