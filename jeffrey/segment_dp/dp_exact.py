"""Exact-piece DP targets on the truth probe pairs (segment-DP task 8, diagnostic only; never trains or selects).

For the probe pairs (i, j) of truth.py in segment m (3 or 4), the DP target y = min_k T(i,k) + T(k,j) over the same
split candidates as dp.targets (pool poses in the lens |i-k|, |k-j| <= e_(m-1) + blend; half nearest the straight line,
half uniform; 384 candidates), with the pieces T taken from
  exact   local_truth.local_truth (path-optimized, not a network): the target a perfect lower level would give
  <label> the first --base-models models of each listed segment field (the base of a DP chain), for level 3
Reports median y / T* and mean |log y / T*| per source, the share of pairs with an empty lens, the detour of the
winning split (|i-k| + |k-j|) / |i-j|, and how often the exact pieces pick the same split as each field.
"""
import sys
sys.path.insert(0, '/clean/method'); sys.path.insert(0, '/clean/jeffrey/segment_dp')
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from common import sha, write
from fields import SegmentField, load_field
from sampler import DUPLICATE, load_pool, pairwise
from local_truth import local_truth


@torch.no_grad()
def candidates(pool, square, i, j, radius, count, generator):
    a, b = pool[i], pool[j]; da = pairwise(a, pool, square); db = pairwise(b, pool, square)
    lens = (da <= radius) & (db <= radius) & (da > DUPLICATE) & (db > DUPLICATE); half = count//2
    near = (da+db).masked_fill_(~lens, torch.inf).topk(half, dim=1, largest=False)
    rand = torch.rand(lens.shape, generator=generator, device=pool.device).masked_fill_(~lens, -1.0).topk(half, dim=1)
    index = torch.cat((near.indices, rand.indices), 1); valid = torch.cat((torch.isfinite(near.values), rand.values >= 0), 1)
    return index, valid, da.gather(1, index), db.gather(1, index)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--prepared', type=Path, required=True); p.add_argument('--truth', type=Path, required=True)
    p.add_argument('--fields', type=Path, nargs='*', default=[]); p.add_argument('--labels', nargs='*', default=[]); p.add_argument('--base-models', type=int, default=2)
    p.add_argument('--segments', default='3,4'); p.add_argument('--max-pairs', default='1024,256'); p.add_argument('--candidates', type=int, default=384)
    p.add_argument('--blend', type=float, default=0.016); p.add_argument('--output', type=Path, required=True); p.add_argument('--seed', type=int, default=7); p.add_argument('--chunk', type=int, default=4096)
    p.add_argument('--smoke', action='store_true'); a = p.parse_args(); started = time.time()
    pool, speeds, speed, info = load_pool(a.prepared, 'cuda:0'); speed.requires_grad_(False); square = pool.square().sum(1)
    z = np.load(a.truth); edges = z['edges'].tolist(); gen = torch.Generator(device='cuda:0').manual_seed(a.seed)
    bases = {label: SegmentField(list(load_field(f, 'cuda:0').models[:a.base_models]), edges[:a.base_models+1], a.blend).eval() for label, f in zip(a.labels, a.fields)}
    report = dict(truth=str(a.truth), truth_sha256=sha(a.truth), fields={l: str(f) for l, f in zip(a.labels, a.fields)}, candidates=a.candidates, levels=[], **info)
    for seg, cap in zip(map(int, a.segments.split(',')), map(int, a.max_pairs.split(','))):
        m = seg-1; radius = edges[m]+a.blend   # level seg (1-based) = dp3's m = seg-1: pieces <= e_(seg-1) + blend
        sel = np.flatnonzero(z['segment'] == m)[:16 if a.smoke else cap]
        ti = torch.as_tensor(z['source_index'][sel], device='cuda:0'); tj = torch.as_tensor(z['goal_index'][sel], device='cuda:0'); ts = torch.as_tensor(z['truth'][sel], device='cuda:0')
        index, valid, da, db = candidates(pool, square, ti, tj, radius, a.candidates, gen)
        rows, cols = torch.nonzero(valid, as_tuple=True); k = index[rows, cols]
        # unique pieces (i,k) and (k,j): exact local travel time once per piece
        left = torch.stack((ti[rows], k), 1); right = torch.stack((k, tj[rows]), 1)
        pieces, inverse = torch.unique(torch.cat((left, right)).sort(1).values, dim=0, return_inverse=True)
        t0 = time.time(); exact = []
        for s in range(0, len(pieces), a.chunk):
            exact.append(local_truth(speed, pool[pieces[s:s+a.chunk, 0]], pool[pieces[s:s+a.chunk, 1]])[0])
        exact = torch.cat(exact); n = len(rows); piece_seconds = time.time()-t0
        sources = dict(exact=exact[inverse[:n]]+exact[inverse[n:]])
        with torch.no_grad():
            for label, base in bases.items():
                if seg != a.base_models+1: continue   # a network base is the teacher of level base_models+1 only
                v = lambda x, y: torch.cat([base(pool[x[s:s+65536]], pool[y[s:s+65536]])[:, 0] for s in range(0, len(x), 65536)])
                sources[label] = v(ti[rows], k)+v(k, tj[rows])
        gap = torch.linalg.vector_norm(pool[ti]-pool[tj], dim=1); level = dict(segment=seg, radius=radius, pairs=len(sel), unique_pieces=len(pieces), piece_seconds=piece_seconds,
                                                                           lens_valid_median=float(valid.sum(1).float().median()), sources={})
        win = {}
        for label, total in sources.items():
            full = torch.full(valid.shape, torch.inf, device='cuda:0'); full[rows, cols] = total; best, which = full.min(1); ok = torch.isfinite(best); r = (best[ok]/ts[ok])
            detour = ((da+db)[torch.arange(len(sel), device='cuda:0'), which]/gap)[ok]; win[label] = which
            level['sources'][label] = dict(median_over_truth=float(r.median()), mean_abs_log=float(r.log().abs().mean()), p10=float(r.quantile(.1)), p90=float(r.quantile(.9)),
                                           empty=float((~ok).float().mean()), detour_median=float(detour.median()), below_truth=float((r < .995).float().mean()))
        for label in sources:
            if label != 'exact': level['sources'][label]['same_split_as_exact'] = float((win[label] == win['exact']).float().mean())
        report['levels'].append(level); print(json.dumps(level), flush=True); write(a.output, report)
    report.update(complete=True, seconds=time.time()-started); write(a.output, report)


if __name__ == '__main__': main()
