"""Path-optimized local travel times for short pool pairs: the labels of the "correct local field" (segment-DP task 8).

For a pair (q, g) with |q - g| <= ~0.18 the optimal path under the frozen speed S is close to the straight line, so
T*(q, g) = min over paths of the integral of ds / S is computed by optimizing a polyline that starts on the straight
line (truth.py's optimizer: Adam on the interior waypoints, re-resampled by arc length every 50 steps), here with
--waypoints points and --steps steps and the speed network under bf16 autocast while optimizing (the final cost is
re-measured densely in float32: 20 points by arc length, 17 points per segment, as truth.py). The label is the
cheaper of that path and the straight line.
Gradient labels by the envelope theorem: with the optimized interior fixed, d T*/d q and d T*/d g are the gradients
of the dense float32 path cost with respect to its endpoints (S |grad| ~ 1 is reported as a check).

    local_truth.py --prepared P --output O [--pairs 400000] [--segments 1,2]   label set for the first segments
                                                                             (pairs drawn by sampler.PairSampler
                                                                             exactly as the eikonal specialists')
    local_truth.py --prepared P --output O --check TRUTH.npz                  accuracy vs truth.py's T* only
Writes labels.npz (i, j pool indices, segment, gap, T, straight, grad_q, grad_g) and labels.json.
"""
import sys
sys.path.insert(0, '/clean/method'); sys.path.insert(0, '/clean/jeffrey/segment_dp')
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from common import write
from sampler import PairSampler, load_pool
from truth import path_cost, resample, segment_cost
from eikonal import EDGES


def optimize(speed, x, steps, lr, amp=True):
    """truth.optimize with the path cost under bf16 autocast (endpoints fixed, per-path step lr x length / w)."""
    x = x.clone(); n, w, _ = x.shape
    for start in range(0, steps, 50):
        length = torch.linalg.vector_norm(x[:, 1:]-x[:, :-1], dim=2).sum(1)
        scale = (lr*length/(w-1))[:, None, None]
        inner = x[:, 1:-1].clone().requires_grad_(True); m = torch.zeros_like(inner); v = torch.zeros_like(inner)
        for it in range(1, min(50, steps-start)+1):
            path = torch.cat((x[:, :1], inner, x[:, -1:]), 1)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                cost = path_cost(speed, path, 5)
            grad = torch.autograd.grad(cost.float().sum(), inner)[0].float()
            with torch.no_grad():
                m.mul_(.9).add_(.1*grad); v.mul_(.999).add_(.001*grad.square())
                decay = 0.5*(1+np.cos(np.pi*(start+it)/steps))
                inner -= decay*scale*(m/(1-.9**it))/((v/(1-.999**it)).sqrt()+1e-12)
                inner.clamp_(-.5, .5)
        with torch.no_grad():
            x = resample(torch.cat((x[:, :1], inner.detach(), x[:, -1:]), 1), w)
    return x


def local_truth(speed, q, g, waypoints=8, steps=100, lr=0.05, amp=True):
    """-> T (float32 dense cost of the better of optimized path and straight line), straight-line cost, dT/dq, dT/dg."""
    line = torch.stack((q, g), 1)
    x = optimize(speed, resample(line, waypoints), steps, lr, amp)
    dense = resample(x, 20).detach(); straight = resample(line, 20)
    with torch.no_grad():
        c_line = path_cost(speed, straight, 17)
    use = path_cost(speed, dense, 17).detach() > c_line          # the straight line is cheaper: take it
    dense[use] = straight[use]
    with torch.no_grad():
        cost = path_cost(speed, dense, 17)
    # only the first and the last segment of the path depend on its endpoints
    q = dense[:, 0].clone().requires_grad_(True); g = dense[:, -1].clone().requires_grad_(True)
    end_cost = segment_cost(speed, q, dense[:, 1], 17)+segment_cost(speed, dense[:, -2], g, 17)
    grad_q, grad_g = torch.autograd.grad(end_cost.sum(), (q, g))
    return cost, c_line, grad_q, grad_g


def run_batches(speed, q, g, chunk, **kw):
    out = [[], [], [], []]
    for s in range(0, len(q), chunk):
        for o, v in zip(out, local_truth(speed, q[s:s+chunk], g[s:s+chunk], **kw)): o.append(v)
    return [torch.cat(o) for o in out]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--prepared', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pairs', type=int, default=400000, help='per segment'); p.add_argument('--segments', default='1,2')
    p.add_argument('--edges', default=EDGES); p.add_argument('--blend', type=float, default=0.016)
    p.add_argument('--waypoints', type=int, default=8); p.add_argument('--steps', type=int, default=100); p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--no-amp', action='store_true'); p.add_argument('--chunk', type=int, default=4096); p.add_argument('--seed', type=int, default=20261031)
    p.add_argument('--part', default='0/1', help='k/K: this process labels the k-th of K interleaved shares of the pairs'); p.add_argument('--check', type=Path)
    p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True); started = time.time(); kw = dict(waypoints=a.waypoints, steps=a.steps, lr=a.lr, amp=not a.no_amp)
    pool, speeds, speed, info = load_pool(a.prepared, 'cuda:0'); speed.requires_grad_(False)
    if a.check:
        z = np.load(a.check); report = dict(truth=str(a.check), **kw, segments=[])
        for s in range(3):
            sel = np.flatnonzero(z['segment'] == s)[:64 if a.smoke else None]
            q = torch.as_tensor(z['q'][sel], device='cuda:0'); g = torch.as_tensor(z['g'][sel], device='cuda:0')
            t0 = time.time(); T, line, gq, gg = run_batches(speed, q, g, a.chunk, **kw); r = (T/torch.as_tensor(z['truth'][sel], device='cuda:0')).cpu().numpy()
            sq = torch.cat([speed(q[i:i+65536]).reshape(-1) for i in range(0, len(q), 65536)])
            report['segments'].append(dict(segment=s+1, pairs=len(sel), median_over_truth=float(np.median(r)), mean_abs_log=float(np.abs(np.log(r)).mean()),
                                           p90=float(np.quantile(r, .9)), below_truth=float((r < .995).mean()), straight_won=float((line <= T*1.0001).float().mean()),
                                           speed_times_grad=float((sq*torch.linalg.vector_norm(gq, dim=1)).median()), seconds_per_pair=(time.time()-t0)/len(sel)))
            print(json.dumps(report['segments'][-1]), flush=True)
        write(a.output/'check.json', report); return
    k, K = map(int, a.part.split('/')); edges = [float(x) for x in a.edges.split(',')]; segs = [int(x)-1 for x in a.segments.split(',')]
    pairs = 512 if a.smoke else a.pairs
    generator = torch.Generator(device='cuda:0').manual_seed(a.seed); sampler = PairSampler(pool, edges, a.blend, generator)
    seg = torch.tensor(segs, device='cuda:0').repeat_interleave(pairs)
    i, j = [], []
    for s in range(0, len(seg), 8192):
        qi, gi, _, _ = sampler.sample(seg[s:s+8192]); i.append(qi); j.append(gi)
    i, j = torch.cat(i), torch.cat(j); mine = torch.arange(k, len(i), K, device='cuda:0'); i, j, seg = i[mine], j[mine], seg[mine]
    out = dict(T=[], straight=[], grad_q=[], grad_g=[])
    for s in range(0, len(i), a.chunk):
        for key, v in zip(out, local_truth(speed, pool[i[s:s+a.chunk]], pool[j[s:s+a.chunk]], **kw)): out[key].append(v)
        if (s//a.chunk) % 10 == 0: print(json.dumps(dict(done=s+len(out['T'][-1]), of=len(i), seconds=round(time.time()-started))), flush=True)
    out = {key: torch.cat(v).cpu().numpy() for key, v in out.items()}; gap = torch.linalg.vector_norm(pool[i]-pool[j], dim=1).cpu().numpy()
    name = 'labels.npz' if K == 1 else f'labels_part{k}of{K}.npz'
    np.savez(a.output/name, i=i.cpu().numpy(), j=j.cpu().numpy(), segment=seg.cpu().numpy(), gap=gap, **out)
    write(a.output/(name[:-4]+'.json'), dict(**kw, pairs_per_segment=pairs, segments=segs, edges=edges, blend=a.blend, seed=a.seed, part=a.part, rows=len(i), **info,
                                              T_over_gap_median=float(np.median(out['T']/gap)), straight_over_T_median=float(np.median(out['straight']/out['T'])),
                                              seconds=time.time()-started, smoke=a.smoke))


if __name__ == '__main__': main()
