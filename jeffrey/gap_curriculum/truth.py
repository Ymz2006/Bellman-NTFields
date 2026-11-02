"""Graph ground truth for the travel cost (diagnostic only; never selects a checkpoint).

A k-nearest-neighbour graph over the bank sources (k = 24, symmetrized), edge cost = trapezoid integral of ds / speed
along the straight segment (the frozen speed network, 9 points, the bank's own local-cost rule with fewer points).
Dijkstra from 64 seeded goal sources gives D(q, g) and the shortest-path tree; the same graph with Euclidean edge
costs gives E(q, g), so D / E is the travel time per unit of graph path length (the kNN zig-zag divides out) and is
the reference for the field's T / |q - g|. For each checkpoint, over probe pairs (q a source reached from g, half of
them within 0.7 of g, the test range) and per gap bin:
  ratio     median T(q, g)/|q-g| next to the reference median D/E
  spearman  rank correlation of T(q, g)/|q-g| with D/E (does the field know which pairs are slow / need detours)
  cosine    cos(-grad_q T(q, g), direction from q to the node 3 hops along the shortest path to g)
The same cosine of the straight line g - q is the Euclidean reference. The graph has no exact collision veto, so
D can cut thin panels at the speed floor; it is a common reference for comparing fields, not the true optimum.
"""
import sys
sys.path.insert(0, '/clean/method')
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.stats import spearmanr
from common import write
from network import load_checkpoint
from speed import load_speed
BINS = [0, .1, .2, .3, .4, .5, .7, 1.0, 1.4, 2.0]


def knn(pool, k):
    square = pool.square().sum(1); out = torch.empty((len(pool), k), dtype=torch.long, device=pool.device); dist = torch.empty((len(pool), k), device=pool.device)
    for i in range(0, len(pool), 1024):
        q = pool[i:i+1024]
        d = torch.addmm(square[None], q, pool.T, alpha=-2.0).add_(q.square().sum(1, keepdim=True)).clamp_min_(0).sqrt_()
        d.masked_fill_(d < 1e-4, torch.inf)
        top = d.topk(k, dim=1, largest=False); out[i:i+1024] = top.indices; dist[i:i+1024] = top.values
    return out, dist


def edge_costs(speed, pool, src, dst, points=9):
    cost = torch.empty(len(src), device=pool.device); w = torch.ones(points, device=pool.device); w[0] = w[-1] = .5; w /= (points-1)
    t = torch.linspace(0, 1, points, device=pool.device)
    for i in range(0, len(src), 65536):
        a, b = pool[src[i:i+65536]], pool[dst[i:i+65536]]
        x = a[:, None]+t[None, :, None]*(b-a)[:, None]
        with torch.no_grad():
            s = speed(x.reshape(-1, 6)).reshape(len(a), points)
        cost[i:i+65536] = torch.linalg.vector_norm(b-a, dim=1)*(w/s).sum(1)
    return cost


def main():
    p = argparse.ArgumentParser(); p.add_argument('--prepared', type=Path, required=True); p.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    p.add_argument('--labels', nargs='+'); p.add_argument('--output', type=Path, required=True); p.add_argument('--k', type=int, default=24)
    p.add_argument('--goals', type=int, default=64); p.add_argument('--probes', type=int, default=256); a = p.parse_args()
    started = time.time(); source = np.load(a.prepared/'bank/source_q.npy'); pool = torch.as_tensor(source, device='cuda:0')
    speed = load_speed(a.prepared/'speed.pt', 'cuda:0')
    index, dist = knn(pool, a.k); src = torch.arange(len(pool), device='cuda:0').repeat_interleave(a.k); dst = index.reshape(-1)
    cost = edge_costs(speed, pool, src, dst).cpu().numpy(); src, dst = src.cpu().numpy(), dst.cpu().numpy()
    graph = csr_matrix((cost, (src, dst)), shape=(len(pool), len(pool)))   # directed kNN lists; Dijkstra below walks them both ways
    rng = np.random.default_rng(20261027); goals = rng.choice(len(pool), a.goals, replace=False)
    D, pred = dijkstra(graph, directed=False, indices=goals, return_predecessors=True)
    E = dijkstra(csr_matrix((dist.reshape(-1).cpu().numpy().astype(np.float64), (src, dst)), shape=(len(pool), len(pool))), directed=False, indices=goals)
    pairs = []
    for gi, g in enumerate(goals):
        reach = np.flatnonzero(np.isfinite(D[gi]) & (np.arange(len(pool)) != g)); gap = np.linalg.norm(source[reach]-source[g], axis=1)
        near = reach[gap <= .7]; far = reach
        pick = np.concatenate([rng.choice(near, min(len(near), a.probes//2), replace=False), rng.choice(far, a.probes//2, replace=False)])
        for q in pick:
            ahead = q
            for _ in range(3):
                if pred[gi, ahead] < 0: break
                ahead = pred[gi, ahead]
            pairs.append((q, g, gi, ahead))
    pairs = np.array(pairs); qi, gidx, gi, ahead = pairs.T
    truth = D[gi, qi]; reference = truth/E[gi, qi]; q = pool[qi]; g = pool[gidx]; gap = torch.linalg.vector_norm(q-g, dim=1).cpu().numpy()
    step = pool[ahead]-q; step = step/torch.linalg.vector_norm(step, dim=1, keepdim=True).clamp_min(1e-9)
    line = (g-q)/torch.linalg.vector_norm(g-q, dim=1, keepdim=True)
    which = np.digitize(gap, BINS[1:-1])
    result = dict(k=a.k, goals=a.goals, nodes=len(pool), reached_fraction=float(np.isfinite(D).mean()), pairs=len(pairs), bins=BINS,
                  counts=[int((which == b).sum()) for b in range(len(BINS)-1)], graph_seconds=time.time()-started,
                  truth_ratio=[float(np.median(truth[which == b]/gap[which == b])) if (which == b).any() else None for b in range(len(BINS)-1)],
                  reference_ratio=[float(np.median(reference[which == b])) if (which == b).any() else None for b in range(len(BINS)-1)],
                  euclid_cosine=[float((line*step).sum(1).cpu().numpy()[which == b].mean()) if (which == b).any() else None for b in range(len(BINS)-1)], fields={})
    for label, path in zip(a.labels or [str(c) for c in a.checkpoints], a.checkpoints):
        net = load_checkpoint(path, 'cuda:0'); values = []; cos = []
        for i in range(0, len(q), 4096):
            x = q[i:i+4096].clone().requires_grad_(True); v = net(x, g[i:i+4096])[:, 0]; grad = torch.autograd.grad(v.sum(), x)[0]
            values.append(v.detach()); cos.append((-grad/torch.linalg.vector_norm(grad, dim=1, keepdim=True).clamp_min(1e-12)*step[i:i+4096]).sum(1).detach())
        value = torch.cat(values).cpu().numpy(); cos = torch.cat(cos).cpu().numpy(); row = dict(ratio=[], spearman=[], cosine=[], reference_over_ratio=[])
        for b in range(len(BINS)-1):
            sel = which == b
            if sel.sum() < 10:
                for k in row: row[k].append(None)
                continue
            ratio = value[sel]/gap[sel]; row['ratio'].append(float(np.median(ratio)))
            row['spearman'].append(float(spearmanr(ratio, reference[sel]).statistic)); row['cosine'].append(float(cos[sel].mean()))
            row['reference_over_ratio'].append(float(np.median(reference[sel]/ratio)))
        result['fields'][label] = row; print(label, json.dumps(row), flush=True)
    write(a.output, result)


if __name__ == '__main__': main()
