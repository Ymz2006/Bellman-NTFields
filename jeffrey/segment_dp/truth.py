"""Ground-truth travel times for probe pairs of the pool, per distance segment (diagnostic; never selects a model).

T*(q, g) = min over paths of the integral of ds / S, S the frozen speed network: the quantity the eikonal residual
and the Bellman local cost both refer to. Built in two steps:
1. a 24-nearest-neighbour graph over the pool (edge cost: trapezoid integral of ds / S along the straight edge, 9
   points); Dijkstra from --goals seeded goals gives a route for every pair and its cost D (an upper bound: the
   graph zig-zags);
2. each route, resampled to --waypoints points by arc length, is optimized (Adam on the interior waypoints, path
   cost with 5 points per segment, re-resampled by arc length every 50 steps) and its cost re-measured densely (17
   points per segment); the straight line is measured the same way, and where it is within 10 % of the optimized
   route it is optimized too (the better of the two is kept). T* = min(D, optimized, straight) is the cost of a
   concrete path, an upper bound of the true optimum that is close to it when the optimizer converges.
Probe pairs: for every segment of --edges and every goal, up to --per-goal sources q with |q - g| in the segment.
Saved (truth.npz): q, g, gap, segment, T*, D, optimized and straight-line costs, the initial direction of the best
path at q (unit vector to the path point min(0.02, gap/4) ahead), goal index; truth.json: per-segment summary.
"""
import sys
sys.path.insert(0, '/clean/method'); sys.path.insert(0, '/clean/jeffrey/segment_dp')
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from common import write
from sampler import DUPLICATE, load_pool, pairwise
from eikonal import EDGES


def knn(pool, k):
    square = pool.square().sum(1); index = torch.empty((len(pool), k), dtype=torch.long, device=pool.device); dist = torch.empty((len(pool), k), device=pool.device)
    for i in range(0, len(pool), 256):
        d = pairwise(pool[i:i+256], pool, square); d.masked_fill_(d < DUPLICATE, torch.inf)   # no self loops
        top = d.topk(k, dim=1, largest=False); index[i:i+256] = top.indices; dist[i:i+256] = top.values
    return index, dist


def segment_cost(speed, a, b, points):
    """Trapezoid integral of ds / S along straight segments a -> b ([n, 6] each) with the given number of points."""
    t = torch.linspace(0, 1, points, device=a.device); w = torch.ones(points, device=a.device); w[0] = w[-1] = .5; w /= (points-1)
    x = a[:, None]+t[None, :, None]*(b-a)[:, None]
    s = speed(x.reshape(-1, 6)).reshape(len(a), points)
    return torch.linalg.vector_norm(b-a, dim=1)*(w/s).sum(1)


def path_cost(speed, x, points):
    n, w, _ = x.shape
    return segment_cost(speed, x[:, :-1].reshape(-1, 6), x[:, 1:].reshape(-1, 6), points).view(n, w-1).sum(1)


def resample(x, count):
    """Waypoints [n, w, 6] -> [n, count, 6] evenly spaced in arc length (endpoints kept)."""
    seg = torch.linalg.vector_norm(x[:, 1:]-x[:, :-1], dim=2); cum = torch.cat((seg.new_zeros((len(x), 1)), seg.cumsum(1)), 1)
    total = cum[:, -1:].clamp_min(1e-12); s = torch.linspace(0, 1, count, device=x.device)[None]*total
    k = torch.searchsorted(cum.contiguous(), s.contiguous()).clamp(1, x.shape[1]-1)
    lo, hi = cum.gather(1, k-1), cum.gather(1, k); f = ((s-lo)/(hi-lo).clamp_min(1e-12)).clamp(0, 1)[..., None]
    a = x.gather(1, (k-1)[..., None].expand(-1, -1, 6)); b = x.gather(1, k[..., None].expand(-1, -1, 6))
    return a+f*(b-a)


def optimize(speed, x, steps, lr):
    """Adam on the interior waypoints of x [n, w, 6] (endpoints fixed); per-path step size lr x (path length / w)."""
    x = x.clone(); n, w, _ = x.shape
    for start in range(0, steps, 50):
        length = torch.linalg.vector_norm(x[:, 1:]-x[:, :-1], dim=2).sum(1)
        scale = (lr*length/(w-1))[:, None, None]
        inner = x[:, 1:-1].clone().requires_grad_(True); m = torch.zeros_like(inner); v = torch.zeros_like(inner)
        for it in range(1, min(50, steps-start)+1):
            path = torch.cat((x[:, :1], inner, x[:, -1:]), 1)
            cost = path_cost(speed, path, 5); grad = torch.autograd.grad(cost.sum(), inner)[0]
            with torch.no_grad():
                m.mul_(.9).add_(.1*grad); v.mul_(.999).add_(.001*grad.square())
                decay = 0.5*(1+np.cos(np.pi*(start+it)/steps))
                inner -= decay*scale*(m/(1-.9**it))/((v/(1-.999**it)).sqrt()+1e-12)
                inner.clamp_(-.5, .5)
        with torch.no_grad():
            x = resample(torch.cat((x[:, :1], inner.detach(), x[:, -1:]), 1), w)
    return x


def main():
    p = argparse.ArgumentParser(); p.add_argument('--prepared', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--edges', default=EDGES); p.add_argument('--goals', type=int, default=128); p.add_argument('--per-goal', type=int, default=8)
    p.add_argument('--k', type=int, default=24); p.add_argument('--waypoints', type=int, default=20); p.add_argument('--steps', type=int, default=250)
    p.add_argument('--lr', type=float, default=0.05); p.add_argument('--chunk', type=int, default=256); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False); started = time.time(); edges = np.array([float(x) for x in a.edges.split(',')])
    if a.smoke: a.goals, a.per_goal, a.steps = 4, 2, 20
    pool, _, speed, pool_info = load_pool(a.prepared, 'cuda:0'); source = pool.cpu().numpy(); n = len(pool)
    index, dist = knn(pool, a.k); src = torch.arange(n, device='cuda:0').repeat_interleave(a.k); dst = index.reshape(-1)
    with torch.no_grad():
        cost = torch.cat([segment_cost(speed, pool[src[i:i+8192]], pool[dst[i:i+8192]], 9) for i in range(0, len(src), 8192)]).cpu().numpy()
    src, dst = src.cpu().numpy(), dst.cpu().numpy(); graph = csr_matrix((cost.astype(np.float64), (src, dst)), shape=(n, n))
    rng = np.random.default_rng(20261027); goals = rng.choice(n, a.goals, replace=False)
    D, pred = dijkstra(graph, directed=False, indices=goals, return_predecessors=True); graph_seconds = time.time()-started
    rows = []
    for gi, g in enumerate(goals):
        gap = np.linalg.norm(source-source[g], axis=1); seg = np.digitize(gap, edges[1:-1]); ok = np.isfinite(D[gi]) & (gap > 1e-4) & (gap < edges[-1])
        for s in range(len(edges)-1):
            cand = np.flatnonzero(ok & (seg == s))
            for q in rng.choice(cand, min(len(cand), a.per_goal), replace=False): rows.append((q, g, gi, s))
    rows = np.array(rows); qi, gidx, gi, seg = rows.T
    routes = []
    for q, g, k in zip(qi, gidx, gi):
        node, path = q, [q]
        while node != g: node = pred[k, node]; path.append(node)
        routes.append(source[path])
    x0 = torch.stack([resample(torch.as_tensor(r, device='cuda:0')[None], a.waypoints)[0] for r in routes])

    def initial_direction(x, gap):
        f = resample(x, 200); arc = torch.cat((f.new_zeros((len(f), 1)), torch.linalg.vector_norm(f[:, 1:]-f[:, :-1], dim=2).cumsum(1)), 1)
        ahead = torch.minimum(torch.full_like(gap, .02), gap/4)
        k = torch.searchsorted(arc.contiguous(), ahead[:, None].contiguous()).clamp(1, f.shape[1]-1)[:, 0]
        d = f[torch.arange(len(f)), k]-f[:, 0]; return d/torch.linalg.vector_norm(d, dim=1, keepdim=True).clamp_min(1e-12)

    best, opt_cost, line_cost, direction, refined = [], [], [], [], 0
    for s in range(0, len(x0), a.chunk):
        x = optimize(speed, x0[s:s+a.chunk], a.steps, a.lr)
        with torch.no_grad():
            c_opt = path_cost(speed, x, 17); line = resample(torch.stack((x[:, 0], x[:, -1]), 1), a.waypoints); c_line = path_cost(speed, line, 17)
        # where the straight line is competitive, also optimize from it (the graph route may sit in a worse homotopy)
        again = torch.nonzero(c_line < 1.1*c_opt).flatten()
        if len(again):
            y = optimize(speed, line[again], a.steps, a.lr)
            with torch.no_grad():
                c_y = path_cost(speed, y, 17); better = c_y < c_opt[again]
                x[again[better]] = y[better]; c_opt[again[better]] = c_y[better]; refined += int(better.sum())
        with torch.no_grad():
            c_graph = torch.as_tensor(D[gi[s:s+a.chunk], qi[s:s+a.chunk]], device='cuda:0', dtype=torch.float32)
            value, which = torch.stack((c_graph, c_opt, c_line), 1).min(1)
            gap = torch.linalg.vector_norm(x[:, -1]-x[:, 0], dim=1)
            # a graph route that beats both is followed by the optimized path's direction (it started from that route)
            direction.append(torch.where((which == 2)[:, None], initial_direction(line, gap), initial_direction(x, gap)))
            best.append(value); opt_cost.append(c_opt); line_cost.append(c_line)
        print(json.dumps(dict(done=min(s+a.chunk, len(x0)), of=len(x0), refined_from_straight=refined, seconds=round(time.time()-started))), flush=True)
    best, opt_cost, line_cost, direction = [torch.cat(v).cpu().numpy() for v in (best, opt_cost, line_cost, direction)]
    graph_cost = D[gi, qi]; gap = np.linalg.norm(source[qi]-source[gidx], axis=1)
    np.savez(a.output/'truth.npz', q=source[qi], g=source[gidx], gap=gap, segment=seg, truth=best, graph=graph_cost, optimized=opt_cost, straight=line_cost,
             direction=direction, goal_index=gidx, source_index=qi, edges=edges)
    summary = dict(edges=edges.tolist(), goals=a.goals, per_goal=a.per_goal, k=a.k, waypoints=a.waypoints, steps=a.steps, lr=a.lr, pairs=len(qi), refined_from_straight=refined, **pool_info,
                   graph_seconds=graph_seconds, seconds=time.time()-started, smoke=a.smoke, segments=[])
    for s in range(len(edges)-1):
        sel = seg == s
        if not sel.any(): summary['segments'].append(dict(segment=s+1, pairs=0)); continue
        summary['segments'].append(dict(segment=s+1, range=[float(edges[s]), float(edges[s+1])], pairs=int(sel.sum()),
            truth_over_gap=float(np.median(best[sel]/gap[sel])), graph_over_truth=float(np.median(graph_cost[sel]/best[sel])),
            straight_over_truth=float(np.median(line_cost[sel]/best[sel])), optimized_over_truth=float(np.median(opt_cost[sel]/best[sel])),
            straight_is_best=float(np.mean(line_cost[sel] <= best[sel]*1.0001))))
    write(a.output/'truth.json', summary); print(json.dumps(summary['segments']), flush=True)


if __name__ == '__main__': main()
