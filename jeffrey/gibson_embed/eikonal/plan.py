"""Planning evaluation: start/goal pairs, the P-NTFields bidirectional gradient planner (test/gib_plan.py: both end
points move by step * (-grad T / |grad T|^2) until they are within tol, at most max_iter steps), collision check of the
polyline on a fine occupancy grid, and path length against the grid geodesic."""
import math
import numpy as np
import torch


def choose_pairs(free_c, clear_c, labels_c, free_g, dist_fn, n_starts, margin_cells, seed, min_euclid=0.1, wall_ratio=1.5,
                 wall_euclid=(0.05, 0.3), max_candidates=200000, wall_per_start=3, log=print):
    """Pairs on the collision grid (res_c): both end points free with clearance >= margin_cells in the largest component.

    'uniform': random goal at Euclidean distance >= min_euclid; 'wall': goal whose geodesic / Euclidean ratio >= wall_ratio
    with Euclidean distance in wall_euclid (locally separated by walls). dist_fn(cells_g) -> (n, *shape_g) geodesic
    fields on the geodesic grid (res_g). Returns dict of arrays: start, goal (units), geodesic, euclid, kind."""
    rng = np.random.default_rng(seed)
    res_c, res_g = free_c.shape[0], free_g.shape[0]
    sizes = np.bincount(labels_c.ravel())[1:]
    main = int(np.argmax(sizes)) + 1
    ok = free_c & (clear_c >= margin_cells) & (labels_c == main)
    cand = np.stack(np.nonzero(ok), 1)
    cand_g = (cand * res_g) // res_c
    cand_g = cand_g[free_g[tuple(cand_g.T)]]
    cand = cand_g * res_c // res_g + (res_c // res_g) // 2  # back to collision-grid cells (centre of the geodesic cell)
    if len(cand) > max_candidates:
        cand = cand[rng.choice(len(cand), max_candidates, replace=False)]
    starts = cand[rng.choice(len(cand), n_starts, replace=False)]
    starts_g = (starts * res_g) // res_c
    dist = dist_fn(torch.from_numpy(starts_g))
    idx_g = tuple(torch.from_numpy(cand * res_g // res_c).T.to(dist.device))
    pos = lambda cells: (cells.astype(np.float32) + 0.5) / res_c - 0.5
    cand_pos = pos(cand)
    out = dict(start=[], goal=[], geodesic=[], euclid=[], kind=[])
    for i in range(n_starts):
        d = dist[i][idx_g].cpu().numpy()
        s = pos(starts[i:i + 1])[0]
        e = np.linalg.norm(cand_pos - s, axis=1)
        fin = np.isfinite(d) & (e > 0)
        sets = {'uniform': fin & (e >= min_euclid), 'wall': fin & (d / np.maximum(e, 1e-9) >= wall_ratio) & (e >= wall_euclid[0]) & (e <= wall_euclid[1])}
        for kind, sel in sets.items():
            idx = np.nonzero(sel)[0]
            if len(idx) == 0:
                continue
            take = min(len(idx), wall_per_start if kind == 'wall' else 1)
            for j in rng.choice(idx, take, replace=False):
                out['start'].append(s)
                out['goal'].append(cand_pos[j])
                out['geodesic'].append(d[j])
                out['euclid'].append(e[j])
                out['kind'].append(kind)
    out = {k: np.array(v) for k, v in out.items()}
    log(f'  pairs: {dict(zip(*np.unique(out["kind"], return_counts=True)))} from {n_starts} starts')
    return out


@torch.no_grad()
def plan(direction_fn, start, goal, step=0.03, tol=0.06, max_iter=500):
    """direction_fn(X (n, 2 dim)) -> (n, 2 dim) planner update. Returns trajectories (max_iter+1, n, dim) for both
    fronts, the number of steps each pair took, and whether the fronts met."""
    X = torch.cat([start, goal], 1).clone()
    n, dim = X.shape[0], X.shape[1] // 2
    traj0, traj1 = [X[:, :dim].clone()], [X[:, dim:].clone()]
    steps = torch.full((n,), max_iter, dtype=torch.long, device=X.device)
    active = torch.ones(n, dtype=torch.bool, device=X.device)
    for it in range(1, max_iter + 1):
        with torch.enable_grad():
            d = direction_fn(X[active])
        X[active] = (X[active] + step * d).clamp(-0.5, 0.5)
        traj0.append(X[:, :dim].clone())
        traj1.append(X[:, dim:].clone())
        gap = (X[:, dim:] - X[:, :dim]).norm(dim=1)
        done = active & (gap < tol)
        steps[done] = it
        active &= ~done
        if not active.any():
            break
    return torch.stack(traj0), torch.stack(traj1), steps, ~active


def path_points(traj0, traj1, steps, i):
    k = int(steps[i])
    return torch.cat([traj0[:k + 1, i], traj1[:k + 1, i].flip(0)], 0)


def collision_free(points, occ, spacing=0.5):
    """points (m, dim) in units; occ (res, ...) bool occupancy. Samples every segment at <= spacing cells."""
    res = occ.shape[0]
    seg = points[1:] - points[:-1]
    length = seg.norm(dim=1)
    n_sub = (length * res / spacing).ceil().long().clamp(min=1)
    t = torch.cat([torch.linspace(0, 1, int(k) + 1, device=points.device) for k in n_sub])
    base = torch.repeat_interleave(torch.arange(len(seg), device=points.device), n_sub + 1)
    samples = points[:-1][base] + t[:, None] * seg[base]
    cells = ((samples + 0.5) * res).floor().long().clamp(0, res - 1)
    return not bool(occ[tuple(cells.T)].any())


def evaluate(direction_fn, pairs, occ, device, step=0.03, tol=0.06, max_iter=500, log=print, label=''):
    start = torch.tensor(pairs['start'], device=device)
    goal = torch.tensor(pairs['goal'], device=device)
    traj0, traj1, steps, met = plan(direction_fn, start, goal, step, tol, max_iter)
    n = len(start)
    free_path = np.zeros(n, dtype=bool)
    length = np.zeros(n)
    for i in range(n):
        pts = path_points(traj0, traj1, steps, i)
        free_path[i] = collision_free(pts, occ)
        length[i] = float((pts[1:] - pts[:-1]).norm(dim=1).sum())
    met = met.cpu().numpy()
    success = met & free_path
    ratio = length / pairs['geodesic']
    out = dict(n=int(n))
    for kind in np.unique(pairs['kind']):
        sel = pairs['kind'] == kind
        s = success[sel]
        r = ratio[sel & success]
        out[kind] = dict(n=int(sel.sum()), success=float(s.mean()), met=float(met[sel].mean()), collision_among_met=float((~free_path[sel & met]).mean()) if (sel & met).any() else None,
                         length_ratio_mean=float(r.mean()) if len(r) else None, length_ratio_median=float(np.median(r)) if len(r) else None,
                         length_ratio_p90=float(np.percentile(r, 90)) if len(r) else None, steps_mean=float(steps.cpu().numpy()[sel].mean()))
        log(f"  {label} {kind}: success {out[kind]['success']:.3f} (met {out[kind]['met']:.3f}, collision among met {out[kind]['collision_among_met']}) "
            f"length/geodesic median {out[kind]['length_ratio_median']} p90 {out[kind]['length_ratio_p90']}")
    out['per_pair'] = dict(success=success.tolist(), met=met.tolist(), free=free_path.tolist(), length=length.tolist())
    return out
