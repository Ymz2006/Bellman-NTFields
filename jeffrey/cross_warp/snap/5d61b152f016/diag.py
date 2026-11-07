"""Field diagnostics for the cross-warp runs (inference only). near_wall is jeffrey/maze_scale/wallfeel.measure, and
segments / profiles / bumps are jeffrey/grid_sync/mechanism.py's, copied here so a run imports neither study's report
modules; where_plus and warp_stats are new.

* near_wall   at held-out near-wall x0 (trained speed S < 0.3): median S |grad_x0 T| (eikonal target 1) and
              cos(grad_x0 T, -n) (target 1: the field rises into the wall), by pair gap in cells.
* profiles    learned speed across every interior grid-line segment (doorways: a dip is a phantom wall; walls: speed at
              0.35 and 0.5 cells from the line), median over 8 prepared goals.
* bumps       local minima of T(x, goal) along the true paths of queries >= 8 cells (an MPPI run can stall in each).
* warp_stats  stretch of the warp: log det J_psi and the heading log-scale in the open pool, inside / outside the +,
              next to walls; singular values of J_psi; a stretch map at heading 0 (warp_map.npz).
"""
import math
from pathlib import Path
import numpy as np
import torch

import warp as W

OFFSETS = np.round(np.linspace(-0.5, 0.5, 21), 3)


def near_wall(maze, prepared, grad):
    hx = torch.tensor(np.load(prepared / 'heldout/sampled_points.npy'), device=maze.device)
    hd = torch.tensor(np.load(prepared / 'heldout/clearance_cells.npy'), device=maze.device)[:, 0]
    _, normal = maze.clearance_normal(hx[:, :3])
    S = maze.train_speed(hd)
    gap = torch.linalg.vector_norm(hx[:, :3] - hx[:, 3:], dim=1) / maze.pitch
    _, g = grad(hx)
    slope = S * torch.linalg.vector_norm(g, dim=1)
    cos = torch.nn.functional.cosine_similarity(g, -normal, dim=1)
    out = {}
    for lo, hi in [(0, 0.5), (0.5, 2), (2, 8), (8, 1e9)]:
        m = (gap >= lo) & (gap < hi) & (S < 0.3)
        if int(m.sum()) > 20:
            out[f'{lo:g}-{hi:g}' if hi < 1e8 else f'>{lo:g}'] = dict(n=int(m.sum()), slope=float(slope[m].median()), cos=float(cos[m].median()))
    return out


def segments(maze):
    out = []
    n = maze.n
    for i in range(1, n):
        for j in range(n):
            out.append((float(i), j + 0.5, 0.0, bool(maze.V[i, j])))
    for i in range(n):
        for j in range(1, n):
            out.append((i + 0.5, float(j), math.pi / 2, bool(maze.H[i, j])))
    return out


def profiles(maze, prepared, grad, goals_k=8):
    dev = maze.device
    goals = torch.tensor(np.load(prepared / 'goals.npy')[:goals_k], device=dev)
    seg = segments(maze)
    cx = torch.tensor([s[0] for s in seg], device=dev)
    cy = torch.tensor([s[1] for s in seg], device=dev)
    th = torch.tensor([s[2] for s in seg], device=dev)
    wall = torch.tensor([s[3] for s in seg], device=dev)
    u = torch.tensor(OFFSETS, device=dev, dtype=torch.float32)
    vert = (th == 0)[:, None]
    x = torch.where(vert, cx[:, None] + u[None], cx[:, None].expand(-1, len(u)))
    y = torch.where(vert, cy[:, None].expand(-1, len(u)), cy[:, None] + u[None])
    q = maze.from_cells(x.reshape(-1), y.reshape(-1), th[:, None].expand(-1, len(u)).reshape(-1))
    q[:, 2] = q[:, 2].clamp(-maze.phi_max, maze.phi_max)
    clear = maze.clearance(q).reshape(len(seg), len(u))
    label = maze.train_speed(clear).reshape(len(seg), len(u))
    S = []
    for g in goals:
        pairs = torch.cat([q, g[None].expand(len(q), 3)], 1)
        _, gr = grad(pairs)
        S.append(1 / torch.linalg.vector_norm(gr, dim=1).clamp(min=1e-6))
    S = torch.stack(S).median(0).values.reshape(len(seg), len(u))
    door = ~wall
    free_door = door & (label > 0.999).all(1)
    mid = len(u) // 2
    edge = [0, len(u) - 1]
    k35 = [int(np.argmin(np.abs(OFFSETS + 0.35))), int(np.argmin(np.abs(OFFSETS - 0.35)))]
    res = dict(doorways=int(door.sum()), doorways_free=int(free_door.sum()), walls=int(wall.sum()))
    sd = S[free_door]
    dip = sd[:, mid] / sd[:, edge].mean(1)
    res['doorway_dip_median'] = float(dip.median())
    res['doorway_phantom_share'] = float((dip < 0.5).float().mean())
    sw = S[wall]
    res['wall_S_at_035'] = float(sw[:, k35].median())
    res['wall_label_at_035'] = float(label[wall][:, k35].median())
    res['wall_S_at_05'] = float(sw[:, edge].median())
    return res


def bumps(maze, prepared, cost, P, min_bin=3):
    q = np.load(prepared / 'queries.npz')
    tp = np.load(prepared / 'truth_paths.npy', allow_pickle=True)
    counts = []
    for k in np.nonzero(q['bin'] >= min_bin)[0]:
        p = P.densify(torch.tensor(np.asarray(tp[k], np.float32), device=maze.device), 0.02 * maze.pitch)
        g = torch.tensor(q['goal'][k], device=maze.device, dtype=torch.float32).expand(len(p), 3)
        t = cost(torch.cat([p, g], 1)).cpu().numpy()
        dt = np.diff(t)
        keep = np.abs(dt) > 1e-6 * t[0]
        sgn = np.sign(dt[keep])
        counts.append(int(((sgn[:-1] < 0) & (sgn[1:] > 0)).sum()))
    return dict(paths=len(counts), bumps_per_path=float(np.mean(counts)) if counts else None, paths_with_bump=int(np.sum(np.array(counts) > 0)))


def where_plus(maze, x, width):
    """True per configuration (N, >= 2, normalized) inside the + (by cell)."""
    import plus as PL
    cells = PL.plus_cells(maze.n, width)
    c = np.clip(np.floor(x[:, 0] * maze.E + maze.n / 2).astype(int), 0, maze.n - 1)
    r = np.clip(np.floor(x[:, 1] * maze.E + maze.n / 2).astype(int), 0, maze.n - 1)
    return cells[c, r]


@torch.no_grad()
def warp_stats(maze, warp, free_pool, prepared, width, out_dir):
    dev = maze.device
    gen = torch.Generator(dev).manual_seed(5)
    u = maze.uniform(200_000, gen)
    u = u[maze.clearance(u) >= 0]
    out = {}
    lf, gf = warp.logdet(free_pool[:50_000])
    out['open'] = dict(logdet_median=float(lf.median()), heading_median=float(gf.median()))
    if width:
        inside = torch.tensor(where_plus(maze, u.cpu().numpy(), width), device=dev)
        for name, m in (('plus', inside), ('quadrants', ~inside)):
            l, g = warp.logdet(u[m])
            out[name] = dict(logdet_median=float(l.median()), logdet_mean=float(l.mean()), heading_median=float(g.median()), n=int(m.sum()))
    x0 = torch.tensor(np.load(prepared / 'data/sampled_points.npy', mmap_mode='r')[:50_000, :3].copy(), device=dev)
    l, g = warp.logdet(x0)
    out['near_wall'] = dict(logdet_median=float(l.median()), heading_median=float(g.median()))
    J = W.jacobian(warp, u[:20_000])
    sv = torch.linalg.svdvals(J[:, :2, :2] / warp.s)
    out['sv_xy'] = dict(min=float(sv.min()), q01=float(sv[:, 1].quantile(0.01)), q99=float(sv[:, 0].quantile(0.99)), max=float(sv.max()))
    res = 8 * maze.n + 1
    xs = torch.linspace(-0.5, 0.5, res, device=dev)
    X, Y = torch.meshgrid(xs, xs, indexing='ij')
    q = torch.stack([X.reshape(-1), Y.reshape(-1), torch.zeros(res * res, device=dev)], 1)
    ld, gp = warp.logdet(q)
    mapped = warp(q)
    np.savez(Path(out_dir) / 'warp_map.npz', logdet=ld.reshape(res, res).cpu().numpy(), heading=gp.reshape(res, res).cpu().numpy(),
             mapped=mapped.reshape(res, res, 3).cpu().numpy(), xs=xs.cpu().numpy())
    return out
