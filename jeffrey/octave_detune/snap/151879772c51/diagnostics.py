"""Field diagnostics for the octave-detune runs (inference only; copies of the grid-sync / maze-scale measures so the
frozen snapshot is self-contained).

wall_measure   = jeffrey/maze_scale/wallfeel.measure: at near-wall x0 of the held-out pairs (trained speed S < 0.3), median
                 S |grad_x0 T| (eikonal target 1) and cos(grad T, -n) (normal target 1), by the pair's gap in cells.
segments, profiles, bumps = jeffrey/grid_sync/mechanism.py: learned speed across every interior grid-line segment
                 (doorway dip = phantom wall, wall S at 0.35 cells); local minima of T(x, goal) along the true paths of
                 queries >= 8 cells (densified to 0.02 cells).
accuracy       = jeffrey/grid_sync/report.accuracy: median |log(T / T*)| over the queries, all and by distance bin.
failures       = failures split by arrival (plan.mppi appends the goal to every path, so the collision flag alone
                 counts stalls as wall contacts): never arrived = ~converged; arrived after touching a wall =
                 converged & collision.
"""
import math
from pathlib import Path
import numpy as np
import torch
import plan as P

GAPS = [(0, 0.5), (0.5, 2), (2, 8), (8, 1e9)]
OFFSETS = np.round(np.linspace(-0.5, 0.5, 21), 3)
BINS = ['0.5-2', '2-4', '4-8', '8-16', '16-32', '32-64', '64-128', '128-256']


def wall_measure(maze, prepared, grad):
    hx = torch.tensor(np.load(prepared / 'heldout/sampled_points.npy'), device=maze.device)
    hd = torch.tensor(np.load(prepared / 'heldout/clearance_cells.npy'), device=maze.device)[:, 0]
    _, normal = maze.clearance_normal(hx[:, :3])
    S = maze.train_speed(hd)
    gap = torch.linalg.vector_norm(hx[:, :3] - hx[:, 3:], dim=1) / maze.pitch
    _, g = grad(hx)
    slope = S * torch.linalg.vector_norm(g, dim=1)
    cos = torch.nn.functional.cosine_similarity(g, -normal, dim=1)
    out = {}
    for lo, hi in GAPS:
        near = (gap >= lo) & (gap < hi) & (S < 0.3)
        if int(near.sum()) > 20:
            out[f'{lo:g}-{hi:g}' if hi < 1e8 else f'>{lo:g}'] = dict(n=int(near.sum()), slope=float(slope[near].median()), cos=float(cos[near].median()))
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
    res['doorway_dip_q10'] = float(dip.quantile(0.1))
    res['doorway_phantom_share'] = float((dip < 0.5).float().mean())
    sw = S[wall]
    res['wall_S_at_035'] = float(sw[:, k35].median())
    res['wall_label_at_035'] = float(label[wall][:, k35].median())
    res['wall_S_at_05'] = float(sw[:, edge].median())
    res['profile_doorway_median'] = sd.median(0).values.tolist()
    res['profile_wall_median'] = sw.median(0).values.tolist()
    return res


def bumps(maze, prepared, cost, min_bin=3):
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


def accuracy(prepared, rows_path):
    q = np.load(Path(prepared) / 'queries.npz')
    rows = np.load(rows_path)
    err = np.abs(np.log(np.maximum(rows['t_pred'], 1e-9) / q['t_star']))
    by = {BINS[b]: float(np.median(err[q['bin'] == b])) for b in sorted(set(q['bin'].tolist()))}
    return dict(log_err_median=float(np.median(err)), log_err_by_bin=by)


def failures(rows_path):
    r = np.load(rows_path)
    ok, conv, col = r['success'], r['converged'], r['collision']
    return dict(runs=int(ok.size), success=int(ok.sum()), never_arrived=int((~ok & ~conv).sum()),
                arrived_after_wall=int((~ok & conv & col).sum()), other=int((~ok & conv & ~col).sum()))
