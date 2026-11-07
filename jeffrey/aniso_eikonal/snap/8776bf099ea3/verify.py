"""Does the idea hold? Checks on ground truth (no network, no training) on one gap maze and its constructed warp.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/aniso_eikonal/verify.py --prepared <cross_warp gapprepare run> --out <dir>

Two exact travel-time fields on maze_scale's C-space grid (1/15 cell, 13 headings; Dijkstra in C):
  T*   the maze's travel time (cross_warp/truthfast.FastGrid): what a planner should minimize.
  Tw   the warped maze's travel time (cross_warp/warptruth.WarpGrid): the same speeds, every length measured in the warp's
       coordinates z. This is what 'training on the warped maze' (|grad_z H| = 1 / S) is asked to learn.

1. metric     aniso.metric_diag against the autograd Jacobian of the warp.
2. equations  at random grid nodes (central differences, by region): S |grad_x T*| (the maze's eikonal equation, the
              numerical reference); S sqrt(grad_z T*^T M grad_z T*), the same field differentiated on the warp's own node
              coordinates with the inner product M = J J^T (must give the same number: the anisotropic equation is the
              maze's equation); S |grad_z (m T*)|, the plain length in z in the quadrants' time unit (1 where the warp is
              a uniform magnification, off elsewhere: T* does not solve the warped maze's equation); and the mirror for Tw.
              Also the angle between the steepest descent of T* in z (carried back to the maze, J^-1 grad_z T*) and the
              fastest direction (-grad_x T* = -J^T grad_z T*): where J is not a multiple of the identity they differ.
3. the price  per query of the prepared set (60 per region pair): Tw(start, goal) / m against T* (the warped maze's time
              read as the maze's), and the maze travel time of the warped maze's own shortest path over T*. The second is
              the most a perfect planner on the warped maze loses on the maze; the certified true paths' time over T* is
              the grid's own error, for reference.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = next((q for q in HERE.parents if (q / 'baselines/ntrl-demo').is_dir()), Path(os.environ.get('NTFIELD_ROOT', '/home/jeffrey/bellman-ntfield')))
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
if not (HERE / 'warp.py').exists():
    sys.path.insert(0, str(ROOT / 'jeffrey/cross_warp'))
sys.path.insert(0, str(HERE))


@torch.no_grad()
def descend_coords(grid, fields, gi, start, goal, cx, ct, max_steps=200000):
    """truthfast.FastGrid.descend with every edge length measured in per-axis node coordinates cx (x and y nodes) and ct
    (heading nodes), and grid.seed_block as the grid defines it. With cx = grid.xs, ct = grid.ts it is FastGrid.descend."""
    dev = start.device
    nx, ny, nt = grid.shape
    P = len(start)
    flat = fields.reshape(len(fields), -1)
    k, cost = grid.seed_block(start)
    tot = cost + flat[gi[:, None].expand_as(cost), grid.flat(k)]
    ar = torch.arange(P, device=dev)
    cur = k[ar, tot.argmin(1)]
    gk, _ = grid.seed_block(goal)
    trail = [cur]
    active = torch.ones(P, dtype=torch.bool, device=dev)
    for _ in range(max_steps):
        near = (cur[:, None] == gk).all(-1).any(1)
        active = active & ~near
        if not bool(active.any()):
            break
        nb = cur[:, None] + grid.off_t[None]
        ok = (nb >= 0).all(-1) & (nb[..., 0] < nx) & (nb[..., 1] < ny) & (nb[..., 2] < nt)
        nbc = torch.stack([nb[..., 0].clamp(0, nx - 1), nb[..., 1].clamp(0, ny - 1), nb[..., 2].clamp(0, nt - 1)], -1)
        d = torch.stack([cx[nbc[..., 0]] - cx[cur[:, 0]][:, None], cx[nbc[..., 1]] - cx[cur[:, 1]][:, None], ct[nbc[..., 2]] - ct[cur[:, 2]][:, None]], -1)
        edge = torch.linalg.vector_norm(d, dim=-1) * 0.5 * (grid.inv[grid.flat(cur)][:, None] + grid.inv[grid.flat(nbc)])
        c = edge + flat[gi[:, None].expand(-1, len(grid.offsets)), grid.flat(nbc)]
        c = torch.where(ok, c, torch.full_like(c, float('inf')))
        nxt = nbc[ar, c.argmin(1)]
        cur = torch.where(active[:, None], nxt, cur)
        trail.append(cur)
    trail = torch.stack(trail, 1)
    paths = []
    for i in range(P):
        t = trail[i]
        keep = torch.ones(len(t), dtype=torch.bool, device=dev)
        keep[1:] = (t[1:] != t[:-1]).any(-1)
        paths.append(torch.cat([start[i:i + 1], grid.node(t[keep]), goal[i:i + 1]]))
    return paths, int(active.sum())


@torch.no_grad()
def node_gradients(grid, field, idx, cx, ct):
    """Central differences of one field (nx, nx, nt) at nodes idx (N, 3) on node coordinates cx, ct -> (N, 3)."""
    i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
    gx = (field[i + 1, j, k] - field[i - 1, j, k]) / (cx[i + 1] - cx[i - 1])
    gy = (field[i, j + 1, k] - field[i, j - 1, k]) / (cx[j + 1] - cx[j - 1])
    gt = (field[i, j, k + 1] - field[i, j, k - 1]) / (ct[k + 1] - ct[k - 1])
    return torch.stack([gx, gy, gt], 1)


def stats(v):
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    return dict(median=float(np.median(v)), p10=float(np.quantile(v, 0.1)), p90=float(np.quantile(v, 0.9)), n=int(len(v))) if len(v) else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--m-lo', type=float, default=None)
    p.add_argument('--eq-goals', type=int, default=6, help='goals for the equation residuals (spread over the goal list)')
    p.add_argument('--nodes', type=int, default=400_000, help='random grid nodes per goal for the residuals')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from common import Logger, dump
    log = Logger(a.out / 'log.txt')
    info = json.load(open(a.prepared / 'maze.json'))
    import gap as GP
    k, base = info['gap_scale'], GP.base_name(info)
    GP.install(k, base)
    import maze as MZ
    import plan as P
    import warp as W
    import truthfast as TF
    import warptruth as WT
    import aniso as AN
    dev = 'cuda:0'
    t0 = time.time()
    maze = MZ.Maze(info['n'], dev)
    assert np.array_equal(maze.V.astype(int), np.array(info['V'])) and np.array_equal(maze.H.astype(int), np.array(info['H'])), 'prepared maze differs'
    pres = json.load(open(a.prepared / 'results.json'))
    drawn = (pres.get('data') or {}).get('warp') or {}
    m_hi = 6 * maze.E / 22.1
    m_lo = a.m_lo if a.m_lo is not None else float(drawn.get('m_lo', 1.0))
    warp = W.constructed(maze, GP.BASE_WIDTH * k, m_hi, m_lo).to(dev)
    res = dict(stage='aniso-verify', maze=maze.describe(), gap_scale=k, gap_base=base, warp=dict(m_hi=m_hi, m_lo=m_lo), smoke=a.smoke, prepared=str(a.prepared))
    res['metric_check'] = AN.check_metric(warp, maze)
    log(f'gap maze k={k} base {base}; warp m_hi {m_hi:.4f} m_lo {m_lo:.5f}; M against autograd: {res["metric_check"]}')

    goals = torch.tensor(np.load(a.prepared / 'goals.npy'), device=dev)
    q = np.load(a.prepared / 'queries.npz')
    S, G = torch.tensor(q['start'], device=dev), torch.tensor(q['goal'], device=dev)
    gi = torch.tensor(q['goal_index'], device=dev).long()
    cat, t_star = q['category'], q['t_star']
    truth_paths = np.load(a.prepared / 'truth_paths.npy', allow_pickle=True)
    if a.smoke:
        keepg = torch.unique(gi)[:4]
        sel = torch.isin(gi, keepg).cpu().numpy()
        remap = {int(g): n for n, g in enumerate(keepg)}
        S, G, cat, t_star, truth_paths = S[sel], G[sel], cat[sel], t_star[sel], truth_paths[sel]
        gi = torch.tensor([remap[int(g)] for g in gi[torch.tensor(sel, device=dev)]], device=dev)
        goals = goals[keepg]
        a.eq_goals, a.nodes = 2, 50_000

    # 2. the equations
    fg = TF.FastGrid(maze, a.out / 'build', threads=a.threads)
    wg = WT.WarpGrid(maze, warp, a.out / 'build', threads=a.threads)
    eq_ids = torch.linspace(0, len(goals) - 1, a.eq_goals).round().long().to(dev)
    tq = time.time()
    Ts, _ = fg.solve(goals[eq_ids])
    Tw_eq, _ = wg.solve(goals[eq_ids])
    log(f'T* and Tw for {len(eq_ids)} goals: {time.time() - tq:.0f}s, grid {fg.shape}')
    nx, _, nt = fg.shape
    gen = torch.Generator(dev).manual_seed(5)
    rows = {key: [] for key in ('region', 'rx', 'raniso', 'riso', 'w_iso', 'w_aniso', 'angle', 'free')}
    for n_, g in enumerate(eq_ids):
        idx = torch.stack([torch.randint(2, nx - 2, (a.nodes,), generator=gen, device=dev), torch.randint(2, nx - 2, (a.nodes,), generator=gen, device=dev),
                           torch.randint(1, nt - 1, (a.nodes,), generator=gen, device=dev)], 1)
        node = fg.node(idx)
        far = torch.linalg.vector_norm((node - goals[g])[:, :2], dim=1) > 1.0 * maze.pitch          # away from the source's kink
        sp = fg.speed[idx[:, 0], idx[:, 1], idx[:, 2]]
        nb_ok = torch.ones(len(idx), dtype=torch.bool, device=dev)
        for ax in range(3):                                                # all six neighbours reachable and not blocked
            for sgn in (-1, 1):
                j = idx.clone()
                j[:, ax] += sgn
                nb_ok &= torch.isfinite(Ts[n_][j[:, 0], j[:, 1], j[:, 2]]) & (fg.speed[j[:, 0], j[:, 1], j[:, 2]] > 0.05)
        keep = far & nb_ok & (sp > 0.05)
        idx, node, sp = idx[keep], node[keep], sp[keep]
        gx = node_gradients(fg, Ts[n_], idx, fg.xs, fg.ts)
        gz = node_gradients(fg, Ts[n_], idx, wg.zx, wg.zt)
        gw = node_gradients(fg, Tw_eq[n_], idx, wg.zx, wg.zt)
        M = AN.metric_diag(warp, node)
        rows['region'].append(AN.region4(maze, GP.BLOCK, node))
        rows['rx'].append((sp * torch.linalg.vector_norm(gx, dim=1)).cpu().numpy())
        rows['raniso'].append((sp * torch.sqrt((gz * gz * M).sum(1))).cpu().numpy())
        rows['riso'].append((sp * m_hi * torch.linalg.vector_norm(gz, dim=1)).cpu().numpy())
        rows['w_iso'].append((sp * torch.linalg.vector_norm(gw, dim=1)).cpu().numpy())
        rows['w_aniso'].append((sp * torch.sqrt((gw * gw * M).sum(1)) / m_hi).cpu().numpy())
        v1 = torch.nn.functional.normalize(gz / torch.sqrt(M), dim=1)      # J^-1 grad_z T*  (steepest descent in z, on the maze)
        v2 = torch.nn.functional.normalize(gx, dim=1)                      # grad_x T* (the fastest direction, reversed)
        rows['angle'].append(torch.rad2deg(torch.acos((v1 * v2).sum(1).clamp(-1, 1))).cpu().numpy())
        rows['free'].append((sp > 0.999).cpu().numpy())
    rows = {key: np.concatenate(v) for key, v in rows.items()}
    eq = {}
    for r, name in enumerate(AN.REGIONS):
        mk = rows['region'] == r
        if not mk.any():
            continue
        eq[name] = dict(nodes=int(mk.sum()),
                        maze_equation_on_x=stats(rows['rx'][mk]), inner_product_on_z=stats(rows['raniso'][mk]), plain_length_on_z=stats(rows['riso'][mk]),
                        inner_product_minus_maze_max_abs=float(np.abs(rows['raniso'][mk] - rows['rx'][mk]).max()),
                        warped_truth_plain_length_on_z=stats(rows['w_iso'][mk]), warped_truth_inner_product=stats(rows['w_aniso'][mk]),
                        steepest_z_vs_fastest_deg=stats(rows['angle'][mk]), steepest_z_vs_fastest_deg_free=stats(rows['angle'][mk & rows['free']]))
    res['equations'] = dict(goals=len(eq_ids), nodes_per_goal=a.nodes, by_region=eq,
                            note='S x gradient norm at random grid nodes (central differences); 1 = the equation holds. plain_length_on_z is in the quadrants\' time unit (m T*).')
    for name, e in eq.items():
        log(f'{name}: maze eq. {e["maze_equation_on_x"]["median"]:.3f}, inner product on z {e["inner_product_on_z"]["median"]:.3f} (max |diff| {e["inner_product_minus_maze_max_abs"]:.2e}), '
            f'plain length on z {e["plain_length_on_z"]["median"]:.3f} [{e["plain_length_on_z"]["p10"]:.2f}, {e["plain_length_on_z"]["p90"]:.2f}]; warped truth: plain {e["warped_truth_plain_length_on_z"]["median"]:.3f}, '
            f'inner product {e["warped_truth_inner_product"]["median"]:.3f}; steepest-z vs fastest {e["steepest_z_vs_fastest_deg"]["median"]:.1f} deg')
    del Ts, Tw_eq
    torch.cuda.empty_cache()

    # 3. the price of planning on the warped maze with a perfect field
    tq = time.time()
    fields = []
    for i in range(0, len(goals), 2 * a.threads):
        f, _ = wg.solve(goals[i:i + 2 * a.threads])
        fields.append(f)
        log(f'Tw goals {i}-{i + len(f) - 1}: {time.time() - tq:.0f}s')
    Tw = torch.cat(fields)
    del fields
    tw_q = wg.query_time(Tw, gi, S).cpu().numpy() / m_hi
    ratio_T = tw_q / np.maximum(t_star, 1e-9)
    paths_w = []
    left = 0
    for i in range(0, len(S), 100):
        pw, l = descend_coords(wg, Tw, gi[i:i + 100], S[i:i + 100], G[i:i + 100], wg.zx, wg.zt)
        paths_w += pw
        left += l
    rw = P.score(maze, paths_w, [True] * len(paths_w))
    rt = P.score(maze, [torch.tensor(np.asarray(t), device=dev) for t in truth_paths], [True] * len(truth_paths))
    time_w = np.array([r['time'] for r in rw]) / np.maximum(t_star, 1e-9)
    time_t = np.array([r['time'] for r in rt]) / np.maximum(t_star, 1e-9)
    len_w = np.array([r['length_cells'] for r in rw]) / np.maximum(np.array([r['length_cells'] for r in rt]), 1e-9)
    names = GP.CATEGORIES
    res['price'] = dict(queries=len(S), unfinished_descents=left, warped_paths_colliding=int(sum(r['collision'] for r in rw)), seconds=time.time() - tq,
                        note='Tw / m over T*: the warped maze\'s travel time read as the maze\'s. path time: the maze travel time of the warped maze\'s shortest path '
                             '(grid descent of Tw) over T*; true path: the same for the certified true path (the grid\'s own error).',
                        all=dict(warped_time_over_true=stats(ratio_T), warped_path_time_over_true=stats(time_w), true_path_time_over_true=stats(time_t),
                                 warped_path_over_true_path=stats(time_w / time_t), warped_path_length_over_true=stats(len_w),
                                 share_over_5pct=float(((time_w / time_t) > 1.05).mean()), share_over_20pct=float(((time_w / time_t) > 1.2).mean()),
                                 worst=float((time_w / time_t).max())),
                        by_category={names[c]: dict(warped_time_over_true=stats(ratio_T[cat == c]), warped_path_time_over_true=stats(time_w[cat == c]),
                                                    true_path_time_over_true=stats(time_t[cat == c]), warped_path_over_true_path=stats((time_w / time_t)[cat == c]),
                                                    share_over_5pct=float(((time_w / time_t)[cat == c] > 1.05).mean()), worst=float((time_w / time_t)[cat == c].max()))
                                     for c in range(len(names)) if (cat == c).any()})
    for name, e in res['price']['by_category'].items():
        log(f'{name}: Tw / m over T* {e["warped_time_over_true"]["median"]:.3f} [{e["warped_time_over_true"]["p10"]:.3f}, {e["warped_time_over_true"]["p90"]:.3f}]; '
            f'warped path / true path time {e["warped_path_over_true_path"]["median"]:.3f} (p90 {e["warped_path_over_true_path"]["p90"]:.3f}, worst {e["worst"]:.3f}, > 5 %: {e["share_over_5pct"]:.2f})')
    np.savez(a.out / 'verify_rows.npz', category=cat, t_star=t_star, warped_time=tw_q, warped_path_time=time_w, true_path_time=time_t,
             eq_region=rows['region'], eq_rx=rows['rx'].astype(np.float32), eq_raniso=rows['raniso'].astype(np.float32), eq_riso=rows['riso'].astype(np.float32),
             eq_angle=rows['angle'].astype(np.float32), eq_free=rows['free'])
    order = np.argsort(-(time_w / time_t))[:8]
    ex = np.empty(len(order), dtype=object)
    ex[:] = [dict(query=int(i), category=int(cat[i]), ratio=float((time_w / time_t)[i]), warped=paths_w[i].cpu().numpy().astype(np.float32),
                  true=np.asarray(truth_paths[i], np.float32)) for i in order]
    np.save(a.out / 'example_paths.npy', ex, allow_pickle=True)
    import shutil
    shutil.rmtree(a.out / 'build', ignore_errors=True)
    res['elapsed_seconds'] = time.time() - t0
    res['complete'] = True
    dump(a.out / 'results.json', res)
    log(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
