"""Prepare a gap maze (gap.py: the 22 x 22 cross maze's four 8 x 8 quadrants pushed apart by a band of 6 k cells):
maze, ntrl-demo training pairs, ground truth and queries stratified by where their ends lie.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/cross_warp/gapprepare.py --scale 4 --out <dir>
    ... --scale 1 --source <maze_scale plus-prepare run>     (k = 1: a query set on that run's maze; train on the source run)

1. maze     gap.install(k); Maze(16 + 6 k); maze.obj / maze.json as maze_scale/prepare.py (variant 'gap').
2. data     ntrl-demo's point recipe (maze_scale/data.py, seed 0, 500,000 pairs) and 50,000 held-out pairs (seed 1);
            none with --source (the run is then a query set for fields trained on the source run).
3. goals    --goals poses with clearance >= 0.1 cells, stratified by region: goals / 8 in each quadrant, the rest in the
            open band (generator seed 7).
4. truth    T*(., goal) on maze_scale's C-space grid (1/15 cell, 13 headings) by truthfast.FastGrid (Dijkstra in C,
            the fixed point of truth.Grid's Jacobi sweeps).
5. queries  --candidates starts (clearance >= 0.1, seed 11); every (start, goal) pair gets a region-pair category
            (gap.CATEGORIES: same quadrant, adjacent quadrants, diagonal quadrants, quadrant - open, open - open) and its
            true travel distance; per category --quota pairs spread over the goals (at most ceil(quota / 8) per goal before
            relaxing), each certified by its truth path (collision-free under the exact scorer, prepare.certify).
            queries.npz: prepare.py's fields plus category, region_start, region_goal.
6. controls straight segment; oracle = the same MPPI on T* (planning seed 0), by category.

Open-pair control (mechanism test for the open band): ntrl-demo's recipe keeps x0 only within its wall margin (0.175
cells), so no training pair has both ends far from walls. --open-pairs N appends N pairs drawn by the same recipe with
x0 beyond the margin instead (speed 1 at x0; generator seed 2) to the unchanged seed-0 pairs. Local-pair control: the
recipe's step is a fixed share of the maze side, so as the maze grows fewer pairs stay inside one 8 x 8 quadrant (34 %
of pairs at 22 x 22, 7 % at 88 x 88); --local-pairs N appends N recipe pairs whose x, y step is scaled by 22.1 / E (the
22 x 22 maze's pair lengths in cells; seed 3). --queries-from <run>
reuses that gapprepare run's goals, queries, truth paths and controls (same maze, checked), so fields trained on the
two data sets are planned on the same queries; steps 3-6 are then skipped.

Pairs drawn on the warped maze (--z-draw): ntrl-demo's recipe run in the constructed warp's coordinates z = phi(x)
(warp.constructed: m_hi = 6 E / 22.1 over the quadrants and the band's edge cells, x 1 over the band's interior, heading
x m_hi; the gap study's design) instead of x: the recipe's draws are read as normalized coordinates of the warped box, so
the pairs are uniform over the warped maze, and each is mapped back by the exact inverse (warp.SepInverse) before the
recipe's rules (sample_z). Seeds 0 / 1 as the recipe; with --queries-from the same queries.

--base maze10: the gap mazes of gap.py's 'maze10' base (maze_scale's 10 x 10 maze cut into four 5 x 5 quadrants, the
same bands of 6 k cells; n = 10 + 6 k) instead of the 22 x 22 cross maze's.
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
sys.path.insert(0, str(HERE))


def sample_open(maze, n, seed, r_max=math.sqrt(3), batch=2_000_000, open_x0=True, xy_scale=1.0):
    """maze_scale/data.sample (ntrl-demo's pair recipe) with x0 kept beyond the wall margin (d(x0) >= margin, speed 1)
    instead of within it (open_x0; False: within it, as the recipe); the step's x, y components are multiplied by
    xy_scale (1: the recipe). x1, speed, normals and the stats as there."""
    from maze import MARGIN_CELLS, OFFSET_CELLS
    step_scale = torch.tensor([xy_scale, xy_scale, 1.0], device=maze.device)
    dev = maze.device
    gen = torch.Generator(dev).manual_seed(seed)
    scale = torch.tensor([1.0, 1.0, 2 * maze.phi_max], device=dev)
    xs, ys, ns = [], [], []
    drawn, total = 0, 0
    while total < n:
        v0 = torch.rand(batch, 3, generator=gen, device=dev) - 0.5
        dp = torch.rand(batch, 3, generator=gen, device=dev) - 0.5
        rl = torch.rand(batch, 1, generator=gen, device=dev) * r_max
        v1 = v0 + torch.nn.functional.normalize(dp, dim=1) * rl * step_scale
        drawn += batch
        keep = (v1.abs() <= 0.5).all(1)
        p, q = v0[keep] * scale, v1[keep] * scale
        d0, n0 = maze.clearance_normal(p)
        keep = d0 >= MARGIN_CELLS if open_x0 else (d0 < MARGIN_CELLS) & (d0 > OFFSET_CELLS)
        p, q, d0, n0 = p[keep], q[keep], d0[keep], n0[keep]
        d1, n1 = maze.clearance_normal(q)
        keep = (d1 > -OFFSET_CELLS) & ~n0.isnan().any(1) & ~n1.isnan().any(1)
        xs.append(torch.cat([p, q], 1)[keep].cpu())
        ys.append(torch.stack([d0, d1], 1)[keep].cpu())
        ns.append(torch.cat([n0, n1], 1)[keep].cpu())
        total += int(keep.sum())
    x = torch.cat(xs)[:n].numpy().astype(np.float32)
    d = torch.cat(ys)[:n].numpy()
    normal = torch.cat(ns)[:n].numpy().astype(np.float32)
    speed = np.clip(d / MARGIN_CELLS, OFFSET_CELLS / MARGIN_CELLS, 1).astype(np.float32)
    gap_cells = np.linalg.norm(x[:, :2] - x[:, 3:5], axis=1) / maze.pitch
    stats = dict(n=len(x), drawn=drawn, r_max=r_max, seed=seed, x0_rule='d(x0) >= margin' if open_x0 else 'offset < d(x0) < margin', xy_scale=xy_scale, x1_blocked=float((d[:, 1] <= OFFSET_CELLS).mean()),
                 x1_free=float((d[:, 1] >= MARGIN_CELLS).mean()), gap_cells_quantiles=np.quantile(gap_cells, [0.1, 0.5, 0.9]).tolist())
    return x, speed, normal, d, stats


def sample_z(maze, warp, n, seed, r_max=math.sqrt(3), batch=2_000_000, chunk=262_144):
    """maze_scale/data.sample (ntrl-demo's pair recipe) on the warped maze: the recipe's draws v0, v1 (the same generator
    calls) are normalized coordinates of z = phi(x) over the warped box phi([-0.5, 0.5]^2) for x and y (the heading map
    is linear, so its normalized coordinate is the recipe's), mapped back by the exact inverse x = phi^-1(z)
    (warp.SepInverse); the recipe's rules (offset < d(x0) < margin, d(x1) > -offset, finite normals) then apply on the
    real maze as there. So the pairs are uniform over the warped maze instead of the real one; with phi = id this is
    data.sample itself (check_sample_z)."""
    from maze import MARGIN_CELLS, OFFSET_CELLS
    import warp as W
    inv = W.SepInverse(warp)
    dev = maze.device
    with torch.no_grad():                         # the box's corners in z from the inverse's float64 copy of the warp
        corner = torch.tensor([[-0.5, -0.5, 0.0], [0.5, 0.5, 0.0]], device=dev, dtype=torch.float64)
        zlo, zhi = inv.w(corner)
    span = zhi - zlo

    def to_x(v):                                  # normalized z -> normalized x (heading unchanged)
        out = []
        for i in range(0, len(v), chunk):
            vi = v[i:i + chunk].double()
            z = torch.stack([zlo[0] + (vi[:, 0] + 0.5) * span[0], zlo[1] + (vi[:, 1] + 0.5) * span[1], torch.zeros_like(vi[:, 0])], 1)
            x = inv(z)
            out.append(torch.stack([x[:, 0].clamp(-0.5, 0.5), x[:, 1].clamp(-0.5, 0.5), vi[:, 2]], 1).float())
        return torch.cat(out)

    gen = torch.Generator(dev).manual_seed(seed)
    scale = torch.tensor([1.0, 1.0, 2 * maze.phi_max], device=dev)
    xs, ys, ns = [], [], []
    drawn, total = 0, 0
    while total < n:
        v0 = torch.rand(batch, 3, generator=gen, device=dev) - 0.5
        dp = torch.rand(batch, 3, generator=gen, device=dev) - 0.5
        rl = torch.rand(batch, 1, generator=gen, device=dev) * r_max
        v1 = v0 + torch.nn.functional.normalize(dp, dim=1) * rl
        drawn += batch
        keep = (v1.abs() <= 0.5).all(1)
        p, q = to_x(v0[keep]) * scale, to_x(v1[keep]) * scale
        d0, n0 = maze.clearance_normal(p)
        keep = (d0 < MARGIN_CELLS) & (d0 > OFFSET_CELLS)
        p, q, d0, n0 = p[keep], q[keep], d0[keep], n0[keep]
        d1, n1 = maze.clearance_normal(q)
        keep = (d1 > -OFFSET_CELLS) & ~n0.isnan().any(1) & ~n1.isnan().any(1)
        xs.append(torch.cat([p, q], 1)[keep].cpu())
        ys.append(torch.stack([d0, d1], 1)[keep].cpu())
        ns.append(torch.cat([n0, n1], 1)[keep].cpu())
        total += int(keep.sum())
    x = torch.cat(xs)[:n].numpy().astype(np.float32)
    d = torch.cat(ys)[:n].numpy()
    normal = torch.cat(ns)[:n].numpy().astype(np.float32)
    speed = np.clip(d / MARGIN_CELLS, OFFSET_CELLS / MARGIN_CELLS, 1).astype(np.float32)
    gap_cells = np.linalg.norm(x[:, :2] - x[:, 3:5], axis=1) / maze.pitch
    stats = dict(n=len(x), drawn=drawn, r_max=r_max, seed=seed, drawn_on='warped maze (z = phi(x))', z_box=[zlo[:2].tolist(), zhi[:2].tolist()],
                 x1_blocked=float((d[:, 1] <= OFFSET_CELLS).mean()), x1_free=float((d[:, 1] >= MARGIN_CELLS).mean()),
                 gap_cells_quantiles=np.quantile(gap_cells, [0.1, 0.5, 0.9]).tolist())
    return x, speed, normal, d, stats


def pair_regions(maze, x):
    """Where training pairs (N, 6) lie: shares with x0 in a quadrant, x1 in a quadrant, both ends in the same quadrant."""
    import gap
    c0 = x[:, :2] * maze.E + maze.n / 2
    c1 = x[:, 3:5] * maze.E + maze.n / 2
    r0, r1 = gap.region_of(maze.n, c0[:, 0], c0[:, 1]), gap.region_of(maze.n, c1[:, 0], c1[:, 1])
    return dict(x0_quadrant=float((r0 < 4).mean()), x1_quadrant=float((r1 < 4).mean()), same_quadrant=float(((r0 < 4) & (r0 == r1)).mean()))


def check_sample_z(maze, n=20_000):
    """sample_z with a x 1 constructed warp (phi = id) against data.sample: same pairs (largest difference)."""
    import data as D
    import warp as W
    import gap
    x0, *_ = D.sample(maze, n, 5)
    x1, *_ = sample_z(maze, W.constructed(maze, gap.BASE_WIDTH * gap._installed['k'], 1.0, 1.0), n, 5)
    m = min(len(x0), len(x1))
    return float(np.abs(x0[:m] - x1[:m]).max())


def map_from_k1(maze, k, pts):
    """Poses (N, 3) normalized on the 22 x 22 cross maze -> the gap maze of scale k: along x and along y, cells c < 9
    unchanged, the band's interior [9, 13] stretched linearly onto [9, 6 k + 7], c > 13 shifted by 6 (k - 1); the
    heading angle kept. The quadrants and the band's edge cells (one cell inside its outline, where the constructed warp
    still magnifies) keep every pose at the same place relative to its walls, so in the constructed warp's coordinates
    a mapped pair there is exactly the 22 x 22 pair; only the band's interior is wider."""
    E1, n1 = 22.1, 22
    c = pts[:, :2].astype(np.float64) * E1 + n1 / 2
    f = (6 * k - 2) / 4
    out = np.where(c < 9, c, np.where(c <= 13, 9 + (c - 9) * f, c + 6 * (k - 1)))
    return np.concatenate([(out - maze.n / 2) / maze.E, pts[:, 2:3].astype(np.float64) * E1 / maze.E], 1).astype(np.float32)


def mapped_pairs(maze, k, pts, chunk=262144):
    """The 22 x 22 pairs (N, 6) mapped by map_from_k1, relabelled on the gap maze (clearance, speed, normals) and kept
    where the recipe's rule still holds (offset < d(x0) < margin, d(x1) > -offset, finite normals)."""
    from maze import MARGIN_CELLS, OFFSET_CELLS
    a, b = map_from_k1(maze, k, pts[:, :3]), map_from_k1(maze, k, pts[:, 3:6])
    xs, ds, ns = [], [], []
    for i in range(0, len(a), chunk):
        p, q = torch.tensor(a[i:i + chunk], device=maze.device), torch.tensor(b[i:i + chunk], device=maze.device)
        d0, n0 = maze.clearance_normal(p)
        d1, n1 = maze.clearance_normal(q)
        keep = (d0 < MARGIN_CELLS) & (d0 > OFFSET_CELLS) & (d1 > -OFFSET_CELLS) & ~n0.isnan().any(1) & ~n1.isnan().any(1)
        xs.append(torch.cat([p, q], 1)[keep].cpu())
        ds.append(torch.stack([d0, d1], 1)[keep].cpu())
        ns.append(torch.cat([n0, n1], 1)[keep].cpu())
    x = torch.cat(xs).numpy().astype(np.float32)
    d = torch.cat(ds).numpy()
    normal = torch.cat(ns).numpy().astype(np.float32)
    speed = np.clip(d / MARGIN_CELLS, OFFSET_CELLS / MARGIN_CELLS, 1).astype(np.float32)
    gap_cells = np.linalg.norm(x[:, :2] - x[:, 3:5], axis=1) / maze.pitch
    stats = dict(n=len(x), source_pairs=len(pts), kept=len(x) / len(pts), x1_blocked=float((d[:, 1] <= OFFSET_CELLS).mean()),
                 x1_free=float((d[:, 1] >= MARGIN_CELLS).mean()), gap_cells_quantiles=np.quantile(gap_cells, [0.1, 0.5, 0.9]).tolist())
    return x, speed, normal, d, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scale', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source', type=Path, default=None)
    p.add_argument('--n-data', type=int, default=500_000)
    p.add_argument('--open-pairs', type=int, default=0, help='extra pairs with x0 beyond the wall margin (open space), seed 2')
    p.add_argument('--local-pairs', type=int, default=0, help='extra recipe pairs (x0 near a wall) with the x, y step scaled to the 22 x 22 maze (x 22.1 / E), seed 3')
    p.add_argument('--queries-from', type=Path, default=None, help="reuse this gapprepare run's goals, queries, truth paths and controls")
    p.add_argument('--data-from', type=Path, default=None, help="the 22 x 22 cross maze's prepare run: its training and held-out pairs mapped into this "
                   "maze (map_from_k1) instead of new ones (needs --queries-from)")
    p.add_argument('--map-queries', type=Path, default=None, help="the k = 1 region query set (gapprepare --scale 1): its goals, queries and truth paths "
                   "mapped into this maze by map_from_k1 (a query set only: no data, no truth fields, t_star unknown)")
    p.add_argument('--base', choices=['cross22', 'maze10'], default='cross22', help="gap.py's base maze")
    p.add_argument('--z-draw', action='store_true', help='draw the training and held-out pairs on the warped maze (sample_z; the constructed warp)')
    p.add_argument('--z-m-hi', type=float, default=None, help='--z-draw: the warp\'s magnification (default 6 E / 22.1, the gap study\'s)')
    p.add_argument('--z-m-lo', type=float, default=1.0, help="--z-draw: the warp's density over the band's interior (1: the gap study's; < 1 shrinks it)")
    p.add_argument('--check-z', action='store_true', help='--z-draw: first check sample_z against the recipe with phi = id')
    p.add_argument('--goals', type=int, default=48)
    p.add_argument('--quota', type=int, default=60)
    p.add_argument('--candidates', type=int, default=200_000)
    p.add_argument('--threads', type=int, default=12)
    p.add_argument('--save-fields-gb', type=float, default=1.0, help='store truth_fields.npy (float16) only below this size')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from common import BINS, Logger, bin_labels, bin_of, dump, save_paths
    log = Logger(a.out / 'log.txt')
    import gap
    gap.install(a.scale, a.base)
    import maze as MZ
    import data as D
    import plan as P
    import prepare as PR
    import truthfast as TF
    dev = 'cuda:0'
    t0 = time.time()
    if a.smoke:
        a.n_data, a.goals, a.quota, a.candidates = 20_000, 16, 4, 20_000
    n = gap.BLOCK * 2 + gap.BASE_WIDTH * a.scale
    maze = MZ.Maze(n, dev)
    res = dict(stage='gapprepare', maze=maze.describe(), scale=a.scale, base=a.base, smoke=a.smoke, gpu=torch.cuda.get_device_name(0), source=str(a.source) if a.source else None,
               categories=gap.CATEGORIES, quota=a.quota, query_clearance_cells=PR.QUERY_CLEAR)

    if a.map_queries is not None:                   # the 22 x 22 query set carried into this maze (same queries in the warped space)
        assert a.base == 'cross22', '--map-queries maps the 22 x 22 cross maze'
        src = a.map_queries
        sres = json.load(open(src / 'results.json'))
        assert sres.get('stage') == 'gapprepare' and sres.get('scale') == 1 and sres.get('complete'), '--map-queries must be the finished k = 1 query set'
        q = np.load(src / 'queries.npz')
        tp = np.load(src / 'truth_paths.npy', allow_pickle=True)
        S, Gq = map_from_k1(maze, a.scale, q['start']), map_from_k1(maze, a.scale, q['goal'])
        paths = [map_from_k1(maze, a.scale, np.asarray(t)) for t in tp]
        clen = np.array([np.linalg.norm(np.diff(t, axis=0), axis=1).sum() for t in paths], np.float32)
        pcells = np.array([np.linalg.norm(np.diff(t[:, :2], axis=0), axis=1).sum() * maze.E for t in paths], np.float32)
        cS, cG = maze.clearance(torch.tensor(S, device=dev)), maze.clearance(torch.tensor(Gq, device=dev))
        okpaths = [bool((maze.clearance(P.densify(torch.tensor(t, device=dev), 0.005 * maze.pitch)) >= 0).all()) for t in paths]
        np.save(a.out / 'goals.npy', map_from_k1(maze, a.scale, np.load(src / 'goals.npy')))
        np.savez(a.out / 'queries.npz', start=S, goal=Gq, goal_index=q['goal_index'], t_star=np.full(len(S), np.nan, np.float32), d_cells=np.full(len(S), np.nan, np.float32),
                 bin=q['bin'], category=q['category'], region_start=q['region_start'], region_goal=q['region_goal'],
                 euclid_cells=np.linalg.norm((S[:, :2] - Gq[:, :2]) * maze.E, axis=1).astype(np.float32), path_cells=pcells, path_clen=clen, true_dir=q['true_dir'])
        arr = np.empty(len(paths), dtype=object)
        arr[:] = paths
        np.save(a.out / 'truth_paths.npy', arr, allow_pickle=True)
        dump(a.out / 'maze.json', dict(maze.describe(), V=maze.V.astype(int).tolist(), H=maze.H.astype(int).tolist(), boxes_cells=maze.boxes.tolist()))
        res.update(queries_from=str(src), queries_mapped_from=str(src), mapped_rule='map_from_k1', queries=len(S),
                   start_clearance_min=float(cS.min()), goal_clearance_min=float(cG.min()), mapped_truth_paths_collision_free=int(sum(okpaths)),
                   path_cells_median=float(np.median(pcells)), elapsed_seconds=time.time() - t0, complete=True)
        dump(a.out / 'results.json', res)
        log(f'k = 1 queries mapped into gap maze k={a.scale}: {len(S)} queries, start / goal clearance min {float(cS.min()):.3f} / {float(cG.min()):.3f} cells, '
            f'mapped truth paths collision-free {sum(okpaths)}/{len(paths)}, path length median {np.median(pcells):.1f} cells')
        return

    # 1-2. maze and data
    if a.source is not None:
        info = json.load(open(a.source / 'maze.json'))
        assert np.array_equal(maze.V.astype(int), np.array(info['V'])) and np.array_equal(maze.H.astype(int), np.array(info['H'])), 'source maze differs'
        res['data'] = dict(source=str(a.source), note='query set only: train on the source run (no data written here)')
    else:
        MZ.write_obj(a.out / 'maze.obj', maze.boxes)
        centre, extent = MZ.to_off_transform(MZ.read_obj_vertices(a.out / 'maze.obj'))
        assert abs(extent - maze.E) < 1e-5 and abs(centre[0] - n / 2) < 1e-5 and abs(centre[1] - n / 2) < 1e-5, (centre, extent)
        if a.data_from is not None:
            assert a.queries_from is not None and not a.open_pairs and not a.local_pairs and not a.z_draw and a.base == 'cross22', '--data-from needs --queries-from and no extra pairs'
            src = json.load(open(a.data_from / 'maze.json'))
            assert src['n'] == 22 and src.get('plus_width') == 6, '--data-from must be the 22 x 22 cross maze'
            x, speed, normal, d, stats = mapped_pairs(maze, a.scale, np.load(a.data_from / 'data/sampled_points.npy'))
            stats = dict(stats, mapped_from=str(a.data_from), rule='map_from_k1: quadrants and band edge cells unchanged, band interior [9, 13] -> [9, 6k + 7]')
        elif a.z_draw:
            import warp as W
            m_hi = a.z_m_hi or 6 * maze.E / 22.1
            zwarp = W.constructed(maze, gap.BASE_WIDTH * a.scale, m_hi, a.z_m_lo)
            if a.check_z:
                res['check_sample_z_max_diff'] = check_sample_z(maze)
                log(f'sample_z with phi = id vs the recipe: largest pose difference {res["check_sample_z_max_diff"]:.2e}')
            x, speed, normal, d, stats = sample_z(maze, zwarp, a.n_data, 0)
            stats['warp'] = dict(kind='constructed (plusmag)', m_hi=m_hi, m_lo=a.z_m_lo, inset=1.0, band=gap.BASE_WIDTH * a.scale, s_p=m_hi)
        else:
            x, speed, normal, d, stats = D.sample(maze, a.n_data, 0)
        if a.open_pairs:
            xo, so, no, do, ostats = sample_open(maze, a.open_pairs, 2)
            x, speed, normal, d = np.concatenate([x, xo]), np.concatenate([speed, so]), np.concatenate([normal, no]), np.concatenate([d, do])
            stats = dict(stats, open=ostats, n_total=len(x))
            del xo, so, no, do
        if a.local_pairs:
            xo, so, no, do, lstats = sample_open(maze, a.local_pairs, 3, open_x0=False, xy_scale=22.1 / maze.E)
            x, speed, normal, d = np.concatenate([x, xo]), np.concatenate([speed, so]), np.concatenate([normal, no]), np.concatenate([d, do])
            stats = dict(stats, local=lstats, n_total=len(x))
            del xo, so, no, do
        stats['regions'] = pair_regions(maze, x)
        (a.out / 'data').mkdir(exist_ok=True)
        np.save(a.out / 'data/sampled_points.npy', x)
        np.save(a.out / 'data/speed.npy', speed)
        np.save(a.out / 'data/normal.npy', normal)
        res['data'] = stats
        if a.data_from is not None:
            hx, _, _, hd, _ = mapped_pairs(maze, a.scale, np.load(a.data_from / 'heldout/sampled_points.npy'))
        elif a.z_draw:
            hx, _, _, hd, _ = sample_z(maze, zwarp, 5_000 if a.smoke else 50_000, 1)
        else:
            hx, _, _, hd, _ = D.sample(maze, 5_000 if a.smoke else 50_000, 1)
        (a.out / 'heldout').mkdir(exist_ok=True)
        np.save(a.out / 'heldout/sampled_points.npy', hx)
        np.save(a.out / 'heldout/clearance_cells.npy', hd.astype(np.float32))
        log(f'data: {stats}')
        del x, speed, normal, d
    dump(a.out / 'maze.json', dict(maze.describe(), V=maze.V.astype(int).tolist(), H=maze.H.astype(int).tolist(), boxes_cells=maze.boxes.tolist()))
    log(f'gap maze k={a.scale}: n={n}, {len(maze.boxes)} walls, E={maze.E:g} cells, pitch {maze.pitch:.5f}')

    if a.queries_from is not None:                 # same goals, queries and controls as that run; only the data are new
        import shutil
        src = a.queries_from
        info = json.load(open(src / 'maze.json'))
        assert np.array_equal(maze.V.astype(int), np.array(info['V'])) and np.array_equal(maze.H.astype(int), np.array(info['H'])), 'queries-from maze differs'
        sres = json.load(open(src / 'results.json'))
        assert sres.get('stage') == 'gapprepare' and sres.get('scale') == a.scale and sres.get('complete'), 'queries-from must be a finished gapprepare run of the same scale'
        for f in ('goals.npy', 'queries.npz', 'truth_paths.npy'):
            shutil.copyfile(src / f, a.out / f)
        res['queries_from'] = str(src)
        for key in ('truth', 'queries', 'planner', 'straight', 'oracle'):
            if key in sres:
                res[key] = sres[key]
        res['elapsed_seconds'] = time.time() - t0
        res['complete'] = True
        dump(a.out / 'results.json', res)
        log(f'goals, queries, truth paths and controls copied from {src}; done in {(time.time() - t0) / 60:.1f} min')
        return

    def regions(q):
        x, y, _ = maze.to_cells(q)
        return gap.region_of(n, x.cpu().numpy(), y.cpu().numpy())

    # 3. goals, stratified by region
    gen = torch.Generator(dev).manual_seed(7)
    per_quad = a.goals // 8
    want = [per_quad] * 4 + [a.goals - 4 * per_quad]
    have = [[] for _ in range(5)]
    while any(len(h) < w for h, w in zip(have, want)):
        cand = maze.free(65_536, gen, PR.QUERY_CLEAR)
        rg = regions(cand)
        for r in range(5):
            need = want[r] - len(have[r])
            if need > 0:
                have[r] += list(cand[torch.tensor(rg == r, device=dev)][:need])
    goals = torch.stack([g for h in have for g in h])
    np.save(a.out / 'goals.npy', goals.cpu().numpy())

    # 4. truth
    tg = time.time()
    fg = TF.FastGrid(maze, a.out / 'build', threads=a.threads)
    fields = []
    for i in range(0, len(goals), 4 * a.threads):
        f, _ = fg.solve(goals[i:i + 4 * a.threads])
        fields.append(f)
        log(f'truth goals {i}-{i + len(f) - 1}: Dijkstra {time.time() - tg:.0f}s')
    fields = torch.cat(fields)
    size_gb = fields.numel() * 2 / 1e9
    if size_gb < a.save_fields_gb:
        np.save(a.out / 'truth_fields.npy', fields.half().cpu().numpy())
    res['truth'] = dict(grid_shape=list(fg.shape), hx=fg.hx, ht=fg.ht, solver='dijkstra_full.c', seconds=time.time() - tg,
                        unreachable_fraction=float((~torch.isfinite(fields)).float().mean()), saved=size_gb < a.save_fields_gb)

    # 5. queries by region-pair category
    starts = maze.free(a.candidates, torch.Generator(dev).manual_seed(11), PR.QUERY_CLEAR)
    G = len(goals)
    T = torch.empty(G, len(starts), device=dev)
    for g in range(G):
        for i in range(0, len(starts), 50_000):
            s = starts[i:i + 50_000]
            T[g, i:i + len(s)] = fg.query_time(fields, torch.full((len(s),), g, device=dev, dtype=torch.long), s)
    rs, rgl = regions(starts), regions(goals)
    cat = gap.category(rs[None, :].repeat(G, 0), rgl[:, None].repeat(len(starts), 1))
    Tn = T.cpu().numpy()
    valid = np.isfinite(Tn) & (Tn / maze.pitch >= 0.5)
    rng = np.random.default_rng(3)
    picked, available = [], {}
    for c, name in enumerate(gap.CATEGORIES):
        gi, si = np.nonzero(valid & (cat == c))
        available[name] = int(len(gi))
        if not len(gi):
            continue
        order = rng.permutation(len(gi))[:400_000]
        cap = math.ceil(a.quota / 8)
        want_n = min(int(1.5 * a.quota) + 4, len(order))
        chosen, used, per_goal = [], set(), {}
        while len(chosen) < want_n:
            for k in order:
                if len(chosen) >= want_n:
                    break
                g = int(gi[k])
                if k not in used and per_goal.get(g, 0) < cap:
                    chosen.append(int(k))
                    used.add(int(k))
                    per_goal[g] = per_goal.get(g, 0) + 1
            cap *= 2
        picked += [(int(gi[k]), int(si[k]), c) for k in chosen]
    picked = np.array(picked)
    gi_t = torch.tensor(picked[:, 0], device=dev)
    S, Gq = starts[torch.tensor(picked[:, 1], device=dev)], goals[gi_t]
    tpaths, ok, plen, clen = [], [], [], []
    for i in range(0, len(picked), 128):
        pt, o, pl, cl = PR.certify(maze, fg, fields, gi_t[i:i + 128], S[i:i + 128], Gq[i:i + 128])
        tpaths += pt
        ok.append(o)
        plen.append(pl)
        clen.append(cl)
    ok, plen, clen = np.concatenate(ok), np.concatenate(plen), np.concatenate(clen)
    keep = []
    for c in range(len(gap.CATEGORIES)):
        keep += [i for i in np.nonzero(picked[:, 2] == c)[0] if ok[i]][:a.quota]
    keep = np.array(sorted(keep, key=lambda i: (picked[i, 2], i)))
    kq = torch.tensor(keep, device=dev)
    S, Gq, gi_t = S[kq], Gq[kq], gi_t[kq]
    t_star = T[gi_t, torch.tensor(picked[keep, 1], device=dev)]
    d_cells = (t_star / maze.pitch).cpu().numpy()
    true_dir = torch.nn.functional.normalize(-fg.gradient(fields, gi_t, S), dim=1)
    euclid = torch.linalg.vector_norm(S[:, :2] - Gq[:, :2], dim=1) / maze.pitch
    category = picked[keep, 2]
    np.savez(a.out / 'queries.npz', start=S.cpu().numpy(), goal=Gq.cpu().numpy(), goal_index=gi_t.cpu().numpy(), t_star=t_star.cpu().numpy(),
             d_cells=d_cells, bin=bin_of(d_cells), category=category, region_start=regions(S), region_goal=regions(Gq),
             euclid_cells=euclid.cpu().numpy(), path_cells=plen[keep], path_clen=clen[keep], true_dir=true_dir.cpu().numpy())
    save_paths(a.out / 'truth_paths.npy', [tpaths[i] for i in keep])
    res['queries'] = dict(available_pairs=available, picked=int(len(picked)), certified=int(ok.sum()), kept=int(len(keep)),
                          per_category={name: int((category == c).sum()) for c, name in enumerate(gap.CATEGORIES)},
                          d_cells_by_category={name: [float(np.quantile(d_cells[category == c], q)) for q in (0.1, 0.5, 0.9)]
                                               for c, name in enumerate(gap.CATEGORIES) if (category == c).any()},
                          per_bin={l: int((bin_of(d_cells) == b).sum()) for b, l in enumerate(bin_labels()) if (bin_of(d_cells) == b).any()})
    log(f'queries: {res["queries"]}')

    # 6. controls
    st = P.settings(maze)
    budget = torch.tensor(np.maximum(st['min_steps'], np.ceil(st['budget_factor'] * clen[keep] / st['step'])), device=dev)
    res['planner'] = dict(st, budget_median=float(budget.median()), budget_max=float(budget.max()))

    def by_cat(ok_rows):
        ok_rows = np.asarray(ok_rows)
        return {name: [int(ok_rows[category == c].sum()), int((category == c).sum())] for c, name in enumerate(gap.CATEGORIES)}

    rows = P.score(maze, P.straight(S, Gq), [True] * len(S))
    res['straight'] = dict(success=int(sum(r['success'] for r in rows)), by_category=by_cat([r['success'] for r in rows]))

    def oracle(pairs, ids):
        g = gi_t[ids].repeat_interleave(len(pairs) // len(ids))
        return fg.oracle_cost(fields, g, pairs[:, :3], goals)

    tp = time.time()
    paths, stop, conv = P.mppi(oracle, S, Gq, 0, budget, maze.lo, maze.hi, **st)
    rows = P.score(maze, [paths[i] for i in range(len(S))], conv)
    res['oracle'] = dict(success=int(sum(r['success'] for r in rows)), by_category=by_cat([r['success'] for r in rows]), seconds=time.time() - tp)
    log(f'straight: {res["straight"]}; oracle: {res["oracle"]}')
    res['elapsed_seconds'] = time.time() - t0
    res['complete'] = True
    dump(a.out / 'results.json', res)
    log(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
