"""Prepare stage for an embed_warp scene (scene.py: disks, Gibson): jeffrey/maze_scale/prepare.py's steps 2-5 on the scene (ntrl-demo
training pairs, ground truth, distance-binned certified queries, straight and oracle controls), the same files, so every maze_scale
stage-2 pipeline (and embed_warp's) runs on it. maze.json holds the scene spec (scene.build rebuilds it) and the geometry preview.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/embed_warp/prepare.py --scene disks --out <dir>
    CUDA_VISIBLE_DEVICES=1 python jeffrey/embed_warp/prepare.py --scene gibson --idx 1 --h-cells 0.125 --out <dir>
    CUDA_VISIBLE_DEVICES=1 python jeffrey/embed_warp/prepare.py --from <prepare run> --inside-share 0.2 --out <dir>

--from: a data variant of an existing prepare run (any scene): the scene, goals, truth, queries and controls are copied, the training
pairs are redrawn with a share of pairs whose x1 lies INSIDE an obstacle (ntrl-demo's recipe keeps d(x1) > -offset, so its eikonal
never integrates the floor speed across a wall; the label there is the floor speed and the normal the oracle's clearance gradient).
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
sys.path.insert(0, str(HERE))
import scene as SC  # noqa: E402
import data as D  # noqa: E402
import truth as TR  # noqa: E402
import plan as P  # noqa: E402
from common import BINS, PLAN_SEEDS, Logger, bin_labels, by_bin, dump, save_paths  # noqa: E402

QUERY_CLEAR = 0.1


def certify(scene, grid, fields, gi, start, goal):
    paths = grid.descend(fields, gi, start, goal)
    ok, plen, clen = [], [], []
    for p in paths:
        dense = P.densify(p, 0.005 * scene.pitch)
        ok.append(bool((scene.clearance(dense) >= 0).all()))
        seg = p[1:] - p[:-1]
        plen.append(float(torch.linalg.vector_norm(seg[:, :2], dim=1).sum() / scene.pitch))
        clen.append(float(torch.linalg.vector_norm(seg, dim=1).sum()))
    return paths, np.array(ok), np.array(plen), np.array(clen)


def preview(scene, out, res=1024):
    """Occupancy of the robot centre at heading 0 (clearance < 0) as a PNG, for the report."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = torch.linspace(-0.5, 0.5, res, device=scene.device)
    X, Y = torch.meshgrid(t, t, indexing='ij')
    q = torch.stack([X.reshape(-1), Y.reshape(-1), torch.zeros(res * res, device=scene.device)], 1)
    d = scene.clearance(q).reshape(res, res).cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow((d.T < 0), origin='lower', cmap='Greys', extent=[-0.5, 0.5, -0.5, 0.5])
    ax.set_axis_off()
    fig.savefig(out / 'scene.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    np.save(out / 'clearance_heading0.npy', d.astype(np.float16))


def sample_inside(scene, n, seed, batch=2_000_000):
    """n pairs: x0 in ntrl-demo's band (offset < d0 < margin), x1 uniform in the C-space box with d(x1) < 0 (inside an obstacle)."""
    from maze import MARGIN_CELLS, OFFSET_CELLS
    dev = scene.device
    gen = torch.Generator(dev).manual_seed(seed)
    xs, ds, ns = [], [], []
    total = 0
    while total < n:
        v0 = scene.uniform(batch, gen)
        d0, n0 = scene.clearance_normal(v0)
        keep = (d0 < MARGIN_CELLS) & (d0 > OFFSET_CELLS) & ~n0.isnan().any(1)
        v0, d0, n0 = v0[keep], d0[keep], n0[keep]
        v1 = scene.uniform(batch, gen)
        d1, n1 = scene.clearance_normal(v1)
        keep = (d1 < 0) & ~n1.isnan().any(1)
        v1, d1, n1 = v1[keep], d1[keep], n1[keep]
        m = min(len(v0), len(v1))
        xs.append(torch.cat([v0[:m], v1[:m]], 1).cpu())
        ds.append(torch.stack([d0[:m], d1[:m]], 1).cpu())
        ns.append(torch.cat([n0[:m], n1[:m]], 1).cpu())
        total += m
    x = torch.cat(xs)[:n].numpy().astype(np.float32)
    d = torch.cat(ds)[:n].numpy()
    normal = torch.cat(ns)[:n].numpy().astype(np.float32)
    speed = np.clip(d / MARGIN_CELLS, OFFSET_CELLS / MARGIN_CELLS, 1).astype(np.float32)
    return x, speed, normal, d


def data_variant(a):
    import shutil
    src = a.from_run
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    log = Logger(out / 'log.txt')
    dev = 'cuda:0'
    t0 = time.time()
    scene, info = SC.load_scene(src, dev)
    for f in ('maze.json', 'goals.npy', 'queries.npz', 'truth_paths.npy', 'truth_fields.npy', 'oracle_paths_seed0.npy', 'scene.png', 'clearance_heading0.npy'):
        if (src / f).exists():
            shutil.copyfile(src / f, out / f)
    res = json.loads((src / 'results.json').read_text())
    if not (out / 'scene.png').exists():
        preview(scene, out)
    n_data = 20_000 if a.smoke else a.n_data
    n_in = int(round(a.inside_share * n_data))
    x, speed, normal, d, stats = D.sample(scene, n_data - n_in, 0)
    xi, si, ni, di = sample_inside(scene, n_in, 2)
    rng = np.random.default_rng(0)
    order = rng.permutation(n_data)
    x, speed, normal = np.concatenate([x, xi])[order], np.concatenate([speed, si])[order], np.concatenate([normal, ni])[order]
    (out / 'data').mkdir(exist_ok=True)
    np.save(out / 'data/sampled_points.npy', x)
    np.save(out / 'data/speed.npy', speed)
    np.save(out / 'data/normal.npy', normal)
    hx, hspeed, _, hd, hstats = D.sample(scene, 5_000 if a.smoke else 50_000, 1)
    (out / 'heldout').mkdir(exist_ok=True)
    np.save(out / 'heldout/sampled_points.npy', hx)
    np.save(out / 'heldout/clearance_cells.npy', hd.astype(np.float32))
    res.update(stage='prepare', data=dict(stats, inside_pairs=n_in, inside_share=a.inside_share, inside_d1_quantiles=np.quantile(di[:, 1], [0.1, 0.5, 0.9]).tolist() if n_in else None),
               variant_of=str(src), smoke=a.smoke, elapsed_seconds=time.time() - t0, complete=True)
    dump(out / 'results.json', res)
    log(f'data variant of {src}: {n_data} pairs, {n_in} with x1 inside an obstacle; {res["data"]}; done in {(time.time() - t0) / 60:.1f} min')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', choices=['disks', 'gibson'], default=None)
    p.add_argument('--from', dest='from_run', type=Path, default=None, help='data variant of this prepare run')
    p.add_argument('--inside-share', type=float, default=0.2)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--n', type=int, default=22, help='disks: box size in cells')
    p.add_argument('--count', type=int, default=40)
    p.add_argument('--r-lo', type=float, default=0.5)
    p.add_argument('--r-hi', type=float, default=1.4)
    p.add_argument('--gap', type=float, default=0.7)
    p.add_argument('--scene-seed', type=int, default=5)
    p.add_argument('--idx', type=int, default=1, help='gibson: scene index')
    p.add_argument('--res', type=int, default=1024)
    p.add_argument('--E', type=float, default=64.0, help='gibson: cells across the box')
    p.add_argument('--h-cells', type=float, default=TR.H_CELLS, help='truth grid step (cells)')
    p.add_argument('--n-data', type=int, default=500_000)
    p.add_argument('--goals', type=int, default=48)
    p.add_argument('--quota', type=int, default=40)
    p.add_argument('--candidates', type=int, default=200_000)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if a.from_run is not None:
        return data_variant(a)
    assert a.scene, '--scene or --from'
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    log = Logger(out / 'log.txt')
    dev = 'cuda:0'
    t0 = time.time()
    if a.smoke:
        a.n_data, a.goals, a.quota, a.candidates = 20_000, 8, 6, 20_000
    scene = SC.Disks(a.n, a.count, a.r_lo, a.r_hi, a.gap, a.scene_seed, dev) if a.scene == 'disks' else SC.Gibson(a.idx, a.res, a.E, 0.0, dev)
    res = dict(stage='prepare', maze=scene.describe(), smoke=a.smoke, gpu=torch.cuda.get_device_name(0), bins=BINS, query_clearance_cells=QUERY_CLEAR, scene_kind=a.scene)
    dump(out / 'maze.json', dict(scene.describe(), variant=scene.kind))
    preview(scene, out)
    log(f'scene {scene.describe()}')

    x, speed, normal, d, stats = D.sample(scene, a.n_data, 0)
    (out / 'data').mkdir(exist_ok=True)
    np.save(out / 'data/sampled_points.npy', x)
    np.save(out / 'data/speed.npy', speed)
    np.save(out / 'data/normal.npy', normal)
    res['data'] = stats
    hx, hspeed, _, hd, hstats = D.sample(scene, 5_000 if a.smoke else 50_000, 1)
    (out / 'heldout').mkdir(exist_ok=True)
    np.save(out / 'heldout/sampled_points.npy', hx)
    np.save(out / 'heldout/clearance_cells.npy', hd.astype(np.float32))
    log(f'data: {stats}')
    del x, speed, normal, d
    torch.cuda.empty_cache()

    gen = torch.Generator(dev).manual_seed(7)
    goals = scene.free(a.goals, gen, QUERY_CLEAR)
    tg = time.time()
    grid = TR.Grid(scene, h_cells=a.h_cells)
    log(f'truth grid {grid.shape}, {time.time() - tg:.0f}s to build')
    fields, sweeps = [], []
    for i in range(0, len(goals), 8):
        f, it = grid.solve(goals[i:i + 8])
        fields.append(f)
        sweeps.append(it)
        log(f'truth goals {i}-{i + len(f) - 1}: {it} sweeps, {time.time() - tg:.0f}s')
    fields = torch.cat(fields)
    np.save(out / 'truth_fields.npy', fields.half().cpu().numpy())
    res['truth'] = dict(grid_shape=list(grid.shape), hx=grid.hx, ht=grid.ht, h_cells=a.h_cells, n_theta=TR.N_THETA, sweeps=sweeps, seconds=time.time() - tg,
                        unreachable_fraction=float((~torch.isfinite(fields)).float().mean()))
    np.save(out / 'goals.npy', goals.cpu().numpy())

    starts = scene.free(a.candidates, gen, QUERY_CLEAR)
    G = len(goals)
    T = torch.empty(G, len(starts), device=dev)
    for g in range(G):
        for i in range(0, len(starts), 50_000):
            s = starts[i:i + 50_000]
            T[g, i:i + len(s)] = grid.query_time(fields, torch.full((len(s),), g, device=dev, dtype=torch.long), s)
    Dc = (T / scene.pitch).cpu().numpy()
    rng = np.random.default_rng(3)
    picked = []
    available = {}
    for b in range(len(BINS) - 1):
        gi, si = np.nonzero((Dc >= BINS[b]) & (Dc < BINS[b + 1]) & np.isfinite(Dc))
        available[bin_labels()[b]] = int(len(gi))
        if not len(gi):
            continue
        order = rng.permutation(len(gi))[:200_000]
        cap = math.ceil(a.quota / 12)
        want = min(int(1.5 * a.quota) + 4, len(order))
        chosen, used, per_goal = [], set(), {}
        while len(chosen) < want:
            for k in order:
                if len(chosen) >= want:
                    break
                g = int(gi[k])
                if k not in used and per_goal.get(g, 0) < cap:
                    chosen.append(int(k))
                    used.add(int(k))
                    per_goal[g] = per_goal.get(g, 0) + 1
            cap *= 2
        picked += [(int(gi[k]), int(si[k]), b) for k in chosen]
    picked = np.array(picked)
    gi = torch.tensor(picked[:, 0], device=dev)
    S, Gq = starts[torch.tensor(picked[:, 1], device=dev)], goals[gi]
    tpaths, ok, plen, clen = [], [], [], []
    for i in range(0, len(picked), 256):
        pt, o, pl, cl = certify(scene, grid, fields, gi[i:i + 256], S[i:i + 256], Gq[i:i + 256])
        tpaths += pt
        ok.append(o)
        plen.append(pl)
        clen.append(cl)
    ok, plen, clen = np.concatenate(ok), np.concatenate(plen), np.concatenate(clen)
    keep = []
    for b in range(len(BINS) - 1):
        idx = [i for i in np.nonzero(picked[:, 2] == b)[0] if ok[i]][:a.quota]
        keep += idx
    keep = np.array(sorted(keep, key=lambda i: (picked[i, 2], i)))
    res['queries'] = dict(available_pairs=available, picked=int(len(picked)), certified=int(ok.sum()), uncertified=int((~ok).sum()), kept=int(len(keep)),
                          per_bin={bin_labels()[b]: int((picked[keep, 2] == b).sum()) for b in range(len(BINS) - 1)})
    kq = torch.tensor(keep, device=dev)
    S, Gq, gi = S[kq], Gq[kq], gi[kq]
    t_star = T[gi, torch.tensor(picked[keep, 1], device=dev)]
    true_dir = torch.nn.functional.normalize(-grid.gradient(fields, gi, S), dim=1)
    euclid = torch.linalg.vector_norm(S[:, :2] - Gq[:, :2], dim=1) / scene.pitch
    np.savez(out / 'queries.npz', start=S.cpu().numpy(), goal=Gq.cpu().numpy(), goal_index=gi.cpu().numpy(), t_star=t_star.cpu().numpy(),
             d_cells=(t_star / scene.pitch).cpu().numpy(), bin=picked[keep, 2], euclid_cells=euclid.cpu().numpy(), path_cells=plen[keep],
             path_clen=clen[keep], true_dir=true_dir.cpu().numpy())
    save_paths(out / 'truth_paths.npy', [tpaths[i] for i in keep])
    log(f'queries: {res["queries"]}')
    bins = picked[keep, 2]

    st = P.settings(scene)
    budget = torch.tensor(np.maximum(st['min_steps'], np.ceil(st['budget_factor'] * clen[keep] / st['step'])), device=dev)
    res['planner'] = dict(st, budget_median=float(budget.median()), budget_max=float(budget.max()))
    rows = P.score(scene, P.straight(S, Gq), [True] * len(S))
    res['straight'] = dict(success=int(sum(r['success'] for r in rows)), by_bin=by_bin(rows, bins))

    def oracle(pairs, ids):
        g = gi[ids].repeat_interleave(len(pairs) // len(ids))
        return grid.oracle_cost(fields, g, pairs[:, :3], goals)

    res['oracle'] = {}
    for ps in PLAN_SEEDS:
        tp = time.time()
        paths, stop, conv = P.mppi(oracle, S, Gq, ps, budget, scene.lo, scene.hi, **st)
        rows = P.score(scene, [paths[i] for i in range(len(S))], conv)
        res['oracle'][str(ps)] = dict(success=int(sum(r['success'] for r in rows)), converged=int(sum(r['converged'] for r in rows)),
                                      collision=int(sum(r['collision'] for r in rows)), by_bin=by_bin(rows, bins), seconds=time.time() - tp)
        if ps == 0:
            save_paths(out / 'oracle_paths_seed0.npy', [paths[i] for i in range(len(S))])
        log(f'oracle MPPI seed {ps}: {res["oracle"][str(ps)]["success"]}/{len(S)} by bin {res["oracle"][str(ps)]["by_bin"]} ({time.time() - tp:.0f}s)')
    log(f'straight: {res["straight"]}')
    res['elapsed_seconds'] = time.time() - t0
    res['complete'] = True
    dump(out / 'results.json', res)
    log(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
