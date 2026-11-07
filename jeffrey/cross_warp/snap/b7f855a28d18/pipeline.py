"""One cross-warp run: ntrl-demo with a C-space warp in front of its encoder (warp.py), trained (trainer.py), planned
and diagnosed on a jeffrey/maze_scale maze, by default its plus ('cross') variant.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/cross_warp/pipeline.py --prepared <maze_scale plus-prepare run> --seed 87358 \
        --s 1 --xy --learn --free-c 4 --free-weight 1 --out <dir>

1. maze      the prepared maze (plus.install(width) first when --plus-width > 0, as maze_scale/plusntrl.py does),
             checked against the prepared maze.json.
2. warp      phi = (s psi(x, y), s_p exp(g) p): --s / --s-p global factors; --xy the box-preserving coupling grids,
             --heading the heading-scale grid; --design open sets the grids from the geometry (density ratio
             --ratio between the maze and the open area, --heading-ratio for the heading), --learn makes them
             trainable (README_warp's learned warp; with --design the design is the initialization).
3. train     trainer.py (ntrl-demo's loop from scratch, 5,000 epochs) with the warp terms.
4. plan      maze_scale/evaluate.py: ntrl-demo's MPPI per cell, planning seeds 0/1/2, exact scorer, success by bin;
             field T / T*, Spearman, descent direction; held-out speed fit; slice T(., goal 0).
5. extras    success by where the query's ends lie (both / one / neither inside the +), collisions and stalls;
             diag.py: near-wall slope / direction, grid-line profiles, local minima along true paths, the warp's
             stretch (log det J_psi, heading scale) in the open area, the maze and next to walls; warp_map.npz.
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
import warp as W  # noqa: E402
import trainer as TR  # noqa: E402
import diag as DG  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=5000)
    p.add_argument('--plus-width', type=int, default=6, help='0: the unmodified maze')
    # warp
    p.add_argument('--s', type=float, default=1.0)
    p.add_argument('--s-p', type=float, default=None, help='heading factor (default: --s)')
    p.add_argument('--xy', action='store_true')
    p.add_argument('--heading', action='store_true')
    p.add_argument('--nodes', type=int, default=None, help='grid intervals per axis (default: n, one node per cell)')
    p.add_argument('--k', type=float, default=4.0, help='xy stretch bound per coupling step')
    p.add_argument('--k-p', type=float, default=8.0, help='heading scale bound')
    p.add_argument('--learn', action='store_true')
    p.add_argument('--design', choices=['none', 'open'], default='none')
    p.add_argument('--ratio', type=float, default=3.0, help='design: maze / open density ratio')
    p.add_argument('--heading-ratio', type=float, default=None, help='design: maze / open heading scale ratio')
    p.add_argument('--r0', type=float, default=0.5, help='design: free radius (cells) up to which a pose counts as in the maze')
    p.add_argument('--r1', type=float, default=1.5, help='design: free radius (cells) from which a pose counts as open')
    # learned-warp terms
    p.add_argument('--warp-lr', type=float, default=3e-3)
    p.add_argument('--warp-start', type=int, default=0)
    p.add_argument('--ramp', type=int, default=500)
    p.add_argument('--free-c', type=float, default=4.0)
    p.add_argument('--free-weight', type=float, default=0.0)
    p.add_argument('--free-radius', type=float, default=1.0, help='cells: free-compression pool = free radius at least this')
    p.add_argument('--band-c', type=float, default=2.0)
    p.add_argument('--band-weight', type=float, default=0.0)
    p.add_argument('--heading-c', type=float, default=2.0)
    p.add_argument('--heading-weight', type=float, default=0.0)
    p.add_argument('--smooth-weight', type=float, default=0.0)
    p.add_argument('--term-batch', type=int, default=2000)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from common import Logger, dump
    log = Logger(a.out / 'log.txt')
    if a.plus_width:
        import plus as PL
        PL.install(a.plus_width)
    import maze as MZ
    import plan as P
    import evaluate as EV
    dev = 'cuda:0'
    t0 = time.time()
    info = json.load(open(a.prepared / 'maze.json'))
    maze = MZ.Maze(info['n'], dev)
    assert maze.describe()['walls'] == info['walls'] and np.array_equal(maze.V.astype(int), np.array(info['V'])), 'prepared maze differs'
    assert np.array_equal(maze.H.astype(int), np.array(info['H'])), 'prepared maze differs'
    if a.plus_width:
        assert info.get('variant') == 'plus' and info.get('plus_width') == a.plus_width, 'prepared run is not this plus variant'
    epochs = 3 if a.smoke else a.epochs
    nodes = a.nodes or maze.n
    wcfg = dict(s=a.s, s_p=a.s if a.s_p is None else a.s_p, xy=a.xy, heading=a.heading, nodes=nodes, k=a.k, k_p=a.k_p, learn=a.learn)
    tcfg = dict(warp_lr=a.warp_lr, warp_start=a.warp_start, ramp=a.ramp, free_c=a.free_c, free_weight=a.free_weight, band_c=a.band_c,
                band_weight=a.band_weight, heading_c=a.heading_c, heading_weight=a.heading_weight, smooth_weight=a.smooth_weight, batch=a.term_batch)
    res = dict(stage='cross-warp', maze=maze.describe(), seed=a.seed, epochs=epochs, prepared=str(a.prepared), gpu=torch.cuda.get_device_name(0),
               smoke=a.smoke, warp=wcfg, trainer=tcfg, design=dict(kind=a.design, ratio=a.ratio, heading_ratio=a.heading_ratio, r0=a.r0, r1=a.r1),
               free_radius_cells=a.free_radius)
    # free pool: uniform configurations whose free radius is at least --free-radius cells
    gen = torch.Generator(dev).manual_seed(13)
    pool = []
    have = 0
    while have < 200_000:
        q = maze.uniform(400_000, gen)
        q = q[W.free_radius(maze, q) >= a.free_radius]
        pool.append(q)
        have += len(q)
    free_pool = torch.cat(pool)[:200_000]
    res['free_pool'] = len(free_pool)
    design = None
    if a.design == 'open':
        design = lambda w: W.design_open(w, maze, a.ratio, a.r0, a.r1, a.heading_ratio)
    log(f'maze n={maze.n} ({info.get("variant", "plain")}), warp {wcfg}, trainer {tcfg}, design {a.design}, free pool {len(free_pool)}')
    ck, hist = TR.train(a.prepared / 'data', a.out / 'training', a.seed, epochs, wcfg, tcfg, log, maze=maze, free_pool=free_pool, design=design)
    res['train_seconds'] = time.time() - t0
    res['final_loss'] = hist['final_loss']
    res['rollbacks'] = int(sum(hist['rollbacks']))
    log(f'trained: {res["train_seconds"] / 60:.1f} min, final ntrl loss {res["final_loss"]}, rollbacks {res["rollbacks"]}')
    net = W.load_field(ck, dev)
    cost, grad = P.field_cost(net), lambda pairs: P.field_grad(net, pairs)
    res.update(EV.evaluate(maze, a.prepared, cost, grad, a.out, log, label=f'warp n={maze.n} s{a.seed}'))
    res['speed_fit'] = EV.speed_fit(maze, a.prepared, grad)
    goals = torch.tensor(np.load(a.prepared / 'goals.npy'), device=dev)
    np.save(a.out / 'slice_goal0.npy', EV.slice_values(maze, cost, goals[0]))
    log(f'speed fit: {res["speed_fit"]}')
    # extras
    rows = np.load(a.out / 'rows.npz')
    q = np.load(a.prepared / 'queries.npz')
    if a.plus_width:
        where = DG.where_plus(maze, q['start'], a.plus_width).astype(int) + DG.where_plus(maze, q['goal'], a.plus_width).astype(int)
        res['by_where'] = {name: dict(success=int(rows['success'][:, where == k].sum()), collision=int(rows['collision'][:, where == k].sum()),
                                      stalled=int((~rows['converged'][:, where == k]).sum()), runs=int(rows['success'][:, where == k].size))
                           for k, name in ((2, 'both'), (1, 'one'), (0, 'neither'))}
    res['failures'] = dict(collision=int(rows['collision'].sum()), stalled=int((~rows['converged']).sum()),
                           stalled_no_collision=int(((~rows['converged']) & ~rows['collision']).sum()), runs=int(rows['success'].size))
    res['near_wall'] = DG.near_wall(maze, a.prepared, grad)
    res['profiles'] = DG.profiles(maze, a.prepared, grad)
    res['bumps'] = DG.bumps(maze, a.prepared, cost, P)
    res['warp_stats'] = DG.warp_stats(maze, net.warp, free_pool, a.prepared, a.plus_width, a.out) if (a.xy or a.heading) else None
    log(f'by where {res.get("by_where")}, failures {res["failures"]}, near wall {res["near_wall"]}, bumps {res["bumps"]}, profiles {res["profiles"]}')
    if res['warp_stats']:
        log(f'warp stats {res["warp_stats"]}')
    res['elapsed_seconds'] = time.time() - t0
    res['complete'] = True
    dump(a.out / 'results.json', res)
    log(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
