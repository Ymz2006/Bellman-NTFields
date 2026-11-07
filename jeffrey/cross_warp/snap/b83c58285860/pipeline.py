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
    p.add_argument('--gap-scale', type=int, default=0, help='k > 0: the gap maze of scale k (gap.py; replaces the plus variant)')
    # warp
    p.add_argument('--s', type=float, default=1.0)
    p.add_argument('--s-p', type=float, default=None, help='heading factor (default: --s)')
    p.add_argument('--xy', action='store_true')
    p.add_argument('--heading', action='store_true')
    p.add_argument('--sep', action='store_true', help='separable 1-D warps of x and y (no shear)')
    p.add_argument('--sep-nodes', type=int, default=None, help='separable grid intervals per axis (default 8 n)')
    p.add_argument('--k-sep', type=float, default=4.0, help='separable stretch bound per axis')
    p.add_argument('--sep-free', action='store_true', help='separable maps not normalized to the box (local magnification)')
    p.add_argument('--sep-scale', type=float, default=1.0, help='unnormalized separable: initial / central density')
    p.add_argument('--sep-init', type=float, default=None, help='separable: initial density (default: --sep-scale)')
    p.add_argument('--xy-free', action='store_true', help='coupling steps not normalized to the box (row-dependent local magnification)')
    p.add_argument('--xy-scale', type=float, default=1.0, help='unnormalized coupling: central density')
    p.add_argument('--xy-init', type=float, default=None, help='unnormalized coupling: initial density')
    p.add_argument('--m-hi', type=float, default=6.0, help='design plusmag: density outside the + interior')
    p.add_argument('--m-lo', type=float, default=1.0, help='design plusmag: density inside the + interior')
    p.add_argument('--inset', type=float, default=1.0, help='design plusmag: cells kept at m_hi inside the + outline')
    p.add_argument('--warp-from', type=Path, default=None, help='take the warp (grids) from this run\'s training/checkpoint.pt (stage 2 of a learned warp)')
    p.add_argument('--nodes', type=int, default=None, help='grid intervals per axis (default: n, one node per cell)')
    p.add_argument('--k', type=float, default=4.0, help='xy stretch bound per coupling step')
    p.add_argument('--k-p', type=float, default=8.0, help='heading scale bound')
    p.add_argument('--learn', action='store_true')
    p.add_argument('--design', choices=['none', 'open', 'slow', 'grid', 'plusmag'], default='none')
    p.add_argument('--ratio', type=float, default=3.0, help='design: maze / open density ratio')
    p.add_argument('--heading-ratio', type=float, default=None, help='design: maze / open heading scale ratio')
    p.add_argument('--heading-base', type=float, default=1.0, help='design: geometric mean of the heading scale (maze and open)')
    p.add_argument('--r0', type=float, default=0.5, help='design: free radius (cells) up to which a pose counts as in the maze')
    p.add_argument('--r1', type=float, default=1.5, help='design: free radius (cells) from which a pose counts as open')
    p.add_argument('--band', type=float, default=0.4, help='design slow: stretch within this many cells of a wall')
    p.add_argument('--kd', type=float, default=3.0, help='design slow: largest density ratio from a wall')
    p.add_argument('--alpha', type=float, default=1.0, help='design slow: exponent of band / d')
    p.add_argument('--open-ratio', type=float, default=None, help='design slow: extra density ratio maze / open area')
    # learned-warp terms
    p.add_argument('--warp-lr', type=float, default=3e-3)
    p.add_argument('--warp-start', type=int, default=0)
    p.add_argument('--warp-stop', type=int, default=10 ** 9, help='freeze the learned warp after this epoch')
    p.add_argument('--normal-k', type=float, default=6.0, help='anisotropic near-wall stretch target along the clearance normal')
    p.add_argument('--normal-weight', type=float, default=0.0)
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
    if a.gap_scale:
        import gap
        gap.install(a.gap_scale)
        band = gap.BASE_WIDTH * a.gap_scale          # the open band between the quadrants (the plus helpers' width)
    else:
        band = a.plus_width
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
    if a.gap_scale:
        assert info.get('variant') == 'gap' and info.get('gap_scale') == a.gap_scale, 'prepared run is not this gap maze'
    elif a.plus_width:
        assert info.get('variant') == 'plus' and info.get('plus_width') == a.plus_width, 'prepared run is not this plus variant'
    epochs = 3 if a.smoke else a.epochs
    nodes = a.nodes or maze.n
    wcfg = dict(s=a.s, s_p=a.s if a.s_p is None else a.s_p, xy=a.xy, heading=a.heading, nodes=nodes, k=a.k, k_p=a.k_p, learn=a.learn,
                sep=a.sep, sep_nodes=a.sep_nodes or 8 * maze.n, k_sep=a.k_sep, sep_norm=not a.sep_free, sep_scale=a.sep_scale,
                xy_norm=not a.xy_free, xy_scale=a.xy_scale)
    tcfg = dict(warp_lr=a.warp_lr, warp_start=a.warp_start, warp_stop=a.warp_stop, normal_k=a.normal_k, normal_weight=a.normal_weight, ramp=a.ramp, free_c=a.free_c, free_weight=a.free_weight, band_c=a.band_c,
                band_weight=a.band_weight, heading_c=a.heading_c, heading_weight=a.heading_weight, smooth_weight=a.smooth_weight, batch=a.term_batch)
    res = dict(stage='cross-warp', maze=maze.describe(), seed=a.seed, epochs=epochs, prepared=str(a.prepared), gpu=torch.cuda.get_device_name(0),
               smoke=a.smoke, gap_scale=a.gap_scale, warp=wcfg, trainer=tcfg, design=dict(kind=a.design, ratio=a.ratio, heading_ratio=a.heading_ratio, heading_base=a.heading_base, r0=a.r0, r1=a.r1, band=a.band, kd=a.kd, alpha=a.alpha, open_ratio=a.open_ratio, m_hi=a.m_hi, m_lo=a.m_lo, inset=a.inset),
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
    elif a.design == 'slow':
        design = lambda w: W.design_slow(w, maze, a.band, a.kd, a.alpha, a.open_ratio, a.r0, a.r1)
    elif a.design == 'grid':
        design = lambda w: W.design_grid(w, maze, a.band, a.kd, a.open_ratio, band)
    elif a.design == 'plusmag':
        design = lambda w: W.design_plusmag(w, maze, band, a.m_hi, a.m_lo, a.inset)
    if a.sep_init is not None:
        first_design = design

        def design(w):
            if first_design is not None:
                first_design(w)
            with torch.no_grad():
                off = (math.log(a.sep_init) - w.sep_log0) / w.hmax_sep
                assert abs(off) < 1, '--sep-init outside the separable bound'
                raw = w.hmax_sep * math.atanh(off)
                w.raw_sx.fill_(raw)
                w.raw_sy.fill_(raw)
        res['sep_init'] = a.sep_init
    if a.xy_init is not None:
        prev_design = design

        def design(w):
            if prev_design is not None:
                prev_design(w)
            with torch.no_grad():
                off = (math.log(a.xy_init) - w.xy_log0) / w.hmax
                assert abs(off) < 1, '--xy-init outside the coupling bound'
                w.raw_f.fill_(w.hmax * math.atanh(off))
                w.raw_g.fill_(w.hmax * math.atanh(off))
        res['xy_init'] = a.xy_init
    if a.warp_from is not None:
        src = torch.load(a.warp_from, map_location=dev, weights_only=False)
        src_cfg = src['warp_config']
        for key in ('s', 's_p', 'xy', 'heading', 'nodes', 'k', 'k_p', 'sep', 'sep_nodes', 'k_sep', 'sep_norm', 'sep_scale'):
            assert src_cfg.get(key) == wcfg.get(key), f'--warp-from: warp config differs in {key}: {src_cfg.get(key)} vs {wcfg.get(key)}'
        warp_state = {k[len('warp.'):]: v for k, v in src['model_state_dict'].items() if k.startswith('warp.')}
        res['warp_from'] = str(a.warp_from)

        def design(w):
            w.load_state_dict(warp_state, strict=True)
    if a.heading and a.heading_ratio is not None and a.design != 'open':
        first = design
        def design(w):
            if first is not None:
                first(w)
            W.design_heading(w, maze, a.heading_ratio, a.r0, a.r1, a.heading_base)
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
    if 'category' in q.files:
        import gap as GP
        res['by_category'] = {name: dict(success=int(rows['success'][:, q['category'] == c].sum()), collision=int(rows['collision'][:, q['category'] == c].sum()),
                                         stalled=int((~rows['converged'][:, q['category'] == c]).sum()), runs=int(rows['success'][:, q['category'] == c].size))
                              for c, name in enumerate(GP.CATEGORIES) if (q['category'] == c).any()}
    if band:
        where = DG.where_plus(maze, q['start'], band).astype(int) + DG.where_plus(maze, q['goal'], band).astype(int)
        res['by_where'] = {name: dict(success=int(rows['success'][:, where == k].sum()), collision=int(rows['collision'][:, where == k].sum()),
                                      stalled=int((~rows['converged'][:, where == k]).sum()), runs=int(rows['success'][:, where == k].size))
                           for k, name in ((2, 'both'), (1, 'one'), (0, 'neither'))}
    res['failures'] = dict(collision=int(rows['collision'].sum()), stalled=int((~rows['converged']).sum()),
                           stalled_no_collision=int(((~rows['converged']) & ~rows['collision']).sum()), runs=int(rows['success'].size))
    res['near_wall'] = DG.near_wall(maze, a.prepared, grad)
    res['profiles'] = DG.profiles(maze, a.prepared, grad)
    res['bumps'] = DG.bumps(maze, a.prepared, cost, P)
    res['warp_stats'] = DG.warp_stats(maze, net.warp, free_pool, a.prepared, band, a.out) if (a.xy or a.heading or a.sep) else None
    log(f'by category {res.get("by_category")}; by where {res.get("by_where")}, failures {res["failures"]}, near wall {res["near_wall"]}, bumps {res["bumps"]}, profiles {res["profiles"]}')
    if res['warp_stats']:
        log(f'warp stats {res["warp_stats"]}')
    res['elapsed_seconds'] = time.time() - t0
    res['complete'] = True
    dump(a.out / 'results.json', res)
    log(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
