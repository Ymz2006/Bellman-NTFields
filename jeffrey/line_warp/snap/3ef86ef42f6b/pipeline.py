"""Stage 2: ntrl-demo through a fitted warp (a fit run's phi, frozen), trained and planned exactly as cross_warp's runs.

    python pipeline.py --prepared <prepare run> --fit <fit run> --seed 87358 --s 4 --out <dir> [--z-space]

--fit none: no warp (the control). s = the warped maze's width in unit boxes; --s-p the heading factor (default s).
Default: the loss is taken in x through phi (the chain rule), MPPI in x, as cross_warp's plus study. --z-space: the
network is trained on the warped maze (trainer.py z_space): fixed coordinates z = phi(x), the loss in z.
"""
import argparse
import json
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
sys.path.insert(0, str(ROOT / 'jeffrey/cross_warp'))
sys.path.insert(0, str(HERE))
import warp as W  # noqa: E402
import trainer as TR  # noqa: E402
import diag as DG  # noqa: E402
import netwarp as NW  # noqa: E402
import fit as FT  # noqa: E402
import checks as CK  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--fit', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--s', type=float, default=1.0)
    p.add_argument('--s-p', type=float, default=None)
    p.add_argument('--m', type=float, default=None, help='instead of --s: scale so that the median stretch inside the quadrants is m, and the heading factor m (a like-for-like match with a constructed warp x m)')
    p.add_argument('--epochs', type=int, default=5000)
    p.add_argument('--z-space', action='store_true')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from common import Logger, dump
    log = Logger(a.out / 'log.txt')
    dev = 'cuda:0'
    t0 = time.time()
    maze, info = FT.load_maze(a.prepared, dev)
    import plan as P
    import evaluate as EV
    import warpplan as WP
    NW.install(W, P)
    band = info.get('plus_width', 0)
    epochs = 3 if a.smoke else a.epochs
    use_fit = str(a.fit) != 'none'
    design = None
    if use_fit:
        state, netcfg, fit_args = NW.load_fit(a.fit, dev)
        wcfg = dict(net=netcfg, s=a.s, s_p=a.s if a.s_p is None else a.s_p)

        def design(w):
            w.set_state(state, maze)
            if a.m is not None:
                w.s = 1.0
                w.s = w.cfg['s'] = a.m / WP.quadrant_stretch(w, maze)
                w.s_p = w.cfg['s_p'] = a.m
                res['s'], res['s_p'] = w.s, w.s_p
                log(f'quadrant stretch {a.m}: s = {w.s:.4f}, heading factor {w.s_p}')
    else:
        wcfg = dict(s=a.s, s_p=a.s if a.s_p is None else a.s_p, xy=False, heading=False, sep=False)
    res = {}
    tcfg = dict(warp_start=0, ramp=1, lb_weight=0, los_weight=0, td_len=None, z_space=a.z_space, free_weight=0, band_weight=0, heading_weight=0, smooth_weight=0)
    res.update(stage='line-warp', maze=maze.describe(), seed=a.seed, epochs=epochs, prepared=str(a.prepared), fit=str(a.fit), s=a.s, s_p=wcfg['s_p'], m=a.m, z_space=a.z_space,
               gpu=torch.cuda.get_device_name(0), smoke=a.smoke)
    log(f'maze n={maze.n} ({info.get("variant", "plain")}), fit {a.fit}, s {a.s}, s_p {wcfg["s_p"]}, z_space {a.z_space}')
    ck, hist = TR.train(a.prepared / 'data', a.out / 'training', a.seed, epochs, wcfg, tcfg, log, maze=maze, free_pool=None, design=design)
    res['train_seconds'] = time.time() - t0
    res['final_loss'] = hist['final_loss']
    res['rollbacks'] = int(sum(hist['rollbacks']))
    log(f'trained: {res["train_seconds"] / 60:.1f} min, final ntrl loss {res["final_loss"]}, rollbacks {res["rollbacks"]}')
    net = W.load_field(ck, dev)
    cost, grad = P.field_cost(net), lambda pairs: P.field_grad(net, pairs)
    if a.z_space:
        t_scale = WP.quadrant_stretch(net.warp, maze)
        base_cost, base_grad = cost, grad
        cost = lambda pairs, ids=None: base_cost(pairs, ids) / t_scale

        def grad(pairs):
            t, g = base_grad(pairs)
            return t / t_scale, g / t_scale
        res['z_scale'] = t_scale
    res.update(EV.evaluate(maze, a.prepared, cost, grad, a.out, log, label=f'line warp n={maze.n} s{a.seed}'))
    res['speed_fit'] = EV.speed_fit(maze, a.prepared, grad)
    log(f'speed fit: {res["speed_fit"]}')
    rows = np.load(a.out / 'rows.npz')
    q = np.load(a.prepared / 'queries.npz')
    if 'category' in q.files:
        import gap as GP
        res['by_category'] = {name: dict(success=int(rows['success'][:, q['category'] == c].sum()), runs=int(rows['success'][:, q['category'] == c].size))
                              for c, name in enumerate(GP.CATEGORIES) if (q['category'] == c).any()}
    if band:
        where = DG.where_plus(maze, q['start'], band).astype(int) + DG.where_plus(maze, q['goal'], band).astype(int)
        res['by_where'] = {name: dict(success=int(rows['success'][:, where == k].sum()), runs=int(rows['success'][:, where == k].size)) for k, name in ((2, 'both'), (1, 'one'), (0, 'neither'))}
    conv, coll, ok = rows['converged'], rows['collision'], rows['success']
    # by arrival: plan.mppi appends the goal to every path, so the scorer's collision flag also covers the final jump of runs that never arrived
    res['failures'] = dict(arrived_wall=int((~ok & conv & coll).sum()), not_arrived=int((~ok & ~conv).sum()), runs=int(ok.size))
    res['near_wall'] = DG.near_wall(maze, a.prepared, grad)
    res['bumps'] = DG.bumps(maze, a.prepared, cost, P)
    if use_fit:
        res['warp'] = CK.measure(net.warp, maze)
    log(f'by category {res.get("by_category")}; by where {res.get("by_where")}, failures {res["failures"]}, near wall {res["near_wall"]["all"]}, bumps {res["bumps"]}, warp {res.get("warp")}')
    res['elapsed_seconds'] = time.time() - t0
    res['complete'] = True
    dump(a.out / 'results.json', res)
    log(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
