"""Representation test: the pair-lift network fitted directly to the true travel time (supervised), so that what a pair coordinate lets
the network REPRESENT is measured apart from what ntrl-demo's loss, the push and the cap find.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/pair_lift/repfit.py --prepared <prepare run> --seed 87358 --lift time --pair count ... --out <dir>

Training pairs: (free configuration, one of the prepare run's goals; `--band b`: a share b of the configurations are the prepare run's
band points, where ntrl-demo puts every first end) with the prepare run's ground truth T* (diag.Truth), `--pairs` of
them, redrawn never; loss = mean of ((T - T*) / (T* + 0.5 cells))^2, AdamW lr 5e-4 (ntrl-demo's weight decay 0.5 is kept: the network's
layers are scale-normalized), `--steps` minibatches of 4,000. Then, on held-out configurations and the same goals: travel times within
10 %, the detour index, the step held across walls (jeffrey/lift_scale/jumps.py's pairs), and ntrl-demo's MPPI on the prepared queries
(planning seeds 0 / 1 / 2). The network and its options are pairpipe.py's.
"""
import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pairnet as PN  # noqa: E402
import pairtrain as PT  # noqa: E402
import scenes as SN  # noqa: E402
import diag as DG  # noqa: E402
import diagnostics as EDG  # noqa: E402
LF = PN.LF

_spec = importlib.util.spec_from_file_location('lift_scale_jumps', PN.ROOT / 'jeffrey/lift_scale/jumps.py')
JP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(JP)


def jumps(scene, prepared, net):
    W = JP.wall_pairs(scene, prepared)
    with torch.no_grad():
        ta = net.out(torch.cat([W['A'], W['goal']], 1))[0][:, 0]
        tb = net.out(torch.cat([W['B'], W['goal']], 1))[0][:, 0]
    step, st = (ta - tb).abs() / scene.pitch, W['step_true']
    out = dict(pairs=int(len(st)), planar_gap_median=float(W['planar_gap'].median()), by_true={})
    for i, lab in enumerate(JP.labels()):
        m = (st >= JP.BINS[i]) & (st < JP.BINS[i + 1])
        if int(m.sum()) > 15:
            out['by_true'][lab] = dict(n=int(m.sum()), true=float(st[m].median()), learned=float(step[m].median()), learned_q10=float(step[m].quantile(0.1)), learned_q90=float(step[m].quantile(0.9)))
    big = st >= 3.0
    out['big'] = dict(n=int(big.sum()), true=float(st[big].median()), learned=float(step[big].median()), ratio=float((step[big] / st[big]).median())) if int(big.sum()) > 15 else None
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--blocked', type=Path, default=None)       # accepted and ignored (the queue passes it)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--pairs', type=int, default=600000)
    p.add_argument('--lift', choices=['none', 'time', 'vec'], default='none')
    p.add_argument('--depth', type=float, default=LF.OFFSET_CELLS)
    p.add_argument('--octaves', default='')
    p.add_argument('--pair', choices=['none', 'bit', 'count', 'len'], default='none')
    p.add_argument('--pair-sign', choices=['anti', 'same'], default='anti')
    p.add_argument('--pair-norm', choices=['tanh', 'frac', 'log', 'raw'], default='tanh')
    p.add_argument('--pair-scale', default='2')
    p.add_argument('--pair-what', choices=['excess', 'length', 'updown'], default='excess')
    p.add_argument('--pair-place', choices=['input', 'rows', 'pre', 'post'], default='input')
    p.add_argument('--pair-copies', type=int, default=16)
    p.add_argument('--pair-amp', type=float, default=1.0)
    p.add_argument('--pair-freq', type=float, default=1.0)
    p.add_argument('--pair-rows', type=int, default=16)
    p.add_argument('--band', type=float, default=0.0, help="share of the supervised configurations taken from the prepare run's band points (ntrl-demo's x0) instead of uniformly free ones")
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--no-plan', action='store_true')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from common import Logger, dump
    import plan as P
    import evaluate as EV
    log = Logger(a.out / 'log.txt')
    dev = 'cuda:0'
    t0 = time.time()
    scene, info = SN.load_scene(a.prepared, dev)
    truth = DG.Truth(scene, a.prepared)
    steps = 20 if a.smoke else a.steps
    octaves = [float(v) for v in a.octaves.split(',') if v.strip()]
    cfg = dict(octaves=octaves, rows=16, base_scale=1.0, whiten=True, gate=dict(kind='const' if octaves else 'none', mu_min=0.02, const=1.0),
               lift=dict(kind=a.lift, gain=1.0, depth=a.depth, freq=1.0, seed=a.seed), zero_cols=[],
               pair=dict(kind=a.pair, sign=a.pair_sign, norm=a.pair_norm, scale=scene.E if a.pair_scale == 'box' else float(a.pair_scale), what=a.pair_what,
                         place=a.pair_place, copies=a.pair_copies, amp=a.pair_amp, freq=a.pair_freq, rows=a.pair_rows, inside='seg', grad=False, seed=a.seed))
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    B = torch.normal(0, 1, size=(128, 3))
    torch.nn.init.trunc_normal_(B, mean=0.0, std=1, a=-2.0, b=2.0)
    mt, mf, mn, db = PT.LT.ntrl_modules()
    net = PN.build_network(mn, dev, B, cfg, scene)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.5)
    n_pairs = 20000 if a.smoke else a.pairs
    gen = torch.Generator(dev).manual_seed(a.seed + 11)
    q = scene.free(n_pairs, gen, 0.0)
    if a.band > 0:
        x0 = np.load(a.prepared / 'data/sampled_points.npy', mmap_mode='r')
        rows = np.sort(np.random.default_rng(a.seed).choice(len(x0), size=int(a.band * n_pairs), replace=False))
        xb = torch.tensor(np.asarray(x0[rows, :3]), device=dev)
        xb = xb[scene.clearance(xb) >= 0]
        q = torch.cat([xb, q])[:n_pairs]
        q = q[torch.randperm(len(q), generator=gen, device=dev)]
    gi = torch.randint(0, len(truth.goals), (n_pairs,), generator=gen, device=dev)
    ts = truth.time(gi, q)
    keep = torch.isfinite(ts) & (torch.linalg.vector_norm(q - truth.goals[gi], dim=1) > 0.25 * scene.pitch)
    X, Y = torch.cat([q, truth.goals[gi]], 1)[keep], ts[keep]
    log(f'scene {SN.scene_name(info)}, lift {a.lift}, pair {cfg["pair"]}, octaves {octaves}, {len(X)} supervised pairs, {steps} steps')
    hist = []
    for it in range(1, steps + 1):
        idx = torch.randint(0, len(X), (4000,), generator=gen, device=dev)
        T = net.out(X[idx])[0][:, 0]
        loss = (((T - Y[idx]) / (Y[idx] + 0.5 * scene.pitch)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 200 == 0 or it == steps:
            hist.append((it, float(loss)))
            if it % 1000 == 0 or it == steps:
                log(f'step {it}: relative squared error {float(loss):.4e} ({(time.time() - t0) / 60:.1f} min)')
    (a.out / 'training').mkdir(exist_ok=True)
    ck = a.out / 'training/checkpoint.pt'
    torch.save({'epoch': steps, 'model_state_dict': net.state_dict(), 'B_state_dict': B, 'embed_config': net.cfg}, ck)
    net = PN.load_field(ck, scene, dev)
    cost, grad = P.field_cost(net), lambda pairs: P.field_grad(net, pairs)
    res = dict(stage='pair-lift-repfit', scene=scene.describe(), scene_name=SN.scene_name(info), seed=a.seed, steps=steps, prepared=str(a.prepared), smoke=a.smoke, config=cfg,
               args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, history=hist, final_loss=hist[-1][1], train_seconds=time.time() - t0)
    pairs, gi2, t_star, D = DG.truth_pairs(scene, truth, 2000 if a.smoke else 24000)
    blk = LF.segment_blocked(scene, pairs)
    res['accuracy'] = DG.accuracy(scene, cost, pairs, t_star, D, blk)
    res['jumps'] = jumps(scene, a.prepared, net)
    acc = res['accuracy']
    log(f'held-out: within 10 % {acc["within10"]:.3f}, median T/T* {acc["ratio_median"]:.3f}, line of sight {acc.get("los")}, detour {({k: v for k, v in acc.get("detour", {}).items() if k != "by_length"})}')
    log('step across walls, true -> learned (cells): ' + '; '.join(f"{k}: {v['true']:.1f} -> {v['learned']:.1f}" for k, v in res['jumps']['by_true'].items()))
    if not a.no_plan:
        res.update(EV.evaluate(scene, a.prepared, cost, grad, a.out, log, label=f'fit {a.lift}+{a.pair}'))
        res['failures'] = EDG.failures_by_arrival(np.load(a.out / 'rows.npz'))
        log(f'planning {res["success_total"]}, failures {res["failures"]}')
    res['elapsed_seconds'] = time.time() - t0
    res['complete'] = True
    dump(a.out / 'results.json', res)
    log(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
