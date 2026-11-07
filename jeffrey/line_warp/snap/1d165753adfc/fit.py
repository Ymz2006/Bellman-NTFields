"""Stage 1: fit a diffeomorphism phi of the maze plane from the geometry alone (no travel-time field, no planner).

    python fit.py --prepared <maze prepare run> --out <dir> [--mode cspace --headings 9 --kind hard --steps 6000 ...]

Every step: the boundary polylines of a few planes (obstacles.py) go through phi, `--lines` axis-aligned lines per axis are
drawn uniformly over the image, and the line loss (lineloss.py) is averaged over them. The fold term keeps phi orientation
preserving and its singular values inside [1 / kappa, kappa] on points drawn uniformly in the maze box.
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
sys.path.insert(0, str(ROOT / 'jeffrey/cross_warp'))
sys.path.insert(0, str(HERE))
import lineloss as LL  # noqa: E402
import obstacles as OB  # noqa: E402
import warpnet as WN  # noqa: E402


def install_maze(info):
    """Patch the maze module as the prepare run did (plus.py / gap.py) from its maze.json."""
    if info.get('variant') == 'gap':
        import gap
        gap.install(info['gap_scale'], gap.base_name(info))
    elif info.get('variant') == 'plus':
        import plus
        plus.install(info['plus_width'])


def load_maze(prepared, dev):
    info = json.load(open(Path(prepared) / 'maze.json'))
    install_maze(info)
    import maze as MZ
    maze = MZ.Maze(info['n'], dev)
    assert np.array_equal(maze.V.astype(int), np.array(info['V'])) and np.array_equal(maze.H.astype(int), np.array(info['H'])), 'maze differs'
    return maze, info


def fold_terms(phi, x, kappa, eps):
    J = WN.jacobian_fd(phi, x, eps)
    s1, s2, det = WN.singular_values(J)
    fold = torch.relu(0.02 - det).pow(2).mean()
    lim = math.log(kappa)
    bound = (torch.relu(torch.log(s1.clamp(min=1e-6)) - lim).pow(2) + torch.relu(-lim - torch.log(s2.clamp(min=1e-6))).pow(2)).mean()
    return fold, bound, s1.detach(), s2.detach(), det.detach(), J


def rotation_deg(J):
    """Rotation angle of the polar factor of 2 x 2 matrices, in degrees (N,)."""
    return torch.rad2deg(torch.atan2(J[:, 1, 0] - J[:, 0, 1], J[:, 0, 0] + J[:, 1, 1]))


def build(a):
    scales = [float(s) for s in a.scales.split(',')]
    if a.arch == 'resnet':
        return WN.FourierResNet(a.blocks, a.hidden, scales, a.per_scale, a.amp, a.seed)
    return WN.TriMap(a.nodes_per_cell * a.n, a.rows_per_cell * a.n, a.hidden, scales, a.per_scale, a.hmax, a.seed, a.arch == 'sep', a.tri_steps)


def fit(a, maze, log, phi=None):
    dev = maze.device
    a.n = maze.n
    pl = OB.planes(maze, a.mode, a.headings, a.spacing)
    P = [torch.tensor(b.points, device=dev) for b, _ in pl]
    S = [torch.tensor(b.segments, device=dev) for b, _ in pl]
    phi = (phi or build(a)).to(dev)
    opt = torch.optim.Adam(phi.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=a.lr * 0.02)
    gen = torch.Generator(dev).manual_seed(a.seed + 1)
    eps = 0.05 / maze.E
    hist = []
    t0 = time.time()
    for step in range(1, a.steps + 1):
        if a.window > 0:
            phi.window(min(phi.groups, 1 + (phi.groups - 1) * step / (a.window * a.steps)))
        idx = torch.randperm(len(pl), generator=gen, device=dev)[:a.planes_per_step].tolist()
        losses, ratios, runs, odd, pooled = [], [], [], [], []
        for i in idx:
            Z = phi(P[i])
            for axis in (0, 1):
                ang = float(torch.rand((), generator=gen, device=dev)) * math.pi if a.any_angle else 0.0
                l, st = LL.line_batch(Z, S[i], a.lines, axis, gen, a.kind, a.floor, a.graze, ang, a.big_runs)
                losses.append(l)
                ratios.append(st['ratio'])
                runs.append(st['runs'])
                odd.append(st['crossings_odd'])
                pooled.append(st['all_runs'])
        line = torch.cat(losses).mean()
        if a.kind in ('gvar', 'gmix'):
            lg = torch.log(torch.cat(pooled).clamp(min=1e-9))
            line = (a.global_weight * lg.var()) + (line if a.kind == 'gmix' else 0.0)
        x = torch.rand(a.fold_points, 2, generator=gen, device=dev) - 0.5
        fold, bound, s1, s2, det, J = fold_terms(phi, x, a.kappa, eps)
        loss = line
        if a.arch == 'resnet':
            lip = phi.block_lipschitz(x, eps)
            loss = loss + a.fold_weight * fold + a.bound_weight * bound + a.lip_weight * torch.relu(lip - a.lip).pow(2).mean()
        else:                                                # tri / sep: det J > 0 by construction, the density bound is hmax
            lip = torch.zeros(1, device=dev)
            if a.cond_weight > 0 and a.arch == 'tri':
                loss = loss + a.cond_weight * phi.conditional_variation()
            if a.curv_weight > 0:
                loss = loss + a.curv_weight * phi.curvature()
        if a.shear_weight > 0:
            loss = loss + a.shear_weight * ((J[:, 0, 1] ** 2 + J[:, 1, 0] ** 2) / (J[:, 0, 0] ** 2 + J[:, 1, 1] ** 2).clamp(min=1e-6)).mean()
        opt.zero_grad()
        loss.backward()
        if a.clip > 0:
            torch.nn.utils.clip_grad_norm_(phi.parameters(), a.clip)
        opt.step()
        sched.step()
        if step % a.log_every == 0 or step == 1:
            r = torch.cat(ratios)
            h = dict(step=step, line=float(line), fold=float(fold), bound=float(bound), ratio_median=float(r.median()), ratio_q90=float(r.quantile(0.9)),
                     runs=float(torch.cat(runs).mean()), odd=float(np.mean(odd)), s1_max=float(s1.max()), s2_min=float(s2.min()), det_min=float(det.min()),
                     lip_max=float(lip.max()), rot_abs_median=float(rotation_deg(J.detach()).abs().median()), seconds=time.time() - t0)
            hist.append(h)
            log(f"step {step}: line {h['line']:.4f}, ratio median {h['ratio_median']:.2f} (q90 {h['ratio_q90']:.1f}), runs/line {h['runs']:.1f}, odd {h['odd']:.3f}, "
                f"sv {h['s2_min']:.3f}..{h['s1_max']:.2f}, det min {h['det_min']:.4f}, lip {h['lip_max']:.2f}, rot {h['rot_abs_median']:.1f} deg, fold {h['fold']:.1e}, bound {h['bound']:.1e}, {h['seconds']:.0f}s")
    return phi, hist, pl


def arguments(p=None):
    p = p or argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=['walls', 'cspace'], default='cspace')
    p.add_argument('--headings', type=int, default=9)
    p.add_argument('--spacing', type=float, default=0.1, help='boundary points every this many cells')
    p.add_argument('--kind', default='hard', choices=['hard', 'hard2', 'logr', 'logr2', 'var', 'gvar', 'gmix'])
    p.add_argument('--global-weight', type=float, default=1.0, help='weight of the variance of log runs over the batch')
    p.add_argument('--floor', type=float, default=0.1)
    p.add_argument('--graze', type=float, default=0.05)
    p.add_argument('--lines', type=int, default=128, help='lines per axis and plane per step')
    p.add_argument('--planes-per-step', type=int, default=3)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--blocks', type=int, default=8)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--scales', default='1,2,4,8,16')
    p.add_argument('--per-scale', type=int, default=32)
    p.add_argument('--amp', type=float, default=0.1)
    p.add_argument('--window', type=float, default=0.5, help='share of the steps over which the higher Fourier scales fade in (0: all on)')
    p.add_argument('--kappa', type=float, default=16.0)
    p.add_argument('--fold-weight', type=float, default=100.0)
    p.add_argument('--bound-weight', type=float, default=1.0)
    p.add_argument('--fold-points', type=int, default=4096)
    p.add_argument('--lip', type=float, default=0.7, help='per-block Lipschitz limit (i-ResNet invertibility)')
    p.add_argument('--lip-weight', type=float, default=100.0)
    p.add_argument('--any-angle', action='store_true', help='lines at a random angle instead of axis-aligned')
    p.add_argument('--big-runs', choices=['all', 'free'], default='all', help='largest run over all runs, or over the free runs only')
    p.add_argument('--shear-weight', type=float, default=0.0, help='penalty on the off-diagonal of J relative to its diagonal')
    p.add_argument('--arch', choices=['resnet', 'tri', 'sep'], default='resnet')
    p.add_argument('--nodes-per-cell', type=int, default=8)
    p.add_argument('--rows-per-cell', type=int, default=2)
    p.add_argument('--hmax', type=float, default=2.0, help='tri / sep: bound on the log-density')
    p.add_argument('--tri-steps', type=int, default=2)
    p.add_argument('--cond-weight', type=float, default=0.0, help='tri: penalty on the dependence of the density on the other coordinate')
    p.add_argument('--curv-weight', type=float, default=0.0, help='tri / sep: penalty on the second difference of the log-density')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--log-every', type=int, default=250)
    return p


if __name__ == '__main__':
    a = arguments().parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from common import Logger
    log = Logger(a.out / 'log.txt')
    dev = 'cuda:0'
    maze, info = load_maze(a.prepared, dev)
    phi, hist, pl = fit(a, maze, log)
    torch.save(dict(state=phi.state_dict(), cfg=phi.cfg, args={k: str(v) for k, v in vars(a).items()}), a.out / 'phi.pt')
    json.dump(hist, open(a.out / 'history.json', 'w'))
