"""Fit psi = phi^-1 from the collision oracle alone (no travel-time field, no planner).

Every step: `--lines` axis-aligned lines of the z box (random axis, random slice) are read through psi and the oracle (scan.py), the runs go to the loss (losses.py), and the gradient reaches the network only
through the crossings. Limits on the map itself: --kappa bounds each singular value of J to [1 / kappa, kappa], --aniso their ratio. --trust bounds how far a single update may move the map.
"""
import argparse
import math
import os
import time
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np
import torch

torch.set_float32_matmul_precision('high')
import losses as LS
import maps as M
import measure as ME
import scan as S


def build(a, scene):
    """--scales are cycles per cell (a cell of the scene), so a map sees the same structure on every maze size."""
    scales = [float(s) * scene.E for s in a.scales.split(',')]
    d = scene.dim
    if a.arch == 'identity':
        return M.Identity()
    if a.arch in ('flow', 'res'):
        return M.FlowMap(d, a.flow_steps, a.hidden, a.depth, scales, a.per_scale, a.seed, a.arch == 'flow', a.amp)
    knots = int(round(a.knots_per_cell * scene.E))
    return M.CouplingMap(d, a.layers, knots, a.hidden, a.depth, scales, a.per_scale, a.hmax, a.seed, a.arch == 'coupling')


def reg_terms(psi, z, eps, a):
    """Bound and anisotropy hinge on log singular values of J = d psi / d z, fold hinge on det J. Returns the sum and the numbers it came from."""
    J = M.jacobian_fd(psi, z, eps)
    sv = M.singular_values(J)
    l = torch.log(sv.clamp(min=1e-6))
    det = torch.linalg.det(J)
    lim = math.log(a.kappa)
    bound = (torch.relu(l - lim).pow(2) + torch.relu(-l - lim).pow(2)).sum(1).mean()
    fold = torch.relu(0.02 - det).pow(2).mean()
    total = a.bound_weight * bound + a.fold_weight * fold
    if a.aniso > 0:
        total = total + a.bound_weight * torch.relu(l[:, 0] - l[:, -1] - math.log(a.aniso)).pow(2).mean()
    return total, dict(bound=float(bound.detach()), fold=float(fold.detach()), s_max=float(sv[:, 0].max().detach()), s_min=float(sv[:, -1].min().detach()), det_min=float(det.min().detach()), l=l, J=J)


def tv_term(psi, z, l, eps, delta, near=None):
    """Mean |log s(z + d) - log s(z)| / |d| over the singular values, d random of length delta: the rise and fall of log stretch across the box, in any dimension. near: per-point weights."""
    d = torch.randn_like(z)
    d = d / d.norm(dim=1, keepdim=True).clamp(min=1e-9) * delta
    y = (z + d).clamp(-0.5 + eps, 0.5 - eps)
    l2 = torch.log(M.singular_values(M.jacobian_fd(psi, y, eps)).clamp(min=1e-6))
    v = (l2 - l).abs().sum(1)
    return ((v * near).sum() / near.sum().clamp(min=1e-9) if near is not None else v.mean()) / delta


def near_walls(psi, scene, z, r, gen):
    """Stiffness of each point: exp(-|clearance| / r), r in the oracle's units. Stiff where the speed has structure to represent, soft in open space, and it needs only the oracle."""
    with torch.no_grad():
        k = torch.randint(scene.planes, (len(z),), generator=gen, device=z.device)
        return torch.exp(-scene.sdf(psi(z), k).abs() / r)


def line_loss(a, r, lines):
    if a.kind == 'equal':
        return LS.equal(r['ell'], r['lid'], lines, a.form, a.scope, a.weight, a.target, a.top, a.rest, a.floor, a.power)
    if a.kind == 'cap':
        return LS.cap(r['ell'], r['lid'], lines, a.target, a.top, a.rest, a.power)
    return LS.hard(r['ell'], r['lid'], lines, a.floor)


def batch_rays(a, scene, gen, dev):
    """Chords of one batch: along the axes, at any angle, or half and half."""
    if a.directions == 'axis':
        return S.lines_uniform(a.lines, scene.dim, scene.planes, gen, dev)
    if a.directions == 'any':
        return S.any_rays(a.lines, scene.dim, scene.planes, gen, dev)
    h = a.lines // 2
    x, y = S.lines_uniform(h, scene.dim, scene.planes, gen, dev), S.any_rays(a.lines - h, scene.dim, scene.planes, gen, dev)
    return S.Rays(*[torch.cat([p, q]) for p, q in zip(x, y)])


def anchor_step(a, psi, scene, rays, opt, params):
    """The limit on the sections that are not the worst: on this batch of lines, `inner` updates push the worst runs while the change of every other run against the start of the update is penalized (weight --anchor).
    The worst runs are the `anchor_top` share of the batch by error, or the largest run of each line when anchor_top is 0."""
    with torch.no_grad():
        r0 = S.read(psi, scene, rays, a.samples, a.graze, differentiate=False)
        v = r0['ell'].clamp(min=1e-6)
        ref = a.target if a.target > 0 else float(v.median())
        err = torch.log(v / ref) ** 2 * (v / ref).clamp(min=1.0)
        if a.anchor_top > 0:
            free_to_move = err >= err.topk(max(1, int(a.anchor_top * len(err)))).values[-1]
        else:
            worst = err.new_zeros(a.lines).scatter_reduce(0, r0['lid'], err, 'amax', include_self=True)
            free_to_move = err >= worst[r0['lid']]
    for _ in range(a.inner):
        r = S.read(psi, scene, rays, a.samples, a.graze)
        loss = line_loss(a, r, a.lines) + a.anchor * LS.anchored(r['ell'], r['lid'], r['count'], r0['ell'], r0['count'], ~free_to_move)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if a.clip > 0:
            torch.nn.utils.clip_grad_norm_(params, a.clip)
        opt.step()


def fit(a, scene, log, psi=None, on_step=None):
    dev = scene.table.v.device
    psi = (psi if psi is not None else build(a, scene)).to(dev)
    params = [p for p in psi.parameters() if p.requires_grad]
    if not params:                                      # the identity control: nothing to fit
        return psi, []
    opt = torch.optim.Adam(params, lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=a.lr * 0.02)
    gen = torch.Generator(dev).manual_seed(a.seed + 1)
    eps = 0.02 / scene.E
    hist = []
    t0 = time.time()
    groups = len(a.scales.split(','))
    for step in range(1, a.steps + 1):
        if a.window > 0:
            psi.window(min(groups, 1 + (groups - 1) * step / (a.window * a.steps)))
        if a.kind == 'pair':
            sigma = a.sigma * (a.sigma_end / a.sigma) ** (step / a.steps)
            line = LS.pair(psi, scene, a.pairs, a.target, sigma, gen, dev)
            with torch.no_grad():
                r = S.read(psi, scene, S.lines_uniform(128, scene.dim, scene.planes, gen, dev), a.samples, a.graze, differentiate=False)
        else:
            rays = batch_rays(a, scene, gen, dev)
            if a.anchor > 0:
                anchor_step(a, psi, scene, rays, opt, params)
            r = S.read(psi, scene, rays, a.samples, a.graze)
            line = line_loss(a, r, a.lines)
        z = (torch.rand(a.reg_points, scene.dim, generator=gen, device=dev) - 0.5) * (1 - 2 * eps)
        reg, info = reg_terms(psi, z, eps, a)
        loss = line + reg
        lam = near_walls(psi, scene, z, a.near, gen) if a.near > 0 else None
        if a.tv_weight > 0:
            loss = loss + a.tv_weight * tv_term(psi, z, info['l'], eps, a.tv_delta / scene.E, lam)
        if a.iso_weight > 0:
            v = (info['l'] ** 2).sum(1)
            loss = loss + a.iso_weight * ((v * lam).sum() / lam.sum().clamp(min=1e-9) if lam is not None else v.mean())
        if a.shape_weight > 0:
            v = (info['l'][:, 0] - info['l'][:, -1]) ** 2
            loss = loss + a.shape_weight * ((v * lam).sum() / lam.sum().clamp(min=1e-9) if lam is not None else v.mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if a.clip > 0:
            torch.nn.utils.clip_grad_norm_(params, a.clip)
        if a.trust > 0:
            with torch.no_grad():
                old = [p.detach().clone() for p in params]
                before = psi(z)
        opt.step()
        moved = 0.0
        if a.trust > 0:
            with torch.no_grad():
                moved = float((psi(z) - before).norm(dim=1).max()) * scene.E
                if moved > a.trust:
                    f = a.trust / moved
                    for p, o in zip(params, old):
                        p.mul_(f).add_(o, alpha=1 - f)
        sched.step()
        if step % a.log_every == 0 or step == 1:
            v = r['ell'].detach().clamp(min=1e-6)
            w = v / v.sum()
            lg = torch.log(v)
            h = dict(step=step, line=float(line), runs_per_line=float(r['count'].float().mean()), log_std_run=float(lg.std()), log_std_len=float((w * (lg - (w * lg).sum()) ** 2).sum().sqrt()), s_max=info['s_max'], s_min=info['s_min'], det_min=info['det_min'],
                     bound=info['bound'], fold=info['fold'], kept=float(r['keep'].float().mean()), moved=moved, seconds=time.time() - t0)
            hist.append(h)
            log(f"step {step}: line {h['line']:.4f}, runs/line {h['runs_per_line']:.1f}, log std run {h['log_std_run']:.3f} len {h['log_std_len']:.3f}, sv {h['s_min']:.3f}..{h['s_max']:.2f}, det min {h['det_min']:.3f}, bound {h['bound']:.1e}, kept {h['kept']:.3f}, moved {moved:.3f}, {h['seconds']:.0f}s")
        if on_step is not None:
            on_step(step, psi)
    if a.window > 0:
        psi.window(groups)
    return psi, hist


def arguments(p=None):
    p = p or argparse.ArgumentParser()
    p.add_argument('--scene', default='plain:10', help="'plain:10', 'gap:maze10:12', 'gap:cross22:12', 'plus:22:6', 'se2/plain:10', 'rooms:0'")
    p.add_argument('--out', required=True)
    p.add_argument('--headings', type=int, default=9)
    p.add_argument('--res', type=int, default=32, help='table steps per cell')
    p.add_argument('--arch', choices=['flow', 'res', 'coupling', 'sep', 'identity'], default='coupling')
    p.add_argument('--kind', choices=['equal', 'hard', 'cap', 'pair'], default='equal')
    p.add_argument('--form', choices=['lv', 'cv'], default='lv')
    p.add_argument('--scope', choices=['line', 'batch'], default='batch')
    p.add_argument('--weight', choices=['run', 'len'], default='len')
    p.add_argument('--target', type=float, default=0.0, help='fixed target run length (box units); 0: the batch sets it')
    p.add_argument('--pairs', type=int, default=65536, help='pair loss: point pairs per step')
    p.add_argument('--sigma', type=float, default=0.3, help='pair loss: width of the soft step, in clearance units (cells), at the start')
    p.add_argument('--sigma-end', type=float, default=0.05, help='... and at the end')
    p.add_argument('--iso-weight', type=float, default=0.0, help='penalty on (log singular value)^2: stay close to an isometry unless the line force is worth it')
    p.add_argument('--power', type=float, default=0.0, help='a run of a line is weighted by (its error / the largest error of the line)^power')
    p.add_argument('--anchor', type=float, default=0.0, help='weight on the change of every run that is not among the worst, within a batch of inner updates')
    p.add_argument('--anchor-top', type=float, default=0.0, help='share of runs free to move (0: the largest run of each line)')
    p.add_argument('--inner', type=int, default=3, help='inner updates per batch with --anchor')
    p.add_argument('--near', type=float, default=0.0, help='stiffness terms act near obstacles only: exp(-|clearance| / near), clearance units (cells)')
    p.add_argument('--shape-weight', type=float, default=0.0, help='penalty on (log largest - log smallest singular value)^2')
    p.add_argument('--top', type=float, default=1.0, help='share of runs, ranked by error, that keeps full weight')
    p.add_argument('--rest', type=float, default=1.0, help='weight of the other runs')
    p.add_argument('--floor', type=float, default=0.25, help='runs below this x the target get no push')
    p.add_argument('--graze', type=float, default=0.05)
    p.add_argument('--directions', choices=['axis', 'any', 'mix'], default='axis', help='lines along the axes, at any angle, or half and half')
    p.add_argument('--lines', type=int, default=512)
    p.add_argument('--samples', type=int, default=256)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--trust', type=float, default=0.0, help='largest move of the map in one update, cells (0: none)')
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--scales', default='0.05,0.1,0.2,0.4,0.8', help='Fourier scales, cycles per cell')
    p.add_argument('--per-scale', type=int, default=32)
    p.add_argument('--window', type=float, default=0.5, help='share of the steps over which the higher Fourier scales fade in (0: all on)')
    p.add_argument('--flow-steps', type=int, default=8)
    p.add_argument('--amp', type=float, default=4.0, help='flow / res: bound on the velocity')
    p.add_argument('--layers', type=int, default=6)
    p.add_argument('--knots-per-cell', type=float, default=8.0)
    p.add_argument('--hmax', type=float, default=3.0)
    p.add_argument('--kappa', type=float, default=16.0)
    p.add_argument('--aniso', type=float, default=0.0)
    p.add_argument('--bound-weight', type=float, default=1.0)
    p.add_argument('--fold-weight', type=float, default=100.0)
    p.add_argument('--tv-weight', type=float, default=0.0)
    p.add_argument('--tv-delta', type=float, default=0.25, help='cells')
    p.add_argument('--reg-points', type=int, default=1024)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--log-every', type=int, default=250)
    return p
