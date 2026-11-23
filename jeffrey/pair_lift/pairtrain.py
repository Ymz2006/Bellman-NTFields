"""jeffrey/lift_scale/trainer.py's loop (ntrl-demo's Model.train() step for step, plus the largest-solution push) on the pair network
(pairnet.py), with one more term that the pair coordinate makes necessary:

    triangle cap   for every pair of the minibatch and each of its two ends, a random step delta (uniform in a ball of radius `tri_r`
                   box lengths) to q' = q + delta whose straight segment stays out of obstacles:
                       v = relu(|T(q, g) - T(q', g)| - tau(q, q')),   tau = |delta| x mean of 1 / S along the step  (an upper bound of T*(q, q'))
                   loss += tri * mean(v^2) + tri1 * mean(v)
                   The pair coordinate is discontinuous in q (it jumps where the straight segment to the other end starts to cross an
                   obstacle), so the eikonal term, a derivative at a point, no longer ties the field's two branches together there; a
                   finite step does. Without it the push can lift the whole 'blocked' branch by a constant at no cost.

Everything else (seeding, B, the loss, beta, the epoch-repeat rule, the push and its one-sided term) is lift_scale's trainer; with pair
kind 'none' and tri = tri1 = 0 the run is that trainer's. The random steps come from a private generator, so the global stream the
network's draws and the rollbacks use is not touched.

    free-step TD   `td_free`: ntrl-demo's temporal-difference term is dropped for pairs whose step crosses an obstacle (pair_loss)
    Bellman backup `bell`: T(q, g) is regressed on the smallest of tau(q, q') + T(q', g) over K short steps that stay out of obstacles
                   (bellman_term): the selection term that needs no push. tau is the straight step's travel time, i.e. the length of the
                   lifted curve of the segment between the two configurations, used here as the local cost instead of as a network input.
"""
import copy
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = next((q for q in HERE.parents if (q / 'baselines/ntrl-demo').is_dir()), Path(os.environ.get('NTFIELD_ROOT', '/home/jeffrey/bellman-ntfield')))
sys.path.insert(0, str(HERE))
import pairnet as PN  # noqa: E402
from maze import OFFSET_CELLS, MARGIN_CELLS, ALPHA  # noqa: E402


def _speed(d):
    s = (d / MARGIN_CELLS).clamp(OFFSET_CELLS / MARGIN_CELLS, 1.0)
    return ALPHA * s * s * (2 - s) * (2 - s) + 1 - ALPHA


PN.LF_speed = _speed

_spec = importlib.util.spec_from_file_location('lift_scale_trainer', ROOT / 'jeffrey/lift_scale/trainer.py')
LT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LT)


def tri_term(net, scene, points, T, radius, gen, samples=6):
    """Mean squared and mean absolute violation of |T(q, g) - T(q', g)| <= tau over random free steps at both ends, and the share of steps used."""
    N, dev = len(points), points.device
    sq, lin, used, jump = 0.0, 0.0, 0.0, 0.0
    for end in (0, 1):
        sl = slice(3 * end, 3 * end + 3)
        delta = torch.randn(N, 3, generator=gen, device=dev)
        delta = delta / torch.linalg.vector_norm(delta, dim=1, keepdim=True) * radius * torch.rand(N, 1, generator=gen, device=dev) ** (1 / 3)
        a = points[:, sl]
        b = torch.minimum(torch.maximum(a + delta, scene.lo), scene.hi)
        with torch.no_grad():
            tau, ok = step_time(scene, a, b, samples, exact=False)
        new = points.clone()
        new[:, sl] = b
        Tn = net.out(new)[0][:, 0]
        v = torch.relu((T - Tn).abs() - tau)[ok]
        if len(v):
            sq = sq + (v ** 2).mean()
            lin = lin + v.mean()
            jump = max(jump, float(v.max()))
        used += float(ok.float().mean()) / 2
    return sq, lin, used, jump


class Slowness:
    """F(d) = int_d^margin dd' / S(d') (cells), continued linearly below the offset (floor speed) and above the margin (speed 1): the mean
    of 1 / S over a stretch along which the clearance runs linearly from d1 to d2 is (F(d1) - F(d2)) / (d2 - d1), exactly. 1 / S falls from 83
    to 9 within 0.0175 cells of the offset, which no fixed sampling of a step resolves; the antiderivative does."""

    def __init__(self, device, nodes=16385):
        d = np.linspace(OFFSET_CELLS, MARGIN_CELLS, nodes)
        inv = 1 / PN.LF.train_speed_np(d)
        seg = 0.5 * (inv[1:] + inv[:-1]) * np.diff(d)
        F = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
        self.table = torch.tensor(F, device=device, dtype=torch.float64)
        self.nodes, self.inv_floor, self.F0 = nodes, float(inv[0]), float(F[0])

    def F(self, d):
        d = d.double()
        dc = d.clamp(OFFSET_CELLS, MARGIN_CELLS)
        u = (dc - OFFSET_CELLS) / (MARGIN_CELLS - OFFSET_CELLS) * (self.nodes - 1)
        i = u.floor().long().clamp(0, self.nodes - 2)
        w = u - i
        out = (1 - w) * self.table[i] + w * self.table[i + 1]
        out = torch.where(d < OFFSET_CELLS, self.F0 + (OFFSET_CELLS - d) * self.inv_floor, out)
        return torch.where(d > MARGIN_CELLS, MARGIN_CELLS - d, out)

    def mean(self, d1, d2):
        """Mean of 1 / S between clearances d1 and d2 (cells)."""
        dd = (d2 - d1).double()
        flat = dd.abs() < 1e-5
        m = (self.F(d1) - self.F(d2)) / torch.where(flat, torch.ones_like(dd), dd)
        return torch.where(flat, 1 / PN.LF_speed(0.5 * (d1 + d2)).double(), m).float()


_slow = {}


def step_time(scene, a, b, samples=9, exact=True, clear=None):
    """Straight step a -> b (..., 3): its travel time (the step's length x the mean of 1 / S along it; `exact`: by Slowness between
    consecutive samples, otherwise the trapezoid rule on 1 / S) and whether it stays out of obstacles (clearance above -offset at every sample).
    clear: a clearance function other than the scene's exact one (pairnet.ClearTable: ten times faster for the rectangle robot)."""
    shape = a.shape[:-1]
    ts = torch.linspace(0, 1, samples, device=a.device)
    p = a[..., None, :] + ts[:, None] * (b - a)[..., None, :]
    d = (scene.clearance if clear is None else clear)(p.reshape(-1, 3)).reshape(*shape, samples)
    if exact:
        if a.device not in _slow:
            _slow[a.device] = Slowness(a.device)
        mean = _slow[a.device].mean(d[..., :-1], d[..., 1:]).mean(-1)
    else:
        inv = 1 / scene.train_speed(d)
        mean = (0.5 * (inv[..., 1:] + inv[..., :-1])).mean(-1)
    return torch.linalg.vector_norm(b - a, dim=-1) * mean, d.amin(-1) > -OFFSET_CELLS


def learned_step_time(net, points, sl, a, b, n, gen):
    """Step cost from the field's own speed: |b - a| x the mean of |grad T| (the predicted slowness, gradient at the moving end with the
    pair's other end as the goal) at n stratified random points of the straight step a -> b (an unbiased estimate of the line integral).
    a (N, 3), b (N, K, 3); no gradient is kept."""
    N, K = b.shape[:2]
    out = torch.empty(N, K, device=a.device)
    chunk = max(1, 120000 // (K * n))
    for i in range(0, N, chunk):
        aa, bb = a[i:i + chunk], b[i:i + chunk]
        m = len(aa)
        t = (torch.arange(n, device=a.device) + torch.rand(m, K, n, generator=gen, device=a.device)) / n
        p = aa[:, None, None, :] + t[..., None] * (bb - aa[:, None, :])[:, :, None, :]
        x = points[i:i + chunk, None, None, :].repeat(1, K, n, 1)
        x[..., sl] = p
        with torch.enable_grad():
            tau, _, xp = net.out(x.reshape(-1, 6))
            g, = torch.autograd.grad(tau.sum(), xp)
        slow = torch.linalg.vector_norm(g[:, sl], dim=1).reshape(m, K, n).mean(2)
        out[i:i + chunk] = torch.linalg.vector_norm(bb - aa[:, None, :], dim=2) * slow
    return out.detach()


def pair_loss(function, scene, points, Yobs, normal, beta, td_len=0.03, td_free=False, td_weight=1e-3, pair_w=None):
    """lift_scale.trainer.ntrl_loss (ntrl-demo's metric loss, line for line) with one option: td_free drops the temporal-difference term of
    a pair whose step (0.03 box lengths down the field's own gradient, taken by ntrl-demo with no collision check) does not stay out of
    obstacles. Such a step equates the travel time on one side of a wall with the travel time on the other plus the step's cost at the
    starting point's speed. Also returns the gradients at both ends."""
    net, dim = function.network, function.dim
    tau, w, Xp = net.out(points)
    dtau = function.gradient(tau, Xp)
    DT0 = dtau[:, :dim]
    DT1 = dtau[:, dim:]
    S0 = torch.einsum('ij,ij->i', DT0, DT0)
    S1 = torch.einsum('ij,ij->i', DT1, DT1)
    free0 = free1 = None
    with torch.no_grad():
        length0 = td_len / (Yobs[:, 0]).unsqueeze(1)
        Dir0 = length0 * (DT0 * Yobs[:, 0].unsqueeze(1) ** 2).clone().detach()
        Xp_new0 = Xp.clone().detach()
        Xp_new0[:, :dim] = Xp_new0[:, :dim] - Dir0
        if td_free:
            _, free0 = step_time(scene, Xp.detach()[:, :dim], Xp_new0[:, :dim], 6, exact=False)
        tau_new0, w, Xp_new0 = net.out(Xp_new0)
        tau_new1 = length0
        del Xp_new0, Dir0
    tau_loss0 = td_weight * ((tau - (tau_new0 + tau_new1)) ** 2).squeeze()
    with torch.no_grad():
        length1 = td_len / (Yobs[:, 1]).unsqueeze(1)
        Dir1 = length1 * (DT1 * Yobs[:, 1].unsqueeze(1) ** 2).clone().detach()
        Xp_new0 = Xp.clone().detach()
        Xp_new0[:, dim:] = Xp_new0[:, dim:] - Dir1
        if td_free:
            _, free1 = step_time(scene, Xp.detach()[:, dim:], Xp_new0[:, dim:], 6, exact=False)
        tau_new0, w, Xp_new0 = net.out(Xp_new0)
        tau_new1 = length1
        del Xp_new0, Dir1
    tau_loss1 = td_weight * ((tau - (tau_new0 + tau_new1)) ** 2).squeeze()
    where_d0 = (tau[:, 0] < length0.squeeze())
    where_d1 = (tau[:, 0] < length1.squeeze())
    tau_loss0[where_d0] = 0
    tau_loss1[where_d1] = 0
    if td_free:
        tau_loss0[~free0] = 0
        tau_loss1[~free1] = 0
    tau_loss = tau_loss0 + tau_loss1
    Ypred0 = torch.sqrt(S0 + 1e-8)
    Ypred1 = torch.sqrt(S1 + 1e-8)
    l0 = ((Yobs[:, 0] * (Ypred0)))
    l1 = ((Yobs[:, 1] * (Ypred1)))
    l0_2 = (torch.sqrt(l0))
    l1_2 = (torch.sqrt(l1))
    loss_weight = 1e-2
    loss0 = loss_weight * (l0_2 - 1) ** 2
    loss1 = loss_weight * (l1_2 - 1) ** 2
    T = tau[:, 0]
    diff = loss0 + loss1
    normal_weight = 1e-3
    normal0 = normal[:, :dim]
    normal1 = normal[:, dim:]
    n_loss0 = (1.001 - Yobs[:, 0].unsqueeze(1)) * (Yobs[:, 0].unsqueeze(1) * DT0 + normal0) ** 2
    n_loss1 = (1.001 - Yobs[:, 1].unsqueeze(1)) * (Yobs[:, 1].unsqueeze(1) * DT1 + normal1) ** 2
    n_loss = normal_weight * (torch.sum(n_loss0, dim=1) + torch.sum(n_loss1, dim=1))
    # pair_w: a weight per pair (`--in-weight`: the pairs with an end inside an obstacle); None is ntrl-demo's loss
    loss_n = (torch.sum((diff + n_loss + tau_loss) * torch.exp(-0.5 * T) * (1.0 if pair_w is None else pair_w))) / Yobs.shape[0]
    loss = beta * loss_n
    td_blocked = float(1 - 0.5 * (free0.float().mean() + free1.float().mean())) if td_free else float('nan')
    return loss, loss_n, T, l0_2, l1_2, DT0.detach(), DT1.detach(), td_blocked


def bellman_term(net, scene, points, T, grads, radius, K, gen, one_sided=False, cone=0, exact=True, huber=0.0, clear=None, long=1.0, keep_cone=False, free_goal=False, tau_mode='true', ls=4, nofree=False, speed=None, speed_check=0.0):
    """Free-step Bellman backup at both ends of every pair (q, g):

        y = min over K steps q -> q' that stay out of obstacles of [ tau(q, q') + T(q', g) ]          (no gradient through y)
        term = mean (T(q, g) - y)^2

    tau is the travel time of the straight step (|step| x mean of 1 / S along it: for a short step this is the length of the lifted curve
    of the segment), the steps have lengths in [radius / 2, radius] box lengths, K - 2 random directions plus two along the field's own
    descent direction (-grad T). A field that descends into a wall has no free step that lowers it there, so y > T: the backup raises
    exactly the configurations a see-through field gets wrong, and its fixed point is the shortest-path field. Pairs closer than 2 radius
    are left to the eikonal term."""
    N, dev = len(points), points.device
    total, used, gap = 0.0, 0.0, 0.0
    for end in (0, 1):
        sl = slice(3 * end, 3 * end + 3)
        other = slice(3 - 3 * end, 6 - 3 * end)
        a = points[:, sl]
        dirs = torch.randn(N, K, 3, generator=gen, device=dev)
        g = -grads[end]
        g = g / torch.linalg.vector_norm(g, dim=1, keepdim=True).clamp(min=1e-12)
        length = radius * (0.5 + 0.5 * torch.rand(N, K, 1, generator=gen, device=dev))
        dirs[:, 0] = g
        dirs[:, 1] = g
        length[:, 0] = radius
        length[:, 1] = 0.5 * radius
        if cone:
            # `cone` more candidates around the descent direction (one of them at a quarter of the radius, straight down the gradient)
            dirs = dirs / torch.linalg.vector_norm(dirs, dim=2, keepdim=True).clamp(min=1e-12)
            dirs[:, 2:2 + cone] = g[:, None, :] + 0.5 * dirs[:, 2:2 + cone]
            dirs[:, 2] = g
            length[:, 2] = 0.25 * radius
            length[:, 3:2 + cone] = radius * (0.25 + 0.75 * torch.rand(N, cone - 1, 1, generator=gen, device=dev))
        dirs = dirs / torch.linalg.vector_norm(dirs, dim=2, keepdim=True).clamp(min=1e-12)
        if long > 1:
            # every candidate but the two fixed descent steps gets a length log-uniform in [radius / 2, long x radius]: in open space a
            # long straight step is free and carries the backup `long` times farther per update
            lo, hi = np.log(0.5 * radius), np.log(long * radius)
            # (keep_cone: the candidates around the descent direction keep their short lengths, which is where the near-optimal step is)
            first = 2 + cone if keep_cone else 2
            length[:, first:] = torch.exp(lo + (hi - lo) * torch.rand(N, K - first, 1, generator=gen, device=dev))
        with torch.no_grad():
            b = torch.minimum(torch.maximum(a[:, None, :] + dirs * length, scene.lo), scene.hi)
            tau, free = step_time(scene, a[:, None, :].expand(-1, K, -1), b, (int(min(17 * max(long, 1.0), 137)) if exact else 6), exact=exact, clear=clear)
            if tau_mode == 'learned':
                # the step's cost from the field's own predicted speed in place of the true speed along the step (the lifted length)
                tau = learned_step_time(net, points, sl, a, b, int(min(ls * max(long, 1.0), 32)), gen)
            if tau_mode == 'field':
                # the step's cost from the speed another (frozen) field predicts, read from its table (pairnet.SpeedTable)
                tau, smax = speed.step_time(a[:, None, :].expand(-1, K, -1), b, int(min(33 * max(long, 1.0), 265)), with_max=True)
                if speed_check > 0:
                    # collision check from the predicted speed: a step is a candidate if the frozen field's speed stays above speed_check along it
                    free = smax < 1.0 / speed_check
            if nofree and not (tau_mode == 'field' and speed_check > 0):
                free = torch.ones_like(free)
            new = points[:, None, :].repeat(1, K, 1)
            new[:, :, sl] = b
            Tn = net.out(new.reshape(-1, 6))[0][:, 0].reshape(N, K)
            cand = torch.where(free, tau + Tn, torch.full_like(Tn, float('inf')))
            y, _ = cand.min(1)
            ok = torch.isfinite(y) & (torch.linalg.vector_norm(a - points[:, other], dim=1) > 2 * radius)
            if free_goal:
                # obstacle-space sampling: a pair whose other end is inside an obstacle is left to ntrl-demo's loss
                ok = ok & ((scene.clearance if clear is None else clear)(points[:, other]) > -OFFSET_CELLS)
        if bool(ok.any()):
            r = (T - y)[ok]
            r2 = torch.relu(r) if one_sided else r
            total = total + (torch.nn.functional.huber_loss(r2, torch.zeros_like(r2), delta=huber) * 2 if huber > 0 else (r2 ** 2).mean())
            gap += float(r.mean()) / 2
        used += float(ok.float().mean()) / 2
    return total, used, gap


def train(data_dir, out, seed, epochs, cfg, scene, log, blocked=None, schedule=None, mu=0.1, bound=3.0, push_all=False, save_at=(), mu1=0.0, b_scale=1.0,
          tri=0.0, tri1=0.0, tri_r=0.03, bell=0.0, bell_r=0.03, bell_k=8, bell_from=0, td_free=False, td_weight=1e-3, bell_cone=0, bell_exact=True, bell_huber=0.0, bell_table=False, bell_long=1.0, bell_keep=False, bell_free=False, in_weight=1.0, bell_tau='true', bell_ls=4, bell_nofree=False, bell_speed=None, init=None, bell_speed_check=0.0):
    mt, mf, mn, db = LT.ntrl_modules()
    dev = 'cuda:0'
    out = Path(out)
    (out / 'checkpoints').mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    B = torch.normal(0, 1, size=(128, 3))
    torch.nn.init.trunc_normal_(B, mean=0.0, std=1, a=-2.0, b=2.0)
    if b_scale != 1.0:
        B = B * b_scale
    cfg = dict(cfg)
    cfg['lift'] = dict(cfg.get('lift') or {}, seed=seed)
    cfg['pair'] = dict(cfg.get('pair') or {}, seed=seed)
    network = PN.build_network(mn, dev, B, cfg, scene)
    if init is not None:
        # warm start from a finished field of the same network (same seed, so the same Fourier matrix): the new loss terms are added to it
        payload = torch.load(init, map_location=dev, weights_only=False)
        assert torch.equal(payload['B_state_dict'].cpu(), B.cpu()), 'the warm start has another Fourier matrix (seed or scale)'
        network.load_state_dict(payload['model_state_dict'], strict=True)
        log(f'warm start from {init} (epoch {payload["epoch"]}, loss {payload["val_loss"]:.4e})')
        del payload
    function = mf.Function(str(out), dev, network, 3)
    optimizer = torch.optim.AdamW([{'params': list(network.parameters())}], lr=1e-3, weight_decay=0.5)
    dataset = db.Database(str(data_dir) + '/')
    data = dataset.data
    schedule = list(schedule or [])
    if blocked is not None:
        assert len(blocked) == len(data), 'the blocked flags belong to other pairs'
        data = torch.cat([data, torch.as_tensor(blocked, dtype=data.dtype)[:, None]], 1)
    else:
        assert not schedule, 'the push needs the blocked flag of every training pair'
    loader = mt.FastTensorDataLoader(data, batch_size=2000, shuffle=True)
    tri_gen = torch.Generator(dev).manual_seed(seed + 99)
    bell_clear = PN.ClearTable(scene, 32, 49) if (bell > 0 and bell_table) else None
    use_tri = tri > 0 or tri1 > 0
    beta = 1.0
    current_diff = 1000.0
    keys = ('epochs', 'ntrl_loss', 'rollbacks', 'lam', 'push', 'one_sided', 'pushed_share', 't_blocked', 't_free', 'tri_sq', 'tri_lin', 'tri_used', 'tri_max', 'bell', 'bell_gap', 'bell_used', 'td_blocked')
    history = {k: [] for k in keys}
    history['all_loss'] = []
    current_state = copy.deepcopy(network.state_dict())
    current_opt = copy.deepcopy(optimizer.state_dict())
    prev_states, prev_opts = [], []
    train_loss = []
    t_start = time.time()
    save_at = set(int(e) for e in save_at) | {epochs}
    log(f'network: lift {network.lift_kind} gain {network.lift_gain}, pair {cfg["pair"]}, octaves {network.octaves}, parameters {sum(p.numel() for p in network.parameters())}, '
        f'push schedule {schedule}, mu {mu}, mu1 {mu1}, bound {bound}, triangle cap {tri} / {tri1} radius {tri_r}, Bellman backup {bell} radius {bell_r} K {bell_k} cone {bell_cone} exact step time {bell_exact} clearance table {bell_table} huber {bell_huber} longest step x{bell_long} (cone kept short: {bell_keep}), backup only for pairs with both ends free {bell_free}, weight of the pairs with an end inside an obstacle {in_weight}, step cost {bell_tau} ({bell_ls} samples), collision check of the steps {not bell_nofree} from epoch {bell_from}, '
        f'TD weight {td_weight} free steps only {td_free}')
    for epoch in range(1, epochs + 1):
        gamma = 0.001
        alpha = 1.025
        prev_states.append(current_state)
        prev_opts.append(current_opt)
        if len(prev_states) > 5:
            prev_states.pop(0)
            prev_opts.pop(0)
        current_state = copy.deepcopy(network.state_dict())
        current_opt = copy.deepcopy(optimizer.state_dict())
        optimizer.param_groups[0]['lr'] = 5e-4
        prev_diff = current_diff
        rollbacks = 0
        lam = LT.lam_at(schedule, epoch)
        ramp = 1.0 if len(schedule) < 2 else min(1.0, max(0.0, (epoch - schedule[0][0]) / max(schedule[1][0] - schedule[0][0], 1)))
        while True:
            total_train_loss, total_diff = 0.0, 0.0
            acc = dict(push=0.0, one=0.0, share=0.0, tb=0.0, tf=0.0, tsq=0.0, tlin=0.0, tused=0.0, tmax=0.0, bell=0.0, bgap=0.0, bused=0.0, tdb=0.0)
            ii = 0
            for i, whole in enumerate(loader, 0):
                if ii > 4:
                    break
                ii += 1
                batch = whole[0].to(dev)
                points = batch[:, :6]
                speed = batch[:, 6:8]
                normal = batch[:, 8:14]
                speed = speed * speed * (2 - speed) * (2 - speed)
                speed = alpha * speed + 1 - alpha
                pair_w = None
                if in_weight != 1.0:
                    with torch.no_grad():
                        cl = scene.clearance if bell_clear is None else bell_clear
                        inside = (cl(points[:, :3]) <= -OFFSET_CELLS) | (cl(points[:, 3:6]) <= -OFFSET_CELLS)
                        pair_w = torch.where(inside, in_weight, 1.0)
                loss_value, loss_n, T, r0, r1, g0, g1, tdb = pair_loss(function, scene, points, speed, normal, beta, td_free=td_free, td_weight=td_weight, pair_w=pair_w)
                acc['tdb'] += tdb
                extra = 0.0
                if blocked is not None:
                    blk = batch[:, 14] > 0.5
                    with torch.no_grad():
                        acc['tb'] += float(T[blk].mean()) if bool(blk.any()) else 0.0
                        acc['tf'] += float(T[~blk].mean()) if bool((~blk).any()) else 0.0
                    if lam > 0:
                        m = (torch.ones_like(blk) if push_all else blk) & (T.detach() < bound)
                        push = -lam * T[m].mean() if bool(m.any()) else T.sum() * 0
                        one = mu * ramp * (torch.relu(r0 - 1) ** 2 + torch.relu(r1 - 1) ** 2).mean()
                        if mu1 > 0:
                            one = one + mu1 * ramp * (torch.relu(r0 - 1) + torch.relu(r1 - 1)).mean()
                        extra = extra + push + one
                        acc['push'] += float(push)
                        acc['one'] += float(one)
                        acc['share'] += float(m.float().mean())
                if use_tri:
                    sq, lin, used, vmax = tri_term(network, scene, points, T, tri_r, tri_gen)
                    extra = extra + tri * sq + tri1 * lin
                    acc['tsq'] += float(sq)
                    acc['tlin'] += float(lin)
                    acc['tused'] += used
                    acc['tmax'] = max(acc['tmax'], vmax)
                if bell > 0 and epoch >= bell_from:
                    bt, bused, bgap = bellman_term(network, scene, points, T, (g0, g1), bell_r, bell_k, tri_gen, cone=bell_cone, exact=bell_exact, huber=bell_huber, clear=bell_clear, long=bell_long, keep_cone=bell_keep, free_goal=bell_free, tau_mode=bell_tau, ls=bell_ls, nofree=bell_nofree, speed=bell_speed, speed_check=bell_speed_check)
                    extra = extra + bell * bt
                    acc['bell'] += float(bt)
                    acc['bgap'] += bgap
                    acc['bused'] += bused
                if not isinstance(extra, float):
                    loss_value = loss_value + beta * extra
                loss_value.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_train_loss += loss_value.item()
                total_diff += loss_n.item()
            total_train_loss /= 4
            total_diff /= 4
            current_diff = total_diff
            ratio = current_diff / prev_diff
            if 0 < ratio < 1.2:
                break
            rollbacks += 1
            if rollbacks > 2000:
                raise RuntimeError(f'epoch {epoch}: 2000 rollbacks, the epoch-repeat rule does not let the training continue')
            with torch.no_grad():
                k = random.randint(0, len(prev_states) - 1)
                network.load_state_dict(prev_states[k], strict=True)
                optimizer.load_state_dict(prev_opts[k])
        train_loss.append(total_train_loss)
        beta = 1.0 / total_diff
        history['all_loss'].append(total_diff)
        if epoch % 10 == 0 or rollbacks:
            for k, v in zip(keys, (epoch, total_diff, rollbacks, lam, acc['push'] / 5, acc['one'] / 5, acc['share'] / 5, acc['tb'] / 5, acc['tf'] / 5,
                                   acc['tsq'] / 5, acc['tlin'] / 5, acc['tused'] / 5, acc['tmax'], acc['bell'] / 5, acc['bgap'] / 5, acc['bused'] / 5, acc['tdb'] / 5)):
                history[k].append(v)
        if epoch % 100 == 0:
            el = time.time() - t_start
            msg = f'epoch {epoch}: ntrl loss {total_diff:.4e}, rollbacks {rollbacks}, {el / 60:.1f} min ({el / epoch:.3f} s/epoch)'
            if blocked is not None:
                msg += f', lambda {lam:.2e}, push {acc["push"] / 5:.3e}, one-sided {acc["one"] / 5:.3e}, T blocked {acc["tb"] / 5:.3f} / free {acc["tf"] / 5:.3f}'
            if use_tri:
                msg += f', triangle sq {acc["tsq"] / 5:.3e} lin {acc["tlin"] / 5:.3e} max {acc["tmax"]:.3f} (steps used {acc["tused"] / 5:.2f})'
            if bell > 0:
                msg += f', Bellman sq {acc["bell"] / 5:.3e} mean T - y {acc["bgap"] / 5:+.4f} (pairs used {acc["bused"] / 5:.2f})'
            if td_free:
                msg += f', TD steps into obstacles {acc["tdb"] / 5:.2f}'
            log(msg)
        if epoch in save_at:
            torch.save({'epoch': epoch, 'model_state_dict': network.state_dict(), 'B_state_dict': B, 'val_loss': total_diff, 'embed_config': network.cfg},
                       out / 'checkpoints' / f'epoch_{epoch:05d}.pt')
    final = out / 'checkpoint.pt'
    torch.save({'epoch': epochs, 'model_state_dict': network.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'B_state_dict': B,
                'train_loss': train_loss, 'val_loss': total_diff, 'embed_config': network.cfg}, final)
    history['final_loss'] = total_diff
    history['seconds'] = time.time() - t_start
    with open(out / 'history.json', 'w') as f:
        json.dump(history, f)
    return final, history
