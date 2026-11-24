"""LBS trainer for a joint space of any dimension: jeffrey/lbs/trainer.py with the dimension taken from the scene (there: 3), the
pieces of core.py in place of lift.py / liftfast.py / bell.py / traverse.py, and three additions that the 6-joint arm needs:

    freq        the Fourier matrix of the joints is multiplied by cfg['freq'] (one factor per joint; 1 = ntrl-demo's draw)
    bell_n      the Bellman backup is computed on that many random pairs of each minibatch (0 = all): its step cost needs exact
                clearances along every candidate step, and the arm has no clearance table
    pair_max    traversal pairs (a, b) are at most this far apart in the normalized joint space (0 = any); the frozen cabinet
                queries are 0.02 to 0.57 apart

Everything else (ntrl-demo's loop: 5 minibatches of 2,000 pairs per epoch, eikonal + TD + normal terms with weight exp(-0.5 T),
beta = 1 / previous loss, AdamW, rollback when the loss rises 20 %; the Bellman term from epoch 1; the midpoint term from the start
epoch with refreshed traversals) is jeffrey/lbs/trainer.py's, which documents it.

    loss = beta * (loss_ntrl + bell * mean (T - y)^2) + lambda(epoch) * mean | log T - log y_mid |
"""
import copy
import json
import math
import os
import pickle
import random
import sys
import time
import types
import warnings
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = next((q for q in HERE.parents if (q / 'baselines/ntrl-demo').is_dir()), Path(os.environ.get('NTFIELD_ROOT', '/home/jeffrey/bellman-ntfield')))
NTRL = ROOT / 'baselines/ntrl-demo'
sys.path.insert(0, str(HERE))
import core as C  # noqa: E402
import tree as TR  # noqa: E402  (verbatim copy of jeffrey/lbs/tree.py)


def ntrl_modules():
    warnings.filterwarnings('ignore', message="input's size at dim=0 does not match num_features")
    warnings.filterwarnings('ignore', message='You are using `torch.load` with `weights_only=False`')
    sys.modules['pickle5'] = pickle
    igl = types.ModuleType('igl')
    igl.write_triangle_mesh = lambda *a, **k: True
    sys.modules['igl'] = igl
    if str(NTRL) not in sys.path:
        sys.path.insert(0, str(NTRL))
    from models.metric import model_train_metric as mt
    from models.metric import model_function_metric as mf
    from models.metric import model_network_metric as mn
    from models import data_mlp as db
    for m in (mt, mf, mn, db):
        assert Path(m.__file__).resolve().is_relative_to(NTRL / 'models'), 'ntrl-demo models package shadowed'
    return mt, mf, mn, db


def ntrl_loss(function, points, Yobs, normal, beta, td_len=0.03, td_free=False, scene=None, clear=None):
    """ntrl-demo's metric loss, line for line as jeffrey/lbs/trainer.py ntrl_loss (dimension from function.dim)."""
    net, dim = function.network, function.dim
    tau, w, Xp = net.out(points)
    dtau = function.gradient(tau, Xp)
    DT0 = dtau[:, :dim]
    DT1 = dtau[:, dim:]
    S0 = torch.einsum('ij,ij->i', DT0, DT0)
    S1 = torch.einsum('ij,ij->i', DT1, DT1)
    td_weight = 1e-3
    with torch.no_grad():
        length0 = td_len / (Yobs[:, 0]).unsqueeze(1)
        Dir0 = length0 * (DT0 * Yobs[:, 0].unsqueeze(1) ** 2).clone().detach()
        Xp_new0 = Xp.clone().detach()
        Xp_new0[:, :dim] = Xp_new0[:, :dim] - Dir0
        if td_free:
            _, free0 = C.step_time(scene, Xp.detach()[:, :dim], Xp_new0[:, :dim], 6, clear, exact=False)
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
            _, free1 = C.step_time(scene, Xp.detach()[:, dim:], Xp_new0[:, dim:], 6, clear, exact=False)
        tau_new0, w, Xp_new0 = net.out(Xp_new0)
        tau_new1 = length1
        del Xp_new0, Dir1
    tau_loss1 = td_weight * ((tau - (tau_new0 + tau_new1)) ** 2).squeeze()
    where_d0 = (tau[:, 0] < length0.squeeze())
    where_d1 = (tau[:, 0] < length1.squeeze())
    tau_loss0[where_d0] = 0
    tau_loss1[where_d1] = 0
    if td_free:
        # ntrl-demo's TD step (0.03 box lengths down the gradient, no collision check) is dropped where it crosses an obstacle
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
    loss_n = (torch.sum((diff + n_loss + tau_loss) * torch.exp(-0.5 * T))) / Yobs.shape[0]
    loss = beta * loss_n
    return loss, loss_n, T, DT0.detach(), DT1.detach()


class Schedule:
    def __init__(self, start, weight, ramp):
        self.start, self.weight, self.ramp = start, weight, max(ramp, 1)

    def __call__(self, epoch):
        if self.weight <= 0 or epoch <= self.start:
            return 0.0
        return self.weight * min(1.0, (epoch - self.start) / self.ramp)


class Counted:
    def __init__(self, fn):
        self.fn, self.n = fn, 0

    def __call__(self, q):
        self.n += len(q)
        return self.fn(q)


def cup_pool(scene, clear, count, gen, min_clear, batch=1 << 20):
    """count free configurations (uniform joints, rejection) whose grasp point is inside the cupboard box, clearance >= contact +
    min_clear cells. The frozen suites have half of their starts and goals there; uniform free configurations about 1 %."""
    out, have = [], 0
    cup = scene.cupboard
    while have < count:
        x = scene.uniform(batch, gen)
        keep = []
        for i in range(0, len(x), 1 << 18):
            e = scene.ee(x[i:i + (1 << 18)])
            keep.append(((e >= cup[:, 0]) & (e <= cup[:, 1])).all(1))
        x = x[torch.cat(keep)]
        x = x[scene.in_box(x)]
        x = x[clear(x) >= scene.contact + min_clear]
        out.append(x)
        have += len(x)
    return torch.cat(out)[:count]


def data_pairs(scene, clear, data_dir, min_clear, pair_max, bins=10):
    """Traversal pairs taken from the training pairs (cfg pair_data): those with both ends free as scene.free asks (grasp points in
    the sampling box, clearance >= contact + min_clear cells) and at most pair_max apart, grouped into `bins` equal-width length bins.
    For scenes where two random free configurations are never close (two arms: median distance 1.36), so that draw_pairs returns only
    pairs near pair_max, longer than almost every test query."""
    X = torch.tensor(np.load(Path(data_dir) / 'sampled_points.npy'), device=scene.device)
    D = X.shape[1] // 2
    x0, x1 = X[:, :D], X[:, D:]
    g = torch.linalg.vector_norm(x0 - x1, dim=1)
    keep = scene.in_box(x0) & scene.in_box(x1) & (g <= (pair_max or float('inf')))
    idx = torch.nonzero(keep).squeeze(1)
    idx = idx[(clear(x0[idx]) >= scene.contact + min_clear) & (clear(x1[idx]) >= scene.contact + min_clear)]
    top = float(g[idx].max()) * (1 + 1e-6)
    b = (g[idx] / top * bins).long().clamp(max=bins - 1)
    groups = [idx[b == k] for k in range(bins)]
    return dict(x0=x0, x1=x1, groups=[q for q in groups if len(q)], top=top, sizes=[len(q) for q in groups])


def draw_data_pairs(dp, count, gen):
    """count pairs: a length bin uniformly, a pair of the bin uniformly, either direction."""
    dev = dp['x0'].device
    which = torch.randint(len(dp['groups']), (count,), device=dev, generator=gen)
    pick = torch.zeros(count, dtype=torch.long, device=dev)
    for k, q in enumerate(dp['groups']):
        m = which == k
        pick[m] = q[torch.randint(len(q), (int(m.sum()),), device=dev, generator=gen)]
    flip = torch.rand(count, device=dev, generator=gen) < 0.5
    a, b = dp['x0'][pick], dp['x1'][pick]
    return torch.where(flip[:, None], b, a), torch.where(flip[:, None], a, b)


def draw_pairs(scene, clear, count, gen, min_clear, pair_max, shifts=32, cup=0.0, pool=None):
    """count pairs (a, b) of free configurations (scene.free: grasp point in the sampling box, clearance >= contact + min_clear cells),
    at most pair_max apart in the normalized joint space (0 = any): every a takes the first of `shifts` candidate partners in range.
    cup > 0: every endpoint is replaced, with that probability, by a configuration of `pool` (cup_pool: grasp point in the cupboard)."""
    def free(n):
        x = scene.free(n, gen, min_clear, clear=clear)
        if cup > 0:
            m = torch.rand(n, device=x.device, generator=gen) < cup
            x = torch.where(m[:, None], pool[torch.randint(len(pool), (n,), device=x.device, generator=gen)], x)
        return x

    if not pair_max:
        return free(count), free(count)
    A, B, have = [], [], 0
    while have < count:
        a = free(2 * count)
        b = free(2 * count)
        done = torch.zeros(len(a), dtype=torch.bool, device=a.device)
        partner = torch.zeros_like(a)
        for sft in range(shifts):
            bb = torch.roll(b, sft, 0)
            ok = ~done & (torch.linalg.vector_norm(a - bb, dim=1) <= pair_max)
            partner[ok] = bb[ok]
            done |= ok
        A.append(a[done])
        B.append(partner[done])
        have += int(done.sum())
    return torch.cat(A)[:count], torch.cat(B)[:count]


class Bank:
    """Inner nodes of the current midpoint trees (and sub-paths): endpoints, their input rows, targets, depth."""

    def __init__(self):
        self.size = 0

    def build(self, ev, scene, a, b, gen, cfg):
        t0 = time.time()
        st = cfg['planner']
        q0 = ev.n_clear
        tab = ev.travel(torch.cat([a, b], 1))
        budget = torch.clamp(torch.ceil(cfg['budget_factor'] * tab / st['step']), cfg['min_steps'], cfg['max_steps'])
        paths, stop, conv, stalled = C.mppi(ev, a, b, gen, budget, scene.lo, scene.hi, stall_window=cfg['stall_window'],
                                            stall_drop=cfg['stall_drop_cells'] * scene.pitch, safe=cfg['traverse'] == 'safe',
                                            safe_margin=cfg['safe_margin'], **st)
        t1 = time.time()
        d, first = C.collisions(scene, paths, stop, ev.clearance)
        collided = first <= stop
        end = torch.where(collided, (first - 1).clamp(min=0), stop)
        if cfg['trim_hover'] > 0:
            far = torch.linalg.vector_norm(paths - paths[torch.arange(len(paths)), end][:, None], dim=2) > cfg['trim_hover'] * scene.pitch
            idx = torch.arange(paths.shape[1], device=paths.device)[None]
            far &= idx <= end[:, None]
            last_far = torch.where(far.any(1), paths.shape[1] - 1 - torch.argmax(far.flip(1).int(), 1), torch.zeros_like(end))
            end = torch.where(stalled & ~collided, torch.minimum(end, last_far + 1), end)
        cum = TR.path_time(scene, paths, d)
        tau = cfg['leaf_cells'] * scene.pitch
        leaves, inner = TR.split_tree(cum, end, tau)
        reached = conv & ~collided
        n_tree = len(inner)
        if cfg['nodes'] in ('sub', 'both') and len(leaves):
            sub = TR.sub_intervals(leaves, int(cfg['sub_factor'] * max(len(inner), 1)), gen)
            inner = sub if cfg['nodes'] == 'sub' else torch.cat([inner, sub])
        if cfg['leaf'] == 'step':     # the true time of the straight step between the leaf's ends where that step is free (never the network)
            pid, ls, le = leaves[:, 0], leaves[:, 1], leaves[:, 2]
            own = cum[pid, le] - cum[pid, ls]
            chord, free = C.step_time(scene, paths[pid, ls], paths[pid, le], 17, ev.clear)
            lv = torch.where(free, torch.minimum(chord.to(own.dtype), own), own)
            self.leaf_step = dict(free=float(free.float().mean()), chord_over_path=float((chord[free] / own[free].clamp(min=1e-9)).median()) if bool(free.any()) else None)
        else:
            lv = TR.leaf_values(ev, paths, cum, leaves, cfg['leaf'])
        tg = TR.node_targets(paths, cum, leaves, inner, lv)
        self.wa = paths[inner[:, 0], inner[:, 1]].contiguous()
        self.wb = paths[inner[:, 0], inner[:, 2]].contiguous()
        self.ra, self.rb = ev.rows(self.wa), ev.rows(self.wb)
        self.target = tg
        self.depth = inner[:, 3]
        self.size = len(inner)
        with torch.no_grad():
            tn = ev.travel(torch.cat([self.wa, self.wb], 1)) if self.size else tg
        r = torch.log(tn / tg) if self.size else tg
        L_path = torch.linalg.vector_norm(paths[:, 1:] - paths[:, :-1], dim=2).sum(1)
        L_line = torch.linalg.vector_norm(paths[torch.arange(len(paths)), end] - a, dim=1)
        ratio = L_path[reached] / L_line[reached].clamp(min=1e-4)
        stats = dict(traversals=len(a), reached=int(reached.sum()), collided=int(collided.sum()), stalled=int(stalled.sum()),
                     converged=int(conv.sum()), steps_mean=float(stop.float().mean()), steps_max=int(stop.max()), leaves=len(leaves), nodes=self.size,
                     tree_nodes=n_tree, depth_max=int(self.depth.max()) if self.size else 0,
                     log_net_over_target_median=float(r.median()) if self.size else None,
                     log_net_over_target_by_depth={int(k): float(r[self.depth == k].median()) for k in range(-1, min(8, int(self.depth.max()) + 1)) if bool((self.depth == k).any())} if self.size else {},
                     reached_path_over_line_median=float(ratio.median()) if bool(reached.any()) else None,
                     reached_long_detour_share=float((ratio > 1.3).float().mean()) if bool(reached.any()) else None,
                     pair_distance_median=float(torch.linalg.vector_norm(a - b, dim=1).median()),
                     traverse_seconds=t1 - t0, seconds=time.time() - t0, clearance_queries=ev.n_clear - q0)
        if cfg['leaf'] == 'step':
            stats['leaf_step'] = self.leaf_step
        del d, paths, cum
        return stats

    def sample(self, m, gen):
        i = torch.randint(self.size, (m,), generator=gen, device=self.target.device)
        return self.ra[i], self.rb[i], self.target[i]


def rho(r, kind):
    if kind == 'l2':
        return r * r
    if kind == 'huber':
        a = r.abs()
        return torch.where(a < 0.1, 0.5 * r * r / 0.1, a - 0.05)
    return r.abs()


def train(scene, data_dir, out, cfg, log, validate=None):
    """Train from epoch 1 to cfg['epochs']; returns (final checkpoint, history). cfg: seed, lift, lift_gain, lift_depth, freq, epochs,
    start, weight, ramp, refresh, traversals, batch, leaf, leaf_cells, loss, nodes, sub_factor, min_clear, pair_max, budget_factor,
    min_steps, max_steps, stall_window, stall_drop_cells, trim_hover, traverse, safe_margin, chunk, compile, max_rollbacks, save_at,
    validate_at, planner, bell, bell_r, bell_k, bell_cone, bell_long, bell_n, td_free."""
    mt, mf, mn, db = ntrl_modules()
    dev = 'cuda:0'
    D = len(scene.lo)
    out = Path(out)
    (out / 'checkpoints').mkdir(parents=True, exist_ok=True)
    seed = cfg['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    B = torch.normal(0, 1, size=(128, D))
    torch.nn.init.trunc_normal_(B, mean=0.0, std=1, a=-2.0, b=2.0)
    B = B * torch.tensor(cfg['freq'], dtype=B.dtype)[None, :]
    ncfg = dict(lift=dict(kind=cfg['lift'], gain=cfg['lift_gain'], depth=cfg['lift_depth'], seed=seed), freq=list(cfg['freq']))
    network = C.build_network(mn, dev, D, B, ncfg, scene)
    function = mf.Function(str(out), dev, network, D)
    optimizer = torch.optim.AdamW([{'params': list(network.parameters())}], lr=1e-3, weight_decay=0.5)
    dataset = db.Database(str(data_dir) + '/')
    loader = mt.FastTensorDataLoader(dataset.data, batch_size=2000, shuffle=True)
    beta = 1.0
    current_diff = 1000.0
    sched = Schedule(cfg['start'], cfg['weight'], cfg['ramp'])
    gen = torch.Generator(dev).manual_seed(seed + 101)
    pool, dpairs = None, None
    W_train = C._rows_weight(network)
    use_bell = cfg.get('bell', 0) > 0
    bell_gen = torch.Generator(dev).manual_seed(seed + 99)
    clear = C.Clearance(scene, compile=cfg['compile'])
    if use_bell:
        log(f'Bellman backup: weight {cfg["bell"]:.4g} (argument {cfg.get("bell_arg")}), radius {cfg["bell_r"]:.5f} = {cfg["bell_r"] * scene.E:.3f} cells, K {cfg["bell_k"]}, cone {cfg["bell_cone"]}, '
            f'longest step x{cfg["bell_long"]}, on {cfg["bell_n"] or 2000} pairs per minibatch, TD only on free steps {cfg.get("td_free", False)}')
    bank, ev = Bank(), None
    history = dict(epochs=[], ntrl_loss=[], mid_loss=[], bell_loss=[], bell_used=[], weight=[], rollbacks=[], refresh=[], validate=[], all_loss=[])
    t_validate = 0.0
    current_state = copy.deepcopy(network.state_dict())
    current_opt = copy.deepcopy(optimizer.state_dict())
    prev_states, prev_opts = [], []
    train_loss = []
    t_start = time.time()
    use_mid = cfg['weight'] > 0
    save_at = set(int(e) for e in cfg['save_at']) | {cfg['epochs']}
    if network.lift_kind != 'none':
        log(f'lift: gain {network.lift_gain}, depth {network.profile.depth}, G_max {network.profile.G_max:.3f} cells, h range {network.lift_gain * network.profile.G_max / scene.E:.4f} (joints: 1)')
    log(f'network: lift {network.lift_kind}, Fourier factors {cfg["freq"]}, parameters {sum(p.numel() for p in network.parameters())}; midpoint from epoch {cfg["start"]} '
        f'weight {cfg["weight"]} ramp {cfg["ramp"]}, refresh {cfg["refresh"]} x {cfg["traversals"]} traversals ({cfg["traverse"]}, pairs <= {cfg["pair_max"]}), leaves {cfg["leaf"]} {cfg["leaf_cells"]} cells, nodes {cfg["nodes"]}')
    for epoch in range(1, cfg['epochs'] + 1):
        if use_mid and epoch > cfg['start'] and ((epoch - cfg['start'] - 1) % cfg['refresh'] == 0):
            if ev is None:
                ev = C.Compiled(network, scene, clear, chunk=cfg['chunk'], compile=cfg['compile'])
            else:
                ev.update(network)
            n0 = clear.n
            if cfg.get('pair_cup', 0) > 0 and pool is None:
                t_pool = time.time()
                pool = cup_pool(scene, clear, cfg['pool'], gen, cfg['min_clear'])
                log(f'in-cupboard pool: {len(pool)} configurations in {time.time() - t_pool:.0f}s; clearance cells q10/50/90 '
                    f'{[round(float(v), 3) for v in torch.quantile(clear(pool[:50000]), torch.tensor([.1, .5, .9], device=dev))]}')
            if cfg.get('pair_data') and dpairs is None:
                dpairs = data_pairs(scene, clear, data_dir, cfg['min_clear'], cfg['pair_max'])
                log(f'traversal pairs from the training pairs: {sum(dpairs["sizes"])} usable, by length bin (width {dpairs["top"] / len(dpairs["sizes"]):.3f}) {dpairs["sizes"]}')
            if cfg.get('pair_data'):
                a, b = draw_data_pairs(dpairs, cfg['traversals'], gen)
            else:
                a, b = draw_pairs(scene, clear, cfg['traversals'], gen, cfg['min_clear'], cfg['pair_max'], cup=cfg.get('pair_cup', 0), pool=pool)
            n_ends = clear.n - n0
            for attempt in range(4):
                try:
                    stats = bank.build(ev, scene, a, b, gen, cfg)
                    break
                except torch.OutOfMemoryError:
                    bank.__dict__.clear()
                    bank.size = 0
                    torch.cuda.empty_cache()
                    ev.set_chunk(max(1 << 14, ev.chunk // 2))
                    log(f'epoch {epoch}: CUDA out of memory in the refresh; retrying with chunk {ev.chunk}')
            else:
                raise RuntimeError('refresh failed 4 times (CUDA memory)')
            stats['epoch'] = epoch
            stats['clearance_queries_ends'] = n_ends
            history['refresh'].append(stats)
            log(f'epoch {epoch}: refresh {stats["traversals"]} traversals in {stats["seconds"]:.1f}s (MPPI {stats["traverse_seconds"]:.1f}s, steps mean {stats["steps_mean"]:.0f} max {stats["steps_max"]}): '
                f'reached {stats["reached"]}, collided {stats["collided"]}, stalled {stats["stalled"]}; nodes {stats["nodes"]} (tree {stats["tree_nodes"]}, depth <= {stats["depth_max"]}), '
                f'median log(T/target) {stats["log_net_over_target_median"]}, by depth {json.dumps({k: round(v, 3) for k, v in stats["log_net_over_target_by_depth"].items()})}, '
                f'reached path/line {stats["reached_path_over_line_median"]}, detour share {stats["reached_long_detour_share"]}')
            torch.cuda.empty_cache()
        lam = sched(epoch)
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
        while True:
            total_train_loss, total_diff, total_mid, total_bell, total_used = 0.0, 0.0, 0.0, 0.0, 0.0
            ii = 0
            for i, whole in enumerate(loader, 0):
                if ii > 4:
                    break
                ii += 1
                batch = whole[0].to(dev)
                points = batch[:, :2 * D]
                speed = batch[:, 2 * D:2 * D + 2]
                normal = batch[:, 2 * D + 2:4 * D + 2]
                speed = speed * speed * (2 - speed) * (2 - speed)
                speed = 1.025 * speed + 1 - 1.025
                loss_value, loss_n, T, g0, g1 = ntrl_loss(function, points, speed, normal, beta, td_free=bool(cfg.get('td_free')) and use_bell, scene=scene, clear=clear)
                if use_bell:
                    bt, bused, _ = C.bellman_term(network, scene, points, T, (g0, g1), cfg['bell_r'], cfg['bell_k'], bell_gen, clear, cone=cfg['bell_cone'], long=cfg['bell_long'], subset=cfg['bell_n'], goal=cfg.get('bell_goal', False))
                    if not isinstance(bt, float):
                        loss_value = loss_value + beta * cfg['bell'] * bt
                        total_bell += float(bt)
                    total_used += bused
                if lam > 0 and bank.size:
                    ra, rb, y = bank.sample(cfg['batch'], gen)
                    t = C.travel_train(network, W_train, ra, rb)
                    lmid = rho(torch.log(t.clamp(min=1e-6)) - torch.log(y.clamp(min=1e-6)), cfg['loss']).mean()
                    loss_value = loss_value + lam * lmid
                    total_mid += float(lmid)
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
            if epoch > cfg['start'] and rollbacks >= cfg['max_rollbacks']:
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
            history['epochs'].append(epoch)
            history['ntrl_loss'].append(total_diff)
            history['mid_loss'].append(total_mid / 5)
            history['bell_loss'].append(total_bell / 5)
            history['bell_used'].append(total_used / 5)
            history['weight'].append(lam)
            history['rollbacks'].append(rollbacks)
        if epoch % 100 == 0:
            el = time.time() - t_start
            log(f'epoch {epoch}: ntrl loss {total_diff:.4e}, Bellman sq {total_bell / 5:.3e} (pair ends used {total_used / 5:.2f}), mid loss {total_mid / 5:.4f}, lambda {lam:.3g}, rollbacks {rollbacks}, {el / 60:.1f} min ({el / epoch:.3f} s/epoch)')
        if epoch in save_at:
            torch.save({'epoch': epoch, 'model_state_dict': network.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'B_state_dict': B,
                        'val_loss': total_diff, 'embed_config': network.cfg}, out / 'checkpoints' / f'epoch_{epoch:05d}.pt')
        if validate is not None and epoch in set(cfg['validate_at']):
            tv = time.time()
            if ev is None:
                ev = C.Compiled(network, scene, clear, chunk=cfg['chunk'], compile=cfg['compile'])
            else:
                ev.update(network)
            v = validate(ev, epoch)
            t_validate += time.time() - tv
            v['epoch'] = epoch
            v['train_seconds'] = time.time() - t_start - t_validate
            history['validate'].append(v)
        if epoch % 100 == 0:
            with open(out / 'history.json', 'w') as f:
                json.dump(history, f)
    final = out / 'checkpoint.pt'
    torch.save({'epoch': cfg['epochs'], 'model_state_dict': network.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'B_state_dict': B,
                'train_loss': train_loss, 'val_loss': total_diff, 'embed_config': network.cfg}, final)
    history['final_loss'] = total_diff
    history['seconds'] = time.time() - t_start
    history['validate_seconds'] = t_validate
    with open(out / 'history.json', 'w') as f:
        json.dump(history, f)
    return final, history
