"""LBS trainer: ntrl-demo's training loop with any of the travel-time Lift, the free-step Bellman backup (true step cost) and the
travel-time midpoint (Segment) loss. jeffrey/lift_midpoint/trainer.py with the Bellman term of jeffrey/pair_lift added (bell.py):

    loss = beta * (loss_ntrl + bell * mean (T - y)^2) + lambda(epoch) * mean | log T - log y_mid |

with cfg['bell'] > 0 (from epoch 1; ntrl-demo's own TD term then skips steps that cross an obstacle, cfg['td_free']). Everything below
is lift_midpoint's description of the loop and of the midpoint term.

ntrl-demo's training loop with the travel-time lift and, from a start epoch on, the travel-time midpoint loss. No push.

Epochs 1 .. start are jeffrey/lift_scale/trainer.py's loop with no push, call for call: the same global RNG draws in the same order
(seed -> B -> network -> init_weights -> per-epoch reshuffle -> rollback draws), 5 minibatches of 2,000 pairs per epoch, ntrl-demo's
loss (eikonal + TD + normal terms, weight exp(-0.5 T), beta = 1 / previous loss), AdamW lr 5e-4 weight decay 0.5, rollback to one of the
last 5 states when ntrl-demo's own loss rises 20 %. So up to the start epoch a run is the registered lift-alone field of the same
seed (--lift time) or ntrl-demo itself (--lift none).

After the start epoch every update adds jeffrey/midpoint_dp's term (its MAIN setting by default):

    loss = beta * loss_ntrl + lambda(epoch) * mean over 8,192 sampled nodes of | log T(w_s, w_e) - log y(s, e) |

Every --refresh epochs, --traversals random pairs (a, b) of configurations with exact clearance >= 0.1 cells are driven by ntrl-demo's
MPPI on the current field, all at once (liftfast.mppi: the lifted network, h from the exact clearance of every rollout end; 'safe':
rollouts whose first state comes within 0.02 cells of a wall get +100). Each drive is cut where it first collides or stalls (its end is
then the pair's goal), parametrized by travel time along it (exact speed), split recursively at travel-time midpoints down to one cell
of travel time (midpoint_dp/tree.py); a leaf is worth min(T(leaf ends), the travel time the drive took over it); every inner node of the
recursion and an equal number of random sub-paths between leaf boundaries get the sum of their leaves as target y. lambda ramps
linearly from 0 at the start epoch to --weight over --ramp epochs. Rollbacks are capped at --max-rollbacks per epoch after the start
(midpoint_dp's rule); before it, as lift_scale (raise after 2,000).
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
import lift as LF  # noqa: E402  (verbatim copy of jeffrey/lift_scale/lift.py)
import liftfast as LFT  # noqa: E402
import traverse as TV  # noqa: E402  (verbatim copy of jeffrey/midpoint_dp/traverse.py)
import tree as TR  # noqa: E402  (verbatim copy of jeffrey/midpoint_dp/tree.py)
import bell as BL  # noqa: E402  (the Bellman backup of jeffrey/pair_lift)


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
    """ntrl-demo's metric loss, line for line as jeffrey/lift_scale/trainer.py.ntrl_loss (checked there against the package's
    Function.Loss, gradients included)."""
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
            _, free0 = BL.step_time(scene, Xp.detach()[:, :dim], Xp_new0[:, :dim], 6, exact=False, clear=clear)
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
            _, free1 = BL.step_time(scene, Xp.detach()[:, dim:], Xp_new0[:, dim:], 6, exact=False, clear=clear)
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


def free_counted(scene, clear, count, gen, min_clear, batch=262144):
    """scene.free with the clearance queries counted (same draws as scene.free)."""
    out, have = [], 0
    while have < count:
        q = scene.uniform(batch, gen)
        q = q[clear(q) >= min_clear]
        out.append(q)
        have += len(q)
    return torch.cat(out)[:count]


class Bank:
    """Inner nodes of the current midpoint trees (and sub-paths): endpoints, their input rows, targets, depth."""

    def __init__(self):
        self.size = 0

    def build(self, ev, scene, a, b, gen, cfg, drive=None):
        """drive: the evaluator whose field the traversals follow (default ev, the network being trained; a frozen driver field for
        the driver control). Leaf values, targets and input rows always come from ev."""
        t0 = time.time()
        st = cfg['planner']
        drive = ev if drive is None else drive
        q0 = ev.n_clear + (drive.n_clear if drive is not ev else 0)
        tab = drive.travel(torch.cat([a, b], 1))
        budget = torch.clamp(torch.ceil(cfg['budget_factor'] * tab / st['step']), st['min_steps'], cfg['max_steps'])
        paths, stop, conv, stalled = LFT.mppi(drive, a, b, gen, budget, scene.lo, scene.hi, stall_window=cfg['stall_window'],
                                              stall_drop=cfg['stall_drop_cells'] * scene.pitch, safe=cfg['traverse'] == 'safe',
                                              safe_margin=cfg['safe_margin'], safe_mode=cfg['safe_mode'], **st)
        t1 = time.time()
        d, first = TV.collisions(scene, paths, stop, clear=ev.clearance)
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
        # how much of each drive's straight-line distance the drive needed (detour share of the traversals)
        L_path = torch.linalg.vector_norm(paths[:, 1:, :2] - paths[:, :-1, :2], dim=2).sum(1) / scene.pitch
        L_line = torch.linalg.vector_norm(paths[torch.arange(len(paths)), end, :2] - a[:, :2], dim=1) / scene.pitch
        stats = dict(traversals=len(a), reached=int(reached.sum()), collided=int(collided.sum()), stalled=int(stalled.sum()),
                     converged=int(conv.sum()), steps_mean=float(stop.float().mean()), steps_max=int(stop.max()), leaves=len(leaves), nodes=self.size,
                     tree_nodes=n_tree, depth_max=int(self.depth.max()) if self.size else 0,
                     log_net_over_target_median=float(r.median()) if self.size else None,
                     log_net_over_target_by_depth={int(k): float(r[self.depth == k].median()) for k in range(-1, min(8, int(self.depth.max()) + 1)) if bool((self.depth == k).any())} if self.size else {},
                     reached_path_over_line_median=float((L_path[reached] / L_line[reached].clamp(min=1e-3)).median()) if bool(reached.any()) else None,
                     reached_long_detour_share=float(((L_path[reached] / L_line[reached].clamp(min=1e-3)) > 1.3).float().mean()) if bool(reached.any()) else None,
                     traverse_seconds=t1 - t0, seconds=time.time() - t0, clearance_queries=ev.n_clear + (drive.n_clear if drive is not ev else 0) - q0)
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
    """Train from epoch 1 to cfg['epochs']; returns (final checkpoint, history). cfg: seed, lift, epochs, start, weight, ramp, refresh,
    traversals, batch, leaf, leaf_cells, loss, nodes, sub_factor, min_clear, budget_factor, max_steps, stall_window, stall_drop_cells,
    trim_hover, traverse, safe_margin, safe_mode, chunk, compile, max_rollbacks, save_at, validate_at, planner."""
    mt, mf, mn, db = ntrl_modules()
    dev = 'cuda:0'
    out = Path(out)
    (out / 'checkpoints').mkdir(parents=True, exist_ok=True)
    seed = cfg['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    B = torch.normal(0, 1, size=(128, 3))
    torch.nn.init.trunc_normal_(B, mean=0.0, std=1, a=-2.0, b=2.0)
    ncfg = dict(octaves=[], rows=16, base_scale=1.0, whiten=True, gate=dict(kind='none', mu_min=0.02, const=1.0),
                lift=dict(kind=cfg['lift'], gain=1.0, depth=LF.OFFSET_CELLS, freq=1.0), zero_cols=[])
    ncfg['lift'] = dict(ncfg['lift'], seed=seed)
    network = LF.build_network(mn, dev, B, ncfg, scene)
    function = mf.Function(str(out), dev, network, 3)
    optimizer = torch.optim.AdamW([{'params': list(network.parameters())}], lr=1e-3, weight_decay=0.5)
    dataset = db.Database(str(data_dir) + '/')
    loader = mt.FastTensorDataLoader(dataset.data, batch_size=2000, shuffle=True)
    beta = 1.0
    current_diff = 1000.0
    sched = Schedule(cfg['start'], cfg['weight'], cfg['ramp'])
    gen = torch.Generator(dev).manual_seed(seed + 101)
    W_train = LFT.train_weight(network)
    use_bell = cfg.get('bell', 0) > 0
    bell_gen = torch.Generator(dev).manual_seed(seed + 99)
    bell_clear = BL.ClearTable(scene, 32, 49) if use_bell else None
    if use_bell:
        log(f'Bellman backup: weight {cfg["bell"]:.4g} (argument {cfg.get("bell_arg")}), radius {cfg["bell_r"]:.5f} box = {cfg["bell_r"] * scene.E:.3f} cells, K {cfg["bell_k"]}, cone {cfg["bell_cone"]}, '
            f'longest step x{cfg["bell_long"]}, TD only on free steps {cfg.get("td_free", False)}, clearance table {bell_clear.per_cell} nodes per cell')
    bank, ev, drive = Bank(), None, None
    if cfg.get('driver'):
        # control: the traversals follow a frozen field (a registered run's final network) instead of the network being trained
        dp = torch.load(cfg['driver'], map_location=dev, weights_only=False)
        if 'embed_config' in dp:
            dnet = LF.load_field(cfg['driver'], scene, dev)
        else:
            dnet = mn.NN(dev, 3, dp['B_state_dict'])
            dnet.load_state_dict(dp['model_state_dict'], strict=True)
            dnet.to(dev).float().eval()
            for q_ in dnet.parameters():
                q_.requires_grad_(False)
        drive = LFT.Compiled(dnet, scene, chunk=cfg['chunk'], compile=cfg['compile'])
        log(f'traversals follow the frozen field {cfg["driver"]} (lift {getattr(dnet, "lift_kind", "none")})')
    history = dict(epochs=[], ntrl_loss=[], mid_loss=[], bell_loss=[], bell_used=[], weight=[], rollbacks=[], refresh=[], validate=[], all_loss=[])
    t_validate = 0.0
    current_state = copy.deepcopy(network.state_dict())
    current_opt = copy.deepcopy(optimizer.state_dict())
    prev_states, prev_opts = [], []
    train_loss = []
    t_start = time.time()
    use_mid = cfg['weight'] > 0
    save_at = set(int(e) for e in cfg['save_at']) | {cfg['epochs']}
    log(f'network: lift {network.lift_kind} gain {network.lift_gain}, parameters {sum(p.numel() for p in network.parameters())}; midpoint from epoch {cfg["start"]} '
        f'weight {cfg["weight"]} ramp {cfg["ramp"]}, refresh {cfg["refresh"]} x {cfg["traversals"]} traversals ({cfg["traverse"]}), leaves {cfg["leaf"]} {cfg["leaf_cells"]} cells, nodes {cfg["nodes"]}')
    for epoch in range(1, cfg['epochs'] + 1):
        if use_mid and epoch > cfg['start'] and ((epoch - cfg['start'] - 1) % cfg['refresh'] == 0):
            if ev is None:
                ev = LFT.Compiled(network, scene, chunk=cfg['chunk'], compile=cfg['compile'])
            else:
                ev.update(network)
            fc = Counted(scene.clearance)
            a = free_counted(scene, fc, cfg['traversals'], gen, cfg['min_clear'])
            b = free_counted(scene, fc, cfg['traversals'], gen, cfg['min_clear'])
            for attempt in range(4):
                try:
                    stats = bank.build(ev, scene, a, b, gen, cfg, drive=drive)
                    break
                except torch.OutOfMemoryError:
                    bank.__dict__.clear()
                    bank.size = 0
                    torch.cuda.empty_cache()
                    ev.set_chunk(max(1 << 14, ev.chunk // 2))
                    if drive is not None:
                        drive.set_chunk(ev.chunk)
                    log(f'epoch {epoch}: CUDA out of memory in the refresh; retrying with chunk {ev.chunk}')
            else:
                raise RuntimeError('refresh failed 4 times (CUDA memory)')
            stats['epoch'] = epoch
            stats['clearance_queries_ends'] = fc.n
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
                points = batch[:, :6]
                speed = batch[:, 6:8]
                normal = batch[:, 8:14]
                speed = speed * speed * (2 - speed) * (2 - speed)
                speed = 1.025 * speed + 1 - 1.025
                loss_value, loss_n, T, g0, g1 = ntrl_loss(function, points, speed, normal, beta, td_free=bool(cfg.get('td_free')) and use_bell, scene=scene, clear=bell_clear)
                if use_bell:
                    bt, bused, _ = BL.bellman_term(network, scene, points, T, (g0, g1), cfg['bell_r'], cfg['bell_k'], bell_gen, cone=cfg['bell_cone'], clear=bell_clear, long=cfg['bell_long'])
                    if not isinstance(bt, float):
                        loss_value = loss_value + beta * cfg['bell'] * bt
                        total_bell += float(bt)
                    total_used += bused
                if lam > 0 and bank.size:
                    ra, rb, y = bank.sample(cfg['batch'], gen)
                    t = LFT.travel_train(network, W_train, ra, rb)
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
            v = validate(network, epoch)
            t_validate += time.time() - tv
            v['epoch'] = epoch
            v['train_seconds'] = time.time() - t_start - t_validate          # training time so far, without the planning of the validations
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
