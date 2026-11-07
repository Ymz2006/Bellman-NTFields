"""ntrl-demo training with the travel-time midpoint loss added after a start epoch.

The loop is ntrl-demo's models/metric Model.train() (baselines/ntrl-demo/models/metric/model_train_metric.py), resumed
from one of its own checkpoints (network, AdamW state and B; the baseline runs keep epochs 1000/2000/3000/4000/5000):
5 minibatches of 2,000 pairs per epoch from its FastTensorDataLoader, speed stretch alpha = 1.025, the package's own
Function.Loss (eikonal + TD + normal terms, weight exp(-0.5 T), normalized by beta = 1 / previous epoch's loss),
AdamW lr 5e-4 weight decay 0.5, epoch rollback when the loss rises by 20 % (random one of the last 5 states). The
rollback test and beta use the package's loss only. The only change is one more term in every update:

    loss = beta * loss_ntrl + lambda(epoch) * mean_nodes rho(log T(w_s, w_e) - log y(s, e))

with (w_s, w_e, y) inner nodes of travel-time midpoint trees (tree.py) over MPPI traversals (traverse.py) of random
collision-free (a, b) pairs with the current network: every --refresh epochs, --traversals pairs are traversed, each
path is cut where it first collides (exact clearance) or stalls (the reached end becomes the pair's goal), split at
travel-time midpoints down to --leaf-cells, and every inner node gets the sum of its leaf values as target. Targets
can be re-evaluated with the current network every --retarget epochs (same paths and leaves). lambda ramps linearly
from 0 at --start to --weight over --ramp epochs. --weight 0 is the control (the same resumed loop without the term).
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
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
import fastnet as F  # noqa: E402
import traverse as TV  # noqa: E402
import tree as TR  # noqa: E402


def ntrl_modules():
    """ntrl-demo's models/metric modules (unchanged on disk), with the pickle5 / igl shims of jeffrey/maze_scale/train.py."""
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


class Schedule:
    def __init__(self, start, weight, ramp):
        self.start, self.weight, self.ramp = start, weight, max(ramp, 1)

    def __call__(self, epoch):
        if self.weight <= 0 or epoch <= self.start:
            return 0.0
        return self.weight * min(1.0, (epoch - self.start) / self.ramp)


class Bank:
    """Inner nodes of the current midpoint trees: endpoints, targets, depth; plus the leaves to re-evaluate targets."""

    def __init__(self):
        self.size = 0

    def build(self, ev, maze, a, b, gen, cfg, log):
        t0 = time.time()
        st = cfg['planner']
        tab = ev.travel(torch.cat([a, b], 1))
        budget = torch.clamp(torch.ceil(cfg['budget_factor'] * tab / st['step']), st['min_steps'], cfg['max_steps'])
        if getattr(self, 'clear', None) is None:
            self.clear = TV.Clearance(maze, compile=cfg['compile'])
        safe = dict(clearance=self.clear, safe_margin=cfg['safe_margin'], safe_mode=cfg['safe_mode']) if cfg['traverse'] == 'safe' else {}
        paths, stop, conv, stalled = TV.mppi(ev, a, b, gen, budget, maze.lo, maze.hi, stall_window=cfg['stall_window'],
                                             stall_drop=cfg['stall_drop_cells'] * maze.pitch, **safe, **st)
        t1 = time.time()
        d, first = TV.collisions(maze, paths, stop, clear=self.clear)
        collided = first <= stop
        end = torch.where(collided, (first - 1).clamp(min=0), stop)
        if cfg['trim_hover'] > 0:
            # drop the hover at the end of a stalled path: cut at the first waypoint after which it stays near its end
            far = torch.linalg.vector_norm(paths - paths[torch.arange(len(paths)), end][:, None], dim=2) > cfg['trim_hover'] * maze.pitch
            idx = torch.arange(paths.shape[1], device=paths.device)[None]
            far &= idx <= end[:, None]
            last_far = torch.where(far.any(1), paths.shape[1] - 1 - torch.argmax(far.flip(1).int(), 1), torch.zeros_like(end))
            end = torch.where(stalled & ~collided, torch.minimum(end, last_far + 1), end)
        cum = TR.path_time(maze, paths, d)
        tau = cfg['leaf_cells'] * maze.pitch
        leaves, inner = TR.split_tree(cum, end, tau)
        reached = conv & ~collided
        if cfg['reached_only']:
            keep = reached[inner[:, 0]]
            inner = inner[keep]
            leaves = leaves[reached[leaves[:, 0]]]
        if cfg['nodes'] in ('sub', 'both'):
            sub = TR.sub_intervals(leaves, int(cfg['sub_factor'] * max(len(inner), 1)), gen)
            inner = sub if cfg['nodes'] == 'sub' else torch.cat([inner, sub])
        lv = TR.leaf_values(ev, paths, cum, leaves, cfg['leaf'])
        tg = TR.node_targets(paths, cum, leaves, inner, lv)
        self.paths, self.cum, self.leaves, self.inner = paths, cum, leaves, inner
        self.wa = paths[inner[:, 0], inner[:, 1]].contiguous()
        self.wb = paths[inner[:, 0], inner[:, 2]].contiguous()
        self.target = tg
        self.depth = inner[:, 3]
        self.reached = reached[inner[:, 0]]
        self.size = len(inner)
        with torch.no_grad():
            tn = ev.travel(torch.cat([self.wa, self.wb], 1))
        r = torch.log(tn / tg)
        stats = dict(traversals=len(a), reached=int(reached.sum()), collided=int(collided.sum()), stalled=int(stalled.sum()),
                     converged=int(conv.sum()), steps_mean=float(stop.float().mean()), leaves=len(leaves), nodes=self.size,
                     depth_max=int(self.depth.max()) if self.size else 0,
                     log_net_over_target_median=float(r.median()) if self.size else None,
                     log_net_over_target_by_depth={int(k): float(r[self.depth == k].median()) for k in range(-1, min(8, int(self.depth.max()) + 1)) if bool((self.depth == k).any())} if self.size else {},
                     traverse_seconds=t1 - t0, seconds=time.time() - t0)
        del d
        return stats

    def retarget(self, ev, mode):
        lv = TR.leaf_values(ev, self.paths, self.cum, self.leaves, mode)
        self.target = TR.node_targets(self.paths, self.cum, self.leaves, self.inner, lv)

    def sample(self, m, gen):
        i = torch.randint(self.size, (m,), generator=gen, device=self.target.device)
        return self.wa[i], self.wb[i], self.target[i]


def rho(r, kind):
    if kind == 'l2':
        return r * r
    if kind == 'huber':
        a = r.abs()
        return torch.where(a < 0.1, 0.5 * r * r / 0.1, a - 0.05)
    return r.abs()


def train(maze, data_dir, resume, out, cfg, log, validate=None):
    """Resume ntrl-demo from checkpoint `resume` and train to cfg['epochs'] with the midpoint term. Returns the final
    checkpoint path. validate(net, epoch) -> dict is called at cfg['validate_every'] epochs (diagnostics only)."""
    mt, mf, mn, db = ntrl_modules()
    dev = 'cuda:0'
    out = Path(out)
    (out / 'checkpoints').mkdir(parents=True, exist_ok=True)
    seed = cfg['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    payload = torch.load(resume, map_location=dev, weights_only=False)
    B = payload['B_state_dict']
    network = mn.NN(dev, 3, B)
    network.load_state_dict(payload['model_state_dict'], strict=True)
    network.to(dev)
    optimizer = torch.optim.AdamW(network.parameters(), lr=5e-4, weight_decay=0.5)
    optimizer.load_state_dict(payload['optimizer_state_dict'])
    function = mf.Function(str(out), dev, network, 3)
    dataset = db.Database(str(data_dir) + '/')
    loader = mt.FastTensorDataLoader(dataset.data, batch_size=2000, shuffle=True)
    e0 = int(payload['epoch'])
    if isinstance(payload['val_loss'], (int, float)):
        total_diff = float(payload['val_loss'])
    else:                                           # ntrl-demo stores the epoch's loss only in the file name
        total_diff = float(Path(resume).stem.split('_ValLoss_')[1])
    beta, current_diff = 1.0 / total_diff, total_diff
    alpha, gamma = 1.025, 0.001
    train_loss = [float(x) for x in payload['train_loss']]
    sched = Schedule(cfg['start'], cfg['weight'], cfg['ramp'])
    gen = torch.Generator(dev).manual_seed(seed + 101)
    bank = Bank()
    ev = None
    history = dict(epochs=[], ntrl_loss=[], mid_loss=[], weight=[], rollbacks=[], refresh=[], validate=[], grad_ratio=[])
    prev_states, prev_opts = [], []
    current_state = copy.deepcopy(network.state_dict())
    current_opt = copy.deepcopy(optimizer.state_dict())
    t_start = time.time()
    use_mid = cfg['weight'] > 0
    for epoch in range(e0 + 1, cfg['epochs'] + 1):
        # midpoint bank refresh (traversals with the current network) and target re-evaluation
        if use_mid and epoch > cfg['start'] and ((epoch - cfg['start'] - 1) % cfg['refresh'] == 0):
            frozen = F.Frozen(network)
            if ev is None:
                ev = TV.Compiled(frozen, chunk=cfg['chunk'], compile=cfg['compile'])
            else:
                for (w, b), (w2, b2) in zip(ev.fz.layers, frozen.layers):
                    w.copy_(w2)
                    b.copy_(b2)
                for g, g2 in zip(ev.fz.gates, frozen.gates):
                    g.copy_(g2)
            a = maze.free(cfg['traversals'], gen, cfg['min_clear'])
            b = maze.free(cfg['traversals'], gen, cfg['min_clear'])
            if cfg['local_share'] > 0:
                k = int(cfg['local_share'] * len(a))
                off = torch.nn.functional.normalize(torch.randn(k, 3, generator=gen, device=dev), dim=1)
                rad = torch.rand(k, 1, generator=gen, device=dev) * cfg['local_cells'] * maze.pitch
                cand = a[:k] + off * rad * torch.tensor([1.0, 1.0, maze.phi_max / 0.5], device=dev)
                ok = maze.inside(cand) & (maze.clearance(cand) >= cfg['min_clear'])
                b[:k] = torch.where(ok[:, None], cand, b[:k])
            stats = bank.build(ev, maze, a, b, gen, cfg, log)
            stats['epoch'] = epoch
            history['refresh'].append(stats)
            log(f'epoch {epoch}: refresh {stats["traversals"]} traversals in {stats["seconds"]:.1f}s (MPPI {stats["traverse_seconds"]:.1f}s): reached {stats["reached"]}, '
                f'collided {stats["collided"]}, stalled {stats["stalled"]}, steps {stats["steps_mean"]:.0f}; nodes {stats["nodes"]} (depth <= {stats["depth_max"]}), '
                f'median log(T/target) {stats["log_net_over_target_median"]}, by depth {json.dumps({k: round(v, 3) for k, v in stats["log_net_over_target_by_depth"].items()})}')
            torch.cuda.empty_cache()
        elif use_mid and cfg['retarget'] and epoch > cfg['start'] and bank.size and ((epoch - cfg['start'] - 1) % cfg['retarget'] == 0):
            frozen = F.Frozen(network)
            for (w, b), (w2, b2) in zip(ev.fz.layers, frozen.layers):
                w.copy_(w2)
                b.copy_(b2)
            for g, g2 in zip(ev.fz.gates, frozen.gates):
                g.copy_(g2)
            bank.retarget(ev, cfg['leaf'])
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
            total_train_loss, total_diff, total_mid = 0.0, 0.0, 0.0
            for i, whole in enumerate(loader, 0):
                if i > 4:
                    break
                data = whole[0].to(dev)
                points = data[:, :6]
                speed = data[:, 6:8]
                normal = data[:, 8:]
                speed = speed * speed * (2 - speed) * (2 - speed)
                speed = alpha * speed + 1 - alpha
                loss_value, loss_n, _ = function.Loss(points, speed, normal, beta, gamma, epoch)
                if lam > 0 and bank.size:
                    wa, wb, y = bank.sample(cfg['batch'], gen)
                    t = F.travel(network, torch.cat([wa, wb], 1))
                    lmid = rho(torch.log(t.clamp(min=1e-6)) - torch.log(y.clamp(min=1e-6)), cfg['loss']).mean()
                    if cfg['grad_probe'] and epoch % cfg['grad_probe'] == 0 and i == 0 and rollbacks == 0:
                        gn = torch.autograd.grad(loss_value, [p for p in network.parameters() if p.requires_grad], retain_graph=True, allow_unused=True)
                        gm = torch.autograd.grad(lam * lmid, [p for p in network.parameters() if p.requires_grad], retain_graph=True, allow_unused=True)
                        nn_ = math.sqrt(sum(float((g ** 2).sum()) for g in gn if g is not None))
                        nm_ = math.sqrt(sum(float((g ** 2).sum()) for g in gm if g is not None))
                        history['grad_ratio'].append([epoch, nn_, nm_])
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
            if (0 < ratio < 1.2) or rollbacks >= cfg['max_rollbacks']:
                break
            rollbacks += 1
            with torch.no_grad():
                k = random.randint(0, len(prev_states) - 1)
                network.load_state_dict(prev_states[k], strict=True)
                optimizer.load_state_dict(prev_opts[k])
        train_loss.append(total_train_loss)
        beta = 1.0 / total_diff
        if epoch % 10 == 0 or rollbacks:
            history['epochs'].append(epoch)
            history['ntrl_loss'].append(total_diff)
            history['mid_loss'].append(total_mid / 5)
            history['weight'].append(lam)
            history['rollbacks'].append(rollbacks)
        if epoch % 100 == 0:
            el = time.time() - t_start
            log(f'epoch {epoch}: ntrl loss {total_diff:.4e}, mid loss {total_mid / 5:.4f}, lambda {lam:.3g}, rollbacks {rollbacks}, '
                f'{el / 60:.1f} min ({el / (epoch - e0):.2f} s/epoch)')
        if epoch % cfg['save_every'] == 0 or epoch == cfg['epochs']:
            path = out / 'checkpoints' / f'epoch_{epoch:05d}.pt'
            torch.save({'epoch': epoch, 'model_state_dict': network.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'B_state_dict': B,
                        'train_loss': train_loss, 'val_loss': total_diff}, path)
        if validate is not None and cfg['validate_every'] and (epoch % cfg['validate_every'] == 0 or epoch == cfg['epochs']):
            v = validate(network, epoch)
            v['epoch'] = epoch
            history['validate'].append(v)
        if epoch % 100 == 0:
            with open(out / 'history.json', 'w') as f:
                json.dump(history, f)
    with open(out / 'history.json', 'w') as f:
        json.dump(history, f)
    final = out / 'checkpoint.pt'
    torch.save({'epoch': cfg['epochs'], 'model_state_dict': network.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'B_state_dict': B,
                'train_loss': train_loss, 'val_loss': total_diff}, final)
    return final, history
