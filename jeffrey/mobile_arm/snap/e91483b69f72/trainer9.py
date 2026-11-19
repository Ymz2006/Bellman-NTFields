"""ntrl-demo's training loop (models/metric Model.train(), dim 9) with, from a start epoch on, the travel-time midpoint loss.

Epochs 1 .. start are ntrl-demo's Model.train() call for call (checked against the package itself in check_ntrl(): same losses to the last
digit): seed -> B = N(0, 1) truncated at 2 (128 x 9) -> NN(dim 9) -> init_weights -> AdamW (lr 1e-3, then 5e-4 every epoch, weight decay
0.5) -> FastTensorDataLoader reshuffle -> 5 minibatches of 2,000 pairs per epoch through the package's own Function.Loss (eikonal + TD +
normal terms, weight exp(-0.5 T), beta = 1 / previous loss; speed stretch 1.025 s^2 (2 - s)^2 - 0.025) -> rollback to one of the last 5
states when the loss rises 20 %. With --weight 0 the whole run is ntrl-demo (the baseline).

After the start epoch every update adds jeffrey/midpoint_dp's term (its MAIN setting; midpoint_dp/trainer.py and lift_midpoint):

    loss = beta * loss_ntrl + lambda(epoch) * mean over 8,192 sampled nodes of | log T(w_s, w_e) - log y(s, e) |

Every --refresh epochs, --traversals random pairs (a, b) of 9-D configurations with exact clearance >= 0.1 cells are driven by ntrl-demo's
MPPI on the current field all at once (traverse9.mppi; 'safe': rollouts whose first state comes within 0.02 cells of anything get +100),
cut where they first collide or stall, parametrized by travel time along the drive (exact speed of scene9's clearance), split at
travel-time midpoints down to one cell of travel time (tree.py, verbatim midpoint_dp); a leaf is worth min(T(leaf ends), the drive's time
over it); every inner node and as many random sub-paths between leaf boundaries get the sum of their leaves as target y. lambda ramps
linearly from 0 at the start epoch to --weight over --ramp epochs; after the start rollbacks are capped at --max-rollbacks per epoch.
"""
import copy
import json
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
import fastnet as F  # noqa: E402  (verbatim copy of jeffrey/midpoint_dp/fastnet.py)
import traverse9 as TV  # noqa: E402
import tree as TR  # noqa: E402  (verbatim copy of jeffrey/midpoint_dp/tree.py; path_time is always given the clearance)

DIM = 9


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


def free_counted(scene, clear, count, gen, min_clear, batch=262144):
    out, have = [], 0
    while have < count:
        q = scene.uniform(batch, gen)
        q = q[clear(q) >= min_clear]
        out.append(q)
        have += len(q)
    return torch.cat(out)[:count]


class Bank:
    """Inner nodes of the current midpoint trees and sub-paths: endpoints, targets, depth."""

    def __init__(self):
        self.size = 0

    def build(self, ev, scene, a, b, gen, cfg, clear):
        t0 = time.time()
        st = cfg['planner']
        q0 = clear.n
        tab = ev.travel(torch.cat([a, b], 1))
        budget = torch.clamp(torch.ceil(cfg['budget_factor'] * tab / st['step']), st['min_steps'], cfg['max_steps'])
        safe = dict(clearance=clear, safe_margin=cfg['safe_margin'], safe_mode=cfg['safe_mode']) if cfg['traverse'] == 'safe' else {}
        paths, stop, conv, stalled = TV.mppi(ev, a, b, gen, budget, scene, stall_window=cfg['stall_window'],
                                             stall_drop=cfg['stall_drop_cells'] * scene.pitch, **safe, **st)
        t1 = time.time()
        d, first = TV.collisions(scene, paths, stop, clear=clear)
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
        self.target = tg
        self.depth = inner[:, 3]
        self.size = len(inner)
        with torch.no_grad():
            tn = ev.travel(torch.cat([self.wa, self.wb], 1)) if self.size else tg
        r = torch.log(tn / tg) if self.size else tg
        L_path = torch.linalg.vector_norm(paths[:, 1:, :2] - paths[:, :-1, :2], dim=2).sum(1) / scene.pitch
        L_line = torch.linalg.vector_norm(paths[torch.arange(len(paths)), end, :2] - a[:, :2], dim=1) / scene.pitch
        stats = dict(traversals=len(a), reached=int(reached.sum()), collided=int(collided.sum()), stalled=int(stalled.sum()),
                     converged=int(conv.sum()), steps_mean=float(stop.float().mean()), steps_max=int(stop.max()), leaves=len(leaves), nodes=self.size,
                     tree_nodes=n_tree, depth_max=int(self.depth.max()) if self.size else 0,
                     log_net_over_target_median=float(r.median()) if self.size else None,
                     log_net_over_target_by_depth={int(k): float(r[self.depth == k].median()) for k in range(-1, min(8, int(self.depth.max()) + 1)) if bool((self.depth == k).any())} if self.size else {},
                     reached_path_over_line_median=float((L_path[reached] / L_line[reached].clamp(min=1e-3)).median()) if bool(reached.any()) else None,
                     traverse_seconds=t1 - t0, seconds=time.time() - t0, clearance_queries=clear.n - q0)
        del d, paths, cum
        return stats

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


def train(scene, data_dir, out, cfg, log, validate=None):
    """Train from epoch 1 to cfg['epochs']; returns (final checkpoint, history)."""
    mt, mf, mn, db = ntrl_modules()
    dev = 'cuda:0'
    out = Path(out)
    (out / 'checkpoints').mkdir(parents=True, exist_ok=True)
    seed = cfg['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    e0, train_loss = 0, []
    if cfg.get('resume'):
        # resume a run's checkpoint (network, AdamW state, B, its last loss as beta), as jeffrey/midpoint_dp/trainer.py does
        payload = torch.load(cfg['resume'], map_location='cpu', weights_only=False)
        B = payload['B_state_dict'].cpu()
        network = mn.NN(dev, DIM, B)
        network.load_state_dict(payload['model_state_dict'], strict=True)
        network.to(dev)
        optimizer = torch.optim.AdamW(network.parameters(), lr=5e-4, weight_decay=0.5)
        optimizer.load_state_dict(payload['optimizer_state_dict'])
        e0 = int(payload['epoch'])
        current_diff = float(payload['val_loss'])
        beta = 1.0 / current_diff
        train_loss = [float(x) for x in payload.get('train_loss', [])]
        log(f'resumed {cfg["resume"]} at epoch {e0} (loss {current_diff:.4e})')
    else:
        # ntrl-demo Model.train(), call for call
        B = torch.normal(0, 1, size=(128, DIM))
        torch.nn.init.trunc_normal_(B, mean=0.0, std=1, a=-2.0, b=2.0)
        network = mn.NN(dev, DIM, B)
        network.apply(network.init_weights)
        network.to(dev)
        optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3, weight_decay=0.5)
        beta = 1.0
        current_diff = 1000.0
    function = mf.Function(str(out), dev, network, DIM)
    dataset = db.Database(str(data_dir) + '/')
    loader = mt.FastTensorDataLoader(dataset.data, batch_size=2000, shuffle=True)
    alpha, gamma = 1.025, 0.001
    sched = Schedule(cfg['start'], cfg['weight'], cfg['ramp'])
    gen = torch.Generator(dev).manual_seed(seed + 101)
    bank, ev, clear = Bank(), None, None
    history = dict(epochs=[], ntrl_loss=[], mid_loss=[], weight=[], rollbacks=[], refresh=[], validate=[], all_loss=[])
    current_state = copy.deepcopy(network.state_dict())
    current_opt = copy.deepcopy(optimizer.state_dict())
    prev_states, prev_opts = [], []
    t_start = time.time()
    use_mid = cfg['weight'] > 0
    save_at = set(int(e) for e in cfg['save_at']) | {cfg['epochs']}
    log(f'ntrl-demo NN dim {DIM}, parameters {sum(p.numel() for p in network.parameters())}; midpoint '
        + (f'from epoch {cfg["start"]} weight {cfg["weight"]} ramp {cfg["ramp"]}, refresh {cfg["refresh"]} x {cfg["traversals"]} traversals ({cfg["traverse"]}), '
           f'leaves {cfg["leaf"]} {cfg["leaf_cells"]} cells, nodes {cfg["nodes"]}' if use_mid else 'off (ntrl-demo baseline)'))
    for epoch in range(e0 + 1, cfg['epochs'] + 1):
        if use_mid and epoch > cfg['start'] and ((epoch - cfg['start'] - 1) % cfg['refresh'] == 0):
            if ev is None:
                ev = TV.Compiled(F.Frozen(network), chunk=cfg['chunk'], compile=cfg['compile'])
                clear = TV.Clearance(scene, compile=cfg['compile'])
            else:
                ev.update(network)
            n0 = clear.n
            a = free_counted(scene, clear, cfg['traversals'], gen, cfg['min_clear'])
            b = free_counted(scene, clear, cfg['traversals'], gen, cfg['min_clear'])
            q_ends = clear.n - n0
            for attempt in range(4):
                try:
                    stats = bank.build(ev, scene, a, b, gen, cfg, clear)
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
            stats['clearance_queries_ends'] = q_ends
            history['refresh'].append(stats)
            log(f'epoch {epoch}: refresh {stats["traversals"]} traversals in {stats["seconds"]:.1f}s (MPPI {stats["traverse_seconds"]:.1f}s, steps mean {stats["steps_mean"]:.0f} max {stats["steps_max"]}): '
                f'reached {stats["reached"]}, collided {stats["collided"]}, stalled {stats["stalled"]}; nodes {stats["nodes"]} (tree {stats["tree_nodes"]}, depth <= {stats["depth_max"]}), '
                f'median log(T/target) {stats["log_net_over_target_median"]}, by depth {json.dumps({k: round(v, 3) for k, v in stats["log_net_over_target_by_depth"].items()})}, '
                f'reached path/line {stats["reached_path_over_line_median"]}')
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
            total_train_loss, total_diff, total_mid = 0.0, 0.0, 0.0
            ii = 0
            for i, whole in enumerate(loader, 0):
                if ii > 4:
                    break
                ii += 1
                batch = whole[0].to(dev)
                points = batch[:, :2 * DIM]
                speed = batch[:, 2 * DIM:2 * DIM + 2]
                normal = batch[:, 2 * DIM + 2:]
                speed = speed * speed * (2 - speed) * (2 - speed)
                speed = alpha * speed + 1 - alpha
                loss_value, loss_n, _ = function.Loss(points, speed, normal, beta, gamma, epoch)
                if lam > 0 and bank.size:
                    wa, wb, y = bank.sample(cfg['batch'], gen)
                    t = F.travel(network, torch.cat([wa, wb], 1))
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
            if epoch > cfg['start'] and use_mid and rollbacks >= cfg['max_rollbacks']:
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
            history['weight'].append(lam)
            history['rollbacks'].append(rollbacks)
        if epoch % 100 == 0 or epoch <= e0 + 3:
            el = time.time() - t_start
            log(f'epoch {epoch}: ntrl loss {total_diff:.4e}, mid loss {total_mid / 5:.4f}, lambda {lam:.3g}, rollbacks {rollbacks}, {el / 60:.1f} min ({el / (epoch - e0):.3f} s/epoch)')
        if epoch in save_at:
            torch.save({'epoch': epoch, 'model_state_dict': network.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'B_state_dict': B,
                        'train_loss': train_loss, 'val_loss': total_diff}, out / 'checkpoints' / f'epoch_{epoch:05d}.pt')
        if validate is not None and epoch in set(cfg['validate_at']):
            v = validate(network, epoch)
            v['epoch'] = epoch
            history['validate'].append(v)
        if epoch % 100 == 0:
            with open(out / 'history.json', 'w') as f:
                json.dump(history, f)
    final = out / 'checkpoint.pt'
    torch.save({'epoch': cfg['epochs'], 'model_state_dict': network.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'B_state_dict': B,
                'train_loss': train_loss, 'val_loss': total_diff}, final)
    history['final_loss'] = total_diff
    history['seconds'] = time.time() - t_start
    with open(out / 'history.json', 'w') as f:
        json.dump(history, f)
    return final, history


def check_ntrl(data_dir, out, epochs=3, seed=87358):
    """Losses of ntrl-demo's own Model.train() (dim 9) and of train() with the midpoint off, for the first epochs."""
    mt, mf, mn, db = ntrl_modules()
    import io
    import contextlib
    out = Path(out)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cwd = os.getcwd()
    os.chdir(NTRL)
    buf = io.StringIO()
    try:
        model = mt.Model(str(out / 'ntrl_model'), str(Path(data_dir).resolve()) + '/', DIM, [0.0] * DIM, device='cuda:0')
        model.Params['Training']['Number of Epochs'] = epochs
        (out / 'ntrl_model').mkdir(parents=True, exist_ok=True)
        with contextlib.redirect_stdout(buf):
            model.train()
    finally:
        os.chdir(cwd)
    ref = [float(l.split('Loss = ')[1].split(' ')[0]) for l in buf.getvalue().splitlines() if l.startswith('Epoch = ')]
    mine = []
    cfg = dict(seed=seed, epochs=epochs, start=10 ** 9, weight=0.0, ramp=1, save_at=[], validate_at=[])
    _, hist = train(None, data_dir, out / 'mine', cfg, lambda m: None)
    mine = hist['all_loss']
    return dict(ntrl=ref, mine=[float(f'{x:.4e}') for x in mine], equal=[f'{x:.4e}' for x in mine] == [f'{x:.4e}' for x in ref])
