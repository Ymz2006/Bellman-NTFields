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
from maze import OFFSET_CELLS  # noqa: E402

_spec = importlib.util.spec_from_file_location('lift_scale_trainer', ROOT / 'jeffrey/lift_scale/trainer.py')
LT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LT)


def tri_term(net, scene, points, T, radius, gen, samples=6):
    """Mean squared and mean absolute violation of |T(q, g) - T(q', g)| <= tau over random free steps at both ends, and the share of steps used."""
    N, dev = len(points), points.device
    sq, lin, used, jump = 0.0, 0.0, 0.0, 0.0
    ts = torch.linspace(0, 1, samples, device=dev)
    for end in (0, 1):
        sl = slice(3 * end, 3 * end + 3)
        delta = torch.randn(N, 3, generator=gen, device=dev)
        delta = delta / torch.linalg.vector_norm(delta, dim=1, keepdim=True) * radius * torch.rand(N, 1, generator=gen, device=dev) ** (1 / 3)
        a = points[:, sl]
        b = torch.minimum(torch.maximum(a + delta, scene.lo), scene.hi)
        with torch.no_grad():
            p = a[:, None, :] + ts[None, :, None] * (b - a)[:, None, :]
            d = scene.clearance(p.reshape(-1, 3)).reshape(N, samples)
            ok = d.amin(1) > -OFFSET_CELLS
            inv = 1 / scene.train_speed(d)
            tau = torch.linalg.vector_norm(b - a, dim=1) * (0.5 * (inv[:, 1:] + inv[:, :-1])).mean(1)
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


def train(data_dir, out, seed, epochs, cfg, scene, log, blocked=None, schedule=None, mu=0.1, bound=3.0, push_all=False, save_at=(), mu1=0.0, b_scale=1.0,
          tri=0.0, tri1=0.0, tri_r=0.03):
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
    use_tri = tri > 0 or tri1 > 0
    beta = 1.0
    current_diff = 1000.0
    keys = ('epochs', 'ntrl_loss', 'rollbacks', 'lam', 'push', 'one_sided', 'pushed_share', 't_blocked', 't_free', 'tri_sq', 'tri_lin', 'tri_used', 'tri_max')
    history = {k: [] for k in keys}
    history['all_loss'] = []
    current_state = copy.deepcopy(network.state_dict())
    current_opt = copy.deepcopy(optimizer.state_dict())
    prev_states, prev_opts = [], []
    train_loss = []
    t_start = time.time()
    save_at = set(int(e) for e in save_at) | {epochs}
    log(f'network: lift {network.lift_kind} gain {network.lift_gain}, pair {cfg["pair"]}, octaves {network.octaves}, parameters {sum(p.numel() for p in network.parameters())}, '
        f'push schedule {schedule}, mu {mu}, mu1 {mu1}, bound {bound}, triangle cap {tri} / {tri1} radius {tri_r}')
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
            acc = dict(push=0.0, one=0.0, share=0.0, tb=0.0, tf=0.0, tsq=0.0, tlin=0.0, tused=0.0, tmax=0.0)
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
                loss_value, loss_n, T, r0, r1 = LT.ntrl_loss(function, points, speed, normal, beta)
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
                                   acc['tsq'] / 5, acc['tlin'] / 5, acc['tused'] / 5, acc['tmax'])):
                history[k].append(v)
        if epoch % 100 == 0:
            el = time.time() - t_start
            msg = f'epoch {epoch}: ntrl loss {total_diff:.4e}, rollbacks {rollbacks}, {el / 60:.1f} min ({el / epoch:.3f} s/epoch)'
            if blocked is not None:
                msg += f', lambda {lam:.2e}, push {acc["push"] / 5:.3e}, one-sided {acc["one"] / 5:.3e}, T blocked {acc["tb"] / 5:.3f} / free {acc["tf"] / 5:.3f}'
            if use_tri:
                msg += f', triangle sq {acc["tsq"] / 5:.3e} lin {acc["tlin"] / 5:.3e} max {acc["tmax"]:.3f} (steps used {acc["tused"] / 5:.2f})'
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
