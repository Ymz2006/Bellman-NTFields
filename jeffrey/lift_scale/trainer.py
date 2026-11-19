"""ntrl-demo's models/metric training loop (Model.train(), step for step, as jeffrey/embed_warp/trainer.py) with the lifted network
(lift.py) and the largest-solution push.

Same global RNG calls in the same order as the package: seed -> B -> network -> init_weights -> per-epoch reshuffle -> rollback draws;
5 minibatches of 2,000 pairs per epoch, the package's loss (eikonal + TD + normal terms, weight exp(-0.5 T), beta = 1 / previous loss),
AdamW lr 5e-4 weight decay 0.5, rollback to one of the last 5 states when ntrl-demo's own loss rises 20 %. With lift 'none' and no push
this is ntrl-demo itself.

The push (the selection term of the lift page), added inside the bracket that beta multiplies:

    - lambda(epoch) * mean of T over the pairs of the minibatch whose straight segment is blocked and whose T is below `bound`
    + mu * mean of relu(sqrt(S |grad T|) - 1)^2 at both ends            (mu applies while lambda > 0, faded in over the first ramp)
    + mu1 * mean of relu(sqrt(S |grad T|) - 1) at both ends            (optional exact penalty, same fade)

lambda(epoch) is piecewise linear through `schedule` (epoch, value) knots, 0 before the first knot (the page's: 0 at 200, 1e-2 from 600
to 2,000, 3e-3 from 3,000 on). The blocked flag of every training pair is computed once (stage 'blocked') and travels with the pair as
a fifteenth data column. ntrl-demo's own loss value (its beta and its epoch-repeat rule) never sees the push.
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
import lift as LF  # noqa: E402

PAGE_SCHEDULE = [(200, 0.0), (600, 1e-2), (2000, 1e-2), (3000, 3e-3)]


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


def parse_schedule(text, epochs):
    """'page' | 'none' | 'const:<v>' (faded in over epochs 200-600) | 'e:v,e:v,...'; knots given for 5,000 epochs are kept as they are."""
    if not text or text == 'none':
        return []
    if text == 'page':
        return list(PAGE_SCHEDULE)
    if text.startswith('const:'):
        v = float(text.split(':')[1])
        return [(200, 0.0), (600, v)]
    return [(int(a), float(b)) for a, b in (t.split(':') for t in text.split(','))]


def lam_at(schedule, epoch):
    if not schedule or epoch < schedule[0][0]:
        return 0.0
    for (e0, v0), (e1, v1) in zip(schedule[:-1], schedule[1:]):
        if e0 <= epoch <= e1:
            return v0 + (v1 - v0) * (epoch - e0) / max(e1 - e0, 1)
    return schedule[-1][1]


def ntrl_loss(function, points, Yobs, normal, beta, td_len=0.03):
    """ntrl-demo's metric loss (models/metric/model_function_metric.py, Function.Loss) line for line (jeffrey/embed_warp/trainer.py
    checked the equality, gradients included). Also returns T and sqrt(S |grad T|) at both ends for the push."""
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
        tau_new0, w, Xp_new0 = net.out(Xp_new0)
        tau_new1 = length0
        del Xp_new0, Dir0
    tau_loss0 = td_weight * ((tau - (tau_new0 + tau_new1)) ** 2).squeeze()
    with torch.no_grad():
        length1 = td_len / (Yobs[:, 1]).unsqueeze(1)
        Dir1 = length1 * (DT1 * Yobs[:, 1].unsqueeze(1) ** 2).clone().detach()
        Xp_new0 = Xp.clone().detach()
        Xp_new0[:, dim:] = Xp_new0[:, dim:] - Dir1
        tau_new0, w, Xp_new0 = net.out(Xp_new0)
        tau_new1 = length1
        del Xp_new0, Dir1
    tau_loss1 = td_weight * ((tau - (tau_new0 + tau_new1)) ** 2).squeeze()
    where_d0 = (tau[:, 0] < length0.squeeze())
    where_d1 = (tau[:, 0] < length1.squeeze())
    tau_loss0[where_d0] = 0
    tau_loss1[where_d1] = 0
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
    return loss, loss_n, T, l0_2, l1_2


def train(data_dir, out, seed, epochs, cfg, scene, log, blocked=None, schedule=None, mu=0.1, bound=3.0, push_all=False, save_at=(), use_package_loss=False, mu_abrupt=False, mu1=0.0, b_scale=1.0):
    """Returns (final checkpoint path, history). cfg: lift.LiftNN config; blocked: (N,) bool for the training pairs (needed for a push);
    schedule: [(epoch, lambda)]; bound: pairs are pushed only while T < bound (normalized units = box lengths)."""
    mt, mf, mn, db = ntrl_modules()
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
        B = B * b_scale                  # ntrl-demo's Fourier matrix times a constant (jeffrey/maze_scale/fourier.py); same draws
    cfg = dict(cfg)
    cfg['lift'] = dict(cfg.get('lift') or {}, seed=seed)
    network = LF.build_network(mn, dev, B, cfg, scene)
    function = mf.Function(str(out), dev, network, 3)
    optimizer = torch.optim.AdamW([{'params': list(network.parameters())}], lr=1e-3, weight_decay=0.5)
    dataset = db.Database(str(data_dir) + '/')
    data = dataset.data
    schedule = list(schedule or [])
    if schedule:
        assert blocked is not None and len(blocked) == len(data), 'the push needs the blocked flag of every training pair'
        data = torch.cat([data, torch.as_tensor(blocked, dtype=data.dtype)[:, None]], 1)
    loader = mt.FastTensorDataLoader(data, batch_size=2000, shuffle=True)
    beta = 1.0
    current_diff = 1000.0
    history = dict(epochs=[], ntrl_loss=[], rollbacks=[], lam=[], push=[], one_sided=[], pushed_share=[], t_blocked=[], t_free=[], all_loss=[])
    current_state = copy.deepcopy(network.state_dict())
    current_opt = copy.deepcopy(optimizer.state_dict())
    prev_states, prev_opts = [], []
    train_loss = []
    t_start = time.time()
    save_at = set(int(e) for e in save_at) | {epochs}
    log(f'network: lift {network.lift_kind} gain {network.lift_gain}, octaves {network.octaves}, parameters {sum(p.numel() for p in network.parameters())}, '
        f'push schedule {schedule}, mu {mu}, mu1 {mu1}, bound {bound}')
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
        lam = lam_at(schedule, epoch)
        # the one-sided term is faded in over the push's first ramp (at full weight from its first epoch it trips the epoch-repeat rule)
        ramp = 1.0 if (mu_abrupt or len(schedule) < 2) else min(1.0, max(0.0, (epoch - schedule[0][0]) / max(schedule[1][0] - schedule[0][0], 1)))
        while True:
            total_train_loss, total_diff = 0.0, 0.0
            acc = dict(push=0.0, one=0.0, share=0.0, tb=0.0, tf=0.0)
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
                if use_package_loss:
                    loss_value, loss_n, _ = function.Loss(points, speed, normal, beta, gamma, epoch)
                else:
                    loss_value, loss_n, T, r0, r1 = ntrl_loss(function, points, speed, normal, beta)
                    if schedule:
                        blk = batch[:, 14] > 0.5
                        with torch.no_grad():
                            acc['tb'] += float(T[blk].mean()) if bool(blk.any()) else 0.0
                            acc['tf'] += float(T[~blk].mean()) if bool((~blk).any()) else 0.0
                        if lam > 0:
                            m = (torch.ones_like(blk) if push_all else blk) & (T.detach() < bound)
                            push = -lam * T[m].mean() if bool(m.any()) else T.sum() * 0
                            one = mu * ramp * (torch.relu(r0 - 1) ** 2 + torch.relu(r1 - 1) ** 2).mean()
                            if mu1 > 0:
                                # exact (linear) penalty: its gradient does not vanish at the constraint, so the push cannot buy a
                                # violation proportional to its strength
                                one = one + mu1 * ramp * (torch.relu(r0 - 1) + torch.relu(r1 - 1)).mean()
                            loss_value = loss_value + beta * (push + one)
                            acc['push'] += float(push)
                            acc['one'] += float(one)
                            acc['share'] += float(m.float().mean())
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
            history['epochs'].append(epoch)
            history['ntrl_loss'].append(total_diff)
            history['rollbacks'].append(rollbacks)
            history['lam'].append(lam)
            history['push'].append(acc['push'] / 5)
            history['one_sided'].append(acc['one'] / 5)
            history['pushed_share'].append(acc['share'] / 5)
            history['t_blocked'].append(acc['tb'] / 5)
            history['t_free'].append(acc['tf'] / 5)
        if epoch % 100 == 0:
            el = time.time() - t_start
            msg = f'epoch {epoch}: ntrl loss {total_diff:.4e}, rollbacks {rollbacks}, {el / 60:.1f} min ({el / epoch:.3f} s/epoch)'
            if schedule:
                msg += f', lambda {lam:.2e}, push {acc["push"] / 5:.3e}, one-sided {acc["one"] / 5:.3e}, pushed share {acc["share"] / 5:.3f}, T blocked {acc["tb"] / 5:.3f} / free {acc["tf"] / 5:.3f}'
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
