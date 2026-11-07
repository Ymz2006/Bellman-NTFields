"""ntrl-demo's models/metric training loop with a warp in front of its encoder (warp.py), from scratch.

This is baselines/ntrl-demo/models/metric/model_train_metric.py Model.train() step for step, with the same global RNG
calls in the same order (seed -> B ~ N(0, 1) truncated at 2 -> NN(...) -> apply(init_weights) -> per-epoch reshuffle
of FastTensorDataLoader -> random.randint for rollbacks): 5 minibatches of 2,000 pairs per epoch, the package's own
Function.Loss (eikonal + TD + normal terms, weight exp(-0.5 T), normalized by beta = 1 / previous epoch's loss), AdamW
lr 5e-4 weight decay 0.5, epoch rollback to a random one of the last 5 states when the loss rises by 20 %. With an
identity warp (s = 1, no grids) it is ntrl-demo itself. Plotting and the copy of the package sources are skipped
(they draw no random numbers).

The warp is attached after the network's initialization and draws no global random numbers. Its grids (when learned)
are a second AdamW parameter group (lr --warp-lr, weight decay 0), frozen until --warp-start. Warp terms added to
every minibatch's loss (the rollback test and beta use the package's loss only):

* free compression (README_warp's 'suck in free space'): free_weight * mean relu(log det J_psi + log free_c)^2 over
  configurations whose free radius >= free_radius cells (a fixed pool, sampled per minibatch): their area is pushed
  to at most 1 / free_c; since psi preserves the box, the area goes to the rest of the maze.
* band expansion: band_weight * mean relu(log band_c - log det J_psi)^2 over the minibatch's x0 (all within the margin
  of a wall): near-wall area pushed to at least band_c.
* heading: heading_weight * mean relu(log heading_c - g)^2 over x0 and relu(g + log heading_c)^2 over the free pool
  (heading stretched near walls, compressed in the open), when the heading grid is learned.
* normal stretch: normal_weight * mean relu(log normal_k - log |J_psi n| / |n|)^2 over x0 with n its clearance normal
  (xy part): stretched by at least normal_k across the wall it is near (anisotropic; README_warp section 10).
* smoothness: smooth_weight * mean squared second difference of the bounded grids.
* temporal-difference step (td_len, optional): ntrl-demo's loss steps 0.03 normalized units along -grad T for its TD
  term, a fixed share of the maze (0.66 cells at 22 x 22, 2.6 cells at 88 x 88, where corridors are 1 cell wide);
  td_len replaces it in ntrl_loss, a line-for-line copy of the package's loss (td_len = 0.03 gives it exactly).
* straight-line lower bound (any field, warped or not; lb_weight > 0): the speed is at most 1, so the travel time is
  at least the straight-line distance in the C-space metric, T(a, b) >= |a - b|. lb_weight * mean relu(|a - b| -
  T(a, b))^2 over lb_batch pose pairs drawn uniformly in the C-space box per minibatch (no labels; the warp-term
  generator). Added from epoch 1 with full weight, outside the warp ramp.
* line of sight (los_weight > 0): if the straight segment a -> b keeps the robot beyond the wall margin (speed 1 all
  along), its time |a - b| is the travel time exactly (no path is shorter). los_weight * mean over such pairs of
  (T(a, b) - |a - b|)^2, with the lower bound above on all pairs, from a pool of lb_pool uniform pairs drawn once
  (seed + 11) and tested by line_of_sight (conservative: the xy segment misses every wall box inflated by the
  robot's circumscribed radius plus the margin, so any heading along it is clear).
Their weights ramp in linearly over --ramp epochs from --warp-start; after --warp-stop the warp is frozen (the field
finishes training on fixed coordinates).
Warped-space training (z_space): a fixed warp only. The training poses are mapped once to z = phi(x) (no_grad), the
clearance normals to the unit vectors along J^-T n, and the network bypasses its warp while training, so ntrl-demo's
loss is taken in z (the network is trained on the warped maze as if it were the maze: it never sees x, and no
gradient reaches the warp or passes through its Jacobian). The saved field is T(x0, x1) = H(phi(x0), phi(x1)), in
z-units of time (inside a region the warp magnifies m times, m times ntrl-demo's units).
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
import warp as W  # noqa: E402


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


def ramp(epoch, start, length):
    if epoch <= start:
        return 0.0
    return min(1.0, (epoch - start) / max(length, 1))


def train(data_dir, out, seed, epochs, warp_cfg, cfg, log, maze=None, free_pool=None, design=None, save_every=1000):
    """Returns (final checkpoint path, history). warp_cfg: warp.Warp config; cfg: trainer options (warp_lr, warp_start,
    ramp, free_c, free_weight, band_c, band_weight, heading_c, heading_weight, smooth_weight); free_pool: (M, 3) free
    configurations for the compression term; design(warp) sets designed grids before training."""
    mt, mf, mn, db = ntrl_modules()
    dev = 'cuda:0'
    out = Path(out)
    (out / 'checkpoints').mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Model.train(): B, network, init (same calls as the package)
    B = torch.normal(0, 1, size=(128, 3))
    torch.nn.init.trunc_normal_(B, mean=0.0, std=1, a=-2.0, b=2.0)
    network = W.warped_class(mn)(dev, 3, B)
    network.apply(network.init_weights)
    network.to(dev)
    base_params = list(network.parameters())
    warp = W.Warp(warp_cfg).to(dev)
    if design is not None:
        design(warp)
    network.warp = warp
    warp_params = [p for p in warp.parameters() if p.requires_grad]
    groups = [{'params': base_params}]
    if warp_params:
        groups.append({'params': warp_params, 'lr': cfg['warp_lr'], 'weight_decay': 0.0})
    function = mf.Function(str(out), dev, network, 3)
    optimizer = torch.optim.AdamW(groups, lr=1e-3, weight_decay=0.5)
    dataset = db.Database(str(data_dir) + '/')
    if cfg.get('z_space'):
        # Warped-space training: the network never sees x. Every training pose is replaced once by z = phi(x) (the
        # fixed warp, under no_grad: no gradient reaches the warp's parameters or flows through its Jacobian), the
        # clearance normals by their warped unit vectors (J^-T n), and the warp is bypassed inside the network, so
        # ntrl-demo's loss (eikonal, TD step 0.03, normal term) is taken in z exactly as it is taken in x without a
        # warp. The saved checkpoint keeps the warp, so T(x0, x1) = H(phi(x0), phi(x1)) with H trained on z.
        assert not warp_params and not any(p.requires_grad for p in warp.parameters()), 'warped-space training needs a fixed warp'
        assert not cfg.get('lb_weight') and not cfg.get('los_weight') and cfg.get('td_len') is None, 'warped-space training: lb / los / td_len not supported'
        d = dataset.data
        with torch.no_grad():
            x = d[:, :6].to(dev)
            nrm = d[:, 8:14].to(dev)
            z0, n0 = W.to_warped(warp, x[:, :3], nrm[:, :3])
            z1, n1 = W.to_warped(warp, x[:, 3:], nrm[:, 3:])
            d[:, :6] = torch.cat([z0, z1], 1).cpu()
            d[:, 8:14] = torch.cat([n0, n1], 1).cpu()
        network.bypass = True
        warp.cfg['z_space'] = True
        log(f'warped-space training: {len(d)} pairs mapped to z (z range x {float(d[:, 0].min()):.3f}..{float(d[:, 0].max()):.3f}, heading '
            f'{float(d[:, 2].min()):.4f}..{float(d[:, 2].max()):.4f}); warp bypassed in the network, no warp parameter trainable')
    loader = mt.FastTensorDataLoader(dataset.data, batch_size=2000, shuffle=True)
    beta = 1.0
    current_diff = 1000.0
    gen = torch.Generator(dev).manual_seed(seed + 7)          # warp-term sampling (not the global RNG)
    history = dict(epochs=[], ntrl_loss=[], warp_terms=[], rollbacks=[], warp_stats=[])
    current_state = copy.deepcopy(network.state_dict())
    current_opt = copy.deepcopy(optimizer.state_dict())
    prev_states, prev_opts = [], []
    train_loss = []
    t_start = time.time()
    use_terms = bool(warp_params)
    if cfg.get('lb_weight', 0) > 0:
        assert maze is not None, 'the lower bound needs the maze (C-space box)'
        lb_scale = torch.tensor([1.0, 1.0, 2 * maze.phi_max] * 2, device=dev)
    if cfg.get('los_weight', 0) > 0:
        assert maze is not None, 'line of sight needs the maze'
        pg = torch.Generator(dev).manual_seed(seed + 11)
        pool_u = (torch.rand(cfg.get('lb_pool', 1_000_000), 6, generator=pg, device=dev) - 0.5) * torch.tensor([1.0, 1.0, 2 * maze.phi_max] * 2, device=dev)
        pool_d = torch.linalg.vector_norm(pool_u[:, :3] - pool_u[:, 3:], dim=1)
        pool_vis = line_of_sight(maze, pool_u[:, :3], pool_u[:, 3:])
        log(f'line-of-sight pool: {len(pool_u)} uniform pairs, {float(pool_vis.float().mean()):.3f} visible')
    for epoch in range(1, epochs + 1):
        alpha, gamma = 1.025, 0.001
        prev_states.append(current_state)
        prev_opts.append(current_opt)
        if len(prev_states) > 5:
            prev_states.pop(0)
            prev_opts.pop(0)
        current_state = copy.deepcopy(network.state_dict())
        current_opt = copy.deepcopy(optimizer.state_dict())
        optimizer.param_groups[0]['lr'] = 5e-4
        active = use_terms and cfg['warp_start'] < epoch <= cfg.get('warp_stop', 10 ** 9)
        for p in warp_params:
            p.requires_grad_(active)
        lam = ramp(epoch, cfg['warp_start'], cfg['ramp']) if active else 0.0
        prev_diff = current_diff
        rollbacks = 0
        while True:
            total_train_loss, total_diff = 0.0, 0.0
            terms = dict(free=0.0, band=0.0, heading=0.0, smooth=0.0)
            ii = 0
            for i, whole in enumerate(loader, 0):
                if ii > 4:
                    break
                ii += 1
                data = whole[0].to(dev)
                points = data[:, :6]
                speed = data[:, 6:8]
                normal = data[:, 8:]
                speed = speed * speed * (2 - speed) * (2 - speed)
                speed = alpha * speed + 1 - alpha
                if cfg.get('td_len') is None:
                    loss_value, loss_n, _ = function.Loss(points, speed, normal, beta, gamma, epoch)
                else:
                    loss_value, loss_n, _ = ntrl_loss(function, points, speed, normal, beta, gamma, epoch, cfg['td_len'])
                if cfg.get('los_weight', 0) > 0:
                    j = torch.randint(len(pool_u), (cfg['lb_batch'],), generator=gen, device=dev)
                    tl = network.out(pool_u[j])[0][:, 0]
                    dl, vl = pool_d[j], pool_vis[j]
                    llb = torch.relu(dl - tl).pow(2).mean()
                    llos = ((tl - dl).pow(2) * vl).sum() / vl.sum().clamp(min=1)
                    loss_value = loss_value + cfg.get('lb_weight', 0) * llb + cfg['los_weight'] * llos
                    terms['lb'] = terms.get('lb', 0.0) + float(llb)
                    terms['los'] = terms.get('los', 0.0) + float(llos)
                elif cfg.get('lb_weight', 0) > 0:
                    u = (torch.rand(cfg['lb_batch'], 6, generator=gen, device=dev) - 0.5) * lb_scale
                    tl = network.out(u)[0][:, 0]
                    llb = torch.relu(torch.linalg.vector_norm(u[:, :3] - u[:, 3:], dim=1) - tl).pow(2).mean()
                    loss_value = loss_value + cfg['lb_weight'] * llb
                    terms['lb'] = terms.get('lb', 0.0) + float(llb)
                if active and lam > 0:
                    extra = 0.0
                    if cfg['free_weight'] > 0 and free_pool is not None and (warp.spatial or warp.heading):
                        fq = free_pool[torch.randint(len(free_pool), (cfg['batch'],), generator=gen, device=dev)]
                        ld, gp = warp.logdet(fq)
                        if warp.spatial:
                            lf = torch.relu(ld + math.log(cfg['free_c'])).pow(2).mean()
                            extra = extra + cfg['free_weight'] * lf
                            terms['free'] += float(lf)
                        if warp.heading and cfg['heading_weight'] > 0:
                            lh = torch.relu(gp + math.log(cfg['heading_c'])).pow(2).mean()
                            extra = extra + cfg['heading_weight'] * lh
                            terms['heading'] += float(lh)
                    if cfg.get('normal_weight', 0) > 0 and warp.spatial:
                        # anisotropic near-wall stretch (README_warp section 10): |J n| >= normal_k along the clearance normal at x0
                        ln = warp.normal_stretch(points[:, :3], normal[:, :3])
                        lnn = torch.relu(math.log(cfg['normal_k']) - ln).pow(2).mean()
                        extra = extra + cfg['normal_weight'] * lnn
                        terms['normal'] = terms.get('normal', 0.0) + float(lnn)
                    if (cfg['band_weight'] > 0 and warp.spatial) or (cfg['heading_weight'] > 0 and warp.heading):
                        ld, gp = warp.logdet(points[:, :3])
                        if cfg['band_weight'] > 0 and warp.spatial:
                            lb = torch.relu(math.log(cfg['band_c']) - ld).pow(2).mean()
                            extra = extra + cfg['band_weight'] * lb
                            terms['band'] += float(lb)
                        if cfg['heading_weight'] > 0 and warp.heading:
                            lh = torch.relu(math.log(cfg['heading_c']) - gp).pow(2).mean()
                            extra = extra + cfg['heading_weight'] * lh
                            terms['heading'] += float(lh)
                    if cfg['smooth_weight'] > 0:
                        ls = warp.smoothness()
                        extra = extra + cfg['smooth_weight'] * ls
                        terms['smooth'] += float(ls)
                    loss_value = loss_value + lam * extra
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
            with torch.no_grad():
                k = random.randint(0, len(prev_states) - 1)
                network.load_state_dict(prev_states[k], strict=True)
                optimizer.load_state_dict(prev_opts[k])
        train_loss.append(total_train_loss)
        beta = 1.0 / total_diff
        if epoch % 10 == 0 or rollbacks:
            history['epochs'].append(epoch)
            history['ntrl_loss'].append(total_diff)
            history['warp_terms'].append({k: v / 5 for k, v in terms.items()})
            history['rollbacks'].append(rollbacks)
        if epoch % 100 == 0:
            el = time.time() - t_start
            msg = f'epoch {epoch}: ntrl loss {total_diff:.4e}, rollbacks {rollbacks}, {el / 60:.1f} min ({el / epoch:.3f} s/epoch)'
            if active:
                msg += ', warp terms ' + ', '.join(f'{k} {v / 5:.3g}' for k, v in terms.items() if v) + f' (ramp {lam:.2f})'
            elif terms.get('lb') or terms.get('los'):
                msg += f', lower-bound term {terms.get("lb", 0) / 5:.3g}, line-of-sight term {terms.get("los", 0) / 5:.3g}'
            if (warp.spatial or warp.heading) and free_pool is not None and epoch % 500 == 0 and not cfg.get('z_space'):
                st = warp_summary(warp, free_pool, points[:, :3].detach())
                st['epoch'] = epoch
                history['warp_stats'].append(st)
                msg += f'; log det free {st["logdet_free_median"]:.3f}, near wall {st["logdet_band_median"]:.3f}'
            log(msg)
        if epoch % save_every == 0 or epoch == epochs:
            torch.save({'epoch': epoch, 'model_state_dict': network.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                        'B_state_dict': B, 'train_loss': train_loss, 'val_loss': total_diff, 'warp_config': warp.cfg},
                       out / 'checkpoints' / f'epoch_{epoch:05d}.pt')
    final = out / 'checkpoint.pt'
    torch.save({'epoch': epochs, 'model_state_dict': network.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'B_state_dict': B,
                'train_loss': train_loss, 'val_loss': total_diff, 'warp_config': warp.cfg}, final)
    history['final_loss'] = total_diff
    history['seconds'] = time.time() - t_start
    with open(out / 'history.json', 'w') as f:
        json.dump(history, f)
    return final, history


def ntrl_loss(function, points, Yobs, normal, beta, gamma, epoch, td_len=0.03):
    """ntrl-demo's metric loss (baselines/ntrl-demo/models/metric/model_function_metric.py, Function.Loss) line for
    line, with its temporal-difference step (0.03 in normalized units, i.e. 0.03 E cells: 0.66 cells on the 22 x 22
    maze, 2.6 on the 88 x 88 one) as a parameter. td_len = 0.03 reproduces it exactly (checked: check_ntrl_loss)."""
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
    return loss, loss_n, diff


@torch.no_grad()
def line_of_sight(maze, a, b, chunk=20_000):
    """Conservative line of sight between poses a, b (normalized, (N, 3)): the xy segment misses every wall box
    inflated by the robot's circumscribed radius plus the margin, so the robot keeps clearance >= the margin (speed 1)
    at any heading along the straight C-space segment. Slab test against all boxes, in chunks."""
    from maze import MARGIN_CELLS, ROBOT_L, ROBOT_W
    R = 0.5 * math.hypot(ROBOT_L, ROBOT_W) + MARGIN_CELLS
    boxes = torch.as_tensor(maze.boxes, device=a.device, dtype=torch.float32)
    lo, hi = boxes[:, :2] - R, boxes[:, 2:] + R
    P, Q = a[:, :2] * maze.E + maze.n / 2, b[:, :2] * maze.E + maze.n / 2
    out = []
    for i in range(0, len(P), chunk):
        p, d = P[i:i + chunk, None, :], (Q - P)[i:i + chunk, None, :]
        d = torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
        t0, t1 = (lo[None] - p) / d, (hi[None] - p) / d
        tmin, tmax = torch.minimum(t0, t1).amax(-1), torch.maximum(t0, t1).amin(-1)
        hit = (tmax >= tmin) & (tmax >= 0) & (tmin <= 1)
        out.append(~hit.any(1))
    return torch.cat(out)


@torch.no_grad()
def warp_summary(warp, free_pool, band):
    ldf, gpf = warp.logdet(free_pool[:4096])
    ldb, gpb = warp.logdet(band[:4096])
    return dict(logdet_free_median=float(ldf.median()), logdet_band_median=float(ldb.median()),
                heading_free_median=float(gpf.median()), heading_band_median=float(gpb.median()))
