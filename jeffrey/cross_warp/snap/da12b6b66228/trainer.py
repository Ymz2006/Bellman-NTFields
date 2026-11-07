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
* straight-line lower bound (any field, warped or not; lb_weight > 0): the speed is at most 1, so the travel time is
  at least the straight-line distance in the C-space metric, T(a, b) >= |a - b|. lb_weight * mean relu(|a - b| -
  T(a, b))^2 over lb_batch pose pairs drawn uniformly in the C-space box per minibatch (no labels; the warp-term
  generator). Added from epoch 1 with full weight, outside the warp ramp.
Their weights ramp in linearly over --ramp epochs from --warp-start; after --warp-stop the warp is frozen (the field
finishes training on fixed coordinates).
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
                loss_value, loss_n, _ = function.Loss(points, speed, normal, beta, gamma, epoch)
                if cfg.get('lb_weight', 0) > 0:
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
            elif terms.get('lb'):
                msg += f', lower-bound term {terms["lb"] / 5:.3g}'
            if (warp.spatial or warp.heading) and free_pool is not None and epoch % 500 == 0:
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


@torch.no_grad()
def warp_summary(warp, free_pool, band):
    ldf, gpf = warp.logdet(free_pool[:4096])
    ldb, gpb = warp.logdet(band[:4096])
    return dict(logdet_free_median=float(ldf.median()), logdet_band_median=float(ldb.median()),
                heading_free_median=float(gpf.median()), heading_band_median=float(gpb.median()))
