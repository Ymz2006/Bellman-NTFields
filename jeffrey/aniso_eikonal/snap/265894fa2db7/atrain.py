"""ntrl-demo's models/metric training loop with the eikonal term only, on the maze or in a fixed warp's coordinates.

The loop is baselines/ntrl-demo/models/metric/model_train_metric.py Model.train() as jeffrey/cross_warp/trainer.py copies
it (same global RNG calls in the same order: seed -> B ~ N(0, 1) truncated at 2 -> NN -> init_weights -> the loader's
per-epoch reshuffle -> random.randint on rollbacks; 5 minibatches of 2,000 pairs per epoch; AdamW lr 5e-4, weight decay
0.5; beta = 1 / the previous epoch's loss; rollback to a random one of the last 5 states when the loss rises by 20 %).
The loss is aniso.eikonal_loss: the package's eikonal term with its exp(-T / 2) weights and nothing else (the package's
temporal-difference and normal terms are left out: the base of this study).

Modes
  id      no warp: the network on the maze's coordinates x, |grad_x T| = 1 / S (ntrl-demo without TD / normal terms).
  ziso    'trained on the warped maze': poses mapped once to z = phi(x), the network on z, |grad_z H| = 1 / S. The field
          is the travel time of the warped maze (z-units).
  zaniso  the same inputs, the inner product: grad_z H^T (J J^T / c^2) grad_z H = 1 / S^2 with J = d phi / d x at each
          pose, precomputed (the warp is fixed). The field is c times the travel time of the maze.
  zsplit  zaniso with the split head (aniso.split_class): H' = d_net + c |u(z0) - u(z1)|, the equation asked of the sum,
          the pair weights taken on d_net.
  zboth   the split head with both equations (aniso.both_loss): the warped maze's plain equation on d_net and the
          anisotropic equation on the sum.
lb_weight > 0 adds the straight-line lower bound on the maze as a relative hinge (see the loop).
The checkpoint has cross_warp/trainer.py's format (warp.load_field reads it): the warp inside the network, so
T(x0, x1) = H(phi(x0), phi(x1)) for callers that pass x, and H itself with net.bypass = True.
"""
import copy
import json
import random
import time
from pathlib import Path
import numpy as np
import torch

import aniso as AN
import trainer as TR
import warp as W


def load_pairs(data_dir):
    d = Path(data_dir)
    return np.load(d / 'sampled_points.npy'), np.load(d / 'speed.npy')


def train(data_dir, out, seed, epochs, warp, mode, c, log, tau_weight=0.5, save_every=1000, check=True, split_m=None, lb_weight=0.0,
          head_scale=1.0, emb_scale=1.0, pair_weight='tau'):
    """warp: a fixed warp.Warp on the device (identity for mode 'id'). Returns (checkpoint path, history)."""
    assert mode in ('id', 'ziso', 'zaniso', 'zsplit', 'zboth')
    mt, mf, mn, db = TR.ntrl_modules()
    dev = 'cuda:0'
    out = Path(out)
    (out / 'checkpoints').mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    B = torch.normal(0, 1, size=(128, 3))
    torch.nn.init.trunc_normal_(B, mean=0.0, std=1, a=-2.0, b=2.0)
    network = AN.split_class(W.warped_class(mn))(dev, 3, B)
    network.apply(network.init_weights)
    network.to(dev)
    base_params = list(network.parameters())
    assert not any(p.requires_grad for p in warp.parameters()), 'the warp must be fixed'
    network.warp = warp
    optimizer = torch.optim.AdamW([{'params': base_params}], lr=1e-3, weight_decay=0.5)
    pts, spd = load_pairs(data_dir)
    x = torch.tensor(pts, device=dev)
    info = dict(mode=mode, c=float(c), tau_weight=tau_weight, pairs=len(x), head_scale=float(head_scale), emb_scale=float(emb_scale), pair_weight=pair_weight)
    ones6 = lambda n: torch.cat([torch.ones(n, 3, device=dev), torch.zeros(n, 3, device=dev)], 1)       # the identity as a full symmetric matrix
    with torch.no_grad():
        if mode == 'id':
            assert warp.identity(), "mode 'id' takes no warp"
            p = x
            M = torch.cat([ones6(len(x)), ones6(len(x))], 1)
        else:
            z0, _ = W.to_warped(warp, x[:, :3])
            z1, _ = W.to_warped(warp, x[:, 3:])
            p = torch.cat([z0, z1], 1)
            if mode in ('zaniso', 'zsplit', 'zboth'):
                # the general form: J J^T from the warp's autograd Jacobian (for this warp it is diagonal; nothing below uses that)
                M = torch.cat([torch.cat([AN.metric_full(warp, x[i:i + 65536, :3]), AN.metric_full(warp, x[i:i + 65536, 3:])], 1) for i in range(0, len(x), 65536)])
                offd = float(M[:, [3, 4, 5, 9, 10, 11]].abs().max() / M[:, [0, 1, 2, 6, 7, 8]].abs().max())
                info['metric_offdiagonal_max_rel'] = offd
            else:
                M = torch.cat([ones6(len(x)), ones6(len(x))], 1)
            network.bypass = True
            warp.cfg['z_space'] = True
            if mode in ('zsplit', 'zboth'):
                assert mode == 'zsplit' or abs(c - split_m) < 1e-9, "zboth: the network's unit must be the warp's (--unit-cells quad)"
                network.set_split(c, split_m)
                info.update(split=float(c), m_ref=float(split_m))
        info.update(input_range=[p[:, :3].amin(0).tolist(), p[:, :3].amax(0).tolist()],
                    metric_over_c2=dict(min=(M[:, :3].amin(0) / c ** 2).tolist(), max=(M[:, :3].amax(0) / c ** 2).tolist()))
        info['c_from_warp'] = float(torch.sqrt(M[:, :2].amax())) if mode != 'id' else 1.0        # the warp's largest stretch along x or y over the training poses
    data = torch.cat([p, torch.tensor(spd, device=dev), M, x], 1).cpu()          # 0-5 inputs, 6-7 speeds, 8-13 / 14-19 the metric at the two ends, 20-25 the poses on the maze
    lb_gen = torch.Generator(dev).manual_seed(seed + 23)
    info['lb_weight'] = lb_weight
    log(f'training {mode}: {len(data)} pairs, c = {c:.4f}, input range {info["input_range"]}, M / c^2 range {info["metric_over_c2"]}')
    if check:
        function = mf.Function(str(out), dev, network, 3)         # (the checks run on the network as ntrl-demo builds it: scales are set after them)
        j = torch.arange(2000)
        sp = data[j, 6:8].to(dev)
        sp = 1.025 * (sp * sp * (2 - sp) * (2 - sp)) + 1 - 1.025
        keep_split, network.split = network.split, 0.0          # the package's term is the plain network's
        info['base_loss_check'] = AN.check_base_loss(function, network, data[j, :6].to(dev), sp, torch.zeros(2000, 6, device=dev))
        network.split = keep_split
        log(f'eikonal term with M = 1 against the package\'s (untrained network, 2,000 pairs): {info["base_loss_check"]}')
        if mode != 'id':
            saved = network.bypass
            info['chain_check'] = AN.check_chain(network, warp, _box(warp, x))
            network.bypass = saved
            log(f'chain rule on the untrained network: {info["chain_check"]}')
    network.set_scales(head_scale, emb_scale)
    loader = mt.FastTensorDataLoader(data, batch_size=2000, shuffle=True)
    beta, current_diff = 1.0, 1000.0
    history = dict(epochs=[], loss=[], rollbacks=[])
    current_state = copy.deepcopy(network.state_dict())
    current_opt = copy.deepcopy(optimizer.state_dict())
    prev_states, prev_opts, train_loss = [], [], []
    t_start = time.time()
    alpha = 1.025
    stopped = 0
    for epoch in range(1, epochs + 1):
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
            total_train_loss, total_diff, lb_sum = 0.0, 0.0, 0.0
            ii = 0
            for i, whole in enumerate(loader, 0):
                if ii > 4:
                    break
                ii += 1
                batch = whole[0].to(dev)
                speed = batch[:, 6:8]
                speed = speed * speed * (2 - speed) * (2 - speed)
                speed = alpha * speed + 1 - alpha
                loss_value, loss_n = (AN.both_loss if mode == 'zboth' else AN.eikonal_loss)(network, batch[:, :6], speed, batch[:, 8:14], batch[:, 14:20], beta, c, tau_weight, **({} if mode == 'zboth' else dict(pair_weight=pair_weight)))
                if lb_weight > 0:
                    # straight-line lower bound on the maze (label-free: the speed is at most 1): H'(a, b) >= c |x_a - x_b| for
                    # the batch's starts paired with randomly permuted ends; relative hinge, so the weight has no unit
                    perm = torch.randperm(len(batch), generator=lb_gen, device=dev)
                    tl = network.out(torch.cat([batch[:, :3], batch[perm, 3:6]], 1))[0][:, 0]
                    dl = c * torch.linalg.vector_norm(batch[:, 20:23] - batch[perm, 23:26], dim=1)
                    llb = torch.relu(1 - tl / dl.clamp(min=1e-9)).pow(2).mean()
                    loss_value = loss_value + lb_weight * llb
                    lb_sum += float(llb)
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
            if rollbacks > 200:                  # the package would roll back for ever: stop on the epoch's starting state
                stopped = epoch
                break
            with torch.no_grad():
                k = random.randint(0, len(prev_states) - 1)
                network.load_state_dict(prev_states[k], strict=True)
                optimizer.load_state_dict(prev_opts[k])
        if stopped:
            network.load_state_dict(current_state, strict=True)
            total_diff = prev_diff
            log(f'epoch {epoch}: more than 200 rollbacks in a row; training stopped on the state the epoch started from (loss {prev_diff:.4e})')
            break
        train_loss.append(total_train_loss)
        beta = 1.0 / total_diff
        if epoch % 10 == 0 or rollbacks:
            history['epochs'].append(epoch)
            history['loss'].append(total_diff)
            history['rollbacks'].append(rollbacks)
        if epoch % 500 == 0:
            el = time.time() - t_start
            log(f'epoch {epoch}: loss {total_diff:.4e}, rollbacks {rollbacks}, {el / 60:.1f} min ({el / epoch:.3f} s/epoch)' + (f', lower-bound hinge {lb_sum / 5:.3e}' if lb_weight > 0 else ''))
        if epoch % save_every == 0 or epoch == epochs:
            torch.save(_payload(network, optimizer, B, epoch, train_loss, total_diff, warp, info), out / 'checkpoints' / f'epoch_{epoch:05d}.pt')
    final = out / 'checkpoint.pt'
    info['stopped_at_epoch'] = stopped
    torch.save(_payload(network, optimizer, B, epochs, train_loss, total_diff, warp, info), final)
    history.update(final_loss=total_diff, seconds=time.time() - t_start, info=info, stopped_at_epoch=stopped)
    with open(out / 'history.json', 'w') as f:
        json.dump(history, f)
    return final, history


def _payload(network, optimizer, B, epoch, train_loss, loss, warp, info):
    return {'epoch': epoch, 'model_state_dict': network.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'B_state_dict': B,
            'train_loss': train_loss, 'val_loss': loss, 'warp_config': warp.cfg, 'aniso': info}


class _box:
    """The C-space box of the training poses (for the checks' uniform samples), without building the maze again."""

    def __init__(self, warp, x):
        self.device = x.device
        self.lo = torch.tensor([-0.5, -0.5, float(x[:, 2].min())], device=x.device)
        self.hi = torch.tensor([0.5, 0.5, float(x[:, 2].max())], device=x.device)

    def uniform(self, count, gen):
        return self.lo + (self.hi - self.lo) * torch.rand(count, 3, generator=gen, device=self.device)
