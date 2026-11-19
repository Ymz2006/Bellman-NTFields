"""ntrl-demo's models/metric training loop (Model.train(), step for step, as jeffrey/cross_warp/trainer.py) with features.EmbedNN.

Same global RNG calls in the same order as the package (seed -> B -> network -> init_weights -> per-epoch reshuffle -> rollback draws):
5 minibatches of 2,000 pairs per epoch, the package's own Function.Loss (eikonal + TD + normal terms, weight exp(-0.5 T), beta = 1 /
previous loss), AdamW lr 5e-4 weight decay 0.5, rollback to one of the last 5 states when the loss rises 20 %. With no extra octaves
this is ntrl-demo itself (the network is the package's class with an unchanged input mapping).

The gate module is attached after the package's initialization. A GridGate / ConstGate has no parameters. A LearnedGate's parameters
are a second AdamW group (lr gate_lr, no weight decay) trained by the same loss, optionally with a budget term
budget_weight * relu(mean mu over uniform C-space poses - budget)^2 (a private generator; the package's RNG sequence is untouched).
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
import features as FE  # noqa: E402


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


def train(data_dir, out, seed, epochs, embed_cfg, log, scene=None, gate_table=None, gate_lr=1e-3, budget=None, budget_weight=0.0, save_every=1000, td_len=None):
    """Returns (final checkpoint path, history). embed_cfg: features.EmbedNN config incl. 'gate' (kind, mu_min, shape, lo, hi, ...);
    gate_table: (K, nx, ny, nt) tensor for a GridGate."""
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
    network = FE.build_network(mn, dev, B, embed_cfg)
    gate = network.embed_gate
    if gate is not None and gate.kind == 'grid':
        assert gate_table is not None, 'a grid gate needs its table'
        with torch.no_grad():
            gate.table.copy_(gate_table.to(dev))
    gate_params = [p for p in (gate.parameters() if gate is not None else []) if p.requires_grad]
    base_params = [p for n, p in network.named_parameters() if not n.startswith('embed_gate.')]
    groups = [{'params': base_params}]
    if gate_params:
        groups.append({'params': gate_params, 'lr': gate_lr, 'weight_decay': 0.0})
    function = mf.Function(str(out), dev, network, 3)
    optimizer = torch.optim.AdamW(groups, lr=1e-3, weight_decay=0.5)
    dataset = db.Database(str(data_dir) + '/')
    loader = mt.FastTensorDataLoader(dataset.data, batch_size=2000, shuffle=True)
    beta = 1.0
    current_diff = 1000.0
    gen = torch.Generator(dev).manual_seed(seed + 7)
    history = dict(epochs=[], ntrl_loss=[], gate_terms=[], rollbacks=[], gate_stats=[])
    current_state = copy.deepcopy(network.state_dict())
    current_opt = copy.deepcopy(optimizer.state_dict())
    prev_states, prev_opts = [], []
    train_loss = []
    t_start = time.time()
    use_budget = bool(gate_params) and budget is not None and budget_weight > 0
    if use_budget:
        assert scene is not None
    log(f'network: input dim {network.in_dim}, octaves {network.octaves}, rows {network.rows} (base {network.base_rows}), gate {None if gate is None else gate.kind}, '
        f'parameters {sum(p.numel() for p in base_params)} + gate {sum(p.numel() for p in gate_params)}')
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
        prev_diff = current_diff
        rollbacks = 0
        while True:
            total_train_loss, total_diff = 0.0, 0.0
            terms = dict(budget=0.0, mu_band=0.0, mu_uniform=0.0)
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
                if td_len is None:
                    loss_value, loss_n, _ = function.Loss(points, speed, normal, beta, gamma, epoch)
                else:
                    loss_value, loss_n, _ = ntrl_loss(function, points, speed, normal, beta, gamma, epoch, td_len)
                if gate_params:
                    with torch.no_grad():
                        terms['mu_band'] += float(network.gates(points[:, :3]).mean())
                    if use_budget:
                        u = scene.uniform(2000, gen)
                        mu_u = network.gates(u)
                        terms['mu_uniform'] += float(mu_u.mean())
                        lb = torch.relu(mu_u.mean() - budget).pow(2)
                        loss_value = loss_value + budget_weight * lb
                        terms['budget'] += float(lb)
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
            history['gate_terms'].append({k: v / 5 for k, v in terms.items()})
            history['rollbacks'].append(rollbacks)
        if epoch % 100 == 0:
            el = time.time() - t_start
            msg = f'epoch {epoch}: ntrl loss {total_diff:.4e}, rollbacks {rollbacks}, {el / 60:.1f} min ({el / epoch:.3f} s/epoch)'
            if gate_params:
                msg += f', gate mean mu band {terms["mu_band"] / 5:.3f}' + (f', uniform {terms["mu_uniform"] / 5:.3f}, budget term {terms["budget"] / 5:.3g}' if use_budget else '')
            log(msg)
        if epoch % save_every == 0 or epoch == epochs:
            torch.save({'epoch': epoch, 'model_state_dict': network.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                        'B_state_dict': B, 'train_loss': train_loss, 'val_loss': total_diff, 'embed_config': network.cfg},
                       out / 'checkpoints' / f'epoch_{epoch:05d}.pt')
    final = out / 'checkpoint.pt'
    torch.save({'epoch': epochs, 'model_state_dict': network.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'B_state_dict': B,
                'train_loss': train_loss, 'val_loss': total_diff, 'embed_config': network.cfg}, final)
    history['final_loss'] = total_diff
    history['seconds'] = time.time() - t_start
    with open(out / 'history.json', 'w') as f:
        json.dump(history, f)
    return final, history


def ntrl_loss(function, points, Yobs, normal, beta, gamma, epoch, td_len=0.03):
    """ntrl-demo's metric loss (models/metric/model_function_metric.py, Function.Loss) line for line, with its temporal-difference step
    (0.03 normalized units = 0.03 E cells, 0.66 cells on a 22-cell maze, longer than its 0.1-cell walls) as a parameter; td_len = 0.03 is
    the package's loss exactly (jeffrey/cross_warp/trainer.py checked the equality including gradients)."""
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
