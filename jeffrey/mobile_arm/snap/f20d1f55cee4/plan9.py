"""ntrl-demo's MPPI and the exact scorer in the 9-D C-space (jeffrey/maze_scale/plan.py with dim 9).

mppi() is maze_scale's plan.mppi (ntrl-demo tests/arm_plan_stat.py MPPI: 50 samples, horizon 5, dP = step N(0, 1) (first step) + step
N(0, 1) (every step) + 2 x previous update, every step clipped to norm 'step'; cost 10 T(first) + T(last); softmax(-temperature cost);
weighted first step; stop within tol) with every coordinate of the 9-D configuration; the angles (heading and the six joints) are clamped
to the C-space box, x and y are not (the house's outside is an obstacle). settings(): maze_scale's (step, tol x pitch / pitch_ref,
temperature / it). The goal is appended to every path, as maze_scale's.

score(): a path succeeds if it converged and every configuration on it, interpolated at <= 0.005 cells per coordinate, has non-negative
clearance (walls for base and arm, floor, base box, self). The first contact is attributed to the part that is negative there.
"""
import importlib.util
from pathlib import Path
import numpy as np
import torch
from scene9 import K_INT

NTRL = Path(__file__).resolve().parents[2] / 'baselines/ntrl-demo'
BASE = dict(sample_num=50, horizon=5, step=0.015, temperature=50.0, tol=0.01, min_steps=200, budget_factor=3.0)
PARTS = ('rect', 'arm', 'floor', 'base', 'self')


def settings(scene, **override):
    s = scene.planner_scale()
    out = dict(BASE, step=BASE['step'] * s, temperature=BASE['temperature'] / s, tol=BASE['tol'] * s)
    out.update(override)
    return out


def network_class(variant='metric'):
    path = NTRL / f'models/{variant}/model_network_metric.py'
    spec = importlib.util.spec_from_file_location(f'ntrl_demo_{variant}_network', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NN


def load_field(checkpoint, dim=9, device='cuda:0'):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    net = network_class('metric')(device, dim, payload['B_state_dict'])
    net.load_state_dict(payload['model_state_dict'], strict=True)
    net.to(torch.device(device)).float().eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net


def field_cost(net, chunk=200_000):
    def cost(pairs, ids=None):
        out = []
        for i in range(0, len(pairs), chunk):
            with torch.no_grad():
                out.append(net.out(pairs[i:i + chunk])[0][:, 0].detach())
        return torch.cat(out) if out else pairs.new_zeros(0)
    return cost


def field_grad(net, pairs, chunk=100_000):
    """T and dT/dx0."""
    D = pairs.shape[1] // 2
    ts, gs = [], []
    for i in range(0, len(pairs), chunk):
        with torch.enable_grad():
            tau, _, xp = net.out(pairs[i:i + chunk])
            g, = torch.autograd.grad(tau.sum(), xp)
        ts.append(tau.detach()[:, 0])
        gs.append(g[:, :D].detach())
    return torch.cat(ts), torch.cat(gs)


@torch.no_grad()
def mppi(cost, start, goal, seed, budget, scene, sample_num=50, horizon=5, step=0.015, temperature=50.0, tol=0.01, keep_trace=None, **_):
    """budget (N,) step limits. Returns per query the positions (start ... last MPPI position, goal appended), the stop step, converged."""
    torch.manual_seed(seed)
    n, dev = len(start), start.device
    D = start.shape[1]
    lo, hi, mask = scene.lo, scene.hi, scene.clamp_mask
    xp = torch.cat([start, goal], 1).clone()
    prior = torch.zeros(n, D, device=dev)
    active = torch.ones(n, dtype=torch.bool, device=dev)
    budget = budget.to(dev).long()
    stop = budget.clone()
    converged = torch.zeros(n, dtype=torch.bool, device=dev)
    keep = torch.arange(n, device=dev) if keep_trace is None else torch.as_tensor(keep_trace, device=dev).long()
    trace = [xp[keep, :D].clone()]
    for it in range(int(budget.max())):
        active = active & (budget > it)
        ids = torch.nonzero(active).squeeze(1)
        if not len(ids):
            break
        cur = xp[ids]
        tmp = cur[:, None, None, :].repeat(1, sample_num, horizon, 1)
        dp = step * torch.randn(len(ids), sample_num, 1, D, device=dev) + step * torch.randn(len(ids), sample_num, horizon, D, device=dev)
        dp = dp + 2 * prior[ids][:, None, None, :]
        dp = dp / (torch.linalg.vector_norm(dp, dim=3, keepdim=True).clamp(min=step) / step)
        roll = tmp[..., :D] + torch.cumsum(dp, dim=2)
        tmp[..., :D] = torch.where(mask, torch.maximum(torch.minimum(roll, hi), lo), roll)
        c = cost(tmp[:, :, [0, -1], :].reshape(-1, 2 * D), ids).reshape(len(ids), sample_num, 2)
        c = 10 * c[..., 0] + c[..., 1]
        weight = torch.softmax(-temperature * c, dim=1)
        update = torch.einsum('bs,bsd->bd', weight, dp[:, :, 0, :])
        prior[ids] = update
        new = cur[:, :D] + update
        new = torch.where(mask, torch.maximum(torch.minimum(new, hi), lo), new)
        xp[ids, :D] = new
        trace.append(xp[keep, :D].clone())
        done = torch.linalg.vector_norm(xp[ids, D:] - xp[ids, :D], dim=1) < tol
        stop[ids[done]] = it + 1
        converged[ids[done]] = True
        active[ids[done]] = False
    trace = torch.stack(trace)
    pos = {int(k): i for i, k in enumerate(keep.tolist())}
    paths = {}
    for k, i in pos.items():
        paths[k] = torch.cat([start[k:k + 1], trace[1:int(stop[k]) + 1, i], goal[k:k + 1]])
    return paths, stop, converged


def densify(path, res):
    seg = path[1:] - path[:-1]
    k = torch.clamp(torch.ceil(seg.abs().amax(1) / res), min=1).long()
    j = torch.repeat_interleave(torch.arange(len(seg), device=path.device), k)
    first = torch.cumsum(k, 0) - k
    t = (torch.arange(int(k.sum()), device=path.device) - first[j] + 1).to(path.dtype) / k[j].to(path.dtype)
    return torch.cat([path[:1], path[j] + t[:, None] * seg[j]])


def path_time(scene, dense, d=None):
    d = scene.clearance(dense) if d is None else d
    s = scene.train_speed(d)
    seg = torch.linalg.vector_norm(dense[1:] - dense[:-1], dim=1)
    return float((seg * 0.5 * (1 / s[1:] + 1 / s[:-1])).sum())


def score(scene, paths, converged, res_cells=0.005):
    """Per path: success, collision, the part touched first, min clearance (cells), xy length (cells), C-space length, travel time."""
    rows = []
    for i, path in enumerate(paths):
        dense = densify(path, res_cells * scene.pitch)
        d = scene.clearance(dense)
        bad = torch.nonzero(d < 0).squeeze(1)
        seg = torch.linalg.vector_norm(dense[1:] - dense[:-1], dim=1)
        s = scene.train_speed(d)
        row = dict(converged=bool(converged[i]), collision=bool(len(bad)), min_clearance=float(d.min()),
                   length_cells=float(torch.linalg.vector_norm(dense[1:, :2] - dense[:-1, :2], dim=1).sum() / scene.pitch),
                   clen=float(seg.sum()), time=float((seg * 0.5 * (1 / s[1:] + 1 / s[:-1])).sum()), waypoints=int(len(path)))
        if len(bad):
            k = int(bad[0])
            row['first_contact_fraction'] = k / max(len(dense) - 1, 1)
            pt = scene.parts(dense[k:k + 1])
            row['contact_part'] = min(PARTS, key=lambda p: float(pt[p][0]) * (K_INT if p in ('floor', 'base', 'self') else 1.0))
        row['success'] = row['converged'] and not row['collision']
        rows.append(row)
    return rows


def straight(start, goal):
    return [torch.stack([start[i], goal[i]]) for i in range(len(start))]
