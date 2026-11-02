"""ntrl-demo's MPPI on the parking scene, the exact path scorer, and the straight-line baseline.

mppi() is ntrl-demo tests/arm_plan_stat.py MPPI with dim 3, batched over queries (each query evolves as in the
single-query loop): per step 'sample_num' rollouts of 'horizon' steps, dP = step N(0, 1) (first step) + step N(0, 1)
(every step) + 2 x previous update, each step clipped to norm 'step'; cost 10 T(first) + T(last); weights
softmax(-temperature cost); the update is the weighted first step; stop within 'tol' of the goal. Defaults are that
file's settings (50 samples, horizon 5, 200 steps, step 0.015, temperature 50, tol 0.01). The goal is appended to
every path and the scorer judges every segment, the last one included.

score(): a path succeeds if it converged and every configuration on it, interpolated at <= 0.001 per coordinate,
has non-negative exact clearance. The first contact's wall box is recorded.
"""
import importlib.util
import math
from pathlib import Path
import numpy as np
import torch

NTRL = Path(__file__).resolve().parents[2] / 'baselines/ntrl-demo'
DEFAULT = dict(steps=200, sample_num=50, horizon=5, step=0.015, temperature=50.0, tol=0.01)
WIDE = dict(steps=400, sample_num=200, horizon=10, step=0.015, temperature=50.0, tol=0.01)
FINE = dict(steps=600, sample_num=50, horizon=5, step=0.005, temperature=50.0, tol=0.01)   # same MPPI, 3x smaller steps (tight bays)
SETTINGS = dict(default=DEFAULT, wide=WIDE, fine=FINE)


def network_class(variant):
    path = NTRL / f'models/{variant}/model_network_metric.py'
    spec = importlib.util.spec_from_file_location(f'ntrl_demo_{variant}_network', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NN


def load_field(checkpoint, variant='metric', device='cuda:0'):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    net = network_class(variant)(device, 3, payload['B_state_dict'])
    net.load_state_dict(payload['model_state_dict'], strict=True)
    net.to(torch.device(device)).float().eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net


def field_cost(net, chunk=200_000):
    """T(x0, x1) for (P, 6) pairs, the network's own NN.out."""
    def cost(pairs, ids=None):
        out = []
        for i in range(0, len(pairs), chunk):
            out.append(net.out(pairs[i:i + chunk])[0][:, 0])
        return torch.cat(out)
    return cost


def field_grad(net, pairs, chunk=100_000):
    """T and dT/dx0 (gradient with respect to the first configuration)."""
    ts, gs = [], []
    for i in range(0, len(pairs), chunk):
        with torch.enable_grad():
            tau, _, xp = net.out(pairs[i:i + chunk])
            g, = torch.autograd.grad(tau.sum(), xp)
        ts.append(tau.detach()[:, 0])
        gs.append(g[:, :3].detach())
    return torch.cat(ts), torch.cat(gs)


@torch.no_grad()
def mppi(cost, start, goal, seed, steps=200, sample_num=50, horizon=5, step=0.015, temperature=50.0, tol=0.01):
    """Returns positions (steps + 1, N, 3), the step index each query stopped at, and the converged flags."""
    torch.manual_seed(seed)
    n, dev = len(start), start.device
    xp = torch.cat([start, goal], 1).clone()
    prior = torch.zeros(n, 3, device=dev)
    active = torch.ones(n, dtype=torch.bool, device=dev)
    stop = torch.full((n,), steps, dtype=torch.long, device=dev)
    trace = [xp[:, :3].clone()]
    for it in range(steps):
        ids = torch.nonzero(active).squeeze(1)
        if not len(ids):
            break
        cur = xp[ids]
        tmp = cur[:, None, None, :].repeat(1, sample_num, horizon, 1)
        dp = step * torch.randn(len(ids), sample_num, 1, 3, device=dev) + step * torch.randn(len(ids), sample_num, horizon, 3, device=dev)
        dp = dp + 2 * prior[ids][:, None, None, :]
        dp = dp / (torch.linalg.vector_norm(dp, dim=3, keepdim=True).clamp(min=step) / step)
        tmp[..., :3] = tmp[..., :3] + torch.cumsum(dp, dim=2)
        c = cost(tmp[:, :, [0, -1], :].reshape(-1, 6), ids).reshape(len(ids), sample_num, 2)
        c = 10 * c[..., 0] + c[..., 1]
        weight = torch.softmax(-temperature * c, dim=1)
        update = torch.einsum('bs,bsd->bd', weight, dp[:, :, 0, :])
        prior[ids] = update
        xp[ids, :3] = cur[:, :3] + update
        trace.append(xp[:, :3].clone())
        done = torch.linalg.vector_norm(xp[ids, 3:] - xp[ids, :3], dim=1) < tol
        stop[ids[done]] = it + 1
        active[ids[done]] = False
    trace = torch.stack(trace)
    converged = stop < steps
    converged |= torch.linalg.vector_norm(xp[:, 3:] - xp[:, :3], dim=1) < tol
    stop = stop.clamp(max=len(trace) - 1)
    return trace, stop, converged


def paths_from_trace(trace, stop, start, goal):
    """Per query: start, MPPI positions up to its stop step, goal."""
    out = []
    for i in range(trace.shape[1]):
        p = trace[:int(stop[i]) + 1, i]
        out.append(torch.cat([start[i:i + 1], p[1:], goal[i:i + 1]]))
    return out


def densify(path, res=0.001):
    """Linear interpolation with at most res per coordinate between consecutive configurations."""
    seg = path[1:] - path[:-1]
    k = torch.clamp(torch.ceil(seg.abs().amax(1) / res), min=1).long()
    j = torch.repeat_interleave(torch.arange(len(seg), device=path.device), k)
    first = torch.cumsum(k, 0) - k
    t = (torch.arange(int(k.sum()), device=path.device) - first[j] + 1).to(path.dtype) / k[j].to(path.dtype)
    return torch.cat([path[:1], path[j] + t[:, None] * seg[j]])


def score(scene, paths, converged, speed_fn=None):
    """Per path: success, collision, first-contact box, min clearance, C-space length, travel time under speed_fn."""
    rows = []
    for i, path in enumerate(paths):
        dense = densify(path)
        d, idx = scene.clearance(dense)
        bad = torch.nonzero(d < 0).squeeze(1)
        seg = torch.linalg.vector_norm(dense[1:] - dense[:-1], dim=1)
        row = dict(converged=bool(converged[i]), collision=bool(len(bad)), min_clearance=float(d.min()),
                   length=float(seg.sum()), waypoints=int(len(path)))
        if len(bad):
            row['first_contact_box'] = int(idx[bad[0]])
            row['first_contact_fraction'] = float(bad[0]) / max(len(dense) - 1, 1)
            row['divider_contact'] = bool((idx[bad] == 4).any())
        if speed_fn is not None:
            s = speed_fn(d)
            row['time'] = float((seg * 0.5 * (1 / s[1:] + 1 / s[:-1])).sum())
        row['success'] = row['converged'] and not row['collision']
        rows.append(row)
    return rows


def straight(start, goal):
    return [torch.stack([start[i], goal[i]]) for i in range(len(start))]
