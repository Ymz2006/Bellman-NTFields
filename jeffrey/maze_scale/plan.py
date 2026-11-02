"""ntrl-demo's MPPI on a maze, the exact path scorer, the straight-line control and the ntrl-demo field loader.

mppi() is ntrl-demo tests/arm_plan_stat.py MPPI with dim 3 (as jeffrey/parking/plan.py), batched over queries: per step
'sample_num' rollouts of 'horizon' steps, dP = step N(0, 1) (first step) + step N(0, 1) (every step) + 2 x previous
update, each step clipped to norm 'step'; cost 10 T(first) + T(last); weights softmax(-temperature cost); the update
is the weighted first step; stop within 'tol' of the goal. The file's settings (50 samples, horizon 5, step 0.015,
temperature 50, tol 0.01) hold for the REF_N = 4 maze; on a maze with pitch p the step and tol are multiplied by
s = p / p_ref and the temperature divided by it (cost differences between rollouts scale with the step), so the planner
is the same in cell units at every size. The step budget is per query: max(200, 3 x the true path's C-space length /
step) (ntrl-demo: 200 steps, i.e. 3.0 normalized units, about 12 cells at n = 4). phi is clamped to the C-space box
(theta in [-pi/2, pi/2]); x, y are not (leaving them means hitting the outer wall). The goal is appended to every path.

score(): a path succeeds if it converged and every configuration on it, interpolated at <= 0.005 cells per coordinate,
has non-negative exact clearance.
"""
import importlib.util
from pathlib import Path
import numpy as np
import torch

NTRL = Path(__file__).resolve().parents[2] / 'baselines/ntrl-demo'
BASE = dict(sample_num=50, horizon=5, step=0.015, temperature=50.0, tol=0.01, min_steps=200, budget_factor=3.0)


def settings(maze, **override):
    s = maze.planner_scale()
    out = dict(BASE, step=BASE['step'] * s, temperature=BASE['temperature'] / s, tol=BASE['tol'] * s)
    out.update(override)
    return out


def network_class(variant='metric'):
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
    """T(x0, x1) for (P, 6) pairs, the network's own NN.out (no gradient graph)."""
    def cost(pairs, ids=None):
        out = []
        for i in range(0, len(pairs), chunk):
            with torch.no_grad():
                out.append(net.out(pairs[i:i + chunk])[0][:, 0].detach())
        return torch.cat(out) if out else pairs.new_zeros(0)
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
def mppi(cost, start, goal, seed, budget, lo, hi, sample_num=50, horizon=5, step=0.015, temperature=50.0, tol=0.01, keep_trace=None, **_):
    """budget: (N,) step limits. Returns per query the positions (list of (L, 3) tensors, start ... last MPPI position)
    for the rows in keep_trace (all if None), the step index each query stopped at, and the converged flags."""
    torch.manual_seed(seed)
    n, dev = len(start), start.device
    xp = torch.cat([start, goal], 1).clone()
    prior = torch.zeros(n, 3, device=dev)
    active = torch.ones(n, dtype=torch.bool, device=dev)
    budget = budget.to(dev).long()
    stop = budget.clone()
    converged = torch.zeros(n, dtype=torch.bool, device=dev)
    keep = torch.arange(n, device=dev) if keep_trace is None else torch.as_tensor(keep_trace, device=dev).long()
    trace = [xp[keep, :3].clone()]
    steps = int(budget.max())
    for it in range(steps):
        active = active & (budget > it)
        ids = torch.nonzero(active).squeeze(1)
        if not len(ids):
            break
        cur = xp[ids]
        tmp = cur[:, None, None, :].repeat(1, sample_num, horizon, 1)
        dp = step * torch.randn(len(ids), sample_num, 1, 3, device=dev) + step * torch.randn(len(ids), sample_num, horizon, 3, device=dev)
        dp = dp + 2 * prior[ids][:, None, None, :]
        dp = dp / (torch.linalg.vector_norm(dp, dim=3, keepdim=True).clamp(min=step) / step)
        tmp[..., :3] = tmp[..., :3] + torch.cumsum(dp, dim=2)
        tmp[..., 2] = tmp[..., 2].clamp(float(lo[2]), float(hi[2]))
        c = cost(tmp[:, :, [0, -1], :].reshape(-1, 6), ids).reshape(len(ids), sample_num, 2)
        c = 10 * c[..., 0] + c[..., 1]
        weight = torch.softmax(-temperature * c, dim=1)
        update = torch.einsum('bs,bsd->bd', weight, dp[:, :, 0, :])
        prior[ids] = update
        new = cur[:, :3] + update
        new[:, 2] = new[:, 2].clamp(float(lo[2]), float(hi[2]))
        xp[ids, :3] = new
        trace.append(xp[keep, :3].clone())
        done = torch.linalg.vector_norm(xp[ids, 3:] - xp[ids, :3], dim=1) < tol
        stop[ids[done]] = it + 1
        converged[ids[done]] = True
        active[ids[done]] = False
    trace = torch.stack(trace)                                          # (L, K, 3)
    pos = {int(k): i for i, k in enumerate(keep.tolist())}
    paths = {}
    for k, i in pos.items():
        paths[k] = torch.cat([start[k:k + 1], trace[1:int(stop[k]) + 1, i], goal[k:k + 1]])
    return paths, stop, converged


def densify(path, res):
    """Linear interpolation with at most res per coordinate between consecutive configurations."""
    seg = path[1:] - path[:-1]
    k = torch.clamp(torch.ceil(seg.abs().amax(1) / res), min=1).long()
    j = torch.repeat_interleave(torch.arange(len(seg), device=path.device), k)
    first = torch.cumsum(k, 0) - k
    t = (torch.arange(int(k.sum()), device=path.device) - first[j] + 1).to(path.dtype) / k[j].to(path.dtype)
    return torch.cat([path[:1], path[j] + t[:, None] * seg[j]])


def score(maze, paths, converged, res_cells=0.005):
    """Per path: success, collision, first-contact fraction, min clearance (cells), xy length (cells), travel time."""
    rows = []
    for i, path in enumerate(paths):
        dense = densify(path, res_cells * maze.pitch)
        d = maze.clearance(dense)
        bad = torch.nonzero(d < 0).squeeze(1)
        seg = torch.linalg.vector_norm(dense[1:] - dense[:-1], dim=1)
        s = maze.train_speed(d)
        row = dict(converged=bool(converged[i]), collision=bool(len(bad)), min_clearance=float(d.min()),
                   length_cells=float(torch.linalg.vector_norm(dense[1:, :2] - dense[:-1, :2], dim=1).sum() / maze.pitch),
                   time=float((seg * 0.5 * (1 / s[1:] + 1 / s[:-1])).sum()), waypoints=int(len(path)))
        if len(bad):
            row['first_contact_fraction'] = float(bad[0]) / max(len(dense) - 1, 1)
        row['success'] = row['converged'] and not row['collision']
        rows.append(row)
    return rows


def straight(start, goal):
    return [torch.stack([start[i], goal[i]]) for i in range(len(start))]
