"""Only Bellman, Eikonal and confident Bellman-winner direction supervision."""
import torch
from torch.nn import functional as F

WEIGHTS = dict(bellman=6.0, eikonal=2.0, direction=2.0)


@torch.no_grad()
def bellman_backup(target, successors, goal, cost, safe):
    if not bool(safe.any(dim=1).all()):
        raise ValueError('Each source must have at least one safe action')
    repeated = goal[:, None, :].expand_as(successors)
    tail = target(successors.reshape(-1, 6), repeated.reshape(-1, 6))
    candidates = (cost + tail.reshape_as(cost)).masked_fill(~safe, torch.inf)
    return candidates.amin(dim=1), candidates


def eikonal_loss(model, speed, q, goal):
    q = q.detach().requires_grad_(True)
    goal = goal.detach().requires_grad_(True)
    value = model(q, goal)[:, 0]
    dq, dg = torch.autograd.grad(value.sum(), (q, goal), create_graph=True)
    with torch.no_grad():
        sq = speed(q.detach()).reshape(-1)
        sg = speed(goal.detach()).reshape(-1)
    nq = torch.sqrt(dq.square().sum(dim=1) + 1e-12)
    ng = torch.sqrt(dg.square().sum(dim=1) + 1e-12)
    residual = torch.cat((torch.log((sq*nq).clamp_min(1e-8)),
                          torch.log((sg*ng).clamp_min(1e-8))))
    return residual.square().mean(), dq


def direction_loss(gradient, q, successors, cost, safe, candidates):
    best_two = candidates.topk(k=2, largest=False, dim=1)
    rows = torch.arange(len(q), device=q.device)
    winner = best_two.indices[:, 0]
    gap = (best_two.values[:, 1] - best_two.values[:, 0]) / cost[rows, winner].clamp_min(1e-8)
    confident = (safe.sum(dim=1) >= 2) & torch.isfinite(gap) & (gap >= 0.10)
    delta = successors[rows, winner] - q.detach()
    descent = -gradient
    norm = torch.sqrt(descent.square().sum(dim=1) + 1e-12)
    cosine = (descent*delta).sum(dim=1) / (norm*torch.linalg.vector_norm(delta, dim=1)).clamp_min(1e-8)
    return (1-cosine[confident]).mean() if bool(confident.any()) else gradient.sum()*0.0


def training_loss(model, target, speed, q, goal, successors, cost, safe):
    backup, candidates = bellman_backup(target, successors, goal, cost, safe)
    value = model(q, goal)[:, 0]
    bellman = F.smooth_l1_loss(value, backup)
    eikonal, gradient = eikonal_loss(model, speed, q, goal)
    direction = direction_loss(gradient, q, successors, cost, safe, candidates)
    # Preserve the established accumulation order and its floating-point behavior.
    total = sum((6.0*bellman, 2.0*eikonal, 2.0*direction), value.new_zeros(()))
    return total, dict(bellman=bellman, eikonal=eikonal, direction=direction)
