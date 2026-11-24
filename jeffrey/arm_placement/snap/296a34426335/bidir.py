"""Planners on a trained field that use its symmetry T(a, b) = T(b, a): same field, same MPPI step as core.mppi (ntrl-demo's), no
collision checker.

Why: with the forward planner a query whose GOAL is inside the cupboard is solved far less often than the same kind of query with the
START inside (free -> compartment against compartment -> free), although the field is symmetric: the planner leaves a compartment
easily and fails on the last stretch into one.

    plan_reverse   plans goal -> start with core.plan and returns the reversed path
    plan_bidir     two heads, one from the start and one from the goal; in every iteration each head takes one MPPI step towards the
                   other head's current position (the goal features are re-encoded every iteration). Stops when the heads are within
                   the planner tolerance. Path = start head's path + the goal head's path reversed. `steps` = iterations, so the
                   number of moves and of field evaluations is 2 x steps (compare with the forward planner at 2 x steps).
"""
import torch
import core as C


@torch.no_grad()
def step(ev, cur, prior, target_feat, gen, lo, hi, sample_num, horizon, step, temperature):
    """One MPPI step of core.mppi for every row of cur towards the configuration whose features are target_feat."""
    k, D = cur.shape
    dev = cur.device
    dp = step * torch.randn(k, sample_num, 1, D, device=dev, generator=gen) + step * torch.randn(k, sample_num, horizon, D, device=dev, generator=gen)
    dp = dp + 2 * prior[:, None, None, :]
    dp = dp / (torch.linalg.vector_norm(dp, dim=3, keepdim=True).clamp(min=step) / step)
    roll = torch.maximum(torch.minimum(cur[:, None, None, :] + torch.cumsum(dp, dim=2), hi), lo)
    flat = roll[:, :, [0, -1], :].reshape(-1, D)
    d = ev.clearance(flat) if ev.lift else None
    gf = target_feat[:, None, None, :].expand(k, sample_num, 2, target_feat.shape[1])
    c = ev.to_goal(flat, gf.reshape(-1, target_feat.shape[1]), d=d).reshape(k, sample_num, 2)
    c = 10 * c[..., 0] + c[..., 1]
    weight = torch.softmax(-temperature * c, dim=1)
    update = torch.einsum('bs,bsd->bd', weight, dp[:, :, 0, :])
    return torch.maximum(torch.minimum(cur + update, hi), lo), update


@torch.no_grad()
def plan_bidir(ev, start, goal, seed, steps, sample_num=50, horizon=5, step=0.015, temperature=50.0, tol=0.01, **_):
    """Returns (list of paths start ... goal, met (n,) bool, where the heads met as the share of the path's waypoints)."""
    sc = ev.scene
    dev, n, D = start.device, len(start), start.shape[1]
    gen = torch.Generator(dev).manual_seed(int(seed))
    A = start.new_empty(n, steps + 1, D)
    B = start.new_empty(n, steps + 1, D)
    A[:, 0], B[:, 0] = start, goal
    a, b = start.clone(), goal.clone()
    pa, pb = torch.zeros(n, D, device=dev), torch.zeros(n, D, device=dev)
    na = torch.zeros(n, dtype=torch.long, device=dev)
    nb = torch.zeros(n, dtype=torch.long, device=dev)
    met = torch.linalg.vector_norm(a - b, dim=1) < tol
    kw = dict(sample_num=sample_num, horizon=horizon, step=step, temperature=temperature)
    for it in range(steps):
        ids = torch.nonzero(~met).squeeze(1)
        if not len(ids):
            break
        new, up = globals()['step'](ev, a[ids], pa[ids], ev.encode(b[ids]), gen, sc.lo, sc.hi, **kw)
        a[ids], pa[ids] = new, up
        na[ids] += 1
        A[ids, na[ids]] = new
        met[ids] = torch.linalg.vector_norm(new - b[ids], dim=1) < tol
        ids = torch.nonzero(~met).squeeze(1)
        if not len(ids):
            break
        new, up = globals()['step'](ev, b[ids], pb[ids], ev.encode(a[ids]), gen, sc.lo, sc.hi, **kw)
        b[ids], pb[ids] = new, up
        nb[ids] += 1
        B[ids, nb[ids]] = new
        met[ids] = torch.linalg.vector_norm(new - a[ids], dim=1) < tol
    paths = [torch.cat([A[i, :int(na[i]) + 1], B[i, :int(nb[i]) + 1].flip(0)]) for i in range(n)]
    return paths, met, (na.float() + 1) / (na + nb + 2).float()


@torch.no_grad()
def plan_reverse(ev, start, goal, seed, steps, **planner):
    paths, conv = C.plan(ev, goal, start, seed, steps, **planner)
    return [p.flip(0) for p in paths], conv


@torch.no_grad()
def hub_pool(scene, count, seed, clear=None):
    """count free configurations for hubs: every grasp point in the sampling box and outside the cupboard's outer box, clearance at
    least the speed margin (full speed, 0.05 m from everything)."""
    from maze import MARGIN_CELLS, OFFSET_CELLS
    gen = torch.Generator(scene.device).manual_seed(int(seed) + 777)
    out, have = [], 0
    while have < count:
        x = scene.free(4 * count, gen, MARGIN_CELLS - OFFSET_CELLS, clear=clear)
        x = x[~scene.in_cupboard(x).any(1)]
        out.append(x)
        have += len(x)
    return torch.cat(out)[:count]


@torch.no_grad()
def choose_hub(ev, start, goal, pool, chunk=16):
    """For every query the pool configuration with the smallest T(start, hub) + T(goal, hub) by the field itself."""
    fp = ev.encode(pool)
    fs, fg = ev.encode(start), ev.encode(goal)
    pick = []
    for i in range(0, len(start), chunk):
        a = fs[i:i + chunk][:, None, :].expand(-1, len(pool), -1).reshape(-1, fs.shape[1])
        b = fg[i:i + chunk][:, None, :].expand(-1, len(pool), -1).reshape(-1, fs.shape[1])
        p = fp[None].expand(len(fs[i:i + chunk]), -1, -1).reshape(-1, fs.shape[1])
        t = (C.F.metric(a, p) + C.F.metric(b, p)).reshape(-1, len(pool))
        pick.append(t.argmin(1))
    return pool[torch.cat(pick)]


@torch.no_grad()
def plan_hub(ev, start, goal, seed, steps, pool, **planner):
    """start -> hub and goal -> hub with core.plan (`steps` each), the second reversed: both legs go the easy way, out of the cupboard.
    Returns (paths, arrived at the hub from both ends)."""
    hub = choose_hub(ev, start, goal, pool)
    pa, ca = C.plan(ev, start, hub, seed, steps, **planner)
    pb, cb = C.plan(ev, goal, hub, int(seed) + 1000, steps, **planner)
    return [torch.cat([a, b.flip(0)[1:]]) for a, b in zip(pa, pb)], ca & cb
