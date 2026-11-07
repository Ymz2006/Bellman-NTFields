"""MPPI in a warp's coordinates z = phi(x): maze_scale's plan.mppi itself, run on z.

The planner never sees x. Start and goal are mapped to z, plan.mppi runs on z with the field's network evaluated on z
directly (the warp bypassed, WarpNN.bypass), and only the finished path is mapped back to x, exactly (warp.SepInverse
for the separable warps, a division for the uniform stretch, the identity without a warp), to be scored by
maze_scale's exact scorer in x. Settings are plan.settings(maze) carried into z with the quadrant stretch m
(warpplan.quadrant_stretch: the warp's median magnification inside the four quadrant mazes): step and tolerance x m,
so a step inside a quadrant is the x-planner's step there and the same number of z-units at every gap, and the step
budget is plan's rule with the certified true path measured in z.

Temperature: the softmax compares travel times, so it needs the field's time unit. A field trained in x (the gap
study's fields) has ntrl-demo's unit; a field trained in z (trainer.py z_space) has z-units, m times larger inside
the quadrants. t_scale divides the cost before the softmax (1 for x-trained fields, m for z-trained ones), so the
softmax is as selective per step inside the quadrants in every case. The same t_scale is used when such a field is
planned in x, so the x and z planners differ only in the space they step in.
"""
import numpy as np
import torch

import plan as P
import warp as W
import warpplan as WP


def inverse(warp):
    """x = phi^-1(z) for (N, 3) z: identity (no warp), division (uniform stretch), SepInverse (separable grids)."""
    if warp is None or isinstance(warp, WP.Identity) or warp.identity():
        return lambda z: z
    if not warp.spatial and not warp.heading:
        return lambda z: z / torch.tensor([warp.s, warp.s, warp.s_p], device=z.device, dtype=z.dtype)
    return W.SepInverse(warp)


def field_warp(net):
    w = getattr(net, 'warp', None)
    return WP.Identity() if w is None else w


def z_trained(net):
    w = getattr(net, 'warp', None)
    return bool(w is not None and getattr(w, 'cfg', {}).get('z_space'))


def cost_x(net, t_scale, chunk=200_000):
    """T(x0, x1) / t_scale (the field as trained: the warp applied inside the network)."""
    base = P.field_cost(net, chunk)
    return lambda pairs, ids=None: base(pairs, ids) / t_scale


def cost_z(net, t_scale, chunk=200_000):
    """H(z0, z1) / t_scale for warped pairs (P, 6): the network with its warp bypassed (inputs are already z)."""
    def cost(pairs, ids=None):
        out = []
        prev = getattr(net, 'bypass', False)
        net.bypass = True
        try:
            for i in range(0, len(pairs), chunk):
                with torch.no_grad():
                    out.append(net.out(pairs[i:i + chunk])[0][:, 0].detach())
        finally:
            net.bypass = prev
        return (torch.cat(out) if out else pairs.new_zeros(0)) / t_scale
    if getattr(net, 'warp', None) is None:          # a plain ntrl-demo network: z = x
        return cost_x(net, t_scale, chunk)
    return cost


def cost_lookup(net, inv, t_scale, chunk=200_000):
    """T(x0, x1) / t_scale looked up at x = phi^-1(z) for warped pairs (P, 6): a field trained in x (ntrl-demo, no warp of
    its own) planned in another warp's coordinates (plan_z with pwarp)."""
    base = cost_x(net, t_scale, chunk)

    def cost(pairs, ids=None):
        return base(torch.cat([inv(pairs[:, :3]), inv(pairs[:, 3:])], 1), ids)
    return cost


@torch.no_grad()
def plan_z(net, maze, S, G, seed, path_z_len, t_scale, st=None, pwarp=None):
    """plan.mppi on z for queries S, G (N, 3) in x. Returns (x paths, stop, converged, settings used, z paths).
    pwarp: plan in this (fixed, separable) warp's coordinates instead of the field's own, the field looked up at
    x = pwarp^-1(z) (for fields without a warp of their own: ntrl-demo).
    As plan.mppi's own paths, every path ends with the goal appended (also when MPPI did not arrive), so the scorer's
    collision flag includes that last jump: success is unaffected (it needs arrival), the wall / stall split is not."""
    st = dict(st or P.settings(maze))
    warp = field_warp(net) if pwarp is None else pwarp
    m = WP.quadrant_stretch(warp, maze)
    inv = inverse(getattr(net, 'warp', None) if pwarp is None else pwarp)
    Sz, Gz = warp(S), warp(G)
    lo, hi = warp(maze.lo[None])[0], warp(maze.hi[None])[0]          # only the heading clamp (index 2) is used by plan.mppi
    stz = dict(st, step=st['step'] * m, tol=st['tol'] * m)
    budget = torch.tensor(np.maximum(st['min_steps'], np.ceil(st['budget_factor'] * path_z_len / stz['step'])), device=S.device)
    cost = cost_z(net, t_scale) if pwarp is None else cost_lookup(net, inv, t_scale)
    paths_z, stop, conv = P.mppi(cost, Sz, Gz, seed, budget, lo, hi, **stz)
    paths = []
    for i in range(len(S)):
        x = inv(paths_z[i])
        x[0], x[-1] = S[i], G[i]
        paths.append(x)
    return paths, stop, conv, dict(stz, quadrant_stretch=m, budget_median=float(budget.float().median()), t_scale=t_scale), paths_z


@torch.no_grad()
def plan_x(net, maze, S, G, seed, path_clen, t_scale, st=None):
    """plan.mppi on x (the gap study's planner) with the field's cost divided by t_scale."""
    st = dict(st or P.settings(maze))
    budget = torch.tensor(np.maximum(st['min_steps'], np.ceil(st['budget_factor'] * path_clen / st['step'])), device=S.device)
    paths, stop, conv = P.mppi(cost_x(net, t_scale), S, G, seed, budget, maze.lo, maze.hi, **st)
    return [paths[i] for i in range(len(S))], stop, conv, dict(st, budget_median=float(budget.float().median()), t_scale=t_scale)
