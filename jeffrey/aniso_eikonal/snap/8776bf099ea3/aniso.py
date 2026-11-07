"""The eikonal equation in a warp's coordinates, with an inner product in place of the gradient's length.

A warp z = phi(x) (a diffeomorphism of the robot's C-space, here jeffrey/cross_warp's constructed warp) is put in front of
ntrl-demo's network, T(x0, x1) = H(phi(x0), phi(x1)). The real travel time obeys the isotropic eikonal equation on the
maze, S(x) |grad_x T| = 1. By the chain rule grad_x T = J^T grad_z H with J = d phi / d x, so in the warp's coordinates

    S(x)^2 * grad_z H^T (J J^T) grad_z H = 1 .                                                            (anisotropic)

M = J J^T is symmetric positive definite (phi is a diffeomorphism): the equation is the eikonal equation of the Riemannian
metric G = (J J^T)^-1 / S^2 on z. In z the robot's speed depends on its direction u (|u| = 1): S / |J^-1 u|; the fastest
direction out of a point is -M grad_z H, not -grad_z H. Training H on z with the plain length |grad_z H| = 1 / S instead
(cross_warp's 'trained on the warped maze') solves the eikonal equation of another metric, I / S^2 on z: the travel time
of the warped maze, not of the maze.

Time unit: the network may learn H' = c T (c > 0) instead of T; then S^2 grad H'^T (M / c^2) grad H' = 1. c = E / U for a
unit of U cells of free travel (U = E: ntrl-demo's normalized unit, c = 1).

For the constructed warp phi = (F(x), F(y), s_p p) the Jacobian is diagonal, J = diag(F'(x), F'(y), s_p), and so is M.
"""
import math
import numpy as np
import torch


def metric_diag(warp, x):
    """Diagonal of M = J J^T at poses x (N, 3) for a separable warp without coupling or heading grids."""
    assert not warp.xy and not warp.heading, 'metric_diag: separable warps only'
    if warp.sep:
        _, _, la, lb = warp.psi(x[:, 0], x[:, 1])
    else:
        la = lb = torch.zeros_like(x[:, 0])
    return torch.stack([(warp.s * torch.exp(la)) ** 2, (warp.s * torch.exp(lb)) ** 2, torch.full_like(la, warp.s_p ** 2)], 1)


def eikonal_loss(net, points, speed, M0, M1, beta, c=1.0, tau_weight=0.5):
    """ntrl-demo's metric loss with its eikonal term only (no temporal-difference term, no normal term), the squared
    gradient length replaced by the inner product g^T M g / c^2 (M0, M1: (N, 3) diagonals at the two ends; ones and
    c = 1 give ntrl-demo's own eikonal term). Weighting as there: exp(-tau_weight T) per pair (T the network's output),
    the whole multiplied by beta. Returns (loss, loss / beta)."""
    tau, _, Xp = net.out(points)
    dtau, = torch.autograd.grad(tau, Xp, torch.ones_like(tau), create_graph=True)
    g0, g1 = dtau[:, :3], dtau[:, 3:]
    S0 = (g0 * g0 * M0).sum(1) / (c * c)
    S1 = (g1 * g1 * M1).sum(1) / (c * c)
    l0 = speed[:, 0] * torch.sqrt(S0 + 1e-8)
    l1 = speed[:, 1] * torch.sqrt(S1 + 1e-8)
    diff = 1e-2 * (torch.sqrt(l0) - 1) ** 2 + 1e-2 * (torch.sqrt(l1) - 1) ** 2
    loss_n = torch.sum(diff * torch.exp(-tau_weight * tau[:, 0])) / points.shape[0]
    return beta * loss_n, loss_n


# --- regions of a gap maze ----------------------------------------------------------------------------------------------
REGIONS = ['quadrants', 'band edge cells', 'arms', 'centre']


def region4(maze, block, x, inset=1.0):
    """0 = inside a quadrant block, 1 = the band's edge cells (within `inset` cells of the band's outline, where the
    constructed warp still magnifies), 2 = arm (band interior along one axis, quadrant rows / columns along the other),
    3 = centre (band interior along both). x: (N, 3) normalized poses; returns a numpy int array."""
    n, E = maze.n, maze.E
    c = (x[:, :2].detach().cpu().numpy().astype(np.float64)) * E + n / 2
    inq = (c < block) | (c >= n - block)                       # per axis: quadrant columns / rows
    inner = (c > block + inset) & (c < n - block - inset)      # per axis: the band's interior
    out = np.full(len(c), 1)
    out[inq[:, 0] & inq[:, 1]] = 0
    out[(inner[:, 0] & inq[:, 1]) | (inner[:, 1] & inq[:, 0])] = 2
    out[inner[:, 0] & inner[:, 1]] = 3
    return out


# --- checks -------------------------------------------------------------------------------------------------------------
def check_metric(warp, maze, n=20_000, seed=0):
    """Largest relative difference between metric_diag and the autograd Jacobian's J J^T over uniform poses (and the
    largest off-diagonal entry of J relative to its diagonal)."""
    import warp as W
    gen = torch.Generator(maze.device).manual_seed(seed)
    q = maze.uniform(n, gen).double()
    w = __import__('copy').deepcopy(warp).double()
    J = W.jacobian(w, q)
    M = torch.einsum('nij,nkj->nik', J, J)
    d = metric_diag(w, q)
    diag = torch.diagonal(M, dim1=1, dim2=2)
    off = (M - torch.diag_embed(diag)).abs().amax((1, 2)) / diag.amax(1)
    return dict(max_rel_diag=float(((diag - d).abs() / diag).max()), max_rel_offdiag=float(off.max()), poses=n,
                metric_min=d.amin(0).tolist(), metric_max=d.amax(0).tolist())


def check_chain(net, warp, maze, n=4096, seed=1):
    """The identity the formulation rests on, on a network: |grad_x T|^2 (autograd through the warp) against
    grad_z H^T M grad_z H (the network on z, the warp bypassed). Returns the largest relative differences (float64)."""
    import copy
    gen = torch.Generator(maze.device).manual_seed(seed)
    pairs = torch.cat([maze.uniform(n, gen), maze.uniform(n, gen)], 1).double()
    net = copy.deepcopy(net).double()
    net.B = net.B.double()
    w = copy.deepcopy(warp).double()
    net.warp, net.bypass = w, False
    with torch.enable_grad():
        tau, _, xp = net.out(pairs)
        gx, = torch.autograd.grad(tau.sum(), xp)
    z = torch.cat([w(pairs[:, :3]), w(pairs[:, 3:])], 1)
    net.bypass = True
    with torch.enable_grad():
        tau_z, _, zp = net.out(z)
        gz, = torch.autograd.grad(tau_z.sum(), zp)
    M0, M1 = metric_diag(w, pairs[:, :3]), metric_diag(w, pairs[:, 3:])
    a0, a1 = (gx[:, :3] ** 2).sum(1), (gx[:, 3:] ** 2).sum(1)
    b0, b1 = (gz[:, :3] ** 2 * M0).sum(1), (gz[:, 3:] ** 2 * M1).sum(1)
    Jt0 = torch.sqrt(M0) * gz[:, :3]                              # J^T grad_z H for a diagonal J
    return dict(value_max_abs=float((tau - tau_z).abs().max()), sq_norm_max_rel=float(torch.maximum((a0 - b0).abs() / a0, (a1 - b1).abs() / a1).max()),
                gradient_max_rel=float(((gx[:, :3] - Jt0).abs().amax(1) / gx[:, :3].abs().amax(1)).max()), pairs=n)


def check_base_loss(function, net, points, speed, normal, tol=1e-5):
    """eikonal_loss with M = 1, c = 1 against ntrl-demo's own eikonal term: the vendored loss copy (trainer.ntrl_loss)
    returns its per-pair eikonal term `diff`; summed with the same exp(-T / 2) weights it must equal eikonal_loss."""
    import trainer as TR
    _, _, diff = TR.ntrl_loss(function, points, speed, normal, 1.0, 0.001, 1)
    with torch.no_grad():
        T = net.out(points)[0][:, 0]
    ref = float((diff * torch.exp(-0.5 * T)).sum() / len(points))
    one = torch.ones(len(points), 3, device=points.device)
    mine = float(eikonal_loss(net, points, speed, one, one, 1.0)[1])
    assert abs(mine - ref) <= tol * max(abs(ref), 1e-12), (mine, ref)
    return dict(package_eikonal_term=ref, eikonal_loss=mine)
