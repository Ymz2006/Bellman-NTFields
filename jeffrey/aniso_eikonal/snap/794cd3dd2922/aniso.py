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

Split head (split_class). The warp magnifies the quadrants m times and the band less, so against a uniform magnification
m x it leaves a deficit u(x) = x - phi(x) / m along x and y (zero slope where the warp magnifies m times, slope
1 - F' / m where it does not; none along the heading, which is magnified m times everywhere). The split head adds that
deficit back in closed form,

    H'(z0, z1) = d_net(z0, z1) + c |u(z0) - u(z1)| ,

a sum of two metrics. The anisotropic equation is asked of the sum; the network d_net is left with what the warp did not
compress. Where the path stays in one kind of region the split is exact with d_net the warped maze's travel time (inside
a quadrant u is constant; in the band's centre both terms are straight lines in the same direction); in the arms, where
the warp magnifies one axis only, the sum of the two lengths overestimates an oblique segment.
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
    the whole multiplied by beta. With a split head the gradient is the sum's and the pair weight the learned part's.
    Returns (loss, loss / beta)."""
    tau, _, Xp = net.out(points)
    dtau, = torch.autograd.grad(tau, Xp, torch.ones_like(tau), create_graph=True)
    tau_w = getattr(net, 'tau_net', None)          # split head: the pair weight is taken on the learned part d_net
    tau_w = tau if tau_w is None else tau_w
    if hasattr(net, 'tau_net'):
        net.tau_net = None                         # do not keep the graph on the module
    g0, g1 = dtau[:, :3], dtau[:, 3:]
    S0 = (g0 * g0 * M0).sum(1) / (c * c)
    S1 = (g1 * g1 * M1).sum(1) / (c * c)
    l0 = speed[:, 0] * torch.sqrt(S0 + 1e-8)
    l1 = speed[:, 1] * torch.sqrt(S1 + 1e-8)
    diff = 1e-2 * (torch.sqrt(l0) - 1) ** 2 + 1e-2 * (torch.sqrt(l1) - 1) ** 2
    loss_n = torch.sum(diff * torch.exp(-tau_weight * tau_w[:, 0])) / points.shape[0]
    return beta * loss_n, loss_n


# --- split head ---------------------------------------------------------------------------------------------------------
def split_class(base):
    """Subclass of cross_warp's WarpNN (warp.warped_class) with the closed-form deficit term (module docstring).
    split = 0: the plain network. set_split(coeff, m) needs the warp attached; coeff = c for H' = c T."""
    class SplitNN(base):
        def __init__(self, device, dim, B):
            super().__init__(device, dim, B)
            self.split, self.m_ref, self.tau_net = 0.0, 1.0, None
            self.u_z0, self.u_dz = 0.0, 1.0
            self.register_buffer('u_grid', torch.zeros(2), persistent=False)

        def set_split(self, coeff, m_ref, n=1 << 18, nz=1 << 17):
            """Tabulate u as a function of z along one axis (the constructed warp maps x and y alike): u = x - z / m_ref
            on n nodes of x (float64), resampled on nz uniform nodes of z; linear in between, constant beyond the box."""
            import copy
            w = copy.deepcopy(self.warp).double()
            dev = next(self.parameters()).device
            with torch.no_grad():
                t = torch.linspace(-0.5, 0.5, n, device=dev, dtype=torch.float64)
                z = w(torch.stack([t, t, torch.zeros_like(t)], 1))
                assert torch.allclose(z[:, 0], z[:, 1]), 'the split head needs the same map along x and y'
                zx = z[:, 0].cpu().numpy()
                assert (np.diff(zx) > 0).all()
                zg = np.linspace(zx[0], zx[-1], nz)
                ug = np.interp(zg, zx, t.cpu().numpy() - zx / m_ref)
            self.u_grid = torch.tensor(ug, device=dev, dtype=next(self.parameters()).dtype)
            self.u_z0, self.u_dz = float(zg[0]), float(zg[1] - zg[0])
            self.split, self.m_ref = float(coeff), float(m_ref)

        def u_of_z(self, z):
            t = ((z - self.u_z0) / self.u_dz).clamp(0, len(self.u_grid) - 1)
            i = torch.floor(t).clamp(max=len(self.u_grid) - 2).long()
            f = t - i
            return self.u_grid[i] * (1 - f) + self.u_grid[i + 1] * f

        def u_of_x(self, x):
            return x[:, :2] - self.warp(x)[:, :2] / self.m_ref

        def out(self, coords):
            tau, w, xp = super().out(coords)
            self.tau_net = tau
            if self.split:
                if self.bypass:
                    u0, u1 = self.u_of_z(xp[:, 0:2]), self.u_of_z(xp[:, 3:5])
                else:
                    u0, u1 = self.u_of_x(xp[:, :3]), self.u_of_x(xp[:, 3:])
                tau = tau + self.split * torch.sqrt(((u0 - u1) ** 2).sum(1, keepdim=True) + 1e-12)
            return tau, w, xp
    return SplitNN


def load_field(checkpoint, device='cuda:0', split=None):
    """A field trained by atrain.py, frozen. split: (coeff, m) to add the deficit term to a checkpoint trained without
    it (the control: the warped maze's field plus the closed-form term); False: the network alone; default: as the
    checkpoint was trained."""
    import warp as W
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    net = split_class(W.warped_class(W.network_module()))(device, 3, payload['B_state_dict'])
    net.warp = W.Warp(payload['warp_config'])
    net.load_state_dict(payload['model_state_dict'], strict=True)
    net.to(torch.device(device)).float().eval()
    info = payload.get('aniso', {})
    if split is None and info.get('split'):
        split = (info['split'], info['m_ref'])
    if split:                                           # split=False: the network alone, whatever it was trained with
        net.set_split(*split)
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net


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
    split = (net.split, net.m_ref) if getattr(net, 'split', 0) else None
    if hasattr(net, 'tau_net'):
        net.tau_net = None
    net = copy.deepcopy(net).double()
    net.B = net.B.double()
    w = copy.deepcopy(warp).double()
    net.warp, net.bypass = w, False
    if split:
        net.set_split(*split)
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
