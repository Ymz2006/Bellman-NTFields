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


def metric_full(warp, x):
    """M = J J^T at poses x (N, 3) for ANY warp, from its autograd Jacobian J = d phi / d x (no assumption that J is
    diagonal, uniform or isotropic): (N, 6) = [Mxx, Myy, Mpp, Mxy, Mxp, Myp]."""
    import warp as W
    J = W.jacobian(warp, x)                       # J[n, k, i] = d z_k / d x_i
    M = J @ J.transpose(1, 2)
    return torch.stack([M[:, 0, 0], M[:, 1, 1], M[:, 2, 2], M[:, 0, 1], M[:, 0, 2], M[:, 1, 2]], 1)


def quad(g, M):
    """g^T M g for gradients g (N, 3) and M as a diagonal (N, 3) or a full symmetric matrix (N, 6: xx, yy, pp, xy, xp, yp)."""
    if M.shape[1] == 3:
        return (g * g * M).sum(1)
    return (M[:, 0] * g[:, 0] ** 2 + M[:, 1] * g[:, 1] ** 2 + M[:, 2] * g[:, 2] ** 2
            + 2 * (M[:, 3] * g[:, 0] * g[:, 1] + M[:, 4] * g[:, 0] * g[:, 2] + M[:, 5] * g[:, 1] * g[:, 2]))


def eikonal_loss(net, points, speed, M0, M1, beta, c=1.0, tau_weight=0.5, pair_weight='tau', xpair=None):
    """ntrl-demo's metric loss with its eikonal term only (no temporal-difference term, no normal term), the squared
    gradient length replaced by the inner product g^T M g / c^2 (M0, M1: (N, 3) diagonals at the two ends; ones and
    c = 1 give ntrl-demo's own eikonal term; M0, M1 may also be full symmetric matrices, quad()). Weighting as there:
    exp(-tau_weight T) per pair (T the network's output; pair_weight 'z': the distance between the inputs instead; 'none'),
    the whole multiplied by beta. With a split head the gradient is the sum's and the pair weight the learned part's.
    Returns (loss, loss / beta)."""
    tau, _, Xp = net.out(points)
    dtau, = torch.autograd.grad(tau, Xp, torch.ones_like(tau), create_graph=True)
    tau_w = getattr(net, 'tau_net', None)          # split head: the pair weight is taken on the learned part d_net
    tau_w = tau if tau_w is None else tau_w
    if hasattr(net, 'tau_net'):
        net.tau_net = None                         # do not keep the graph on the module
    g0, g1 = dtau[:, :3], dtau[:, 3:]
    S0 = quad(g0, M0) / (c * c)
    S1 = quad(g1, M1) / (c * c)
    l0 = speed[:, 0] * torch.sqrt(S0 + 1e-8)
    l1 = speed[:, 1] * torch.sqrt(S1 + 1e-8)
    diff = 1e-2 * (torch.sqrt(l0) - 1) ** 2 + 1e-2 * (torch.sqrt(l1) - 1) ** 2
    if pair_weight == 'tau':            # ntrl-demo: on the field's own output
        wgt = torch.exp(-tau_weight * tau_w[:, 0])
    elif pair_weight == 'z':            # on the straight distance between the two inputs (no unit of time in it, no gradient)
        wgt = torch.exp(-tau_weight * torch.linalg.vector_norm(points[:, :3] - points[:, 3:], dim=1))
    elif pair_weight == 'tauz':
        # ntrl-demo's weight on the travel time as the warp shortens it: T x rho, rho = |z0 - z1| / (c |x0 - x1|) <= 1, the
        # pair's straight distance on the warp over its straight distance on the maze in the same unit. Inside a region the
        # warp magnifies fully rho = 1 (ntrl-demo's weight, as on the warped maze); across a compressed region rho is small,
        # so pairs the warp brought close weigh as near pairs, as they do when training on the warped maze.
        dz = torch.linalg.vector_norm(points[:, :3] - points[:, 3:], dim=1)
        dx = c * torch.linalg.vector_norm(xpair[:, :3] - xpair[:, 3:], dim=1)
        rho = (dz / dx.clamp(min=1e-9)).clamp(max=1.0)
        wgt = torch.exp(-tau_weight * tau_w[:, 0] * rho)
    else:
        assert pair_weight == 'none'
        wgt = torch.ones_like(diff)
    loss_n = torch.sum(diff * wgt) / points.shape[0]
    return beta * loss_n, loss_n


def both_loss(net, points, speed, M0, M1, beta, c, tau_weight=0.5):
    """Split head trained with two equations at once: the plain eikonal equation of the warped maze on the network part
    (S |grad_z d_net| = 1, in the network's unit) and the anisotropic equation on the sum (as eikonal_loss), each with
    half the weight; pair weights on the network part. The closed-form term has no parameters, so its gradient is taken
    without a graph and added to the network's."""
    tau, _, Xp = net.out(points)
    tau_net = net.tau_net
    net.tau_net = None
    g_net, = torch.autograd.grad(tau_net, Xp, torch.ones_like(tau_net), create_graph=True)
    g_def, = torch.autograd.grad((tau - tau_net).sum(), Xp, retain_graph=True)
    g_sum = g_net + g_def.detach()
    diff = 0.0
    for k, sl in enumerate((slice(0, 3), slice(3, 6))):
        Mk = M0 if k == 0 else M1
        iso = speed[:, k] * torch.sqrt((g_net[:, sl] ** 2).sum(1) + 1e-8)
        ani = speed[:, k] * torch.sqrt(quad(g_sum[:, sl], Mk) / (c * c) + 1e-8)
        diff = diff + 0.5e-2 * (torch.sqrt(iso) - 1) ** 2 + 0.5e-2 * (torch.sqrt(ani) - 1) ** 2
    loss_n = torch.sum(diff * torch.exp(-tau_weight * tau_net[:, 0])) / points.shape[0]
    return beta * loss_n, loss_n


# --- a messy warp: the generality test ------------------------------------------------------------------------------------
class MessyWarp(torch.nn.Module):
    """A fixed warp with none of the constructed warp's structure: the base warp followed by a smooth, non-separable
    displacement of (x, y),

        z' = z + sum_j A_j sin(2 pi K_j . z + P_j),        z = base(x),

    `modes` random plane waves (wave numbers `kmin`..`kmax` cycles per warp unit, random directions, phases and
    polarizations A_j), scaled so that the displacement's Jacobian is at most `amp` in norm anywhere (sum_j 2 pi |A_j||K_j|
    = amp < 1: z -> z' is then a diffeomorphism, and the fixed point z = z' - d(z) converges at rate amp). Its Jacobian is
    full (shear, rotation, local anisotropy that changes from place to place), as a learned warp's would be. The heading
    coordinate is left as the base warp makes it. Stands in for a learned warp in the loss and the planners:
    forward(x), inverse_fn(z'), cfg (with cfg['messy'])."""
    xy, heading = True, False        # (not separable: nothing may take the separable shortcuts)

    def __init__(self, base, amp=0.4, modes=8, seed=0, kmin=0.2, kmax=0.8):
        super().__init__()
        assert 0 <= amp < 1
        self.base = base
        self.cfg = dict(base.cfg, messy=dict(amp=float(amp), modes=int(modes), seed=int(seed), kmin=float(kmin), kmax=float(kmax)))
        self.s, self.s_p, self.sep = base.s, base.s_p, False
        g = torch.Generator().manual_seed(int(seed))
        ang = 2 * math.pi * torch.rand(modes, generator=g)
        mag = kmin + (kmax - kmin) * torch.rand(modes, generator=g)
        K = torch.stack([mag * torch.cos(ang), mag * torch.sin(ang)], 1)
        pol = 2 * math.pi * torch.rand(modes, generator=g)
        A = torch.stack([torch.cos(pol), torch.sin(pol)], 1) * (0.5 + torch.rand(modes, 1, generator=g))
        A = A * (amp / float((2 * math.pi * A.norm(dim=1) * K.norm(dim=1)).sum())) if amp > 0 else A * 0
        self.register_buffer('K', K)
        self.register_buffer('A', A)
        self.register_buffer('P', 2 * math.pi * torch.rand(modes, generator=g))
        self._inv = None

    def disp(self, zxy):
        return torch.sin(2 * math.pi * zxy @ self.K.T.to(zxy.dtype) + self.P.to(zxy.dtype)) @ self.A.to(zxy.dtype)

    def forward(self, q):
        z = self.base(q)
        return torch.cat([z[:, :2] + self.disp(z[:, :2]), z[:, 2:3]], 1)

    def identity(self):
        return False

    @property
    def spatial(self):
        return True

    @torch.no_grad()
    def inverse_fn(self, zp):
        """x with forward(x) = zp: fixed point for the displacement (float64), then the base warp's exact inverse."""
        import warp as W
        if self._inv is None:
            self._inv = W.SepInverse(self.base)
        z = zp.double().clone()
        for _ in range(60):
            z[:, :2] = zp[:, :2].double() - self.disp(z[:, :2])
        return self._inv(z).to(zp.dtype)


def make_warp(cfg):
    """The warp of a checkpoint's warp_config: cross_warp's Warp, wrapped in MessyWarp when the config says so."""
    import warp as W
    cfg = dict(cfg)
    messy = cfg.pop('messy', None)
    base = W.Warp(cfg)
    return base if messy is None else MessyWarp(base, **messy)


# --- split head ---------------------------------------------------------------------------------------------------------
def split_class(base):
    """Subclass of cross_warp's WarpNN (warp.warped_class) with the closed-form deficit term (module docstring).
    split = 0: the plain network. set_split(coeff, m) needs the warp attached; coeff = c for H' = c T."""
    class ScaledNorm(torch.nn.Module):
        """ntrl-demo's InstanceNorm of the embedding followed by a constant factor (no parameters: the state dict is unchanged)."""

        def __init__(self, norm, scale, groups=None):
            super().__init__()
            self.norm, self.scale = norm, float(scale)
            self.groups = None if groups is None else [float(g) for g in groups]

        def forward(self, y):
            y = self.scale * self.norm(y)
            if self.groups is not None:               # one factor per group of 16 features (the head takes a soft maximum inside each group)
                y = y * torch.tensor(self.groups, device=y.device, dtype=y.dtype).repeat_interleave(y.shape[1] // len(self.groups))
            return y

    class SplitNN(base):
        def __init__(self, device, dim, B):
            super().__init__(device, dim, B)
            self.split, self.m_ref, self.tau_net = 0.0, 1.0, None
            self.head_scale, self.emb_scale, self.group_scales = 1.0, 1.0, None

        def set_scales(self, head_scale=1.0, emb_scale=1.0, group_scales=None):
            """Output range. head_scale h: the distance head's scalar 0.2 becomes 0.2 h (the output is multiplied by h).
            emb_scale k: the normalized embeddings are multiplied by k before the distance head (larger differences, and a
            sharper soft maximum inside each group of 16). group_scales (16 numbers): one factor per group of the head, so
            some groups measure fine distances and others coarse ones (a multi-scale head: the largest output grows with
            the coarse groups while the fine groups keep their amplitude)."""
            self.head_scale, self.emb_scale = float(head_scale), float(emb_scale)
            self.group_scales = None if group_scales is None else [float(g) for g in group_scales]
            norm = self.encoder_norm.norm if isinstance(self.encoder_norm, ScaledNorm) else self.encoder_norm
            self.encoder_norm = norm if (self.emb_scale == 1.0 and self.group_scales is None) else ScaledNorm(norm, self.emb_scale, self.group_scales)
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
            if self.head_scale != 1.0:
                tau = self.head_scale * tau
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
    net.warp = make_warp(payload['warp_config'])
    net.load_state_dict(payload['model_state_dict'], strict=True)
    net.to(torch.device(device)).float().eval()
    if 'z_space' in payload['warp_config']:
        net.warp.cfg['z_space'] = payload['warp_config']['z_space']
    info = payload.get('aniso', {})
    net.set_scales(info.get('head_scale', 1.0), info.get('emb_scale', 1.0), info.get('group_scales'))
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
def check_metric_full(warp, maze, n=4096, seed=0, eps=1e-5):
    """metric_full (autograd) against central differences of the warp, float64: largest relative difference of J J^T;
    also how far the warp is from the constructed one's structure (largest off-diagonal entry of J J^T relative to its
    diagonal, and the spread of the local anisotropy sigma_max / sigma_min of the x, y block)."""
    import copy
    gen = torch.Generator(maze.device).manual_seed(seed)
    q = maze.uniform(n, gen).double()
    w = copy.deepcopy(warp).double()
    M = metric_full(w, q)
    cols = []
    with torch.no_grad():
        for a in range(3):
            e = torch.zeros(1, 3, dtype=torch.float64, device=q.device)
            e[0, a] = eps
            cols.append((w(q + e) - w(q - e)) / (2 * eps))
    J = torch.stack(cols, 2)
    Mf = J @ J.transpose(1, 2)
    ref = torch.stack([Mf[:, 0, 0], Mf[:, 1, 1], Mf[:, 2, 2], Mf[:, 0, 1], Mf[:, 0, 2], Mf[:, 1, 2]], 1)
    sv = torch.linalg.svdvals(J[:, :2, :2])
    return dict(max_rel=float((M - ref).abs().max() / ref.abs().max()), offdiagonal_max_rel=float(M[:, 3:].abs().max() / M[:, :3].abs().max()),
                anisotropy_median=float((sv[:, 0] / sv[:, 1]).median()), anisotropy_p90=float(torch.quantile(sv[:, 0] / sv[:, 1], 0.9)),
                stretch_min=float(sv.min()), stretch_max=float(sv.max()), det_min=float(torch.linalg.det(J[:, :2, :2]).min()), poses=n)


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
    M0, M1 = metric_full(w, pairs[:, :3]), metric_full(w, pairs[:, 3:])
    a0, a1 = (gx[:, :3] ** 2).sum(1), (gx[:, 3:] ** 2).sum(1)
    b0, b1 = quad(gz[:, :3], M0), quad(gz[:, 3:], M1)
    import warp as W
    Jt0 = torch.einsum('nki,nk->ni', W.jacobian(w, pairs[:, :3]), gz[:, :3])   # J^T grad_z H
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
    full = torch.cat([one, torch.zeros_like(one)], 1)
    assert abs(float(eikonal_loss(net, points, speed, full, full, 1.0)[1]) - mine) <= tol * max(abs(mine), 1e-12)
    assert abs(mine - ref) <= tol * max(abs(ref), 1e-12), (mine, ref)
    return dict(package_eikonal_term=ref, eikonal_loss=mine)
