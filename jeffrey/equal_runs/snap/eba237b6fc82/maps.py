"""Diffeomorphisms of the box [-0.5, 0.5]^dim, any dimension, shear allowed, all with Fourier features.

The network is psi: z -> x (warped space to the scene): a line drawn in z is a curve in x, and that curve is what the collision oracle reads (scan.py). Forward and
backward maps are the same object, so nothing is lost: x -> z is `inverse`.

    FlowMap      a residual network with many small steps: u <- u + (1 / K) w(u) v(u, tau). shared=True: one Fourier velocity field integrated with midpoint steps
                 (a neural ODE); shared=False: one block per step (the ResNet of line_warp). w vanishes on the faces of the box, so faces stay on themselves and the box
                 maps onto itself. A smooth field's flow cannot fold; a finite step can, so det J is measured (measure.py).
    CouplingMap  a stack of monotone conditional maps, one axis at a time: u_a <- F(u_a | other coordinates), F the normalized integral of a positive density exp(h),
                 h = a Fourier-conditioned network on `knots` knots along the axis. Diffeomorphic by construction: every layer is strictly monotone in its own
                 coordinate. Two or more layers shear. conditioned=False leaves out the other coordinates: the separable map of line_warp, as a control.
    Identity     the control with no warp.
"""
import math
import torch
import torch.nn as nn


def fourier_matrix(scales, per_scale, dim, gen):
    """(len(scales) * per_scale, dim): each scale's rows are random directions of that many cycles per unit, jittered +-30 %."""
    rows = []
    for s in scales:
        d = torch.randn(per_scale, dim, generator=gen)
        d = d / d.norm(dim=1, keepdim=True).clamp(min=1e-9)
        rows.append(d * (s * (0.7 + 0.6 * torch.rand(per_scale, 1, generator=gen))))
    return torch.cat(rows)


class Features(nn.Module):
    """gamma(u) = [w sin 2 pi B u, w cos 2 pi B u], w = coarse-to-fine weights per scale."""

    def __init__(self, dim, scales, per_scale, gen):
        super().__init__()
        self.register_buffer('B', fourier_matrix(scales, per_scale, dim, gen))
        self.register_buffer('w', torch.ones(len(scales) * per_scale))
        self.groups, self.per_scale, self.width = len(scales), per_scale, 2 * len(scales) * per_scale

    def window(self, alpha):
        g = torch.arange(self.groups, device=self.B.device, dtype=self.B.dtype)
        self.w.copy_((0.5 * (1 - torch.cos(math.pi * (alpha - g).clamp(0, 1)))).repeat_interleave(self.per_scale))

    def forward(self, u):
        p = (2 * math.pi) * (u @ self.B.T)
        return torch.cat([self.w * torch.sin(p), self.w * torch.cos(p)], -1)


def mlp(width, hidden, depth, out):
    layers, d = [], width
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    last = nn.Linear(d, out)
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    return nn.Sequential(*layers, last)


class Identity(nn.Module):
    cfg = dict(kind='identity')

    def forward(self, z):
        return z

    inverse = forward

    def window(self, alpha):
        pass


class FlowMap(nn.Module):
    def __init__(self, dim=2, steps=8, hidden=128, depth=2, scales=(1, 2, 4, 8), per_scale=32, seed=0, shared=True, amp=4.0):
        super().__init__()
        self.cfg = dict(kind='flow', dim=dim, steps=steps, hidden=hidden, depth=depth, scales=list(scales), per_scale=per_scale, seed=seed, shared=shared, amp=amp)
        gen = torch.Generator().manual_seed(seed)
        self.dim, self.steps, self.shared, self.amp = dim, steps, shared, amp
        self.feat = Features(dim, scales, per_scale, gen)
        state = torch.get_rng_state()
        torch.manual_seed(seed)
        self.net = nn.ModuleList([mlp(self.feat.width + (2 if shared else 0), hidden, depth, dim) for _ in range(1 if shared else steps)])
        torch.set_rng_state(state)

    def window(self, alpha):
        self.feat.window(alpha)

    def velocity(self, u, k, tau):
        f = self.feat(u)
        if self.shared:
            f = torch.cat([f, torch.full_like(f[:, :1], math.sin(math.pi * tau)), torch.full_like(f[:, :1], math.cos(math.pi * tau))], 1)
            v = self.net[0](f)
        else:
            v = self.net[k](f)
        return self.amp * torch.tanh(v) * (1 - (2 * u) ** 2)              # zero normal velocity on every face

    def forward(self, z):
        h, u = 1.0 / self.steps, z
        for k in range(self.steps):
            tau = (k + 0.5) * h
            if self.shared:
                mid = u + 0.5 * h * self.velocity(u, k, k * h)
                u = u + h * self.velocity(mid, k, tau)
            else:
                u = u + h * self.velocity(u, k, tau)
        return u.clamp(-0.5, 0.5)

    @torch.no_grad()
    def inverse(self, x, iters=12):
        """x -> z, each step undone by fixed-point iteration (converges while a step's Lipschitz constant is below 1)."""
        h, u = 1.0 / self.steps, x
        for k in reversed(range(self.steps)):
            tau = (k + 0.5) * h
            target, w = u, u
            for _ in range(iters):
                if self.shared:
                    mid = w + 0.5 * h * self.velocity(w, k, k * h)
                    w = target - h * self.velocity(mid, k, tau)
                else:
                    w = target - h * self.velocity(w, k, tau)
            u = w
        return u


def density_map(h, t):
    """F(t) in [-0.5, 0.5] for rows of a log-density h (N, K + 1) on K equal cells of [-0.5, 0.5], linear density inside a cell, and dF / dt (N,). F is the normalized integral of exp(h)."""
    K = h.shape[1] - 1
    rho = torch.exp(h)
    dt = 1.0 / K
    C = torch.cumsum(0.5 * (rho[:, :-1] + rho[:, 1:]) * dt, 1)
    C = torch.cat([torch.zeros_like(C[:, :1]), C], 1)
    R = C[:, -1]
    s = (t.clamp(-0.5, 0.5) + 0.5) * K
    i = s.floor().clamp(max=K - 1).long()
    u = (s - i).unsqueeze(1)
    i = i.unsqueeze(1)
    r0, r1 = rho.gather(1, i), rho.gather(1, i + 1)
    F = C.gather(1, i) + dt * (r0 * u + 0.5 * (r1 - r0) * u * u)
    return (F[:, 0] / R) - 0.5, (r0 + (r1 - r0) * u)[:, 0] / R


def density_inverse(h, y):
    """t with F(t) = y, F as in density_map: find the cell from the cumulative table, then the quadratic inside it."""
    K = h.shape[1] - 1
    rho = torch.exp(h)
    dt = 1.0 / K
    C = torch.cumsum(0.5 * (rho[:, :-1] + rho[:, 1:]) * dt, 1)
    C = torch.cat([torch.zeros_like(C[:, :1]), C], 1)
    R = C[:, -1:]
    target = (y.clamp(-0.5, 0.5).unsqueeze(1) + 0.5) * R
    i = (torch.searchsorted(C.contiguous(), target.contiguous(), right=True) - 1).clamp(0, K - 1)
    r0, r1 = rho.gather(1, i), rho.gather(1, i + 1)
    a = 0.5 * (r1 - r0) * dt
    b = r0 * dt
    c = C.gather(1, i) - target
    disc = (b * b - 4 * a * c).clamp(min=0)
    u = (-2 * c) / (b + torch.sqrt(disc)).clamp(min=1e-20)             # the stable root of a u^2 + b u + c = 0
    return ((i + u.clamp(0, 1)) / K - 0.5)[:, 0]


class CouplingMap(nn.Module):
    def __init__(self, dim=2, layers=6, knots=96, hidden=128, depth=2, scales=(1, 2, 4, 8), per_scale=32, hmax=3.0, seed=0, conditioned=True):
        super().__init__()
        self.cfg = dict(kind='coupling', dim=dim, layers=layers, knots=knots, hidden=hidden, depth=depth, scales=list(scales), per_scale=per_scale, hmax=hmax, seed=seed, conditioned=conditioned)
        gen = torch.Generator().manual_seed(seed)
        self.dim, self.layers, self.knots, self.hmax, self.conditioned = dim, layers, knots, hmax, conditioned
        self.feat = nn.ModuleList([Features(dim - 1, scales, per_scale, gen) for _ in range(layers)])
        state = torch.get_rng_state()
        torch.manual_seed(seed)
        if conditioned:
            self.net = nn.ModuleList([mlp(self.feat[k].width, hidden, depth, knots + 1) for k in range(layers)])
        else:
            # separable control: a Fourier network of the position along the axis alone gives the log-density at the knots
            self.f1 = nn.ModuleList([Features(1, scales, per_scale, gen) for _ in range(layers)])
            self.net = nn.ModuleList([mlp(self.f1[k].width, hidden, depth, 1) for k in range(layers)])
            self.register_buffer('nodes', torch.linspace(-0.5, 0.5, knots + 1)[:, None])
        torch.set_rng_state(state)

    def window(self, alpha):
        for f in self.feat:
            f.window(alpha)
        if not self.conditioned:
            for f in self.f1:
                f.window(alpha)

    def density(self, k, u):
        a = k % self.dim
        if self.conditioned:
            c = torch.cat([u[:, :a], u[:, a + 1:]], 1)
            h = self.net[k](self.feat[k](c))
        else:
            h = self.net[k](self.f1[k](self.nodes))[:, 0][None].expand(len(u), -1)
        return self.hmax * torch.tanh(h / self.hmax)

    def forward(self, z):
        u = z
        for k in range(self.layers):
            a = k % self.dim
            F, _ = density_map(self.density(k, u), u[:, a])
            u = torch.cat([u[:, :a], F[:, None], u[:, a + 1:]], 1)
        return u

    def forward_logdet(self, z):
        """The map and log det J exactly: a layer's Jacobian is triangular with dF / dt on the diagonal."""
        u, ld = z, z.new_zeros(len(z))
        for k in range(self.layers):
            a = k % self.dim
            F, dF = density_map(self.density(k, u), u[:, a])
            u = torch.cat([u[:, :a], F[:, None], u[:, a + 1:]], 1)
            ld = ld + torch.log(dF.clamp(min=1e-12))
        return u, ld

    @torch.no_grad()
    def inverse(self, x):
        u = x
        for k in reversed(range(self.layers)):
            a = k % self.dim
            t = density_inverse(self.density(k, u), u[:, a])               # the conditioner does not read coordinate a, so it is the same before and after
            u = torch.cat([u[:, :a], t[:, None], u[:, a + 1:]], 1)
        return u


def build(cfg):
    c = dict(cfg)
    kind = c.pop('kind')
    if kind == 'identity':
        return Identity()
    if kind == 'flow':
        return FlowMap(**c)
    if kind == 'coupling':
        return CouplingMap(**c)
    raise ValueError(kind)


def jacobian_fd(psi, z, eps):
    """(N, d, d) central differences of psi at z, differentiable with respect to psi's parameters."""
    cols = []
    d = z.shape[1]
    for a in range(d):
        e = torch.zeros(d, device=z.device, dtype=z.dtype)
        e[a] = eps
        cols.append((psi(z + e) - psi(z - e)) / (2 * eps))
    return torch.stack(cols, 2)


def singular_values(J):
    """Singular values (N, d), largest first. Closed form for 2 x 2."""
    if J.shape[1] == 2:
        a, b, c, d = J[:, 0, 0], J[:, 0, 1], J[:, 1, 0], J[:, 1, 1]
        det = a * d - b * c
        f = a * a + b * b + c * c + d * d
        root = torch.sqrt((f * f - 4 * det * det).clamp(min=1e-20))
        s1 = torch.sqrt((0.5 * (f + root)).clamp(min=1e-20))
        return torch.stack([s1, det.abs() / s1.clamp(min=1e-12)], 1)
    return torch.linalg.svdvals(J)
