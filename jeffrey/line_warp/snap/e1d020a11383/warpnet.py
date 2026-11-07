"""A Fourier residual network as a diffeomorphism of the maze plane.

    u_0 = x,   u_{k+1} = u_k + amp * tanh(f_k(gamma(u_k))),   gamma(u) = [sin 2 pi B u, cos 2 pi B u]

as README_warp.md's phi, on (x, y). The last layer of every block starts at zero, so phi = id at the start. B holds one block of
random directions per frequency scale; `window` fades the higher scales in during training (coarse to fine).

A residual step is not invertible by construction. fit.py penalizes det J <= 0 on sampled points and checks the finished map
(checks.py), so 'diffeomorphism' is a measured property of a trained map, not a guarantee of the architecture.
"""
import math
import torch
import torch.nn as nn


def fourier_matrix(scales, per_scale, gen):
    """(len(scales) * per_scale, 2): each scale's rows are random directions of that many cycles per unit, jittered +-30 %."""
    rows = []
    for s in scales:
        ang = 2 * math.pi * torch.rand(per_scale, generator=gen)
        mag = s * (0.7 + 0.6 * torch.rand(per_scale, generator=gen))
        rows.append(torch.stack([mag * torch.cos(ang), mag * torch.sin(ang)], 1))
    return torch.cat(rows)


class Block(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.a = nn.Linear(width, hidden)
        self.b = nn.Linear(hidden, hidden)
        self.c = nn.Linear(hidden, 2)
        nn.init.zeros_(self.c.weight)
        nn.init.zeros_(self.c.bias)

    def forward(self, f):
        return self.c(nn.functional.silu(self.b(nn.functional.silu(self.a(f)))))


class FourierResNet(nn.Module):
    def __init__(self, blocks=8, hidden=128, scales=(1, 2, 4, 8, 16), per_scale=32, amp=0.1, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.cfg = dict(kind='resnet', blocks=blocks, hidden=hidden, scales=list(scales), per_scale=per_scale, amp=amp, seed=seed)
        self.register_buffer('B', fourier_matrix(scales, per_scale, gen))
        self.register_buffer('weights', torch.ones(len(scales) * per_scale))
        self.amp, self.groups, self.per_scale = amp, len(scales), per_scale
        state = torch.get_rng_state()
        torch.manual_seed(seed)
        self.net = nn.ModuleList([Block(4 * 0 + 2 * len(scales) * per_scale, hidden) for _ in range(blocks)])
        torch.set_rng_state(state)

    def window(self, alpha):
        """Coarse-to-fine: scale group g has weight clamp(alpha - g, 0, 1) smoothed as in BARF (alpha = groups: all on)."""
        g = torch.arange(self.groups, device=self.B.device, dtype=self.B.dtype)
        w = 0.5 * (1 - torch.cos(math.pi * (alpha - g).clamp(0, 1)))
        self.weights.copy_(w.repeat_interleave(self.per_scale))

    def gamma(self, u):
        p = 2 * math.pi * (u @ self.B.T)
        return torch.cat([self.weights * torch.sin(p), self.weights * torch.cos(p)], -1)

    def step(self, k, u):
        """The residual of block k at u."""
        return self.amp * torch.tanh(self.net[k](self.gamma(u)))

    def forward(self, x):
        u = x
        for k in range(len(self.net)):
            u = u + self.step(k, u)
        return u

    def block_lipschitz(self, x, eps):
        """Largest singular value (N, K) of each residual step's Jacobian along the path of x (N, 2). A step u + g(u) is invertible
        wherever this stays below 1 (the i-ResNet condition), so the whole map is."""
        out = []
        u = x
        for k in range(len(self.net)):
            cols = []
            for a in range(2):
                e = torch.zeros(1, 2, device=x.device, dtype=x.dtype)
                e[0, a] = eps
                cols.append((self.step(k, u + e) - self.step(k, u - e)) / (2 * eps))
            out.append(singular_values(torch.stack(cols, 2))[0])
            u = (u + self.step(k, u)).detach()
        return torch.stack(out, 1)


def jacobian_fd(phi, x, eps):
    """(N, 2, 2) central differences of phi at x (N, 2); differentiable with respect to phi's parameters."""
    cols = []
    for a in range(2):
        e = torch.zeros(1, 2, device=x.device, dtype=x.dtype)
        e[0, a] = eps
        cols.append((phi(x + e) - phi(x - e)) / (2 * eps))
    return torch.stack(cols, 2)


def singular_values(J):
    """Larger and smaller singular value of 2 x 2 matrices, and the determinant."""
    a, b, c, d = J[:, 0, 0], J[:, 0, 1], J[:, 1, 0], J[:, 1, 1]
    det = a * d - b * c
    f = a * a + b * b + c * c + d * d
    root = torch.sqrt((f * f - 4 * det * det).clamp(min=1e-20))
    s1 = torch.sqrt(0.5 * (f + root))
    return s1, det.abs() / s1.clamp(min=1e-12), det


def build(cfg):
    """A plane warp from its config (FourierResNet.cfg)."""
    if cfg['kind'] == 'resnet':
        return FourierResNet(cfg['blocks'], cfg['hidden'], cfg['scales'], cfg['per_scale'], cfg['amp'], cfg['seed'])
    raise ValueError(cfg['kind'])
