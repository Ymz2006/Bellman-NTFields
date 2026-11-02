"""P-NTFields network (baselines/P-NTFields/models/model_res_sigmoid_multi.NN) re-implemented with autograd.

Same architecture, initialisation and factorisation (T(x0, x1) = |x1 - x0| / tau(x0, x1), tau symmetric through
elementwise log-sum-exp max/min of the two encodings, sigmoid(0.1 y) output), same Eikonal loss and planner update, but
the per-point input is a list of feature modules instead of the hard-wired Fourier mapping, so a frozen embedding grid
(masked trilinear lookup) can be concatenated to the Fourier features. Derivatives come from autograd instead of the
hand-derived out_laplace / out_backgrad passes.
"""
import math
import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import MaskedGridFeature  # noqa: E402,F401  (gibson_embed/train.py: masked bilinear/trilinear grid lookup)


class FourierB(nn.Module):
    """P-NTFields input mapping [sin(2 pi x B), cos(2 pi x B)] with the dataset's fixed B (dim x 128)."""

    def __init__(self, B):
        super().__init__()
        self.register_buffer('B', torch.as_tensor(B, dtype=torch.float32))
        self.channels = 2 * self.B.shape[1]

    def forward(self, x):
        p = x @ (2 * math.pi * self.B)
        return torch.cat([torch.sin(p), torch.cos(p)], -1)

    def jet(self, x):
        """f, df/dx_i (n, dim, F), d2f/dx_i^2 (n, dim, F)."""
        W = 2 * math.pi * self.B
        p = x @ W
        s, c = torch.sin(p), torch.cos(p)
        f = torch.cat([s, c], -1)
        Wt = W[None]  # (1, dim, 128)
        d = torch.cat([c[:, None] * Wt, -s[:, None] * Wt], -1)
        h = -torch.cat([Wt * Wt, Wt * Wt], -1) * f[:, None]
        return f, d, h


def grid_jet(feat, x):
    """Masked trilinear lookup of a MaskedGridFeature with its first and (diagonal) second derivatives in x.

    f = acc / den with acc = sum_c m_c w_c g_c, den = sum_c m_c w_c, w_c multilinear in the cell coordinates, so
    d2acc = d2den = 0 and f'' = -2 f' den' / den."""
    res, dim = feat.res, feat.dim
    u = (x + 0.5) * res - 0.5
    i0 = torch.floor(u)
    t = u - i0
    i0 = i0.long()
    acc, den = 0.0, 0.0
    dacc, dden = 0.0, 0.0
    for c in feat.corners:
        idx = (i0 + c).clamp(0, res - 1)
        tw = torch.where(c.bool(), t, 1 - t)  # (n, dim)
        w = torch.prod(tw, 1)
        # dw/dx_j = res * sign_j * prod_{k != j} tw_k
        prod_others = torch.stack([torch.prod(torch.cat([tw[:, :j], tw[:, j + 1:]], 1), 1) for j in range(dim)], 1)
        dw = res * torch.where(c.bool(), 1.0, -1.0)[None] * prod_others  # (n, dim)
        lin = (idx * feat.strides).sum(1)
        m = feat.free[lin]
        g = feat.grid[lin].float()  # (n, F)
        acc = acc + (m * w)[:, None] * g
        den = den + m * w
        dacc = dacc + (m[:, None] * dw)[:, :, None] * g[:, None]  # (n, dim, F)
        dden = dden + m[:, None] * dw
    valid = den > 1e-6
    den = den.clamp(min=1e-6)
    f = acc / den[:, None]
    df = (dacc - f[:, None] * dden[:, :, None]) / den[:, None, None]
    d2f = -2 * df * (dden / den[:, None])[:, :, None]
    df = df * valid[:, None, None]
    d2f = d2f * valid[:, None, None]
    return f, df, d2f


def feature_jet(feat, x):
    if hasattr(feat, 'jet'):
        return feat.jet(x)
    return grid_jet(feat, x)


class Jet:
    """Value v (n, W), first derivatives d (n, K, W) and diagonal second derivatives h (n, K, W) along K coordinates."""
    __slots__ = ('v', 'd', 'h')

    def __init__(self, v, d, h):
        self.v, self.d, self.h = v, d, h

    def linear(self, l):
        return Jet(l(self.v), self.d @ l.weight.T, self.h @ l.weight.T)

    def softplus(self, beta):
        s1 = torch.sigmoid(beta * self.v)
        s2 = beta * s1 * (1 - s1)
        return Jet(torch.nn.functional.softplus(self.v, beta), s1[:, None] * self.d, s2[:, None] * self.d * self.d + s1[:, None] * self.h)

    def __add__(self, o):
        return Jet(self.v + o.v, self.d + o.d, self.h + o.h)


class EikNet(nn.Module):
    def __init__(self, dim, features, hidden=128, nl1=3, nl2=3, scale=10.0):
        super().__init__()
        self.dim, self.scale = dim, scale
        self.features = nn.ModuleList(features)
        width = sum(f.channels for f in features)
        self.act = nn.Softplus(beta=scale)
        self.enc0 = nn.Linear(width, hidden)
        self.enc = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(nl1 - 1)])
        self.enc1 = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(nl1 - 1)])
        self.enc_last = nn.Linear(hidden, hidden)
        self.gen = nn.ModuleList([nn.Linear(2 * hidden, 2 * hidden) for _ in range(nl2)])
        self.gen1 = nn.ModuleList([nn.Linear(2 * hidden, 2 * hidden) for _ in range(nl2)])
        self.gen_out1 = nn.Linear(2 * hidden, hidden)
        self.gen_out2 = nn.Linear(hidden, 1)
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            stdv = 2.0 / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)
            m.bias.data.uniform_(-stdv, stdv)

    def encode(self, x):
        h = self.act(self.enc0(torch.cat([f(x) for f in self.features], -1)))
        for l, l1 in zip(self.enc, self.enc1):
            t = h
            h = self.act(l(h))
            h = self.act(l1(h) + t)
        return self.enc_last(h)

    def tau(self, x0, x1):
        n = x0.shape[0]
        h = self.encode(torch.cat([x0, x1], 0))
        xx = torch.stack([h[:n], h[n:]], 1)
        z = torch.cat([torch.logsumexp(self.scale * xx, 1) / self.scale, -torch.logsumexp(-self.scale * xx, 1) / self.scale], 1)
        for l, l1 in zip(self.gen, self.gen1):
            t = z
            z = self.act(l(z))
            z = self.act(l1(z) + t)
        y = self.gen_out2(self.act(self.gen_out1(z)))
        return torch.sigmoid(0.1 * y).squeeze(-1)

    def travel_time(self, x0, x1):
        return (x1 - x0).norm(dim=1) / self.tau(x0, x1)

    def encode_jet(self, x):
        parts = [feature_jet(f, x) for f in self.features]
        j = Jet(*[torch.cat([p[i] for p in parts], -1) for i in range(3)])
        h = j.linear(self.enc0).softplus(self.scale)
        for l, l1 in zip(self.enc, self.enc1):
            t = h
            h = h.linear(l).softplus(self.scale)
            h = (h.linear(l1) + t).softplus(self.scale)
        return h.linear(self.enc_last)

    def tau_jet(self, x0, x1):
        """tau, its gradient (n, 2 dim) and its Laplacians with respect to x0 and x1, by forward-mode propagation of
        value / first derivative / diagonal second derivative through every layer (one pass, no double backward)."""
        n, dim, beta = x0.shape[0], self.dim, self.scale
        e = self.encode_jet(torch.cat([x0, x1], 0))
        a, b = Jet(e.v[:n], e.d[:n], e.h[:n]), Jet(e.v[n:], e.d[n:], e.h[n:])
        # elementwise log-sum-exp max and min of the two encodings; a depends on x0 only, b on x1 only
        pa = torch.sigmoid(beta * (a.v - b.v))
        pb = 1 - pa
        qa = pb
        qb = pa
        zmax_v = torch.logsumexp(beta * torch.stack([a.v, b.v], 1), 1) / beta
        zmin_v = -torch.logsumexp(-beta * torch.stack([a.v, b.v], 1), 1) / beta
        c1, c2 = (beta * pa * pb)[:, None], (-beta * qa * qb)[:, None]
        zmax = Jet(zmax_v, torch.cat([pa[:, None] * a.d, pb[:, None] * b.d], 1),
                   torch.cat([c1 * a.d * a.d + pa[:, None] * a.h, c1 * b.d * b.d + pb[:, None] * b.h], 1))
        zmin = Jet(zmin_v, torch.cat([qa[:, None] * a.d, qb[:, None] * b.d], 1),
                   torch.cat([c2 * a.d * a.d + qa[:, None] * a.h, c2 * b.d * b.d + qb[:, None] * b.h], 1))
        z = Jet(torch.cat([zmax.v, zmin.v], -1), torch.cat([zmax.d, zmin.d], -1), torch.cat([zmax.h, zmin.h], -1))
        for l, l1 in zip(self.gen, self.gen1):
            t = z
            z = z.linear(l).softplus(beta)
            z = (z.linear(l1) + t).softplus(beta)
        y = z.linear(self.gen_out1).softplus(beta).linear(self.gen_out2)
        s = torch.sigmoid(0.1 * y.v)
        s1 = 0.1 * s * (1 - s)
        s2 = 0.01 * s * (1 - s) * (1 - 2 * s)
        tau = s.squeeze(-1)
        g = (s1[:, None] * y.d).squeeze(-1)
        lap = (s2[:, None] * y.d * y.d + s1[:, None] * y.h).squeeze(-1)
        return tau, g, lap[:, :dim].sum(1), lap[:, dim:].sum(1)


def load_pntfields_state(net, state):
    """Load a baseline checkpoint ('model_state_dict' of model_res_sigmoid_multi.NN) into an EikNet with Fourier input."""
    nl1 = len(net.enc) + 1
    names = {'encoder.0': 'enc0', f'encoder.{nl1}': 'enc_last', f'generator.{len(net.gen)}': 'gen_out1', f'generator.{len(net.gen) + 1}': 'gen_out2'}
    for i in range(1, nl1):
        names[f'encoder.{i}'] = f'enc.{i - 1}'
        names[f'encoder1.{i}'] = f'enc1.{i - 1}'
    for i in range(len(net.gen)):
        names[f'generator.{i}'] = f'gen.{i}'
        names[f'generator1.{i}'] = f'gen1.{i}'
    new = {}
    for k, v in state.items():
        base, leaf = k.rsplit('.', 1)
        if base in names:
            new[f'{names[base]}.{leaf}'] = v
    missing, unexpected = net.load_state_dict(new, strict=False)
    assert not unexpected and all(m.startswith('features.') for m in missing), (missing, unexpected)
    return net


def fd_laplacians(net, x0, x1, h):
    """Central-difference Laplacians of tau at scale h (units) with respect to x0 and x1, in one batched pass."""
    n, dim = x0.shape[0], x0.shape[1]
    E = h * torch.eye(dim, device=x0.device)
    shifts = torch.cat([E, -E], 0)  # (2 dim, dim)
    X0 = torch.cat([x0[None] + shifts[:, None], x0[None].expand(2 * dim, n, dim)], 0).reshape(-1, dim)
    X1 = torch.cat([x1[None].expand(2 * dim, n, dim), x1[None] + shifts[:, None]], 0).reshape(-1, dim)
    t = net.tau(X0, X1).reshape(2, 2 * dim, n).sum(1)
    tau = net.tau(x0, x1)
    return (t[0] - 2 * dim * tau) / h ** 2, (t[1] - 2 * dim * tau) / h ** 2


def tau_derivatives(net, X, laplacian=True, mode='jet', fd_h=2.0 / 128):
    """tau, grad tau (N, 2 dim), Laplacians w.r.t. x0 and x1 (0 if not requested). mode 'jet': forward-mode pass;
    'fd': exact gradient (jet) and central-difference Laplacians at scale fd_h; 'autograd': reverse-mode double
    backward (reference, 10-40x slower with the Laplacian)."""
    dim = net.dim
    if mode in ('jet', 'fd'):
        X = X.detach()
        tau, g, lap0, lap1 = net.tau_jet(X[:, :dim], X[:, dim:])
        if not laplacian:
            lap0 = lap1 = 0.0
        elif mode == 'fd':
            lap0, lap1 = fd_laplacians(net, X[:, :dim], X[:, dim:], fd_h)
        return X, tau, g, lap0, lap1
    X = X.detach().requires_grad_(True)
    tau = net.tau(X[:, :dim], X[:, dim:])
    g = torch.autograd.grad(tau.sum(), X, create_graph=True)[0]
    lap0 = lap1 = 0.0
    if laplacian:
        lap0 = sum(torch.autograd.grad(g[:, i].sum(), X, create_graph=True)[0][:, i] for i in range(dim))
        lap1 = sum(torch.autograd.grad(g[:, i].sum(), X, create_graph=True)[0][:, i] for i in range(dim, 2 * dim))
    return X, tau, g, lap0, lap1


def eikonal_loss(net, X, speed, gamma, laplacian=True, mode='jet', fd_h=2.0 / 128):
    """Mean of the P-NTFields residual  s_pred/s + s/s_pred - 2  at both end points of each pair (X: (N, 2 dim))."""
    dim = net.dim
    X, tau, g, lap0, lap1 = tau_derivatives(net, X, laplacian and bool(gamma), mode, fd_h)
    x0, x1 = X[:, :dim], X[:, dim:]
    D = x1 - x0
    T0 = (D * D).sum(1)
    DT0, DT1 = g[:, :dim], g[:, dim:]
    T3 = tau ** 2
    S0 = (T0 * (DT0 * DT0).sum(1) + 2 * tau * (DT0 * D).sum(1) + T3).clamp(min=1e-12)
    S1 = (T0 * (DT1 * DT1).sum(1) - 2 * tau * (DT1 * D).sum(1) + T3).clamp(min=1e-12)
    pred0 = 1 / (torch.sqrt(S0) / T3 + gamma * lap0)
    pred1 = 1 / (torch.sqrt(S1) / T3 + gamma * lap1)
    s0, s1 = speed[:, 0], speed[:, 1]
    diff = pred0 / s0 + s0 / pred0 + pred1 / s1 + s1 / pred1 - 4
    return diff.mean()


def descent_direction(net, X):
    """P-NTFields planner update for both end points: -grad T / |grad T|^2 (length = local speed), shape (N, 2 dim)."""
    dim = net.dim
    X = X.detach().requires_grad_(True)
    T = net.travel_time(X[:, :dim], X[:, dim:])
    g = torch.autograd.grad(T.sum(), X)[0]
    out = torch.empty_like(g)
    for sl in (slice(0, dim), slice(dim, 2 * dim)):
        gs = g[:, sl]
        out[:, sl] = -gs / (gs * gs).sum(1, keepdim=True).clamp(min=1e-12)
    return out.detach()
