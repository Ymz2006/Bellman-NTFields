"""A diffeomorphism of ntrl-demo's embedding space, instead of a warp of the C-space.

ntrl-demo's travel time is T(x0, x1) = metric(enc(gamma(x0)), enc(gamma(x1))) with gamma(x) = [sin 2 pi B x, cos 2 pi B x] in R^256
(B: 128 x 3, N(0, 1) truncated at 2). The encoder is Lipschitz in gamma (row-normalized weights, sine activations), so how fast T can
change along a direction of the C-space is set by how much arc length the embedding has there: the Gram matrix J^T J of
J = d gamma / dx. For random rows it is (2 pi)^2 B^T B ~ (2 pi)^2 128 sigma^2 I, the same in every direction and the same everywhere:
the field has one resolution for the whole scene, too coarse at a wall and fine enough in the open.

Every earlier warp (jeffrey/cross_warp, line_warp, equal_runs) changed x -> phi(x) before gamma. A map of the 3-D C-space onto itself
can only stretch a region by stretching it along some direction more than others (a conformal stretch by a non-constant factor does not
exist in three dimensions), which is the anisotropy those studies ran into. Here the base embedding is kept and the map acts on the
embedding space E = R^256 x R^(2 m) x ... x R^(2 m):

    Gamma(x) = (gamma_0(x), gamma_1(x), ..., gamma_K(x))         gamma_k: m of ntrl-demo's 128 rows at 2^k (or any) times their scale
    Psi(u_0, u_1, ..., u_K) = (u_0, mu_1(u_0) u_1, ..., mu_K(u_0) u_K)        mu_k in [mu_min, 1], mu_min > 0

Psi is a coupling layer: triangular with a positive diagonal, hence a diffeomorphism of E for any smooth gate mu, and the network sees
Psi(Gamma(x)). Its induced metric on the C-space is

    J^T J = g_0 + sum_k mu_k^2 g_k + sum_k m |grad mu_k|^2 n n^T        g_k = (2 pi)^2 B_k^T B_k ~ (2 pi)^2 m sigma_k^2 I

(the cross terms vanish exactly: J_k^T gamma_k = sum_rows (cos sin - sin cos) = 0, and |gamma_k|^2 = m), i.e. an isotropic stretch
of the whole tangent space by the factor lambda(x) = sqrt(1 + sum_k mu_k(x)^2 m sigma_k^2 / (128 sigma_0^2)), plus one rank-1 term
along the gate's gradient that is small whenever the gate changes over a longer distance than the gated octave's wavelength
(measured by isotropy() below). Where the gate is open (near obstacles) the embedding has lambda times the arc length per unit of
C-space in every direction; where it is closed the octaves are compressed to mu_min and the network sees ntrl-demo's own low
frequencies (the 128 - K m base rows): no aliasing in the open, which is what sank the uniform magnifications (B x s) and the
compressing warps. The network's first residual block adds the input to a 256-wide layer, so the input stays 2 x 128 and the octaves
are a reassignment of ntrl-demo's rows (architecture and parameter count unchanged, 656,900).

Gates (the conditioner of the coupling layer; on the manifold any function of x is a function of u_0 = gamma_0(x), since gamma_0
is injective on the box):
    GridGate      a table w(x) in [0, 1] on a C-space grid, trilinear, fixed; filled from the geometry (gates.py: clearance band,
                  band-sample density, masked diffusion from the walls) or from a trained field's curvature. Not trained by T.
    LearnedGate   w = sigmoid(MLP(u_0)) (or of x), trained with the network by the travel-time loss (its own parameter group).
    ConstGate     mu = 1 everywhere: the extra octaves always on (the 'uniform multi-scale' control).
With no extra octaves the network is ntrl-demo's, bit for bit.
"""
import math
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = next((q for q in HERE.parents if (q / 'baselines/ntrl-demo').is_dir()), Path(os.environ.get('NTFIELD_ROOT', '/home/jeffrey/bellman-ntfield')))


def network_module():
    """ntrl-demo's models/metric network module (unchanged on disk)."""
    ntrl = ROOT / 'baselines/ntrl-demo'
    if str(ntrl) not in sys.path:
        sys.path.insert(0, str(ntrl))
    from models.metric import model_network_metric as mn
    assert Path(mn.__file__).resolve() == ntrl / 'models/metric/model_network_metric.py', 'ntrl-demo models package shadowed'
    return mn


def draw_rows(seed, rows, dim=3, scale=1.0):
    """rows x dim, N(0, 1) truncated at +-2 (ntrl-demo's distribution) times scale, from a private generator (the global RNG is untouched,
    so the network's own draws stay those of ntrl-demo)."""
    gen = torch.Generator().manual_seed(int(seed))
    out = torch.randn(rows, dim, generator=gen)
    for _ in range(20):
        bad = out.abs() > 2
        if not bool(bad.any()):
            break
        out[bad] = torch.randn(int(bad.sum()), generator=gen)
    return out.clamp(-2, 2) * scale


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


class GridGate(nn.Module):
    """mu(x) = mu_min + (1 - mu_min) w(x), w a (K, nx, ny, nt) table over the C-space box, trilinear with border clamp."""

    def __init__(self, K, shape, lo, hi, mu_min):
        super().__init__()
        self.K, self.mu_min = int(K), float(mu_min)
        self.register_buffer('table', torch.zeros(K, *shape))
        self.register_buffer('lo', torch.as_tensor(lo, dtype=torch.float32))
        self.register_buffer('hi', torch.as_tensor(hi, dtype=torch.float32))
        self.kind = 'grid'

    def w(self, x):
        nx, ny, nt = self.table.shape[1:]
        c = (x - self.lo) / (self.hi - self.lo) * torch.tensor([nx - 1, ny - 1, nt - 1], device=x.device, dtype=x.dtype)
        c = torch.stack([c[:, 0].clamp(0, nx - 1), c[:, 1].clamp(0, ny - 1), c[:, 2].clamp(0, nt - 1)], 1)
        k0 = torch.floor(c.detach()).long()
        k0 = torch.stack([k0[:, 0].clamp(max=nx - 2), k0[:, 1].clamp(max=ny - 2), k0[:, 2].clamp(max=nt - 2)], 1)
        f = c - k0.to(c.dtype)
        flat = self.table.reshape(self.K, -1)
        out = 0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    idx = ((k0[:, 0] + dx) * ny + (k0[:, 1] + dy)) * nt + (k0[:, 2] + dz)
                    wt = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                    out = out + wt[None] * flat[:, idx]
        return out.T                                                                    # (N, K)

    def forward(self, x, u0=None):
        return self.mu_min + (1 - self.mu_min) * self.w(x)


class LearnedGate(nn.Module):
    """mu = mu_min + (1 - mu_min) sigmoid(MLP(u_0 or x)); the last layer starts at zero, the bias at logit(init)."""

    def __init__(self, K, mu_min, inp='u0', in_dim=256, hidden=64, init=0.5):
        super().__init__()
        self.K, self.mu_min, self.inp = int(K), float(mu_min), inp
        d = in_dim if inp == 'u0' else 3
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, K))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, logit(init))
        self.kind = 'learn'

    def w(self, x, u0=None):
        return torch.sigmoid(self.net(u0 if self.inp == 'u0' else x))

    def forward(self, x, u0=None):
        return self.mu_min + (1 - self.mu_min) * self.w(x, u0)


class ConstGate(nn.Module):
    def __init__(self, K, value=1.0):
        super().__init__()
        self.K, self.value, self.mu_min = int(K), float(value), float(value)
        self.kind = 'const'

    def w(self, x, u0=None):
        return torch.full((x.shape[0], self.K), 1.0, device=x.device, dtype=x.dtype)

    def forward(self, x, u0=None):
        return torch.full((x.shape[0], self.K), self.value, device=x.device, dtype=x.dtype)


def embed_class(mn):
    class EmbedNN(mn.NN):
        """ntrl-demo's network, architecture and parameter count unchanged, with Psi(Gamma(x)) as its input mapping. The first residual
        block adds the input features to a 256-wide layer, so the input stays 2 x 128: the octaves are a reassignment of ntrl-demo's own
        rows. cfg: octaves (scales), rows (per octave); the first 128 - K rows rows keep ntrl-demo's frequency (times base_scale, default
        1), the remaining K groups of `rows` rows are multiplied by their octave's scale and gated. With no octaves the mapping is
        ntrl-demo's. The gate module is attached after init_weights (set_gate) and named embed_gate (the package's NN owns 'gate')."""

        def __init__(self, device, dim, B, cfg):
            super().__init__(device, dim, B)
            self.cfg = dict(cfg)
            self.octaves = [float(s) for s in cfg.get('octaves', [])]
            self.rows = int(cfg.get('rows', 16))
            self.K = len(self.octaves)
            self.base_scale = float(cfg.get('base_scale', 1.0))
            M = B.shape[0]
            self.base_rows = M - self.K * self.rows
            assert self.base_rows > 0, 'octave rows exceed the Fourier matrix'
            self.embed_gate = None
            self.in_dim = 2 * M
            scale = torch.full((M,), self.base_scale, device=B.device)
            group = torch.full((M,), -1, dtype=torch.long, device=B.device)   # -1: base row, k: octave k
            Bw = B.detach().double().clone()
            for k, sc in enumerate(self.octaves):
                i0 = self.base_rows + k * self.rows
                scale[i0:i0 + self.rows] = sc
                group[i0:i0 + self.rows] = k
                if cfg.get('whiten', True):
                    # whiten the octave's rows: R <- R C^-1/2 sqrt(tr C / dim), so R^T R = (tr C / dim) I exactly and the octave's induced
                    # metric is isotropic by construction (random directions of 16 rows in 3-D would otherwise spread its eigenvalues ~1.4x)
                    R = Bw[i0:i0 + self.rows]
                    C = R.T @ R
                    ev, V = torch.linalg.eigh(C)
                    Bw[i0:i0 + self.rows] = R @ (V @ torch.diag(ev.clamp(min=1e-12) ** -0.5) @ V.T) * math.sqrt(float(C.trace()) / C.shape[0])
            self.register_buffer('B_eff', (Bw.to(B.dtype).T * scale[None]).contiguous().to(device))      # (dim, M): rows (whitened) and scaled
            self.register_buffer('row_group', group.to(device))

        def set_gate(self, gate):
            self.embed_gate = gate

        def base_mapping(self, x):
            """ntrl-demo's own features of x (the unscaled rows): the gate's conditioner u_0 and the isotropy reference."""
            return super().input_mapping(x)

        def gates(self, x, base=None):
            if self.embed_gate is None:
                return torch.ones(x.shape[0], max(self.K, 1), device=x.device, dtype=x.dtype)
            return self.embed_gate(x, base if base is not None else self.base_mapping(x))

        def row_scale(self, x, base=None):
            """(N, M) multiplier of every row: 1 for base rows, mu_k(x) for octave k."""
            mu = self.gates(x, base)
            ones = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
            return torch.cat([ones, mu], 1)[:, self.row_group + 1]

        def input_mapping(self, x):
            if not self.K:
                return super().input_mapping(x)
            proj = x @ (2.0 * math.pi * self.B_eff)
            m = self.row_scale(x, self.base_mapping(x) if (self.embed_gate is not None and getattr(self.embed_gate, 'inp', None) == 'u0') else None)
            return torch.cat([torch.sin(proj) * m, torch.cos(proj) * m], dim=-1)
    return EmbedNN


def make_gate(kind, K, cfg, lo=None, hi=None, shape=None):
    if kind in ('none',) or K == 0:
        return None
    if kind == 'const':
        return ConstGate(K, cfg.get('const', 1.0))
    if kind == 'learn':
        return LearnedGate(K, cfg['mu_min'], cfg.get('gate_input', 'u0'), cfg.get('in_dim', 256), cfg.get('hidden', 64), cfg.get('init', 0.5))
    return GridGate(K, shape, lo, hi, cfg['mu_min'])


def build_network(mn, device, B, cfg):
    """Network as the trainer constructs it: package init first, then the gate module (never re-initialized by init_weights)."""
    net = embed_class(mn)(device, 3, B, cfg)
    net.apply(net.init_weights)
    net.to(device)
    gate_cfg = cfg.get('gate', {}) or {}
    kind = gate_cfg.get('kind', 'none')
    gate = make_gate(kind, net.K, gate_cfg, gate_cfg.get('lo'), gate_cfg.get('hi'), gate_cfg.get('shape'))
    if gate is not None:
        gate.to(device)
        net.set_gate(gate)
    return net


def load_field(checkpoint, device='cuda:0'):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    mn = network_module()
    net = build_network(mn, device, payload['B_state_dict'], payload['embed_config'])
    net.load_state_dict(payload['model_state_dict'], strict=True)
    net.to(torch.device(device)).float().eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


@torch.no_grad()
def isotropy(net, x, eps=2e-4, chunk=4096):
    """Metric the network's input sees at configurations x (N, 3): Gram J^T J of the full input mapping by central differences.
    Returns per point the stretch sqrt(trace / 3) relative to ntrl-demo's base features, the anisotropy sqrt(lambda_max / lambda_min)
    and the share of the arc length in the gated octaves."""
    out = dict(stretch=[], aniso=[], aniso_base=[], share=[])
    for i in range(0, len(x), chunk):
        q = x[i:i + chunk]
        cols, cols0 = [], []
        for j in range(3):
            e = torch.zeros(3, device=q.device, dtype=q.dtype)
            e[j] = eps
            fp, fm = net.input_mapping(q + e), net.input_mapping(q - e)
            cols.append((fp - fm) / (2 * eps))
            bp, bm = net.base_mapping(q + e), net.base_mapping(q - e)
            cols0.append((bp - bm) / (2 * eps))
        J = torch.stack(cols, 2)                                     # (N, D, 3)
        J0 = torch.stack(cols0, 2)
        G = J.transpose(1, 2) @ J
        G0 = J0.transpose(1, 2) @ J0
        ev = torch.linalg.eigvalsh(G.double()).clamp(min=1e-12)
        ev0 = torch.linalg.eigvalsh(G0.double()).clamp(min=1e-12)
        tr, tr0 = ev.sum(1), ev0.sum(1)
        out['stretch'].append(torch.sqrt(tr / tr0).float())
        out['aniso'].append(torch.sqrt(ev[:, -1] / ev[:, 0]).float())
        out['aniso_base'].append(torch.sqrt(ev0[:, -1] / ev0[:, 0]).float())
        out['share'].append(((tr - tr0) / tr).float())
    return {k: torch.cat(v) for k, v in out.items()}


def summarize(t):
    t = t.detach().float().cpu()
    return dict(median=float(t.median()), q10=float(t.quantile(0.1)), q90=float(t.quantile(0.9)), mean=float(t.mean()))
