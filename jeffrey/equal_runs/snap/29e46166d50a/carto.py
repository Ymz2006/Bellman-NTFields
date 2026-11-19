"""Equal runs as a density: a cartogram of the obstacle boundary.

Along one line, 'every run has the same length' says the crossings are evenly spaced: the cumulative number of crossings grows linearly, which is histogram equalization of the crossings. Over all lines the crossing rate at a point is, by
Crofton's formula, the density of obstacle boundary there. So the same statement for a whole box in any dimension is: push the boundary density to uniform. That is a density-equalizing map, a cartogram in the sense of Gastner and Newman (2004), with obstacle boundary as the population:

    rho(x, 0) = eps + blur_h(boundary density of the scene, mean 1)
    d rho / d t = laplacian rho                     Neumann on the box: solved exactly mode by mode in a cosine (Fourier) basis
    v = - grad rho / rho                            the velocity that carries mass along with the diffusion
    d x / d t = v(x, t)                             from t = 0 to the time rho is flat: phi(x) = x(t_end), and psi = phi^-1 integrates back

No training and nothing to tune against the planner. What it needs from a scene is only crossing points of random lines, which the collision oracle gives in any dimension. Properties that the line losses have to be forced into:
a lattice is already uniform once blurred, so it stays near the identity; v is a gradient field at every time, so there is no rotation to find; boundary density is blurred past each wall, so nothing is compressed right next to one; a smooth flow cannot fold, and the box maps onto itself.

    eps  share of open space: the boundary gets 1 / (1 + eps) of the z box, open space eps / (1 + eps)
    h    blur, in cells: structure finer than h is left alone
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as Fn

import scan as S


@torch.no_grad()
def boundary_density(scene, G, lines=1 << 16, samples=2048, seed=31, directions='any', frame=1.0, frame_weight=0.0):
    """Histogram (G, ... , G) of crossing points of random lines through the identity, over all slices, mean 1. Only the oracle is used.
    Crossings within `frame` cells of a face of the box count `frame_weight` (default: left out): every line ends at the box anyway, so its frame is not structure to resolve, and on an open scene its long perimeter would otherwise take most of the z box."""
    dev = scene.table.v.device
    gen = torch.Generator(dev).manual_seed(seed)
    d = scene.dim
    H = torch.zeros(G ** d, device=dev)
    done = 0
    while done < lines:
        n = min(8192, lines - done)
        r = S.any_rays(n, d, scene.planes, gen, dev) if directions == 'any' else S.lines_uniform(n, d, scene.planes, gen, dev)
        line, t, _ = S.crossings(lambda z: z, scene, r, samples)
        x = S.at(r, line, t)
        w = torch.where(x.abs().amax(1) < 0.5 - frame / scene.E, torch.ones_like(x[:, 0]), torch.full_like(x[:, 0], frame_weight))
        i = ((x + 0.5) * G).long().clamp(0, G - 1)
        idx = torch.zeros(len(x), device=dev, dtype=torch.long)
        for a in range(d):
            idx = idx * G + i[:, a]
        H.index_add_(0, idx, w)
        done += n
    H = H.view(*([G] * d))
    return H / H.mean().clamp(min=1e-12)


def even(a):
    """Even extension of a (G, ..., G) to (2G, ..., 2G): its Fourier series is the cosine series with no flux through the faces."""
    for k in range(a.dim()):
        a = torch.cat([a, a.flip(k)], k)
    return a


class Diffusion:
    """rho(., t) and v(., t) on the cell-centred grid."""

    def __init__(self, rho):
        self.d, self.G = rho.dim(), rho.shape[0]
        self.F = torch.fft.fftn(even(rho))
        G2 = 2 * self.G
        f = torch.fft.fftfreq(G2, d=1.0 / self.G, device=rho.device)        # cycles per unit length of the box: the mirrored box has length 2
        self.w = [2 * math.pi * f.view(*([1] * a + [-1] + [1] * (self.d - 1 - a))) for a in range(self.d)]
        self.w2 = sum(w ** 2 for w in self.w)

    def crop(self, a):
        return a[tuple(slice(0, self.G) for _ in range(self.d))]

    def blurred(self, t):
        return self.crop(torch.fft.ifftn(self.F * torch.exp(-self.w2 * t)).real)

    def velocity(self, t):
        """(d, G, ..., G): v_a = - d_a rho / rho at time t, spectral derivative."""
        E = self.F * torch.exp(-self.w2 * t)
        rho = self.crop(torch.fft.ifftn(E).real).clamp(min=1e-9)
        return torch.stack([-self.crop(torch.fft.ifftn(E * (1j * w)).real) / rho for w in self.w])


def sample(field, x):
    """field (C, G, ..., G) at points x (N, d) in [-0.5, 0.5]^d, multilinear, cell-centred. grid_sample's coordinates run over the last axis first."""
    d = x.shape[1]
    g = (2 * x).flip(1).view(1, *([1] * (d - 1)), -1, d)
    out = Fn.grid_sample(field[None], g, mode='bilinear', padding_mode='border', align_corners=False)
    return out.reshape(field.shape[0], -1).T


def times(t0, t1, steps):
    """0, then log-spaced from t0 to t1: the field's length scale grows like sqrt(t), so equal steps in log t keep the update's stretch small."""
    r = torch.logspace(math.log10(t0), math.log10(t1), steps).tolist()
    return [0.0] + r


def wall_safe(v, x, G):
    """No flow through a face: the normal velocity falls to zero over the last grid cell."""
    return v * ((0.5 - x.abs()) * G).clamp(0, 1)


def advect(D, x, ts, reverse=False, chunk=1 << 21):
    """Points x carried from ts[0] to ts[-1], or back. Midpoint in time, RK2 in space."""
    x = x.clone()
    G = D.G
    pairs = list(zip(ts[:-1], ts[1:]))
    if reverse:
        pairs = [(b, a) for a, b in reversed(pairs)]
    for a, b in pairs:
        dt = b - a
        v = D.velocity(0.5 * (a + b))
        for i in range(0, len(x), chunk):
            c = x[i:i + chunk]
            m = (c + 0.5 * dt * wall_safe(sample(v, c), c, G)).clamp(-0.5, 0.5)
            x[i:i + chunk] = (c + dt * wall_safe(sample(v, m), m, G)).clamp(-0.5, 0.5)
    return x


class TableMap(nn.Module):
    """A map held as tables on a node grid of the box: psi (z -> x) and phi (x -> z), multilinear between nodes."""
    cfg = dict(kind='table')

    def __init__(self, psi, phi):
        super().__init__()
        self.register_buffer('psi_table', psi)
        self.register_buffer('phi_table', phi)

    @staticmethod
    def read(table, p):
        d = p.shape[1]
        g = (2 * p).flip(1).view(1, *([1] * (d - 1)), -1, d)
        out = Fn.grid_sample(table[None], g, mode='bilinear', padding_mode='border', align_corners=True)
        return out.reshape(table.shape[0], -1).T

    def forward(self, z):
        return self.read(self.psi_table, z)

    def inverse(self, x):
        return self.read(self.phi_table, x)

    def window(self, alpha):
        pass


def nodes(N, d, dev):
    t = torch.linspace(-0.5, 0.5, N, device=dev)
    g = torch.meshgrid(*([t] * d), indexing='ij')
    return torch.stack([a.reshape(-1) for a in g], 1)


def build(scene, G=512, N=1024, h=1.0, eps=1.0, lines=1 << 16, samples=2048, steps=320, t_end=1.0, directions='any', frame=1.0, log=print, frame_weight=0.0):
    """The cartogram of a scene. G: density grid; N: nodes of the output tables; h: blur in cells."""
    dev = scene.table.v.device
    d = scene.dim
    H = boundary_density(scene, G, lines, samples, directions=directions, frame=frame, frame_weight=frame_weight)
    D0 = Diffusion(H)
    hb = h / scene.E                                           # a Gaussian of standard deviation hb is diffusion for time hb^2 / 2
    B = D0.blurred(0.5 * hb * hb).clamp(min=0)
    rho = eps + B / B.mean()
    D = Diffusion(rho)
    ts = times(0.02 * hb * hb, t_end, steps)
    X = nodes(N, d, dev)
    psi = advect(D, X, ts, reverse=True)
    phi = advect(D, X, ts)
    shape = [N] * d
    m = TableMap(psi.T.reshape(d, *shape).contiguous(), phi.T.reshape(d, *shape).contiguous()).to(dev)
    with torch.no_grad():
        z = torch.rand(1 << 18, d, device=dev) - 0.5
        err = (m.inverse(m(z)) - z).norm(dim=1)
    log(f'cartogram: rho range {float(rho.min()):.2f} .. {float(rho.max()):.1f}, round trip error median {float(err.median()) * scene.E:.3f} cells, q99 {float(err.quantile(0.99)) * scene.E:.3f} cells')
    return m, rho
