"""Iterated free-space ball averaging on an occupancy grid (the embedding under test) and the stencil machinery.

phi_{k+1}(x) = mean of phi_k over { x + o : |o| <= r, x + o free [and the line x -> x + o free if los] }, for free x.
`masked=False` removes the free-space condition (plain box blur, the control). Snapshots at several iteration counts
give a multi-scale embedding (diffusion length ~ sqrt(k) cells). Every channel is standardized over free cells.
"""
import itertools
import math
import numpy as np
import torch
import torch.nn.functional as F


def ball_offsets(dim, r):
    span = range(-int(math.floor(r)), int(math.floor(r)) + 1)
    return [o for o in itertools.product(span, repeat=dim) if sum(c * c for c in o) <= r * r + 1e-9]


def line_cells(offset):
    """Cells strictly between the origin and `offset` that the centre-to-centre segment may visit.

    Unit-box offsets use the whole box (conservative for diagonals); longer offsets use the sampled nearest-cell line."""
    o = np.asarray(offset)
    m = int(np.abs(o).max())
    if m == 0:
        return []
    if m == 1:
        cells = set(itertools.product(*[range(min(0, int(c)), max(0, int(c)) + 1) for c in o]))
    else:
        cells = {tuple(int(v) for v in np.floor(t * o + 0.5)) for t in np.linspace(0, 1, 8 * m + 1)}
    cells.discard(tuple(int(v) for v in np.zeros_like(o)))
    cells.discard(tuple(int(c) for c in o))
    return sorted(cells)


def shifted(padded, offset, shape, pad):
    """View of padded[..., x + offset] for x over `shape` (padded by `pad` on every side)."""
    sl = tuple(slice(pad + o, pad + o + n) for o, n in zip(offset, shape))
    return padded[(...,) + sl]


def pad(t, p, value=0.0):
    if t.dtype == torch.bool:
        return F.pad(t.to(torch.uint8), [p, p] * t.ndim, value=0).bool()
    return F.pad(t, [p, p] * t.ndim, value=value)


class Stencil:
    """Offsets of a ball with their validity masks m_o(x) = support(x + o) [and the line x -> x + o inside support]."""

    def __init__(self, support, r, los=False, dtype=torch.float32):
        self.shape = tuple(support.shape)
        self.dim = support.ndim
        self.offsets = ball_offsets(self.dim, r)
        self.pad = max(1, int(math.ceil(r)))
        self.device = support.device
        sp = pad(support, self.pad)
        masks = []
        for o in self.offsets:
            m = shifted(sp, o, self.shape, self.pad)
            if los:
                for c in line_cells(o):
                    m = m & shifted(sp, c, self.shape, self.pad)
            masks.append(m)
        self.masks = torch.stack(masks).to(dtype)  # (O, *shape)
        self.support = support.to(dtype)

    def pad_field(self, phi, value=0.0):
        return F.pad(phi, [self.pad, self.pad] * self.dim, value=value)

    def shifted(self, padded, offset):
        return shifted(padded, offset, self.shape, self.pad)


class Averager:
    def __init__(self, free, r=1.0, los=False, masked=True, dtype=torch.float32):
        support = free if masked else torch.ones_like(free)
        self.stencil = Stencil(support, r, los=los, dtype=dtype)
        self.denominator = self.stencil.masks.sum(0).clamp(min=1.0)
        self.keep = self.stencil.support
        self.free = free

    def step(self, phi):
        padded = self.stencil.pad_field(phi)
        out = torch.zeros_like(phi)
        for o, m in zip(self.stencil.offsets, self.stencil.masks):
            out += m * self.stencil.shifted(padded, o)
        return out / self.denominator * self.keep

    def run(self, phi, snapshots, orthogonalize_every=0, callback=None):
        """Iterate; return {k: standardized phi_k} for k in snapshots (and the final state)."""
        out = {}
        last = max(snapshots)
        for k in range(1, last + 1):
            phi = self.step(phi)
            if orthogonalize_every and k % orthogonalize_every == 0:
                phi = self.orthogonalize(phi)
            if k in snapshots:
                out[k] = self.standardize(phi)
            if callback and k % 512 == 0:
                callback(k)
        return out, phi

    def orthogonalize(self, phi):
        flat = phi[:, self.free].T  # (N_free, D)
        q, _ = torch.linalg.qr(flat)
        new = torch.zeros_like(phi)
        new[:, self.free] = q.T * math.sqrt(flat.shape[0])
        return new

    def standardize(self, phi):
        vals = phi[:, self.free]
        mean = vals.mean(1, keepdim=True)
        std = vals.std(1, keepdim=True).clamp(min=1e-8)
        out = torch.zeros_like(phi)
        out[:, self.free] = (vals - mean) / std
        return out


def cell_centres(shape, device):
    axes = [(torch.arange(n, device=device, dtype=torch.float32) + 0.5) / n - 0.5 for n in shape]
    return torch.stack(torch.meshgrid(*axes, indexing='ij'))  # (dim, *shape)


def initial(kind, free, channels, seed, sigma=0.5):
    """Initial field phi_0 (channels, *shape), zero outside free."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    shape = tuple(free.shape)
    dim = free.ndim
    if kind == 'random':
        phi = torch.randn((channels,) + shape, generator=g).to(free.device)
    elif kind == 'coords':
        phi = cell_centres(shape, free.device)
    elif kind == 'fourier':
        B = sigma * torch.randn((dim, channels // 2), generator=g).to(free.device)
        x = cell_centres(shape, free.device).reshape(dim, -1).T
        proj = 2 * math.pi * x @ B
        phi = torch.cat([torch.sin(proj), torch.cos(proj)], 1).T.reshape((channels,) + shape)
    else:
        raise ValueError(kind)
    return phi * free.to(phi.dtype)


def embed(free, kind='random', r=1.0, los=False, masked=True, snapshots=(8, 32, 128, 512, 2048, 8192), channels=16, seed=0,
          spectral=False, sigma=0.5, log=print):
    """Multi-scale averaged embedding: concatenation over snapshots of standardized phi_k (channels each).

    kind 'coords' stores phi_k - x (the centroid shift). spectral=True orthogonalizes every 16 steps and keeps the last snapshot only."""
    av = Averager(free, r=r, los=los, masked=masked)
    phi = initial(kind, free, channels, seed, sigma=sigma)
    if kind == 'coords':
        x = phi.clone()
        snaps = {}
        for k in range(1, max(snapshots) + 1):
            phi = av.step(phi)
            if k in snapshots:
                snaps[k] = av.standardize((phi - x) * free)
            if k % 512 == 0:
                log(f'  iteration {k}')
    else:
        snaps, _ = av.run(phi, set(snapshots), orthogonalize_every=16 if spectral else 0, callback=lambda k: log(f'  iteration {k}'))
    keys = [max(snapshots)] if spectral else sorted(snaps)
    feat = torch.cat([snaps[k] for k in keys], 0)
    return feat, keys
