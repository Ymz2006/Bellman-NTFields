"""The free-step Bellman backup of jeffrey/pair_lift (its recipe `bxt1nL`), as one self-contained module for the LBS trainer.

For each end q of a training pair (q, g):

    y = min over K straight steps q -> q' that stay out of obstacles of [ tau(q, q') + T(q', g) ]          (no gradient through y)
    term = mean (T(q, g) - y)^2

tau is the travel time of the step from the TRUE speed along it: its length x the mean of 1 / S, integrated with the antiderivative of
1 / S between samples (`Slowness`; this is the length of the lifted curve of the step). The clearance along the steps comes from a table
of the scene's exact clearance (`ClearTable`, 32 nodes per cell, 49 headings, trilinear). Candidates: two steps down the field's own
gradient (radius and half of it), `cone` more around that direction, the rest random; every candidate but the first two gets a length
log-uniform in [radius / 2, long x radius]. Pairs closer than 2 radius are left to the eikonal term.

Copied from jeffrey/pair_lift/pairnet.py (ClearTable) and pairtrain.py (Slowness, step_time, bellman_term), with the options this study
does not use removed (learned speed, pair inputs, one-sided / Huber variants).
"""
import numpy as np
import torch
from maze import ALPHA, MARGIN_CELLS, OFFSET_CELLS


def train_speed(d):
    s = (d / MARGIN_CELLS).clamp(OFFSET_CELLS / MARGIN_CELLS, 1.0)
    return ALPHA * s * s * (2 - s) * (2 - s) + 1 - ALPHA


def train_speed_np(d):
    s = np.clip(d / MARGIN_CELLS, OFFSET_CELLS / MARGIN_CELLS, 1.0)
    return ALPHA * s * s * (2 - s) * (2 - s) + 1 - ALPHA


class ClearTable:
    """Signed clearance (cells) of the scene on a regular grid over the C-space box, trilinear."""

    def __init__(self, scene, per_cell=32, nt=49, max_nodes=480_000_000):
        dev = scene.device
        per_cell = min(per_cell, max(8, int((max_nodes / nt) ** 0.5 / scene.E)))      # large scenes: fewer nodes per cell (GPU memory)
        self.per_cell = per_cell
        self.nx = int(round(scene.E * per_cell)) + 1
        self.nt = int(nt)
        xs = torch.linspace(-0.5, 0.5, self.nx, device=dev)
        ts = torch.linspace(-scene.phi_max, scene.phi_max, self.nt, device=dev)
        tab = torch.empty(self.nx, self.nx, self.nt, device=dev)
        rows = max(1, 400_000 // (self.nx * self.nt))
        for i in range(0, self.nx, rows):
            X, Y, Tt = torch.meshgrid(xs[i:i + rows], xs, ts, indexing='ij')
            tab[i:i + rows] = scene.clearance(torch.stack([X, Y, Tt], -1).reshape(-1, 3)).reshape(X.shape)
        self.tab = tab.clamp(max=2.0).reshape(-1).contiguous()
        self.lo, self.hi = scene.lo, scene.hi
        self.size = torch.tensor([self.nx - 1, self.nx - 1, self.nt - 1], device=dev, dtype=torch.float32)

    def __call__(self, q):
        c = (q - self.lo) / (self.hi - self.lo) * self.size
        c = torch.minimum(torch.maximum(c, torch.zeros_like(self.size)), self.size)
        k = torch.minimum(torch.floor(c.detach()), self.size - 1).long()
        f = c - k.to(c.dtype)
        base = (k[:, 0] * self.nx + k[:, 1]) * self.nt + k[:, 2]
        out = 0
        for dx in (0, 1):
            wx = f[:, 0] if dx else 1 - f[:, 0]
            for dy in (0, 1):
                wy = f[:, 1] if dy else 1 - f[:, 1]
                for dz in (0, 1):
                    wz = f[:, 2] if dz else 1 - f[:, 2]
                    out = out + wx * wy * wz * self.tab[base + (dx * self.nx + dy) * self.nt + dz]
        return out


class Slowness:
    """F(d) = int_d^margin dd' / S(d') (cells), continued linearly below the offset (floor speed) and above the margin (speed 1): the mean
    of 1 / S over a stretch along which the clearance runs linearly from d1 to d2 is (F(d1) - F(d2)) / (d2 - d1), exactly."""

    def __init__(self, device, nodes=16385):
        d = np.linspace(OFFSET_CELLS, MARGIN_CELLS, nodes)
        inv = 1 / train_speed_np(d)
        seg = 0.5 * (inv[1:] + inv[:-1]) * np.diff(d)
        F = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
        self.table = torch.tensor(F, device=device, dtype=torch.float64)
        self.nodes, self.inv_floor, self.F0 = nodes, float(inv[0]), float(F[0])

    def F(self, d):
        d = d.double()
        dc = d.clamp(OFFSET_CELLS, MARGIN_CELLS)
        u = (dc - OFFSET_CELLS) / (MARGIN_CELLS - OFFSET_CELLS) * (self.nodes - 1)
        i = u.floor().long().clamp(0, self.nodes - 2)
        w = u - i
        out = (1 - w) * self.table[i] + w * self.table[i + 1]
        out = torch.where(d < OFFSET_CELLS, self.F0 + (OFFSET_CELLS - d) * self.inv_floor, out)
        return torch.where(d > MARGIN_CELLS, MARGIN_CELLS - d, out)

    def mean(self, d1, d2):
        dd = (d2 - d1).double()
        flat = dd.abs() < 1e-5
        m = (self.F(d1) - self.F(d2)) / torch.where(flat, torch.ones_like(dd), dd)
        return torch.where(flat, 1 / train_speed(0.5 * (d1 + d2)).double(), m).float()


_slow = {}


def step_time(scene, a, b, samples=17, exact=True, clear=None, chunk=6_000_000):
    """Straight step a -> b (..., 3): its travel time at the true speed (length x mean of 1 / S) and whether it stays out of obstacles
    (clearance above -offset at every sample)."""
    shape = a.shape[:-1]
    ts = torch.linspace(0, 1, samples, device=a.device)
    fn = scene.clearance if clear is None else clear
    af, bf = a.reshape(-1, 3), b.reshape(-1, 3)
    rows = max(1, chunk // samples)
    ds = []
    for i in range(0, len(af), rows):
        p = af[i:i + rows, None, :] + ts[:, None] * (bf[i:i + rows] - af[i:i + rows])[:, None, :]
        ds.append(fn(p.reshape(-1, 3)).reshape(-1, samples))
    d = torch.cat(ds).reshape(*shape, samples)
    if exact:
        if a.device not in _slow:
            _slow[a.device] = Slowness(a.device)
        mean = _slow[a.device].mean(d[..., :-1], d[..., 1:]).mean(-1)
    else:
        inv = 1 / train_speed(d)
        mean = (0.5 * (inv[..., 1:] + inv[..., :-1])).mean(-1)
    return torch.linalg.vector_norm(b - a, dim=-1) * mean, d.amin(-1) > -OFFSET_CELLS


def bellman_term(net, scene, points, T, grads, radius, K, gen, cone=6, clear=None, long=8.0):
    """Mean squared backup residual over both ends of the pairs, the share of pair ends used, and the mean T - y."""
    N, dev = len(points), points.device
    total, used, gap = 0.0, 0.0, 0.0
    samples = int(min(17 * max(long, 1.0), 137))
    for end in (0, 1):
        sl = slice(3 * end, 3 * end + 3)
        other = slice(3 - 3 * end, 6 - 3 * end)
        a = points[:, sl]
        dirs = torch.randn(N, K, 3, generator=gen, device=dev)
        g = -grads[end]
        g = g / torch.linalg.vector_norm(g, dim=1, keepdim=True).clamp(min=1e-12)
        length = radius * (0.5 + 0.5 * torch.rand(N, K, 1, generator=gen, device=dev))
        dirs[:, 0] = g
        dirs[:, 1] = g
        length[:, 0] = radius
        length[:, 1] = 0.5 * radius
        if cone:
            dirs = dirs / torch.linalg.vector_norm(dirs, dim=2, keepdim=True).clamp(min=1e-12)
            dirs[:, 2:2 + cone] = g[:, None, :] + 0.5 * dirs[:, 2:2 + cone]
            dirs[:, 2] = g
            length[:, 2] = 0.25 * radius
            length[:, 3:2 + cone] = radius * (0.25 + 0.75 * torch.rand(N, cone - 1, 1, generator=gen, device=dev))
        dirs = dirs / torch.linalg.vector_norm(dirs, dim=2, keepdim=True).clamp(min=1e-12)
        if long > 1:
            lo, hi = np.log(0.5 * radius), np.log(long * radius)
            length[:, 2:] = torch.exp(lo + (hi - lo) * torch.rand(N, K - 2, 1, generator=gen, device=dev))
        with torch.no_grad():
            b = torch.minimum(torch.maximum(a[:, None, :] + dirs * length, scene.lo), scene.hi)
            tau, free = step_time(scene, a[:, None, :].expand(-1, K, -1), b, samples, clear=clear)
            new = points[:, None, :].repeat(1, K, 1)
            new[:, :, sl] = b
            Tn = net.out(new.reshape(-1, 6))[0][:, 0].reshape(N, K)
            cand = torch.where(free, tau + Tn, torch.full_like(Tn, float('inf')))
            y, _ = cand.min(1)
            ok = torch.isfinite(y) & (torch.linalg.vector_norm(a - points[:, other], dim=1) > 2 * radius)
        if bool(ok.any()):
            r = (T - y)[ok]
            total = total + (r ** 2).mean()
            gap += float(r.mean()) / 2
        used += float(ok.float().mean()) / 2
    return total, used, gap
