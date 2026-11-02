"""Ground-truth travel time T*(., source) on a C-space grid, with the speed the metric trainer fits.

Grid x, y in [-0.5, 0.5] at hx = hy = 0.005 and theta/2pi in [-0.12, 0.12] (+-43 deg) at 0.01; the speed at each
node is ALPHA s^2 (2 - s)^2 + 1 - ALPHA of the exact clearance (scene.train_speed). Shortest paths over the
98-neighbourhood (index offsets in {-2..2}^3 with gcd 1; edge cost = length x mean of 1/S at both ends), relaxed
in parallel (Jacobi) until no value changes by more than 1e-6, batched over sources. The theta window is a
restriction (paths that rotate further are not represented); optimal paths here translate at theta ~ 0.
Interpolation is trilinear (grid_sample, align_corners=True, border clamp).
"""
import math
import torch
import torch.nn.functional as F

HX, HT, THETA_MAX, RADIUS = 0.005, 0.01, 0.12, 2


class Grid:
    def __init__(self, scene):
        dev = scene.device
        self.scene = scene
        self.xs = torch.linspace(-0.5, 0.5, int(round(1 / HX)) + 1, device=dev)
        self.ts = torch.linspace(-THETA_MAX, THETA_MAX, int(round(2 * THETA_MAX / HT)) + 1, device=dev)
        self.shape = (len(self.xs), len(self.xs), len(self.ts))
        q = torch.stack(torch.meshgrid(self.xs, self.xs, self.ts, indexing='ij'), -1).reshape(-1, 3)
        d, _ = scene.clearance(q)
        self.clear = d.reshape(self.shape)
        self.speed = scene.train_speed(d).reshape(self.shape)
        inv = 1 / self.speed
        r = RADIUS
        pad = F.pad(inv[None, None], (r, r, r, r, r, r), value=float('inf'))[0, 0]
        nx, ny, nt = self.shape
        self.offsets, self.costs = [], []
        for a in range(-r, r + 1):
            for b in range(-r, r + 1):
                for c in range(-r, r + 1):
                    if (a, b, c) == (0, 0, 0) or math.gcd(math.gcd(abs(a), abs(b)), abs(c)) != 1:
                        continue
                    length = math.sqrt((a * HX) ** 2 + (b * HX) ** 2 + (c * HT) ** 2)
                    nb = pad[r + a:r + a + nx, r + b:r + b + ny, r + c:r + c + nt]
                    self.offsets.append((a, b, c))
                    self.costs.append(length * 0.5 * (inv + nb))

    def index(self, q):
        return torch.stack([(q[:, 0] + 0.5) / HX, (q[:, 1] + 0.5) / HX, (q[:, 2] + THETA_MAX) / HT], 1)

    def solve(self, sources, tol=1e-6, max_iter=3000):
        """T*(node, source) for each source pose, (B, nx, ny, nt), and the number of sweeps."""
        r, dev = RADIUS, sources.device
        nx, ny, nt = self.shape
        B = len(sources)
        T = torch.full((B, nx + 2 * r, ny + 2 * r, nt + 2 * r), float('inf'), device=dev)
        inner = T[:, r:r + nx, r:r + ny, r:r + nt]
        # seed the 4x4x4 nodes around each source with the straight-segment cost
        base = torch.floor(self.index(sources)).long()
        off = torch.stack(torch.meshgrid(*[torch.arange(-1, 3, device=dev)] * 3, indexing='ij'), -1).reshape(-1, 3)
        k = base[:, None] + off[None]                                          # (B, 64, 3)
        ok = (k >= 0).all(-1) & (k[..., 0] < nx) & (k[..., 1] < ny) & (k[..., 2] < nt)
        bi = torch.arange(B, device=dev)[:, None].expand_as(ok)[ok]
        k = k[ok]
        node = torch.stack([self.xs[k[:, 0]], self.xs[k[:, 1]], self.ts[k[:, 2]]], 1)
        s0 = 1 / self.scene.train_speed(self.scene.clearance(sources)[0])
        cost = torch.linalg.vector_norm(node - sources[bi], dim=1) * 0.5 * (s0[bi] + 1 / self.speed[k[:, 0], k[:, 1], k[:, 2]])
        inner[bi, k[:, 0], k[:, 1], k[:, 2]] = cost
        for it in range(max_iter):
            best = inner.clone()
            for (a, b, c), cost in zip(self.offsets, self.costs):
                torch.minimum(best, T[:, r + a:r + a + nx, r + b:r + b + ny, r + c:r + c + nt] + cost, out=best)
            fin = torch.isfinite(inner)
            newly = bool((~fin & torch.isfinite(best)).any())
            change = float((inner[fin] - best[fin]).max()) if bool(fin.any()) else 0.0
            inner.copy_(best)
            if not newly and change < tol:
                break
        return inner.clone(), it + 1

    def interp(self, field, q):
        """Trilinear T*(q) from fields (B, nx, ny, nt) at q (B, P, 3) -> (B, P)."""
        g = torch.stack([q[..., 2] / THETA_MAX, 2 * q[..., 1], 2 * q[..., 0]], -1)   # (W=theta, H=y, D=x)
        out = F.grid_sample(field[:, None], g[:, None, None], mode='bilinear', padding_mode='border', align_corners=True)
        return out[:, 0, 0, 0]

    def descent(self, field, q, h=0.004):
        """-grad T* at q by central differences of the interpolated field (theta step h as well)."""
        e = torch.eye(3, device=q.device) * h
        grads = []
        for j in range(3):
            plus = self.interp(field, (q + e[j])[:, None])[:, 0]
            minus = self.interp(field, (q - e[j])[:, None])[:, 0]
            grads.append((plus - minus) / (2 * h))
        return -torch.stack(grads, 1)
