"""Ground-truth travel time T*(., goal) on a C-space grid of a maze, with the speed the metric trainer fits.

Nodes: x, y over [-0.5, 0.5] every H_CELLS cells (10 E + 1 intervals), phi over [-phi_max, phi_max] with N_THETA
nodes (theta every pi / (N_THETA - 1)); speed at a node = ALPHA s^2 (2 - s)^2 + 1 - ALPHA of the exact clearance
(maze.train_speed). Shortest paths over the 98-neighbourhood (index offsets in {-2..2}^3 with gcd 1; edge cost =
C-space length x mean of 1/S at both ends), relaxed in parallel (Jacobi) until no value changes by more than tol,
batched over goals (jeffrey/parking/truth.py, with the grid in cell units so every maze is resolved alike). The
C-space is the box (no theta wrap-around), as the network sees it. Interpolation is trilinear with border clamp.
descend() extracts a path by following the argmin neighbour from the start to the goal's seeded block.
"""
import math
import torch

H_CELLS, N_THETA, RADIUS = 1 / 15, 13, 2   # within 1 % (median) / 3 % (p99) of a 4x finer grid on the 4- and 7-cell mazes


class Grid:
    def __init__(self, maze, h_cells=H_CELLS, n_theta=N_THETA, radius=RADIUS):
        dev = maze.device
        self.maze, self.r = maze, radius
        nx = int(round(maze.E / h_cells)) + 1
        self.xs = torch.linspace(-0.5, 0.5, nx, device=dev)
        self.ts = torch.linspace(-maze.phi_max, maze.phi_max, n_theta, device=dev)
        self.hx, self.ht = 1.0 / (nx - 1), 2 * maze.phi_max / (n_theta - 1)
        self.shape = (nx, nx, n_theta)
        q = torch.stack(torch.meshgrid(self.xs, self.xs, self.ts, indexing='ij'), -1).reshape(-1, 3)
        d = maze.clearance(q)
        self.clear = d.reshape(self.shape)
        self.speed = maze.train_speed(d).reshape(self.shape)
        del q, d
        inv = 1 / self.speed
        r = radius
        pad = torch.nn.functional.pad(inv[None, None], (r, r, r, r, r, r), value=float('inf'))[0, 0]
        self.offsets, self.lengths, self.costs = [], [], []
        for a in range(-r, r + 1):
            for b in range(-r, r + 1):
                for c in range(-r, r + 1):
                    if (a, b, c) == (0, 0, 0) or math.gcd(math.gcd(abs(a), abs(b)), abs(c)) != 1:
                        continue
                    length = math.sqrt((a * self.hx) ** 2 + (b * self.hx) ** 2 + (c * self.ht) ** 2)
                    nb = pad[r + a:r + a + nx, r + b:r + b + nx, r + c:r + c + n_theta]
                    self.offsets.append((a, b, c))
                    self.lengths.append(length)
                    self.costs.append(length * 0.5 * (inv + nb))
        self.off_t = torch.tensor(self.offsets, device=dev)
        self.len_t = torch.tensor(self.lengths, device=dev)
        self.cost_flat = torch.stack([c.reshape(-1) for c in self.costs])        # (98, N), inf where the neighbour is off the grid

    def index(self, q):
        """Continuous node coordinates of q."""
        return torch.stack([(q[:, 0] + 0.5) / self.hx, (q[:, 1] + 0.5) / self.hx, (q[:, 2] + self.maze.phi_max) / self.ht], 1)

    def node(self, k):
        return torch.stack([self.xs[k[..., 0]], self.xs[k[..., 1]], self.ts[k[..., 2]]], -1)

    def flat(self, k):
        nx, ny, nt = self.shape
        return (k[..., 0] * ny + k[..., 1]) * nt + k[..., 2]

    def seed_block(self, src):
        """Nodes of the 4 x 4 x 4 block around each source and the straight-segment cost to them."""
        dev = src.device
        nx, ny, nt = self.shape
        base = torch.floor(self.index(src)).long()
        off = torch.stack(torch.meshgrid(*[torch.arange(-1, 3, device=dev)] * 3, indexing='ij'), -1).reshape(-1, 3)
        k = base[:, None] + off[None]
        ok = (k >= 0).all(-1) & (k[..., 0] < nx) & (k[..., 1] < ny) & (k[..., 2] < nt)
        kc = torch.stack([k[..., 0].clamp(0, nx - 1), k[..., 1].clamp(0, ny - 1), k[..., 2].clamp(0, nt - 1)], -1)
        s0 = 1 / self.maze.train_speed(self.maze.clearance(src))
        sn = self.speed[kc[..., 0], kc[..., 1], kc[..., 2]]
        cost = torch.linalg.vector_norm(self.node(kc) - src[:, None], dim=-1) * 0.5 * (s0[:, None] + 1 / sn)
        return kc, torch.where(ok, cost, torch.full_like(cost, float('inf')))

    def solve(self, sources, tol=1e-6, max_iter=20000, check_every=8):
        """T*(node, source) for each source, (B, nx, ny, nt), and the number of sweeps."""
        r, dev = self.r, sources.device
        nx, ny, nt = self.shape
        B = len(sources)
        T = torch.full((B, nx + 2 * r, ny + 2 * r, nt + 2 * r), float('inf'), device=dev)
        inner = T[:, r:r + nx, r:r + ny, r:r + nt]
        k, cost = self.seed_block(sources)
        bi = torch.arange(B, device=dev)[:, None].expand_as(cost)
        ok = torch.isfinite(cost)        # off-grid block entries are clamped onto boundary nodes: never write them
        inner[bi[ok], k[..., 0][ok], k[..., 1][ok], k[..., 2][ok]] = cost[ok]
        best = torch.empty_like(inner)
        prev = inner.clone()
        for it in range(max_iter):
            best.copy_(inner)
            for (a, b, c), cst in zip(self.offsets, self.costs):
                torch.minimum(best, T[:, r + a:r + a + nx, r + b:r + b + ny, r + c:r + c + nt] + cst, out=best)
            inner.copy_(best)
            if (it + 1) % check_every == 0:
                fin = torch.isfinite(inner)
                newly = bool((fin & ~torch.isfinite(prev)).any())
                change = float(torch.where(fin & torch.isfinite(prev), prev - inner, torch.zeros_like(inner)).max())
                if not newly and change < tol:
                    break
                prev.copy_(inner)
        return inner.clone(), it + 1

    def interp(self, fields, gi, q):
        """Trilinear T*(q) for query rows q (P, 3) on fields[gi] (fields (G, nx, ny, nt), gi (P,))."""
        nx, ny, nt = self.shape
        c = self.index(q)
        c = torch.stack([c[:, 0].clamp(0, nx - 1), c[:, 1].clamp(0, ny - 1), c[:, 2].clamp(0, nt - 1)], 1)
        k0 = torch.floor(c).long()
        k0 = torch.stack([k0[:, 0].clamp(max=nx - 2), k0[:, 1].clamp(max=ny - 2), k0[:, 2].clamp(max=nt - 2)], 1)
        w = c - k0
        flat = fields.reshape(len(fields), -1)
        out = torch.zeros(len(q), device=q.device, dtype=fields.dtype)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    kk = k0 + torch.tensor([dx, dy, dz], device=q.device)
                    wt = (w[:, 0] if dx else 1 - w[:, 0]) * (w[:, 1] if dy else 1 - w[:, 1]) * (w[:, 2] if dz else 1 - w[:, 2])
                    out = out + wt * flat[gi, self.flat(kk)]
        return out

    def oracle_cost(self, fields, gi, q, goals):
        """The oracle planner's T*(q): trilinear on the grid, and within two nodes of the goal (and inside the goal's
        clearance) also the straight segment's cost to the goal, |q - g| x mean of 1/S at both ends, whichever is lower.
        (The interpolated cone's minimum sits at the node nearest the goal, up to sqrt(3)/2 nodes = 0.058 cells away,
        farther than the MPPI tolerance on every maze; the straight term puts it at the goal.)"""
        t = self.interp(fields, gi, q)
        g = goals[gi]
        dist = torch.linalg.vector_norm(q - g, dim=1)
        dg = self.maze.clearance(g)
        near = (dist <= 2 * self.hx) & (1.5 * dist / self.maze.pitch <= dg)   # robot points move <= ~1.5 x the C-space step
        if bool(near.any()):
            sq = self.maze.train_speed(self.maze.clearance(q[near]))
            direct = dist[near] * 0.5 * (1 / sq + 1 / self.maze.train_speed(dg[near]))
            t = t.clone()
            t[near] = torch.minimum(t[near], direct)
        return t

    def query_time(self, fields, gi, q):
        """T*(q) as the grid would extend it to an off-grid point: min over the 4 x 4 x 4 block around q of the straight
        segment's cost to a node plus T*(node) (no interpolation across slow nodes next to q)."""
        k, cost = self.seed_block(q)
        flat = fields.reshape(len(fields), -1)
        return (cost + flat[gi[:, None].expand_as(cost), self.flat(k)]).amin(1)

    def gradient(self, fields, gi, q, h=None):
        """grad T* at q by central differences of the interpolated field (step h, default half a node)."""
        h = 0.5 * self.hx if h is None else h
        e = torch.eye(3, device=q.device)
        scale = torch.tensor([1.0, 1.0, self.ht / self.hx], device=q.device)
        g = []
        for j in range(3):
            step = e[j] * h * scale[j]
            g.append((self.interp(fields, gi, q + step) - self.interp(fields, gi, q - step)) / (2 * h * scale[j]))
        return torch.stack(g, 1)

    @torch.no_grad()
    def descend(self, fields, gi, start, goal, max_steps=100000):
        """Grid path from start to goal on fields[gi] (argmin neighbour), as a list of (L, 3) tensors (start ... goal)."""
        dev = start.device
        nx, ny, nt = self.shape
        P = len(start)
        flat = fields.reshape(len(fields), -1)
        k, cost = self.seed_block(start)                                   # first node: straight cost + T*
        tot = cost + flat[gi[:, None].expand_as(cost), self.flat(k)]
        cur = k[torch.arange(P, device=dev), tot.argmin(1)]
        gk, _ = self.seed_block(goal)
        trail = [cur]
        active = torch.ones(P, dtype=torch.bool, device=dev)
        for _ in range(max_steps):
            near = (cur[:, None] == gk).all(-1).any(1)
            active = active & ~near
            if not bool(active.any()):
                break
            nb = cur[:, None] + self.off_t[None]                            # (P, 98, 3)
            ok = (nb >= 0).all(-1) & (nb[..., 0] < nx) & (nb[..., 1] < ny) & (nb[..., 2] < nt)
            nbc = torch.stack([nb[..., 0].clamp(0, nx - 1), nb[..., 1].clamp(0, ny - 1), nb[..., 2].clamp(0, nt - 1)], -1)
            c = self.cost_flat[:, self.flat(cur)].T + flat[gi[:, None].expand(-1, len(self.offsets)), self.flat(nbc)]
            c = torch.where(ok, c, torch.full_like(c, float('inf')))
            nxt = nbc[torch.arange(P, device=dev), c.argmin(1)]
            cur = torch.where(active[:, None], nxt, cur)
            trail.append(cur)
        trail = torch.stack(trail, 1)                                       # (P, L, 3)
        paths = []
        for i in range(P):
            t = trail[i]
            keep = torch.ones(len(t), dtype=torch.bool, device=dev)
            keep[1:] = (t[1:] != t[:-1]).any(-1)
            paths.append(torch.cat([start[i:i + 1], self.node(t[keep]), goal[i:i + 1]]))
        return paths
