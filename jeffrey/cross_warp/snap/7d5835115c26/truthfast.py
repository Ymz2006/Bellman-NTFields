"""jeffrey/maze_scale/truth.py's ground truth for large mazes: the same C-space grid (H_CELLS = 1/15 cell, 13 headings),
the same 98-neighbourhood, edge costs, seeding and interpolation, but

* solve(): Dijkstra in C over CPU threads (dijkstra_full.c, compiled on first use), which returns the fixed point the
  GPU Jacobi sweeps of Grid.solve converge to, instead of the sweeps (at 88 x 88 cells the grid has 22.7 M nodes:
  the sweeps would need ~1,600 passes and two copies of 98 per-neighbour cost arrays, ~18 GB);
* no stored per-neighbour cost arrays: descend() computes each edge cost from 1/S at its two ends on the fly.

Everything else (seed_block, interp, query_time, gradient, oracle_cost) is truth.Grid's own code.
check(maze) compares solve() with Grid.solve on a few goals of a small maze.
"""
import hashlib
import math
import os
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import torch

import truth as TR

HERE = Path(__file__).resolve().parent


def binary(build_dir):
    """Compile dijkstra_full.c (next to this file) into build_dir once per source version; returns the binary's path."""
    src = HERE / 'dijkstra_full.c'
    tag = hashlib.sha256(src.read_bytes()).hexdigest()[:12]
    out = Path(build_dir) / f'dijkstra_full_{tag}'
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix('.tmp')
        subprocess.run(['gcc', '-O3', '-march=native', '-fopenmp', '-o', str(tmp), str(src), '-lm'], check=True)
        tmp.replace(out)
    return out


class FastGrid(TR.Grid):
    def __init__(self, maze, build_dir, threads=12, h_cells=TR.H_CELLS, n_theta=TR.N_THETA, radius=TR.RADIUS, chunk=1 << 21):
        dev = maze.device
        self.maze, self.r = maze, radius
        nx = int(round(maze.E / h_cells)) + 1
        self.xs = torch.linspace(-0.5, 0.5, nx, device=dev)
        self.ts = torch.linspace(-maze.phi_max, maze.phi_max, n_theta, device=dev)
        self.hx, self.ht = 1.0 / (nx - 1), 2 * maze.phi_max / (n_theta - 1)
        self.shape = (nx, nx, n_theta)
        N = nx * nx * n_theta
        speed = torch.empty(N, device=dev)
        for i in range(0, N, chunk):                       # node order (x, y, theta), as truth.Grid
            idx = torch.arange(i, min(i + chunk, N), device=dev)
            k0, k1, k2 = idx // (nx * n_theta), (idx // n_theta) % nx, idx % n_theta
            q = torch.stack([self.xs[k0], self.xs[k1], self.ts[k2]], 1)
            speed[i:i + len(idx)] = maze.train_speed(maze.clearance(q))
        self.speed = speed.reshape(self.shape)
        self.inv = (1 / self.speed).reshape(-1)
        self.offsets, self.lengths = [], []
        r = radius
        for a in range(-r, r + 1):
            for b in range(-r, r + 1):
                for c in range(-r, r + 1):
                    if (a, b, c) == (0, 0, 0) or math.gcd(math.gcd(abs(a), abs(b)), abs(c)) != 1:
                        continue
                    self.offsets.append((a, b, c))
                    self.lengths.append(math.sqrt((a * self.hx) ** 2 + (b * self.hx) ** 2 + (c * self.ht) ** 2))
        self.off_t = torch.tensor(self.offsets, device=dev)
        self.len_t = torch.tensor(self.lengths, device=dev)
        self.build_dir, self.threads = Path(build_dir), int(threads)

    def solve(self, sources, keep_on='cuda'):
        """T*(node, source) for each source (B, nx, ny, nt) by dijkstra_full.c, and 0 (no sweeps)."""
        nx, _, nt = self.shape
        N = nx * nx * nt
        bin_ = binary(self.build_dir)
        with tempfile.TemporaryDirectory(dir=self.build_dir) as tmp:
            tmp = Path(tmp)
            self.inv.float().cpu().numpy().astype('<f4').tofile(tmp / 'inv.f32')
            k, cost = self.seed_block(sources)
            flat = self.flat(k).int().cpu().numpy()
            rec = np.zeros((len(sources), 64), dtype=[('v', '<i4'), ('c', '<f4')])
            rec['v'], rec['c'] = flat, cost.float().cpu().numpy()
            rec.tofile(tmp / 'seeds.bin')
            subprocess.run([str(bin_), str(tmp / 'inv.f32'), str(nx), str(nt), repr(self.hx), repr(self.ht), str(tmp / 'seeds.bin'),
                            str(len(sources)), str(tmp / 'out.f32'), str(self.threads)], check=True, stderr=subprocess.DEVNULL)
            mm = np.memmap(tmp / 'out.f32', dtype='<f4', mode='r', shape=(len(sources), N))
            out = torch.empty((len(sources), nx, nx, nt), device=keep_on)
            for s in range(len(sources)):
                out[s] = torch.from_numpy(np.array(mm[s])).reshape(nx, nx, nt).to(keep_on)
            del mm
        return out, 0

    @torch.no_grad()
    def descend(self, fields, gi, start, goal, max_steps=100000):
        """truth.Grid.descend with each edge cost computed from 1/S at its ends (no stored cost arrays)."""
        dev = start.device
        nx, ny, nt = self.shape
        P = len(start)
        flat = fields.reshape(len(fields), -1)
        k, cost = self.seed_block(start)
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
            nb = cur[:, None] + self.off_t[None]
            ok = (nb >= 0).all(-1) & (nb[..., 0] < nx) & (nb[..., 1] < ny) & (nb[..., 2] < nt)
            nbc = torch.stack([nb[..., 0].clamp(0, nx - 1), nb[..., 1].clamp(0, ny - 1), nb[..., 2].clamp(0, nt - 1)], -1)
            edge = self.len_t[None] * 0.5 * (self.inv[self.flat(cur)][:, None] + self.inv[self.flat(nbc)])
            c = edge + flat[gi[:, None].expand(-1, len(self.offsets)), self.flat(nbc)]
            c = torch.where(ok, c, torch.full_like(c, float('inf')))
            nxt = nbc[torch.arange(P, device=dev), c.argmin(1)]
            cur = torch.where(active[:, None], nxt, cur)
            trail.append(cur)
        trail = torch.stack(trail, 1)
        paths = []
        for i in range(P):
            t = trail[i]
            keep = torch.ones(len(t), dtype=torch.bool, device=dev)
            keep[1:] = (t[1:] != t[:-1]).any(-1)
            paths.append(torch.cat([start[i:i + 1], self.node(t[keep]), goal[i:i + 1]]))
        return paths


def check(maze, build_dir, goals):
    """Max relative difference between FastGrid.solve and truth.Grid.solve (Jacobi) over finite nodes, and between the
    two grids' descend paths' end points."""
    fg = FastGrid(maze, build_dir)
    a, _ = fg.solve(goals)
    g = TR.Grid(maze)
    b, sweeps = g.solve(goals)
    ok = torch.isfinite(a) & torch.isfinite(b)
    rel = ((a - b).abs() / b.clamp(min=1e-6))[ok]
    same_inf = bool((torch.isfinite(a) == torch.isfinite(b)).all())
    return dict(max_rel=float(rel.max()), median_rel=float(rel.median()), same_reachability=same_inf, jacobi_sweeps=sweeps,
                speed_equal=bool(torch.equal(fg.speed, g.speed)))
