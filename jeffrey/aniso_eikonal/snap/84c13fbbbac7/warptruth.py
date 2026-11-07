"""Ground truth of the warped maze: the travel time a network trained in z = phi(x) is asked to learn.

truthfast.FastGrid (maze_scale's C-space grid, 1/15 cell, 13 headings, 98-neighbourhood, edge cost = length x mean of
1/S at its ends, Dijkstra in C) with every length measured in z instead of x: dijkstra_warp.c takes the warp's
per-axis map at the grid nodes (a separable warp: z_x = F(x), z_y = F(y), z_p = s_p p), and the seed block's straight
segments are measured in z too. The speed S is the maze's (the robot's clearance in x), as in the warped-space training,
where only the coordinates are warped. With the identity warp this is FastGrid itself (check()).
"""
import hashlib
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import torch

import truthfast as TF

HERE = Path(__file__).resolve().parent


def binary(build_dir):
    src = HERE / 'dijkstra_warp.c'
    tag = hashlib.sha256(src.read_bytes()).hexdigest()[:12]
    out = Path(build_dir) / f'dijkstra_warp_{tag}'
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix('.tmp')
        subprocess.run(['gcc', '-O3', '-march=native', '-fopenmp', '-o', str(tmp), str(src), '-lm'], check=True)
        tmp.replace(out)
    return out


class WarpGrid(TF.FastGrid):
    def __init__(self, maze, warp, build_dir, threads=4, **kw):
        super().__init__(maze, build_dir, threads=threads, **kw)
        self.warp = warp
        with torch.no_grad():
            q = torch.stack([self.xs, self.xs, torch.zeros_like(self.xs)], 1)
            z = warp(q)
            assert torch.allclose(z[:, 0], z[:, 1]), 'WarpGrid needs the same map along x and y'
            self.zx = z[:, 0].float()
            qt = torch.stack([torch.zeros_like(self.ts), torch.zeros_like(self.ts), self.ts], 1)
            self.zt = warp(qt)[:, 2].float()
            assert bool((self.zx[1:] > self.zx[:-1]).all()) and bool((self.zt[1:] > self.zt[:-1]).all())

    def seed_block(self, src):
        """FastGrid.seed_block with the straight segments to the 4 x 4 x 4 block measured in z."""
        dev = src.device
        nx, ny, nt = self.shape
        base = torch.floor(self.index(src)).long()
        off = torch.stack(torch.meshgrid(*[torch.arange(-1, 3, device=dev)] * 3, indexing='ij'), -1).reshape(-1, 3)
        k = base[:, None] + off[None]
        ok = (k >= 0).all(-1) & (k[..., 0] < nx) & (k[..., 1] < ny) & (k[..., 2] < nt)
        kc = torch.stack([k[..., 0].clamp(0, nx - 1), k[..., 1].clamp(0, ny - 1), k[..., 2].clamp(0, nt - 1)], -1)
        s0 = 1 / self.maze.train_speed(self.maze.clearance(src))
        sn = self.speed[kc[..., 0], kc[..., 1], kc[..., 2]]
        with torch.no_grad():
            zs = self.warp(src)
        zn = torch.stack([self.zx[kc[..., 0]], self.zx[kc[..., 1]], self.zt[kc[..., 2]]], -1)
        cost = torch.linalg.vector_norm(zn - zs[:, None], dim=-1) * 0.5 * (s0[:, None] + 1 / sn)
        return kc, torch.where(ok, cost, torch.full_like(cost, float('inf')))

    def solve(self, sources, keep_on='cuda'):
        nx, _, nt = self.shape
        N = nx * nx * nt
        bin_ = binary(self.build_dir)
        with tempfile.TemporaryDirectory(dir=self.build_dir) as tmp:
            tmp = Path(tmp)
            self.inv.float().cpu().numpy().astype('<f4').tofile(tmp / 'inv.f32')
            self.zx.cpu().numpy().astype('<f4').tofile(tmp / 'zx.f32')
            self.zt.cpu().numpy().astype('<f4').tofile(tmp / 'zt.f32')
            k, cost = self.seed_block(sources)
            rec = np.zeros((len(sources), 64), dtype=[('v', '<i4'), ('c', '<f4')])
            rec['v'], rec['c'] = self.flat(k).int().cpu().numpy(), cost.float().cpu().numpy()
            rec.tofile(tmp / 'seeds.bin')
            subprocess.run([str(bin_), str(tmp / 'inv.f32'), str(nx), str(nt), str(tmp / 'zx.f32'), str(tmp / 'zt.f32'), str(tmp / 'seeds.bin'),
                            str(len(sources)), str(tmp / 'out.f32'), str(self.threads)], check=True, stderr=subprocess.DEVNULL)
            mm = np.memmap(tmp / 'out.f32', dtype='<f4', mode='r', shape=(len(sources), N))
            out = torch.empty((len(sources), nx, nx, nt), device=keep_on)
            for s in range(len(sources)):
                out[s] = torch.from_numpy(np.array(mm[s])).reshape(nx, nx, nt).to(keep_on)
            del mm
        return out, 0


def check(maze, build_dir, goals):
    """WarpGrid with the identity warp against FastGrid (max relative difference over finite nodes)."""
    import warpplan as WP
    a, _ = TF.FastGrid(maze, build_dir).solve(goals)
    b, _ = WarpGrid(maze, WP.Identity(), build_dir).solve(goals)
    ok = torch.isfinite(a) & torch.isfinite(b)
    return float(((a - b).abs() / a.clamp(min=1e-6))[ok].max())
