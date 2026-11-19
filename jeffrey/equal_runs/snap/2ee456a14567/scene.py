"""Scenes as a collision oracle: the only thing the fit knows about obstacles.

A scene is a box [-0.5, 0.5]^dim and a signed clearance sdf(x, k) (> 0: free, < 0: obstacle), with k the index of a slice when the
scene is a family of slices that share one map (planes of a heading). Nothing here is a polygon or a mesh, so the same fit runs on a 2-D
floor plan, a 3-D workspace, or a C-space: line scans (scan.py) only ever call sdf and its gradient. A robot arm's clearance from its collision
checker has the same shape.

The maze scenes tabulate maze.Maze.clearance (exact robot-to-wall signed distance, cells) on a grid and interpolate, only to save time: the zero set
of the table agrees with the exact oracle to a fraction of a grid step (checked in checks.py).

    MazePlanes  dim 2: the maze's C-space at `headings` fixed headings (slices); one map of the plane is shared by all of them
    MazeSE2     dim 3: the whole C-space (x, y, theta), theta in [-pi/2, pi/2] on the third box axis
    Boxes3D     dim 3: a workspace of axis-aligned or rotated boxes (exact signed distance on a grid)
"""
import math
import os
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = next((q for q in HERE.parents if (q / 'baselines/ntrl-demo').is_dir()), Path(os.environ.get('NTFIELD_ROOT', '/home/jeffrey/bellman-ntfield')))
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
sys.path.insert(0, str(ROOT / 'jeffrey/cross_warp'))


def make_maze(spec, dev):
    """'plain:10', 'gap:maze10:12' (10 x 10 quadrants, band 6 k), 'gap:cross22:4', 'plus:22:6'."""
    kind, *rest = spec.split(':')
    if kind == 'gap':
        import gap
        base, k = rest[0], int(rest[1])
        gap.install(k, base)
        import maze as MZ
        return MZ.Maze(2 * gap.BASES[base]['block'] + gap.BASE_WIDTH * k, dev)
    if kind == 'plus':
        import plus
        plus.install(int(rest[1]))
        import maze as MZ
        return MZ.Maze(int(rest[0]), dev)
    import maze as MZ
    return MZ.Maze(int(rest[0]), dev)


class Table:
    """Multilinear interpolation of a table over [-0.5, 0.5]^dim, one table per slice: values (K, G, ..., G)."""

    def __init__(self, values, dim):
        self.dim, self.K, self.G = dim, values.shape[0], values.shape[1]
        self.v = values.reshape(-1).contiguous()
        self.step = 1.0 / (self.G - 1)
        self.stride = [self.G ** (dim - 1 - a) for a in range(dim)]

    def __call__(self, x, k):
        G = self.G
        u = ((x + 0.5) * (G - 1)).clamp(0, G - 1 - 1e-3)
        i = u.floor().long()
        f = u - i
        base = k * (G ** self.dim)
        for a in range(self.dim):
            base = base + i[:, a] * self.stride[a]
        out = torch.zeros(len(x), device=x.device, dtype=x.dtype)
        for corner in range(1 << self.dim):
            w = torch.ones_like(out)
            idx = base
            for a in range(self.dim):
                if corner >> a & 1:
                    w = w * f[:, a]
                    idx = idx + self.stride[a]
                else:
                    w = w * (1 - f[:, a])
            out = out + w * self.v[idx]
        return out


class Scene:
    dim = 2
    planes = 1
    margin = 0.175                                              # the speed label's margin, in the oracle's units (cells)

    def sdf(self, x, k):
        return self.table(x, k)

    def grad(self, x, k):
        """Central difference of the table, one grid step wide: the direction of the obstacle's normal at a crossing."""
        h = self.table.step
        cols = []
        for a in range(self.dim):
            e = torch.zeros(self.dim, device=x.device, dtype=x.dtype)
            e[a] = h
            cols.append(self.table(x + e, k) - self.table(x - e, k))
        return torch.stack(cols, 1) / (2 * h)

    def plane_of(self, count, gen, dev):
        return torch.randint(self.planes, (count,), generator=gen, device=dev)


class MazePlanes(Scene):
    dim = 2

    def __init__(self, maze, headings=9, res=32, chunk=1 << 19):
        self.maze, self.n, self.E, self.headings = maze, maze.n, maze.E, headings
        self.planes = headings
        self.thetas = [(-0.5 + i / headings) * math.pi for i in range(headings)]
        dev = maze.device
        G = int(round(res * maze.E)) + 1
        t = torch.linspace(-0.5, 0.5, G, device=dev)
        X, Y = torch.meshgrid(t, t, indexing='ij')
        xy = torch.stack([X.reshape(-1), Y.reshape(-1)], 1)
        out = torch.empty(headings, G * G, device=dev)
        for k, th in enumerate(self.thetas):
            for i in range(0, len(xy), chunk):
                c = xy[i:i + chunk]
                q = torch.cat([c, torch.full((len(c), 1), th * 0.25 / maze.E, device=dev)], 1)      # third coordinate: RHO theta / E
                out[k, i:i + chunk] = maze.clearance(q)
        self.table = Table(out.view(headings, G, G), 2)

    def exact(self, x, k):
        """maze.clearance at normalized positions and slice indices (no table)."""
        th = torch.tensor(self.thetas, device=x.device, dtype=x.dtype)[k]
        return self.maze.clearance(torch.cat([x, (th * 0.25 / self.E)[:, None]], 1))


class MazeSE2(Scene):
    """The whole C-space of the maze: third box axis = heading, -0.5 .. 0.5 is -pi/2 .. pi/2."""
    dim = 3

    def __init__(self, maze, res=12, res_theta=None, chunk=1 << 19):
        self.maze, self.n, self.E = maze, maze.n, maze.E
        dev = maze.device
        G = int(round(res * maze.E)) + 1
        t = torch.linspace(-0.5, 0.5, G, device=dev)
        vol = torch.empty(G, G, G, device=dev)
        X, Y = torch.meshgrid(t, t, indexing='ij')
        xy = torch.stack([X.reshape(-1), Y.reshape(-1)], 1)
        for j, th in enumerate((t * math.pi).tolist()):
            vals = []
            for i in range(0, len(xy), chunk):
                c = xy[i:i + chunk]
                vals.append(maze.clearance(torch.cat([c, torch.full((len(c), 1), th * 0.25 / maze.E, device=dev)], 1)))
            vol[:, :, j] = torch.cat(vals).view(G, G)
        self.table = Table(vol[None], 3)

    def exact(self, x, k=None):
        return self.maze.clearance(torch.cat([x[:, :2], (x[:, 2:3] * math.pi) * 0.25 / self.E], 1))


class Gibson(Scene):
    """A scanned building (P-NTFields' Gibson meshes): a 2-D floor plan at height 0 or a 3-D voxelization. The largest free component is free space, everything else obstacle, and a ball robot of `radius` box units inflates it.
    Signed distance by distance transforms: the same oracle interface as a maze, from a mesh."""

    def __init__(self, idx, dim, res, dev, radius=0.008):
        sys.path.insert(0, str(ROOT / 'jeffrey/gibson_embed'))
        import grid as G
        from scipy import ndimage
        self.dim, self.res = dim, res
        mesh = G.load_mesh(ROOT / f'baselines/P-NTFields/datasets/gibson/{idx}/mesh_z_up_scaled.off')
        occ, _ = G.build(mesh, dim, res, 0.0)
        labels, sizes = G.components(~occ)
        free = labels == (1 + int(np.argmax(sizes)))
        free[tuple([0] * dim)] = free[tuple([0] * dim)]
        d = (ndimage.distance_transform_edt(free) - ndimage.distance_transform_edt(~free)) / res - radius
        self.free_share = float(free.mean())
        self.table = Table(torch.tensor(d, device=dev, dtype=torch.float32)[None], dim)
        self.E = self.n = 64.0 if dim == 2 else 32.0                     # a nominal cell, for the Fourier scales and knots
        self.margin = 0.175 / self.E


class Boxes3D(Scene):
    """A 3-D workspace: a closed room with box obstacles of any orientation. boxes: list of (centre (3,), half (3,), rotation (3, 3)); the free space
    is the room minus the boxes, inflated by `radius` (a ball robot)."""
    dim = 3

    def __init__(self, boxes, dev, res=128, radius=0.0, wall=0.02):
        t = torch.linspace(-0.5, 0.5, res + 1, device=dev)
        X, Y, Z = torch.meshgrid(t, t, t, indexing='ij')
        p = torch.stack([X, Y, Z], -1).reshape(-1, 3)
        d = 0.5 - wall - p.abs().amax(1)                                 # inside the room: positive
        for c, h, R in boxes:
            c, h, R = [torch.as_tensor(v, device=dev, dtype=p.dtype) for v in (c, h, R)]
            q = ((p - c) @ R).abs() - h
            out = torch.linalg.norm(q.clamp(min=0), dim=1) + q.amax(1).clamp(max=0)
            d = torch.minimum(d, out)
        self.table = Table((d - radius).view(1, res + 1, res + 1, res + 1), 3)
        self.E, self.n = 1.0 * res / 8, res / 8                          # a nominal 'cell' of 8 grid steps
        self.margin = 0.175 / self.E


class Rotated(Scene):
    """A plane scene turned by `deg` degrees inside the box and shrunk to fit: everything outside the turned square is obstacle. Its corridors and bands are no longer along the axes,
    so a separable map cannot follow them and a shear has to."""
    dim = 2

    def __init__(self, inner, deg, chunk=1 << 20):
        self.inner, self.deg = inner, deg
        self.planes, self.E, self.n = inner.planes, inner.E, inner.n
        self.maze = inner.maze
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        self.shrink = 1.0 / (abs(c) + abs(s))
        G = inner.table.G
        dev = inner.table.v.device
        t = torch.linspace(-0.5, 0.5, G, device=dev)
        X, Y = torch.meshgrid(t, t, indexing='ij')
        xy = torch.stack([X.reshape(-1), Y.reshape(-1)], 1)
        R = torch.tensor([[c, s], [-s, c]], device=dev)
        p = (xy @ R.T) / self.shrink                          # inner coordinates of each box point
        out = torch.empty(self.planes, G * G, device=dev)
        beyond = (p.abs().amax(1) - 0.5).clamp(min=0) * self.E
        for k in range(self.planes):
            kk = torch.full((len(p),), k, device=dev, dtype=torch.long)
            v = torch.cat([inner.sdf(p[i:i + chunk], kk[i:i + chunk]) for i in range(0, len(p), chunk)])
            out[k] = torch.where(beyond > 0, torch.minimum(v, -beyond), v)
        self.table = Table(out.view(self.planes, G, G), 2)

    def to_inner(self, x):
        """Box coordinates -> the unturned maze's coordinates. Points of the box outside the turned square land outside the maze, where the maze reads as obstacle."""
        a = math.radians(self.deg)
        R = torch.tensor([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]], device=x.device, dtype=x.dtype)
        return (x @ R.T) / self.shrink

    def exact(self, x, k):
        return self.inner.exact(self.to_inner(x), k)


def build(spec, dev, headings=9, res=32):
    """'<maze spec>' (planes), 'rot<deg>/<maze spec>' (planes, turned), 'se2/<maze spec>' (whole C-space), 'rooms:<seed>' (3-D boxes)."""
    if spec.startswith('se2/'):
        return MazeSE2(make_maze(spec[4:], dev), res)
    if spec.startswith('rot'):
        head, tail = spec.split('/', 1)
        return Rotated(MazePlanes(make_maze(tail, dev), headings, res), float(head[3:]))
    if spec.startswith('rooms:'):
        return rooms(int(spec.split(':')[1]), dev, res)
    if spec.startswith('gibson:'):
        _, idx, dim = spec.split(':')
        return Gibson(int(idx), int(dim), 512 if dim == '2' else 128, dev)
    return MazePlanes(make_maze(spec, dev), headings, res)


def rooms(seed, dev, res=128, count=24):
    """A 3-D scene that no axis-aligned or separable map can straighten: `count` thin slabs at random orientations in a closed room."""
    rng = np.random.default_rng(seed)
    boxes = []
    for _ in range(count):
        c = rng.uniform(-0.38, 0.38, 3)
        h = np.array([rng.uniform(0.06, 0.22), rng.uniform(0.06, 0.22), 0.012])
        q = rng.normal(size=(3, 3))
        R, _ = np.linalg.qr(q)
        boxes.append((c, h, R))
    return Boxes3D(boxes, dev, res)
