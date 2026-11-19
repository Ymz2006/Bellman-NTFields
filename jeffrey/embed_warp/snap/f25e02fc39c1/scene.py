"""Scenes with jeffrey/maze_scale's Maze interface, so its sampler (data.py), truth (truth.py), planner and scorer (plan.py)
and evaluation (evaluate.py) run unchanged on them: the same SE(2) rectangle robot (0.5 x 0.2 cells, heading phi = RHO theta / E),
the same ntrl-demo speed label (margin 0.175 cells), normalized configurations q = (x, y, phi) in [-0.5, 0.5]^2 x [-phi_max, phi_max].

    Disks    a box of n x n cells with `count` disks (radii in cells) that do not overlap and leave a gap of at least `gap` cells
             between any two; the box frame is an obstacle. Exact signed clearance: the signed distance of each disk centre to the
             robot rectangle minus the radius (the rectangle inflated by a disk is a rounded rectangle), and the rectangle's corners
             to the frame.
    Gibson   P-NTFields' scanned apartment (baselines/P-NTFields/datasets/gibson/<idx>/mesh_z_up_scaled.off) as a 2-D floor plan:
             occupancy at `res` x `res` (jeffrey/gibson_embed/grid.py, conservative rasterization), the largest free component is free
             space, signed distance by distance transforms (box units), bilinear table. The robot's clearance is the minimum of the
             point signed distance over a lattice of points on the rectangle (9 x 5): exact up to half the lattice spacing (0.03 cells),
             the same function for the labels, the truth, the planner and the scorer. A 'cell' is 1 / E of the box (E = 64 by default:
             rooms are a few cells wide, the scan walls a quarter of a cell thick).

load_scene(prepared) rebuilds the scene a prepare run was made on from its maze.json: a maze_scale maze (plain, plus or gap variant,
installing the variant patch as cross_warp's pipeline does) or one of the scenes above (spec in maze.json['scene']).
"""
import json
import math
import os
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = next((q for q in HERE.parents if (q / 'baselines/ntrl-demo').is_dir()), Path(os.environ.get('NTFIELD_ROOT', '/home/jeffrey/bellman-ntfield')))
if str(ROOT / 'jeffrey/maze_scale') not in sys.path:
    sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))          # maze, data, truth, plan, evaluate, common, plus
if str(ROOT / 'jeffrey/cross_warp') not in sys.path:
    sys.path.append(str(ROOT / 'jeffrey/cross_warp'))             # gap (gap variants only); appended: its trainer.py must not shadow ours
import maze as MZ  # noqa: E402
from maze import MARGIN_CELLS, OFFSET_CELLS, ALPHA, RHO, ROBOT_L, ROBOT_W, REF_N, T_WALL  # noqa: E402


class SceneBase:
    """Everything maze_scale needs from a 'maze', given signed_cells(q) (differentiable signed clearance in cells)."""
    kind = 'scene'

    def _setup(self, n, E, device, dtype=torch.float32):
        self.n, self.E = int(n), float(E)
        self.pitch = 1.0 / self.E
        self.phi_max = RHO * (math.pi / 2) / self.E
        self.margin = MARGIN_CELLS * self.pitch
        self.offset = OFFSET_CELLS * self.pitch
        self.device, self.dtype = device, dtype
        self.lo = torch.tensor([-0.5, -0.5, -self.phi_max], device=device, dtype=dtype)
        self.hi = torch.tensor([0.5, 0.5, self.phi_max], device=device, dtype=dtype)

    def to_cells(self, q):
        return q[:, 0] * self.E + self.n / 2, q[:, 1] * self.E + self.n / 2, q[:, 2] * self.E / RHO

    def from_cells(self, x, y, th):
        return torch.stack([(x - self.n / 2) / self.E, (y - self.n / 2) / self.E, th * RHO / self.E], -1)

    def planner_scale(self):
        return self.pitch / (1.0 / (REF_N + T_WALL))

    def clearance(self, q, chunk=65536):
        with torch.no_grad():
            return torch.cat([self.signed_cells(q[i:i + chunk]) for i in range(0, len(q), chunk)]) if len(q) else q.new_zeros(0)

    def clearance_normal(self, q, chunk=65536):
        ds, ns = [], []
        for i in range(0, len(q), chunk):
            x = q[i:i + chunk].detach().clone().to(self.dtype).requires_grad_(True)
            with torch.enable_grad():
                d = self.signed_cells(x)
                g, = torch.autograd.grad(d.sum(), x)
            ds.append(d.detach())
            ns.append(torch.nn.functional.normalize(g, dim=1))
        return torch.cat(ds), torch.cat(ns)

    unit_speed = staticmethod(MZ.Maze.unit_speed)
    train_speed = staticmethod(MZ.Maze.train_speed)

    def uniform(self, count, gen):
        return self.lo + (self.hi - self.lo) * torch.rand(count, 3, generator=gen, device=self.device, dtype=self.dtype)

    def free(self, count, gen, min_clear=0.0, batch=262144):
        out, have = [], 0
        while have < count:
            q = self.uniform(batch, gen)
            q = q[self.clearance(q) >= min_clear]
            out.append(q)
            have += len(q)
        return torch.cat(out)[:count]

    def inside(self, q):
        return ((q >= self.lo) & (q <= self.hi)).all(1)

    def describe(self):
        return dict(n=self.n, E_cells=self.E, pitch=self.pitch, phi_max=self.phi_max, margin=self.margin, offset=self.offset, robot=[ROBOT_L, ROBOT_W],
                    rho=RHO, planner_scale=self.planner_scale(), variant=self.kind, scene=self.spec())


def rect_frame(p, th, hl, hw):
    """Robot corners (N, 4, 2) in cell units."""
    c, s = torch.cos(th), torch.sin(th)
    u = torch.stack([c, s], -1)
    v = torch.stack([-s, c], -1)
    return torch.stack([p + a * hl * u + b * hw * v for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))], 1)


class Disks(SceneBase):
    kind = 'disks'

    def __init__(self, n=22, count=40, r_lo=0.5, r_hi=1.4, gap=0.7, seed=5, device='cuda:0', dtype=torch.float32, disks=None):
        self.count, self.r_lo, self.r_hi, self.gap, self.seed = int(count), float(r_lo), float(r_hi), float(gap), int(seed)
        self._setup(n, n, device, dtype)
        if disks is None:
            disks = self.layout(self.n, self.count, self.r_lo, self.r_hi, self.gap, self.seed)
        self.disks = np.asarray(disks, dtype=np.float64)                        # (K, 3): centre x, centre y, radius, cells
        self.ct = torch.tensor(self.disks[:, :2], device=device, dtype=dtype)
        self.rt = torch.tensor(self.disks[:, 2], device=device, dtype=dtype)

    @staticmethod
    def layout(n, count, r_lo, r_hi, gap, seed):
        """Rejection sampling: disks inside the box with `gap` cells to the frame and between surfaces."""
        rng = np.random.default_rng(seed)
        out = []
        tries = 0
        while len(out) < count and tries < 200_000:
            tries += 1
            r = rng.uniform(r_lo, r_hi)
            c = rng.uniform(r + gap, n - r - gap, 2)
            if all(np.hypot(*(c - np.array(o[:2]))) >= r + o[2] + gap for o in out):
                out.append([c[0], c[1], r])
        assert len(out) == count, f'only {len(out)} disks placed'
        return out

    def signed_cells(self, q):
        x, y, th = self.to_cells(q.to(self.dtype))
        p = torch.stack([x, y], -1)
        c, s = torch.cos(th), torch.sin(th)
        rel = self.ct[None] - p[:, None]                                        # (N, K, 2) centre relative to the robot
        a = rel[..., 0] * c[:, None] + rel[..., 1] * s[:, None]                 # along the robot's length
        b = -rel[..., 0] * s[:, None] + rel[..., 1] * c[:, None]
        ea = a.abs() - ROBOT_L / 2
        eb = b.abs() - ROBOT_W / 2
        outside = torch.sqrt(ea.clamp(min=0) ** 2 + eb.clamp(min=0) ** 2 + 1e-12)
        insided = torch.maximum(ea, eb).clamp(max=0)                           # negative inside the rectangle
        sd = torch.where((ea > 0) | (eb > 0), outside, insided) - self.rt[None]  # signed distance of the centre to the rounded rectangle
        d_disks = sd.amin(1)
        corners = rect_frame(p, th, ROBOT_L / 2, ROBOT_W / 2)                   # frame: every corner inside the box
        d_frame = torch.minimum(torch.minimum(corners[..., 0], self.n - corners[..., 0]), torch.minimum(corners[..., 1], self.n - corners[..., 1])).amin(1)
        return torch.minimum(d_disks, d_frame)

    def spec(self):
        return dict(kind='disks', n=self.n, count=self.count, r_lo=self.r_lo, r_hi=self.r_hi, gap=self.gap, seed=self.seed, disks=self.disks.tolist())

    def describe(self):
        d = super().describe()
        area = float((math.pi * self.disks[:, 2] ** 2).sum() / self.n ** 2)
        d.update(disks=self.count, disk_area_fraction=area, radius_range=[float(self.disks[:, 2].min()), float(self.disks[:, 2].max())])
        return d


class Gibson(SceneBase):
    kind = 'gibson'
    LATTICE = (9, 5)

    def __init__(self, idx=1, res=1024, E=64, z=0.0, device='cuda:0', dtype=torch.float32, table=None):
        self.idx, self.res, self.z = int(idx), int(res), float(z)
        self._setup(E, E, device, dtype)
        if table is None:
            table = self.build_table(self.idx, self.res, self.z)
        self.table = torch.as_tensor(table, device=device, dtype=dtype)         # (res, res) signed distance, box units, index [i (x), j (y)]
        self.free_share = float((self.table > 0).float().mean())
        a = torch.linspace(-ROBOT_L / 2, ROBOT_L / 2, self.LATTICE[0], device=device, dtype=dtype)
        b = torch.linspace(-ROBOT_W / 2, ROBOT_W / 2, self.LATTICE[1], device=device, dtype=dtype)
        A, Bq = torch.meshgrid(a, b, indexing='ij')
        self.lat = torch.stack([A.reshape(-1), Bq.reshape(-1)], 1)             # (L, 2) robot-frame offsets, cells

    @staticmethod
    def mesh_path(idx):
        return ROOT / f'baselines/P-NTFields/datasets/gibson/{idx}/mesh_z_up_scaled.off'

    @staticmethod
    def build_table(idx, res, z):
        import importlib.util
        spec = importlib.util.spec_from_file_location('gibson_grid', ROOT / 'jeffrey/gibson_embed/grid.py')
        G = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(G)
        from scipy import ndimage
        mesh = G.load_mesh(Gibson.mesh_path(idx))
        occ, _ = G.build(mesh, 2, res, z)
        labels, sizes = G.components(~occ)
        free = labels == (1 + int(np.argmax(sizes)))
        d = (ndimage.distance_transform_edt(free) - ndimage.distance_transform_edt(~free)) / res
        return d.astype(np.float32)

    def point_sdf_cells(self, xy):
        """Bilinear signed distance (cells) at normalized xy (N, 2); outside the box the border value continues minus the overshoot."""
        G = self.res
        u = (xy + 0.5) * (G - 1)
        uc = u.clamp(0, G - 1 - 1e-4)
        i = uc.floor().long()
        f = uc - i
        i1 = (i + 1).clamp(max=G - 1)
        t = self.table
        v = ((1 - f[:, 0]) * (1 - f[:, 1]) * t[i[:, 0], i[:, 1]] + f[:, 0] * (1 - f[:, 1]) * t[i1[:, 0], i[:, 1]]
             + (1 - f[:, 0]) * f[:, 1] * t[i[:, 0], i1[:, 1]] + f[:, 0] * f[:, 1] * t[i1[:, 0], i1[:, 1]])
        over = ((u - uc).abs().sum(1)) / (G - 1)
        return (v - over) * self.E

    def signed_cells(self, q):
        x, y, th = self.to_cells(q.to(self.dtype))
        c, s = torch.cos(th), torch.sin(th)
        px = x[:, None] + self.lat[None, :, 0] * c[:, None] - self.lat[None, :, 1] * s[:, None]
        py = y[:, None] + self.lat[None, :, 0] * s[:, None] + self.lat[None, :, 1] * c[:, None]
        xy = torch.stack([(px - self.n / 2) / self.E, (py - self.n / 2) / self.E], -1).reshape(-1, 2)
        return self.point_sdf_cells(xy).reshape(len(q), -1).amin(1)

    def spec(self):
        return dict(kind='gibson', idx=self.idx, res=self.res, E=self.E, z=self.z, lattice=list(self.LATTICE), mesh=str(self.mesh_path(self.idx).relative_to(ROOT)))

    def describe(self):
        d = super().describe()
        d.update(free_share=self.free_share)
        return d


def build(spec, device='cuda:0'):
    """A scene from its spec dict (maze.json['scene'])."""
    kind = spec['kind']
    if kind == 'disks':
        return Disks(spec['n'], spec['count'], spec['r_lo'], spec['r_hi'], spec['gap'], spec['seed'], device, disks=spec.get('disks'))
    if kind == 'gibson':
        return Gibson(spec['idx'], spec['res'], spec['E'], spec.get('z', 0.0), device)
    raise ValueError(kind)


def load_scene(prepared, device='cuda:0'):
    """(scene, maze.json) of a prepare run: maze_scale maze (plain / plus / gap) or an embed_warp scene."""
    info = json.load(open(Path(prepared) / 'maze.json'))
    if 'scene' in info:
        scene = build(info['scene'], device)
        assert scene.n == info['n'], 'prepared scene differs'
        return scene, info
    variant = info.get('variant', 'plain')
    if variant == 'gap':
        import gap
        gap.install(info['gap_scale'], gap.base_name(info))
    elif variant == 'plus':
        import plus
        plus.install(info['plus_width'])
    maze = MZ.Maze(info['n'], device)
    assert maze.describe()['walls'] == info['walls'] and np.array_equal(maze.V.astype(int), np.array(info['V'])) and np.array_equal(maze.H.astype(int), np.array(info['H'])), 'prepared maze differs'
    return maze, info
