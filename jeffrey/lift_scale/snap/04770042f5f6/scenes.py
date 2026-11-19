"""Scenes of the lift-scale study with jeffrey/maze_scale's Maze interface (the SE(2) rectangle robot of every local maze study).

    House    a Gibson house (~/Downloads/gibson_tiny/<name>/mesh_z_up.obj, metres, z up) as a floor plan: the mesh is cut by horizontal
             planes at the heights `zs` (metres), the cut segments are rasterized conservatively (jeffrey/gibson_embed/grid.py) on a
             res x res grid over a square box that holds the plan with `pad` metres around it. Scans have holes (glass, open outer
             doors), so the inside is found on the occupancy dilated by `seal_m` metres (gaps narrower than 2 seal_m are closed): the
             largest free component of the dilated map that does not touch the box border, grown back by seal_m and cut by the
             undilated occupancy, is free space; everything else (the outside, wall interiors, rooms reached only through such a gap)
             is an obstacle. Signed distance by distance transforms, bilinear table. One cell is `cell_m` metres, so E = box side / cell_m grows with the
             house: a larger house is the same world with more cells (as in jeffrey/maze_scale). Robot clearance = minimum of the
             point signed distance over the 9 x 5 lattice on the rectangle (embed_warp's Gibson scene).

load_scene(prepared) rebuilds any scene a prepare run was made on: a House (maze.json['scene']['kind'] == 'house'), otherwise whatever
jeffrey/embed_warp/scene.py knows (maze_scale mazes, plus / gap variants, disks, the P-NTFields apartment).
"""
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = next((q for q in HERE.parents if (q / 'baselines/ntrl-demo').is_dir()), Path(os.environ.get('NTFIELD_ROOT', '/home/jeffrey/bellman-ntfield')))
if str(ROOT / 'jeffrey/embed_warp') not in sys.path:
    sys.path.append(str(ROOT / 'jeffrey/embed_warp'))                 # appended: this study's modules come first
import scene as SC  # noqa: E402  (embed_warp/scene.py; puts maze_scale at the front of sys.path)

GIBSON_TINY = Path(os.environ.get('GIBSON_TINY', str(Path.home() / 'Downloads/gibson_tiny')))


def read_obj(path):
    """Vertices (V, 3) and triangles (F, 3) of a triangle .obj (v / f lines only; faces may carry /vt/vn indices)."""
    vs, fs = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                a = line.split()
                vs.append((float(a[1]), float(a[2]), float(a[3])))
            elif line.startswith('f '):
                a = [int(t.split('/')[0]) - 1 for t in line.split()[1:]]
                for k in range(1, len(a) - 1):
                    fs.append((a[0], a[k], a[k + 1]))
    return np.asarray(vs, np.float64), np.asarray(fs, np.int64)


def plane_segments(v, f, z):
    """2-D segments (S, 2, 2) where the triangles cross the plane at height z."""
    t = v[f]                                            # (F, 3, 3)
    s = t[:, :, 2] - z
    cross = (s.min(1) < 0) & (s.max(1) > 0)
    t, s = t[cross], s[cross]
    segs = []
    pts = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        m = (s[:, a] * s[:, b]) < 0
        w = s[:, a] / np.where(m, s[:, a] - s[:, b], 1.0)
        p = t[:, a, :2] + w[:, None] * (t[:, b, :2] - t[:, a, :2])
        pts.append((m, p))
    out = np.zeros((len(t), 2, 2))
    fill = np.zeros(len(t), int)
    for m, p in pts:
        idx = np.nonzero(m & (fill < 2))[0]
        out[idx, fill[idx]] = p[idx]
        fill[idx] += 1
    return out[fill == 2]


class House(SC.SceneBase):
    kind = 'house'
    LATTICE = (9, 5)

    def __init__(self, name, cell_m=0.25, zs=(1.0,), res=None, pad=0.5, seal_m=0.15, device='cuda:0', dtype=torch.float32, table=None, side_m=None, centre=None):
        self.name, self.cell_m, self.zs, self.pad, self.seal_m = str(name), float(cell_m), [float(z) for z in zs], float(pad), float(seal_m)
        if table is None:
            table, side_m, centre, res = self.build_table(self.name, self.cell_m, self.zs, res, self.pad, self.seal_m)
        self.side_m, self.centre, self.res = float(side_m), [float(c) for c in centre], int(table.shape[0])
        E = int(round(self.side_m / self.cell_m))
        self._setup(E, E, device, dtype)
        self.table = torch.as_tensor(table, device=device, dtype=dtype)         # signed distance, box units, [i (x), j (y)]
        self.free_share = float((self.table > 0).float().mean())
        a = torch.linspace(-SC.ROBOT_L / 2, SC.ROBOT_L / 2, self.LATTICE[0], device=device, dtype=dtype)
        b = torch.linspace(-SC.ROBOT_W / 2, SC.ROBOT_W / 2, self.LATTICE[1], device=device, dtype=dtype)
        A, Bq = torch.meshgrid(a, b, indexing='ij')
        self.lat = torch.stack([A.reshape(-1), Bq.reshape(-1)], 1)

    @staticmethod
    def mesh_path(name):
        return GIBSON_TINY / name / 'mesh_z_up.obj'

    @staticmethod
    def build_table(name, cell_m, zs, res, pad, seal_m):
        import importlib.util
        from scipy import ndimage
        spec = importlib.util.spec_from_file_location('gibson_grid', ROOT / 'jeffrey/gibson_embed/grid.py')
        G = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(G)
        v, f = read_obj(House.mesh_path(name))
        segs = np.concatenate([plane_segments(v, f, z) for z in zs])
        lo, hi = segs.reshape(-1, 2).min(0), segs.reshape(-1, 2).max(0)
        centre = (lo + hi) / 2
        side = float(np.ceil(((hi - lo).max() + 2 * pad) / cell_m) * cell_m)        # a whole number of cells
        if res is None:
            res = int(2 ** np.ceil(np.log2(side / cell_m * 16)))                # >= 16 table nodes per cell
            res = min(max(res, 1024), 4096)
        occ = G.rasterize_segments((segs - centre) / side, res)
        k = max(1, int(round(seal_m / side * res)))
        yy, xx = np.mgrid[-k - 1:k + 2, -k - 1:k + 2]
        sealed = ndimage.binary_dilation(occ, structure=(xx * xx + yy * yy) <= k * k)
        labels, sizes = G.components(~sealed)
        border = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))) - {0}
        order = [c for c in np.argsort(-sizes) if (c + 1) not in border]
        core = labels == (order[0] + 1)
        grown = ndimage.binary_dilation(core, structure=(xx * xx + yy * yy) <= (k + 1) ** 2) & ~occ
        labels, _ = G.components(grown)
        free = labels == np.bincount(labels[core]).argmax()
        d = (ndimage.distance_transform_edt(free) - ndimage.distance_transform_edt(~free)) / res
        return d.astype(np.float32), side, centre.tolist(), res

    def point_sdf_cells(self, xy):
        G = self.res
        u = (xy + 0.5) * (G - 1)
        uc = u.clamp(0, G - 1 - 1e-4)
        i = uc.floor().long()
        fr = uc - i
        i1 = (i + 1).clamp(max=G - 1)
        t = self.table
        v = ((1 - fr[:, 0]) * (1 - fr[:, 1]) * t[i[:, 0], i[:, 1]] + fr[:, 0] * (1 - fr[:, 1]) * t[i1[:, 0], i[:, 1]]
             + (1 - fr[:, 0]) * fr[:, 1] * t[i[:, 0], i1[:, 1]] + fr[:, 0] * fr[:, 1] * t[i1[:, 0], i1[:, 1]])
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
        return dict(kind='house', name=self.name, cell_m=self.cell_m, zs=self.zs, res=self.res, pad=self.pad, seal_m=self.seal_m, side_m=self.side_m, centre=self.centre,
                    lattice=list(self.LATTICE))

    def describe(self):
        d = super().describe()
        d.update(free_share=self.free_share)
        return d


def build(spec, device='cuda:0'):
    if spec['kind'] == 'house':
        return House(spec['name'], spec['cell_m'], spec['zs'], spec['res'], spec['pad'], spec.get('seal_m', 0.15), device)
    return SC.build(spec, device)


def load_scene(prepared, device='cuda:0'):
    info = json.load(open(Path(prepared) / 'maze.json'))
    if info.get('scene', {}).get('kind') == 'house':
        scene = build(info['scene'], device)
        assert abs(scene.E - info['E_cells']) < 1e-6 and abs(scene.free_share - info['free_share']) < 1e-6, 'prepared house differs'
        return scene, info
    return SC.load_scene(prepared, device)


def scene_name(info):
    """Short label of a prepared scene."""
    sc = info.get('scene', {})
    if sc.get('kind') == 'house':
        return f"house {sc['name']} ({info['E_cells']:.0f} cells)"
    if sc.get('kind') == 'gibson':
        return f"apartment ({info['E_cells']:.0f} cells)"
    if sc.get('kind') == 'disks':
        return 'disks'
    v = info.get('variant', 'plain')
    return f"{'plus maze' if v == 'plus' else v + ' maze'} {info['n']} x {info['n']}"


def camera_floor(name):
    """Camera height (metres) of the floor with the most scan positions (camera_poses.csv)."""
    import csv
    cz = np.array([float(r[3]) for r in csv.reader(open(GIBSON_TINY / name / 'camera_poses.csv'))])
    hist, edges = np.histogram(cz, bins=np.arange(cz.min() - 0.25, cz.max() + 0.5, 0.5))
    k = hist.argmax()
    return float(np.median(cz[(cz >= edges[k] - 0.3) & (cz < edges[k + 1] + 0.3)]))


def house_slices(name, offsets=(-0.9, -0.6, -0.3, 0.0, 0.3)):
    """Slice heights: the main floor's camera height plus offsets (the camera is about 1.3 m above the floor, so 0.4 .. 1.6 m)."""
    zc = camera_floor(name)
    return [round(zc + o, 3) for o in offsets]
