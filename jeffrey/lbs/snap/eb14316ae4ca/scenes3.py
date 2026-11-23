"""Gibson houses as 3-D scenes for a point robot in R^3 (asked 2026-11-20: "3d gibson R3 planning"), with the Maze interface of every
local study, so the sampler, truth solver, planner, scorer and the LBS trainer run on them unchanged.

    House3   the house mesh (~/Downloads/gibson_tiny/<name>/mesh_z_up.obj, metres, z up) voxelized at `vox_per_cell` voxels per cell
             (one cell = cell_m metres, 0.25 as in the 2-D houses) inside a slab of the main floor: from slab[0] to slab[1] metres above
             the floor (the floor is the largest horizontal surface 0.7 to 2.2 m below the scan cameras of the floor with the most scan
             positions). Voxels that hold a piece of the mesh are obstacles (every triangle is sampled at a third of a voxel, so a
             surface between two face-adjacent voxels marks one of them). The outside of the house is an obstacle as well: the
             footprint is found as in the 2-D House (scenes.py: slices 0.4 to 1.6 m above the floor, gaps narrower than 2 seal_m
             closed, everything connected to the box border is outside), and so is everything below and above the slab. Free space is
             the largest 6-connected component of what is left (the air of the rooms: above and below furniture, through doors).
             Signed distance by distance transforms between voxel centres (half a voxel taken off, so the zero is on the voxel
             faces), trilinear table.

Configuration q = (x, y, z) / side_m about the box centre: x, y in [-0.5, 0.5], z in [-z_half, z_half] (z_half = half the slab height
/ side_m), the same metric in all three directions. The maze interface calls the half-range of the third coordinate `phi_max`; here it
is z_half, and the truth grid's "heading" nodes are z nodes. The robot is a point: its clearance is the signed distance itself.

load_scene(prepared) rebuilds a House3 from maze.json and the table the prepare run stored (house3_table.npy), and hands every other
prepare run to scenes.load_scene.
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scenes as SN  # noqa: E402

SC = SN.SC
ROOT = SN.ROOT
TABLE = 'house3_table.npy'


def floor_height(v, f, zc):
    """Height (metres) of the largest horizontal surface 0.7 to 2.2 m below the camera height zc."""
    t = v[f]
    nrm = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    area = 0.5 * np.linalg.norm(nrm, axis=1)
    flat = np.abs(nrm[:, 2]) > 0.9 * 2 * area + 1e-12
    zt = t[:, :, 2].mean(1)
    h, e = np.histogram(zt[flat], bins=np.arange(zc - 2.2, zc - 0.7 + 1e-9, 0.05), weights=area[flat])
    k = int(np.argmax(h))
    return float(0.5 * (e[k] + e[k + 1]))


def voxelize(v, f, origin, vox, shape, budget=4_000_000):
    """Occupancy (bool, shape) of the triangles: voxel (i, j, k) covers origin + vox [i, i + 1) x [j, j + 1) x [k, k + 1)."""
    origin = np.asarray(origin, np.float64)
    hi = origin + vox * np.asarray(shape)
    t = v[f]
    t = t[(t.max(1) >= origin).all(1) & (t.min(1) <= hi).all(1)]
    edge = np.stack([np.linalg.norm(t[:, a] - t[:, b], axis=1) for a, b in ((0, 1), (1, 2), (2, 0))], 1).max(1)
    n = np.maximum(1, np.ceil(edge / (vox / 3)).astype(np.int64))
    occ = np.zeros(shape, bool)
    shp = np.asarray(shape)
    for nn in np.unique(n):
        tt = t[n == nn]
        i, j = np.meshgrid(np.arange(nn + 1), np.arange(nn + 1), indexing='ij')
        m = (i + j) <= nn
        a, b = i[m] / nn, j[m] / nn
        c = 1 - a - b
        step = max(1, budget // len(a))
        for s in range(0, len(tt), step):
            q = tt[s:s + step]
            pts = a[None, :, None] * q[:, None, 0] + b[None, :, None] * q[:, None, 1] + c[None, :, None] * q[:, None, 2]
            idx = np.floor((pts.reshape(-1, 3) - origin) / vox).astype(np.int64)
            ok = ((idx >= 0) & (idx < shp)).all(1)
            idx = idx[ok]
            occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return occ


class House3(SC.SceneBase):
    kind = 'house3'
    pos_dims = 3
    PAD_Z = 4                                        # obstacle voxels below and above the slab in the table

    def __init__(self, name, cell_m=0.25, vox_per_cell=10, slab=(0.15, 2.15), pad=0.5, seal_m=0.15, device='cuda:0', dtype=torch.float32, built=None):
        self.name, self.cell_m, self.vpc, self.slab, self.pad, self.seal_m = str(name), float(cell_m), int(vox_per_cell), [float(s) for s in slab], float(pad), float(seal_m)
        if built is None:
            built = self.build(self.name, self.cell_m, self.vpc, self.slab, self.pad, self.seal_m)
        table, meta = built
        self.side_m, self.centre, self.floor_z, self.zs = float(meta['side_m']), [float(c) for c in meta['centre']], float(meta['floor_z']), [float(z) for z in meta['zs']]
        self.build_stats = meta.get('stats', {})
        E = int(round(self.side_m / self.cell_m))
        self._setup(E, E, device, dtype)
        self.z_cells = (self.slab[1] - self.slab[0]) / self.cell_m
        self.phi_max = 0.5 * (self.slab[1] - self.slab[0]) / self.side_m          # half-range of the third coordinate (z)
        self.lo = torch.tensor([-0.5, -0.5, -self.phi_max], device=device, dtype=dtype)
        self.hi = torch.tensor([0.5, 0.5, self.phi_max], device=device, dtype=dtype)
        table = torch.as_tensor(np.asarray(table), device=device, dtype=dtype)   # (nx, ny, nz) signed distance in cells at voxel centres
        self.shape = tuple(int(s) for s in table.shape)
        self.free_share = float((table[:, :, self.PAD_Z:-self.PAD_Z] > 0).float().mean())
        self.table = table.reshape(-1).contiguous()
        vox_q = 1.0 / (E * self.vpc)                                              # voxel size in normalized units
        self.q0 = torch.tensor([-0.5, -0.5, -self.phi_max - self.PAD_Z * vox_q], device=device, dtype=dtype)
        self.per_q = float(E * self.vpc)
        self.umax = torch.tensor([s - 1 - 1e-4 for s in self.shape], device=device, dtype=dtype)
        self.uzero = torch.zeros(3, device=device, dtype=dtype)

    # --- construction --------------------------------------------------------------------------------------------
    @staticmethod
    def build(name, cell_m, vpc, slab, pad, seal_m):
        import importlib.util
        from scipy import ndimage
        spec = importlib.util.spec_from_file_location('gibson_grid', ROOT / 'jeffrey/gibson_embed/grid.py')
        G = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(G)
        v, f = SN.read_obj(SN.House.mesh_path(name))
        zc = SN.camera_floor(name)
        floor = floor_height(v, f, zc)
        zs = [round(floor + h, 3) for h in (0.4, 0.7, 1.0, 1.3, 1.6)]
        segs = np.concatenate([SN.plane_segments(v, f, z) for z in zs])
        lo, hi = segs.reshape(-1, 2).min(0), segs.reshape(-1, 2).max(0)
        centre = (lo + hi) / 2
        side = float(np.ceil(((hi - lo).max() + 2 * pad) / cell_m) * cell_m)
        vox = cell_m / vpc
        nxy = int(round(side / vox))
        nzs = int(round((slab[1] - slab[0]) / vox))
        assert abs(nzs * vox - (slab[1] - slab[0])) < 1e-9, 'the slab must be a whole number of voxels'
        # footprint: everything that is not connected to the box border once gaps narrower than 2 seal_m are closed
        occ2 = G.rasterize_segments((segs - centre) / side, nxy)
        k = max(1, int(round(seal_m / vox)))
        yy, xx = np.mgrid[-k - 1:k + 2, -k - 1:k + 2]
        sealed = ndimage.binary_dilation(occ2, structure=(xx * xx + yy * yy) <= k * k)
        labels, sizes = G.components(~sealed)
        border = np.array(sorted(set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))) - {0}))
        outside = np.isin(labels, border)
        outside = ndimage.binary_dilation(outside, structure=(xx * xx + yy * yy) <= (k + 1) ** 2) & ~occ2
        foot = ~outside
        # the slab
        z0 = floor + slab[0]
        occ3 = voxelize(v, f, [centre[0] - side / 2, centre[1] - side / 2, z0], vox, (nxy, nxy, nzs))
        cand = foot[:, :, None] & ~occ3
        lab3, n3 = ndimage.label(cand)
        counts = np.bincount(lab3.reshape(-1))
        counts[0] = 0
        main = int(np.argmax(counts))
        free = np.zeros((nxy, nxy, nzs + 2 * House3.PAD_Z), bool)
        free[:, :, House3.PAD_Z:-House3.PAD_Z] = lab3 == main
        stats = dict(voxels=[nxy, nxy, nzs], vox_m=vox, mesh_voxels_share=float(occ3.mean()), footprint_share=float(foot.mean()), candidates_share=float(cand.mean()),
                     components=int(n3), free_share=float(counts[main] / cand.size), second_component_share=float(np.sort(counts)[-2] / cand.size) if n3 > 1 else 0.0,
                     free_columns_share=float((lab3 == main).any(2).mean()))
        del lab3, cand, occ3
        d = ndimage.distance_transform_edt(free).astype(np.float32)
        d -= 0.5 * free
        dn = ndimage.distance_transform_edt(~free).astype(np.float32)
        d -= np.where(free, 0, dn - 0.5).astype(np.float32)
        del dn
        table = (d / vpc).astype(np.float32)                                    # cells
        return table, dict(side_m=side, centre=[float(centre[0]), float(centre[1])], floor_z=floor, zs=zs, stats=stats)

    # --- clearance -----------------------------------------------------------------------------------------------
    def signed_cells(self, q):
        """Signed clearance (cells) of points q (N, 3), normalized; beyond the table the border value minus the overshoot."""
        u = (q.to(self.dtype) - self.q0) * self.per_q - 0.5                      # in voxel centres
        uc = torch.minimum(torch.maximum(u, self.uzero), self.umax)
        i = uc.detach().floor().long()
        fr = uc - i
        nx, ny, nz = self.shape
        base = (i[:, 0] * ny + i[:, 1]) * nz + i[:, 2]
        t = self.table
        fx, fy, fz = fr[:, 0], fr[:, 1], fr[:, 2]
        v = ((1 - fx) * (1 - fy) * (1 - fz) * t[base] + (1 - fx) * (1 - fy) * fz * t[base + 1]
             + (1 - fx) * fy * (1 - fz) * t[base + nz] + (1 - fx) * fy * fz * t[base + nz + 1]
             + fx * (1 - fy) * (1 - fz) * t[base + ny * nz] + fx * (1 - fy) * fz * t[base + ny * nz + 1]
             + fx * fy * (1 - fz) * t[base + ny * nz + nz] + fx * fy * fz * t[base + ny * nz + nz + 1])
        return v - (u - uc).abs().sum(1) / self.vpc

    def bell_table(self):
        """The clearance table of the Bellman backup: this scene's own table (bell.ClearTable would only resample it)."""
        scene = self

        class Table:
            per_cell = scene.vpc

            def __call__(self, q):
                return scene.signed_cells(q).clamp(max=2.0)
        return Table()

    def to_cells(self, q):
        return q[:, 0] * self.E + self.n / 2, q[:, 1] * self.E + self.n / 2, q[:, 2] * self.E

    def spec(self):
        return dict(kind='house3', name=self.name, cell_m=self.cell_m, vox_per_cell=self.vpc, slab=self.slab, pad=self.pad, seal_m=self.seal_m, side_m=self.side_m, centre=self.centre,
                    floor_z=self.floor_z, zs=self.zs, table_shape=list(self.shape), z_cells=self.z_cells, robot='point', build=self.build_stats)

    def describe(self):
        d = super().describe()
        d.update(free_share=self.free_share, robot='point', z_cells=self.z_cells)
        return d


def load_scene(prepared, device='cuda:0'):
    info = json.load(open(Path(prepared) / 'maze.json'))
    sc = info.get('scene', {})
    if sc.get('kind') == 'house3':
        table = np.load(Path(prepared) / TABLE)
        meta = dict(side_m=sc['side_m'], centre=sc['centre'], floor_z=sc['floor_z'], zs=sc['zs'], stats=sc.get('build', {}))
        scene = House3(sc['name'], sc['cell_m'], sc['vox_per_cell'], sc['slab'], sc['pad'], sc['seal_m'], device, built=(table, meta))
        assert abs(scene.E - info['E_cells']) < 1e-6 and abs(scene.free_share - info['free_share']) < 1e-6, 'prepared house differs'
        return scene, info
    return SN.load_scene(prepared, device)


def scene_name(info):
    sc = info.get('scene', {})
    if sc.get('kind') == 'house3':
        return f"house {sc['name']} in 3-D ({info['E_cells']:.0f} x {info['E_cells']:.0f} x {sc['z_cells']:.0f} cells)"
    return SN.scene_name(info)
