"""Occupancy grids of Gibson scenes: 2-D floor-plan slices and 3-D voxelizations over [-0.5, 0.5]^dim.

Rasterization is conservative for face-adjacent cells: every segment (2-D) or triangle (3-D) is sampled at a spacing
of a quarter cell, so any surface crossing the segment between the centres of two face-adjacent cells marks one of
them (the crossing point is at least half a cell from the side faces). Diagonal moves are handled by the
line-of-sight masks in diffuse.py. `leak_check` verifies the claim against the exact mesh on random pairs.
"""
import itertools
import numpy as np
import trimesh
from scipy import ndimage


def load_mesh(path):
    return trimesh.load(str(path), force='mesh', process=False)


def cell_index(points, res):
    return np.clip(np.floor((points + 0.5) * res).astype(np.int64), 0, res - 1)


def section_segments(mesh, z):
    segs = trimesh.intersections.mesh_plane(mesh, plane_normal=[0, 0, 1.0], plane_origin=[0, 0, z])
    return np.asarray(segs)[:, :, :2]


def rasterize_segments(segs, res, spacing=0.25):
    cell = 1.0 / res
    a, b = segs[:, 0], segs[:, 1]
    n = np.ceil(np.linalg.norm(b - a, axis=1) / (spacing * cell)).astype(np.int64) + 2
    rep = np.repeat(np.arange(len(segs)), n)
    start = np.cumsum(n) - n
    t = (np.arange(n.sum()) - start[rep]) / (n[rep] - 1)
    pts = a[rep] + t[:, None] * (b - a)[rep]
    occ = np.zeros((res, res), bool)
    ij = cell_index(pts, res)
    occ[ij[:, 0], ij[:, 1]] = True
    return occ


def rasterize_triangles(vertices, faces, res, spacing=0.25, chunk_points=4_000_000):
    cell = 1.0 / res
    tri = vertices[faces]
    longest = np.linalg.norm(tri[:, [1, 2, 0]] - tri, axis=2).max(1)
    n = np.ceil(longest / (spacing * cell)).astype(np.int64) + 1
    occ = np.zeros((res, res, res), bool)
    for k in np.unique(n):
        sel = np.nonzero(n == k)[0]
        a, b = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing='ij')
        keep = (a + b) <= k
        a, b = a[keep] / k, b[keep] / k
        for part in np.array_split(sel, max(1, len(sel) * len(a) // chunk_points + 1)):
            t = tri[part]
            pts = t[:, None, 0] + a[None, :, None] * (t[:, None, 1] - t[:, None, 0]) + b[None, :, None] * (t[:, None, 2] - t[:, None, 0])
            ijk = cell_index(pts.reshape(-1, 3), res)
            occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    return occ


def build(mesh, dim, res, z=0.0):
    """Occupancy (bool, res^dim) of the scene: the slice at height z for dim 2, the voxelization for dim 3."""
    if dim == 2:
        return rasterize_segments(section_segments(mesh, z), res), section_segments(mesh, z)
    return rasterize_triangles(np.asarray(mesh.vertices), np.asarray(mesh.faces), res), None


def clearance_cells(free):
    """Distance (in cells) from each free cell to the nearest occupied cell."""
    return ndimage.distance_transform_edt(free)


def wall_thickness_cells(occ):
    """Thickness estimate at occupied cells: twice the distance to the nearest free cell, in cells."""
    return 2 * ndimage.distance_transform_edt(occ)


def components(free):
    labels, n = ndimage.label(free, ndimage.generate_binary_structure(free.ndim, 1))
    sizes = np.bincount(labels.ravel())[1:]
    return labels, sizes


def face_adjacent_free_pairs(free, count, rng):
    """Random pairs of face-adjacent free cells (as integer index arrays)."""
    dim = free.ndim
    idx = np.stack(np.nonzero(free), 1)
    pick = idx[rng.integers(0, len(idx), count * 3)]
    axis = rng.integers(0, dim, len(pick))
    other = pick.copy()
    other[np.arange(len(pick)), axis] += 1
    inside = (other < free.shape[0]).all(1)
    pick, other = pick[inside], other[inside]
    ok = free[tuple(other.T)]
    return pick[ok][:count], other[ok][:count]


def _segments_intersect(p, q, a, b):
    """Proper intersection test of segments p-q (N) against a-b (M), vectorized (N, M)."""
    def orient(o, d, e):
        return (d[..., 0] - o[..., 0]) * (e[..., 1] - o[..., 1]) - (d[..., 1] - o[..., 1]) * (e[..., 0] - o[..., 0])
    p, q = p[:, None], q[:, None]
    a, b = a[None], b[None]
    o1, o2 = orient(p, q, a), orient(p, q, b)
    o3, o4 = orient(a, b, p), orient(a, b, q)
    return ((o1 * o2) <= 0) & ((o3 * o4) <= 0)


def leak_check(free, res, mesh=None, segs=None, count=3000, seed=0):
    """Fraction of random face-adjacent free pairs whose centre-to-centre segment crosses the exact surface."""
    rng = np.random.default_rng(seed)
    p, q = face_adjacent_free_pairs(free, count, rng)
    cp = (p + 0.5) / res - 0.5
    cq = (q + 0.5) / res - 0.5
    if free.ndim == 2:
        hits = np.zeros(len(p), bool)
        for i in range(0, len(p), 500):
            hits[i:i + 500] = _segments_intersect(cp[i:i + 500], cq[i:i + 500], segs[:, 0], segs[:, 1]).any(1)
    else:
        direction = cq - cp
        locations, index_ray, _ = mesh.ray.intersects_location(cp, direction, multiple_hits=False)
        hits = np.zeros(len(p), bool)
        if len(index_ray):
            dist = np.linalg.norm(locations - cp[index_ray], axis=1)
            hits[index_ray[dist <= np.linalg.norm(direction[index_ray], axis=1) + 1e-9]] = True
    return dict(pairs=int(len(p)), leaks=int(hits.sum()), leak_fraction=float(hits.mean()) if len(p) else float('nan'))


def summary(occ):
    free = ~occ
    labels, sizes = components(free)
    thick = wall_thickness_cells(occ)
    return dict(res=int(occ.shape[0]), dim=int(occ.ndim), occupied_fraction=float(occ.mean()), free_cells=int(free.sum()),
                components=int(len(sizes)), largest_component_fraction=float(sizes.max() / free.sum()) if len(sizes) else 0.0,
                wall_thickness_cells=dict(p10=float(np.percentile(thick[occ], 10)), median=float(np.median(thick[occ])), p90=float(np.percentile(thick[occ], 90))))
