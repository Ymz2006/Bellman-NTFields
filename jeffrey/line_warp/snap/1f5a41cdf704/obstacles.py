"""Exact obstacle boundaries of a maze, as polylines, for the line loss (lineloss.py).

The loss needs the places where a line changes from free space to obstacle space. Instead of sampling an indicator it
pushes the obstacle boundary itself through the warp (one forward pass) and intersects the image with the line, so a run
length is a piecewise-linear function of the warp's output and the gradient is exact. This module supplies that boundary.

Two definitions of 'obstacle space' on a plane:

* mode 'walls': the wall boxes themselves (workspace; 0.1 cells thick).
* mode 'cspace': the configuration-space obstacle of the robot at a fixed heading theta, i.e. every centre position where
  the robot touches a wall: wall box (+) R(theta) rectangle (Minkowski sum, exact for convex shapes; the union over walls
  by shapely). This is clearance(x, y, theta) < 0 of maze.Maze, the set the planner must avoid (check_cspace).

Free space = the maze box minus the obstacles. Its boundary rings are returned, so a line running from outside enters free
space at its first crossing and leaves it at its last: border walls and everything outside the maze form one unbounded
obstacle that is not a run (lineloss.py counts only the closed runs between the first and the last crossing).
"""
import math
import numpy as np
import shapely
from shapely.geometry import MultiPoint, box as sbox
from shapely.ops import unary_union


def footprint(theta, length, width):
    """Corners (4, 2) of the robot rectangle at heading theta, centred on the origin (cells)."""
    c, s = math.cos(theta), math.sin(theta)
    u, v = np.array([c, s]) * length / 2, np.array([-s, c]) * width / 2
    return np.array([u + v, u - v, -u - v, -u + v])


def obstacle_union(boxes, theta=None, length=0.5, width=0.2):
    """Union of the boxes [xmin, ymin, xmax, ymax] (cells), each dilated by the robot at heading theta (None: not dilated)."""
    if theta is None:
        return unary_union([sbox(*b) for b in boxes])
    f = footprint(theta, length, width)
    out = []
    for x0, y0, x1, y1 in boxes:
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        out.append(MultiPoint((corners[:, None, :] + f[None, :, :]).reshape(-1, 2)).convex_hull)
    return unary_union(out)


def free_space(boxes, n, t_wall, theta=None, length=0.5, width=0.2):
    """Free polygons (shapely) of the maze box [-t/2, n + t/2]^2 minus the obstacles at heading theta (None: the walls)."""
    h = t_wall / 2
    return sbox(-h, -h, n + h, n + h).difference(obstacle_union(boxes, theta, length, width))


def rings_of(free, min_area=1e-3):
    """Boundary rings of the free polygons above min_area (cells^2): closed coordinate arrays, first point repeated last."""
    polys = list(free.geoms) if hasattr(free, 'geoms') else [free]
    rings = []
    for p in polys:
        if p.area < min_area:
            continue
        rings.append(np.asarray(p.exterior.coords))
        rings += [np.asarray(r.coords) for r in p.interiors]
    return rings


def densify(ring, spacing):
    """A closed ring with every edge cut into pieces of at most `spacing` cells (the warp bends what was straight)."""
    out = []
    for a, b in zip(ring[:-1], ring[1:]):
        k = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / spacing)))
        out.append(a + (b - a) * (np.arange(k)[:, None] / k))
    return np.concatenate(out)


class Boundary:
    """The boundary polylines of one plane: points (N, 2) in normalized maze coordinates and segments (S, 2) of indices."""

    def __init__(self, rings, n, E, spacing):
        pts, seg, k = [], [], 0
        for r in rings:
            d = densify(r, spacing)
            m = len(d)
            pts.append(d)
            seg.append(np.stack([k + np.arange(m), k + (np.arange(m) + 1) % m], 1))
            k += m
        self.points = ((np.concatenate(pts) - n / 2) / E).astype(np.float32)
        self.segments = np.concatenate(seg).astype(np.int64)
        self.rings = len(rings)


def planes(maze, mode, headings=9, spacing=0.1):
    """(Boundary, theta) per plane of `maze` (maze.Maze): 'walls' one plane; 'cspace' `headings` planes evenly over
    [-pi/2, pi/2) (the rectangle is pi-periodic); theta None for the walls."""
    import maze as MZ
    thetas = [None] if mode == 'walls' else [(-0.5 + i / headings) * math.pi for i in range(headings)]
    out = []
    for th in thetas:
        free = free_space(maze.boxes, maze.n, MZ.T_WALL, th, MZ.ROBOT_L, MZ.ROBOT_W)
        out.append((Boundary(rings_of(free), maze.n, maze.E, spacing), th))
    return out


def check_cspace(maze, theta, count=200_000, seed=0):
    """Share of random positions where shapely's free space at theta disagrees with maze.clearance(x, y, theta) >= 0,
    outside a thin shell around the boundary (float32 clearance)."""
    import maze as MZ
    import torch
    rng = np.random.default_rng(seed)
    c = rng.uniform(0, maze.n, (count, 2))
    free = free_space(maze.boxes, maze.n, MZ.T_WALL, theta, MZ.ROBOT_L, MZ.ROBOT_W)
    inside = shapely.contains_xy(free, c[:, 0], c[:, 1])
    q = maze.from_cells(torch.tensor(c[:, 0], device=maze.device, dtype=torch.float32), torch.tensor(c[:, 1], device=maze.device, dtype=torch.float32),
                        torch.full((count,), theta, device=maze.device))
    d = maze.clearance(q).cpu().numpy()
    sure = np.abs(d) > 1e-3
    return float(((d >= 0) != inside)[sure].mean())
