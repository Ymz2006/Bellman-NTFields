"""Ray rows for a jeffrey/maze_scale maze: step functions of position (part 4 of the study).

From a point, shoot a ray in a direction and count how many times it hits an obstacle in front: a hit is a passage from free space into
a wall (the maze's wall boxes, thickness T_WALL; the outer wall counts). The count is a step function of the point: it drops by one when
the point passes a wall on its ray, and it changes when the point moves sideways past the shadow line of a wall end.

One row per ray direction. K rows = K / 2 line directions theta_k = pi k / (K / 2), each with the ray along it and the ray against it, so
the rays cover the full turn in steps of 2 pi / K. Kinds:

  hits            hits ahead, to the edge of the maze                                              [the request]
  hits, reach D   hits within the next D cells only (a local count)
  depth, reach D  distance in cells to the first hit ahead, at most D (a range finder; a ramp with steps, not a step function)

Computed like the line rows of linefit.py: the exact wall occupancy is sampled on the rotated grid of direction theta (lines t, position
s along the line, spacing h / SUB along the line), hits are counted along every line, and the result is read back at the nodes with
linear interpolation (no overshoot at the steps). CPU, numpy, no linear algebra: the table does not depend on the thread count.

table(maze, kind, n_rows, reach) -> (R, R, n_rows) float32 on the node grid of linefit.node_axis, and a description.
"""
import math
import numpy as np
from scipy import ndimage
import linefit as LF

SUB = 4                     # samples along the line per node spacing: 1 / 64 cell, a wall is 6.4 samples thick


def occupied(x, y, V, H, half):
    """Is (x, y) (cell units) inside a wall box? V[i, j]: wall on x = i for y in [j, j + 1]; H[i, j]: wall on y = j for x in [i, i + 1];
    every box is longer by `half` at both ends (maze.wall_boxes)."""
    V, H = np.asarray(V, bool), np.asarray(H, bool)

    def get(W, a, b):
        ok = (a >= 0) & (a < W.shape[0]) & (b >= 0) & (b < W.shape[1])
        return ok & W[np.clip(a, 0, W.shape[0] - 1), np.clip(b, 0, W.shape[1] - 1)]

    def along(c):                                        # the one or two wall segments whose (lengthened) span holds coordinate c
        first = np.floor(c + half).astype(np.int64)
        return first, (c + half - first) <= 2 * half

    i = np.rint(x).astype(np.int64)
    j1, two = along(y)
    occ = (np.abs(x - i) <= half) & (get(V, i, j1) | (two & get(V, i, j1 - 1)))
    j = np.rint(y).astype(np.int64)
    i1, two = along(x)
    return occ | ((np.abs(y - j) <= half) & (get(H, i1, j) | (two & get(H, i1 - 1, j))))


def frame(theta, h):
    """Rotated sample grid of direction theta covering the box: positions along the line (spacing h / SUB) and lines (spacing h)."""
    half = 0.5 * (abs(math.cos(theta)) + abs(math.sin(theta)))
    count = int(math.ceil(2 * half / h)) + 1
    u_t = (np.arange(count) - (count - 1) / 2) * h
    u_s = (np.arange((count - 1) * SUB + 1) - (count - 1) * SUB / 2) * (h / SUB)
    return u_s, u_t


def line_rows(maze, theta, kind, reach, h):
    """(ahead, behind) on the rotated grid (lines, s): the row of the ray along theta and of the ray against it."""
    u_s, u_t = frame(theta, h)
    S, T = np.meshgrid(u_s, u_t)
    x = (S * math.cos(theta) - T * math.sin(theta)) * maze.E + maze.n / 2
    y = (S * math.sin(theta) + T * math.cos(theta)) * maze.E + maze.n / 2
    occ = occupied(x, y, maze.V, maze.H, (maze.E - maze.n) / 2)
    step = (h / SUB) * maze.E                                                  # cells per sample
    k = None if reach is None else int(round(reach / step))
    out = []
    for o in (occ, occ[:, ::-1]):                                              # the ray along +s, then the ray along -s (reversed line)
        enter = np.zeros_like(o)
        enter[:, 1:] = o[:, 1:] & ~o[:, :-1]                                   # sample i is the first inside a wall: a hit at i
        c = np.cumsum(enter, 1)                                                # hits at samples <= i
        if kind == 'hits':
            total = c[:, -1:] if k is None else np.concatenate([c[:, k:], np.repeat(c[:, -1:], k, 1)], 1)     # hits at samples <= i + k
            row = (total - c).astype(np.float64)                               # hits at samples in (i, i + k]
        else:                                                                  # depth: samples to the next hit, at most k
            idx = np.arange(o.shape[1])
            nxt = np.where(enter, idx[None], o.shape[1] + 10 * (k or 0))
            nxt = np.minimum.accumulate(nxt[:, ::-1], 1)[:, ::-1]              # the first hit at a sample >= i
            nxt = np.concatenate([nxt[:, 1:], np.full((len(nxt), 1), o.shape[1] + 10 * (k or 0))], 1)        # ... at a sample > i
            row = np.minimum(nxt - idx[None], k if k is not None else o.shape[1]) * step
        out.append(row)
    return out[0], out[1][:, ::-1], u_s, u_t


def to_nodes(rows, theta, u_s, u_t, ax):
    X, Y = np.meshgrid(ax, ax, indexing='ij')
    s = X * math.cos(theta) + Y * math.sin(theta)
    t = -X * math.sin(theta) + Y * math.cos(theta)
    return ndimage.map_coordinates(np.ascontiguousarray(rows), [(t - u_t[0]) / (u_t[1] - u_t[0]), (s - u_s[0]) / (u_s[1] - u_s[0])], order=1, mode='nearest')


def table(maze, kind='hits', n_rows=16, reach=None, per_cell=LF.NODES_PER_CELL):
    assert kind in ('hits', 'depth') and n_rows % 2 == 0 and (kind == 'hits' or reach is not None)
    ax = LF.node_axis(maze, per_cell)
    h = 1.0 / (per_cell * maze.E)
    half = n_rows // 2
    out = np.zeros((len(ax), len(ax), n_rows), np.float32)
    for k, theta in enumerate(LF.angles(half)):
        ahead, behind, u_s, u_t = line_rows(maze, theta, kind, reach, h)
        out[..., k], out[..., half + k] = to_nodes(ahead, theta, u_s, u_t, ax), to_nodes(behind, theta, u_s, u_t, ax)
    return out, dict(kind=kind, rows=n_rows, reach=reach, directions=half, value_max=float(out.max()), value_mean=float(out.mean()),
                     constant_rows=int((out.reshape(-1, n_rows).std(0) < 1e-9).sum()))


# --- rows that put a norm on the integral (or on the hit counts): part 4, test 2 -----------------------------------------------------

DIRECTIONS = 64             # directions / rays behind every landmark row
NORMS = dict(l1=lambda d: np.abs(d).mean(-1), linf=lambda d: np.abs(d).max(-1))


def integral_rows(maze, z, dirs, per_cell=LF.NODES_PER_CELL):
    """The exact integral of z along the line from where it enters the box, `dirs` directions over a half turn: (R, R, dirs) float64
    (linefit.fit_integral with M = 0, the rows of part 2 before standardization; normalized time units when z = 1 / speed)."""
    h = 1.0 / (per_cell * maze.E)
    ax = LF.node_axis(maze, per_cell)
    out = np.zeros((len(ax), len(ax), dirs))
    for k, th in enumerate(LF.angles(dirs)):
        _, _, total, _, u_s, u_t, _ = LF.fit_integral(z, th, 0, h)
        out[..., k] = LF.to_table(total, th, u_s, u_t, ax)
    return out


def landmarks(n, count):
    """`count` landmark cells, farthest-point sampling of the cell centres in straight-line distance from cell (0, 0): no maze graph."""
    c = np.stack(np.meshgrid(np.arange(n), np.arange(n), indexing='ij'), -1).reshape(-1, 2).astype(np.float64)
    marks = [0]
    d = np.linalg.norm(c - c[0], axis=1)
    while len(marks) < count:
        marks.append(int(d.argmax()))
        d = np.minimum(d, np.linalg.norm(c - c[marks[-1]], axis=1))
    return marks


def landmark_rows(base, maze, marks, norm, per_cell=LF.NODES_PER_CELL):
    """Row m = norm over the base rows of (base at the point - base at the centre of landmark cell m): a distance to the landmark measured
    in the base rows. l1: the mean absolute difference; linf: the largest."""
    out = np.zeros(base.shape[:2] + (len(marks),))
    for m, cell in enumerate(marks):
        i, j = divmod(cell, maze.n)
        out[..., m] = NORMS[norm](base - base[per_cell // 2 + per_cell * i, per_cell // 2 + per_cell * j])
    return out


SCOPES = ['hits', 'depth', 'markint', 'markhits', 'intnorm']


def rows(maze, scope, n_rows, reach=None, norm=None, z=None):
    """The (R, R, n_rows) table of a part-4 scope and its description.
      hits, depth   ray rows (table)
      markint       landmark rows from DIRECTIONS exact integrals of z, norm l1 | linf
      markhits      landmark rows from DIRECTIONS hit counts (to the edge, or within `reach`), norm l1 | linf
      intnorm       n_rows exact integrals of z, each point's rows divided by their l1 | linf norm over the directions"""
    assert scope in SCOPES, scope
    if scope in ('hits', 'depth'):
        return table(maze, scope, n_rows, reach)
    assert norm in NORMS, 'this scope needs --norm l1 | linf'
    if scope == 'intnorm':
        base = integral_rows(maze, z, n_rows)
        size = np.abs(base).mean(-1, keepdims=True) if norm == 'l1' else np.abs(base).max(-1, keepdims=True)
        out = base / np.maximum(size, 1e-9)
        return out.astype(np.float32), dict(kind=scope, rows=n_rows, norm=norm, directions=n_rows)
    marks = landmarks(maze.n, n_rows)
    base = integral_rows(maze, z, DIRECTIONS) if scope == 'markint' else table(maze, 'hits', DIRECTIONS, reach)[0].astype(np.float64)
    out = landmark_rows(base, maze, marks, norm)
    return out.astype(np.float32), dict(kind=scope, rows=n_rows, norm=norm, reach=reach, directions=DIRECTIONS, landmarks=marks, value_max=float(out.max()))
