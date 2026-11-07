"""Speed-shaped embedding rows for a jeffrey/maze_scale maze: line-by-line sums of sin / cos fitted to the obstacle signal.

Obstacle signal. o(x, y) = 1 - mean over headings of the trained speed label (maze.train_speed of the exact clearance), so
o = 0 where the robot is fast and o -> 1 where it is slow. Outside the maze box o = 1 (the outer wall continues). The speed
can also be a learned one: a field's 1 / |dT/dq| (speed_map(..., field=...)).

One row per direction. Direction k has angle theta_k; s = x cos theta_k + y sin theta_k runs along the line and
t = -x sin theta_k + y cos theta_k says which of the parallel lines. On every line the signal is fitted by least squares with

    S_k(s, t) = c_0(t) + sum_{m=1..M} a_m(t) sin(2 pi f_km s) + b_m(t) cos(2 pi f_km s)

The M frequencies f_km are chosen from the environment: the M largest peaks of the line-summed power spectrum of direction k
(zero-padded, at least one window harmonic apart), so they are not harmonics of the box and not a Gaussian draw. Two scopes:

  line   a_m(t), b_m(t) belong to the line through (x, y): S_k(x, y) is the signal on that line, as a short sum of sin / cos
  ridge  one coefficient set per direction, fitted to all lines at once (the mean over t): S_k depends on s only, so it is a
         fixed sum of plane waves, a B matrix with M frequencies per row and fitted amplitudes and phases
  terms  as line, but the sum is left open: one channel per term, a_m(t) sin(2 pi f_km s) + b_m(t) cos(2 pi f_km s). Summing a
         direction's channels gives its line row, so the network can form it in its first layer. K // M directions.
  map    the obstacle signal itself in every row [control: is a row more than the speed map?]

With an accurate fit every line row reconstructs the same number o(x, y), whatever its direction, so the rows of `line` merge
into one map as M grows (effective rank of 32 rows 1.4 at M = 30 on 13 x 13); `terms` and `ridge` do not.

fit() returns the table S (R, R, K) of the K rows on the node grid x_i = -0.5 + i / (R - 1), the fit's quality per row and
the frequencies. The table only caches the sums so that the network does not evaluate K M sines per point.
"""
import math
import numpy as np
import torch
from scipy import ndimage

NODES_PER_CELL = 16
HEADINGS = 16


def angles(n):
    """n directions over a half turn (a line and its reverse are one row), starting on the x axis."""
    return np.pi * np.arange(n) / n


def node_axis(maze, per_cell=NODES_PER_CELL):
    R = int(round(per_cell * maze.E)) + 1
    return np.linspace(-0.5, 0.5, R)


@torch.no_grad()
def speed_map(maze, per_cell=NODES_PER_CELL, headings=HEADINGS, field=None, goals=None):
    """Mean over headings of the speed on the node grid, (R, R), axis 0 = x. field=None: the trained speed label from the exact
    clearance. field=grad(pairs) -> (T, dT/dq0): the learned speed 1 / |dT/dq0|, median over `goals`, clipped to [0, 1]."""
    ax = torch.tensor(node_axis(maze, per_cell), device=maze.device, dtype=maze.dtype)
    X, Y = torch.meshgrid(ax, ax, indexing='ij')
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], 1)
    th = (torch.arange(headings, device=maze.device, dtype=maze.dtype) + 0.5) / headings * math.pi - math.pi / 2
    phi = th * (maze.hi[2] / (math.pi / 2))
    out = torch.zeros(len(xy), device=maze.device, dtype=maze.dtype)
    for p in phi:
        q = torch.cat([xy, p.expand(len(xy), 1)], 1)
        if field is None:
            out += maze.train_speed(maze.clearance(q))
        else:
            with torch.enable_grad():
                s = torch.stack([1 / torch.linalg.vector_norm(field(torch.cat([q, g.expand(len(q), 3)], 1))[1], dim=1).clamp(min=1e-6) for g in goals])
            out += s.median(0).values.clamp(0, 1)
    return (out / headings).reshape(len(ax), len(ax)).cpu().numpy()


@torch.no_grad()
def speed_cube(maze, per_cell=NODES_PER_CELL, knots=33):
    """The trained speed on the node grid at `knots` headings from -pi/2 to pi/2, (R, R, knots)."""
    ax = torch.tensor(node_axis(maze, per_cell), device=maze.device, dtype=maze.dtype)
    X, Y = torch.meshgrid(ax, ax, indexing='ij')
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], 1)
    out = []
    for p in torch.linspace(-1, 1, knots, device=maze.device, dtype=maze.dtype) * maze.hi[2]:
        out.append(maze.train_speed(maze.clearance(torch.cat([xy, p.expand(len(xy), 1)], 1))).reshape(len(ax), len(ax)))
    return torch.stack(out, -1).cpu().numpy()


def corridor_distance(maze):
    """(n * n, n * n) steps between cells through the open passages, breadth-first over the cell graph (cell index i * n + j)."""
    n = maze.n
    V, H = (np.asarray(w.cpu() if torch.is_tensor(w) else w) for w in (maze.V, maze.H))
    nb = [[] for _ in range(n * n)]
    for i in range(n):
        for j in range(n):
            if i + 1 < n and not V[i + 1, j]:
                nb[i * n + j].append((i + 1) * n + j)
                nb[(i + 1) * n + j].append(i * n + j)
            if j + 1 < n and not H[i, j + 1]:
                nb[i * n + j].append(i * n + j + 1)
                nb[i * n + j + 1].append(i * n + j)
    D = np.full((n * n, n * n), -1, np.int32)
    for s in range(n * n):
        D[s, s] = 0
        todo = [s]
        while todo:
            nxt = []
            for a in todo:
                for b in nb[a]:
                    if D[s, b] < 0:
                        D[s, b] = D[s, a] + 1
                        nxt.append(b)
            todo = nxt
    return D


def landmark_table(maze, count, per_cell=NODES_PER_CELL):
    """Corridor distance in cells from `count` landmark cells, on the node grid: (R, R, count), and the cells. Landmarks: farthest-point sampling over
    the cell graph from cell (0, 0). A point p of cell c is min over the open sides of c of |p - door| + 0.5 + D(neighbour, landmark): continuous through
    every doorway, with a jump across a wall of the difference between the two ways round."""
    n = maze.n
    D = corridor_distance(maze)
    marks = [0]
    while len(marks) < count:
        marks.append(int(D[marks].min(0).argmax()))
    V, H = (np.asarray(w.cpu() if torch.is_tensor(w) else w) for w in (maze.V, maze.H))
    ax = torch.tensor(node_axis(maze, per_cell), device=maze.device, dtype=maze.dtype)
    X, Y = torch.meshgrid(ax, ax, indexing='ij')
    cells = maze.to_cells(torch.stack([X.reshape(-1), Y.reshape(-1), torch.zeros_like(X.reshape(-1))], 1))
    x, y = (c.double().cpu().numpy().clip(0, n - 1e-6) for c in cells[:2])
    i, j = np.floor(x).astype(int), np.floor(y).astype(int)
    c = i * n + j
    best = np.full((len(x), count), np.inf)
    sides = [(V[i, j] | (i == 0), i, j + 0.5, -n), (V[np.minimum(i + 1, n - 1), j] | (i == n - 1), i + 1.0, j + 0.5, n),
             (H[i, j] | (j == 0), i + 0.5, j, -1), (H[i, np.minimum(j + 1, n - 1)] | (j == n - 1), i + 0.5, j + 1.0, 1)]
    for closed, dx, dy, step in sides:
        k = np.nonzero(~closed)[0]
        door = np.hypot(x[k] - np.broadcast_to(dx, x.shape)[k], y[k] - np.broadcast_to(dy, y.shape)[k])
        best[k] = np.minimum(best[k], door[:, None] + 0.5 + D[c[k] + step][:, marks])
    for m, cell in enumerate(marks):
        k = np.nonzero(c == cell)[0]
        best[k, m] = np.hypot(x[k] - (i[k] + 0.5), y[k] - (j[k] + 0.5))
    R = len(ax)
    return best.reshape(R, R, count).astype(np.float32), marks


def line_frame(theta, h):
    """Sample points of direction theta: s_i, t_j on the h-spaced rotated grid that covers the box, and the window length."""
    c, s = abs(math.cos(theta)), abs(math.sin(theta))
    half = 0.5 * (c + s)
    count = int(math.ceil(2 * half / h)) + 1
    u = (np.arange(count) - (count - 1) / 2) * h
    return u, u, 2 * half


def sample_rotated(o, theta, u_s, u_t):
    """o on the rotated grid: rows = lines (t), columns = position along the line (s). Outside the box o = 1."""
    R = o.shape[0]
    S, T = np.meshgrid(u_s, u_t)
    x = S * math.cos(theta) - T * math.sin(theta)
    y = S * math.sin(theta) + T * math.cos(theta)
    return ndimage.map_coordinates(o, [(x + 0.5) * (R - 1), (y + 0.5) * (R - 1)], order=1, mode='constant', cval=1.0)


def pick_frequencies(signal, h, M, window, pad=8):
    """The M largest peaks of the power spectrum summed over the rows of signal (mean removed), at least 1 / window apart."""
    z = signal - signal.mean(1, keepdims=True)
    size = 1 << int(math.ceil(math.log2(pad * z.shape[1])))
    power = (np.abs(np.fft.rfft(z, n=size, axis=1)) ** 2).sum(0)
    freqs = np.fft.rfftfreq(size, h)
    sep = max(1, int(round((1 / window) / (freqs[1] - freqs[0]))))
    keep = np.zeros(len(power), bool)
    p = power.copy()
    p[0:sep] = 0                                 # the mean is c_0; the first harmonic is the slowest frequency allowed
    picked = []
    for i in np.argsort(-p):
        if p[i] <= 0 or len(picked) == M:
            break
        if not keep[i]:
            picked.append(freqs[i])
            keep[max(0, i - sep + 1):i + sep] = True
    return np.sort(np.array(picked))


def basis(s, f):
    """[1, sin 2 pi f s, cos 2 pi f s] (len(s), 2 len(f) + 1)."""
    w = 2 * np.pi * s[:, None] * f[None]
    return np.concatenate([np.ones((len(s), 1)), np.sin(w), np.cos(w)], 1)


def fit_direction(o, theta, M, h, scope):
    """Least-squares sum of sin / cos of direction theta. Returns the fitted signal on the rotated grid, its grid and the
    frequencies. Row weights: the lstsq runs over the whole window, every line shares the frequencies of the direction."""
    u_s, u_t, window = line_frame(theta, h)
    signal = sample_rotated(o, theta, u_s, u_t)
    if scope == 'ridge':
        inside = sample_rotated(np.zeros_like(o), theta, u_s, u_t) < 0.5        # (lines, s): inside the box
        target = (signal * inside).sum(0) / np.maximum(inside.sum(0), 1)
        target = np.where(inside.sum(0) > 0, target, 1.0)
        f = pick_frequencies(target[None], h, M, window)
        A = basis(u_s, f)
        coef = np.linalg.lstsq(A, target, rcond=1e-8)[0]
        return np.broadcast_to(A @ coef, signal.shape), u_s, u_t, f, signal
    f = pick_frequencies(signal, h, M, window)
    A = basis(u_s, f)
    coef = np.linalg.lstsq(A, signal.T, rcond=1e-8)[0]
    return (A @ coef).T, u_s, u_t, f, signal


def fit_terms(o, theta, M, h):
    """Each term of the line fit of direction theta on the rotated grid: (M, lines, s), and the fit (lines, s)."""
    u_s, u_t, window = line_frame(theta, h)
    signal = sample_rotated(o, theta, u_s, u_t)
    f = pick_frequencies(signal, h, M, window)
    A = basis(u_s, f)
    coef = np.linalg.lstsq(A, signal.T, rcond=1e-8)[0]               # (2 M' + 1, lines)
    m = len(f)
    terms = [np.outer(coef[1 + j], A[:, 1 + j]) + np.outer(coef[1 + m + j], A[:, 1 + m + j]) for j in range(m)]
    return terms, (A @ coef).T, u_s, u_t, f


def to_table(fitted, theta, u_s, u_t, ax):
    """Fitted rotated signal -> values on the (x, y) node grid."""
    X, Y = np.meshgrid(ax, ax, indexing='ij')
    s = X * math.cos(theta) + Y * math.sin(theta)
    t = -X * math.sin(theta) + Y * math.cos(theta)
    h = u_s[1] - u_s[0]
    return ndimage.map_coordinates(np.ascontiguousarray(fitted), [(t - u_t[0]) / h, (s - u_s[0]) / h], order=3, mode='nearest')


def fit(o, E, n_rows, M, scope, per_cell=NODES_PER_CELL, log=None):
    """All K = n_rows rows. o: (R, R) obstacle signal on the node grid. Returns the table (R, R, K) float32 and a description."""
    R = o.shape[0]
    ax = np.linspace(-0.5, 0.5, R)
    h = 1.0 / (per_cell * E)
    table = np.zeros((R, R, n_rows), np.float32)
    rows, freqs = [], []
    if scope == 'map':
        table[:] = o[..., None]
        return table, dict(scope=scope, M=0, rows=n_rows, r2_median=1.0, r2_min=1.0, r2_max=1.0, f_over_grid_quantiles=None)
    if scope == 'terms':
        dirs = n_rows // M
        assert dirs >= 1, (n_rows, M)
        c = 0
        for theta in angles(dirs):
            terms, fitted, u_s, u_t, f = fit_terms(o, theta, M, h)
            for term in terms:
                table[..., c] = to_table(term, theta, u_s, u_t, ax)
                c += 1
            err = to_table(fitted, theta, u_s, u_t, ax) - o
            rows.append(dict(theta=float(theta), r2=float(1 - (err ** 2).mean() / o.var()), frequencies=len(f)))
            freqs.append(f)
            if log:
                log(f'direction {theta:.3f}: terms M={M}, R^2 of the sum against o {rows[-1]["r2"]:.3f}, frequencies up to {f.max():.1f}')
        r2 = np.array([r['r2'] for r in rows])
        allf = np.concatenate(freqs)
        return table, dict(scope=scope, M=M, rows=c, directions=dirs, r2_median=float(np.median(r2)), r2_min=float(r2.min()), r2_max=float(r2.max()),
                           f_over_grid_quantiles=(np.quantile(allf, [0, 0.1, 0.5, 0.9, 1]) / (E / 2)).tolist())
    for k, theta in enumerate(angles(n_rows)):
        fitted, u_s, u_t, f, signal = fit_direction(o, theta, M, h, scope)
        table[..., k] = to_table(fitted, theta, u_s, u_t, ax)
        err = table[..., k] - o
        rows.append(dict(theta=float(theta), r2=float(1 - (err ** 2).mean() / o.var()), frequencies=len(f)))
        freqs.append(f)
        if log and k % 32 == 0:
            log(f'row {k}/{n_rows}: {scope} M={M}, R^2 against o {rows[-1]["r2"]:.3f}, frequencies up to {f.max():.1f}')
    r2 = np.array([r['r2'] for r in rows])
    allf = np.concatenate(freqs)
    return table, dict(scope=scope, M=M, rows=n_rows, r2_median=float(np.median(r2)), r2_min=float(r2.min()), r2_max=float(r2.max()),
                       f_over_grid_quantiles=(np.quantile(allf, [0, 0.1, 0.5, 0.9, 1]) / (E / 2)).tolist())
