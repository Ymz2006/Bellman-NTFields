"""Grid-anchored Fourier rows: sin / cos waves that start and end on the maze's grid lines.

jeffrey/grid_sync's `octave` rows fix each wave's period (half period L cells along its main axis) but not where it starts: the origin of
q is the maze centre, most rows are tilted, so a row's phase drifts along a wall. Here the feature of row r is

    [sin, cos](2 pi (b_r . q) + delta_r),   delta_r = pi (b_x + b_y) n / E

so that at heading 0 the phase is 2 pi (b_x x_c + b_y y_c) / E in cell coordinates (x_c, y_c) measured from the maze's lower-left grid
corner. The directions, the octave of every row and the heading column are those of `octave` (same seeded draw).

  anchor    the octave rows, only shifted (delta_r). sin and cos of a row span every phase, so this changes the features the network
            starts from but not what a linear layer can build from them                                          [control]
  axis      every row turned onto its main axis (minor component 0): the phase is pi x_c / L (or y_c), so sin is 0 on every L-th grid
            line along its whole length and cos is +-1 there: the wave starts and ends on grid lines                [main]
  axiscell  axis with one more octave, L = 1/2: one full period per cell, the same in every cell (cos = 1 on every grid line)
  lattice   the minor component snapped to a multiple of half the row's own grid frequency (tilt 0, 26.6 or 45 degrees): tilted waves
            that still repeat on the grid (phase a multiple of pi at every 2L-th grid corner)

rows(draw, E, n, mode) -> (B (128, 3) float64, delta (128,), L (128,) half periods in cells).
"""
import numpy as np

MODES = ['anchor', 'axis', 'axiscell', 'lattice']


def ladder(n, extra=False):
    L = [1.0]
    while L[-1] * 2 <= n:
        L.append(L[-1] * 2)
    return L[::-1] + ([0.5] if extra else [])


def rows(draw, E, n, mode):
    assert mode in MODES + ['octave'], mode
    b = np.asarray(draw, np.float64)
    xy, ph = b[:, :2], b[:, 2]
    grid = E / 2.0
    norm = np.linalg.norm(xy, axis=1)
    dom = np.abs(xy).max(1)
    unit = xy / norm[:, None]
    s_eff = grid / float(np.median(dom))
    Ls = ladder(n, mode == 'axiscell')
    L = np.empty(len(b))
    for k, idx in enumerate(np.array_split(np.argsort(norm, kind='stable'), len(Ls))):   # features.transform's grouping
        L[idx] = Ls[k]
    f = unit * ((grid / (dom / norm)) / L)[:, None]              # octave: the larger component is exactly grid / L
    if mode in ('axis', 'axiscell', 'lattice'):
        major = np.abs(f).argmax(1)
        minor = 1 - major
        r = np.arange(len(f))
        step = grid / L
        f[r, minor] = np.round(f[r, minor] / step * 2) / 2 * step if mode == 'lattice' else 0.0
    delta = np.zeros(len(b)) if mode == 'octave' else np.pi * f.sum(1) * n / E
    return np.concatenate([f, (ph * s_eff)[:, None]], 1), delta, L
