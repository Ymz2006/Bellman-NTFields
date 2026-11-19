"""Detuned grid-synced octaves: jeffrey/grid_sync's `octave` matrix with its xy frequencies moved off the octave values.

The grid-sync octave matrix (jeffrey/grid_sync/features.py, mode `octave`) takes ntrl-demo's seeded draw B (128 x 3),
keeps every row's xy direction, sorts the rows by ntrl-demo's |f_xy| into K equal groups, and puts group k on the square
max(|f_x|, |f_y|) = f_grid / L_k, L = 2^(K-1) .. 2, 1 cells (f_grid = E / 2 cycles per normalized unit: half period one
cell). Every row then has an exact octave frequency and runs in step with every L-th grid line: across the grid lines
x_i = (i - n/2) / E its phase is pi (i - n/2) / L, the same pattern at every line. The heading column is b_phi x s_eff.

This module rebuilds that matrix (bit for bit, checked against grid_sync's transform) and multiplies each row's xy
frequency vector by a factor c_i (direction, heading column and row grouping unchanged):

  octave            c = 1                       the grid-sync method, exact
  shift<s>          c = s for every row         coherent detuning: every level keeps the ratio 2 to the next, but no
                                                level sits on f_grid / L any more (s = 2^k is locked again, one
                                                octave up or down); phase error at the maze edge pi |s - 1| n / (2 L)
  jitter<d>         c_i = exp(u_i), u_i = v_i ln(1 + d) minus its octave group's mean, v_i ~ U(-1, 1) from numpy
                    default_rng(0) (the same v for every d and every training seed): every row a little off its
                    octave, each octave group keeping its geometric-mean frequency exactly; incoherent detuning
  levels<d>         c = (1 + d)^(+1, -1, +1, ...) per octave group, the top group (L = 1) first: each level coherent
                    but the levels no longer a factor 2 apart (ratios 2 (1 + d)^2 and 2 / (1 + d)^2)
  top<s>            c = s for the top octave group only (L = 1, the rows on f_grid), every other row exact and locked
  rest<s>           c = s for every row except the top group, which stays exactly on f_grid (top<s> and rest<s>
                    together make shift<s>: which part of the band carries a shift's effect)

`shift1.0001` is the null perturbation: the same frequencies to 1e-4 (phase error 0.007 rad at the 22x22 edge for the
top octave) but a different floating-point trajectory, the paired-noise floor of training.
"""
import re
import numpy as np
import torch

KINDS = ('octave', 'shift', 'jitter', 'levels', 'top', 'rest')


def parse(spec):
    if spec == 'octave':
        return 'octave', None
    m = re.fullmatch(r'(shift|jitter|levels|top|rest)([0-9]*\.?[0-9]+)', spec)
    assert m, f'bad spec {spec!r}'
    value = float(m.group(2))
    assert value > 0, spec
    return m.group(1), value


def draw(seed):
    """ntrl-demo's B for a training seed without touching the global generator (= jeffrey/grid_sync/features.draw)."""
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        B = torch.normal(0, 1, size=(128, 3))
        torch.nn.init.trunc_normal_(B, mean=0.0, std=1, a=-2.0, b=2.0)
    finally:
        torch.random.set_rng_state(state)
    return B


def octaves(n):
    """Half periods L (cells) of the octave levels, largest first: 2^k <= n."""
    L = [1]
    while L[-1] * 2 <= n:
        L.append(L[-1] * 2)
    return L[::-1]


def parts(B, E):
    """The grid-sync octave construction, step by step (same operations and order as jeffrey/grid_sync/features.py)."""
    b = B.detach().cpu().double().numpy().copy()
    xy, ph = b[:, :2].copy(), b[:, 2].copy()
    grid = E / 2.0
    dom = np.abs(xy).max(1)
    norm = np.linalg.norm(xy, axis=1)
    cos = dom / norm
    unit = xy / norm[:, None]
    s_eff = grid / float(np.median(dom))
    Ls = octaves(int(round(E - 0.1)))
    order = np.argsort(norm, kind='stable')
    L = np.empty(len(b))
    groups = np.array_split(order, len(Ls))
    for k, idx in enumerate(groups):
        L[idx] = Ls[k]
    mag = grid / cos
    return dict(unit=unit, mag=mag, L=L, groups=groups, Ls=Ls, ph=ph, s_eff=s_eff, grid=grid)


def factors(spec, pr):
    """Per-row factor on the xy frequency (None for the exact octave)."""
    kind, value = parse(spec)
    count = len(pr['L'])
    if kind == 'octave':
        return None
    if kind == 'shift':
        return np.full(count, value)
    if kind == 'jitter':
        v = np.random.default_rng(0).uniform(-1.0, 1.0, count)
        u = v * np.log1p(value)
        for idx in pr['groups']:
            u[idx] -= u[idx].mean()
        return np.exp(u)
    if kind == 'levels':
        c = np.empty(count)
        for k, idx in enumerate(pr['groups'][::-1]):          # top group (L = 1) first
            c[idx] = (1.0 + value) ** (1 if k % 2 == 0 else -1)
        return c
    if kind in ('top', 'rest'):
        top = pr['L'] == 1
        return np.where(top, value, 1.0) if kind == 'top' else np.where(top, 1.0, value)
    raise AssertionError(kind)


def transform(B, E, spec):
    """The spec's Fourier matrix (128 x 3, B's dtype and device) from ntrl-demo's draw B for a maze of extent E cells."""
    pr = parts(B, E)
    c = factors(spec, pr)
    if c is None:
        xy = pr['unit'] * (pr['mag'] / pr['L'])[:, None]       # grid_sync octave, the same expression
    else:
        xy = pr['unit'] * (pr['mag'] / pr['L'] * c)[:, None]
    out = np.concatenate([xy, (pr['ph'] * pr['s_eff'])[:, None]], 1)
    return torch.tensor(out, dtype=B.dtype, device=B.device)


def describe(Bm, E, n, spec, B0=None):
    """Frequency statistics in grid units (dominant xy component / f_grid) and the lock to the octave values.

    edge phase error of a row = pi |d - d*| n / 2, d = its dominant component / f_grid, d* = the nearest power of two
    (in log scale): the phase mismatch, relative to an exactly locked row, at the outermost grid line (all rows agree at
    the maze centre x = 0). 'locked' = edge error < pi / 4. With B0 (ntrl-demo's draw), also the factor statistics."""
    b = Bm.detach().cpu().double().numpy()
    grid = E / 2.0
    d = np.abs(b[:, :2]).max(1) / grid
    mag = np.linalg.norm(b[:, :2], axis=1) / grid
    k = np.round(-np.log2(d))
    target = 2.0 ** (-k)
    edge = np.pi * np.abs(d - target) * n / 2.0
    out = dict(spec=spec, f_grid=grid, dom_over_grid_quantiles=np.quantile(d, [0, 0.1, 0.5, 0.9, 1]).tolist(),
               mag_over_grid_quantiles=np.quantile(mag, [0, 0.1, 0.5, 0.9, 1]).tolist(),
               dom_top_mean=float(d[d >= np.quantile(d, 0.8)].mean()),
               edge_phase_error_median=float(np.median(edge)), edge_phase_error_mean=float(edge.mean()),
               edge_phase_error_top_octave_median=float(np.median(edge[k == k.min()])),
               locked_rows=int((edge < np.pi / 4).sum()), rows=int(len(b)),
               log2_offset_from_octave_quantiles=np.quantile(np.log2(d / target), [0, 0.1, 0.5, 0.9, 1]).tolist(),
               phi_col_std=float(b[:, 2].std()), b_abs_max=float(np.abs(b).max()))
    if B0 is not None:
        pr = parts(B0, E)
        c = factors(spec, pr)
        c = np.ones(len(b)) if c is None else c
        out['factor_quantiles'] = np.quantile(c, [0, 0.1, 0.5, 0.9, 1]).tolist()
        out['factor_log2_rms'] = float(np.sqrt(np.mean(np.log2(c) ** 2)))
        out['levels'] = {str(L): int((pr['L'] == L).sum()) for L in pr['Ls']}
    return out
