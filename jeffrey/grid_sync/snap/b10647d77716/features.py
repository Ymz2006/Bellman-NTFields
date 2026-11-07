"""Grid-synchronised Fourier matrices for ntrl-demo on the jeffrey/maze_scale mazes.

ntrl-demo's network embeds a configuration q = (x, y, phi) (normalized: x, y in [-0.5, 0.5], phi in [-phi_max, phi_max])
as [sin(2 pi B q), cos(2 pi B q)] with B (128 x 3) ~ N(0, 1) truncated at 2. A row b is a plane wave with frequency
vector (b_x, b_y, b_phi) in cycles per normalized unit. The maze's walls lie on the grid lines x = i and y = j of
pitch p = 1 / E (E = n + 0.1, maze.py), so a wave whose dominant xy component is f_grid = 1 / (2 p) = E / 2 has a half
period of exactly one cell along that axis: shifting by one cell flips its sign, and every grid line sees the same
feature (up to sign). That is the grid resonance tested here.

Direction: a wave with xy direction at angle theta' to the nearest grid axis (theta' in [0, 45 deg]) keeps its
crests one cell apart along that axis only if |f_xy| cos theta' = f_grid, i.e. |f_xy| = f_grid / cos theta': its crest
spacing along the axis (the hypotenuse) is one cell and its half period along its own direction is p cos theta'. In
the frequency plane these rows lie on the square max(|f_x|, |f_y|) = f_grid.

Modes (all keep the row directions and the phi column of the same seeded draw; the network, data and training are
unchanged; s_eff = f_grid / median_i max(|b_x|, |b_y|), so the median row of B x s_eff is on the grid frequency):

  ring      |f_xy| = f_grid / cos theta'   every row in sync with the grid lines it is closest to perpendicular to
                                            (rows at 0 / 45 deg with both families)                          [main]
  circle    |f_xy| = c f_grid               same directions, constant magnitude c = mean over rows of 1 / cos theta'
                                            (the ring's mean): the usual single-frequency encoding, in sync only
                                            where c cos theta' = 1                                          [control]
  literal   |f_xy| = f_grid cos theta'      half period along the wave = one cell / cos theta' (the formula read
                                            literally); in sync only at theta' = 0                          [variant]
  scaled    B x s_eff                        Gaussian disk at the grid scale (maze_scale/fourier.py with s = s_eff)
  snap      B x s_eff, then every row whose dominant component is >= f_grid / 2 rescaled so that component is the
            nearest grid harmonic m f_grid (m >= 1); lower rows unchanged (the disk, with its upper part locked)

phi column: b_phi x s_eff in every mode (s_eff is proportional to E and phi = RHO theta / E, so the heading
frequencies per radian are the same at every maze size).
"""
import numpy as np
import torch

MODES = ['ring', 'circle', 'literal', 'scaled', 'snap']


def draw(seed):
    """ntrl-demo's B for a training seed (models/metric Model.train: torch.manual_seed(seed) in train.py, then
    torch.normal(0, 1, (128, 3)) and trunc_normal_(0, 1, -2, 2) before any other draw), without touching the global
    generator. Checked equal to the B of the registered maze-scale ntrl runs of seeds 87358, 1, 2."""
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        B = torch.normal(0, 1, size=(128, 3))
        torch.nn.init.trunc_normal_(B, mean=0.0, std=1, a=-2.0, b=2.0)
    finally:
        torch.random.set_rng_state(state)
    return B


def transform(B, E, mode):
    """The mode's Fourier matrix (128 x 3, float32, same device) from ntrl-demo's draw B for a maze of extent E cells."""
    assert mode in MODES, mode
    b = B.detach().cpu().double().numpy().copy()
    xy, ph = b[:, :2].copy(), b[:, 2].copy()
    grid = E / 2.0
    dom = np.abs(xy).max(1)
    norm = np.linalg.norm(xy, axis=1)
    cos = dom / norm
    unit = xy / norm[:, None]
    s_eff = grid / float(np.median(dom))
    if mode == 'ring':
        xy = unit * (grid / cos)[:, None]
    elif mode == 'circle':
        xy = unit * (grid * float(np.mean(1.0 / cos)))
    elif mode == 'literal':
        xy = unit * (grid * cos)[:, None]
    elif mode == 'scaled':
        xy = xy * s_eff
    elif mode == 'snap':
        xy = xy * s_eff
        d = np.abs(xy).max(1)
        m = np.round(d / grid)
        hi = m >= 1
        xy[hi] *= (m[hi] * grid / d[hi])[:, None]
    out = np.concatenate([xy, (ph * s_eff)[:, None]], 1)
    return torch.tensor(out, dtype=B.dtype, device=B.device)


def describe(B, E, n):
    """Frequency statistics of a Fourier matrix in grid units (f_grid = E / 2 cycles per unit = half period one cell)."""
    b = B.detach().cpu().double().numpy()
    grid = E / 2.0
    dom = np.abs(b[:, :2]).max(1) / grid
    mag = np.linalg.norm(b[:, :2], axis=1) / grid
    harmonic = np.maximum(np.round(dom), 1)
    lock = np.abs(dom - harmonic)                    # 0 = dominant component exactly on a grid harmonic
    return dict(f_grid=grid, dom_over_grid_quantiles=np.quantile(dom, [0, 0.1, 0.5, 0.9, 1]).tolist(),
                mag_over_grid_quantiles=np.quantile(mag, [0, 0.1, 0.5, 0.9, 1]).tolist(),
                lock_error_median=float(np.median(lock)), locked_rows=int((lock < 1e-6).sum()),
                phase_drift_over_maze_median_rad=float(np.median(lock) * np.pi * n),
                low_rows=int((dom < 0.5).sum()), b_abs_max=float(np.abs(b).max()), phi_col_std=float(b[:, 2].std()))
