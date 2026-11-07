"""The 'gap' mazes: the four 8 x 8 maze quadrants of the 22 x 22 cross maze, pushed apart.

The 22 x 22 plus maze (jeffrey/maze_scale/plus.py: maze seed 1022, every interior wall inside a centred 6-cell + removed)
is four 8 x 8 maze blocks in the corners (cell columns / rows 0-7 and 14-21) around an open band 6 cells wide. Gap scale
k widens that band to G = 6 k cells: the maze is n = 16 + G cells a side, the four blocks keep every wall they had
(their interior walls and their outline walls towards the band, doorways included), the band holds no walls, and the
outer border is closed all round. k = 1 is the 22 x 22 plus maze itself (checked in check()).

install(k) patches this process's maze module (maze.py is unchanged on disk), as plus.py does: Maze(16 + 6 k) builds
the gap maze, and Maze.describe() reports variant 'gap', gap_scale, plus_width = G (the band, so the plus helpers and
the plusmag design see the open band) and the quadrant blocks.
"""
import numpy as np

import maze as MZ

BASE_N, BASE_SEED, BASE_WIDTH, BLOCK = 22, 1022, 6, 8
_installed = {}


def base_walls(generate):
    """V (23, 22), H (22, 23) of the 22 x 22 plus maze, from the unpatched generator."""
    import plus as PL
    V, H = generate(BASE_N, BASE_SEED)
    return PL.open_plus(V, H, BASE_WIDTH)


def expand(V, H, k):
    """Walls of gap scale k from the plus maze's V, H."""
    n0 = V.shape[1]
    G = BASE_WIDTH * k
    n = 2 * BLOCK + G
    d = G - BASE_WIDTH

    def line(i):                      # a grid line of the base maze -> the gap maze (None: strictly inside the band)
        if i <= BLOCK:
            return i
        if i >= BLOCK + BASE_WIDTH:
            return i + d
        return None

    def cells(j):                     # a cell row / column of the base maze -> the gap maze's cells
        if j < BLOCK:
            return [j]
        if j >= BLOCK + BASE_WIDTH:
            return [j + d]
        return list(range(BLOCK, BLOCK + G))

    V2 = np.zeros((n + 1, n), bool)
    H2 = np.zeros((n, n + 1), bool)
    for i, j in zip(*np.nonzero(V)):
        li = line(i)
        assert li is not None, f'vertical wall inside the band at line {i}'
        cj = cells(j)
        assert len(cj) == 1 or i in (0, n0), f'vertical wall at line {i} across band row {j} that is not the border'
        V2[li, cj] = True
    for i, j in zip(*np.nonzero(H)):
        lj = line(j)
        assert lj is not None, f'horizontal wall inside the band at line {j}'
        ci = cells(i)
        assert len(ci) == 1 or j in (0, n0), f'horizontal wall at line {j} across band column {i} that is not the border'
        H2[ci, lj] = True
    V2[0, :] = V2[n, :] = True                      # outer border
    H2[:, 0] = H2[:, n] = True
    return V2, H2


def region_of(n, x_cells, y_cells):
    """Region per position in cell units: 0..3 = quadrant (0 bottom-left, 1 bottom-right, 2 top-left, 3 top-right), 4 = open."""
    G = n - 2 * BLOCK
    cx = np.floor(np.asarray(x_cells)).astype(int)
    cy = np.floor(np.asarray(y_cells)).astype(int)
    left, right = cx < BLOCK, cx >= BLOCK + G
    low, high = cy < BLOCK, cy >= BLOCK + G
    out = np.full(cx.shape, 4)
    out[left & low] = 0
    out[right & low] = 1
    out[left & high] = 2
    out[right & high] = 3
    return out


CATEGORIES = ['same quadrant', 'adjacent quadrants', 'diagonal quadrants', 'quadrant - open', 'open - open']


def category(ra, rb):
    """Region-pair category per query (indices into CATEGORIES) from the start and goal regions."""
    ra, rb = np.asarray(ra), np.asarray(rb)
    out = np.empty(ra.shape, int)
    qa, qb = ra < 4, rb < 4
    out[qa & qb & (ra == rb)] = 0
    diag = ((ra == 0) & (rb == 3)) | ((ra == 3) & (rb == 0)) | ((ra == 1) & (rb == 2)) | ((ra == 2) & (rb == 1))
    out[qa & qb & (ra != rb) & ~diag] = 1
    out[qa & qb & diag] = 2
    out[qa ^ qb] = 3
    out[~qa & ~qb] = 4
    return out


def install(k):
    if _installed:
        assert _installed['k'] == k
        return
    generate, describe = MZ.generate, MZ.Maze.describe
    n_gap = 2 * BLOCK + BASE_WIDTH * k
    V0, H0 = base_walls(generate)

    def gap_generate(n, seed, **kw):
        if n == n_gap:
            return expand(V0, H0, k)
        return generate(n, seed, **kw)

    def gap_describe(self):
        d = describe(self)
        if self.n == n_gap:
            d.update(variant='gap', gap_scale=k, plus_width=BASE_WIDTH * k, block=BLOCK, base=dict(n=BASE_N, seed=BASE_SEED, plus_width=BASE_WIDTH))
        return d

    MZ.generate = gap_generate
    MZ.Maze.describe = gap_describe
    _installed['k'] = k
    _installed['n'] = n_gap


def check():
    """k = 1 reproduces the plus maze; at every k the four blocks keep their walls (interior and outline) and only the
    border grows."""
    V0, H0 = base_walls(MZ.generate)
    V1, H1 = expand(V0, H0, 1)
    assert np.array_equal(V0, V1) and np.array_equal(H0, H1), 'k = 1 differs from the plus maze'
    for k in (2, 4, 8, 12):
        V, H = expand(V0, H0, k)
        n = 2 * BLOCK + BASE_WIDTH * k
        d = n - BASE_N
        for lo in (False, True):                 # block position along x: left (cells 0-7) or right (14-21)
            for bo in (False, True):             # along y: bottom or top
                cx0, cy0 = (14 if lo else 0), (14 if bo else 0)
                sx, sy = (d if lo else 0), (d if bo else 0)
                assert np.array_equal(V0[cx0:cx0 + 9, cy0:cy0 + 8], V[cx0 + sx:cx0 + sx + 9, cy0 + sy:cy0 + sy + 8]), (k, lo, bo, 'V')
                assert np.array_equal(H0[cx0:cx0 + 8, cy0:cy0 + 9], H[cx0 + sx:cx0 + sx + 8, cy0 + sy:cy0 + sy + 9]), (k, lo, bo, 'H')
        assert V.sum() + H.sum() == V0.sum() + H0.sum() + 4 * d, (k, int(V.sum() + H.sum()))
        assert not V[1:n, BLOCK:BLOCK + BASE_WIDTH * k].any() and not H[BLOCK:BLOCK + BASE_WIDTH * k, 1:n].any(), 'walls inside the band'
    return True
