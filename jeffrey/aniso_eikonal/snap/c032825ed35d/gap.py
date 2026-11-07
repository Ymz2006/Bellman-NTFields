"""The 'gap' mazes: the four maze quadrants of a base maze, pushed apart by an open band of 6 k cells.

Base 'cross22' (the default, the gap study): the 22 x 22 plus maze (jeffrey/maze_scale/plus.py: maze seed 1022, every
interior wall inside a centred 6-cell + removed) is four 8 x 8 maze blocks in the corners (cell columns / rows 0-7 and
14-21) around an open band 6 cells wide. Gap scale k widens that band to G = 6 k cells: the maze is n = 16 + G cells a
side, the four blocks keep every wall they had (their interior walls and their outline walls towards the band,
doorways included), the band holds no walls, and the outer border is closed all round. k = 1 is the 22 x 22 plus maze
itself (checked in check()).

Base 'maze10': jeffrey/maze_scale's 10 x 10 maze (seed 1010, no walls removed) cut along its two centre lines into four
5 x 5 blocks, with the same bands between them (G = 6 k cells, n = 10 + G). Every wall of the 10 x 10 maze is kept;
the walls on the two centre lines (the blocks' sides towards each other, doorways included) become the outline of the
blocks on both sides of the band. So the blocks together are exactly the 10 x 10 maze; k = 1 is a 16 x 16 maze.

install(k, base) patches this process's maze module (maze.py is unchanged on disk), as plus.py does: Maze(2 BLOCK + 6 k)
builds the gap maze, and Maze.describe() reports variant 'gap', gap_scale, plus_width = G (the band, so the plus
helpers and the plusmag design see the open band), the block size and the base (with its name for every base but the
default). BASE_WIDTH is the band per unit of k (6 for every base); BASE_OPEN is the + width opened in the base maze.
"""
import numpy as np

import maze as MZ

BASES = {'cross22': dict(n=22, seed=1022, open_width=6, block=8),
         'maze10': dict(n=10, seed=1010, open_width=0, block=5)}
BASE_WIDTH = 6                                     # band per unit of the gap scale: G = 6 k cells
BASE = 'cross22'
BASE_N, BASE_SEED, BASE_OPEN, BLOCK = 22, 1022, 6, 8
_installed = {}


def use_base(base):
    """Set the module's base constants (install does this; also for tools that only build the walls)."""
    global BASE, BASE_N, BASE_SEED, BASE_OPEN, BLOCK
    cfg = BASES[base]
    BASE, BASE_N, BASE_SEED, BASE_OPEN, BLOCK = base, cfg['n'], cfg['seed'], cfg['open_width'], cfg['block']


def base_name(info):
    """The base of a maze.json / describe() dict of a gap maze ('cross22' when not named)."""
    return (info.get('base') or {}).get('name', 'cross22')


def base_walls(generate):
    """V (n0 + 1, n0), H (n0, n0 + 1) of the base maze, from the unpatched generator."""
    V, H = generate(BASE_N, BASE_SEED)
    if BASE_OPEN:
        import plus as PL
        return PL.open_plus(V, H, BASE_OPEN)
    return V, H


def expand(V, H, k):
    """Walls of gap scale k from the base maze's V, H."""
    n0 = V.shape[1]
    G = BASE_WIDTH * k
    n = 2 * BLOCK + G
    d = G - BASE_OPEN

    def lines(i):                     # a grid line of the base maze -> the gap maze's lines ([]: strictly inside the band)
        if BASE_OPEN == 0 and i == BLOCK:
            return [BLOCK, BLOCK + G]               # the cut line: the blocks on both sides keep its walls
        if i <= BLOCK:
            return [i]
        if i >= BLOCK + BASE_OPEN:
            return [i + d]
        return []

    def cells(j):                     # a cell row / column of the base maze -> the gap maze's cells
        if j < BLOCK:
            return [j]
        if j >= BLOCK + BASE_OPEN:
            return [j + d]
        return list(range(BLOCK, BLOCK + G))

    V2 = np.zeros((n + 1, n), bool)
    H2 = np.zeros((n, n + 1), bool)
    for i, j in zip(*np.nonzero(V)):
        li = lines(i)
        assert li, f'vertical wall inside the band at line {i}'
        cj = cells(j)
        assert len(cj) == 1 or i in (0, n0), f'vertical wall at line {i} across band row {j} that is not the border'
        for l in li:
            V2[l, cj] = True
    for i, j in zip(*np.nonzero(H)):
        lj = lines(j)
        assert lj, f'horizontal wall inside the band at line {j}'
        ci = cells(i)
        assert len(ci) == 1 or j in (0, n0), f'horizontal wall at line {j} across band column {i} that is not the border'
        for l in lj:
            H2[ci, l] = True
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


def install(k, base='cross22'):
    if _installed:
        assert _installed['k'] == k and _installed['base'] == base, (_installed, k, base)
        return
    use_base(base)
    generate, describe = MZ.generate, MZ.Maze.describe
    n_gap = 2 * BLOCK + BASE_WIDTH * k
    V0, H0 = base_walls(generate)
    base_info = dict(n=BASE_N, seed=BASE_SEED, plus_width=BASE_OPEN)
    if base != 'cross22':
        base_info['name'] = base

    def gap_generate(n, seed, **kw):
        if n == n_gap:
            return expand(V0, H0, k)
        return generate(n, seed, **kw)

    def gap_describe(self):
        d = describe(self)
        if self.n == n_gap:
            d.update(variant='gap', gap_scale=k, plus_width=BASE_WIDTH * k, block=BLOCK, base=dict(base_info))
        return d

    MZ.generate = gap_generate
    MZ.Maze.describe = gap_describe
    _installed.update(k=k, n=n_gap, base=base)


def check_blocks(V0, H0, V, H, k):
    """Every cell of the four blocks has the same four walls as its base cell, and the band holds no walls but the border."""
    n0 = V0.shape[1]
    G = BASE_WIDTH * k
    n = 2 * BLOCK + G
    d = G - BASE_OPEN
    base_cells = [c for c in range(n0) if c < BLOCK or c >= BLOCK + BASE_OPEN]
    to = lambda c: c if c < BLOCK else c + d
    for c in base_cells:
        for r in base_cells:
            c2, r2 = to(c), to(r)
            assert (V0[c, r], V0[c + 1, r], H0[c, r], H0[c, r + 1]) == (V[c2, r2], V[c2 + 1, r2], H[c2, r2], H[c2, r2 + 1]), (k, c, r)
    assert not V[1:n, BLOCK:BLOCK + G].any() and not H[BLOCK:BLOCK + G, 1:n].any(), 'walls inside the band'
    assert not V[BLOCK + 1:BLOCK + G, :].any() and not H[:, BLOCK + 1:BLOCK + G].any(), 'walls inside the band'
    return True


def check(base='cross22'):
    """cross22: k = 1 reproduces the plus maze; at every k and for every base the four blocks keep their walls (interior
    and outline) and only the border grows."""
    prev = BASE
    use_base(base)
    try:
        V0, H0 = base_walls(MZ.generate)
        if base == 'cross22':
            V1, H1 = expand(V0, H0, 1)
            assert np.array_equal(V0, V1) and np.array_equal(H0, H1), 'k = 1 differs from the plus maze'
        for k in (1, 2, 4, 8, 12):
            V, H = expand(V0, H0, k)
            check_blocks(V0, H0, V, H, k)
            if base == 'cross22':
                assert V.sum() + H.sum() == V0.sum() + H0.sum() + 4 * (BASE_WIDTH * k - BASE_OPEN), (k, int(V.sum() + H.sum()))
    finally:
        use_base(prev)
    return True
