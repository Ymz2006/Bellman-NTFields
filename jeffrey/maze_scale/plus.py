"""The 'plus' variant of a maze: a centred + whose interior walls are removed (plusprepare.py / plusntrl.py).

install(width) patches this process's maze module (maze.py is unchanged on disk): maze.generate is wrapped so that, after
the usual Kruskal + loops maze, every interior wall between two cells of the + is removed, and Maze.describe() reports
variant='plus' with the + geometry. The + is the union of a vertical and a horizontal band of `width` cells through the
centre, running the full length of the maze (cells with column or row in [c0, c0 + width), c0 = (n - width) / 2).
Walls on the +'s outline (between a + cell and an outside cell) and the border are left as the maze had them.
Area of the + = 2 width n - width^2 cells; half the maze needs width = n (1 - 1/sqrt 2) (6.44 for n = 22). Walls lie on
the cell grid and a centred + on an even n needs an even width, so the default is the nearest, 6 (228 of 484 cells,
47 %).
"""
import numpy as np

import maze as MZ

_installed = {}


def plus_cells(n, width):
    c0 = (n - width) // 2
    band = np.zeros(n, bool)
    band[c0:c0 + width] = True
    return band[:, None] | band[None, :]                   # (column, row)


def open_plus(V, H, width):
    """Copies of V (n+1, n) and H (n, n+1) with walls between two + cells removed (border kept)."""
    n = H.shape[0]
    P = plus_cells(n, width)
    V, H = V.copy(), H.copy()
    for i in range(1, n):                                     # V[i, j]: between cells (i - 1, j) and (i, j)
        for j in range(n):
            if P[i - 1, j] and P[i, j]:
                V[i, j] = False
    for i in range(n):                                        # H[i, j]: between cells (i, j - 1) and (i, j)
        for j in range(1, n):
            if P[i, j - 1] and P[i, j]:
                H[i, j] = False
    return V, H


def install(width=6):
    if _installed:
        assert _installed['width'] == width
        return
    generate, describe = MZ.generate, MZ.Maze.describe

    def plus_generate(n, seed, **kw):
        V, H = generate(n, seed, **kw)
        V2, H2 = open_plus(V, H, width)
        _installed.setdefault('removed', {})[n] = int(V.sum() + H.sum() - V2.sum() - H2.sum())
        return V2, H2

    def plus_describe(self):
        d = describe(self)
        P = plus_cells(self.n, width)
        d.update(variant='plus', plus_width=width, plus_cells=int(P.sum()), plus_area_fraction=float(P.mean()),
                 plus_walls_removed=_installed.get('removed', {}).get(self.n))
        return d

    MZ.generate = plus_generate
    MZ.Maze.describe = plus_describe
    _installed['width'] = width
