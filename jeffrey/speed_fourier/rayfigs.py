"""Figure of the part-4 rows on the 13 x 13 maze, no training: report/rays.png.

    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= ~/.venvs/bellman-ntfield/bin/python jeffrey/speed_fourier/rayfigs.py

Panels: the walls with one ray and one landmark; the hit-count row of the ray pointing along +x and of the ray at 45 degrees; the exact integral of
1 / speed along +x; the landmark row (L1 norm over 64 directions of the integral differences to the landmark cell); the same landmark row with
1 / speed replaced by 1 (no walls, the control).
"""
import math
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
sys.path.insert(0, str(ROOT / 'jeffrey/grid_sync'))
import linefit as LF  # noqa: E402
import rays as RY  # noqa: E402
import speedfourier as SF  # noqa: E402
import maze as MZ  # noqa: E402

INK, ORANGE = '#15181d', '#eb6834'


def main():
    n = 13
    maze = MZ.Maze(n, 'cpu')
    s = LF.speed_map(maze)
    z = 1 / np.maximum(s, SF.SPEED_FLOOR)
    ax = LF.node_axis(maze)
    hits = RY.table(maze, 'hits', 16)[0]
    integral = RY.integral_rows(maze, z, 16)
    marks = RY.landmarks(n, 16)
    m = min(range(16), key=lambda k: sum((c - (n - 1) / 2) ** 2 for c in divmod(marks[k], n)))      # the landmark nearest the centre
    i, j = divmod(marks[m], n)
    lx, ly = ax[8 + 16 * i], ax[8 + 16 * j]
    mark = RY.landmark_rows(RY.integral_rows(maze, z, RY.DIRECTIONS), maze, marks, 'l1')[..., m]
    flat = RY.landmark_rows(RY.integral_rows(maze, np.ones_like(z), RY.DIRECTIONS), maze, marks, 'l1')[..., m]
    panels = [(1 - s, 'Walls (1 − speed), one ray, one landmark', 'Greys'), (hits[..., 0], 'Hit count, ray along +x: walls entered ahead', 'Blues'),
              (hits[..., 2], 'Hit count, ray at 45°', 'Blues'), (integral[..., 0], 'Integral of 1 / speed along +x (part 2)', 'Blues'),
              (mark, 'Landmark row: L1 norm of integral differences', 'Blues'), (flat, 'Same landmark row with no walls (control)', 'Blues')]
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.6), dpi=170)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02, wspace=0.06, hspace=0.14)
    for a, (img, title, cmap) in zip(axes.ravel(), panels):
        a.imshow(img.T, origin='lower', extent=[-0.5, 0.5, -0.5, 0.5], cmap=cmap, vmin=0 if cmap == 'Greys' else None, vmax=1.6 if cmap == 'Greys' else None)
        a.set_title(title, fontsize=10, color=INK, loc='left')
        a.set_xticks([]); a.set_yticks([])
    a = axes[0, 0]
    x0, y0 = ax[8 + 16 * 2], ax[8 + 16 * 9]
    a.annotate('', xy=(0.49, y0), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', color=ORANGE, lw=2))
    a.plot([x0], [y0], 'o', color=ORANGE, ms=5)
    for a in (axes[0, 0], axes[1, 1], axes[1, 2]):
        a.plot([lx], [ly], marker='*', color=ORANGE, ms=13, mec='white', mew=0.8)
    fig.savefig(HERE / 'report/rays.png', facecolor='white')
    print('ok', hits[..., 0].max(), mark.max(), flat.max(), 'landmark cell', (i, j))


if __name__ == '__main__':
    main()
