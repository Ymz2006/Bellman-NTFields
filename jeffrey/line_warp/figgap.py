"""One gap maze and its warped space as a two-panel image (a maze module is patched per process, so report/gaps.png is stitched from these).

    python figgap.py <prepared run> <fit run> <out.png> <title>
    python figgap.py --stitch <out.png> <row1 panels ...> -- <row2 panels ...>
"""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / 'jeffrey/maze_scale'), str(ROOT / 'jeffrey/cross_warp')]


def panels(prepared, fit, out, title):
    import figs as G
    import fit as FT
    import view as V
    G.style()
    maze, _ = FT.load_maze(Path(prepared), 'cuda:0')
    w = G.load_warp(Path(fit), maze)
    fig, ax = plt.subplots(1, 2, figsize=(8, 4.1))
    V.draw_maze(ax[0], lambda x: x, maze, 0.0, grid=False)
    V.draw_maze(ax[1], w.plane, maze, 0.0, grid=False)
    ax[0].set_title(f'{title}: the maze', fontsize=11, loc='left')
    ax[1].set_title('its learned warp', fontsize=11, loc='left')
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def stitch(out, rows):
    imgs = [[plt.imread(p) for p in r] for r in rows]
    h = sum(max(i.shape[0] for i in r) for r in imgs)
    w = max(sum(i.shape[1] for i in r) for r in imgs)
    canvas = np.ones((h, w, 4))
    y = 0
    for r in imgs:
        x = 0
        for i in r:
            canvas[y:y + i.shape[0], x:x + i.shape[1], :i.shape[2]] = i
            x += i.shape[1]
        y += max(i.shape[0] for i in r)
    plt.imsave(out, canvas)


if __name__ == '__main__':
    if sys.argv[1] == '--stitch':
        out, rest = sys.argv[2], sys.argv[3:]
        rows, cur = [], []
        for a in rest:
            if a == '--':
                rows.append(cur)
                cur = []
            else:
                cur.append(a)
        rows.append(cur)
        stitch(out, rows)
    else:
        panels(*sys.argv[1:5])
