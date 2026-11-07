"""Travel-time fields of individual segment-DP chain levels (report/levelfields13.png, levelfields22.png).

    CUDA_VISIBLE_DEVICES=0 ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/levelfields.py

For each travel-distance bin, one query of that bin (the one whose straight-line gap is closest to the bin's median)
picks its goal, its start heading and the chain level that serves its gap. Each column shows T(x, goal) at that
heading (theta of the nearest truth-grid node), in cells, cropped to the level's band around the goal: the truth T*
(truth_fields.npy of the prepared run, bilinear onto the same grid) and the level network alone from the segment DP +
wall term (weight 0.3) chain on the plain ntrl-demo base and on the B x s base (fourier.py), training seed 87358. A level
is trained only for pairs whose gap lies in its band; outside [e_lo (1 - blend), e_hi (1 + blend)] the map is faded.
Colour range (truth inside the band, 1st-90th percentile) and contour spacing are per column, the same for all three rows. Grey =
C-space obstacle at that heading. Start (circle), goal (square), true path (dashed white), and the chain's planned path
for this query (planning seed 0, red; the whole chain drives it, not only this level). Inference only.
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import maze as MZ  # noqa: E402
import report as R  # noqa: E402
from common import bin_labels  # noqa: E402
from fields import load_chain  # noqa: E402

OUT = HERE / 'report'
SEED = 87358
RES = 260
WALL = '#8f8e89'


def chains(n):
    """{'plain' | 'scaled': (run, base scale)} for the w0.3 chain at seed SEED on maze n."""
    out = {}
    for d in R.runs('segdp'):
        c = d.get('config', {})
        if d['maze']['n'] != n or d['seed'] != SEED or not d.get('normal_term') or c.get('normal_weight') != 0.3:
            continue
        if c.get('ratio', 1.35) != 1.35 or c.get('candidates', 128) != 128:
            continue
        b = R.base_scale(d)
        out['plain' if b == 1.0 else 'scaled'] = (d, b)
    return out


def nice_step(span):
    raw = span / 14
    for s in (0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5):
        if s >= raw:
            return s
    return 10.0


def figure(n):
    prep = {d['maze']['n']: d for d in R.runs('prepare')}[n]
    pd = prep['_dir']
    maze = MZ.Maze(n)
    dev = maze.device
    q = np.load(pd / 'queries.npz')
    tp = np.load(pd / 'truth_paths.npy', allow_pickle=True)
    truth = np.load(pd / 'truth_fields.npy', mmap_mode='r')
    ths = np.linspace(-maze.phi_max, maze.phi_max, truth.shape[3])
    ch = chains(n)
    rows = [('truth', None, None)] + [(k, *ch[k]) for k in ('plain', 'scaled') if k in ch]
    loaded = {k: (load_chain(d['_dir'] / 'chain/field.pt')[0], np.load(d['_dir'] / 'paths_seed0.npy', allow_pickle=True),
                  np.load(d['_dir'] / 'rows.npz')) for k, d, _ in rows[1:]}
    ref = loaded['plain'][0]
    edges = (ref.edges / maze.pitch).cpu().numpy()
    blend = ref.blend
    labels = bin_labels()
    bins = sorted(set(q['bin'].tolist()))
    fig, axes = plt.subplots(len(rows), len(bins), figsize=(2.75 * len(bins), 3.3 * len(rows)))
    axes = np.array(axes).reshape(len(rows), -1)
    meta = []
    for c, b in enumerate(bins):
        idx = np.nonzero(q['bin'] == b)[0]
        k = int(idx[np.argmin(np.abs(q['euclid_cells'][idx] - np.median(q['euclid_cells'][idx])))])
        gi = int(q['goal_index'][k])
        it = int(np.argmin(np.abs(ths - q['start'][k][2])))
        theta = float(ths[it])
        gap0 = float(np.linalg.norm(q['start'][k] - q['goal'][k])) / maze.pitch
        m = int(np.searchsorted(edges, gap0))                         # level serving this gap (0 = base)
        lo = 0.0 if m == 0 else float(edges[m - 1] * (1 - blend))
        hi = float(edges[m] * (1 + blend)) if m < len(edges) else float('inf')
        # window: the band around the goal (and the true path), clipped to the maze
        g = q['goal'][k][:2]
        path = np.asarray(tp[k])
        rad = max(min(hi, 30.0) * 1.08, 1.6) * maze.pitch
        x0, x1 = max(-0.5, min(g[0] - rad, path[:, 0].min() - 0.5 * maze.pitch)), min(0.5, max(g[0] + rad, path[:, 0].max() + 0.5 * maze.pitch))
        y0, y1 = max(-0.5, min(g[1] - rad, path[:, 1].min() - 0.5 * maze.pitch)), min(0.5, max(g[1] + rad, path[:, 1].max() + 0.5 * maze.pitch))
        side = max(x1 - x0, y1 - y0)                                   # square window
        cx, cy = np.clip((x0 + x1) / 2, -0.5 + side / 2, 0.5 - side / 2), np.clip((y0 + y1) / 2, -0.5 + side / 2, 0.5 - side / 2)
        x0, x1, y0, y1 = cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2
        X, Y = torch.meshgrid(torch.linspace(x0, x1, RES, device=dev), torch.linspace(y0, y1, RES, device=dev), indexing='ij')
        pts = torch.stack([X.reshape(-1), Y.reshape(-1), torch.full((RES * RES,), theta, device=dev)], 1)
        wall = (maze.clearance(pts) < 0).reshape(RES, RES).cpu().numpy()
        goal = torch.tensor(q['goal'][k], device=dev, dtype=torch.float32)
        gap = (torch.linalg.vector_norm(pts[:, :2] - goal[None, :2], dim=1) / maze.pitch).reshape(RES, RES).cpu().numpy()
        out_band = (gap < lo) | (gap > hi)
        # truth slice, bilinear onto the window (grid nodes span [-0.5, 0.5])
        ts = torch.tensor(np.asarray(truth[gi, :, :, it], np.float32), device=dev)
        grid = torch.stack([pts[:, 1], pts[:, 0]], 1).reshape(1, RES, RES, 2) * 2   # grid_sample: x -> last dim (y index)
        tt = F.grid_sample(ts[None, None], grid, mode='bilinear', align_corners=True)[0, 0].cpu().numpy() / maze.pitch
        free_in = ~wall & ~out_band
        vmin, vmax = np.percentile(tt[free_in], [1, 90]) if free_in.any() else (tt.min(), tt.max())
        step = nice_step(vmax - vmin)
        col = dict(bin=labels[b], query=k, travel=float(q['d_cells'][k]), gap=gap0, theta=theta, level=m, band=[lo, hi],
                   window_cells=float(side / maze.pitch), contour_step=step)
        for r, (name, d, bscale) in enumerate(rows):
            ax = axes[r, c]
            if name == 'truth':
                img = tt
                title = f'{labels[b]} cells — truth T*\ntravel {q["d_cells"][k]:.1f}, gap {gap0:.1f}; level {m} band {lo:.1f}–{hi:.1f}'
            else:
                field, paths, rr = loaded[name]
                with torch.no_grad():
                    img = (field.models[m].out(torch.cat([pts, goal[None].expand(len(pts), 3)], 1))[0][:, 0] / maze.pitch).reshape(RES, RES).cpu().numpy()
                ok, conv = bool(rr['success'][0, k]), bool(rr['converged'][0, k])
                base = 'plain base' if name == 'plain' else f'B × {bscale:g} base'
                title = f'DP, {base}, level {m}\nplanned: {"success" if ok else ("stall" if not conv else "collision")}'
                col[name] = dict(success=ok, converged=conv)
            ext = (x0, x1, y0, y1)
            im = ax.imshow(np.ma.masked_where(wall, img).T, origin='lower', extent=ext, cmap='viridis_r', vmin=vmin, vmax=vmax, interpolation='nearest')
            xs, ys = np.linspace(x0, x1, RES), np.linspace(y0, y1, RES)
            ax.contour(xs, ys, np.where(wall, np.nan, img).T, levels=np.arange(np.floor(vmin / step) * step, vmax + step, step),
                       colors='white', linewidths=0.55, alpha=0.9)
            fade = np.zeros((RES, RES, 4))
            fade[out_band & ~wall] = (1, 1, 1, 0.62)
            fade[wall] = matplotlib.colors.to_rgba(WALL)
            ax.imshow(fade.transpose(1, 0, 2), origin='lower', extent=ext, interpolation='nearest')
            if name != 'truth':
                p = np.asarray(loaded[name][1][k])
                drv = p if loaded[name][2]['converged'][0, k] else p[:-1]
                ax.plot(drv[:, 0], drv[:, 1], color='#e34948', lw=1.5)
            ax.plot(path[:, 0], path[:, 1], color='white', lw=1.1, ls='--')
            ax.plot(*q['start'][k][:2], 'o', color='#e34948', ms=6, mec='white', mew=1)
            ax.plot(*q['goal'][k][:2], 's', color='#e34948', ms=6, mec='white', mew=1)
            ax.set_title(title, fontsize=7.5, pad=3)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == len(rows) - 1:
                cb = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.045, pad=0.03, extend='both')
                cb.ax.tick_params(labelsize=6.5)
                cb.set_label(f'T (cells); window {side / maze.pitch:.1f} cells; contours every {step:g}', fontsize=7)
        meta.append(col)
    fig.suptitle(f'{n} × {n}: travel time to the goal (cells) of one chain level per distance range, at the query\'s start heading; '
                 f'faded = outside the level\'s gap band', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=1.6)
    fig.savefig(OUT / f'levelfields{n}.png', dpi=115)
    plt.close(fig)
    return meta


def main():
    res = {n: figure(n) for n in (13, 22)}
    (OUT / 'levelfields.json').write_text(json.dumps(res, indent=1, default=float))
    for n, cols in res.items():
        for c in cols:
            print(n, c['bin'], 'level', c['level'], 'band', [round(v, 1) for v in c['band']], 'plain', c.get('plain'), 'scaled', c.get('scaled'))


if __name__ == '__main__':
    main()
