"""Data for the chain-level field viewer (report/levelexplorer/meta.json + <n>_<level>.bin).

    CUDA_VISIBLE_DEVICES=0 ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/levelexplorer.py

For every level of the segment DP + wall term (weight 0.3) chain (training seed 87358) on the 13 x 13 and 22 x 22
mazes: one query whose straight-line gap lies in the level's band (closest to the band's geometric centre; the closest
gap overall if none does) gives the goal and the start heading (nearest truth-grid theta). On a square window around
the goal covering the band, T(x, goal) in cells is evaluated for: the truth T* (truth_fields.npy, bilinear), this
level of the chain on the plain ntrl-demo base, the same level of the chain on the B x s base (fourier.py; s = 3 for
13 x 13, 6 for 22 x 22), and for reference the single ntrl-demo network and ntrl-demo with B x s. Two views per level: the window ('crop') and the whole maze ('full'). Grids are uint16 packed into lossless RGB PNGs (R = high byte, G = low byte, the five fields stacked vertically),
quantised per window over [tmin, tmax]; 65535 marks a C-space obstacle at that heading. Inference only.
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import maze as MZ  # noqa: E402
import plan as P  # noqa: E402
import report as R  # noqa: E402
from common import bin_labels  # noqa: E402
from fields import load_chain  # noqa: E402
from levelfields import chains  # noqa: E402

OUT = HERE / 'report' / 'levelexplorer'
SEED = 87358
SCALE = {13: 3.0, 22: 6.0}
METHODS = ['truth', 'dp_plain', 'dp_scaled', 'ntrl', 'ntrl_scaled']


def thin(p, k=400):
    p = np.asarray(p, np.float64)[:, :2]
    if len(p) > k:
        p = p[np.linspace(0, len(p) - 1, k).round().astype(int)]
    return p.round(5).tolist()


def build(n):
    prep = {d['maze']['n']: d for d in R.runs('prepare')}[n]
    pd = prep['_dir']
    maze = MZ.Maze(n)
    dev = maze.device
    q = np.load(pd / 'queries.npz')
    tp = np.load(pd / 'truth_paths.npy', allow_pickle=True)
    truth = np.load(pd / 'truth_fields.npy', mmap_mode='r')
    ths = np.linspace(-maze.phi_max, maze.phi_max, truth.shape[3])
    _, ntrl, _ = R.collect()
    fb = [d for d in R.runs('fourier') if d['maze']['n'] == n and d['seed'] == SEED and float(d.get('b_scale', 1)) == SCALE[n]]
    ch = chains(n)
    runs = dict(dp_plain=ch['plain'][0]['_dir'], dp_scaled=ch['scaled'][0]['_dir'], ntrl=ntrl[n][SEED]['_dir'], ntrl_scaled=fb[0]['_dir'])
    chain_plain = load_chain(runs['dp_plain'] / 'chain/field.pt')[0]
    chain_scaled = load_chain(runs['dp_scaled'] / 'chain/field.pt')[0]
    singles = dict(ntrl=P.load_field(runs['ntrl'] / 'training/checkpoint.pt'), ntrl_scaled=P.load_field(runs['ntrl_scaled'] / 'training/checkpoint.pt'))
    plans = {k: (np.load(v / 'paths_seed0.npy', allow_pickle=True), np.load(v / 'rows.npz')) for k, v in runs.items()}
    edges = (chain_plain.edges / maze.pitch).cpu().numpy()
    blend = chain_plain.blend
    max_gap = float(json.loads((runs['dp_plain'] / 'results.json').read_text())['max_gap_cells'])
    gaps = np.linalg.norm(q['start'] - q['goal'], axis=1) / maze.pitch
    labels = bin_labels()
    levels = []
    for m in range(len(edges) + 1):
        lo = 0.0 if m == 0 else float(edges[m - 1] * (1 - blend))
        hi = float(edges[m] * (1 + blend)) if m < len(edges) else max_gap
        centre = np.sqrt(max(lo, 0.3) * hi)
        inside = np.nonzero((gaps >= lo) & (gaps <= hi))[0]
        pool = inside if len(inside) else np.arange(len(gaps))
        k = int(pool[np.argmin(np.abs(np.log(gaps[pool] / centre)))])
        gi = int(q['goal_index'][k])
        it = int(np.argmin(np.abs(ths - q['start'][k][2])))
        theta = float(ths[it])
        g = q['goal'][k][:2]
        path = np.asarray(tp[k])
        rad = max(hi * 1.08, 1.6) * maze.pitch
        x0, x1 = max(-0.5, min(g[0] - rad, path[:, 0].min() - 0.5 * maze.pitch)), min(0.5, max(g[0] + rad, path[:, 0].max() + 0.5 * maze.pitch))
        y0, y1 = max(-0.5, min(g[1] - rad, path[:, 1].min() - 0.5 * maze.pitch)), min(0.5, max(g[1] + rad, path[:, 1].max() + 0.5 * maze.pitch))
        side = max(x1 - x0, y1 - y0)
        cx, cy = np.clip((x0 + x1) / 2, -0.5 + side / 2, 0.5 - side / 2), np.clip((y0 + y1) / 2, -0.5 + side / 2, 0.5 - side / 2)
        x0, x1, y0, y1 = cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2
        goal = torch.tensor(q['goal'][k], device=dev, dtype=torch.float32)
        ts = torch.tensor(np.asarray(truth[gi, :, :, it], np.float32), device=dev)

        def view(x0, x1, y0, y1, res, fname):
            X, Y = torch.meshgrid(torch.linspace(x0, x1, res, device=dev), torch.linspace(y0, y1, res, device=dev), indexing='ij')
            pts = torch.stack([X.reshape(-1), Y.reshape(-1), torch.full((res * res,), theta, device=dev)], 1)
            wall = (maze.clearance(pts) < 0).reshape(res, res).cpu().numpy()
            pairs = torch.cat([pts, goal[None].expand(len(pts), 3)], 1)
            grid = torch.stack([pts[:, 1], pts[:, 0]], 1).reshape(1, res, res, 2) * 2
            grids = dict(truth=F.grid_sample(ts[None, None], grid, mode='bilinear', align_corners=True)[0, 0].cpu().numpy() / maze.pitch)
            with torch.no_grad():
                for name, net in (('dp_plain', chain_plain.models[m]), ('dp_scaled', chain_scaled.models[m]), *singles.items()):
                    grids[name] = torch.cat([net.out(c)[0][:, 0] for c in pairs.split(100_000)]).div(maze.pitch).reshape(res, res).cpu().numpy()
            gap = np.linalg.norm(np.stack([X.cpu().numpy() - g[0], Y.cpu().numpy() - g[1]], -1), axis=-1) / maze.pitch
            band = ~wall & (gap >= lo) & (gap <= hi)
            tt = grids['truth']
            vb = np.percentile(tt[band], [1, 90]) if band.sum() > 20 else np.percentile(tt[~wall], [1, 90])
            va = np.percentile(tt[~wall], [0, 99])
            allv = np.concatenate([v[~wall] for v in grids.values()])
            tmin, tmax = float(np.percentile(allv, 0.05)), float(np.percentile(allv, 99.95))
            buf = []
            for name in METHODS:
                v = np.clip((grids[name] - tmin) / (tmax - tmin), 0, 1) * 65534
                v = v.round().astype(np.uint16)
                v[wall] = 65535
                buf.append(v.T.copy())                                   # rows = y, columns = x
            u = np.concatenate(buf, 0)                                   # (5 res, res): methods stacked vertically
            rgb = np.stack([(u >> 8).astype(np.uint8), (u & 255).astype(np.uint8), np.zeros_like(u, np.uint8)], -1)
            Image.fromarray(rgb, 'RGB').save(OUT / fname, optimize=True)
            return dict(file=fname, res=res, window=[float(x0), float(x1), float(y0), float(y1)], tmin=tmin, tmax=tmax,
                        band_range=[float(vb[0]), float(vb[1])], all_range=[float(va[0]), float(va[1])])

        crop = view(x0, x1, y0, y1, int(np.clip(side / maze.pitch * 16, 140, 260)), f'{n}_{m}.png')
        full = view(-0.5, 0.5, -0.5, 0.5, int(round(maze.E * (20 if n <= 13 else 16))), f'{n}_{m}_full.png')
        outcome = {}
        for name, (paths, rows) in plans.items():
            p = np.asarray(paths[k])
            conv, ok = bool(rows['converged'][0, k]), bool(rows['success'][0, k])
            outcome[name] = dict(result='success' if ok else ('stall' if not conv else 'collision'), path=thin(p if conv else p[:-1]))
        levels.append(dict(level=m, band=[lo, hi], crop=crop, full=full, query=k, bin=labels[int(q['bin'][k])], travel=float(q['d_cells'][k]),
                           gap=float(gaps[k]), gap_in_band=bool(len(inside)), theta=theta, start=q['start'][k][:2].tolist(),
                           goal=g.tolist(), truth_path=thin(path), outcome=outcome))
        print(n, m, f'band {lo:.2f}-{hi:.2f}', 'res', crop['res'], full['res'], 'query', k, labels[int(q['bin'][k])], {a: b['result'] for a, b in outcome.items()}, flush=True)
        torch.cuda.empty_cache()
    return dict(n=n, pitch=maze.pitch, scale=SCALE[n], ids={k: v.name.replace('registered_', '') for k, v in runs.items()}, levels=levels)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = dict(methods=METHODS, mazes={str(n): build(n) for n in (13, 22)})
    (OUT / 'meta.json').write_text(json.dumps(meta))


if __name__ == '__main__':
    main()
