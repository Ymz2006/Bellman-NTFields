"""How much of the speed does an embedding capture, and does that predict planning? Inference only, CPU, from registered runs.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/speed_fourier/capture.py

Per trained field on the maze_scale mazes: the embedding phi(q) is fixed before training, so without the network it can be
scored directly. All scores on the prepared held-out pairs (x0, label), never on training points:

  capture    R^2 of the linear readout o(q) = 1 - trained speed from phi(q), ridge-regressed on half the points and scored on 40 %
             of them: the share of the speed variance the embedding carries linearly
  layer      the same readout after one random layer sin(phi W + b) with the network's own initial scale (4,096 units, W ~ N(0, 1 / 16^2)
             truncated at 2 sigma, b ~ U(-pi, pi)): what the network's first layer has available before any training
  cells      share of held-out points whose cell index a one-hot ridge readout gets right, after the same random layer: does the
             embedding say which cell a point is in? T(q, g) sees only the two endpoints, so the walls between them have to be
             learned per cell; the embedding has to tell the cells apart
  resolve    median over free points (trained speed 1) of the smaller singular value of the 256 x 2 Jacobian d phi / d(x, y), per
             cell of movement: how far the embedding moves in the direction it is least sensitive to, away from walls
  rank       effective rank exp(entropy of the normalized squared singular values) of phi over the points
  coord      mean R^2 of the linear readout of x and of y from phi: are the plain coordinates in the embedding? A field T(q, g) of two
             endpoints has to grow with how far apart they are, which rows that follow the walls need not tell it
  coordlayer the same after the random layer

report/capture.json holds every field; report/capture.md the per-maze rank correlations of capture and resolve with success.
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
sys.path.insert(0, str(ROOT / 'jeffrey/grid_sync'))
import embed as EM  # noqa: E402
import speedfourier as SF  # noqa: E402
import maze as MZ  # noqa: E402

PREPARED = {7: '20261101T192446Z-9b1569db', 10: '20261102T032150Z-b0baa8e9', 13: '20261101T192504Z-13cf1d04', 22: '20261101T193403Z-cea05796'}
POINTS = 40000
JPOINTS = 2000


def runs():
    want = {str(ROOT / f'results/registered_{v}'): n for n, v in PREPARED.items()}
    out = []
    for r in sorted(ROOT.glob('results/registered_*/results.json')):
        try:
            d = json.loads(r.read_text())
        except Exception:
            continue
        if d.get('stage') not in ('ntrl', 'fourier', 'gridsync', 'speedfourier') or not d.get('complete') or d.get('smoke'):
            continue
        if str(Path(d.get('prepared', '')).resolve()) not in want or d.get('epochs') != 5000 or d.get('variant', 'metric') != 'metric':
            continue
        d['_dir'], d['_n'] = r.parent, want[str(Path(d['prepared']).resolve())]
        out.append(d)
    return out


def key(d):
    if d['stage'] == 'ntrl':
        return 'ntrl-demo'
    if d['stage'] == 'fourier':
        return f'B x {d["b_scale"]:g}'
    if d['stage'] == 'gridsync':
        return d['mode']
    e = d['embedding']
    extra = (f' r{e["reach"]:g}' if e.get('reach') else '') + (f' {e["norm"]}' if e.get('norm') else '') + (f' +head {e["head"]}' + (f' w{e["head_fixed"]:g}' if e.get('head_fixed') is not None else '') if e.get('head', 'none') != 'none' else '')
    return f'{e["form"]} {e["scope"]} M{e["terms"]} {e["walls"]} {e["rows_from"]}{" " + e["target"] if e.get("target", "obstacle") != "obstacle" else ""}{extra}' if e['form'] != 'plain' else f'plain {e["rows_from"]} {e["walls"]}'


def embedding_of(d, maze, cache={}):
    """A function q -> phi(q) for the run."""
    if d['stage'] != 'speedfourier':
        B = torch.load(d['_dir'] / ('training/checkpoint.pt'), map_location='cpu', weights_only=False)['B_state_dict'].float()
        return EM.Embed(B.numpy(), None, 'cpu')
    e = d['embedding']
    k = (d['_n'], d['seed'], e['form'], e['scope'], e['terms'], e['walls'], e['rows_from'], e.get('target', 'obstacle'), e.get('reach'), e.get('norm'))
    if k not in cache:
        cache.clear()
        cache[k] = SF.build(maze, d['seed'], e['form'], e['scope'], e['terms'], e['walls'], None, None, e['rows_from'], target=e.get('target', 'obstacle'), reach=e.get('reach'), norm=e.get('norm'))[0]
    return cache[k]


def ridge(F, Y):
    """Ridge readouts of the columns of Y from F: fit on the first 50 %, ridge strength chosen on the next 10 % (from 1e-4 to 1e2 x the
    mean diagonal), returns the predictions on the last 40 %."""
    a, b = int(0.5 * len(Y)), int(0.6 * len(Y))
    mean = F[:a].mean(0, keepdim=True)
    ybar = Y[:a].mean(0, keepdim=True)
    A = F - mean
    G = A[:a].T @ A[:a]
    lam, V = torch.linalg.eigh(G)
    r = V.T @ (A[:a].T @ (Y[:a] - ybar))
    best = None
    scale = float(G.diagonal().mean())
    for k in (1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100):
        w = V @ (r / (lam.clamp(min=0) + k * scale)[:, None])
        err = float(((Y[a:b] - ybar - A[a:b] @ w) ** 2).mean())
        if best is None or err < best[0]:
            best = (err, w)
    return ybar + A[b:] @ best[1]


def readout(F, o):
    """Held-out R^2 of the ridge readout of o from the columns of F."""
    y = o.double()[:, None]
    b = int(0.6 * len(y))
    return 1 - float(((y[b:] - ridge(F, y)) ** 2).mean() / y[b:].var())


def cell_accuracy(F, cell, count):
    Y = torch.nn.functional.one_hot(cell, count).double()
    b = int(0.6 * len(Y))
    return float((ridge(F, Y).argmax(1) == cell[b:]).double().mean())


def random_layer(F, width=4096, seed=0):
    g = torch.Generator().manual_seed(seed)
    W = (torch.randn(F.shape[1], width, generator=g, dtype=F.dtype) / 16).clamp(-2 / 16, 2 / 16)
    b = (torch.rand(width, generator=g, dtype=F.dtype) * 2 - 1) * np.pi
    return torch.sin(F @ W + b)


def score(phi, q, o, free, pitch, cell, count):
    F = torch.cat([phi(q[i:i + 4096]) for i in range(0, len(q), 4096)]).double()
    capture = readout(F, o)
    coord = (readout(F, q[:, 0]) + readout(F, q[:, 1])) / 2
    H = random_layer(F)
    layer = readout(H, o)
    coordlayer = (readout(H, q[:, 0]) + readout(H, q[:, 1])) / 2
    cells = cell_accuracy(H, cell, count)
    del H
    s = torch.linalg.svdvals(F - F.mean(0))
    p = s ** 2 / (s ** 2).sum()
    rank = float(torch.exp(-(p * torch.log(p + 1e-30)).sum()))
    x = q[free][:JPOINTS].clone()
    h = 0.02 * pitch
    cols = []
    for ax in (0, 1):
        e = torch.zeros(3)
        e[ax] = h
        cols.append((phi(x + e) - phi(x - e)) / (2 * h) * pitch)        # per cell of movement
    J = torch.stack(cols, -1)                                           # (P, 256, 2)
    resolve = float(torch.linalg.svdvals(J)[:, -1].median())
    return capture, layer, cells, resolve, rank, coord, coordlayer


def main():
    torch.set_num_threads(4)
    store = HERE / 'report/capture.json'
    table = [t for t in (json.loads(store.read_text()) if store.exists() else []) if 'coord' in t]      # finished runs are kept, new runs added
    have = {t['run'] for t in table}
    for n, rid in PREPARED.items():
        prepared = ROOT / f'results/registered_{rid}'
        maze = MZ.Maze(n, 'cpu')
        pts = np.load(prepared / 'heldout/sampled_points.npy', mmap_mode='r')
        clr = np.load(prepared / 'heldout/clearance_cells.npy', mmap_mode='r')
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(len(pts), min(POINTS, len(pts)), replace=False))
        q = torch.tensor(np.array(pts[idx, :3]), dtype=torch.float32)
        speed = maze.train_speed(torch.tensor(np.array(clr[idx, 0]), dtype=torch.float32))
        perm = torch.tensor(rng.permutation(len(q)))
        q, speed = q[perm], speed[perm]
        o = 1 - speed
        free = speed > 0.95
        cx, cy = maze.to_cells(q)[:2]
        cell = cx.floor().long().clamp(0, n - 1) * n + cy.floor().long().clamp(0, n - 1)
        for d in [r for r in runs() if r['_n'] == n and r['_dir'].name.replace('registered_', '') not in have]:
            phi = embedding_of(d, maze)
            c, l, a, r, k, xy, xl = score(phi, q, o, free, maze.pitch, cell, n * n)
            ok, tot = d['success_total']
            table.append(dict(n=n, field=key(d), seed=d['seed'], stage=d['stage'], run=d['_dir'].name.replace('registered_', ''), capture=c, layer=l, cells=a, resolve=r, rank=k, coord=xy, coordlayer=xl, success=ok / tot))
            print(n, table[-1]['field'], d['seed'], f'capture {c:.3f} layer {l:.3f} cells {a:.3f} resolve {r:.3f} rank {k:.1f} coord {xy:.3f} {xl:.3f} success {ok / tot:.3f}', flush=True)
            store.write_text(json.dumps(table, indent=1) + '\n')
    lines = ['| maze | fields | Spearman(capture, success) | Spearman(layer, success) | Spearman(cells, success) | Spearman(resolve, success) |', '|---|---|---|---|---|---|']
    for n in PREPARED:
        rows = {}
        for t in table:
            if t['n'] == n:
                rows.setdefault(t['field'], []).append(t)
        m = [tuple(np.mean([x[k] for x in v]) for k in ('capture', 'layer', 'cells', 'resolve', 'success')) for v in rows.values()]
        if len(m) > 2:
            a = np.array(m)
            lines.append(f'| {n}×{n} | {len(m)} | {spearmanr(a[:, 0], a[:, 4])[0]:.2f} | {spearmanr(a[:, 1], a[:, 4])[0]:.2f} | {spearmanr(a[:, 2], a[:, 4])[0]:.2f} | {spearmanr(a[:, 3], a[:, 4])[0]:.2f} |')
    (HERE / 'report/capture.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
