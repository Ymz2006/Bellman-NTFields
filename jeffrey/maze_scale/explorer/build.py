"""Data for the findings page's travel-time explorer: pick a pose in a maze, see T from it along x, y and heading.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/explorer/build.py
    ... build.py --check 13          # Dijkstra vs the prepare run's Jacobi truth fields, then exit

Read-only analysis of registered runs (report.collect(): the mazes and seed-87358 fields the page reports); nothing is
trained or registered. Sources: the grid node nearest every cell centre at headings 0, 45, 90, -45 degrees. Truth:
truth.py's grid (1/15 cell, 13 headings, 98-neighbourhood, same edge costs and source seeding), solved by Dijkstra on
the CPU (dijkstra.c, the fixed point Grid.solve's Jacobi sweeps converge to) instead of the GPU. Reported at every grid
node of the source's row (along x), column (along y) and the 13 headings at the source. Learned: T(source, q) of each
field at the same x / y nodes and at 37 headings (every 5 degrees). All times in cells at full speed (T / pitch).

Writes report/explorer/: meta.json (mazes, walls, node coordinates, fields) and per maze and heading
m<n>_h<deg>.b64.txt (float16, base64), per maze m<n>_speed.b64.txt (true unit speed along every row and column at
each source heading, uint8), for page_text/explorer.html.
"""
import argparse
import base64
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MS = HERE.parent
sys.path.insert(0, str(MS))
import maze as MZ  # noqa: E402
import plan as P  # noqa: E402
import report as R  # noqa: E402
import truth as TR  # noqa: E402
from fields import load_chain  # noqa: E402

OUT = MS / 'report' / 'explorer'
SCRATCH = Path(os.environ.get('EXPLORER_SCRATCH', tempfile.gettempdir()))
BIN = Path(os.environ.get('DIJKSTRA_BIN', SCRATCH / 'dijkstra'))
SERIES = ['ntrl-demo', 'segment DP', 'segment DP + wall term (weight 0.3)']
HEADINGS = [0, 45, 90, -45]          # degrees; grid headings every 15 degrees, so all are nodes
SEED = 87358
N_THETA_LEARNED = 37
THREADS = 12


class Nodes:
    """truth.Grid's node layout and speed without its GPU edge tensors."""

    def __init__(self, maze):
        self.maze = maze
        nx = int(round(maze.E / TR.H_CELLS)) + 1
        self.nx, self.nt = nx, TR.N_THETA
        self.xs = torch.linspace(-0.5, 0.5, nx, dtype=torch.float64).float()
        self.ts = torch.linspace(-maze.phi_max, maze.phi_max, self.nt, dtype=torch.float64).float()
        self.hx, self.ht = 1.0 / (nx - 1), 2 * maze.phi_max / (self.nt - 1)
        q = torch.stack(torch.meshgrid(self.xs, self.xs, self.ts, indexing='ij'), -1).reshape(-1, 3)
        self.unit = MZ.Maze.unit_speed(maze.clearance(q)).reshape(nx, nx, self.nt)
        self.speed = MZ.Maze.train_speed(maze.clearance(q)).reshape(nx, nx, self.nt)

    def flat(self, i, j, k):
        return (i * self.nx + j) * self.nt + k

    def seeds(self, src):
        """Grid.seed_block: the 4 x 4 x 4 nodes around each source and the straight-segment cost to them."""
        nx, nt = self.nx, self.nt
        idx = torch.stack([(src[:, 0] + 0.5) / self.hx, (src[:, 1] + 0.5) / self.hx, (src[:, 2] + self.maze.phi_max) / self.ht], 1)
        base = torch.floor(idx).long()
        off = torch.stack(torch.meshgrid(*[torch.arange(-1, 3)] * 3, indexing='ij'), -1).reshape(-1, 3)
        k = base[:, None] + off[None]
        ok = (k >= 0).all(-1) & (k[..., 0] < nx) & (k[..., 1] < nx) & (k[..., 2] < nt)
        kc = torch.stack([k[..., 0].clamp(0, nx - 1), k[..., 1].clamp(0, nx - 1), k[..., 2].clamp(0, nt - 1)], -1)
        s0 = 1 / MZ.Maze.train_speed(self.maze.clearance(src))
        sn = self.speed[kc[..., 0], kc[..., 1], kc[..., 2]]
        node = torch.stack([self.xs[kc[..., 0]], self.xs[kc[..., 1]], self.ts[kc[..., 2]]], -1)
        cost = torch.linalg.vector_norm(node - src[:, None], dim=-1) * 0.5 * (s0[:, None] + 1 / sn)
        cost = torch.where(ok, cost, torch.full_like(cost, float('inf')))
        return self.flat(kc[..., 0], kc[..., 1], kc[..., 2]).int(), cost.float()

    def solve(self, src, lines):
        """Travel time from each source (S, 3) to the flat node indices lines (S, L), by dijkstra.c."""
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            tmp = Path(tmp)
            (1 / self.speed).float().numpy().astype('<f4').tofile(tmp / 'inv.f32')
            fi, cost = self.seeds(src)
            rec = np.zeros((len(src), 64), dtype=[('v', '<i4'), ('c', '<f4')])
            rec['v'], rec['c'] = fi.numpy(), cost.numpy()
            rec.tofile(tmp / 'seeds.bin')
            np.ascontiguousarray(lines, dtype='<i4').tofile(tmp / 'lines.i32')
            subprocess.run([str(BIN), str(tmp / 'inv.f32'), str(self.nx), str(self.nt), repr(self.hx), repr(self.ht), str(tmp / 'seeds.bin'),
                            str(len(src)), str(tmp / 'lines.i32'), str(lines.shape[1]), str(tmp / 'out.f32'), str(THREADS)], check=True)
            return np.fromfile(tmp / 'out.f32', dtype='<f4').reshape(lines.shape)


def check(n):
    """Dijkstra from the prepare run's goals vs its stored Jacobi fields at 20,000 random nodes per goal."""
    prep, _, _ = R.collect()
    d = prep[n]['_dir']
    maze = MZ.Maze(n, 'cpu')
    g = Nodes(maze)
    goals = torch.tensor(np.load(d / 'goals.npy')[:6], dtype=torch.float32)
    tf = np.load(d / 'truth_fields.npy', mmap_mode='r')
    rng = np.random.default_rng(0)
    N = g.nx * g.nx * g.nt
    lines = rng.integers(0, N, (len(goals), 20000))
    t = g.solve(goals, lines)
    for i in range(len(goals)):
        ref = np.asarray(tf[i]).reshape(-1)[lines[i]].astype(np.float64)
        ok = np.isfinite(ref) & np.isfinite(t[i])
        rel = np.abs(t[i][ok] - ref[ok]) / np.maximum(ref[ok], 1e-9)
        print(f'goal {i}: truth dtype {tf.dtype}, finite {ok.sum()}/{len(ok)}, rel err median {np.median(rel):.2e} p99 {np.percentile(rel, 99):.2e} max {rel.max():.2e}')


def fields_for(n, dev):
    """[(name, cost(pairs))] for the reported seed-87358 fields of maze n."""
    _, ntrl, seg = R.collect()
    out = []
    if SEED in ntrl.get(n, {}):
        out.append(('ntrl-demo', ntrl[n][SEED]['_id'], P.field_cost(P.load_field(ntrl[n][SEED]['_dir'] / 'training/checkpoint.pt', 'metric', dev))))
    for name in SERIES[1:]:
        d = seg.get(name, {}).get(n, {}).get(SEED)
        if d:
            chain, _ = load_chain(d['_dir'] / 'chain/field.pt', dev)
            out.append((name, d['_id'], chain.cost))
    return out


def b64(path, arr):
    path.write_text(base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode())


def build(n, dev):
    maze = MZ.Maze(n, 'cpu')
    g = Nodes(maze)
    nx, nt = g.nx, g.nt
    xc = (g.xs * maze.E + n / 2).numpy()                                         # node x (= y) in cells
    ci = np.array([int(np.argmin(np.abs(xc - (i + 0.5)))) for i in range(n)])   # node nearest each cell centre
    hk = [int(np.argmin(np.abs(np.degrees(g.ts.numpy() * maze.E / MZ.RHO) - h))) for h in HEADINGS]
    th_learned = torch.linspace(-maze.phi_max, maze.phi_max, N_THETA_LEARNED)
    fields = fields_for(n, dev)
    meta = dict(n=n, nx=nx, nt=nt, xs=[round(float(v), 4) for v in xc], cells=ci.tolist(), headings=HEADINGS,
                head_nodes=hk, theta_truth=[round(float(v), 2) for v in np.degrees(g.ts.numpy() * maze.E / MZ.RHO)],
                theta_learned=[round(float(v), 2) for v in np.degrees(th_learned.numpy() * maze.E / MZ.RHO)],
                V=maze.V.astype(int).tolist(), H=maze.H.astype(int).tolist(), pitch=maze.pitch,
                fields=[dict(name=f[0], run=f[1]) for f in fields], prepared=R.collect()[0][n]['_id'],
                layout=dict(truth=[nx, nx, nt], learned=[nx, nx, N_THETA_LEARNED]))
    speed = np.zeros((len(HEADINGS), 2, n, nx), np.uint8)                      # [heading, row/col, cell, node]
    stats = {}
    for h, k in enumerate(hk):
        speed[h, 0] = np.round(255 * g.unit[:, ci, k].T.numpy())                # row j: nodes (all x, ci[j], k)
        speed[h, 1] = np.round(255 * g.unit[ci, :, k].numpy())                  # column i: nodes (ci[i], all y, k)
        I, J = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        I, J = I.reshape(-1), J.reshape(-1)                                       # source order: cell i (x) major, j (y) minor
        kx, ky = ci[I], ci[J]
        src = torch.stack([g.xs[kx], g.xs[ky], g.ts[k].expand(len(I))], 1)
        ar, art = np.arange(nx), np.arange(nt)
        lines = np.concatenate([g.flat(ar[None], ky[:, None], k), g.flat(kx[:, None], ar[None], k), g.flat(kx[:, None], ky[:, None], art[None])], 1)
        truth = g.solve(src, lines) / maze.pitch
        blocks = [truth]
        srcd = src.to(dev)
        for name, rid, cost in fields:
            xs_d = g.xs.to(dev)
            S = len(src)
            qx = torch.stack([xs_d[None].expand(S, nx), srcd[:, 1:2].expand(S, nx), srcd[:, 2:3].expand(S, nx)], -1)
            qy = torch.stack([srcd[:, 0:1].expand(S, nx), xs_d[None].expand(S, nx), srcd[:, 2:3].expand(S, nx)], -1)
            qt = torch.stack([srcd[:, 0:1].expand(S, N_THETA_LEARNED), srcd[:, 1:2].expand(S, N_THETA_LEARNED), th_learned.to(dev)[None].expand(S, -1)], -1)
            q = torch.cat([qx, qy, qt], 1)
            L = q.shape[1]
            pairs = torch.cat([srcd[:, None].expand(-1, L, -1), q], -1).reshape(-1, 6)
            with torch.no_grad():
                t = cost(pairs).reshape(S, L).cpu().numpy() / maze.pitch
            blocks.append(t)
            # agreement on the truth's x / y nodes, for the log
            tt = truth[:, :2 * nx]
            ok = np.isfinite(tt) & (tt > 0.5)
            stats.setdefault(name, []).append(float(np.median(t[:, :2 * nx][ok] / tt[ok])))
        arr = np.concatenate(blocks, 1).astype(np.float16)
        b64(OUT / f'm{n}_h{HEADINGS[h]}.b64.txt', arr)
        print(f'  n={n} heading {HEADINGS[h]}: {len(src)} sources, truth max {np.nanmax(truth):.1f} cells, {arr.nbytes / 1e6:.2f} MB', flush=True)
    b64(OUT / f'm{n}_speed.b64.txt', speed)
    meta['median_ratio'] = {k: float(np.median(v)) for k, v in stats.items()}
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', type=int)
    ap.add_argument('--mazes', type=int, nargs='*')
    a = ap.parse_args()
    torch.set_num_threads(4)
    if a.check:
        check(a.check)
        return
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    OUT.mkdir(parents=True, exist_ok=True)
    prep, _, _ = R.collect()
    mazes = a.mazes or sorted(prep)
    path = OUT / 'meta.json'
    meta = json.loads(path.read_text()) if path.exists() else {}
    for n in mazes:
        meta[str(n)] = build(n, dev)
        path.write_text(json.dumps(meta))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
