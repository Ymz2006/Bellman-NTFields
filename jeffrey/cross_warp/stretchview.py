"""Data for the stretched-field viewer (report/stretchview/): the field trained on gap x1, stretched onto gap x k by
stretch.py's map, next to the true travel time of gap x k. Inference and ground truth only.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/cross_warp/stretchview.py [base ...]

Per base and gap k (one process each: gap.install patches the maze module once; k = 1 first, its truth is reused):
four goals at the same place relative to the quadrants at every gap (heading 0). Fields on every 8th node of maze_scale's
truth grid (1/15 cell, 13 headings, so 0.53 cells at every gap), the minimum over the 13 headings at each (x, y):
  truth        T* of gap x k (truthfast.FastGrid), cells;
  truth1       T* of gap x1 at psi(x), the nearest truth node (the stretch without any fitting error), cells;
  ntrl1        ntrl-demo trained on gap x1, at psi(x), psi(goal), cells of gap x1;
  warp1        the x1 field trained on the warped maze, pairs drawn on the warped maze (label pmzw), at psi(x), quadrant cells;
  ntrlk, warpk the same two trainings on gap x k itself: the controls.
Against truth, by region (quadrants, band; free nodes farther than one cell from the goal): median |T - T*| / T*, Spearman
rank correlation, and the median angle between the two fields' descent directions (central differences on this grid,
nodes whose neighbours are all free).
zoom: the bottom-left and top-right quadrants with their edge cell on every 2nd node (0.13 cells), the same fields.
report/stretchview/<base>_k<k>.json.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
sys.path.insert(0, str(HERE))
OUT = HERE / 'report/stretchview'
BUILD = Path(tempfile.gettempdir()) / 'cross_warp_build'      # compiled solver and gap x1's truth, rebuilt when absent
SEED = 87358
STRIDE = 8
ZSTRIDE = 2
MAIN22 = {1: '20261103T233014Z-ae84b379', 2: '20261103T232925Z-119e9a37', 4: '20261103T233057Z-2c83639d', 8: '20261103T233625Z-021c8182', 12: '20261103T233747Z-3c48b7dc'}
GAPBASE = dict(c22='cross22', m10='maze10')
LAYERS = ['truth1', 'ntrl1', 'warp1', 'ntrlk', 'warpk']


def goal_cells(base, n):
    arm = 4.0 if base == 'c22' else 2.5
    return [[n / 2, n / 2], [n / 2, arm], [n - 2.5, n - 2.5], [2.5, 2.5]], ['band centre', 'lower band arm', 'top-right quadrant', 'bottom-left quadrant']


def run_of(R, base, k, lab):
    if lab == 'id' and base == 'c22':
        return R['id22'][str(k)][str(SEED)]
    return R['fields'][f'{base}/{k}'][lab][str(SEED)]['_id']


def spearman(a, b):
    ra, rb = torch.argsort(torch.argsort(a)).double(), torch.argsort(torch.argsort(b)).double()
    return float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])


def grad(t):
    """Central differences of (nd, nd) t; NaN wherever a neighbour is missing."""
    gx, gy = torch.full_like(t, float('nan')), torch.full_like(t, float('nan'))
    gx[1:-1], gy[:, 1:-1] = t[2:] - t[:-2], t[:, 2:] - t[:, :-2]
    return gx, gy


def compare(f, truth, keep, region):
    """Stats of field f against truth (both (nd, nd))."""
    fx, fy = grad(f)
    tx, ty = grad(truth)
    dot = fx * tx + fy * ty
    ang = torch.rad2deg(torch.acos((dot / (torch.sqrt(fx ** 2 + fy ** 2) * torch.sqrt(tx ** 2 + ty ** 2)).clamp(min=1e-12)).clamp(-1, 1)))
    out = {}
    for name, sel in (('all', torch.ones_like(keep)), ('quadrants', region < 4), ('band', region == 4)):
        m = keep & sel & torch.isfinite(f) & torch.isfinite(truth)
        a = ang[m & torch.isfinite(ang)]
        out[name] = dict(rel=float(((f[m] - truth[m]).abs() / truth[m].clamp(min=1e-6)).median()), rank=spearman(f[m], truth[m]), angle=float(a.median()) if len(a) else None)
    return out


def compute(base, k):
    import gap
    gap.install(k, GAPBASE[base])
    import maze as MZ
    import plan as P
    import stretch as ST
    import truthfast as TF
    import warp as W
    import fieldspaces as FS
    R = json.loads((HERE / 'report/zdraw10.json').read_text())
    dev = 'cuda:0'
    prep = ROOT / 'results' / (f'registered_{MAIN22[k]}' if base == 'c22' else R['prepares']['m10_main'][str(k)])
    info = json.loads((prep / 'maze.json').read_text())
    n = info['n']
    maze = MZ.Maze(n, dev)
    assert np.array_equal(maze.V.astype(int), np.array(info['V'])) and np.array_equal(maze.H.astype(int), np.array(info['H'])), 'maze differs'
    E = maze.E
    psi = ST.Stretch(maze, k).to(dev)
    cells, names = goal_cells(base, n)
    cells = torch.tensor(cells, device=dev)
    goals = maze.from_cells(cells[:, 0], cells[:, 1], torch.zeros(len(cells), device=dev))
    assert bool((maze.clearance(goals) > 0.1).all()), f'goal not free {maze.clearance(goals)}'
    BUILD.mkdir(parents=True, exist_ok=True)
    fg = TF.FastGrid(maze, BUILD, threads=3)
    tk, _ = fg.solve(goals)
    tk = tk.amin(3)                                 # (G, nx, nx): minimum over headings, normalized time
    t1file = BUILD / f'stretch_truth1_{base}.npz'
    if k == 1:
        np.savez(t1file, t=tk.cpu().numpy() * E, xs=fg.xs.cpu().numpy() * E + n / 2)
    t1 = np.load(t1file)
    T1, xs1 = torch.tensor(t1['t'], device=dev), torch.tensor(t1['xs'], device=dev)
    nt = fg.shape[2]

    def load(k_, lab):
        r = ROOT / 'results' / f'registered_{run_of(R, base, k_, lab)}'
        d = json.loads((r / 'results.json').read_text())
        return (W.load_field(r / 'training/checkpoint.pt', dev) if d.get('stage') == 'cross-warp' else P.load_field(r / 'training/checkpoint.pt')), r.name.replace('registered_', '')

    nets, runs = {}, {}
    nets['ntrl1'], runs['ntrl1'] = load(1, 'id')
    nets['warp1'], runs['warp1'] = load(1, 'pmzw')
    nets['ntrlk'], runs['ntrlk'] = load(k, 'id')
    nets['warpk'], runs['warpk'] = load(k, 'pmzw')
    m1, mk = 6 * psi.E1 / 22.1, 6 * E / 22.1        # the constructed warp's magnification on gap x1 and on gap x k (m = 6 E / 22.1)
    h1 = float(xs1[1] - xs1[0])

    def grid(ix):
        """Poses, free mask and gap x1's truth node at the grid nodes ix (the same along x and y)."""
        nd = len(ix)
        X, Y = torch.meshgrid(fg.xs[ix], fg.xs[ix], indexing='ij')
        poses = torch.stack([X[..., None].expand(nd, nd, nt), Y[..., None].expand(nd, nd, nt), fg.ts[None, None].expand(nd, nd, nt)], -1).reshape(-1, 3)
        wall = maze.clearance(poses).reshape(nd, nd, nt).amax(2) < 0
        z = psi(torch.stack([X.reshape(-1), Y.reshape(-1), torch.zeros(nd * nd, device=dev)], 1))
        c1 = z[:, :2].double() * psi.E1 + psi.n1 / 2
        i1 = ((c1 - float(xs1[0])) / h1).round().long().clamp(0, len(xs1) - 1)
        return dict(ix=ix, nd=nd, X=X, Y=Y, poses=poses, zp=psi(poses), wall=wall, i1=i1)

    def layers(G, g):
        """Every field of goal g on grid G, in the units of the module docstring (walls not masked)."""
        ix, nd = G['ix'], G['nd']
        out = dict(truth=tk[g][ix][:, ix] * E, truth1=T1[g][G['i1'][:, 0], G['i1'][:, 1]].reshape(nd, nd))
        gq = goals[g][None]
        zg = psi(gq)
        for key, scale, stretched in (('ntrl1', psi.E1, True), ('warp1', psi.E1 / m1, True), ('ntrlk', E, False), ('warpk', E / mk, False)):
            a, b = (G['zp'], zg) if stretched else (G['poses'], gq)
            out[key] = P.field_cost(nets[key])(torch.cat([a, b.expand(len(a), 3)], 1)).reshape(nd, nd, nt).amin(2) * scale
        return out

    pack = lambda f, G: {key: FS.q16(torch.where(G['wall'], torch.full_like(v, float('nan')), v).cpu().numpy().ravel()) for key, v in f.items()}
    cellsx = lambda ix: [float(v) for v in (fg.xs[ix] * E + n / 2).cpu()]
    block = gap.BLOCK
    full = grid(torch.arange(0, fg.shape[0], STRIDE, device=dev))
    wall, nd = full['wall'], full['nd']
    xc, yc = full['X'] * E + n / 2, full['Y'] * E + n / 2
    region = torch.tensor(gap.region_of(n, xc.cpu().numpy(), yc.cpu().numpy()), device=dev)
    # zoom: the two quadrants at the corners of the diagonal, with their edge cell, on every 2nd node (0.13 cells)
    xall = fg.xs * E + n / 2
    win = dict(q0=[0.0, block + 1.0], q3=[n - block - 1.0, float(n)])
    zooms = {name: grid(torch.nonzero((xall >= lo - 0.2) & (xall <= hi + 0.2)).squeeze(1)[::ZSTRIDE]) for name, (lo, hi) in win.items()}
    out = dict(base=base, k=k, n=n, E=E, block=block, band=6 * k, stride_cells=float(STRIDE * fg.hx * E), nd=nd, boxes=info['boxes_cells'],
               rigid=[block + 1, block + 6 * k - 1], stretch=psi.f, x=cellsx(full['ix']), runs=runs,
               zoom={name: dict(win=win[name], x=cellsx(G['ix']), nd=G['nd']) for name, G in zooms.items()}, goals=[])
    for g, name in enumerate(names):
        fields = layers(full, g)
        truth = fields['truth']
        dist = torch.sqrt((xc - cells[g, 0]) ** 2 + (yc - cells[g, 1]) ** 2)
        keep = ~wall & (dist > 1.0) & torch.isfinite(truth)
        stats = {key: compare(fields[key], truth, keep, region) for key in LAYERS}
        out['goals'].append(dict(name=name, cell=[float(cells[g, 0]), float(cells[g, 1])], stats=stats, fields=pack(fields, full),
                                 zoom={zn: pack(layers(G, g), G) for zn, G in zooms.items()}))
        print(f'{base} x{k} {name}: ' + '; '.join(f'{key} rel {s["all"]["rel"]:.3f} rank {s["all"]["rank"]:.3f} angle q {s["quadrants"]["angle"]} b {s["band"]["angle"]}' for key, s in stats.items()), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{base}_k{k}.json').write_text(json.dumps(out))
    print(f'{base} x{k}: n={n}, grid {nd}^2, {(OUT / f"{base}_k{k}.json").stat().st_size / 1e6:.2f} MB', flush=True)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if '--child' in sys.argv:
        compute(args[0], int(args[1]))
        return
    for base in args or ['c22', 'm10']:
        for k in (1, 2, 4, 8, 12):
            subprocess.run([sys.executable, __file__, base, str(k), '--child'], check=True)


if __name__ == '__main__':
    main()
