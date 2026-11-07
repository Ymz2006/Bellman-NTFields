"""Data for the wall-speed viewer (report/speedlines/): the speed a field implies, 1 / |dT/dx0|, along lines across the
gap mazes and near the walls, next to the target speed. Inference only, no training.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/cross_warp/speedlines.py [c22 m10]

Fields (training seed 87358, one process per maze because gap.install patches the maze module once):
  idw   ntrl-demo, trained on the maze, pairs drawn on the constructed warped maze (the field under study);
  id    ntrl-demo, trained on the maze, pairs drawn on the maze (control: the pairs);
  pmzw  same pairs, same labels, same network, trained on the warped maze (control: the coordinates). Its loss is taken in
        z = phi(x), so its speed is 1 / |dH/dz| at z = phi(x), the quantity that loss pulls towards the same label S(x).
Speed uses the full gradient over (x, y, phi), as evaluate.speed_fit does; speed_fit's stored numbers are the check.

Lines: heading 0, x-lines (fixed y) and y-lines (fixed x). Along a line 1/16-cell steps over the blocks and their edge cells
and 1/2-cell steps over the band's interior; lines every 1/8 cell over the blocks, every cell over the interior. Layers per
field: the four named goals of stretchview, then the mean, p10 and p90 over the prepare run's 48 goals. Speeds are stored as
uint8 codes of log10(speed) on [LOG_LO, LOG_HI].

Near walls: the held-out pairs of idw's prepare run (x0 inside the 0.175-cell margin, never trained on), the same poses for
every field, with the pair's own x1 ('pair') or one of the 48 goals at random ('far'); free poses (clearance >= 0.35) with far goals
as the open-space reference. report/speedlines/<base>_k<k>.json, and index.json with the distributions of every maze.
"""
import base64
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
sys.path.insert(0, str(HERE))
OUT = HERE / 'report/speedlines'
SEED = '87358'
GAPS = (1, 2, 4, 8, 12)
GAPBASE = dict(c22='cross22', m10='maze10')
MAIN22 = {1: '20261103T233014Z-ae84b379', 2: '20261103T232925Z-119e9a37', 4: '20261103T233057Z-2c83639d', 8: '20261103T233625Z-021c8182', 12: '20261103T233747Z-3c48b7dc'}
LOG_LO, LOG_HI = -3.0, 0.8          # stored line speeds
HIST_LO = -2.2                       # histogram range (near-wall poses do not reach lower)
FINE, FINE_LINE, COARSE, COARSE_LINE = 1 / 16, 1 / 8, 0.5, 1.0
HIST_BINS = 60
CLEAR_EDGES = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175]
FIELDS = ('idw', 'id', 'pmzw')


def axis(n, block, fine, coarse):
    """Cell coordinates 0..n: fine over [0, block + 1] and [n - block - 1, n], coarse over the band's interior."""
    a, b = block + 1.0, n - block - 1.0
    return np.concatenate([np.arange(0, a, fine), np.arange(a, b, coarse), np.arange(b, n + 1e-9, fine)])


def code(s):
    return torch.round(((torch.log10(s.clamp(min=1e-9)) - LOG_LO) / (LOG_HI - LOG_LO)).clamp(0, 1) * 254).to(torch.uint8)


def b64(t):
    return base64.b64encode(t.contiguous().cpu().numpy().tobytes()).decode()


def run_id(R, base, k, lab):
    if lab == 'id' and base == 'c22':
        return R['id22'][str(k)][SEED]
    return R['fields'][f'{base}/{k}'][lab][SEED]['_id']


def compute(base, k):
    import gap
    gap.install(k, GAPBASE[base])
    import maze as MZ
    import plan as P
    import warp as W
    R = json.loads((HERE / 'report/zdraw10.json').read_text())
    dev = 'cuda:0'
    main = ROOT / 'results' / (f'registered_{MAIN22[k]}' if base == 'c22' else R['prepares']['m10_main'][str(k)])
    info = json.loads((main / 'maze.json').read_text())
    n = info['n']
    maze = MZ.Maze(n, dev)
    assert np.array_equal(maze.V.astype(int), np.array(info['V'])) and np.array_equal(maze.H.astype(int), np.array(info['H'])), 'maze differs'
    block = gap.BLOCK
    nets, runs, stored = {}, {}, {}
    for lab in FIELDS:
        rid = run_id(R, base, k, lab)
        r = ROOT / 'results' / f'registered_{rid}'
        d = json.loads((r / 'results.json').read_text())
        nets[lab] = W.load_field(r / 'training/checkpoint.pt', dev) if d.get('stage') == 'cross-warp' else P.load_field(r / 'training/checkpoint.pt')
        tot = d.get('success_total')
        runs[lab] = dict(id=rid, prepared=Path(d['prepared']).name if d.get('prepared') else None, success=tot)
        stored[lab] = d.get('speed_fit')
    zprep = ROOT / 'results' / runs['idw']['prepared']
    assert runs['pmzw']['prepared'] == runs['idw']['prepared'], 'pmzw and idw must share their pairs'
    zs = nets['pmzw']
    assert zs.warp.cfg.get('z_space'), 'pmzw is not a warped-space field'
    top = float(torch.linalg.vector_norm(nets['idw'].B, dim=0).max())

    def speed(lab, q, g):
        """1 / |grad| at poses q (N, 3) towards goals g (N, 3)."""
        net = nets[lab]
        if lab == 'pmzw':
            with torch.no_grad():
                a, b = net.warp(q), net.warp(g)
            net.bypass = True
            _, gr = P.field_grad(net, torch.cat([a, b], 1))
            net.bypass = False
        else:
            _, gr = P.field_grad(net, torch.cat([q, g], 1))
        return 1 / torch.linalg.vector_norm(gr, dim=1).clamp(min=1e-6)

    # the bypass path is the saved field
    gen = torch.Generator(device=dev).manual_seed(11)
    q0, q1 = maze.free(256, gen), maze.free(256, gen)
    t_full = P.field_cost(zs)(torch.cat([q0, q1], 1))
    zs.bypass = True
    t_z = P.field_cost(zs)(torch.cat([zs.warp(q0), zs.warp(q1)], 1))
    zs.bypass = False
    assert float((t_full - t_z).abs().max()) < 1e-4, 'bypass differs from the saved field'

    # goals
    arm = 4.0 if base == 'c22' else 2.5
    gcells = [[n / 2, n / 2], [n / 2, arm], [n - 2.5, n - 2.5], [2.5, 2.5]]
    gnames = ['band centre', 'lower band arm', 'top-right block', 'bottom-left block']
    c = torch.tensor(gcells, device=dev)
    named = maze.from_cells(c[:, 0], c[:, 1], torch.zeros(len(c), device=dev))
    assert bool((maze.clearance(named) > 0.1).all()), 'goal not free'
    pool = torch.tensor(np.load(main / 'goals.npy'), device=dev, dtype=named.dtype)
    pool_cells = torch.stack(maze.to_cells(pool)[:2], 1)

    # lines
    us = torch.tensor(axis(n, block, FINE, COARSE), device=dev, dtype=named.dtype)
    ps = torch.tensor(axis(n, block, FINE_LINE, COARSE_LINE), device=dev, dtype=named.dtype)
    U, Pn = len(us), len(ps)
    grids, strip = {}, dict(cells=0, collision=0, mean={f: [] for f in FIELDS})
    for ax in 'xy':
        A, B = torch.meshgrid(ps, us, indexing='ij')          # A: fixed coordinate of the line, B: along it
        X, Y = (B, A) if ax == 'x' else (A, B)
        q = maze.from_cells(X.reshape(-1), Y.reshape(-1), torch.zeros(Pn * U, device=dev))
        clr = maze.clearance(q)
        tl = maze.train_speed(clr)
        low = tl < 0.05
        strip['cells'] += int(low.sum())
        strip['collision'] += int((low & (clr < 0)).sum())
        lay = {'target': b64(code(tl))}
        clip = {}
        for lab in FIELDS:
            nm = [speed(lab, q, g[None].expand(len(q), 3)) for g in named]
            al = torch.stack([speed(lab, q, g[None].expand(len(q), 3)) for g in pool])
            mean = al.mean(0)
            strip['mean'][lab].append(mean[low])
            p10, p90 = torch.quantile(al, torch.tensor([0.1, 0.9], device=dev, dtype=al.dtype), dim=0)
            lay[lab] = [b64(code(v)) for v in nm + [mean, p10, p90]]
            clip[lab] = [float((mean > 10 ** LOG_HI).float().mean()), float((mean < 10 ** LOG_LO).float().mean())]
        grids[ax] = lay
        print(f'{base} x{k} {ax}-lines done, mean above the code range: {clip}', flush=True)

    strip_out = dict(collision_share=strip['collision'] / max(strip['cells'], 1), median={f: float(torch.cat(v).median()) for f, v in strip['mean'].items()})

    # near walls
    hx = torch.tensor(np.load(zprep / 'heldout/sampled_points.npy'), device=dev, dtype=named.dtype)
    hd = torch.tensor(np.load(zprep / 'heldout/clearance_cells.npy'), device=dev, dtype=named.dtype)[:, 0]
    tgt = maze.train_speed(hd)
    x0, x1 = hx[:, :3], hx[:, 3:]
    gen = torch.Generator(device=dev).manual_seed(12)
    far = pool[torch.randint(len(pool), (len(x0),), generator=gen, device=dev)]
    free = maze.free(20000, gen, min_clear=0.35)
    free_goal = pool[torch.randint(len(pool), (len(free),), generator=gen, device=dev)]
    edges = np.linspace(HIST_LO, LOG_HI, HIST_BINS + 1)

    def hist(s):
        v = torch.log10(s.clamp(min=1e-9)).clamp(HIST_LO, LOG_HI).cpu().numpy()
        return np.histogram(v, bins=edges)[0].tolist()

    def by_clear(s):
        rows = []
        for a, b in zip(CLEAR_EDGES[:-1], CLEAR_EDGES[1:]):
            m = (hd >= a) & (hd < b)
            if int(m.sum()) < 20:
                rows.append(None)
                continue
            v = torch.quantile(s[m], torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=dev, dtype=s.dtype))
            rows.append([float(x) for x in v] + [int(m.sum())])
        return rows

    centres = [(a + b) / 2 for a, b in zip(CLEAR_EDGES[:-1], CLEAR_EDGES[1:])]
    fine = torch.linspace(0, 0.2, 81, device=dev)
    floor = tgt < 0.05
    dist = dict(edges=edges.tolist(), clear_edges=CLEAR_EDGES, clear_centres=centres, count=len(x0), target_hist=hist(tgt),
                target_curve=[fine.tolist(), maze.train_speed(fine).tolist()], target_median=float(tgt.median()), target_floor_share=float((tgt < 0.05).float().mean()),
                target_mean=float(tgt.mean()), target_floor_median=float(tgt[floor].median()), fields={})
    for lab in FIELDS:
        e = {}
        for variant, g in (('pair', x1), ('far', far)):
            s = speed(lab, x0, g)
            lp, lt = torch.log(s), torch.log(tgt)
            e[variant] = dict(hist=hist(s), clear=by_clear(s), median=float(s.median()), mean=float(s.mean()), floor_median=float(s[floor].median()),
                              corr=float(torch.corrcoef(torch.stack([lp, lt]))[0, 1]), mae=float((lp - lt).abs().mean()),
                              below_02=float((s < 0.2).float().mean()), ratio_median=float((s / tgt).median()))
        s = speed(lab, free, free_goal)
        e['free'] = dict(hist=hist(s), median=float(s.median()))
        e['contrast'] = e['free']['median'] / e['far']['floor_median']
        e['stored_speed_fit'] = stored[lab]
        dist['fields'][lab] = e
        print(f'{base} x{k} {lab}: pair corr {e["pair"]["corr"]:.3f} floor med {e["pair"]["floor_median"]:.3f} | far floor {e["far"]["floor_median"]:.3f} free {e["free"]["median"]:.3f} | stored {stored[lab]}', flush=True)

    meta = dict(strip=strip_out, base=base, k=k, n=n, E=float(maze.E), block=block, band=6 * k, runs=runs, boxes=info['boxes_cells'], margin_cells=0.175, top_freq_cycles_per_unit=top,
                goals=[dict(name=a, cell=b) for a, b in zip(gnames, gcells)], pool_cells=pool_cells.tolist(), wall_band_norm=0.175 / float(maze.E))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{base}_k{k}.json').write_text(json.dumps(dict(meta, us=us.tolist(), ps=ps.tolist(), log_lo=LOG_LO, log_hi=LOG_HI, lines=grids)))
    (OUT / f'{base}_k{k}.dist.json').write_text(json.dumps(dict(meta, dist=dist)))
    print(f'{base} x{k}: n={n}, U={U}, P={Pn}, {(OUT / f"{base}_k{k}.json").stat().st_size / 1e6:.2f} MB', flush=True)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if '--child' in sys.argv:
        compute(args[0], int(args[1]))
        return
    for base in args or ['c22', 'm10']:
        for k in GAPS:
            subprocess.run([sys.executable, __file__, base, str(k), '--child'], check=True)
    idx = [json.loads(p.read_text()) for p in sorted(OUT.glob('*.dist.json'))]
    (OUT / 'index.json').write_text(json.dumps(idx))


if __name__ == '__main__':
    main()
