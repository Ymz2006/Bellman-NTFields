"""Is the field still a travel-time field on pairs ntrl-demo's sampler never draws? Gap study, inference only:
report/openfit.json (cached by run id) and printed tables.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/cross_warp/openfit.py [k ...]

ntrl-demo's pair recipe (maze_scale/data.py) keeps x0 within its wall margin (0.175 cells), so every training pair has
one end next to a wall. Its network is symmetric, T(x0, x1) = d(f(x0), f(x1)) in feature space, so a pair (open pose,
near-wall pose) is a trained pair read the other way round, while a pair of two open poses is never trained.

For every field of the gap study (gapreport.collect(), training seed 87358, main and open-pair data) and gap scale k,
at open poses x (clearance >= the margin: true speed 1, so |grad_x T| should be 1) and partner poses y:
  wall partner  y within the margin: the recipe's own pairs (seed 5) with their ends swapped, kept if x is open
  open partner  y beyond the margin too: gapprepare.sample_open (seed 6), kept if both are open
Reported per partner type (all pairs and partner distance >= 24 cells): median | |grad_x T(x, y)| - 1 |, the share of x
where |grad_x T| < 0.5 (a flat field: MPPI's steps stall), and the share where the descent direction -grad_x T points
away from y (cos < 0) for pairs joined by a free straight segment (the true descent points towards y there).
"""
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
sys.path.insert(0, str(HERE))
OUT = HERE / 'report/openfit.json'
N_PAIRS = 60_000
TRAP_CELLS = 0.3          # MPPI's look-ahead: 5 steps of 0.015 x 4.1 cells (maze_scale/plan.py BASE, REF_N = 4)


def segment_free(maze, a, b, steps=24):
    """Every pose on the straight segment a -> b (linear in x, y and heading) has clearance >= the margin."""
    from maze import MARGIN_CELLS
    ok = torch.ones(len(a), dtype=torch.bool, device=a.device)
    for t in torch.linspace(0, 1, steps, device=a.device):
        d, _ = maze.clearance_normal(a + (b - a) * t)
        ok &= d >= MARGIN_CELLS
    return ok


def grad_first(net, pairs, chunk=50_000):
    gs = []
    for i in range(0, len(pairs), chunk):
        with torch.enable_grad():
            tau, _, xp = net.out(pairs[i:i + chunk])
            g, = torch.autograd.grad(tau.sum(), xp)
        gs.append(g[:, :3].detach())
    return torch.cat(gs)


def path_minima(net, maze, prepared, category=4):
    """Spurious local minima of T(., goal) along the truth paths of the prepared queries of one category (default
    open - open): the true T falls monotonically along a truth path, so every minimum is a defect. Per query: the path
    densified to 0.02 cells, T evaluated at every point (heading as on the path), a minimum = T falling then rising
    (changes below 1e-6 T ignored). Returns minima per 100 cells of path and the share of queries with at least one,
    and the traps: minima whose basin (path distance until T falls below the minimum again) is wider than MPPI's
    look-ahead (TRAP_CELLS), and the share of queries with at least one."""
    import plan as P
    q = np.load(prepared / 'queries.npz')
    tp = np.load(prepared / 'truth_paths.npy', allow_pickle=True)
    idx = np.nonzero(q['category'] == category)[0]
    count, length, hit, traps, trapped = 0, 0.0, 0, 0, 0
    for i in idx:
        p = P.densify(torch.tensor(np.asarray(tp[i], np.float32), device=maze.device), 0.02 * maze.pitch)
        g = torch.tensor(q['goal'][i], device=maze.device, dtype=torch.float32).expand(len(p), 3)
        with torch.no_grad():
            t = net.out(torch.cat([p, g], 1))[0][:, 0]
        dt = t[1:] - t[:-1]
        keep = torch.nonzero(dt.abs() > 1e-6 * t[0].abs()).squeeze(1)
        sg = torch.sign(dt[keep])
        mins = keep[1:][(sg[:-1] < 0) & (sg[1:] > 0)].cpu().numpy()
        count += len(mins)
        hit += len(mins) > 0
        seg = (torch.linalg.vector_norm(p[1:, :2] - p[:-1, :2], dim=1) / maze.pitch).cpu().numpy()
        s_along = np.concatenate([[0.0], np.cumsum(seg)])
        length += float(s_along[-1])
        tc = t.cpu().numpy()
        wide = 0
        for j in mins:                    # basin width: path distance until T falls below the minimum again
            below = np.nonzero(tc[j + 1:] < tc[j])[0]
            end = j + 1 + below[0] if len(below) else len(tc) - 1
            if s_along[end] - s_along[j] > TRAP_CELLS:
                wide += 1
        traps += wide
        trapped += wide > 0
    return dict(queries=int(len(idx)), minima=count, per_100_cells=100 * count / max(length, 1e-9), share_with_minimum=hit / max(len(idx), 1),
                traps=traps, share_with_trap=trapped / max(len(idx), 1), trap_cells=TRAP_CELLS)


def sink_map(net, maze, goal, res_cells=0.25, H=12):
    """For one goal: (sink, ok, spurious, extent) on a grid of res_cells over the open band (outside the four quadrant
    blocks). Tmin(x, y) = min over H headings with clearance >= the margin of T((x, y, heading), goal); each grid cell
    points to its lowest 8-neighbour (or itself); following the pointers gives the sink (flat index) every cell drains
    to; ok = cells with a free heading; spurious = ok cells whose sink is farther than 1 cell from the goal."""
    import gap
    from maze import MARGIN_CELLS
    n, E, dev = maze.n, maze.E, maze.device
    c = torch.arange(res_cells / 2, n, res_cells, device=dev)
    X, Y = torch.meshgrid(c, c, indexing='ij')
    G = n - 2 * gap.BLOCK
    inb = lambda z: (z >= gap.BLOCK) & (z < gap.BLOCK + G)
    band = inb(X) | inb(Y)
    phi = torch.linspace(-maze.phi_max, maze.phi_max, H + 1, device=dev)[:-1]
    xy = torch.stack([(X - n / 2) / E, (Y - n / 2) / E], -1)[band]
    poses = torch.cat([xy[:, None, :].expand(-1, H, -1), phi[None, :, None].expand(len(xy), -1, -1)], -1).reshape(-1, 3)
    key = (n, res_cells, H)
    if _FREE.get('key') != key:
        d, _ = maze.clearance_normal(poses)
        _FREE.update(key=key, free=(d >= MARGIN_CELLS).reshape(len(xy), H))
    free = _FREE['free']
    shp = X.shape
    goal = torch.as_tensor(goal, device=dev, dtype=torch.float32)
    with torch.no_grad():
        T = torch.cat([net.out(torch.cat([poses[i:i + 200_000], goal.expand(len(poses[i:i + 200_000]), 3)], 1))[0][:, 0]
                       for i in range(0, len(poses), 200_000)]).reshape(len(xy), H)
    T = torch.where(free, T, torch.full_like(T, float('inf'))).min(1).values
    F = torch.full(shp, float('inf'), device=dev)
    F[band] = T
    ok = torch.isfinite(F)
    Fp = torch.nn.functional.pad(F[None, None], (1, 1, 1, 1), value=float('inf'))[0, 0]
    nb = torch.stack([Fp[1 + dx:1 + dx + shp[0], 1 + dy:1 + dy + shp[1]] for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
    arg = nb.argmin(0)
    flat = torch.arange(shp[0] * shp[1], device=dev).reshape(shp)
    dxs = torch.tensor([dx for dx in (-1, 0, 1) for dy in (-1, 0, 1)], device=dev)
    dys = torch.tensor([dy for dx in (-1, 0, 1) for dy in (-1, 0, 1)], device=dev)
    ptr = ((torch.arange(shp[0], device=dev)[:, None] + dxs[arg]) * shp[1] + torch.arange(shp[1], device=dev)[None, :] + dys[arg]).reshape(-1)
    ptr = torch.where(ok.reshape(-1), ptr, flat.reshape(-1))
    for _ in range(int(math.log2(shp[0] * shp[1])) + 2):
        ptr = ptr[ptr]
    sink = ptr.reshape(shp)
    sx, sy = (sink // shp[1]).float() * res_cells + res_cells / 2, (sink % shp[1]).float() * res_cells + res_cells / 2
    gx, gy = float(goal[0]) * E + n / 2, float(goal[1]) * E + n / 2
    spurious = ((sx - gx) ** 2 + (sy - gy) ** 2 > 1.0) & ok
    return sink, ok, spurious


_FREE = {}


def sinks(net, maze, prepared, res_cells=0.25, H=12, category=4):
    """Where does descent on the field end? For each goal of the category's queries (default open - open), sink_map;
    a sink farther than 1 cell from the goal is spurious (the true T has none in open space). Returns the mean share
    of the open band's free cells that drain into spurious sinks and the mean number of spurious sinks per goal."""
    q = np.load(prepared / 'queries.npz')
    idx = np.nonzero(q['category'] == category)[0]
    gids, first = np.unique(q['goal_index'][idx], return_index=True)
    shares, counts = [], []
    for j in first:
        sink, ok, spurious = sink_map(net, maze, q['goal'][idx[j]], res_cells, H)
        shares.append(float(spurious.sum()) / max(int(ok.sum()), 1))
        counts.append(int(torch.unique(sink[spurious]).numel()))
    return dict(goals=int(len(first)), trapped_share=float(np.mean(shares)), spurious_sinks=float(np.mean(counts)), res_cells=res_cells, headings=H)


def aliasing(net, maze, prepared, res_cells=0.25, goals=12, category=4):
    """Learned T against the straight-line distance (the true travel time when the straight segment keeps the robot
    beyond the margin, up to the heading term): median T / |a - b| over such pairs (> 4 cells apart) from the first
    `goals` open - open goals to (i) 400 random open-band cells each and (ii) every spurious sink of sink_map.
    Ratio well below 1 = the field puts far poses close to the goal (their codes alias)."""
    q = np.load(prepared / 'queries.npz')
    idx = np.nonzero(q['category'] == category)[0]
    gids, first = np.unique(q['goal_index'][idx], return_index=True)
    n, E, dev = maze.n, maze.E, maze.device
    rng = np.random.default_rng(0)
    at_sink, at_random = [], []
    for j in first[:goals]:
        goal = torch.tensor(q['goal'][idx[j]], device=dev, dtype=torch.float32)
        sink, ok, sp = sink_map(net, maze, goal, res_cells)
        sk, spn, okn = sink.cpu().numpy(), sp.cpu().numpy(), ok.cpu().numpy()
        ny = sk.shape[1]

        def ratio(cells):
            if len(cells) == 0:
                return []
            sx, sy = (cells // ny) * res_cells + res_cells / 2, (cells % ny) * res_cells + res_cells / 2
            P = torch.tensor(np.stack([(sx - n / 2) / E, (sy - n / 2) / E], 1), device=dev, dtype=torch.float32)
            best = torch.full((len(P),), float('inf'), device=dev)
            for th in torch.linspace(-maze.phi_max, maze.phi_max, 13, device=dev):
                X = torch.cat([P, th.expand(len(P), 1)], 1)
                with torch.no_grad():
                    best = torch.minimum(best, net.out(torch.cat([X, goal.expand(len(X), 3)], 1))[0][:, 0])
            d = torch.linalg.vector_norm(P - goal[:2], dim=1)
            free = segment_free(maze, torch.cat([P, goal[2].expand(len(P), 1)], 1), goal.expand(len(P), 3))
            keep = free & (d / maze.pitch > 4)
            return (best / d)[keep].cpu().numpy().tolist()
        at_sink += ratio(np.unique(sk[spn]))
        at_random += ratio(rng.choice(np.nonzero(okn.reshape(-1))[0], 400, replace=False))
    med = lambda v: float(np.median(v)) if len(v) else None
    return dict(random_median=med(at_random), random_p10=float(np.quantile(at_random, 0.1)) if at_random else None, n_random=len(at_random),
                sink_median=med(at_sink), sink_p10=float(np.quantile(at_sink, 0.1)) if at_sink else None, n_sink=len(at_sink))


def figure(k, methods, out=HERE / 'report/gap_sinks.png'):
    """Basin map for one open - open goal (the goal of the longest open - open query) at gap scale k: every open-band
    cell coloured by the sink it drains to (pale grey: the goal; colours: spurious sinks), quadrant mazes in black."""
    import gapreport as GR
    import gap
    gap.install(k)
    import maze as MZ
    import warp as W
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data, preps = GR.collect()
    dev = 'cuda:0'
    n = gap.BLOCK * 2 + gap.BASE_WIDTH * k
    maze = MZ.Maze(n, dev)
    q = np.load(preps[k]['_dir'] / 'queries.npz')
    idx = np.nonzero(q['category'] == 4)[0]
    i = idx[np.argmax(q['d_cells'][idx])]
    goal, start = q['goal'][i], q['start'][i]
    info = json.load(open(preps[k]['_dir'] / 'maze.json'))
    fig, axes = plt.subplots(1, len(methods), figsize=(3.3 * len(methods), 3.6))
    rng = np.random.default_rng(0)
    for ax, m in zip(np.atleast_1d(axes), methods):
        e = data[k][m][87358]
        net = W.load_field(ROOT / 'results' / f'registered_{e["_id"]}' / 'training/checkpoint.pt', dev)
        sink, ok, spurious = sink_map(net, maze, goal)
        sk, okc, sp = sink.cpu().numpy(), ok.cpu().numpy(), spurious.cpu().numpy()
        ids = np.unique(sk[sp])
        img = np.ones(sk.shape + (3,))
        img[okc] = (0.90, 0.91, 0.93)
        pal = rng.uniform(0.25, 0.85, (len(ids), 3))
        for j, s_ in enumerate(ids):
            img[(sk == s_) & sp] = pal[j]
        ax.imshow(np.transpose(img, (1, 0, 2)), origin='lower', extent=(0, n, 0, n), interpolation='nearest')
        for x0, y0, x1, y1 in info['boxes_cells']:
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color='#3b3a37', lw=0))
        E = n + 0.1
        ax.plot(goal[0] * E + n / 2, goal[1] * E + n / 2, '*', color='#d6293b', ms=11, mec='white', mew=0.8)
        ax.set_title(f'{m}: {100 * sp.sum() / max(okc.sum(), 1):.0f} % of the band trapped', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_aspect('equal')
        del net
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print('wrote', out)


def stats(g, x, y, free, far):
    gn = torch.linalg.vector_norm(g, dim=1)
    res = (gn - 1).abs()
    toward = y[:, :2] - x[:, :2]
    cos = torch.nn.functional.cosine_similarity(-g[:, :2], toward, dim=1)
    out = {}
    for name, m in (('all', torch.ones_like(far)), ('far', far)):
        mf = m & free
        out[name] = dict(n=int(m.sum()), residual_median=float(res[m].median()), flat_share=float((gn[m] < 0.5).float().mean()),
                         n_free=int(mf.sum()), away_share=float((cos[mf] < 0).float().mean()) if mf.any() else None)
    return out


def compute(k):
    """One gap scale per process (gap.install patches the maze module once)."""
    import gapreport as GR
    import gap
    gap.install(k)
    import maze as MZ
    import data as D
    import warp as W
    import gapprepare as GPR
    from maze import MARGIN_CELLS
    data, preps = GR.collect()
    C = json.loads(OUT.read_text()) if OUT.exists() else {}
    dev = 'cuda:0'
    n = gap.BLOCK * 2 + gap.BASE_WIDTH * k
    maze = MZ.Maze(n, dev)
    todo = [(m, by_seed[87358]) for m, by_seed in sorted(data[k].items()) if 87358 in by_seed and 'aliasing' not in C.get(f'{k}|{by_seed[87358]["_id"]}', {})]
    if not todo:
        return
    x, _, _, d, _ = D.sample(maze, N_PAIRS, 5)
    keep = d[:, 1] >= MARGIN_CELLS
    wall = torch.tensor(np.concatenate([x[keep, 3:], x[keep, :3]], 1), device=dev)        # (open x, near-wall y)
    x, _, _, d, _ = GPR.sample_open(maze, N_PAIRS, 6)
    keep = d[:, 1] >= MARGIN_CELLS
    opn = torch.tensor(x[keep], device=dev)                                                   # (open x, open y)
    sets = {}
    for name, P in (('wall partner', wall), ('open partner', opn)):
        a, b = P[:, :3], P[:, 3:]
        far = torch.linalg.vector_norm(a[:, :2] - b[:, :2], dim=1) / maze.pitch >= 24
        sets[name] = (P, a, b, segment_free(maze, a, b), far)
    for m, e in todo:
        ck = ROOT / 'results' / f'registered_{e["_id"]}' / 'training/checkpoint.pt'
        net = W.load_field(ck, dev)
        C = json.loads(OUT.read_text()) if OUT.exists() else {}
        r = C.get(f'{k}|{e["_id"]}') or dict(k=k, method=m, run=e['_id'], **{name: stats(grad_first(net, P), a, b, free, far) for name, (P, a, b, free, far) in sets.items()})
        if 'traps' not in r.get('path minima', {}):
            r['path minima'] = path_minima(net, maze, preps[k]['_dir'])
        if 'sinks' not in r:
            r['sinks'] = sinks(net, maze, preps[k]['_dir'])
        r['aliasing'] = aliasing(net, maze, preps[k]['_dir'])
        C[f'{k}|{e["_id"]}'] = r
        OUT.write_text(json.dumps(C, indent=1))
        del net
        torch.cuda.empty_cache()


SEEDS_OUT = HERE / 'report/openfit_seeds.json'


def compute_seeds(k, seeds, methods):
    """Sinks and aliasing only, for other training seeds' fields (report/openfit_seeds.json, keyed k|run id, with the seed):
    kept apart from openfit.json, whose records the page indexes by (k, method) for training seed 87358."""
    import gapreport as GR
    import gap
    gap.install(k)
    import maze as MZ
    import warp as W
    data, preps = GR.collect()
    C = json.loads(SEEDS_OUT.read_text()) if SEEDS_OUT.exists() else {}
    maze = MZ.Maze(gap.BLOCK * 2 + gap.BASE_WIDTH * k, 'cuda:0')
    for m in methods:
        for s_ in seeds:
            e = data[k].get(m, {}).get(s_)
            if e is None or f'{k}|{e["_id"]}' in C:
                continue
            net = W.load_field(ROOT / 'results' / f'registered_{e["_id"]}' / 'training/checkpoint.pt', 'cuda:0')
            r = dict(k=k, method=m, seed=s_, run=e['_id'], sinks=sinks(net, maze, preps[k]['_dir']), aliasing=aliasing(net, maze, preps[k]['_dir']))
            C = json.loads(SEEDS_OUT.read_text()) if SEEDS_OUT.exists() else {}
            C[f'{k}|{e["_id"]}'] = r
            SEEDS_OUT.write_text(json.dumps(C, indent=1))
            print(k, m, s_, 'trapped', round(r['sinks']['trapped_share'], 3), 'sinks/goal', round(r['sinks']['spurious_sinks'], 1), 'T/|a-b| at sinks', r['aliasing']['sink_median'], flush=True)
            del net
            torch.cuda.empty_cache()


def main():
    import subprocess
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if '--seeds' in sys.argv:            # openfit.py 8 12 --seeds: training seeds 1 and 2 of warp + scaling and constructed
        if '--child' in sys.argv:
            return compute_seeds(int(args[0]), [1, 2], ['warp + scaling', 'constructed'])
        for k in [int(a) for a in args]:
            subprocess.run([sys.executable, __file__, str(k), '--child', '--seeds'], check=True)
        return
    ks = [int(a) for a in args] or [2, 4, 8, 12]
    if '--figure' in sys.argv:
        figure(int(args[0]) if args else 8, ['ntrl-demo', 'README warp', 'warp + scaling', 'constructed', 'uniform'])
        return
    if '--child' in sys.argv:
        compute(ks[0])
        return
    for k in ks:
        subprocess.run([sys.executable, __file__, str(k), '--child'], check=True)
    C = json.loads(OUT.read_text()) if OUT.exists() else {}
    print(f'{"k":>3} {"method":34} | {"wall partner: resid  flat  away":32} | {"open partner: resid  flat  away":32} | {"open, >= 24 cells: resid flat away":34} | path minima /100 cells, share')
    for key, r in sorted(C.items(), key=lambda t: (t[1]['k'], t[1]['method'])):
        w, o = r['wall partner']['all'], r['open partner']['all']
        of = r['open partner']['far']
        f = lambda s: f'{s["residual_median"]:6.2f} {100 * s["flat_share"]:5.1f} {"  -  " if s["away_share"] is None else f"{100 * s["away_share"]:5.1f}"}'
        pm = r.get('path minima')
        pm = f'{pm["per_100_cells"]:5.2f} {100 * pm["share_with_minimum"]:5.1f}  traps {100 * pm["share_with_trap"]:5.1f}' if pm else ''
        sk = r.get('sinks')
        pm += f' | open band draining to spurious sinks {100 * sk["trapped_share"]:5.1f} %, sinks/goal {sk["spurious_sinks"]:6.1f}' if sk else ''
        al = r.get('aliasing')
        fm = lambda v: ' -  ' if v is None else f'{v:.2f}'
        pm += f' | T/|a-b| random {fm(al["random_median"])}, at sinks {fm(al["sink_median"])}' if al else ''
        print(f'{r["k"]:>3} {r["method"]:34} | {f(w):32} | {f(o):32} | {f(of):34} | {pm}')


if __name__ == '__main__':
    main()
