"""Does a field feel the walls of a quadrant when the goal is far away?  (post-hoc, on registered runs' checkpoints)

    ~/.venvs/bellman-ntfield/bin/python jeffrey/aniso_eikonal/wallfeel.py --run <run id>      # one run -> report/wallfeel/<id>.json
    ~/.venvs/bellman-ntfield/bin/python jeffrey/aniso_eikonal/wallfeel.py --all [--gpu 1]     # the runs the page uses (seed 87358), one process each

MPPI only compares the field's values at the rollouts' points, so it stays off a wall only where T(x, goal) rises as x
moves into the wall's margin. The eikonal term fixes the size of the gradient, not its sign along the wall's normal.

Poses x: free poses inside the four quadrant mazes, binned by clearance. Goals: free poses (clearance >= 0.1 cells) in the
same quadrant / an adjacent quadrant / the diagonal quadrant / the open band. Per (bin, goal class):
  speed    median of  t_scale / |grad_x T|  over the maze's speed  (1 = the equation holds)
  rises    share of poses where T rises into the wall  (grad_x T . n < 0, n the clearance normal)
  normal   quantiles of  -S grad_x T . n / t_scale  (about +1 deep in the margin for the true travel time: out of the margin first)
Also where the plans of planning seed 0 fail: first wall contact (or stall) by stage of the trip; and, for queries from the
open band into a quadrant, how far from the true path's entrance a plan first comes within one cell of the goal quadrant.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'report/wallfeel'
BINS = [(0.02, 0.06), (0.06, 0.12), (0.12, 0.175), (0.175, 9.0)]
BIN_NAMES = ['0.02-0.06', '0.06-0.12', '0.12-0.175', 'beyond the margin']
CLS = ['same quadrant', 'adjacent quadrant', 'diagonal quadrant', 'open band']
STAGES = ['start quadrant', 'band', 'goal quadrant', 'other quadrant']


def one(rid, dev='cuda:0'):
    import torch
    sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
    sys.path.insert(0, str(ROOT / 'jeffrey/cross_warp'))
    sys.path.insert(0, str(HERE))
    import gap as GP
    import plan as P
    import aniso as AN
    out = ROOT / f'results/registered_{rid}'
    res = json.load(open(out / 'results.json'))
    prep = Path(res['prepared'])
    info = json.load(open(prep / 'maze.json'))
    k, base = info['gap_scale'], GP.base_name(info)
    GP.install(k, base)
    import maze as MZ
    maze = MZ.Maze(info['n'], dev)
    ck = Path(res['checkpoint_from']) if res.get('checkpoint_from') else out / 'training/checkpoint.pt'
    net = AN.load_field(ck, dev)
    t_scale = res['t_scale']
    n, E = maze.n, maze.E
    cells = lambda x: (np.asarray(x[:, 0].cpu() if hasattr(x, 'cpu') else x[:, 0]) * E + n / 2, np.asarray(x[:, 1].cpu() if hasattr(x, 'cpu') else x[:, 1]) * E + n / 2)
    gen = torch.Generator(dev).manual_seed(7)
    X = maze.free(400_000, gen, 0.02)
    rX = GP.region_of(n, *cells(X))
    X, rX = X[torch.tensor(rX < 4, device=dev)], rX[rX < 4]
    d = maze.clearance(X).cpu().numpy()
    keep = np.concatenate([np.nonzero((d >= lo) & (d < hi))[0][:6000] for lo, hi in BINS])
    X, rX, d = X[keep], rX[keep], d[keep]
    Gf = maze.free(200_000, gen, 0.1)
    rG = GP.region_of(n, *cells(Gf))
    diag = {0: 3, 3: 0, 1: 2, 2: 1}
    rng = np.random.default_rng(3)
    pools = {r: np.nonzero(rG == r)[0] for r in range(5)}
    dcl, nrm = maze.clearance_normal(X)
    nrm = nrm / torch.linalg.vector_norm(nrm, dim=1, keepdim=True).clamp(min=1e-9)
    s_true = maze.train_speed(dcl)
    feel = {}
    for ci, cname in enumerate(CLS):
        gi = np.empty(len(X), int)
        for i in range(len(X)):
            r = int(rX[i])
            pool = pools[r] if ci == 0 else pools[[q for q in range(4) if q != r and q != diag[r]][rng.integers(2)]] if ci == 1 else pools[diag[r]] if ci == 2 else pools[4]
            gi[i] = pool[rng.integers(len(pool))]
        _, g = P.field_grad(net, torch.cat([X, Gf[torch.tensor(gi, device=dev)]], 1))
        sp = (t_scale / torch.linalg.vector_norm(g, dim=1).clamp(min=1e-12) / s_true).cpu().numpy()
        gn = (g * nrm).sum(1)
        rises = (gn < 0).cpu().numpy()
        nor = (-s_true * gn / t_scale).cpu().numpy()
        feel[cname] = {}
        for (lo, hi), bname in zip(BINS, BIN_NAMES):
            m = (d >= lo) & (d < hi)
            feel[cname][bname] = dict(poses=int(m.sum()), speed_median=float(np.median(sp[m])), rises=float(rises[m].mean()),
                                      normal_quantiles=[float(v) for v in np.quantile(nor[m], [0.1, 0.25, 0.5, 0.75, 0.9])])
    # where the plans fail (planning seed 0)
    where = {}
    planner = 'z' if (out / 'paths_z_seed0.npy').exists() else 'x'
    if (out / f'paths_{planner}_seed0.npy').exists():
        q = np.load(prep / 'queries.npz')
        S, G, cat = q['start'], q['goal'], q['category']
        rows = np.load(out / 'rows.npz')
        ok, conv = rows[f'{planner}_success'][0], rows[f'{planner}_converged'][0]
        paths = np.load(out / f'paths_{planner}_seed0.npy', allow_pickle=True)
        rS, rGq = GP.region_of(n, *cells(S)), GP.region_of(n, *cells(G))
        stage = lambda reg, a, b: 1 if reg == 4 else 0 if reg == a else 2 if reg == b else 3
        for c in range(len(GP.CATEGORIES)):
            idx = np.nonzero(cat == c)[0]
            contact, stall, last = np.zeros(4, int), np.zeros(4, int), 0
            for i in idx:
                if ok[i]:
                    continue
                path = torch.tensor(paths[i][:-1], device=dev)               # without the goal plan.mppi appends
                dense = P.densify(path, 0.005 * maze.pitch) if len(path) > 1 else path
                dd = maze.clearance(dense).cpu().numpy()
                reg = GP.region_of(n, *cells(dense.cpu().numpy()))
                bad = np.nonzero(dd < 0)[0]
                st = stage(reg[bad[0] if len(bad) else len(reg) - 1], rS[i], rGq[i])
                if len(bad):
                    contact[st] += 1
                elif conv[i]:
                    last += 1
                else:
                    stall[st] += 1
            where[GP.CATEGORIES[c]] = dict(queries=int(len(idx)), solved=int(ok[idx].sum()), first_contact=dict(zip(STAGES, contact.tolist())), stalled=dict(zip(STAGES, stall.tolist())), contact_only_in_last_jump=int(last))
    # open -> quadrant queries: how far from the true path's entrance a plan first reaches the goal quadrant's outline
    arrival = {}
    if (prep / 'truth_paths.npy').exists():
        q = np.load(prep / 'queries.npz')
        S, G, cat = q['start'], q['goal'], q['category']
        truth = np.load(prep / 'truth_paths.npy', allow_pickle=True)
        xy = lambda a_: np.stack([a_[:, 0] * E + n / 2, a_[:, 1] * E + n / 2], 1)
        rS, rGq = GP.region_of(n, *xy(S).T), GP.region_of(n, *xy(G).T)
        Bk, Gw = GP.BLOCK, n - 2 * GP.BLOCK

        def to_block(pts, r):
            x0 = 0 if r in (0, 2) else Bk + Gw
            y0 = 0 if r in (0, 1) else Bk + Gw
            dx = np.maximum(np.maximum(x0 - pts[:, 0], pts[:, 0] - (x0 + Bk)), 0)
            dy = np.maximum(np.maximum(y0 - pts[:, 1], pts[:, 1] - (y0 + Bk)), 0)
            return np.hypot(dx, dy)
        idx = np.nonzero((cat == 3) & (rS == 4) & (rGq < 4))[0]
        rows = np.load(out / 'rows.npz')
        for pl in ('z', 'zh'):
            f = out / f'paths_{pl}_seed0.npy'
            if not f.exists():
                continue
            paths = np.load(f, allow_pickle=True)
            off = []
            for i in idx:
                tp = xy(np.asarray(truth[i], float))
                inside = np.nonzero(to_block(tp, int(rGq[i])) == 0)[0]
                pp = xy(np.asarray(paths[i][:-1], float))
                near = np.nonzero(to_block(pp, int(rGq[i])) <= 1.0)[0]
                if len(inside) and len(near):
                    off.append(float(np.hypot(*(pp[near[0]] - tp[inside[0]]))))
            if off:
                arrival[pl] = dict(queries=int(len(idx)), reached=len(off), median_cells=float(np.median(off)), p75_cells=float(np.quantile(off, 0.75)),
                                   solved_seed0=int(rows[f'{pl}_success'][0][idx].sum()))
    OUT.mkdir(parents=True, exist_ok=True)
    result = dict(id=rid, base=base, k=k, mode=res['mode'], planner=planner, feel=feel, fail_where=where, arrival=arrival,
                  note='feel: poses inside the quadrant mazes by clearance bin (cells) and goal class; fail_where: planning seed 0, first wall contact by stage of the trip')
    json.dump(result, open(OUT / f'{rid}.json', 'w'), indent=1)
    row = lambda c: ' | '.join(f'{100 * feel[c][b]["rises"]:5.1f}% {feel[c][b]["normal_quantiles"][2]:+.2f}' for b in BIN_NAMES)
    print(f'{rid} {base} x{k} {res["mode"]}: T rises into the wall (share, median normal slope) by clearance bin')
    for c in CLS:
        print(f'   {c:18s} {row(c)}')
    for c, w in where.items():
        print(f'   {c:20s} solved {w["solved"]}/{w["queries"]}; first contact {w["first_contact"]}; stalled {sum(w["stalled"].values())}')


def wanted(S):
    """Runs the page shows: seed 87358, the reference, the inner product alone, every recipe / far variant with open-start pairs."""
    out = []
    for r in S['runs']:
        v = r['variant']
        if r['seed'] != 87358 or 'messy' in v or r.get('checkpoint_from'):
            continue
        if v in ('ziso-open', 'ziso', 'zaniso-Uquad', 'zres-Uquad-open-far-lb10', 'zaniso-Uquad-open-pwtauz-lb10', 'zres-Uquad-open-far-lb10-full', 'zres-Uquad-open-far-lb10-full1s') or ('-open' in v and 'pwtauz' in v and 'lb10' in v and ('gs8x1+8x' in v or 'far' in v)):
            out.append(r['id'])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run')
    p.add_argument('--all', action='store_true')
    p.add_argument('--gpu', default='1')
    p.add_argument('--force', action='store_true', help='--all: recompute the runs already analysed')
    a = p.parse_args()
    if a.run:
        one(a.run)
        return
    S = json.load(open(HERE / 'report/summary.json'))
    todo = [rid for rid in wanted(S) if a.force or not (OUT / f'{rid}.json').exists()]
    print(f'{len(todo)} runs to analyse')
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=a.gpu)
    for rid in todo:
        r = subprocess.run([sys.executable, str(HERE / 'wallfeel.py'), '--run', rid], env=env, capture_output=True, text=True)
        print(r.stdout.strip() if r.returncode == 0 else f'{rid} FAILED: {r.stderr[-600:]}')


if __name__ == '__main__':
    main()
