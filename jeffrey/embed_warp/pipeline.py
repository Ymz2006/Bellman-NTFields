"""One embed-warp run: ntrl-demo with a gated multi-octave embedding (features.py), trained (trainer.py) and planned on a prepared scene
(a jeffrey/maze_scale maze, plus or gap variant, or an embed_warp prepare run: disks, Gibson).

    CUDA_VISIBLE_DEVICES=1 python jeffrey/embed_warp/pipeline.py --prepared <prepare run> --seed 87358 --octaves 2,4,8,16 --gate geo --out <dir>

--octaves ''          ntrl-demo unchanged (the control)
--gate const          octaves always on (uniform multi-scale control)
--gate geo|kde|diff   fixed grid gates from the geometry / the band samples / masked diffusion (gates.py)
--gate curv           grid gate from a trained field's curvature: --curv-from <run dir> (its training/checkpoint.pt)
--gate learn          MLP gate on u_0 (or x) trained with the field
Then maze_scale/evaluate.py: MPPI planning seeds 0/1/2, exact scoring, success by bin, field diagnostics; speed fit; near-wall slope and
direction; local minima on true paths; failures by arrival; the metric the input sees (features.isotropy) at band and open poses;
gate statistics; a slice of T(., goal 0) and the gate at heading 0 for the viewer.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import features as FE  # noqa: E402
import gates as GT  # noqa: E402
import trainer as TR  # noqa: E402
import diagnostics as DG  # noqa: E402
import scene as SC  # noqa: E402  (adds maze_scale to the front of sys.path, cross_warp to the back)


def parse_octaves(s):
    return [float(v) for v in s.split(',') if v.strip()] if s else []


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=5000)
    p.add_argument('--octaves', default='', help="scales of the extra Fourier octaves relative to ntrl-demo's rows, e.g. 2,4,8,16 ('' = ntrl-demo)")
    p.add_argument("--rows", type=int, default=16, help="rows per octave (taken from ntrl-demo's 128; the rest stay base rows)")
    p.add_argument("--base-scale", type=float, default=1.0, help="frequency factor of the base rows")
    p.add_argument('--no-whiten', action='store_true', help="keep the octave rows' random directions (default: whitened so each octave's metric is isotropic)")
    p.add_argument('--gate', choices=['none', 'const', 'geo', 'kde', 'diff', 'curv', 'learn'], default='none')
    p.add_argument('--mu-min', type=float, default=0.02, help='the gated octaves are compressed to this where the gate is closed')
    p.add_argument('--grid-res', type=int, default=4, help='gate grid nodes per cell (x, y)')
    p.add_argument('--grid-theta', type=int, default=13, help='gate grid nodes over the heading range')
    p.add_argument('--r0', type=float, default=0.5, help='geo: clearance (cells) up to which the gate is fully open')
    p.add_argument('--r1', type=float, default=1.5, help='geo: clearance (cells) from which the gate is closed')
    p.add_argument('--per-octave', action='store_true', help='geo: octave k gets radii / 2^k')
    p.add_argument('--kde-sigma', type=float, default=0.5, help='kde: blur (cells)')
    p.add_argument('--kde-q', type=float, default=0.5, help='kde: density quantile over the occupied cells that maps to w = 1')
    p.add_argument('--diff-src', type=float, default=0.5, help='diff: clearance (cells) of the heat source')
    p.add_argument('--diff-steps', type=int, default=32)
    p.add_argument('--diff-gain', type=float, default=1.0)
    p.add_argument('--curv-from', type=Path, default=None, help='curv: run whose training/checkpoint.pt gives the field')
    p.add_argument('--curv-kind', choices=['hess', 'resid'], default='hess')
    p.add_argument('--curv-q', type=float, default=0.9)
    p.add_argument('--curv-blur', type=float, default=0.5)
    p.add_argument('--curv-goals', type=int, default=8)
    p.add_argument('--gate-input', choices=['u0', 'x'], default='u0')
    p.add_argument('--gate-hidden', type=int, default=64)
    p.add_argument('--gate-init', type=float, default=0.5)
    p.add_argument('--gate-lr', type=float, default=1e-3)
    p.add_argument('--gate-budget', type=float, default=None, help='learn: mean mu over uniform poses pushed below this')
    p.add_argument('--gate-budget-weight', type=float, default=0.0)
    p.add_argument('--td-cells', type=float, default=None, help="ntrl-demo's TD step in cells (its own: 0.03 E, i.e. 0.66 cells at n = 22)")
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from common import Logger, dump
    import plan as P
    import evaluate as EV
    log = Logger(a.out / 'log.txt')
    dev = 'cuda:0'
    t0 = time.time()
    scene, info = SC.load_scene(a.prepared, dev)
    epochs = 3 if a.smoke else a.epochs
    octaves = parse_octaves(a.octaves)
    K = len(octaves)
    gate_kind = a.gate if K else 'none'
    xs, ts = GT.axes(scene, a.grid_res, a.grid_theta)
    q, shape = GT.grid_points(scene, xs, ts)
    gcfg = dict(kind=gate_kind, mu_min=a.mu_min)
    table, gate_info = None, {}
    if gate_kind in ('geo', 'kde', 'diff', 'curv'):
        gcfg.update(shape=list(shape), lo=scene.lo.tolist(), hi=scene.hi.tolist())
        if gate_kind == 'geo':
            table, gate_info = GT.geo_table(scene, q, shape, a.r0, a.r1, K, a.per_octave)
        elif gate_kind == 'kde':
            band = torch.tensor(np.load(a.prepared / 'data/sampled_points.npy', mmap_mode='r')[:, :3], device=dev)
            table, gate_info = GT.kde_table(scene, band, shape, xs, ts, a.kde_sigma, a.grid_res, a.kde_q, K)
            del band
        elif gate_kind == 'diff':
            table, gate_info = GT.diff_table(scene, q, shape, a.diff_src, a.diff_steps, a.diff_gain, K)
        else:
            assert a.curv_from is not None, '--gate curv needs --curv-from'
            prev = FE.load_field(a.curv_from / 'training/checkpoint.pt', dev)
            goals = torch.tensor(np.load(a.prepared / 'goals.npy')[:a.curv_goals], device=dev)
            table, gate_info = GT.curv_table(prev, scene, q, shape, goals, a.curv_kind, a.curv_q, a.curv_blur, a.grid_res, K)
            del prev
        gate_info['by_clearance'] = GT.table_stats(scene, table, q, shape)
        np.save(a.out / 'gate_table.npy', table[0].half().cpu().numpy())
        log(f'gate {gate_kind}: {gate_info}')
    elif gate_kind == 'learn':
        gcfg.update(gate_input=a.gate_input, in_dim=256, hidden=a.gate_hidden, init=a.gate_init)
    elif gate_kind == 'const':
        gcfg.update(const=1.0)
    ecfg = dict(octaves=octaves, rows=a.rows, base_scale=a.base_scale, whiten=not a.no_whiten, gate=gcfg)
    res = dict(stage='embed-warp', scene=scene.describe(), seed=a.seed, epochs=epochs, prepared=str(a.prepared), gpu=torch.cuda.get_device_name(0), smoke=a.smoke,
               embed=ecfg, gate_info=gate_info, args=dict(vars(a), prepared=str(a.prepared), out=str(a.out), curv_from=str(a.curv_from) if a.curv_from else None))
    log(f'scene {info.get("variant", "plain")} n={scene.n}, octaves {octaves} rows {a.rows} (base rows {128 - K * a.rows} x {a.base_scale}), gate {gate_kind} mu_min {a.mu_min}')
    ck, hist = TR.train(a.prepared / 'data', a.out / 'training', a.seed, epochs, ecfg, log, scene=scene, gate_table=table, gate_lr=a.gate_lr,
                        budget=a.gate_budget, budget_weight=a.gate_budget_weight, td_len=None if a.td_cells is None else a.td_cells / scene.E)
    res['train_seconds'] = time.time() - t0
    res['final_loss'] = hist['final_loss']
    res['rollbacks'] = int(sum(hist['rollbacks']))
    log(f'trained: {res["train_seconds"] / 60:.1f} min, final ntrl loss {res["final_loss"]:.4e}, rollbacks {res["rollbacks"]}')
    net = FE.load_field(ck, dev)
    cost, grad = P.field_cost(net), lambda pairs: P.field_grad(net, pairs)
    res.update(EV.evaluate(scene, a.prepared, cost, grad, a.out, log, label=f'embed n={scene.n} s{a.seed}'))
    res['speed_fit'] = EV.speed_fit(scene, a.prepared, grad)
    goals = torch.tensor(np.load(a.prepared / 'goals.npy'), device=dev)
    np.save(a.out / 'slice_goal0.npy', EV.slice_values(scene, cost, goals[0]))
    log(f'speed fit: {res["speed_fit"]}')
    rows = np.load(a.out / 'rows.npz')
    Q = np.load(a.prepared / 'queries.npz')
    res['failures'] = DG.failures_by_arrival(rows)
    if 'category' in Q.files:
        import gap as GP
        res['by_category'] = {name: dict(success=int(rows['success'][:, Q['category'] == c].sum()), runs=int(rows['success'][:, Q['category'] == c].size))
                              for c, name in enumerate(GP.CATEGORIES) if (Q['category'] == c).any()}
    if info.get('variant') == 'plus':
        import plus as PL
        cells = PL.plus_cells(scene.n, info['plus_width'])

        def inside(x):
            c = np.clip(np.floor(x[:, 0] * scene.E + scene.n / 2).astype(int), 0, scene.n - 1)
            r = np.clip(np.floor(x[:, 1] * scene.E + scene.n / 2).astype(int), 0, scene.n - 1)
            return cells[c, r]
        where = inside(Q['start']).astype(int) + inside(Q['goal']).astype(int)
        res['by_where'] = {name: dict(success=int(rows['success'][:, where == k].sum()), runs=int(rows['success'][:, where == k].size)) for k, name in ((2, 'both'), (1, 'one'), (0, 'neither'))}
    res['near_wall'] = DG.near_wall(scene, a.prepared, grad)
    res['bumps'] = DG.bumps(scene, a.prepared, cost, P)
    # the metric the input sees: band poses (held-out x0) vs open poses (clearance >= 2 cells)
    hx = torch.tensor(np.load(a.prepared / 'heldout/sampled_points.npy')[:4096, :3], device=dev)
    gen = torch.Generator(dev).manual_seed(99)
    pool = scene.free(40_000, gen, 0.0)
    d = scene.clearance(pool)
    open_q = pool[d >= 2.0][:4096]
    iso = {}
    for name, pts in (('band', hx), ('open', open_q), ('uniform', pool[:4096])):
        if len(pts) > 10:
            m = FE.isotropy(net, pts)
            iso[name] = {k: FE.summarize(v) for k, v in m.items()}
    res['isotropy'] = iso
    if net.K:
        with torch.no_grad():
            res['gate_mu'] = {name: FE.summarize(net.gates(pts).mean(1)) for name, pts in (('band', hx), ('open', open_q), ('uniform', pool[:4096])) if len(pts) > 10}
        # gate map at heading 0 for the viewer (res 8 n + 1, as the slice)
        r = 8 * scene.n + 1
        g1 = torch.linspace(-0.5, 0.5, r, device=dev)
        X, Y = torch.meshgrid(g1, g1, indexing='ij')
        qq = torch.stack([X.reshape(-1), Y.reshape(-1), torch.zeros(r * r, device=dev)], 1)
        with torch.no_grad():
            mu = torch.cat([net.gates(qq[i:i + 65536]).mean(1) for i in range(0, len(qq), 65536)])
        np.save(a.out / 'gate_slice.npy', mu.reshape(r, r).half().cpu().numpy())
    log(f'failures {res["failures"]}, by where {res.get("by_where")}, near wall {res["near_wall"]["all"]}, bumps {res["bumps"]}')
    log(f'isotropy band stretch {iso.get("band", {}).get("stretch")}, aniso {iso.get("band", {}).get("aniso")}; open stretch {iso.get("open", {}).get("stretch")}, aniso {iso.get("open", {}).get("aniso")}; gate mu {res.get("gate_mu")}')
    res['elapsed_seconds'] = time.time() - t0
    res['complete'] = True
    dump(a.out / 'results.json', res)
    log(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
