"""One field of the mobile-arm study: ntrl-demo (dim 9) trained on a prepared house, with or without the travel-time midpoint loss,
then planned and diagnosed.

    CUDA_VISIBLE_DEVICES=0 python jeffrey/mobile_arm/pipeline9.py --prepared <prepare run> --seed 87358 --weight 1 --out <dir>

--weight 0   ntrl-demo itself (the baseline); --weight 1 = jeffrey/midpoint_dp's MAIN setting from epoch 2,000 (safe traversals, min leaves,
             tree + sub-paths, ramp over 500 epochs, leaves of 1 cell of travel time, 8,192 traversals every 100 epochs, 8,192 nodes per update).
--resume     continue a checkpoint (as midpoint_dp: the baseline's epoch-2,000 network, AdamW state and B): the midpoint run with --weight 1,
             the control (the same resumed loop without the term) with --weight 0.
Planning: ntrl-demo's MPPI (plan9.py) on the prepared queries for planning seeds 0 / 1 / 2, per-query budget 3 x the reference path's
C-space length / step; exact scoring. Failures are split by arrival (the planner appends the goal to every path): arrived through
something (with the part touched first: base, arm against a wall, floor, base box, self) or did not arrive. Field: T(start, goal) against
the two bounds T*_SE2 <= T* <= T_ref; speed fit 1 / |dT/dx0| against the label on held-out pairs, by the constraint that sets the label
(base against a wall, arm against a wall, internal). During training (diagnostics only) the current field plans the queries with
planning seed 0 at --validate-at epochs (RNG states saved and restored).
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')        # no pool of compile worker processes (host RAM)
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scene9 as S9  # noqa: E402
import trainer9 as TRN  # noqa: E402
import plan9 as P9  # noqa: E402
import fastnet as F  # noqa: E402
from common import BINS, PLAN_SEEDS, Logger, bin_labels, by_bin, dump, save_paths  # noqa: E402  (maze_scale)

SAVE_AT = (1000, 2000, 3000, 4000)
VALIDATE_AT = (1000, 2000, 3000, 4000)


def load_queries(prepared, dev):
    z = np.load(prepared / 'queries.npz')
    Q = {k: z[k] for k in z.files}
    Q['start_t'] = torch.tensor(Q['start'], device=dev)
    Q['goal_t'] = torch.tensor(Q['goal'], device=dev)
    return Q


def classify(rows):
    ok = np.array([r['success'] for r in rows])
    cv = np.array([r['converged'] for r in rows])
    cl = np.array([r['collision'] for r in rows])
    parts = {p: int(sum(1 for r in rows if r['converged'] and r['collision'] and r.get('contact_part') == p)) for p in P9.PARTS}
    return dict(success=int(ok.sum()), runs=len(rows), arrived_through=int((~ok & cv & cl).sum()), not_arrived=int((~cv).sum()), arrived_through_parts=parts)


def evaluate(scene, Q, net, out, log, label, plan_seeds=PLAN_SEEDS):
    S, G, bins = Q['start_t'], Q['goal_t'], Q['bin']
    st = P9.settings(scene)
    budget = torch.tensor(np.maximum(st['min_steps'], np.ceil(st['budget_factor'] * Q['ref_clen'] / st['step'])), device=scene.device)
    res = dict(planner=dict(st, budget_median=float(budget.median()), budget_max=float(budget.max())), planning={})
    n = len(S)
    success = np.zeros((len(plan_seeds), n), bool)
    collision = np.zeros_like(success)
    converged = np.zeros_like(success)
    time_ratio = np.full(success.shape, np.nan)
    time_ratio_se2 = np.full(success.shape, np.nan)
    part = np.full(success.shape, '', dtype=object)
    cost = P9.field_cost(net)
    for k, ps in enumerate(plan_seeds):
        tp = time.time()
        paths, stop, conv = P9.mppi(cost, S, G, ps, budget, scene, **st)
        rows = P9.score(scene, [paths[i] for i in range(n)], conv)
        success[k] = [r['success'] for r in rows]
        collision[k] = [r['collision'] for r in rows]
        converged[k] = [r['converged'] for r in rows]
        time_ratio[k] = [r['time'] / Q['t_ref'][i] if r['success'] else np.nan for i, r in enumerate(rows)]
        time_ratio_se2[k] = [r['time'] / Q['t_se2'][i] if r['success'] else np.nan for i, r in enumerate(rows)]
        part[k] = [r.get('contact_part', '') if (r['converged'] and r['collision']) else '' for r in rows]
        c = classify(rows)
        res['planning'][str(ps)] = dict(c, by_bin=by_bin(rows, bins), steps_median=float(stop.float().median()), seconds=time.time() - tp)
        if k == 0:
            save_paths(out / 'paths_seed0.npy', [paths[i] for i in range(n)])
        log(f'{label} MPPI seed {ps}: {c["success"]}/{n} (arrived through {c["arrived_through"]} {c["arrived_through_parts"]}, did not arrive {c["not_arrived"]}) '
            f'by bin {res["planning"][str(ps)]["by_bin"]} ({time.time() - tp:.0f}s)')
    labels = bin_labels()
    res['success_by_bin'] = {labels[b]: [int(success[:, bins == b].sum()), int(len(plan_seeds) * (bins == b).sum())] for b in sorted(set(bins.tolist()))}
    res['success_total'] = [int(success.sum()), int(success.size)]
    res['arrived_through_total'] = int((~success & converged & collision).sum())
    res['not_arrived_total'] = int((~converged).sum())
    res['arrived_through_parts'] = {p: int((part == p).sum()) for p in P9.PARTS}
    res['time_over_ref_median'] = float(np.nanmedian(time_ratio)) if np.isfinite(time_ratio).any() else None
    res['time_over_ref_by_bin'] = {labels[b]: (float(np.nanmedian(time_ratio[:, bins == b])) if np.isfinite(time_ratio[:, bins == b]).any() else None) for b in sorted(set(bins.tolist()))}
    np.savez(out / 'rows.npz', success=success, collision=collision, converged=converged, time_ratio=time_ratio, time_ratio_se2=time_ratio_se2,
             part=part.astype(str), bin=bins)
    return res


def field_diag(scene, Q, net, prepared):
    """T(start, goal) against T*_SE2 (lower bound) and T_ref (upper bound); speed fit on the held-out pairs by active constraint."""
    from scipy.stats import spearmanr
    labels = bin_labels()
    pairs = torch.cat([Q['start_t'], Q['goal_t']], 1)
    t = P9.field_cost(net)(pairs).cpu().numpy()
    lo, hi, bins = Q['t_se2'], Q['t_ref'], Q['bin']
    out = dict(t_over_ref_median=float(np.median(t / hi)), t_over_se2_median=float(np.median(t / lo)), below_lower_bound=float((t < lo).mean()),
               above_upper_bound=float((t > hi).mean()), spearman_ref=float(spearmanr(t, hi)[0]), by_bin={})
    for b in sorted(set(bins.tolist())):
        m = bins == b
        out['by_bin'][labels[b]] = dict(t_over_ref=float(np.median(t[m] / hi[m])), t_over_se2=float(np.median(t[m] / lo[m])), below_lower=float((t[m] < lo[m]).mean()),
                                        spearman_ref=float(spearmanr(t[m], hi[m])[0]) if m.sum() > 2 else None)
    hx = torch.tensor(np.load(prepared / 'heldout/sampled_points.npy'), device=scene.device)
    hd = torch.tensor(np.load(prepared / 'heldout/clearance_cells.npy'), device=scene.device)
    _, g = P9.field_grad(net, hx)
    s_pred = 1 / torch.linalg.vector_norm(g, dim=1).clamp(min=1e-6)
    s_true = scene.train_speed(hd[:, 0])
    pt = scene.parts(hx[:, :9])
    act = torch.stack([pt['rect'], pt['arm'], S9.K_INT * pt['int']], 1).argmin(1)
    lp, lt = torch.log(s_pred), torch.log(s_true)
    sf = dict(log_speed_corr=float(torch.corrcoef(torch.stack([lp, lt]))[0, 1]), log_speed_mae=float((lp - lt).abs().mean()), by_constraint={})
    for k, name in enumerate(('base_rect', 'arm_walls', 'internal')):
        m = act == k
        if int(m.sum()) > 10:
            sf['by_constraint'][name] = dict(n=int(m.sum()), log_speed_corr=float(torch.corrcoef(torch.stack([lp[m], lt[m]]))[0, 1]), log_speed_mae=float((lp[m] - lt[m]).abs().mean()),
                                             pred_over_true_median=float((s_pred[m] / s_true[m]).median()))
    out['speed_fit'] = sf
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=5000)
    p.add_argument('--start', type=int, default=2000)
    p.add_argument('--weight', type=float, default=1.0)
    p.add_argument('--ramp', type=int, default=500)
    p.add_argument('--refresh', type=int, default=100)
    p.add_argument('--traversals', type=int, default=8192)
    p.add_argument('--batch', type=int, default=8192)
    p.add_argument('--leaf', choices=['net', 'min', 'path'], default='min')
    p.add_argument('--leaf-cells', type=float, default=1.0)
    p.add_argument('--loss', choices=['l1', 'l2', 'huber'], default='l1')
    p.add_argument('--nodes', choices=['tree', 'sub', 'both'], default='both')
    p.add_argument('--sub-factor', type=float, default=1.0)
    p.add_argument('--min-clear', type=float, default=0.1)
    p.add_argument('--budget-factor', type=float, default=2.0)
    p.add_argument('--max-steps', type=int, default=0, help='traversal step cap; 0 = 5 n cells of motion')
    p.add_argument('--stall-window', type=int, default=60)
    p.add_argument('--stall-drop-cells', type=float, default=0.05)
    p.add_argument('--trim-hover', type=float, default=0.25)
    p.add_argument('--traverse', choices=['plain', 'safe'], default='safe')
    p.add_argument('--safe-margin', type=float, default=0.02)
    p.add_argument('--safe-mode', choices=['first', 'ends'], default='first')
    p.add_argument('--chunk', type=int, default=1 << 20)
    p.add_argument('--no-compile', action='store_true')
    p.add_argument('--max-rollbacks', type=int, default=20)
    p.add_argument('--validate-at', default=','.join(map(str, VALIDATE_AT)))
    p.add_argument('--resume', type=Path, default=None, help='continue this checkpoint (the baseline run\'s epoch 2,000) instead of training from epoch 1')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    log = Logger(a.out / 'log.txt')
    dev = 'cuda:0'
    t0 = time.time()
    scene, info = S9.load_scene(a.prepared, dev)
    if a.smoke:
        a.epochs, a.start, a.refresh, a.traversals, a.ramp, a.batch = 40, 20, 10, 512, 10, 2048
    st = P9.settings(scene)
    if a.max_steps <= 0:
        a.max_steps = int(math.ceil(5 * scene.n * scene.pitch / st['step']))
    save_at = (10, 20, 30) if a.smoke else SAVE_AT
    validate_at = (30,) if a.smoke else tuple(int(x) for x in a.validate_at.split(',') if x)
    cfg = dict(seed=a.seed, epochs=a.epochs, start=a.start, weight=a.weight, ramp=a.ramp, refresh=a.refresh, traversals=a.traversals, batch=a.batch,
               leaf=a.leaf, leaf_cells=a.leaf_cells, loss=a.loss, nodes=a.nodes, sub_factor=a.sub_factor, min_clear=a.min_clear, budget_factor=a.budget_factor,
               max_steps=a.max_steps, stall_window=a.stall_window, stall_drop_cells=a.stall_drop_cells, trim_hover=a.trim_hover, traverse=a.traverse,
               safe_margin=a.safe_margin, safe_mode=a.safe_mode, chunk=a.chunk, compile=not a.no_compile, max_rollbacks=a.max_rollbacks,
               save_at=list(save_at), validate_at=list(validate_at), planner=st)
    if a.resume is not None:
        cfg['resume'] = str(a.resume)
    house = info['scene']['house']['name']
    res = dict(stage='mobile-arm-field', scene=scene.describe(), house=house, seed=a.seed, prepared=str(a.prepared), gpu=torch.cuda.get_device_name(0),
               smoke=a.smoke, config={k: v for k, v in cfg.items() if k != 'planner'}, planner=st, args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()})
    log(f'house {house} E={scene.E:g}, 9-DOF mobile arm; midpoint weight {a.weight} from epoch {a.start}; config {json.dumps(res["config"])}')
    Q = load_queries(a.prepared, dev)
    budget = torch.tensor(np.maximum(st['min_steps'], np.ceil(st['budget_factor'] * Q['ref_clen'] / st['step'])), device=dev)

    def validate(network, epoch):
        cpu_state, cuda_state = torch.get_rng_state(), torch.cuda.get_rng_state()     # plan9.mppi reseeds the global RNGs
        tv = time.time()
        paths, stop, conv = P9.mppi(P9.field_cost(network), Q['start_t'], Q['goal_t'], 0, budget, scene, **st)
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state)
        rows = P9.score(scene, [paths[i] for i in range(len(Q['start_t']))], conv)
        out = dict(classify(rows), by_bin=by_bin(rows, Q['bin']), seconds=time.time() - tv)
        log(f'validate epoch {epoch} (planning seed 0): {out["success"]}/{out["runs"]}, arrived through {out["arrived_through"]} {out["arrived_through_parts"]}, '
            f'did not arrive {out["not_arrived"]}; by bin {out["by_bin"]} ({out["seconds"]:.0f}s)')
        return out

    ck, hist = TRN.train(scene, a.prepared / 'data', a.out / 'training', cfg, log, validate=validate)
    res['train_seconds'] = time.time() - t0
    res['final_loss'] = hist['final_loss']
    res['rollbacks'] = int(sum(hist['rollbacks']))
    res['refresh'] = hist['refresh']
    res['validate'] = hist['validate']
    log(f'trained: {res["train_seconds"] / 60:.1f} min, final ntrl loss {res["final_loss"]:.4e}, rollbacks {res["rollbacks"]}')
    net = P9.load_field(ck, 9, dev)
    fz = F.Frozen(net)
    g = torch.Generator(dev).manual_seed(0)
    pr = torch.cat([scene.uniform(4096, g), scene.uniform(4096, g)], 1)
    with torch.no_grad():
        res['fast_check'] = float((fz.travel(pr) - net.out(pr)[0][:, 0]).abs().max())
    label = f'{("ctl" if a.resume else "ntrl") if a.weight <= 0 else "mid"} w{a.weight:g} s{a.seed}'
    res.update(evaluate(scene, Q, net, a.out, log, label, plan_seeds=(0,) if a.smoke else PLAN_SEEDS))
    res['field'] = field_diag(scene, Q, net, a.prepared)
    f = res['field']
    log(f'{label}: solved {res["success_total"]}, arrived through {res["arrived_through_total"]} {res["arrived_through_parts"]}, did not arrive {res["not_arrived_total"]}; '
        f'time/T_ref {res["time_over_ref_median"]}; field T/T_ref {f["t_over_ref_median"]:.3f}, T/T_SE2 {f["t_over_se2_median"]:.3f}, below the lower bound {f["below_lower_bound"]:.3f}, '
        f'Spearman {f["spearman_ref"]:.3f}; speed fit {json.dumps(f["speed_fit"])}')
    res['elapsed_seconds'] = time.time() - t0
    res['complete'] = True
    dump(a.out / 'results.json', res)
    log(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
