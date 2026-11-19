"""One lift + midpoint field: ntrl-demo with the travel-time lift (no push) trained on a prepared scene, the travel-time midpoint loss
added after --start, then planned and diagnosed as jeffrey/lift_scale/pipeline.py does it.

    CUDA_VISIBLE_DEVICES=0 python jeffrey/lift_midpoint/pipeline.py --prepared <prepare run> --seed 87358 --lift time --out <dir>

--lift time|none     the travel-time lift (lift.py) or ntrl-demo's own input (the midpoint-only control)
--weight 0           the same loop without the midpoint term (the lift-alone / ntrl-demo control)
Midpoint settings default to jeffrey/midpoint_dp's MAIN run (safe traversals, min leaves, tree + sub-paths, weight 1 ramped over 500
epochs from epoch 2,000, leaves of 1 cell, 8,192 traversals every 100 epochs, 8,192 nodes per update, |log T - log y|).
Then maze_scale/evaluate.py (MPPI planning seeds 0/1/2, exact scoring) and lift_scale's diag.py (wall-layer fit by speed, travel times
against the truth, line of sight, detour index, where the runs that did not arrive stopped), and the same field measures at the saved
epochs. During training (diagnostics only) the current field plans the prepared queries with planning seed 0 at --validate-at epochs.
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
import trainer as TRN  # noqa: E402  (imports lift -> scenes, which put maze_scale first on sys.path)
import liftfast as LFT  # noqa: E402
import lift as LF  # noqa: E402
import scenes as SN  # noqa: E402
import diag as DG  # noqa: E402  (verbatim copy of jeffrey/lift_scale/diag.py)
import diagnostics as EDG  # noqa: E402  (embed_warp/diagnostics.py)

SAVE_AT = (1000, 2000, 2500, 3000, 3500, 4000, 4500)
VALIDATE_AT = (2000, 2500, 3000, 3500, 4000, 4500)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--lift', choices=['time', 'none'], default='time')
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
    p.add_argument('--driver', type=Path, default=None, help='control: traversals follow this frozen field (checkpoint of a registered run) instead of the network being trained')
    p.add_argument('--no-course', action='store_true')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from common import Logger, dump, by_bin
    import plan as P
    import evaluate as EV
    log = Logger(a.out / 'log.txt')
    dev = 'cuda:0'
    t0 = time.time()
    scene, info = SN.load_scene(a.prepared, dev)
    if a.smoke:
        a.epochs, a.start, a.refresh, a.traversals, a.ramp = 40, 20, 10, 512, 10
    if a.max_steps <= 0:
        a.max_steps = int(math.ceil(5 * scene.n * scene.pitch / P.settings(scene)['step']))
    save_at = (10, 20, 30) if a.smoke else SAVE_AT
    validate_at = (30,) if a.smoke else VALIDATE_AT
    cfg = dict(seed=a.seed, lift=a.lift, epochs=a.epochs, start=a.start, weight=a.weight, ramp=a.ramp, refresh=a.refresh, traversals=a.traversals, batch=a.batch,
               leaf=a.leaf, leaf_cells=a.leaf_cells, loss=a.loss, nodes=a.nodes, sub_factor=a.sub_factor, min_clear=a.min_clear, budget_factor=a.budget_factor,
               max_steps=a.max_steps, stall_window=a.stall_window, stall_drop_cells=a.stall_drop_cells, trim_hover=a.trim_hover, traverse=a.traverse,
               safe_margin=a.safe_margin, safe_mode=a.safe_mode, chunk=a.chunk, compile=not a.no_compile, max_rollbacks=a.max_rollbacks,
               save_at=list(save_at), validate_at=list(validate_at), planner=P.settings(scene))
    if a.driver is not None:
        cfg['driver'] = str(a.driver)
    res = dict(stage='lift-midpoint-field', scene=scene.describe(), scene_name=SN.scene_name(info), seed=a.seed, prepared=str(a.prepared), gpu=torch.cuda.get_device_name(0),
               smoke=a.smoke, config={k: v for k, v in cfg.items() if k != 'planner'}, planner=cfg['planner'], args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()})
    log(f'scene {res["scene_name"]} E={scene.E:g}: lift {a.lift}, midpoint weight {a.weight} from epoch {a.start}; config {json.dumps(res["config"])}')
    Q = EV.load_queries(a.prepared, dev)
    st = P.settings(scene)
    budget = torch.tensor(np.maximum(st['min_steps'], np.ceil(st['budget_factor'] * Q['path_clen'] / st['step'])), device=dev)

    def validate(network, epoch):
        cpu_state, cuda_state = torch.get_rng_state(), torch.cuda.get_rng_state()     # plan.mppi reseeds the global RNGs
        paths, stop, conv = P.mppi(P.field_cost(network), Q['start_t'], Q['goal_t'], 0, budget, scene.lo, scene.hi, **st)
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state)
        rows = P.score(scene, [paths[i] for i in range(len(Q['start_t']))], conv)
        ok = np.array([r['success'] for r in rows])
        cv = np.array([r['converged'] for r in rows])
        cl = np.array([r['collision'] for r in rows])
        out = dict(success=int(ok.sum()), runs=len(rows), arrived_wall=int((~ok & cv & cl).sum()), not_arrived=int((~cv).sum()), by_bin=by_bin(rows, Q['bin']))
        log(f'validate epoch {epoch} (planning seed 0): {out["success"]}/{len(rows)}, arrived through a wall {out["arrived_wall"]}, did not arrive {out["not_arrived"]}; by bin {out["by_bin"]}')
        return out

    ck, hist = TRN.train(scene, a.prepared / 'data', a.out / 'training', cfg, log, validate=validate)
    res['train_seconds'] = time.time() - t0
    res['final_loss'] = hist['final_loss']
    res['rollbacks'] = int(sum(hist['rollbacks']))
    res['refresh'] = hist['refresh']
    res['validate'] = hist['validate']
    log(f'trained: {res["train_seconds"] / 60:.1f} min, final ntrl loss {res["final_loss"]:.4e}, rollbacks {res["rollbacks"]}')

    net = LF.load_field(ck, scene, dev)
    res['fast_check'] = LFT.check(net, scene)
    log(f'evaluator check against NN.out: {res["fast_check"]}')
    cost, grad = P.field_cost(net), lambda pairs: P.field_grad(net, pairs)
    res.update(EV.evaluate(scene, a.prepared, cost, grad, a.out, log, label=f'{a.lift} w{a.weight:g} s{a.seed}'))
    res['speed_fit'] = EV.speed_fit(scene, a.prepared, grad)
    goals = torch.tensor(np.load(a.prepared / 'goals.npy'), device=dev)
    np.save(a.out / 'slice_goal0.npy', EV.slice_values(scene, cost, goals[0]).astype(np.float16))
    rows = np.load(a.out / 'rows.npz')
    res['failures'] = EDG.failures_by_arrival(rows)
    if info.get('variant') == 'plus':
        import plus as PL
        Qn = np.load(a.prepared / 'queries.npz')
        cells = PL.plus_cells(scene.n, info['plus_width'])

        def inside(x):
            c = np.clip(np.floor(x[:, 0] * scene.E + scene.n / 2).astype(int), 0, scene.n - 1)
            r = np.clip(np.floor(x[:, 1] * scene.E + scene.n / 2).astype(int), 0, scene.n - 1)
            return cells[c, r]
        where = inside(Qn['start']).astype(int) + inside(Qn['goal']).astype(int)
        res['by_where'] = {name: dict(success=int(rows['success'][:, where == k].sum()), runs=int(rows['success'][:, where == k].size)) for k, name in ((2, 'both'), (1, 'one'), (0, 'neither'))}
    truth = DG.Truth(scene, a.prepared)
    n_pairs = 2000 if a.smoke else 24000
    pairs, gi, t_star, D = DG.truth_pairs(scene, truth, n_pairs)
    blk = LF.segment_blocked(scene, pairs)
    res['wall_fit'] = DG.wall_fit(scene, a.prepared, grad)
    res['accuracy'] = DG.accuracy(scene, cost, pairs, t_star, D, blk)
    st_, table = DG.stalls(scene, truth, cost, grad, a.prepared, a.out, lambda pr: LF.segment_blocked(scene, pr))
    res['stalls'] = st_
    if table is not None:
        np.savez(a.out / 'stalls.npz', **table)
    det = torch.nonzero(t_star >= 1.3 * D).squeeze(1)[:1500]
    if len(det) > 20:
        res['descent_detour'] = DG.descent_trap(scene, cost, grad, pairs[det], t_star[det])
    res['bumps'] = EDG.bumps(scene, a.prepared, cost, P)
    acc = res['accuracy']
    log(f'wall fit innermost {res["wall_fit"]["innermost"]}; accuracy: within 10 % {acc["within10"]:.3f}, median T/T* {acc["ratio_median"]:.3f}, line of sight {acc.get("los")}, '
        f'detour {({k: v for k, v in acc.get("detour", {}).items() if k != "by_length"})}')
    log(f'detours by length {acc.get("detour", {}).get("by_length")}')
    log(f'failures {res["failures"]}, stalls {st_}, descent on detour pairs {res.get("descent_detour")}, bumps {res["bumps"]}')
    course = {}
    if not a.no_course and not a.smoke:
        for e in sorted(set(save_at) | {a.epochs}):
            f = a.out / 'training/checkpoints' / f'epoch_{e:05d}.pt'
            if not f.exists():
                continue
            n_e = LF.load_field(f, scene, dev)
            c_e, g_e = P.field_cost(n_e), (lambda pr, n_e=n_e: P.field_grad(n_e, pr))
            row = dict(wall_innermost=DG.wall_fit(scene, a.prepared, g_e, 10000)['innermost'])
            ac = DG.accuracy(scene, c_e, pairs[:8000], t_star[:8000], D[:8000])
            row.update(within10=ac['within10'], ratio_median=ac['ratio_median'], los_within10=ac.get('los', {}).get('within10'), los_ratio=ac.get('los', {}).get('ratio_median'),
                       rho=ac.get('detour', {}).get('rho_median'), seen_through=ac.get('detour', {}).get('seen_through'), t_max=ac['t_max'],
                       detour_by_length={k: v['rho_median'] for k, v in ac.get('detour', {}).get('by_length', {}).items()})
            course[str(e)] = row
            log(f'epoch {e}: {({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "detour_by_length"})}')
            del n_e
    res['course'] = course
    res['elapsed_seconds'] = time.time() - t0
    res['complete'] = True
    dump(a.out / 'results.json', res)
    log(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
