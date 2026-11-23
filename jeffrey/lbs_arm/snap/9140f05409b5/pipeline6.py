"""One LBS field for the cabinet arm (6 joints): ntrl-demo with any of the travel-time Lift (onto 7 dimensions: 6 joints + h), the
free-step Bellman backup (true step cost) and the travel-time midpoint (Segment) loss, trained on a prepared scene (prepare.py), with
the frozen queries planned during and after training by ntrl-demo's MPPI and scored by the release scorer's rule (core.score; the
official numbers come from evaluate6.py, the unchanged cabinet evaluator).

    CUDA_VISIBLE_DEVICES=1 python jeffrey/lbs_arm/pipeline6.py --prepared <prepare run> --seed 87358 --lift time --bell 1 --out <dir>

--lift time|none     L: the travel-time lift or ntrl-demo's own input
--lift-gain g        h = g G(d) / E. g = 1 is the lift of the maze and house tests (its length along the clearance gradient is the travel
                     time for the scene's median gradient); the range of h is g G_max / E in units where every joint has range 1
--freq f [...]       factor on ntrl-demo's Fourier matrix of the joints: one number, six numbers (one per joint), or 'sens' (each joint
                     in proportion to its clearance sensitivity, root mean square of d clearance / d joint, mean 1)
--bell w             B: the Bellman backup with weight w (0 = off), radius --bell-r-cells, on --bell-n pairs of each minibatch
--weight w           S: the midpoint loss (0 = off) from epoch --start
Arms: ntrl-demo (--lift none --weight 0), LBS (--lift time --bell 1), and the three ablations as in jeffrey/lbs.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arm as A  # noqa: E402
import core as C  # noqa: E402
import trainer6 as TRN  # noqa: E402
from common import Logger, dump  # noqa: E402  (jeffrey/maze_scale/common.py)


def summarize(rows, Q):
    ok = np.array([r['success'] for r in rows])
    cv = np.array([r['converged'] for r in rows])
    names = [str(t) for t in Q['task_names']]
    out = dict(success=int(ok.sum()), runs=len(rows), arrived_contact=int((~ok & cv).sum()), not_arrived=int((~cv).sum()), not_arrived_solved=int((ok & ~cv).sum()),
               contact_self=int(sum(1 for r in rows if r.get('contact') == 'self')), contact_obstacle=int(sum(1 for r in rows if r.get('contact') == 'obstacle')),
               by_task={n: int(ok[Q['task'] == k].sum()) for k, n in enumerate(names)},
               line_feasible=[int(ok[Q['line']].sum()), int(Q['line'].sum())], detour=[int(ok[~Q['line']].sum()), int((~Q['line']).sum())],
               by_ends_in_cupboard={str(k): [int(ok[Q['ends_in_cupboard'] == k].sum()), int((Q['ends_in_cupboard'] == k).sum())] for k in (0, 1, 2)})
    return out, ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--lift', choices=['time', 'none'], default='time')
    p.add_argument('--lift-gain', type=float, default=1.0)
    p.add_argument('--freq', nargs='+', default=['1'])
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
    p.add_argument('--pair-max', type=float, default=0.6)
    p.add_argument('--budget-factor', type=float, default=2.0)
    p.add_argument('--min-steps', type=int, default=200)
    p.add_argument('--max-steps', type=int, default=600)
    p.add_argument('--stall-window', type=int, default=60)
    p.add_argument('--stall-drop-cells', type=float, default=0.05)
    p.add_argument('--trim-hover', type=float, default=0.25)
    p.add_argument('--traverse', choices=['plain', 'safe'], default='safe')
    p.add_argument('--safe-margin', type=float, default=0.02)
    p.add_argument('--chunk', type=int, default=1 << 19)
    p.add_argument('--no-compile', action='store_true')
    p.add_argument('--max-rollbacks', type=int, default=20)
    p.add_argument('--bell', type=float, default=0.0)
    p.add_argument('--bell-r-cells', type=float, default=0.663)
    p.add_argument('--bell-k', type=int, default=16)
    p.add_argument('--bell-cone', type=int, default=6)
    p.add_argument('--bell-long', type=float, default=8.0)
    p.add_argument('--bell-n', type=int, default=512)
    p.add_argument('--td-all', action='store_true')
    p.add_argument('--validate-every', type=int, default=500)
    p.add_argument('--plan-steps', type=int, default=600)
    p.add_argument('--plan-samples', type=int, default=50)
    p.add_argument('--plan-horizon', type=int, default=5)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    log = Logger(a.out / 'log.txt')
    dev = 'cuda:0'
    t0 = time.time()
    scene, info = A.load_scene(a.prepared, dev)
    if a.smoke:
        a.epochs, a.start, a.refresh, a.traversals, a.ramp, a.plan_steps = 40, 20, 10, 512, 10, 100
    if a.freq == ['sens']:
        s = np.array(info['measure']['per_joint_rms'])
        freq = (s / s.mean()).tolist()
    else:
        freq = [float(v) for v in a.freq] * (A.D if len(a.freq) == 1 else 1)
    assert len(freq) == A.D, freq
    validate_at = (20, 40) if a.smoke else tuple(range(a.validate_every, a.epochs + 1, a.validate_every))
    planner = C.planner_settings(scene, sample_num=a.plan_samples, horizon=a.plan_horizon)
    bell_r = a.bell_r_cells / scene.E
    cfg = dict(seed=a.seed, lift=a.lift, lift_gain=a.lift_gain, lift_depth=0.0, freq=freq, epochs=a.epochs, start=a.start, weight=a.weight, ramp=a.ramp, refresh=a.refresh,
               traversals=a.traversals, batch=a.batch, leaf=a.leaf, leaf_cells=a.leaf_cells, loss=a.loss, nodes=a.nodes, sub_factor=a.sub_factor, min_clear=a.min_clear,
               pair_max=a.pair_max, budget_factor=a.budget_factor, min_steps=a.min_steps, max_steps=a.max_steps, stall_window=a.stall_window,
               stall_drop_cells=a.stall_drop_cells, trim_hover=a.trim_hover, traverse=a.traverse, safe_margin=a.safe_margin, chunk=a.chunk, compile=not a.no_compile,
               max_rollbacks=a.max_rollbacks, save_at=list(validate_at), validate_at=list(validate_at), planner=planner,
               bell=a.bell * (0.03 / bell_r) ** 2, bell_arg=a.bell, bell_r=bell_r, bell_k=a.bell_k, bell_cone=a.bell_cone, bell_long=a.bell_long, bell_n=a.bell_n, td_free=not a.td_all)
    prof = C.Profile(0.0, device=dev)
    res = dict(stage='lbsarm-field', scene=scene.describe(), seed=a.seed, prepared=str(a.prepared), gpu=torch.cuda.get_device_name(0), smoke=a.smoke,
               config={k: v for k, v in cfg.items() if k != 'planner'}, planner=dict(planner, steps=a.plan_steps),
               args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
               arm=dict(L=a.lift == 'time', B=a.bell > 0, S=a.weight > 0),
               lift=dict(gain=a.lift_gain, G_max_cells=prof.G_max, h_range=(a.lift_gain * prof.G_max / scene.E) if a.lift == 'time' else 0.0, input_dims=A.D + (a.lift == 'time')))
    log(f'cabinet arm E={scene.E:g}: lift {a.lift} gain {a.lift_gain} (h range {res["lift"]["h_range"]:.4f}), Fourier factors {freq}, Bellman weight {a.bell}, midpoint weight {a.weight} from epoch {a.start}; '
        f'planner {res["planner"]}')
    Q = dict(np.load(a.prepared / 'queries.npz'))
    qs, qg = torch.tensor(Q['start'], device=dev), torch.tensor(Q['goal'], device=dev)
    if a.smoke:
        keep = np.arange(0, len(qs), 10)
        Q = {k: (v[keep] if getattr(v, 'shape', (0,))[:1] == (len(qs),) else v) for k, v in Q.items()}
        qs, qg = qs[keep], qg[keep]

    def run_plan(ev, seed):
        paths, conv = C.plan(ev, qs, qg, seed, a.plan_steps, **planner)
        rows = C.score(scene, paths, conv.cpu().numpy())
        return paths, rows

    def validate(ev, epoch):
        _, rows = run_plan(ev, 0)
        out, _ = summarize(rows, Q)
        log(f'validate epoch {epoch} (planning seed 0): {out["success"]}/{out["runs"]}; by task {out["by_task"]}; detour queries {out["detour"]}; '
            f'arrived in contact {out["arrived_contact"]}, did not arrive {out["not_arrived"]}')
        return out

    ck, hist = TRN.train(scene, a.prepared / 'data', a.out / 'training', cfg, log, validate=validate)
    res['train_seconds'] = time.time() - t0
    res['final_loss'] = hist['final_loss']
    res['rollbacks'] = int(sum(hist['rollbacks']))
    res['refresh'] = hist['refresh']
    res['validate'] = hist['validate']
    res['train_only_seconds'] = hist['seconds'] - hist['validate_seconds']
    log(f'trained: {res["train_seconds"] / 60:.1f} min, final ntrl loss {res["final_loss"]:.4e}, rollbacks {res["rollbacks"]}')

    mn = TRN.ntrl_modules()[2]
    net = C.load_field(mn, ck, scene, dev)
    clear = C.Clearance(scene, compile=not a.no_compile)
    ev = C.Compiled(net, scene, clear, chunk=a.chunk, compile=not a.no_compile)
    # the compiled evaluator against the network itself
    gen = torch.Generator(dev).manual_seed(5)
    xa, xb = scene.free(2048, gen, 0.0), scene.free(2048, gen, 0.0)
    with torch.no_grad():
        ref = net.out(torch.cat([xa, xb], 1))[0][:, 0]
    res['evaluator_max_abs_error'] = float((ev.travel(torch.cat([xa, xb], 1)) - ref).abs().max())
    res['field'] = dict(T_test_queries_median=float(ev.travel(torch.cat([qs, qg], 1)).median()), T_random_pairs_median=float(ref.median()),
                        distance_test_queries_median=float(torch.linalg.vector_norm(qs - qg, dim=1).median()))
    res['plan'] = {}
    total = np.zeros(len(qs), int)
    for seed in ([0] if a.smoke else [0, 1, 2]):
        paths, rows = run_plan(ev, seed)
        out, ok = summarize(rows, Q)
        total += ok
        res['plan'][str(seed)] = out
        arr = np.empty(len(paths), dtype=object)
        arr[:] = [(pth * scene.q_range).cpu().numpy().astype(np.float32) for pth in paths]
        np.save(a.out / f'paths_seed{seed}.npy', arr, allow_pickle=True)
        log(f'planning seed {seed}: {out["success"]}/{out["runs"]}; by task {out["by_task"]}; detour {out["detour"]}; line-feasible {out["line_feasible"]}')
    res['success'] = [int(total.sum()), int(len(qs) * len(res['plan']))]
    res['success_by_task'] = {n: int(sum(res['plan'][s]['by_task'][n] for s in res['plan'])) for n in [str(t) for t in Q['task_names']]}
    res['detour'] = [int(sum(res['plan'][s]['detour'][0] for s in res['plan'])), int(sum(res['plan'][s]['detour'][1] for s in res['plan']))]
    res['elapsed_seconds'] = time.time() - t0
    dump(a.out / 'results.json', res)
    log(f'done: {res["success"][0]}/{res["success"][1]} (this scene\'s scorer), detour queries {res["detour"]}, {res["elapsed_seconds"] / 60:.1f} min')


if __name__ == '__main__':
    main()
