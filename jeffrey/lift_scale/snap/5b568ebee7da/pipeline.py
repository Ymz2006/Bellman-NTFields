"""One lift-scale field: ntrl-demo with (optionally) the travel-time lift and the largest-solution push, trained and planned on a
prepared scene (maze_scale maze / plus maze, embed_warp disks / apartment, lift_scale house).

    CUDA_VISIBLE_DEVICES=1 python jeffrey/lift_scale/pipeline.py --prepared <prepare run> --seed 87358 --lift time --push page --blocked <blocked run> --out <dir>

--lift none                 ntrl-demo unchanged (the control)
--lift time|clear|dead      the travel-time lift / the clearance coordinate of the same range / a dead extra input (lift.py)
--lift vec                  the vector lift h n (three extra inputs; the two faces of a wall get opposite vectors)
--gain g                    h multiplied by g ('cell': E / 22, the lifted band keeps the input length it has on a 22-cell scene)
--push none|page|const:v|e:v,e:v   the push schedule (trainer.py); needs --blocked (stage 'blocked' of the same prepare run)
--check dense|p48           which blocked flag drives the push: a sample every 0.1 cells, or the page's 48 points per segment
--bound b                   pairs are pushed while T < b box lengths ('auto': 1.7 x the largest true travel time of the prepare run)
--octaves 2,4,8             embed_warp's whitened octaves, always on (the local stand-in for 'finer position features')
Then maze_scale/evaluate.py (MPPI planning seeds 0/1/2, exact scoring) and diag.py (wall-layer fit by speed, travel times against the
truth, detour index, stalls), and the same field measures at the saved epochs (with planning seed 0 at a few of them).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lift as LF  # noqa: E402
import scenes as SN  # noqa: E402
import trainer as TR  # noqa: E402
import diag as DG  # noqa: E402
import diagnostics as EDG  # noqa: E402  (embed_warp/diagnostics.py)

SAVE_AT = (100, 300, 600, 1000, 1500, 2000, 3000, 4000)
PLAN_AT = (1000, 2000, 3000)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--blocked', type=Path, default=None)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=5000)
    p.add_argument('--lift', choices=['none', 'time', 'clear', 'dead', 'vec'], default='none')
    p.add_argument('--gain', default='1')
    p.add_argument('--depth', type=float, default=LF.OFFSET_CELLS)
    p.add_argument('--lift-freq', type=float, default=1.0, help='scale of the Fourier column of h')
    p.add_argument('--octaves', default='')
    p.add_argument('--rows', type=int, default=16)
    p.add_argument('--push', default='none')
    p.add_argument('--mu', type=float, default=0.1)
    p.add_argument('--mu1', type=float, default=0.0, help='weight of the linear (exact) one-sided penalty')
    p.add_argument('--bound', default='3')
    p.add_argument('--check', choices=['dense', 'p48'], default='dense')
    p.add_argument('--push-all', action='store_true', help='control: push every pair, not only the blocked ones')
    p.add_argument('--mu-abrupt', action='store_true', help='one-sided term at full weight from the first push epoch (the first runs; trips the epoch-repeat rule)')
    p.add_argument('--package-loss', action='store_true', help="use the package's Function.Loss object (equality check of the copied loss)")
    p.add_argument('--zero-heading', action='store_true', help='control: zero the Fourier column of the heading coordinate (the network sees heading only through h)')
    p.add_argument('--no-course', action='store_true')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from common import Logger, dump
    import plan as P
    import evaluate as EV
    log = Logger(a.out / 'log.txt')
    dev = 'cuda:0'
    t0 = time.time()
    scene, info = SN.load_scene(a.prepared, dev)
    epochs = 3 if a.smoke else a.epochs
    gain = scene.E / 22.0 if a.gain == 'cell' else float(a.gain)
    octaves = [float(v) for v in a.octaves.split(',') if v.strip()]
    cfg = dict(octaves=octaves, rows=a.rows, base_scale=1.0, whiten=True, gate=dict(kind='const' if octaves else 'none', mu_min=0.02, const=1.0),
               lift=dict(kind=a.lift, gain=gain, depth=a.depth, freq=a.lift_freq), zero_cols=[2] if a.zero_heading else [])
    schedule = TR.parse_schedule(a.push, epochs)
    blocked, bstats = None, None
    if schedule or a.blocked is not None:
        assert a.blocked is not None, '--push needs --blocked'
        bstats = json.load(open(a.blocked / 'results.json'))
        assert Path(bstats['prepared']).name == a.prepared.name, 'the blocked flags belong to another prepare run'
        blocked = np.load(a.blocked / ('blocked.npy' if a.check == 'dense' else 'blocked48.npy'))
    if a.bound == 'auto':
        assert bstats is not None
        bound = 1.7 * bstats['truth']['t_star_max']
    else:
        bound = float(a.bound)
    truth = DG.Truth(scene, a.prepared)
    res = dict(stage='lift-scale-field', scene=scene.describe(), scene_name=SN.scene_name(info), seed=a.seed, epochs=epochs, prepared=str(a.prepared), gpu=torch.cuda.get_device_name(0),
               smoke=a.smoke, config=cfg, push=dict(schedule=schedule, mu=a.mu, mu1=a.mu1, bound=bound, check=a.check, push_all=a.push_all),
               args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()})
    log(f'scene {res["scene_name"]} E={scene.E:g}, lift {a.lift} gain {gain:.3f}, octaves {octaves}, push {schedule} bound {bound:.2f} check {a.check}')
    save_at = () if a.smoke else SAVE_AT
    ck, hist = TR.train(a.prepared / 'data', a.out / 'training', a.seed, epochs, cfg, scene, log, blocked=blocked, schedule=schedule, mu=a.mu, bound=bound,
                        push_all=a.push_all, save_at=save_at, use_package_loss=a.package_loss, mu_abrupt=a.mu_abrupt, mu1=a.mu1)
    res['train_seconds'] = time.time() - t0
    res['final_loss'] = hist['final_loss']
    res['rollbacks'] = int(sum(hist['rollbacks']))
    log(f'trained: {res["train_seconds"] / 60:.1f} min, final ntrl loss {res["final_loss"]:.4e}, rollbacks {res["rollbacks"]}')

    net = LF.load_field(ck, scene, dev)
    cost, grad = P.field_cost(net), lambda pairs: P.field_grad(net, pairs)
    res.update(EV.evaluate(scene, a.prepared, cost, grad, a.out, log, label=f'{a.lift} s{a.seed}'))
    res['speed_fit'] = EV.speed_fit(scene, a.prepared, grad)
    goals = torch.tensor(np.load(a.prepared / 'goals.npy'), device=dev)
    np.save(a.out / 'slice_goal0.npy', EV.slice_values(scene, cost, goals[0]).astype(np.float16))
    rows = np.load(a.out / 'rows.npz')
    res['failures'] = EDG.failures_by_arrival(rows)
    if info.get('variant') == 'plus':
        import plus as PL
        Q = np.load(a.prepared / 'queries.npz')
        cells = PL.plus_cells(scene.n, info['plus_width'])

        def inside(x):
            c = np.clip(np.floor(x[:, 0] * scene.E + scene.n / 2).astype(int), 0, scene.n - 1)
            r = np.clip(np.floor(x[:, 1] * scene.E + scene.n / 2).astype(int), 0, scene.n - 1)
            return cells[c, r]
        where = inside(Q['start']).astype(int) + inside(Q['goal']).astype(int)
        res['by_where'] = {name: dict(success=int(rows['success'][:, where == k].sum()), runs=int(rows['success'][:, where == k].size)) for k, name in ((2, 'both'), (1, 'one'), (0, 'neither'))}
    # field against the truth
    n_pairs = 2000 if a.smoke else 24000
    pairs, gi, t_star, D = DG.truth_pairs(scene, truth, n_pairs)
    blk = LF.segment_blocked(scene, pairs)
    res['wall_fit'] = DG.wall_fit(scene, a.prepared, grad)
    res['nu'] = DG.nu_stats(scene, a.prepared)
    res['accuracy'] = DG.accuracy(scene, cost, pairs, t_star, D, blk)
    st, table = DG.stalls(scene, truth, cost, grad, a.prepared, a.out, lambda pr: LF.segment_blocked(scene, pr))
    res['stalls'] = st
    if table is not None:
        np.savez(a.out / 'stalls.npz', **table)
    det = torch.nonzero(t_star >= 1.3 * D).squeeze(1)[:1500]
    if len(det) > 20:
        res['descent_detour'] = DG.descent_trap(scene, cost, grad, pairs[det], t_star[det])
    res['bumps'] = EDG.bumps(scene, a.prepared, cost, P)
    log(f'wall fit innermost {res["wall_fit"]["innermost"]}, by speed {({k: round(v["slope"], 3) for k, v in res["wall_fit"]["by_speed"].items()})}')
    acc = res['accuracy']
    log(f'accuracy: within 10 % {acc["within10"]:.3f}, median T/T* {acc["ratio_median"]:.3f}, line of sight {acc.get("los")}, detour {({k: v for k, v in acc.get("detour", {}).items() if k != "by_length"})}')
    log(f'detours by length {acc.get("detour", {}).get("by_length")}')
    log(f'failures {res["failures"]}, stalls {st}, descent on detour pairs {res.get("descent_detour")}, bumps {res["bumps"]}')

    # the same field measures along the training (saved epochs), planning seed 0 at a few of them
    course = {}
    if not a.no_course and not a.smoke:
        for e in sorted(set(SAVE_AT) | {epochs}):
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
            if e in PLAN_AT:
                sub = a.out / f'course_{e:05d}'
                sub.mkdir(exist_ok=True)
                r = EV.evaluate(scene, a.prepared, c_e, g_e, sub, lambda m: None, label=f'epoch {e}', plan_seeds=[0])
                rr = np.load(sub / 'rows.npz')
                row.update(success=r['success_total'], success_by_bin=r['success_by_bin'], failures=EDG.failures_by_arrival(rr))
                (sub / 'paths_seed0.npy').unlink()
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
