"""One pair-lift field: ntrl-demo on the travel-time lift with a per-pair input coordinate (pairnet.py), trained (pairtrain.py) and
planned on a prepared scene, then jeffrey/lift_scale's diagnostics. jeffrey/lift_scale/pipeline.py with these additions:

--pair none|bit|count|len   no pair coordinate (lift_scale's network) | 1 if the straight segment is blocked | how many obstacles it enters |
                            the lifted-curve excess
--pair-sign anti|same       -psi / +psi at the two ends | psi at both
--pair-norm tanh|frac|log|raw, --pair-scale s (cells; 'box' = E), --pair-what excess|length|updown
--pair-place input|rows|pre|post, --pair-copies k, --pair-amp a, --pair-freq f, --pair-rows r
--pair-inside seg|zero      'zero': psi = 0 when either end is inside an obstacle (kind bit)
--pair-nograd               do not put the feature's derivative in the graph (kind len)
--tri w, --tri1 w, --tri-r r   the triangle cap over random free steps of radius r box lengths (pairtrain.py)
--bell w, --bell-r r, --bell-k K, --bell-from e   the free-step Bellman backup (pairtrain.bellman_term)
--td-free, --td-weight w    ntrl-demo's TD term only over steps that stay out of obstacles; its weight (ntrl-demo: 1e-3)
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
import pairnet as PN  # noqa: E402
import pairtrain as PT  # noqa: E402
LF = PN.LF
import scenes as SN  # noqa: E402
TR = PT.LT
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
    p.add_argument('--b-scale', default='1', help="factor on ntrl-demo's Fourier matrix and on the lift column ('cell': E / 22.1)")
    p.add_argument('--octaves', default='')
    p.add_argument('--rows', type=int, default=16)
    p.add_argument('--push', default='none')
    p.add_argument('--mu', type=float, default=0.1)
    p.add_argument('--mu1', type=float, default=0.0, help='weight of the linear (exact) one-sided penalty')
    p.add_argument('--lam-scale', default='1', help="factor on the push schedule ('cell': E / 22.1, the push per cell of travel time a 22-cell scene gets)")
    p.add_argument('--bound', default='3')
    p.add_argument('--check', choices=['dense', 'p48'], default='dense')
    p.add_argument('--push-all', action='store_true', help='control: push every pair, not only the blocked ones')
    p.add_argument('--zero-heading', action='store_true', help='control: zero the Fourier column of the heading coordinate (the network sees heading only through h)')
    p.add_argument('--no-course', action='store_true')
    p.add_argument('--pair', choices=['none', 'bit', 'count', 'len'], default='none')
    p.add_argument('--pair-sign', choices=['anti', 'same'], default='anti')
    p.add_argument('--pair-norm', choices=['tanh', 'frac', 'log', 'raw'], default='tanh')
    p.add_argument('--pair-scale', default='2')
    p.add_argument('--pair-what', choices=['excess', 'length', 'updown'], default='excess')
    p.add_argument('--pair-place', choices=['input', 'rows', 'pre', 'post'], default='input')
    p.add_argument('--pair-copies', type=int, default=16)
    p.add_argument('--pair-amp', type=float, default=1.0)
    p.add_argument('--pair-freq', type=float, default=1.0)
    p.add_argument('--pair-rows', type=int, default=16)
    p.add_argument('--pair-inside', choices=['seg', 'zero'], default='seg')
    p.add_argument('--pair-nograd', action='store_true')
    p.add_argument('--tri', type=float, default=0.0)
    p.add_argument('--tri1', type=float, default=0.0)
    p.add_argument('--tri-r', type=float, default=0.03)
    p.add_argument('--bell', type=float, default=0.0, help='weight of the free-step Bellman backup')
    p.add_argument('--bell-r', default='0.03', help="radius of the backup's steps in box lengths ('cell': 0.03 x 22.1 / E, the 0.66 cells it is on the 22-cell maze; 'cell:x': x cells)")
    p.add_argument('--bell-k', type=int, default=8)
    p.add_argument('--bell-norm', action='store_true', help='weight times (0.03 / radius)^2: the backup residual is measured in units of the step, so the term keeps the strength it has at radius 0.03 box lengths')
    p.add_argument('--bell-from', type=int, default=0)
    p.add_argument('--bell-cone', type=int, default=0, help='candidates around the descent direction (of the K)')
    p.add_argument('--bell-trapz', action='store_true', help='step time by the 6-point trapezoid rule (the first Bellman runs; it overestimates steps that start deep in a speed band)')
    p.add_argument('--bell-huber', type=float, default=0.0)
    p.add_argument('--bell-long', default='1', help="candidate steps up to this many radii long (log-uniform lengths); 'box:f' = up to f box lengths")
    p.add_argument('--bell-keep', action='store_true', help='with --bell-long: the candidates around the descent direction keep their short lengths')
    p.add_argument('--bell-tau', choices=['true', 'learned', 'field'], default='true', help="cost of a candidate step: 'true' = the true speed along it (its lifted length), 'learned' = the field's own predicted speed 1 / |grad T| integrated along it, 'field' = the speed predicted by the frozen field of --speed-field")
    p.add_argument('--speed-field', type=Path, default=None, help='--bell-tau field: a finished field run (run.py --field); its predicted speed is tabulated (pairnet.SpeedTable)')
    p.add_argument('--speed-goals', type=int, default=4, help='--bell-tau field: partners per table node')
    p.add_argument('--init-field', type=Path, default=None, help='warm start: a finished field run of the same network and seed (run.py --init)')
    p.add_argument('--bell-ls', type=int, default=4, help='--bell-tau learned: stratified random samples per step')
    p.add_argument('--bell-nofree', action='store_true', help='no collision check of the candidate steps (a step through an obstacle costs its integral)')
    p.add_argument('--bell-free', action='store_true', help='obstacle-space sampling: the backup only for pairs with both ends out of obstacles')
    p.add_argument('--in-weight', type=float, default=1.0, help="weight of ntrl-demo's loss for the pairs with an end inside an obstacle (1 = ntrl-demo)")
    p.add_argument('--bell-table', action='store_true', help="clearance along the backup's steps from pairnet.ClearTable instead of the exact rectangle clearance")
    p.add_argument('--td-free', action='store_true', help="drop ntrl-demo's TD term for pairs whose step crosses an obstacle")
    p.add_argument('--td-weight', type=float, default=1e-3)
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
    octaves = [round(k * scene.E / 22.1, 2) for k in (2, 4, 8)] if a.octaves == 'cell' else [float(v) for v in a.octaves.split(',') if v.strip()]
    b_scale = scene.E / 22.1 if a.b_scale == 'cell' else float(a.b_scale)
    cfg = dict(octaves=octaves, rows=a.rows, base_scale=1.0, whiten=True, gate=dict(kind='const' if octaves else 'none', mu_min=0.02, const=1.0),
               lift=dict(kind=a.lift, gain=gain, depth=a.depth, freq=a.lift_freq * b_scale), zero_cols=[2] if a.zero_heading else [],
               pair=dict(kind=a.pair, sign=a.pair_sign, norm=a.pair_norm, scale=scene.E if a.pair_scale == 'box' else float(a.pair_scale), what=a.pair_what,
                         place=a.pair_place, copies=a.pair_copies, amp=a.pair_amp, freq=a.pair_freq, rows=a.pair_rows, inside=a.pair_inside, grad=not a.pair_nograd))
    schedule = TR.parse_schedule(a.push, epochs)
    lam_scale = scene.E / 22.1 if a.lam_scale == 'cell' else float(a.lam_scale)
    schedule = [(e, v * lam_scale) for e, v in schedule]
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
    bell_r = 0.03 * 22.1 / scene.E if a.bell_r == 'cell' else float(a.bell_r.split(':')[1]) / scene.E if a.bell_r.startswith('cell:') else float(a.bell_r)
    bell_w = a.bell * (0.03 / bell_r) ** 2 if a.bell_norm else a.bell
    bell_long = float(a.bell_long.split(':')[1]) / bell_r if a.bell_long.startswith('box:') else float(a.bell_long)
    truth = DG.Truth(scene, a.prepared)
    res = dict(stage='pair-lift-field', scene=scene.describe(), scene_name=SN.scene_name(info), seed=a.seed, epochs=epochs, prepared=str(a.prepared), gpu=torch.cuda.get_device_name(0),
               smoke=a.smoke, config=cfg, push=dict(schedule=schedule, mu=a.mu, mu1=a.mu1, bound=bound, check=a.check, push_all=a.push_all), tri=dict(tri=a.tri, tri1=a.tri1, radius=a.tri_r), bellman=dict(weight=bell_w, weight_argument=a.bell, normalized=a.bell_norm, radius=bell_r, radius_cells=bell_r * scene.E, k=a.bell_k, cone=a.bell_cone, table=a.bell_table, td_free=a.td_free, long=bell_long, long_cells=bell_long * bell_r * scene.E, keep_cone=a.bell_keep, free_pairs=a.bell_free, in_weight=a.in_weight, tau=a.bell_tau, tau_samples=a.bell_ls, collision_check=not a.bell_nofree),
               args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()})
    res['b_scale'] = b_scale
    log(f'scene {res["scene_name"]} E={scene.E:g}, lift {a.lift} gain {gain:.3f}, pair {cfg["pair"]}, octaves {octaves}, B x {b_scale:.3f}, push {schedule} bound {bound:.2f} check {a.check}')
    save_at = () if a.smoke else tuple(sorted(set(SAVE_AT) | set(range(5000, epochs, 1000))))
    bell_speed = None
    if a.bell_tau == 'field':
        assert a.speed_field is not None, '--bell-tau field needs --speed-field (run.py --field)'
        ts = time.time()
        fnet = PN.load_field(a.speed_field / 'training/checkpoint.pt', scene)
        bell_speed = PN.SpeedTable(fnet, scene, 32, 49, goals=a.speed_goals, seed=a.seed)
        del fnet
        torch.cuda.empty_cache()
        res['speed_field'] = dict(run=a.speed_field.name, goals=a.speed_goals, nodes=[bell_speed.nx, bell_speed.nx, bell_speed.nt], seconds=time.time() - ts)
        log(f'speed table of {a.speed_field.name}: {res["speed_field"]}')
    ck, hist = PT.train(a.prepared / 'data', a.out / 'training', a.seed, epochs, cfg, scene, log, blocked=blocked, schedule=schedule, mu=a.mu, bound=bound,
                        push_all=a.push_all, save_at=save_at, mu1=a.mu1, b_scale=b_scale, tri=a.tri, tri1=a.tri1, tri_r=a.tri_r,
                        bell=bell_w, bell_r=bell_r, bell_k=a.bell_k, bell_from=a.bell_from, td_free=a.td_free, td_weight=a.td_weight,
                        bell_cone=a.bell_cone, bell_exact=not a.bell_trapz, bell_huber=a.bell_huber, bell_table=a.bell_table, bell_long=bell_long, bell_keep=a.bell_keep, bell_free=a.bell_free, in_weight=a.in_weight, bell_tau=a.bell_tau, bell_ls=a.bell_ls, bell_nofree=a.bell_nofree, bell_speed=bell_speed, init=(a.init_field / 'training/checkpoint.pt') if a.init_field is not None else None)
    res['tri_final'] = dict(sq=hist['tri_sq'][-1], lin=hist['tri_lin'][-1], max=hist['tri_max'][-1], used=hist['tri_used'][-1]) if hist['tri_sq'] else None
    res['t_blocked_free'] = [hist['t_blocked'][-1], hist['t_free'][-1]] if hist['t_blocked'] else None
    res['train_seconds'] = time.time() - t0
    res['final_loss'] = hist['final_loss']
    res['rollbacks'] = int(sum(hist['rollbacks']))
    log(f'trained: {res["train_seconds"] / 60:.1f} min, final ntrl loss {res["final_loss"]:.4e}, rollbacks {res["rollbacks"]}')

    net = PN.load_field(ck, scene, dev)
    cost, grad = P.field_cost(net), lambda pairs: P.field_grad(net, pairs)
    res.update(EV.evaluate(scene, a.prepared, cost, grad, a.out, log, label=f'{a.lift}+{a.pair} s{a.seed}'))
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
        for e in sorted(set(save_at) | {epochs}):
            f = a.out / 'training/checkpoints' / f'epoch_{e:05d}.pt'
            if not f.exists():
                continue
            n_e = PN.load_field(f, scene, dev)
            c_e, g_e = P.field_cost(n_e), (lambda pr, n_e=n_e: P.field_grad(n_e, pr))
            row = dict(wall_innermost=DG.wall_fit(scene, a.prepared, g_e, 10000)['innermost'])
            ac = DG.accuracy(scene, c_e, pairs[:8000], t_star[:8000], D[:8000])
            row.update(within10=ac['within10'], ratio_median=ac['ratio_median'], los_within10=ac.get('los', {}).get('within10'), los_ratio=ac.get('los', {}).get('ratio_median'),
                       rho=ac.get('detour', {}).get('rho_median'), seen_through=ac.get('detour', {}).get('seen_through'), t_max=ac['t_max'],
                       detour_by_length={k: v['rho_median'] for k, v in ac.get('detour', {}).get('by_length', {}).items()})
            if e in PLAN_AT or (e >= 5000 and e % 2500 == 0 and e < epochs):
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
