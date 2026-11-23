"""Why do plans stop short of the goal? Analysis of a registered field run (no training; reads its checkpoint).

    CUDA_VISIBLE_DEVICES=1 python stalls.py results/registered_<field> [--only planner|ends|witness]

1. planner: the 600 frozen queries (planning seed 0) with the run's planner and with one setting changed at a time (steps, samples,
   horizon, temperature, step, rollouts in contact rejected), scored by core.score.
2. ends: where the default plans that do not arrive stop: remaining distance, clearance, and whether the stop is a local minimum of
   T(., goal) (share of random moves of 1 / 3 / 8 planner steps that lower T; all moves and moves that stay out of contact).
3. witness: the suites' certified witness paths. T(w(t), goal) along each: largest rise as a share of T(start, goal); T(start, goal)
   against the travel time along the witness (an upper bound of the truth) and against the straight distance (a lower bound).
Writes report/stalls_<run>.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import arm as A  # noqa: E402
import core as C  # noqa: E402
import trainer6 as TRN  # noqa: E402


def summary(rows, Q):
    ok = np.array([r['success'] for r in rows])
    cv = np.array([r['converged'] for r in rows])
    return dict(success=int(ok.sum()), arrived_contact=int((~ok & cv).sum()), not_arrived=int((~cv).sum()), detour=int(ok[~Q['line']].sum()),
                by_task=[int(ok[Q['task'] == k].sum()) for k in range(len(Q['task_names']))]), ok, cv


@torch.no_grad()
def run(ev, scene, qs, qg, seed, steps, **kw):
    gen = torch.Generator(qs.device).manual_seed(int(seed))
    budget = torch.full((len(qs),), int(steps), device=qs.device)
    paths, stop, conv, _ = C.mppi(ev, qs, qg, gen, budget, scene.lo, scene.hi, **kw)
    plist = [torch.cat([paths[i, :int(stop[i]) + 1], qg[i:i + 1]]) for i in range(len(qs))]
    return plist, conv, stop


def main():
    p = argparse.ArgumentParser()
    p.add_argument('field', type=Path)
    p.add_argument('--only', default='')
    p.add_argument('--tag', default='')
    a = p.parse_args()
    dev = 'cuda:0'
    fr = json.loads((a.field / 'results.json').read_text())
    prepared = Path(fr['prepared'])
    scene, info = A.load_scene(prepared, dev)
    mn = TRN.ntrl_modules()[2]
    net = C.load_field(mn, a.field / 'training/checkpoint.pt', scene, dev)
    ev = C.Compiled(net, scene, C.Clearance(scene, compile=False), compile=False)
    Q = dict(np.load(prepared / 'queries.npz'))
    qs, qg = torch.tensor(Q['start'], device=dev), torch.tensor(Q['goal'], device=dev)
    pl = {k: v for k, v in fr['planner'].items() if k != 'steps'}
    steps = fr['planner']['steps']
    outp = HERE / 'report' / f'stalls_{a.field.name[-8:]}{a.tag}.json'
    out = json.loads(outp.read_text()) if outp.exists() else {}
    out.update(field=a.field.name, arm=fr['arm'], planner=fr['planner'])
    t0 = time.time()

    paths, conv, stop = run(ev, scene, qs, qg, 0, steps, **pl)
    rows = C.score(scene, paths, conv.cpu().numpy())
    base, ok, cv = summary(rows, Q)
    print('default', base, f'{time.time() - t0:.0f}s', flush=True)

    if a.only in ('', 'planner'):
        res = dict(default=base)
        variants = {
            'steps x3': (steps * 3, {}),
            'steps x3, step x2': (steps * 3, dict(step=pl['step'] * 2)),
            'samples 400': (steps, dict(sample_num=400)),
            'samples 400, steps x3': (steps * 3, dict(sample_num=400)),
            'horizon 10': (steps, dict(horizon=10)),
            'temperature / 4': (steps, dict(temperature=pl['temperature'] / 4)),
            'temperature x 4': (steps, dict(temperature=pl['temperature'] * 4)),
            'no contact rollouts': (steps, dict(safe=True, safe_margin=0.02)),
            'no contact rollouts, steps x3': (steps * 3, dict(safe=True, safe_margin=0.02)),
            'no contact rollouts, samples 400, steps x3': (steps * 3, dict(safe=True, safe_margin=0.02, sample_num=400)),
        }
        for name, (st, kw) in variants.items():
            t1 = time.time()
            pp, cc, _ = run(ev, scene, qs, qg, 0, st, **dict(pl, **kw))
            res[name], _, _ = summary(C.score(scene, pp, cc.cpu().numpy()), Q)
            print(name, res[name], f'{time.time() - t1:.0f}s', flush=True)
            out['planner_variants'] = res
            outp.write_text(json.dumps(out, indent=1))

    if a.only in ('', 'ends'):
        na = np.nonzero(~cv)[0]
        end = torch.stack([paths[i][-2] for i in na])
        g, s = qg[na], qs[na]
        gf = ev.encode(g)
        T_end, T_start = ev.to_goal(end, gf), ev.to_goal(s, gf)
        d_end = ev.clearance(end)
        left = torch.linalg.vector_norm(end - g, dim=1) / torch.linalg.vector_norm(s - g, dim=1)
        gen = torch.Generator(dev).manual_seed(1)
        loc = {}
        for mult in (1, 3, 8):
            r = pl['step'] * mult
            u = torch.randn(len(na), 2048, A.D, device=dev, generator=gen)
            u = u / torch.linalg.vector_norm(u, dim=2, keepdim=True)
            x = scene.clamp((end[:, None, :] + r * u).reshape(-1, A.D))
            Tn = ev.to_goal(x, gf[:, None, :].expand(-1, 2048, -1).reshape(-1, gf.shape[1])).reshape(len(na), 2048)
            dn = ev.clearance(x).reshape(len(na), 2048)
            lower = Tn < T_end[:, None]
            free = dn > scene.contact
            loc[str(mult)] = dict(lower_share_median=float(lower.float().mean(1).median()), none_lower=int((~lower.any(1)).sum()),
                                  none_lower_free=int((~(lower & free).any(1)).sum()),
                                  best_drop_over_T_median=float(((T_end[:, None] - Tn).amax(1) / T_end).median()),
                                  best_free_drop_over_T_median=float(((T_end[:, None] - Tn.masked_fill(~free, float('inf'))).amax(1) / T_end).median()))
        stopped = np.array([int(stop[i]) for i in na])
        moved = torch.stack([torch.linalg.vector_norm(paths[i][-2] - paths[i][max(len(paths[i]) - 102, 0)]) for i in na]) / (pl['step'] * 100)
        out['ends'] = dict(count=int(len(na)), remaining_over_line=np.quantile(left.cpu().numpy(), [.1, .25, .5, .75, .9]).tolist(),
                           clearance_cells=np.quantile(d_end.cpu().numpy(), [.1, .25, .5, .75, .9]).tolist(), in_contact=int((d_end <= scene.contact).sum()),
                           within_margin=int((d_end < C.MARGIN_CELLS).sum()), T_end_over_T_start=np.quantile((T_end / T_start).cpu().numpy(), [.1, .25, .5, .75, .9]).tolist(),
                           used_all_steps=int((stopped >= steps).sum()), last_100_steps_net_move_over_straight=np.quantile(moved.cpu().numpy(), [.1, .5, .9]).tolist(),
                           local=loc, by_task=[int((Q['task'][na] == k).sum()) for k in range(len(Q['task_names']))])
        print('ends', json.dumps(out['ends']), flush=True)
        outp.write_text(json.dumps(out, indent=1))

    if a.only in ('', 'witness'):
        qr = scene.q_range.cpu().numpy()
        names = [str(t) for t in Q['task_names']]
        rise, t_net, t_wit, line, wlen, okw = [], [], [], [], [], []
        for k, n in enumerate(names):
            z = np.load(ROOT / 'experiments/cabinet/tests/suites' / f'cabinet_{n}_100.npz', allow_pickle=True)
            idx = np.nonzero(Q['task'] == k)[0]
            assert np.abs(z['starts_rad'] / qr - Q['start'][idx]).max() < 1e-5
            for j, i in enumerate(idx):
                w = torch.tensor(np.asarray(z['witness_trajectories'][j], dtype=np.float64) / qr, device=dev, dtype=torch.float32)
                dense = C.densify(w, torch.full((A.D,), 0.001, device=dev))
                d = ev.clearance(dense)
                gf = ev.encode(qg[i:i + 1]).expand(len(dense), -1)
                T = ev.to_goal(dense, gf, d=d if ev.lift else None)
                seg = torch.linalg.vector_norm(dense[1:] - dense[:-1], dim=1)
                sp = C.train_speed(0.5 * (d[1:] + d[:-1]))
                run_min = torch.cummin(T, 0).values
                rise.append(float((T - run_min).max() / T[0]))
                t_net.append(float(T[0]))
                t_wit.append(float((seg / sp).sum()))
                wlen.append(float(seg.sum()))
                line.append(float(torch.linalg.vector_norm(w[0] - w[-1])))
                okw.append(bool((d > scene.contact - C.OFFSET_CELLS * 0 - 1e-4).all()))
        rise, t_net, t_wit, line, wlen = map(np.array, (rise, t_net, t_wit, line, wlen))
        order = np.concatenate([np.nonzero(Q['task'] == k)[0] for k in range(len(names))])
        inv = np.empty(len(order), dtype=int)
        inv[order] = np.arange(len(order))
        rise, t_net, t_wit, line, wlen = (v[inv] for v in (rise, t_net, t_wit, line, wlen))
        groups = {'line feasible': Q['line'], 'detour, solved': ~Q['line'] & ok, 'detour, arrived in contact': ~Q['line'] & ~ok & cv, 'detour, did not arrive': ~Q['line'] & ~cv}
        for k, n in enumerate(names):
            groups['task ' + n] = Q['task'] == k
        wt = {}
        for name, m in groups.items():
            if not m.sum():
                continue
            wt[name] = dict(count=int(m.sum()), witness_over_line_median=float(np.median(wlen[m] / line[m])), T_witness_over_line_median=float(np.median(t_wit[m] / line[m])),
                            T_net_over_line_median=float(np.median(t_net[m] / line[m])), T_net_over_T_witness=np.quantile(t_net[m] / t_wit[m], [.1, .25, .5, .75, .9]).tolist(),
                            rise_median=float(np.median(rise[m])), rise_below_5pct=float((rise[m] < 0.05).mean()), rise_above_25pct=float((rise[m] > 0.25).mean()),
                            witness_steps_median=float(np.median(wlen[m] / pl['step'])), witness_time_steps_median=float(np.median(t_wit[m] / pl['step'])))
            print(name, json.dumps(wt[name]), flush=True)
        out['witness'] = dict(groups=wt, witness_valid=int(sum(okw)))
        outp.write_text(json.dumps(out, indent=1))
    if a.only in ('', 'dock'):
        # Start on the certified witness path, a given arc length before the goal, and plan to the goal with the run's planner.
        qr = scene.q_range.cpu().numpy()
        names = [str(t) for t in Q['task_names']]
        dg = ev.clearance(qg).cpu().numpy()
        wit = [None] * len(qs)
        for k, n in enumerate(names):
            z = np.load(ROOT / 'experiments/cabinet/tests/suites' / f'cabinet_{n}_100.npz', allow_pickle=True)
            for j, i in enumerate(np.nonzero(Q['task'] == k)[0]):
                w = torch.tensor(np.asarray(z['witness_trajectories'][j], dtype=np.float64) / qr, device=dev, dtype=torch.float32)
                wit[i] = C.densify(w, torch.full((A.D,), 0.0005, device=dev))
        end = torch.stack([paths[i][-2] for i in range(len(qs))])
        gf = ev.encode(qg)
        T_stop = ev.to_goal(end, gf).cpu().numpy()
        left = torch.linalg.vector_norm(end - qg, dim=1).cpu().numpy() * scene.E
        groups = {'all': np.ones(len(qs), bool), 'default plan arrived': cv, 'default plan did not arrive': ~cv, 'goal in band': dg < C.MARGIN_CELLS,
                  'goal clearance < 0.05 cells': dg < 0.05, 'goal outside band': dg >= C.MARGIN_CELLS}
        dock = {}
        for cells in (0.25, 0.5, 1.0, 2.0, 4.0):
            st = []
            for i in range(len(qs)):
                w = wit[i]
                back = torch.flip(torch.cumsum(torch.flip(torch.linalg.vector_norm(w[1:] - w[:-1], dim=1), [0]), 0), [0])   # arc length from point j to the goal
                j = int(torch.nonzero(back <= cells / scene.E)[0]) if bool((back <= cells / scene.E).any()) else len(w) - 1
                st.append(w[j])
            st = torch.stack(st)
            pp, cc, _ = run(ev, scene, st, qg, 0, 300, **pl)
            rows_d = C.score(scene, pp, cc.cpu().numpy())
            okd = np.array([r['success'] for r in rows_d])
            cvd = cc.cpu().numpy()
            T_w = ev.to_goal(st, gf).cpu().numpy()
            eu = torch.linalg.vector_norm(st - qg, dim=1).cpu().numpy() * scene.E
            dock[str(cells)] = {name: dict(count=int(m.sum()), solved=int(okd[m].sum()), arrived=int(cvd[m].sum()), start_to_goal_cells_median=float(np.median(eu[m])),
                                           T_witness_point_median=float(np.median(T_w[m]))) for name, m in groups.items()}
            print('dock', cells, {n: (v['solved'], v['arrived'], v['count']) for n, v in dock[str(cells)].items()}, flush=True)
        out['dock'] = dict(by_arc_cells=dock, stop=dict(T_median_not_arrived=float(np.median(T_stop[~cv])), left_cells_median_not_arrived=float(np.median(left[~cv]))))
        print('stop', out['dock']['stop'], flush=True)
        outp.write_text(json.dumps(out, indent=1))
    if a.only in ('', 'goal'):
        # Arrival of the default plans by the clearance of the goal / start, and where the segment loss's traversal endpoints lie.
        dg, ds = ev.clearance(qg).cpu().numpy(), ev.clearance(qs).cpu().numpy()
        cup = scene.cupboard
        in_cup = lambda x: ((scene.ee(x) >= cup[:, 0]) & (scene.ee(x) <= cup[:, 1])).all(1)  # noqa: E731
        bins = [0, 0.03, 0.05, 0.08, 0.12, C.MARGIN_CELLS, 100]
        det = ~Q['line']
        rows_g, rows_s = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            for d, rows_ in ((dg, rows_g), (ds, rows_s)):
                m = (d >= lo) & (d < hi)
                rows_.append(dict(lo=lo, hi=hi, lo_mm=1000 * lo * A.CELL_M, hi_mm=1000 * hi * A.CELL_M, n=int(m.sum()), arrived=int(cv[m].sum()), solved=int(ok[m].sum()),
                                  detour_n=int((m & det).sum()), detour_arrived=int(cv[m & det].sum()), detour_solved=int(ok[m & det].sum())))
        gen = torch.Generator(dev).manual_seed(3)
        ends = {}
        for mc in (float(fr['config']['min_clear']), 0.0):
            x = scene.free(200000, gen, mc)
            d = ev.clearance(x)
            ends[str(mc)] = dict(in_cupboard=float(in_cup(x).float().mean()), below_band_share=float((d < C.MARGIN_CELLS).float().mean()), below_005=float((d < 0.05).float().mean()),
                                 clearance_q=np.quantile(d.cpu().numpy(), [.1, .5, .9]).tolist())
        ig = in_cup(qg).cpu().numpy()
        out['goal'] = dict(by_goal_clearance=rows_g, by_start_clearance=rows_s, contact_cells=float(scene.contact), margin_cells=float(C.MARGIN_CELLS), cell_m=float(A.CELL_M),
                           traversal_endpoints=ends, test=dict(goals_in_cupboard=float(ig.mean()), starts_in_cupboard=float(in_cup(qs).float().mean()),
                                                              goal_clearance_in_cupboard_q=np.quantile(dg[ig], [.1, .5, .9]).tolist(), goals_below_band=float((dg < C.MARGIN_CELLS).mean())),
                           arrived=int(cv.sum()))
        print('goal', json.dumps(out['goal'])[:900], flush=True)
        outp.write_text(json.dumps(out, indent=1))
    if a.only in ('', 'block'):
        # What separates the stop from the goal: clearance along the straight move, and how far the grasp point still is.
        na = np.nonzero(~cv)[0]
        e = torch.stack([paths[i][-2] for i in na])
        g = qg[na]
        t = torch.linspace(0, 1, 201, device=dev)[None, :, None]
        seg = (e[:, None, :] + (g - e)[:, None, :] * t).reshape(-1, A.D)
        pt = scene.parts(seg)
        obs, slf = pt['obs'].reshape(len(na), 201), pt['self'].reshape(len(na), 201)
        hand = torch.linalg.vector_norm(scene.ee(e) - scene.ee(g), dim=1).cpu().numpy()
        rad = torch.linalg.vector_norm((g - e) * scene.q_range, dim=1).cpu().numpy()
        out['block'] = dict(count=int(len(na)), obstacle_blocks=int((obs < A.OBS_VALID_M).any(1).sum()), self_blocks=int((slf < A.SELF_VALID_M).any(1).sum()),
                            obstacle_min_mm_q=np.quantile(1000 * obs.min(1).values.cpu().numpy(), [.1, .25, .5, .75, .9]).tolist(),
                            in_contact_share_q=np.quantile((obs < A.OBS_VALID_M).float().mean(1).cpu().numpy(), [.1, .25, .5, .75, .9]).tolist(),
                            grasp_point_to_goal_mm_q=np.quantile(1000 * hand, [.1, .25, .5, .75, .9]).tolist(), joint_move_rad_q=np.quantile(rad, [.1, .25, .5, .75, .9]).tolist())
        print('block', json.dumps(out['block']), flush=True)
        outp.write_text(json.dumps(out, indent=1))
    print('wrote', outp, f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
