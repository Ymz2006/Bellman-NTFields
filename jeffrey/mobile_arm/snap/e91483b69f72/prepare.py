"""Stage 'prepare' of the mobile-arm study: training pairs and certified queries for the 9-DOF robot in a Gibson house.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/mobile_arm/prepare.py --house results/registered_<lift_scale house prepare> --out <dir>

The house is a registered jeffrey/lift_scale house prepare run (floor plan, SE(2) truth fields of the bare rectangle for its 48 goals).
1. ntrl-demo pairs (data9.py): 500,000 (data seed 0) + 50,000 held out (seed 1).
2. Queries. Start = a random configuration with clearance >= 0.1 cells (base pose and arm uniform in the box); goal = the base pose of
   one of the house run's 48 truth goals with its own random arm (clearance >= 0.1). Difficulty = D = T*_SE2(start base, goal base) / pitch,
   the bare rectangle's (= the tucked robot's, scene9.TUCK) true travel distance in cells, binned as the house studies (0.5-2, 2-4, ...,
   40 queries per bin, spread over goals). Each query is certified by a reference path: fold the arm to TUCK with the base standing
   (straight in joint space; if that touches anything, first the shoulder lift / elbow / wrist 1 / wrist 2 then the pan and wrist 3),
   drive the truth path of the SE(2) field (grid descent) tucked, unfold at the goal (the reverse); every configuration on it, at <= 0.005
   cells per coordinate, has clearance >= 0. Its travel time T_ref (exact speed) is an upper bound on the 9-D optimum.
3. Controls: the straight segment, scored like the planner.
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
import scene9 as S9  # noqa: E402  (puts lift_scale, embed_warp, maze_scale on the path)
import robot as RB  # noqa: E402
import data9 as D9  # noqa: E402
import plan9 as P9  # noqa: E402
import truth as TR  # noqa: E402  (maze_scale)
from common import BINS, Logger, bin_labels, by_bin, dump, save_paths  # noqa: E402

QUERY_CLEAR = 0.1
STAGE_A = [1, 2, 3, 4]          # shoulder lift, elbow, wrist 1, wrist 2: the arm straightens up first
STAGE_B = [0, 5]                # pan, wrist 3


def arm_moves(q_from, q_to, order):
    """Waypoints (k, 9) after q_from moving the joints to q_to: order 'one' = all at once, 'AB' = STAGE_A then STAGE_B, 'BA' the reverse."""
    if order == 'one':
        return q_to[None]
    first, second = (STAGE_A, STAGE_B) if order == 'AB' else (STAGE_B, STAGE_A)
    mid = q_from.clone()
    idx = torch.tensor([3 + j for j in first], device=q_from.device)
    mid[idx] = q_to[idx]
    return torch.stack([mid, q_to])


def clean(scene, pts):
    dense = P9.densify(pts, 0.005 * scene.pitch)
    d = scene.clearance(dense)
    return bool((d >= 0).all()), dense, d


def certify(scene, grid, fields, gi, start, goal, log=None):
    """Reference path for each (start, goal) (9-D). Returns lists: ok, waypoints, t_ref, clen, (tuck order, untuck order), parts times."""
    base_paths = grid.descend(fields, gi, start[:, :3], goal[:, :3])
    out = []
    for i in range(len(start)):
        s, g = start[i], goal[i]
        ts, tg = scene.with_tuck(s[None, :3])[0], scene.with_tuck(g[None, :3])[0]
        drive = scene.with_tuck(base_paths[i])
        ok_drive, dense_drive, d_drive = clean(scene, drive)
        best = None
        if ok_drive:
            for o1 in ('one', 'AB'):
                w1 = torch.cat([s[None], arm_moves(s, ts, o1)])
                ok1, dense1, d1 = clean(scene, w1)
                if ok1:
                    break
            for o2 in ('one', 'BA'):
                w2 = torch.cat([tg[None], arm_moves(tg, g, o2)])
                ok2, dense2, d2 = clean(scene, w2)
                if ok2:
                    break
            if ok1 and ok2:
                t1, t0, t2 = P9.path_time(scene, dense1, d1), P9.path_time(scene, dense_drive, d_drive), P9.path_time(scene, dense2, d2)
                way = torch.cat([w1, drive[1:], w2[1:]])
                clen = float(torch.linalg.vector_norm(way[1:] - way[:-1], dim=1).sum())
                best = dict(ok=True, way=way, t_ref=t1 + t0 + t2, t_tuck=t1, t_drive=t0, t_untuck=t2, clen=clen, tuck=o1, untuck=o2)
        out.append(best if best is not None else dict(ok=False, drive_ok=ok_drive))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--house', type=Path, required=True, help='registered lift_scale house prepare run')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--n-data', type=int, default=500_000)
    p.add_argument('--quota', type=int, default=40)
    p.add_argument('--candidates', type=int, default=200_000)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    log = Logger(out / 'log.txt')
    dev = 'cuda:0'
    t0 = time.time()
    if a.smoke:
        a.n_data, a.quota, a.candidates = 20_000, 4, 20_000
    house, hinfo = S9.house_from_prepare(a.house, dev)
    hres = json.loads((a.house / 'results.json').read_text())
    scene = S9.MobileArm(house, dev)
    res = dict(stage='prepare', maze=scene.describe(), house_prepare=str(a.house), smoke=a.smoke, gpu=torch.cuda.get_device_name(0), bins=BINS,
               query_clearance_cells=QUERY_CLEAR)
    dump(out / 'maze.json', dict(scene.describe(), house_prepare=str(a.house)))
    log(f'scene {scene.describe()}')

    td = time.time()
    x, speed, normal, d, stats = D9.sample(scene, a.n_data, 0)
    (out / 'data').mkdir(exist_ok=True)
    np.save(out / 'data/sampled_points.npy', x)
    np.save(out / 'data/speed.npy', speed)
    np.save(out / 'data/normal.npy', normal)
    res['data'] = dict(stats, seconds=time.time() - td)
    hx, hspeed, _, hd, hstats = D9.sample(scene, 5_000 if a.smoke else 50_000, 1)
    (out / 'heldout').mkdir(exist_ok=True)
    np.save(out / 'heldout/sampled_points.npy', hx)
    np.save(out / 'heldout/clearance_cells.npy', hd.astype(np.float32))
    log(f'data: {res["data"]}')
    del x, speed, normal, d
    torch.cuda.empty_cache()

    # --- queries ----------------------------------------------------------------------------------------------------------------------
    goals_b = torch.tensor(np.load(a.house / 'goals.npy'), device=dev)
    G = len(goals_b)
    fields = torch.tensor(np.load(a.house / 'truth_fields.npy', mmap_mode='r')[:], device=dev, dtype=torch.float32)
    grid = TR.Grid(house, h_cells=hres['truth']['h_cells'])
    assert tuple(grid.shape) == tuple(fields.shape[1:]), 'truth grid differs from the house run'
    gen = torch.Generator(dev).manual_seed(7)
    starts = scene.free(a.candidates, gen, QUERY_CLEAR)
    T = torch.empty(G, len(starts), device=dev)
    for g in range(G):
        for i in range(0, len(starts), 50_000):
            s = starts[i:i + 50_000, :3]
            T[g, i:i + len(s)] = grid.query_time(fields, torch.full((len(s),), g, device=dev, dtype=torch.long), s)
    Dc = (T / scene.pitch).cpu().numpy()
    rng = np.random.default_rng(3)
    labels = bin_labels()
    available, kept_rows, tried = {}, [], {}
    for b in range(len(BINS) - 1):
        gi_all, si_all = np.nonzero((Dc >= BINS[b]) & (Dc < BINS[b + 1]) & np.isfinite(Dc))
        available[labels[b]] = int(len(gi_all))
        if not len(gi_all):
            continue
        order = rng.permutation(len(gi_all))[:200_000]
        cap = math.ceil(a.quota / 12)
        used, per_goal, got = set(), {}, []
        rounds = 0
        while len(got) < a.quota and rounds < 6:
            rounds += 1
            want = int(1.5 * (a.quota - len(got))) + 4
            chosen = []
            while len(chosen) < want:
                before = len(chosen)
                for k in order:
                    if len(chosen) >= want:
                        break
                    g = int(gi_all[k])
                    if int(k) not in used and per_goal.get(g, 0) < cap:
                        chosen.append(int(k))
                        used.add(int(k))
                        per_goal[g] = per_goal.get(g, 0) + 1
                if len(chosen) == before and len(used) >= len(order):
                    break
                cap *= 2
            if not chosen:
                break
            ch = np.array(chosen)
            gi = torch.tensor(gi_all[ch], device=dev)
            S = starts[torch.tensor(si_all[ch], device=dev)]
            # goal arm: per query, the first of 64 random joint vectors whose goal configuration has clearance >= 0.1
            Gq = torch.empty(len(ch), 9, device=dev)
            for i in range(len(ch)):
                for _ in range(20):
                    cand = scene.uniform(64, gen)
                    cand[:, :3] = goals_b[gi[i]]
                    dc = scene.clearance(cand)
                    j = torch.nonzero(dc >= QUERY_CLEAR).squeeze(1)
                    if len(j):
                        Gq[i] = cand[j[0]]
                        break
                else:
                    Gq[i] = float('nan')
            okg = ~Gq.isnan().any(1)
            for i0 in range(0, len(ch), 64):
                sl = slice(i0, min(i0 + 64, len(ch)))
                idx = torch.nonzero(okg[sl]).squeeze(1) + i0
                if not len(idx):
                    continue
                cert = certify(scene, grid, fields, gi[idx], S[idx], Gq[idx])
                for k, c in zip(idx.tolist(), cert):
                    tried[labels[b]] = tried.get(labels[b], 0) + 1
                    if c['ok'] and len(got) < a.quota:
                        got.append(dict(c, b=b, g=int(gi[k]), start=S[k], goal=Gq[k], t_se2=float(T[gi[k], int(si_all[ch[k]])])))
            log(f'bin {labels[b]} round {rounds}: {len(got)}/{a.quota} certified of {tried.get(labels[b], 0)} tried')
        kept_rows += got
    Q = kept_rows
    res['queries'] = dict(available_pairs=available, tried=tried, kept=len(Q), per_bin={labels[b]: sum(1 for q in Q if q['b'] == b) for b in range(len(BINS) - 1)},
                          tuck_orders={o: sum(1 for q in Q if q['tuck'] == o) for o in ('one', 'AB')}, untuck_orders={o: sum(1 for q in Q if q['untuck'] == o) for o in ('one', 'BA')})
    S = torch.stack([q['start'] for q in Q])
    Gq = torch.stack([q['goal'] for q in Q])
    bins = np.array([q['b'] for q in Q])
    t_se2 = np.array([q['t_se2'] for q in Q])
    np.savez(out / 'queries.npz', start=S.cpu().numpy(), goal=Gq.cpu().numpy(), goal_index=np.array([q['g'] for q in Q]), bin=bins,
             t_se2=t_se2, d_cells=t_se2 / scene.pitch, t_ref=np.array([q['t_ref'] for q in Q]), t_tuck=np.array([q['t_tuck'] for q in Q]),
             t_drive=np.array([q['t_drive'] for q in Q]), t_untuck=np.array([q['t_untuck'] for q in Q]), ref_clen=np.array([q['clen'] for q in Q]),
             euclid_cells=(torch.linalg.vector_norm(S[:, :2] - Gq[:, :2], dim=1) / scene.pitch).cpu().numpy(),
             arm_cells=(torch.linalg.vector_norm(S[:, 3:] - Gq[:, 3:], dim=1) / scene.pitch).cpu().numpy(),
             cspace_cells=(torch.linalg.vector_norm(S - Gq, dim=1) / scene.pitch).cpu().numpy())
    save_paths(out / 'ref_paths.npy', [q['way'] for q in Q])
    log(f'queries: {res["queries"]}')
    st = P9.settings(scene)
    clen = np.array([q['clen'] for q in Q])
    budget = np.maximum(st['min_steps'], np.ceil(st['budget_factor'] * clen / st['step']))
    res['planner'] = dict(st, budget_median=float(np.median(budget)), budget_max=float(budget.max()))
    rows = P9.score(scene, P9.straight(S, Gq), [True] * len(S))
    res['straight'] = dict(success=int(sum(r['success'] for r in rows)), by_bin=by_bin(rows, bins),
                           contact_parts={p: sum(1 for r in rows if r.get('contact_part') == p) for p in P9.PARTS})
    tr = np.array([q['t_ref'] for q in Q]) / t_se2
    res['reference'] = dict(t_ref_over_t_se2_median=float(np.median(tr)), t_ref_over_t_se2_by_bin={labels[b]: float(np.median(tr[bins == b])) for b in sorted(set(bins.tolist()))},
                            tuck_share_of_t_ref_median=float(np.median((np.array([q['t_tuck'] + q['t_untuck'] for q in Q])) / np.array([q['t_ref'] for q in Q]))))
    log(f'straight: {res["straight"]}; reference: {res["reference"]}')
    res['elapsed_seconds'] = time.time() - t0
    res['complete'] = True
    dump(out / 'results.json', res)
    log(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
