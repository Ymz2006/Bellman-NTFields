"""Prepare a scene with several arms (real_world_12dof) for the LBS runs: prepare.py for arm12.ArmSet. Host venv.

    python prepare12.py --export <export run>/robot_model.npz --scene scenes/real_world_12dof.yml --tests tests_real_world_12dof --out <dir>

Writes what prepare.py writes, in the same layout: robot_model.npz (arm12.build_model), scene.json (E = the median clearance-gradient
norm in the band; the check against jeffrey/real_world/robot.py), data/{sampled_points,speed,normal}.npy, queries.npz, results.json.

Training pairs: prepare.py's recipe in 6 A joints. x0 uniform in the joint box with every grasp point in the sampling box and
offset < d(x0) < margin (drawn through arm12's candidates(): the same distribution, without the 60 times longer rejection loop); x1 = x0 + unit(rand - 0.5) x U(0, r_max) inside the joint box with d(x1) > contact - offset. r_max is
0.5 sqrt(A): the same step per arm as the one-arm scenes' 0.5.

Queries: the frozen suites (make_tests12.py). Checked here: every stored witness passes core.score at 0.001 rad per joint in this
scene, and the stored straight-line flags are core.score's.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arm12 as A  # noqa: E402
import core as C  # noqa: E402
from maze import MARGIN_CELLS, OFFSET_CELLS  # noqa: E402

STEP = 1e-3


def sample_pairs(scene, n, seed, r_max, batch=1 << 18):
    dev, D = scene.device, scene.D
    gen = torch.Generator(dev).manual_seed(seed)
    xs, ys, ns = [], [], []
    drawn, total = 0, 0
    while total < n:
        x0 = scene.candidates(batch, gen)
        drawn += batch
        d0 = scene.clearance(x0)
        x0 = x0[(d0 > OFFSET_CELLS) & (d0 < MARGIN_CELLS)]
        dp = torch.rand(len(x0), D, generator=gen, device=dev) - 0.5
        rl = torch.rand(len(x0), 1, generator=gen, device=dev) * r_max
        x1 = x0 + torch.nn.functional.normalize(dp, dim=1) * rl
        keep = scene.inside(x1)
        x0, x1 = x0[keep], x1[keep]
        d0, n0 = scene.clearance_normal(x0)
        d1, n1 = scene.clearance_normal(x1)
        keep = (d1 > scene.contact - OFFSET_CELLS) & ~n0.isnan().any(1) & ~n1.isnan().any(1)
        xs.append(torch.cat([x0, x1], 1)[keep].cpu())
        ys.append(torch.stack([d0, d1], 1)[keep].cpu())
        ns.append(torch.cat([n0, n1], 1)[keep].cpu())
        total += int(keep.sum())
    x = torch.cat(xs)[:n].numpy().astype(np.float32)
    d = torch.cat(ys)[:n].numpy()
    normal = torch.cat(ns)[:n].numpy().astype(np.float32)
    speed = np.clip(d / MARGIN_CELLS, OFFSET_CELLS / MARGIN_CELLS, 1).astype(np.float32)
    gap = np.linalg.norm(x[:, :D] - x[:, D:], axis=1)
    c0 = scene.in_cupboard(torch.tensor(x[:, :D], device=dev))
    c1 = scene.in_cupboard(torch.tensor(x[:, D:], device=dev))
    stats = dict(n=len(x), candidates_drawn=drawn, r_max=r_max, x1_floor=float((d[:, 1] <= OFFSET_CELLS).mean()), x1_full_speed=float((d[:, 1] >= MARGIN_CELLS).mean()),
                 gap_quantiles=np.quantile(gap, [0.1, 0.5, 0.9]).tolist(), gap_cells_quantiles=(np.quantile(gap, [0.1, 0.5, 0.9]) * scene.E).tolist(),
                 x1_grasp_points_in_box=float(scene.in_box(torch.tensor(x[:, D:], device=dev)).float().mean()),
                 x0_in_cupboard_per_arm=c0.float().mean(0).tolist(), x0_any_arm_in_cupboard=float(c0.any(1).float().mean()), x0_all_arms_in_cupboard=float(c0.all(1).float().mean()),
                 x1_any_arm_in_cupboard=float(c1.any(1).float().mean()))
    return x, speed, normal, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--tests', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--pairs', type=int, default=500_000)
    p.add_argument('--r-max', type=float, default=0.0, help='0 = 0.5 sqrt(arms)')
    p.add_argument('--seed', type=int, default=20261120)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'data').mkdir(exist_ok=True)
    t0 = time.time()
    dev = 'cuda:0'
    model = A.build_model(a.export, a.scene, a.out / 'robot_model.npz')
    scene = A.ArmSet(model, dev)
    check = scene.check(a.scene, 50_000 if a.smoke else 200_000)
    assert check['valid_agree'] == check['n'] and max(check[k] for k in ('obs', 'self', 'inter', 'ee')) < 1e-5, check
    meas = scene.measure(20_000 if a.smoke else 200_000)
    E = round(meas['norm_quantiles'][2], 1)
    scene = A.ArmSet(model, dev, E=E)
    gen = torch.Generator(dev).manual_seed(a.seed + 1)
    x = scene.uniform(1 << 23, gen)
    xb = x[scene.in_box(x)]
    d = scene.clearance(xb)
    v = scene.valid_cells(xb) >= 0
    occ = dict(in_box_of_uniform=len(xb) / len(x), valid_of_in_box=float(v.float().mean()), contact_rule_equals_scorer=float(((d >= scene.contact) == v).float().mean()),
               full_speed_of_valid=float((d[v] >= MARGIN_CELLS).float().mean()))
    r_max = a.r_max or 0.5 * math.sqrt(scene.A)
    X, speed, normal, stats = sample_pairs(scene, 4096 if a.smoke else a.pairs, a.seed, r_max)
    np.save(a.out / 'data/sampled_points.npy', X)
    np.save(a.out / 'data/speed.npy', speed)
    np.save(a.out / 'data/normal.npy', normal)
    man = json.loads((a.tests / 'suite_manifest.json').read_text())
    assert man['scene'] == a.scene.stem, (man['scene'], a.scene.stem)
    qr = scene.q_range.cpu().numpy()
    S, G, task, line, wit, source = [], [], [], [], [], []
    for k, item in enumerate(man['suites']):
        z = np.load(a.tests / 'suites' / item['file'], allow_pickle=True)
        S.append(z['starts_rad'] / qr)
        G.append(z['goals_rad'] / qr)
        task += [k] * len(z['starts_rad'])
        line.append(z['direct_line_strict_feasible'].astype(bool))
        wit += [np.asarray(w, np.float32) / qr for w in z['witness_trajectories']]
        source += [str(s) for s in z['witness_source']]
    S, G = np.concatenate(S).astype(np.float32), np.concatenate(G).astype(np.float32)
    st, gt = torch.tensor(S, device=dev), torch.tensor(G, device=dev)
    ends_in = (scene.in_cupboard(st).any(1).long() + scene.in_cupboard(gt).any(1).long()).cpu().numpy()
    arms_in = (scene.in_cupboard(st).long().sum(1) + scene.in_cupboard(gt).long().sum(1)).cpu().numpy()
    line = np.concatenate(line)
    ones = np.ones(len(st), bool)
    mine = np.array([r['success'] for r in C.score(scene, [torch.stack([st[i], gt[i]]) for i in range(len(st))], ones, step_rad=STEP)])
    wok = np.array([r['success'] for r in C.score(scene, [torch.tensor(w, device=dev) for w in wit], ones, step_rad=STEP)])
    assert all(np.abs(w[0] - S[i]).max() < 1e-6 and np.abs(w[-1] - G[i]).max() < 1e-6 for i, w in enumerate(wit)), 'a witness does not join its endpoints'
    assert wok.all() and (line == mine).all(), f'suite and scorer disagree: witnesses certified {int(wok.sum())}/{len(wok)}, straight-line flags equal {int((line == mine).sum())}'
    np.savez(a.out / 'queries.npz', start=S, goal=G, task=np.array(task), line=line, line_here=mine, ends_in_cupboard=ends_in, arm_ends_in_cupboard=arms_in,
             task_names=np.array([i['task'] for i in man['suites']]), dist=np.linalg.norm(S - G, axis=1), witness_source=np.array(source))
    info = dict(E_cells=E, scene=scene.spec(), describe=scene.describe(), measure=meas, check=check, occupancy=occ, tests=str(a.tests), tests_scene=man['scene'], scene_file=str(a.scene),
                arms=[str(v) for v in np.load(model)['arm_names']], score_step_rad=STEP)
    (a.out / 'scene.json').write_text(json.dumps(info, indent=1))
    names = [i['task'] for i in man['suites']]
    res = dict(stage='lbsarm12-prepare', smoke=a.smoke, E_cells=E, check=check, measure=meas, occupancy=occ, data=stats,
               queries=dict(n=len(S), per_task={t: int((np.array(task) == k).sum()) for k, t in enumerate(names)}, straight_line_suite=int(line.sum()), straight_line_here=int(mine.sum()),
                            straight_line_by_task={t: int(line[np.array(task) == k].sum()) for k, t in enumerate(names)}, witnesses_certified=int(wok.sum()),
                            witness_sources={k: int(v) for k, v in zip(*np.unique(source, return_counts=True))},
                            ends_in_cupboard={str(k): int((ends_in == k).sum()) for k in (0, 1, 2)}, dist_quantiles=np.quantile(np.linalg.norm(S - G, axis=1), [0, 0.1, 0.5, 0.9, 1]).tolist()),
               planner=C.planner_settings(scene), seconds=time.time() - t0)
    (a.out / 'results.json').write_text(json.dumps(res, indent=1))
    (a.out / 'log.txt').write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
