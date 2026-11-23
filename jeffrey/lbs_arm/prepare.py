"""Prepare a cabinet scene for the LBS arm runs (host venv): the scene of arm.py from an export run's robot_model.npz, its checks, the
training pairs and the frozen test queries in the scene's coordinates.

    python jeffrey/lbs_arm/prepare.py --export results/registered_<export run> --tests experiments/cabinet/tests --out <dir>

Writes robot_model.npz (copied), scene.json (E = the median clearance-gradient norm in the band, cells per unit; checks against the label
code and the scorer), data/{sampled_points,speed,normal}.npy (ntrl-demo's layout), queries.npz, results.json.

Training pairs (ntrl-demo's arm recipe, dataprocessing/speed_sampling_arm_normal.py, as jeffrey/mobile_arm/data9.py applies it):
x0 uniform in the joint box with the grasp point in the scene's sampling box and offset < d(x0) < margin; x1 = x0 + unit(rand - 0.5) x
U(0, r_max) inside the joint box with d(x1) > contact - offset; speed = clip(d / margin, offset / margin, 1); normal = the unit
gradient of d in the normalized joint space.

Queries: the 600 frozen queries of the scene's six suites (start, goal in normalized coordinates, task, the suite's own flag "the
straight line passes the release scorer"). As a check of this scene's scorer the straight lines are scored here too.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arm as A  # noqa: E402
import core as C  # noqa: E402
from maze import MARGIN_CELLS, OFFSET_CELLS  # noqa: E402


def sample_pairs(scene, n, seed, r_max, batch=1 << 21, cup_share=0.0):
    """cup_share > 0: that share of the pairs has the grasp point of x0 inside the cupboard box (the other pairs are thinned to match)."""
    dev = scene.device
    gen = torch.Generator(dev).manual_seed(seed)
    xs, ys, ns = [], [], []
    drawn, total = 0, 0
    while total < n:
        x0 = scene.uniform(batch, gen)
        drawn += batch
        x0 = x0[scene.in_box(x0)]
        d0 = scene.clearance(x0)
        x0 = x0[(d0 > OFFSET_CELLS) & (d0 < MARGIN_CELLS)]
        if cup_share > 0:
            e0 = scene.ee(x0)
            c0 = ((e0 >= scene.cupboard[:, 0]) & (e0 <= scene.cupboard[:, 1])).all(1)
            others = torch.nonzero(~c0).squeeze(1)[:int(round(int(c0.sum()) * (1 - cup_share) / cup_share))]
            x0 = torch.cat([x0[c0], x0[others]])[torch.randperm(int(c0.sum()) + len(others), generator=gen, device=dev)]
        dp = torch.rand(len(x0), A.D, generator=gen, device=dev) - 0.5
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
    gap = np.linalg.norm(x[:, :A.D] - x[:, A.D:], axis=1)
    with torch.no_grad():
        e1 = scene.in_box(torch.tensor(x[:, A.D:], device=dev))
        cup = scene.cupboard
        in_cup = lambda q: torch.cat([((scene.ee(q[i:i + (1 << 17)]) >= cup[:, 0]) & (scene.ee(q[i:i + (1 << 17)]) <= cup[:, 1])).all(1) for i in range(0, len(q), 1 << 17)])  # noqa: E731
        c0, c1 = in_cup(torch.tensor(x[:, :A.D], device=dev)), in_cup(torch.tensor(x[:, A.D:], device=dev))
    stats = dict(n=len(x), drawn=drawn, r_max=r_max, x1_floor=float((d[:, 1] <= OFFSET_CELLS).mean()), x1_full_speed=float((d[:, 1] >= MARGIN_CELLS).mean()),
                 gap_quantiles=np.quantile(gap, [0.1, 0.5, 0.9]).tolist(), gap_cells_quantiles=(np.quantile(gap, [0.1, 0.5, 0.9]) * scene.E).tolist(),
                 x1_grasp_point_in_box=float(e1.float().mean()), cup_share_arg=cup_share, x0_in_cupboard=float(c0.float().mean()), x1_in_cupboard=float(c1.float().mean()),
                 both_in_cupboard=float((c0 & c1).float().mean()))
    return x, speed, normal, d, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--tests', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--pairs', type=int, default=500_000)
    p.add_argument('--r-max', type=float, default=0.5)
    p.add_argument('--cup-share', type=float, default=0.0, help='share of the pairs whose x0 has the grasp point inside the cupboard')
    p.add_argument('--seed', type=int, default=20261120)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'data').mkdir(exist_ok=True)
    t0 = time.time()
    dev = 'cuda:0'
    shutil.copyfile(a.export / 'robot_model.npz', a.out / 'robot_model.npz')
    scene = A.CabinetArm(a.out / 'robot_model.npz', dev)
    check = scene.check()
    assert max(check[k] for k in ('obs', 'obs_all', 'self', 'ee')) < 1e-5, check
    meas = scene.measure(100_000 if a.smoke else 400_000)
    E = round(meas['norm_quantiles'][2], 1)
    scene = A.CabinetArm(a.out / 'robot_model.npz', dev, E=E)
    gen = torch.Generator(dev).manual_seed(a.seed + 1)
    x = scene.uniform(1 << 22, gen)
    x = x[scene.in_box(x)]
    d = scene.clearance(x)
    v = scene.valid_cells(x) >= 0
    occ = dict(in_box_of_uniform=len(x) / (1 << 22), valid_of_in_box=float(v.float().mean()), contact_rule_equals_scorer=float(((d >= scene.contact) == v).float().mean()),
               full_speed_of_valid=float((d[v] >= MARGIN_CELLS).float().mean()))
    n = 4096 if a.smoke else a.pairs
    X, speed, normal, dd, stats = sample_pairs(scene, n, a.seed, a.r_max, cup_share=a.cup_share)
    np.save(a.out / 'data/sampled_points.npy', X)
    np.save(a.out / 'data/speed.npy', speed)
    np.save(a.out / 'data/normal.npy', normal)
    man = json.loads((a.tests / 'suite_manifest.json').read_text())
    qr = scene.q_range.cpu().numpy()
    S, G, task, line, both = [], [], [], [], []
    cup = scene.cupboard
    for k, item in enumerate(man['suites']):
        z = np.load(a.tests / 'suites' / item['file'], allow_pickle=True)
        S.append(z['starts_rad'] / qr)
        G.append(z['goals_rad'] / qr)
        task += [k] * len(z['starts_rad'])
        line.append(z['direct_line_strict_feasible'].astype(bool))
    S, G = np.concatenate(S).astype(np.float32), np.concatenate(G).astype(np.float32)
    st, gt = torch.tensor(S, device=dev), torch.tensor(G, device=dev)
    es, eg = scene.ee(st), scene.ee(gt)
    in_cup = lambda e: ((e >= cup[:, 0]) & (e <= cup[:, 1])).all(1)  # noqa: E731
    ends_in = (in_cup(es).long() + in_cup(eg).long()).cpu().numpy()
    line = np.concatenate(line)
    rows = C.score(scene, [torch.stack([st[i], gt[i]]) for i in range(len(st))], np.ones(len(st), bool))
    mine = np.array([r['success'] for r in rows])
    np.savez(a.out / 'queries.npz', start=S, goal=G, task=np.array(task), line=line, line_here=mine, ends_in_cupboard=ends_in,
             task_names=np.array([i['task'] for i in man['suites']]), dist=np.linalg.norm(S - G, axis=1))
    info = dict(E_cells=E, scene=scene.spec(), describe=scene.describe(), measure=meas, check=check, occupancy=occ, tests=str(a.tests), tests_scene=man['scene'])
    (a.out / 'scene.json').write_text(json.dumps(info, indent=1))
    res = dict(stage='lbsarm-prepare', smoke=a.smoke, E_cells=E, check=check, measure=meas, occupancy=occ, data=stats,
               queries=dict(n=len(S), per_task={t: int((np.array(task) == k).sum()) for k, t in enumerate(i['task'] for i in man['suites'])},
                            straight_line_suite=int(line.sum()), straight_line_here=int(mine.sum()), straight_line_agree=int((line == mine).sum()),
                            ends_in_cupboard={str(k): int((ends_in == k).sum()) for k in (0, 1, 2)}, dist_quantiles=np.quantile(np.linalg.norm(S - G, axis=1), [0, 0.1, 0.5, 0.9, 1]).tolist(),
                            start_valid=float((scene.valid_cells(st) >= 0).float().mean()), goal_valid=float((scene.valid_cells(gt) >= 0).float().mean())),
               planner=C.planner_settings(scene), seconds=time.time() - t0)
    (a.out / 'results.json').write_text(json.dumps(res, indent=1))
    (a.out / 'log.txt').write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
