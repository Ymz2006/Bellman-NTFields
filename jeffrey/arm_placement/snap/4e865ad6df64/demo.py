"""Hard queries of one placement and the paths the trained fields plan for them, in 12 joints.

Hard = both arms' grasp points inside the cupboard at the start AND at the goal (the suites' compartment_to_compartment and
same_compartment strata). For every field run given, the saved plans of its three planning seeds (paths_seed{0,1,2}.npy, radians) are
scored again here by core.score at 0.001 rad per joint (the rule that certified the suites), and a plan counts as arrived when its last
two consecutive waypoints are more than 1.05 planner steps apart (a plan that stops short has a long last piece to the goal, or between
the two heads; it can still pass when that piece is free; such plans are not picked).

Picked: `--pick` queries that the first field solves with all three planning seeds, hardest first (not solvable by a straight line,
every arm changes zone, longest joint distance), at most `--per-signature` per (start zones -> goal zones) signature. For each: start and
goal (radians, [left 6, right 6]), the seed-0 plan of every field, grasp-point and link-origin positions along it, clearances.

A field is given as name=<field run> (its own plans, paths_seed{k}.npy: ntrl-demo's planner) or as name=<evalfield run>:<prefix> (the
plans an evalfield run saved, <prefix>_seed{k}.npy, e.g. paths_lift_bidir_600_50_5 for the two-ended planner).

    CUDA_VISIBLE_DEVICES=0 python demo.py --prepared <prepare run> --fields lift=<field run> ntrl=<field run> --out <dir>
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
LBS = HERE if (HERE / 'pipeline6.py').is_file() else HERE.parent / 'lbs_arm'
sys.path.insert(0, str(LBS))
import arm12  # noqa: E402
sys.modules['arm'] = arm12
import arm as A  # noqa: E402
import arm as A12  # noqa: E402,F401
import core as C  # noqa: E402
from common import Logger, dump  # noqa: E402

HARD = ('compartment_to_compartment', 'same_compartment')


def skeleton(scene, x):
    """(N, A, 8, 3): base, the six link-frame origins, the grasp point of every arm at normalized configurations x."""
    q = (x * scene.q_range).reshape(len(x) * scene.A, 6)
    F = arm12.A1.link_frames(q, scene.O, scene.T0)
    o = F[:, :, :3, 3].reshape(len(x), scene.A, 6, 3)
    Rt = scene.base_R.transpose(1, 2)
    o = o @ Rt[None] + scene.base_t[None, :, None, :]
    ee = scene.ee(x)
    base = scene.base_t[None, :, None, :].expand(len(x), -1, 1, -1)
    return torch.cat([base, o, ee[:, :, None, :]], 2)


def zone_of(ee, comp):
    """(N, A) zone index of each grasp point (-1 outside the zone boxes); comp (Z, 3, 2)."""
    inside = ((ee[:, :, None, :] >= comp[None, None, :, :, 0]) & (ee[:, :, None, :] <= comp[None, None, :, :, 1])).all(3)
    return torch.where(inside.any(2), inside.float().argmax(2), torch.full(inside.shape[:2], -1, device=ee.device))


def resample(path, n):
    """n points at equal arc length along a polyline (K, D)."""
    seg = torch.linalg.vector_norm(path[1:] - path[:-1], dim=1)
    s = torch.cat([seg.new_zeros(1), torch.cumsum(seg, 0)])
    t = torch.linspace(0, float(s[-1]), n, device=path.device)
    i = torch.searchsorted(s, t, right=True).clamp(1, len(s) - 1)
    w = ((t - s[i - 1]) / (s[i] - s[i - 1]).clamp(min=1e-12))[:, None]
    return path[i - 1] + w * (path[i] - path[i - 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--fields', nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--pick', type=int, default=10)
    p.add_argument('--per-signature', type=int, default=2)
    p.add_argument('--points', type=int, default=80)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    log = Logger(a.out / 'log.txt')
    dev, t0 = 'cuda:0', time.time()
    scene, info = A.load_scene(a.prepared, dev)
    m = np.load(a.prepared / 'robot_model.npz')
    comp = torch.tensor(m['compartments'], device=dev)
    zone_names = [str(v) for v in m['compartment_names']]
    Q = dict(np.load(a.prepared / 'queries.npz'))
    names = [str(t) for t in Q['task_names']]
    qs, qg = torch.tensor(Q['start'], device=dev), torch.tensor(Q['goal'], device=dev)
    n = len(qs)
    qr = scene.q_range
    stp = C.planner_settings(scene)['step']
    cup_s, cup_g = scene.in_cupboard(qs), scene.in_cupboard(qg)
    hard = (cup_s.all(1) & cup_g.all(1)).cpu().numpy() & np.isin(Q['task'], [names.index(t) for t in HARD if t in names])
    zs, zg = zone_of(scene.ee(qs), comp).cpu().numpy(), zone_of(scene.ee(qg), comp).cpu().numpy()
    fields, ok, arrived, plans = [], {}, {}, {}
    for item in a.fields:
        name, run = item.split('=', 1)
        run, prefix = (run.rsplit(':', 1) + ['paths'])[:2] if ':' in run else (run, 'paths')
        fields.append(name)
        ok[name], arrived[name], plans[name] = np.zeros((3, n), bool), np.zeros((3, n), bool), []
        for seed in range(3):
            raw = np.load(Path(run) / f'{prefix}_seed{seed}.npy', allow_pickle=True)
            paths = [torch.tensor(r, device=dev) / qr for r in raw]
            assert len(paths) == n, (len(paths), n)
            rows = C.score(scene, paths, np.ones(n, bool), step_rad=1e-3)
            ok[name][seed] = [r['success'] for r in rows]
            arrived[name][seed] = [float(torch.linalg.vector_norm(pth[1:] - pth[:-1], dim=1).max()) <= 1.05 * stp for pth in paths]
            plans[name].append(paths)
        log(f'{name}: solved per planning seed {ok[name].sum(1).tolist()} of {n}; hard queries {int(hard.sum())}: solved {ok[name][:, hard].sum(1).tolist()}, '
            f'with all three seeds {int(ok[name][:, hard].all(0).sum())}, arrived and solved {int((ok[name] & arrived[name])[:, hard].all(0).sum())}')
    first = fields[0]
    robust = (ok[first] & arrived[first]).all(0) & hard
    dist = np.linalg.norm((Q['start'] - Q['goal']) * qr.cpu().numpy(), axis=1)
    change = (zs != zg).sum(1)
    differ = (zs[:, 0] != zs[:, 1]) | (zg[:, 0] != zg[:, 1])
    score = 4.0 * (~Q['line']) + 1.0 * change + 0.5 * differ + dist / max(float(dist.max()), 1e-9)
    order = [i for i in np.argsort(-score) if robust[i]]
    seen, picked = {}, []
    for i in order:
        sig = (tuple(zs[i]), tuple(zg[i]))
        if seen.get(sig, 0) >= a.per_signature:
            continue
        seen[sig] = seen.get(sig, 0) + 1
        picked.append(int(i))
        if len(picked) >= a.pick:
            break
    if len(picked) < a.pick:
        picked += [int(i) for i in order if i not in picked][:a.pick - len(picked)]
    zn = lambda z: [zone_names[k] if k >= 0 else 'opening' for k in z]  # noqa: E731
    demo, store = [], {}
    for rank, i in enumerate(picked):
        item = dict(rank=rank + 1, query=i, task=names[int(Q['task'][i])], straight_line_feasible=bool(Q['line'][i]), witness_source=str(Q['witness_source'][i]),
                    joint_distance_rad=float(dist[i]), start_zones=zn(zs[i]), goal_zones=zn(zg[i]), start_rad=(qs[i] * qr).tolist(), goal_rad=(qg[i] * qr).tolist(),
                    start_grasp_points=scene.ee(qs[i:i + 1])[0].tolist(), goal_grasp_points=scene.ee(qg[i:i + 1])[0].tolist(), plans={})
        pe = scene.parts(torch.stack([qs[i], qg[i]]))
        item['endpoint_clearance_m'] = dict(obstacle=pe['obs_only'].tolist(), inter_arm=pe['inter'].tolist(), self=pe['self'].tolist())
        for name in fields:
            item['plans'][name] = dict(solved_seeds=[bool(v) for v in ok[name][:, i]], arrived_seeds=[bool(v) for v in arrived[name][:, i]])
            for seed in range(3):
                store[f'q{i}_{name}_seed{seed}'] = (plans[name][seed][i] * qr).cpu().numpy().astype(np.float32)
            pth = plans[name][0][i]
            rs = resample(pth, a.points)
            pt = scene.parts(C.densify(pth, 1e-3 / qr))
            sk = skeleton(scene, rs)
            item['plans'][name].update(waypoints=int(len(pth)), length_rad=float(torch.linalg.vector_norm((pth[1:] - pth[:-1]) * qr, dim=1).sum()),
                                       min_obstacle_m=float(pt['obs_only'].min()), min_inter_arm_m=float(pt['inter'].min()), min_self_m=float(pt['self'].min()),
                                       joints_rad=[[round(float(v), 4) for v in r] for r in (rs * qr)], skeleton=[[[[round(float(c), 3) for c in pnt] for pnt in arm] for arm in fr] for fr in sk[:: max(1, a.points // 16)]],
                                       grasp=[[[round(float(c), 3) for c in arm] for arm in fr] for fr in sk[:, :, 7]])
        demo.append(item)
        log(f'pick {rank + 1}: query {i} ({item["task"]}), zones {item["start_zones"]} -> {item["goal_zones"]}, distance {dist[i]:.2f} rad, straight line {"free" if Q["line"][i] else "blocked"}; '
            + '; '.join(f'{f} solved {sum(item["plans"][f]["solved_seeds"])}/3' for f in fields))
    np.savez(a.out / 'demo.npz', query=np.array(picked), start_rad=np.stack([d['start_rad'] for d in demo]) if demo else np.zeros((0, scene.D)),
             goal_rad=np.stack([d['goal_rad'] for d in demo]) if demo else np.zeros((0, scene.D)), **store)
    doc = yaml_scene(info)
    res = dict(stage='armplace-demo', prepared=str(a.prepared), fields={i.split('=', 1)[0]: i.split('=', 1)[1] for i in a.fields}, scene=doc, zone_names=zone_names,
               hard=dict(n=int(hard.sum()), by_task={t: int((hard & (Q['task'] == names.index(t))).sum()) for t in HARD if t in names},
                         solved={f: dict(per_seed=ok[f][:, hard].sum(1).tolist(), all_seeds=int(ok[f][:, hard].all(0).sum()), any_seed=int(ok[f][:, hard].any(0).sum()),
                                         arrived_and_solved_all_seeds=int((ok[f] & arrived[f])[:, hard].all(0).sum())) for f in fields}),
               all=dict(n=n, solved={f: dict(per_seed=ok[f].sum(1).tolist(), all_seeds=int(ok[f].all(0).sum())) for f in fields}),
               picked=demo, seconds=time.time() - t0)
    dump(a.out / 'results.json', res)
    log(f'done: {len(demo)} picked, {time.time() - t0:.0f}s')


def yaml_scene(info):
    import yaml
    f = Path(info['scene_file'])
    doc = yaml.safe_load(f.read_text())
    return dict(file=str(f), robots=doc['robots'], cuboids=[dict(name=c['name'], pose=c['pose'][:3], dims=c['dims']) for c in doc['cuboids']], compartments=doc['compartments'], sampling=doc['sampling'])


if __name__ == '__main__':
    main()
