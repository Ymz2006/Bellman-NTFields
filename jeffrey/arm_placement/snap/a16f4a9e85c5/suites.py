"""Certified test suites of one placement: jeffrey/lbs_arm/make_tests12.py's recipe (same validity rule, endpoint pools, strata, pair
range, RRT-Connect, one-arm-at-a-time witnesses, certification at 0.001 rad, seeds) with what a sweep over placements needs:

    --per N            queries per stratum (make_tests12: 100)
    round sizes        per stratum, from the number still missing and the stratum's certification rate so far (make_tests12: 160 for
                       every unfinished stratum in every round), so that the RRT batch is large and few rounds are needed
    zones              an arm draws its zone among the zones it can reach at this placement (make_tests12 assumes all three; far from the
                       cupboard an arm may lose one). The reachable zones per arm are written to generation.json.
    --iterations       RRT-Connect iterations per round (make_tests12: 10,000)

Everything else is make_tests12's code, imported. Output: the same files (suite_manifest.json, generation.json, suites/*.npz).

    CUDA_VISIBLE_DEVICES=0 python suites.py --export arm_export.npz --scene scenes/<name>.yml --output <dir> --per 50
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
LBS = HERE if (HERE / 'make_tests12.py').is_file() else HERE.parent / 'lbs_arm'
sys.path.insert(0, str(LBS))
import make_tests12 as M  # noqa: E402
import arm12 as A  # noqa: E402

PRIOR = dict(free_to_free=0.49, free_to_compartment=0.29, compartment_to_free=0.33, compartment_to_compartment=0.09, same_compartment=0.56, anywhere=0.49)   # tests 0defb4fb


@torch.no_grad()
def endpoint_pool(scene, pools, kind, count, gen, zones_ok, batch=1 << 18):
    """make_tests12.endpoint_pool with the zone of an arm drawn among the zones that arm reaches (zones_ok[a]: list of zone ids)."""
    dev = scene.device
    X, Z, drawn = [], [], 0
    while sum(len(x) for x in X) < count:
        cols, zs = [], []
        for a in range(scene.A):
            if kind == 'compartment':
                zo = torch.tensor(zones_ok[a], device=dev)
                z = zo[torch.randint(len(zo), (batch,), generator=gen, device=dev)]
                col = torch.zeros(batch, 6, device=dev)
                for k in zones_ok[a]:
                    m = z == k
                    pk = pools[a][f'zone{k}']
                    col[m] = pk[torch.randint(len(pk), (int(m.sum()),), generator=gen, device=dev)]
            else:
                pk = pools[a][kind]
                z = torch.full((batch,), -1, device=dev)
                col = pk[torch.randint(len(pk), (batch,), generator=gen, device=dev)]
            cols.append(col)
            zs.append(z)
        x, z = torch.cat(cols, 1), torch.stack(zs, 1)
        ok = scene.valid_cells(x) >= 0
        drawn += batch
        X.append(x[ok])
        Z.append(z[ok])
        assert drawn < 400 * batch or sum(len(x) for x in X) > 0, f'no valid {kind} endpoint in {drawn} draws'
    return torch.cat(X)[:count], torch.cat(Z)[:count], sum(len(x) for x in X) / drawn


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--per', type=int, default=50)
    p.add_argument('--iterations', type=int, default=10000)
    p.add_argument('--max-rounds', type=int, default=8)
    p.add_argument('--round-max', type=int, default=640)
    p.add_argument('--min-zone-pool', type=int, default=200)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--no-compile', action='store_true')
    a = p.parse_args()
    started = time.monotonic()
    dev = 'cuda:0'
    name = a.scene.stem
    out = a.output
    (out / 'suites').mkdir(parents=True, exist_ok=True)
    (out / 'work').mkdir(exist_ok=True)

    def log(msg):
        print(msg, flush=True)
        with open(out / 'log.txt', 'a') as f:
            f.write(msg + '\n')

    scene = A.ArmSet(A.build_model(a.export, a.scene, out / 'robot_model.npz'), dev)
    scene.inter_chunk = 1 << 16
    m = np.load(out / 'robot_model.npz')
    scene.compartments = torch.tensor(m['compartments'], device=dev)
    zone_names = [str(v) for v in m['compartment_names']]
    arm_names = [str(v) for v in m['arm_names']]
    max_l2 = M.MAX_L2_PER_ARM * math.sqrt(scene.A)
    per, iterations = (3, 2000) if a.smoke else (a.per, a.iterations)
    samples, n_end = ((1 << 22), 20_000) if a.smoke else ((1 << 26), 300_000)
    capacity = iterations + 500
    strata = [M.STRATA[1], M.STRATA[4]] if a.smoke else M.STRATA
    qr = scene.q_range.cpu().numpy()
    V = M.Validator(scene, compile=not a.no_compile)
    gen = torch.Generator(dev).manual_seed(M.SEEDS['pool'])
    pools = M.arm_pools(scene, samples, gen, 400_000, batch=1 << 22)
    sizes = {arm_names[i]: {k: len(v) for k, v in pl.items()} for i, pl in enumerate(pools)}
    log(f'arm pools ({samples} uniform samples per arm): {json.dumps(sizes)}')
    zones_ok = [[k for k in range(len(zone_names)) if len(pools[i][f'zone{k}']) >= a.min_zone_pool] for i in range(scene.A)]
    log(f'zones each arm reaches (at least {a.min_zone_pool} of {samples} uniform samples valid alone with the grasp point in the zone): '
        f'{json.dumps({arm_names[i]: [zone_names[k] for k in z] for i, z in enumerate(zones_ok)})}')
    assert all(len(z) >= 2 for z in zones_ok), 'an arm reaches fewer than two zones: compartment_to_compartment has no query'
    ends, accept = {}, {}
    for kind in ('free', 'compartment', 'box', 'near'):
        x, z, acc = endpoint_pool(scene, pools, kind, n_end, gen, zones_ok)
        ends[kind], accept[kind] = (x, z), acc
    zc = ends['compartment'][1]
    zone_pairs = {f'{zone_names[i]}|{zone_names[j]}': int(((zc[:, 0] == i) & (zc[:, 1] == j)).sum()) for i in range(len(zone_names)) for j in range(len(zone_names))} if scene.A == 2 else {}
    targets = ends['near'][0] * scene.q_range
    timing = dict(pools=time.monotonic() - started)
    log(f'endpoint pools of {n_end}: share of the draws whose arms are clear of each other {json.dumps({k: round(v, 4) for k, v in accept.items()})}; '
        f'zone pairs ({"|".join(arm_names)}) {json.dumps(zone_pairs)}; {timing["pools"]:.0f}s')
    pgen = torch.Generator(dev).manual_seed(M.SEEDS['pair'])
    rgen = torch.Generator(dev).manual_seed(M.SEEDS['rrt'])
    state = {s[0]: dict(rows=[], status=[]) for s in strata}
    report = dict(scene=name, rounds=[], pair_in_range_share={})
    for r in range(a.max_rounds):
        have = {s[0]: sum(x == 'certified' for x in state[s[0]]['status']) for s in strata}
        todo = [s for s in strata if have[s[0]] < per]
        if not todo:
            break
        S, G, ZS, ZG, labels, sizes_r = [], [], [], [], [], {}
        for stratum, start, goal, relation in todo:
            n_seen = len(state[stratum]['status'])
            rate = max((have[stratum] + 1) / (n_seen + 2), 0.03) if n_seen else PRIOR[stratum]
            size = int(min(a.round_max, max(16, math.ceil(1.3 * (per - have[stratum]) / rate))))
            sizes_r[stratum] = size
            ij, share = M.candidates(scene, ends[start], ends[goal], relation, size, pgen, max_l2)
            report['pair_in_range_share'][stratum] = share
            S.append(ends[start][0][ij[:, 0]])
            G.append(ends[goal][0][ij[:, 1]])
            ZS.append(ends[start][1][ij[:, 0]])
            ZG.append(ends[goal][1][ij[:, 1]])
            labels += [stratum] * len(ij)
        starts = (torch.cat(S) * scene.q_range).cpu().numpy().astype(np.float32)
        goals = (torch.cat(G) * scene.q_range).cpu().numpy().astype(np.float32)
        zs, zg = torch.cat(ZS).cpu().numpy(), torch.cat(ZG).cpu().numpy()
        endpoint_ok = M.certify(scene, [starts[i][None] for i in range(len(starts))]) & M.certify(scene, [goals[i][None] for i in range(len(goals))])
        line = M.certify(scene, [np.stack([s, g]) for s, g in zip(starts, goals)]) & endpoint_ok
        todo_ids = np.flatnonzero(~line & endpoint_ok)
        begin = time.monotonic()
        found, used = M.rrt_connect(V, torch.tensor(starts[todo_ids], device=dev), torch.tensor(goals[todo_ids], device=dev), targets, rgen, iterations, capacity=capacity, log=log)
        paths, source = [None] * len(starts), [''] * len(starts)
        for j, i in enumerate(todo_ids):
            paths[i], source[i] = found[j], 'rrtc'
        left = np.array([i for j, i in enumerate(todo_ids) if found[j] is None], int)
        used2, found2 = 0, []
        if scene.A == 2 and len(left):
            found2, used2 = M.one_arm_at_a_time(V, torch.tensor(starts[left], device=dev), torch.tensor(goals[left], device=dev), targets, rgen, iterations, capacity, log=log)
            for j, i in enumerate(left):
                paths[i], source[i] = found2[j], 'rrtc_one_arm_at_a_time'
        checked = M.certify(scene, paths)
        report['rounds'].append(dict(round=r, sizes=sizes_r, candidates=len(starts), endpoints_valid=int(endpoint_ok.sum()), direct=int(line.sum()), rrt_queries=len(todo_ids), rrt_found=sum(x is not None for x in found),
                                     one_arm_queries=len(left), one_arm_found=sum(x is not None for x in found2), one_arm_iterations=used2,
                                     rrt_certified=int(checked.sum()), rrt_iterations=used, rrt_seconds=time.monotonic() - begin))
        log('ROUND ' + json.dumps(report['rounds'][-1]))
        for i in range(len(starts)):
            if line[i]:
                status, witness, src = 'certified', np.stack([starts[i], goals[i]]), 'direct'
            elif checked[i]:
                status, witness, src = 'certified', paths[i], source[i]
            else:
                status, witness, src = ('uncertified' if paths[i] is None else 'rejected_by_scorer'), None, ''
            state[labels[i]]['rows'].append(dict(start=starts[i], goal=goals[i], zs=zs[i], zg=zg[i], round=r, witness=witness, source=src, direct=bool(line[i])))
            state[labels[i]]['status'].append(status)
    select = np.random.default_rng(M.SEEDS['select'])
    manifest = dict(scene=name, scene_sha256=M.sha(a.scene), generator_sha256=M.sha(Path(__file__)), base_generator_sha256=M.sha(Path(M.__file__)), seeds=M.SEEDS, smoke=a.smoke, arms=arm_names, total=0, suites=[])
    report['strata'] = {}
    validator = json.dumps(dict(step=M.STEP, obs_margin=A.OBS_VALID_M, self_margin=A.SELF_VALID_M, inter_arm_margin=A.INTER_VALID_M, rule='arm12.ArmSet valid, core.score'))
    region = lambda pool, z: '|'.join(zone_names[k] if pool == 'compartment' else pool for k in z)  # noqa: E731
    with torch.no_grad():
        for stratum, start, goal, relation in strata:
            rows, status = state[stratum]['rows'], np.array(state[stratum]['status'])
            ok = np.flatnonzero(status == 'certified')
            assert len(ok) >= per or (a.smoke and len(ok)), f'{stratum}: only {len(ok)} certified of {len(rows)} candidates'
            keep = select.choice(ok, min(per, len(ok)), replace=False)
            q = [rows[i] for i in keep]
            n = len(q)
            starts, goals = np.stack([x['start'] for x in q]).astype(np.float32), np.stack([x['goal'] for x in q]).astype(np.float32)
            es = scene.ee(torch.tensor(starts / qr, device=dev)).cpu().numpy().astype(np.float32)
            eg = scene.ee(torch.tensor(goals / qr, device=dev)).cpu().numpy().astype(np.float32)
            witnesses = np.empty(n, dtype=object)
            for i, x in enumerate(q):
                witnesses[i] = x['witness']
            wlen = np.array([np.linalg.norm(np.diff(x['witness'], axis=0), axis=1).sum() for x in q], np.float32)
            fields = dict(starts_rad=starts, goals_rad=goals, starts_norm=starts / qr, goals_norm=goals / qr, ee_start=es, ee_goal=eg,
                          start_region=np.array([region(start, x['zs']) for x in q]), goal_region=np.array([region(goal, x['zg']) for x in q]),
                          collision_margin=np.array(A.OBS_VALID_M), self_collision_margin=np.array(A.SELF_VALID_M), inter_arm_margin=np.array(A.INTER_VALID_M),
                          orientation_rule=np.array('none'), min_cspace_l2=np.array(M.MIN_L2), max_cspace_l2=np.array(max_l2), seed=np.array(M.SEEDS['pair']),
                          suite_name=np.array(f'{name}_{stratum}'), scene=np.array(f'jeffrey/arm_placement/scenes/{name}.yml'), scene_sha256=np.array(M.sha(a.scene)),
                          sampling_box=scene.box.cpu().numpy(), certified=np.array(True), witness_trajectories=witnesses, witness_source=np.array([x['source'] for x in q]),
                          witness_cspace_length=wlen, witness_n_waypoints=np.array([len(x['witness']) for x in q], np.int32), candidate_index=keep.astype(np.int32),
                          n_candidates=np.array(len(rows)), n_certified=np.array(len(ok)), select_seed=np.array(M.SEEDS['select']), validator=np.array(validator),
                          query_group_label=np.array([stratum] * n), direct_line_strict_feasible=np.array([x['direct'] for x in q]), start_pool=np.array(start), goal_pool=np.array(goal),
                          arm_names=np.array(arm_names))
            path = out / 'suites' / f'{name}_{stratum}_{n}.npz'
            np.savez(path, **fields)
            counts = {k: int(v) for k, v in zip(*np.unique(status, return_counts=True))}
            report['strata'][stratum] = dict(start_pool=start, goal_pool=goal, relation=relation, candidates=len(rows), **counts, certified_direct=int(sum(rows[i]['direct'] for i in ok)), certified_one_arm_at_a_time=int(sum(rows[i]['source'] == 'rrtc_one_arm_at_a_time' for i in ok)),
                                             selected_sources={k: int(v) for k, v in zip(*np.unique(fields['witness_source'], return_counts=True))},
                                             selected_direct=int(fields['direct_line_strict_feasible'].sum()), selected_distance_rad_quantiles=np.quantile(np.linalg.norm(starts - goals, axis=1), [0, .1, .5, .9, 1]).tolist(),
                                             selected_witness_length_rad_median=float(np.median(wlen)))
            manifest['suites'].append(dict(task=stratum, file=path.name, sha256=M.sha(path), n=n, keys=sorted(fields)))
            manifest['total'] += n
    for stratum in state:
        rows = state[stratum]['rows']
        np.savez(out / 'work' / f'candidates_{stratum}.npz', starts=np.stack([x['start'] for x in rows]), goals=np.stack([x['goal'] for x in rows]),
                 status=np.array(state[stratum]['status']), direct=np.array([x['direct'] for x in rows]), round=np.array([x['round'] for x in rows]))
    timing['total'] = time.monotonic() - started
    report.update(arm_pool_sizes=sizes, zones_reached={arm_names[i]: [zone_names[k] for k in z] for i, z in enumerate(zones_ok)}, endpoint_accept=accept, zone_pairs=zone_pairs, rrt_targets=len(targets), timing=timing,
                  parameters=dict(samples_per_arm=samples, endpoint_pool=n_end, round_max=a.round_max, max_rounds=a.max_rounds, rrt_iterations=iterations, rrt_step=.2, rrt_capacity=capacity,
                                  goal_bias=.1, min_cspace_l2=M.MIN_L2, max_cspace_l2=max_l2, per_stratum=per, certificate_step_rad=M.STEP))
    (out / 'suite_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'generation.json').write_text(json.dumps(dict(report, complete=True), indent=2) + '\n')
    log(json.dumps(report['strata'], indent=1))


if __name__ == '__main__':
    main()
