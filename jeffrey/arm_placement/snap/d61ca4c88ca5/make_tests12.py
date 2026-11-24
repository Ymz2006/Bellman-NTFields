"""Certified test suites for a scene with several arms (real_world_12dof: two arms, 12 joints). Host venv.

    CUDA_VISIBLE_DEVICES=1 python make_tests12.py --export <export run>/robot_model.npz --scene scenes/real_world_12dof.yml --output <dir>

The cabinet recipe (experiments/cabinet/make_tests.py) carried to A arms. What is the same: strict endpoints, random pairs, a witness
path for every kept query (the straight joint-space line, or bidirectional RRT-Connect; never a learned method), certification of
every witness at 0.001 rad per joint, 100 queries per stratum chosen at random among the certified candidates, six strata, the seeds.
What differs, because the release scorer and cuRobo's robot are one arm:

- Validity = arm12.ArmSet: each arm by the release scorer's rule (every sphere >= 0.005 m from every cuboid, self distance >= 0.002 m,
  joint limits) and the arms >= 0.005 m from each other (all sphere pairs). The certificate is core.score with that rule at 0.001 rad
  per joint, the function that scores the planners.
- Endpoints: per arm, uniform joint samples valid for that arm alone, by the region of its grasp point (`free` = sampling box outside
  the cupboard's outer box, `compartment` = one of the zone interiors, a zone drawn uniformly, `box` = anywhere in the sampling box).
  An endpoint takes one configuration per arm from the same kind of region and is kept if the arms are clear of each other.
- Strata, the regions holding for every arm: free_to_free, free_to_compartment, compartment_to_free, compartment_to_compartment (every
  arm changes zone), same_compartment (every arm stays in its zone), anywhere.
- Pairs: joint-space distance 0.02 to 4 sqrt(A) rad (the cabinet suites' 4 rad per arm), uniform among the pairs of random blocks
  of the endpoint pools that are in range.
- RRT-Connect is experiments/obstacle_scenes/make_tests.py's (rrt_connect, Validator.prefix / reach), copied with the dimension taken
  from the query; its random targets are valid configurations with every grasp point within 0.2 m of the sampling box.
- Witness search, in this order: the straight line; RRT-Connect in all joints; for two arms, one arm at a time (one arm moves from its
  start to its goal by RRT-Connect in its six joints while the other is held, then the other arm; either order). In all joints
  RRT-Connect with the cabinet budget finds few witnesses (about 1 in 8 in a probe), which would leave only the easiest queries.
"""
import argparse
import hashlib
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

STEP, COARSE = 1e-3, 1e-2
MIN_L2, MAX_L2_PER_ARM = 0.02, 4.0
SEEDS = dict(pool=20261014, pair=20261015, rrt=20261016, select=20261017)
STRATA = [('free_to_free', 'free', 'free', None), ('free_to_compartment', 'free', 'compartment', None), ('compartment_to_free', 'compartment', 'free', None),
          ('compartment_to_compartment', 'compartment', 'compartment', 'different'), ('same_compartment', 'compartment', 'compartment', 'same'), ('anywhere', 'box', 'box', None)]
WIDEN = 0.2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Validator:
    """The scene's valid rule on configurations in radians, batched (obstacle_scenes/make_tests.py Validator's prefix and reach)."""

    def __init__(self, scene, compile=True):
        self.scene, self.dev = scene, scene.device
        holder = type('Rule', (), dict(signed_cells=staticmethod(scene.valid_margin)))
        self.margin = C.Clearance(holder, chunk=1 << 17, compile=compile)
        self.lo, self.hi, self.qr = scene.lo * scene.q_range, scene.hi * scene.q_range, scene.q_range

    @torch.no_grad()
    def __call__(self, q):
        return (self.margin(q / self.qr) >= 0) & ((q >= self.lo - 1e-5) & (q <= self.hi + 1e-5)).all(1)

    def prefix(self, qa, qb, step):
        """Leading valid points of each segment densified like the scorer (k/n, k=1..n), and n."""
        n = torch.ceil((qb - qa).abs().amax(1) / step).clamp(min=1).long()
        seg = torch.repeat_interleave(torch.arange(len(n), device=self.dev), n)
        k = torch.arange(len(seg), device=self.dev) - (torch.cumsum(n, 0) - n)[seg] + 1
        ok = self(qa[seg] + (qb - qa)[seg] * (k / n[seg]).unsqueeze(1))
        return (n + 1).scatter_reduce(0, seg, torch.where(ok, n[seg] + 1, k), reduce='amin') - 1, n

    def reach(self, qa, qb):
        """Valid fraction of each segment from qa: a coarse pass, then the scorer's resolution."""
        kc, nc = self.prefix(qa, qb, COARSE)
        fc = kc / nc
        f = torch.zeros_like(fc)
        sel = kc > 0
        if sel.any():
            kf, nf = self.prefix(qa[sel], qa[sel] + (qb[sel] - qa[sel]) * fc[sel, None], STEP)
            f[sel] = fc[sel] * kf / nf
        return f


def nearest(nodes, count, target):
    d = (nodes - target[:, None]).square().sum(2)
    d.masked_fill_(torch.arange(nodes.shape[1], device=nodes.device)[None] >= count[:, None], float('inf'))
    i = d.argmin(1)
    return i, nodes[torch.arange(len(i), device=nodes.device), i]


@torch.no_grad()
def rrt_connect(V, starts, goals, targets, generator, iterations, eps=.2, capacity=4000, goal_bias=.1, log=None, moving=None):
    """Bidirectional RRT-Connect for a batch of queries (radians). Every tree edge passes Validator.reach.
    moving (B, D) bool: the joints a query may move (its start and goal agree on the others); the random targets take the start's
    value on the others, so both trees stay in that subspace."""
    dev = starts.device
    B, D = starts.shape
    tree = torch.zeros(2, B, capacity, D, device=dev)
    parent = torch.full((2, B, capacity), -1, dtype=torch.long, device=dev)
    count = torch.ones(2, B, dtype=torch.long, device=dev)
    tree[0, :, 0] = starts
    tree[1, :, 0] = goals
    link = torch.full((B, 2), -1, dtype=torch.long, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    it = -1
    t0 = time.time()
    for it in range(iterations):
        a, b = it % 2, 1 - it % 2
        ids = torch.nonzero(~done & (count.amax(0) < capacity - 1)).squeeze(1)
        if not len(ids):
            break
        if log is not None and it % 500 == 0:
            log(f'  rrt iteration {it}: {int(done.sum())}/{B} connected, {time.time() - t0:.0f}s')
        target = targets[torch.randint(len(targets), (len(ids),), device=dev, generator=generator)]
        if moving is not None:
            target = torch.where(moving[ids], target, tree[0, ids, 0])
        bias = torch.rand(len(ids), device=dev, generator=generator) < goal_bias
        target = torch.where(bias[:, None], tree[b, ids, 0], target)
        near, qn = nearest(tree[a, ids], count[a, ids], target)
        step = target - qn
        qs = qn + step * (eps / step.norm(dim=1, keepdim=True).clamp_min(1e-9)).clamp(max=1)
        qnew = qn + (qs - qn) * V.reach(qn, qs)[:, None]
        grew = (qnew - qn).abs().amax(1) >= 5 * STEP
        if not grew.any():
            continue
        ids, near, qnew = ids[grew], near[grew], qnew[grew]
        slot = count[a, ids]
        tree[a, ids, slot] = qnew
        parent[a, ids, slot] = near
        count[a, ids] += 1
        # Connect: the other tree grows straight towards the new node as far as it is valid.
        other, qo = nearest(tree[b, ids], count[b, ids], qnew)
        f = V.reach(qo, qnew)
        full = f >= 1
        link[ids[full], a] = slot[full]
        link[ids[full], b] = other[full]
        done[ids[full]] = True
        q2 = qo + (qnew - qo) * f[:, None]
        part = ~full & ((q2 - qo).abs().amax(1) >= 5 * STEP)
        if part.any():
            p = ids[part]
            s2 = count[b, p]
            tree[b, p, s2] = q2[part]
            parent[b, p, s2] = other[part]
            count[b, p] += 1
    L = link.cpu().numpy()
    paths = [None] * B
    for i in np.flatnonzero(done.cpu().numpy()):
        T, P = tree[:, i].cpu().numpy(), parent[:, i].cpu().numpy()          # one query's trees at a time (host memory)
        branch = []
        for t in (0, 1):
            node, chain = L[i, t], []
            while node >= 0:
                chain.append(T[t, node])
                node = P[t, node]
            branch.append(chain[::-1] if t == 0 else chain)
        paths[i] = np.array(branch[0] + branch[1], dtype=np.float32)
    return paths, int(it) + 1


@torch.no_grad()
def one_arm_at_a_time(V, starts, goals, targets, generator, iterations, capacity, log=None):
    """Witness for two arms by moving one arm, then the other (either order): each leg is RRT-Connect in the moving arm's six joints
    with the other arm held, checked in the whole scene. Returns paths (None = no witness) and the iterations used."""
    dev, n = starts.device, len(starts)
    arm0 = torch.arange(starts.shape[1], device=dev) < 6
    S, G, M, owner = [], [], [], []
    for order in (0, 1):
        first = arm0 if order == 0 else ~arm0
        mid = torch.where(first, goals, starts)
        ok = torch.nonzero(V(mid)).squeeze(1)
        for leg, (qa, qb, mv) in enumerate(((starts, mid, first), (mid, goals, ~first))):
            S.append(qa[ok])
            G.append(qb[ok])
            M.append(mv[None].expand(len(ok), -1))
            owner.append(torch.stack([ok, torch.full_like(ok, order), torch.full_like(ok, leg)], 1))
    owner = torch.cat(owner).cpu().numpy()
    paths = [None] * n
    if not len(owner):
        return paths, 0
    found, used = rrt_connect(V, torch.cat(S), torch.cat(G), targets, generator, iterations, capacity=capacity, log=log, moving=torch.cat(M))
    legs = {(int(q), int(o), int(l)): f for (q, o, l), f in zip(owner, found)}
    for q in range(n):
        for order in (0, 1):
            a, b = legs.get((q, order, 0)), legs.get((q, order, 1))
            if a is not None and b is not None:
                paths[q] = np.concatenate([a, b[1:]])
                break
    return paths, used


def arm_pools(scene, samples, gen, cap, batch=1 << 20):
    """Per arm: configurations valid for that arm alone (normalized, (n, 6)), by the region of the grasp point."""
    box, cup = scene.box, scene.cupboard
    comp = scene.compartments
    wide = torch.stack([box[:, 0] - WIDEN, box[:, 1] + WIDEN], 1)
    ins = lambda e, b: ((e >= b[:, 0]) & (e <= b[:, 1])).all(1)  # noqa: E731
    names = ['free', 'box', 'near'] + [f'zone{k}' for k in range(len(comp))]
    pools = []
    for a in range(scene.A):
        found = {k: [] for k in names}
        have = {k: 0 for k in names}
        for _ in range(samples // batch):
            x = scene.lo[:6] + (scene.hi[:6] - scene.lo[:6]) * torch.rand(batch, 6, generator=gen, device=scene.device)
            keep = []
            for i in range(0, batch, 1 << 17):
                v, e = scene.arm_parts(a, x[i:i + (1 << 17)])
                keep.append(torch.cat([e, (v >= 0).float()[:, None]], 1))
            keep = torch.cat(keep)
            ok = keep[:, 3] > 0
            x, e = x[ok], keep[ok, :3]
            inb = ins(e, box)
            masks = dict(free=inb & ~ins(e, cup), box=inb, near=ins(e, wide), **{f'zone{k}': ins(e, comp[k]) for k in range(len(comp))})
            for k, m in masks.items():
                if have[k] < cap and m.any():
                    found[k].append(x[m])
                    have[k] += int(m.sum())
        pools.append({k: (torch.cat(v)[:cap] if v else torch.zeros(0, 6, device=scene.device)) for k, v in found.items()})
    return pools


@torch.no_grad()
def endpoint_pool(scene, pools, kind, count, gen, batch=1 << 18):
    """count valid configurations of the whole scene, every arm drawn from its `kind` pool; zone ids (count, A) (-1 outside zones)."""
    dev = scene.device
    nz = len(scene.compartments)
    X, Z, drawn = [], [], 0
    while sum(len(x) for x in X) < count:
        cols, zs = [], []
        for a in range(scene.A):
            if kind == 'compartment':
                z = torch.randint(nz, (batch,), generator=gen, device=dev)
                col = torch.zeros(batch, 6, device=dev)
                for k in range(nz):
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
    return torch.cat(X)[:count], torch.cat(Z)[:count], sum(len(x) for x in X) / drawn


@torch.no_grad()
def candidates(scene, start, goal, relation, count, gen, max_l2, block=4096, max_blocks=4000):
    """count (start index, goal index) pairs, uniform among the in-range pairs of random blocks of the two endpoint pools."""
    (xs, zs), (xg, zg) = start, goal
    dev = xs.device
    got, n, blocks = [], 0, 0
    while n < 4 * count:
        i = torch.randint(len(xs), (block,), generator=gen, device=dev)
        j = torch.randint(len(xg), (block,), generator=gen, device=dev)
        d = torch.cdist((xs[i] * scene.q_range).double(), (xg[j] * scene.q_range).double())
        ok = (d >= MIN_L2) & (d <= max_l2)
        if relation == 'same':
            ok &= (zs[i][:, None, :] == zg[j][None, :, :]).all(2)
        elif relation == 'different':
            ok &= (zs[i][:, None, :] != zg[j][None, :, :]).all(2)
        a, b = torch.nonzero(ok, as_tuple=True)
        got.append(torch.stack([i[a], j[b]], 1))
        n += len(a)
        blocks += 1
        assert blocks < max_blocks, f'{n} pairs in range after {blocks} blocks'
    got = torch.cat(got)
    got = got[torch.randperm(len(got), generator=gen, device=dev)[:count]]
    return got, n / (blocks * block * block)


def certify(scene, paths):
    """core.score at 0.001 rad per joint on paths in radians (None = no path): success flags."""
    qr = scene.q_range.cpu().numpy()
    ids = [i for i, p in enumerate(paths) if p is not None]
    out = np.zeros(len(paths), bool)
    if ids:
        rows = C.score(scene, [torch.tensor(np.asarray(paths[i], np.float32) / qr, device=scene.device) for i in ids], np.ones(len(ids), bool), step_rad=STEP)
        out[ids] = [r['success'] for r in rows]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--export', type=Path, required=True, help='robot_model.npz of an export run (the arm)')
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--probe', action='store_true', help='one small round of every stratum with a real RRT budget (rates and timing; not a suite to freeze)')
    p.add_argument('--no-compile', action='store_true')
    a = p.parse_args()
    started = time.monotonic()
    dev = 'cuda:0'
    name = a.scene.stem
    out = a.output
    (out / 'suites').mkdir(parents=True, exist_ok=False)
    (out / 'work').mkdir()

    def log(msg):
        print(msg, flush=True)
        with open(out / 'log.txt', 'a') as f:
            f.write(msg + '\n')

    scene = A.ArmSet(A.build_model(a.export, a.scene, out / 'robot_model.npz'), dev)
    m = np.load(out / 'robot_model.npz')
    scene.compartments = torch.tensor(m['compartments'], device=dev)
    zone_names = [str(v) for v in m['compartment_names']]
    arm_names = [str(v) for v in m['arm_names']]
    max_l2 = MAX_L2_PER_ARM * math.sqrt(scene.A)
    per, round_size, max_rounds = (3, 16, 2) if a.smoke else (100, 160, 24)
    samples, n_end, iterations, capacity = ((1 << 22), 20_000, 3000, 3100) if a.smoke else ((1 << 26), 300_000, 10000, 10500)
    strata = [STRATA[1], STRATA[4]] if a.smoke else STRATA
    if a.probe:
        per, round_size, max_rounds, samples, n_end, iterations, capacity, a.smoke = 5, 32, 1, 1 << 24, 100_000, 10000, 10500, True
    qr = scene.q_range.cpu().numpy()
    V = Validator(scene, compile=not a.no_compile)
    gen = torch.Generator(dev).manual_seed(SEEDS['pool'])
    pools = arm_pools(scene, samples, gen, 400_000)
    sizes = {arm_names[i]: {k: len(v) for k, v in pl.items()} for i, pl in enumerate(pools)}
    log(f'arm pools ({samples} uniform samples per arm): {json.dumps(sizes)}')
    ends, accept = {}, {}
    for kind in ('free', 'compartment', 'box', 'near'):
        x, z, acc = endpoint_pool(scene, pools, kind, n_end, gen)
        ends[kind], accept[kind] = (x, z), acc
    zc = ends['compartment'][1]
    zone_pairs = {f'{zone_names[i]}|{zone_names[j]}': int(((zc[:, 0] == i) & (zc[:, 1] == j)).sum()) for i in range(len(zone_names)) for j in range(len(zone_names))} if scene.A == 2 else {}
    targets = ends['near'][0] * scene.q_range
    timing = dict(pools=time.monotonic() - started)
    log(f'endpoint pools of {n_end}: share of the draws whose arms are clear of each other {json.dumps({k: round(v, 4) for k, v in accept.items()})}; '
        f'zone pairs ({"|".join(arm_names)}) {json.dumps(zone_pairs)}; {timing["pools"]:.0f}s')
    pgen = torch.Generator(dev).manual_seed(SEEDS['pair'])
    rgen = torch.Generator(dev).manual_seed(SEEDS['rrt'])
    state = {s[0]: dict(rows=[], status=[]) for s in strata}
    report = dict(scene=name, rounds=[], pair_in_range_share={})
    for r in range(max_rounds):
        todo = [s for s in strata if sum(x == 'certified' for x in state[s[0]]['status']) < per]
        if not todo:
            break
        S, G, ZS, ZG, labels = [], [], [], [], []
        for stratum, start, goal, relation in todo:
            ij, share = candidates(scene, ends[start], ends[goal], relation, round_size, pgen, max_l2)
            report['pair_in_range_share'][stratum] = share
            S.append(ends[start][0][ij[:, 0]])
            G.append(ends[goal][0][ij[:, 1]])
            ZS.append(ends[start][1][ij[:, 0]])
            ZG.append(ends[goal][1][ij[:, 1]])
            labels += [stratum] * len(ij)
        # endpoints as stored: float32 radians
        starts = (torch.cat(S) * scene.q_range).cpu().numpy().astype(np.float32)
        goals = (torch.cat(G) * scene.q_range).cpu().numpy().astype(np.float32)
        zs, zg = torch.cat(ZS).cpu().numpy(), torch.cat(ZG).cpu().numpy()
        endpoint_ok = certify(scene, [starts[i][None] for i in range(len(starts))]) & certify(scene, [goals[i][None] for i in range(len(goals))])
        line = certify(scene, [np.stack([s, g]) for s, g in zip(starts, goals)]) & endpoint_ok
        todo_ids = np.flatnonzero(~line & endpoint_ok)
        begin = time.monotonic()
        found, used = rrt_connect(V, torch.tensor(starts[todo_ids], device=dev), torch.tensor(goals[todo_ids], device=dev), targets, rgen, iterations, capacity=capacity, log=log)
        paths, source = [None] * len(starts), [''] * len(starts)
        for j, i in enumerate(todo_ids):
            paths[i], source[i] = found[j], 'rrtc'
        left = np.array([i for j, i in enumerate(todo_ids) if found[j] is None], int)
        used2, found2 = 0, []
        if scene.A == 2 and len(left):
            found2, used2 = one_arm_at_a_time(V, torch.tensor(starts[left], device=dev), torch.tensor(goals[left], device=dev), targets, rgen, iterations, capacity, log=log)
            for j, i in enumerate(left):
                paths[i], source[i] = found2[j], 'rrtc_one_arm_at_a_time'
        checked = certify(scene, paths)
        report['rounds'].append(dict(round=r, candidates=len(starts), endpoints_valid=int(endpoint_ok.sum()), direct=int(line.sum()), rrt_queries=len(todo_ids), rrt_found=sum(x is not None for x in found),
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
    select = np.random.default_rng(SEEDS['select'])
    manifest = dict(scene=name, scene_sha256=sha(a.scene), generator_sha256=sha(Path(__file__)), seeds=SEEDS, smoke=a.smoke, arms=arm_names, total=0, suites=[])
    report['strata'] = {}
    validator = json.dumps(dict(step=STEP, obs_margin=A.OBS_VALID_M, self_margin=A.SELF_VALID_M, inter_arm_margin=A.INTER_VALID_M, rule='arm12.ArmSet valid, core.score'))
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
                          orientation_rule=np.array('none'), min_cspace_l2=np.array(MIN_L2), max_cspace_l2=np.array(max_l2), seed=np.array(SEEDS['pair']),
                          suite_name=np.array(f'{name}_{stratum}'), scene=np.array(f'jeffrey/lbs_arm/scenes/{name}.yml'), scene_sha256=np.array(sha(a.scene)),
                          sampling_box=scene.box.cpu().numpy(), certified=np.array(True), witness_trajectories=witnesses, witness_source=np.array([x['source'] for x in q]),
                          witness_cspace_length=wlen, witness_n_waypoints=np.array([len(x['witness']) for x in q], np.int32), candidate_index=keep.astype(np.int32),
                          n_candidates=np.array(len(rows)), n_certified=np.array(len(ok)), select_seed=np.array(SEEDS['select']), validator=np.array(validator),
                          query_group_label=np.array([stratum] * n), direct_line_strict_feasible=np.array([x['direct'] for x in q]), start_pool=np.array(start), goal_pool=np.array(goal),
                          arm_names=np.array(arm_names))
            path = out / 'suites' / f'{name}_{stratum}_{n}.npz'
            np.savez(path, **fields)
            counts = {k: int(v) for k, v in zip(*np.unique(status, return_counts=True))}
            report['strata'][stratum] = dict(start_pool=start, goal_pool=goal, relation=relation, candidates=len(rows), **counts, certified_direct=int(sum(rows[i]['direct'] for i in ok)), certified_one_arm_at_a_time=int(sum(rows[i]['source'] == 'rrtc_one_arm_at_a_time' for i in ok)),
                                             selected_sources={k: int(v) for k, v in zip(*np.unique(fields['witness_source'], return_counts=True))},
                                             selected_direct=int(fields['direct_line_strict_feasible'].sum()), selected_distance_rad_quantiles=np.quantile(np.linalg.norm(starts - goals, axis=1), [0, .1, .5, .9, 1]).tolist(),
                                             selected_witness_length_rad_median=float(np.median(wlen)))
            manifest['suites'].append(dict(task=stratum, file=path.name, sha256=sha(path), n=n, keys=sorted(fields)))
            manifest['total'] += n
    for stratum in state:
        rows = state[stratum]['rows']
        np.savez(out / 'work' / f'candidates_{stratum}.npz', starts=np.stack([x['start'] for x in rows]), goals=np.stack([x['goal'] for x in rows]),
                 status=np.array(state[stratum]['status']), direct=np.array([x['direct'] for x in rows]), round=np.array([x['round'] for x in rows]))
    timing['total'] = time.monotonic() - started
    report.update(arm_pool_sizes=sizes, endpoint_accept=accept, zone_pairs=zone_pairs, rrt_targets=len(targets), timing=timing,
                  parameters=dict(samples_per_arm=samples, endpoint_pool=n_end, round_size=round_size, max_rounds=max_rounds, rrt_iterations=iterations, rrt_step=.2, rrt_capacity=capacity,
                                  goal_bias=.1, min_cspace_l2=MIN_L2, max_cspace_l2=max_l2, per_stratum=per, certificate_step_rad=STEP))
    (out / 'suite_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'generation.json').write_text(json.dumps(dict(report, complete=True), indent=2) + '\n')
    log(json.dumps(report['strata'], indent=1))


if __name__ == '__main__':
    main()
