"""Test 4 training pairs (jeffrey/joint_range/README.md, Test 4): mixed pairs, one grasp point inside the cupboard and
the other outside, plus collision pairs, whose first configuration touches or is inside the cupboard or the table.

mixed      data.py's in-area recipe in release units, as cabinet_data.py: x0 uniform over the joint limits with its grasp
           point in the sampling box and 0 < d - 0.01 < margin (release label d = min(obstacle distance, 0.5 x self
           distance)); x1 = x0 + unit(rand - 0.5) * U(0, 0.5) inside the joint limits with its grasp point in the
           sampling box. Kept when exactly one of the two grasp points is in the cupboard's outer box (cabinet_data.py:
           at least one).
collision  x0 uniform over the joint limits with its grasp point in the sampling box and the label's obstacle distance
           <= 0: the arm touches or is inside the cupboard or the table (self contact alone does not count; it does not
           depend on the cabinet and the random steps reach it often). x1 is the recipe's random step from x0, as above.
Labels as in the recipe for every end: speed clip(d, offset, margin) / margin, so each colliding end (every collision x0)
gets the lowest speed, offset / margin = 0.1; normals by central differences (eps 1e-4); pairs with a NaN or infinite
normal are dropped (data.py).

Sets in <output>/ (release coordinates, the full-range scaling; the data.py file set plus source.npy, 0 mixed / 1 collision):
  test4/    --pairs pairs: round(pairs / 3) collision pairs and the rest mixed, shuffled (seed)
  control/  --pairs mixed pairs (test4's mixed pairs first, then the next ones), shuffled (seed)
  heldout/  --heldout mixed and --heldout collision pairs that neither set contains, for field checks
Speed-ups as cabinet_data.py (none changes which pairs are accepted): a URDF grasp-frame pre-screen (experiments/face_speed/
robot.py; final tests on cuRobo positions; start-up check of the two kinematics), --batch uniform draws per step, one
worker per visible GPU (seeds seed + 1000 k); the parent merges and shuffles.
"""
import argparse, importlib.util, json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, '/clean/method'); sys.path.insert(0, '/clean')
from common import sha, write
import numpy as np
import torch
_spec = importlib.util.spec_from_file_location('ntrl_cabinet_data', '/clean/experiments/ntrl_cabinet/data.py')
D = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(D)
FILES = ['sampled_points', 'speed', 'normal', 'clearance', 'source']


def inside(pos, b):
    return ((pos >= b[:, 0]) & (pos <= b[:, 1])).all(1)


class Labels(D.Clearance):
    """The release label (D.Clearance, unchanged) plus its obstacle term and per-group obstacle distances."""

    @torch.no_grad()
    def parts(self, q, groups):
        out = []
        for i in range(0, len(q), 25000):                  # half data.py's chunk: three obstacle terms besides the self distances
            st = self.robot.get_state(q[i:i+25000]); sp = st.link_spheres_tensor; pos = st.ee_pose.position.clone()
            ob = [D.geom.sphere_cuboid_distance(sp[:, :, :3], sp[:, :, 3], c, h, ignore_sphere_indices=D.geom.OBSTACLE_IGNORE_SPHERE_INDICES) for c, h in [(self.centers, self.half)]+groups]
            d = torch.minimum(ob[0], D.SELF_SCALE*D.geom.sphere_sphere_distance(sp[:, :, :3], sp[:, :, 3], self.mask))
            out.append([d, pos]+ob)
        return [torch.cat(x) for x in zip(*out)]


def worker(a, k):
    device = 'cuda:0'; torch.manual_seed(a.seed+1000*k); started = time.monotonic()
    from experiments.face_speed import robot as R
    box, cupboard, _ = D.scene_boxes(a.scene); box_t = torch.tensor(box, device=device); cup_t = torch.tensor(cupboard, device=device)
    grow = lambda b: b+torch.tensor([[-a.screen_margin, a.screen_margin]], device=device)
    box_s, cup_s = grow(box_t), grow(cup_t)
    unit, qmax = D.units('release'); unit_t = torch.tensor(unit, device=device); bound = torch.tensor(qmax/unit, device=device)
    C = Labels(a.scene, 'release', device); steps = torch.eye(6, device=device)*D.EPS; kin = R.Kinematics().to(device)
    names = [c['name'] for c in D.geom.load_scene_cuboids(str(a.scene))]
    assert len(names) == len(C.centers), 'cuboid order'
    table = torch.tensor([n == 'table' for n in names], device=device)
    groups = [(C.centers[table], C.half[table]), (C.centers[~table], C.half[~table])]       # table, cupboard panels

    @torch.no_grad()
    def ee(x):
        return torch.cat([kin.transforms(x[s:s+250000]*unit_t, [R.EE_LINK])[R.EE_LINK][:, :3, 3] for s in range(0, len(x), 250000)]) if len(x) else x[:, :3]

    def gradient(x):
        plus = torch.stack([C((x+s)*unit_t) for s in steps], 1); minus = torch.stack([C((x-s)*unit_t) for s in steps], 1)
        return (plus-minus)/(2*D.EPS)

    def step(x0):
        dP = torch.rand(len(x0), 6, device=device)-.5; rL = torch.rand(len(x0), 1, device=device)*a.step
        return x0+torch.nn.functional.normalize(dP, dim=1)*rL

    def normals(x0, x1):
        g0 = gradient(x0); g1 = gradient(x1)
        n0 = g0/torch.linalg.vector_norm(g0, dim=1, keepdim=True); n1 = g1/torch.linalg.vector_norm(g1, dim=1, keepdim=True)
        good = ~(n0.isnan().any(1) | n1.isnan().any(1) | n0.isinf().any(1) | n1.isinf().any(1))
        return torch.cat([n0, n1], 1), good

    # start-up check of the pre-screen against cuRobo's end-effector positions (as cabinet_data.py)
    xv = (torch.rand(500000, 6, device=device)-.5)*2*bound; _, pv = C(xv*unit_t, positions=True); fv = ee(xv)
    deviation = float(torch.linalg.vector_norm(pv-fv, dim=1).max()); missed = int((inside(pv, box_t) & ~inside(fv, box_s)).sum())
    assert deviation < a.screen_margin/10 and missed == 0, (deviation, missed)
    out = {}; stats = {}
    # the short collision stream first: after the ~2,000 batches of the mixed stream the 12 GB card had no room left for it
    for stream, target in (('collision', a.collision_part), ('mixed', a.mixed_part)):
        xs, ds, ns = [], [], []; have = 0; drawn = 0; s = stats[stream] = dict(screen_pass=0, x0_in_box=0, x0_accepted=0, x1_inside_domain=0, accepted=0, x0_cupboard=0, x1_cupboard=0)
        if stream == 'collision': s.update(x0_hits_cupboard=0, x0_hits_table=0, x0_hits_both=0, x0_grasp_in_cupboard=0)
        while have < target:
            x0 = (torch.rand(a.batch, 6, device=device)-.5)*2*bound; drawn += a.batch
            x0 = x0[inside(ee(x0), box_s)]; s['screen_pass'] += len(x0)
            if stream == 'mixed':
                d0, p0 = C(x0*unit_t, positions=True); keep = inside(p0, box_t); s['x0_in_box'] += int(keep.sum())
                y0 = d0-.01; keep &= (y0 > 0) & (y0 < a.margin)
            else:
                d0, p0, ob, obt, obc = C.parts(x0*unit_t, groups); keep = inside(p0, box_t); s['x0_in_box'] += int(keep.sum())
                keep &= ob <= 0
            s['x0_accepted'] += int(keep.sum()); x0, d0, p0 = x0[keep], d0[keep], p0[keep]
            hits = (obt[keep] <= 0, obc[keep] <= 0) if stream == 'collision' else None
            x1 = step(x0); ok = ((x1 <= bound) & (x1 >= -bound)).all(1); s['x1_inside_domain'] += int(ok.sum())
            x0, x1, d0, p0 = x0[ok], x1[ok], d0[ok], p0[ok]; c0 = inside(p0, cup_t)
            if hits: hits = (hits[0][ok], hits[1][ok])
            f1 = ee(x1); ok = inside(f1, box_s)
            if stream == 'mixed': ok &= c0 | inside(f1, cup_s)               # exactly one inside needs x1 near the cupboard when x0 is outside
            x0, x1, d0, c0 = x0[ok], x1[ok], d0[ok], c0[ok]
            if hits: hits = (hits[0][ok], hits[1][ok])
            if len(x0) <= 1: continue
            d1, p1 = C(x1*unit_t, positions=True); c1 = inside(p1, cup_t); ok = inside(p1, box_t)
            if stream == 'mixed': ok &= c0 ^ c1
            x0, x1, d0, d1, c0, c1 = x0[ok], x1[ok], d0[ok], d1[ok], c0[ok], c1[ok]
            if hits: hits = (hits[0][ok], hits[1][ok])
            if len(x0) <= 1: continue
            n, good = normals(x0, x1)
            s['accepted'] += int(good.sum()); s['x0_cupboard'] += int((c0 & good).sum()); s['x1_cupboard'] += int((c1 & good).sum())
            if hits:
                t_, c_ = hits[0] & good, hits[1] & good
                s['x0_hits_table'] += int(t_.sum()); s['x0_hits_cupboard'] += int(c_.sum()); s['x0_hits_both'] += int((t_ & c_).sum()); s['x0_grasp_in_cupboard'] += int((c0 & good).sum())
            xs.append(torch.cat([x0, x1], 1)[good].cpu()); ds.append(torch.stack([d0, d1], 1)[good].cpu()); ns.append(n[good].cpu()); have += int(good.sum())
            if len(xs) % 20 == 0: print(json.dumps(dict(worker=k, stream=stream, drawn=drawn, collected=have, seconds=round(time.monotonic()-started))), flush=True)
        s['uniform_draws'] = drawn
        out[stream] = [torch.cat(v)[:target].numpy().astype(np.float32) for v in (xs, ds, ns)]; del xs, ds, ns; torch.cuda.empty_cache()
    np.savez(a.output/f'part{k}.npz', **{f'{st}_{f}': v for st, arrs in out.items() for f, v in zip('XDN', arrs)})
    write(a.output/f'part{k}.json', dict(worker=k, seed=a.seed+1000*k, gpu=torch.cuda.get_device_name(0), counts=stats, kinematics_deviation_m=deviation,
                                         prescreen_missed=missed, seconds=time.monotonic()-started))
    print('WORKER_DONE', k, json.dumps(stats), round(time.monotonic()-started), flush=True)


def save(folder, X, Dc, N, src, a, extra):
    """One training set in the data.py file layout (release coordinates), with its composition."""
    folder.mkdir(); speed = (np.clip(Dc, a.offset, a.margin)/a.margin).astype(np.float32)
    for f, v in zip(FILES, [X, speed, N, Dc, src.astype(np.int8)]): np.save(folder/(f+'.npy'), np.ascontiguousarray(v))
    from experiments.face_speed import robot as R
    unit, _ = D.units('release'); kin = R.Kinematics().to('cuda:0'); Xt = torch.as_tensor(X, device='cuda:0'); u = torch.tensor(unit, device='cuda:0')
    with torch.no_grad():
        pos = [torch.cat([kin.transforms(Xt[s:s+100000, c:c+6]*u, [R.EE_LINK])[R.EE_LINK][:, :3, 3] for s in range(0, len(Xt), 100000)]).cpu().numpy() for c in (0, 6)]
    box, cupboard, _ = D.scene_boxes(a.scene); ins = lambda q, b: ((q >= b[:, 0]) & (q <= b[:, 1])).all(1)
    c0, c1 = ins(pos[0], cupboard), ins(pos[1], cupboard); col = src == 1
    comp = dict(pairs=len(X), pairs_collision=int(col.sum()), pairs_mixed=int((~col).sum()), pair_with_cupboard_point=float((c0 | c1).mean()),
                pair_one_end_in_cupboard=float((c0 ^ c1).mean()), points_in_cupboard=float((c0.sum()+c1.sum())/(2*len(X))),
                points_in_box=float((ins(pos[0], box).sum()+ins(pos[1], box).sum())/(2*len(X))),
                endpoints_in_collision=float((Dc <= 0).mean()), endpoints_at_floor_speed=float((speed <= a.offset/a.margin+1e-7).mean()),
                x0_in_collision=float((Dc[:, 0] <= 0).mean()), x1_in_collision=float((Dc[:, 1] <= 0).mean()),
                mixed_x1_in_collision=float((Dc[~col, 1] <= 0).mean()) if (~col).any() else None, collision_x1_in_collision=float((Dc[col, 1] <= 0).mean()) if col.any() else None,
                speed_quantiles={str(t): float(np.quantile(speed, t)) for t in (.05, .25, .5, .75, .95)})
    write(folder/'complete.json', dict(complete=True, smoke=a.smoke, coords='full', center_rad=[0.0]*6, radians_per_unit=list(map(float, unit)), scene=str(a.scene),
                                       scene_sha256=sha(a.scene), label='release', margin=a.margin, offset=a.offset, **comp, **extra,
                                       files={f: sha(folder/(f+'.npy')) for f in FILES}))
    return comp


def main():
    p = argparse.ArgumentParser(); p.add_argument('--scene', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pairs', type=int, default=200000); p.add_argument('--heldout', type=int, default=20000); p.add_argument('--margin', type=float, default=.05)
    p.add_argument('--offset', type=float, default=.005); p.add_argument('--step', type=float, default=.5); p.add_argument('--seed', type=int, default=20261103)
    p.add_argument('--batch', type=int, default=1000000); p.add_argument('--screen-margin', type=float, default=.01); p.add_argument('--workers', type=int, default=0, help='0: one per visible GPU')
    p.add_argument('--worker', type=int); p.add_argument('--mixed-part', type=int); p.add_argument('--collision-part', type=int); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    if a.worker is not None: worker(a, a.worker); return
    a.output.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    pairs, heldout = (2048, 256) if a.smoke else (a.pairs, a.heldout); n_col = int(round(pairs/3)); n_mix = pairs-n_col
    need_mix, need_col = pairs+heldout, n_col+heldout
    workers = a.workers or torch.cuda.device_count(); mix_part, col_part = -(-need_mix//workers), -(-need_col//workers)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES'); ids = visible.split(',') if visible else [str(i) for i in range(torch.cuda.device_count())]
    procs = []
    for k in range(workers):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=ids[k % len(ids)], PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        cmd = [sys.executable, '-u', __file__, '--scene', str(a.scene), '--output', str(a.output), '--margin', str(a.margin), '--offset', str(a.offset), '--step', str(a.step),
               '--seed', str(a.seed), '--batch', str(a.batch), '--screen-margin', str(a.screen_margin), '--worker', str(k), '--mixed-part', str(mix_part), '--collision-part', str(col_part)]
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [q.wait() for q in procs]; assert all(c == 0 for c in codes), codes
    parts = [np.load(a.output/f'part{k}.npz') for k in range(workers)]; meta = [json.loads((a.output/f'part{k}.json').read_text()) for k in range(workers)]
    rng = np.random.default_rng(a.seed); stream = {}
    for st, need in (('mixed', need_mix), ('collision', need_col)):
        arrs = [np.concatenate([q[f'{st}_{f}'] for q in parts]) for f in 'XDN']; perm = rng.permutation(len(arrs[0]))[:need]; stream[st] = [v[perm] for v in arrs]
    (Xm, Dm, Nm), (Xc, Dc, Nc) = stream['mixed'], stream['collision']
    assert (Dc[:, 0] <= 0).all() and len(Xm) == need_mix and len(Xc) == need_col
    sets = {}
    X = np.concatenate([Xm[:n_mix], Xc[:n_col]]); perm = rng.permutation(len(X)); src = np.concatenate([np.zeros(n_mix), np.ones(n_col)])[perm]
    sets['test4'] = save(a.output/'test4', X[perm], np.concatenate([Dm[:n_mix], Dc[:n_col]])[perm], np.concatenate([Nm[:n_mix], Nc[:n_col]])[perm], src, a,
                         dict(recipe=f'{n_col} collision pairs (x0 touches or is inside the cupboard or table) + {n_mix} mixed pairs (exactly one grasp point in the cupboard)'))
    perm = rng.permutation(pairs)
    sets['control'] = save(a.output/'control', Xm[:pairs][perm], Dm[:pairs][perm], Nm[:pairs][perm], np.zeros(pairs), a,
                           dict(recipe=f'{pairs} mixed pairs: test4\'s {n_mix} mixed pairs and the next {pairs-n_mix}'))
    sets['heldout'] = save(a.output/'heldout', np.concatenate([Xm[pairs:], Xc[n_col:]]), np.concatenate([Dm[pairs:], Dc[n_col:]]), np.concatenate([Nm[pairs:], Nc[n_col:]]),
                           np.concatenate([np.zeros(heldout), np.ones(heldout)]), a, dict(recipe=f'{heldout} mixed + {heldout} collision pairs in neither training set'))
    counts = {st: {key: sum(m['counts'][st][key] for m in meta) for key in meta[0]['counts'][st]} for st in meta[0]['counts']}
    write(a.output/'complete.json', dict(complete=True, smoke=a.smoke, scene=str(a.scene), scene_sha256=sha(a.scene), seed=a.seed, pairs=pairs, heldout=heldout,
                                         collision_pairs=n_col, mixed_pairs=n_mix, workers=meta, counts=counts, sets=sets, margin=a.margin, offset=a.offset, step=a.step,
                                         screen_margin=a.screen_margin, elapsed_seconds=time.monotonic()-started))
    for k in range(workers): (a.output/f'part{k}.npz').unlink()
    print('COLLIDE_DATA_COMPLETE', json.dumps(dict(sets={k: {kk: v[kk] for kk in ('pairs', 'pairs_collision', 'pair_one_end_in_cupboard', 'endpoints_in_collision')} for k, v in sets.items()},
                                                   counts=counts, seconds=round(time.monotonic()-started))), flush=True)


if __name__ == '__main__': main()
