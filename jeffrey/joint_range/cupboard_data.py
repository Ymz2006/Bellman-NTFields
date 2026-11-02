"""Both-ends-in-the-cupboard training pairs (jeffrey/joint_range/README.md, "Both ends in the cupboard"): the Test 4 mixed
recipe (collide_data.py) with the membership test changed, so that both grasp points lie in the cupboard's outer box.

both       x0 uniform over the joint limits with its grasp point in the cupboard's outer box and 0 < d - 0.01 < margin
           (release label d = min(obstacle distance, 0.5 x self distance)); x1 = x0 + unit(rand - 0.5) * U(0, 0.5) inside
           the joint limits with its grasp point in the cupboard's outer box. Every other setting is the mixed recipe's.
mixed      collide_data.py's mixed stream, unchanged (exactly one grasp point in the cupboard), for the combined set.
Labels as the recipe for every end: speed clip(d, offset, margin) / margin; normals by central differences; pairs with
a NaN or infinite normal are dropped.

Sets in <output>/ (release coordinates, the data.py file set plus source.npy, 0 mixed / 2 both):
  both/     --pairs both pairs, shuffled (seed)
  combo/    --pairs / 2 mixed pairs + the first --pairs / 2 both pairs of both/, shuffled (seed)
  heldout/  --heldout both pairs that neither set contains, for field checks
complete.json of every set adds the gripper displacement of its pairs (median length, median |unit components|, share
of pairs whose largest component is x / y / z), the quantity the axis-sensitivity probe pointed at.
Speed-ups as collide_data.py: URDF grasp-frame pre-screen (final tests on cuRobo positions), --batch draws per step,
one worker per visible GPU (seeds seed + 1000 k); the parent merges and shuffles.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import collide_data as CD
from collide_data import D, inside, Labels, write, sha

BOTH = 2


def worker(a, k):
    device = 'cuda:0'; torch.manual_seed(a.seed+1000*k); started = time.monotonic()
    from experiments.face_speed import robot as R
    box, cupboard, _ = D.scene_boxes(a.scene); box_t = torch.tensor(box, device=device); cup_t = torch.tensor(cupboard, device=device)
    grow = lambda b: b+torch.tensor([[-a.screen_margin, a.screen_margin]], device=device)
    box_s, cup_s = grow(box_t), grow(cup_t)
    unit, qmax = D.units('release'); unit_t = torch.tensor(unit, device=device); bound = torch.tensor(qmax/unit, device=device)
    C = Labels(a.scene, 'release', device); steps = torch.eye(6, device=device)*D.EPS; kin = R.Kinematics().to(device)

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

    xv = (torch.rand(500000, 6, device=device)-.5)*2*bound; _, pv = C(xv*unit_t, positions=True); fv = ee(xv)
    deviation = float(torch.linalg.vector_norm(pv-fv, dim=1).max()); missed = int((inside(pv, cup_t) & ~inside(fv, cup_s)).sum())
    assert deviation < a.screen_margin/10 and missed == 0, (deviation, missed)
    out = {}; stats = {}
    for stream, target in (('both', a.both_part), ('mixed', a.mixed_part)):
        xs, ds, ns = [], [], []; have = 0; drawn = 0; s = stats[stream] = dict(screen_pass=0, x0_in_region=0, x0_accepted=0, x1_inside_domain=0, x1_screen_pass=0, accepted=0, x0_cupboard=0, x1_cupboard=0)
        region_s, region_t = (cup_s, cup_t) if stream == 'both' else (box_s, box_t)
        while have < target:
            x0 = (torch.rand(a.batch, 6, device=device)-.5)*2*bound; drawn += a.batch
            x0 = x0[inside(ee(x0), region_s)]; s['screen_pass'] += len(x0)
            if len(x0) <= 1: continue
            d0, p0 = C(x0*unit_t, positions=True); keep = inside(p0, region_t); s['x0_in_region'] += int(keep.sum())
            y0 = d0-.01; keep &= (y0 > 0) & (y0 < a.margin); s['x0_accepted'] += int(keep.sum())
            x0, d0, p0 = x0[keep], d0[keep], p0[keep]
            x1 = step(x0); ok = ((x1 <= bound) & (x1 >= -bound)).all(1); s['x1_inside_domain'] += int(ok.sum())
            x0, x1, d0, p0 = x0[ok], x1[ok], d0[ok], p0[ok]; c0 = inside(p0, cup_t)
            f1 = ee(x1); ok = inside(f1, cup_s) if stream == 'both' else inside(f1, box_s) & (c0 | inside(f1, cup_s))
            s['x1_screen_pass'] += int(ok.sum()); x0, x1, d0, c0 = x0[ok], x1[ok], d0[ok], c0[ok]
            if len(x0) <= 1: continue
            d1, p1 = C(x1*unit_t, positions=True); c1 = inside(p1, cup_t); ok = (c0 & c1) if stream == 'both' else inside(p1, box_t) & (c0 ^ c1)
            x0, x1, d0, d1, c0, c1 = x0[ok], x1[ok], d0[ok], d1[ok], c0[ok], c1[ok]
            if len(x0) <= 1: continue
            n, good = normals(x0, x1)
            s['accepted'] += int(good.sum()); s['x0_cupboard'] += int((c0 & good).sum()); s['x1_cupboard'] += int((c1 & good).sum())
            xs.append(torch.cat([x0, x1], 1)[good].cpu()); ds.append(torch.stack([d0, d1], 1)[good].cpu()); ns.append(n[good].cpu()); have += int(good.sum())
            if len(xs) % 20 == 0: print(json.dumps(dict(worker=k, stream=stream, drawn=drawn, collected=have, seconds=round(time.monotonic()-started))), flush=True)
        s['uniform_draws'] = drawn
        out[stream] = [torch.cat(v)[:target].numpy().astype(np.float32) for v in (xs, ds, ns)]; del xs, ds, ns; torch.cuda.empty_cache()
    np.savez(a.output/f'part{k}.npz', **{f'{st}_{f}': v for st, arrs in out.items() for f, v in zip('XDN', arrs)})
    write(a.output/f'part{k}.json', dict(worker=k, seed=a.seed+1000*k, gpu=torch.cuda.get_device_name(0), counts=stats, kinematics_deviation_m=deviation,
                                         prescreen_missed=missed, seconds=time.monotonic()-started))
    print('WORKER_DONE', k, json.dumps(stats), round(time.monotonic()-started), flush=True)


def displacement(X):
    """Gripper displacement x1 - x0 of release-coordinate pairs: median length, median |unit components|, dominant-axis shares."""
    from experiments.face_speed import robot as R
    unit, _ = D.units('release'); kin = R.Kinematics().to('cuda:0'); Xt = torch.as_tensor(X, device='cuda:0'); u = torch.tensor(unit, device='cuda:0')
    with torch.no_grad():
        pos = [torch.cat([kin.transforms(Xt[s:s+100000, c:c+6]*u, [R.EE_LINK])[R.EE_LINK][:, :3, 3] for s in range(0, len(Xt), 100000)]).cpu().numpy() for c in (0, 6)]
    d = pos[1]-pos[0]; L = np.linalg.norm(d, axis=1); un = np.abs(d/np.maximum(L, 1e-12)[:, None]); share = np.bincount(un.argmax(1), minlength=3)/len(d)
    return dict(gripper_displacement_median_m=float(np.median(L)), gripper_displacement_p90_m=float(np.quantile(L, .9)), unit_components_median_abs=[float(v) for v in np.median(un, 0)],
                dominant_axis_share=dict(zip('xyz', map(float, share))))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--scene', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pairs', type=int, default=200000); p.add_argument('--heldout', type=int, default=20000); p.add_argument('--margin', type=float, default=.05)
    p.add_argument('--offset', type=float, default=.005); p.add_argument('--step', type=float, default=.5); p.add_argument('--seed', type=int, default=20261104)
    p.add_argument('--batch', type=int, default=1000000); p.add_argument('--screen-margin', type=float, default=.01); p.add_argument('--workers', type=int, default=0, help='0: one per visible GPU')
    p.add_argument('--worker', type=int); p.add_argument('--mixed-part', type=int); p.add_argument('--both-part', type=int); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    if a.worker is not None: worker(a, a.worker); return
    a.output.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    pairs, heldout = (2048, 256) if a.smoke else (a.pairs, a.heldout); half = pairs//2
    need_both, need_mix = pairs+heldout, pairs-half
    workers = a.workers or torch.cuda.device_count(); both_part, mix_part = -(-need_both//workers), -(-need_mix//workers)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES'); ids = visible.split(',') if visible else [str(i) for i in range(torch.cuda.device_count())]
    procs = []
    for k in range(workers):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=ids[k % len(ids)], PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        cmd = [sys.executable, '-u', __file__, '--scene', str(a.scene), '--output', str(a.output), '--margin', str(a.margin), '--offset', str(a.offset), '--step', str(a.step),
               '--seed', str(a.seed), '--batch', str(a.batch), '--screen-margin', str(a.screen_margin), '--worker', str(k), '--mixed-part', str(mix_part), '--both-part', str(both_part)]
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [q.wait() for q in procs]; assert all(c == 0 for c in codes), codes
    parts = [np.load(a.output/f'part{k}.npz') for k in range(workers)]; meta = [json.loads((a.output/f'part{k}.json').read_text()) for k in range(workers)]
    rng = np.random.default_rng(a.seed); stream = {}
    for st, need in (('both', need_both), ('mixed', need_mix)):
        arrs = [np.concatenate([q[f'{st}_{f}'] for q in parts]) for f in 'XDN']; perm = rng.permutation(len(arrs[0]))[:need]; stream[st] = [v[perm] for v in arrs]
    (Xb, Db, Nb), (Xm, Dm, Nm) = stream['both'], stream['mixed']
    assert len(Xb) == need_both and len(Xm) == need_mix
    sets = {}
    perm = rng.permutation(pairs)
    extra = {'both': dict(recipe=f'{pairs} pairs with both grasp points in the cupboard (mixed recipe, membership test changed)', **displacement(Xb[:pairs]))}
    sets['both'] = {**CD.save(a.output/'both', Xb[:pairs][perm], Db[:pairs][perm], Nb[:pairs][perm], np.full(pairs, BOTH), a, extra['both']), **extra['both']}
    X = np.concatenate([Xm, Xb[:half]]); perm = rng.permutation(len(X)); src = np.concatenate([np.zeros(len(Xm)), np.full(half, BOTH)])[perm]
    extra['combo'] = dict(recipe=f'{len(Xm)} mixed pairs (exactly one grasp point in the cupboard) + the first {half} pairs of both/', pairs_both=int(half), **displacement(X))
    sets['combo'] = {**CD.save(a.output/'combo', X[perm], np.concatenate([Dm, Db[:half]])[perm], np.concatenate([Nm, Nb[:half]])[perm], src, a, extra['combo']), **extra['combo']}
    extra['heldout'] = dict(recipe=f'{heldout} both-in-cupboard pairs in neither training set', **displacement(Xb[pairs:]))
    sets['heldout'] = {**CD.save(a.output/'heldout', Xb[pairs:], Db[pairs:], Nb[pairs:], np.full(heldout, BOTH), a, extra['heldout']), **extra['heldout']}
    counts = {st: {key: sum(m['counts'][st][key] for m in meta) for key in meta[0]['counts'][st]} for st in meta[0]['counts']}
    write(a.output/'complete.json', dict(complete=True, smoke=a.smoke, scene=str(a.scene), scene_sha256=sha(a.scene), seed=a.seed, pairs=pairs, heldout=heldout, both_pairs=pairs, combo_mixed_pairs=len(Xm),
                                         combo_both_pairs=half, workers=meta, counts=counts, sets=sets, margin=a.margin, offset=a.offset, step=a.step, screen_margin=a.screen_margin,
                                         elapsed_seconds=time.monotonic()-started))
    for k in range(workers): (a.output/f'part{k}.npz').unlink()
    print('CUPBOARD_DATA_COMPLETE', json.dumps(dict(sets={k: {kk: v[kk] for kk in ('pairs', 'pair_one_end_in_cupboard', 'points_in_cupboard', 'endpoints_in_collision', 'gripper_displacement_median_m', 'dominant_axis_share')} for k, v in sets.items()},
                                                    counts=counts, seconds=round(time.monotonic()-started))), flush=True)


if __name__ == '__main__': main()
