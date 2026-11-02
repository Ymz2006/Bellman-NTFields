"""In-cabinet ntrl-demo pairs (joint-range test 3): the in-area recipe of data.py, kept only when at least one of the
two grasp points is inside the cupboard (the scene's cupboard outer box, as the test suites' compartment regions).

The recipe is data.py's, in release units: x0 uniform over the joint limits with its grasp point in the sampling box and
0 < d - 0.01 < margin (release label: obstacle and 0.5 x self distance); x1 = x0 + unit(rand - 0.5) * U(0, 0.5) inside
the joint limits with its grasp point in the sampling box; speed clip(d, offset, margin) / margin; normals by central
differences (eps 1e-4). The extra condition: x0's or x1's grasp point inside the cupboard. With one x1 draw per x0, as
the recipe, the accepted pairs are the recipe's pairs conditioned on that event.

Speed-ups (none of them changes which pairs are accepted):
  pre-screen   the grasp frame alone by forward kinematics of the release URDF (experiments/face_speed/robot.py) for
               every uniform draw; only draws within --screen-margin of the sampling box reach the clearance (cuRobo
               spheres, obstacle and self distances), which then decides with cuRobo's own end-effector position, as
               data.py. x1 goes on to its clearance and the normals only when x0 is in the cupboard (cuRobo position)
               or x1's kinematic grasp point is within the margin of the cupboard (and of the box); the final tests
               use cuRobo positions. A start-up check measures the two kinematics' difference (must be < margin / 10)
               and counts draws the pre-screen would have dropped although cuRobo puts them in the box (must be 0).
  batches      --batch uniform draws per step, so the clearance calls stay large after the pre-screen
  workers      --workers processes, one per visible GPU (CUDA_VISIBLE_DEVICES, expandable CUDA segments), seeds seed + 1000 k; the parent merges
               the parts and shuffles them (seed)
Writes cabinet/ (sampled_points.npy (N, 12) release coordinates, speed.npy, normal.npy, clearance.npy, complete.json).
"""
import argparse, importlib.util, json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, '/clean/method'); sys.path.insert(0, '/clean')
from common import sha, write
import numpy as np
import torch
_spec = importlib.util.spec_from_file_location('ntrl_cabinet_data', '/clean/experiments/ntrl_cabinet/data.py')
D = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(D)


def inside(pos, b):
    return ((pos >= b[:, 0]) & (pos <= b[:, 1])).all(1)


def worker(a, k):
    device = 'cuda:0'; torch.manual_seed(a.seed+1000*k); started = time.monotonic()
    from experiments.face_speed import robot as R
    box, cupboard, _ = D.scene_boxes(a.scene); box_t = torch.tensor(box, device=device); cup_t = torch.tensor(cupboard, device=device)
    grow = lambda b: b+torch.tensor([[-a.screen_margin, a.screen_margin]], device=device)
    box_s, cup_s = grow(box_t), grow(cup_t)
    unit, qmax = D.units('release'); unit_t = torch.tensor(unit, device=device); bound = torch.tensor(qmax/unit, device=device)
    C = D.Clearance(a.scene, 'release', device); steps = torch.eye(6, device=device)*D.EPS
    kin = R.Kinematics().to(device)

    @torch.no_grad()
    def ee(x):
        return torch.cat([kin.transforms(x[s:s+250000]*unit_t, [R.EE_LINK])[R.EE_LINK][:, :3, 3] for s in range(0, len(x), 250000)]) if len(x) else x[:, :3]

    def gradient(x):
        plus = torch.stack([C((x+s)*unit_t) for s in steps], 1); minus = torch.stack([C((x-s)*unit_t) for s in steps], 1)
        return (plus-minus)/(2*D.EPS)

    # start-up check of the pre-screen against cuRobo's end-effector positions
    xv = (torch.rand(500000, 6, device=device)-.5)*2*bound; _, pv = C(xv*unit_t, positions=True); fv = ee(xv)
    deviation = float(torch.linalg.vector_norm(pv-fv, dim=1).max()); missed = int((inside(pv, box_t) & ~inside(fv, box_s)).sum())
    assert deviation < a.screen_margin/10 and missed == 0, (deviation, missed)
    target = a.part; xs, ds, ns = [], [], []; have = 0; drawn = 0
    stats = dict(screen_pass=0, x0_in_box=0, x0_window=0, x1_inside_domain=0, x1_screen_pass=0, accepted=0, x0_cupboard=0, x1_cupboard=0, both_cupboard=0)
    while have < target:
        x0 = (torch.rand(a.batch, 6, device=device)-.5)*2*bound; drawn += a.batch
        x0 = x0[inside(ee(x0), box_s)]; stats['screen_pass'] += len(x0)
        d0, p0 = C(x0*unit_t, positions=True); keep = inside(p0, box_t); stats['x0_in_box'] += int(keep.sum())
        y0 = d0-.01; keep &= (y0 > 0) & (y0 < a.margin); stats['x0_window'] += int(keep.sum())
        x0, d0, p0 = x0[keep], d0[keep], p0[keep]
        dP = torch.rand(len(x0), 6, device=device)-.5; rL = torch.rand(len(x0), 1, device=device)*a.step
        x1 = x0+torch.nn.functional.normalize(dP, dim=1)*rL
        ok = ((x1 <= bound) & (x1 >= -bound)).all(1); stats['x1_inside_domain'] += int(ok.sum())
        x0, x1, d0, p0 = x0[ok], x1[ok], d0[ok], p0[ok]
        c0 = inside(p0, cup_t); f1 = ee(x1); ok = inside(f1, box_s) & (c0 | inside(f1, cup_s)); stats['x1_screen_pass'] += int(ok.sum())
        x0, x1, d0, c0 = x0[ok], x1[ok], d0[ok], c0[ok]
        if len(x0) <= 1: continue
        d1, p1 = C(x1*unit_t, positions=True); c1 = inside(p1, cup_t); ok = inside(p1, box_t) & (c0 | c1)
        x0, x1, d0, d1, c0, c1 = x0[ok], x1[ok], d0[ok], d1[ok], c0[ok], c1[ok]
        if len(x0) <= 1: continue
        g0 = gradient(x0); g1 = gradient(x1)
        n0 = g0/torch.linalg.vector_norm(g0, dim=1, keepdim=True); n1 = g1/torch.linalg.vector_norm(g1, dim=1, keepdim=True)
        good = ~(n0.isnan().any(1) | n1.isnan().any(1) | n0.isinf().any(1) | n1.isinf().any(1))
        stats['accepted'] += int(good.sum()); stats['x0_cupboard'] += int((c0 & good).sum()); stats['x1_cupboard'] += int((c1 & good).sum()); stats['both_cupboard'] += int((c0 & c1 & good).sum())
        xs.append(torch.cat([x0, x1], 1)[good].cpu()); ds.append(torch.stack([d0, d1], 1)[good].cpu()); ns.append(torch.cat([n0, n1], 1)[good].cpu())
        have += int(good.sum())
        if len(xs) % 20 == 0: print(json.dumps(dict(worker=k, drawn=drawn, collected=have, seconds=round(time.monotonic()-started))), flush=True)
    X = torch.cat(xs)[:target].numpy().astype(np.float32); Dc = torch.cat(ds)[:target].numpy().astype(np.float32); N = torch.cat(ns)[:target].numpy().astype(np.float32)
    np.savez(a.output/f'part{k}.npz', X=X, D=Dc, N=N)
    write(a.output/f'part{k}.json', dict(worker=k, seed=a.seed+1000*k, gpu=torch.cuda.get_device_name(0), pairs=len(X), uniform_draws=drawn, counts=stats,
                                         kinematics_deviation_m=deviation, prescreen_missed=missed, seconds=time.monotonic()-started))
    print('WORKER_DONE', k, drawn, have, round(time.monotonic()-started), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--scene', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pairs', type=int, default=200000); p.add_argument('--margin', type=float, default=.05); p.add_argument('--offset', type=float, default=.005)
    p.add_argument('--step', type=float, default=.5); p.add_argument('--seed', type=int, default=20261101); p.add_argument('--batch', type=int, default=1000000)
    p.add_argument('--screen-margin', type=float, default=.01); p.add_argument('--workers', type=int, default=0, help='0: one per visible GPU')
    p.add_argument('--worker', type=int); p.add_argument('--part', type=int); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    if a.worker is not None: worker(a, a.worker); return
    a.output.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    n_pairs = 2048 if a.smoke else a.pairs; workers = a.workers or torch.cuda.device_count(); part = -(-n_pairs//workers)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES'); ids = visible.split(',') if visible else [str(i) for i in range(torch.cuda.device_count())]
    procs = []
    for k in range(workers):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=ids[k % len(ids)], PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')   # varying batch sizes fragment the 12 GB card otherwise
        cmd = [sys.executable, '-u', __file__, '--scene', str(a.scene), '--output', str(a.output), '--pairs', str(n_pairs), '--margin', str(a.margin), '--offset', str(a.offset),
               '--step', str(a.step), '--seed', str(a.seed), '--batch', str(a.batch), '--screen-margin', str(a.screen_margin), '--worker', str(k), '--part', str(part)]
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [q.wait() for q in procs]; assert all(c == 0 for c in codes), codes
    parts = [np.load(a.output/f'part{k}.npz') for k in range(workers)]; meta = [json.loads((a.output/f'part{k}.json').read_text()) for k in range(workers)]
    X = np.concatenate([q['X'] for q in parts]); Dc = np.concatenate([q['D'] for q in parts]); N = np.concatenate([q['N'] for q in parts])
    perm = np.random.default_rng(a.seed).permutation(len(X))[:n_pairs]; X, Dc, N = X[perm], Dc[perm], N[perm]
    speed = (np.clip(Dc, a.offset, a.margin)/a.margin).astype(np.float32)
    out = a.output/'cabinet'; out.mkdir()
    np.save(out/'sampled_points.npy', X); np.save(out/'speed.npy', speed); np.save(out/'normal.npy', N); np.save(out/'clearance.npy', Dc)
    counts = {key: sum(m['counts'][key] for m in meta) for key in meta[0]['counts']}
    unit, _ = D.units('release')
    write(out/'complete.json', dict(complete=True, smoke=a.smoke, scene=str(a.scene), scene_sha256=sha(a.scene), seed=a.seed, pairs=len(X), workers=meta,
        uniform_draws=sum(m['uniform_draws'] for m in meta), counts=counts, label='release', margin=a.margin, offset=a.offset, screen_margin=a.screen_margin,
        condition='x0 or x1 grasp point inside the cupboard outer box (scene_boxes)', x0_filter=f'grasp point in the sampling box, 0 < d - 0.01 < {a.margin}',
        x1_step=f'unit(rand-0.5) * U(0, {a.step}) release units, inside the joint limits, grasp point in the sampling box',
        share_x0_cupboard=counts['x0_cupboard']/counts['accepted'], share_x1_cupboard=counts['x1_cupboard']/counts['accepted'], share_both_cupboard=counts['both_cupboard']/counts['accepted'],
        x1_contact_share=float((Dc[:, 1] <= 0).mean()), speed_quantiles={str(t): float(np.quantile(speed, t)) for t in (.05, .25, .5, .75, .95)},
        coords='full', center_rad=[0.0]*6, radians_per_unit=list(map(float, unit)), elapsed_seconds=time.monotonic()-started, evaluation_rows_read=False,
        files={f: sha(out/(f+'.npy')) for f in ['sampled_points', 'speed', 'normal', 'clearance']}))
    for k in range(workers): (a.output/f'part{k}.npz').unlink()
    print('CABINET_DATA_COMPLETE', json.dumps(dict(pairs=len(X), draws=sum(m['uniform_draws'] for m in meta), counts=counts, seconds=round(time.monotonic()-started))), flush=True)


if __name__ == '__main__': main()
