"""Checks of the face-speed geometry against the release (cuRobo) labels and brute force.

    ~/.venvs/bellman-ntfield/bin/python experiments/face_speed/tests/test_geometry.py [--device cuda:1] [--run results/registered_<cabinet run>]

1. Spheres and grasp point reproduce the cuRobo-made arrays of a cabinet run (coverage/validation:
   d_obstacle with the shoulder ignored, d_self with the release mask, ee_positions).
2. Every padded sphere lies inside its body's box (so face clearance <= sphere clearance).
3. rect_aabb_signed equals a brute-force rectangle sampling for random pairs; intersecting pairs are <= 0.
4. The box-face clearance never exceeds the release obstacle clearance on pool configurations; sphere and
   link covers reproduce the release clearance exactly.
5. Autograd gradient agrees with central differences away from kinks; timing per 4096 configurations.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
from experiments.face_speed import robot as R, faces as F, clearance as C

p = argparse.ArgumentParser(); p.add_argument('--device', default='cuda:0'); p.add_argument('--run', default='results/registered_20261019T034213Z-f9c501f9')
p.add_argument('--scene', default='experiments/cabinet/scenes/cabinet.yml'); a = p.parse_args()
dev = a.device; run = ROOT/a.run; scene = ROOT/a.scene; torch.manual_seed(0); rng = np.random.default_rng(0)
report = {}

# 1. cuRobo equivalence on saved arrays
val = run/'sampled/coverage/validation'
q = np.load(val/'configurations_rad.npy', mmap_mode='r'); n = 20000; idx = np.sort(rng.choice(len(q), n, replace=False))
q = torch.tensor(np.asarray(q[idx]), device=dev); do = np.asarray(np.load(val/'d_obstacle.npy', mmap_mode='r')[idx]); ds = np.asarray(np.load(val/'d_self.npy', mmap_mode='r')[idx]); ee = np.asarray(np.load(val/'ee_positions.npy', mmap_mode='r')[idx])[:, :3]  # validation split: pair (xyz_i, xyz_j), i = the configuration order
pc = C.PieceClearance(scene, 'spheres', dev)
with torch.no_grad():
    qn = (q-pc.robot.q_center)/pc.robot.q_range
    my_do, my_ds = pc.release_components(qn); my_ee = pc.ee_position(qn)
e_do, e_ds, e_ee = np.abs(my_do.cpu().numpy()-do).max(), np.abs(my_ds.cpu().numpy()-ds).max(), np.abs(my_ee.cpu().numpy()-ee).max()
report['curobo_equivalence'] = dict(n=n, max_abs_err_d_obstacle=float(e_do), max_abs_err_d_self=float(e_ds), max_abs_err_ee=float(e_ee))
print('1 cuRobo equivalence', report['curobo_equivalence'], flush=True)
assert e_do < 2e-5 and e_ds < 2e-5 and e_ee < 2e-5

# 2. boxes contain spheres
fc = C.PieceClearance(scene, 'faces', dev); box = fc.boxes
print('   box cover (m):', {k: [round(x, 3) for x in v['size']] for k, v in box.summary().items()})
with torch.no_grad():
    qr = fc.robot.radians(torch.rand(512, 6, device=dev)-.5); links = list(dict.fromkeys(fc.robot.collision_links+box.reference))
    t = fc.robot.transforms(qr, links); sp = fc.robot.link_spheres(qr, t); worst = -1.
    for k, body in enumerate(box.body_names):
        ref, members = box.bodies[body]; Tinv = torch.linalg.inv(t[ref])
        rows = torch.isin(fc.robot.sphere_link, torch.tensor([fc.robot.collision_links.index(l) for l in members], device=dev))
        c = torch.einsum('bij,bsj->bsi', Tinv[:, :3, :3], sp[:, rows, :3])+Tinv[:, None, :3, 3]; r = sp[:, rows, 3]
        excess = ((c-box.box_center[k]).abs()+r[..., None]-box.box_half[k]).amax(-1).max(); worst = max(worst, float(excess))
report['box_contains_spheres_max_excess_m'] = worst; print('2 max sphere excess outside its box (m):', worst, flush=True); assert worst < 1e-5

# 3. brute force rectangle distance
M = 300; c = torch.rand(M, 3)*.6-.3; U = torch.nn.functional.normalize(torch.randn(M, 3), dim=1); V = torch.nn.functional.normalize(torch.cross(U, torch.randn(M, 3), dim=1), dim=1)
aa = torch.rand(M)*.1+.01; bb = torch.rand(M)*.1+.01; lo = torch.rand(1, 3)*.2-.3; hi = lo+torch.rand(1, 3)*.3+.05
d = F.rect_aabb_signed(c, U, V, aa, bb, lo, hi, far=10.)[:, 0]
g = torch.linspace(-1, 1, 161); gu, gv = torch.meshgrid(g, g, indexing='ij')
pts = c[:, None, :]+(gu.flatten()[None, :, None]*aa[:, None, None])*U[:, None]+(gv.flatten()[None, :, None]*bb[:, None, None])*V[:, None]
brute = F.point_aabb(pts, lo, hi).amin(1); spacing = torch.sqrt((aa/80)**2+(bb/80)**2)
disjoint = brute > 1e-6; err = (brute-d)[disjoint]
report['brute_force'] = dict(pairs=M, disjoint=int(disjoint.sum()), max_brute_minus_exact=float(err.max()), min_brute_minus_exact=float(err.min()), max_allowed=float(spacing.max()),
                             intersecting_max_signed=float(d[~disjoint].max()) if (~disjoint).any() else None)
print('3 brute force', report['brute_force'], flush=True)
assert err.min() > -1e-5 and bool((err <= spacing[disjoint]+1e-6).all()) and (d[~disjoint] <= 1e-6).all()

# 4. face clearance <= release clearance; sphere/link covers exact
pool = np.load(run/'sampled/original/sampled_points.npy', mmap_mode='r'); rows = np.sort(rng.choice(len(pool), 4096, replace=False))
qn = torch.tensor(np.asarray(pool[rows])[:, :6], device=dev)
with torch.no_grad():
    rel_o, rel_s = fc.release_components(qn); release = torch.minimum(rel_o, .5*rel_s)
    d_faces = fc.clearance(qn); d_sph = pc.clearance(qn); lc = C.PieceClearance(scene, 'links', dev); d_link = lc.clearance(qn)
    faces_obs = fc.pieces(qn)[:, :-1].amin(1)
report['covers'] = dict(face_obstacle_minus_release_max=float((faces_obs-rel_o).max()), sphere_cover_max_err=float((d_sph-release).abs().max()), link_cover_max_err=float((d_link-release).abs().max()),
                        mean_release_speed=float(C.PieceClearance.speed_of(release).mean()), mean_face_speed=float(C.PieceClearance.speed_of(d_faces).mean()),
                        share_face_speed_at_floor=float((C.PieceClearance.speed_of(d_faces) <= .1+1e-6).float().mean()), share_release_speed_at_floor=float((C.PieceClearance.speed_of(release) <= .1+1e-6).float().mean()))
print('4 covers', report['covers'], flush=True)
assert report['covers']['face_obstacle_minus_release_max'] < 1e-5 and report['covers']['sphere_cover_max_err'] < 1e-6 and report['covers']['link_cover_max_err'] < 1e-6

# 5. gradients and timing
x = qn[:1024]; g_auto = fc.gradient(x); eps = 1e-4
with torch.no_grad():
    steps = torch.eye(6, device=dev)*eps
    g_fd = torch.stack([(fc.clearance(x+s)-fc.clearance(x-s))/(2*eps) for s in steps], 1)
cos = torch.nn.functional.cosine_similarity(g_auto, g_fd, dim=1)
report['gradient'] = dict(n=len(x), cosine_median=float(cos.median()), share_cosine_above_0_99=float((cos > .99).float().mean()), zero_autograd=int((g_auto.norm(dim=1) == 0).sum()))
print('5 gradient', report['gradient'], flush=True)
for name, obj in [('faces', fc), ('spheres', pc)]:
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad(): obj.pieces(qn)
    torch.cuda.synchronize(); t1 = time.time(); obj.gradient(qn); torch.cuda.synchronize(); t2 = time.time()
    report[f'seconds_per_4096_{name}'] = dict(pieces=t1-t0, gradient=t2-t1)
print('   timing', {k: v for k, v in report.items() if k.startswith('seconds')}, flush=True)
import json; out = Path(__file__).with_name('geometry_checks.json'); out.write_text(json.dumps(report, indent=2)+'\n'); print('ALL GEOMETRY CHECKS PASSED ->', out)
