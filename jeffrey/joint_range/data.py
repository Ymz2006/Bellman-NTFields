"""Cabinet ntrl-demo training pairs with the grasp point of BOTH configurations in the test area, in two input scalings.

The user's joint-range test (2026-10-31). One physical data set, written in two coordinate systems:

  full   x = q / q_range, the release normalization: every joint's full range of motion (+-2 pi, elbow +-pi)
         maps onto [-0.5, 0.5]. This is the coordinate system of the registered ntrl-demo cabinet baseline.
  range  y = (q - c) / u per joint, with [lo_j, hi_j] the min / max of joint j over every configuration of the
         data (x0 and x1 of all pairs), c = (lo + hi) / 2 and u = hi - lo: the data's own range of motion
         maps onto [-0.5, 0.5].

The pairs follow the ntrl-demo baseline recipe of experiments/ntrl_cabinet/data.py (its Clearance and units are
imported unchanged; its defaults reproduce the registered cabinet-metric data), in release units:
  x0  uniform over the joint limits, grasp point inside the scene's sampling box (the test area: every frozen
      test endpoint lies in it), clearance window 0 < d - 0.01 < margin with the release label
      d = min(obstacle distance, 0.5 self distance) (collision-checked, near an obstacle)
  x1  x0 + unit(rand - 0.5) * U(0, --step), inside the joint limits, and - the one addition to the recipe - its
      grasp point also inside the sampling box. As in the recipe x1 is not rejected for contact; it is labelled.
  speed clip(d, offset, margin) / margin; normals the unit clearance gradient by central differences (eps 1e-4,
      release units). For the range coordinates the same gradient is mapped by the chain rule
      (dd/dy_j = dd/dx_j * u_j / q_range_j) and renormalised, so both sets describe the same physical labels.
Writes full/ and range/ (sampled_points.npy (N, 12), speed.npy (N, 2), normal.npy (N, 12), clearance.npy,
complete.json with the coordinate map) and range.json (per-joint ranges).
"""
import argparse, importlib.util, json, sys, time
from pathlib import Path
sys.path.insert(0, '/clean/method')
from common import sha, write
import numpy as np
import torch
_spec = importlib.util.spec_from_file_location('ntrl_cabinet_data', '/clean/experiments/ntrl_cabinet/data.py')
D = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(D)

JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3']


def main():
    p = argparse.ArgumentParser(); p.add_argument('--scene', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pairs', type=int, default=400000); p.add_argument('--margin', type=float, default=.05); p.add_argument('--offset', type=float, default=.005)
    p.add_argument('--step', type=float, default=.5); p.add_argument('--seed', type=int, default=20261019); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False); started = time.monotonic(); device = 'cuda:0'
    box, cupboard, _ = D.scene_boxes(a.scene); box_t = torch.tensor(box, device=device); cup_t = torch.tensor(cupboard, device=device)
    inside = lambda pos, b: ((pos >= b[:, 0]) & (pos <= b[:, 1])).all(1)
    n_pairs = 4096 if a.smoke else a.pairs; margin, offset = a.margin, a.offset; batch = 200000
    unit, qmax = D.units('release'); unit_t = torch.tensor(unit, device=device); bound = torch.tensor(qmax/unit, device=device)
    torch.manual_seed(a.seed); C = D.Clearance(a.scene, 'release', device); steps = torch.eye(6, device=device)*D.EPS

    def gradient(x):
        plus = torch.stack([C((x+s)*unit_t) for s in steps], 1); minus = torch.stack([C((x-s)*unit_t) for s in steps], 1)
        return (plus-minus)/(2*D.EPS)

    xs, ds, ns = [], [], []; drawn = 0; have = 0
    stats = dict(x0_in_box=0, x0_window=0, x1_inside_domain=0, x1_in_box=0, nonzero_normals=0)
    while have < n_pairs:
        x0 = (torch.rand(batch, 6, device=device)-.5)*2*bound; drawn += batch        # uniform over the joint limits
        d0, pos = C(x0*unit_t, positions=True); keep = inside(pos, box_t); stats['x0_in_box'] += int(keep.sum())
        y0 = d0-.01; keep &= (y0 > 0) & (y0 < margin); stats['x0_window'] += int(keep.sum())
        x0, d0 = x0[keep], d0[keep]
        dP = torch.rand(len(x0), 6, device=device)-.5; rL = torch.rand(len(x0), 1, device=device)*a.step
        x1 = x0+torch.nn.functional.normalize(dP, dim=1)*rL
        ok = ((x1 <= bound) & (x1 >= -bound)).all(1); x0, x1, d0 = x0[ok], x1[ok], d0[ok]; stats['x1_inside_domain'] += int(ok.sum())
        if len(x0) <= 1: continue
        d1, p1 = C(x1*unit_t, positions=True); ok = inside(p1, box_t)
        x0, x1, d0, d1 = x0[ok], x1[ok], d0[ok], d1[ok]; stats['x1_in_box'] += int(ok.sum())
        if len(x0) <= 1: continue
        g0 = gradient(x0); g1 = gradient(x1)
        n0 = g0/torch.linalg.vector_norm(g0, dim=1, keepdim=True); n1 = g1/torch.linalg.vector_norm(g1, dim=1, keepdim=True)
        good = ~(n0.isnan().any(1) | n1.isnan().any(1) | n0.isinf().any(1) | n1.isinf().any(1)); stats['nonzero_normals'] += int(good.sum())
        xs.append(torch.cat([x0, x1], 1)[good].cpu()); ds.append(torch.stack([d0, d1], 1)[good].cpu()); ns.append(torch.cat([n0, n1], 1)[good].cpu())
        have += int(good.sum()); print({'drawn': drawn, 'collected': have}, flush=True)
    X = torch.cat(xs)[:n_pairs].numpy().astype(np.float32); Dc = torch.cat(ds)[:n_pairs].numpy().astype(np.float32)
    N = torch.cat(ns)[:n_pairs].numpy().astype(np.float32)
    speed = (np.clip(Dc, offset, margin)/margin).astype(np.float32)

    # the data's range of motion per joint, over x0 and x1
    q = X.reshape(-1, 2, 6).astype(np.float64)*unit                                  # radians, [N, 2, 6]
    lo, hi = q.min((0, 1)), q.max((0, 1)); center, width = (lo+hi)/2, hi-lo
    full_width = 2*qmax.astype(np.float64)
    with torch.no_grad():
        p0 = torch.cat([C(torch.tensor(X[i:i+65536, :6], device=device)*unit_t, positions=True)[1].cpu() for i in range(0, len(X), 65536)])
        p1 = torch.cat([C(torch.tensor(X[i:i+65536, 6:], device=device)*unit_t, positions=True)[1].cpu() for i in range(0, len(X), 65536)])
    common = dict(complete=True, smoke=a.smoke, scene=str(a.scene), scene_sha256=sha(a.scene), seed=a.seed, pairs=len(X), uniform_draws=drawn, counts=stats,
                  label='release', margin=margin, offset=offset, x0_filter=f'grasp point in the sampling box, 0 < d - 0.01 < {margin}',
                  x1_step=f'unit(rand-0.5) * U(0, {a.step}) release units, inside the joint limits, grasp point in the sampling box',
                  sampling_box=box.tolist(), x0_cupboard_share=float(inside(p0, torch.tensor(cupboard)).float().mean()),
                  x1_cupboard_share=float(inside(p1, torch.tensor(cupboard)).float().mean()), x1_contact_share=float((Dc[:, 1] <= 0).mean()),
                  speed_quantiles={str(t): float(np.quantile(speed, t)) for t in (.05, .25, .5, .75, .95)}, evaluation_rows_read=False)
    ranges = dict(joints=JOINTS, lo_rad=lo.tolist(), hi_rad=hi.tolist(), center_rad=center.tolist(), width_rad=width.tolist(), full_width_rad=full_width.tolist(),
                  zoom=(full_width/width).tolist(), occupied_share=[float((np.histogram(q[:, :, j].ravel(), bins=72, range=(-qmax[j], qmax[j]))[0] > q[:, :, j].size*5e-4).mean()) for j in range(6)])
    write(a.output/'range.json', ranges); print('RANGE', json.dumps(ranges), flush=True)

    sets = {'full': (np.zeros(6), unit.astype(np.float64)), 'range': (center, width)}
    for name, (c, u) in sets.items():
        out = a.output/name; out.mkdir()
        y = ((q-c)/u).reshape(-1, 12).astype(np.float32)
        if name == 'full':
            y = X                                                                     # exactly the recipe's values
            normal = N
        else:
            g = N.reshape(-1, 2, 6).astype(np.float64)*(u/unit.astype(np.float64))    # chain rule, then unit length
            normal = (g/np.linalg.norm(g, axis=2, keepdims=True)).reshape(-1, 12).astype(np.float32)
        np.save(out/'sampled_points.npy', y); np.save(out/'speed.npy', speed); np.save(out/'normal.npy', normal); np.save(out/'clearance.npy', Dc)
        write(out/'complete.json', dict(common, coords=name, center_rad=list(map(float, c)), radians_per_unit=list(map(float, u)),
                                        coordinate_bounds=[float(y.min()), float(y.max())], elapsed_seconds=time.monotonic()-started,
                                        files={f: sha(out/(f+'.npy')) for f in ['sampled_points', 'speed', 'normal', 'clearance']}))
    print('DATA_COMPLETE', json.dumps(dict(pairs=len(X), drawn=drawn, counts=stats, seconds=time.monotonic()-started)), flush=True)


if __name__ == '__main__': main()
