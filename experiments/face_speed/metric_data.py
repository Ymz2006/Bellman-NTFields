"""Metric NTFields training data for the cabinet task with the per-piece (box-face) clearance.

The recipe of experiments/cabinet/metric/data.py is kept (ntrl-demo arm sampling adapted to the cabinet:
500,000 pairs; x0 uniform in the normalized joint cube with the grasp point in the sampling box and
0 < d - 0.01 < 0.05; x1 = x0 + unit(rand - 0.5) * U(0, 0.5) inside [-0.5, 0.5]^6; per endpoint the speed
clip(d, 0.005, 0.05) / 0.05 and the unit clearance gradient as normal; seed 20261019). Only the clearance
d changes (experiments/face_speed/clearance.py):

  --cover faces    d = min(min over box faces of the signed face-cuboid distance, 0.5 d_self): the robot is
                   covered by one box per rigid body and each face has its own clearance; the body's speed
                   is the minimum over the faces. Boxes contain the release spheres, so d <= release d.
  --cover spheres  d = the release clearance exactly (the baseline's own labels).
  --cover environment  d = the environment clearance only, no self-collision term.

The normal is autograd through the kinematics instead of a central difference with eps 1e-4 (the two agree
to cosine > 0.99 on 99.9% of configurations, tests/geometry_checks.json). Rows whose gradient is zero are
dropped, as the baseline drops NaN normals. Writes sampled_points.npy (N, 12), speed.npy (N, 2),
normal.npy (N, 12) and clearance.npy (N, 2), the files ntrl_demo/models/data_mlp.py reads, plus
release_speed.npy (N, 2), the release label at the same points, for the report.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import yaml
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from experiments.face_speed.clearance import PieceClearance, MARGIN, OFFSET

SEED = 20261019


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scene_boxes(scene):
    doc = yaml.safe_load(Path(scene).read_text())
    box = np.array([doc['sampling'][k] for k in 'xyz'], np.float32)
    panels = [c for c in doc['cuboids'] if c['name'].startswith('shelf_')]
    lo = np.min([np.subtract(c['pose'][:3], np.divide(c['dims'], 2)) for c in panels], 0)
    hi = np.max([np.add(c['pose'][:3], np.divide(c['dims'], 2)) for c in panels], 0)
    return box, np.stack([lo, hi], 1).astype(np.float32)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--scene', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cover', choices=['faces', 'spheres', 'links', 'environment'], default='faces'); p.add_argument('--smoke', action='store_true')
    p.add_argument('--filter', choices=['cover', 'release'], default='cover', help='clearance used by the x0 near-obstacle filter: the cover clearance (default) or the release clearance (same x0 distribution as the baseline)')
    p.add_argument('--margin', type=float, default=MARGIN, help='speed clip(d, offset, margin)/margin and x0 filter 0 < d - 0.01 < margin (default 0.05)')
    p.add_argument('--offset', type=float, default=OFFSET, help='lower clip of the speed label: floor speed offset/margin (default 0.005)')
    p.add_argument('--pairs', type=int, default=500000, help='number of (x0, x1) pairs (default 500000)')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False); started = time.monotonic(); device = 'cuda:0'
    box, cupboard = scene_boxes(a.scene); box_t = torch.tensor(box, device=device)
    numsamples = 4096 if a.smoke else a.pairs; batch = 200000; margin, offset = a.margin, a.offset; assert 0 < offset < margin
    torch.manual_seed(SEED); C = PieceClearance(a.scene, a.cover, device)
    xs, ys, ns, rs = [], [], [], []; drawn = 0; stats = dict(x0_in_box=0, x0_near_obstacle=0, x1_inside_domain=0, nonzero_normals=0)

    def gradient(x):
        out = []
        for s in range(0, len(x), 8192): out.append(C.gradient(x[s:s+8192]))
        return torch.cat(out)

    while sum(len(x) for x in xs) < numsamples:
        x0 = torch.rand(batch, 6, device=device)-.5; drawn += batch
        with torch.no_grad():
            pos = torch.cat([C.ee_position(x0[s:s+65536]) for s in range(0, batch, 65536)])
            keep = ((pos >= box_t[:, 0]) & (pos <= box_t[:, 1])).all(1); stats['x0_in_box'] += int(keep.sum()); x0 = x0[keep]
            d0 = torch.cat([C.clearance(x0[s:s+8192]) for s in range(0, len(x0), 8192)])
            if a.filter == 'release':
                o, sd = zip(*[C.release_components(x0[s:s+8192]) for s in range(0, len(x0), 8192)]); d_filter = torch.minimum(torch.cat(o), .5*torch.cat(sd))
            else: d_filter = d0
            y0 = d_filter-.01; keep = (y0 > 0) & (y0 < margin); stats['x0_near_obstacle'] += int(keep.sum()); x0, d0 = x0[keep], d0[keep]
            dP = torch.rand(len(x0), 6, device=device)-.5; rL = torch.rand(len(x0), 1, device=device)*.5
            x1 = x0+torch.nn.functional.normalize(dP, dim=1)*rL
            inside = ((x1 <= .5) & (x1 >= -.5)).all(1); x0, x1, d0 = x0[inside], x1[inside], d0[inside]; stats['x1_inside_domain'] += int(inside.sum())
        if len(x0) <= 1: continue
        with torch.no_grad(): d1 = torch.cat([C.clearance(x1[s:s+8192]) for s in range(0, len(x1), 8192)])
        g0 = gradient(x0); g1 = gradient(x1)
        n0 = g0/torch.linalg.vector_norm(g0, dim=1, keepdim=True); n1 = g1/torch.linalg.vector_norm(g1, dim=1, keepdim=True)
        good = ~(n0.isnan().any(1) | n1.isnan().any(1) | n0.isinf().any(1) | n1.isinf().any(1)); stats['nonzero_normals'] += int(good.sum())
        with torch.no_grad():
            rel = []
            for x in (x0, x1):
                o, s = zip(*[C.release_components(x[k:k+8192]) for k in range(0, len(x), 8192)])
                rel.append(torch.minimum(torch.cat(o), .5*torch.cat(s)))
            rs.append(torch.stack(rel, 1)[good].cpu())
        xs.append(torch.cat([x0, x1], 1)[good].cpu()); ys.append(torch.stack([d0, d1], 1)[good].cpu()); ns.append(torch.cat([n0, n1], 1)[good].cpu())
        print({'drawn': drawn, 'collected': sum(len(x) for x in xs), 'seconds': round(time.monotonic()-started, 1)}, flush=True)
    X = torch.cat(xs)[:numsamples].numpy().astype(np.float32); D = torch.cat(ys)[:numsamples].numpy(); N = torch.cat(ns)[:numsamples].numpy().astype(np.float32)
    REL = torch.cat(rs)[:numsamples].numpy()
    speed = (np.clip(D, offset, margin)/margin).astype(np.float32); release_speed = (np.clip(REL, OFFSET, MARGIN)/MARGIN).astype(np.float32)
    np.save(a.output/'sampled_points.npy', X); np.save(a.output/'speed.npy', speed); np.save(a.output/'normal.npy', N)
    np.save(a.output/'clearance.npy', D.astype(np.float32)); np.save(a.output/'release_speed.npy', release_speed)
    with torch.no_grad():
        pos = torch.cat([C.ee_position(torch.tensor(X[i:i+65536, :6], device=device)).cpu() for i in range(0, len(X), 65536)]).numpy()
    report = dict(complete=True, smoke=a.smoke, scene=str(a.scene), scene_sha256=sha(a.scene), seed=SEED, cover=a.cover, pieces=C.names, n_pieces=C.n_pieces,
        box_cover=C.boxes.summary() if a.cover == 'faces' else None, pairs=len(X), uniform_draws=drawn, counts=stats,
        recipe='experiments/cabinet/metric/data.py with the cover clearance (experiments/face_speed/metric_data.py docstring)', margin=margin, offset=offset, release_label=dict(margin=MARGIN, offset=OFFSET),
        x0_filter=f'0 < d - 0.01 < margin with d = {a.filter} clearance', filter=a.filter, x1_step='unit(rand-0.5) * U(0, 0.5)', normal='autograd gradient of the cover clearance',
        speed_quantiles={str(q): float(np.quantile(speed, q)) for q in (.05, .25, .5, .75, .95)},
        release_speed_quantiles={str(q): float(np.quantile(release_speed, q)) for q in (.05, .25, .5, .75, .95)},
        cover_vs_release=dict(mean_speed=float(speed.mean()), mean_release_speed=float(release_speed.mean()), share_below_release=float((speed < release_speed-1e-6).mean()),
                              share_above_release=float((speed > release_speed+1e-6).mean()), mean_abs_difference=float(np.abs(speed-release_speed).mean())),
        x0_cupboard_share=float(((pos >= cupboard[:, 0]) & (pos <= cupboard[:, 1])).all(1).mean()), x1_speed_floor_share=float((speed[:, 1] <= offset/margin+1e-6).mean()),
        x1_release_speed_floor_share=float((release_speed[:, 1] <= .1+1e-6).mean()), elapsed_seconds=time.monotonic()-started, evaluation_rows_read=False,
        files={n: sha(a.output/(n+'.npy')) for n in ['sampled_points', 'speed', 'normal', 'clearance', 'release_speed']})
    (a.output/'complete.json').write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report, indent=1), flush=True)


if __name__ == '__main__': main()
