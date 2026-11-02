"""Test 2b of the joint-range test: the range coordinates fitted to the collision-checked configurations only.

data.py's range/ set fits [-0.5, 0.5] to the min / max over every configuration of the pairs. x1 is a random step
that may be in contact (the ntrl-demo recipe labels it with the floor speed), and those steps reach every joint
limit, so that map is the identity. The robot's actual range of motion in the data is that of the collision-checked
x0 (clearance > 1 cm, self clearance > 2 cm): there the elbow stays within about +-2.76 rad because of self-contact.
This writes <output>/anchor/: the pairs of a data run's full/ set in y = (q - c) / u with c, u from the min / max of x0
only. x1 values outside that range map slightly outside [-0.5, 0.5] (reported). Speeds and clearances are copied;
normals are mapped by the chain rule and renormalised, as data.py does for range/.
"""
import sys
sys.path.insert(0, '/clean/method')
import argparse, json, time
from pathlib import Path
import numpy as np
from common import sha, write

JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3']


def main():
    p = argparse.ArgumentParser(); p.add_argument('--data', type=Path, required=True, help="a data run's data/ folder"); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); started = time.monotonic(); src = a.data/'full'; meta = json.loads((src/'complete.json').read_text()); assert meta['coords'] == 'full' and meta['complete']
    unit = np.asarray(meta['radians_per_unit'], np.float64); assert np.all(np.asarray(meta['center_rad']) == 0)
    X = np.load(src/'sampled_points.npy'); N = np.load(src/'normal.npy')
    q = X.reshape(-1, 2, 6).astype(np.float64)*unit
    lo, hi = q[:, 0].min(0), q[:, 0].max(0); c, u = (lo+hi)/2, hi-lo
    y = ((q-c)/u).reshape(-1, 12).astype(np.float32)
    g = N.reshape(-1, 2, 6).astype(np.float64)*(u/unit); normal = (g/np.linalg.norm(g, axis=2, keepdims=True)).reshape(-1, 12).astype(np.float32)
    out = a.output/'anchor'; out.mkdir(parents=True, exist_ok=False)
    np.save(out/'sampled_points.npy', y); np.save(out/'normal.npy', normal)
    for f in ['speed', 'clearance']: np.save(out/(f+'.npy'), np.load(src/(f+'.npy')))
    outside = (np.abs(y.reshape(-1, 2, 6)[:, 1]) > .5).mean(0)
    ranges = dict(joints=JOINTS, fitted_on='x0 (collision-checked) only', lo_rad=lo.tolist(), hi_rad=hi.tolist(), center_rad=c.tolist(), width_rad=u.tolist(),
                  full_width_rad=unit.tolist(), zoom=(unit/u).tolist(), x1_outside_unit_interval_share=outside.tolist(), coordinate_bounds=[float(y.min()), float(y.max())])
    write(a.output/'anchor_range.json', ranges)
    write(out/'complete.json', dict({k: v for k, v in meta.items() if k not in ('coords', 'center_rad', 'radians_per_unit', 'coordinate_bounds', 'files', 'elapsed_seconds')},
                                    coords='anchor', center_rad=c.tolist(), radians_per_unit=u.tolist(), coordinate_bounds=ranges['coordinate_bounds'],
                                    source=str(src), source_files=meta['files'], elapsed_seconds=time.monotonic()-started,
                                    files={f: sha(out/(f+'.npy')) for f in ['sampled_points', 'speed', 'normal', 'clearance']}))
    print('ANCHOR_RANGE', json.dumps(ranges), flush=True)


if __name__ == '__main__': main()
