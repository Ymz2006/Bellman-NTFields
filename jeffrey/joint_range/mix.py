"""One mixture of the in-cabinet sweep (joint-range test 3): --fraction of the pairs from the in-cabinet set, the rest
from the in-area set, --pairs in total, shuffled (seed). Both sets are in release coordinates (the full-range
scaling); the first round(fraction * pairs) in-cabinet pairs and the first pairs - that many in-area pairs are taken, so
the mixtures are nested. Grasp points of the mixture are recomputed by forward kinematics of the release URDF
(experiments/face_speed/robot.py) to report its cupboard shares. Writes <output>/mix/ (the data.py file set).
"""
import sys
sys.path.insert(0, '/clean/method'); sys.path.insert(0, '/clean')
import argparse, importlib.util, json, time
from pathlib import Path
import numpy as np
import torch
from common import sha, write
_spec = importlib.util.spec_from_file_location('ntrl_cabinet_data', '/clean/experiments/ntrl_cabinet/data.py')
D = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(D)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--cabinet', type=Path, required=True); p.add_argument('--area', type=Path, required=True)
    p.add_argument('--scene', type=Path, required=True); p.add_argument('--fraction', type=float, required=True); p.add_argument('--pairs', type=int, default=200000)
    p.add_argument('--seed', type=int, default=20261102); p.add_argument('--output', type=Path, required=True); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    started = time.monotonic(); n_cab = int(round(a.fraction*a.pairs)); n_area = a.pairs-n_cab
    mc, ma = json.loads((a.cabinet/'complete.json').read_text()), json.loads((a.area/'complete.json').read_text())
    for m in (mc, ma): assert m['complete'] and m['coords'] == 'full' and (a.smoke or not m['smoke'])
    assert mc['pairs'] >= n_cab and ma['pairs'] >= n_area
    parts = {}
    for f in ['sampled_points', 'speed', 'normal', 'clearance']:
        parts[f] = np.concatenate([np.load(a.cabinet/(f+'.npy'), mmap_mode='r')[:n_cab], np.load(a.area/(f+'.npy'), mmap_mode='r')[:n_area]])
    source = np.concatenate([np.ones(n_cab, bool), np.zeros(n_area, bool)])
    perm = np.random.default_rng(a.seed).permutation(a.pairs)
    out = a.output/'mix'; out.mkdir(parents=True, exist_ok=False)
    for f, v in parts.items(): np.save(out/(f+'.npy'), np.ascontiguousarray(v[perm]))
    from experiments.face_speed import robot as R
    unit, _ = D.units('release'); kin = R.Kinematics().to('cuda:0'); X = torch.as_tensor(parts['sampled_points'], device='cuda:0'); u = torch.tensor(unit, device='cuda:0')
    with torch.no_grad():
        pos = [torch.cat([kin.transforms(X[s:s+100000, c:c+6]*u, [R.EE_LINK])[R.EE_LINK][:, :3, 3] for s in range(0, len(X), 100000)]).cpu().numpy() for c in (0, 6)]
    box, cupboard, _ = D.scene_boxes(a.scene); inside = lambda q, b: ((q >= b[:, 0]) & (q <= b[:, 1])).all(1)
    c0, c1 = inside(pos[0], cupboard), inside(pos[1], cupboard)
    write(out/'complete.json', dict(complete=True, smoke=a.smoke, coords='full', center_rad=[0.0]*6, radians_per_unit=ma['radians_per_unit'], fraction=a.fraction, pairs=a.pairs,
        pairs_from_cabinet=n_cab, pairs_from_area=n_area, cabinet_source=str(a.cabinet), area_source=str(a.area), cabinet_files=mc['files'], area_files=ma['files'], seed=a.seed,
        pair_with_cupboard_point=float((c0 | c1).mean()), points_in_cupboard=float((c0.sum()+c1.sum())/(2*a.pairs)), x0_cupboard=float(c0.mean()), x1_cupboard=float(c1.mean()),
        points_in_box=float((inside(pos[0], box).sum()+inside(pos[1], box).sum())/(2*a.pairs)), elapsed_seconds=time.monotonic()-started,
        files={f: sha(out/(f+'.npy')) for f in parts}))
    print('MIX', json.dumps(dict(fraction=a.fraction, cab=n_cab, area=n_area, pair_with_cupboard_point=float((c0 | c1).mean()))), flush=True)


if __name__ == '__main__': main()
