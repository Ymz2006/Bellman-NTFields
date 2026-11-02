"""Evaluate a joint-range checkpoint with the ntrl-demo cabinet protocol (experiments/ntrl_cabinet/evaluate.py, unchanged).

That module is imported as is and run with --coords release; only its two field hooks are replaced:
FIELD.load_metric -> field.load_affine with the training data's coordinate map (--data-map, the data's complete.json),
FIELD.ntrl_mppi   -> field.ntrl_mppi (ntrl-demo's MPPI in release coordinates).
So native RK4, the release MPPI (1536 samples, horizon 7, <= 400 steps, step 0.01, beta 12.5, goal weight 4),
ntrl-demo's MPPI (50 samples, horizon 5, 200 steps, step 0.015, temperature 50, tol 0.01), the planning seeds
0/1/2 and the release scorer (tilt gate off) are identical for both scalings, in radians.
Checks written to <output>/scaling_check.json:
  identity map (full data): the replaced ntrl MPPI and field reproduce the original ones exactly (one suite, seed 0)
  any map: the field equals the network evaluated on the mapped coordinates at the test endpoints, and the share of
           frozen test endpoints outside the training data's range per joint.
"""
import sys
sys.path.insert(0, '/clean/method'); sys.path.insert(0, '/clean/jeffrey/joint_range')
import argparse, importlib.util, json
from pathlib import Path
import numpy as np
import torch
from common import write
import field as AF
_spec = importlib.util.spec_from_file_location('ntrl_cabinet_evaluate', '/clean/experiments/ntrl_cabinet/evaluate.py')
E = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(E)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--data-map', type=Path, required=True); p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--variant', default='metric'); p.add_argument('--output', type=Path, required=True); p.add_argument('--tests', type=Path, required=True)
    a, rest = p.parse_known_args()
    m = json.loads(a.data_map.read_text()); center, unit = m['center_rad'], m['radians_per_unit']
    original = (E.FIELD.load_metric, E.FIELD.ntrl_mppi)
    E.FIELD.load_metric = lambda path, device='cuda:0', variant='metric', coords='release', metric_scale=None: AF.load_affine(path, center, unit, device, variant, metric_scale)
    E.FIELD.ntrl_mppi = AF.ntrl_mppi
    sys.argv = ['evaluate.py', '--checkpoint', str(a.checkpoint), '--variant', a.variant, '--coords', 'release', '--output', str(a.output), '--tests', str(a.tests), *rest]
    E.main()

    # checks
    field = AF.load_affine(a.checkpoint, center, unit, 'cuda:0', a.variant)
    manifest = json.loads((a.tests/'suite_manifest.json').read_text()); first = a.tests/'suites'/manifest['suites'][0]['file']
    suites = [np.load(a.tests/'suites'/s['file'], allow_pickle=True) for s in manifest['suites']]
    ends = np.concatenate([np.concatenate([s['starts_rad'], s['goals_rad']]) for s in suites]).astype(np.float64)
    lo, hi = np.asarray(center)-np.asarray(unit)/2, np.asarray(center)+np.asarray(unit)/2
    R = np.load(AF.NORM)['q_range']
    q = torch.tensor(np.concatenate([np.concatenate([s['starts_rad'], s['goals_rad']], 1) for s in suites])/np.tile(R, 2), dtype=torch.float32, device='cuda:0')
    with torch.no_grad():
        mine = field(q[:, :6], q[:, 6:])[:, 0]
        manual = field.net.out(((q.double()*torch.tensor(np.tile(R, 2), device='cuda:0').double()-torch.tensor(np.tile(center, 2), device='cuda:0').double())
                                /torch.tensor(np.tile(unit, 2), device='cuda:0').double()).float())[0][:, 0]
    check = dict(map=dict(center_rad=center, radians_per_unit=unit, identity=field.identity), planner_coordinates='release (x = q / q_range) for every planner',
                 field_vs_manual_map_max_abs=float((mine-manual).abs().max()), field_value_median=float(mine.median()),
                 test_endpoints_outside_data_range=[float(((ends[:, j] < lo[j]-1e-6) | (ends[:, j] > hi[j]+1e-6)).mean()) for j in range(6)])
    if field.identity:
        suite = np.load(first, allow_pickle=True); ref = original[0](a.checkpoint, 'cuda:0', a.variant, 'release', None)
        p0, c0 = original[1](ref, suite, 100, 0); p1, c1 = AF.ntrl_mppi(field, suite, 100, 0)
        check['identity_ntrl_mppi_paths_equal'] = bool(all(np.array_equal(x, y) for x, y in zip(p0, p1)) and np.array_equal(c0, c1))
        with torch.no_grad():
            check['identity_field_max_abs'] = float((ref(q[:, :6], q[:, 6:])[:, 0]-mine).abs().max())
    write(a.output/'scaling_check.json', check); print('SCALING_CHECK', json.dumps(check), flush=True)


if __name__ == '__main__': main()
