"""Score an LBS arm field with the cabinet protocol of the earlier baselines (experiments/ntrl_cabinet/evaluate.py, unchanged; container
layout). That module is imported as is and run with --coords release --no-release; only its two field hooks are replaced, as
jeffrey/joint_range/evaluate.py does:

    FIELD.load_metric -> the (lifted) network of core.py with the scene of arm.py attached
    FIELD.ntrl_mppi   -> ntrl-demo's MPPI in the training coordinates (core.plan: the same sampling, cost and update as
                         experiments/ntrl_cabinet/field.py ntrl_mppi; every joint clamped to its limits), with the settings given

So the frozen suites, the planning seeds 0 / 1 / 2, the native RK4 and the release scorer (obstacle clearance >= 0.005 m, self clearance
>= 0.002 m, joint limits, endpoints; tilt gate off) are those of the registered ntrl-demo and P-NTFields runs. The release MPPI is not
run. Writes <output>/evaluation/evaluation.json in the baselines' format and <output>/results.json.
"""
import sys
sys.path.insert(0, '/clean/method')
import argparse
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arm as A  # noqa: E402
import core as C  # noqa: E402

_spec = importlib.util.spec_from_file_location('ntrl_cabinet_evaluate', '/clean/experiments/ntrl_cabinet/evaluate.py')
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)
_spec = importlib.util.spec_from_file_location('ntrl_demo_metric_network', '/clean/baselines/ntrl-demo/models/metric/model_network_metric.py')
MN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MN)


class Field(nn.Module):
    def __init__(self, net, scene):
        super().__init__()
        self.net, self.scene = net, scene
        self.unit = scene.q_range.cpu().numpy().astype(np.float32)             # planning unit: release coordinates = training coordinates
        self.ev = C.Compiled(net, scene, C.Clearance(scene, compile=False), compile=False)

    def forward(self, q, goal):
        coords = torch.cat((q, goal), dim=1)
        if torch.is_grad_enabled() and (q.requires_grad or goal.requires_grad):
            return E.FIELD._Value.apply(coords, self.net)
        with torch.no_grad():
            return self.net.out(coords)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--field', type=Path, required=True, help='a field run (results.json, training/checkpoint.pt)')
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--tests', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    t0 = time.time()
    fr = json.loads((a.field / 'results.json').read_text())
    pl = fr['planner']
    scene, info = A.load_scene(a.prepared, 'cuda:0')

    def load(path, device='cuda:0', variant='metric', coords='release', metric_scale=None):
        return Field(C.load_field(MN, path, scene, device), scene).eval()

    @torch.no_grad()
    def ntrl_mppi(field, suite, count, seed, steps=200, sample_num=50, horizon=5, step=.015, temperature=50., tol=.01, device='cuda:0'):
        unit = field.unit
        s = torch.tensor(suite['starts_rad'][:count] / unit, dtype=torch.float32, device=device)
        g = torch.tensor(suite['goals_rad'][:count] / unit, dtype=torch.float32, device=device)
        paths, conv = C.plan(field.ev, s, g, seed, steps, sample_num=sample_num, horizon=horizon, step=step, temperature=temperature, tol=tol)
        result = np.empty(count, dtype=object)
        for i, path in enumerate(paths):
            rad = path.cpu().numpy().astype(np.float64) * unit
            rad[0] = suite['starts_rad'][i]
            rad[-1] = suite['goals_rad'][i]
            result[i] = rad.astype(np.float32)
        return result, conv.cpu().numpy()

    E.FIELD.load_metric = load
    E.FIELD.ntrl_mppi = ntrl_mppi
    sys.argv = ['evaluate.py', '--checkpoint', str(a.field / 'training/checkpoint.pt'), '--variant', 'metric', '--coords', 'release', '--output', str(a.output / 'evaluation'),
                '--tests', str(a.tests), '--scene', str(a.scene), '--no-release', '--ntrl-steps', str(pl['steps']), '--ntrl-samples', str(pl['sample_num']),
                '--ntrl-horizon', str(pl['horizon']), '--ntrl-step', repr(pl['step']), '--ntrl-temperature', repr(pl['temperature']), '--ntrl-tol', repr(pl['tol'])] + (['--smoke'] if a.smoke else [])
    E.main()
    ev = json.loads((a.output / 'evaluation/evaluation.json').read_text())
    res = dict(stage='lbsarm-eval', field=str(a.field), arm=fr['arm'], lift=fr['lift'], args=fr['args'], seed=fr['seed'], planner=pl, totals=ev['totals'],
               own_scorer=dict(success=fr['success'], by_task=fr['success_by_task'], detour=fr['detour']), smoke=a.smoke, seconds=time.time() - t0)
    (a.output / 'results.json').write_text(json.dumps(res, indent=1))
    print('LBSARM_EVAL', json.dumps({k: {kk: vv for kk, vv in v.items()} for k, v in ev['totals'].items()}), flush=True)


if __name__ == '__main__':
    main()
