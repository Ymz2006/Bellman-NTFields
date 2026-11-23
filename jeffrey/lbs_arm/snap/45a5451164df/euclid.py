"""The planner on the Euclidean field T(x, g) = |x - g| (no network, no training): the control that says what the planner settings
alone do on a prepared scene. Same planner (core.plan, ntrl-demo's MPPI, planning seeds 0/1/2), same queries and same scorer as the
field run of the scene (pipeline6.py / pipeline12.py). --steps, --samples: one result per pair, to see what a larger budget changes.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/lbs_arm/euclid.py --prepared <prepare run> --out <dir> [--steps 600 2400] [--samples 50 400]
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


class Euclid:
    lift = False

    def __init__(self, scene):
        self.scene = scene

    def encode(self, g):
        return g

    def to_goal(self, x, g, d=None):
        return torch.linalg.vector_norm(x - g, dim=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, nargs='+', default=[600])
    p.add_argument('--samples', type=int, nargs='+', default=[50])
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    info = json.loads((a.prepared / 'scene.json').read_text())
    if info['describe']['dim'] > 6:
        import arm12
        sys.modules['arm'] = arm12
    import arm as A
    import core as C
    from pipeline6 import summarize
    from common import Logger, dump
    a.out.mkdir(parents=True, exist_ok=True)
    log = Logger(a.out / 'log.txt')
    dev, t0 = 'cuda:0', time.time()
    scene, info = A.load_scene(a.prepared, dev)
    step_rad = info.get('score_step_rad', 0.002)
    Q = dict(np.load(a.prepared / 'queries.npz'))
    qs, qg = torch.tensor(Q['start'], device=dev), torch.tensor(Q['goal'], device=dev)
    if a.smoke:
        keep = np.arange(0, len(qs), 10)
        Q = {k: (v[keep] if getattr(v, 'shape', (0,))[:1] == (len(qs),) else v) for k, v in Q.items()}
        qs, qg = qs[keep], qg[keep]
    ev = Euclid(scene)
    dist = torch.linalg.vector_norm(qs - qg, dim=1)
    res = dict(stage='lbsarm-euclid', scene=scene.describe(), prepared=str(a.prepared), smoke=a.smoke, score_step_rad=step_rad, distance_median=float(dist.median()), runs=[])
    for steps in a.steps:
        for samples in a.samples:
            planner = C.planner_settings(scene, sample_num=samples, horizon=5)
            row = dict(planner=dict(planner, steps=steps), plan={})
            total = np.zeros(len(qs), int)
            for seed in ([0] if a.smoke else [0, 1, 2]):
                paths, conv = C.plan(ev, qs, qg, seed, steps, **planner)
                rows = C.score(scene, paths, conv.cpu().numpy(), step_rad=step_rad)
                out, ok = summarize(rows, Q)
                cv = conv.cpu().numpy()
                n = np.array([len(pth) - 2 for pth in paths])
                out['steps_per_straight_step_median'] = float(np.median((n / (dist.cpu().numpy() / planner['step']))[cv])) if cv.any() else None
                total += ok
                row['plan'][str(seed)] = out
                log(f'steps {steps} samples {samples} seed {seed}: {out["success"]}/{out["runs"]}; arrived in contact {out["arrived_contact"]}, did not arrive {out["not_arrived"]}; '
                    f'line-feasible {out["line_feasible"]}; detour {out["detour"]}; steps / straight steps (arrived) {out["steps_per_straight_step_median"]}')
            row['success'] = [int(total.sum()), int(len(qs) * len(row['plan']))]
            res['runs'].append(row)
    res['elapsed_seconds'] = time.time() - t0
    dump(a.out / 'results.json', res)
    log(f'done in {res["elapsed_seconds"] / 60:.1f} min')


if __name__ == '__main__':
    main()
