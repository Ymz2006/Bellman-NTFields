"""How deep inside the cupboard the test queries put the grasp points, per placement: report/depth.json.

For every placement saved under models/ (finalize.py): forward kinematics of the start and goal configurations of its test queries
(demo.skeleton), depth = grasp-point x minus the x of the cupboard's front face (base x + dx). Reported for the ends that the suite
says are inside the cupboard (arm_ends_in_cupboard), over all queries and over the zone -> another zone stratum.

Control for "does distance help, or do the queries just get shallower": every query gets the largest depth of any arm at either end
(<= 0: no arm inside), and the lift field's solved share with the two-ended planner (the evalfield runs' flags.npz, all planning seeds
and training seeds, --steps per end) is reported per depth bin, so placements can be compared at equal depth.

    CUDA_VISIBLE_DEVICES=0 ~/.venvs/bellman-ntfield/bin/python jeffrey/arm_placement/depth.py [--steps 2400]
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'lbs_arm'))
import arm12  # noqa: E402
sys.modules['arm'] = arm12
import demo as DM  # noqa: E402


def stats(d):
    return dict(n=int(len(d)), median=float(np.median(d)), p90=float(np.percentile(d, 90)), max=float(d.max()), min=float(d.min())) if len(d) else None


BINS = [(-9.0, 0.0, 'no arm inside'), (0.0, 0.07, 'up to 7 cm inside'), (0.07, 0.13, '7 to 13 cm'), (0.13, 0.25, '13 to 25 cm'), (0.25, 9.0, 'deeper than 25 cm')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=2400)
    a = ap.parse_args()
    S = {p['name']: p for p in json.loads((HERE / 'report/summary.json').read_text())['placements']}
    M = HERE / 'models'
    idx = json.loads((M / 'index.json').read_text())['placements']
    out = {}
    for name, row in idx.items():
        scene, _ = arm12.load_scene(M / name, 'cuda:0')
        Q = np.load(M / name / 'queries.npz', allow_pickle=True)
        names = [str(t) for t in Q['task_names']]

        def depth(x):
            sk = DM.skeleton(scene, torch.tensor(x, device='cuda:0', dtype=torch.float32)).cpu().numpy()
            return sk[:, :, 7, 0] - (sk[:, :, 0, 0] + row['dx'])
        ds, dg = depth(Q['start']), depth(Q['goal'])
        both = np.concatenate([ds, dg])                                     # (2N, arms)
        inside = both > 0
        c2c = np.tile(Q['task'] == names.index('compartment_to_compartment'), 2)
        out[name] = dict(dx=row['dx'], half_spacing=row['half_spacing'], base_height=row['base_height'], inside_all=stats(both[inside]), zone_to_zone=stats(both[c2c].ravel()))
        print(name, {k: round(v, 3) for k, v in out[name]['zone_to_zone'].items()})
        qd = np.maximum(ds.max(1), dg.max(1))                               # per query: deepest arm at either end
        flags = []
        for fr in S.get(name, {}).get('fields', []):
            if fr['label'] != 'lift':
                continue
            for q in fr.get('planners', []):
                if q['kind'] == 'bidir' and q['steps'] == a.steps and q['samples'] == 50 and q.get('suite') == 'placement':
                    z = np.load(HERE.parents[1] / f'results/registered_{q["run"]}/flags.npz')
                    flags += [z[f'lift|bidir|{a.steps}|50|5|{k}'] for k in (0, 1, 2)]
        if flags:
            F = np.stack(flags)                                             # (runs, queries)
            out[name]['by_depth'] = dict(steps=a.steps, runs_per_query=int(len(F)), bins=[dict(label=lab, lo=lo, hi=hi, queries=int(((qd > lo) & (qd <= hi)).sum()), solved=int(F[:, (qd > lo) & (qd <= hi)].sum()),
                                                                                         runs=int(F[:, (qd > lo) & (qd <= hi)].size)) for lo, hi, lab in BINS])
            print('   by depth', [(b['label'], b['solved'], b['runs']) for b in out[name]['by_depth']['bins']])
    (HERE / 'report/depth.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
