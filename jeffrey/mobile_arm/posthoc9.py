"""Read-only forensics of finished mobile-arm runs (planning seed 0 paths, paths_seed0.npy) -> report/posthoc_runs.json.

Per run and query: the first contact (base or arm against a wall; floor / base box / self), the base's own clearance there, the arm's
horizontal reach there (how far the farthest sphere sits from the base centre); for every run that touched nothing: the arm's reach while
the base was within 0.5 cells of a wall vs in the open (does the field fold the arm next to walls?).

    CUDA_VISIBLE_DEVICES=1 python jeffrey/mobile_arm/posthoc9.py
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scene9 as S9  # noqa: E402
import plan9 as P9  # noqa: E402

ROOT = HERE.parents[1]


def reach(scene, q):
    """Largest horizontal distance (cells) of an arm sphere's footprint from the base centre."""
    _, _, _, qj = scene.to_cells(q)
    p = scene.arm.spheres(qj)
    return (torch.linalg.vector_norm(p[..., :2], dim=-1) + scene.arm.radius).amax(1)


def analyse(scene, run_dir, Q, limit=None):
    paths = np.load(run_dir / 'paths_seed0.npy', allow_pickle=True)
    rows = np.load(run_dir / 'rows.npz', allow_pickle=True)
    succ, conv, coll = rows['success'][0], rows['converged'][0], rows['collision'][0]
    out = dict(contact=[], near_reach=[], open_reach=[])
    near_all, open_all, closest = [], [], []
    for i, w in enumerate(paths[:limit]):
        path = torch.tensor(w, device=scene.device)
        if conv[i] and coll[i]:
            dense = P9.densify(path, 0.005 * scene.pitch)
            d = scene.clearance(dense)
            k = int(torch.nonzero(d < 0)[0])
            pt = scene.parts(dense[k:k + 1])
            x, y, _, _ = scene.to_cells(dense[k:k + 1])
            out['contact'].append(dict(query=i, rect=float(pt['rect'][0]), arm=float(pt['arm'][0]), int=float(pt['int'][0]), reach=float(reach(scene, dense[k:k + 1])[0]),
                                       frac=k / max(len(dense) - 1, 1), depth=float(-d.min()), x=float(x[0]), y=float(y[0])))
        elif succ[i]:
            dense = P9.densify(path, 0.02 * scene.pitch)
            closest.append(float(scene.clearance(dense).min()))
            pt = scene.parts(path[:-1])
            rch = reach(scene, path[:-1])
            near = pt['rect'] < 0.5
            if bool(near.any()):
                near_all.append(rch[near].cpu().numpy())
            if bool((~near).any()):
                open_all.append(rch[~near].cpu().numpy())
    near_all = np.concatenate(near_all) if near_all else np.zeros(0)
    open_all = np.concatenate(open_all) if open_all else np.zeros(0)
    c = out['contact']
    arm_first = [x for x in c if x['arm'] < 0 and x['arm'] <= x['rect']]
    base_first = [x for x in c if x['rect'] < 0 and x['rect'] < x['arm']]
    return dict(contacts=len(c), arm_first=len(arm_first), base_first=len(base_first), internal=len(c) - len(arm_first) - len(base_first),
                arm_first_base_clear=float(np.mean([x['rect'] > 0 for x in arm_first])) if arm_first else None,
                arm_first_base_clearance_median=float(np.median([x['rect'] for x in arm_first])) if arm_first else None,
                arm_first_reach_median=float(np.median([x['reach'] for x in arm_first])) if arm_first else None,
                reach_near_wall_median=float(np.median(near_all)) if len(near_all) else None, reach_open_median=float(np.median(open_all)) if len(open_all) else None,
                tucked_share_near_wall=float((near_all < 0.3).mean()) if len(near_all) else None, tucked_share_open=float((open_all < 0.3).mean()) if len(open_all) else None,
                waypoints_near=int(len(near_all)), waypoints_open=int(len(open_all)),
                contact_depth_median=float(np.median([x['depth'] for x in c])) if c else None, contact_depth_q90=float(np.quantile([x['depth'] for x in c], 0.9)) if c else None,
                shallow_contacts=int(sum(x['depth'] < 0.1 for x in c)),
                success_closest_median=float(np.median(closest)) if closest else None, success_closest_q10=float(np.quantile(closest, 0.1)) if closest else None,
                contact_xy=[[round(x['x'], 1), round(x['y'], 1)] for x in c])


def main():
    q = HERE / 'queue'
    houses = json.loads((q / 'houses.json').read_text())
    state = json.loads((q / 'state.json').read_text())
    out_f = HERE / 'report/posthoc_runs.json'
    out = json.loads(out_f.read_text()) if out_f.exists() else {}
    scenes = {}
    for job, rid in state['done'].items():
        label, key, seed, extra = job.split('|')
        if rid in out or label not in ('ntrl', 'ctl', 'mid'):
            continue
        run_dir = ROOT / 'results' / f'registered_{rid}'
        if not (run_dir / 'results.json').exists():
            continue
        if key not in scenes:
            scenes[key] = S9.load_scene(ROOT / houses[key])[0]
        sc = scenes[key]
        Q = np.load(ROOT / houses[key] / 'queries.npz')
        r = analyse(sc, run_dir, Q)
        r.update(house=key, label=label, seed=seed)
        out[rid] = r
        print(job, rid, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
        out_f.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
