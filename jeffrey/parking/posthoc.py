"""Post-hoc analysis of registered parking runs (reads their saved paths and profiles; trains nothing).

    CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/parking/posthoc.py [--out jeffrey/parking/report]

Per (variant, car width, wall, training seed), for the default-MPPI bay-to-bay paths of planning seeds 0/1/2:
  - divider clearance: the minimum exact clearance between the car and the divider box along each path (interpolated
    at <= 0.001 per coordinate); median over successful paths and the share of paths that pass within 0.01 of it;
  - bay-B tilt: along the profile line (x = 0.29, theta = 0), the drop of T(bay-A pose, .) across bay B's free band
    from its far side to the side next to the divider, for the field and for T* (a field that pulls through the
    wall is lower next to the divider; T* is not).
Writes posthoc.json and posthoc.md next to the report tables.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scene as SC  # noqa: E402
import plan as P  # noqa: E402
from report import find_runs, DEFAULT_W  # noqa: E402


def divider_clearance(scene, path):
    dense = P.densify(torch.tensor(path, device=scene.device, dtype=torch.float32))
    d = torch.cat([scene.signed(dense[i:i + 65536])[:, 4] for i in range(0, len(dense), 65536)])
    return float(d.min())


def tilt(profile_y, values, wall, car_w):
    """T at the far edge of bay B's free band minus T next to the divider (0.01 inside the band on each side)."""
    yb = SC.bay_y(wall, 'B', car_w)
    far, near = yb - SC.CLEAR + 0.01, yb + SC.CLEAR - 0.01
    return float(np.interp(far, profile_y, values) - np.interp(near, profile_y, values))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='*', default=[])
    p.add_argument('--out', type=Path, default=HERE / 'report')
    a = p.parse_args()
    rows = []
    for rid, out, res in find_runs(a.runs):
        car_w = res.get('car_width', DEFAULT_W)
        scene = SC.Scene(res['wall'], 'cuda:0', car_w=car_w)
        truth = np.load(out / 'truth.npz')
        tilt_star = tilt(truth['profile_y'], truth['profile_star'], res['wall'], car_w)
        for seed, sr in res.get('seeds_results', {}).items():
            paths = np.load(out / f'seed{seed}' / 'paths.npz', allow_pickle=True)
            field = np.load(out / f'seed{seed}' / 'field.npz')
            clear, close = [], 0
            total = 0
            for ps in ['0', '1', '2']:
                ok = set(sr['planning'][f'default/spot_to_spot/{ps}']['success_ids'])
                for i, path in enumerate(paths[f'default_spot_to_spot_{ps}']):
                    c = divider_clearance(scene, path)
                    total += 1
                    close += c < 0.01
                    if i in ok:
                        clear.append(c)
            rows.append(dict(run=rid, variant=res['variant'], car_w=car_w, wall=res['wall'], seed=seed, paths=total,
                             divider_clearance_median=float(np.median(clear)) if clear else None,
                             divider_clearance_p10=float(np.percentile(clear, 10)) if clear else None,
                             within_001=int(close), tilt_pred=tilt(truth['profile_y'], field['profile_pred'], res['wall'], car_w), tilt_star=tilt_star))
            print(json.dumps(rows[-1]), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r['variant'], -r['car_w'], -r['wall'], r['seed']))
    json.dump(rows, open(a.out / 'posthoc.json', 'w'), indent=2)
    lines = ['|variant|car W|wall w|training seed|median divider clearance of successful paths|10th percentile|paths within 0.01 of the divider /300|'
             'bay-B tilt toward the divider: field|truth|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        f = lambda v: 'n/a' if v is None else f'{v:.3f}'
        lines.append(f"|{r['variant']}|{r['car_w']:g}|{r['wall']:g}|{r['seed']}|{f(r['divider_clearance_median'])}|{f(r['divider_clearance_p10'])}|{r['within_001']}|"
                     f"{r['tilt_pred']:.3f}|{r['tilt_star']:.3f}|")
    (a.out / 'posthoc.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
