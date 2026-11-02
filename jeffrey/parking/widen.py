"""Widening-car sweep: bays fixed at 0.23, divider 0.01, car width W, so the free space on each side of a centred
parked car (the side gap) is (0.23 - W) / 2. Tables and a summary from the registered runs (run.py --bay-width).

    ~/.venvs/bellman-ntfield/bin/python jeffrey/parking/widen.py [--out jeffrey/parking/report]

Per training seed: bay-to-bay successes with the fine MPPI (plan.FINE: step 0.005, 600 steps) and ntrl-demo's default
MPPI, both also driven by the true T* grid (oracle); aisle-to-bay control; T/T*; the slope of T(bay-A pose, car) along
bay B's centre line (theta = 0, x in [0, 0.25]) for field and truth; on the line x = 0.29 through both bays, the bay-B
level, the peak inside the divider and the slope inside it; and how the fine-MPPI failures of planning seed 0 end (left
the bay before the first contact / first contact on the way to the other bay / scraped a bay side / no contact).
Writes widen_tables.md and widen_summary.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plan as P  # noqa: E402
import scene as SC  # noqa: E402
from report import PS, collect, find_runs  # noqa: E402

GX = np.linspace(-0.5, 0.5, 201)      # pipeline slice grid (x index, y index), theta = 0


def total(planning, key):
    return sum(planning[f'{key}/{k}']['success'] for k in PS)


def endings(sc, paths, ok, gap):
    """How failed paths end: left the bay first / headed for the other bay / scraped a bay side / no contact."""
    modes = dict(left_bay=0, toward_other_bay=0, bay_side=0, no_contact=0)
    for i, pth in enumerate(paths):
        if i in ok:
            continue
        x = P.densify(torch.tensor(pth, dtype=torch.float64), 0.001)
        d, idx = sc.clearance(x)
        hit = torch.nonzero(d < 0)
        if not len(hit):
            modes['no_contact'] += 1
            continue
        k = int(hit[0])
        before = x[:k + 1].numpy()
        if (before[:, 0] < SC.X_D - SC.CAR_L / 2).any():
            modes['left_bay'] += 1
        elif np.sign(before[0, 1]) != np.sign(before[-1, 1]) or (int(idx[k]) == 4 and abs(before[-1, 1]) < abs(before[0, 1]) - 0.5 * gap):
            modes['toward_other_bay'] += 1
        else:
            modes['bay_side'] += 1
    return modes


def line_stats(y, t, C, yB):
    inside = np.abs(y) <= C / 2
    mid = inside & (np.abs(y) <= C / 4) & (y > 0)
    return dict(bay_b=float(np.interp(yB, y, t)), peak=float(t[inside].max()),
                wall_slope=float(abs(np.polyfit(y[mid], t[mid], 1)[0])) if mid.sum() > 2 else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=HERE / 'report')
    a = p.parse_args()
    groups = collect(find_runs([]), fixed_bays=True)
    rows = []
    for key, g in sorted(groups.items(), key=lambda kv: kv[0][1]):
        res = json.loads((Path(g['dirs'][0]) / 'results.json').read_text())
        car_w, wall, bay_w, gap = res['car_width'], res['wall'], res['bay_width'], res['side_gap']
        C = wall + car_w
        sc = SC.Scene(wall, 'cpu', torch.float64, car_w, bay_w)
        yB = SC.bay_y(wall, 'B', car_w, bay_w)
        tr = np.load(Path(g['dirs'][0]) / 'truth.npz')
        jB = int(round((yB + 0.5) / 0.005))
        xs = (GX >= 0.0) & (GX <= 0.25)
        truth = dict(corridor_slope=float(np.polyfit(GX[xs], tr['slice_star'][xs, jB], 1)[0]), **line_stats(tr['profile_y'], tr['profile_star'], C, yB))
        per = {}
        for sd in sorted(g['seeds'], key=lambda s: ['87358', '1', '2'].index(s)):
            s = g['seeds'][sd]
            fl = np.load(Path(s['dir']) / 'field.npz')
            paths = np.load(Path(s['dir']) / 'paths.npz', allow_pickle=True)['fine_spot_to_spot_0']
            per[sd] = dict(fine=total(s['planning'], 'fine/spot_to_spot'), default=total(s['planning'], 'default/spot_to_spot'),
                           aisle_fine=total(s['planning'], 'fine/aisle_to_spot'), aisle_default=total(s['planning'], 'default/aisle_to_spot'),
                           t_over_tstar=s['field']['t_over_tstar_median'],
                           corridor_slope=float(np.polyfit(GX[xs], fl['slice_pred'][xs, jB], 1)[0]),
                           **line_stats(tr['profile_y'], fl['profile_pred'], C, yB),
                           fine_failures_seed0=endings(sc, paths, set(s['planning']['fine/spot_to_spot/0']['success_ids']), gap), run=s['run'])
        rows.append(dict(car_w=car_w, side_gap=gap, wall=wall, bay_w=bay_w, cspace_wall=C, slope_needed=1 / float(sc.train_speed(torch.tensor(gap))),
                         oracle={st: sum(o[k]['success'] for k in PS) for st, o in res['oracle_settings'].items()},
                         straight=res['straight']['spot_to_spot']['success'], t_star_median=res['truth']['t_star_median'], truth=truth,
                         seeds=list(per), per_seed=per, runs=g['runs']))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'widen_summary.json').write_text(json.dumps(rows, indent=2))
    cell = lambda r, k: ' · '.join(f'{sd}: {r["per_seed"][sd][k]}' for sd in r['seeds'])
    lines = ['### ntrl-demo `metric`, bays fixed at 0.23, divider 0.01: car width sweep (side gap = (0.23 - W) / 2)', '',
             'Bay-to-bay suite (100 queries x planning seeds 0/1/2). fine = the same MPPI with step 0.005 and 600 steps; default = ntrl-demo\'s '
             'MPPI (step 0.015, 200 steps); oracle = that MPPI on the true T* grid. Slope needed = 1/S(side gap).', '',
             '|side gap|car width|slope needed|T* median|oracle fine / default /300|fine MPPI /300, training seed: successes|default MPPI /300|aisle control, fine|',
             '|---:|---:|---:|---:|---:|---|---|---|']
    for r in sorted(rows, key=lambda r: -r['side_gap']):
        lines.append(f"|{r['side_gap']:.4f}|{r['car_w']:g}|{r['slope_needed']:.2f}|{r['t_star_median']:.2f}|{r['oracle'].get('fine', '-')} / {r['oracle']['default']}|"
                     f"{cell(r, 'fine')}|{cell(r, 'default')}|{cell(r, 'aisle_fine')}|")
    lines += ['', '|side gap|seed|T/T*|slope along bay B, truth / field|bay B level on x = 0.29, truth / field|peak inside the divider, truth / field|'
              'slope inside the divider, truth / field|failed fine paths, planning seed 0: left bay first / toward other bay / bay side / no contact|',
              '|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in sorted(rows, key=lambda r: -r['side_gap']):
        t = r['truth']
        for sd in r['seeds']:
            f = r['per_seed'][sd]
            m = f['fine_failures_seed0']
            ws = f"{f['wall_slope']:.1f}" if f['wall_slope'] is not None else '-'
            lines.append(f"|{r['side_gap']:.4f}|{sd}|{f['t_over_tstar']:.2f}|{t['corridor_slope']:.2f} / {f['corridor_slope']:.2f}|{t['bay_b']:.2f} / {f['bay_b']:.2f}|"
                         f"{t['peak']:.2f} / {f['peak']:.2f}|{t['wall_slope']:.1f} / {ws}|{m['left_bay']} / {m['toward_other_bay']} / {m['bay_side']} / {m['no_contact']}|")
    (a.out / 'widen_tables.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
