"""Training-seed summary of gap-curriculum runs (read-only).

    summary.py --group B10=registered_<a>,registered_<b>,registered_<c> --group C10=... [--reference B10]

Per group: RK4 /600 and release MPPI /1800 for each training seed, their mean and range; with --reference, the
per-seed difference to the reference group's run of the same training seed (same initialization and source order).
"""
import argparse, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[2]


def totals(run):
    root = HERE/'results'/Path(run).name; s = json.loads((root/'status.json').read_text()); c = json.loads((root/'training/config.json').read_text())
    return c['seed'], s['totals']['native_rk4']['success'], s['totals']['matched_mppi']['success']


def main():
    p = argparse.ArgumentParser(); p.add_argument('--group', action='append', required=True); p.add_argument('--reference'); a = p.parse_args()
    groups = {}
    for g in a.group:
        name, runs = g.split('=', 1); groups[name] = {seed: (rk4, mppi) for seed, rk4, mppi in map(totals, runs.split(','))}
    ref = groups.get(a.reference)
    print('|Group|Seeds|RK4 /600 per seed|RK4 mean|MPPI /1800 per seed|MPPI mean|' + ('Δ RK4 vs ref per seed|Δ MPPI vs ref per seed|' if ref else ''))
    print('|---|---|---|---:|---|---:|' + ('---|---|' if ref else ''))
    for name, rows in groups.items():
        seeds = sorted(rows); rk4 = [rows[s][0] for s in seeds]; mppi = [rows[s][1] for s in seeds]
        line = f"|{name}|{', '.join(map(str, seeds))}|{', '.join(map(str, rk4))}|{np.mean(rk4):.1f}|{', '.join(map(str, mppi))}|{np.mean(mppi):.1f}|"
        if ref:
            common = [s for s in seeds if s in ref]
            line += ', '.join(f'{rows[s][0]-ref[s][0]:+d}' for s in common) + '|' + ', '.join(f'{rows[s][1]-ref[s][1]:+d}' for s in common) + '|'
        print(line)


if __name__ == '__main__': main()
