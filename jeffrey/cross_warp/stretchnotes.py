"""Registry notes for the final test's runs (zq --stretch): one line per field in each run, with its control.
Usage: stretchnotes.py [--dry]. Skips runs that already have notes."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
import stretchreport as SR

REG = ['python3', str(ROOT / 'tools/experiment_registry/registry.py')]
LAB = dict(id='ntrl-demo', pmzw='field trained on the warped maze with warped-maze pairs')


def main():
    R, runs = SR.collect()
    T = SR.table_data(R, runs)
    by = {}
    for (base, k, kind, lab, seed), e in runs.items():
        p = e['plans']
        c = lambda pl: f'{SR.pct(p[pl]["success_total"]):.1f} %' + (f' (arrived after a wall {p[pl]["arrived_wall"]}, never arrived {p[pl]["not_arrived"]})')
        row = T[f'{base}/{k}/{lab}']
        ctl = row['own_x'] if lab == 'id' else row['own_z']
        line = f'{LAB[lab]} trained on gap x1, seed {seed}, {SR.BNAME[base]}, gap x{k}: MPPI on the maze {c("x")}'
        if 'z' in p:
            line += f'; constant step in its own coordinates {c("z")}'
        line += f'; control, the same training on this gap (seed 87358): {ctl:.1f} %' if ctl is not None else ''
        by.setdefault(e['run'], []).append((kind, line))
    for rid, lines in sorted(by.items()):
        kind = lines[0][0]
        head = ('Final test (2026-11-05): field trained on gap x1 planned on this gap through stretch.py\'s map. ' if kind == 'plus' else
                'Final test control (2026-11-05): gap x1 checkpoint at the same normalized coordinates, no plus-shaped stretch. ')
        note = head + ' | '.join(l for _, l in lines) + '. Report: jeffrey/cross_warp/README_gap.md "The x1 field on the wider gaps".'
        shown = subprocess.run(REG + ['show', rid], capture_output=True, text=True).stdout
        if '"notes": ""' not in shown and '"notes": null' not in shown and '"notes"' in shown:
            print('has notes, skipped', rid)
            continue
        print(rid, note[:200])
        if '--dry' not in sys.argv:
            subprocess.run(REG + ['note', rid, note], check=True, capture_output=True)


if __name__ == '__main__':
    main()
