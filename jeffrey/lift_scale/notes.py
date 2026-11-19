"""Write a one-line conclusion on every finished lift-scale run in the experiment registry (registry.py note), once per run.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/lift_scale/notes.py
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REG = ROOT / 'tools/experiment_registry/registry.py'


def main():
    done_f = HERE / 'report/.noted.json'
    noted = set(json.load(open(done_f))) if done_f.exists() else set()
    d = json.load(open(HERE / 'report/runs.json'))
    n = 0
    for r in d['runs']:
        if r['id'] in noted:
            continue
        rho = '-' if r['rho'] is None else f"{r['rho']:.2f}"
        note = (f"lift_scale {r['scene']} {r['label']} seed {r['seed']}: {r['solved']:.1f} % solved ({r['wall']:.1f} % through walls, {r['stall']:.1f} % stopped short), "
                f"{r['within10']:.1f} % of travel times within 10 %, detour index {rho}, S x slope innermost {r['wall_innermost']:.2f}. See jeffrey/lift_scale/README.md.")
        p = subprocess.run([sys.executable, str(REG), 'note', r['id'], note], capture_output=True, text=True)
        if p.returncode == 0:
            noted.add(r['id'])
            n += 1
        else:
            print('note failed', r['id'], p.stderr[-200:])
    for key, b in d['blocked'].items():
        if b['id'] in noted:
            continue
        note = f"lift_scale blocked flags {key}: {100 * b['blocked_share']:.1f} % of the training pairs blocked (48-point check misses {100 * b['missed_by_48']:.1f} % of them)."
        if subprocess.run([sys.executable, str(REG), 'note', b['id'], note], capture_output=True, text=True).returncode == 0:
            noted.add(b['id'])
            n += 1
    json.dump(sorted(noted), open(done_f, 'w'))
    print(f'{n} notes written, {len(noted)} runs noted in all')


if __name__ == '__main__':
    main()
