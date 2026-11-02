"""Registered Bellman pipeline with the hierarchical point-field speed (bellman.py), ending on the frozen cabinet suites.

prepare (bellman.py prepare: baseline seed states and bank transitions, local costs from the hierarchical speed)
-> training (bellman.py train: face_speed/training_pieces.py unchanged) -> experiments/cabinet/evaluate.py --method bellman.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, '/clean/method')
from common import sha, write
HERE = '/clean/experiments/point_field'; CABINET = '/clean/experiments/cabinet'
p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); p.add_argument('--scene', type=Path, required=True)
p.add_argument('--tests', type=Path, required=True); p.add_argument('--sampled', type=Path, required=True); p.add_argument('--cover', choices=['learned', 'exact'], required=True)
p.add_argument('--field', type=Path); p.add_argument('--reference', type=Path); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
started = time.monotonic(); timing = {}; smoke = ['--smoke'] if a.smoke else []; o = a.output
steps = [('prepare', f'{HERE}/bellman.py', ['prepare', '--output', o/'prepared', '--sampled', a.sampled, '--scene', a.scene, '--cover', a.cover]
          +(['--field', a.field] if a.field else [])+(['--reference', a.reference] if a.reference else [])+smoke),
         ('training', f'{HERE}/bellman.py', ['train', '--output', o/'training', '--prepared', o/'prepared']+smoke),
         ('evaluate', f'{CABINET}/evaluate.py', ['--method', 'bellman', '--checkpoint', o/'training/checkpoint.pt', '--scene', a.scene, '--tests', a.tests, '--output', o/'evaluation']+smoke)]
base = dict(method='bellman', cover='point-field-'+a.cover, field=str(a.field) if a.field else None, reference=str(a.reference) if a.reference else None, smoke=a.smoke, scene=str(a.scene), sampled=str(a.sampled))
for stage, script, extra in steps:
    write(o/'status.json', dict(complete=False, stage=stage, timing=timing, **base))
    begin = time.monotonic()
    with (o/(stage+'.log')).open('w') as log:
        subprocess.run([sys.executable, '-u', script, *map(str, extra)], stdout=log, stderr=subprocess.STDOUT, check=True)
    timing[stage] = time.monotonic()-begin
    print('STAGE', stage, timing[stage], flush=True)
result = json.loads((o/'evaluation/evaluation.json').read_text())
write(o/'status.json', dict(complete=True, stage='complete', scene_sha256=sha(a.scene), tests_manifest_sha256=sha(a.tests/'suite_manifest.json'), timing=timing,
                             elapsed_seconds=time.monotonic()-started, totals=result['totals'], mppi_successes=result['mppi_successes'], native_success=result['native_success'],
                             mode=f'point-field-bellman-{a.cover}', **base))
