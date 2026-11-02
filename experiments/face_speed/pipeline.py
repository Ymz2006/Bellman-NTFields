"""Registered per-piece speed pipelines on the cabinet task, ending on the frozen cabinet test suites.

metric   (Task 1)  metric_data.py with the cover clearance -> experiments/cabinet/metric/train.py (ntrl-demo,
                   unchanged) -> experiments/cabinet/evaluate.py --method metric (unchanged). With --data an
                   existing metric data directory is reused instead (the same-environment baseline rerun).
bellman  (Task 3)  prepare_pieces.py (per-piece labels of an existing cabinet FK pool and coverage set, the
                   per-piece speed network, the offline bank with it) -> training_pieces.py (the unchanged
                   338,177-parameter TravelCost, losses 6/2/2, seed 87358, 5 rounds, with the per-piece speed
                   as the frozen coefficient) -> evaluate.py --method bellman (unchanged).

Each stage is logged; status.json records the stage, so a run that stops after training can be scored by
`run.py face-speed-evaluate --run ...`.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, '/clean/method')
from common import sha, write
HERE = '/clean/experiments/face_speed'; CABINET = '/clean/experiments/cabinet'
p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); p.add_argument('--scene', type=Path, required=True)
p.add_argument('--tests', type=Path, required=True); p.add_argument('--method', choices=['bellman', 'metric'], required=True)
p.add_argument('--cover', choices=['faces', 'spheres', 'links', 'split', 'environment', 'release'], required=True)
p.add_argument('--speed-model', choices=['shared', 'separate'], default='shared', help='bellman: one trunk with an output per piece, or one network per piece')
p.add_argument('--data', type=Path, help='metric: reuse this data directory (baseline rerun)')
p.add_argument('--sampled', type=Path, help='bellman: sampled/ directory of a cabinet run (FK pool, coverage, inputs)')
p.add_argument('--speed-steps', type=int, default=20000); p.add_argument('--filter', choices=['cover', 'release'], default='cover', help='metric: clearance of the x0 filter')
p.add_argument('--margin', type=float); p.add_argument('--offset', type=float); p.add_argument('--pairs', type=int)
p.add_argument('--smoke', action='store_true'); a = p.parse_args()
label = sum(([f'--{k}', str(v)] for k, v in (('margin', a.margin), ('offset', a.offset), ('pairs', a.pairs)) if v is not None), [])
started = time.monotonic(); timing = {}; smoke = ['--smoke'] if a.smoke else []; o = a.output
evaluate = (f'{CABINET}/evaluate.py', ['--method', a.method, '--checkpoint', o/'training/checkpoint.pt', '--scene', a.scene, '--tests', a.tests, '--output', o/'evaluation']+smoke)
if a.method == 'metric':
    data = a.data or o/'data'
    steps = ([] if a.data else [('data', f'{HERE}/metric_data.py', ['--scene', a.scene, '--output', data, '--cover', a.cover, '--filter', a.filter]+label+smoke)])
    steps += [('training', f'{CABINET}/metric/train.py', ['--data', data, '--output', o/'training']+smoke), ('evaluate', *evaluate)]
else:
    assert a.sampled, 'bellman needs --sampled'
    steps = [('prepare', f'{HERE}/prepare_pieces.py', ['--sampled', a.sampled, '--output', o/'prepared', '--scene', a.scene, '--cover', a.cover, '--speed-steps', str(a.speed_steps), '--speed-model', a.speed_model]+smoke),
             ('training', f'{HERE}/training_pieces.py', ['--prepared', o/'prepared', '--output', o/'training']+smoke), ('evaluate', *evaluate)]
base = dict(method=a.method, cover=a.cover, speed_model=a.speed_model if a.method == 'bellman' else None, filter=a.filter if a.method == 'metric' else None, margin=a.margin, offset=a.offset, pairs=a.pairs, smoke=a.smoke, scene=str(a.scene), data=str(a.data) if a.data else None, sampled=str(a.sampled) if a.sampled else None)
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
                             mode=f'face-speed-{a.method}-{a.cover}', **base))
