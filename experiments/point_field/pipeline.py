"""Registered hierarchical point-field experiment on the cabinet scene (experiments/point_field/README.md).

field_mixed  train.py --sources pool+uniform: a PointField on sphere centers of the FK pool's training rows and of
             uniform joint draws (plus jittered and uniform workspace points), exact scene distances.
field_pool   train.py --sources pool: the same with the FK pool's training rows only, the configurations the speed
             networks were fitted on (control: the gain is not from more configurations).
compare      compare.py: both fields through FK (hierarchical), the exact hierarchy, the learned-self variants and the
             given speed networks, on the sample sets of face-speed-accuracy; distance and speed metrics.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, '/clean/method')
from common import sha, write
HERE = '/clean/experiments/point_field'
p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); p.add_argument('--scene', type=Path, required=True)
p.add_argument('--tests', type=Path, required=True); p.add_argument('--sampled', type=Path, required=True); p.add_argument('--runs', type=Path, nargs='*', default=[])
p.add_argument('--self-net', type=Path); p.add_argument('--steps', type=int, default=20000); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
started = time.monotonic(); timing = {}; smoke = ['--smoke'] if a.smoke else []; o = a.output
train = lambda sources: ['--scene', a.scene, '--sampled', a.sampled, '--output', o/'fields'/sources.replace('+', '_'), '--sources', sources, '--steps', str(a.steps)]+smoke
steps = [('field_mixed', f'{HERE}/train.py', train('pool+uniform')), ('field_pool', f'{HERE}/train.py', train('pool')),
         ('compare', f'{HERE}/compare.py', ['--scene', a.scene, '--tests', a.tests, '--sampled', a.sampled, '--output', o/'comparison',
                                            '--fields', f"mixed={o/'fields/pool_uniform/field.pt'}", f"pool={o/'fields/pool/field.pt'}"]
                                           +(['--self-net', a.self_net] if a.self_net else [])+(['--runs', *a.runs] if a.runs else [])+smoke)]
base = dict(smoke=a.smoke, scene=str(a.scene), sampled=str(a.sampled), runs=[str(r) for r in a.runs], self_net=str(a.self_net) if a.self_net else None, steps=a.steps)
for stage, script, extra in steps:
    write(o/'status.json', dict(complete=False, stage=stage, timing=timing, **base))
    begin = time.monotonic()
    with (o/(stage+'.log')).open('w') as log:
        subprocess.run([sys.executable, '-u', script, *map(str, extra)], stdout=log, stderr=subprocess.STDOUT, check=True)
    timing[stage] = time.monotonic()-begin
    print('STAGE', stage, timing[stage], flush=True)
write(o/'status.json', dict(complete=True, stage='complete', scene_sha256=sha(a.scene), tests_manifest_sha256=sha(a.tests/'suite_manifest.json'), timing=timing,
                             elapsed_seconds=time.monotonic()-started, **base))
