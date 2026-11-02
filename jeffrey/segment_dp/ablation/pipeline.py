"""Stages of one loss-ablation run, inside the launch.py mounts (started by ../launch.py, registered by run.py here).

  training   train.py --prepared P --weights B,E,D --seed S     5 rounds of the original method
  evaluate   no-wall: method/evaluate.py, the frozen 600 queries (native RK4, release MPPI seeds 0/1/2)
             cabinet: experiments/cabinet/evaluate.py, the frozen cabinet suites (same planners, tilt gate off)
             on the fifth-round checkpoint, unchanged evaluators
"""
import sys
sys.path.insert(0, '/clean/method')
import argparse, json, subprocess, time
from pathlib import Path
from common import sha, write
HERE = Path('/clean/jeffrey/segment_dp/ablation')


def run(stage, args, out, status):
    status['stage'] = stage; write(out/'status.json', status); begin = time.monotonic()
    with (out/(stage+'.log')).open('w') as log:
        subprocess.run([sys.executable, '-u', *map(str, args)], stdout=log, stderr=subprocess.STDOUT, check=True)
    status['timing'][stage] = time.monotonic()-begin


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--scene', type=Path, help='cabinet scene file; omitted: the no-wall frozen 600 queries'); p.add_argument('--tests', type=Path)
    p.add_argument('--weights', default='6,2,2'); p.add_argument('--seed', type=int, default=87358); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    o = a.output; o.mkdir(parents=True, exist_ok=True); started = time.monotonic(); smoke = ['--smoke'] if a.smoke else []
    status = dict(complete=False, settings={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, timing={})
    run('training', [HERE/'train.py', '--prepared', a.prepared, '--output', o/'training', '--weights', a.weights, '--seed', a.seed, *smoke], o, status)
    checkpoint = o/'training/checkpoint.pt'
    if a.scene:
        run('evaluate', ['/clean/experiments/cabinet/evaluate.py', '--method', 'bellman', '--checkpoint', checkpoint, '--scene', a.scene, '--tests', a.tests,
                         '--output', o/'evaluation', *smoke], o, status)
    else:
        run('evaluate', ['/clean/method/evaluate.py', '--checkpoint', checkpoint, '--output', o/'evaluation', *smoke], o, status)
    ev = json.loads((o/'evaluation/evaluation.json').read_text())
    totals = ev.get('totals') or dict(native_rk4=dict(success=ev['native_success'], n=sum(r['n'] for r in ev['scores'] if r['method'] == 'native_rk4'), per_seed=None),
                                     matched_mppi=dict(success=sum(ev['mppi_successes']), n=sum(r['n'] for r in ev['scores'] if r['method'] == 'matched_mppi'),
                                                       per_seed=ev['mppi_successes']))
    status.update(complete=True, stage='complete', totals=totals, checkpoint_sha256=sha(checkpoint), elapsed_seconds=time.monotonic()-started,
                  scene_sha256=sha(a.scene) if a.scene else None, tests_manifest_sha256=sha(a.tests/'suite_manifest.json') if a.tests else None)
    write(o/'status.json', status)


if __name__ == '__main__': main()
