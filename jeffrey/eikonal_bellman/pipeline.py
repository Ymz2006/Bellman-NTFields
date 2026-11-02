"""Stages of one eikonal-foundation Bellman-extension run, inside the launch.py mounts (started by segment_dp/launch.py,
registered by run.py here).

  training   chain.py (the foundation is --base, a registered eikonal field)
  evaluate   jeffrey/segment_dp/evaluate.py: the unchanged frozen evaluators on the final field (no-wall: the frozen
             600 queries; cabinet: the frozen cabinet suites, tilt gate off), native RK4 + release MPPI seeds 0/1/2;
             skipped with --no-evaluate (diagnostic chains that stop before the longest segment)
"""
import sys
sys.path.insert(0, '/clean/method')
import argparse, json, shlex, subprocess, time
from pathlib import Path
from common import sha, write
HERE = Path('/clean/jeffrey/eikonal_bellman'); SEGMENT_DP = Path('/clean/jeffrey/segment_dp')


def run(stage, args, out, status):
    status['stage'] = stage; write(out/'status.json', status); begin = time.monotonic()
    with (out/(stage+'.log')).open('w') as log:
        subprocess.run([sys.executable, '-u', *map(str, args)], stdout=log, stderr=subprocess.STDOUT, check=True)
    status['timing'][stage] = time.monotonic()-begin


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--scene', type=Path, help='cabinet scene file; omitted: the no-wall frozen 600 queries'); p.add_argument('--tests', type=Path)
    p.add_argument('--smoke', action='store_true'); p.add_argument('--no-evaluate', action='store_true')
    p.add_argument('--train-args', default='', help='arguments of chain.py, one shell-quoted string'); a = p.parse_args()
    o = a.output; o.mkdir(parents=True, exist_ok=True); started = time.monotonic(); smoke = ['--smoke'] if a.smoke else []
    status = dict(complete=False, settings={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, timing={})
    status['settings']['trainer'] = 'chain'
    run('training', [HERE/'chain.py', '--prepared', a.prepared, '--output', o/'training', *shlex.split(a.train_args), *smoke], o, status)
    field = o/'training/field.pt'
    if not a.no_evaluate:
        if a.scene:
            run('evaluate', [SEGMENT_DP/'evaluate.py', 'cabinet', '--method', 'bellman', '--checkpoint', field, '--scene', a.scene, '--tests', a.tests, '--output', o/'evaluation', *smoke], o, status)
        else:
            run('evaluate', [SEGMENT_DP/'evaluate.py', 'nowall', '--checkpoint', field, '--output', o/'evaluation', *smoke], o, status)
        ev = json.loads((o/'evaluation/evaluation.json').read_text())
        totals = ev.get('totals') or dict(native_rk4=dict(success=ev['native_success'], n=sum(r['n'] for r in ev['scores'] if r['method'] == 'native_rk4'), per_seed=None),
                                         matched_mppi=dict(success=sum(ev['mppi_successes']), n=sum(r['n'] for r in ev['scores'] if r['method'] == 'matched_mppi'),
                                                           per_seed=ev['mppi_successes']))
        status['totals'] = totals
    status.update(complete=True, stage='complete', field_sha256=sha(field), elapsed_seconds=time.monotonic()-started,
                  scene_sha256=sha(a.scene) if a.scene else None, tests_manifest_sha256=sha(a.tests/'suite_manifest.json') if a.tests else None)
    write(o/'status.json', status)


if __name__ == '__main__': main()
