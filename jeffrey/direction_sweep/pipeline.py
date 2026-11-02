"""Stages of one direction-sweep run, inside the launch.py mounts (started by jeffrey/segment_dp/launch.py, registered by run.py here).

  --stage extend   extend.py: the nested extension bank (frozen sources, frozen 4 pairs + new pairs)
  --stage train    train.py --pairs D       5 rounds of the original method, Bellman min over 8*D actions
                   evaluate                 no-wall: method/evaluate.py (frozen 600 queries, RK4 + release MPPI seeds 0/1/2)
                                            cabinet: experiments/cabinet/evaluate.py (frozen cabinet suites)
                   fit                      jeffrey/segment_dp/fit.py against the ground-truth travel times
                   diag                     diag.py (implied speed, Bellman self-consistency, direction coverage)
"""
import sys
sys.path.insert(0, '/clean/method')
import argparse, json, subprocess, time
from pathlib import Path
from common import sha, write
HERE = Path('/clean/jeffrey/direction_sweep')


def run(stage, args, out, status):
    status['stage'] = stage; write(out/'status.json', status); begin = time.monotonic()
    with (out/(stage+'.log')).open('w') as log:
        subprocess.run([sys.executable, '-u', *map(str, args)], stdout=log, stderr=subprocess.STDOUT, check=True)
    status['timing'][stage] = time.monotonic()-begin


def main():
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=['extend', 'train'], required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True); p.add_argument('--scene', type=Path); p.add_argument('--tests', type=Path)
    p.add_argument('--ext', type=Path); p.add_argument('--truth', type=Path); p.add_argument('--pairs', type=int); p.add_argument('--seed', type=int, default=87358)
    p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    o = a.output; o.mkdir(parents=True, exist_ok=True); started = time.monotonic(); smoke = ['--smoke'] if a.smoke else []
    status = dict(complete=False, settings={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, timing={})
    if a.stage == 'extend':
        run('extend', [HERE/'extend.py', '--prepared', a.prepared, *(['--scene', a.scene] if a.scene else []), '--pairs', a.pairs, '--output', o, *smoke], o, status)
        ext = json.loads((o/'ext/manifest.json').read_text())
        status.update(complete=True, stage='complete', audit=ext['audit'], by_pairs=ext['by_pairs'], elapsed_seconds=time.monotonic()-started); write(o/'status.json', status); return
    run('training', [HERE/'train.py', '--prepared', a.prepared, '--ext', a.ext, '--pairs', a.pairs, '--seed', a.seed, '--output', o/'training', *smoke], o, status)
    checkpoint = o/'training/checkpoint.pt'
    if a.scene:
        run('evaluate', ['/clean/experiments/cabinet/evaluate.py', '--method', 'bellman', '--checkpoint', checkpoint, '--scene', a.scene, '--tests', a.tests,
                         '--output', o/'evaluation', *smoke], o, status)
    else:
        run('evaluate', ['/clean/method/evaluate.py', '--checkpoint', checkpoint, '--output', o/'evaluation', *smoke], o, status)
    run('fit', ['/clean/jeffrey/segment_dp/fit.py', '--truth', a.truth, '--prepared', a.prepared, '--fields', checkpoint, '--labels', f'D{a.pairs}', '--output', o/'fit.json'], o, status)
    run('diag', [HERE/'diag.py', '--checkpoint', checkpoint, '--prepared', a.prepared, '--ext', a.ext, '--truth', a.truth, *(['--samples', '2048'] if a.smoke else []),
                 '--output', o/'diag.json'], o, status)
    ev = json.loads((o/'evaluation/evaluation.json').read_text())
    totals = ev.get('totals') or dict(native_rk4=dict(success=ev['native_success'], n=sum(r['n'] for r in ev['scores'] if r['method'] == 'native_rk4'), per_seed=None),
                                     matched_mppi=dict(success=sum(ev['mppi_successes']), n=sum(r['n'] for r in ev['scores'] if r['method'] == 'matched_mppi'),
                                                       per_seed=ev['mppi_successes']))
    progress = json.loads((o/'training/progress.json').read_text())
    status.update(complete=True, stage='complete', totals=totals, checkpoint_sha256=sha(checkpoint), rows_without_safe_action=progress['rows_without_safe_action'],
                  elapsed_seconds=time.monotonic()-started)
    write(o/'status.json', status)


if __name__ == '__main__': main()
