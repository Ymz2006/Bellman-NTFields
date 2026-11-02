"""Stages of the jeffrey/ experiments, run inside the launch.py mounts by jeffrey/launch.py (registered by jeffrey/run.py).

  guard  [--train-distance --sampled DIR | --distance distance.pt] --field K --checkpoint C --variants JSON
         trains the hierarchical distance model (distance.py) when asked, then runs evaluate.py once per variant
         ({"name": ..., "args": "...", "script": "evaluate" (default) | "guard_only"}) into /outputs/<name>
  sharp  --checkpoint P-NTFields.pt --bank DIR --sharp-args "..." --eval-generations 0 5 10
         Bellman-sharp training (bellman_sharp.py) into /outputs/sharp, the tunnelling diagnostic (sharp_diagnose.py,
         /outputs/sharp_diagnostic.json), then the unchanged P-NTFields cabinet
         evaluation (experiments/pntfields_cabinet/evaluate.py) of each listed generation (0 = the pretrained field)
"""
import sys
sys.path.insert(0, '/clean/method')
import argparse, json, shlex, subprocess, time
from pathlib import Path
from common import sha, write
HERE = Path('/clean/jeffrey')


def run(stage, script, args, out, status):
    status['stage'] = stage; write(out/'status.json', status); begin = time.monotonic()
    with (out/(stage+'.log')).open('w') as log:
        subprocess.run([sys.executable, '-u', str(script), *map(str, args)], stdout=log, stderr=subprocess.STDOUT, check=True)
    status['timing'][stage] = time.monotonic()-begin


def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['guard', 'sharp']); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scene', type=Path, required=True); p.add_argument('--tests', type=Path, required=True); p.add_argument('--checkpoint', type=Path)
    p.add_argument('--field', default='bellman'); p.add_argument('--distance', type=Path); p.add_argument('--train-distance', action='store_true'); p.add_argument('--sampled', type=Path)
    p.add_argument('--distance-args', default=''); p.add_argument('--variants', default='[]'); p.add_argument('--bank', type=Path); p.add_argument('--sharp-args', default='')
    p.add_argument('--eval-generations', type=int, nargs='*', default=[]); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    o = a.output; o.mkdir(parents=True, exist_ok=True); started = time.monotonic(); smoke = ['--smoke'] if a.smoke else []
    status = dict(complete=False, mode=a.mode, settings={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, timing={})
    common = ['--scene', a.scene, '--tests', a.tests]
    if a.mode == 'guard':
        distance = a.distance
        if a.train_distance:
            run('distance', HERE/'distance.py', ['--scene', a.scene, '--sampled', a.sampled, '--output', o/'distance', *shlex.split(a.distance_args), *smoke], o, status)
            distance = o/'distance/distance.pt'
        results = {}
        for v in json.loads(a.variants):
            extra = shlex.split(v['args']); dist = [] if '--exact' in extra else ['--distance', distance]
            ck = ['--checkpoint', a.checkpoint] if a.checkpoint and '--field' not in extra else []
            field = [] if '--field' in extra else ['--field', a.field]
            run(v['name'], HERE/(v.get('script', 'evaluate')+'.py'), ['--output', o/v['name'], *common, *field, *ck, *dist, '--label', v['name'], *extra, *smoke], o, status)
            results[v['name']] = json.loads((o/v['name']/'evaluation.json').read_text())['totals']
        status['results'] = results
    else:
        run('sharp', HERE/'bellman_sharp.py', ['--checkpoint', a.checkpoint, '--bank', a.bank, '--scene', a.scene, '--output', o/'sharp', *shlex.split(a.sharp_args), *smoke], o, status)
        run('diagnose', HERE/'sharp_diagnose.py', ['--run', o/'sharp', '--base', a.checkpoint, '--bank', a.bank, '--scene', a.scene, '--output', o/'sharp_diagnostic.json'], o, status)
        results = {}
        for g in a.eval_generations:
            ck = a.checkpoint if g == 0 else o/'sharp'/f'gen_{g:02d}.pt'
            run(f'evaluate_gen{g:02d}', '/clean/experiments/pntfields_cabinet/evaluate.py', ['--checkpoint', ck, '--coords', 'ntrl', '--output', o/f'evaluation_gen{g:02d}', *common, *smoke], o, status)
            results[g] = json.loads((o/f'evaluation_gen{g:02d}'/'evaluation.json').read_text())['totals']
        status['results'] = results
    status.update(complete=True, stage='complete', scene_sha256=sha(a.scene), tests_manifest_sha256=sha(a.tests/'suite_manifest.json'), elapsed_seconds=time.monotonic()-started)
    write(o/'status.json', status)


if __name__ == '__main__': main()
