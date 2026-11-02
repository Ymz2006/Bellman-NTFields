"""Joint-range Test 4 pipeline (jeffrey/joint_range/README.md, Test 4).

  pipeline4.py data --scene S --output O [--pairs N]   collide_data.py on every visible GPU: test4/ (1/3 collision pairs
                                                       + mixed pairs), control/ (mixed pairs only), heldout/
  pipeline4.py cupdata --scene S --output O [--pairs N] cupboard_data.py on every visible GPU: both/ (both grasp points in
                                                       the cupboard), combo/ (half mixed, half both), heldout/
  pipeline4.py train --data D --set test4|control|both|combo
                                                       experiments/ntrl_cabinet/train.py (unchanged, models/metric, 5,000
                     --scene S --tests T --seed N      epochs: the baseline recipe) on D/<set>, then evaluate.py (the
                     --output O                        ntrl-demo cabinet protocol, planners in release coordinates) on the
                                                       scene's frozen suites
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, '/clean/method')
from common import sha, write
HERE = Path(__file__).resolve().parent
NTRL = Path('/clean/experiments/ntrl_cabinet')


def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['data', 'cupdata', 'train']); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scene', type=Path, required=True); p.add_argument('--tests', type=Path); p.add_argument('--data', type=Path); p.add_argument('--set', choices=['test4', 'control', 'both', 'combo'])
    p.add_argument('--seed', type=int, default=87358); p.add_argument('--epochs', type=int, default=5000); p.add_argument('--pairs', type=int, default=200000)
    p.add_argument('--log-eikonal', action='store_true'); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    o = a.output; o.mkdir(parents=True, exist_ok=True); smoke = ['--smoke'] if a.smoke else []; started = time.monotonic(); timing = {}
    settings = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    if a.mode in ('data', 'cupdata'):
        steps = [('data', HERE/('collide_data.py' if a.mode == 'data' else 'cupboard_data.py'), ['--scene', a.scene, '--output', o/'data', '--pairs', a.pairs]+smoke)]
    else:
        data = a.data/a.set; meta = json.loads((data/'complete.json').read_text()); assert meta['complete'] and meta['coords'] == 'full'
        assert a.smoke or not meta['smoke'], 'a full run needs full data'
        assert Path(meta['scene']).name == a.scene.name and meta['scene_sha256'] == sha(a.scene), 'data from another scene'
        steps = [('training', NTRL/'train.py', ['--data', data, '--output', o/'training', '--variant', 'metric', '--epochs', a.epochs, '--seed', a.seed]+(['--log-eikonal'] if a.log_eikonal else [])+smoke),
                 ('evaluate', HERE/'evaluate.py', ['--data-map', data/'complete.json', '--checkpoint', o/'training/checkpoint.pt', '--variant', 'metric',
                                                   '--output', o/'evaluation', '--tests', a.tests, '--scene', a.scene]+smoke)]
    for stage, script, extra in steps:
        write(o/'status.json', dict(complete=False, stage=stage, settings=settings, timing=timing))
        begin = time.monotonic()
        with (o/(stage+'.log')).open('w') as log:
            subprocess.run([sys.executable, '-u', str(script), *map(str, extra)], stdout=log, stderr=subprocess.STDOUT, check=True)
        timing[stage] = time.monotonic()-begin
    status = dict(complete=True, stage='complete', mode=a.mode, settings=settings, scene_sha256=sha(a.scene), timing=timing, elapsed_seconds=time.monotonic()-started)
    if a.mode in ('data', 'cupdata'):
        m = json.loads((o/'data/complete.json').read_text()); status.update(data={k: v for k, v in m.items() if k not in ('workers',)})
    else:
        status.update(totals=json.loads((o/'evaluation/evaluation.json').read_text())['totals'], scaling_check=json.loads((o/'evaluation/scaling_check.json').read_text()),
                      training_set={k: v for k, v in json.loads((a.data/a.set/'complete.json').read_text()).items() if k != 'files'})
    write(o/'status.json', status)


if __name__ == '__main__': main()
