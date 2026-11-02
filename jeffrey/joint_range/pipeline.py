"""Joint-range test pipeline (jeffrey/joint_range/README.md).

  pipeline.py data --scene S --output O            data.py: one pair set, written as full/ and range/
  pipeline.py cabdata --scene S --output O         cabinet_data.py: in-cabinet pairs on every visible GPU
  pipeline.py mix --cabinet A --data B --fraction F  mix.py, then train.py and evaluate.py on the mixture
  pipeline.py derive --data D --output O          derive.py: Test 2b's anchor/ set (range of the collision-checked x0)
  pipeline.py test --data D --scaling full|range|anchor  experiments/ntrl_cabinet/train.py (unchanged, models/metric, 5,000
                   --seed N --scene S --tests T    epochs: the registered baseline recipe) on D/<scaling>, then
                   --output O                      evaluate.py (the ntrl-demo cabinet protocol, planners in release
                                                   coordinates)
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, '/clean/method')
from common import sha, write
HERE = Path(__file__).resolve().parent
NTRL = Path('/clean/experiments/ntrl_cabinet')


def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['data', 'derive', 'test', 'cabdata', 'mix']); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scene', type=Path, required=True); p.add_argument('--tests', type=Path); p.add_argument('--data', type=Path); p.add_argument('--cabinet', type=Path); p.add_argument('--fraction', type=float)
    p.add_argument('--scaling', choices=['full', 'range', 'anchor']); p.add_argument('--seed', type=int, default=87358); p.add_argument('--epochs', type=int, default=5000)
    p.add_argument('--pairs', type=int, default=400000); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    o = a.output; o.mkdir(parents=True, exist_ok=True); smoke = ['--smoke'] if a.smoke else []; started = time.monotonic(); timing = {}
    settings = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    if a.mode == 'data':
        steps = [('data', HERE/'data.py', ['--scene', a.scene, '--output', o/'data', '--pairs', a.pairs]+smoke)]
    elif a.mode == 'cabdata':
        steps = [('cabdata', HERE/'cabinet_data.py', ['--scene', a.scene, '--output', o/'data', '--pairs', a.pairs]+smoke)]
    elif a.mode == 'mix':
        steps = [('mix', HERE/'mix.py', ['--cabinet', a.cabinet, '--area', a.data, '--scene', a.scene, '--fraction', a.fraction, '--pairs', a.pairs, '--output', o/'data']+smoke),
                 ('training', NTRL/'train.py', ['--data', o/'data/mix', '--output', o/'training', '--variant', 'metric', '--epochs', a.epochs, '--seed', a.seed]+smoke),
                 ('evaluate', HERE/'evaluate.py', ['--data-map', o/'data/mix/complete.json', '--checkpoint', o/'training/checkpoint.pt', '--variant', 'metric',
                                                   '--output', o/'evaluation', '--tests', a.tests, '--scene', a.scene]+smoke)]
    elif a.mode == 'derive':
        steps = [('derive', HERE/'derive.py', ['--data', a.data, '--output', o/'data'])]
    else:
        data = a.data/a.scaling; meta = json.loads((data/'complete.json').read_text()); assert meta['complete'] and meta['coords'] == a.scaling
        assert a.smoke or not meta['smoke'], 'full test needs full data'
        steps = [('training', NTRL/'train.py', ['--data', data, '--output', o/'training', '--variant', 'metric', '--epochs', a.epochs, '--seed', a.seed]+smoke),
                 ('evaluate', HERE/'evaluate.py', ['--data-map', data/'complete.json', '--checkpoint', o/'training/checkpoint.pt', '--variant', 'metric',
                                                   '--output', o/'evaluation', '--tests', a.tests, '--scene', a.scene]+smoke)]
    for stage, script, extra in steps:
        write(o/'status.json', dict(complete=False, stage=stage, settings=settings, timing=timing))
        begin = time.monotonic()
        with (o/(stage+'.log')).open('w') as log:
            subprocess.run([sys.executable, '-u', str(script), *map(str, extra)], stdout=log, stderr=subprocess.STDOUT, check=True)
        timing[stage] = time.monotonic()-begin
    status = dict(complete=True, stage='complete', mode=a.mode, settings=settings, scene_sha256=sha(a.scene), timing=timing, elapsed_seconds=time.monotonic()-started)
    if a.mode == 'mix':
        status.update(totals=json.loads((o/'evaluation/evaluation.json').read_text())['totals'], scaling_check=json.loads((o/'evaluation/scaling_check.json').read_text()),
                      mixture={k: v for k, v in json.loads((o/'data/mix/complete.json').read_text()).items() if k not in ('files', 'cabinet_files', 'area_files')})
    elif a.mode == 'cabdata':
        status.update(cabinet={k: v for k, v in json.loads((o/'data/cabinet/complete.json').read_text()).items() if k not in ('files', 'workers')})
    elif a.mode == 'test':
        status.update(totals=json.loads((o/'evaluation/evaluation.json').read_text())['totals'], scaling_check=json.loads((o/'evaluation/scaling_check.json').read_text()))
    elif a.mode == 'derive':
        status.update(range=json.loads((o/'data/anchor_range.json').read_text()))
    else:
        status.update(range=json.loads((o/'data/range.json').read_text()))
    write(o/'status.json', status)


if __name__ == '__main__': main()
