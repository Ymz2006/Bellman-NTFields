"""Control (one maze, one training seed): ntrl-demo on zoomed coordinates, planned and diagnosed as stage 2.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/maze_scale/zoom.py --prepared <stage-1 dir> --seed 87358 --zoom 5.7 --out <dir>

ntrl-demo's network sees x @ 2 pi B with B ~ N(0, 1): a fixed set of input frequencies over the unit box. On an n x n
maze a cell is 1 / (n + 0.1) of the box, so the frequencies per cell fall as the maze grows. Here the training pairs'
coordinates are multiplied by --zoom s (the box becomes [-s/2, s/2]^2 x [-s phi_max, s phi_max]); speeds and unit
normals are unchanged. ntrl-demo's Model.train() (unchanged package) then fits T_net with |grad T_net| = 1 / S in zoomed
units, i.e. T_net(s x) = s T(x), and its TD step 0.03 covers 0.03 / s of the original box. The trained network is
used as the equivalent field on the original coordinates: an NN with B' = s B and the same weights computes
T_net(s x) exactly (the first operation is x @ 2 pi B), and its output is divided by s (fields.Scaled). Planning,
diagnostics and outputs are those of stage 2 (evaluate.py); results.json records the zoom.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import maze as MZ  # noqa: E402
import plan as P  # noqa: E402
import train as TN  # noqa: E402
import evaluate as EV  # noqa: E402
from zoomfield import load_zoomed  # noqa: E402
from common import Logger, dump  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--zoom', type=float, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=5000)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    log = Logger(a.out / 'log.txt')
    dev = 'cuda:0'
    t0 = time.time()
    info = json.load(open(a.prepared / 'maze.json'))
    maze = MZ.Maze(info['n'], dev)
    assert np.array_equal(maze.V.astype(int), np.array(info['V'])), 'prepared maze differs'
    epochs = 3 if a.smoke else a.epochs
    goals = torch.tensor(np.load(a.prepared / 'goals.npy'), device=dev)
    (a.out / 'data').mkdir(exist_ok=True)
    np.save(a.out / 'data/sampled_points.npy', (np.load(a.prepared / 'data/sampled_points.npy') * a.zoom).astype(np.float32))
    for f in ('speed.npy', 'normal.npy'):
        np.save(a.out / 'data' / f, np.load(a.prepared / 'data' / f))
    res = dict(stage='zoom', zoom=a.zoom, maze=maze.describe(), seed=a.seed, epochs=epochs, prepared=str(a.prepared), gpu=torch.cuda.get_device_name(0), smoke=a.smoke)
    ck = TN.train(a.out / 'data', a.out / 'training', 'metric', epochs, a.seed, source=(goals[0] * a.zoom).tolist())
    (a.out / 'data/sampled_points.npy').unlink()                      # 24 MB copy of the prepared pairs x zoom
    res['train_seconds'] = time.time() - t0
    res['final_loss'] = json.load(open(a.out / 'training/progress.json'))['final_loss']
    log(f'trained n={maze.n} seed {a.seed} zoom {a.zoom:g}: {res["train_seconds"] / 60:.1f} min, final loss {res["final_loss"]:.4e}')
    net = load_zoomed(ck, a.zoom, dev)
    cost, grad = P.field_cost(net), lambda pairs: P.field_grad(net, pairs)
    res.update(EV.evaluate(maze, a.prepared, cost, grad, a.out, log, label=f'zoom{a.zoom:g} n={maze.n} s{a.seed}'))
    res['speed_fit'] = EV.speed_fit(maze, a.prepared, grad)
    np.save(a.out / 'slice_goal0.npy', EV.slice_values(maze, cost, goals[0]))
    log(f'speed fit: {res["speed_fit"]}')
    res['elapsed_seconds'] = time.time() - t0
    res['complete'] = True
    dump(a.out / 'results.json', res)
    log(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
