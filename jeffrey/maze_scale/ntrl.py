"""Stage 2 of the maze-scale test (one maze, one training seed): ntrl-demo on the prepared data, planned and diagnosed.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/maze_scale/ntrl.py --prepared <stage-1 dir> --seed 87358 --out <dir>

1. train     ntrl-demo models/metric through its own Model.train() (train.py, unchanged package), dim 3, 5,000 epochs.
2. plan      ntrl-demo's MPPI scaled to the maze's pitch (plan.py), planning seeds 0/1/2, exact scorer, success by bin.
3. field     T / T*, Spearman, descent direction at the start by bin (evaluate.py); speed fit on held-out pairs;
             the field's slice T((x, y, phi_goal), goal 0) for figures.
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
from common import Logger, dump  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=5000)
    p.add_argument('--variant', choices=['metric', 'metric_arm'], default='metric')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    log = Logger(a.out / 'log.txt')
    dev = 'cuda:0'
    t0 = time.time()
    info = json.load(open(a.prepared / 'maze.json'))
    maze = MZ.Maze(info['n'], dev)
    assert maze.describe()['walls'] == info['walls'] and np.array_equal(maze.V.astype(int), np.array(info['V'])), 'prepared maze differs'
    epochs = 3 if a.smoke else a.epochs
    goals = torch.tensor(np.load(a.prepared / 'goals.npy'), device=dev)
    res = dict(stage='ntrl', maze=maze.describe(), seed=a.seed, epochs=epochs, variant=a.variant, prepared=str(a.prepared), gpu=torch.cuda.get_device_name(0), smoke=a.smoke)
    ck = TN.train(a.prepared / 'data', a.out / 'training', a.variant, epochs, a.seed, source=goals[0].tolist())
    res['train_seconds'] = time.time() - t0
    res['final_loss'] = json.load(open(a.out / 'training/progress.json'))['final_loss']
    log(f'trained n={maze.n} seed {a.seed}: {res["train_seconds"] / 60:.1f} min, final loss {res["final_loss"]:.4e}')
    net = P.load_field(ck, a.variant, dev)
    cost, grad = P.field_cost(net), lambda pairs: P.field_grad(net, pairs)
    res.update(EV.evaluate(maze, a.prepared, cost, grad, a.out, log, label=f'ntrl n={maze.n} s{a.seed}'))
    res['speed_fit'] = EV.speed_fit(maze, a.prepared, grad)
    np.save(a.out / 'slice_goal0.npy', EV.slice_values(maze, cost, goals[0]))
    log(f'speed fit: {res["speed_fit"]}')
    res['elapsed_seconds'] = time.time() - t0
    res['complete'] = True
    dump(a.out / 'results.json', res)
    log(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
