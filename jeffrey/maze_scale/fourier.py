"""ntrl-demo with a scaled Fourier matrix (one maze, one training seed): B' = s B, trained, planned and diagnosed as stage 2.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/maze_scale/fourier.py --prepared <stage-1 dir> --seed 87358 --b-scale 3 --out <dir>

ntrl-demo's network sees x @ 2 pi B with B ~ N(0, 1) truncated at 2 (128 x 3, drawn in Model.train()). Here B is
multiplied by --b-scale s once, right after the package draws it and before its network is built (a wrapper around
models.metric.model_network_metric.NN that scales the B tensor in place on the first construction, so the package's
self.B, the network and the checkpoint all hold s B; removed again after training). Same seed -> the same draw, so
B' is exactly s times the B of the plain ntrl-demo run with that seed; the network init, data, losses, TD step, time
units and coordinates are unchanged. Unlike the zoom control (zoom.py), only the input frequencies change.
Planning, diagnostics and outputs are those of stage 2 (evaluate.py); results.json records b_scale and max |B|.
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


def scaled_network_class(scale):
    """Patch the package's NN so the first network built scales its B argument in place; returns (module, original)."""
    if str(TN.NTRL) not in sys.path:
        sys.path.insert(0, str(TN.NTRL))
    from models.metric import model_network_metric as mn
    assert Path(mn.__file__).resolve() == TN.NTRL / 'models/metric/model_network_metric.py'
    original = mn.NN
    state = dict(done=False)

    def first_network(device, dim, B):
        # the package's NN.__init__ calls super(NN, self), so the module global must be the real class while it runs:
        # scale B, restore the class (later constructions are unpatched), then build
        B.mul_(scale)
        state['done'] = True
        mn.NN = original
        return original(device, dim, B)

    mn.NN = first_network
    return mn, original, state


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--b-scale', type=float, required=True)
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
    assert maze.describe()['walls'] == info['walls'] and np.array_equal(maze.V.astype(int), np.array(info['V'])), 'prepared maze differs'
    epochs = 3 if a.smoke else a.epochs
    goals = torch.tensor(np.load(a.prepared / 'goals.npy'), device=dev)
    res = dict(stage='fourier', b_scale=a.b_scale, maze=maze.describe(), seed=a.seed, epochs=epochs, prepared=str(a.prepared),
               gpu=torch.cuda.get_device_name(0), smoke=a.smoke)
    mn, original, state = scaled_network_class(a.b_scale)
    try:
        ck = TN.train(a.prepared / 'data', a.out / 'training', 'metric', epochs, a.seed, source=goals[0].tolist())
    finally:
        mn.NN = original
    assert state['done'], 'B was never scaled'
    B = torch.load(ck, map_location='cpu', weights_only=False)['B_state_dict']
    res['b_abs_max'] = float(B.abs().max())
    res['b_std'] = float(B.std())
    res['train_seconds'] = time.time() - t0
    res['final_loss'] = json.load(open(a.out / 'training/progress.json'))['final_loss']
    log(f'trained n={maze.n} seed {a.seed} B x {a.b_scale:g} (max |B| {res["b_abs_max"]:.3f}, std {res["b_std"]:.3f}): '
        f'{res["train_seconds"] / 60:.1f} min, final loss {res["final_loss"]:.4e}')
    net = P.load_field(ck, 'metric', dev)
    cost, grad = P.field_cost(net), lambda pairs: P.field_grad(net, pairs)
    res.update(EV.evaluate(maze, a.prepared, cost, grad, a.out, log, label=f'fourier x{a.b_scale:g} n={maze.n} s{a.seed}'))
    res['speed_fit'] = EV.speed_fit(maze, a.prepared, grad)
    np.save(a.out / 'slice_goal0.npy', EV.slice_values(maze, cost, goals[0]))
    log(f'speed fit: {res["speed_fit"]}')
    res['elapsed_seconds'] = time.time() - t0
    res['complete'] = True
    dump(a.out / 'results.json', res)
    log(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
