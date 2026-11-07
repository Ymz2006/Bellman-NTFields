"""ntrl-demo with a grid-synchronised Fourier matrix (one maze, one training seed, one mode of features.py), trained,
planned and diagnosed exactly as jeffrey/maze_scale's ntrl and fourier stages.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/grid_sync/gridsync.py --prepared <maze_scale prepare run> --seed 87358 --mode ring --out <dir>

ntrl-demo draws B (128 x 3) inside Model.train() and builds its network from it. A wrapper around the package's
models.metric.model_network_metric.NN replaces that B in place by features.transform(B, E, mode) on the first
construction (as maze_scale/fourier.py does for B x s), so the package's self.B, the network and the checkpoint all
hold the new matrix; the network init, data, losses, TD step, time units and coordinates are those of plain
ntrl-demo with the same seed. Planning (ntrl-demo MPPI per cell, planning seeds 0/1/2, exact scorer), the field
diagnostics and the held-out speed fit are maze_scale/evaluate.py's. results.json records the mode, s_eff and the
frequency statistics of the matrix (features.describe).
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
ROOT = next((q for q in HERE.parents if (q / 'baselines/ntrl-demo').is_dir()), Path(os.environ.get('NTFIELD_ROOT', '/home/jeffrey/bellman-ntfield')))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
import features as FT  # noqa: E402
import maze as MZ  # noqa: E402
import plan as P  # noqa: E402
import train as TN  # noqa: E402
import evaluate as EV  # noqa: E402
from common import Logger, dump  # noqa: E402


def patched_network_class(make):
    """Patch the package's NN so the first network built gets B replaced in place by make(B); returns (module, original, state)."""
    if str(TN.NTRL) not in sys.path:
        sys.path.insert(0, str(TN.NTRL))
    from models.metric import model_network_metric as mn
    assert Path(mn.__file__).resolve() == TN.NTRL / 'models/metric/model_network_metric.py'
    original = mn.NN
    state = dict(done=False)

    def first_network(device, dim, B):
        # the package's NN.__init__ calls super(NN, self): restore the real class before building (later constructions unpatched)
        state['drawn'] = B.clone()
        B.copy_(make(B))
        state['done'] = True
        mn.NN = original
        return original(device, dim, B)

    mn.NN = first_network
    return mn, original, state


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--mode', choices=FT.MODES, required=True)
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
    res = dict(stage='gridsync', mode=a.mode, maze=maze.describe(), seed=a.seed, epochs=epochs, prepared=str(a.prepared),
               gpu=torch.cuda.get_device_name(0), smoke=a.smoke)
    expected = FT.transform(FT.draw(a.seed), maze.E, a.mode)
    mn, original, state = patched_network_class(lambda B: FT.transform(B, maze.E, a.mode))
    try:
        ck = TN.train(a.prepared / 'data', a.out / 'training', 'metric', epochs, a.seed, source=goals[0].tolist())
    finally:
        mn.NN = original
    assert state['done'], 'B was never replaced'
    assert torch.equal(state['drawn'], FT.draw(a.seed)), 'ntrl-demo drew a different B than features.draw'
    B = torch.load(ck, map_location='cpu', weights_only=False)['B_state_dict']
    assert torch.equal(B, expected), 'checkpoint B differs from features.transform'
    grid = maze.E / 2
    res['s_eff'] = float(grid / np.median(np.abs(state['drawn'][:, :2].double().numpy()).max(1)))
    res['features'] = FT.describe(B, maze.E, maze.n)
    res['features_baseline'] = FT.describe(state['drawn'], maze.E, maze.n)
    res['train_seconds'] = time.time() - t0
    res['final_loss'] = json.load(open(a.out / 'training/progress.json'))['final_loss']
    log(f'trained n={maze.n} seed {a.seed} mode {a.mode} (s_eff {res["s_eff"]:.2f}, |f_xy| / f_grid median '
        f'{res["features"]["mag_over_grid_quantiles"][2]:.3f}, locked rows {res["features"]["locked_rows"]}): '
        f'{res["train_seconds"] / 60:.1f} min, final loss {res["final_loss"]:.4e}')
    net = P.load_field(ck, 'metric', dev)
    cost, grad = P.field_cost(net), lambda pairs: P.field_grad(net, pairs)
    res.update(EV.evaluate(maze, a.prepared, cost, grad, a.out, log, label=f'gridsync {a.mode} n={maze.n} s{a.seed}'))
    res['speed_fit'] = EV.speed_fit(maze, a.prepared, grad)
    np.save(a.out / 'slice_goal0.npy', EV.slice_values(maze, cost, goals[0]))
    log(f'speed fit: {res["speed_fit"]}')
    res['elapsed_seconds'] = time.time() - t0
    res['complete'] = True
    dump(a.out / 'results.json', res)
    log(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
