"""ntrl-demo with sharp features on the grid-synced octave embedding (one maze, one training seed, one features.py spec),
trained, planned and diagnosed as jeffrey/octave_detune/octdetune.py.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/sharp_features/sharp.py --prepared <maze_scale prepare run> --seed 87358 --spec Rpow4@half --out <dir>

Training: ntrl-demo's own loop (maze_scale/train.py, unchanged package). On the first network construction the patch
replaces B in place by the octave matrix (features.octave, = grid_sync / octave_detune 'octave') and builds
sharpnet.SharpNN with the spec's plan instead of the package's NN; the draw (directions), data, losses, optimizer and
schedule are those of plain ntrl-demo with the same seed (and, for replace specs, the initialization and data order
too). The package saves a checkpoint every 100 epochs; the ones at --keep-epochs and the final one are kept.

Final checkpoint (epoch 5,000): maze_scale/evaluate.py (MPPI planning seeds 0/1/2, exact scorer, field vs truth),
held-out speed fit, diagnostics.py (near-wall slope / cos, grid-line profiles, local minima along true paths, median
|log T/T*|, failures split by arrival) and the input mapping's own arc / fold measures (diagnostics.feature_metrics).
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
import sharpnet as SN  # noqa: E402
import maze as MZ  # noqa: E402
import plan as P  # noqa: E402
import train as TN  # noqa: E402
import evaluate as EV  # noqa: E402
from common import Logger, dump  # noqa: E402
import diagnostics as DG  # noqa: E402

KEEP_EPOCHS = '1000,2000,3000,4000'


def patched_network_class(make):
    """Patch the package's NN so that the first network built is make(original NN class, device, dim, B)."""
    if str(TN.NTRL) not in sys.path:
        sys.path.insert(0, str(TN.NTRL))
    from models.metric import model_network_metric as mn
    assert Path(mn.__file__).resolve() == TN.NTRL / 'models/metric/model_network_metric.py'
    original = mn.NN
    state = dict(done=False)

    def first_network(device, dim, B):
        state['drawn'] = B.clone()
        mn.NN = original
        net = make(original, device, dim, B)
        state['done'] = True
        return net

    mn.NN = first_network
    return mn, original, state


def sharp_columns(pl):
    """(D,) bool: the input columns that are not the plain octave sin / cos."""
    cols = np.zeros(256 + pl['extra_width'], bool)
    for blk in pl['replace']:
        cols[:128] |= blk['mask']
        cols[128:256] |= blk['mask']
    cols[256:] = True
    return cols


def measure(maze, prepared, net, out, log, label, plan_seeds):
    out.mkdir(parents=True, exist_ok=True)
    cost, grad = P.field_cost(net), (lambda pairs: P.field_grad(net, pairs))
    t0 = time.time()
    res = EV.evaluate(maze, prepared, cost, grad, out, log, label=label, plan_seeds=plan_seeds)
    res['speed_fit'] = EV.speed_fit(maze, prepared, grad)
    res['wall'] = DG.wall_measure(maze, prepared, grad)
    res['bumps'] = DG.bumps(maze, prepared, cost)
    res['profiles'] = DG.profiles(maze, prepared, grad)
    res['accuracy'] = DG.accuracy(prepared, out / 'rows.npz')
    res['failures'] = DG.failures(out / 'rows.npz')
    res['measure_seconds'] = time.time() - t0
    w = res['wall'].get('2-8', {})
    log(f'{label}: success {res["success_total"]}, |log T/T*| {res["accuracy"]["log_err_median"]:.3f}, speed corr '
        f'{res["speed_fit"]["log_speed_corr"]:.3f}, near-wall slope {w.get("slope", float("nan")):.2f} cos {w.get("cos", float("nan")):.2f}, '
        f'bumps {res["bumps"]["bumps_per_path"]:.2f}, never arrived {res["failures"]["never_arrived"]}, wall {res["failures"]["arrived_after_wall"]} '
        f'({res["measure_seconds"]:.0f}s)')
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--spec', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=5000)
    p.add_argument('--keep-epochs', default=KEEP_EPOCHS)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    FT.parse(a.spec)
    a.out.mkdir(parents=True, exist_ok=True)
    log = Logger(a.out / 'log.txt')
    dev = 'cuda:0'
    t0 = time.time()
    info = json.load(open(a.prepared / 'maze.json'))
    maze = MZ.Maze(info['n'], dev)
    assert maze.describe()['walls'] == info['walls'] and np.array_equal(maze.V.astype(int), np.array(info['V'])), 'prepared maze differs'
    epochs = 3 if a.smoke else a.epochs
    keep = [int(e) for e in a.keep_epochs.split(',') if e] if not a.smoke else []
    keep = [e for e in keep if e < epochs]
    goals = torch.tensor(np.load(a.prepared / 'goals.npy'), device=dev)
    drawn = FT.draw(a.seed)
    Bo = FT.octave(drawn, maze.E)
    pl = FT.plan(a.spec, Bo, maze.E, maze.n, a.seed)
    res = dict(stage='sharp', spec=a.spec, parts=FT.parse(a.spec), plan=pl['summary'], maze=maze.describe(), seed=a.seed, epochs=epochs,
               prepared=str(a.prepared), gpu=torch.cuda.get_device_name(0), smoke=a.smoke, keep_epochs=keep)
    log(f'spec {a.spec}: {json.dumps(pl["summary"])}')

    def make(NN, device, dim, B):
        B.copy_(FT.octave(B, maze.E))
        return SN.build(NN, device, B, FT.plan(a.spec, B, maze.E, maze.n, a.seed))

    mn, original, state = patched_network_class(make)
    try:
        ck = TN.train(a.prepared / 'data', a.out / 'training', 'metric', epochs, a.seed, source=goals[0].tolist(), keep_every=100)
    finally:
        mn.NN = original
    assert state['done'], 'the network was never built'
    assert torch.equal(state['drawn'], drawn), 'ntrl-demo drew a different B than features.draw'
    payload = torch.load(ck, map_location='cpu', weights_only=False)
    assert torch.equal(payload['B_state_dict'], Bo), 'checkpoint B differs from features.octave'
    folders = [q for q in (a.out / 'training/model').iterdir() if q.is_dir()]
    assert len(folders) == 1, folders
    kept = {}
    for c in sorted(folders[0].glob('Model_Epoch_*.pt')):
        e = int(c.name.split('_')[2])
        if e in keep:
            kept[e] = c
        elif e != epochs:
            c.unlink()
    res['kept_checkpoints'] = {str(e): str(c.relative_to(a.out)) for e, c in sorted(kept.items())}
    res['train_seconds'] = time.time() - t0
    res['final_loss'] = json.load(open(a.out / 'training/progress.json'))['final_loss']
    net = SN.load_field(ck, a.spec, maze.E, maze.n, a.seed, dev)
    res['parameters'] = SN.parameter_count(net)
    log(f'trained n={maze.n} seed {a.seed} spec {a.spec}: input width {256 + pl["extra_width"]}, {res["parameters"]} parameters, '
        f'final loss {res["final_loss"]:.4f}, {res["train_seconds"] / 60:.1f} min')
    res['feature_metrics'] = DG.feature_metrics(maze, a.prepared, net.input_mapping, torch.tensor(sharp_columns(pl), device=dev))
    log(f'features: {json.dumps(res["feature_metrics"])}')
    res.update(measure(maze, a.prepared, net, a.out, log, f'sharp {a.spec} n={maze.n} s{a.seed} e{epochs}', [0, 1, 2]))
    np.save(a.out / 'slice_goal0.npy', EV.slice_values(maze, P.field_cost(net), goals[0]))
    res['elapsed_seconds'] = time.time() - t0
    res['complete'] = True
    dump(a.out / 'results.json', res)
    log(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
