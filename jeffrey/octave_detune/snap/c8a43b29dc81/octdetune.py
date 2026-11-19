"""ntrl-demo with a detuned grid-synced octave matrix (one maze, one training seed, one features.py spec), trained,
planned and diagnosed as jeffrey/grid_sync/gridsync.py, plus the convergence record: the same field measures at
intermediate checkpoints.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/octave_detune/octdetune.py --prepared <maze_scale prepare run> --seed 87358 --spec shift1.05 --out <dir>

Training: ntrl-demo's own loop (maze_scale/train.py, unchanged package) with B replaced in place by
features.transform(B, E, spec) on the first network construction (as grid_sync/gridsync.py), so the draw (directions),
network initialization, data order, losses and optimizer are those of plain ntrl-demo with the same seed. The package
saves a checkpoint every 100 epochs; the ones at --conv-epochs and the final one are kept.

Final checkpoint (epoch 5,000): maze_scale/evaluate.py (MPPI planning seeds 0/1/2, exact scorer, field vs truth),
held-out speed fit, and diagnostics.py (near-wall slope / cos, grid-line profiles, local minima along true paths,
median |log T/T*|, failures split by arrival). The checkpoints at --keep-epochs stay in training/model/ for the
convergence record (convergence.py evaluates them, and the older runs' 1,000-epoch checkpoints, with one evaluator);
--conv-eval evaluates them here instead (planning seed(s) --conv-plan-seeds, outputs in conv/e<epoch>/). Per-epoch loss
and epoch rollbacks are in the registry console log (ntrl-demo prints them).
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
import diagnostics as DG  # noqa: E402

KEEP_EPOCHS = '250,500,750,1000,1250,1500,1750,2000,2500,3000,3500,4000,4500'


def patched_network_class(make):
    """Patch the package's NN so the first network built gets B replaced in place by make(B) (= grid_sync/gridsync.py)."""
    if str(TN.NTRL) not in sys.path:
        sys.path.insert(0, str(TN.NTRL))
    from models.metric import model_network_metric as mn
    assert Path(mn.__file__).resolve() == TN.NTRL / 'models/metric/model_network_metric.py'
    original = mn.NN
    state = dict(done=False)

    def first_network(device, dim, B):
        state['drawn'] = B.clone()
        B.copy_(make(B))
        state['done'] = True
        mn.NN = original
        return original(device, dim, B)

    mn.NN = first_network
    return mn, original, state


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
    p.add_argument('--conv-eval', action='store_true')
    p.add_argument('--conv-plan-seeds', default='0')
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
    conv = [int(e) for e in a.keep_epochs.split(',') if e] if not a.smoke else []
    conv = [e for e in conv if e < epochs]
    goals = torch.tensor(np.load(a.prepared / 'goals.npy'), device=dev)
    kind, value = FT.parse(a.spec)
    res = dict(stage='octdetune', spec=a.spec, kind=kind, value=value, maze=maze.describe(), seed=a.seed, epochs=epochs, prepared=str(a.prepared),
               gpu=torch.cuda.get_device_name(0), smoke=a.smoke, keep_epochs=conv, conv_eval=a.conv_eval, conv_plan_seeds=a.conv_plan_seeds)
    drawn = FT.draw(a.seed)
    expected = FT.transform(drawn, maze.E, a.spec)
    mn, original, state = patched_network_class(lambda B: FT.transform(B, maze.E, a.spec))
    try:
        ck = TN.train(a.prepared / 'data', a.out / 'training', 'metric', epochs, a.seed, source=goals[0].tolist(), keep_every=100)
    finally:
        mn.NN = original
    assert state['done'], 'B was never replaced'
    assert torch.equal(state['drawn'], drawn), 'ntrl-demo drew a different B than features.draw'
    B = torch.load(ck, map_location='cpu', weights_only=False)['B_state_dict']
    assert torch.equal(B, expected), 'checkpoint B differs from features.transform'
    folders = [q for q in (a.out / 'training/model').iterdir() if q.is_dir()]
    assert len(folders) == 1, folders
    folder = folders[0]
    kept = {}
    for c in sorted(folder.glob('Model_Epoch_*.pt')):
        e = int(c.name.split('_')[2])
        if e in conv:
            kept[e] = c
        elif e != epochs:
            c.unlink()
    res['s_eff'] = float(maze.E / 2 / np.median(np.abs(drawn[:, :2].double().numpy()).max(1)))
    res['features'] = FT.describe(B, maze.E, maze.n, a.spec, drawn)
    res['features_octave'] = FT.describe(FT.transform(drawn, maze.E, 'octave'), maze.E, maze.n, 'octave', drawn)
    res['train_seconds'] = time.time() - t0
    res['final_loss'] = json.load(open(a.out / 'training/progress.json'))['final_loss']
    f = res['features']
    log(f'trained n={maze.n} seed {a.seed} spec {a.spec} (factor q {np.round(f["factor_quantiles"], 4).tolist()}, locked rows {f["locked_rows"]}, '
        f'edge phase error median {f["edge_phase_error_median"]:.3f} rad, top octave {f["edge_phase_error_top_octave_median"]:.3f}): '
        f'{res["train_seconds"] / 60:.1f} min')
    net = P.load_field(ck, 'metric', dev)
    res.update(measure(maze, a.prepared, net, a.out, log, f'octdetune {a.spec} n={maze.n} s{a.seed} e{epochs}', [0, 1, 2]))
    np.save(a.out / 'slice_goal0.npy', EV.slice_values(maze, P.field_cost(net), goals[0]))
    del net
    res['convergence'] = {}
    res['kept_checkpoints'] = {str(e): str(c.relative_to(a.out)) for e, c in sorted(kept.items())}
    seeds = [int(s) for s in a.conv_plan_seeds.split(',')]
    for e, c in (sorted(kept.items()) if a.conv_eval else []):
        payload = torch.load(c, map_location='cpu', weights_only=False)
        assert payload['epoch'] == e and torch.equal(payload['B_state_dict'], expected)
        net = P.load_field(c, 'metric', dev)
        d = a.out / 'conv' / f'e{e:05d}'
        r = measure(maze, a.prepared, net, d, log, f'octdetune {a.spec} n={maze.n} s{a.seed} e{e}', seeds)
        (d / 'paths_seed0.npy').unlink(missing_ok=True)
        r['checkpoint_loss'] = float(c.name.split('ValLoss_')[1][:-3])
        res['convergence'][str(e)] = r
        del net
        torch.cuda.empty_cache()
    res['elapsed_seconds'] = time.time() - t0
    res['complete'] = True
    dump(a.out / 'results.json', res)
    log(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
