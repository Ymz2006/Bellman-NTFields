"""ntrl-demo with a speed-shaped embedding (one maze, one training seed, one embedding), trained, planned and diagnosed exactly
as jeffrey/grid_sync's gridsync stage.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/speed_fourier/speedfourier.py --prepared <maze_scale prepare run> --seed 87358 \
        --form mix --scope line --terms 30 --walls true --out <dir>

--form   mix | top | topzero | oracle | geo | pure | plain | half (embed.py); plain with --rows-from spectrum uses the environment's frequencies,
         with --rows-from anchor | axis | axiscell | lattice the grid-anchored rows of gridrows.py
--scope  line | ridge | terms | map | integral | iterms | both | share (linefit.py);  --terms M sin / cos pairs per row (0 with integral: exact)
         hits | depth | markint | markhits | intnorm (rays.py, part 4): ray hit counts and rows that put a norm on the integral; --reach, --norm
--head   l1 | linf: T gets an added norm of the difference of the two endpoints' rows, with a trained weight (embed.install_head)
--target obstacle (1 - speed, the default) | slow (1 / speed: the integral along a line is then the travel time along it)
         | unit (1 everywhere: the integral is the length along the line, a position-only control)
--walls  true: the map of this maze; flip: the same rows read at (-x, -y) (same spectrum, walls in the wrong places);
         other: the map of a different maze of the same size (same statistics, not this environment)
--speed-from <checkpoint>: the map comes from a trained field's learned speed instead of the exact clearance

The network, its seeded initialization, data, losses and time units are ntrl-demo's: only NN.input_mapping changes. Planning,
field diagnostics and the speed fit are maze_scale/evaluate.py's.
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
sys.path.insert(0, str(ROOT / 'jeffrey/grid_sync'))
import embed as EM  # noqa: E402
import linefit as LF  # noqa: E402
import gridrows as GR  # noqa: E402
import rays as RY  # noqa: E402
import spectrum as SP  # noqa: E402
import features as FT  # noqa: E402
import maze as MZ  # noqa: E402
import plan as P  # noqa: E402
import train as TN  # noqa: E402
import evaluate as EV  # noqa: E402
from common import Logger, dump  # noqa: E402


SPEED_FLOOR = MZ.ALPHA * 0.1 ** 2 * 1.9 ** 2 + 1 - MZ.ALPHA        # the trained label's floor (0.012)


def obstacle(maze, walls, speed_from=None, goals=None, other_seed=None, target='obstacle'):
    """The target signal on the node grid: o = 1 - speed (target obstacle) or the slowness 1 / speed (target slow)."""
    if speed_from is not None:
        net = P.load_field(speed_from, 'metric', maze.device)
        s = LF.speed_map(maze, field=lambda pairs: P.field_grad(net, pairs), goals=goals, headings=4)
    else:
        src = maze
        if walls == 'other':
            src = MZ.Maze(maze.n, maze.device, seed=maze.seed + (other_seed or 7919))
        s = LF.speed_map(src)
    if target == 'unit':                                # no speed at all: the integral is then the length along the line (control)
        return np.ones_like(s)
    return 1 / np.maximum(s, SPEED_FLOOR) if target == 'slow' else 1 - s


def build(maze, seed, form, scope, terms, walls, speed_from=None, goals=None, rows_from='octave', log=None, other_seed=None, target='obstacle', reach=None, norm=None):
    """Returns the Embed, the description and the base-matrix for the checkpoint."""
    draw = FT.draw(seed)
    octave = FT.transform(draw, maze.E, 'octave').numpy()
    r, K = EM.SPLIT[form]
    info = dict(form=form, scope=scope, terms=terms, walls=walls, rows_from=rows_from, target=target, base_rows=r, speed_rows=K, speed_from=str(speed_from) if speed_from else None)
    if scope in RY.SCOPES:
        info.update(reach=reach, norm=norm)
    table = None
    if form == 'plain' and rows_from in GR.MODES:             # grid-anchored rows: frequencies and a phase per row, nothing else
        rows, phase, L = GR.rows(draw.double().numpy(), maze.E, maze.n, rows_from)
        info.update(row_digest=EM.digest(np.c_[rows, phase].astype(np.float32)), half_periods=sorted(set(L.tolist()), reverse=True), axis_rows=int((np.abs(rows[:, :2]).min(1) < 1e-9).sum()))
        return EM.Embed(rows, None, maze.device, phase=phase), info, None
    if form == 'oracle':
        cube = 1 - LF.speed_cube(maze)
        cube = ((cube - cube.mean()) / cube.std() * EM.RMS).astype(np.float32)
        if walls == 'flip':                               # the rectangle is pi-symmetric, so a half turn of the map keeps its heading
            cube = np.ascontiguousarray(cube[::-1, ::-1])
        info['table_digest'], info['table_shape'] = EM.digest(cube), list(cube.shape)
        return EM.Embed(np.delete(octave, np.arange(0, 128, 16), axis=0), None, maze.device, cube, float(maze.hi[2]), K), info, cube
    if form == 'geo':
        table, marks = LF.landmark_table(maze, K)
        table = EM.standardize(table)
        if walls == 'flip':
            table = np.ascontiguousarray(table[::-1, ::-1])
        info['landmarks'], info['table_digest'], info['table_shape'] = marks, EM.digest(table), list(table.shape)
        return EM.Embed(np.delete(octave, np.arange(0, 128, 16), axis=0), table, maze.device), info, table
    if form != 'topzero' and (K or rows_from == 'spectrum'):
        o = obstacle(maze, walls, speed_from, goals, other_seed, target)
        info['o_mean'] = float(o.mean())
    if form == 'plain':
        if rows_from == 'spectrum':
            f = SP.peaks(o, 128)
            s_eff = (maze.E / 2) / float(np.median(np.abs(draw[:, :2].double().numpy()).max(1)))
            rows = np.concatenate([f, draw[:, 2:3].double().numpy() * s_eff], 1)
            if walls == 'flip':                          # sin / cos pairs: flipping the map does not change the plain spectrum
                raise SystemExit('flip has no effect on plain rows')
            info['f_over_grid_quantiles'] = (np.quantile(np.linalg.norm(f, axis=1), [0, .1, .5, .9, 1]) / (maze.E / 2)).tolist()
        else:
            rows = octave
    elif form == 'pure':
        order = np.argsort(np.abs(draw[:, 2].numpy()), kind='stable')[::16]
        s_eff = (maze.E / 2) / float(np.median(np.abs(draw[:, :2].double().numpy()).max(1)))
        rows = np.zeros((r, 3))
        rows[:, 2] = draw[order, 2].numpy() * s_eff
    elif form in ('top', 'topzero'):
        rows = np.delete(octave, np.arange(0, 128, 16), axis=0)
    else:
        rows = octave[::2]
    if form == 'topzero':
        R = len(LF.node_axis(maze))
        return EM.Embed(rows, np.zeros((R, R, K), np.float32), maze.device), info, None
    if K:
        o_line = obstacle(maze, walls, speed_from, goals, other_seed, 'obstacle') if scope == 'both' and target != 'obstacle' else None
        table, fit = RY.rows(maze, scope, K, reach, norm, o) if scope in RY.SCOPES else LF.fit(o, maze.E, K, terms, scope, log=log, o_line=o_line)
        info['fit'] = fit
        raw = np.ascontiguousarray(table[::-1, ::-1]) if walls == 'flip' else table          # the rows before standardization (the head term reads these)
        table = EM.standardize(table)
        if walls == 'flip':
            table = np.ascontiguousarray(table[::-1, ::-1])
        if form == 'half':
            table = np.zeros_like(table)
        info['table_digest'] = EM.digest(table)
        info['table_shape'] = list(table.shape)
    phi = EM.Embed(rows, table, maze.device)
    if K:
        phi.raw = raw
    return phi, info, table


def head_scale(embedding, kind, prepared):
    """Calibration of the head term: half the median over the prepared queries of T* / N(Z(start) - Z(goal)), so that with a = 1 the term
    supplies about half of the true travel time of a typical query."""
    q = np.load(Path(prepared) / 'queries.npz')
    Z = torch.as_tensor(embedding.raw, dtype=torch.float32).reshape(embedding.raw.shape[0] ** 2, -1)
    cpu = EM.Embed(None, embedding.raw, 'cpu')
    d = (cpu.speed_rows(torch.tensor(q['start'][:, :2]), Z) - cpu.speed_rows(torch.tensor(q['goal'][:, :2]), Z)).abs()
    size = d.mean(1) if kind == 'l1' else d.amax(1)
    return float(0.5 * np.median(q['t_star'] / np.maximum(size.numpy(), 1e-9)))


def load_field(checkpoint, device='cuda:0'):
    """maze_scale/plan.load_field, for checkpoints that may carry the head weight: (net, weight or None)."""
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = dict(payload['model_state_dict'])
    weight = state.pop('skip_weight', None)
    net = P.network_class('metric')(device, 3, payload['B_state_dict'])
    net.load_state_dict(state, strict=True)
    net.to(torch.device(device)).float().eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net, weight


def patched_network_class(embedding, head=None):
    """Wrap the package's NN so the network built by Model.train() carries the embedding."""
    if str(TN.NTRL) not in sys.path:
        sys.path.insert(0, str(TN.NTRL))
    from models.metric import model_network_metric as mn
    assert Path(mn.__file__).resolve() == TN.NTRL / 'models/metric/model_network_metric.py'
    original = mn.NN
    state = dict(done=False)

    def first_network(device, dim, B):
        state['drawn'] = B.clone()
        mn.NN = original                                 # the package's NN.__init__ calls super(NN, self)
        net = original(device, dim, B)
        EM.install(net, embedding)
        if head:
            EM.install_head(net, embedding, embedding.raw, head[0], head[1], weight=head[2])
        state['done'] = True
        return net

    mn.NN = first_network
    return mn, original, state


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--form', choices=list(EM.SPLIT), required=True)
    p.add_argument('--scope', choices=['line', 'ridge', 'terms', 'map', 'integral', 'iterms', 'both', 'share'] + RY.SCOPES, default='line')
    p.add_argument('--reach', type=float, default=0.0, help='ray scopes: count hits within this many cells only (0: to the edge of the maze)')
    p.add_argument('--norm', choices=['l1', 'linf'], help='scopes markint, markhits, intnorm: the norm over the directions')
    p.add_argument('--head-weight', type=float, help='--head: keep the weight a fixed at this value instead of training it from 1')
    p.add_argument('--head', choices=['none', 'l1', 'linf'], default='none', help='add |a| * scale * norm of the difference of the two endpoints rows to T (embed.install_head)')
    p.add_argument('--target', choices=['obstacle', 'slow', 'unit'], default='obstacle', help='obstacle: 1 - speed; slow: 1 / speed (travel time per unit length); unit: 1 (control: the integral is the length along the line)')
    p.add_argument('--terms', type=int, default=30)
    p.add_argument('--walls', choices=['true', 'flip', 'other'], default='true')
    p.add_argument('--rows-from', choices=['octave', 'spectrum'] + GR.MODES, default='octave', help='form plain: octave, the maze spectrum, or a grid-anchored set (gridrows.py)')
    p.add_argument('--speed-from', type=Path)
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
    embedding, einfo, table = build(maze, a.seed, a.form, a.scope, a.terms, a.walls, a.speed_from, goals[:4], a.rows_from, log, target=a.target, reach=a.reach or None, norm=a.norm)
    head = None
    if a.head != 'none':
        head = (a.head, head_scale(embedding, a.head, a.prepared), a.head_weight)
        einfo.update(head=a.head, head_scale=head[1], head_fixed=a.head_weight)
    log(f'embedding: {json.dumps({k: v for k, v in einfo.items() if k != "fit"})}; fit {einfo.get("fit")}')
    res = dict(stage='speedfourier', embedding=einfo, form=a.form, scope=a.scope, terms=a.terms, walls=a.walls, rows_from=a.rows_from, target=a.target, maze=maze.describe(), seed=a.seed,
               epochs=epochs, prepared=str(a.prepared), gpu=torch.cuda.get_device_name(0), smoke=a.smoke)
    mn, original, state = patched_network_class(embedding, head)
    try:
        ck = TN.train(a.prepared / 'data', a.out / 'training', 'metric', epochs, a.seed, source=goals[0].tolist())
    finally:
        mn.NN = original
    assert state['done'], 'the embedding was never installed'
    assert torch.equal(state['drawn'], FT.draw(a.seed)), 'ntrl-demo drew a different B than features.draw'
    res['train_seconds'] = time.time() - t0
    res['final_loss'] = json.load(open(a.out / 'training/progress.json'))['final_loss']
    log(f'trained n={maze.n} seed {a.seed}: {res["train_seconds"] / 60:.1f} min, final loss {res["final_loss"]:.4e}')
    net, weight = load_field(ck, dev)
    EM.install(net, embedding)
    if head:
        EM.install_head(net, embedding, embedding.raw, head[0], head[1], weight=weight)
        res['head_weight'] = float(weight.abs())
        log(f'head term {head[0]}: scale {head[1]:.4e}, ' + (f'fixed weight {res["head_weight"]:.4f}' if head[2] is not None else f'trained weight {res["head_weight"]:.4f} (started at 1)'))
    cost, grad = P.field_cost(net), lambda pairs: P.field_grad(net, pairs)
    res.update(EV.evaluate(maze, a.prepared, cost, grad, a.out, log, label=f'speedfourier {a.form} {a.scope} M={a.terms} {a.walls} {a.target} n={maze.n} s{a.seed}'))
    res['speed_fit'] = EV.speed_fit(maze, a.prepared, grad)
    np.save(a.out / 'slice_goal0.npy', EV.slice_values(maze, cost, goals[0]))
    log(f'speed fit: {res["speed_fit"]}')
    res['elapsed_seconds'] = time.time() - t0
    res['complete'] = True
    dump(a.out / 'results.json', res)
    log(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
