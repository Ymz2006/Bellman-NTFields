"""Midpoint-loss run (one maze, one training seed): resume a registered ntrl-demo run at --resume-epoch, train to
--epochs with the travel-time midpoint loss (trainer.py; --weight 0 = control), then plan and diagnose exactly as the
ntrl-demo stage does (jeffrey/maze_scale/evaluate.py: ntrl-demo MPPI on the prepared queries, planning seeds 0/1/2,
exact scorer; T / T*, Spearman, descent cosine by bin; held-out speed fit).

    CUDA_VISIBLE_DEVICES=1 python jeffrey/midpoint_dp/pipeline.py --prepared <prepare run> --base-run <ntrl run> \
        --seed 87358 --out <dir> [--weight 1 --leaf net ...]

Diagnostics during training (not used by the method): every --validate-every epochs the current field plans the
prepared queries with planning seed 0.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')        # no pool of compile worker processes (host RAM)
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = next((q for q in HERE.parents if (q / 'baselines/ntrl-demo').is_dir()), Path(os.environ.get('NTFIELD_ROOT', '/home/jeffrey/bellman-ntfield')))       # repository (also from snap/<hash>/)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
import maze as MZ  # noqa: E402
import plan as P  # noqa: E402
import evaluate as EV  # noqa: E402
from common import Logger, dump, by_bin  # noqa: E402
import trainer as TRN  # noqa: E402


def resume_checkpoint(base_run, epoch):
    folder = Path(base_run) / 'training/model'
    found = sorted(folder.glob(f'*/Model_Epoch_{epoch:05d}_*.pt'))
    assert len(found) == 1, f'no epoch-{epoch} checkpoint under {folder}'
    return found[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--base-run', type=Path, required=True, help='registered ntrl-demo run (maze_scale ntrl or fourier stage) whose checkpoint is resumed')
    p.add_argument('--resume-epoch', type=int, default=2000)
    p.add_argument('--seed', type=int, default=87358)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=5000)
    p.add_argument('--start', type=int, default=2000, help='first epoch of the midpoint term (after the resume epoch)')
    p.add_argument('--weight', type=float, default=1.0)
    p.add_argument('--ramp', type=int, default=500)
    p.add_argument('--refresh', type=int, default=100)
    p.add_argument('--retarget', type=int, default=0)
    p.add_argument('--traversals', type=int, default=8192)
    p.add_argument('--batch', type=int, default=8192)
    p.add_argument('--leaf', choices=['net', 'min', 'path'], default='net')
    p.add_argument('--leaf-cells', type=float, default=1.0)
    p.add_argument('--loss', choices=['l1', 'l2', 'huber'], default='l1')
    p.add_argument('--reached-only', action='store_true', help='train only on paths that reached their goal without collision')
    p.add_argument('--nodes', choices=['tree', 'sub', 'both'], default='tree', help='tree: the midpoint recursion\'s inner nodes; sub: random sub-paths between leaf boundaries')
    p.add_argument('--sub-factor', type=float, default=1.0, help='sub-paths per tree inner node')
    p.add_argument('--local-share', type=float, default=0.0, help='share of traversal pairs with b within --local-cells of a')
    p.add_argument('--local-cells', type=float, default=4.0)
    p.add_argument('--min-clear', type=float, default=0.1)
    p.add_argument('--budget-factor', type=float, default=2.0)
    p.add_argument('--max-steps', type=int, default=0, help='traversal step cap; 0 = 5 n cells of motion (2x the longest maze path)')
    p.add_argument('--stall-window', type=int, default=60)
    p.add_argument('--stall-drop-cells', type=float, default=0.05)
    p.add_argument('--trim-hover', type=float, default=0.25)
    p.add_argument('--traverse', choices=['plain', 'safe'], default='plain', help='safe: traversal rollouts that enter a wall are penalized')
    p.add_argument('--safe-margin', type=float, default=0.02, help='cells')
    p.add_argument('--safe-mode', choices=['first', 'ends'], default='first')
    p.add_argument('--chunk', type=int, default=1 << 20)
    p.add_argument('--no-compile', action='store_true')
    p.add_argument('--max-rollbacks', type=int, default=20)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--validate-every', type=int, default=500)
    p.add_argument('--grad-probe', type=int, default=250)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    log = Logger(a.out / 'log.txt')
    dev = 'cuda:0'
    t0 = time.time()
    info = json.load(open(a.prepared / 'maze.json'))
    maze = MZ.Maze(info['n'], dev)
    assert maze.describe()['walls'] == info['walls'] and np.array_equal(maze.V.astype(int), np.array(info['V'])), 'prepared maze differs'
    base_res = json.load(open(a.base_run / 'results.json'))
    assert base_res['stage'] in ('ntrl', 'fourier') and base_res['seed'] == a.seed and Path(base_res['prepared']).resolve() == a.prepared.resolve(), 'base run does not match'
    resume = resume_checkpoint(a.base_run, a.resume_epoch)
    if a.smoke:
        a.epochs, a.start, a.refresh, a.traversals, a.ramp, a.save_every, a.validate_every = a.resume_epoch + 30, a.resume_epoch, 10, 512, 10, 10, 30
    if a.max_steps <= 0:
        a.max_steps = int(math.ceil(5 * maze.n * maze.pitch / P.settings(maze)['step']))
    cfg = dict(seed=a.seed, epochs=a.epochs, start=a.start, weight=a.weight, ramp=a.ramp, refresh=a.refresh, retarget=a.retarget, traversals=a.traversals,
               batch=a.batch, leaf=a.leaf, leaf_cells=a.leaf_cells, loss=a.loss, reached_only=a.reached_only, nodes=a.nodes, sub_factor=a.sub_factor, local_share=a.local_share,
               local_cells=a.local_cells, min_clear=a.min_clear, budget_factor=a.budget_factor, max_steps=a.max_steps, stall_window=a.stall_window,
               stall_drop_cells=a.stall_drop_cells, trim_hover=a.trim_hover, traverse=a.traverse, safe_margin=a.safe_margin, safe_mode=a.safe_mode, chunk=a.chunk, compile=not a.no_compile, max_rollbacks=a.max_rollbacks,
               save_every=a.save_every, validate_every=a.validate_every, grad_probe=a.grad_probe, planner=P.settings(maze))
    res = dict(stage='midpoint', maze=maze.describe(), seed=a.seed, prepared=str(a.prepared), base_run=str(a.base_run), resume=str(resume),
               base_stage=base_res['stage'], b_scale=base_res.get('b_scale', 1.0),
               resume_epoch=a.resume_epoch, config={k: v for k, v in cfg.items() if k != 'planner'}, planner=cfg['planner'], gpu=torch.cuda.get_device_name(0), smoke=a.smoke)
    log(f'midpoint run n={maze.n} seed {a.seed}: resume {resume} -> epoch {a.epochs}; config {json.dumps(res["config"])}')
    Q = EV.load_queries(a.prepared, dev)
    st = P.settings(maze)
    budget = torch.tensor(np.maximum(st['min_steps'], np.ceil(st['budget_factor'] * Q['path_clen'] / st['step'])), device=dev)

    def validate(network, epoch):
        cpu_state, cuda_state = torch.get_rng_state(), torch.cuda.get_rng_state()     # plan.mppi reseeds the global RNGs
        paths, stop, conv = P.mppi(P.field_cost(network), Q['start_t'], Q['goal_t'], 0, budget, maze.lo, maze.hi, **st)
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state)
        rows = P.score(maze, [paths[i] for i in range(len(Q['start_t']))], conv)
        out = dict(success=int(sum(r['success'] for r in rows)), collision=int(sum(r['collision'] for r in rows)), converged=int(conv.sum()),
                   by_bin=by_bin(rows, Q['bin']))
        log(f'validate epoch {epoch} (planning seed 0): {out["success"]}/{len(rows)} collisions {out["collision"]} by bin {out["by_bin"]}')
        return out

    ck, hist = TRN.train(maze, a.prepared / 'data', resume, a.out / 'training', cfg, log, validate=validate)
    res['train_seconds'] = time.time() - t0
    res['refresh'] = hist['refresh']
    res['validate'] = hist['validate']
    net = P.load_field(ck, 'metric', dev)
    cost, grad = P.field_cost(net), lambda pairs: P.field_grad(net, pairs)
    res.update(EV.evaluate(maze, a.prepared, cost, grad, a.out, log, label=f'midpoint n={maze.n} s{a.seed} w{a.weight:g}'))
    res['speed_fit'] = EV.speed_fit(maze, a.prepared, grad)
    goals = torch.tensor(np.load(a.prepared / 'goals.npy'), device=dev)
    np.save(a.out / 'slice_goal0.npy', EV.slice_values(maze, cost, goals[0]))
    log(f'speed fit: {res["speed_fit"]}')
    res['elapsed_seconds'] = time.time() - t0
    res['complete'] = True
    dump(a.out / 'results.json', res)
    log(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
