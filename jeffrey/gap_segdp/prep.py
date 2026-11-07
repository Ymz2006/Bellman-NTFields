"""Prepare a gap maze (jeffrey/cross_warp/gap.py: the 22 x 22 cross maze's quadrants pushed apart by a band of 6 k cells)
for jeffrey/maze_scale's segment DP.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/gap_segdp/prep.py --queries <cross_warp gapprepare run> [--heldout-from <run>] --out <dir>

segdp.py / segdp3.py read a maze_scale prepare run: maze.json, goals.npy, queries.npz, heldout/, pool.npy and
truth_fields.npy. The gap study's prepare runs (cross_warp gapprepare.py) hold the same maze, goals and region-stratified
queries but no pool and, above about 1 GB, no truth fields. This writes a folder in that layout:

  maze.json, goals.npy, queries.npz, truth_paths.npy   copied from --queries (the gap study's own query set for this k)
  heldout/                                            copied from --heldout-from (default --queries; for k = 1 the
                                                      maze_scale plus-prepare run the k = 1 fields were trained on)
  pool.npy                                            --pool collision-free poses, maze.free(seed 11), as maze_scale
                                                      prepare.py
  truth_fields.npy (float16)                          copied from --queries if it has them, else T*(., goal) for every
                                                      goal by cross_warp truthfast.FastGrid (the same grid and truth)

and checks the truth fields against the queries' certified T* (query_time at each query's start).
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('NTFIELD_ROOT', '/home/jeffrey/bellman-ntfield'))
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
sys.path.insert(0, str(HERE))


def same_maze(info, other):
    return info['n'] == other['n'] and np.array_equal(np.array(info['V']), np.array(other['V'])) and np.array_equal(np.array(info['H']), np.array(other['H']))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--queries', type=Path, required=True)
    p.add_argument('--heldout-from', type=Path, default=None)
    p.add_argument('--pool', type=int, default=200_000)
    p.add_argument('--threads', type=int, default=12)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from common import Logger, dump
    log = Logger(a.out / 'log.txt')
    t0 = time.time()
    info = json.load(open(a.queries / 'maze.json'))
    assert info.get('variant') == 'gap', 'not a gap maze'
    k = int(info['gap_scale'])
    import gap as GP
    GP.install(k)
    import maze as MZ
    import truthfast as TF
    dev = 'cuda:0'
    maze = MZ.Maze(info['n'], dev)
    assert np.array_equal(maze.V.astype(int), np.array(info['V'])) and np.array_equal(maze.H.astype(int), np.array(info['H'])), 'maze differs'
    held = a.heldout_from or a.queries
    assert same_maze(info, json.load(open(held / 'maze.json'))), 'held-out run is on another maze'
    for f in ('maze.json', 'goals.npy', 'queries.npz', 'truth_paths.npy'):
        shutil.copyfile(a.queries / f, a.out / f)
    shutil.copytree(held / 'heldout', a.out / 'heldout', dirs_exist_ok=True)
    res = dict(stage='gapsegprep', scale=k, maze=maze.describe(), queries=str(a.queries.resolve()), heldout_from=str(held.resolve()), gpu=torch.cuda.get_device_name(0))
    # pool (maze_scale prepare.py step 6)
    pool = maze.free(a.pool, torch.Generator(dev).manual_seed(11), 0.0)
    np.save(a.out / 'pool.npy', pool.cpu().numpy())
    res['pool'] = dict(n=len(pool), per_cell=len(pool) / maze.n ** 2)
    log(f'gap k={k}, n={maze.n}: pool {len(pool)} ({len(pool) / maze.n ** 2:.1f} per cell)')
    # truth fields
    goals = torch.tensor(np.load(a.out / 'goals.npy'), device=dev)
    fg = TF.FastGrid(maze, a.out / 'build', threads=a.threads)
    tt = time.time()
    if (a.queries / 'truth_fields.npy').exists():
        src = np.load(a.queries / 'truth_fields.npy', mmap_mode='r')
        assert src.shape[0] == len(goals) and tuple(src.shape[1:]) == tuple(fg.shape), f'truth fields shape {src.shape} vs grid {fg.shape}'
        shutil.copyfile(a.queries / 'truth_fields.npy', a.out / 'truth_fields.npy')
        res['truth'] = dict(source='copied', grid_shape=list(fg.shape))
        del src
    else:
        out = np.lib.format.open_memmap(a.out / 'truth_fields.npy', mode='w+', dtype=np.float16, shape=(len(goals), *fg.shape))
        step = 4 * a.threads
        for i in range(0, len(goals), step):
            f, _ = fg.solve(goals[i:i + step])
            out[i:i + len(f)] = f.half().cpu().numpy()
            log(f'truth goals {i}-{i + len(f) - 1}: Dijkstra {time.time() - tt:.0f}s')
            del f
        out.flush()
        del out
        res['truth'] = dict(source='truthfast.FastGrid (dijkstra_full.c)', grid_shape=list(fg.shape))
    res['truth']['seconds'] = time.time() - tt
    res['truth']['gb'] = (a.out / 'truth_fields.npy').stat().st_size / 1e9
    # check against the certified query times, goal by goal (memory-mapped)
    q = np.load(a.out / 'queries.npz')
    fm = np.load(a.out / 'truth_fields.npy', mmap_mode='r')
    rel = np.zeros(len(q['t_star']))
    for g in np.unique(q['goal_index']):
        m = q['goal_index'] == g
        fld = torch.tensor(np.asarray(fm[g]), device=dev).float()[None]
        t = fg.query_time(fld, torch.zeros(int(m.sum()), dtype=torch.long, device=dev), torch.tensor(q['start'][m], device=dev))
        rel[m] = np.abs(t.cpu().numpy() / q['t_star'][m] - 1)
    res['truth']['query_check'] = dict(max_rel=float(rel.max()), median_rel=float(np.median(rel)), n=len(rel))
    log(f'truth fields {res["truth"]["gb"]:.2f} GB ({res["truth"]["source"]}); against the certified T*: median rel {np.median(rel):.2e}, max {rel.max():.2e}')
    assert rel.max() < 2e-3, 'truth fields disagree with the queries'
    res['elapsed_seconds'] = time.time() - t0
    res['complete'] = True
    dump(a.out / 'results.json', res)
    log(f'done in {res["elapsed_seconds"]:.0f}s')


if __name__ == '__main__':
    main()
