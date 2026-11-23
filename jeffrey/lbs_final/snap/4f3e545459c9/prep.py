"""Prepare one scene of the LBS final benchmark with 500 random test queries (seeds.py): the prepare script of the earlier study run unchanged
in this process, with the maze seed and the query bins patched.

    python prep.py maze  --n 14 --instance 2 --lbs-src <snapshot of jeffrey/lbs> --out <dir>      maze_scale/prepare.py, maze seed 1000 i + n
    python prep.py gap   --scale 8 --instance 2 --lbs-src ... --out <dir>                         cross_warp/gapprepare.py --base maze10, base seed 1000 i + 10
    python prep.py house3 --name Onaga --lbs-src ... --out <dir>                                  prepare3r.py (jeffrey/lbs/prepare3.py + --quota)

Training data, truth, controls and the rest are exactly those scripts' (data seed 0, 48 goals, ...). Gap mazes keep gapprepare's goals
(half in the four maze blocks, half in the open band); their region-pair categories are replaced by one category, so the 500 queries are
random over (start, goal) pairs as in the other scenes.
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(q for q in HERE.parents if (q / 'baselines/ntrl-demo').is_dir())
N_QUERIES = 500


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('kind', choices=['maze', 'gap', 'house3'])
    p.add_argument('--n', type=int)
    p.add_argument('--scale', type=int)
    p.add_argument('--instance', type=int, default=1)
    p.add_argument('--name')
    p.add_argument('--lbs-src', required=True)
    p.add_argument('--threads', type=int, default=6)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    sys.path.insert(0, str(HERE))
    import seeds as SD
    smoke = ['--smoke'] if a.smoke else []
    if a.kind == 'maze':
        sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
        import maze as MZ
        seed = SD.maze_seed(a.n, a.instance)
        MZ.maze_seed = lambda n, s=seed: s
        PR = load('ms_prepare', ROOT / 'jeffrey/maze_scale/prepare.py')
        saved = SD.random_bins()
        assert PR.BINS == SD.RANDOM_BINS
        sys.argv = ['prepare.py', '--n', str(a.n), '--quota', str(N_QUERIES), '--out', str(a.out)] + smoke
        PR.main()
        note = dict(kind='maze', n=a.n, instance=a.instance, maze_seed=seed, script='jeffrey/maze_scale/prepare.py')
    elif a.kind == 'gap':
        GP = load('cw_gapprepare', ROOT / 'jeffrey/cross_warp/gapprepare.py')        # its module code puts maze_scale and cross_warp on sys.path
        import gap
        import numpy as np
        seed = SD.maze_seed(10, a.instance)
        gap.BASES['maze10']['seed'] = seed
        gap.CATEGORIES = ['random pairs']
        gap.category = lambda ra, rb: np.zeros(np.broadcast(ra, rb).shape, dtype=np.int64)
        saved = SD.random_bins()
        sys.argv = ['gapprepare.py', '--scale', str(a.scale), '--base', 'maze10', '--quota', str(N_QUERIES), '--threads', str(a.threads), '--out', str(a.out)] + smoke
        GP.main()
        info = json.load(open(a.out / 'maze.json'))
        assert info['base']['seed'] == seed and info['base']['name'] == 'maze10', info['base']
        note = dict(kind='gap', scale=a.scale, instance=a.instance, base_seed=seed, script='jeffrey/cross_warp/gapprepare.py --base maze10',
                    goals='gapprepare: 1/8 of the goals in each maze block, the rest in the open band')
    else:
        P3 = load('prepare3r', HERE / 'prepare3r.py')                                   # imports scenes3 from --lbs-src (puts maze_scale on sys.path)
        saved = SD.random_bins()
        sys.argv = ['prepare3r.py', '--name', a.name, '--quota', str(N_QUERIES), '--lbs-src', a.lbs_src, '--out', str(a.out)] + smoke
        P3.main()
        note = dict(kind='house3', name=a.name, script='jeffrey/lbs_final/prepare3r.py (jeffrey/lbs/prepare3.py + --quota)')
    rq = SD.rebin(a.out, saved, note)
    print(f'random queries: {json.dumps(rq)}', flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
