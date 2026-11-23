"""Maze instances other than maze_scale's default layout, and the random query protocol, for the LBS final benchmark.

A maze_scale maze of side n is generated from seed 1000 + n (maze.maze_seed); the gap mazes of cross_warp/gap.py with base 'maze10' cut the
10 x 10 maze of seed 1010. This study makes three instances per size: maze seeds 1000 + n (the layout every earlier study used),
2000 + n and 3000 + n; gap-maze bases 1010, 2010 and 3010. Every prepare run stores the seed in maze.json (maze: 'seed', gap maze:
'base'.'seed'), and the loaders of the earlier studies rebuild the maze from n alone and assert the walls equal; apply(info) patches
the generator seed in this process so that they rebuild the stored instance. Nothing on disk is changed.

random_bins(): the prepare scripts pick their test queries per distance bin of common.BINS (a quota per bin). With the bins replaced by
one bin [0.5 cells, inf) and quota 500 they pick 500 (start, goal) pairs at random from all candidate pairs (starts uniform over the free
space with clearance >= 0.1 cells, goals the scene's 48 random goal poses, at most ceil(500 / 12) per goal before the cap is relaxed;
every pair certified by its truth path). rebin() restores the bins and writes each query's true distance bin back into queries.npz.
"""
import json
from pathlib import Path
import numpy as np

DEFAULT_MAZE_SEED = 1000          # + n
DEFAULT_GAP_BASE_SEED = 1010
RANDOM_BINS = [0.5, 1e9]


def maze_seed(n, instance):
    """Seed of instance 1, 2, 3 of the n x n maze."""
    return 1000 * instance + n


def apply(info):
    """Patch this process so that embed_warp/scene.load_scene (and Maze(n)) rebuild the maze stored in maze.json."""
    if 'scene' in info:
        return 'scene'                                       # houses: built from their own spec
    variant = info.get('variant', 'plain')
    if variant == 'gap':
        import gap
        base = info.get('base') or {}
        name = base.get('name', 'cross22')
        seed = int(base['seed'])
        if gap.BASES[name]['seed'] != seed:
            assert not getattr(gap, '_installed', None), 'gap maze already installed with another base seed'
            gap.BASES[name]['seed'] = seed
        return f'gap base {name} seed {seed}'
    import maze as MZ
    seed = int(info['seed'])
    if seed != MZ.maze_seed(int(info['n'])):
        MZ.maze_seed = lambda n, s=seed: s
    return f'maze seed {seed}'


def random_bins():
    import common
    saved = list(common.BINS)
    common.BINS[:] = RANDOM_BINS
    return saved


def rebin(out, saved, note):
    import common
    common.BINS[:] = saved
    out = Path(out)
    q = dict(np.load(out / 'queries.npz'))
    b = np.clip(np.digitize(q['d_cells'], saved) - 1, 0, len(saved) - 2)
    q['bin'] = b.astype(np.int64)
    np.savez(out / 'queries.npz', **q)
    res = json.load(open(out / 'results.json'))
    res['random_queries'] = dict(note, n=int(len(q['start'])), bins=saved, per_bin={f'{saved[i]:g}-{saved[i + 1]:g}': int((b == i).sum()) for i in range(len(saved) - 1)},
                                 d_cells_quantiles={str(p): float(np.quantile(q['d_cells'], p)) for p in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)})
    json.dump(res, open(out / 'results.json', 'w'), indent=1)
    return res['random_queries']
