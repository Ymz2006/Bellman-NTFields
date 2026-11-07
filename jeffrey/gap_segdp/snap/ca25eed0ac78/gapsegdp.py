"""jeffrey/maze_scale's segment DP (segdp3.py: DP chain + ntrl-demo's wall-direction term per level; or segdp.py, the
plain chain) on a gap maze, on top of an ntrl-demo field trained on that maze.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/gap_segdp/gapsegdp.py --method segdp3 --prepared <prep.py run> --base <checkpoint.pt> \
        --seed 87358 --normal-weight 0.3 --out <dir>

The maze module is patched for gap scale k (gap.install, read from the prepared maze.json) and the chosen script's main()
runs unchanged with the remaining arguments. One substitution: truth.Grid (used by the DP scripts only for the per-level
truth probes) becomes cross_warp's truthfast.FastGrid, the same grid, interpolation and query_time without the stored
per-neighbour cost arrays (~18 GB at 88 x 88; the scripts delete them right away anyway). The base must be ntrl-demo:
a maze_scale ntrl / plus-ntrl checkpoint or a cross-warp run with the identity warp (same network and state dict; checked).
Afterwards results.json gets the gap study's breakdown: success, collisions and stalls per region-pair category.
"""
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('NTFIELD_ROOT', '/home/jeffrey/bellman-ntfield'))
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
sys.path.insert(0, str(HERE))

IDENTITY = dict(s=1.0, s_p=1.0, xy=False, heading=False, sep=False)


def pop(flag, default=None):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        v = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return v
    return default


def arg(flag):
    return sys.argv[sys.argv.index(flag) + 1]


def main():
    method = pop('--method', 'segdp3')
    threads = int(pop('--threads', '12'))
    assert method in ('segdp', 'segdp3')
    prepared, base, out = Path(arg('--prepared')), Path(arg('--base')), Path(arg('--out'))
    info = json.load(open(prepared / 'maze.json'))
    k = int(info['gap_scale'])
    import gap as GP
    GP.install(k)
    import truth as TR
    import truthfast as TF

    class Grid(TF.FastGrid):
        def __init__(self, maze):
            super().__init__(maze, out / 'build', threads=threads)
            self.costs = self.cost_flat = None

    TR.Grid = Grid
    # the base is ntrl-demo
    base_res = json.load(open(base.parents[1] / 'results.json'))
    if base_res.get('stage') == 'cross-warp':
        import torch
        wc = torch.load(base, map_location='cpu', weights_only=False)['warp_config']
        assert all(wc.get(key) == v for key, v in IDENTITY.items()), f'base warp is not the identity: {wc}'
    else:
        assert base_res.get('stage') in ('ntrl', 'plus-ntrl'), base_res.get('stage')
    base_maze = base_res['maze']
    assert base_maze['n'] == info['n'] and base_maze.get('gap_scale', 1) == k, 'base trained on another maze'
    mod = __import__(method)
    mod.main()
    res = json.loads((out / 'results.json').read_text())
    assert res['maze'].get('variant') == 'gap' and res['maze'].get('gap_scale') == k, 'gap patch not applied'
    rows = np.load(out / 'rows.npz')
    q = np.load(prepared / 'queries.npz')
    res['by_category'] = {name: dict(success=int(rows['success'][:, q['category'] == c].sum()), collision=int(rows['collision'][:, q['category'] == c].sum()),
                                     stalled=int((~rows['converged'][:, q['category'] == c]).sum()), runs=int(rows['success'][:, q['category'] == c].size))
                          for c, name in enumerate(GP.CATEGORIES) if (q['category'] == c).any()}
    res['failures'] = dict(collision=int(rows['collision'].sum()), stalled=int((~rows['converged']).sum()),
                           stalled_no_collision=int(((~rows['converged']) & ~rows['collision']).sum()), runs=int(rows['success'].size))
    res.update(stage='gap-' + method, gap_scale=k, method=method, base_stage=base_res.get('stage'), base_run=base.parents[1].name.replace('registered_', ''))
    (out / 'results.json').write_text(json.dumps(res, indent=1))
    print('by category', {c: f'{v["success"]}/{v["runs"]}' for c, v in res['by_category'].items()}, 'failures', res['failures'], flush=True)


if __name__ == '__main__':
    main()
