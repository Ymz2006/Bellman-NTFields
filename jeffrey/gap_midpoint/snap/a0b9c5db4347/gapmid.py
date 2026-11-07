"""jeffrey/midpoint_dp's travel-time midpoint loss on a gap maze, resumed from the gap study's ntrl-demo run at epoch 2000.

    CUDA_VISIBLE_DEVICES=0 python jeffrey/gap_midpoint/gapmid.py --prepared <gap_segdp prep run> --base-run <ntrl-demo run> \
        --seed 87358 --weight 1 --leaf min --traverse safe --nodes both --out <dir>      (--weight 0 = control)

The maze module is patched for gap scale k (gap.install, read from the prepared maze.json) and jeffrey/midpoint_dp's
pipeline.py main() runs unchanged with the remaining arguments, on two small adapter folders (symlinks, removed at the end):

  <out>/layout: the gap study's maze, goals, queries and held-out data (the gap_segdp prep run) next to the training data
  of the base run (its own prepare run's data/). <out>/base: the base's epoch-2000 checkpoint under the name pipeline.py
  looks for, with a results.json that names the layout. The base must be ntrl-demo: a plus-ntrl run, or a cross-warp run
  with the identity warp (same network, state dict and optimizer; checked).

Afterwards results.json gets the gap study's breakdown: success, collisions and stalls per region-pair category.
"""
import json
import os
import shutil
import sys
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next((q for q in HERE.parents if (q / 'baselines/ntrl-demo').is_dir()), Path(os.environ.get('NTFIELD_ROOT', '/home/jeffrey/bellman-ntfield')))
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
sys.path.insert(0, str(HERE))

IDENTITY = dict(s=1.0, s_p=1.0, xy=False, heading=False, sep=False)


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def setarg(flag, value):
    sys.argv[sys.argv.index(flag) + 1] = str(value)


def base_checkpoint(base, epoch):
    """(path, ntrl-demo style file name) of the base run's checkpoint at this epoch."""
    found = sorted((base / 'training/model').glob(f'*/Model_Epoch_{epoch:05d}_*.pt'))
    if found:
        assert len(found) == 1
        return found[0], found[0].name
    path = base / f'training/checkpoints/epoch_{epoch:05d}.pt'
    assert path.exists(), f'no epoch-{epoch} checkpoint in {base}'
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=False)
    wc = payload['warp_config']
    assert all(wc.get(key) == v for key, v in IDENTITY.items()), f'base warp is not the identity: {wc}'
    assert int(payload['epoch']) == epoch
    return path, f'Model_Epoch_{epoch:05d}_ValLoss_{float(payload["val_loss"]):.6e}.pt'


def main():
    prepared, base, out = Path(arg('--prepared')).resolve(), Path(arg('--base-run')).resolve(), Path(arg('--out')).resolve()
    seed, epoch = int(arg('--seed', 87358)), int(arg('--resume-epoch', 2000))
    info = json.load(open(prepared / 'maze.json'))
    k = int(info['gap_scale'])
    import gap as GP
    GP.install(k)
    base_res = json.load(open(base / 'results.json'))
    assert base_res.get('stage') in ('cross-warp', 'plus-ntrl') and base_res['seed'] == seed, (base_res.get('stage'), base_res['seed'])
    assert base_res['maze']['n'] == info['n'] and base_res['maze'].get('gap_scale', 1) == k, 'base trained on another maze'
    data = Path(base_res['prepared'])
    data = (data if data.is_absolute() else ROOT / data) / 'data'
    base_info = json.load(open(data.parent / 'maze.json'))
    assert np.array_equal(base_info['V'], info['V']) and np.array_equal(base_info['H'], info['H']), 'base data from another maze'
    ck, ck_name = base_checkpoint(base, epoch)

    out.mkdir(parents=True, exist_ok=True)
    layout, adapter = out / 'layout', out / 'base'
    layout.mkdir(exist_ok=True)
    for f in ('maze.json', 'goals.npy', 'queries.npz', 'heldout'):
        (layout / f).symlink_to(prepared / f)
    (layout / 'data').symlink_to(data)
    folder = adapter / 'training/model/resume'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / ck_name).symlink_to(ck.resolve())
    (adapter / 'results.json').write_text(json.dumps(dict(stage='ntrl', seed=seed, prepared=str(layout), b_scale=1.0)))
    setarg('--prepared', layout)
    setarg('--base-run', adapter)

    import pipeline as PL
    PL.main()

    res = json.loads((out / 'results.json').read_text())
    assert res['maze'].get('variant') == 'gap' and res['maze'].get('gap_scale') == k, 'gap patch not applied'
    rows = np.load(out / 'rows.npz')
    q = np.load(prepared / 'queries.npz')
    res['by_category'] = {name: dict(success=int(rows['success'][:, q['category'] == c].sum()), collision=int(rows['collision'][:, q['category'] == c].sum()),
                                     stalled=int((~rows['converged'][:, q['category'] == c]).sum()), runs=int(rows['success'][:, q['category'] == c].size))
                          for c, name in enumerate(GP.CATEGORIES) if (q['category'] == c).any()}
    res['failures'] = dict(collision=int(rows['collision'].sum()), stalled=int((~rows['converged']).sum()),
                           stalled_no_collision=int(((~rows['converged']) & ~rows['collision']).sum()), runs=int(rows['success'].size))
    res.update(stage='gap-midpoint', gap_scale=k, prepared=str(prepared), base_run=str(base), base_stage=base_res['stage'], resume=str(ck), data=str(data))
    (out / 'results.json').write_text(json.dumps(res, indent=1))
    shutil.rmtree(layout)
    shutil.rmtree(adapter)
    print('by category', {c: f'{v["success"]}/{v["runs"]}' for c, v in res['by_category'].items()}, 'failures', res['failures'], flush=True)


if __name__ == '__main__':
    main()
