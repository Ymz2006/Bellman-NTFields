"""Register a gap-maze segment-DP run with the repository's experiment registry.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/gap_segdp/run.py prep \
        --queries results/registered_<cross_warp gapprepare id> [--heldout-from results/registered_<id>]
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/gap_segdp/run.py segdp \
        --prepared results/registered_<prep id> --base results/registered_<ntrl run>/training/checkpoint.pt --seed 87358 \
        --method segdp3 --extra '--normal-weight 0.3'

Outputs go to results/registered_<id>/. Each run executes a frozen copy of this study's sources and of the cross_warp
modules it uses (gap.py, truthfast.py, dijkstra_full.c) in snap/<hash>/, so working files can change while runs are in
flight. The jeffrey/maze_scale modules (the segment DP itself), the ntrl-demo model files, the prepared files and the
base checkpoint are declared inputs: do not edit them while a run is in flight.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / 'jeffrey/gibson_embed'))
from run import REGISTRY, environment  # noqa: E402

DIR = 'jeffrey/gap_segdp'
OWN = ['prep.py', 'gapsegdp.py']
CROSS = ['gap.py', 'truthfast.py', 'dijkstra_full.c']                       # from jeffrey/cross_warp
MAZE = dict(prep=['maze.py', 'common.py', 'plus.py', 'truth.py'],
            segdp=['maze.py', 'common.py', 'plus.py', 'truth.py', 'plan.py', 'evaluate.py', 'fields.py', 'segdp.py', 'segdp3.py'])
NTRL = ['models/metric/model_train_metric.py', 'models/metric/model_network_metric.py', 'models/metric/model_function_metric.py', 'models/data_mlp.py']
REQUIRED = dict(prep=['results.json', 'log.txt', 'maze.json', 'goals.npy', 'queries.npz', 'pool.npy', 'truth_fields.npy', 'heldout/sampled_points.npy'],
                segdp=['results.json', 'log.txt', 'rows.npz', 'chain/field.pt'])


def snapshot():
    files = [(HERE / DIR / s, s) for s in OWN] + [(HERE / 'jeffrey/cross_warp' / s, s) for s in CROSS]
    h = hashlib.sha256()
    for src, s in files:
        h.update(s.encode() + b'\0' + src.read_bytes())
    folder = HERE / DIR / 'snap' / h.hexdigest()[:12]
    if not folder.exists():
        tmp = folder.with_name(folder.name + f'.tmp{os.getpid()}')
        tmp.mkdir(parents=True, exist_ok=True)
        for src, s in files:
            shutil.copyfile(src, tmp / s)
        try:
            tmp.rename(folder)
        except OSError:
            shutil.rmtree(tmp)                    # another launch made it first
    for src, s in files:
        assert (folder / s).read_bytes() == src.read_bytes(), f'snapshot {folder} differs in {s}'
    return folder


def rel(p):
    p = Path(p).resolve()
    return str(p.relative_to(HERE)) if p.is_relative_to(HERE) else str(p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prep', 'segdp'])
    p.add_argument('--queries', type=Path, help='prep: the gap study prepare run (cross_warp gapprepare) of this k')
    p.add_argument('--heldout-from', type=Path, help='prep: run whose heldout/ to use (default --queries)')
    p.add_argument('--prepared', type=Path, help='segdp: a prep run')
    p.add_argument('--base', type=Path, help='segdp: ntrl-demo checkpoint.pt on this maze')
    p.add_argument('--method', choices=['segdp', 'segdp3'], default='segdp3')
    p.add_argument('--seed', type=int, default=87358)
    p.add_argument('--extra', default='')
    p.add_argument('--label', default='')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--repeat-reason')
    a = p.parse_args()
    py = sys.executable
    snap = snapshot()
    inputs = dict(registry=str(REGISTRY), **{'source_' + s: str(snap / s) for s in OWN + CROSS},
                  **{'maze_' + s: str(HERE / 'jeffrey/maze_scale' / s) for s in MAZE[a.stage]})
    extra = a.extra.split()
    if a.stage == 'prep':
        q = a.queries.resolve()
        info = json.load(open(q / 'maze.json'))
        k = info['gap_scale']
        inputs.update({'queries_' + f: str(q / f) for f in ('maze.json', 'goals.npy', 'queries.npz', 'truth_paths.npy', 'results.json')})
        if (q / 'truth_fields.npy').exists():
            inputs['queries_truth_fields.npy'] = str(q / 'truth_fields.npy')
        held = (a.heldout_from or a.queries).resolve()
        inputs.update({'heldout_' + f: str(held / f) for f in ('maze.json', 'heldout')})
        args = ['--queries', str(q), '--heldout-from', str(held)] + extra
        config = dict(action='gap-segdp-prep', k=k, queries=rel(q), heldout_from=rel(held), extra=' '.join(extra), code=snap.name)
        name = f'Gap segdp prep k={k}'
        script = 'prep.py'
    else:
        pr = a.prepared.resolve()
        info = json.load(open(pr / 'maze.json'))
        k = info['gap_scale']
        base = a.base.resolve()
        inputs.update({'prepared_' + f: str(pr / f) for f in ('maze.json', 'goals.npy', 'queries.npz', 'heldout', 'pool.npy', 'truth_fields.npy', 'results.json')})
        inputs.update({'ntrl_' + s.replace('/', '_'): str(HERE / 'baselines/ntrl-demo' / s) for s in NTRL})
        inputs.update(base_checkpoint=str(base), base_results=str(base.parents[1] / 'results.json'))
        args = ['--method', a.method, '--prepared', str(pr), '--base', str(base), '--seed', str(a.seed)] + extra
        config = dict(action='gap-segdp', k=k, method=a.method, seed=a.seed, prepared=rel(pr), base=rel(base), extra=' '.join(extra), code=snap.name)
        name = f'Gap segdp k={k} {a.method} seed {a.seed}' + (f' {a.label}' if a.label else '')
        script = 'gapsegdp.py'
    if a.smoke:
        args.append('--smoke')
        config['smoke'] = True
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    spec = dict(name=f'{name} {tag}' + (' smoke' if a.smoke else ''), kind='pipeline', root=str(HERE), config=config, environment=environment(), inputs=inputs,
                command=[py, str(snap / script), *args, '--out', '{output}'], required_outputs=REQUIRED[a.stage], validation='files')
    directory = HERE / 'specs'
    directory.mkdir(exist_ok=True)
    path = directory / f'gap_segdp_{a.stage}_{tag}{"_smoke" if a.smoke else ""}.json'
    path.write_text(json.dumps(spec, indent=2) + '\n')
    if a.prepare_only:
        print(path)
        return
    check = subprocess.run([py, str(REGISTRY), 'check', str(path)], capture_output=True, text=True)
    (directory / (path.stem + '.check.json')).write_text(check.stdout)
    if check.returncode and not (check.returncode == 3 and a.repeat_reason):
        raise SystemExit(check.stderr or 'Duplicate experiment; supply a concrete --repeat-reason.')
    command = [py, str(REGISTRY), 'run', str(path)] + (['--repeat-reason', a.repeat_reason] if a.repeat_reason else [])
    sys.stdout.flush()
    os.execv(py, command)


if __name__ == '__main__':
    main()
