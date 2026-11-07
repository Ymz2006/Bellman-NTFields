"""Register a grid-sync run (ntrl-demo with a grid-synchronised Fourier matrix on a jeffrey/maze_scale maze) with the
repository's experiment registry.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/grid_sync/run.py \
        --prepared results/registered_<maze_scale prepare id> --seed 87358 --mode ring

Outputs go to results/registered_<id>/. Each run executes a frozen copy of this study's sources (snap/<hash>/, made
here), so the working files can change while runs are in flight. The jeffrey/maze_scale modules it imports, the
ntrl-demo model files and the prepared maze files are declared inputs (do not edit them while a run is in flight).
Nothing here touches the frozen Bellman600 pipeline, assets or baseline.
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

DIR = 'jeffrey/grid_sync'
SOURCES = ['features.py', 'gridsync.py']
MAZE = ['maze.py', 'plan.py', 'evaluate.py', 'common.py', 'train.py']
NTRL = ['models/metric/model_train_metric.py', 'models/metric/model_network_metric.py', 'models/metric/model_function_metric.py', 'models/data_mlp.py']
PREPARED = ['maze.json', 'goals.npy', 'queries.npz', 'data', 'heldout']
REQUIRED = ['results.json', 'log.txt', 'rows.npz', 'training/checkpoint.pt', 'paths_seed0.npy']
MODES = ['ring', 'circle', 'literal', 'scaled', 'snap', 'ringhalf', 'circlehalf', 'octave', 'octavecircle', 'octaveperm']


def snapshot(src):
    """Copy the sources to jeffrey/grid_sync/snap/<hash>/ (never edited afterwards); returns that folder."""
    h = hashlib.sha256()
    for s in SOURCES:
        h.update(s.encode() + b'\0' + (src / s).read_bytes())
    folder = HERE / DIR / 'snap' / h.hexdigest()[:12]
    if not folder.exists():
        tmp = folder.with_name(folder.name + '.tmp')
        tmp.mkdir(parents=True, exist_ok=True)
        for s in SOURCES:
            shutil.copyfile(src / s, tmp / s)
        tmp.rename(folder)
    for s in SOURCES:
        assert (folder / s).read_bytes() == (src / s).read_bytes(), f'snapshot {folder} differs in {s}'
    return folder


def rel(p):
    p = Path(p).resolve()
    return str(p.relative_to(HERE)) if p.is_relative_to(HERE) else str(p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--seed', type=int, default=87358)
    p.add_argument('--mode', choices=MODES, required=True)
    p.add_argument('--extra', default='', help='extra pipeline arguments (one string)')
    p.add_argument('--label', default='')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--repeat-reason')
    p.add_argument('--src', type=Path, default=HERE / DIR, help='source folder to snapshot (default: the working copy)')
    a = p.parse_args()
    py = sys.executable
    assert (a.prepared / 'results.json').exists(), '--prepared must be a finished maze_scale prepare run'
    n = json.load(open(a.prepared / 'maze.json'))['n']
    snap = snapshot(a.src.resolve())
    inputs = dict(registry=str(REGISTRY), **{'source_' + s: str(snap / s) for s in SOURCES},
                  **{'maze_' + s: str(HERE / 'jeffrey/maze_scale' / s) for s in MAZE},
                  **{'ntrl_' + s.replace('/', '_'): str(HERE / 'baselines/ntrl-demo' / s) for s in NTRL},
                  **{'prepared_' + f.replace('/', '_'): str(a.prepared.resolve() / f) for f in PREPARED})
    config = dict(action='grid-sync', n=n, seed=a.seed, mode=a.mode, prepared=rel(a.prepared), extra=a.extra, smoke=a.smoke, code=snap.name)
    args = ['--prepared', str(a.prepared.resolve()), '--seed', str(a.seed), '--mode', a.mode] + a.extra.split()
    if a.smoke:
        args.append('--smoke')
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    name = f'Grid sync n={n} seed {a.seed} {a.mode}' + (f' {a.label}' if a.label else '') + f' {tag}' + (' smoke' if a.smoke else '')
    spec = dict(name=name, kind='pipeline', root=str(HERE), config=config, environment=environment(), inputs=inputs,
                command=[py, str(snap / 'gridsync.py'), *args, '--out', '{output}'], required_outputs=REQUIRED, validation='files')
    directory = HERE / 'specs'
    directory.mkdir(exist_ok=True)
    path = directory / f'grid_sync_{tag}{"_smoke" if a.smoke else ""}.json'
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
