"""Register a run of the anisotropic-eikonal study with the repository's experiment registry.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/aniso_eikonal/run.py \
        --prepared results/registered_<cross_warp gapprepare id> --seed 87358 --label zaniso22 --extra '--mode zaniso --unit-cells 22.1'
    ... --stage verify --prepared <gapprepare id> --label verify     (the idea's checks on ground truth, no training)

Outputs go to results/registered_<id>/. Each run executes a frozen copy of the sources (snap/<hash>/: this study's files
and the jeffrey/cross_warp modules they import, copied at registration), so working files can change while runs are in
flight. The jeffrey/maze_scale modules, the ntrl-demo model files and the prepared maze files are declared inputs (do not
edit them while a run is in flight). Nothing here touches the frozen Bellman600 pipeline, assets or baseline.
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

DIR = 'jeffrey/aniso_eikonal'
OWN = ['aniso.py', 'atrain.py', 'pipeline.py', 'verify.py', 'nplan.py']
CROSS = ['warp.py', 'trainer.py', 'gap.py', 'zplan.py', 'warpplan.py', 'truthfast.py', 'warptruth.py', 'dijkstra_full.c', 'dijkstra_warp.c']
MAZE = ['maze.py', 'plan.py', 'common.py', 'plus.py', 'truth.py']
NTRL = ['models/metric/model_train_metric.py', 'models/metric/model_network_metric.py', 'models/metric/model_function_metric.py', 'models/data_mlp.py']
PREPARED = ['maze.json', 'goals.npy', 'queries.npz', 'truth_paths.npy', 'results.json', 'data', 'heldout']
REQUIRED = dict(train=['results.json', 'log.txt', 'rows.npz', 'training/checkpoint.pt'], verify=['results.json', 'log.txt'], evalq=['results.json', 'log.txt', 'rows.npz'])


def snapshot():
    """Copy the sources to jeffrey/aniso_eikonal/snap/<hash>/ (never edited afterwards); returns that folder."""
    files = [(HERE / DIR / s, s) for s in OWN if (HERE / DIR / s).exists()] + [(HERE / 'jeffrey/cross_warp' / s, s) for s in CROSS]
    h = hashlib.sha256()
    for path, s in files:
        h.update(s.encode() + b'\0' + path.read_bytes())
    folder = HERE / DIR / 'snap' / h.hexdigest()[:12]
    if not folder.exists():
        tmp = folder.with_name(folder.name + f'.tmp{os.getpid()}')        # per process: several queue slots may register at once
        tmp.mkdir(parents=True, exist_ok=True)
        for path, s in files:
            shutil.copyfile(path, tmp / s)
        try:
            tmp.rename(folder)
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)
            assert folder.exists()
    for path, s in files:
        assert (folder / s).read_bytes() == path.read_bytes(), f'snapshot {folder} differs in {s}'
    return folder, [s for _, s in files]


def rel(p):
    p = Path(p).resolve()
    return str(p.relative_to(HERE)) if p.is_relative_to(HERE) else str(p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['train', 'verify', 'evalq'], default='train')
    p.add_argument('--prepared', type=Path, required=True, help='a cross_warp gapprepare run (maze, pairs, queries)')
    p.add_argument('--seed', type=int, default=87358)
    p.add_argument('--label', required=True)
    p.add_argument('--extra', default='', help='pipeline arguments (one string)')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--repeat-reason')
    a = p.parse_args()
    py = sys.executable
    snap, names = snapshot()
    prepared = a.prepared.resolve()
    assert (prepared / 'results.json').exists(), '--prepared must be a finished gapprepare run'
    info = json.load(open(prepared / 'maze.json'))
    inputs = dict(registry=str(REGISTRY), **{'source_' + s: str(snap / s) for s in names},
                  **{'maze_' + s: str(HERE / 'jeffrey/maze_scale' / s) for s in MAZE},
                  **{'ntrl_' + s.replace('/', '_'): str(HERE / 'baselines/ntrl-demo' / s) for s in NTRL},
                  **{'prepared_' + f: str(prepared / f) for f in PREPARED if (prepared / f).exists()})
    extra = a.extra.split()
    if '--extra-data' in extra:                       # a second set of training pairs: its files are inputs
        i = extra.index('--extra-data')
        src = Path(extra[i + 1]).resolve()
        extra[i + 1] = str(src)
        inputs.update({'extra_data_' + f.replace('/', '_'): str(src / f) for f in ('maze.json', 'data/sampled_points.npy', 'data/speed.npy')})
    for flag in ('--checkpoint', '--runs', '--head-from'):
        if flag in extra:
            i = extra.index(flag)
            paths = [str(Path(x).resolve()) for x in extra[i + 1].split(',')]
            extra[i + 1] = ','.join(paths)
            inputs.update({f'{flag[2:]}_{j}': x if x.endswith('.pt') else x + '/results.json' for j, x in enumerate(paths)})
    config = dict(action='aniso-eikonal-' + a.stage, n=info['n'], gap_scale=info.get('gap_scale'), prepared=rel(prepared), label=a.label, extra=' '.join(extra),
                  smoke=a.smoke, code=snap.name)
    if a.stage != 'verify':
        config['seed'] = a.seed
    script = dict(train='pipeline.py', verify='verify.py', evalq='pipeline.py')[a.stage]      # evalq: pipeline.py --checkpoint (no training)
    args = ['--prepared', str(prepared)] + (['--seed', str(a.seed)] if a.stage != 'verify' else []) + extra + (['--smoke'] if a.smoke else [])
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    name = f'Aniso eikonal {a.stage} n={info["n"]} {a.label}' + (f' seed {a.seed}' if a.stage != 'verify' else '') + f' {tag}' + (' smoke' if a.smoke else '')
    spec = dict(name=name, kind='pipeline', root=str(HERE), config=config, environment=environment(), inputs=inputs,
                command=[py, str(snap / script), *args, '--out', '{output}'], required_outputs=REQUIRED[a.stage], validation='files')
    directory = HERE / 'specs'
    directory.mkdir(exist_ok=True)
    path = directory / f'aniso_eikonal_{a.stage}_{tag}{"_smoke" if a.smoke else ""}.json'
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
