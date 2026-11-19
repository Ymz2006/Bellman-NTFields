"""Register an embed-warp run with the repository's experiment registry.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/embed_warp/run.py \
        --stage prepare --label disks --extra '--scene disks'
    ... --stage field --prepared results/registered_<prepare id> --seed 87358 --label geo --extra '--octaves 2,4,8,16 --gate geo'

Outputs go to results/registered_<id>/. Each run executes a frozen copy of this study's sources (snap/<hash>/), so the working files can
change while runs are in flight. maze_scale's modules, cross_warp's gap.py (gap variants), ntrl-demo's model files and the prepared files
are declared inputs. Nothing here touches the frozen Bellman600 pipeline, assets or baseline.
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

DIR = 'jeffrey/embed_warp'
SOURCES = ['scene.py', 'features.py', 'gates.py', 'trainer.py', 'pipeline.py', 'prepare.py', 'diagnostics.py', 'evalq.py']
MAZE = ['maze.py', 'plan.py', 'evaluate.py', 'common.py', 'plus.py', 'data.py', 'truth.py']
CROSS = ['gap.py']
GIBSON = ['grid.py']
NTRL = ['models/metric/model_train_metric.py', 'models/metric/model_network_metric.py', 'models/metric/model_function_metric.py', 'models/data_mlp.py']
PREPARED = ['maze.json', 'goals.npy', 'queries.npz', 'truth_paths.npy', 'data', 'heldout']
REQUIRED = dict(prepare=['results.json', 'log.txt', 'maze.json', 'goals.npy', 'queries.npz', 'truth_paths.npy', 'data/sampled_points.npy', 'heldout/sampled_points.npy', 'scene.png'],
                field=['results.json', 'log.txt', 'rows.npz', 'training/checkpoint.pt', 'paths_seed0.npy'], evalq=['results.json', 'log.txt'])


def snapshot(src):
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
    p.add_argument('--stage', choices=list(REQUIRED), required=True)
    p.add_argument('--prepared', type=Path, default=None, help='field: the prepare run; evalq: the query set')
    p.add_argument('--runs', default='', help='evalq: comma-separated run directories')
    p.add_argument('--seed', type=int, default=87358)
    p.add_argument('--label', required=True)
    p.add_argument('--extra', default='')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--repeat-reason')
    a = p.parse_args()
    py = sys.executable
    snap = snapshot(HERE / DIR)
    extra = a.extra.split()
    inputs = dict(registry=str(REGISTRY), **{'source_' + s: str(snap / s) for s in SOURCES}, **{'maze_' + s: str(HERE / 'jeffrey/maze_scale' / s) for s in MAZE},
                  **{'cross_' + s: str(HERE / 'jeffrey/cross_warp' / s) for s in CROSS}, **{'gibson_' + s: str(HERE / 'jeffrey/gibson_embed' / s) for s in GIBSON})
    config = dict(action='embed-warp-' + a.stage, label=a.label, extra=' '.join(extra), smoke=a.smoke, code=snap.name)
    if a.stage == 'prepare':
        args = list(extra)
        if '--from' in extra:
            k = extra.index('--from')
            src = Path(extra[k + 1]).resolve()
            args[k + 1] = str(src)
            inputs.update({'from_' + f.replace('/', '_'): str(src / f) for f in ('maze.json', 'goals.npy', 'queries.npz', 'truth_paths.npy', 'results.json')})
            config['from'] = rel(src)
        if '--scene' in extra and extra[extra.index('--scene') + 1] == 'gibson':
            idx = extra[extra.index('--idx') + 1] if '--idx' in extra else '1'
            inputs['mesh'] = str(HERE / f'baselines/P-NTFields/datasets/gibson/{idx}/mesh_z_up_scaled.off')
        name = f'Embed warp prepare {a.label}'
    elif a.stage == 'evalq':
        queries = a.prepared.resolve()
        runs = [Path(x).resolve() for x in a.runs.split(',') if x]
        assert runs, '--runs'
        info = json.load(open(queries / 'maze.json'))
        config.update(n=info['n'], variant=info.get('variant', 'plain'), queries=rel(queries), runs=sorted(x.name.replace('registered_', '') for x in runs))
        args = ['--queries', str(queries), '--runs', ','.join(map(str, runs))] + extra
        inputs.update({'ntrl_' + s.replace('/', '_'): str(HERE / 'baselines/ntrl-demo' / s) for s in NTRL})
        inputs.update({'queries_' + f: str(queries / f) for f in ('maze.json', 'queries.npz', 'truth_paths.npy', 'goals.npy', 'results.json')})
        for x in runs:
            rid = x.name.replace('registered_', '')
            inputs[f'run_{rid}_results'] = str(x / 'results.json')
            inputs[f'run_{rid}_checkpoint'] = str(x / 'training/checkpoint.pt')
        name = f'Embed warp evalq n={info["n"]} {info.get("variant", "plain")} {a.label}'
    else:
        prepared = a.prepared.resolve()
        assert (prepared / 'results.json').exists(), '--prepared must be a finished prepare run'
        info = json.load(open(prepared / 'maze.json'))
        config.update(seed=a.seed, n=info['n'], variant=info.get('variant', 'plain'), prepared=rel(prepared))
        args = ['--prepared', str(prepared), '--seed', str(a.seed)] + extra
        inputs.update({'ntrl_' + s.replace('/', '_'): str(HERE / 'baselines/ntrl-demo' / s) for s in NTRL})
        inputs.update({'prepared_' + f.replace('/', '_'): str(prepared / f) for f in PREPARED})
        if '--curv-from' in extra:
            k = extra.index('--curv-from')
            src = Path(extra[k + 1]).resolve()
            args[args.index('--curv-from') + 1] = str(src)
            inputs['curv_from_checkpoint'] = str(src / 'training/checkpoint.pt')
            config['curv_from'] = rel(src)
            config['extra'] = ' '.join(args[4:])
        name = f'Embed warp n={info["n"]} {info.get("variant", "plain")}{info.get("gap_scale") or ""} seed {a.seed} {a.label}'
    if a.smoke:
        args.append('--smoke')
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    name += f' {tag}' + (' smoke' if a.smoke else '')
    script = dict(prepare='prepare.py', field='pipeline.py', evalq='evalq.py')[a.stage]
    spec = dict(name=name, kind='pipeline', root=str(HERE), config=config, environment=environment(), inputs=inputs,
                command=[py, str(snap / script), *args, '--out', '{output}'], required_outputs=REQUIRED[a.stage], validation='files')
    directory = HERE / 'specs'
    directory.mkdir(exist_ok=True)
    path = directory / f'embed_warp_{a.stage}_{tag}{"_smoke" if a.smoke else ""}.json'
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
