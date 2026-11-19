"""Register a lift-scale run with the repository's experiment registry.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/lift_scale/run.py --stage house --label Allensville --extra '--name Allensville'
    ... --stage blocked --prepared results/registered_<prepare id> --label plus22
    ... --stage field --prepared results/registered_<prepare id> --blocked results/registered_<blocked id> --seed 87358 --label lift_push --extra '--lift time --push page'

Outputs go to results/registered_<id>/. Each run executes a frozen copy of this study's sources (snap/<hash>/), so the working files can
change while runs are in flight. embed_warp's, maze_scale's, cross_warp's gap.py, gibson_embed's grid.py, ntrl-demo's model files and the
prepared files are declared inputs. Nothing here touches the frozen Bellman600 pipeline, assets or baseline.
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

DIR = 'jeffrey/lift_scale'
SOURCES = ['scenes.py', 'lift.py', 'trainer.py', 'pipeline.py', 'diag.py', 'blocked.py', 'house.py']
EMBED = ['scene.py', 'features.py', 'diagnostics.py', 'prepare.py']
MAZE = ['maze.py', 'plan.py', 'evaluate.py', 'common.py', 'plus.py', 'data.py', 'truth.py']
CROSS = ['gap.py']
GIBSON = ['grid.py']
NTRL = ['models/metric/model_train_metric.py', 'models/metric/model_network_metric.py', 'models/metric/model_function_metric.py', 'models/data_mlp.py']
PREPARED = ['maze.json', 'goals.npy', 'queries.npz', 'truth_paths.npy', 'truth_fields.npy', 'data', 'heldout']
REQUIRED = dict(house=['results.json', 'log.txt', 'maze.json', 'goals.npy', 'queries.npz', 'truth_paths.npy', 'data/sampled_points.npy', 'heldout/sampled_points.npy', 'scene.png'],
                blocked=['results.json', 'log.txt', 'blocked.npy', 'blocked48.npy'],
                field=['results.json', 'log.txt', 'rows.npz', 'training/checkpoint.pt', 'paths_seed0.npy'])


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
    p.add_argument('--prepared', type=Path, default=None)
    p.add_argument('--blocked', type=Path, default=None)
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
    inputs = dict(registry=str(REGISTRY), **{'source_' + s: str(snap / s) for s in SOURCES}, **{'embed_' + s: str(HERE / 'jeffrey/embed_warp' / s) for s in EMBED},
                  **{'maze_' + s: str(HERE / 'jeffrey/maze_scale' / s) for s in MAZE}, **{'cross_' + s: str(HERE / 'jeffrey/cross_warp' / s) for s in CROSS},
                  **{'gibson_' + s: str(HERE / 'jeffrey/gibson_embed' / s) for s in GIBSON})
    config = dict(action='lift-scale-' + a.stage, label=a.label, extra=' '.join(extra), smoke=a.smoke, code=snap.name)
    if a.stage == 'house':
        name_ = extra[extra.index('--name') + 1]
        home = Path(os.environ.get('GIBSON_TINY', str(Path.home() / 'Downloads/gibson_tiny')))
        inputs['mesh'] = str(home / name_ / 'mesh_z_up.obj')
        inputs['cameras'] = str(home / name_ / 'camera_poses.csv')
        args = list(extra)
        name = f'Lift scale house {a.label}'
    else:
        prepared = a.prepared.resolve()
        assert (prepared / 'results.json').exists(), '--prepared must be a finished prepare run'
        info = json.load(open(prepared / 'maze.json'))
        scene = info.get('scene', {}).get('name') or info.get('scene', {}).get('kind') or info.get('variant', 'plain')
        config.update(n=info['n'], scene=scene, prepared=rel(prepared))
        inputs.update({'prepared_' + f.replace('/', '_'): str(prepared / f) for f in PREPARED})
        args = ['--prepared', str(prepared)]
        if a.stage == 'field':
            config['seed'] = a.seed
            args += ['--seed', str(a.seed)]
            inputs.update({'ntrl_' + s.replace('/', '_'): str(HERE / 'baselines/ntrl-demo' / s) for s in NTRL})
            if a.blocked is not None:
                blocked = a.blocked.resolve()
                assert (blocked / 'results.json').exists(), '--blocked must be a finished blocked run'
                config['blocked'] = rel(blocked)
                inputs.update({'blocked_' + f: str(blocked / f) for f in ('results.json', 'blocked.npy', 'blocked48.npy')})
                args += ['--blocked', str(blocked)]
        args += extra
        name = f'Lift scale {a.stage} {scene} n={info["n"]}' + (f' seed {a.seed}' if a.stage == 'field' else '') + f' {a.label}'
    if a.smoke:
        args.append('--smoke')
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    name += f' {tag}' + (' smoke' if a.smoke else '')
    script = dict(house='house.py', blocked='blocked.py', field='pipeline.py')[a.stage]
    spec = dict(name=name, kind='pipeline', root=str(HERE), config=config, environment=environment(), inputs=inputs,
                command=[py, str(snap / script), *args, '--out', '{output}'], required_outputs=REQUIRED[a.stage], validation='files')
    directory = HERE / 'specs'
    directory.mkdir(exist_ok=True)
    path = directory / f'lift_scale_{a.stage}_{tag}{"_smoke" if a.smoke else ""}.json'
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
