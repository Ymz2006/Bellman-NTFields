"""Register a pair-lift field run with the repository's experiment registry.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/pair_lift/run.py \
        --prepared results/registered_<prepare id> --blocked results/registered_<blocked id> --seed 87358 --label bit_push --extra '--lift time --pair bit --push page --tri1 0.5'

Outputs go to results/registered_<id>/. Each run executes a frozen copy of this study's sources (snap/<hash>/), so the working files can
change while runs are in flight. jeffrey/lift_scale's lift.py / scenes.py / diag.py / trainer.py, embed_warp's, maze_scale's, cross_warp's
gap.py, gibson_embed's grid.py, ntrl-demo's model files, the prepared files and the blocked flags (jeffrey/lift_scale stage 'blocked')
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

DIR = 'jeffrey/pair_lift'
SOURCES = ['pairnet.py', 'pairtrain.py', 'pairpipe.py', 'repfit.py', 'wholedata.py']
LIFT = ['lift.py', 'scenes.py', 'diag.py', 'trainer.py', 'jumps.py']
EMBED = ['scene.py', 'features.py', 'diagnostics.py']
MAZE = ['maze.py', 'plan.py', 'evaluate.py', 'common.py', 'plus.py', 'truth.py']
CROSS = ['gap.py']
GIBSON = ['grid.py']
NTRL = ['models/metric/model_train_metric.py', 'models/metric/model_network_metric.py', 'models/metric/model_function_metric.py', 'models/data_mlp.py']
PREPARED = ['maze.json', 'goals.npy', 'queries.npz', 'truth_paths.npy', 'truth_fields.npy', 'data', 'heldout']
REQUIRED = dict(field=['results.json', 'log.txt', 'rows.npz', 'training/checkpoint.pt', 'training/history.json', 'paths_seed0.npy'],
                repfit=['results.json', 'log.txt', 'training/checkpoint.pt'],
                data=['results.json', 'log.txt', 'maze.json', 'data/sampled_points.npy', 'data/speed.npy', 'data/normal.npy', 'heldout/sampled_points.npy'])
SCRIPT = dict(field='pairpipe.py', repfit='repfit.py', data='wholedata.py')


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
    p.add_argument('--stage', choices=['field', 'repfit', 'data'], default='field')
    p.add_argument('--init', type=Path, default=None, help='a finished field run to warm-start from (same network and seed)')
    p.add_argument('--field', type=Path, default=None, help='a finished field run whose checkpoint the run reads (the frozen field of --bell-tau field)')
    p.add_argument('--prepared', type=Path, required=True)
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
    prepared = a.prepared.resolve()
    assert (prepared / 'results.json').exists(), '--prepared must be a finished prepare run'
    info = json.load(open(prepared / 'maze.json'))
    sc = info.get('scene', {})
    scene = sc.get('name') or sc.get('kind') or info.get('variant', 'plain')
    inputs = dict(registry=str(REGISTRY), **{'source_' + s: str(snap / s) for s in SOURCES}, **{'lift_' + s: str(HERE / 'jeffrey/lift_scale' / s) for s in LIFT},
                  **{'embed_' + s: str(HERE / 'jeffrey/embed_warp' / s) for s in EMBED}, **{'maze_' + s: str(HERE / 'jeffrey/maze_scale' / s) for s in MAZE},
                  **{'cross_' + s: str(HERE / 'jeffrey/cross_warp' / s) for s in CROSS}, **{'gibson_' + s: str(HERE / 'jeffrey/gibson_embed' / s) for s in GIBSON},
                  **{'ntrl_' + s.replace('/', '_'): str(HERE / 'baselines/ntrl-demo' / s) for s in NTRL},
                  **{'prepared_' + f.replace('/', '_'): str(prepared / f) for f in PREPARED})
    if sc.get('kind') == 'house':
        home = Path(os.environ.get('GIBSON_TINY', str(Path.home() / 'Downloads/gibson_tiny')))
        inputs['mesh'] = str(home / sc['name'] / 'mesh_z_up.obj')
    config = dict(action='pair-lift-' + a.stage, label=a.label, extra=' '.join(extra), smoke=a.smoke, code=snap.name, n=info['n'], scene=scene, prepared=rel(prepared), seed=a.seed)
    args = ['--prepared', str(prepared), '--seed', str(a.seed)] if a.stage != 'data' else ['--from', str(prepared)]
    if a.field is not None:
        field = a.field.resolve()
        assert (field / 'training/checkpoint.pt').exists(), '--field must be a finished field run'
        config['field'] = rel(field)
        inputs.update({'field_' + f.replace('/', '_'): str(field / f) for f in ('results.json', 'training/checkpoint.pt')})
        args += ['--speed-field', str(field)]
    if a.init is not None:
        init = a.init.resolve()
        assert (init / 'training/checkpoint.pt').exists(), '--init must be a finished field run'
        config['init'] = rel(init)
        inputs.update({'init_' + f.replace('/', '_'): str(init / f) for f in ('results.json', 'training/checkpoint.pt')})
        args += ['--init-field', str(init)]
    if a.blocked is not None:
        blocked = a.blocked.resolve()
        assert (blocked / 'results.json').exists(), '--blocked must be a finished blocked run'
        config['blocked'] = rel(blocked)
        inputs.update({'blocked_' + f: str(blocked / f) for f in ('results.json', 'blocked.npy', 'blocked48.npy')})
        args += ['--blocked', str(blocked)]
    args += extra
    if a.smoke:
        args.append('--smoke')
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    name = f'Pair lift {a.stage} {scene} n={info["n"]} seed {a.seed} {a.label} {tag}' + (' smoke' if a.smoke else '')
    spec = dict(name=name, kind='pipeline', root=str(HERE), config=config, environment=environment(), inputs=inputs,
                command=[py, str(snap / SCRIPT[a.stage]), *args, '--out', '{output}'], required_outputs=REQUIRED[a.stage], validation='files')
    directory = HERE / 'specs'
    directory.mkdir(exist_ok=True)
    path = directory / f'pair_lift_{a.stage}_{tag}{"_smoke" if a.smoke else ""}.json'
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
