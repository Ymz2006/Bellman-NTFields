"""Register the runs of a scene with several arms (real_world_12dof) with the experiment registry. All in the host venv.

    P=~/.venvs/bellman-ntfield/bin/python; export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
    $P jeffrey/lbs_arm/run12.py tests   --scene real_world_12dof --export results/registered_<export run>      # certified suites
    $P jeffrey/lbs_arm/run12.py freeze  --run results/registered_<tests run>                                   # -> tests_<scene>/
    $P jeffrey/lbs_arm/run12.py prepare --scene real_world_12dof --export results/registered_<export run>
    $P jeffrey/lbs_arm/run12.py euclid  --prepared results/registered_<prepare> --label rw12 --extra "--steps 600 2400"   # planner control, T = |x - g|
    $P jeffrey/lbs_arm/run12.py field   --prepared results/registered_<prepare> --seed 87358 --label lbs --extra "--lift time --bell 1"

--export: any export run of run.py (the arm is the same in every scene; its cuboids are not used). Every run executes a frozen copy of
its sources (snap/<hash>/, hashed over the files of its mode), so the working files can change while runs are in flight. There is no
eval mode: the field run scores its own plans (pipeline12.py).
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'jeffrey/gibson_embed'))
from run import REGISTRY, environment  # noqa: E402  (jeffrey/gibson_embed/run.py)

DIR = 'jeffrey/lbs_arm'
SCENE_SOURCES = ['arm.py', 'arm12.py', 'core.py', 'fastnet.py']
SOURCES = dict(tests=SCENE_SOURCES + ['make_tests12.py'], prepare=SCENE_SOURCES + ['prepare12.py'],
               field=SCENE_SOURCES + ['tree.py', 'trainer6.py', 'pipeline6.py', 'pipeline12.py'],
               euclid=SCENE_SOURCES + ['tree.py', 'trainer6.py', 'pipeline6.py', 'euclid.py'],
               closepairs=SCENE_SOURCES + ['tree.py', 'trainer6.py', 'closepairs.py'],
               belldiag=SCENE_SOURCES + ['tree.py', 'trainer6.py', 'belldiag.py'])
COPIES = {'fastnet.py': 'jeffrey/lbs/fastnet.py', 'tree.py': 'jeffrey/lbs/tree.py'}
MAZE = ['maze.py', 'common.py']
NTRL = ['models/metric/model_train_metric.py', 'models/metric/model_network_metric.py', 'models/metric/model_function_metric.py', 'models/data_mlp.py']
CHECK = ['jeffrey/real_world/robot.py', 'jeffrey/real_world/robot_geometry.npz']       # arm12.check's reference model


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(names):
    src = HERE / DIR
    for name, orig in COPIES.items():
        assert (src / name).read_bytes() == (HERE / orig).read_bytes(), f'{name} is no longer a verbatim copy of {orig}'
    h = hashlib.sha256()
    for s in names:
        h.update(s.encode() + b'\0' + (src / s).read_bytes())
    folder = src / 'snap' / h.hexdigest()[:12]
    if not folder.exists():
        tmp = folder.with_name(folder.name + '.tmp')
        tmp.mkdir(parents=True, exist_ok=True)
        for s in names:
            shutil.copyfile(src / s, tmp / s)
        tmp.rename(folder)
    for s in names:
        assert (folder / s).read_bytes() == (src / s).read_bytes(), f'snapshot {folder} differs in {s}'
    return folder


def rel(p):
    p = Path(p).resolve()
    return str(p.relative_to(HERE)) if p.is_relative_to(HERE) else str(p)


def launch(spec, tag, a, py, prefix):
    directory = HERE / 'specs'
    directory.mkdir(exist_ok=True)
    path = directory / f'{prefix}_{tag}{"_smoke" if a.smoke else ""}.json'
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


def scene_paths(scene):
    assert (HERE / DIR / 'scenes' / (scene + '.yml')).is_file(), scene
    return HERE / DIR / 'scenes' / (scene + '.yml'), HERE / DIR / ('tests_' + scene)


def freeze(run):
    """Copy a finished tests run into tests_<scene>/ (once; never overwrites), checking every hash."""
    src = run / 'tests'
    man = json.loads((src / 'suite_manifest.json').read_text())
    gen = json.loads((src / 'generation.json').read_text())
    assert gen['complete'] and not man['smoke'], 'not a complete full generation'
    scene, target = scene_paths(man['scene'])
    assert sha(scene) == man['scene_sha256'], 'scene file changed since generation'
    assert not target.exists(), f'{target} is already frozen'
    (target / 'suites').mkdir(parents=True)
    for name in ('suite_manifest.json', 'generation.json'):
        shutil.copyfile(src / name, target / name)
    for item in man['suites']:
        shutil.copyfile(src / 'suites' / item['file'], target / 'suites' / item['file'])
        assert sha(target / 'suites' / item['file']) == item['sha256'], item['file']
    rec = dict(source_run=run.name, scene=man['scene'], scene_sha256=man['scene_sha256'], suite_manifest_sha256=sha(target / 'suite_manifest.json'), total=man['total'],
               generator='jeffrey/lbs_arm/make_tests12.py', generator_sha256=man['generator_sha256'], note='Frozen testing data: never resample or edit.')
    (target / 'FROZEN.json').write_text(json.dumps(rec, indent=2) + '\n')
    print(json.dumps(rec, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['tests', 'freeze', 'prepare', 'field', 'euclid', 'closepairs', 'belldiag'])
    p.add_argument('--run', type=Path, help='freeze: a finished tests run')
    p.add_argument('--scene', default='real_world_12dof')
    p.add_argument('--export', type=Path)
    p.add_argument('--prepared', type=Path)
    p.add_argument('--seed', type=int, default=87358)
    p.add_argument('--label', default='v1')
    p.add_argument('--extra', default='')
    p.add_argument('--fields', nargs='+', default=[], help='closepairs: name=<field run>')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--probe', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--repeat-reason')
    a = p.parse_args()
    if a.mode == 'freeze':
        return freeze(a.run.resolve())
    py = sys.executable
    snap = snapshot(SOURCES[a.mode])
    extra = a.extra.split()
    src = {'source_' + s: str(snap / s) for s in SOURCES[a.mode]}
    maze = {'maze_' + s: str(HERE / 'jeffrey/maze_scale' / s) for s in MAZE}
    ntrl = {'ntrl_' + s.replace('/', '_'): str(HERE / 'baselines/ntrl-demo' / s) for s in NTRL}
    flags = (['--smoke'] if a.smoke else [])
    if a.mode in ('tests', 'prepare'):
        SCENE, TESTS = scene_paths(a.scene)
        ex = a.export.resolve() / 'robot_model.npz'
        assert json.loads((ex.parent / 'export.json').read_text())['complete']
    if a.mode == 'tests':
        config = dict(action='lbsarm12-tests', scene=a.scene, scene_sha256=sha(SCENE), export=rel(ex), label=a.label, code=snap.name, smoke=a.smoke, probe=a.probe,
                      per_stratum=3 if a.smoke else 100, seeds=dict(pool=20261014, pair=20261015, rrt=20261016, select=20261017), certificate='arm12.ArmSet valid rule, core.score at 0.001 rad')
        tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
        spec = dict(name=f'LBS arm12 tests {a.scene} {a.label} {tag}' + (' smoke' if a.smoke else '') + (' probe' if a.probe else ''), kind='pipeline', root=str(HERE), config=config,
                    environment=environment(), inputs=dict(registry=str(REGISTRY), scene=str(SCENE), robot_model=str(ex), **src, maze_maze=maze['maze_maze.py']),
                    command=[py, str(snap / 'make_tests12.py'), '--export', str(ex), '--scene', str(SCENE), '--output', '{output}/tests', *flags, *(['--probe'] if a.probe else [])],
                    required_outputs=['tests/suite_manifest.json', 'tests/generation.json', 'tests/suites/*.npz'], validation='files')
        return launch(spec, tag, a, py, 'lbsarm12_tests')
    if a.mode == 'prepare':
        config = dict(action='lbsarm12-prepare', scene=a.scene, export=rel(ex), label=a.label, extra=' '.join(extra), code=snap.name, smoke=a.smoke,
                      tests=sha(TESTS / 'suite_manifest.json'))
        tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
        spec = dict(name=f'LBS arm12 prepare {a.scene} {a.label} {tag}' + (' smoke' if a.smoke else ''), kind='pipeline', root=str(HERE), config=config, environment=environment(),
                    inputs=dict(registry=str(REGISTRY), scene=str(SCENE), scene_tests=str(TESTS), robot_model=str(ex), **src, **maze, **{'check_' + Path(c).name: str(HERE / c) for c in CHECK}),
                    command=[py, str(snap / 'prepare12.py'), '--export', str(ex), '--scene', str(SCENE), '--tests', str(TESTS), *extra, *flags, '--out', '{output}'],
                    required_outputs=['results.json', 'scene.json', 'robot_model.npz', 'queries.npz', 'data/sampled_points.npy', 'data/speed.npy', 'data/normal.npy'], validation='files')
        return launch(spec, tag, a, py, 'lbsarm12_prepare')
    prep_files = ['scene.json', 'robot_model.npz', 'queries.npz', 'data/sampled_points.npy', 'data/speed.npy', 'data/normal.npy']
    prepared = a.prepared.resolve()
    assert (prepared / 'results.json').exists(), '--prepared must be a finished prepare run'
    if a.mode in ('closepairs', 'belldiag'):  # predicted against true travel time on close pairs: no training; any number of joints
        fields = {i.split('=', 1)[0]: Path(i.split('=', 1)[1]).resolve() for i in a.fields}
        config = dict(action='lbsarm-' + a.mode, label=a.label, extra=' '.join(extra), smoke=a.smoke, code=snap.name, prepared=rel(prepared), fields={k: rel(v) for k, v in fields.items()})
        tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
        spec = dict(name=('LBS arm close pairs, predicted against true time' if a.mode == 'closepairs' else 'LBS arm Bellman backup residual on trained fields') + f' {a.label} {tag}' + (' smoke' if a.smoke else ''), kind='pipeline', root=str(HERE), config=config, environment=environment(),
                    inputs=dict(registry=str(REGISTRY), **src, **maze, **ntrl, **{'prepared_' + f.replace('/', '_'): str(prepared / f) for f in ('scene.json', 'robot_model.npz') + (('data/sampled_points.npy',) if a.mode == 'belldiag' else ())},
                                **{'field_' + k: str(v / 'training/checkpoint.pt') for k, v in fields.items()}),
                    command=[py, str(snap / (a.mode + '.py')), '--prepared', str(prepared), '--fields', *[f'{k}={v}' for k, v in fields.items()], *extra, *flags, '--out', '{output}'],
                    required_outputs=['results.json', 'log.txt'] + (['pairs.npz'] if a.mode == 'closepairs' else []), validation='files')
        return launch(spec, tag, a, py, 'lbsarm_' + a.mode)
    if a.mode == 'euclid':      # the planner on T = |x - g|: no training; works on a prepared scene of any number of joints
        config = dict(action='lbsarm-euclid', label=a.label, extra=' '.join(extra), smoke=a.smoke, code=snap.name, prepared=rel(prepared))
        tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
        spec = dict(name=f'LBS arm Euclidean-field planner control {a.label} {tag}' + (' smoke' if a.smoke else ''), kind='pipeline', root=str(HERE), config=config, environment=environment(),
                    inputs=dict(registry=str(REGISTRY), **src, **maze, **ntrl, **{'prepared_' + f.replace('/', '_'): str(prepared / f) for f in ('scene.json', 'robot_model.npz', 'queries.npz')}),
                    command=[py, str(snap / 'euclid.py'), '--prepared', str(prepared), *extra, *flags, '--out', '{output}'],
                    required_outputs=['results.json', 'log.txt'], validation='files')
        return launch(spec, tag, a, py, 'lbsarm_euclid')
    config = dict(action='lbsarm12-field', label=a.label, extra=' '.join(extra), smoke=a.smoke, code=snap.name, prepared=rel(prepared), seed=a.seed)
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    spec = dict(name=f'LBS arm12 field seed {a.seed} {a.label} {tag}' + (' smoke' if a.smoke else ''), kind='pipeline', root=str(HERE), config=config, environment=environment(),
                inputs=dict(registry=str(REGISTRY), **src, **maze, **ntrl, **{'prepared_' + f.replace('/', '_'): str(prepared / f) for f in prep_files}),
                command=[py, str(snap / 'pipeline12.py'), '--prepared', str(prepared), '--seed', str(a.seed), *extra, *flags, '--out', '{output}'],
                required_outputs=['results.json', 'log.txt', 'training/checkpoint.pt', 'training/history.json', 'paths_seed0.npy'], validation='files')
    launch(spec, tag, a, py, 'lbsarm12_field')


if __name__ == '__main__':
    main()
