"""Register the runs of the arm-placement study with the experiment registry (host venv; every run executes a frozen copy of its
sources in snap/<hash>/, so the working files can change while runs are in flight).

    P=~/.venvs/bellman-ntfield/bin/python; export CUDA_VISIBLE_DEVICES=0
    $P jeffrey/arm_placement/run.py reach   --extra "--dx 0.25 0.35 ... --s 0.30 0.46 --h 0.60 0.80"
    $P jeffrey/arm_placement/run.py tests   --dx 0.55 --s 0.30 --h 0.60 [--extra "--per 50"]          # certified suites of a placement
    $P jeffrey/arm_placement/run.py prepare --scene <name> --tests results/registered_<tests run>       # training pairs + queries
    $P jeffrey/arm_placement/run.py field   --prepared results/registered_<prepare> --seed 87358 --label lift --extra "--lift time --weight 0 --freq 0.5"
    $P jeffrey/arm_placement/run.py euclid  --prepared results/registered_<prepare>                     # planner control, T = |x - g|
    $P jeffrey/arm_placement/run.py demo    --prepared ... --fields name=<field run> ...                # hard queries, paths in 12-D

--scene real_world_12dof with --tests jeffrey/lbs_arm/tests_real_world_12dof prepares the earlier study's placement on its frozen suites.
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
from run import REGISTRY, environment  # noqa: E402  (jeffrey/gibson_embed/run.py)

DIR = 'jeffrey/arm_placement'
LBS = 'jeffrey/lbs_arm'
OWN = ['place.py', 'suites.py', 'prep.py', 'demo.py', 'evalfield.py', 'bidir.py']
SCENE_SOURCES = ['arm.py', 'arm12.py', 'core.py', 'fastnet.py']
SOURCES = dict(reach=['place.py'], tests=SCENE_SOURCES + ['make_tests12.py', 'suites.py'], prepare=SCENE_SOURCES + ['prepare12.py', 'prep.py'],
               field=SCENE_SOURCES + ['tree.py', 'trainer6.py', 'pipeline6.py', 'pipeline12.py'],
               euclid=SCENE_SOURCES + ['tree.py', 'trainer6.py', 'pipeline6.py', 'euclid.py'],
               demo=SCENE_SOURCES + ['tree.py', 'trainer6.py', 'pipeline6.py', 'make_tests12.py', 'demo.py'],
               evalfield=SCENE_SOURCES + ['tree.py', 'trainer6.py', 'pipeline6.py', 'bidir.py', 'evalfield.py'])
MAZE = ['maze.py', 'common.py']
NTRL = ['models/metric/model_train_metric.py', 'models/metric/model_network_metric.py', 'models/metric/model_function_metric.py', 'models/data_mlp.py']
REAL = ['jeffrey/real_world/robot.py', 'jeffrey/real_world/robot_geometry.npz', 'jeffrey/real_world/make_scenes.py']
EXPORT = HERE / DIR / 'arm_export.npz'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def origin(name):
    return HERE / (DIR if name in OWN else LBS) / name


def frozen(name, names):
    """An existing snapshot (e.g. the one an earlier run used), checked against its own content hash."""
    folder = HERE / DIR / 'snap' / name
    h = hashlib.sha256()
    for s in names:
        h.update(s.encode() + b'\0' + (folder / s).read_bytes())
    assert h.hexdigest()[:12] == name, f'snapshot {folder} does not hash to its name'
    return folder


def snapshot(names):
    h = hashlib.sha256()
    for s in names:
        h.update(s.encode() + b'\0' + origin(s).read_bytes())
    folder = HERE / DIR / 'snap' / h.hexdigest()[:12]
    if not folder.exists():
        tmp = folder.with_name(folder.name + f'.tmp{os.getpid()}')
        tmp.mkdir(parents=True, exist_ok=True)
        for s in names:
            shutil.copyfile(origin(s), tmp / s)
        try:
            tmp.rename(folder)
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)
    for s in names:
        assert (folder / s).read_bytes() == origin(s).read_bytes(), f'snapshot {folder} differs in {s}'
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


def scene_file(name):
    f = HERE / (LBS if name.startswith('real_world') else DIR) / 'scenes' / (name + '.yml')
    assert f.is_file(), f
    return f


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=list(SOURCES))
    p.add_argument('--dx', type=float)
    p.add_argument('--s', type=float, default=0.30)
    p.add_argument('--h', type=float, default=0.60)
    p.add_argument('--scene')
    p.add_argument('--tests', type=Path)
    p.add_argument('--prepared', type=Path)
    p.add_argument('--field', type=Path)
    p.add_argument('--fields', nargs='+', default=[])
    p.add_argument('--seed', type=int, default=87358)
    p.add_argument('--label', default='v1')
    p.add_argument('--extra', default='')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--repeat-reason')
    p.add_argument('--snap', help='run an existing snapshot snap/<hash> instead of the working files (to repeat an earlier run exactly)')
    a = p.parse_args()
    py = sys.executable
    snap = frozen(a.snap, SOURCES[a.mode]) if a.snap else snapshot(SOURCES[a.mode])
    extra = a.extra.split()
    src = {'source_' + s: str(snap / s) for s in SOURCES[a.mode]}
    maze = {'maze_' + s: str(HERE / 'jeffrey/maze_scale' / s) for s in MAZE}
    ntrl = {'ntrl_' + s.replace('/', '_'): str(HERE / 'baselines/ntrl-demo' / s) for s in NTRL}
    real = {'real_' + Path(c).name: str(HERE / c) for c in REAL}
    flags = ['--smoke'] if a.smoke else []

    def spec_of(action, name, config, inputs, command, outputs, prefix):
        config = dict(action=action, label=a.label, extra=' '.join(extra), smoke=a.smoke, code=snap.name, **config)
        tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
        spec = dict(name=f'{name} {a.label} {tag}' + (' smoke' if a.smoke else ''), kind='pipeline', root=str(HERE), config=config, environment=environment(),
                    inputs=dict(registry=str(REGISTRY), **src, **inputs), command=command, required_outputs=outputs, validation='files')
        return launch(spec, tag, a, py, prefix)

    if a.mode == 'reach':
        return spec_of('armplace-reach', 'Arm placement reach grid', {}, real, [py, str(snap / 'place.py'), 'reach', *extra, '--out', '{output}'], ['reach.json', 'log.txt'], 'armplace_reach')
    if a.mode == 'tests':
        sys.path.insert(0, str(HERE / DIR))
        import place
        scene = place.write_scene(a.dx, a.s, a.h)
        return spec_of('armplace-tests', f'Arm placement tests {scene.stem}', dict(scene=scene.stem, scene_sha256=sha(scene), export=sha(EXPORT)),
                       dict(scene=str(scene), robot_model=str(EXPORT), maze_maze=maze['maze_maze.py']),
                       [py, str(snap / 'suites.py'), '--export', str(EXPORT), '--scene', str(scene), '--output', '{output}/tests', *extra, *flags],
                       ['tests/suite_manifest.json', 'tests/generation.json', 'tests/suites/*.npz'], 'armplace_tests')
    if a.mode == 'prepare':
        scene = scene_file(a.scene)
        tests = a.tests.resolve()
        tests = tests / 'tests' if (tests / 'tests').is_dir() else tests
        assert json.loads((tests / 'generation.json').read_text())['complete']
        return spec_of('armplace-prepare', f'Arm placement prepare {a.scene}', dict(scene=a.scene, tests=sha(tests / 'suite_manifest.json'), export=sha(EXPORT)),
                       dict(scene=str(scene), scene_tests=str(tests / 'suites'), tests_manifest=str(tests / 'suite_manifest.json'), robot_model=str(EXPORT), **maze, **{k: v for k, v in real.items() if 'make_scenes' not in k}),
                       [py, str(snap / 'prep.py'), '--export', str(EXPORT), '--scene', str(scene), '--tests', str(tests), *extra, *flags, '--out', '{output}'],
                       ['results.json', 'scene.json', 'robot_model.npz', 'queries.npz', 'data/sampled_points.npy', 'data/speed.npy', 'data/normal.npy'], 'armplace_prepare')
    prep_files = ['scene.json', 'robot_model.npz', 'queries.npz']
    prepared = a.prepared.resolve()
    assert (prepared / 'results.json').exists(), '--prepared must be a finished prepare run'
    pin = lambda files: {'prepared_' + f.replace('/', '_'): str(prepared / f) for f in files}  # noqa: E731
    scene_name = json.loads((prepared / 'scene.json').read_text())['tests_scene']
    if a.mode == 'field':
        return spec_of('armplace-field', f'Arm placement field {scene_name} seed {a.seed}', dict(prepared=rel(prepared), seed=a.seed, scene=scene_name),
                       dict(**maze, **ntrl, **pin(prep_files + ['data/sampled_points.npy', 'data/speed.npy', 'data/normal.npy'])),
                       [py, str(snap / 'pipeline12.py'), '--prepared', str(prepared), '--seed', str(a.seed), *extra, *flags, '--out', '{output}'],
                       ['results.json', 'log.txt', 'training/checkpoint.pt', 'training/history.json', 'paths_seed0.npy'], 'armplace_field')
    if a.mode == 'euclid':
        return spec_of('armplace-euclid', f'Arm placement Euclidean-field planner control {scene_name}', dict(prepared=rel(prepared), scene=scene_name), dict(**maze, **ntrl, **pin(prep_files)),
                       [py, str(snap / 'euclid.py'), '--prepared', str(prepared), *extra, *flags, '--out', '{output}'], ['results.json', 'log.txt'], 'armplace_euclid')
    fields = {i.split('=', 1)[0]: Path(i.split('=', 1)[1].split(':')[0]).resolve() for i in a.fields}
    prefix = {i.split('=', 1)[0]: (i.split('=', 1)[1].split(':') + [''])[1] for i in a.fields}
    fin = {'field_' + k: str(v / 'training/checkpoint.pt') if (v / 'training/checkpoint.pt').exists() else str(v / 'results.json') for k, v in fields.items()}
    if a.mode == 'evalfield':
        return spec_of('armplace-evalfield', f'Arm placement planner settings on trained fields {scene_name}', dict(prepared=rel(prepared), scene=scene_name, fields={k: rel(v) for k, v in fields.items()}),
                       dict(**maze, **ntrl, **pin(prep_files), **fin),
                       [py, str(snap / 'evalfield.py'), '--prepared', str(prepared), '--fields', *[f'{k}={v}' for k, v in fields.items()], *extra, *flags, '--out', '{output}'],
                       ['results.json', 'log.txt'], 'armplace_evalfield')
    return spec_of('armplace-demo', f'Arm placement hard queries and 12-D paths {scene_name}', dict(prepared=rel(prepared), scene=scene_name, fields={k: rel(v) for k, v in fields.items()}),
                   dict(**maze, **ntrl, **pin(prep_files), **fin),
                   [py, str(snap / 'demo.py'), '--prepared', str(prepared), '--fields', *[f'{k}={v}' + (f':{prefix[k]}' if prefix[k] else '') for k, v in fields.items()], *extra, *flags, '--out', '{output}'],
                   ['results.json', 'log.txt', 'demo.npz'], 'armplace_demo')


if __name__ == '__main__':
    main()
