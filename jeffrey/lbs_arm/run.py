"""Register the LBS cabinet-arm runs with the repository's experiment registry (outputs in results/registered_<id>/).

    P=~/.venvs/bellman-ntfield/bin/python; export CUDA_DEVICE_ORDER=PCI_BUS_ID BELLMAN600_BACKEND=host
    $P jeffrey/lbs_arm/run.py export  --scene cabinet --gpu 1                    # release collision model (container layout)
    CUDA_VISIBLE_DEVICES=1 $P jeffrey/lbs_arm/run.py prepare --export results/registered_<export> --scene cabinet
    CUDA_VISIBLE_DEVICES=1 $P jeffrey/lbs_arm/run.py field --prepared results/registered_<prepare> --seed 87358 --label lbs --extra "--lift time --bell 1"
    $P jeffrey/lbs_arm/run.py eval --field results/registered_<field> --gpu 1    # the cabinet protocol of the baselines (container layout)

export and eval run in the container layout through jeffrey/segment_dp/launch.py (bubblewrap on this workstation); prepare and field run
in the host venv. Every run executes a frozen copy of this folder's sources (snap/<hash>/), so the working files can change while runs
are in flight. Nothing here touches the frozen Bellman600 pipeline, assets or baseline.
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
SOURCES = ['arm.py', 'core.py', 'trainer6.py', 'pipeline6.py', 'prepare.py', 'export_robot.py', 'evaluate6.py', 'fastnet.py', 'tree.py']
COPIES = {'fastnet.py': 'jeffrey/lbs/fastnet.py', 'tree.py': 'jeffrey/lbs/tree.py'}
MAZE = ['maze.py', 'common.py']
NTRL = ['models/metric/model_train_metric.py', 'models/metric/model_network_metric.py', 'models/metric/model_function_metric.py', 'models/data_mlp.py']
CONTAINER = ['jeffrey/segment_dp/launch.py', 'launch.py', 'experiments/ntrl_cabinet/data.py', 'experiments/ntrl_cabinet/field.py', 'experiments/ntrl_cabinet/evaluate.py',
             'experiments/cabinet/geometry.py', 'experiments/transition_wall/evaluate.py', 'data_generation/source', 'method', 'runtime_manifest.json']


def snapshot(src):
    for name, orig in COPIES.items():
        assert (src / name).read_bytes() == (HERE / orig).read_bytes(), f'{name} is no longer a verbatim copy of {orig}'
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


def container_environment():
    from launch import IMAGE, HOST_PYTHON
    if os.environ.get('BELLMAN600_BACKEND', 'docker') == 'host':
        return dict(backend='host-bwrap', python=HOST_PYTHON, pinned_image_id=IMAGE, same_image_as_pinned=False)
    return dict(docker_context='default', docker_image_id=IMAGE)


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
    """Scene file and frozen suites: the cabinet scenes of experiments/cabinet, or this study's own (scenes/<name>.yml, tests_<name>/)."""
    if scene.startswith('real_world'):
        assert (HERE / DIR / 'scenes' / (scene + '.yml')).is_file(), scene
        return HERE / DIR / 'scenes' / (scene + '.yml'), HERE / DIR / ('tests_' + scene)
    exp = HERE / 'experiments/cabinet'
    assert scene.startswith('cabinet') and (exp / 'scenes' / (scene + '.yml')).is_file(), scene
    return exp / 'scenes' / (scene + '.yml'), exp / ('tests' + scene[len('cabinet'):])


def freeze(run):
    """Copy a finished tests run into tests_<scene>/ (once; never overwrites), checking every hash."""
    src = run / 'tests'
    man = json.loads((src / 'suite_manifest.json').read_text())
    gen = json.loads((src / 'generation.json').read_text())
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()  # noqa: E731
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
               generator='jeffrey/lbs_arm/make_tests_rw.py', generator_sha256=sha(HERE / DIR / 'make_tests_rw.py'),
               note='Frozen testing data: never resample or edit.')
    (target / 'FROZEN.json').write_text(json.dumps(rec, indent=2) + '\n')
    print(json.dumps(rec, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['export', 'prepare', 'field', 'eval', 'tests', 'freeze'])
    p.add_argument('--run', type=Path, help='freeze: a finished tests run')
    p.add_argument('--scene', default='cabinet')
    p.add_argument('--export', type=Path)
    p.add_argument('--prepared', type=Path)
    p.add_argument('--field', type=Path)
    p.add_argument('--seed', type=int, default=87358)
    p.add_argument('--label', default='v1')
    p.add_argument('--extra', default='')
    p.add_argument('--gpu')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--repeat-reason')
    a = p.parse_args()
    if a.mode == 'freeze':
        return freeze(a.run.resolve())
    py = sys.executable
    snap = snapshot(HERE / DIR)
    csnap = '/clean/' + rel(snap)
    extra = a.extra.split()
    src = {'source_' + s: str(snap / s) for s in SOURCES}
    maze = {'maze_' + s: str(HERE / 'jeffrey/maze_scale' / s) for s in MAZE}
    ntrl = {'ntrl_' + s.replace('/', '_'): str(HERE / 'baselines/ntrl-demo' / s) for s in NTRL}
    cont = {'container_' + s.replace('/', '_'): str(HERE / s) for s in CONTAINER}
    gpus = 'all' if a.gpu is None else f'device={a.gpu}'
    launcher = [py, str(HERE / 'jeffrey/segment_dp/launch.py'), '--output', '{output}', '--gpus', gpus, '--']
    if a.mode == 'tests':
        SCENE, _ = scene_paths(a.scene)
        from launch import RELEASE
        gen = HERE / DIR / 'make_tests_rw.py'
        config = dict(action='lbsarm-tests', scene=a.scene, scene_sha256=hashlib.sha256(SCENE.read_bytes()).hexdigest(), label=a.label, smoke=a.smoke,
                      per_stratum=3 if a.smoke else 100, seeds=dict(pool=20261014, pair=20261015, rrt=20261016, select=20261017), certificate='release scorer, tilt gate off')
        tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
        spec = dict(name=f'LBS arm tests {a.scene} {a.label} {tag}' + (' smoke' if a.smoke else ''), kind='pipeline', root=str(HERE), config=config, environment=container_environment(),
                    inputs=dict(registry=str(REGISTRY), scene=str(SCENE), runtime=str(RELEASE), generator=str(gen), cabinet_generator=str(HERE / 'experiments/cabinet/make_tests.py'),
                                obstacle_generator=str(HERE / 'experiments/obstacle_scenes/make_tests.py'), **cont),
                    command=launcher + [f'/clean/{rel(gen)}', '--scene', f'/clean/{rel(SCENE)}', '--output', '/outputs/tests'] + (['--smoke'] if a.smoke else []),
                    required_outputs=['tests/suite_manifest.json', 'tests/generation.json', 'tests/suites/*.npz'], validation='files')
        return launch(spec, tag, a, py, 'lbsarm_tests')
    if a.mode == 'export':
        SCENE, _ = scene_paths(a.scene)
        from launch import RELEASE
        config = dict(action='lbsarm-export', scene=a.scene, scene_sha256=hashlib.sha256(SCENE.read_bytes()).hexdigest(), label=a.label, code=snap.name, smoke=a.smoke)
        tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
        spec = dict(name=f'LBS arm export {a.scene} {a.label} {tag}', kind='pipeline', root=str(HERE), config=config, environment=container_environment(),
                    inputs=dict(registry=str(REGISTRY), scene=str(SCENE), runtime=str(RELEASE), **src, **cont),
                    command=launcher + [f'{csnap}/export_robot.py', '--scene', f'/clean/{rel(SCENE)}', '--output', '/outputs'],
                    required_outputs=['robot_model.npz', 'export.json'], validation='files')
        return launch(spec, tag, a, py, 'lbsarm_export')
    if a.mode == 'prepare':
        SCENE, TESTS = scene_paths(a.scene)
        ex = a.export.resolve()
        assert json.loads((ex / 'export.json').read_text())['complete']
        config = dict(action='lbsarm-prepare', scene=a.scene, export=rel(ex), label=a.label, extra=' '.join(extra), code=snap.name, smoke=a.smoke)
        tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
        spec = dict(name=f'LBS arm prepare {a.scene} {a.label} {tag}' + (' smoke' if a.smoke else ''), kind='pipeline', root=str(HERE), config=config, environment=environment(),
                    inputs=dict(registry=str(REGISTRY), scene=str(SCENE), scene_tests=str(TESTS), robot_model=str(ex / 'robot_model.npz'), **src, **maze),
                    command=[py, str(snap / 'prepare.py'), '--export', str(ex), '--tests', str(TESTS), *extra, *(['--smoke'] if a.smoke else []), '--out', '{output}'],
                    required_outputs=['results.json', 'scene.json', 'robot_model.npz', 'queries.npz', 'data/sampled_points.npy', 'data/speed.npy', 'data/normal.npy'], validation='files')
        return launch(spec, tag, a, py, 'lbsarm_prepare')
    prep_files = ['scene.json', 'robot_model.npz', 'queries.npz', 'data/sampled_points.npy', 'data/speed.npy', 'data/normal.npy']
    if a.mode == 'field':
        prepared = a.prepared.resolve()
        assert (prepared / 'results.json').exists(), '--prepared must be a finished prepare run'
        config = dict(action='lbsarm-field', label=a.label, extra=' '.join(extra), smoke=a.smoke, code=snap.name, prepared=rel(prepared), seed=a.seed)
        tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
        spec = dict(name=f'LBS arm field seed {a.seed} {a.label} {tag}' + (' smoke' if a.smoke else ''), kind='pipeline', root=str(HERE), config=config, environment=environment(),
                    inputs=dict(registry=str(REGISTRY), **src, **maze, **ntrl, **{'prepared_' + f.replace('/', '_'): str(prepared / f) for f in prep_files}),
                    command=[py, str(snap / 'pipeline6.py'), '--prepared', str(prepared), '--seed', str(a.seed), *extra, *(['--smoke'] if a.smoke else []), '--out', '{output}'],
                    required_outputs=['results.json', 'log.txt', 'training/checkpoint.pt', 'training/history.json', 'paths_seed0.npy'], validation='files')
        return launch(spec, tag, a, py, 'lbsarm_field')
    field = a.field.resolve()
    fr = json.loads((field / 'results.json').read_text())
    prepared = Path(fr['prepared']).resolve()
    tests_scene = json.loads((prepared / 'scene.json').read_text())['tests_scene']
    SCENE, TESTS = scene_paths(tests_scene)
    from launch import RELEASE
    config = dict(action='lbsarm-eval', field=rel(field), label=a.label, code=snap.name, smoke=a.smoke, scene=tests_scene, planner=fr['planner'])
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    spec = dict(name=f'LBS arm eval {field.name[-8:]} {a.label} {tag}' + (' smoke' if a.smoke else ''), kind='pipeline', root=str(HERE), config=config, environment=container_environment(),
                inputs=dict(registry=str(REGISTRY), scene=str(SCENE), scene_tests=str(TESTS), runtime=str(RELEASE), checkpoint=str(field / 'training/checkpoint.pt'),
                            field_results=str(field / 'results.json'), **src, **maze, **cont, **{'ntrl_' + NTRL[1].replace('/', '_'): str(HERE / 'baselines/ntrl-demo' / NTRL[1])},
                            **{'prepared_' + f.replace('/', '_'): str(prepared / f) for f in prep_files[:2]}),
                command=launcher + [f'{csnap}/evaluate6.py', '--field', f'/clean/{rel(field)}', '--prepared', f'/clean/{rel(prepared)}', '--scene', f'/clean/{rel(SCENE)}',
                                    '--tests', f'/clean/{rel(TESTS)}', '--output', '/outputs'] + (['--smoke'] if a.smoke else []),
                required_outputs=['results.json', 'evaluation/evaluation.json', 'evaluation/REPORT.md'], validation='files')
    launch(spec, tag, a, py, 'lbsarm_eval')


if __name__ == '__main__':
    main()
