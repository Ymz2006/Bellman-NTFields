"""Register a mobile-arm run (stage 'prepare' or 'field') with the repository's experiment registry.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/mobile_arm/run.py prepare \
        --house results/registered_<lift_scale house prepare>
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ~/.venvs/bellman-ntfield/bin/python jeffrey/mobile_arm/run.py field \
        --prepared results/registered_<mobile_arm prepare> --seed 87358 --label mid --extra '--weight 1'

Outputs go to results/registered_<id>/. Each run executes a frozen copy of this study's sources (snap/<hash>/), so the working files can
change while runs are in flight. fastnet.py and tree.py are verbatim copies of jeffrey/midpoint_dp's (checked byte for byte here).
lift_scale's scenes.py, embed_warp's scene.py, maze_scale's maze / truth / common, gibson_embed's grid.py, ntrl-demo's model files, the
UR5e URDF and sphere file, the Gibson mesh and the prepared files are declared inputs. Nothing here touches the frozen Bellman600
pipeline, assets or baseline.
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

DIR = 'jeffrey/mobile_arm'
OWN = ['robot.py', 'scene9.py', 'data9.py', 'plan9.py', 'prepare.py', 'traverse9.py', 'trainer9.py', 'pipeline9.py']
COPIES = {'fastnet.py': 'jeffrey/midpoint_dp/fastnet.py', 'tree.py': 'jeffrey/midpoint_dp/tree.py'}
SOURCES = OWN + list(COPIES)
DEPS = {'lift_scenes': 'jeffrey/lift_scale/scenes.py', 'embed_scene': 'jeffrey/embed_warp/scene.py', 'maze_maze': 'jeffrey/maze_scale/maze.py',
        'maze_truth': 'jeffrey/maze_scale/truth.py', 'maze_common': 'jeffrey/maze_scale/common.py', 'maze_plus': 'jeffrey/maze_scale/plus.py',
        'cross_gap': 'jeffrey/cross_warp/gap.py', 'gibson_grid': 'jeffrey/gibson_embed/grid.py',
        'ur5e_urdf': 'runtime/assets/runtime/robot/ur5e.urdf', 'ur5e_spheres': 'runtime/vendor/planner/spheres_ur5e_arm.yml'}
NTRL = ['models/metric/model_train_metric.py', 'models/metric/model_network_metric.py', 'models/metric/model_function_metric.py', 'models/data_mlp.py']
HOUSE_FILES = ['maze.json', 'goals.npy', 'truth_fields.npy', 'results.json']
PREPARED = ['maze.json', 'queries.npz', 'ref_paths.npy', 'results.json', 'data', 'heldout']
REQUIRED = {'prepare': ['results.json', 'log.txt', 'maze.json', 'queries.npz', 'ref_paths.npy', 'data/sampled_points.npy', 'heldout/sampled_points.npy'],
            'field': ['results.json', 'log.txt', 'rows.npz', 'training/checkpoint.pt', 'training/history.json', 'paths_seed0.npy']}


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare', 'field'])
    p.add_argument('--house', type=Path, help='prepare: registered lift_scale house prepare run')
    p.add_argument('--prepared', type=Path, help='field: registered mobile_arm prepare run')
    p.add_argument('--seed', type=int, default=87358)
    p.add_argument('--label', default='')
    p.add_argument('--extra', default='')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--repeat-reason')
    a = p.parse_args()
    py = sys.executable
    snap = snapshot(HERE / DIR)
    extra = a.extra.split()
    inputs = dict(registry=str(REGISTRY), **{'source_' + s: str(snap / s) for s in SOURCES}, **{k: str(HERE / v) for k, v in DEPS.items()},
                  **{'ntrl_' + s.replace('/', '_'): str(HERE / 'baselines/ntrl-demo' / s) for s in NTRL})
    if a.stage == 'prepare':
        house = a.house.resolve()
        assert (house / 'results.json').exists(), '--house must be a finished lift_scale house prepare run'
        info = json.load(open(house / 'maze.json'))
        name_h = info['scene']['name']
        inputs.update({'house_' + f: str(house / f) for f in HOUSE_FILES})
        home = Path(os.environ.get('GIBSON_TINY', str(Path.home() / 'Downloads/gibson_tiny')))
        inputs['mesh'] = str(home / name_h / 'mesh_z_up.obj')
        config = dict(action='mobile-arm-prepare', house=rel(house), name=name_h, code=snap.name, extra=' '.join(extra), smoke=a.smoke)
        args = ['--house', str(house)] + extra + (['--smoke'] if a.smoke else [])
        script = 'prepare.py'
        tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
        name = f'Mobile arm prepare {name_h} {tag}' + (' smoke' if a.smoke else '')
    else:
        prepared = a.prepared.resolve()
        assert (prepared / 'results.json').exists(), '--prepared must be a finished mobile_arm prepare run'
        info = json.load(open(prepared / 'maze.json'))
        name_h = info['scene']['house']['name']
        inputs.update({'prepared_' + f.replace('/', '_'): str(prepared / f) for f in PREPARED})
        hp = Path(info['house_prepare'])
        inputs.update({'house_' + f: str(hp / f) for f in ('maze.json',)})
        home = Path(os.environ.get('GIBSON_TINY', str(Path.home() / 'Downloads/gibson_tiny')))
        inputs['mesh'] = str(home / name_h / 'mesh_z_up.obj')
        config = dict(action='mobile-arm-field', label=a.label, extra=' '.join(extra), smoke=a.smoke, code=snap.name, house=name_h, prepared=rel(prepared), seed=a.seed)
        args = ['--prepared', str(prepared), '--seed', str(a.seed)] + extra + (['--smoke'] if a.smoke else [])
        if '--resume' in extra:
            inputs['resume'] = str(Path(extra[extra.index('--resume') + 1]).resolve())
        script = 'pipeline9.py'
        tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
        name = f'Mobile arm field {name_h} seed {a.seed} {a.label} {tag}' + (' smoke' if a.smoke else '')
    spec = dict(name=name, kind='pipeline', root=str(HERE), config=config, environment=environment(), inputs=inputs,
                command=[py, str(snap / script), *args, '--out', '{output}'], required_outputs=REQUIRED[a.stage], validation='files')
    directory = HERE / 'specs'
    directory.mkdir(exist_ok=True)
    path = directory / f'mobile_arm_{a.stage}_{tag}{"_smoke" if a.smoke else ""}.json'
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
