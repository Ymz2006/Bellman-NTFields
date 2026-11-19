"""Register an equal-runs fit with the repository's experiment registry.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/equal_runs/run.py --label cv07 --extra '--scene plain:10 --kind equal --form cv --target 0.07'

Outputs go to results/registered_<id>/. Each run executes a frozen copy of this study's sources (snap/<hash>/), so working files can change while runs are in flight. The maze modules are declared inputs.
No planner and no travel-time field here (the request: 'no need to even train the ntfield'). Nothing touches the frozen Bellman600 pipeline, assets or baseline.
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

DIR = 'jeffrey/equal_runs'
SOURCES = ['scene.py', 'maps.py', 'scan.py', 'losses.py', 'measure.py', 'view.py', 'fit.py', 'fitrun.py', 'speedfit.py', 'speedcal.py', 'carto.py', 'cartorun.py', 'nf.py', 'nfrun.py', 'remeasure.py', 'speedseeds.py']
OTHER = {'maze_maze.py': 'jeffrey/maze_scale/maze.py', 'maze_plus.py': 'jeffrey/maze_scale/plus.py', 'cross_gap.py': 'jeffrey/cross_warp/gap.py'}
REQUIRED = ['results.json', 'log.txt', 'psi.pt', 'history.json', 'overview.png']


def snapshot(src):
    h = hashlib.sha256()
    for s in SOURCES:
        h.update(s.encode() + b'\0' + (src / s).read_bytes())
    folder = HERE / DIR / 'snap' / h.hexdigest()[:12]
    if not folder.exists():
        tmp = folder.with_name(folder.name + f'.tmp{os.getpid()}')
        tmp.mkdir(parents=True, exist_ok=True)
        for s in SOURCES:
            shutil.copyfile(src / s, tmp / s)
        try:
            tmp.rename(folder)
        except OSError:
            shutil.rmtree(tmp)
    for s in SOURCES:
        assert (folder / s).read_bytes() == (src / s).read_bytes(), f'snapshot {folder} differs in {s}'
    return folder


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--label', required=True)
    p.add_argument('--extra', default='')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--repeat-reason')
    a = p.parse_args()
    py = sys.executable
    snap = snapshot(HERE / DIR)
    extra = a.extra.split()
    inputs = dict(registry=str(REGISTRY), **{'source_' + s: str(snap / s) for s in SOURCES}, **{k: str(HERE / v) for k, v in OTHER.items()})
    if 'arm:' in a.extra:
        name = a.extra.split('arm:')[1].split()[0]
        inputs.update({'arm_' + k: str(HERE / v) for k, v in dict(scene=f'experiments/cabinet/scenes/{name}.yml', spheres='runtime/vendor/planner/spheres_ur5e_arm.yml', robot='runtime/vendor/planner/ur5e_arm.yml', urdf='runtime/assets/runtime/robot/ur5e_robotiq_2f_140.urdf', geometry='data_generation/source/pkl_to_training_data_multi_ur5e.py', collision='data_generation/source/collision_utils.py').items()})
    if 'gibson:' in a.extra:
        inputs['gibson_grid.py'] = str(HERE / 'jeffrey/gibson_embed/grid.py')
        for i in ('0', '1'):
            inputs[f'gibson_mesh_{i}'] = str(HERE / f'baselines/P-NTFields/datasets/gibson/{i}/mesh_z_up_scaled.off')
    config = dict(action='equal-runs-fit', label=a.label, extra=' '.join(extra), smoke=a.smoke, code=snap.name)
    args = list(extra)
    if a.smoke:
        args.append('--smoke')
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    name = f'Equal runs fit {a.label} {tag}' + (' smoke' if a.smoke else '')
    script, required = 'fitrun.py', REQUIRED
    if a.label.startswith('carto_'):                     # the cartogram of a scene: no training
        script = 'cartorun.py'
        name = f'Equal runs cartogram {a.label} {tag}'
        required = ['results.json', 'log.txt', 'psi.pt', 'overview.png']
    if a.label.split('_')[0] in ('nf', 'nf2', 'nf3', 'nf4', 'nf5'):   # the cartogram as a normalizing flow
        script = 'nfrun.py'
        name = f'Equal runs flow {a.label} {tag}'
        required = ['results.json', 'log.txt', 'psi.pt', 'overview.png']
    if a.label.startswith('ss_'):                        # the score's own noise on the key maps
        script = 'speedseeds.py'
        name = f'Equal runs score noise {a.label} {tag}'
        required = ['results.json', 'log.txt']
        for flag, tmpl in (('--runs', 'results/registered_{}/psi.pt'), ('--old', 'results/registered_{}/phi.pt')):
            if flag in extra:
                for pair in extra[extra.index(flag) + 1].split(','):
                    rid = pair.split('=')[1]
                    inputs[f'run_{rid}'] = str(HERE / tmpl.format(rid))
        inputs.update({'line_warp_' + s: str(HERE / 'jeffrey/line_warp' / s) for s in ('netwarp.py', 'warpnet.py')})
        inputs['cross_warp.py'] = str(HERE / 'jeffrey/cross_warp/warp.py')
    if a.label.startswith('re_'):                        # every map of a scene through the final measures
        script = 'remeasure.py'
        name = f'Equal runs remeasure {a.label} {tag}'
        required = ['results.json', 'log.txt']
        i = extra.index('--runs') + 1
        for pair in extra[i].split(','):
            rid = pair.split('=')[1]
            inputs[f'run_{rid}_psi'] = str(HERE / 'results' / f'registered_{rid}' / 'psi.pt')
    if a.label.startswith('cal_'):                       # calibration of the learnability score on line_warp's maps
        script, required = 'speedcal.py', ['results.json', 'log.txt']
        name = f'Equal runs calibration {a.label} {tag}'
        i = extra.index('--fits') + 1
        for pair in extra[i].split(','):
            rid = pair.split('=')[1]
            inputs[f'fit_{rid}_phi'] = str(HERE / 'results' / f'registered_{rid}' / 'phi.pt')
        inputs.update({'line_warp_' + s: str(HERE / 'jeffrey/line_warp' / s) for s in ('netwarp.py', 'warpnet.py')})
        inputs['cross_warp.py'] = str(HERE / 'jeffrey/cross_warp/warp.py')
    spec = dict(name=name, kind='pipeline', root=str(HERE), config=config, environment=environment(), inputs=inputs, command=[py, str(snap / script), *args, '--out', '{output}'],
                required_outputs=required, validation='files')
    directory = HERE / 'specs'
    directory.mkdir(exist_ok=True)
    path = directory / f'equal_runs_{tag}{"_smoke" if a.smoke else ""}.json'
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
