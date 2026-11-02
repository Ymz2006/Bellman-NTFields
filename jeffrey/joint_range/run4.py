"""Register joint-range Test 4 (outputs in results/registered_<id>/; jeffrey/joint_range/README.md, Test 4).

    python3 jeffrey/joint_range/run4.py data  --scene cabinet [--gpu N]
    python3 jeffrey/joint_range/run4.py train --scene cabinet --data-run registered_<data run> --set test4   --gpu 1
    python3 jeffrey/joint_range/run4.py train --scene cabinet --data-run registered_<data run> --set control --gpu 0
    python3 jeffrey/joint_range/run4.py cupdata --scene cabinet
    python3 jeffrey/joint_range/run4.py train --scene cabinet --data-run registered_<cupdata run> --set both|combo --gpu N

data   pipeline4.py data: 200k pairs, 1/3 collision pairs (x0 touches or is inside the cupboard or table) + mixed pairs
       (exactly one grasp point in the cupboard), as test4/; 200k mixed pairs as control/; held-out pairs; one worker per GPU
cupdata pipeline4.py cupdata: 200k pairs with both grasp points in the cupboard (cupboard_data.py: the mixed recipe with the
       membership test changed) as both/; 100k mixed + 100k both pairs as combo/; held-out both pairs
train  pipeline4.py train: ntrl-demo (experiments/ntrl_cabinet/train.py, models/metric, the baseline recipe) on one set of a
       data run of the same scene, then the cabinet protocol on that scene's frozen suites, every planner in release coordinates
A new runner beside run.py (Tests 1-3), so their declared inputs stay unchanged. The command is jeffrey/segment_dp/launch.py
(launch.py's mounts and builders, unedited). Sources, the ntrl-demo model files, scene, frozen suites and (train) the data
files are declared inputs.
"""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
from launch import RELEASE
from experiments.cabinet.run import scene_paths
from experiments.face_speed.run import environment

REGISTRY = HERE/'tools/experiment_registry/registry.py'
SOURCES = ['jeffrey/joint_range/collide_data.py', 'jeffrey/joint_range/cupboard_data.py', 'experiments/ntrl_cabinet/train.py', 'jeffrey/joint_range/pipeline4.py', 'jeffrey/joint_range/run4.py', 'experiments/face_speed/robot.py', 'jeffrey/joint_range/field.py',
           'jeffrey/joint_range/evaluate.py', 'jeffrey/segment_dp/launch.py', 'launch.py', 'experiments/ntrl_cabinet/data.py', 'experiments/ntrl_cabinet/train.py',
           'experiments/ntrl_cabinet/field.py', 'experiments/ntrl_cabinet/evaluate.py', 'experiments/cabinet/geometry.py', 'experiments/transition_wall/evaluate.py',
           'data_generation/source', 'method', 'runtime_manifest.json']
NTRL_FILES = ['models/data_mlp.py']+[f'models/metric/{f}' for f in ('model_train_metric.py', 'model_network_metric.py', 'model_function_metric.py')]
SET_FILES = ['complete.json', 'sampled_points.npy', 'speed.npy', 'normal.npy', 'clearance.npy', 'source.npy']


def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['data', 'cupdata', 'train']); p.add_argument('--scene', default='cabinet'); p.add_argument('--gpu')
    p.add_argument('--data-run'); p.add_argument('--set', choices=['test4', 'control', 'both', 'combo']); p.add_argument('--pairs', type=int, default=200000); p.add_argument('--seed', type=int, default=87358)
    p.add_argument('--log-eikonal', action='store_true'); p.add_argument('--smoke', action='store_true'); p.add_argument('--repeat-reason'); p.add_argument('--prepare-only', action='store_true'); a = p.parse_args()
    SCENE, TESTS = scene_paths(a.scene); gpus = 'all' if a.gpu is None else f'device={a.gpu}'
    inputs = dict(scene=str(SCENE), runtime=str(RELEASE), registry=str(REGISTRY), **{'source_'+s.replace('/', '_'): str(HERE/s) for s in SOURCES},
                  **{'ntrl_demo_'+f.replace('/', '_'): str(HERE/'baselines/ntrl-demo'/f) for f in NTRL_FILES})
    config = dict(action='joint-range-test4-'+a.mode, scene=a.scene, scene_sha256=hashlib.sha256(SCENE.read_bytes()).hexdigest(), gpus=gpus, smoke=a.smoke,
                  method='ntrl-demo metric NTFields (models/metric, baseline recipe)')
    args = [a.mode, '--scene', f'/clean/{SCENE.relative_to(HERE)}']
    if a.mode == 'data':
        n = 2048 if a.smoke else a.pairs
        config.update(pairs=n, heldout=256 if a.smoke else 20000, collision_share='round(pairs / 3)', data_seed=20261103, workers='one per visible GPU',
                      recipe='mixed: data.py in-area recipe kept when exactly one grasp point is in the cupboard; collision: x0 with obstacle distance <= 0 (cupboard or table), x1 the recipe step')
        args += ['--pairs', str(n)]
        required = ['status.json', 'data/complete.json']+[f'data/{s}/{f}' for s in ('test4', 'control', 'heldout') for f in SET_FILES]
        label = 'data'
    elif a.mode == 'cupdata':
        n = 2048 if a.smoke else a.pairs
        config.update(pairs=n, heldout=256 if a.smoke else 20000, combo='pairs / 2 mixed + pairs / 2 both', data_seed=20261104, workers='one per visible GPU',
                      recipe='both: the mixed recipe (data.py in-area x0 window, random step x1) kept when both grasp points are in the cupboard; mixed: exactly one')
        args += ['--pairs', str(n)]
        required = ['status.json', 'data/complete.json']+[f'data/{s}/{f}' for s in ('both', 'combo', 'heldout') for f in SET_FILES]
        label = 'cupdata'
    else:
        assert a.data_run and a.set, '--data-run and --set are required'
        if not (TESTS/'suite_manifest.json').exists(): raise SystemExit(f'Freeze the {a.scene} test suites into {TESTS} first (run.py cabinet-tests, freeze_tests.py).')
        run = HERE/'results'/Path(a.data_run).name; st = json.loads((run/'status.json').read_text())
        assert st['complete'] and st['mode'] in ('data', 'cupdata') and (a.smoke or not st['settings']['smoke']), run
        assert Path(st['settings']['scene']).name == SCENE.name, f'{run.name} holds data of {st["settings"]["scene"]}, not {SCENE.name}'
        assert (run/'data'/a.set/'complete.json').exists(), f'{run.name} has no {a.set} set'
        inputs.update(scene_tests=str(TESTS), **{f'data_{f}': str(run/'data'/a.set/f) for f in SET_FILES})
        config.update(data_run=run.name, set=a.set, train_seed=a.seed, epochs=5000, variant='metric', scaling='full', planning_seeds=[0] if a.smoke else [0, 1, 2], log_eikonal=a.log_eikonal,
                      planners='native RK4, release MPPI, ntrl-demo MPPI, all in release coordinates', scorer_tilt_deg=180)
        args += ['--tests', f'/clean/{TESTS.relative_to(HERE)}', '--data', f'/clean/{(run/"data").relative_to(HERE)}', '--set', a.set, '--seed', str(a.seed)]+(['--log-eikonal'] if a.log_eikonal else [])
        required = ['status.json', 'training/checkpoint.pt', 'training/progress.json', 'evaluation/evaluation.json', 'evaluation/REPORT.md', 'evaluation/scaling_check.json']
        label = f'{a.set} s{a.seed}'+(' log-eikonal' if a.log_eikonal else '')
    config['label'] = label
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    spec = dict(name=f'Clean Bellman600 joint-range test4 {a.mode} {a.scene} {label} {tag}'+(' smoke' if a.smoke else ' full'), kind='pipeline', root=str(HERE), config=config,
                environment=environment(), inputs=inputs,
                command=[sys.executable, str(HERE/'jeffrey/segment_dp/launch.py'), '--output', '{output}', '--gpus', gpus, '--', '/clean/jeffrey/joint_range/pipeline4.py', *args,
                         '--output', '/outputs']+(['--smoke'] if a.smoke else []),
                required_outputs=required, validation='files')
    directory = HERE/'specs'; directory.mkdir(exist_ok=True); path = directory/f'joint-range-test4_{a.mode}_{a.scene}_{tag}{"_smoke" if a.smoke else ""}.json'
    path.write_text(json.dumps(spec, indent=2)+'\n')
    if a.prepare_only: print(path); return
    check = subprocess.run([sys.executable, str(REGISTRY), 'check', str(path)], capture_output=True, text=True); (directory/(path.stem+'.check.json')).write_text(check.stdout)
    if check.returncode and not (check.returncode == 3 and a.repeat_reason): raise SystemExit(check.stderr or 'Duplicate experiment; supply a concrete --repeat-reason.')
    subprocess.run([sys.executable, str(REGISTRY), 'run', str(path)]+(['--repeat-reason', a.repeat_reason] if a.repeat_reason else []), check=True)


if __name__ == '__main__': main()
