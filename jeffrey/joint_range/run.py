"""Register the joint-range test (outputs in results/registered_<id>/; jeffrey/joint_range/README.md).

    python3 jeffrey/joint_range/run.py data --gpu 1
    python3 jeffrey/joint_range/run.py test --data-run registered_<data run> --scaling full  --seed 87358 --gpu 1
    python3 jeffrey/joint_range/run.py test --data-run registered_<data run> --scaling range --seed 87358 --gpu 0

data  pipeline.py data: 400k ntrl-demo cabinet pairs with both grasp points in the test area, as full/ and range/
cabdata pipeline.py cabdata: 200k in-cabinet pairs (x0 or x1 grasp point in the cupboard), one worker per GPU
mix   pipeline.py mix: --fraction of a cabdata run's pairs + the rest from a data run's in-area pairs, 200k total; train + evaluate
derive pipeline.py derive: Test 2b's coordinates (range of the collision-checked x0 of a data run)
test  pipeline.py test: ntrl-demo (experiments/ntrl_cabinet/train.py, models/metric, the baseline recipe) on one
      scaling of a data run, then the cabinet protocol with every planner in release coordinates
The command is jeffrey/segment_dp/launch.py (launch.py's mounts and builders, unedited). Sources, the ntrl-demo
model files, scene, frozen suites and (test) the data files are declared inputs.
"""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
from launch import RELEASE
from experiments.cabinet.run import scene_paths
from experiments.face_speed.run import environment

REGISTRY = HERE/'tools/experiment_registry/registry.py'
SOURCES = ['jeffrey/joint_range/data.py', 'jeffrey/joint_range/derive.py', 'jeffrey/joint_range/cabinet_data.py', 'jeffrey/joint_range/mix.py', 'experiments/face_speed/robot.py', 'jeffrey/joint_range/field.py', 'jeffrey/joint_range/evaluate.py', 'jeffrey/joint_range/pipeline.py', 'jeffrey/joint_range/run.py',
           'jeffrey/segment_dp/launch.py', 'launch.py', 'experiments/ntrl_cabinet/data.py', 'experiments/ntrl_cabinet/train.py', 'experiments/ntrl_cabinet/field.py',
           'experiments/ntrl_cabinet/evaluate.py', 'experiments/cabinet/geometry.py', 'experiments/transition_wall/evaluate.py', 'data_generation/source', 'method', 'runtime_manifest.json']
NTRL_FILES = ['models/data_mlp.py']+[f'models/metric/{f}' for f in ('model_train_metric.py', 'model_network_metric.py', 'model_function_metric.py')]


def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['data', 'derive', 'test', 'cabdata', 'mix']); p.add_argument('--scene', default='cabinet'); p.add_argument('--gpu')
    p.add_argument('--data-run'); p.add_argument('--cab-run'); p.add_argument('--fraction', type=float); p.add_argument('--pairs', type=int, default=200000); p.add_argument('--scaling', choices=['full', 'range', 'anchor']); p.add_argument('--seed', type=int, default=87358)
    p.add_argument('--smoke', action='store_true'); p.add_argument('--repeat-reason'); p.add_argument('--prepare-only', action='store_true'); a = p.parse_args()
    SCENE, TESTS = scene_paths(a.scene); gpus = 'all' if a.gpu is None else f'device={a.gpu}'
    inputs = dict(scene=str(SCENE), runtime=str(RELEASE), registry=str(REGISTRY), **{'source_'+s.replace('/', '_'): str(HERE/s) for s in SOURCES},
                  **{'ntrl_demo_'+f.replace('/', '_'): str(HERE/'baselines/ntrl-demo'/f) for f in NTRL_FILES})
    config = dict(action='joint-range-'+a.mode, scene=a.scene, scene_sha256=hashlib.sha256(SCENE.read_bytes()).hexdigest(), gpus=gpus, smoke=a.smoke,
                  method='ntrl-demo metric NTFields (models/metric, baseline recipe)')
    args = [a.mode, '--scene', f'/clean/{SCENE.relative_to(HERE)}']
    if a.mode == 'data':
        config.update(pairs=400000, recipe='experiments/ntrl_cabinet/data.py defaults + x1 grasp point in the sampling box', data_seed=20261019)
        required = ['status.json', 'data/range.json', 'data/full/complete.json', 'data/range/complete.json', 'data/full/sampled_points.npy', 'data/range/sampled_points.npy']
        label = 'data'
    elif a.mode == 'cabdata':
        n = 2048 if a.smoke else a.pairs
        config.update(pairs=n, recipe='data.py in-area recipe, kept when x0 or x1 grasp point is in the cupboard', data_seed=20261101, workers='one per visible GPU')
        args += ['--pairs', str(n)]; gpus = 'all' if a.gpu is None else gpus
        required = ['status.json', 'data/cabinet/complete.json', 'data/cabinet/sampled_points.npy', 'data/cabinet/speed.npy', 'data/cabinet/normal.npy', 'data/cabinet/clearance.npy']
        label = 'cabinet-data'
    elif a.mode == 'mix':
        assert a.cab_run and a.data_run and a.fraction is not None, '--cab-run, --data-run and --fraction are required'
        cab = HERE/'results'/Path(a.cab_run).name; area = HERE/'results'/Path(a.data_run).name
        for r, mode in ((cab, 'cabdata'), (area, 'data')):
            st = json.loads((r/'status.json').read_text()); assert st['complete'] and st['mode'] == mode and (a.smoke or not st['settings']['smoke']), r
        files = ['complete.json', 'sampled_points.npy', 'speed.npy', 'normal.npy', 'clearance.npy']; n = 2048 if a.smoke else a.pairs
        inputs.update(scene_tests=str(TESTS), **{f'cabinet_{f}': str(cab/'data/cabinet'/f) for f in files}, **{f'area_{f}': str(area/'data/full'/f) for f in files})
        config.update(cab_run=cab.name, area_run=area.name, fraction=a.fraction, pairs=n, mix_seed=20261102, train_seed=a.seed, epochs=5000, variant='metric', scaling='full',
                      planning_seeds=[0] if a.smoke else [0, 1, 2], planners='native RK4, release MPPI, ntrl-demo MPPI, all in release coordinates', scorer_tilt_deg=180)
        args += ['--tests', f'/clean/{TESTS.relative_to(HERE)}', '--data', f'/clean/{(area/"data/full").relative_to(HERE)}', '--cabinet', f'/clean/{(cab/"data/cabinet").relative_to(HERE)}',
                 '--fraction', f'{a.fraction:g}', '--pairs', str(n), '--seed', str(a.seed)]
        required = ['status.json', 'data/mix/complete.json', 'training/checkpoint.pt', 'training/progress.json', 'evaluation/evaluation.json', 'evaluation/REPORT.md', 'evaluation/scaling_check.json']
        label = f'mix{int(round(100*a.fraction)):03d} s{a.seed}'
    elif a.mode == 'derive':
        assert a.data_run, '--data-run (a data run) is required'
        run = HERE/'results'/Path(a.data_run).name; status = json.loads((run/'status.json').read_text()); assert status['complete'] and status['mode'] == 'data', run
        inputs.update(**{f'data_full_{f}': str(run/'data/full'/f) for f in ['complete.json', 'sampled_points.npy', 'speed.npy', 'normal.npy', 'clearance.npy']})
        config.update(data_run=run.name, fitted_on='x0 (collision-checked) only')
        args += ['--data', f'/clean/{(run/"data").relative_to(HERE)}']
        required = ['status.json', 'data/anchor_range.json', 'data/anchor/complete.json', 'data/anchor/sampled_points.npy']; label = 'derive-anchor'
    else:
        assert a.data_run and a.scaling, '--data-run and --scaling are required'
        run = HERE/'results'/Path(a.data_run).name; data = run/'data'/a.scaling
        status = json.loads((run/'status.json').read_text()); assert status['complete'] and status['mode'] == ('derive' if a.scaling == 'anchor' else 'data'), run
        assert a.smoke or not status['settings']['smoke'], 'full test needs a full data run'
        inputs.update(scene_tests=str(TESTS), **{f'data_{f}': str(data/f) for f in ['complete.json', 'sampled_points.npy', 'speed.npy', 'normal.npy']})
        config.update(data_run=run.name, scaling=a.scaling, train_seed=a.seed, epochs=5000, variant='metric', planning_seeds=[0] if a.smoke else [0, 1, 2],
                      planners='native RK4, release MPPI, ntrl-demo MPPI, all in release coordinates', scorer_tilt_deg=180)
        args += ['--tests', f'/clean/{TESTS.relative_to(HERE)}', '--data', f'/clean/{(run/"data").relative_to(HERE)}', '--scaling', a.scaling, '--seed', str(a.seed)]
        required = ['status.json', 'training/checkpoint.pt', 'training/progress.json', 'evaluation/evaluation.json', 'evaluation/REPORT.md', 'evaluation/scaling_check.json']
        label = f'{a.scaling} s{a.seed}'
    config['label'] = label
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    spec = dict(name=f'Clean Bellman600 joint-range {a.mode} {a.scene} {label} {tag}'+(' smoke' if a.smoke else ' full'), kind='pipeline', root=str(HERE), config=config,
                environment=environment(), inputs=inputs,
                command=[sys.executable, str(HERE/'jeffrey/segment_dp/launch.py'), '--output', '{output}', '--gpus', gpus, '--', '/clean/jeffrey/joint_range/pipeline.py', *args,
                         '--output', '/outputs']+(['--smoke'] if a.smoke else []),
                required_outputs=required, validation='files')
    directory = HERE/'specs'; directory.mkdir(exist_ok=True); path = directory/f'joint-range_{a.mode}_{a.scene}_{tag}{"_smoke" if a.smoke else ""}.json'
    path.write_text(json.dumps(spec, indent=2)+'\n')
    if a.prepare_only: print(path); return
    check = subprocess.run([sys.executable, str(REGISTRY), 'check', str(path)], capture_output=True, text=True); (directory/(path.stem+'.check.json')).write_text(check.stdout)
    if check.returncode and not (check.returncode == 3 and a.repeat_reason): raise SystemExit(check.stderr or 'Duplicate experiment; supply a concrete --repeat-reason.')
    subprocess.run([sys.executable, str(REGISTRY), 'run', str(path)]+(['--repeat-reason', a.repeat_reason] if a.repeat_reason else []), check=True)


if __name__ == '__main__': main()
