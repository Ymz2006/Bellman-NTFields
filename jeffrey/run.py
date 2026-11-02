"""Register jeffrey/ experiments with the repository's experiment registry (outputs in results/registered_<id>/).

    BELLMAN600_BACKEND=host ~/.venvs/bellman-ntfield/bin/python jeffrey/run.py guard --field bellman \
        --field-run registered_<id> --train-distance --variants '[{"name": "bi_k4", "args": ""}]' --gpu 1
    BELLMAN600_BACKEND=host ~/.venvs/bellman-ntfield/bin/python jeffrey/run.py sharp --pnt-run registered_<id> \
        --sharp-args "--generations 10" --eval-generations 0 5 10 --gpu 1

The spec's command is jeffrey/launch.py, which runs jeffrey/pipeline.py with launch.py's own mounts and command
builders, so the top-level run.py / launch.py (declared inputs of other sessions' in-flight runs) stay unedited.
Every source file, checkpoint, bank and scene used is a declared input; exact repeats are blocked as usual.
"""
import argparse, hashlib, json, shlex, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from launch import RELEASE
from experiments.cabinet.run import scene_paths
from experiments.face_speed.run import environment

REGISTRY = HERE/'tools/experiment_registry/registry.py'
SOURCES = ['jeffrey/__init__.py', 'jeffrey/geometry.py', 'jeffrey/distance.py', 'jeffrey/fields.py', 'jeffrey/planner.py', 'jeffrey/evaluate.py', 'jeffrey/bellman_sharp.py', 'jeffrey/sharp_diagnose.py', 'jeffrey/distance_guard.py', 'jeffrey/guard_only.py',
           'jeffrey/pipeline.py', 'jeffrey/launch.py', 'jeffrey/run.py', 'launch.py', 'experiments/face_speed/robot.py', 'experiments/point_field/point_field.py',
           'experiments/point_field/train.py', 'experiments/pntfields_cabinet/field.py', 'experiments/pntfields_cabinet/evaluate.py', 'experiments/ntrl_cabinet/field.py',
           'experiments/transition_wall/evaluate.py', 'baselines/P-NTFields/models/model_res_sigmoid.py', 'baselines/P-NTFields/models/data_mlp.py', 'method', 'runtime_manifest.json']
BANK = ['source_q.npy', 'successor_q.npy', 'safe.npy', 'manifest.json']


def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['guard', 'sharp']); p.add_argument('--scene', default='cabinet'); p.add_argument('--gpu')
    p.add_argument('--field', choices=['bellman', 'pnt', 'euclid'], default='bellman'); p.add_argument('--field-run', help='guard: registered run whose training/checkpoint.pt is the field')
    p.add_argument('--distance-run', help='guard: registered jeffrey guard run whose distance/distance.pt to reuse'); p.add_argument('--train-distance', action='store_true')
    p.add_argument('--sampled', default='registered_20261019T034213Z-f9c501f9', help='run whose sampled/ FK pool trains the distance model and whose prepared/bank Bellman-sharp uses')
    p.add_argument('--distance-args', default=''); p.add_argument('--variants', default='[]')
    p.add_argument('--pnt-run', help='sharp: registered pntfields-cabinet run whose training/checkpoint.pt is the pretrained field')
    p.add_argument('--sharp-args', default=''); p.add_argument('--eval-generations', type=int, nargs='*', default=[0, 10])
    p.add_argument('--smoke', action='store_true'); p.add_argument('--repeat-reason'); p.add_argument('--prepare-only', action='store_true'); a = p.parse_args()
    SCENE, TESTS = scene_paths(a.scene); gpus = 'all' if a.gpu is None else f'device={a.gpu}'
    res = lambda r: HERE/'results'/Path(r).name
    inputs = dict(scene=str(SCENE), scene_tests=str(TESTS), runtime=str(RELEASE), registry=str(REGISTRY), **{'source_'+s.replace('/', '_'): str(HERE/s) for s in SOURCES})
    args = [a.mode, '--scene', f'/clean/{SCENE.relative_to(HERE)}', '--tests', f'/clean/{TESTS.relative_to(HERE)}']
    if a.mode == 'guard':
        if a.field != 'euclid':
            ck = res(a.field_run)/'training/checkpoint.pt'; inputs['field_checkpoint'] = str(ck); args += ['--checkpoint', f'/clean/{ck.relative_to(HERE)}']
        args += ['--field', a.field, '--variants', a.variants]
        if a.train_distance:
            sampled = res(a.sampled)/'sampled'; args += ['--train-distance', '--sampled', f'/clean/{sampled.relative_to(HERE)}', '--distance-args', a.distance_args]
            for kind in ('original', 'augmentation'): inputs[f'pool_{kind}'] = str(sampled/'inputs'/kind/'sampled_points.npy')
        elif a.distance_run:
            d = res(a.distance_run)/'distance/distance.pt'; inputs['distance'] = str(d); args += ['--distance', f'/clean/{d.relative_to(HERE)}']
    else:
        ck = res(a.pnt_run)/'training/checkpoint.pt'; bank = res(a.sampled)/'prepared/bank'; inputs['pnt_checkpoint'] = str(ck)
        inputs.update({'bank_'+f: str(bank/f) for f in BANK})
        args += ['--checkpoint', f'/clean/{ck.relative_to(HERE)}', '--bank', f'/clean/{bank.relative_to(HERE)}', '--sharp-args', a.sharp_args, '--eval-generations', *map(str, a.eval_generations)]
    if a.smoke: args += ['--smoke']
    config = dict(action='jeffrey-'+a.mode, scene=a.scene, gpus=gpus, field=a.field if a.mode == 'guard' else 'pnt', field_run=a.field_run, distance_run=a.distance_run,
                  train_distance=a.train_distance, distance_args=a.distance_args, variants=json.loads(a.variants), pnt_run=a.pnt_run, sharp_args=a.sharp_args,
                  eval_generations=a.eval_generations, smoke=a.smoke, planning_seeds=[0] if a.smoke else [0, 1, 2], scorer_tilt_deg=180)
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    spec = dict(name=f'Clean Bellman600 jeffrey-{a.mode} {a.scene} {tag}'+(' smoke' if a.smoke else ' full'), kind='pipeline', root=str(HERE), config=config,
                environment=environment(), inputs=inputs, command=[sys.executable, str(HERE/'jeffrey/launch.py'), '--output', '{output}', '--gpus', gpus, '--', *args],
                required_outputs=['status.json'], validation='files')
    directory = HERE/'specs'; directory.mkdir(exist_ok=True); path = directory/f'jeffrey-{a.mode}_{a.scene}_{tag}{"_smoke" if a.smoke else ""}.json'
    path.write_text(json.dumps(spec, indent=2)+'\n')
    if a.prepare_only: print(path); return
    check = subprocess.run([sys.executable, str(REGISTRY), 'check', str(path)], capture_output=True, text=True); (directory/(path.stem+'.check.json')).write_text(check.stdout)
    if check.returncode and not (check.returncode == 3 and a.repeat_reason): raise SystemExit(check.stderr or 'Duplicate experiment; supply a concrete --repeat-reason.')
    command = [sys.executable, str(REGISTRY), 'run', str(path)] + (['--repeat-reason', a.repeat_reason] if a.repeat_reason else [])
    subprocess.run(command, check=True)


if __name__ == '__main__': main()
