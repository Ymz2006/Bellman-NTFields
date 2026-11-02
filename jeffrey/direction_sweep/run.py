"""Register direction-sweep runs with the repository's experiment registry (results/registered_<id>/).

Question: does a Bellman min over more random-walk directions make the original method's field more accurate and
the frozen planning benchmarks more successful?

    BELLMAN600_DERIVED_IMAGE=bellman600-env:local python3 jeffrey/direction_sweep/run.py extend --scene nowall --gpu 1
    BELLMAN600_DERIVED_IMAGE=bellman600-env:local python3 jeffrey/direction_sweep/run.py train --scene nowall --ext-run <extend run> --pairs 8 --gpu 1

extend   nested extension bank of the scene's frozen bank: same sources, frozen 4 direction pairs + new pairs up to --pairs
train    the original method (6/2/2, 5 rounds, 15,620 updates, fifth round evaluated) with the min over the first --pairs
         pairs; then the frozen benchmarks, the ground-truth fit and diag.py
--scene nowall   assets/bank + assets/speed.pt, the frozen 600 queries, truth run 20261027T181431Z-bf1c963f
--scene cabinet  the cabinet pipeline run's prepared/ (f9c501f9), the frozen cabinet suites, truth run 20261027T175501Z-5150664b
"""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
from launch import RELEASE, IMAGE
from experiments.cabinet.run import scene_paths

REGISTRY = HERE/'tools/experiment_registry/registry.py'
SOURCES = ['jeffrey/direction_sweep/extend.py', 'jeffrey/direction_sweep/train.py', 'jeffrey/direction_sweep/diag.py', 'jeffrey/direction_sweep/pipeline.py',
           'jeffrey/direction_sweep/run.py', 'jeffrey/segment_dp/launch.py', 'jeffrey/segment_dp/fit.py', 'jeffrey/segment_dp/fields.py', 'launch.py', 'method',
           'runtime_manifest.json', 'experiments/cabinet/evaluate.py', 'experiments/cabinet/geometry.py', 'experiments/no_wall_rebuild/geometry.py']
BANK = ['source_q.npy', 'successor_q.npy', 'local_cost.npy', 'safe.npy', 'in_domain.npy', 'minimum_speed.npy', 'manifest.json']
EXT = ['successor_q.npy', 'local_cost.npy', 'safe.npy', 'manifest.json']
TRUTH = dict(nowall='registered_20261027T181431Z-bf1c963f', cabinet='registered_20261027T175501Z-5150664b')


def environment():
    if os.environ.get('BELLMAN600_BACKEND', 'docker') == 'host':
        return dict(backend='host-bwrap', pinned_image_id=IMAGE, same_image_as_pinned=False)
    image = os.environ.get('BELLMAN600_DERIVED_IMAGE', IMAGE)
    used = subprocess.run(['docker', '--context', 'default', 'image', 'inspect', '--format', '{{.Id}}', image], capture_output=True, text=True).stdout.strip()
    return dict(backend='docker', docker_context='default', docker_image=image, docker_image_id=used, pinned_image_id=IMAGE, same_image_as_pinned=used == IMAGE)


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['extend', 'train']); p.add_argument('--scene', choices=['nowall', 'cabinet'], required=True)
    p.add_argument('--gpu'); p.add_argument('--prepared-run', default='registered_20261019T034213Z-f9c501f9')
    p.add_argument('--ext-run', help='train: the extend run of this scene'); p.add_argument('--pairs', type=int, required=True)
    p.add_argument('--seed', type=int, default=87358); p.add_argument('--smoke', action='store_true'); p.add_argument('--repeat-reason')
    p.add_argument('--prepare-only', action='store_true'); a = p.parse_args()
    gpus = 'all' if a.gpu is None else f'device={a.gpu}'
    inputs = dict(runtime=str(RELEASE), registry=str(REGISTRY), **{'source_'+s.replace('/', '_'): str(HERE/s) for s in SOURCES})
    if a.scene == 'nowall':
        prepared = HERE/'assets'; args = ['--prepared', '/clean/assets']
        if a.stage == 'train': inputs['reference'] = str(HERE/'assets/reference')
    else:
        SCENE, TESTS = scene_paths('cabinet'); prepared = HERE/'results'/a.prepared_run/'prepared'
        inputs.update(scene=str(SCENE), scene_tests=str(TESTS))
        args = ['--prepared', f'/clean/{prepared.relative_to(HERE)}', '--scene', f'/clean/{SCENE.relative_to(HERE)}', '--tests', f'/clean/{TESTS.relative_to(HERE)}']
    inputs.update({'bank_'+f: str(prepared/'bank'/f) for f in BANK}, speed=str(prepared/'speed.pt'))
    config = dict(action='direction-sweep-'+a.stage, scene=a.scene, prepared=str(prepared.relative_to(HERE)), gpus=gpus, smoke=a.smoke)
    if a.stage == 'extend':
        config.update(pairs=a.pairs, frozen_pairs=4, extension_seed=20261028)
        required = ['status.json', 'ext/manifest.json'] + ['ext/'+f for f in EXT]
    else:
        assert a.ext_run, '--ext-run is required for train'
        ext = HERE/'results'/a.ext_run; truth = HERE/'results'/TRUTH[a.scene]/'truth/truth.npz'
        inputs.update({'ext_'+f: str(ext/'ext'/f) for f in EXT}, truth=str(truth))
        args += ['--ext', f'/clean/results/{a.ext_run}', '--truth', f'/clean/results/{TRUTH[a.scene]}/truth/truth.npz', '--seed', str(a.seed)]
        config.update(ext_run=a.ext_run, pairs=a.pairs, actions=8*a.pairs, rounds=5, epochs_per_round=2, loss_weights=dict(bellman=6, eikonal=2, direction=2),
                      training_seed=a.seed, evaluated_round=5, truth_run=TRUTH[a.scene], planning_seeds=[0] if a.smoke else [0, 1, 2],
                      scorer_tilt_deg=180 if a.scene == 'cabinet' else 20)
        required = ['status.json', 'training/checkpoint.pt', 'evaluation/evaluation.json', 'fit.json', 'diag.json']
    args += ['--pairs', str(a.pairs)] + (['--smoke'] if a.smoke else [])
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    label = f'{a.stage} D{a.pairs}' + ('' if a.seed == 87358 or a.stage == 'extend' else f' s{a.seed}')
    spec = dict(name=f'Clean Bellman600 direction-sweep {a.scene} {label} {tag}'+(' smoke' if a.smoke else ' full'), kind='pipeline', root=str(HERE), config=config,
                environment=environment(), inputs=inputs,
                command=[sys.executable, str(HERE/'jeffrey/segment_dp/launch.py'), '--output', '{output}', '--gpus', gpus, '--',
                         '/clean/jeffrey/direction_sweep/pipeline.py', '--stage', a.stage, *args, '--output', '/outputs'],
                required_outputs=required, validation='files')
    directory = HERE/'specs'; directory.mkdir(exist_ok=True); path = directory/f'direction-sweep_{a.scene}_{a.stage}_{tag}{"_smoke" if a.smoke else ""}.json'
    path.write_text(json.dumps(spec, indent=2)+'\n')
    if a.prepare_only: print(path); return
    check = subprocess.run([sys.executable, str(REGISTRY), 'check', str(path)], capture_output=True, text=True); (directory/(path.stem+'.check.json')).write_text(check.stdout)
    if check.returncode and not (check.returncode == 3 and a.repeat_reason): raise SystemExit(check.stderr or 'Duplicate experiment; supply a concrete --repeat-reason.')
    command = [sys.executable, str(REGISTRY), 'run', str(path)] + (['--repeat-reason', a.repeat_reason] if a.repeat_reason else [])
    subprocess.run(command, check=True)


if __name__ == '__main__': main()
