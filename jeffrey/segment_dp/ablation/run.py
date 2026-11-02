"""Register loss-ablation runs of the original method with the repository's experiment registry (results/registered_<id>/).

    BELLMAN600_DERIVED_IMAGE=bellman600-env:local python3 jeffrey/segment_dp/ablation/run.py --scene cabinet --weights 6,0,2 --gpu 1

--scene nowall   the frozen assets (assets/bank, assets/speed.pt) and the frozen 600 queries
--scene cabinet  the prepared speed and bank of the registered cabinet pipeline run (--prepared-run) and the frozen
                 cabinet suites
--weights B,E,D  Bellman, Eikonal, direction weights (6,2,2 = the baseline); 5 rounds, fifth round evaluated

The spec's command is ../launch.py, which runs pipeline.py here with launch.py's own mounts and command builders.
Every source file, bank file and scene is a declared input.
"""
import argparse, hashlib, json, os, shlex, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HERE))
from launch import RELEASE, IMAGE, HOST_PYTHON
from experiments.cabinet.run import scene_paths

REGISTRY = HERE/'tools/experiment_registry/registry.py'
SOURCES = ['jeffrey/segment_dp/ablation/train.py', 'jeffrey/segment_dp/ablation/pipeline.py', 'jeffrey/segment_dp/ablation/run.py', 'jeffrey/segment_dp/launch.py',
           'launch.py', 'method', 'runtime_manifest.json', 'experiments/cabinet/evaluate.py', 'experiments/transition_wall/evaluate.py']
BANK = ['source_q.npy', 'successor_q.npy', 'local_cost.npy', 'safe.npy', 'action_step_size.npy', 'manifest.json']


def environment():
    if os.environ.get('BELLMAN600_BACKEND', 'docker') == 'host':
        return dict(backend='host-bwrap', python=HOST_PYTHON, pinned_image_id=IMAGE, same_image_as_pinned=False)
    image = os.environ.get('BELLMAN600_DERIVED_IMAGE', IMAGE)
    used = subprocess.run(['docker', '--context', 'default', 'image', 'inspect', '--format', '{{.Id}}', image], capture_output=True, text=True).stdout.strip()
    return dict(backend='docker', docker_context='default', docker_image=image, docker_image_id=used, pinned_image_id=IMAGE, same_image_as_pinned=used == IMAGE)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--scene', choices=['nowall', 'cabinet'], required=True); p.add_argument('--gpu')
    p.add_argument('--prepared-run', default='registered_20261019T034213Z-f9c501f9', help='cabinet: run whose prepared/ speed and bank to train on')
    p.add_argument('--weights', default='6,2,2'); p.add_argument('--seed', type=int, default=87358); p.add_argument('--label', default='')
    p.add_argument('--smoke', action='store_true'); p.add_argument('--repeat-reason'); p.add_argument('--prepare-only', action='store_true'); a = p.parse_args()
    gpus = 'all' if a.gpu is None else f'device={a.gpu}'
    weights = [float(x) for x in a.weights.split(',')]; assert len(weights) == 3
    inputs = dict(runtime=str(RELEASE), registry=str(REGISTRY), **{'source_'+s.replace('/', '_'): str(HERE/s) for s in SOURCES})
    if a.scene == 'nowall':
        prepared = HERE/'assets'; inputs['reference'] = str(HERE/'assets/reference'); args = ['--prepared', '/clean/assets']
    else:
        SCENE, TESTS = scene_paths('cabinet'); prepared = HERE/'results'/a.prepared_run/'prepared'
        inputs.update(scene=str(SCENE), scene_tests=str(TESTS))
        args = ['--prepared', f'/clean/{prepared.relative_to(HERE)}', '--scene', f'/clean/{SCENE.relative_to(HERE)}', '--tests', f'/clean/{TESTS.relative_to(HERE)}']
    inputs.update({'bank_'+f: str(prepared/'bank'/f) for f in BANK}, speed=str(prepared/'speed.pt'))
    args += ['--weights', ','.join(f'{w:g}' for w in weights), '--seed', str(a.seed)] + (['--smoke'] if a.smoke else [])
    config = dict(action='loss-ablation', scene=a.scene, prepared=str(prepared.relative_to(HERE)), gpus=gpus, rounds=5, epochs_per_round=2,
                  loss_weights=dict(zip(('bellman', 'eikonal', 'direction'), weights)), training_seed=a.seed, evaluated_round=5,
                  smoke=a.smoke, planning_seeds=[0] if a.smoke else [0, 1, 2], scorer_tilt_deg=180 if a.scene == 'cabinet' else 20)
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    label = a.label or 'w' + '-'.join(f'{w:g}' for w in weights) + ('' if a.seed == 87358 else f's{a.seed}')
    spec = dict(name=f'Clean Bellman600 loss-ablation {a.scene} {label} {tag}'+(' smoke' if a.smoke else ' full'), kind='pipeline', root=str(HERE), config=config,
                environment=environment(), inputs=inputs,
                command=[sys.executable, str(HERE/'jeffrey/segment_dp/launch.py'), '--output', '{output}', '--gpus', gpus, '--', '/clean/jeffrey/segment_dp/ablation/pipeline.py', *args, '--output', '/outputs'],
                required_outputs=['status.json', 'training/checkpoint.pt', 'evaluation/evaluation.json'], validation='files')
    directory = HERE/'specs'; directory.mkdir(exist_ok=True); path = directory/f'loss-ablation_{a.scene}_{tag}{"_smoke" if a.smoke else ""}.json'
    path.write_text(json.dumps(spec, indent=2)+'\n')
    if a.prepare_only: print(path); return
    check = subprocess.run([sys.executable, str(REGISTRY), 'check', str(path)], capture_output=True, text=True); (directory/(path.stem+'.check.json')).write_text(check.stdout)
    if check.returncode and not (check.returncode == 3 and a.repeat_reason): raise SystemExit(check.stderr or 'Duplicate experiment; supply a concrete --repeat-reason.')
    command = [sys.executable, str(REGISTRY), 'run', str(path)] + (['--repeat-reason', a.repeat_reason] if a.repeat_reason else [])
    subprocess.run(command, check=True)


if __name__ == '__main__': main()
