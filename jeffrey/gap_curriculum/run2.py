"""run.py for training2.py (--split decomposition: which loss terms see the curriculum goals). Register gap-curriculum runs with the repository's experiment registry (outputs in results/registered_<id>/).

    BELLMAN600_BACKEND=host ~/.venvs/bellman-ntfield/bin/python jeffrey/gap_curriculum/run.py --scene nowall --rounds 10 --gpu 0
    BELLMAN600_BACKEND=host ~/.venvs/bellman-ntfield/bin/python jeffrey/gap_curriculum/run.py --scene cabinet --rounds 10 \
        --goals ball --radii linear --gpu 1

--scene nowall   the frozen assets (assets/bank, assets/speed.pt) and the frozen 600 queries
--scene cabinet  the prepared speed and bank of the registered cabinet pipeline run (--prepared-run) and the frozen
                 cabinet suites
--radii          linear (r_k = 2 k / rounds), linear:R (r_k = R k / rounds), geometric:A:B, or a comma-separated list, one per round

The spec's command is launch.py here, which runs pipeline.py with launch.py's own mounts and command builders, so
the top-level run.py / launch.py stay unedited. Every source file, bank file and scene is a declared input.
"""
import argparse, hashlib, json, shlex, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
from launch import RELEASE
from experiments.cabinet.run import scene_paths
from experiments.face_speed.run import environment

REGISTRY = HERE/'tools/experiment_registry/registry.py'
SOURCES = ['jeffrey/gap_curriculum/training.py', 'jeffrey/gap_curriculum/training2.py', 'jeffrey/gap_curriculum/diagnose.py', 'jeffrey/gap_curriculum/pipeline2.py', 'jeffrey/gap_curriculum/launch2.py',
           'jeffrey/gap_curriculum/run2.py', 'launch.py', 'method', 'runtime_manifest.json', 'experiments/cabinet/evaluate.py', 'experiments/transition_wall/evaluate.py']
BANK = ['source_q.npy', 'successor_q.npy', 'local_cost.npy', 'safe.npy', 'action_step_size.npy', 'manifest.json']


def schedule(spec, rounds):
    if spec == 'linear' or spec.startswith('linear:'):
        top = float(spec.split(':')[1]) if ':' in spec else 2.0
        return [round(top*(k+1)/rounds, 6) for k in range(rounds)]
    if spec.startswith('geometric:'):
        lo, hi = map(float, spec.split(':')[1:])
        return [round(lo*(hi/lo)**(k/(rounds-1)), 6) for k in range(rounds)]
    return [float(x) for x in spec.split(',')]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--scene', choices=['nowall', 'cabinet'], required=True); p.add_argument('--gpu')
    p.add_argument('--prepared-run', default='registered_20261019T034213Z-f9c501f9', help='cabinet: run whose prepared/ speed and bank to train on')
    p.add_argument('--rounds', type=int, default=10); p.add_argument('--epochs-per-round', type=int, default=2)
    p.add_argument('--goals', choices=['natural', 'ball', 'shell', 'value'], default='natural'); p.add_argument('--radii', default='')
    p.add_argument('--pool-subset', type=int, default=32768, help='ball goals: random pool subset looked in first (same distribution; 0 = whole pool)')
    p.add_argument('--mix', action='store_true'); p.add_argument('--grow-mix', action='store_true'); p.add_argument('--min-step', type=float, default=0.0); p.add_argument('--split', choices=['none', 'eikonal', 'bellman'], default='none'); p.add_argument('--seed', type=int, default=87358); p.add_argument('--label', default='')
    p.add_argument('--smoke', action='store_true'); p.add_argument('--repeat-reason'); p.add_argument('--prepare-only', action='store_true'); a = p.parse_args()
    gpus = 'all' if a.gpu is None else f'device={a.gpu}'
    radii = schedule(a.radii, a.rounds) if a.radii else []
    train = ['--rounds', str(a.rounds), '--epochs-per-round', str(a.epochs_per_round), '--goals', a.goals, '--seed', str(a.seed)]
    if radii: train += ['--radii', ','.join(map(str, radii))]
    if a.mix: train += ['--mix']
    if a.grow_mix: train += ['--grow-mix']
    if a.min_step: train += ['--min-step', str(a.min_step)]
    if a.split != 'none': train += ['--split', a.split]
    subset = a.pool_subset if a.goals == 'ball' else 0
    if subset: train += ['--pool-subset', str(subset)]
    inputs = dict(runtime=str(RELEASE), registry=str(REGISTRY), **{'source_'+s.replace('/', '_'): str(HERE/s) for s in SOURCES})
    if a.scene == 'nowall':
        prepared = HERE/'assets'; inputs['reference'] = str(HERE/'assets/reference'); args = ['--prepared', '/clean/assets']
    else:
        SCENE, TESTS = scene_paths('cabinet'); prepared = HERE/'results'/a.prepared_run/'prepared'
        inputs.update(scene=str(SCENE), scene_tests=str(TESTS))
        args = ['--prepared', f'/clean/{prepared.relative_to(HERE)}', '--scene', f'/clean/{SCENE.relative_to(HERE)}', '--tests', f'/clean/{TESTS.relative_to(HERE)}']
    inputs.update({'bank_'+f: str(prepared/'bank'/f) for f in BANK}, speed=str(prepared/'speed.pt'))
    args += [f'--train-args={shlex.join(train)}'] + (['--smoke'] if a.smoke else [])
    config = dict(action='gap-curriculum', scene=a.scene, prepared=str(prepared.relative_to(HERE)), gpus=gpus, rounds=a.rounds, epochs_per_round=a.epochs_per_round,
                  goals=a.goals, radii=radii, mix=a.mix, grow_mix=a.grow_mix, min_step=a.min_step, split=a.split, trainer='training2.py', pool_subset=subset, training_seed=a.seed, loss_weights={'bellman': 6, 'eikonal': 2, 'direction': 2}, evaluated_round=a.rounds,
                  smoke=a.smoke, planning_seeds=[0] if a.smoke else [0, 1, 2], scorer_tilt_deg=180 if a.scene == 'cabinet' else 20)
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    label = a.label or f"{a.goals}{'-mix' if a.mix else ''}{'-growmix' if a.grow_mix else ''}{f'-minstep{a.min_step:g}' if a.min_step else ''}{f'-split-{a.split}' if a.split != 'none' else ''} r{a.rounds}"
    spec = dict(name=f'Clean Bellman600 gap-curriculum {a.scene} {label} {tag}'+(' smoke' if a.smoke else ' full'), kind='pipeline', root=str(HERE), config=config,
                environment=environment(), inputs=inputs, command=[sys.executable, str(HERE/'jeffrey/gap_curriculum/launch2.py'), '--output', '{output}', '--gpus', gpus, '--', *args],
                required_outputs=['status.json', 'training/checkpoint.pt', 'evaluation/evaluation.json', 'diagnostics.json'], validation='files')
    directory = HERE/'specs'; directory.mkdir(exist_ok=True); path = directory/f'gap-curriculum2_{a.scene}_{tag}{"_smoke" if a.smoke else ""}.json'
    path.write_text(json.dumps(spec, indent=2)+'\n')
    if a.prepare_only: print(path); return
    check = subprocess.run([sys.executable, str(REGISTRY), 'check', str(path)], capture_output=True, text=True); (directory/(path.stem+'.check.json')).write_text(check.stdout)
    if check.returncode and not (check.returncode == 3 and a.repeat_reason): raise SystemExit(check.stderr or 'Duplicate experiment; supply a concrete --repeat-reason.')
    command = [sys.executable, str(REGISTRY), 'run', str(path)] + (['--repeat-reason', a.repeat_reason] if a.repeat_reason else [])
    subprocess.run(command, check=True)


if __name__ == '__main__': main()
