"""Register eikonal-foundation Bellman-extension runs with the repository's experiment registry
(outputs in results/registered_<id>/).

    BELLMAN600_DERIVED_IMAGE=bellman600-env:local python3 jeffrey/eikonal_bellman/run.py --scene cabinet \
        --base-run registered_<EG16 run> --scheme efl --models multi --split pool --label K1 --gpu 1

--scene nowall|cabinet   pool, speed and frozen benchmark as jeffrey/segment_dp/run.py (--prepared-run for the cabinet)
--base-run               the registered eikonal run whose training/field.pt is the frozen foundation (one model)
--truth-run              registered segment-dp-truth run of the same scene (per-level diagnostic input; optional)
chain.py options are passed through (--scheme, --models, --split, --candidates, --continuous, --refine, --refine-count,
--support, --clearance (cabinet: the scene file), --estimator, --init, --weights, --rehearsal, --rehearsal-weight,
--old-share, --pairs, --updates, --chunk, --last-edge, --seed). --no-evaluate skips the frozen benchmark (diagnostic
chains). The spec's command is jeffrey/segment_dp/launch.py (launch.py's mounts and builders, unedited).
"""
import argparse, hashlib, json, os, shlex, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
from launch import RELEASE, IMAGE, HOST_PYTHON
from experiments.cabinet.run import scene_paths
sys.path.insert(0, str(HERE/'jeffrey/segment_dp'))
from run import environment

REGISTRY = HERE/'tools/experiment_registry/registry.py'
SOURCES = ['jeffrey/eikonal_bellman/chain.py', 'jeffrey/eikonal_bellman/pipeline.py', 'jeffrey/eikonal_bellman/run.py',
           'jeffrey/segment_dp/fields.py', 'jeffrey/segment_dp/sampler.py', 'jeffrey/segment_dp/evaluate.py', 'jeffrey/segment_dp/launch.py',
           'launch.py', 'method', 'runtime_manifest.json', 'experiments/cabinet/evaluate.py', 'experiments/transition_wall/evaluate.py']
CLEARANCE_SOURCES = ['jeffrey/geometry.py', 'experiments/face_speed/robot.py', 'experiments/face_speed/clearance.py', 'experiments/face_speed/faces.py', 'experiments/point_field/point_field.py']
PASS = ['scheme', 'models', 'split', 'candidates', 'continuous', 'refine', 'refine_count', 'support', 'estimator', 'init', 'weights', 'rehearsal', 'rehearsal_weight',
        'old_share', 'pairs', 'updates', 'chunk', 'last_edge', 'seed', 'first_range', 'blend']


def main():
    p = argparse.ArgumentParser(); p.add_argument('--scene', choices=['nowall', 'cabinet'], required=True); p.add_argument('--gpu')
    p.add_argument('--prepared-run', default='registered_20261019T034213Z-f9c501f9', help='cabinet: run whose prepared/ speed and pool to use')
    p.add_argument('--base-run', required=True); p.add_argument('--truth-run')
    p.add_argument('--scheme', default='efl'); p.add_argument('--models', default='multi'); p.add_argument('--split', default='pool')
    p.add_argument('--candidates', type=int, default=128); p.add_argument('--continuous', type=int, default=32); p.add_argument('--refine', type=int, default=4); p.add_argument('--refine-count', type=int, default=2)
    p.add_argument('--support', default='auto'); p.add_argument('--clearance', action='store_true', help='cabinet: exact release clearance gate on off-pool split points')
    p.add_argument('--estimator', default='single'); p.add_argument('--init', default='foundation'); p.add_argument('--weights', default='1,0,0')
    p.add_argument('--rehearsal', type=int, default=0); p.add_argument('--rehearsal-weight', type=float, default=2.0); p.add_argument('--old-share', type=float, default=0.5)
    p.add_argument('--pairs', type=int, default=400000); p.add_argument('--updates', type=int, default=15620); p.add_argument('--chunk', type=int, default=128)
    p.add_argument('--last-edge', type=float, default=None); p.add_argument('--first-range', type=float, default=0.176); p.add_argument('--blend', type=float, default=0.016)
    p.add_argument('--seed', type=int, default=87358); p.add_argument('--label', required=True); p.add_argument('--no-evaluate', action='store_true')
    p.add_argument('--smoke', action='store_true'); p.add_argument('--repeat-reason'); p.add_argument('--prepare-only', action='store_true'); a = p.parse_args()
    gpus = 'all' if a.gpu is None else f'device={a.gpu}'
    sources = SOURCES+(CLEARANCE_SOURCES if a.clearance else [])
    inputs = dict(runtime=str(RELEASE), registry=str(REGISTRY), **{'source_'+s.replace('/', '_'): str(HERE/s) for s in sources})
    if a.scene == 'nowall':
        prepared = HERE/'assets'; inputs['reference'] = str(HERE/'assets/reference'); args = ['--prepared', '/clean/assets']; assert not a.clearance, 'clearance gate: cabinet only'
    else:
        SCENE, TESTS = scene_paths('cabinet'); prepared = HERE/'results'/a.prepared_run/'prepared'
        inputs.update(scene=str(SCENE), scene_tests=str(TESTS))
        args = ['--prepared', f'/clean/{prepared.relative_to(HERE)}', '--scene', f'/clean/{SCENE.relative_to(HERE)}', '--tests', f'/clean/{TESTS.relative_to(HERE)}']
    inputs.update(pool=str(prepared/'bank/source_q.npy'), pool_manifest=str(prepared/'bank/manifest.json'), speed=str(prepared/'speed.pt'))
    base_run = HERE/'results'/Path(a.base_run).name; base = base_run/'training/field.pt'
    status = json.loads((base_run/'status.json').read_text()); assert status['complete'] and status['settings']['trainer'] == 'eikonal', a.base_run
    assert status['settings']['prepared'] == f'/clean/{prepared.relative_to(HERE)}', 'foundation of another scene'
    inputs['base_field'] = str(base)
    train = ['--base', f'/clean/{base.relative_to(HERE)}']
    for k in PASS:
        v = getattr(a, k)
        if v is None: continue
        train += [f'--{k.replace("_", "-")}', f'{v:g}' if isinstance(v, float) else str(v)]
    if a.clearance: train += ['--clearance', f'/clean/{SCENE.relative_to(HERE)}']
    if a.truth_run:
        truth_run = HERE/'results'/Path(a.truth_run).name; truth = truth_run/'truth/truth.npz'
        truth_config = json.loads((truth_run/'truth/truth.json').read_text()); assert truth_config['scene_sha256'] == json.loads((prepared/'bank/manifest.json').read_text())['scene_sha256'], 'truth of another scene'
        inputs['truth'] = str(truth); train += ['--truth', f'/clean/{truth.relative_to(HERE)}']
    config = dict(action='eikonal-bellman', scene=a.scene, prepared=str(prepared.relative_to(HERE)), gpus=gpus, base_run=base_run.name, truth_run=a.truth_run, label=a.label,
                  **{k: getattr(a, k) for k in PASS}, clearance=a.clearance, evaluate=not a.no_evaluate, smoke=a.smoke,
                  planning_seeds=[0] if a.smoke else [0, 1, 2], scorer_tilt_deg=180 if a.scene == 'cabinet' else 20, transitions_used=False)
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    required = ['status.json', 'training/field.pt', 'training/progress.json']+([] if a.no_evaluate else ['evaluation/evaluation.json'])
    spec = dict(name=f'Clean Bellman600 eikonal-bellman {a.scene} {a.label} {tag}'+(' smoke' if a.smoke else ' full'), kind='pipeline', root=str(HERE), config=config,
                environment=environment(), inputs=inputs,
                command=[sys.executable, str(HERE/'jeffrey/segment_dp/launch.py'), '--output', '{output}', '--gpus', gpus, '--', '/clean/jeffrey/eikonal_bellman/pipeline.py', *args,
                         *(['--smoke'] if a.smoke else []), *(['--no-evaluate'] if a.no_evaluate else []), '--output', '/outputs', '--train-args='+shlex.join(train)],
                required_outputs=required, validation='files')
    directory = HERE/'specs'; directory.mkdir(exist_ok=True); path = directory/f'eikonal-bellman_{a.scene}_{a.label}_{tag}{"_smoke" if a.smoke else ""}.json'
    path.write_text(json.dumps(spec, indent=2)+'\n')
    if a.prepare_only: print(path); return
    check = subprocess.run([sys.executable, str(REGISTRY), 'check', str(path)], capture_output=True, text=True); (directory/(path.stem+'.check.json')).write_text(check.stdout)
    if check.returncode and not (check.returncode == 3 and a.repeat_reason): raise SystemExit(check.stderr or 'Duplicate experiment; supply a concrete --repeat-reason.')
    command = [sys.executable, str(REGISTRY), 'run', str(path)]+(['--repeat-reason', a.repeat_reason] if a.repeat_reason else [])
    subprocess.run(command, check=True)


if __name__ == '__main__': main()
