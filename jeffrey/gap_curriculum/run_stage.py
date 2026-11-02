"""Register a diagnostic stage over finished gap-curriculum runs (outputs in results/registered_<id>/).

    BELLMAN600_BACKEND=host ~/.venvs/bellman-ntfield/bin/python jeffrey/gap_curriculum/run_stage.py rounds --runs registered_<id> --gpu 1
    BELLMAN600_BACKEND=host ~/.venvs/bellman-ntfield/bin/python jeffrey/gap_curriculum/run_stage.py truth \
        --runs registered_<A>=B10 registered_<B>=C10 [--rounds 5 10] --gpu 1

rounds  rounds.py: native RK4 of every round checkpoint of one run
truth   truth.py: graph ground truth (kNN Dijkstra) compared with the final checkpoints (and --rounds) of several runs
        of one scene
long    long.py (cabinet): native RK4 of the final checkpoints of several runs on 400 uncertified long-gap queries
        built from the frozen suites' endpoints
Both are diagnostics: the evaluated model of a run is always its last round. The command is stage_launch.py here
(launch.py's mounts and builders, unedited); checkpoints, bank, speed and sources are declared inputs.
"""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
from launch import RELEASE
from experiments.cabinet.run import scene_paths
from experiments.face_speed.run import environment

REGISTRY = HERE/'tools/experiment_registry/registry.py'
SOURCES = ['jeffrey/gap_curriculum/rounds.py', 'jeffrey/gap_curriculum/truth.py', 'jeffrey/gap_curriculum/long.py', 'jeffrey/gap_curriculum/run_stage.py', 'jeffrey/gap_curriculum/stage_launch.py',
           'launch.py', 'method', 'runtime_manifest.json', 'experiments/transition_wall/evaluate.py']


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['rounds', 'truth', 'long']); p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--rounds', type=int, nargs='*', default=[], help='truth: also these round checkpoints of every run'); p.add_argument('--gpu')
    p.add_argument('--mppi-seeds', type=int, nargs='*', default=[], help='long: also the release MPPI with these planning seeds')
    p.add_argument('--smoke', action='store_true'); p.add_argument('--repeat-reason'); p.add_argument('--prepare-only', action='store_true'); a = p.parse_args()
    gpus = 'all' if a.gpu is None else f'device={a.gpu}'
    named = [x.split('=', 1) if '=' in x else (x, Path(x).name[11:]) for x in a.runs]; runs = [(HERE/'results'/Path(r).name, label) for r, label in named]
    scenes = set(); prepared = set()
    for run, _ in runs:
        status = json.loads((run/'status.json').read_text()); assert status.get('complete'), f'{run.name} is not complete'
        scenes.add('cabinet' if status['settings'].get('scene') else 'nowall'); prepared.add(status['settings']['prepared'])
    assert len(scenes) == 1 and len(prepared) == 1, 'one scene and one bank per stage'
    scene = scenes.pop(); prep = HERE/prepared.pop()[len('/clean/'):]
    inputs = dict(runtime=str(RELEASE), registry=str(REGISTRY), **{'source_'+s.replace('/', '_'): str(HERE/s) for s in SOURCES})
    if a.stage == 'rounds':
        assert len(runs) == 1, 'rounds: one run'; run = runs[0][0]
        checkpoints = sorted((run/'training').glob('round_*.pt')); inputs.update({'checkpoint_'+c.stem: str(c) for c in checkpoints})
        args = ['/clean/jeffrey/gap_curriculum/rounds.py', '--run', f'/clean/results/{run.name}', '--output', '/outputs/rounds']
        if scene == 'cabinet':
            SCENE, TESTS = scene_paths('cabinet'); inputs.update(scene=str(SCENE), scene_tests=str(TESTS))
            args += ['--scene', f'/clean/{SCENE.relative_to(HERE)}', '--tests', f'/clean/{TESTS.relative_to(HERE)}']
        required = ['rounds/rounds.json']; listed = [c.name for c in checkpoints]
    elif a.stage == 'long':
        assert scene == 'cabinet', 'long: cabinet only'
        SCENE, TESTS = scene_paths('cabinet'); checkpoints = [run/'training/checkpoint.pt' for run, _ in runs]; labels = [label for _, label in runs]
        inputs.update({f'checkpoint_{i}': str(c) for i, c in enumerate(checkpoints)}, scene=str(SCENE), scene_tests=str(TESTS))
        args = ['/clean/jeffrey/gap_curriculum/long.py', '--checkpoints', *[f'/clean/{c.relative_to(HERE)}' for c in checkpoints], '--labels', *labels,
                '--scene', f'/clean/{SCENE.relative_to(HERE)}', '--tests', f'/clean/{TESTS.relative_to(HERE)}', '--output', '/outputs/long'] + (['--count', '8'] if a.smoke else []) + (['--mppi-seeds', *map(str, a.mppi_seeds)] if a.mppi_seeds else [])
        required = ['long/long.json']; listed = [f'{l}={c.relative_to(HERE)}' for l, c in zip(labels, checkpoints)]
    else:
        checkpoints, labels = [], []
        for run, label in runs:
            checkpoints.append(run/'training/checkpoint.pt'); labels.append(label)
            for r in a.rounds:
                checkpoints.append(run/'training'/f'round_{r:02d}.pt'); labels.append(f'{label}@r{r}')
        inputs.update({f'checkpoint_{i}': str(c) for i, c in enumerate(checkpoints)}, bank_source=str(prep/'bank/source_q.npy'), speed=str(prep/'speed.pt'))
        args = ['/clean/jeffrey/gap_curriculum/truth.py', '--prepared', f'/clean/{prep.relative_to(HERE)}', '--checkpoints', *[f'/clean/{c.relative_to(HERE)}' for c in checkpoints],
                '--labels', *labels, '--output', '/outputs/truth.json'] + (['--goals', '8'] if a.smoke else [])
        required = ['truth.json']; listed = [f'{l}={c.relative_to(HERE)}' for l, c in zip(labels, checkpoints)]
    config = dict(action='gap-curriculum-'+a.stage, runs=[r.name for r, _ in runs], scene=scene, gpus=gpus, checkpoints=listed, smoke=a.smoke, diagnostic_only=True, mppi_seeds=a.mppi_seeds)
    if a.smoke and a.stage == 'rounds': args += ['--smoke']
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    spec = dict(name=f'Clean Bellman600 gap-curriculum-{a.stage} {scene} {" ".join(l for _, l in runs)} {tag}'+(' smoke' if a.smoke else ' full'), kind='pipeline',
                root=str(HERE), config=config, environment=environment(), inputs=inputs,
                command=[sys.executable, str(HERE/'jeffrey/gap_curriculum/stage_launch.py'), '--output', '{output}', '--gpus', gpus, '--', *args],
                required_outputs=required, validation='files')
    directory = HERE/'specs'; path = directory/f'gap-curriculum-{a.stage}_{tag}{"_smoke" if a.smoke else ""}.json'; path.write_text(json.dumps(spec, indent=2)+'\n')
    if a.prepare_only: print(path); return
    check = subprocess.run([sys.executable, str(REGISTRY), 'check', str(path)], capture_output=True, text=True); (directory/(path.stem+'.check.json')).write_text(check.stdout)
    if check.returncode and not (check.returncode == 3 and a.repeat_reason): raise SystemExit(check.stderr or 'Duplicate experiment; supply a concrete --repeat-reason.')
    subprocess.run([sys.executable, str(REGISTRY), 'run', str(path)] + (['--repeat-reason', a.repeat_reason] if a.repeat_reason else []), check=True)


if __name__ == '__main__': main()
