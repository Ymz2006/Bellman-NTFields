"""Register a diagnostic stage of the segment-DP experiment (outputs in results/registered_<id>/).

    python3 jeffrey/segment_dp/run_stage.py truth --scene cabinet --gpu 1
    python3 jeffrey/segment_dp/run_stage.py truth --scene cabinet_clutter --prepared-run registered_<pipeline> --gpu 0
    python3 jeffrey/segment_dp/run_stage.py fit --truth-run registered_<T> --runs registered_<A>=ES registered_<B>=EG \
        [registered_<C>:training/field_u03124.pt=ES@1/5] [--by-model ES] --gpu 1
    python3 jeffrey/segment_dp/run_stage.py long --runs registered_<A>=ES ... --mppi-seeds 0 --gpu 1

truth  truth.py: probe pairs per distance segment with path-optimized ground-truth travel times (pool + speed only)
fit    fit.py: every listed field against a truth stage of the same scene (a run's final field: training/field.pt of
       segment runs, training/checkpoint.pt of Bellman runs; or run:relative/path for another snapshot)
long   long.py (cabinet): native RK4 (+ release MPPI with --mppi-seeds) of the listed fields on the 400 uncertified
       long queries of jeffrey/gap_curriculum/long.py
pool   pool_region.py: the configurations of a pipeline run (pool + coverage strata) whose grasp point is inside a
       compartment, as a prepared/ directory for the bank-free trainers (--prepared-run of run.py / run3.py)
derived  derived.py: the control field of the DP chain, the first --first models of one segment field (the last of
       them answering every longer gap), or with --splice DONOR --segments A-B the field with the donor's models on
       segments A..B; and its frozen-benchmark evaluation (evaluate.py)
All are diagnostics (never select a model). The command is launch.py here (launch.py's mounts and builders,
unedited); fields, pool, speed, truth file and sources are declared inputs.
"""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
from launch import RELEASE
from experiments.cabinet.run import scene_paths
sys.path.insert(0, str(HERE/'jeffrey/segment_dp'))
from run import environment, EDGES, PREPARED_RUNS

REGISTRY = HERE/'tools/experiment_registry/registry.py'
SOURCES = ['jeffrey/segment_dp/fields.py', 'jeffrey/segment_dp/sampler.py', 'jeffrey/segment_dp/eikonal.py', 'jeffrey/segment_dp/truth.py', 'jeffrey/segment_dp/fit.py',
           'jeffrey/segment_dp/derived.py', 'jeffrey/segment_dp/evaluate.py', 'experiments/cabinet/evaluate.py',
           'jeffrey/segment_dp/long.py', 'jeffrey/gap_curriculum/long.py', 'jeffrey/segment_dp/pool_region.py', 'jeffrey/segment_dp/run_stage.py', 'jeffrey/segment_dp/run.py', 'jeffrey/segment_dp/launch.py',
           'launch.py', 'method', 'runtime_manifest.json', 'experiments/transition_wall/evaluate.py']
PREPARED = dict(nowall=HERE/'assets', **{k: HERE/'results'/v/'prepared' for k, v in PREPARED_RUNS.items()})


def field_of(spec):
    """'registered_<id>[:relative/path]=label' -> (run dir, field path, label)."""
    name, label = spec.split('=', 1) if '=' in spec else (spec, None)
    run, _, rel = name.partition(':'); run = HERE/'results'/Path(run).name
    if not (run/'status.json').exists():   # a derived stage: its scene is that of the run it was derived from
        derived = json.loads((run/'derived/derived.json').read_text()); assert derived.get('complete'), f'{run.name} is not complete'
        status = json.loads((HERE/'results'/Path(derived['source']).parts[3]/'status.json').read_text())
    else:
        status = json.loads((run/'status.json').read_text()); assert status.get('complete'), f'{run.name} is not complete'
        if 'settings' not in status:   # an original cabinet pipeline run (its checkpoint is the B5 of that scene)
            status = dict(status, settings=dict(scene=status['scene'], trainer='pipeline'))
    path = run/rel if rel else next(p for p in (run/'training/field.pt', run/'derived/field.pt', run/'training/checkpoint.pt') if p.exists())
    assert path.is_file(), path
    return run, path, label or run.name[11:], status


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['truth', 'fit', 'long', 'derived', 'pool']); p.add_argument('--scene', help='nowall, cabinet or a cabinet variant')
    p.add_argument('--runs', nargs='*', default=[]); p.add_argument('--truth-run'); p.add_argument('--prepared-run', help='truth on a scene without a PREPARED_RUNS entry: its pipeline run'); p.add_argument('--by-model', nargs='*', default=[])
    p.add_argument('--mppi-seeds', type=int, nargs='*', default=[]); p.add_argument('--first', type=int, default=2); p.add_argument('--splice', help='derived: donor run (registered_<id>)'); p.add_argument('--segments', help='derived --splice: A-B'); p.add_argument('--count', type=int, default=400, help='long: queries (the first COUNT of the seeded 400)'); p.add_argument('--goals', type=int, default=128); p.add_argument('--per-goal', type=int, default=8)
    p.add_argument('--gpu'); p.add_argument('--smoke', action='store_true'); p.add_argument('--repeat-reason'); p.add_argument('--prepare-only', action='store_true'); a = p.parse_args()
    gpus = 'all' if a.gpu is None else f'device={a.gpu}'
    inputs = dict(runtime=str(RELEASE), registry=str(REGISTRY), **{'source_'+s.replace('/', '_'): str(HERE/s) for s in SOURCES})
    fields = [field_of(x) for x in a.runs]
    scenes = {(Path(f[3]['settings']['scene']).stem if f[3]['settings'].get('scene') else 'nowall') for f in fields}
    if a.stage in ('truth', 'pool'):
        scene = a.scene; assert scene and not fields
    else:
        assert len(scenes) == 1 and fields, 'one scene per stage'; scene = scenes.pop(); assert a.scene in (None, scene)
    if a.prepared_run:   # a scene without a PREPARED_RUNS entry: the pipeline run whose prepared/ pool and speed to use
        assert a.stage in ('truth', 'pool'), '--prepared-run: truth and pool only (other stages use the listed runs\' pool)'; prep = HERE/'results'/Path(a.prepared_run).name/'prepared'
        assert json.loads((prep/'bank/manifest.json').read_text())['scene_sha256'] == hashlib.sha256(scene_paths(scene)[0].read_bytes()).hexdigest(), 'prepared run of another scene'
    elif scene in PREPARED:
        prep = PREPARED[scene]
    else:   # a scene without a PREPARED_RUNS entry (cabinet_clutter, cabinet_platforms): the pool every listed run was trained on
        preps = {(HERE/f[3]['settings']['prepared'].replace('/clean/', '', 1)) if 'prepared' in f[3]['settings'] else f[0]/'prepared' for f in fields}
        assert len(preps) == 1, f'listed runs use different pools: {preps}'; prep = preps.pop()
    inputs.update(pool=str(prep/'bank/source_q.npy'), pool_manifest=str(prep/'bank/manifest.json'), speed=str(prep/'speed.pt'))
    inputs.update({f'field_{i}': str(path) for i, (_, path, _, _) in enumerate(fields)})
    labels = [label for _, _, label, _ in fields]; listed = [f'{label}={path.relative_to(HERE)}' for _, path, label, _ in fields]
    if a.stage == 'truth':
        args = ['/clean/jeffrey/segment_dp/truth.py', '--prepared', f'/clean/{prep.relative_to(HERE)}', '--output', '/outputs/truth', '--edges', EDGES,
                '--goals', str(a.goals), '--per-goal', str(a.per_goal)] + (['--smoke'] if a.smoke else [])
        required = ['truth/truth.npz', 'truth/truth.json']
    elif a.stage == 'pool':
        SCENE, _ = scene_paths(scene); coverage = prep.parent/'sampled/coverage/train'
        inputs.update(scene=str(SCENE), source_speed=str(prep/'speed.pt'), **{f'coverage_{f}': str(coverage/f'{f}.npy') for f in ('configurations_rad', 'd_obstacle', 'd_self')})
        args = ['/clean/jeffrey/segment_dp/pool_region.py', '--prepared', f'/clean/{prep.relative_to(HERE)}', '--coverage', f'/clean/{coverage.relative_to(HERE)}',
                '--scene', f'/clean/{SCENE.relative_to(HERE)}', '--output', '/outputs/prepared']
        required = ['prepared/bank/source_q.npy', 'prepared/bank/manifest.json', 'prepared/speed.pt']
    elif a.stage == 'fit':
        truth_run = HERE/'results'/Path(a.truth_run).name; truth = truth_run/'truth/truth.npz'
        truth_config = json.loads((truth_run/'truth/truth.json').read_text()); assert truth_config['scene_sha256'] == json.loads((prep/'bank/manifest.json').read_text())['scene_sha256']
        inputs['truth'] = str(truth)
        args = ['/clean/jeffrey/segment_dp/fit.py', '--truth', f'/clean/{truth.relative_to(HERE)}', '--prepared', f'/clean/{prep.relative_to(HERE)}',
                '--fields', *[f'/clean/{path.relative_to(HERE)}' for _, path, _, _ in fields], '--labels', *labels, '--output', '/outputs/fit.json'] + (['--by-model', *a.by_model] if a.by_model else [])
        required = ['fit.json']
    elif a.stage == 'derived':
        assert len(fields) == 1, 'derived: one segment field'; path = fields[0][1]
        args = ['/clean/jeffrey/segment_dp/derived.py', '--source', f'/clean/{path.relative_to(HERE)}', '--output', '/outputs/derived']
        if a.splice:   # the models of segments --segments from the donor's final field instead of --first
            donor = field_of(a.splice)[1]; inputs['splice'] = str(donor); args += ['--splice', f'/clean/{donor.relative_to(HERE)}', '--segments', a.segments]
        else:
            args += ['--first', str(a.first)]
        if scene.startswith('cabinet'):
            SCENE, TESTS = scene_paths(scene); inputs.update(scene=str(SCENE), scene_tests=str(TESTS))
            args += ['--scene', f'/clean/{SCENE.relative_to(HERE)}', '--tests', f'/clean/{TESTS.relative_to(HERE)}']
        else:
            inputs['reference'] = str(HERE/'assets/reference')
        args += ['--smoke'] if a.smoke else []
        required = ['derived/field.pt', 'derived/derived.json']
    else:
        assert scene.startswith('cabinet'), 'long: cabinet scenes only'
        SCENE, TESTS = scene_paths(scene); inputs.update(scene=str(SCENE), scene_tests=str(TESTS))
        args = ['/clean/jeffrey/segment_dp/long.py', '--checkpoints', *[f'/clean/{path.relative_to(HERE)}' for _, path, _, _ in fields], '--labels', *labels,
                '--scene', f'/clean/{SCENE.relative_to(HERE)}', '--tests', f'/clean/{TESTS.relative_to(HERE)}', '--output', '/outputs/long'] + (['--count', '8' if a.smoke else str(a.count)]) + \
               (['--mppi-seeds', *map(str, a.mppi_seeds)] if a.mppi_seeds else [])
        required = ['long/long.json']
    config = dict(action='segment-dp-'+a.stage, scene=scene, gpus=gpus, fields=listed, truth_run=a.truth_run, by_model=a.by_model, mppi_seeds=a.mppi_seeds,
                  first=a.first if a.stage == 'derived' and not a.splice else None, count=a.count if a.stage == 'long' else None,
                  goals=a.goals if a.stage == 'truth' else None, per_goal=a.per_goal if a.stage == 'truth' else None, smoke=a.smoke, diagnostic_only=True, **(dict(splice=a.splice, segments=a.segments) if a.splice else {}))
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    spec = dict(name=f'Clean Bellman600 segment-dp-{a.stage} {scene} {" ".join(labels)} {tag}'+(' smoke' if a.smoke else ' full'), kind='pipeline', root=str(HERE), config=config,
                environment=environment(), inputs=inputs,
                command=[sys.executable, str(HERE/'jeffrey/segment_dp/launch.py'), '--output', '{output}', '--gpus', gpus, '--', *args],
                required_outputs=required, validation='files')
    directory = HERE/'specs'; path = directory/f'segment-dp-{a.stage}_{tag}{"_smoke" if a.smoke else ""}.json'; path.write_text(json.dumps(spec, indent=2)+'\n')
    if a.prepare_only: print(path); return
    check = subprocess.run([sys.executable, str(REGISTRY), 'check', str(path)], capture_output=True, text=True); (directory/(path.stem+'.check.json')).write_text(check.stdout)
    if check.returncode and not (check.returncode == 3 and a.repeat_reason): raise SystemExit(check.stderr or 'Duplicate experiment; supply a concrete --repeat-reason.')
    subprocess.run([sys.executable, str(REGISTRY), 'run', str(path)] + (['--repeat-reason', a.repeat_reason] if a.repeat_reason else []), check=True)


if __name__ == '__main__': main()
