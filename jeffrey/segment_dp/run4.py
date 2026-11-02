"""run.py for task 8, the segment DP on a correct local field (outputs in results/registered_<id>/).

    python3 jeffrey/segment_dp/run4.py labels --part 0/2 --gpu 1          local_truth.py: path-optimized labels
    python3 jeffrey/segment_dp/run4.py supervised --label-runs registered_<A> registered_<B> --gpu 1
    python3 jeffrey/segment_dp/run4.py dp3 --base-run registered_<supervised run> --truth-run registered_<T> --gpu 1

labels      local_truth.py on the scene's pool: --pairs per segment of --segments (PairSampler, the eikonal specialists'
            pair distribution), path optimization --waypoints / --steps; --part k/K labels every K-th pair (one
            run per GPU)
supervised  pipeline4.py --trainer supervised: supervised.py on the label runs, then the frozen-benchmark evaluation of
            the resulting field (the SL1-2 control: model 2 answers every longer gap)
exact       dp_exact.py (diagnostic): DP targets of levels 3-4 on the truth probe pairs from path-optimized pieces vs
            the bases of the listed --runs (registered_<id>=label)
select      dp_select.py (diagnostic): target / T* of DP chains (--runs) split into the teacher's bias at its chosen split
            and the true via-split overhead, at --levels
close       close_plan.py (diagnostic): native RK4 on the path-optimized truth field, the straight line and the listed
            --runs (results-relative checkpoint path=label) for --close-pairs pool pairs per segment and the frozen
            queries with gap <= 0.16
dp3         pipeline4.py --trainer dp3: dp3.py unchanged (defaults = dp.py) on the field of --base-run, which may be a
            supervised run of this file or an eikonal specialists run (run.py); then the frozen evaluation
The command is jeffrey/segment_dp/launch.py (launch.py's mounts and builders, unedited); sources, pool, speed, scene,
labels, base field and truth file are declared inputs.
"""
import argparse, hashlib, json, shlex, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
from launch import RELEASE
from experiments.cabinet.run import scene_paths
sys.path.insert(0, str(HERE/'jeffrey/segment_dp'))
from run import environment, EDGES, PREPARED_RUNS

REGISTRY = HERE/'tools/experiment_registry/registry.py'
SOURCES = ['jeffrey/segment_dp/fields.py', 'jeffrey/segment_dp/sampler.py', 'jeffrey/segment_dp/eikonal.py', 'jeffrey/segment_dp/truth.py', 'jeffrey/segment_dp/local_truth.py',
           'jeffrey/segment_dp/supervised.py', 'jeffrey/segment_dp/dp_exact.py', 'jeffrey/segment_dp/dp_select.py', 'jeffrey/segment_dp/close_plan.py', 'jeffrey/segment_dp/dp.py', 'jeffrey/segment_dp/dp3.py', 'jeffrey/segment_dp/evaluate.py', 'jeffrey/segment_dp/pipeline4.py',
           'jeffrey/segment_dp/run4.py', 'jeffrey/segment_dp/run.py', 'jeffrey/segment_dp/launch.py', 'launch.py', 'method', 'runtime_manifest.json',
           'experiments/cabinet/evaluate.py', 'experiments/transition_wall/evaluate.py']


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['labels', 'supervised', 'dp3', 'exact', 'select', 'close']); p.add_argument('--runs', nargs='*', default=[]); p.add_argument('--exact-segments', default='3,4'); p.add_argument('--levels', default='5,7,9,11'); p.add_argument('--exact-pairs', default='1024,256'); p.add_argument('--scene', default='cabinet'); p.add_argument('--gpu')
    p.add_argument('--prepared-run', help='default: the scene\'s pipeline run (run.py PREPARED_RUNS)')
    p.add_argument('--part', default='0/1'); p.add_argument('--pairs', type=int, default=400000); p.add_argument('--segments', default='1,2')
    p.add_argument('--waypoints', type=int, default=8); p.add_argument('--chunk', type=int, default=4096); p.add_argument('--steps', type=int, default=100)
    p.add_argument('--label-runs', nargs='*', default=[]); p.add_argument('--w-grad', type=float, default=1.0); p.add_argument('--updates', type=int, default=15620)
    p.add_argument('--base-run'); p.add_argument('--base-models', type=int, default=2); p.add_argument('--truth-run'); p.add_argument('--last-segment', type=int, default=20)
    p.add_argument('--losses', default='dp'); p.add_argument('--seed', type=int, default=87358); p.add_argument('--label', default='')
    p.add_argument('--close-pairs', type=int, default=100)
    p.add_argument('--smoke', action='store_true'); p.add_argument('--repeat-reason'); p.add_argument('--prepare-only', action='store_true'); a = p.parse_args()
    gpus = 'all' if a.gpu is None else f'device={a.gpu}'
    assert a.scene.startswith('cabinet'), 'task 8 runs on cabinet scenes'
    SCENE, TESTS = scene_paths(a.scene); prepared = HERE/'results'/(a.prepared_run or PREPARED_RUNS[a.scene])/'prepared'; prep = f'/clean/{prepared.relative_to(HERE)}'
    inputs = dict(runtime=str(RELEASE), registry=str(REGISTRY), **{'source_'+s.replace('/', '_'): str(HERE/s) for s in SOURCES},
                  pool=str(prepared/'bank/source_q.npy'), pool_manifest=str(prepared/'bank/manifest.json'), speed=str(prepared/'speed.pt'))
    config = dict(action='segment-dp-'+a.stage, scene=a.scene, prepared=str(prepared.relative_to(HERE)), gpus=gpus, seed=a.seed, smoke=a.smoke, transitions_used=False)
    if a.stage == 'labels':
        k, K = a.part.split('/'); name = 'labels.npz' if K == '1' else f'labels_part{k}of{K}.npz'
        config.update(part=a.part, **({} if a.chunk == 4096 else dict(chunk=a.chunk)), pairs_per_segment=a.pairs, segments=a.segments, waypoints=a.waypoints, steps=a.steps, edges=EDGES, blend=0.016, label_seed=20261031)
        command = ['/clean/jeffrey/segment_dp/local_truth.py', '--prepared', prep, '--output', '/outputs/labels', '--part', a.part, '--pairs', str(a.pairs), '--segments', a.segments,
                   '--waypoints', str(a.waypoints), '--steps', str(a.steps), '--chunk', str(a.chunk)] + (['--smoke'] if a.smoke else [])
        required = [f'labels/{name}', f'labels/{name[:-4]}.json']; label = a.label or f'labels-{k}of{K}'
    elif a.stage == 'select':
        truth_run = HERE/'results'/Path(a.truth_run).name; truth = truth_run/'truth/truth.npz'; inputs['truth'] = str(truth)
        runs = [x.split('=') for x in a.runs]; dirs = [HERE/'results'/Path(r).name/'training' for r, _ in runs]; levels = [int(x) for x in a.levels.split(',')]
        for r, _ in runs: assert json.loads((HERE/'results'/Path(r).name/'status.json').read_text())['settings']['prepared'] == prep, r
        for n, d in enumerate(dirs):
            for m in levels: f = d/f'field_seg{m-1:02d}.pt'; assert f.is_file(), f; inputs[f'chain_{n}_seg{m-1:02d}'] = str(f)
        config.update(levels=a.levels, truth_run=truth_run.name, chains=[f'{l}={Path(r).name}' for r, l in runs], diagnostic_only=True)
        command = ['/clean/jeffrey/segment_dp/dp_select.py', '--prepared', prep, '--truth', f'/clean/{truth.relative_to(HERE)}', '--chains', *[f'/clean/{d.relative_to(HERE)}' for d in dirs],
                   '--labels', *[l for _, l in runs], '--levels', a.levels, '--chunk', str(a.chunk), '--output', '/outputs/select.json'] + (['--smoke'] if a.smoke else [])
        required = ['select.json']; label = a.label or 'select-'+'-'.join(l for _, l in runs)
    elif a.stage == 'close':
        runs = [x.split('=') for x in a.runs]; ckpts = [HERE/'results'/r for r, _ in runs]
        for c in ckpts: assert c.is_file(), c
        inputs.update(scene=str(SCENE), scene_tests=str(TESTS), **{f'field_{n}': str(c) for n, c in enumerate(ckpts)})
        config.update(close_pairs=a.close_pairs, segments=a.segments, truth_waypoints=a.waypoints, truth_steps=a.steps, fields=[f'{l}={r}' for r, l in runs], scorer_tilt_deg=180, diagnostic_only=True)
        command = ['/clean/jeffrey/segment_dp/close_plan.py', '--prepared', prep, '--scene', f'/clean/{SCENE.relative_to(HERE)}', '--tests', f'/clean/{TESTS.relative_to(HERE)}',
                   '--checkpoints', *[f'/clean/{c.relative_to(HERE)}' for c in ckpts], '--labels', *[l for _, l in runs], '--output', '/outputs/close',
                   '--pairs', str(a.close_pairs), '--segments', a.segments, '--waypoints', str(a.waypoints), '--steps', str(a.steps)] + (['--smoke'] if a.smoke else [])
        required = ['close/close.json']; label = a.label or 'close-truth'
    elif a.stage == 'exact':
        truth_run = HERE/'results'/Path(a.truth_run).name; truth = truth_run/'truth/truth.npz'; inputs['truth'] = str(truth)
        runs = [x.split('=') for x in a.runs]; fields = [HERE/'results'/Path(r).name/'training/field.pt' for r, _ in runs]
        for r, _ in runs: assert json.loads((HERE/'results'/Path(r).name/'status.json').read_text())['settings']['prepared'] == prep, r
        inputs.update({f'field_{n}': str(f) for n, f in enumerate(fields)})
        config.update(exact_segments=a.exact_segments, exact_pairs=a.exact_pairs, truth_run=truth_run.name, fields=[f'{l}={Path(r).name}' for r, l in runs], diagnostic_only=True)
        command = ['/clean/jeffrey/segment_dp/dp_exact.py', '--prepared', prep, '--truth', f'/clean/{truth.relative_to(HERE)}', '--fields', *[f'/clean/{f.relative_to(HERE)}' for f in fields],
                   '--labels', *[l for _, l in runs], '--output', '/outputs/exact.json', '--chunk', str(a.chunk), '--segments', a.exact_segments, '--max-pairs', a.exact_pairs] + (['--smoke'] if a.smoke else [])
        required = ['exact.json']; label = a.label or 'exact-'+'-'.join(l for _, l in runs)
    else:
        inputs.update(scene=str(SCENE), scene_tests=str(TESTS)); config.update(planning_seeds=[0] if a.smoke else [0, 1, 2], scorer_tilt_deg=180, updates_per_model=a.updates)
        train = ['--seed', str(a.seed), '--updates', str(a.updates)]
        if a.stage == 'supervised':
            files = []
            for r in a.label_runs:
                d = HERE/'results'/Path(r).name/'labels'
                f = sorted(d.glob('labels*.npz')); assert len(f) == 1, (r, f); files.append(f[0])
                assert json.loads(f[0].with_suffix('.json').read_text())['source_q_sha256'] == json.loads((prepared/'bank/manifest.json').read_text())['file_sha256']['source_q.npy'], 'labels of another pool'
            inputs.update({f'labels_{n}': str(f) for n, f in enumerate(files)}); inputs.update({f'labels_meta_{n}': str(f.with_suffix('.json')) for n, f in enumerate(files)})
            config.update(label_runs=[Path(r).name for r in a.label_runs], models=a.base_models, w_grad=a.w_grad)
            train += ['--labels', *[f'/clean/{f.relative_to(HERE)}' for f in files], '--models', str(a.base_models), '--w-grad', f'{a.w_grad:g}']
            trainer = 'supervised'; label = a.label or ('SL' + ('' if a.w_grad == 1 else f'-g{a.w_grad:g}'))
        else:
            base_run = HERE/'results'/Path(a.base_run).name; base = base_run/'training/field.pt'
            status = json.loads((base_run/'status.json').read_text()); assert status['complete'] and status['settings']['trainer'] in ('eikonal', 'supervised'), a.base_run
            assert status['settings']['prepared'] == prep, 'base field of another scene'
            inputs['base_field'] = str(base)
            config.update(base_run=base_run.name, base_trainer=status['settings']['trainer'], base_models=a.base_models, losses=a.losses.split(','), last_segment=a.last_segment,
                          pairs=400000, candidates=384, regression_loss='smoothl1', decoder='mult', init='fresh')
            train += ['--base', f'/clean/{base.relative_to(HERE)}', '--base-models', str(a.base_models), '--losses', a.losses, '--last-segment', str(a.last_segment)]
            if a.truth_run:
                truth_run = HERE/'results'/Path(a.truth_run).name; truth = truth_run/'truth/truth.npz'
                assert json.loads((truth_run/'truth/truth.json').read_text())['scene_sha256'] == json.loads((prepared/'bank/manifest.json').read_text())['scene_sha256']
                inputs['truth'] = str(truth); config['truth_run'] = truth_run.name; train += ['--truth', f'/clean/{truth.relative_to(HERE)}']
            trainer = 'dp3'; label = a.label or ('DP-' + ('SL' if status['settings']['trainer'] == 'supervised' else 'ES') + ('' if a.losses == 'dp' else '-'+a.losses.replace(',', '+')))
        command = ['/clean/jeffrey/segment_dp/pipeline4.py', '--prepared', prep, '--scene', f'/clean/{SCENE.relative_to(HERE)}', '--tests', f'/clean/{TESTS.relative_to(HERE)}',
                   '--trainer', trainer, *(['--smoke'] if a.smoke else []), '--output', '/outputs', '--train-args='+shlex.join(train)]
        required = ['status.json', 'training/field.pt', 'evaluation/evaluation.json']
    if a.seed != 87358 and not a.label: label += f's{a.seed}'
    config['label'] = label
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    spec = dict(name=f'Clean Bellman600 segment-dp4 {a.stage} {a.scene} {label} {tag}'+(' smoke' if a.smoke else ' full'), kind='pipeline', root=str(HERE), config=config,
                environment=environment(), inputs=inputs,
                command=[sys.executable, str(HERE/'jeffrey/segment_dp/launch.py'), '--output', '{output}', '--gpus', gpus, '--', *command],
                required_outputs=required, validation='files')
    directory = HERE/'specs'; directory.mkdir(exist_ok=True); path = directory/f'segment-dp4_{a.stage}_{a.scene}_{tag}{"_smoke" if a.smoke else ""}.json'
    path.write_text(json.dumps(spec, indent=2)+'\n')
    if a.prepare_only: print(path); return
    check = subprocess.run([sys.executable, str(REGISTRY), 'check', str(path)], capture_output=True, text=True); (directory/(path.stem+'.check.json')).write_text(check.stdout)
    if check.returncode and not (check.returncode == 3 and a.repeat_reason): raise SystemExit(check.stderr or 'Duplicate experiment; supply a concrete --repeat-reason.')
    subprocess.run([sys.executable, str(REGISTRY), 'run', str(path)] + (['--repeat-reason', a.repeat_reason] if a.repeat_reason else []), check=True)


if __name__ == '__main__': main()
