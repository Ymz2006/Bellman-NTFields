"""run.py for dp3 runs (dp3.py: the DP chain with the loss / decoder / split-count / TD / ball-backup knobs of tasks
1-5), through pipeline3.py. Same scenes, inputs and evaluation as run.py; --trainer dp3 takes run.py's dp options plus
--losses dp,td,bell --loss --decoder --w-dp --w-td --w-bell --td-step --ball-step --directions --speed-probe
--last-segment --truth-run (the truth stage whose probe pairs give the per-level accuracy and the candidate sweep).

    BELLMAN600_DERIVED_IMAGE=bellman600-env:local python3 jeffrey/segment_dp/run.py --scene cabinet --trainer eikonal \
        --mode specialists --arch pair --gpu 1
    BELLMAN600_DERIVED_IMAGE=bellman600-env:local python3 jeffrey/segment_dp/run.py --scene cabinet --trainer dp \
        --base-run registered_<specialists run> --gpu 1

--scene nowall     pool and speed of assets/ (bank/source_q.npy positions only, no transitions), the frozen 600 queries
--scene cabinet    pool and speed of the registered cabinet pipeline run (--prepared-run), the frozen cabinet suites
--trainer eikonal  part a, eikonal.py: --mode specialists | generalist, --arch pair | offset, --edges, --blend,
                   --updates, --seed
--trainer dp       part b, dp.py: the DP chain from the field of --base-run (its first --base-models models; edges,
                   blend and architecture are the base's), --pairs, --candidates, --updates, --eikonal-weight, --seed
--trainer bellman  part c on the original method, bellman.py: Bellman + Eikonal + direction on the transition bank
                   (5 rounds, 15,620 updates) with --arch pair | offset and --weights B,E,D

The spec's command is launch.py here, which runs pipeline.py with launch.py's own mounts and command builders, so
the top-level run.py / launch.py stay unedited. Sources, pool, speed, scene and the base field are declared inputs.
"""
import argparse, hashlib, json, os, shlex, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
from launch import RELEASE, IMAGE, HOST_PYTHON
from experiments.cabinet.run import scene_paths

REGISTRY = HERE/'tools/experiment_registry/registry.py'
SOURCES = ['jeffrey/segment_dp/fields.py', 'jeffrey/segment_dp/sampler.py', 'jeffrey/segment_dp/dp.py', 'jeffrey/segment_dp/dp3.py', 'jeffrey/segment_dp/evaluate.py',
           'jeffrey/segment_dp/pipeline3.py', 'jeffrey/segment_dp/run3.py', 'jeffrey/segment_dp/launch.py', 'launch.py', 'method', 'runtime_manifest.json',
           'experiments/cabinet/evaluate.py', 'experiments/transition_wall/evaluate.py']
EDGES = ','.join(f'{0.08*i:g}' for i in range(21))
BANK = ['source_q.npy', 'successor_q.npy', 'local_cost.npy', 'safe.npy', 'action_step_size.npy', 'manifest.json']


def environment():
    if os.environ.get('BELLMAN600_BACKEND', 'docker') == 'host':
        return dict(backend='host-bwrap', python=HOST_PYTHON, pinned_image_id=IMAGE, same_image_as_pinned=False)
    image = os.environ.get('BELLMAN600_DERIVED_IMAGE', IMAGE)
    used = subprocess.run(['docker', '--context', 'default', 'image', 'inspect', '--format', '{{.Id}}', image], capture_output=True, text=True).stdout.strip()
    return dict(backend='docker', docker_context='default', docker_image=image, docker_image_id=used, pinned_image_id=IMAGE, same_image_as_pinned=used == IMAGE)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--scene', required=True, help='nowall, cabinet or a cabinet-task scene (cabinet_plus10, cabinet_clutter, ...; pass its pipeline run as --prepared-run)'); p.add_argument('--gpu')
    p.add_argument('--prepared-run', default='registered_20261019T034213Z-f9c501f9', help='cabinet: run whose prepared/ speed and pool to use')
    p.add_argument('--trainer', choices=['dp3'], default='dp3'); p.add_argument('--weights', default='6,2,2')
    p.add_argument('--mode', choices=['specialists', 'generalist'], default='specialists'); p.add_argument('--arch', default='pair', help='pair | offset | pair-hH[-iI]')
    p.add_argument('--edges', default=EDGES); p.add_argument('--blend', type=float, default=0.016); p.add_argument('--updates', type=int, default=15620)
    p.add_argument('--base-run'); p.add_argument('--base-models', type=int, default=2); p.add_argument('--pairs', type=int, default=400000)
    p.add_argument('--candidates', type=int, default=384); p.add_argument('--losses', default='dp'); p.add_argument('--loss', default='smoothl1'); p.add_argument('--decoder', default='mult')
    p.add_argument('--w-dp', type=float, default=1.0); p.add_argument('--w-td', type=float, default=1.0); p.add_argument('--w-bell', type=float, default=1.0)
    p.add_argument('--td-step', type=float, default=0.01); p.add_argument('--ball-step', type=float, default=0.05); p.add_argument('--directions', type=int, default=8)
    p.add_argument('--speed-probe', type=float, default=0.04); p.add_argument('--last-segment', type=int, default=20); p.add_argument('--truth-run')
    p.add_argument('--init', choices=['fresh', 'warm'], default='fresh', help='warm: each DP level starts from the previous level\'s weights'); p.add_argument('--new-arch', help="the chain models' architecture (pair-hH[-iI]); default: the decoder's")
    p.add_argument('--normal-weight', type=float, default=0., help='dp3: direction term weight at the DP levels'); p.add_argument('--seed', type=int, default=87358); p.add_argument('--label', default='')
    p.add_argument('--smoke', action='store_true'); p.add_argument('--repeat-reason'); p.add_argument('--prepare-only', action='store_true'); a = p.parse_args()
    import re; assert a.arch in ('pair', 'offset') or re.match(r'^pair-h\d+(-i\d+(\.\d+)?)?$', a.arch), a.arch
    gpus = 'all' if a.gpu is None else f'device={a.gpu}'
    inputs = dict(runtime=str(RELEASE), registry=str(REGISTRY), **{'source_'+s.replace('/', '_'): str(HERE/s) for s in SOURCES})
    if a.scene == 'nowall':
        prepared = HERE/'assets'; inputs['reference'] = str(HERE/'assets/reference'); args = ['--prepared', '/clean/assets']
    else:
        assert a.scene.startswith('cabinet'), a.scene; SCENE, TESTS = scene_paths(a.scene); prepared = HERE/'results'/a.prepared_run/'prepared'
        inputs.update(scene=str(SCENE), scene_tests=str(TESTS))
        args = ['--prepared', f'/clean/{prepared.relative_to(HERE)}', '--scene', f'/clean/{SCENE.relative_to(HERE)}', '--tests', f'/clean/{TESTS.relative_to(HERE)}']
    inputs.update(pool=str(prepared/'bank/source_q.npy'), pool_manifest=str(prepared/'bank/manifest.json'), speed=str(prepared/'speed.pt'))
    config = dict(action='segment-dp', scene=a.scene, prepared=str(prepared.relative_to(HERE)), gpus=gpus, trainer=a.trainer, seed=a.seed, updates_per_model=a.updates,
                  smoke=a.smoke, planning_seeds=[0] if a.smoke else [0, 1, 2], scorer_tilt_deg=180 if a.scene.startswith('cabinet') else 20, transitions_used=False)
    train = ['--seed', str(a.seed)] + ([] if a.trainer == 'bellman' else ['--updates', str(a.updates)])
    if a.trainer == 'bellman':
        weights = [float(x) for x in a.weights.split(',')]; assert len(weights) == 3
        config.update(arch=a.arch, loss_weights=dict(zip(('bellman', 'eikonal', 'direction'), weights)), rounds=5, transitions_used=True, updates_per_model=15620)
        inputs.update({'bank_'+f: str(prepared/'bank'/f) for f in BANK})
        train += ['--arch', a.arch, '--weights', ','.join(f'{w:g}' for w in weights)]
        label = a.label or 'B5'+('' if a.arch == 'pair' else '-off')+('' if weights == [6, 2, 2] else 'w'+'-'.join(f'{w:g}' for w in weights))
    elif a.trainer == 'eikonal':
        config.update(mode=a.mode, arch=a.arch, edges=a.edges, blend=a.blend)
        train += ['--mode', a.mode, '--arch', a.arch, '--edges', a.edges, '--blend', f'{a.blend:g}']
        label = a.label or {'specialists': 'ES', 'generalist': 'EG'}[a.mode]+('' if a.arch == 'pair' else '-'+a.arch)
    else:
        base_run = HERE/'results'/Path(a.base_run).name; base = base_run/'training/field.pt'
        status = json.loads((base_run/'status.json').read_text()); assert status['complete'] and status['settings']['trainer'] == 'eikonal', a.base_run
        assert status['settings']['prepared'] == f'/clean/{prepared.relative_to(HERE)}', 'base field of another scene'
        base_config = json.loads((base_run/'training/config.json').read_text()); assert base_config['mode'] == 'specialists'
        inputs['base_field'] = str(base); losses = a.losses.split(',')
        config.update(base_run=base_run.name, base_models=a.base_models, pairs=a.pairs, candidates=a.candidates, losses=losses, regression_loss=a.loss, decoder=a.decoder,
                      weights=dict(dp=a.w_dp, td=a.w_td, bell=a.w_bell), td_step=a.td_step, ball_step=a.ball_step, directions=a.directions, speed_probe=a.speed_probe,
                      last_segment=a.last_segment, init=a.init, arch=base_config['arch'], new_arch=a.new_arch, edges=base_config['edges'], blend=base_config['blend'], normal_weight=a.normal_weight)
        train += ['--base', f'/clean/{base.relative_to(HERE)}', '--base-models', str(a.base_models), '--pairs', str(a.pairs), '--candidates', str(a.candidates),
                  '--losses', a.losses, '--loss', a.loss, '--decoder', a.decoder, '--w-dp', f'{a.w_dp:g}', '--w-td', f'{a.w_td:g}', '--w-bell', f'{a.w_bell:g}',
                  '--td-step', f'{a.td_step:g}', '--ball-step', f'{a.ball_step:g}', '--directions', str(a.directions), '--speed-probe', f'{a.speed_probe:g}', '--last-segment', str(a.last_segment), '--init', a.init]+(['--new-arch', a.new_arch] if a.new_arch else [])+(['--normal-weight', f'{a.normal_weight:g}'] if a.normal_weight > 0 else [])
        if a.truth_run:
            truth_run = HERE/'results'/Path(a.truth_run).name; truth = truth_run/'truth/truth.npz'
            truth_config = json.loads((truth_run/'truth/truth.json').read_text()); assert truth_config['scene_sha256'] == json.loads((prepared/'bank/manifest.json').read_text())['scene_sha256']
            inputs['truth'] = str(truth); config['truth_run'] = truth_run.name; train += ['--truth', f'/clean/{truth.relative_to(HERE)}']
        label = a.label or ('DP3' + ('' if losses == ['dp'] else '-'+'+'.join(losses)) + ('' if a.loss == 'smoothl1' else f'-{a.loss}') + ('' if a.decoder == 'mult' else f'-{a.decoder}')
                            + ('' if a.candidates == 384 else f'-c{a.candidates}') + ('' if a.last_segment >= 20 else f'-L{a.last_segment}')
                            + ('' if (a.w_dp, a.w_td, a.w_bell) == (1, 1, 1) else f'-w{a.w_dp:g}-{a.w_td:g}-{a.w_bell:g}') + ('' if a.init == 'fresh' else '-warm') + (f'-{a.new_arch}' if a.new_arch else '') + (f'-n{a.normal_weight:g}' if a.normal_weight > 0 else ''))
    if a.seed != 87358 and not a.label: label += f's{a.seed}'
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    spec = dict(name=f'Clean Bellman600 segment-dp3 {a.scene} {label} {tag}'+(' smoke' if a.smoke else ' full'), kind='pipeline', root=str(HERE), config=config,
                environment=environment(), inputs=inputs,
                command=[sys.executable, str(HERE/'jeffrey/segment_dp/launch.py'), '--output', '{output}', '--gpus', gpus, '--', '/clean/jeffrey/segment_dp/pipeline3.py', *args,
                         '--trainer', a.trainer, *(['--smoke'] if a.smoke else []), '--output', '/outputs', '--train-args='+shlex.join(train)],
                required_outputs=['status.json', 'training/field.pt', 'evaluation/evaluation.json'], validation='files')
    directory = HERE/'specs'; directory.mkdir(exist_ok=True); path = directory/f'segment-dp3_{a.scene}_{tag}{"_smoke" if a.smoke else ""}.json'
    path.write_text(json.dumps(spec, indent=2)+'\n')
    if a.prepare_only: print(path); return
    check = subprocess.run([sys.executable, str(REGISTRY), 'check', str(path)], capture_output=True, text=True); (directory/(path.stem+'.check.json')).write_text(check.stdout)
    if check.returncode and not (check.returncode == 3 and a.repeat_reason): raise SystemExit(check.stderr or 'Duplicate experiment; supply a concrete --repeat-reason.')
    command = [sys.executable, str(REGISTRY), 'run', str(path)] + (['--repeat-reason', a.repeat_reason] if a.repeat_reason else [])
    subprocess.run(command, check=True)


if __name__ == '__main__': main()
