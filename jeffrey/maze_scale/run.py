"""Register a maze-scale run (ntrl-demo on n x n mazes, SE(2) robot) with the repository's experiment registry.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/run.py prepare --n 13
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/run.py ntrl --prepared results/registered_<id> --seed 87358
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/maze_scale/run.py segdp --prepared ... --base ... [segdp.py options]

Outputs go to results/registered_<id>/. The pipelines run in the host venv; their source files, the ntrl-demo model
files they import and the files they read from earlier runs are declared inputs (do not edit them while a run is in
flight). Nothing here touches the frozen Bellman600 pipeline, assets or baseline.
"""
import argparse
import hashlib
import os
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / 'jeffrey/gibson_embed'))
from run import REGISTRY, environment  # noqa: E402

DIR = 'jeffrey/maze_scale'
COMMON = ['maze.py', 'data.py', 'truth.py', 'plan.py', 'common.py']   # run.py is not an input (it gains stages later)
STAGE_SOURCES = dict(prepare=COMMON + ['prepare.py'], ntrl=COMMON + ['train.py', 'evaluate.py', 'ntrl.py'],
                     segdp=COMMON + ['train.py', 'evaluate.py', 'fields.py', 'segdp.py'],
                     zoom=COMMON + ['train.py', 'evaluate.py', 'zoomfield.py', 'zoom.py'],
                     segdp2=COMMON + ['evaluate.py', 'fields.py', 'segdp.py', 'zoomfield.py', 'segdp2.py'],
                     segdp3=COMMON + ['evaluate.py', 'fields.py', 'segdp.py', 'segdp3.py'],
                     fourier=COMMON + ['train.py', 'evaluate.py', 'fourier.py'],
                     uturn=COMMON + ['uturn.py'],
                     plusprepare=COMMON + ['plus.py', 'prepare.py', 'plusprepare.py'],
                     plusntrl=COMMON + ['plus.py', 'train.py', 'evaluate.py', 'ntrl.py', 'plusntrl.py'],
                     uturneval=COMMON + ['fields.py', 'zoomfield.py', 'uturn.py', 'uturneval.py'])
NTRL = ['models/metric/model_train_metric.py', 'models/metric/model_network_metric.py', 'models/metric/model_function_metric.py', 'models/data_mlp.py']
PREPARED_FILES = dict(prepare=[], ntrl=['maze.json', 'goals.npy', 'queries.npz', 'data', 'heldout'],
                      segdp=['maze.json', 'goals.npy', 'queries.npz', 'heldout', 'pool.npy', 'truth_fields.npy', 'results.json'],
                      zoom=['maze.json', 'goals.npy', 'queries.npz', 'data', 'heldout'],
                      segdp2=['maze.json', 'goals.npy', 'queries.npz', 'heldout', 'pool.npy', 'truth_fields.npy', 'results.json'],
                      segdp3=['maze.json', 'goals.npy', 'queries.npz', 'heldout', 'pool.npy', 'truth_fields.npy', 'results.json'],
                      fourier=['maze.json', 'goals.npy', 'queries.npz', 'data', 'heldout'],
                      uturn=['maze.json', 'results.json'],
                      plusprepare=[],
                      plusntrl=['maze.json', 'goals.npy', 'queries.npz', 'data', 'heldout'],
                      uturneval=['maze.json', 'uturn.npz', 'results.json'])
REQUIRED = dict(prepare=['results.json', 'log.txt', 'queries.npz', 'truth_fields.npy', 'pool.npy', 'maze.obj', 'data/sampled_points.npy'],
                ntrl=['results.json', 'log.txt', 'rows.npz', 'training/checkpoint.pt'],
                segdp=['results.json', 'log.txt', 'rows.npz', 'chain/field.pt'],
                zoom=['results.json', 'log.txt', 'rows.npz', 'training/checkpoint.pt'],
                segdp2=['results.json', 'log.txt', 'rows.npz', 'chain/field.pt'],
                segdp3=['results.json', 'log.txt', 'rows.npz', 'chain/field.pt'],
                fourier=['results.json', 'log.txt', 'rows.npz', 'training/checkpoint.pt'],
                uturn=['results.json', 'log.txt', 'uturn.npz', 'maze.json', 'truth_paths.npy'],
                plusprepare=['results.json', 'log.txt', 'queries.npz', 'truth_fields.npy', 'pool.npy', 'maze.obj', 'data/sampled_points.npy'],
                plusntrl=['results.json', 'log.txt', 'rows.npz', 'training/checkpoint.pt'],
                uturneval=['results.json', 'log.txt'])


def rel(p):
    p = Path(p).resolve()
    return str(p.relative_to(HERE)) if p.is_relative_to(HERE) else str(p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=list(STAGE_SOURCES))
    p.add_argument('--n', type=int)
    p.add_argument('--prepared', type=Path)
    p.add_argument('--seed', type=int, default=87358)
    p.add_argument('--base', type=Path, help='segdp: base checkpoint (ntrl-demo checkpoint.pt)')
    p.add_argument('--zoom', type=float, help='zoom: coordinate scale of the training pairs')
    p.add_argument('--b-scale', type=float, help='fourier: factor on ntrl-demo\'s Fourier matrix B')
    p.add_argument('--runs', default='', help='uturneval: comma-separated registered run directories (fields to plan with)')
    p.add_argument('--extra', default='', help='extra pipeline arguments (one string)')
    p.add_argument('--label', default='')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--repeat-reason')
    a = p.parse_args()
    py = sys.executable
    inputs = dict(registry=str(REGISTRY), **{'source_' + s: str(HERE / DIR / s) for s in STAGE_SOURCES[a.stage]})
    if a.stage in ('ntrl', 'segdp', 'zoom', 'segdp2', 'segdp3', 'fourier', 'uturneval', 'plusntrl'):
        inputs.update({'ntrl_' + s.replace('/', '_'): str(HERE / 'baselines/ntrl-demo' / s) for s in NTRL})
    config = dict(action='maze-scale-' + a.stage, smoke=a.smoke, extra=a.extra)
    if a.stage in ('prepare', 'plusprepare'):
        assert a.n, '--n'
        args = ['--n', str(a.n)]
        config['n'] = a.n
        name = f'Maze scale {a.stage} n={a.n}'
    else:
        assert a.prepared and (a.prepared / 'results.json').exists(), '--prepared must be a finished prepare run'
        n = json.load(open(a.prepared / 'maze.json'))['n']
        inputs.update({'prepared_' + f.replace('/', '_'): str(a.prepared.resolve() / f) for f in PREPARED_FILES[a.stage]})
        args = ['--prepared', str(a.prepared.resolve()), '--seed', str(a.seed)]
        config.update(n=n, seed=a.seed, prepared=rel(a.prepared))
        if a.stage in ('zoom', 'segdp2'):
            assert a.zoom, '--zoom'
            args += ['--zoom', repr(a.zoom)]
            config['zoom'] = a.zoom
        if a.stage == 'uturneval':
            runs = [Path(r).resolve() for r in a.runs.split(',') if r]
            assert runs, '--runs'
            for r in runs:
                d = json.load(open(r / 'results.json'))
                rid = r.name.replace('registered_', '')
                inputs[f'run_{rid}_results'] = str(r / 'results.json')
                if d['stage'] == 'segdp':
                    inputs[f'run_{rid}_chain'] = str(r / 'chain/field.pt')
                    inputs[f'run_{rid}_base'] = str(Path(d['base']).resolve())
                else:
                    inputs[f'run_{rid}_checkpoint'] = str(r / 'training/checkpoint.pt')
            args += ['--runs', ','.join(map(str, runs))]
            config['runs'] = sorted(r.name.replace('registered_', '') for r in runs)
        if a.stage == 'fourier':
            assert a.b_scale, '--b-scale'
            args += ['--b-scale', repr(a.b_scale)]
            config['b_scale'] = a.b_scale
        if a.stage in ('segdp', 'segdp2', 'segdp3'):
            assert a.base and a.base.exists(), '--base'
            inputs['base_checkpoint'] = str(a.base.resolve())
            args += ['--base', str(a.base.resolve())]
            config['base'] = rel(a.base)
        name = f'Maze scale {a.stage} n={n} seed {a.seed}' + (f' zoom {a.zoom:g}' if a.stage in ('zoom', 'segdp2') else '') + (f' B x{a.b_scale:g}' if a.stage == 'fourier' else '') + (f' {a.label}' if a.label else '')
    args += a.extra.split()
    if a.smoke:
        args.append('--smoke')
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    spec = dict(name=f'{name} {tag}' + (' smoke' if a.smoke else ''), kind='pipeline', root=str(HERE), config=config, environment=environment(), inputs=inputs,
                command=[py, str(HERE / DIR / f'{a.stage}.py'), *args, '--out', '{output}'], required_outputs=REQUIRED[a.stage], validation='files')
    directory = HERE / 'specs'
    directory.mkdir(exist_ok=True)
    path = directory / f'maze_scale_{a.stage}_{tag}{"_smoke" if a.smoke else ""}.json'
    path.write_text(json.dumps(spec, indent=2) + '\n')
    if a.prepare_only:
        print(path)
        return
    check = subprocess.run([py, str(REGISTRY), 'check', str(path)], capture_output=True, text=True)
    (directory / (path.stem + '.check.json')).write_text(check.stdout)
    if check.returncode and not (check.returncode == 3 and a.repeat_reason):
        raise SystemExit(check.stderr or 'Duplicate experiment; supply a concrete --repeat-reason.')
    command = [py, str(REGISTRY), 'run', str(path)] + (['--repeat-reason', a.repeat_reason] if a.repeat_reason else [])
    sys.stdout.flush()
    os.execv(py, command)          # replace this process (which imported torch for environment()): ~0.5 GB less per queued run


if __name__ == '__main__':
    main()
