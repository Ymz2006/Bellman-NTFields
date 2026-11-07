"""Register a cross-warp run (ntrl-demo with a C-space warp before its encoder, on a jeffrey/maze_scale maze, by default
the plus variant) with the repository's experiment registry.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/cross_warp/run.py \
        --prepared results/registered_<maze_scale plus-prepare id> --seed 87358 --label lf --extra '--xy --learn --free-weight 1'

Outputs go to results/registered_<id>/. Each run executes a frozen copy of this study's sources (snap/<hash>/, made
here), so the working files can change while runs are in flight. The jeffrey/maze_scale modules it imports, the
ntrl-demo model files and the prepared maze files are declared inputs (do not edit them while a run is in flight).
Nothing here touches the frozen Bellman600 pipeline, assets or baseline.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / 'jeffrey/gibson_embed'))
from run import REGISTRY, environment  # noqa: E402

DIR = 'jeffrey/cross_warp'
SOURCES = ['warp.py', 'trainer.py', 'diag.py', 'pipeline.py', 'queries.py', 'evalq.py', 'gap.py', 'truthfast.py', 'dijkstra_full.c', 'gapprepare.py',
           'warpplan.py', 'stretchq.py', 'stepq.py', 'zplan.py', 'zq.py', 'stretch.py']
MAZE = ['maze.py', 'plan.py', 'evaluate.py', 'common.py', 'plus.py']
MAZE_QUERIES = ['maze.py', 'plan.py', 'common.py', 'plus.py', 'truth.py', 'prepare.py', 'data.py']
MAZE_GAP = ['maze.py', 'plan.py', 'common.py', 'plus.py', 'truth.py', 'prepare.py', 'data.py']
NTRL = ['models/metric/model_train_metric.py', 'models/metric/model_network_metric.py', 'models/metric/model_function_metric.py', 'models/data_mlp.py']
PREPARED = ['maze.json', 'goals.npy', 'queries.npz', 'truth_paths.npy', 'data', 'heldout']
REQUIRED = ['results.json', 'log.txt', 'rows.npz', 'training/checkpoint.pt', 'paths_seed0.npy']
STAGE_REQUIRED = dict(queries=['results.json', 'log.txt', 'queries.npz', 'truth_paths.npy', 'goals.npy'], evalq=['results.json', 'log.txt'], stretchq=['results.json', 'log.txt'], stepq=['results.json', 'log.txt'], zq=['results.json', 'log.txt'],
                      gapprepare=['results.json', 'log.txt', 'queries.npz', 'truth_paths.npy', 'goals.npy', 'maze.json', 'data/sampled_points.npy', 'heldout/sampled_points.npy'])


def snapshot(src):
    """Copy the sources to jeffrey/cross_warp/snap/<hash>/ (never edited afterwards); returns that folder."""
    h = hashlib.sha256()
    for s in SOURCES:
        h.update(s.encode() + b'\0' + (src / s).read_bytes())
    folder = HERE / DIR / 'snap' / h.hexdigest()[:12]
    if not folder.exists():
        tmp = folder.with_name(folder.name + '.tmp')
        tmp.mkdir(parents=True, exist_ok=True)
        for s in SOURCES:
            shutil.copyfile(src / s, tmp / s)
        tmp.rename(folder)
    for s in SOURCES:
        assert (folder / s).read_bytes() == (src / s).read_bytes(), f'snapshot {folder} differs in {s}'
    return folder


def rel(p):
    p = Path(p).resolve()
    return str(p.relative_to(HERE)) if p.is_relative_to(HERE) else str(p)


def submit(spec, path, a, py):
    path.write_text(json.dumps(spec, indent=2) + '\n')
    if a.prepare_only:
        print(path)
        return
    check = subprocess.run([py, str(REGISTRY), 'check', str(path)], capture_output=True, text=True)
    (path.parent / (path.stem + '.check.json')).write_text(check.stdout)
    if check.returncode and not (check.returncode == 3 and a.repeat_reason):
        raise SystemExit(check.stderr or 'Duplicate experiment; supply a concrete --repeat-reason.')
    command = [py, str(REGISTRY), 'run', str(path)] + (['--repeat-reason', a.repeat_reason] if a.repeat_reason else [])
    sys.stdout.flush()
    os.execv(py, command)


def gap_base(info, extra):
    """['--gap-base', name] for a gap maze on another base than the gap study's 22 x 22 cross (gap.py BASES)."""
    name = (info.get('base') or {}).get('name')
    return ['--gap-base', name] if name and '--gap-base' not in extra else []


def other_stage(a, snap, info, py):
    """queries: a second query set on the prepared maze; evalq: plan a query set with the fields of registered runs."""
    n = info['n']
    extra = a.extra.split()
    if info.get('variant') == 'gap' and '--gap-scale' not in extra:
        extra += ['--gap-scale', str(info['gap_scale'])] + gap_base(info, extra)
    elif info.get('variant') != 'plus' and '--plus-width' not in extra:
        extra += ['--plus-width', '0']
    inputs = dict(registry=str(REGISTRY), **{'source_' + s: str(snap / s) for s in SOURCES})
    if a.stage == 'queries':
        inputs.update({'maze_' + s: str(HERE / 'jeffrey/maze_scale' / s) for s in MAZE_QUERIES})
        inputs.update({'prepared_' + f: str(a.prepared.resolve() / f) for f in ('maze.json', 'data', 'heldout', 'results.json')})
        args = ['--prepared', str(a.prepared.resolve())] + extra
        config = dict(action='cross-warp-queries', n=n, label=a.label, prepared=rel(a.prepared), extra=' '.join(extra), smoke=a.smoke, code=snap.name)
    else:
        runs = [Path(r).resolve() for r in a.runs.split(',') if r]
        assert runs, '--runs'
        inputs.update({'maze_' + s: str(HERE / 'jeffrey/maze_scale' / s) for s in MAZE})
        inputs.update({'ntrl_' + s.replace('/', '_'): str(HERE / 'baselines/ntrl-demo' / s) for s in NTRL})
        inputs.update({'queries_' + f: str(a.prepared.resolve() / f) for f in ('maze.json', 'queries.npz', 'truth_paths.npy', 'goals.npy', 'results.json')})
        for r in runs:
            rid = r.name.replace('registered_', '')
            inputs[f'run_{rid}_results'] = str(r / 'results.json')
            inputs[f'run_{rid}_checkpoint'] = str(r / 'training/checkpoint.pt')
        args = ['--queries', str(a.prepared.resolve()), '--runs', ','.join(map(str, runs))] + extra
        config = dict(action='cross-warp-' + a.stage, n=n, label=a.label, queries=rel(a.prepared), runs=sorted(r.name.replace('registered_', '') for r in runs),
                      extra=' '.join(extra), smoke=a.smoke, code=snap.name)
    if a.smoke:
        args.append('--smoke')
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    name = f'Cross warp {a.stage} n={n} {a.label} {tag}' + (' smoke' if a.smoke else '')
    script = {'queries': 'queries.py', 'evalq': 'evalq.py', 'stretchq': 'stretchq.py', 'stepq': 'stepq.py', 'zq': 'zq.py'}[a.stage]
    spec = dict(name=name, kind='pipeline', root=str(HERE), config=config, environment=environment(), inputs=inputs,
                command=[py, str(snap / script), *args, '--out', '{output}'], required_outputs=STAGE_REQUIRED[a.stage], validation='files')
    directory = HERE / 'specs'
    directory.mkdir(exist_ok=True)
    submit(spec, directory / f'cross_warp_{a.stage}_{tag}{"_smoke" if a.smoke else ""}.json', a, py)


def gap_stage(a, snap, py):
    """gapprepare: a gap maze (gap.py) with data, truth and region-stratified queries; --prepared is the k = 1 source (optional)."""
    assert a.scale >= 1, '--scale'
    inputs = dict(registry=str(REGISTRY), **{'source_' + s: str(snap / s) for s in SOURCES},
                  **{'maze_' + s: str(HERE / 'jeffrey/maze_scale' / s) for s in MAZE_GAP})
    extra = a.extra.split()
    if '--queries-from' in extra:                   # open-pair control: that run's queries are inputs
        i = extra.index('--queries-from')
        src = Path(extra[i + 1]).resolve()
        extra[i + 1] = str(src)
        inputs.update({'queries_from_' + f: str(src / f) for f in ('maze.json', 'goals.npy', 'queries.npz', 'truth_paths.npy', 'results.json')})
    if '--map-queries' in extra:                    # mapped-query control: the k = 1 query set is an input
        i = extra.index('--map-queries')
        src = Path(extra[i + 1]).resolve()
        extra[i + 1] = str(src)
        inputs.update({'map_queries_' + f: str(src / f) for f in ('maze.json', 'goals.npy', 'queries.npz', 'truth_paths.npy', 'results.json')})
    if '--data-from' in extra:                      # mapped-data control: the source run's pairs are inputs
        i = extra.index('--data-from')
        src = Path(extra[i + 1]).resolve()
        extra[i + 1] = str(src)
        inputs.update({'data_from_' + f.replace('/', '_'): str(src / f) for f in ('maze.json', 'data/sampled_points.npy', 'heldout/sampled_points.npy', 'results.json')})
    args = ['--scale', str(a.scale)] + extra
    config = dict(action='cross-warp-gapprepare', scale=a.scale, label=a.label, extra=' '.join(extra), smoke=a.smoke, code=snap.name)
    if a.prepared is not None and str(a.prepared) not in ('', 'none'):
        inputs.update({'source_prepared_' + f: str(a.prepared.resolve() / f) for f in ('maze.json', 'data', 'heldout', 'results.json')})
        args += ['--source', str(a.prepared.resolve())]
        config['source'] = rel(a.prepared)
    if a.smoke:
        args.append('--smoke')
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    name = f'Cross warp gapprepare k={a.scale} {a.label} {tag}' + (' smoke' if a.smoke else '')
    spec = dict(name=name, kind='pipeline', root=str(HERE), config=config, environment=environment(), inputs=inputs,
                command=[py, str(snap / 'gapprepare.py'), *args, '--out', '{output}'],
                required_outputs=[f for f in STAGE_REQUIRED['gapprepare'] if not (('--source' in args or '--map-queries' in args) and f.split('/')[0] in ('data', 'heldout'))],
                validation='files')
    directory = HERE / 'specs'
    directory.mkdir(exist_ok=True)
    submit(spec, directory / f'cross_warp_gapprepare_{tag}{"_smoke" if a.smoke else ""}.json', a, py)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['warp', 'queries', 'evalq', 'stretchq', 'stepq', 'zq', 'gapprepare'], default='warp')
    p.add_argument('--scale', type=int, default=0, help='gapprepare: gap scale k')
    p.add_argument('--prepared', type=Path, default=None, help='warp / queries: the maze_scale prepare run; evalq: the query set (queries run); gapprepare: optional k = 1 source')
    p.add_argument('--runs', default='', help='evalq: comma-separated registered run directories to plan with')
    p.add_argument('--seed', type=int, default=87358)
    p.add_argument('--label', required=True, help='short variant name (part of the run name and config)')
    p.add_argument('--extra', default='', help='pipeline arguments (one string)')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--repeat-reason')
    p.add_argument('--src', type=Path, default=HERE / DIR, help='source folder to snapshot (default: the working copy)')
    a = p.parse_args()
    py = sys.executable
    snap = snapshot(a.src.resolve())
    if a.stage == 'gapprepare':
        return gap_stage(a, snap, py)
    assert (a.prepared / 'results.json').exists(), '--prepared must be a finished maze_scale prepare run'
    info = json.load(open(a.prepared / 'maze.json'))
    n = info['n']
    if a.stage != 'warp':
        return other_stage(a, snap, info, py)
    inputs = dict(registry=str(REGISTRY), **{'source_' + s: str(snap / s) for s in SOURCES},
                  **{'maze_' + s: str(HERE / 'jeffrey/maze_scale' / s) for s in MAZE},
                  **{'ntrl_' + s.replace('/', '_'): str(HERE / 'baselines/ntrl-demo' / s) for s in NTRL},
                  **{'prepared_' + f.replace('/', '_'): str(a.prepared.resolve() / f) for f in PREPARED})
    extra = a.extra.split()
    if info.get('variant') == 'gap' and '--gap-scale' not in extra:
        extra += ['--gap-scale', str(info['gap_scale'])] + gap_base(info, extra)
        inputs['source_gap_base_plus'] = str(HERE / 'jeffrey/maze_scale/plus.py')
    elif info.get('variant') != 'plus' and '--plus-width' not in extra:
        extra += ['--plus-width', '0']
    if '--warp-from' in extra:                      # stage 2 of a learned warp: the source checkpoint is an input
        k = extra.index('--warp-from')
        src = Path(extra[k + 1]).resolve()
        extra[k + 1] = str(src)
        inputs['warp_from_checkpoint'] = str(src)
    config = dict(action='cross-warp', n=n, variant=info.get('variant', 'plain'), seed=a.seed, label=a.label, prepared=rel(a.prepared),
                  extra=' '.join(extra), smoke=a.smoke, code=snap.name)
    args = ['--prepared', str(a.prepared.resolve()), '--seed', str(a.seed)] + extra
    if a.smoke:
        args.append('--smoke')
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    name = f'Cross warp n={n} {info.get("variant", "plain")} seed {a.seed} {a.label} {tag}' + (' smoke' if a.smoke else '')
    spec = dict(name=name, kind='pipeline', root=str(HERE), config=config, environment=environment(), inputs=inputs,
                command=[py, str(snap / 'pipeline.py'), *args, '--out', '{output}'], required_outputs=REQUIRED, validation='files')
    directory = HERE / 'specs'
    directory.mkdir(exist_ok=True)
    path = directory / f'cross_warp_{tag}{"_smoke" if a.smoke else ""}.json'
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
    os.execv(py, command)


if __name__ == '__main__':
    main()
