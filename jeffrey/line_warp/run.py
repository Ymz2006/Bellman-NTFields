"""Register a line-warp run with the repository's experiment registry.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/line_warp/run.py \
        --stage fit --prepared results/registered_<prepare id> --label hard --extra '--kind hard'
    ... --stage field --prepared results/registered_<prepare id> --fit results/registered_<fit id> --seed 87358 --label hard_s4 --extra '--s 4'
    ... --stage evalq | zq --prepared results/registered_<query set> --runs <run dir>,... --label ...

Outputs go to results/registered_<id>/. Each run executes a frozen copy of this study's sources (snap/<hash>/), so working files can
change while runs are in flight. cross_warp's and maze_scale's modules, ntrl-demo's model files and the prepared files are declared
inputs. Nothing here touches the frozen Bellman600 pipeline, assets or baseline.
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

DIR = 'jeffrey/line_warp'
SOURCES = ['warpnet.py', 'lineloss.py', 'obstacles.py', 'fit.py', 'fitrun.py', 'view.py', 'checks.py', 'netwarp.py', 'pipeline.py', 'evalq_net.py', 'zq_net.py', 'zdraw.py']
CROSS = ['warp.py', 'trainer.py', 'diag.py', 'warpplan.py', 'zplan.py', 'zq.py', 'evalq.py', 'gap.py', 'stretch.py', 'gapprepare.py']
MAZE = ['maze.py', 'plan.py', 'evaluate.py', 'common.py', 'plus.py', 'data.py']
NTRL = ['models/metric/model_train_metric.py', 'models/metric/model_network_metric.py', 'models/metric/model_function_metric.py', 'models/data_mlp.py']
PREPARED = ['maze.json', 'goals.npy', 'queries.npz', 'truth_paths.npy', 'data', 'heldout']
REQUIRED = dict(fit=['results.json', 'log.txt', 'phi.pt', 'history.json', 'overview.png'], field=['results.json', 'log.txt', 'rows.npz', 'training/checkpoint.pt', 'paths_seed0.npy'],
                evalq=['results.json', 'log.txt'], zq=['results.json', 'log.txt'], zdraw=['results.json', 'log.txt', 'queries.npz', 'maze.json', 'data/sampled_points.npy', 'heldout/sampled_points.npy'])


def snapshot(src):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=list(REQUIRED), required=True)
    p.add_argument('--prepared', type=Path, required=True, help='fit / field: the prepare run; evalq / zq: the query set')
    p.add_argument('--fit', default='none', help='field / zdraw: the fit run (none: no warp)')
    p.add_argument('--runs', default='', help='evalq / zq: comma-separated registered run directories')
    p.add_argument('--seed', type=int, default=87358)
    p.add_argument('--label', required=True)
    p.add_argument('--extra', default='')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--repeat-reason')
    a = p.parse_args()
    py = sys.executable
    snap = snapshot(HERE / DIR)
    prepared = a.prepared.resolve()
    assert (prepared / 'results.json').exists(), '--prepared must be a finished run'
    info = json.load(open(prepared / 'maze.json'))
    extra = a.extra.split()
    inputs = dict(registry=str(REGISTRY), **{'source_' + s: str(snap / s) for s in SOURCES}, **{'cross_' + s: str(HERE / 'jeffrey/cross_warp' / s) for s in CROSS},
                  **{'maze_' + s: str(HERE / 'jeffrey/maze_scale' / s) for s in MAZE}, prepared_maze=str(prepared / 'maze.json'))
    config = dict(action='line-warp-' + a.stage, n=info['n'], variant=info.get('variant', 'plain'), gap_scale=info.get('gap_scale'), label=a.label, prepared=rel(prepared), extra=' '.join(extra), smoke=a.smoke, code=snap.name)
    args = ['--prepared', str(prepared)]
    script = dict(fit='fitrun.py', field='pipeline.py', evalq='evalq_net.py', zq='zq_net.py', zdraw='zdraw.py')[a.stage]
    if a.stage in ('field', 'zdraw'):
        fit = Path(a.fit).resolve() if a.fit != 'none' else None
        config['fit'] = rel(fit) if fit else 'none'
        args += ['--fit', str(fit) if fit else 'none']
        if fit:
            inputs['fit_phi'] = str(fit / 'phi.pt')
    if a.stage == 'field':
        config['seed'] = a.seed
        args += ['--seed', str(a.seed)]
        inputs.update({'ntrl_' + s.replace('/', '_'): str(HERE / 'baselines/ntrl-demo' / s) for s in NTRL})
        inputs.update({'prepared_' + f.replace('/', '_'): str(prepared / f) for f in PREPARED})
    if a.stage in ('evalq', 'zq'):
        runs = [Path(r).resolve() for r in a.runs.split(',') if r]
        assert runs, '--runs'
        args = ['--queries', str(prepared), '--runs', ','.join(map(str, runs))]
        if info.get('variant') == 'gap' and '--gap-scale' not in extra:
            extra += ['--gap-scale', str(info['gap_scale'])]
            if a.stage == 'zq' and (info.get('base') or {}).get('name'):
                extra += ['--gap-base', info['base']['name']]
        inputs.update({'ntrl_' + s.replace('/', '_'): str(HERE / 'baselines/ntrl-demo' / s) for s in NTRL})
        inputs.update({'queries_' + f: str(prepared / f) for f in ('queries.npz', 'truth_paths.npy', 'goals.npy', 'results.json')})
        for r in runs:
            rid = r.name.replace('registered_', '')
            inputs[f'run_{rid}_results'] = str(r / 'results.json')
            inputs[f'run_{rid}_checkpoint'] = str(r / 'training/checkpoint.pt')
        config['runs'] = sorted(r.name.replace('registered_', '') for r in runs)
    args += extra
    if a.smoke:
        args.append('--smoke')
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    name = f'Line warp {a.stage} n={info["n"]} {info.get("variant", "plain")}{info.get("gap_scale") or ""} {a.label}' + (f' seed {a.seed}' if a.stage == 'field' else '') + f' {tag}' + (' smoke' if a.smoke else '')
    spec = dict(name=name, kind='pipeline', root=str(HERE), config=config, environment=environment(), inputs=inputs, command=[py, str(snap / script), *args, '--out', '{output}'],
                required_outputs=REQUIRED[a.stage], validation='files')
    directory = HERE / 'specs'
    directory.mkdir(exist_ok=True)
    path = directory / f'line_warp_{a.stage}_{tag}{"_smoke" if a.smoke else ""}.json'
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
