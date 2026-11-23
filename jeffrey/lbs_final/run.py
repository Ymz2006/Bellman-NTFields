"""Register a stage of the LBS final benchmark with the repository's experiment registry.

    python jeffrey/lbs_final/run.py prep  --kind maze --n 14 --instance 2              (GPU; host venv)
    python jeffrey/lbs_final/run.py prep  --kind gap --scale 8 --instance 1
    python jeffrey/lbs_final/run.py prep  --kind house3 --name Onaga
    python jeffrey/lbs_final/run.py field --prepared results/registered_<prep> --label lbs --extra "--lift time --bell 1 --bell-long 1 --lean"
    ~/.venvs/ompl-bench/bin/python jeffrey/lbs_final/run.py bench --prepared <prep> --methods rrtc,lazyprm
    ~/.venvs/ompl-bench/bin/python jeffrey/lbs_final/run.py bench --prepared <prep> --methods ntrl,lbs --field ntrl=<run> --field lbs=<run>

Each run executes frozen copies: this study's sources in jeffrey/lbs_final/snap/<hash>/ and jeffrey/lbs's in jeffrey/lbs/snap/<hash>/
(jeffrey/lbs/run.py's snapshot, which also checks its verbatim copies). The earlier studies' modules the stages import are declared inputs.
~/.venvs/ompl-bench is a venv whose only package is ompl 2.0.1; a .pth file adds the host venv's site-packages (torch, numpy).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / 'jeffrey/lbs_final'
OWN = ['seeds.py', 'prep.py', 'prepare3r.py', 'field.py', 'bench.py']
_spec = importlib.util.spec_from_file_location('lbs_run', ROOT / 'jeffrey/lbs/run.py')
LR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LR)                       # puts gibson_embed on sys.path: REGISTRY, environment
REGISTRY, environment = LR.REGISTRY, LR.environment
MAZE = LR.MAZE + ['prepare.py', 'data.py']
EMBED = LR.EMBED + ['prepare.py']
CROSS = ['gap.py', 'gapprepare.py', 'truthfast.py', 'dijkstra_full.c']
GIBSON = Path(os.environ.get('GIBSON_TINY', str(Path.home() / 'Downloads/gibson_tiny')))


def snapshot():
    h = hashlib.sha256()
    for s in OWN:
        h.update(s.encode() + b'\0' + (DIR / s).read_bytes())
    folder = DIR / 'snap' / h.hexdigest()[:12]
    if not folder.exists():
        tmp = folder.with_name(folder.name + '.tmp')
        tmp.mkdir(parents=True, exist_ok=True)
        for s in OWN:
            shutil.copyfile(DIR / s, tmp / s)
        tmp.rename(folder)
    for s in OWN:
        assert (folder / s).read_bytes() == (DIR / s).read_bytes(), f'snapshot {folder} differs in {s}'
    return folder


def rel(p):
    p = Path(p).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def common_inputs(snap, lsnap):
    return dict(registry=str(REGISTRY), **{'own_' + s: str(snap / s) for s in OWN}, **{'lbs_' + s: str(lsnap / s) for s in LR.SOURCES},
                **{'maze_' + s: str(ROOT / 'jeffrey/maze_scale' / s) for s in MAZE}, **{'embed_' + s: str(ROOT / 'jeffrey/embed_warp' / s) for s in EMBED},
                **{'cross_' + s: str(ROOT / 'jeffrey/cross_warp' / s) for s in CROSS}, gibson_grid=str(ROOT / 'jeffrey/gibson_embed/grid.py'))


def prepared_inputs(prepared, files):
    out = {}
    for f in files:
        if (prepared / f).exists():
            out['prepared_' + f.replace('/', '_')] = str(prepared / f)
    return out


def submit(spec, a, py):
    tag = hashlib.sha256(json.dumps(spec['config'], sort_keys=True).encode()).hexdigest()[:10]
    spec['name'] = spec['name'] + f' {tag}' + (' smoke' if a.smoke else '')
    directory = ROOT / 'specs'
    directory.mkdir(exist_ok=True)
    path = directory / f'lbsfinal_{spec["config"]["action"]}_{tag}{"_smoke" if a.smoke else ""}.json'
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prep', 'field', 'bench'])
    p.add_argument('--kind', choices=['maze', 'gap', 'house3'])
    p.add_argument('--n', type=int)
    p.add_argument('--scale', type=int)
    p.add_argument('--instance', type=int, default=1)
    p.add_argument('--name')
    p.add_argument('--prepared', type=Path)
    p.add_argument('--seed', type=int, default=87358)
    p.add_argument('--label', default='')
    p.add_argument('--extra', default='')
    p.add_argument('--methods', default='')
    p.add_argument('--field', action='append', default=[])
    p.add_argument('--time-limit', type=float, default=10.0)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--repeat-reason')
    a = p.parse_args()
    py = sys.executable
    snap = snapshot()
    lsnap = LR.snapshot(ROOT / 'jeffrey/lbs')
    inputs = common_inputs(snap, lsnap)
    smoke = ['--smoke'] if a.smoke else []
    if a.stage == 'prep':
        if a.kind == 'maze':
            args, what = ['maze', '--n', str(a.n), '--instance', str(a.instance)], f'maze {a.n} x {a.n} instance {a.instance}'
        elif a.kind == 'gap':
            args, what = ['gap', '--scale', str(a.scale), '--instance', str(a.instance)], f'gap maze x{a.scale} instance {a.instance}'
        else:
            args, what = ['house3', '--name', a.name], f'3-D house {a.name}'
            inputs.update(mesh=str(GIBSON / a.name / 'mesh_z_up.obj'), cameras=str(GIBSON / a.name / 'camera_poses.csv'))
        config = dict(action='prep', kind=a.kind, n=a.n, scale=a.scale, instance=a.instance, name=a.name, smoke=a.smoke, code=snap.name, lbs_code=lsnap.name)
        req = ['results.json', 'log.txt', 'maze.json', 'queries.npz', 'truth_paths.npy', 'goals.npy', 'data/sampled_points.npy'] + (['house3_table.npy'] if a.kind == 'house3' else [])
        spec = dict(name=f'LBS final prep {what}', kind='pipeline', root=str(ROOT), config=config, environment=environment(), inputs=inputs,
                    command=[py, str(snap / 'prep.py'), *args, '--lbs-src', str(lsnap), *smoke, '--out', '{output}'], required_outputs=req, validation='files')
        return submit(spec, a, py)
    prepared = a.prepared.resolve()
    assert (prepared / 'results.json').exists(), '--prepared must be a finished prepare run'
    info = json.load(open(prepared / 'maze.json'))
    if a.stage == 'field':
        inputs.update(prepared_inputs(prepared, LR.PREPARED + ['house3_table.npy']))
        inputs.update({'ntrl_' + s.replace('/', '_'): str(ROOT / 'baselines/ntrl-demo' / s) for s in LR.NTRL})
        sc = info.get('scene', {})
        if sc.get('kind') == 'house':
            inputs['mesh'] = str(GIBSON / sc['name'] / 'mesh_z_up.obj')
        config = dict(action='field', label=a.label, extra=a.extra, seed=a.seed, prepared=rel(prepared), smoke=a.smoke, code=snap.name, lbs_code=lsnap.name)
        spec = dict(name=f'LBS final field {a.label} {prepared.name}', kind='pipeline', root=str(ROOT), config=config, environment=environment(), inputs=inputs,
                    command=[py, str(snap / 'field.py'), '--lbs-src', str(lsnap), '--prepared', str(prepared), '--seed', str(a.seed), *a.extra.split(), *smoke, '--out', '{output}'],
                    required_outputs=LR.REQUIRED, validation='files')
        return submit(spec, a, py)
    inputs.update(prepared_inputs(prepared, ['maze.json', 'queries.npz', 'house3_table.npy']))
    fields = []
    for f in a.field:
        m, run = f.split('=', 1)
        run = Path(run).resolve()
        inputs[f'field_{m}_checkpoint'] = str(run / 'training/checkpoint.pt')
        inputs[f'field_{m}_results'] = str(run / 'results.json')
        fields += ['--field', f'{m}={run}']
    try:
        import ompl
        ompl_version = getattr(ompl, '__version__', '2.0.1')
    except ImportError:
        ompl_version = None
    config = dict(action='bench', methods=a.methods, fields={f.split('=', 1)[0]: rel(f.split('=', 1)[1]) for f in a.field}, prepared=rel(prepared), time_limit=a.time_limit,
                  threads=a.threads, limit=a.limit, code=snap.name, lbs_code=lsnap.name, ompl=ompl_version)
    spec = dict(name=f'LBS final bench {a.methods} {prepared.name}', kind='pipeline', root=str(ROOT), config=config, environment=environment(), inputs=inputs,
                command=[py, str(snap / 'bench.py'), '--prepared', str(prepared), '--methods', a.methods, *fields, '--lbs-src', str(lsnap), '--time-limit', str(a.time_limit),
                         '--threads', str(a.threads), '--limit', str(a.limit), '--out', '{output}'],
                required_outputs=['bench.json', 'log.txt'], validation='files')
    return submit(spec, a, py)


if __name__ == '__main__':
    main()
