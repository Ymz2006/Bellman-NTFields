"""Register a stage of the LBS inference-time study with the repository's experiment registry.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/lbs_fast/run.py bench --scene maze22_1 --device cpu --threads 2 --label ladder \
        --variant 'ref|v=ref' --variant 'fast|c=default' --limit 100
    ~/.venvs/bellman-ntfield/bin/python jeffrey/lbs_fast/run.py bench --scene maze22_1 --device cuda --gpu 1 --variant 'g8|c=ro,k=8,n=bulk'
    ~/.venvs/bellman-ntfield/bin/python jeffrey/lbs_fast/run.py dist --scene maze22_1 --gpu 0 --cfg '{"kind": "grid"}'
    ... bench --dist grid=<dist run id> --variant 'lg|c=ro,k=8,n=bulk,dist=grid'

Scenes are those of jeffrey/lbs_final (scenes.json: scene key -> its prepare run and LBS field run). Each run executes frozen copies:
this study's sources in snap/<hash>/ and jeffrey/lbs's in jeffrey/lbs/snap/<hash>/. seeds.py is a verbatim copy of jeffrey/lbs_final's.
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
DIR = ROOT / 'jeffrey/lbs_fast'
OWN = ['fast.py', 'distnet.py', 'bench.py', 'dist_train.py', 'seeds.py', 'native.py', 'native.cpp']
COPIES = {'seeds.py': 'jeffrey/lbs_final/seeds.py'}
_spec = importlib.util.spec_from_file_location('lbs_run', ROOT / 'jeffrey/lbs/run.py')
LR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LR)
REGISTRY = LR.REGISTRY
MAZE = LR.MAZE
EMBED = LR.EMBED
SCENES = json.load(open(DIR / 'scenes.json'))


def snapshot():
    for name, orig in COPIES.items():
        assert (DIR / name).read_bytes() == (ROOT / orig).read_bytes(), f'{name} is no longer a verbatim copy of {orig}'
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


def run_dir(rid):
    return ROOT / 'results' / f'registered_{rid}'


def submit(spec, a, py, prefix):
    tag = hashlib.sha256(json.dumps(spec['config'], sort_keys=True).encode()).hexdigest()[:10]
    spec['name'] = spec['name'] + f' {tag}'
    directory = ROOT / 'specs'
    directory.mkdir(exist_ok=True)
    path = directory / f'lbsfast_{prefix}_{tag}.json'
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
    p.add_argument('stage', choices=['bench', 'dist'])
    p.add_argument('--scene', required=True)
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--gpu', type=int, default=1, help='PCI index of the GPU (1 = RTX 3090, 0 = RTX 5070)')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--variant', action='append', default=[])
    p.add_argument('--dist', action='append', default=[], help='name=<registered dist run id>')
    p.add_argument('--cfg', default='{}')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--label', default='')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--repeat-reason')
    a = p.parse_args()
    py = sys.executable
    sc = SCENES[a.scene]
    prepared, field = run_dir(sc['prep']), run_dir(sc['field'])
    snap = snapshot()
    lsnap = LR.snapshot(ROOT / 'jeffrey/lbs')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    if a.stage == 'dist' or a.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    inputs = dict(registry=str(REGISTRY), **{'own_' + s: str(snap / s) for s in OWN}, **{'lbs_' + s: str(lsnap / s) for s in LR.SOURCES},
                  **{'maze_' + s: str(ROOT / 'jeffrey/maze_scale' / s) for s in MAZE}, **{'embed_' + s: str(ROOT / 'jeffrey/embed_warp' / s) for s in EMBED},
                  cross_gap=str(ROOT / 'jeffrey/cross_warp/gap.py'), field_checkpoint=str(field / 'training/checkpoint.pt'), field_results=str(field / 'results.json'))
    for f in ('maze.json', 'queries.npz', 'house3_table.npy'):
        if (prepared / f).exists():
            inputs['prepared_' + f] = str(prepared / f)
    env = LR.environment()
    base = [py, None, '--prepared', str(prepared), '--field', str(field), '--lbs-src', str(lsnap)]
    if a.stage == 'dist':
        cfg = json.loads(a.cfg)
        config = dict(action='dist', scene=a.scene, cfg=cfg, prepared=rel(prepared), field=rel(field), code=snap.name, lbs_code=lsnap.name, label=a.label)
        base[1] = str(snap / 'dist_train.py')
        spec = dict(name=f'LBS fast dist {a.scene} {cfg.get("kind", "mlp")}', kind='pipeline', root=str(ROOT), config=config, environment=env, inputs=inputs,
                    command=base + ['--cfg', json.dumps(cfg), '--device', 'cuda:0', '--out', '{output}'], required_outputs=['results.json', 'log.txt', 'dist.pt'], validation='files')
        return submit(spec, a, py, 'dist')
    dists = []
    for d in a.dist:
        name, rid = d.split('=', 1)
        inputs[f'dist_{name}'] = str(run_dir(rid) / 'dist.pt')
        dists += ['--dist', f'{name}={run_dir(rid)}']
    dev = 'cuda:0' if a.device == 'cuda' else 'cpu'
    config = dict(action='bench', scene=a.scene, device=a.device, gpu=a.gpu if a.device == 'cuda' else None, threads=a.threads, variants=a.variant, dists=a.dist, limit=a.limit,
                  prepared=rel(prepared), field=rel(field), code=snap.name, lbs_code=lsnap.name, label=a.label)
    base[1] = str(snap / 'bench.py')
    variants = [x for v in a.variant for x in ('--variant', v)]
    spec = dict(name=f'LBS fast bench {a.scene} {a.device} {a.label}', kind='pipeline', root=str(ROOT), config=config, environment=env, inputs=inputs,
                command=base + ['--device', dev, '--threads', str(a.threads), *variants, *dists, '--limit', str(a.limit), '--out', '{output}'],
                required_outputs=['bench.json', 'log.txt'], validation='files')
    return submit(spec, a, py, 'bench')


if __name__ == '__main__':
    main()
