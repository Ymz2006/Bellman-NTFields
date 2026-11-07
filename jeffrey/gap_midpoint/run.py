"""Register a midpoint-loss run on a gap maze with the repository's experiment registry.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ~/.venvs/bellman-ntfield/bin/python jeffrey/gap_midpoint/run.py \
        --prepared results/registered_<gap_segdp prep id> --base-run results/registered_<ntrl-demo id> --seed 87358 \
        --label main --extra='--weight 1 --leaf min --traverse safe --nodes both'

Outputs go to results/registered_<id>/. Each run executes a frozen copy (snap/<hash>/) of gapmid.py, of jeffrey/midpoint_dp's
pipeline sources and of cross_warp's gap.py, so working files can change while runs are in flight. The jeffrey/maze_scale
modules, the ntrl-demo model files, the prepared files, the base's training data and checkpoint are declared inputs: do
not edit them while a run is in flight.
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

DIR = 'jeffrey/gap_midpoint'
FILES = [('jeffrey/gap_midpoint', 'gapmid.py'), ('jeffrey/cross_warp', 'gap.py')] + [('jeffrey/midpoint_dp', s) for s in ('fastnet.py', 'traverse.py', 'tree.py', 'trainer.py', 'pipeline.py', 'closenet.py')]
MAZE = ['maze.py', 'plus.py', 'plan.py', 'evaluate.py', 'common.py']
NTRL = ['models/metric/model_train_metric.py', 'models/metric/model_network_metric.py', 'models/metric/model_function_metric.py', 'models/data_mlp.py']
PREPARED = ['maze.json', 'goals.npy', 'queries.npz', 'heldout', 'results.json']
REQUIRED = ['results.json', 'log.txt', 'rows.npz', 'training/checkpoint.pt', 'training/history.json']


def snapshot():
    files = [(HERE / d / s, s) for d, s in FILES]
    h = hashlib.sha256()
    for src, s in files:
        h.update(s.encode() + b'\0' + src.read_bytes())
    folder = HERE / DIR / 'snap' / h.hexdigest()[:12]
    if not folder.exists():
        tmp = folder.with_name(folder.name + f'.tmp{os.getpid()}')
        tmp.mkdir(parents=True, exist_ok=True)
        for src, s in files:
            shutil.copyfile(src, tmp / s)
        try:
            tmp.rename(folder)
        except OSError:
            shutil.rmtree(tmp)                    # another launch made it first
    for src, s in files:
        assert (folder / s).read_bytes() == src.read_bytes(), f'snapshot {folder} differs in {s}'
    return folder


def rel(p):
    p = Path(p).resolve()
    return str(p.relative_to(HERE)) if p.is_relative_to(HERE) else str(p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True, help='a jeffrey/gap_segdp prep run')
    p.add_argument('--base-run', type=Path, required=True, help='the gap study\'s ntrl-demo run on this maze and seed')
    p.add_argument('--seed', type=int, default=87358)
    p.add_argument('--resume-epoch', type=int, default=2000)
    p.add_argument('--extra', default='')
    p.add_argument('--label', default='')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--repeat-reason')
    a = p.parse_args()
    py = sys.executable
    pr, base = a.prepared.resolve(), a.base_run.resolve()
    info = json.load(open(pr / 'maze.json'))
    k = info['gap_scale']
    res = json.load(open(base / 'results.json'))
    assert res['seed'] == a.seed, 'base run of another training seed'
    data = Path(res['prepared'])
    data = (data if data.is_absolute() else HERE / data) / 'data'
    found = sorted((base / 'training/model').glob(f'*/Model_Epoch_{a.resume_epoch:05d}_*.pt')) or [base / f'training/checkpoints/epoch_{a.resume_epoch:05d}.pt']
    assert len(found) == 1 and found[0].exists(), 'resume checkpoint'
    snap = snapshot()
    inputs = dict(registry=str(REGISTRY), resume_checkpoint=str(found[0]), base_results=str(base / 'results.json'), base_data=str(data), base_maze=str(data.parent / 'maze.json'),
                  **{'source_' + s: str(snap / s) for _, s in FILES},
                  **{'maze_' + s: str(HERE / 'jeffrey/maze_scale' / s) for s in MAZE},
                  **{'ntrl_' + s.replace('/', '_'): str(HERE / 'baselines/ntrl-demo' / s) for s in NTRL},
                  **{'prepared_' + f: str(pr / f) for f in PREPARED})
    config = dict(action='gap-midpoint', k=k, seed=a.seed, prepared=rel(pr), base_run=rel(base), resume_epoch=a.resume_epoch, extra=a.extra, smoke=a.smoke, code=snap.name)
    args = ['--prepared', str(pr), '--base-run', str(base), '--seed', str(a.seed), '--resume-epoch', str(a.resume_epoch)] + a.extra.split()
    if a.smoke:
        args.append('--smoke')
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    name = f'Gap midpoint k={k} seed {a.seed}' + (f' {a.label}' if a.label else '') + f' {tag}' + (' smoke' if a.smoke else '')
    spec = dict(name=name, kind='pipeline', root=str(HERE), config=config, environment=environment(), inputs=inputs,
                command=[py, str(snap / 'gapmid.py'), *args, '--out', '{output}'], required_outputs=REQUIRED, validation='files')
    directory = HERE / 'specs'
    directory.mkdir(exist_ok=True)
    path = directory / f'gap_midpoint_{tag}{"_smoke" if a.smoke else ""}.json'
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
