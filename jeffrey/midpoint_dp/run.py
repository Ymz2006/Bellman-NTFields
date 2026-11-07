"""Register a midpoint-loss run (ntrl-demo resumed at epoch 2000 with the travel-time midpoint loss) with the
repository's experiment registry.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/midpoint_dp/run.py \
        --prepared results/registered_<prepare id> --base-run results/registered_<ntrl id> --seed 87358 --label "w1 net" \
        --extra="--weight 1 --leaf net"

Outputs go to results/registered_<id>/. Each run executes a frozen copy of the sources (snap/<hash>/, made here) so
the working files can change while runs are in flight. The pipeline runs in the host venv; its source files, the jeffrey/maze_scale
modules it imports, the ntrl-demo model files, the prepared maze files and the resumed checkpoint are declared inputs
(do not edit them while a run is in flight). Nothing here touches the frozen Bellman600 pipeline, assets or baseline.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / 'jeffrey/gibson_embed'))
from run import REGISTRY, environment  # noqa: E402

DIR = 'jeffrey/midpoint_dp'
SOURCES = ['fastnet.py', 'traverse.py', 'tree.py', 'trainer.py', 'pipeline.py', 'closenet.py']
MAZE = ['maze.py', 'plan.py', 'evaluate.py', 'common.py']
NTRL = ['models/metric/model_train_metric.py', 'models/metric/model_network_metric.py', 'models/metric/model_function_metric.py', 'models/data_mlp.py']
PREPARED = ['maze.json', 'goals.npy', 'queries.npz', 'data', 'heldout']
REQUIRED = ['results.json', 'log.txt', 'rows.npz', 'training/checkpoint.pt', 'training/history.json']


def snapshot(src):
    """Copy the pipeline sources from src (default: the working copy) to jeffrey/midpoint_dp/snap/<hash>/ (never edited
    afterwards) so a registered run's declared inputs stay fixed while the working copies change; returns that folder."""
    import shutil
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
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--base-run', type=Path, required=True)
    p.add_argument('--seed', type=int, default=87358)
    p.add_argument('--resume-epoch', type=int, default=2000)
    p.add_argument('--extra', default='', help='extra pipeline arguments (one string)')
    p.add_argument('--label', default='')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--repeat-reason')
    p.add_argument('--src', type=Path, default=HERE / DIR, help='source folder to snapshot (default: the working copy)')
    a = p.parse_args()
    py = sys.executable
    assert (a.prepared / 'results.json').exists(), '--prepared must be a finished prepare run'
    base = json.load(open(a.base_run / 'results.json'))
    assert base['stage'] in ('ntrl', 'fourier') and base['seed'] == a.seed, 'base run must be an ntrl-demo run (ntrl or fourier stage) of this training seed'
    n = json.load(open(a.prepared / 'maze.json'))['n']
    found = sorted((a.base_run / 'training/model').glob(f'*/Model_Epoch_{a.resume_epoch:05d}_*.pt'))
    assert len(found) == 1, 'resume checkpoint'
    ex = a.extra.split()
    early = {}
    if '--tc-mode' in ex and ex[ex.index('--tc-mode') + 1] == 'b':     # (b) reads an early checkpoint of the base run
        e = int(ex[ex.index('--tc-epoch') + 1]) if '--tc-epoch' in ex else 2000
        f2 = sorted((a.base_run / 'training/model').glob(f'*/Model_Epoch_{e:05d}_*.pt'))
        assert len(f2) == 1, f'epoch-{e} checkpoint for --tc-mode b'
        early = dict(tc_checkpoint=str(f2[0].resolve()))
    snap = snapshot(a.src.resolve())
    inputs = dict(registry=str(REGISTRY), resume_checkpoint=str(found[0].resolve()), base_results=str((a.base_run / 'results.json').resolve()), **early,
                  **{'source_' + s: str(snap / s) for s in SOURCES},
                  **{'maze_' + s: str(HERE / 'jeffrey/maze_scale' / s) for s in MAZE},
                  **{'ntrl_' + s.replace('/', '_'): str(HERE / 'baselines/ntrl-demo' / s) for s in NTRL},
                  **{'prepared_' + f.replace('/', '_'): str(a.prepared.resolve() / f) for f in PREPARED})
    config = dict(action='midpoint-dp', n=n, seed=a.seed, prepared=rel(a.prepared), base_run=rel(a.base_run), resume_epoch=a.resume_epoch, b_scale=base.get('b_scale', 1.0),
                  extra=a.extra, smoke=a.smoke, code=snap.name)
    args = ['--prepared', str(a.prepared.resolve()), '--base-run', str(a.base_run.resolve()), '--seed', str(a.seed), '--resume-epoch', str(a.resume_epoch)]
    args += a.extra.split()
    if a.smoke:
        args.append('--smoke')
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    name = f'Midpoint DP n={n} seed {a.seed}' + (f' {a.label}' if a.label else '') + f' {tag}' + (' smoke' if a.smoke else '')
    spec = dict(name=name, kind='pipeline', root=str(HERE), config=config, environment=environment(), inputs=inputs,
                command=[py, str(snap / 'pipeline.py'), *args, '--out', '{output}'], required_outputs=REQUIRED, validation='files')
    directory = HERE / 'specs'
    directory.mkdir(exist_ok=True)
    path = directory / f'midpoint_dp_{tag}{"_smoke" if a.smoke else ""}.json'
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
