"""Register a lift + midpoint field run with the repository's experiment registry.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ~/.venvs/bellman-ntfield/bin/python jeffrey/lift_midpoint/run.py \
        --prepared results/registered_<prepare id> --seed 87358 --label liftmid --extra '--lift time'

Outputs go to results/registered_<id>/. Each run executes a frozen copy of this study's sources (snap/<hash>/), so the working files can
change while runs are in flight. lift.py, scenes.py, diag.py are verbatim copies of jeffrey/lift_scale's and fastnet.py, traverse.py,
tree.py of jeffrey/midpoint_dp's (checked byte for byte here). embed_warp's, maze_scale's, cross_warp's gap.py, gibson_embed's grid.py,
ntrl-demo's model files, the prepared files and (for a house) the Gibson mesh are declared inputs. Nothing here touches the frozen
Bellman600 pipeline, assets or baseline.
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

DIR = 'jeffrey/lift_midpoint'
OWN = ['liftfast.py', 'trainer.py', 'pipeline.py']
COPIES = {'lift.py': 'jeffrey/lift_scale/lift.py', 'scenes.py': 'jeffrey/lift_scale/scenes.py', 'diag.py': 'jeffrey/lift_scale/diag.py',
          'fastnet.py': 'jeffrey/midpoint_dp/fastnet.py', 'traverse.py': 'jeffrey/midpoint_dp/traverse.py', 'tree.py': 'jeffrey/midpoint_dp/tree.py'}
SOURCES = OWN + list(COPIES)
EMBED = ['scene.py', 'features.py', 'diagnostics.py']
MAZE = ['maze.py', 'plan.py', 'evaluate.py', 'common.py', 'plus.py', 'truth.py']
CROSS = ['gap.py']
GIBSON = ['grid.py']
NTRL = ['models/metric/model_train_metric.py', 'models/metric/model_network_metric.py', 'models/metric/model_function_metric.py', 'models/data_mlp.py']
PREPARED = ['maze.json', 'goals.npy', 'queries.npz', 'truth_paths.npy', 'truth_fields.npy', 'data', 'heldout']
REQUIRED = ['results.json', 'log.txt', 'rows.npz', 'training/checkpoint.pt', 'training/history.json', 'paths_seed0.npy']


def snapshot(src):
    for name, orig in COPIES.items():
        assert (src / name).read_bytes() == (HERE / orig).read_bytes(), f'{name} is no longer a verbatim copy of {orig}'
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
    p.add_argument('--seed', type=int, default=87358)
    p.add_argument('--label', required=True)
    p.add_argument('--extra', default='')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--repeat-reason')
    a = p.parse_args()
    py = sys.executable
    snap = snapshot(HERE / DIR)
    extra = a.extra.split()
    prepared = a.prepared.resolve()
    assert (prepared / 'results.json').exists(), '--prepared must be a finished prepare run'
    info = json.load(open(prepared / 'maze.json'))
    sc = info.get('scene', {})
    scene = sc.get('name') or sc.get('kind') or info.get('variant', 'plain')
    inputs = dict(registry=str(REGISTRY), **{'source_' + s: str(snap / s) for s in SOURCES}, **{'embed_' + s: str(HERE / 'jeffrey/embed_warp' / s) for s in EMBED},
                  **{'maze_' + s: str(HERE / 'jeffrey/maze_scale' / s) for s in MAZE}, **{'cross_' + s: str(HERE / 'jeffrey/cross_warp' / s) for s in CROSS},
                  **{'gibson_' + s: str(HERE / 'jeffrey/gibson_embed' / s) for s in GIBSON},
                  **{'ntrl_' + s.replace('/', '_'): str(HERE / 'baselines/ntrl-demo' / s) for s in NTRL},
                  **{'prepared_' + f.replace('/', '_'): str(prepared / f) for f in PREPARED})
    if sc.get('kind') == 'house':
        home = Path(os.environ.get('GIBSON_TINY', str(Path.home() / 'Downloads/gibson_tiny')))
        inputs['mesh'] = str(home / sc['name'] / 'mesh_z_up.obj')
    if '--driver' in extra:
        inputs['driver'] = str(Path(extra[extra.index('--driver') + 1]).resolve())
    config = dict(action='lift-midpoint-field', label=a.label, extra=' '.join(extra), smoke=a.smoke, code=snap.name, n=info['n'], scene=scene, prepared=rel(prepared), seed=a.seed)
    args = ['--prepared', str(prepared), '--seed', str(a.seed)] + extra + (['--smoke'] if a.smoke else [])
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    name = f'Lift midpoint field {scene} n={info["n"]} seed {a.seed} {a.label} {tag}' + (' smoke' if a.smoke else '')
    spec = dict(name=name, kind='pipeline', root=str(HERE), config=config, environment=environment(), inputs=inputs,
                command=[py, str(snap / 'pipeline.py'), *args, '--out', '{output}'], required_outputs=REQUIRED, validation='files')
    directory = HERE / 'specs'
    directory.mkdir(exist_ok=True)
    path = directory / f'lift_midpoint_{tag}{"_smoke" if a.smoke else ""}.json'
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
