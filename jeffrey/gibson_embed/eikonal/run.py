"""Register an Eikonal-only (P-NTFields recipe) training + planning run with the embedding inputs.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/gibson_embed/eikonal/run.py --scene 0

Outputs go to results/registered_<id>/. Declared inputs: the scene mesh, the P-NTFields samples (sampled_points.npy,
speed.npy, generated with their sampler into datasets_retrain/gibson), their B matrix and pretrained checkpoint, and every
source file. Nothing here touches the frozen Bellman600 pipeline, assets or baseline.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HERE / 'jeffrey/gibson_embed'))
from run import REGISTRY, environment  # noqa: E402

SOURCES = {'eikonal/' + s: HERE / 'jeffrey/gibson_embed/eikonal' / s for s in ['model.py', 'training.py', 'plan.py', 'pipeline.py', 'run.py']}
SOURCES.update({s: HERE / 'jeffrey/gibson_embed' / s for s in ['grid.py', 'diffuse.py', 'truth.py', 'train.py']})
PRETRAINED = 'baselines/P-NTFields/Experiments/Gib_multi/Model_Epoch_10000_ValLoss_1.221157e-01.pt'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', choices=['0', '1'], required=True)
    p.add_argument('--res', type=int, default=128)
    p.add_argument('--coll-res', type=int, default=256)
    p.add_argument('--epochs', type=int, default=10000)
    p.add_argument('--gamma', type=float, default=1e-3)
    p.add_argument('--mode', choices=['jet', 'fd', 'autograd'], default='jet')
    p.add_argument('--n-starts', type=int, default=200)
    p.add_argument('--variants', default='')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--extra', default='', help='extra pipeline arguments (one string)')
    p.add_argument('--label', default='')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--repeat-reason')
    a = p.parse_args()
    mesh = f'baselines/P-NTFields/datasets/gibson/{a.scene}/mesh_z_up_scaled.off'
    data = f'baselines/P-NTFields/datasets_retrain/gibson/{a.scene}'
    args = ['--scene', a.scene, '--res', str(a.res), '--coll-res', str(a.coll_res), '--epochs', str(a.epochs), '--gamma', str(a.gamma),
            '--mode', a.mode, '--n-starts', str(a.n_starts), '--seed', str(a.seed)]
    if a.variants:
        args += ['--variants', a.variants]
    args += a.extra.split()
    config = dict(action='gibson-embed-eikonal', scene=a.scene, res=a.res, coll_res=a.coll_res, epochs=a.epochs, gamma=a.gamma, mode=a.mode,
                  n_starts=a.n_starts, variants=a.variants, seed=a.seed, extra=a.extra, mesh=mesh, data=data, pretrained=PRETRAINED)
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    inputs = dict(registry=str(REGISTRY), mesh=str(HERE / mesh), samples=str(HERE / data / 'sampled_points.npy'), speed=str(HERE / data / 'speed.npy'),
                  B=str(HERE / f'baselines/P-NTFields/datasets/gibson/{a.scene}/B.npy'), pretrained=str(HERE / PRETRAINED),
                  **{'source_' + k.replace('/', '_'): str(v) for k, v in SOURCES.items()})
    label = a.label or f'scene{a.scene} epochs{a.epochs} gamma{a.gamma:g} {a.mode}'
    spec = dict(name=f'Gibson embedding Eikonal planning {label} {tag}', kind='pipeline', root=str(HERE), config=config, environment=environment(),
                inputs=inputs, command=[sys.executable, str(HERE / 'jeffrey/gibson_embed/eikonal/pipeline.py'), *args, '--out', '{output}'],
                required_outputs=['grid.json', 'pairs.npz', 'embeddings.json', 'results.json', 'log.txt'], validation='files')
    directory = HERE / 'specs'
    directory.mkdir(exist_ok=True)
    path = directory / f'gibson-embed-eikonal_scene{a.scene}_{tag}.json'
    path.write_text(json.dumps(spec, indent=2) + '\n')
    if a.prepare_only:
        print(path)
        return
    check = subprocess.run([sys.executable, str(REGISTRY), 'check', str(path)], capture_output=True, text=True)
    (directory / (path.stem + '.check.json')).write_text(check.stdout)
    if check.returncode and not (check.returncode == 3 and a.repeat_reason):
        raise SystemExit(check.stderr or 'Duplicate experiment; supply a concrete --repeat-reason.')
    command = [sys.executable, str(REGISTRY), 'run', str(path)] + (['--repeat-reason', a.repeat_reason] if a.repeat_reason else [])
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
