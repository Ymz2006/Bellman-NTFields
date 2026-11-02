"""Register a Gibson free-space-averaging embedding run with the repository's experiment registry.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/gibson_embed/run.py --scene 0 --dim 2 --res 512
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/gibson_embed/run.py --scene 0 --dim 3 --res 128

Outputs go to results/registered_<id>/. The pipeline runs in the host venv (torch 2.7.1+cu128); the scene mesh and
every source file are declared inputs. This does not touch the frozen Bellman600 pipeline, assets or baseline.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
REGISTRY = HERE / 'tools/experiment_registry/registry.py'
SOURCES = ['grid.py', 'diffuse.py', 'truth.py', 'probe.py', 'train.py', 'pipeline.py', 'run.py']
MESHES = {'0': 'baselines/P-NTFields/datasets/gibson/0/mesh_z_up_scaled.off', '1': 'baselines/P-NTFields/datasets/gibson/1/mesh_z_up_scaled.off'}


def environment():
    import torch
    return dict(backend='host-venv', python=sys.executable, torch=torch.__version__, cuda=torch.version.cuda, node=platform.node(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, visible=os.environ.get('CUDA_VISIBLE_DEVICES'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', choices=list(MESHES), required=True)
    p.add_argument('--dim', type=int, choices=[2, 3], default=2)
    p.add_argument('--res', type=int, default=512)
    p.add_argument('--z', type=float, default=0.0)
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--sources', type=int, default=96)
    p.add_argument('--test-sources', type=int, default=24)
    p.add_argument('--snapshots', default='8,32,128,512,2048,8192')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--variants', default='')
    p.add_argument('--embeddings', default='')
    p.add_argument('--extra', default='', help='extra pipeline arguments (one string)')
    p.add_argument('--label', default='')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--repeat-reason')
    a = p.parse_args()
    mesh = HERE / MESHES[a.scene]
    args = ['--mesh', str(mesh), '--dim', str(a.dim), '--res', str(a.res), '--z', str(a.z), '--steps', str(a.steps), '--sources', str(a.sources),
            '--test-sources', str(a.test_sources), '--snapshots', a.snapshots, '--seed', str(a.seed)]
    if a.variants:
        args += ['--variants', a.variants]
    if a.embeddings:
        args += ['--embeddings', a.embeddings]
    args += a.extra.split()
    config = dict(action='gibson-embed', scene=a.scene, dim=a.dim, res=a.res, z=a.z, steps=a.steps, sources=a.sources, test_sources=a.test_sources,
                  snapshots=a.snapshots, seed=a.seed, variants=a.variants, embeddings=a.embeddings, extra=a.extra, mesh=MESHES[a.scene])
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    inputs = dict(registry=str(REGISTRY), mesh=str(mesh), **{'source_' + s: str(HERE / 'jeffrey/gibson_embed' / s) for s in SOURCES})
    label = a.label or f'scene{a.scene} {a.dim}d res{a.res}'
    spec = dict(name=f'Gibson free-space ball-average embedding {label} {tag}', kind='pipeline', root=str(HERE), config=config, environment=environment(),
                inputs=inputs, command=[sys.executable, str(HERE / 'jeffrey/gibson_embed/pipeline.py'), *args, '--out', '{output}'],
                required_outputs=['grid.json', 'truth.npz', 'embeddings.json', 'probe.json', 'results.json', 'log.txt'], validation='files')
    directory = HERE / 'specs'
    directory.mkdir(exist_ok=True)
    path = directory / f'gibson-embed_scene{a.scene}_{a.dim}d_{tag}.json'
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
