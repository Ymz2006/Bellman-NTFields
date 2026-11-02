"""run.py for pipeline2.py (truth2 ground truth for C-space walls below ~0.015); same arguments.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/parking/run.py --wall 0.05

Outputs go to results/registered_<id>/. The pipeline runs in the host venv; every source file and the ntrl-demo
model files it imports are declared inputs (do not edit them while a run is in flight). Nothing here touches the
frozen Bellman600 pipeline, assets or baseline.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / 'jeffrey/gibson_embed'))
from run import REGISTRY, environment  # noqa: E402

SOURCES = ['scene.py', 'data.py', 'train.py', 'plan.py', 'truth.py', 'truth2.py', 'pipeline.py', 'pipeline2.py', 'run2.py']
NTRL = ['models/metric/model_train_metric.py', 'models/metric/model_network_metric.py', 'models/metric/model_function_metric.py',
        'models/metric_arm/model_train_metric.py', 'models/metric_arm/model_network_metric.py', 'models/metric_arm/model_function_metric.py',
        'models/data_mlp.py']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wall', type=float, required=True)
    p.add_argument('--seeds', default='87358,1,2')
    p.add_argument('--variant', choices=['metric', 'metric_arm'], default='metric')
    p.add_argument('--epochs', type=int, default=5000)
    p.add_argument('--n-data', type=int, default=500_000)
    p.add_argument('--car-width', type=float, default=0.13)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--repeat-reason')
    a = p.parse_args()
    args = ['--wall', repr(a.wall), '--seeds', a.seeds, '--variant', a.variant, '--epochs', str(a.epochs), '--n-data', str(a.n_data), '--car-width', repr(a.car_width)]
    if a.smoke:
        args.append('--smoke')
    config = dict(action='parking-thin-wall', truth_version=2, wall=a.wall, car_width=a.car_width, seeds=a.seeds, variant=a.variant, epochs=a.epochs, n_data=a.n_data, smoke=a.smoke,
                  planning_seeds=[0, 1, 2])
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    inputs = dict(registry=str(REGISTRY), **{'source_' + s: str(HERE / 'jeffrey/parking' / s) for s in SOURCES},
                  **{'ntrl_' + s.replace('/', '_'): str(HERE / 'baselines/ntrl-demo' / s) for s in NTRL})
    spec = dict(name=f'Parking thin wall (truth v2) ntrl-demo {a.variant} w={a.wall:g} car_w={a.car_width:g} seeds {a.seeds} {tag}' + (' smoke' if a.smoke else ''), kind='pipeline',
                root=str(HERE), config=config, environment=environment(), inputs=inputs,
                command=[sys.executable, str(HERE / 'jeffrey/parking/pipeline2.py'), *args, '--out', '{output}'],
                required_outputs=['results.json', 'log.txt', 'queries.npz', 'truth.npz'], validation='files')
    directory = HERE / 'specs'
    directory.mkdir(exist_ok=True)
    path = directory / f'parking2_{a.variant}_w{a.wall:g}_W{a.car_width:g}_{tag}{"_smoke" if a.smoke else ""}.json'
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
