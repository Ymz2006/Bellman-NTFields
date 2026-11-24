"""Collect what is needed to reuse this study on another machine into one folder: environment settings, every trained model, the
training data of every placement, the code and the run registry.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/arm_placement/export.py --out /workspace/arm_placement_export_2026-11-23

Nothing is trained or planned. Sources: queue/state.json (job -> registered run), results/registered_<id>/, models/, the host venv.
No environment variables are dumped (they hold access tokens).
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VENV = Path(os.path.expanduser('~/.venvs/bellman-ntfield'))


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=ROOT)
    return (r.stdout + r.stderr).strip()


def copy(src, dst, ignore=None):
    if src.is_dir():
        shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def size(p):
    return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    state = json.loads((HERE / 'queue/state.json').read_text())
    jobs = {j['id']: j for j in (json.loads(l) for l in (HERE / 'queue/jobs.jsonl').read_text().splitlines() if l.strip())}
    res = lambda v: ROOT / f'results/registered_{v["run"]}'  # noqa: E731

    # ---------------------------------------------------------------- environment
    env = out / 'environment'
    env.mkdir(exist_ok=True)
    py = VENV / 'bin/python'
    freeze = sh(f'{py} -m pip freeze') if 'No module named pip' not in sh(f'{py} -m pip --version') else sh(f'uv pip freeze --python {py}')
    (env / 'requirements_freeze.txt').write_text(freeze + '\n')
    (env / 'python_and_torch.txt').write_text(sh(f'{py} -c "import sys, torch, numpy; print(\'python\', sys.version); print(\'torch\', torch.__version__); print(\'torch cuda runtime\', torch.version.cuda); '
                                                 f'print(\'cudnn\', torch.backends.cudnn.version()); print(\'numpy\', numpy.__version__); print(\'gpus\', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"') + '\n')
    (env / 'nvidia_smi.txt').write_text(sh('nvidia-smi') + '\n\n' + sh('nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv') + '\n')
    (env / 'machine.txt').write_text('\n'.join([platform.platform(), sh('cat /etc/os-release | head -4'), 'cpu threads: ' + sh('nproc'), sh('free -g | head -2'), 'venv: ' + str(VENV),
                                                'repository commit: ' + sh('git rev-parse HEAD'), 'baselines/ntrl-demo commit: ' + sh('git -C baselines/ntrl-demo rev-parse HEAD')]) + '\n')
    caps = sh("vast-capabilities | jq '{image: .image, hardware: .hardware}'")
    if caps.startswith('{'):
        (env / 'vast_image_and_hardware.json').write_text(caps + '\n')
    for f in ('environment/local/README.md', 'COLLEAGUE_QUICKSTART.md', 'environment/requirements-gui.txt'):
        if (ROOT / f).exists():
            copy(ROOT / f, env / 'repository' / f)
    (env / 'job_settings.json').write_text(json.dumps(dict(
        per_job_environment=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='<one GPU index>', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TORCHINDUCTOR_COMPILE_THREADS='1'),
        scheduler='jeffrey/arm_placement/queue/sched.py', gpu_load_caps=json.loads((HERE / 'queue/caps.json').read_text()),
        registry='tools/experiment_registry/registry.py, database in registry/ of this export (default location ~/.local/share/ntfields-experiments, override with NTFIELDS_REGISTRY)',
        launcher='jeffrey/arm_placement/run.py <tests|prepare|field|euclid|evalfield|demo|reach> ...'), indent=1) + '\n')

    # ---------------------------------------------------------------- models
    copy(HERE / 'models', out / 'models/weights')
    index, missing = [], []
    for jid, v in sorted(state.items()):
        if v.get('mode') != 'field':
            continue
        if v['status'] != 'done' or not (res(v) / 'training/checkpoint.pt').exists():
            missing.append(dict(job=jid, status=v['status']))
            continue
        d = out / 'models/checkpoints' / jid
        copy(res(v) / 'training/checkpoint.pt', d / 'checkpoint.pt')
        for f in ('training/history.json', 'results.json', 'log.txt'):
            if (res(v) / f).exists():
                copy(res(v) / f, d / Path(f).name)
        index.append(dict(job=jid, run=v['run'], args=v.get('args'), prepared_job=next((x for x in jobs.get(jid, {}).get('deps', []) if x.startswith('prep')), None)))
    (out / 'models/checkpoints/index.json').write_text(json.dumps(dict(trained=index, not_finished=missing), indent=1) + '\n')

    # ---------------------------------------------------------------- training data
    tindex = []
    for jid, v in sorted(state.items()):
        if v.get('mode') not in ('prepare', 'tests') or v['status'] != 'done':
            continue
        d = out / 'training_data' / ('prepared' if v['mode'] == 'prepare' else 'test_suites') / jid
        copy(res(v), d)
        tindex.append(dict(job=jid, kind=v['mode'], run=v['run'], args=v.get('args'), megabytes=round(size(d) / 1e6, 1)))
    (out / 'training_data/index.json').write_text(json.dumps(tindex, indent=1) + '\n')

    # ---------------------------------------------------------------- code and registry
    ig = shutil.ignore_patterns('__pycache__', 'models', 'logs', '*.pyc')
    copy(HERE, out / 'code/jeffrey/arm_placement', ig)
    copy(ROOT / 'jeffrey/lbs_arm', out / 'code/jeffrey/lbs_arm', shutil.ignore_patterns('__pycache__', '*.pyc', 'results*', 'snap'))
    copy(ROOT / 'tools/experiment_registry', out / 'code/tools/experiment_registry', shutil.ignore_patterns('__pycache__', '*.pyc'))
    reg = Path(os.environ.get('NTFIELDS_REGISTRY', os.path.expanduser('~/.local/share/ntfields-experiments')))
    if reg.exists():
        copy(reg, out / 'registry')
    sizes = {d.name: round(size(d) / 1e6) for d in sorted(out.iterdir()) if d.is_dir()}
    (out / 'sizes_mb.json').write_text(json.dumps(sizes, indent=1) + '\n')
    print(json.dumps(dict(models=len(index), not_finished=len(missing), training_sets=len(tindex), megabytes=sizes)))


if __name__ == '__main__':
    sys.exit(main())
