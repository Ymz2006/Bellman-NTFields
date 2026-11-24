"""Re-create p_d055_s46_h60 (0.55 m, bases +-0.46 m, 0.60 m high) on this workstation, from the certified suite on, with the exact
snapshots and arguments of the original registered runs (import/arm_placement_export_2026-11-23). Every run is registered through
jeffrey/arm_placement/run.py --snap <original>. Each stage's outputs are hash-compared with the export (logged, not enforced).

    python jeffrey/arm_placement/repro/chain.py --tests-run <id of the registered suite run>     (state in repro/chain_state.json)
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EXP = ROOT / 'import/arm_placement_export_2026-11-23'
PY = os.path.expanduser('~/.venvs/bellman-ntfield/bin/python')
PL = 'p_d055_s46_h60'
STATE = HERE / ('chain_frozen_state.json' if '--frozen' in sys.argv else 'chain_state.json')
LOGS = HERE / ('logs_frozen' if '--frozen' in sys.argv else 'logs')
LOGS.mkdir(exist_ok=True)
SNAP = dict(prepare='1e4eb1266273', euclid='95f10fa79d11', field='a34d6b85dc3f', evalfield='296a34426335', demo='48e71bf53cf2')
state = json.loads(STATE.read_text()) if STATE.exists() else {}


def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def compare(label, mine, theirs, files):
    for f in files:
        a, b = Path(mine) / f, Path(theirs) / f
        log(f'compare {label} {f}: ' + ('identical' if a.exists() and b.exists() and sha(a) == sha(b) else f'DIFFERENT (here {a.exists()}, export {b.exists()})'))


def run(job, mode, args, gpu, mem='8G'):
    if job in state:
        log(f'{job}: already done, run {state[job]}')
        return state[job]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER='PCI_BUS_ID', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TORCHINDUCTOR_COMPILE_THREADS='1')
    out = LOGS / f'{job}.log'
    cmd = ['systemd-run', '--user', '--scope', '-q', '-p', f'MemoryMax={mem}', '-p', 'MemorySwapMax=1G',
           PY, 'jeffrey/arm_placement/run.py', mode, '--snap', SNAP[mode], *args]
    log(f'{job}: start on GPU {gpu}')
    t = time.time()
    with open(out, 'w') as fh:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    rid = None
    for line in out.read_text().splitlines():
        if line.startswith('{"id"'):
            row = json.loads(line)
            if row.get('state') == 'succeeded':
                rid = row['id']
    log(f'{job}: rc {rc}, run {rid}, {(time.time() - t) / 60:.1f} min')
    if rid:
        state[job] = rid
        STATE.write_text(json.dumps(state, indent=1) + '\n')
    return rid


def res(job):
    if state[job] == 'FROZEN':
        return str(EXP / f'training_data/prepared/prep_{PL}')
    return f'results/registered_{state[job]}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tests-run')
    p.add_argument('--frozen', action='store_true', help='skip suite + prepare: train and evaluate on the exported (original) prepared data and queries')
    p.add_argument('--seeds', type=int, nargs='+', default=[87358, 87359])
    p.add_argument('--gpus', default='1,1', help='GPU for the first and the second of two jobs that run side by side')
    a = p.parse_args()
    g1, g2 = a.gpus.split(',')
    if a.frozen:
        state['prep'] = 'FROZEN'
    else:
        main_chain(a)
    rest(a)

def main_chain(a):
    g1 = a.gpus.split(',')[0]
    state['tests'] = a.tests_run
    STATE.write_text(json.dumps(state, indent=1) + '\n')
    suites = sorted(p.name for p in (EXP / f'training_data/test_suites/tests_{PL}/tests/suites').iterdir())
    compare('suite', ROOT / res('tests') / 'tests/suites', EXP / f'training_data/test_suites/tests_{PL}/tests/suites', suites)
    compare('suite', ROOT / res('tests') / 'tests', EXP / f'training_data/test_suites/tests_{PL}/tests', ['suite_manifest.json'])

    assert run('prep', 'prepare', ['--scene', PL, '--tests', res('tests'), '--label', PL], g1, mem='12G')
    compare('prepare', ROOT / res('prep'), EXP / f'training_data/prepared/prep_{PL}',
            ['queries.npz', 'scene.json', 'robot_model.npz', 'data/sampled_points.npy', 'data/speed.npy', 'data/normal.npy'])


def rest(a):
    g1, g2 = a.gpus.split(',')
    field = lambda f, seed, extra: run(f'{f}_s{seed}', 'field', ['--prepared', res('prep'), '--seed', str(seed), '--label', f'{f}_{PL}', '--extra', extra], g1 if f == 'lift' else g2)  # noqa: E731
    LIFT, NTRL = '--lift time --weight 0 --freq 0.5', '--lift none --weight 0'
    with ThreadPoolExecutor(2) as ex:                     # two trainings at once (workstation RAM rule); the Euclid control is light
        e = ex.submit(run, 'euclid', 'euclid', ['--prepared', res('prep'), '--label', PL], g2)
        jobs = [ex.submit(field, 'lift', 87358, LIFT), ex.submit(field, 'ntrl', 87358, NTRL)]
        e.result()
        [j.result() for j in jobs]
        jobs = [ex.submit(field, 'lift', 87359, LIFT), ex.submit(field, 'ntrl', 87359, NTRL)] if 87359 in a.seeds else []
        ev = lambda job, f, seed, label, extra, g: run(job, 'evalfield', ['--prepared', res('prep'), '--fields', f'{f}={res(f"{f}_s{seed}")}', '--label', label, '--extra', extra], g)  # noqa: E731
        # planner evaluations of the seed-87358 fields, as in the original queue (plan_* and budget_*), then the picks' evaluations
        for f in ('lift', 'ntrl'):
            if f'{f}_s87358' in state:
                ev(f'plan_{f}_s87358', f, 87358, f'plan_{f}_{PL}_s87358', '--planner reverse bidir --steps 300 600', g1)
                ev(f'budget_{f}_s87358', f, 87358, f'budget_{f}_{PL}_s87358', '--planner bidir --steps 1200 2400', g1)
                ev(f'picks_{f}', f, 87358, f'picks_{f}_{PL}', '--planner bidir --steps 2400 --save-paths', g1)
        [j.result() for j in jobs]
    if 'lift_s87359' in state:
        run('budget_lift_s87359', 'evalfield', ['--prepared', res('prep'), '--fields', f'lift={res("lift_s87359")}', '--label', f'budget_lift_{PL}_s87359',
                                                '--extra', '--planner bidir --steps 1200 2400'], g1)
    if 'picks_lift' in state and 'picks_ntrl' in state:
        run('demo', 'demo', ['--prepared', res('prep'), '--fields', f'lift={res("picks_lift")}:paths_lift_bidir_2400_50_5',
                             f'ntrl={res("picks_ntrl")}:paths_ntrl_bidir_2400_50_5', '--label', f'picks_{PL}'], g1)
    log('chain done')


if __name__ == '__main__':
    main()
