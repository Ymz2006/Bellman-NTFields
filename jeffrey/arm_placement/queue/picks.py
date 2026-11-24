"""Pick ten hard start/goal pairs at one placement from its own test queries: two registered evaluations that save the plans of every
query (first lift model on GPU 0, ntrl-demo field on GPU 1; two-ended planner, 2,400 steps per end, planning seeds 0 1 2), then the
registered demo run on them. Copies the result to report/demo_<placement>.json/.npz.     python queue/picks.py p_d055_s46_h60"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

Q = Path(__file__).resolve().parent
ROOT = Q.parents[2]
PY = os.path.expanduser('~/.venvs/bellman-ntfield/bin/python')
pl = sys.argv[1]
state = json.loads((Q / 'state.json').read_text())
res = lambda job: f'results/registered_{state[job]["run"]}'  # noqa: E731


def launch(mode, args, gpu, log):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER='PCI_BUS_ID', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    return subprocess.Popen([PY, 'jeffrey/arm_placement/run.py', mode, *args], cwd=ROOT, env=env, stdout=open(Q / 'logs' / log, 'w'), stderr=subprocess.STDOUT)


def run_id(log):
    rid = None
    for line in (Q / 'logs' / log).read_text().splitlines():
        if line.startswith('{"id"'):
            row = json.loads(line)
            rid = row['id'] if row.get('state') == 'succeeded' else rid
    assert rid, f'{log}: no succeeded run'
    return rid


procs = [launch('evalfield', ['--prepared', res(f'prep_{pl}'), '--fields', f'{f}={res(f"{f}_{pl}_s87358")}', '--label', f'picks_{f}_{pl}', '--extra', '--planner bidir --steps 2400 --save-paths'], g, f'picks_{f}_{pl}.log')
         for f, g in (('lift', '0'), ('ntrl', '1'))]
for p in procs:
    p.wait()
runs = {f: run_id(f'picks_{f}_{pl}.log') for f in ('lift', 'ntrl')}
print('evaluations', runs, flush=True)
launch('demo', ['--prepared', res(f'prep_{pl}'), '--fields', *[f'{f}=results/registered_{r}:paths_{f}_bidir_2400_50_5' for f, r in runs.items()], '--label', f'picks_{pl}'], '0', f'picks_demo_{pl}.log').wait()
rid = run_id(f'picks_demo_{pl}.log')
out = Q.parent / 'report'
shutil.copy(ROOT / f'results/registered_{rid}/results.json', out / f'demo_{pl}.json')
shutil.copy(ROOT / f'results/registered_{rid}/demo.npz', out / f'demo_{pl}.npz')
print('demo', rid, flush=True)
