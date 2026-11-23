"""Launch one job outside the queue runner (used while the queue is paused for the whole-interior tests).

    python queue/direct.py <gpu> '<label>|<scene key>|<seed>|<pipeline arguments>' [--field <run id>] [--init <run id>]

The job line is appended to queue/direct.txt (with ';field=<id>' / ';init=<id>' when given); its log goes to queue/logs/ under the
runner's naming, so status.py and report.py find the registered id there. scenes.json[key] needs 'prepared'; 'blocked' is optional.
Blocks until the run ends (start it as a background task).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PY = os.path.expanduser('~/.venvs/bellman-ntfield/bin/python')


def log_path(j):
    return HERE / 'logs' / (''.join(c if c.isalnum() or c in '-_.' else '_' for c in j)[:180] + '.log')


def main():
    gpu, job = sys.argv[1], sys.argv[2]
    opts = dict(zip(sys.argv[3::2], sys.argv[4::2]))
    label, key, seed, extra = job.split('|')
    sc = json.load(open(HERE / 'scenes.json'))[key]
    cmd = [PY, str(ROOT / 'jeffrey/pair_lift/run.py'), '--prepared', str(ROOT / sc['prepared']), '--seed', seed, '--label', label, '--extra', extra]
    if sc.get('blocked'):
        cmd += ['--blocked', str(ROOT / sc['blocked'])]
    line = job
    if '--repeat-reason' in opts:
        cmd += ['--repeat-reason', opts['--repeat-reason']]
    for k in ('--field', '--init'):
        if k in opts:
            cmd += [k, str(ROOT / f'results/registered_{opts[k]}')]
            line += f';{k[2:]}={opts[k]}'
    have = (HERE / 'direct.txt').read_text().splitlines() if (HERE / 'direct.txt').exists() else []
    if line not in have:
        with open(HERE / 'direct.txt', 'a') as f:
            f.write(line + '\n')
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=gpu)
    with open(HERE / 'runner.log', 'a') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + f'direct start (GPU {gpu}) {line}\n')
    lp = log_path(line)
    lp.parent.mkdir(exist_ok=True)
    code = subprocess.run(cmd, stdout=open(lp, 'w'), stderr=subprocess.STDOUT, env=env, cwd=ROOT).returncode
    with open(HERE / 'runner.log', 'a') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + f'direct {"done" if code == 0 else "FAILED"} {line}\n')
    print(('done ' if code == 0 else 'FAILED ') + line)
    print(lp.read_text()[-300:])


if __name__ == '__main__':
    main()
