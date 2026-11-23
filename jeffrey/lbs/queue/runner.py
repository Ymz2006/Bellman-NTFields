"""Queue runner of the LBS benchmark. jobs.txt lines, processed in file order:

    <label>|<scene key>|<seed>|<pipeline arguments>[|<gpu>]      registers jeffrey/lbs/run.py on scenes.json[key]

One run per GPU (SLOTS; the runs are GPU-bound, a second run on a GPU only slows both). A job with a 5th field runs only on that GPU
(the houses: RTX 3090). No launch below MIN_RAM GB of available RAM. `touch PAUSE` holds new launches; state.json records the registered
id of every finished job; runner.log has one line per start / end.
"""
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PY = os.path.expanduser('~/.venvs/bellman-ntfield/bin/python')
SLOTS = {'1': 1, '0': 1}
MIN_RAM = 6.0
state_f = HERE / 'state.json'
state = json.load(open(state_f)) if state_f.exists() else dict(done={}, failed={})
running = {}


def avail_gb():
    for l in open('/proc/meminfo'):
        if l.startswith('MemAvailable'):
            return int(l.split()[1]) / 1e6
    return 0


def jobs():
    return [l.strip() for l in open(HERE / 'jobs.txt') if l.strip() and not l.startswith('#')]


def log_path(j):
    return HERE / 'logs' / (''.join(c if c.isalnum() or c in '-_.' else '_' for c in j)[:180] + '.log')


def command(j):
    label, key, seed, extra = j.split('|')[:4]
    prep = json.load(open(HERE / 'scenes.json'))[key]
    return [PY, str(ROOT / 'jeffrey/lbs/run.py'), '--prepared', str(ROOT / prep), '--seed', seed, '--label', label, '--extra', extra]


def gpu_of(j):
    f = j.split('|')
    return f[4] if len(f) > 4 and f[4] else None


def stamp(msg):
    with open(HERE / 'runner.log', 'a') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + msg + '\n')


while True:
    for j, (p, gpu, f) in list(running.items()):
        if p.poll() is not None:
            f.close()
            rid = None
            for line in log_path(j).read_text().splitlines():
                if line.startswith('{"id"'):
                    try:
                        rid = json.loads(line)['id']
                    except Exception:
                        pass
            ok = p.returncode == 0 and rid is not None
            (state['done'] if ok else state['failed'])[j] = rid
            del running[j]
            json.dump(state, open(state_f, 'w'), indent=1)
            stamp(f'{"done" if ok else "FAILED"} {j} -> {rid} (exit {p.returncode})')
    todo = [j for j in jobs() if j not in state['done'] and j not in state['failed'] and j not in running]
    if not todo and not running:
        if not (HERE / 'KEEP').exists():
            stamp('queue empty')
            break
        time.sleep(30)
        continue
    for gpu, n in SLOTS.items():
        if not todo:
            break
        if sum(1 for v in running.values() if v[1] == gpu) >= n:
            continue
        if avail_gb() < MIN_RAM or (HERE / 'PAUSE').exists():
            continue
        fit = [x for x in todo if gpu_of(x) in (None, gpu)]
        if not fit:
            continue
        j = fit[0]
        todo.remove(j)
        f = open(log_path(j), 'w')
        env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=gpu)
        running[j] = (subprocess.Popen(command(j), stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env), gpu, f)
        stamp(f'start {j} on GPU {gpu} (RAM {avail_gb():.1f} GB)')
        time.sleep(20)
    time.sleep(30)
