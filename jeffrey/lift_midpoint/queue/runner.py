"""Queue runner of the lift + midpoint study. jobs.txt lines, processed in file order:

    <label>|<scene key>|<seed>|<pipeline arguments>        registers jeffrey/lift_midpoint/run.py on scenes.json[key]

At most MAX_MINE heavy runs of this study at once (the lift-alone maze controls of jeffrey/lift_scale's pipeline count too), one per GPU
slot (SLOTS), and no launch below 8 GB of available RAM. `touch PAUSE` holds new launches; state.json records the registered id of every
finished job.
"""
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PY = os.path.expanduser('~/.venvs/bellman-ntfield/bin/python')
SLOTS = {'0': 1, '1': 1}
MAX_MINE = 2
state_f = HERE / 'state.json'
state = json.load(open(state_f)) if state_f.exists() else dict(done={}, failed={})
running = {}


def avail_gb():
    for l in open('/proc/meminfo'):
        if l.startswith('MemAvailable'):
            return int(l.split()[1]) / 1e6
    return 0


def mine_elsewhere():
    """Heavy runs of this study not started by this runner (the lift-alone maze loop)."""
    out = subprocess.run(['pgrep', '-f', '^' + PY + ' .*jeffrey/lift_scale/snap/[0-9a-f]+/pipeline.py'], capture_output=True, text=True).stdout.split()
    return len(out)


def jobs():
    return [l.strip() for l in open(HERE / 'jobs.txt') if l.strip() and not l.startswith('#')]


def log_path(j):
    return HERE / 'logs' / (''.join(c if c.isalnum() or c in '-_.' else '_' for c in j)[:180] + '.log')


def command(j):
    label, key, seed, extra = j.split('|')
    prep = json.load(open(HERE / 'scenes.json'))[key]
    return [PY, str(ROOT / 'jeffrey/lift_midpoint/run.py'), '--prepared', str(ROOT / prep), '--seed', seed, '--label', label, '--extra', extra]


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
        stamp('queue empty')
        break
    for gpu, n in SLOTS.items():
        if not todo:
            break
        if sum(1 for v in running.values() if v[1] == gpu) >= n or len(running) + mine_elsewhere() >= MAX_MINE:
            continue
        if avail_gb() < 8 or (HERE / 'PAUSE').exists():
            continue
        j = todo.pop(0)
        f = open(log_path(j), 'w')
        env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=gpu)
        running[j] = (subprocess.Popen(command(j), stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env), gpu, f)
        stamp(f'start {j} on GPU {gpu} (RAM {avail_gb():.1f} GB)')
        time.sleep(20)
    time.sleep(30)
