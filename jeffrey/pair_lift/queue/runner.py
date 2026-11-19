"""Queue runner of the pair-lift study. jobs.txt lines, processed in file order:

    <label>|<scene key>|<seed>|<pipeline arguments>        registers jeffrey/pair_lift/run.py on scenes.json[key] (prepared + blocked runs)

Slots: GPU 0 (RTX 5070, drives the display) one run, GPU 1 (RTX 3090) two. No launch below 8 GB of available RAM. `touch PAUSE` holds
new launches; state.json records the registered id of every finished job; runner.log has one line per start / end.
"""
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PY = os.path.expanduser('~/.venvs/bellman-ntfield/bin/python')
SLOTS = {'1': 2, '0': 1}
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
    label, key, seed, extra = j.split('|')
    sc = json.load(open(HERE / 'scenes.json'))[key]
    stage = ['--stage', 'repfit'] if label.startswith('fit_') else []
    return [PY, str(ROOT / 'jeffrey/pair_lift/run.py'), *stage, '--prepared', str(ROOT / sc['prepared']), '--blocked', str(ROOT / sc['blocked']), '--seed', seed, '--label', label, '--extra', extra]


def stamp(msg):
    with open(HERE / 'runner.log', 'a') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + msg + '\n')


def run_id(j):
    rid = None
    if log_path(j).exists():
        for line in log_path(j).read_text().splitlines():
            if line.startswith('{"id"'):
                try:
                    rid = json.loads(line)['id']
                except Exception:
                    pass
    return rid


def alive(rid):
    return bool(subprocess.run(['pgrep', '-f', f'registered_{rid}'], capture_output=True, text=True).stdout.split())


class Adopted:
    """A job started by an earlier runner process: followed by its registered output folder instead of a process handle."""

    def __init__(self, rid):
        self.rid, self.returncode = rid, None

    def poll(self):
        if alive(self.rid):
            return None
        time.sleep(5)
        f = ROOT / 'results' / f'registered_{self.rid}' / 'results.json'
        self.returncode = 0 if (f.exists() and json.load(open(f)).get('complete')) else 1
        return self.returncode


(HERE / 'logs').mkdir(exist_ok=True)
# adopt what an earlier runner left running (its GPU is in runner.log)
if (HERE / 'runner.log').exists():
    started = {}
    for line in open(HERE / 'runner.log'):
        if ' start ' in line and ' on GPU ' in line:
            started[line.split(' start ', 1)[1].rsplit(' on GPU ', 1)[0]] = line.rsplit(' on GPU ', 1)[1].split()[0]
    for j in jobs():
        if j in state['done'] or j in state['failed'] or j not in started:
            continue
        rid = run_id(j)
        if rid is not None:
            running[j] = (Adopted(rid), started[j], None)
            stamp(f'adopted {j} ({rid}) on GPU {started[j]}')
idle = 0
while True:
    for j, (p, gpu, f) in list(running.items()):
        if p.poll() is not None:
            if f is not None:
                f.close()
            rid = run_id(j)
            ok = p.returncode == 0 and rid is not None
            (state['done'] if ok else state['failed'])[j] = rid
            del running[j]
            json.dump(state, open(state_f, 'w'), indent=1)
            stamp(f'{"done" if ok else "FAILED"} {j} -> {rid} (exit {p.returncode})')
    todo = [j for j in jobs() if j not in state['done'] and j not in state['failed'] and j not in running]
    if not todo and not running:
        idle += 1
        if idle > 120:                      # an hour with nothing to do
            stamp('queue empty for an hour, runner exits')
            break
    else:
        idle = 0
    for gpu, n in SLOTS.items():
        while todo and sum(1 for v in running.values() if v[1] == gpu) < n and avail_gb() > 8 and not (HERE / 'PAUSE').exists():
            j = todo.pop(0)
            f = open(log_path(j), 'w')
            env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=gpu)
            running[j] = (subprocess.Popen(command(j), stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env), gpu, f)
            stamp(f'start {j} on GPU {gpu} (RAM {avail_gb():.1f} GB)')
            time.sleep(20)
    time.sleep(30)
