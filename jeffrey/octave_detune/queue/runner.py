"""Queue runner for the octave-detune matrix (jobs.txt lines 'gpu|n|seed|spec', '#' comments; re-read every loop).

Every (maze, seed) group runs on one GPU so its runs pair exactly ('0' = RTX 5070, '1' = RTX 3090, CUDA_DEVICE_ORDER =
PCI_BUS_ID). slots.json (re-read every loop) = {"slots": {gpu: my jobs at once}, "max_foreign": {gpu: launch only while at
most this many other python compute processes use that GPU}} (the machine is shared with other sessions' queues; at
most two of my trainings at once, CLAUDE.md). A job starts only with >= 6 GB RAM available and >= 3 GB free on its GPU;
a PAUSE file holds new starts; STOP ends the runner once its jobs finish. state.json keeps done / failed jobs with run
ids; logs/<job>.log is each registered run's console.

    setsid nohup ~/.venvs/bellman-ntfield/bin/python jeffrey/octave_detune/queue/runner.py > jeffrey/octave_detune/queue/runner.out 2>&1 &
"""
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PY = os.path.expanduser('~/.venvs/bellman-ntfield/bin/python')
DEFAULT = dict(slots={'0': 1, '1': 1}, max_foreign={'0': 1, '1': 2})
PREPARED = {7: '20261101T192446Z-9b1569db', 10: '20261102T032150Z-b0baa8e9', 13: '20261101T192504Z-13cf1d04', 22: '20261101T193403Z-cea05796'}
state_f = HERE / 'state.json'
state = json.load(open(state_f)) if state_f.exists() else dict(done={}, failed={})
running = {}


def avail_gb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable'):
            return int(line.split()[1]) / 1e6
    return 0


def gpu_free_gb(gpu):
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits', '-i', gpu], capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip()) / 1024
    except Exception:
        return 0.0


def config():
    try:
        return json.load(open(HERE / 'slots.json'))
    except Exception:
        return DEFAULT


def foreign(gpu):
    """Other sessions' python compute processes on that GPU (display / browser processes are not counted)."""
    try:
        bus = subprocess.run(['nvidia-smi', '--query-gpu=pci.bus_id', '--format=csv,noheader', '-i', gpu], capture_output=True, text=True, timeout=30).stdout.strip()
        rows = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_bus_id,pid', '--format=csv,noheader'], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return 99
    count = 0
    for row in rows.strip().splitlines():
        b, pid = [x.strip() for x in row.split(',')]
        if b != bus:
            continue
        try:
            cmd = open(f'/proc/{pid}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='replace')
        except Exception:
            continue
        if 'python' in cmd and 'octave_detune' not in cmd:
            count += 1
    return count


def jobs():
    out = []
    for line in open(HERE / 'jobs.txt'):
        line = line.strip()
        if line and not line.startswith('#'):
            out.append(line)
    return out


def logname(job):
    return HERE / 'logs' / (job.replace('|', '_') + '.log')


def stamp(msg):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), msg, flush=True)


stamp(f'runner start, {len(state["done"])} done, {len(state["failed"])} failed')
while True:
    for job, (proc, gpu, f) in list(running.items()):
        if proc.poll() is None:
            continue
        f.close()
        rid = None
        for line in logname(job).read_text().splitlines():
            if line.startswith('{"id"'):
                try:
                    rid = json.loads(line)['id']
                except Exception:
                    pass
        (state['done'] if proc.returncode == 0 else state['failed'])[job] = rid
        del running[job]
        json.dump(state, open(state_f, 'w'), indent=1)
        stamp(f'{"done" if proc.returncode == 0 else "FAILED"} {job} -> {rid}')
    todo = [j for j in jobs() if j not in state['done'] and j not in state['failed'] and j not in running]
    if (not todo or (HERE / 'STOP').exists()) and not running:
        break
    if not (HERE / 'PAUSE').exists() and not (HERE / 'STOP').exists():
        cfg = config()
        for gpu, slots in cfg['slots'].items():
            if sum(1 for v in running.values() if v[1] == gpu) >= slots:
                continue
            mine = [j for j in todo if j.split('|')[0] == gpu]
            if not mine or avail_gb() < 6 or gpu_free_gb(gpu) < 3 or foreign(gpu) > cfg['max_foreign'].get(gpu, 99):
                continue
            job = mine[0]
            _, n, seed, spec = job.split('|')
            f = open(logname(job), 'w')
            env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=gpu)
            cmd = [PY, str(ROOT / 'jeffrey/octave_detune/run.py'), '--prepared', str(ROOT / f'results/registered_{PREPARED[int(n)]}'),
                   '--seed', seed, '--spec', spec]
            running[job] = (subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env), gpu, f)
            stamp(f'start {job} (RAM {avail_gb():.1f} GB, GPU {gpu} free {gpu_free_gb(gpu):.1f} GB, foreign {foreign(gpu)})')
            time.sleep(20)
    time.sleep(30)
stamp('runner end')
