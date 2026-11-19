"""Queue runner for the sharp-features matrix (jobs.txt lines 'gpu|n|seed|spec' (gpu '*' = whichever GPU has a free slot), '#' comments; re-read every loop).

A job runs on its GPU ('0' = RTX 5070, '1' = RTX 3090, CUDA_DEVICE_ORDER = PCI_BUS_ID) or, with '*', on whichever GPU has
a free slot first (a sharp-feature field diverges from the octave run anyway; the noise of one run covers the GPU). slots.json (re-read every loop) = {"slots": {gpu: my jobs at once}, "max_foreign": {gpu: launch only while at
most this many other python compute processes use that GPU}} (the machine is shared with other sessions' queues; at
most two of my trainings at once, CLAUDE.md). A job starts only with >= 6 GB RAM available and >= 3 GB free on its GPU;
a PAUSE file holds new starts; STOP ends the runner once its jobs finish. state.json keeps done / failed jobs with run
ids; logs/<job>.log is each registered run's console.

    setsid nohup ~/.venvs/bellman-ntfield/bin/python jeffrey/sharp_features/queue/runner.py > jeffrey/sharp_features/queue/runner.out 2>&1 &

slots.json 'max_total' caps my trainings across both GPUs (default 2). adopt.json ({job: {"gpu": g, "pid": registry pid}}), read
at start, lists jobs a previous runner launched that are still running: they hold their GPU slot until the pid is gone, then the
registry's final state line in the job's log decides done / failed. (Fixed 2026-11-18 08:05: the job list is recomputed for each
GPU, so one job is never launched on both GPUs in the same poll.)
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
adopted = {}
if (HERE / 'adopt.json').exists():
    for _job, _v in json.load(open(HERE / 'adopt.json')).items():
        if _job not in state['done'] and _job not in state['failed']:
            adopted[_job] = (int(_v['pid']), str(_v['gpu']))


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
        if 'python' in cmd and 'sharp_features' not in cmd:
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
    return HERE / 'logs' / (job.replace('|', '_').replace('*', 'any') + '.log')


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
    for job, (pid, gpu) in list(adopted.items()):
        if os.path.exists(f'/proc/{pid}') and not open(f'/proc/{pid}/stat').read().split()[2] == 'Z':
            continue
        rid, ok = None, False
        for line in logname(job).read_text().splitlines():
            if line.startswith('{"id"'):
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                rid = d.get('id', rid)
                ok = ok or d.get('state') == 'succeeded'
        (state['done'] if ok else state['failed'])[job] = rid
        del adopted[job]
        json.dump(state, open(state_f, 'w'), indent=1)
        stamp(f'{"done" if ok else "FAILED"} (adopted) {job} -> {rid}')
    busy = set(running) | set(adopted)
    if (not [j for j in jobs() if j not in state['done'] and j not in state['failed'] and j not in busy] or (HERE / 'STOP').exists()) and not busy:
        break
    if not (HERE / 'PAUSE').exists() and not (HERE / 'STOP').exists():
        cfg = config()
        for gpu, slots in cfg['slots'].items():
            busy = set(running) | set(adopted)
            if len(busy) >= cfg.get('max_total', 2):
                break
            if sum(1 for v in running.values() if v[1] == gpu) + sum(1 for v in adopted.values() if v[1] == gpu) >= slots:
                continue
            todo = [j for j in jobs() if j not in state['done'] and j not in state['failed'] and j not in busy]
            mine = [j for j in todo if j.split('|')[0] in (gpu, '*')]
            if not mine or avail_gb() < 6 or gpu_free_gb(gpu) < 3 or foreign(gpu) > cfg['max_foreign'].get(gpu, 99):
                continue
            job = mine[0]
            _, n, seed, spec = job.split('|')
            f = open(logname(job), 'w')
            env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=gpu)
            cmd = [PY, str(ROOT / 'jeffrey/sharp_features/run.py'), '--prepared', str(ROOT / f'results/registered_{PREPARED[int(n)]}'),
                   '--seed', seed, '--spec', spec]
            running[job] = (subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env), gpu, f)
            stamp(f'start {job} (RAM {avail_gb():.1f} GB, GPU {gpu} free {gpu_free_gb(gpu):.1f} GB, foreign {foreign(gpu)})')
            time.sleep(20)
    time.sleep(30)
stamp('runner end')
