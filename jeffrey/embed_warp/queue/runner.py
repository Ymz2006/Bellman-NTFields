"""Queue runner for the gap-maze matrix: jobs.txt lines 'label|prepared|seed|extra'; slots GPU0 x1, GPU1 x2; RAM guard; state.json."""
import json, os, subprocess, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PY = os.path.expanduser('~/.venvs/bellman-ntfield/bin/python')
SLOTS = {'0': 1, '1': 2}
state_f = HERE / 'state.json'
state = json.load(open(state_f)) if state_f.exists() else dict(done={}, failed={})
running = {}

def avail_gb():
    for l in open('/proc/meminfo'):
        if l.startswith('MemAvailable'):
            return int(l.split()[1]) / 1e6
    return 0

def jobs():
    out = []
    for l in open(HERE / 'jobs.txt'):
        l = l.strip()
        if l and not l.startswith('#'):
            out.append(l)
    return out

while True:
    for key, (p, gpu, f) in list(running.items()):
        if p.poll() is not None:
            f.close()
            txt = (HERE / 'logs' / f'{key.replace("|", "_").replace("/", "_").replace(" ", "")[:150]}.log').read_text()
            rid = None
            for line in txt.splitlines():
                if line.startswith('{"id"'):
                    try:
                        rid = json.loads(line)['id']
                    except Exception:
                        pass
            (state['done'] if p.returncode == 0 else state['failed'])[key] = rid
            del running[key]
            json.dump(state, open(state_f, 'w'), indent=1)
    todo = [j for j in jobs() if j not in state['done'] and j not in state['failed'] and j not in running]
    if not todo and not running:
        break
    for gpu, n in SLOTS.items():
        while todo and sum(1 for v in running.values() if v[1] == gpu) < n and avail_gb() > 8 and not (HERE / 'PAUSE').exists():
            j = todo.pop(0)
            label, prep, seed, extra = j.split('|')
            log = HERE / 'logs' / f'{j.replace("|", "_").replace("/", "_").replace(" ", "")[:150]}.log'
            f = open(log, 'w')
            env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=gpu)
            cmd = [PY, str(ROOT / 'jeffrey/embed_warp/run.py'), '--stage', 'field', '--prepared', str(ROOT / prep), '--seed', seed, '--label', label, '--extra', extra]
            running[j] = (subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env), gpu, f)
            time.sleep(20)
    time.sleep(30)
