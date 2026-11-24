"""Job scheduler of the arm-placement study: runs registered jobs (run.py) on the GPUs, several per GPU.

queue/jobs.jsonl   one job per line: {"id": "...", "mode": "tests|prepare|field|...", "args": [...], "deps": ["job id", ...], "load": 0.2}
                   An argument may contain {job:<id>}: the result directory (results/registered_<run id>) of that finished job.
queue/caps.json    {"0": 1.0, "1": 1.0}: load a GPU may carry; a job starts on the GPU with the most room. Re-read every loop.
queue/state.json   job id -> status (pending is implicit), run id, gpu, times, pid. Lines can be appended to jobs.jsonl while this runs.
queue/forget.txt   job ids (one per line) whose state is dropped at the next loop, so that they run again; the file is then emptied.
A job whose dependency failed is marked 'blocked'. After a restart, jobs whose process is still alive are followed again.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

Q = Path(__file__).resolve().parent
ROOT = Q.parents[2]
PY = os.path.expanduser('~/.venvs/bellman-ntfield/bin/python')
STATE = Q / 'state.json'


def load_jobs():
    jobs = []
    for line in (Q / 'jobs.jsonl').read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            jobs.append(json.loads(line))
    return jobs


def save(state):
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(STATE)


NEED_MIB = dict(tests=8000, prepare=6000, field=3000, evalfield=5500, demo=6000, euclid=1500, reach=3000)       # VRAM reserved per job (its peak), and free VRAM it must find
LIMIT_MIB = 52000


def free_mib():
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.total,memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
    return {r.split(',')[0].strip(): int(r.split(',')[1]) - int(r.split(',')[2]) for r in out.strip().splitlines()}


def alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        return 'Z' not in Path(f'/proc/{pid}/status').read_text().split('State:')[1].split('\n')[0]
    except Exception:
        return False


class Adopted:
    def __init__(self, pid):
        self.pid = pid

    def poll(self):
        return None if alive(self.pid) else 0


def main():
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    procs = {}
    for k, v in state.items():          # a restart: follow the jobs whose process is still alive
        if v['status'] == 'running':
            if v.get('pid') and alive(v['pid']):
                procs[k] = (Adopted(v['pid']), None)
            else:
                v['status'] = 'lost'

    (Q / 'logs').mkdir(exist_ok=True)
    while True:
        caps = {k: float(v) for k, v in json.loads((Q / 'caps.json').read_text()).items()}
        jobs = load_jobs()
        fg = Q / 'forget.txt'
        if fg.exists() and fg.read_text().strip():
            for jid in fg.read_text().split():
                if jid in state and jid not in procs:
                    del state[jid]
            fg.write_text('')
            save(state)
        for jid, (proc, fh) in list(procs.items()):
            rc = proc.poll()
            if rc is None:
                continue
            if fh is not None:
                fh.close()
            rid = None
            for line in (Q / 'logs' / f'{jid}.log').read_text().splitlines():
                if line.startswith('{"id"'):
                    try:
                        row = json.loads(line)
                        rid = row['id'] if row.get('state') == 'succeeded' else rid
                    except Exception:
                        pass
            state[jid].update(status='done' if rid else 'failed', run=rid, end=time.time(), rc=rc)
            del procs[jid]
            save(state)
        used = {g: 0.0 for g in caps}
        free = free_mib()
        last = getattr(main, 'last', {})
        reserved = {g: 0 for g in caps}
        for jid in procs:
            reserved[state[jid]['gpu']] += NEED_MIB.get(state[jid].get('mode'), 3000)
        for jid in procs:
            used[state[jid]['gpu']] += state[jid]['load']
        for job in jobs:
            jid = job['id']
            if jid in state:
                continue
            deps = [state.get(d, {}).get('status') for d in job.get('deps', [])]
            if any(d in ('failed', 'blocked', 'lost') for d in deps):
                state[jid] = dict(status='blocked')
                save(state)
                continue
            if any(d != 'done' for d in deps):
                continue
            load = float(job.get('load', 0.25))
            need = NEED_MIB.get(job['mode'], 3000)
            fits = [g for g in caps if used[g] + load <= caps[g] + 1e-9 and free.get(g, 0) >= need and reserved[g] + need <= LIMIT_MIB and time.time() - last.get(g, 0) >= 5]      # one start per GPU every 5 s: a new job's memory is not visible at once
            if not fits:
                continue
            gpu = max(fits, key=lambda g: (LIMIT_MIB - reserved[g], caps[g] - used[g]))
            last[gpu] = time.time()
            main.last = last
            free[gpu] -= need
            reserved[gpu] += need
            args = [x if '{job:' not in x else x.replace(x[x.index('{job:'):x.index('}') + 1], f'results/registered_{state[x[x.index("{job:") + 5:x.index("}")]]["run"]}') for x in job['args']]
            fh = open(Q / 'logs' / f'{jid}.log', 'w')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER='PCI_BUS_ID', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TORCHINDUCTOR_COMPILE_THREADS='1')
            proc = subprocess.Popen([PY, 'jeffrey/arm_placement/run.py', job['mode'], *args], cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
            procs[jid] = (proc, fh)
            state[jid] = dict(status='running', gpu=gpu, load=load, start=time.time(), mode=job['mode'], args=args, pid=proc.pid)
            used[gpu] += load
            save(state)
        if (Q / 'STOP').exists() and not procs:
            break
        time.sleep(3)


if __name__ == '__main__':
    sys.exit(main())
