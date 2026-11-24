"""Queue of the LBS inference-time study. jobs.json: list of dict(name, lane, args) with args the arguments of jeffrey/lbs_fast/run.py;
'{run:<job name>}' in an argument is replaced by that finished job's registered run id. Lanes:

    train0 / train1   one job at a time on GPU 0 / GPU 1 (clearance-field trainings; not timed)
    timed             one job at a time, and only while no training job is running or waiting: timing runs own the machine

state.json: done {name: run id}, failed {name: reason}. `touch queue/PAUSE` holds new starts; the runner exits when nothing is left
(unless queue/KEEP exists). jobs.json is re-read every pass, so jobs can be appended while it runs.
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

Q = Path(__file__).resolve().parent
ROOT = Q.parents[2]
PY = sys.executable
RUN = ROOT / 'jeffrey/lbs_fast/run.py'


def load(name, default):
    p = Q / name
    return json.load(open(p)) if p.exists() else default


def save_state(state):
    tmp = Q / 'state.json.tmp'
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(Q / 'state.json')


def log(msg):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(Q / 'runner.log', 'a') as f:
        f.write(line + '\n')


def main():
    state = load('state.json', dict(done={}, failed={}))
    running = {}                                              # lane -> (name, Popen, log file)
    while True:
        jobs = load('jobs.json', [])
        for lane, (name, proc, out) in list(running.items()):
            if proc.poll() is None:
                continue
            text = open(out).read()
            ids = re.findall(r'"id": "([^"]+)", "state": "succeeded"', text)
            if proc.returncode == 0 and ids:
                state['done'][name] = ids[-1]
                log(f'done {name} -> {ids[-1]}')
            else:
                state['failed'][name] = text[-600:]
                log(f'FAILED {name} (exit {proc.returncode})')
            save_state(state)
            del running[lane]
        pending = [j for j in jobs if j['name'] not in state['done'] and j['name'] not in state['failed'] and all(j['name'] != r[0] for r in running.values())]
        if not pending and not running and not (Q / 'KEEP').exists():
            log('queue empty')
            return
        if not (Q / 'PAUSE').exists():
            training = any(l.startswith('train') for l in running) or any(j['lane'].startswith('train') for j in pending)
            for lane in ('train0', 'train1', 'timed'):
                if lane in running or (lane == 'timed' and training):
                    continue
                for j in pending:
                    if j['lane'] != lane:
                        continue
                    deps = re.findall(r'\{run:([^}]+)\}', ' '.join(j['args']))
                    if any(d in state['failed'] for d in deps):
                        state['failed'][j['name']] = 'dependency failed'
                        save_state(state)
                        continue
                    if any(d not in state['done'] for d in deps):
                        continue
                    args = [re.sub(r'\{run:([^}]+)\}', lambda m: state['done'][m.group(1)], x) for x in j['args']]
                    out = Q / 'logs' / (j['name'].replace('/', '_') + '.out')
                    out.parent.mkdir(exist_ok=True)
                    proc = subprocess.Popen([PY, str(RUN), *args], stdout=open(out, 'w'), stderr=subprocess.STDOUT, cwd=ROOT)
                    running[lane] = (j['name'], proc, out)
                    log(f'start {j["name"]} [{lane}]')
                    break
        time.sleep(3)


if __name__ == '__main__':
    main()
