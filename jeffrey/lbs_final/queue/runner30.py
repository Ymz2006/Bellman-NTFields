"""CPU-only runner for jobs30.json (30 s RRT-Connect / LazyPRM pass), next to runner.py, which keeps the main queue.

Launches a job when its deps are done in the main queue's state.json, and only while the number of this study's bench processes (both
runners) is below TOTAL_BUSY while any of this study's field trainings runs, TOTAL_IDLE otherwise, and its own below OWN_BUSY / OWN_IDLE: the machine has 8 physical cores, and
more timed benchmarks than free cores would inflate the per-query times being measured. Same launch environment as runner.py (CUDA hidden,
2 torch threads). On start it adopts the jobs a previous copy started that are still running (their registered run is alive),
so it can be restarted without duplicating them. state30.json / runner30.log; `touch PAUSE30` holds new launches; exits when everything is done.
"""
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OMPL_PY = os.path.expanduser('~/.venvs/ompl-bench/bin/python')
RUN = str(ROOT / 'jeffrey/lbs_final/run.py')
TOTAL_BUSY, TOTAL_IDLE = 4, 6
OWN_BUSY, OWN_IDLE = 3, 5                                         # runner.py adds up to 3 short learned benches of its own
MIN_RAM = 4.0
state_f = HERE / 'state30.json'
state = json.load(open(state_f)) if state_f.exists() else dict(done={}, failed={})
running = {}
dead = {}
adopted = {}                                                       # name -> run id: jobs a previous copy of this runner started


def alive(rid):
    return subprocess.run(['pgrep', '-f', f'registered_{rid}'], capture_output=True).returncode == 0


def outcome(name):
    rid, ok = None, False
    for line in log_path(name).read_text().splitlines() if log_path(name).exists() else []:
        if line.startswith('{"id"'):
            rid = json.loads(line)['id']
            ok = ok or '"succeeded"' in line
    return rid, ok


def avail_gb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable'):
            return int(line.split()[1]) / 1e6
    return 0


def count(pattern):
    out = subprocess.run(['pgrep', '-fc', pattern], capture_output=True, text=True).stdout.strip()
    return int(out or 0)


def stamp(msg):
    with open(HERE / 'runner30.log', 'a') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + msg + '\n')


def save():
    tmp = state_f.with_suffix('.tmp30')
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, state_f)


def log_path(name):
    (HERE / 'logs').mkdir(exist_ok=True)
    return HERE / 'logs' / (''.join(c if c.isalnum() or c in '-_.' else '_' for c in name) + '.log')


def launch(j, main_done):
    args = []
    for x in j['args']:
        for name, rid in main_done.items():
            if x == '@' + name:
                x = str(ROOT / 'results' / f'registered_{rid}')
        assert '@' not in x, x
        args.append(x)
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    f = open(log_path(j['name']), 'w')
    running[j['name']] = (subprocess.Popen([OMPL_PY, RUN, 'bench', *args], stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env), f)
    stamp(f'start {j["name"]} (RAM {avail_gb():.1f} GB)')


for j in json.load(open(HERE / 'jobs30.json')):
    if j['name'] in state['done'] or j['name'] in state['failed']:
        continue
    rid, ok = outcome(j['name'])
    if ok:
        state['done'][j['name']] = rid
        save()
    elif rid and alive(rid):
        adopted[j['name']] = rid
        stamp(f'adopt {j["name"]} ({rid})')

while True:
    jobs = json.load(open(HERE / 'jobs30.json'))
    for name, rid in list(adopted.items()):
        _, ok = outcome(name)
        dead[name] = 0 if alive(rid) else dead.get(name, 0) + 1
        if ok or dead[name] >= 6:                                     # the registry writes its last line after the bench exits
            if ok:
                state['done'][name] = rid
            else:
                state['failed'][name] = 'adopted run ended without success'
            del adopted[name]
            save()
            stamp(f'{"done" if ok else "FAILED"} {name} -> {rid} (adopted)')
    main_done = json.load(open(HERE / 'state.json'))['done']
    for name, (p, f) in list(running.items()):
        if p.poll() is None:
            continue
        f.close()
        rid = None
        for line in log_path(name).read_text().splitlines():
            if line.startswith('{"id"') and '"succeeded"' in line:
                rid = json.loads(line)['id']
        if p.returncode == 0 and rid:
            state['done'][name] = rid
        else:
            state['failed'][name] = f'exit {p.returncode}'
        del running[name]
        save()
        stamp(f'{"done" if rid else "FAILED"} {name} -> {rid} (exit {p.returncode})')
    finished = set(state['done']) | set(state['failed'])
    todo = [j for j in jobs if j['name'] not in finished and j['name'] not in running and j['name'] not in adopted]
    if not todo and not running and not adopted:
        stamp('queue empty')
        break
    if not (HERE / 'PAUSE30').exists():
        for j in todo:
            if not all(d in main_done for d in j['deps']):
                continue
            busy = count('lbs_final/snap/.*/field.py') > 0
            cap, own = (TOTAL_BUSY, OWN_BUSY) if busy else (TOTAL_IDLE, OWN_IDLE)
            if count('lbs_final/snap/.*/bench.py') >= cap or len(running) + len(adopted) >= own or avail_gb() < MIN_RAM:
                break
            launch(j, main_done)
            time.sleep(15)
    time.sleep(30)
