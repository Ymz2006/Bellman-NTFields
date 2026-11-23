"""Queue runner of the LBS final benchmark (jobs.json from make_jobs.py; one json object per job with name, stage, args, deps, gpu, table).

GPU stages (prep, field): at most one of this queue's runs per GPU at a time; the GPUs are shared with other studies' runs, which this
runner neither stops nor waits for. CPU stage (bench): at most CPU_SLOTS at a time, CUDA hidden, 2 torch threads each. No launch below
MIN_RAM_GPU / MIN_RAM_CPU GB of available RAM. '@<job name>' in a job's arguments becomes that job's results directory. A job starts
when all its deps are done; CPU jobs go by table, then file order. `touch PAUSE` holds new launches; `KEEP` keeps the runner alive when
nothing is left. state.json: registered id of every finished job; runner.log: one line per start / end.
"""
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PY = os.path.expanduser('~/.venvs/bellman-ntfield/bin/python')
OMPL_PY = os.path.expanduser('~/.venvs/ompl-bench/bin/python')
RUN = str(ROOT / 'jeffrey/lbs_final/run.py')
GPUS = ['1', '0']
CPU_SLOTS = 3
MIN_RAM_GPU, MIN_RAM_CPU = 6.0, 4.0
state_f = HERE / 'state.json'
state = json.load(open(state_f)) if state_f.exists() else dict(done={}, failed={})
running = {}                                                       # name -> (Popen, resource, file)


def avail_gb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable'):
            return int(line.split()[1]) / 1e6
    return 0


def stamp(msg):
    with open(HERE / 'runner.log', 'a') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + msg + '\n')


def save():
    tmp = state_f.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, state_f)


def resolve(arg):
    for name, rid in state['done'].items():
        token = '@' + name
        if token in arg:
            arg = arg.replace(token, str(ROOT / 'results' / f'registered_{rid}'))
    assert '@' not in arg, arg
    return arg


def log_path(name):
    (HERE / 'logs').mkdir(exist_ok=True)
    return HERE / 'logs' / (''.join(c if c.isalnum() or c in '-_.' else '_' for c in name) + '.log')


def launch(j, resource):
    args = [resolve(x) for x in j['args']]
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID')
    if j['stage'] == 'bench':
        cmd = [OMPL_PY, RUN, 'bench', *args]
        env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    else:
        cmd = [PY, RUN, j['stage'], *args]
        env['CUDA_VISIBLE_DEVICES'] = resource
    f = open(log_path(j['name']), 'w')
    running[j['name']] = (subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env), resource, f)
    stamp(f'start {j["name"]} on {"GPU " + resource if resource != "cpu" else "CPU"} (RAM {avail_gb():.1f} GB)')


while True:
    jobs = json.load(open(HERE / 'jobs.json'))
    for j in jobs:
        if j['stage'] == 'reuse' and j['name'] not in state['done']:
            state['done'][j['name']] = j['run']
            save()
    for name, (p, res, f) in list(running.items()):
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
    todo = [j for j in jobs if j['name'] not in finished and j['name'] not in running]
    ready = [j for j in todo if all(d in state['done'] for d in j['deps'])]
    if not todo and not running:
        if not (HERE / 'KEEP').exists():
            stamp('queue empty')
            break
    paused = (HERE / 'PAUSE').exists()
    if not paused:
        for gpu in GPUS:
            if any(r == gpu for _, r, _ in running.values()) or avail_gb() < MIN_RAM_GPU:
                continue
            fit = [j for j in ready if j['stage'] in ('prep', 'field') and j.get('gpu') in (None, gpu) and j['name'] not in running]
            if fit:
                launch(fit[0], gpu)
                time.sleep(20)
        order = {j['name']: i for i, j in enumerate(jobs)}
        cpu = sorted([j for j in ready if j['stage'] == 'bench'], key=lambda j: (j.get('table', 9), order[j['name']]))
        for j in cpu:
            if sum(1 for _, r, _ in running.values() if r == 'cpu') >= CPU_SLOTS or avail_gb() < MIN_RAM_CPU:
                break
            if j['name'] not in running:
                launch(j, 'cpu')
                time.sleep(5)
    time.sleep(30)
