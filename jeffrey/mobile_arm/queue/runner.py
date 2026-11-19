"""Queue runner of the mobile-arm study. jobs.txt lines, processed in file order:

    <label>|<house key>|<seed>|<pipeline arguments>        registers `jeffrey/mobile_arm/run.py field` on houses.json[key]

'@base' in the arguments stands for the epoch-2,000 checkpoint of the finished job 'ntrl|<house>|<seed>|--weight 0' (the paired
midpoint and control runs resume it); such a job waits until that baseline is done. '@gpuN' pins a job to GPU N (the 5070 runs the
midpoint refresh 1.8x slower than the 3090, so the large houses' midpoint runs go to the 3090); it is not passed on.

One run per GPU (SLOTS), at most MAX_MINE at once, no launch below 8 GB of available RAM. `touch PAUSE` holds new launches; state.json
records the registered id of every finished job. adopt.json ({job: gpu}) lists jobs a previous runner started that are still running: they
hold their GPU until their log shows the registry's final state line.
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


def jobs():
    return [l.strip() for l in open(HERE / 'jobs.txt') if l.strip() and not l.startswith('#')]


def log_path(j):
    return HERE / 'logs' / (''.join(c if c.isalnum() or c in '-_.' else '_' for c in j)[:180] + '.log')


def pinned(j):
    for tok in j.split('|')[3].split():
        if tok.startswith('@gpu'):
            return tok[4:]
    return None


def command(j):
    label, key, seed, extra = j.split('|')
    extra = ' '.join(t for t in extra.split() if not t.startswith('@gpu'))
    prep = json.load(open(HERE / 'houses.json'))[key]
    if '@base' in extra:
        rid = state['done'].get(f'ntrl|{key}|{seed}|--weight 0')
        if rid is None:
            return None
        ck = ROOT / 'results' / f'registered_{rid}' / 'training/checkpoints/epoch_02000.pt'
        if not ck.exists():
            return None
        extra = extra.replace('@base', str(ck))
    return [PY, str(ROOT / 'jeffrey/mobile_arm/run.py'), 'field', '--prepared', str(ROOT / prep), '--seed', seed, '--label', label, '--extra', extra]


def stamp(msg):
    with open(HERE / 'runner.log', 'a') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + msg + '\n')


def outcome(path):
    """(registered id, final state or None) from a job log."""
    rid, final = None, None
    for line in path.read_text(errors='replace').splitlines():
        if line.startswith('{"id"'):
            try:
                d = json.loads(line)
                rid = d['id']
                final = d.get('state', final)
            except Exception:
                pass
    return rid, final


if (HERE / 'adopt.json').exists():
    for j, gpu in json.loads((HERE / 'adopt.json').read_text()).items():
        if j not in state['done'] and j not in state['failed']:
            running[j] = (None, str(gpu), None)
            stamp(f'adopted {j} on GPU {gpu}')

while True:
    for j, (p, gpu, f) in list(running.items()):
        if p is None:                                          # adopted: finished when the registry's final line is in its log
            rid, final = outcome(log_path(j))
            if final is None:
                continue
            ok = final == 'succeeded'
        else:
            if p.poll() is None:
                continue
            f.close()
            rid, final = outcome(log_path(j))
            ok = p.returncode == 0 and rid is not None
        (state['done'] if ok else state['failed'])[j] = rid
        del running[j]
        json.dump(state, open(state_f, 'w'), indent=1)
        stamp(f'{"done" if ok else "FAILED"} {j} -> {rid}' + ('' if p is None else f' (exit {p.returncode})'))
    pending = [j for j in jobs() if j not in state['done'] and j not in state['failed'] and j not in running]
    if not pending and not running:
        stamp('queue empty')
        break
    for gpu, n in SLOTS.items():
        if sum(1 for v in running.values() if v[1] == gpu) >= n or len(running) >= MAX_MINE:
            continue
        if avail_gb() < 8 or (HERE / 'PAUSE').exists():
            continue
        todo = [j for j in pending if j not in running and pinned(j) in (None, gpu) and command(j) is not None]
        if not todo:
            continue
        j = todo[0]
        f = open(log_path(j), 'w')
        env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=gpu)
        running[j] = (subprocess.Popen(command(j), stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env), gpu, f)
        stamp(f'start {j} on GPU {gpu} (RAM {avail_gb():.1f} GB)')
        time.sleep(20)
    time.sleep(30)
