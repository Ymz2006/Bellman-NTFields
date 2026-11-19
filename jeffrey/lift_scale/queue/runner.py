"""Queue runner of the lift-scale study. jobs.txt lines (processed in file order; a job waits until what it needs exists):

    house|<scene key>|<house.py arguments>          registers stage 'house'; scenes.json[key].prepared = its output
    blocked|<scene key>                             registers stage 'blocked' on scenes.json[key].prepared; sets scenes.json[key].blocked
    field|<label>|<scene key>|<seed>|<pipeline arguments>

Slots: GPU 0 (RTX 5070, drives the display) one run, GPU 1 (RTX 3090) two; house prepares only on GPU 1. RAM guard: no launch below
8 GB available. `touch PAUSE` holds new launches; state.json records the registered id of every finished job.
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
state_f, scenes_f = HERE / 'state.json', HERE / 'scenes.json'
state = json.load(open(state_f)) if state_f.exists() else dict(done={}, failed={})
running = {}


def scenes():
    return json.load(open(scenes_f)) if scenes_f.exists() else {}


def set_scene(key, field, value):
    s = scenes()
    s.setdefault(key, {})[field] = value
    tmp = scenes_f.with_suffix('.tmp')
    tmp.write_text(json.dumps(s, indent=1))
    tmp.replace(scenes_f)


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
    """Registry command for a job, or None while its inputs are missing."""
    parts = j.split('|')
    sc = scenes()
    base = [PY, str(ROOT / 'jeffrey/lift_scale/run.py')]
    if parts[0] == 'house':
        return base + ['--stage', 'house', '--label', parts[1], '--extra', parts[2]]
    if parts[0] == 'blocked':
        prep = sc.get(parts[1], {}).get('prepared')
        return base + ['--stage', 'blocked', '--label', parts[1], '--prepared', str(ROOT / prep)] if prep else None
    _, label, key, seed, extra = parts
    prep, blk = sc.get(key, {}).get('prepared'), sc.get(key, {}).get('blocked')
    if not prep or not blk:
        return None
    return base + ['--stage', 'field', '--label', label, '--prepared', str(ROOT / prep), '--blocked', str(ROOT / blk), '--seed', seed, '--extra', extra]


(HERE / 'logs').mkdir(exist_ok=True)
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
            if ok and j.split('|')[0] in ('house', 'blocked'):
                set_scene(j.split('|')[1], 'prepared' if j.startswith('house') else 'blocked', f'results/registered_{rid}')
            del running[j]
            json.dump(state, open(state_f, 'w'), indent=1)
    todo = [j for j in jobs() if j not in state['done'] and j not in state['failed'] and j not in running]
    if not todo and not running:
        break
    for gpu, n in SLOTS.items():
        while sum(1 for v in running.values() if v[1] == gpu) < n and avail_gb() > 8 and not (HERE / 'PAUSE').exists():
            pick = None
            for j in todo:
                if j.startswith('house') and gpu != '1':
                    continue
                cmd = command(j)
                if cmd is not None:
                    pick = (j, cmd)
                    break
            if pick is None:
                break
            j, cmd = pick
            todo.remove(j)
            f = open(log_path(j), 'w')
            env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=gpu)
            running[j] = (subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env), gpu, f)
            time.sleep(15)
    time.sleep(20)
