"""LBS rerun with the fast planner (user request 2026-11-23 ~16:30): jeffrey/lbs_fast fast.Planner(field, settings, variant='goal',
compile='reduce-overhead', block=8, noise='bulk') on the RTX 5070 (GPU 0, given to this study alone), all 500 queries of every scene,
one scene at a time. Each job is a registered `jeffrey/lbs_fast/run.py bench` run (frozen lbs_fast sources; the same prepared scenes and
LBS fields as this study, checked in lbs_fast/scenes.json); its bench.py scores every path with maze_scale/plan.score off the clock.

Waits while any compute process other than the display (chrome / Xorg / gnome) is on GPU 0, so the timed runs have the GPU alone.
statefast.json (job -> run id), runner_fast.log; `touch PAUSEFAST` holds new launches; exits when everything is done.
"""
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PY = os.path.expanduser('~/.venvs/bellman-ntfield/bin/python')
RUN = str(ROOT / 'jeffrey/lbs_fast/run.py')
VARIANT = 'fast|c=ro,k=8,n=bulk'
GPU0 = 'GPU-a366ba00-bbbf-15dd-d6ab-04d19ea7f5cd'
DISPLAY = ('chrome', 'Xorg', 'gnome', 'firefox', 'Xwayland')
HOUSES = ['Collierville', 'Benevolence', 'Onaga', 'Coffeen', 'Forkland', 'Allensville', 'Mifflinburg', 'Noxapater', 'Markleeville', 'Lakeville']
SCENES = ([f'maze{n}_{i}' for n in (10, 14, 18, 22) for i in (1, 2, 3)] + [f'gap{k}_{i}' for k in (1, 2, 4, 8) for i in (1, 2, 3)]
          + [h + '3d' for h in HOUSES])
state_f = HERE / 'statefast.json'
state = json.load(open(state_f)) if state_f.exists() else dict(done={}, failed={})


def stamp(msg):
    with open(HERE / 'runner_fast.log', 'a') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + msg + '\n')


def save():
    tmp = state_f.with_suffix('.tmpfast')
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, state_f)


def others_on_gpu0():
    out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,process_name', '--format=csv,noheader'], capture_output=True, text=True).stdout
    bad = []
    for line in out.splitlines():
        pid, uuid, name = [x.strip() for x in line.split(',', 2)]
        if uuid == GPU0 and not any(d in name for d in DISPLAY):
            bad.append(f'{pid} {name[:60]}')
    return bad


for key in SCENES:
    name = f'fast:{key}'
    if name in state['done']:
        continue
    while (HERE / 'PAUSEFAST').exists() or others_on_gpu0():
        b = others_on_gpu0()
        if b:
            stamp(f'waiting: GPU 0 busy ({"; ".join(b)})')
        time.sleep(60)
    (HERE / 'logs').mkdir(exist_ok=True)
    log = HERE / 'logs' / f'fast_{key}.log'
    stamp(f'start {name}')
    with open(log, 'w') as f:
        p = subprocess.run([PY, RUN, 'bench', '--scene', key, '--device', 'cuda', '--gpu', '0', '--threads', '2', '--variant', VARIANT,
                            '--label', 'lbsfinal'], stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    rid = None
    for line in log.read_text().splitlines():
        if line.startswith('{"id"') and '"succeeded"' in line:
            rid = json.loads(line)['id']
    if p.returncode == 0 and rid:
        state['done'][name] = rid
    else:
        state['failed'][name] = f'exit {p.returncode}'
    save()
    stamp(f'{"done" if rid else "FAILED"} {name} -> {rid} (exit {p.returncode})')
    if not rid:
        stamp('stopping after a failure')
        break
stamp('queue empty' if len(state['done']) == len(SCENES) else 'stopped')
