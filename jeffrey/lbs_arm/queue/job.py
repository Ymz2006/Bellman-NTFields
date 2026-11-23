"""One queue job of the LBS arm study: register the field run (host venv), then the official evaluation of its checkpoint (container
layout) on the same GPU. Prints {"id": <field run>, "eval": <eval run>} as its last line. A scene with several arms (12 joints) has no
official evaluator: its field run (run12.py) is the result, and "eval" is null.

    python job.py <label> <prepared run dir> <seed> "<pipeline arguments>" <gpu>
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PY = os.path.expanduser('~/.venvs/bellman-ntfield/bin/python')
label, prepared, seed, extra, gpu = sys.argv[1:6]


def run(cmd, env):
    out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    sys.stdout.write(out.stdout[-6000:])
    sys.stdout.write(out.stderr[-6000:])
    rid = None
    for line in out.stdout.splitlines():
        if line.startswith('{"id"'):
            try:
                row = json.loads(line)
                if row.get('state') == 'succeeded':
                    rid = row['id']
            except Exception:
                pass
    return rid


env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', BELLMAN600_BACKEND='host')
if json.load(open(Path(prepared) / 'scene.json'))['describe'].get('dim', 6) > 6:
    # several arms (run12.py): the field run scores its own plans, there is no release evaluator for two arms
    fid = run([PY, 'jeffrey/lbs_arm/run12.py', 'field', '--prepared', prepared, '--seed', seed, '--label', label, '--extra', extra], dict(env, CUDA_VISIBLE_DEVICES=gpu))
    print('FIELD FAILED' if fid is None else json.dumps(dict(id=fid, eval=None)))
    sys.exit(0 if fid else 2)
fid = run([PY, 'jeffrey/lbs_arm/run.py', 'field', '--prepared', prepared, '--seed', seed, '--label', label, '--extra', extra], dict(env, CUDA_VISIBLE_DEVICES=gpu))
if fid is None:
    print('FIELD FAILED')
    sys.exit(2)
env.pop('CUDA_VISIBLE_DEVICES', None)
eid = run([PY, 'jeffrey/lbs_arm/run.py', 'eval', '--field', f'results/registered_{fid}', '--gpu', gpu, '--label', label], env)
print(json.dumps(dict(id=fid, eval=eid)))
sys.exit(0 if eid else 3)
