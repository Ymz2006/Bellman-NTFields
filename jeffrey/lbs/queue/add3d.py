"""Register a prepared 3-D house with the queue: scenes.json[<name>3d] = its prepare run, and the LBS and ntrl-demo jobs at the front of
jobs.txt (asked 2026-11-20: 3-D Gibson, "lbs vs original method only for now").   python queue/add3d.py <name> <results/registered_id> [gpu]"""
import fcntl
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
name, prep = sys.argv[1], sys.argv[2]
gpu = sys.argv[3] if len(sys.argv) > 3 else ''
key = name + '3d'
with open(HERE / '.lock', 'w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    sc = json.load(open(HERE / 'scenes.json'))
    sc[key] = prep
    tmp = HERE / 'scenes.json.tmp'
    tmp.write_text(json.dumps(sc, indent=1) + '\n')
    os.replace(tmp, HERE / 'scenes.json')
    lines = (HERE / 'jobs.txt').read_text().split('\n')
    new = [f'lbs|{key}|87358|--lift time --bell 1 --bell-long 8 --lean --validate-every 1000' + (f'|{gpu}' if gpu else ''),
           f'ntrl|{key}|87358|--lift none --weight 0 --lean --validate-every 1000']
    new = [l for l in new if l not in lines]
    first = next((i for i, l in enumerate(lines) if l.strip() and not l.startswith('#')), len(lines))
    last3 = max([i for i, l in enumerate(lines) if '3d|' in l and not l.startswith('#')], default=first - 1)
    lines[last3 + 1:last3 + 1] = new
    tmp = HERE / 'jobs.txt.tmp'
    tmp.write_text('\n'.join(lines))
    os.replace(tmp, HERE / 'jobs.txt')
print('added', key, prep, len(new), 'jobs')
