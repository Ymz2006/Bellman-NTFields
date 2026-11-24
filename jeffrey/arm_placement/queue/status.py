"""Print the queue: python queue/status.py [-v]"""
import json
import sys
import time
from pathlib import Path

Q = Path(__file__).resolve().parent
ROOT = Q.parents[2]
state = json.loads((Q / 'state.json').read_text())
jobs = [json.loads(line) for line in (Q / 'jobs.jsonl').read_text().splitlines() if line.strip() and not line.startswith('#')]
now = time.time()
count = {}
for j in jobs:
    s = state.get(j['id'], dict(status='pending'))
    count[s['status']] = count.get(s['status'], 0) + 1
    if s['status'] == 'pending' and '-v' not in sys.argv:
        continue
    line = f"{j['id']:34s} {s['status']:8s}"
    if 'start' in s:
        line += f" gpu{s['gpu']} {((s.get('end') or now) - s['start']) / 60:6.1f} min"
    if s.get('run'):
        line += f"  {s['run']}"
        r = ROOT / f"results/registered_{s['run']}/results.json"
        if r.exists() and j['mode'] in ('field', 'euclid'):
            d = json.loads(r.read_text())
            suc = d.get('success') or d['runs'][0]['success']
            line += f"  {suc[0]}/{suc[1]} = {100 * suc[0] / suc[1]:.1f}%"
    print(line)
print(count)
