"""Queue the long-budget planner evaluations (two-ended, 1,200 and 2,400 steps per end) for every field job in the queue that has none
yet (lift: every seed; ntrl: the first seed). Idempotent.     python queue/addbudget.py"""
import json
from pathlib import Path

Q = Path(__file__).resolve().parent
jobs = [json.loads(l) for l in (Q / 'jobs.jsonl').read_text().splitlines() if l.strip()]
have = {j['id'] for j in jobs}
new = []
for j in jobs:
    if j['mode'] != 'field':
        continue
    name, rest = j['id'].split('_', 1)
    pl, seed = rest.rsplit('_s', 1)
    jid = f'budget_{name}_{pl}_s{seed}'
    if name not in ('lift', 'ntrl') or (name == 'ntrl' and seed != '87358') or jid in have:
        continue
    new.append(dict(id=jid, mode='evalfield', deps=[f'prep_{pl}', j['id']], load=0.15,
                    args=['--prepared', '{job:prep_' + pl + '}', '--fields', name + '={job:' + j['id'] + '}', '--label', f'budget_{pl}' if jid == 'budget_lift_p_d100_s30_h60_s87358' else jid, '--extra', '--planner bidir --steps 1200 2400']))
out = []
for j in jobs + new:
    if j['id'].startswith('hard_'):
        j['args'][-1] = '--planner forward bidir --steps 600 2400 --save-paths'
    out.append(json.dumps(j))
pr = lambda l: {'tests': 0, 'prepare': 1}.get(json.loads(l)['mode'], 2)  # noqa: E731
(Q / 'jobs.jsonl').write_text('\n'.join(sorted(out, key=pr)) + '\n')
print(len(new), 'budget jobs queued')
