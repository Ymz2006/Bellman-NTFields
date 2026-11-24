"""Queue one evaluation per placement that saves the plans the viewer plays: first lift model and the ntrl-demo field, two-ended planner,
2,400 steps per end, planning seed 0, paths saved. Idempotent.     python queue/addviewer.py"""
import json
from pathlib import Path

Q = Path(__file__).resolve().parent
jobs = [json.loads(l) for l in (Q / 'jobs.jsonl').read_text().splitlines() if l.strip()]
state = json.loads((Q / 'state.json').read_text())
have = {j['id'] for j in jobs}
new = []
for j in jobs:
    if j['mode'] != 'field' or not j['id'].startswith('lift_') or not j['id'].endswith('_s87358') or state.get(j['id'], {}).get('status') != 'done':
        continue
    pl = j['id'][len('lift_'):-len('_s87358')]
    jid = f'viewer_{pl}'
    if jid in have:
        continue
    fields, deps = ['lift={job:' + j['id'] + '}'], [f'prep_{pl}', j['id']]
    if state.get(f'ntrl_{pl}_s87358', {}).get('status') == 'done':
        fields.append('ntrl={job:ntrl_' + pl + '_s87358}')
        deps.append(f'ntrl_{pl}_s87358')
    new.append(dict(id=jid, mode='evalfield', deps=deps, load=0.15, args=['--prepared', '{job:prep_' + pl + '}', '--fields', *fields, '--label', jid, '--extra', '--planner bidir --steps 2400 --seeds 0 --save-paths']))
with open(Q / 'jobs.jsonl', 'a') as f:
    for j in new:
        f.write(json.dumps(j) + '\n')
print(len(new), 'viewer jobs queued:', [j['id'] for j in new])
