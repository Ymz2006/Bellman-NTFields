"""Queue the hard set of one placement: a larger suite of the zone -> another zone stratum only, its prepared queries, and the
planners on the placement's trained fields.     python queue/addhard.py --dx 1.0 --s 0.30 --h 0.60 [--per 200] [--seeds 87358 87359]"""
import argparse
import json
import sys
from pathlib import Path

Q = Path(__file__).resolve().parent
sys.path.insert(0, str(Q.parent))
import place  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--dx', type=float, required=True)
p.add_argument('--s', type=float, default=0.30)
p.add_argument('--h', type=float, default=0.60)
p.add_argument('--per', type=int, default=200)
p.add_argument('--seeds', type=int, nargs='+', default=[87358, 87359])
a = p.parse_args()
pl = place.name_of(a.dx, a.s, a.h)
jobs = [dict(id=f'tests_hard_{pl}', mode='tests', deps=[], load=0.55, args=['--dx', str(a.dx), '--s', str(a.s), '--h', str(a.h), '--label', f'hard_{pl}', '--extra', f'--per {a.per} --iterations 3000 --strata compartment_to_compartment']),
        dict(id=f'prep_hard_{pl}', mode='prepare', deps=[f'tests_hard_{pl}'], load=0.2, args=['--scene', pl, '--tests', '{job:tests_hard_' + pl + '}', '--label', f'hard_{pl}', '--extra', '--pairs 20000'])]
for name, seeds in (('lift', a.seeds), ('ntrl', a.seeds[:1])):
    for seed in seeds:
        jobs.append(dict(id=f'hard_{name}_{pl}_s{seed}', mode='evalfield', deps=[f'prep_hard_{pl}', f'{name}_{pl}_s{seed}'], load=0.15,
                         args=['--prepared', '{job:prep_hard_' + pl + '}', '--fields', name + '={job:' + f'{name}_{pl}_s{seed}' + '}', '--label', f'hard_{name}_{pl}_s{seed}', '--extra', '--planner forward bidir --steps 600 --save-paths']))
have = {json.loads(l)['id'] for l in (Q / 'jobs.jsonl').read_text().splitlines() if l.strip()}
lines = [l for l in (Q / 'jobs.jsonl').read_text().splitlines() if l.strip()] + [json.dumps(j) for j in jobs if j['id'] not in have]
pr = lambda l: {'tests': 0, 'prepare': 1}.get(json.loads(l)['mode'], 2)  # noqa: E731
(Q / 'jobs.jsonl').write_text('\n'.join(sorted(lines, key=pr)) + '\n')
print('queued', [j['id'] for j in jobs if j['id'] not in have])
