"""Append the jobs of placements to queue/jobs.jsonl.

    python queue/add.py place --dx 0.55 --s 0.30 --h 0.60 [--per 50] [--fields ntrl lift] [--seed 87358]
    python queue/add.py base                                   # the earlier study's placement on its frozen 600-query suite
    python queue/add.py field --pl <placement> --name lbs --extra "--lift time --bell 1 ..." [--seed 87359] [--load 0.2]
"""
import argparse
import json
import sys
from pathlib import Path

Q = Path(__file__).resolve().parent
sys.path.insert(0, str(Q.parent))
FIELDS = dict(ntrl='--lift none --weight 0', lift='--lift time --weight 0 --freq 0.5',
              ntrl05='--lift none --weight 0 --freq 0.5', lift1='--lift time --weight 0',
              lbs='--lift time --bell 1 --traversals 4096 --pair-max 0.85 --pair-data --freq 0.5')
LOAD = dict(tests=0.55, prepare=0.2, field=0.08, euclid=0.05)


def field_job(pl, name, seed, extra=None, load=None):
    return dict(id=f'{name}_{pl}_s{seed}', mode='field', deps=[f'prep_{pl}'], load=load or LOAD['field'],
                args=['--prepared', '{job:prep_' + pl + '}', '--seed', str(seed), '--label', f'{name}_{pl}', '--extra', extra or FIELDS[name]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('what', choices=['place', 'base', 'field'])
    p.add_argument('--dx', type=float)
    p.add_argument('--s', type=float, default=0.30)
    p.add_argument('--h', type=float, default=0.60)
    p.add_argument('--per', type=int, default=50)
    p.add_argument('--iterations', type=int, default=3000)
    p.add_argument('--fields', nargs='*', default=['ntrl', 'lift'])
    p.add_argument('--seed', type=int, default=87358)
    p.add_argument('--pl')
    p.add_argument('--name')
    p.add_argument('--extra')
    p.add_argument('--load', type=float)
    a = p.parse_args()
    have = {json.loads(line)['id'] for line in (Q / 'jobs.jsonl').read_text().splitlines() if line.strip() and not line.startswith('#')}
    jobs = []
    if a.what == 'field':
        jobs.append(field_job(a.pl, a.name, a.seed, a.extra, a.load))
    else:
        if a.what == 'base':
            pl = 'base'
            jobs.append(dict(id='prep_base', mode='prepare', deps=[], load=LOAD['prepare'], args=['--scene', 'real_world_12dof', '--tests', 'jeffrey/lbs_arm/tests_real_world_12dof', '--label', 'base']))
        else:
            import place
            pl = place.name_of(a.dx, a.s, a.h)
            jobs.append(dict(id=f'tests_{pl}', mode='tests', deps=[], load=LOAD['tests'], args=['--dx', str(a.dx), '--s', str(a.s), '--h', str(a.h), '--label', pl, '--extra', f'--per {a.per} --iterations {a.iterations}']))
            jobs.append(dict(id=f'prep_{pl}', mode='prepare', deps=[f'tests_{pl}'], load=LOAD['prepare'], args=['--scene', pl, '--tests', '{job:tests_' + pl + '}', '--label', pl]))
        jobs.append(dict(id=f'euclid_{pl}', mode='euclid', deps=[f'prep_{pl}'], load=LOAD['euclid'], args=['--prepared', '{job:prep_' + pl + '}', '--label', pl]))
        jobs += [field_job(pl, f, a.seed) for f in a.fields]
    with open(Q / 'jobs.jsonl', 'a') as f:
        for j in jobs:
            if j['id'] in have:
                print('already queued:', j['id'])
                continue
            f.write(json.dumps(j) + '\n')
            print('queued', j['id'])


if __name__ == '__main__':
    main()
