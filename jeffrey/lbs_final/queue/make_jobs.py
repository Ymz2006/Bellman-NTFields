"""Write jobs.json for the LBS final benchmark queue (runner.py). Job names: prep:<scene>, field:<scene>:<arm>, bench:<scene>:<group>.

Scenes (instance i = maze seed 1000 i + n, gap base seed 1000 i + 10):
  Table 1  maze10/14/18/22, instances 1-3
  Table 2  gap1/2/4/8 (10 x 10 base), instances 1-3
  Table 3  ten 3-D Gibson houses: the five of jeffrey/lbs and the five smallest main floors among the other gibson_tiny houses
Fields reused from jeffrey/lbs (same scene: bench.py checks the walls / table): instance 1 of maze10, maze18, gap1, gap2, gap4 and the five
earlier 3-D houses, both arms (their prepare runs had the binned queries; the new prepare runs of the same scenes carry the random ones).
Order = priority: all preps (short; the CPU benchmarks of the sampling planners start as soon as a scene is prepared), then the trainings
table by table, scene by scene (ntrl-demo, then LBS).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LBS = '--lift time --bell 1 --bell-long {bl} --lean{v}'
NTRL = '--lift none --weight 0 --lean{v}'
HOUSES_OLD = ['Allensville', 'Mifflinburg', 'Noxapater', 'Markleeville', 'Lakeville']
HOUSES_NEW = ['Collierville', 'Benevolence', 'Onaga', 'Coffeen', 'Forkland']
REUSE = {('maze10_1', 'lbs'): '20261120T024416Z-e62e1e8c', ('maze10_1', 'ntrl'): '20261120T024436Z-582ae344',
         ('maze18_1', 'lbs'): '20261120T053456Z-a02310b0', ('maze18_1', 'ntrl'): '20261120T060816Z-2cdfe357',
         ('gap1_1', 'lbs'): '20261120T082336Z-33650f07', ('gap1_1', 'ntrl'): '20261120T082856Z-8764ef43',
         ('gap2_1', 'lbs'): '20261120T095336Z-719315ba', ('gap2_1', 'ntrl'): '20261120T104726Z-007aeff7',
         ('gap4_1', 'lbs'): '20261120T112717Z-11b4d878', ('gap4_1', 'ntrl'): '20261120T124626Z-146b5fcf',
         ('Allensville3d', 'lbs'): '20261120T235517Z-71d6a2c5', ('Allensville3d', 'ntrl'): '20261121T005037Z-585c4d0f',
         ('Mifflinburg3d', 'lbs'): '20261120T232237Z-6dfa59f7', ('Mifflinburg3d', 'ntrl'): '20261120T234857Z-2727e491',
         ('Noxapater3d', 'lbs'): '20261121T010917Z-fd0e606d', ('Noxapater3d', 'ntrl'): '20261121T005627Z-d6eee0aa',
         ('Markleeville3d', 'lbs'): '20261121T021707Z-9645d217', ('Markleeville3d', 'ntrl'): '20261121T010147Z-51711f85',
         ('Lakeville3d', 'lbs'): '20261121T010828Z-3fc6f1af', ('Lakeville3d', 'ntrl'): '20261121T010238Z-6ac4101f'}

scenes = []                                    # (key, table, prep args, bell_long, validate, gpu pin for LBS)
for n in (10, 14, 18, 22):
    for i in (1, 2, 3):
        scenes.append((f'maze{n}_{i}', 1, ['--kind', 'maze', '--n', str(n), '--instance', str(i)], 1, '', None))
for k in (1, 2, 4, 8):
    for i in (1, 2, 3):
        scenes.append((f'gap{k}_{i}', 2, ['--kind', 'gap', '--scale', str(k), '--instance', str(i)], 1 if 10 + 6 * k <= 24 else 8, '', '1' if k == 8 else None))
for h in HOUSES_OLD + HOUSES_NEW:
    scenes.append((f'{h}3d', 3, ['--kind', 'house3', '--name', h], 8, ' --validate-every 1000', None))

jobs = []
for key, table, args, bl, v, pin in scenes:
    jobs.append(dict(name=f'prep:{key}', stage='prep', args=args, deps=[], gpu=None, table=table))
for key, table, args, bl, v, pin in scenes:
    jobs.append(dict(name=f'bench:{key}:sampling', stage='bench', args=['--prepared', f'@prep:{key}', '--methods', 'rrtc,lazyprm'], deps=[f'prep:{key}'], gpu='cpu', table=table))
for key, table, args, bl, v, pin in scenes:
    for arm, extra in (('ntrl', NTRL.format(v=v)), ('lbs', LBS.format(bl=bl, v=v))):
        if (key, arm) in REUSE:
            jobs.append(dict(name=f'field:{key}:{arm}', stage='reuse', run=REUSE[(key, arm)], deps=[], table=table))
        else:
            jobs.append(dict(name=f'field:{key}:{arm}', stage='field', args=['--prepared', f'@prep:{key}', '--label', arm, '--extra', extra], deps=[f'prep:{key}'],
                             gpu=pin if arm == 'lbs' else None, table=table))
    jobs.append(dict(name=f'bench:{key}:learned', stage='bench', args=['--prepared', f'@prep:{key}', '--methods', 'ntrl,lbs', '--field', f'ntrl=@field:{key}:ntrl',
                                                                      '--field', f'lbs=@field:{key}:lbs'], deps=[f'prep:{key}', f'field:{key}:ntrl', f'field:{key}:lbs'], gpu='cpu', table=table))
json.dump(jobs, open(HERE / 'jobs.json', 'w'), indent=1)
print(len(jobs), 'jobs:', {s: sum(j['stage'] == s for j in jobs) for s in ('prep', 'field', 'reuse', 'bench')})
