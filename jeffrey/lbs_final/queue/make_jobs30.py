"""Write jobs30.json: RRT-Connect and LazyPRM again on every scene with a 30 s limit per query (user request 2026-11-23 01:30;
the first pass used 10 s). One job per scene and planner: bench:<scene>:<rrtc|lazyprm>30. Both planners start every query from scratch
(RRT-Connect: SimpleSetup.clear(); LazyPRM: a fresh roadmap in LazyPRM.solve), as in the 10 s pass. Order = expected cost, longest first
(estimated from the 10 s pass), so the long maze runs do not finish last. Deps: the scene's prep job of the main queue (state.json).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOUSES = ['Allensville', 'Mifflinburg', 'Noxapater', 'Markleeville', 'Lakeville', 'Collierville', 'Benevolence', 'Onaga', 'Coffeen', 'Forkland']
groups = [[f'maze22_{i}' for i in (1, 2, 3)], [f'maze18_{i}' for i in (1, 2, 3)], [f'gap8_{i}' for i in (1, 2, 3)],
          [f'maze14_{i}' for i in (1, 2, 3)], [f'gap4_{i}' for i in (1, 2, 3)], [f'gap2_{i}' for i in (1, 2, 3)],
          [f'maze10_{i}' for i in (1, 2, 3)], [f'gap1_{i}' for i in (1, 2, 3)], [f'{h}3d' for h in HOUSES]]
jobs = []
for g in groups:
    for m in ('rrtc', 'lazyprm'):
        for key in g:
            jobs.append(dict(name=f'bench:{key}:{m}30', stage='bench', args=['--prepared', f'@prep:{key}', '--methods', m, '--time-limit', '30'],
                             deps=[f'prep:{key}']))
json.dump(jobs, open(HERE / 'jobs30.json', 'w'), indent=1)
print(len(jobs), 'jobs')
