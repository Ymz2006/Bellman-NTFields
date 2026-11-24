"""Write queue/jobs.json: the full job list of the study (re-run after editing; finished jobs are kept by name in state.json).

Phases
  dist      learned clearance field (grid) of every scene; an MLP of two sizes on three scenes for the comparison
  ladder    Task 1 on four scenes, first 100 queries: the reference and every optimization step, on the CPU (1 / 2 / 4 / 8 threads)
            and on the GPU
  knobs     step multiplier and sample count on four scenes, 500 queries, GPU
  ref       the reference re-timed on an idle machine, first 100 queries of every scene, CPU 2 threads
  final     every scene, 500 queries: the fast planner one-way / bidirectional, exact / learned clearance, on the GPU and on the CPU
"""
import json
from pathlib import Path

Q = Path(__file__).resolve().parent
SCENES = json.load(open(Q.parent / 'scenes.json'))
KEYS = list(SCENES)
LADDER = ['maze10_1', 'maze22_1', 'gap4_1', 'Onaga3d']
GRID = '{"kind": "grid", "steps": 6000, "batch": 32768}'
jobs = []


def add(name, lane, *args):
    jobs.append(dict(name=name, lane=lane, args=[str(x) for x in args]))


def V(*specs):
    return [x for s in specs for x in ('--variant', s)]


# --- dist ------------------------------------------------------------------------------------------------------------
order = LADDER + [k for k in KEYS if k not in LADDER]
for i, k in enumerate(order):
    add(f'dist:{k}:grid', f'train{i % 2}', 'dist', '--scene', k, '--gpu', i % 2, '--cfg', GRID, '--label', 'grid')
for i, k in enumerate(['maze10_1', 'maze22_1', 'Onaga3d']):
    add(f'dist:{k}:mlp128', f'train{i % 2}', 'dist', '--scene', k, '--gpu', i % 2, '--cfg', '{"kind": "mlp", "width": 128, "steps": 6000, "batch": 32768}', '--label', 'mlp128')
    add(f'dist:{k}:mlp512', f'train{(i + 1) % 2}', 'dist', '--scene', k, '--gpu', (i + 1) % 2, '--cfg', '{"kind": "mlp", "width": 512, "depth": 4, "feats": 128, "steps": 6000, "batch": 32768}', '--label', 'mlp512')

# --- ladder (Task 1): four scenes, first 100 queries ------------------------------------------------------------------------
for k in LADDER:
    add(f'ladder:{k}:cpu2', 'timed', 'bench', '--scene', k, '--device', 'cpu', '--threads', 2, '--limit', 100, '--label', 'ladder',
        *V('ref|v=ref', 'cache|c=none', 'compiled|c=default', 'nat1|c=native,t=1', 'nat2|c=native,t=2', 'nat4|c=native,t=4', 'nat8|c=native,t=8',
           'nat8_s25|c=native,t=8,s=25', 'nat8_s12|c=native,t=8,s=12', 'nat8_meet|v=meet,c=native,t=8', 'nat8_meet_m2|v=meet,c=native,t=8,m=2'))
    add(f'ladder:{k}:cpu8', 'timed', 'bench', '--scene', k, '--device', 'cpu', '--threads', 8, '--limit', 100, '--label', 'ladder', *V('ref|v=ref', 'cache|c=none', 'compiled|c=default'))
    add(f'ladder:{k}:cpu1', 'timed', 'bench', '--scene', k, '--device', 'cpu', '--threads', 1, '--limit', 100, '--label', 'ladder', *V('ref|v=ref,q=30', 'compiled|c=default'))
    add(f'ladder:{k}:cpu4', 'timed', 'bench', '--scene', k, '--device', 'cpu', '--threads', 4, '--limit', 100, '--label', 'ladder', *V('ref|v=ref,q=30', 'compiled|c=default'))
    add(f'ladder:{k}:gpu', 'timed', 'bench', '--scene', k, '--device', 'cuda', '--gpu', 1, '--limit', 100, '--label', 'ladder',
        *V('ref|v=ref', 'cache|c=none', 'compiled|c=default', 'graph|c=ro', 'graph_k8|c=ro,k=8', 'graph_k8_bulk|c=ro,k=8,n=bulk', 'graph_k16_bulk|c=ro,k=16,n=bulk',
           'graph_k4_bulk|c=ro,k=4,n=bulk'))

# --- knobs: step multiplier, samples, bidirectional variants; GPU, first 300 queries ------------------------------------------------
G = 'c=ro,k=8,n=bulk'
for k in LADDER:
    add(f'knobs:{k}:gpu', 'timed', 'bench', '--scene', k, '--device', 'cuda', '--gpu', 1, '--limit', 300, '--label', 'knobs', '--dist', f'grid={{run:dist:{k}:grid}}',
        *V(f'goal|{G}', f'goal_m1.5|{G},m=1.5', f'goal_m2|{G},m=2', f'goal_m3|{G},m=3',
           f'meet|v=meet,{G}', f'meet_m1.5|v=meet,{G},m=1.5', f'meet_m2|v=meet,{G},m=2', f'meet_m3|v=meet,{G},m=3',
           f'goal_s12|{G},s=12', f'goal_s25|{G},s=25', f'goal_s200|{G},s=200', f'meet_s25|v=meet,{G},s=25', f'meet_m2_s200|v=meet,{G},m=2,s=200',
           f'both|v=both,{G}', f'both_mt3|v=both,{G},mt=3', f'both_m2|v=both,{G},m=2'))
for k in ['maze10_1', 'maze22_1', 'Onaga3d']:
    d = ['--dist', f'grid={{run:dist:{k}:grid}}', '--dist', f'mlp128={{run:dist:{k}:mlp128}}', '--dist', f'mlp512={{run:dist:{k}:mlp512}}']
    add(f'distcmp:{k}:gpu', 'timed', 'bench', '--scene', k, '--device', 'cuda', '--gpu', 1, '--limit', 300, '--label', 'distcmp', *d,
        *V(f'exact|{G}', f'grid|{G},dist=grid', f'mlp128|{G},dist=mlp128', f'mlp512|{G},dist=mlp512',
           f'meet_exact|v=meet,{G}', f'meet_grid|v=meet,{G},dist=grid', f'meet_mlp128|v=meet,{G},dist=mlp128', f'meet_mlp512|v=meet,{G},dist=mlp512'))
    add(f'distcmp:{k}:cpu2', 'timed', 'bench', '--scene', k, '--device', 'cpu', '--threads', 2, '--limit', 60, '--label', 'distcmp', *d,
        *V('exact|c=default,n=bulk', 'grid|c=default,n=bulk,dist=grid', 'mlp128|c=default,n=bulk,dist=mlp128', 'mlp512|c=default,n=bulk,dist=mlp512'))

# --- final: every scene, first 250 queries; instance 1 and the houses first ----------------------------------------------------------
first = [k for k in KEYS if k.endswith('_1') or k.endswith('3d')]
for k in first + [k for k in KEYS if k not in first]:
    add(f'final:{k}:gpu', 'timed', 'bench', '--scene', k, '--device', 'cuda', '--gpu', 1, '--limit', 250, '--label', 'final', '--dist', f'grid={{run:dist:{k}:grid}}',
        *V(f'goal|{G}', f'meet|v=meet,{G}', f'meet_m2|v=meet,{G},m=2', f'goal_grid|{G},dist=grid', f'meet_m2_grid|v=meet,{G},m=2,dist=grid'))
    add(f'final:{k}:cpu', 'timed', 'bench', '--scene', k, '--device', 'cpu', '--threads', 2, '--limit', 250, '--label', 'final',
        *V('ref|v=ref,q=20', 'nat8|c=native,t=8', 'nat8_meet_m2|v=meet,c=native,t=8,m=2', 'nat8_m2_s25|c=native,t=8,m=2,s=25', 'nat8_meet_m2_s25|v=meet,c=native,t=8,m=2,s=25'))

# --- direction control: is the bidirectional gain "either direction arrives"? scenes where the one-way planner misses queries ---------
for k in ['maze22_1', 'maze22_2', 'maze22_3', 'maze18_1', 'gap8_1', 'Lakeville3d', 'Allensville3d', 'Coffeen3d']:
    add(f'direction:{k}:gpu', 'timed', 'bench', '--scene', k, '--device', 'cuda', '--gpu', 1, '--limit', 250, '--label', 'direction',
        *V(f'goal|{G}', f'goal_rev|{G},rev=1', f'meet|v=meet,{G}', f'both|v=both,{G}'))

json.dump(jobs, open(Q / 'jobs.json.new', 'w'), indent=1)
print(len(jobs), 'jobs', {l: sum(j['lane'] == l for j in jobs) for l in ('train0', 'train1', 'timed')})
