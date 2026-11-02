"""Build the P-NTFields path viewer: Task 1 and Task 2 exports + kinematics.js inlined into template.html -> viewer.html.

  sh baselines/P-NTFields/docker/run.sh python eval/visualize_export.py                 # (in baselines/P-NTFields) Task 1 json
  ~/.venvs/bellman-ntfield/bin/python experiments/pntfields_cabinet/visualizer/export_cabinet.py   # Task 2 json
  python3 experiments/pntfields_cabinet/visualizer/build.py                            # -> viewer.html
Open viewer.html in a browser (three.js loads from cdn.jsdelivr.net). The page checks its kinematics against the
Python reference poses in both exports and prints the result in its footer.
"""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
T1 = json.loads((ROOT/'baselines/P-NTFields/results/visualize_task1.json').read_text())
T2 = json.loads((HERE/'cabinet.json').read_text())

PRETTY = {'free_to_free': 'free → free', 'free_to_compartment': 'free → compartment', 'compartment_to_free': 'compartment → free',
          'compartment_to_compartment': 'compartment → compartment', 'same_compartment': 'same compartment', 'anywhere': 'anywhere'}
NTRL_RUN_NOTE = {'cabinet': 'ntrl-demo run 664b5d50, its best cabinet run (metric arm, ntrl coordinates, environment label)',
                 'cabinet_pi': 'ntrl-demo run 379bcde5', 'cabinet_w25': 'ntrl-demo run b25f368e (baseline settings)',
                 'cabinet_w20': 'ntrl-demo run c32709b5 (baseline settings)', 'cabinet_w15': 'ntrl-demo run b00193b9 (baseline settings)'}
TAB = {'cabinet': 'cabinet', 'cabinet_pi': 'cabinet_pi (±π joints)', 'cabinet_w25': 'cabinet_w25', 'cabinet_w20': 'cabinet_w20', 'cabinet_w15': 'cabinet_w15'}

style = {'retrained': dict(color='--pnt', dash=None), 'pretrained': dict(color='--released', dash=None),
         'witness': dict(color='--witness', dash=[3, 2]), 'pnt_plan': dict(color='--pnt', dash=[7, 3]), 'pnt_ntrl': dict(color='--pnt', dash=None),
         'ntrl_ntrl': dict(color='--ntrl', dash=None), 'pnt_mppi': dict(color='--pnt', dash=[2, 2]), 'ntrl_mppi': dict(color='--ntrl', dash=[2, 2]),
         'pnt_rk4': dict(color='--pnt', dash=[1, 3]), 'ntrl_rk4': dict(color='--ntrl', dash=[1, 3])}
sources = dict(T2['sources'], retrained='Retrained here (10k epochs)', pretrained='Released checkpoint (2023)')


def task1():
    tag = {(False, True): 'retrain fixes', (True, True): 'both solve', (False, False): 'both fail', (True, False): 'retrain breaks'}
    queries = [dict(group='Uniform random pairs' if q['set'] == 'uniform' else 'Both ends near the cabinet', name=f"Query {q['query']}",
                    detail=tag[(q['plans']['pretrained']['success'], q['plans']['retrained']['success'])], plans=q['plans']) for q in T1['queries']]
    t = T1['totals']
    row = lambda k, kind: dict(key=k, label=sources[k], short='Retrained' if k == 'retrained' else 'Released', success=t[k][kind]['success'], n=t[k][kind]['n'])
    start = next(i for i, q in enumerate(queries) if q['group'].startswith('Both') and q['detail'] == 'retrain fixes')
    return dict(id='task1', tabKey='Task 1', tab="P-NTFields' UR5 cabinet", title="P-NTFields' own UR5 cabinet (its released scene)", robot='ur5',
                obstacle=dict(mesh=T1['obstacle']), sourceOrder=['retrained', 'pretrained'], keySources=['retrained', 'pretrained'],
                defaultShown=['retrained', 'pretrained'], defaultActive='retrained', startQuery=start, queries=queries,
                totals=[dict(label='Uniform random pairs, 1000 queries', rows=[row('retrained', 'uniform'), row('pretrained', 'uniform')], ref=dict(value=92.0, label='paper: 92.0% on 2 × 100 pairs')),
                        dict(label='Both ends within 3.5 cm of the cabinet, 1000 queries', rows=[row('retrained', 'near'), row('pretrained', 'near')])],
                totalsNote="P-NTFields' own planner (test/arm_plan.py, β 0.015, goal tolerance 0.03). Success = reaches the goal and the exact link meshes never touch the cabinet mesh.",
                queriesNote='Picked so the two checkpoints agree and disagree in both directions. Marks: retrained, released.',
                contactRule='exact link-mesh contact with the cabinet mesh; self contact is not part of the paper metric', selfScored=False, contactMargin=0, clearanceCap=.15, clearanceTicks=[.05, .1, .15],
                clearanceLabel='link-mesh clearance to the cabinet (exact)', view=dict(eye=[1.15, 2.2, 1.25], target=[0.0, -0.3, 0.25]))


def task2(S):
    queries = [dict(group=PRETTY[q['suite']], suite=q['suite'], name=f"Query {q['query']}", detail=f"{q['start_region']} → {q['goal_region']}", plans=q['plans']) for q in S['queries']]
    t = S['totals']
    row = lambda k: dict(key=k, label=sources[k], short='P-NTFields' if k.startswith('pnt') else 'ntrl-demo', success=t[k]['success'], n=t[k]['n'], per_suite=t[k]['per_suite'])
    start = next((i for i, q in enumerate(queries) if q['suite'] == 'free_to_compartment' and q['plans']['pnt_ntrl']['success'] and not q['plans']['ntrl_ntrl']['success']), 0)
    return dict(id=S['scene'], tabKey='Task 2', tab=TAB[S['scene']], title=f"Your {S['scene']} scene", robot='ur5e', obstacle=dict(cuboids=S['cuboids']),
                sourceOrder=['witness', 'pnt_plan', 'pnt_ntrl', 'ntrl_ntrl', 'pnt_mppi', 'ntrl_mppi', 'pnt_rk4', 'ntrl_rk4'], keySources=['pnt_ntrl', 'ntrl_ntrl', 'pnt_plan'],
                defaultShown=['witness', 'pnt_plan', 'pnt_ntrl', 'ntrl_ntrl'], defaultActive='pnt_ntrl', startQuery=start, queries=queries,
                totals=[dict(label="P-NTFields' own planner, 600 queries", rows=[row('pnt_plan')]),
                        dict(label="ntrl-demo's MPPI, 600 queries × 3 seeds", rows=[row('pnt_ntrl'), row('ntrl_ntrl')]),
                        dict(label='Release MPPI, 600 queries × 3 seeds', rows=[row('pnt_mppi'), row('ntrl_mppi')]),
                        dict(label='Native RK4, 600 queries', rows=[row('pnt_rk4'), row('ntrl_rk4')])],
                totalsNote=f"Release scorer in the scene, tilt gate off. P-NTFields run {S['runs']['pntfields'][-8:]}; {NTRL_RUN_NOTE[S['scene']]}. MPPI paths shown are planning seed 0.",
                queriesNote="Four per suite, picked so the two fields under ntrl-demo's MPPI disagree in both directions, both solve and both fail. Marks: P-NTFields + ntrl-demo MPPI, ntrl-demo + ntrl-demo MPPI, P-NTFields planner.",
                contactRule='release sphere model: cabinet < 5 mm or self < 2 mm, as the scorer', selfScored=True, contactMargin=.005, clearanceCap=.1, clearanceTicks=[.025, .05, .1],
                clearanceLabel='release-sphere clearance to the cabinet', view=dict(eye=[-0.25, -1.5, 1.05], target=[0.55, 0.2, 0.25]))


data = dict(
    lede="Planned paths from Task 1 (P-NTFields reproduced on its own UR5 cabinet) and Task 2 (P-NTFields against ntrl-demo on your cabinets), "
         "replayed on the real arm geometry. Red frames are where the arm touches the cabinet under that scene's collision test.",
    robots=dict(ur5=dict(name='UR5', base=T1['base'], joints=T1['joints'], meshes=T1['meshes'], ee_link='ee_link', reference=T1['reference']),
                ur5e=dict(name='UR5e + Robotiq 2F-140', base=T2['base'], joints=T2['joints'], meshes=T2['meshes'], ee_link=T2['ee_link'], reference=T2['reference'])),
    sources=sources, sourceStyle=style,
    scenes=[task1()]+[task2(S) for S in T2['scenes']],
    about=["Task 1: paths replanned by baselines/P-NTFields/eval/visualize_export.py with the evaluation's planner and queries; every verdict matches results/task1_*_{uniform,near}.json.",
           "Task 2: saved paths of the registered runs, verdicts from the release scorer's .score.json; red frames from the release sphere model at the scorer's margins, sampled every 0.002 rad (agrees with the scorer's collision verdict on all exported paths).",
           "Built by experiments/pntfields_cabinet/visualizer/build.py."])

page = (HERE/'template.html').read_text()
assert page.count('/*DATA*/') == 1 and page.count('/*KINEMATICS_JS*/') == 1
out = HERE/'viewer.html'
out.write_text(page.replace('/*KINEMATICS_JS*/', (HERE/'kinematics.js').read_text()).replace('/*DATA*/', json.dumps(data, separators=(',', ':')).replace('</', '<\\/')))
print(f'wrote {out} ({out.stat().st_size/1e6:.2f} MB)')
