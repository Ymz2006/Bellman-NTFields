"""Evaluate a travel-time field T(q, goal) on the frozen cabinet test suites (tests/).

Same protocol as experiments/obstacle_scenes/evaluate_tests.py, whose native RK4 rollout and planner
adapter (experiments/transition_wall/evaluate.py) are reused unchanged: native RK4 and the release MPPI
with its original settings (1536 samples, horizon 7, at most 400 steps, step 0.01, beta 12.5, goal cost
weight 4) and planning seeds 0/1/2. Differences: the release scorer runs in the cabinet scene with its
tilt gate off (--tilt_deg 180), because the cabinet task has no orientation rule, and the planner gets
no --zone_eval (its own report only; the scorer decides success).

--method bellman  TravelCost checkpoint (method/network.py).
--method metric   metric NTFields checkpoint (ntrl-demo, metric/field.py); additionally runs that
                  method's own MPPI (ntrl-demo tests/arm_plan_stat.py settings) with seeds 0/1/2.
"""
import sys
sys.path.insert(0,'/clean/method')
import argparse,gc,importlib.util,json,runpy
from pathlib import Path
from common import RELEASE,sha,write
import numpy as np
import torch
sys.path.insert(0,'/workspace')
_spec=importlib.util.spec_from_file_location('transition_wall_evaluate','/clean/experiments/transition_wall/evaluate.py')
SHARED=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(SHARED)
native,PlannerAdapter,load_scalar=SHARED.native,SHARED.PlannerAdapter,SHARED.load_scalar
HERE=Path(__file__).resolve().parent
NORM=RELEASE/'configs/joint_normalization.npz'
TILT_OFF='180'
LABELS=dict(native_rk4='RK4 (native)',matched_mppi='Release MPPI',ntrl_mppi='ntrl-demo MPPI')


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['bellman','metric']);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--run',type=Path,help='a cabinet pipeline run (results/registered_<id>): its method and final training checkpoint')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--scene',type=Path,required=True);p.add_argument('--tests',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.run:
        status=json.loads((a.run/'status.json').read_text());progress=json.loads((a.run/'training/progress.json').read_text())
        assert status['smoke']==a.smoke and progress['complete'],f'{a.run.name}: training is not complete'
        a.method,a.checkpoint=status['method'],a.run/'training/checkpoint.pt'
    assert a.method and a.checkpoint,'give --run, or --method and --checkpoint'
    a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.tests/'suite_manifest.json').read_text())
    assert manifest['scene']==a.scene.stem and manifest['scene_sha256']==sha(a.scene),'Suites belong to another scene file'
    assert a.smoke or not manifest['smoke'],'Full evaluation needs full (frozen) suites'
    for item in manifest['suites']:assert sha(a.tests/'suites'/item['file'])==item['sha256'],item['file']
    suites=manifest['suites'][:1] if a.smoke else manifest['suites']
    count=min(2 if a.smoke else 100,*[item['n'] for item in suites]);seeds=[0] if a.smoke else [0,1,2]
    if a.method=='metric':
        spec=importlib.util.spec_from_file_location('cabinet_metric_field',HERE/'metric/field.py')
        FIELD=importlib.util.module_from_spec(spec);spec.loader.exec_module(FIELD);load=FIELD.load_metric
    else:load=load_scalar
    report=dict(complete=False,smoke=a.smoke,method=a.method,checkpoint=str(a.checkpoint),checkpoint_sha256=sha(a.checkpoint),scene=str(a.scene),scene_sha256=sha(a.scene),
                tests_manifest_sha256=sha(a.tests/'suite_manifest.json'),scorer_tilt_deg=float(TILT_OFF),
                experiment='Cabinet task: travel-time field on the frozen cabinet suites',scores=[])

    def invoke(script,args):
        sys.argv=[str(script),*map(str,args)]
        runpy.run_path(str(script),run_name='__main__')

    def score(path,suite,label,task,seed,method):
        result=a.output/(label+'.score.json')
        invoke('/workspace/score_cross_suite.py',['--suite',suite,'--scene',a.scene,'--norm_file',NORM,'--trajectories',path,'--labels',label,'--json_out',result,'--tilt_deg',TILT_OFF])
        row=json.loads(result.read_text())[label];assert row['n']==count and len(row['per_query'])==count
        failures={}
        for r in row['per_query'].values():
            for reason in r['reasons']:failures[reason]=failures.get(reason,0)+1
        report['scores'].append(dict(method=method,task=task,planning_seed=seed,n=count,success=row['success'],label=label,failures=failures,
                                     success_ids=[int(i) for i in row['solved_ids']]))
        write(a.output/'evaluation.json',report);print('CABINET_SCORE',json.dumps({k:v for k,v in report['scores'][-1].items() if k!='success_ids'}),flush=True)

    net=load(a.checkpoint,'cuda:0')
    for item in suites:
        task=item['task'];suite=a.tests/'suites'/item['file']
        paths,reached=native(net,np.load(suite,allow_pickle=True),count)
        label=f'{a.method}_rk4_{task}';path=a.output/(label+'.npz')
        np.savez(path,trajectories=paths,query_ids=np.arange(count),converged=reached)
        score(path,suite,label,task,None,'native_rk4')
    methods=['native_rk4','matched_mppi']
    if a.method=='metric':
        methods.append('ntrl_mppi')
        for seed in seeds:
            for item in suites:
                task=item['task'];suite=a.tests/'suites'/item['file']
                paths,converged=FIELD.ntrl_mppi(net,np.load(suite,allow_pickle=True),count,seed)
                label=f'metric_ntrl_mppi_{task}_seed{seed}';path=a.output/(label+'.npz')
                np.savez(path,trajectories=paths,query_ids=np.arange(count),converged=converged)
                score(path,suite,label,task,seed,'ntrl_mppi')
    del net;gc.collect();torch.cuda.empty_cache()

    import ur5e_plan_eval_multi as planner
    import model as release_model
    release_model.load_checkpoint=lambda path,device='cuda:0':PlannerAdapter(load(path,device))
    base=(planner.MPPI_batch_multi,planner.MultiModel.load,planner.compute_routing_info)
    for seed in seeds:
        for item in suites:
            task=item['task'];suite=a.tests/'suites'/item['file']
            label=f'{a.method}_mppi_{task}_seed{seed}';path=a.output/(label+'.npz')
            planner.MPPI_batch_multi,planner.MultiModel.load,planner.compute_routing_info=base;planner._GRAPH_COST_CACHE.clear()
            invoke(RELEASE/'method/evaluate.py',['--checkpoint',a.checkpoint,'--scene',a.scene,'--norm_file',NORM,'--eval_set',suite,'--n_queries',count,'--batch_size','1',
                '--sample_num','1536','--horizon','7','--mppi_cost_beta','12.5','--mppi_steps','400','--step_size','.010','--seed',seed,'--goal_cost_weight','4',
                '--overlap','.10','--y_divide','0','--save_trajectories',path])
            planner.MPPI_batch_multi,planner.MultiModel.load,planner.compute_routing_info=base;planner._GRAPH_COST_CACHE.clear()
            gc.collect();torch.cuda.empty_cache()
            score(path,suite,label,task,seed,'matched_mppi')
    seeded=[m for m in methods if m!='native_rk4']
    assert len(report['scores'])==len(suites)*(1+len(seeds)*len(seeded))
    n=count*len(suites);report['totals']={}
    for m in methods:
        rows=[r for r in report['scores'] if r['method']==m];runs=1 if m=='native_rk4' else len(seeds)
        report['totals'][m]=dict(success=sum(r['success'] for r in rows),n=n*runs,
            per_seed=None if m=='native_rk4' else [sum(r['success'] for r in rows if r['planning_seed']==s) for s in seeds])
    report['mppi_successes']=report['totals']['matched_mppi']['per_seed'];report['native_success']=report['totals']['native_rk4']['success'];report['complete']=True
    write(a.output/'evaluation.json',report)
    lines=[f'# Cabinet task: {a.method} on the frozen cabinet suites','','SMOKE ONLY; not a success-rate experiment.' if a.smoke else
           'Certified queries (each has a witness path the scorer accepts). Release scorer in the cabinet scene, tilt gate off.','']
    for m in methods:
        t=report['totals'][m];lines.append(f"{LABELS[m]}: {t['success']}/{t['n']} ({100*t['success']/t['n']:.2f}%)"+('' if t['per_seed'] is None else f", per seed {t['per_seed']}")+'.')
    lines+=['','|Suite|'+'|'.join(LABELS[m] for m in methods)+'|','|---|'+'---:|'*len(methods)]
    for item in suites:
        cells=[]
        for m in methods:
            rows=[r for r in report['scores'] if r['method']==m and r['task']==item['task']]
            cells.append(f"{sum(r['success'] for r in rows)}/{count*len(rows)}")
        lines.append(f"|{item['task']}|"+'|'.join(cells)+'|')
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
