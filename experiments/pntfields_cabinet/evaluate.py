"""Evaluate a P-NTFields train.py checkpoint on a scene's frozen cabinet suites (experiments/cabinet/tests*).

Protocol of experiments/ntrl_cabinet/evaluate.py (the ntrl-demo comparison), unchanged: native RK4 and the release
MPPI (1536 samples, horizon 7, <= 400 steps, step 0.01, beta 12.5, goal weight 4) in release coordinates, planning
seeds 0/1/2, the unchanged release scorer in the scene with its tilt gate off. Planners in the training coordinates:
  pnt_plan   P-NTFields' own planner (test/arm_plan.py, field.plan): beta 0.015, dg 0.03, 300 iterations;
             deterministic, so it runs once per query (out of 600)
  ntrl_mppi  ntrl-demo's MPPI (experiments/ntrl_cabinet/field.ntrl_mppi) on this field's travel time, with the
             settings of the best ntrl-demo run (200 samples, horizon 10, 400 steps), so the two fields are also
             compared under the same planner
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
native,PlannerAdapter=SHARED.native,SHARED.PlannerAdapter
_spec=importlib.util.spec_from_file_location('pntfields_cabinet_field',Path(__file__).resolve().parent/'field.py')
FIELD=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(FIELD)
_spec=importlib.util.spec_from_file_location('ntrl_cabinet_field','/clean/experiments/ntrl_cabinet/field.py')
NTRL=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(NTRL)
NORM=RELEASE/'configs/joint_normalization.npz'
TILT_OFF='180'
LABELS=dict(native_rk4='RK4 (native)',pnt_plan='P-NTFields planner',ntrl_mppi='ntrl-demo MPPI',matched_mppi='Release MPPI')


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--coords',choices=['release','ntrl'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--scene',type=Path,required=True);p.add_argument('--tests',type=Path,required=True)
    p.add_argument('--beta',type=float,default=.015);p.add_argument('--dg',type=float,default=.03);p.add_argument('--iters',type=int,default=300)
    p.add_argument('--ntrl-steps',type=int,default=400);p.add_argument('--ntrl-samples',type=int,default=200);p.add_argument('--ntrl-horizon',type=int,default=10)
    p.add_argument('--no-release',action='store_true');p.add_argument('--no-ntrl',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.tests/'suite_manifest.json').read_text())
    assert manifest['scene']==a.scene.stem and manifest['scene_sha256']==sha(a.scene),'Suites belong to another scene file'
    assert a.smoke or not manifest['smoke'],'Full evaluation needs full (frozen) suites'
    for item in manifest['suites']:assert sha(a.tests/'suites'/item['file'])==item['sha256'],item['file']
    suites=manifest['suites'][:1] if a.smoke else manifest['suites']
    count=min(2 if a.smoke else 100,*[item['n'] for item in suites]);seeds=[0] if a.smoke else [0,1,2]
    pnt=dict(beta=a.beta,dg=a.dg,iters=a.iters);ntrl=dict(steps=a.ntrl_steps,sample_num=a.ntrl_samples,horizon=a.ntrl_horizon)
    load=lambda path,device='cuda:0':FIELD.load_pnt(path,device,a.coords)
    report=dict(complete=False,smoke=a.smoke,method='pntfields',coords=a.coords,pnt_planner_settings=pnt,ntrl_mppi_settings=ntrl,checkpoint=str(a.checkpoint),checkpoint_sha256=sha(a.checkpoint),
                scene=str(a.scene),scene_sha256=sha(a.scene),tests_manifest_sha256=sha(a.tests/'suite_manifest.json'),scorer_tilt_deg=float(TILT_OFF),scores=[])

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

    net=load(a.checkpoint);methods=['native_rk4','pnt_plan']
    for item in suites:
        task=item['task'];suite=a.tests/'suites'/item['file'];data=np.load(suite,allow_pickle=True)
        paths,reached=native(net,data,count)
        label=f'pnt_rk4_{task}';path=a.output/(label+'.npz')
        np.savez(path,trajectories=paths,query_ids=np.arange(count),converged=reached)
        score(path,suite,label,task,None,'native_rk4')
        paths,converged=FIELD.plan(net,data,count,**pnt)
        label=f'pnt_plan_{task}';path=a.output/(label+'.npz')
        np.savez(path,trajectories=paths,query_ids=np.arange(count),converged=converged)
        score(path,suite,label,task,None,'pnt_plan')
    if not a.no_ntrl:
        methods.append('ntrl_mppi')
        for seed in seeds:
            for item in suites:
                task=item['task'];suite=a.tests/'suites'/item['file']
                paths,converged=NTRL.ntrl_mppi(net,np.load(suite,allow_pickle=True),count,seed,**ntrl)
                label=f'pnt_ntrl_mppi_{task}_seed{seed}';path=a.output/(label+'.npz')
                np.savez(path,trajectories=paths,query_ids=np.arange(count),converged=converged)
                score(path,suite,label,task,seed,'ntrl_mppi')
    del net;gc.collect();torch.cuda.empty_cache()
    if not a.no_release:
        methods.append('matched_mppi')
        import ur5e_plan_eval_multi as planner
        import model as release_model
        release_model.load_checkpoint=lambda path,device='cuda:0':PlannerAdapter(load(path,device))
        base=(planner.MPPI_batch_multi,planner.MultiModel.load,planner.compute_routing_info)
        for seed in seeds:
            for item in suites:
                task=item['task'];suite=a.tests/'suites'/item['file']
                label=f'pnt_mppi_{task}_seed{seed}';path=a.output/(label+'.npz')
                planner.MPPI_batch_multi,planner.MultiModel.load,planner.compute_routing_info=base;planner._GRAPH_COST_CACHE.clear()
                invoke(RELEASE/'method/evaluate.py',['--checkpoint',a.checkpoint,'--scene',a.scene,'--norm_file',NORM,'--eval_set',suite,'--n_queries',count,'--batch_size','1',
                    '--sample_num','1536','--horizon','7','--mppi_cost_beta','12.5','--mppi_steps','400','--step_size','.010','--seed',seed,'--goal_cost_weight','4',
                    '--overlap','.10','--y_divide','0','--save_trajectories',path])
                planner.MPPI_batch_multi,planner.MultiModel.load,planner.compute_routing_info=base;planner._GRAPH_COST_CACHE.clear()
                gc.collect();torch.cuda.empty_cache()
                score(path,suite,label,task,seed,'matched_mppi')
    n=count*len(suites);report['totals']={}
    for m in methods:
        rows=[r for r in report['scores'] if r['method']==m];runs=1 if m in ('native_rk4','pnt_plan') else len(seeds)
        assert len(rows)==len(suites)*runs
        report['totals'][m]=dict(success=sum(r['success'] for r in rows),n=n*runs,
            per_seed=None if runs==1 else [sum(r['success'] for r in rows if r['planning_seed']==s) for s in seeds],
            per_suite={item['task']:sum(r['success'] for r in rows if r['task']==item['task']) for item in suites},
            failures={k:sum(r['failures'].get(k,0) for r in rows) for k in sorted({k for r in rows for k in r['failures']})})
    report['complete']=True;write(a.output/'evaluation.json',report)
    lines=[f'# P-NTFields ({a.coords} coordinates) on {a.tests.name}','','SMOKE ONLY.' if a.smoke else
           'Certified queries (each has a witness path the scorer accepts). Release scorer in the scene, tilt gate off.','']
    for m in methods:
        t=report['totals'][m];lines.append(f"{LABELS[m]}: {t['success']}/{t['n']} ({100*t['success']/t['n']:.2f}%)"+('' if t['per_seed'] is None else f", per seed {t['per_seed']}")+'.')
    lines+=['','|Suite|'+'|'.join(LABELS[m] for m in methods)+'|','|---|'+'---:|'*len(methods)]
    for item in suites:
        cells=[f"{report['totals'][m]['per_suite'][item['task']]}/{count*(1 if report['totals'][m]['per_seed'] is None else len(seeds))}" for m in methods]
        lines.append(f"|{item['task']}|"+'|'.join(cells)+'|')
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
