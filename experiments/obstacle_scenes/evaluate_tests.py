"""Evaluate a trained checkpoint on one obstacle scene's frozen test suites.

The same round-5 protocol as experiments/transition_wall/evaluate.py, whose RK4 rollout and planner
adapter are reused unchanged: native RK4, the release MPPI with its original settings and planning
seeds 0/1/2, and the release scorer in the scene. Only the query suites differ: the scene's own
certified suites (tests/<scene>/), checked against their manifest hashes before any planning.
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


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--tests',type=Path,required=True,help='frozen suites: tests/<scene> with suite_manifest.json and suites/')
    p.add_argument('--smoke',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.tests/'suite_manifest.json').read_text())
    assert manifest['scene']==a.scene.stem and manifest['scene_sha256']==sha(a.scene) and not manifest['smoke'],'Suites belong to another scene file'
    for item in manifest['suites']:assert sha(a.tests/'suites'/item['file'])==item['sha256'],item['file']
    suites=manifest['suites'][:1] if a.smoke else manifest['suites']
    count=2 if a.smoke else 100;seeds=[0] if a.smoke else [0,1,2]
    report=dict(complete=False,smoke=a.smoke,checkpoint_sha256=sha(a.checkpoint),scene=str(a.scene),scene_sha256=sha(a.scene),
                tests_manifest_sha256=sha(a.tests/'suite_manifest.json'),experiment='Obstacle-scene checkpoint on the scene test suites',scores=[])

    def invoke(script,args):
        sys.argv=[str(script),*map(str,args)]
        runpy.run_path(str(script),run_name='__main__')

    def score(path,suite,label,task,seed,method):
        result=a.output/(label+'.score.json')
        invoke('/workspace/score_cross_suite.py',['--suite',suite,'--scene',a.scene,'--norm_file',RELEASE/'configs/joint_normalization.npz','--trajectories',path,'--labels',label,'--json_out',result])
        row=json.loads(result.read_text())[label];assert row['n']==count and len(row['per_query'])==count
        report['scores'].append(dict(method=method,task=task,planning_seed=seed,n=count,success=row['success'],label=label))
        write(a.output/'evaluation.json',report);print('BELLMAN600_SCORE',json.dumps(report['scores'][-1]),flush=True)

    net=load_scalar(a.checkpoint,'cuda:0')
    for item in suites:
        task=item['task'];suite=a.tests/'suites'/item['file']
        paths,reached=native(net,np.load(suite,allow_pickle=True),count)
        label='bellman_rk4_'+task;path=a.output/(label+'.npz')
        np.savez(path,trajectories=paths,query_ids=np.arange(count),converged=reached)
        score(path,suite,label,task,None,'native_rk4')
    del net;gc.collect();torch.cuda.empty_cache()

    import ur5e_plan_eval_multi as planner
    import model as release_model
    release_model.load_checkpoint=lambda path,device='cuda:0':PlannerAdapter(load_scalar(path,device))
    base=(planner.MPPI_batch_multi,planner.MultiModel.load,planner.compute_routing_info)
    for seed in seeds:
        for item in suites:
            task=item['task'];suite=a.tests/'suites'/item['file']
            label=f'bellman_mppi_{task}_seed{seed}';path=a.output/(label+'.npz')
            planner.MPPI_batch_multi,planner.MultiModel.load,planner.compute_routing_info=base;planner._GRAPH_COST_CACHE.clear()
            invoke(RELEASE/'method/evaluate.py',['--checkpoint',a.checkpoint,'--scene',a.scene,'--norm_file',RELEASE/'configs/joint_normalization.npz','--eval_set',suite,'--n_queries',count,'--batch_size','1','--sample_num','1536','--horizon','7','--mppi_cost_beta','12.5','--mppi_steps','400','--step_size','.010','--seed',seed,'--goal_cost_weight','4','--overlap','.10','--y_divide','0','--zone_eval','--save_trajectories',path])
            planner.MPPI_batch_multi,planner.MultiModel.load,planner.compute_routing_info=base;planner._GRAPH_COST_CACHE.clear()
            gc.collect();torch.cuda.empty_cache()
            score(path,suite,label,task,seed,'matched_mppi')
    assert len(report['scores'])==len(suites)*(1+len(seeds))
    report['mppi_successes']=[sum(r['success'] for r in report['scores'] if r['method']=='matched_mppi' and r['planning_seed']==s) for s in seeds]
    report['native_success']=sum(r['success'] for r in report['scores'] if r['method']=='native_rk4');report['complete']=True
    write(a.output/'evaluation.json',report)
    total=sum(report['mppi_successes']);n=count*len(suites)
    lines=[f'# Obstacle-scene checkpoint on the {a.scene.stem} test suites','','SMOKE ONLY; not a success-rate experiment.' if a.smoke else
           'Certified scene-specific queries: every query has a witness path the scorer accepts. Fixed fifth training round, original planner settings.','',
           f'Native RK4: {report["native_success"]}/{n}.',f'Matched MPPI successes per seed: {report["mppi_successes"]}; {total}/{n*len(seeds)} ({100*total/(n*len(seeds)):.2f}%).','',
           '|Suite|RK4|'+'|'.join(f'MPPI seed {s}' for s in seeds)+'|','|---|---:|'+'---:|'*len(seeds)]
    for item in suites:
        rows={(r['method'],r['planning_seed']):r['success'] for r in report['scores'] if r['task']==item['task']}
        lines.append(f"|{item['task']}|{rows[('native_rk4',None)]}/{count}|"+'|'.join(f"{rows[('matched_mppi',s)]}/{count}" for s in seeds)+'|')
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
