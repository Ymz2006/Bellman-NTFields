"""Frozen600 comparison: native RK4 and unchanged release MPPI/scorer."""
import sys
sys.path.insert(0, "/clean/method")
import argparse
import gc
import json
import runpy
import sys
import time
from pathlib import Path
from common import HERE, ROOT, RELEASE, sha, write
import numpy as np
import torch
from torch import nn
from network import load_checkpoint
SCENE=ROOT/'experiments/wall25/scene.yml'
ENDPOINT_AUDIT=ROOT/'experiments/wall25/endpoint_audit.json'
# The frozen planner imports its own train package.
sys.path.insert(0, "/workspace")


def load_scalar(path, device):
    return load_checkpoint(path,device)


class PlannerAdapter(nn.Module):
    def __init__(self, scalar):
        super().__init__()
        self.scalar = scalar
        self.B = torch.empty((0, 6), device=next(scalar.parameters()).device)
        self.y_divide, self.overlap, self.route_temp = 0., .1, .01
        self.weight_mode, self.zone_mode, self.corner_zones = 'window', 'y', None

    def out(self, coords, **kwargs):
        # Match the planner tuple interface. Routing arguments are ignored:
        # the Bellman scalar is a function ONLY of normalized endpoints.
        value = self.scalar(coords[:, :6], coords[:, 6:])
        return value, coords.new_zeros((len(coords), 2)), coords, value, value


def native(net, suite, count):
    q = torch.tensor(suite['starts_norm'][:count], device='cuda:0')
    goal = torch.tensor(suite['goals_norm'][:count], device='cuda:0')
    paths = [[x.copy()] for x in q.cpu().numpy()]
    reached = np.zeros(count, bool)

    def direction(x, g):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            grad = torch.autograd.grad(net(x, g).sum(), x)[0]
        return -grad / torch.sqrt(grad.square().sum(-1, keepdim=True) + 1e-12)

    for step in range(400):
        dist = (goal-q).norm(dim=1)
        reached |= (dist <= .005).cpu().numpy()
        ids = np.flatnonzero(~reached)
        if not len(ids):
            break
        qt, gt = q[ids], goal[ids]
        h = dist[ids].clamp_max(.01)[:, None]
        k1 = direction(qt,gt)
        k2 = direction((qt+h*.5*k1).clamp(-.5,.5),gt)
        k3 = direction((qt+h*.5*k2).clamp(-.5,.5),gt)
        k4 = direction((qt+h*k3).clamp(-.5,.5),gt)
        q[ids] = (qt+h/6*(k1+2*k2+2*k3+k4)).clamp(-.5,.5).detach()
        for i, x in zip(ids, q[ids].cpu().numpy()):
            paths[i].append(x.copy())
    reached |= ((goal-q).norm(dim=1) <= .005).cpu().numpy()
    norm = np.load(RELEASE/'configs/joint_normalization.npz')
    result = np.empty(count, dtype=object)
    for i, path in enumerate(paths):
        if reached[i]:
            path.append(suite['goals_norm'][i].copy())
        rad = np.array(path)*norm['q_range']+norm['q_center']
        rad[0] = suite['starts_rad'][i]
        if reached[i]:
            rad[-1] = suite['goals_rad'][i]
        result[i] = rad.astype(np.float32)
    return result, reached


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    suites=json.loads((RELEASE/'tests/suite_manifest.json').read_text())['suites']
    assert len(suites)==6 and len({x['task'] for x in suites})==6
    if a.smoke:
        suites=suites[:1]
    count=2 if a.smoke else 100
    audit=json.loads(ENDPOINT_AUDIT.read_text())
    assert audit['complete'] and audit['scene_sha256']==sha(SCENE)
    report=dict(complete=False,smoke=a.smoke,checkpoint_sha256=sha(a.checkpoint),scene_sha256=sha(SCENE),
                experiment='Frozen no-wall checkpoint direct transfer to wall25',endpoint_audit=audit,scores=[])

    def invoke(script,args):
        sys.argv=[str(script),*map(str,args)]
        runpy.run_path(str(script),run_name='__main__')

    def score(path,suite,label,seed,method):
        result=a.output/(label+'.score.json')
        invoke('/workspace/score_cross_suite.py',['--suite',suite,'--scene',SCENE,'--norm_file',RELEASE/'configs/joint_normalization.npz','--trajectories',path,'--labels',label,'--json_out',result])
        doc=json.loads(result.read_text()); row=doc[label]
        assert row['n']==count and len(row['per_query'])==count
        paired = {}
        if seed is not None:
            ref_name=f'bellman_mppi_{task}_seed{seed}'
            ref=json.loads((ROOT/'assets/reference'/(ref_name+'.score.json')).read_text())[ref_name]
            ours=np.array([row['per_query'][str(i)]['success'] for i in range(count)],dtype=bool)
            theirs=np.array([ref['per_query'][str(i)]['success'] for i in range(count)],dtype=bool)
            paired=dict(reference_success=int(theirs.sum()),rescued=int((ours & ~theirs).sum()),regressed=int((~ours & theirs).sum()))
        eligible=[r['query_id'] for r in audit['tasks'][task]['per_query'][:count] if r['wall_endpoints_valid']]
        eligible_success=sum(bool(row['per_query'][str(i)]['success']) for i in eligible)
        assert eligible_success==row['success'], 'Successful path cannot have invalid endpoints'
        report['scores'].append(dict(method=method,task=task,planning_seed=seed,n=count,success=row['success'],
            endpoint_valid_n=len(eligible),endpoint_valid_success=eligible_success,groups=doc.get(label+'_groups',{}),paired_with_reference=paired))
        write(a.output/'evaluation.json',report)
        print('BELLMAN600_SCORE',json.dumps(report['scores'][-1]),flush=True)

    net=load_scalar(a.checkpoint,'cuda:0')
    for item in suites:
        task=item['task']; suite=RELEASE/'tests/suites'/item['file']
        paths,reached=native(net,np.load(suite,allow_pickle=True),count)
        label='bellman_rk4_'+task; path=a.output/(label+'.npz')
        np.savez(path,trajectories=paths,query_ids=np.arange(count),converged=reached)
        score(path,suite,label,None,'native_rk4')
    del net; gc.collect(); torch.cuda.empty_cache()

    import ur5e_plan_eval_multi as planner
    import model as release_model
    release_model.load_checkpoint=lambda path,device='cuda:0': PlannerAdapter(load_scalar(path,device))
    base_mppi=planner.MPPI_batch_multi
    base_load=planner.MultiModel.load
    base_routing=planner.compute_routing_info
    for seed in ([0] if a.smoke else [0,1,2]):
        for item in suites:
            task=item['task']; suite=RELEASE/'tests/suites'/item['file']
            label=f'bellman_mppi_{task}_seed{seed}'; path=a.output/(label+'.npz')
            planner.MPPI_batch_multi=base_mppi
            planner.MultiModel.load=base_load
            planner.compute_routing_info=base_routing
            planner._GRAPH_COST_CACHE.clear()
            invoke(RELEASE/'method/evaluate.py',['--checkpoint',a.checkpoint,'--scene',SCENE,'--norm_file',RELEASE/'configs/joint_normalization.npz','--eval_set',suite,'--n_queries',count,'--batch_size','1','--sample_num','1536','--horizon','7','--mppi_cost_beta','12.5','--mppi_steps','400','--step_size','.010','--seed',seed,'--goal_cost_weight','4','--overlap','.10','--y_divide','0','--zone_eval','--save_trajectories',path])
            planner.MPPI_batch_multi=base_mppi
            planner.MultiModel.load=base_load
            planner.compute_routing_info=base_routing
            planner._GRAPH_COST_CACHE.clear()
            gc.collect(); torch.cuda.empty_cache()
            score(path,suite,label,seed,'matched_mppi')
    report['complete']=True
    report['baseline_successes']=[532,533,532]
    report['mppi_successes']=[sum(r['success'] for r in report['scores'] if r['method']=='matched_mppi' and r['planning_seed']==s) for s in ([0] if a.smoke else [0,1,2])]
    report['native_success']=sum(r['success'] for r in report['scores'] if r['method']=='native_rk4')
    assert len(report['scores'])==(2 if a.smoke else 24)
    report['endpoint_valid_mppi_n']=sum(r['endpoint_valid_n'] for r in report['scores'] if r['method']=='matched_mppi')
    report['endpoint_valid_native_n']=sum(r['endpoint_valid_n'] for r in report['scores'] if r['method']=='native_rk4')
    write(a.output/'evaluation.json',report)
    total=sum(report['mppi_successes']); denominator=count*len(suites)*(1 if a.smoke else 3)
    lines=['# Wall25 direct transfer of frozen no-wall Bellman600', '', 'SMOKE ONLY; not a success-rate experiment.' if a.smoke else 'Unchanged no-wall checkpoint and planner settings; only add the25cm wall. No retraining. Three planning seeds.', '', f'Native RK4: {report["native_success"]}/{count*len(suites)}.', f'Matched MPPI successes: {report["mppi_successes"]}; {total}/{denominator} ({100*total/denominator:.2f}%).', f'Endpoint-valid subset: MPPI {total}/{report["endpoint_valid_mppi_n"]}; RK4 {report["native_success"]}/{report["endpoint_valid_native_n"]}. Endpoints alone do not prove route feasibility.', 'No-wall full600 reference: 532/533/532,1597/1800 (88.72%). This full-set reference is not the endpoint-valid-subset reference.', '', '|Task|Seed|Success|Bellman reference|Rescued|Regressed|', '|---|---:|---:|---:|---:|---:|']
    for row in report['scores']:
        if row['method']=='matched_mppi':
            ref=row['paired_with_reference']
            lines.append(f'|{row["task"]}|{row["planning_seed"]}|{row["success"]}|{ref["reference_success"]}|{ref["rescued"]}|{ref["regressed"]}|')
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
