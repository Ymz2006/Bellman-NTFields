"""Read-only failure breakdown of an evaluate.py output folder (one planner's trajectories on the frozen suites).

For every scored path: the scorer's verdict and reasons (score.json), whether the planner converged (npz
'converged'), and for collision failures the first contact along the path at the scorer's resolution
(densified to 0.001 rad per joint, obstacle clearance < 0.005 m or self clearance < 0.002 m): which cuboid or
self contact, which robot link, where the gripper is, whether the contact lies on the final segment the planner
appends to the goal (a jump when it did not converge), and how deep the path goes (minimum clearance).
"""
import argparse,collections,glob,json,sys
from pathlib import Path
sys.path.insert(0,'/workspace')
import numpy as np
import torch
import yaml
import ur5e_plan_eval_multi as E
from curobo.geom.types import WorldConfig
from collision_utils import build_self_collision_mask,sphere_sphere_distance
sys.path.insert(0,'/clean/experiments/cabinet')
from geometry import scene_boxes

STEP=.001


def densify(q):
    out=[q[:1]];seg=[np.zeros(1,int)]
    for i in range(1,len(q)):
        n=max(1,int(np.ceil(np.abs(q[i]-q[i-1]).max()/STEP)))
        t=np.arange(1,n+1)[:,None]/n;out.append(q[i-1]+(q[i]-q[i-1])*t);seg.append(np.full(n,i))
    return np.concatenate(out).astype(np.float32),np.concatenate(seg)


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--tests',type=Path,required=True);p.add_argument('--prefix',default='metric_ntrl_mppi_');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    dev='cuda:0';doc=yaml.safe_load(a.scene.read_text())
    _,rm=E.setup_robot_world('/workspace/ur5e_arm.yml',WorldConfig.from_dict(doc),dev)
    names=[c['name'] for c in doc['cuboids']]
    cen=torch.tensor([c['pose'][:3] for c in doc['cuboids']],device=dev);half=torch.tensor([c['dims'] for c in doc['cuboids']],device=dev)/2
    rcfg=yaml.safe_load(Path('/workspace/ur5e_arm.yml').read_text())
    mask,link_idx=build_self_collision_mask(E.SPHERE_YML_PATH,rcfg['robot_cfg']['kinematics'].get('self_collision_ignore',{}),E.COLLISION_LINK_NAMES,device=dev)
    _,cupboard,compartments=scene_boxes(a.scene);cup=torch.tensor(cupboard,device=dev)
    manifest=json.loads((a.tests/'suite_manifest.json').read_text());rows=[]
    for item in manifest['suites']:
        task=item['task'];suite=np.load(a.tests/'suites'/item['file'],allow_pickle=True)
        for path in sorted(a.evaluation.glob(f'{a.prefix}{task}_seed*.npz')):
            seed=int(path.stem.rsplit('seed',1)[1]);z=np.load(path,allow_pickle=True)
            label=path.stem;score=json.loads((a.evaluation/(label+'.score.json')).read_text())[label]['per_query']
            for i,traj in enumerate(z['trajectories']):
                s=score[str(i)];row=dict(task=task,seed=seed,query=i,success=s['success'],reasons=s['reasons'],converged=bool(z['converged'][i]),
                    waypoints=int(len(traj)),straight_line_ok=bool(suite['direct_line_strict_feasible'][i]))
                q=np.asarray(traj,np.float32);row['final_jump_rad']=float(np.abs(q[-1]-q[-2]).max()) if len(q)>1 else 0.
                if 'collision' in s['reasons']:
                    d,seg=densify(q);first=None;minimum=dict(obstacle=1e9,self=1e9)
                    for c0 in range(0,len(d),20000):
                        st=rm.get_state(torch.as_tensor(d[c0:c0+20000],device=dev));sph=st.link_spheres_tensor;c,r=sph[:,:,:3],sph[:,:,3]
                        delta=(c[:,:,None,:]-cen[None,None]).abs()-half[None,None]                       # (B,S,K,3)
                        box=torch.linalg.vector_norm(delta.clamp_min(0),dim=-1)+delta.max(-1).values.clamp_max(0)-r[:,:,None]
                        obs=box.flatten(1).min(1).values;selfd=sphere_sphere_distance(c,r,mask)
                        minimum['obstacle']=min(minimum['obstacle'],float(obs.min()));minimum['self']=min(minimum['self'],float(selfd.min()))
                        bad=(obs<.005)|(selfd<.002)
                        if first is None and bool(bad.any()):
                            k=int(torch.nonzero(bad)[0]);flat=int(box[k].flatten().argmin());sphere,cub=divmod(flat,len(names))
                            if float(obs[k])<.005:kind,what,link='obstacle',names[cub],E.COLLISION_LINK_NAMES[int(link_idx[sphere])]
                            else:kind,what,link='self','self',''
                            ee=st.ee_pose.position[k];incup=bool(((ee>=cup[:,0])&(ee<=cup[:,1])).all())
                            first=dict(kind=kind,obstacle=what,link=link,segment=int(seg[c0+k]),fraction=float((c0+k)/max(1,len(d)-1)),
                                       on_final_segment=bool(seg[c0+k]==len(q)-1),ee_in_cupboard=incup,ee=[float(x) for x in ee])
                    row.update(contact=first,min_clearance=minimum)
                rows.append(row)
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'failures.json').write_text(json.dumps(rows))
    # Summary
    out={}
    for task in [x['task'] for x in manifest['suites']]+['ALL']:
        R=[r for r in rows if task in ('ALL',r['task'])];F=[r for r in R if not r['success']];C=[r for r in F if r.get('contact')]
        cnt=lambda key,L:dict(collections.Counter(key(r) for r in L).most_common())
        out[task]=dict(n=len(R),success=sum(r['success'] for r in R),failed=len(F),reasons=cnt(lambda r:'+'.join(r['reasons']),F),
            failed_converged=sum(r['converged'] for r in F),failed_not_converged=sum(not r['converged'] for r in F),
            contact_on_final_jump=sum(r['contact']['on_final_segment'] for r in C),contact_during_motion=sum(not r['contact']['on_final_segment'] for r in C),
            contact_obstacle=cnt(lambda r:r['contact']['obstacle'],C),contact_link=cnt(lambda r:r['contact']['link'] or 'self',C),
            contact_ee_in_cupboard=sum(r['contact']['ee_in_cupboard'] for r in C),
            contact_fraction_quartiles=[float(x) for x in np.quantile([r['contact']['fraction'] for r in C],[.25,.5,.75])] if C else None,
            min_obstacle_clearance_median=float(np.median([r['min_clearance']['obstacle'] for r in C])) if C else None,
            straight_line_ok=sum(r['straight_line_ok'] for r in R),success_where_line_fails=sum(r['success'] and not r['straight_line_ok'] for r in R),
            fail_where_line_ok=sum((not r['success']) and r['straight_line_ok'] for r in R))
    (a.output/'summary.json').write_text(json.dumps(out,indent=1));print(json.dumps(out,indent=1))


if __name__=='__main__':main()
