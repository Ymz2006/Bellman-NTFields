"""Read-only diagnostics of the cabinet comparison (not used by training or evaluation).

    ./docker-python ... (pinned environment, same mounts as launch.py)
    diagnose.py [--scene cabinet_plus10] --bellman results/registered_<cabinet run> --bellman-eval results/registered_<evaluate run>
                --metric results/registered_<cabinet-metric run> --metric-eval results/registered_<evaluate run> --output <dir>

--scene (default cabinet) picks scenes/<scene>.yml and its frozen suites (tests/, tests_plus10/, ...).
--bellman-eval / --metric-eval default to the pipeline runs' own evaluation/.

1. Coverage: share of training states with the grasp point in the cupboard and its compartments
   (Bellman pool, coverage set and bank sources; metric NTFields x0 and x1).
2. Straight-line reference: the suites' own direct-line certificates (the scorer's verdict on the
   two-point path start -> goal).
3. Collisions: for every path the scorer rejected for collision, the first configuration that breaks
   the scorer's rule after 0.001 rad densification, which cuboid (or self-collision) it hits, where the
   grasp point is, and how far along the path it happens.
4. Implied speed: 1 / |grad_q T(q, goal)| of each trained field along the certified witness paths,
   against the true clearance speed there (Eikonal: the two agree for an exact travel time).
5. Query difficulty: the true clearance speed at the suites' endpoints (compartment endpoints also by
   depth past the opening) and its minimum along each witness path (0.02 rad densification), witness
   and endpoint joint-space lengths, direct-line and RRT-Connect certificates.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,'/clean/method')
from common import RELEASE,sha,write
sys.path.insert(0,'/workspace')
sys.path.insert(0,str(Path(__file__).resolve().parent))
import importlib.util
import numpy as np
import torch
import yaml
import ur5e_plan_eval_multi as E
from curobo.geom.types import WorldConfig
from collision_utils import load_cuboid_geometry,build_self_collision_mask,sphere_cuboid_distance,sphere_sphere_distance
from geometry import scene_boxes,ClearanceGate,ClearanceSpeed
from network import load_checkpoint

SCENE=Path('/clean/experiments/cabinet/scenes/cabinet.yml');TESTS=Path('/clean/experiments/cabinet/tests')
NORM=np.load(RELEASE/'configs/joint_normalization.npz');CENTER,RANGE=NORM['q_center'].astype(np.float32),NORM['q_range'].astype(np.float32)


def densify(q,step=1e-3):
    out=[q[0]]
    for i in range(1,len(q)):
        n=max(1,int(np.ceil(np.abs(q[i]-q[i-1]).max()/step)))
        out.extend(q[i-1]+(q[i]-q[i-1])*(k/n) for k in range(1,n+1))
    return np.asarray(out,np.float32)


class Scene:
    def __init__(self,dev='cuda:0'):
        doc=yaml.safe_load(SCENE.read_text());_,self.robot=E.setup_robot_world('/workspace/ur5e_arm.yml',WorldConfig.from_dict(doc),dev)
        self.names=[c['name'] for c in doc['cuboids']];self.centers,self.half=load_cuboid_geometry(str(SCENE),dev)
        rcfg=yaml.safe_load(Path('/workspace/ur5e_arm.yml').read_text())
        self.mask,_=build_self_collision_mask(E.SPHERE_YML_PATH,rcfg['robot_cfg']['kinematics'].get('self_collision_ignore',{}),E.COLLISION_LINK_NAMES,device=dev)
        self.dev=dev;self.box,self.cupboard,self.compartments=scene_boxes(SCENE)

    @torch.no_grad()
    def positions(self,q_rad):
        return np.concatenate([self.robot.get_state(torch.tensor(q_rad[i:i+65536],device=self.dev)).ee_pose.position.cpu().numpy() for i in range(0,len(q_rad),65536)])

    @torch.no_grad()
    def first_violation(self,q_rad):
        """Index of the first configuration breaking obstacle >= 0.005 / self >= 0.002, its culprit and grasp point."""
        for s in range(0,len(q_rad),20000):
            st=self.robot.get_state(torch.tensor(q_rad[s:s+20000],device=self.dev));sp=st.link_spheres_tensor;c,r=sp[:,:,:3],sp[:,:,3]
            per=torch.stack([sphere_cuboid_distance(c,r,self.centers[k:k+1],self.half[k:k+1]) for k in range(len(self.names))],1)
            ds=sphere_sphere_distance(c,r,self.mask);bad=(per.min(1).values<.005)|(ds<.002)
            if bad.any():
                i=int(torch.nonzero(bad)[0]);k=int(per[i].argmin())
                culprit=self.names[k] if per[i,k]<.005 else 'self'
                return s+i,culprit,st.ee_pose.position[i].cpu().numpy()
        return None,None,None

    def region(self,pos):
        inside=lambda b:bool(((pos>=b[:,0])&(pos<=b[:,1])).all())
        for k,v in self.compartments.items():
            if inside(v):return 'compartment_'+k
        return 'cupboard_other' if inside(self.cupboard) else ('box' if inside(self.box) else 'outside_box')

    def share(self,pos):
        inside=lambda b:((pos>=b[:,0])&(pos<=b[:,1])).all(1)
        return dict(n=int(len(pos)),box=float(inside(self.box).mean()),cupboard=float(inside(self.cupboard).mean()),
                    **{'compartment_'+k:float(inside(v).mean()) for k,v in self.compartments.items()})


def metric_loader():
    spec=importlib.util.spec_from_file_location('cabinet_metric_field','/clean/experiments/cabinet/metric/field.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module.load_metric


def main():
    p=argparse.ArgumentParser()
    for k in ['bellman','metric']:p.add_argument('--'+k,type=Path,required=True);p.add_argument(f'--{k}-eval',type=Path)
    p.add_argument('--scene',default='cabinet');p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    global SCENE,TESTS
    SCENE=Path(f'/clean/experiments/cabinet/scenes/{a.scene}.yml');TESTS=Path('/clean/experiments/cabinet/tests'+a.scene[len('cabinet'):])
    a.bellman_eval=a.bellman_eval or a.bellman;a.metric_eval=a.metric_eval or a.metric
    S=Scene();report=dict(scene=a.scene,scene_sha256=sha(SCENE),tests_manifest_sha256=sha(TESTS/'suite_manifest.json'))
    manifest=json.loads((TESTS/'suite_manifest.json').read_text());suites={x['task']:np.load(TESTS/'suites'/x['file'],allow_pickle=True) for x in manifest['suites']}
    rng=np.random.default_rng(0)
    # 1. Coverage of the cupboard in the training states.
    def endpoints(path,count=400000):
        x=np.load(path,mmap_mode='r');rows=np.sort(rng.choice(len(x),min(count,len(x)),replace=False));x=np.asarray(x[rows])
        return np.concatenate([x[:,:6],x[:,6:]])*RANGE+CENTER
    cov={}
    cov['bellman_original_endpoints']=S.share(S.positions(endpoints(a.bellman/'sampled/inputs/original/sampled_points.npy')))
    cov['bellman_coverage_endpoints']=S.share(S.positions(endpoints(a.bellman/'sampled/inputs/augmentation/sampled_points.npy')))
    bank=np.load(a.bellman/'prepared/bank/source_q.npy',mmap_mode='r');rows=np.sort(rng.choice(len(bank),min(400000,len(bank)),replace=False))
    cov['bellman_bank_sources']=S.share(S.positions(np.asarray(bank[rows])*RANGE+CENTER))
    m=np.load(a.metric/'data/sampled_points.npy',mmap_mode='r');rows=np.sort(rng.choice(len(m),min(400000,len(m)),replace=False));m=np.asarray(m[rows])
    cov['metric_x0']=S.share(S.positions(m[:,:6]*RANGE+CENTER));cov['metric_x1']=S.share(S.positions(m[:,6:]*RANGE+CENTER))
    report['coverage']=cov;print('COVERAGE',json.dumps(cov,indent=1),flush=True)
    # 2. Straight-line reference.
    report['straight_line']={t:int(z['direct_line_strict_feasible'].sum()) for t,z in suites.items()};report['straight_line']['total']=sum(report['straight_line'].values())
    # 3. Collision culprits.
    runs={'bellman':(a.bellman_eval,'bellman'),'metric':(a.metric_eval,'metric')};collisions={}
    labels=dict(rk4=lambda m,t:f'{m}_rk4_{t}',mppi_seed0=lambda m,t:f'{m}_mppi_{t}_seed0',ntrl_mppi_seed0=lambda m,t:f'metric_ntrl_mppi_{t}_seed0')
    for method,(run,prefix) in runs.items():
        for planner,name in labels.items():
            if planner=='ntrl_mppi_seed0' and method!='metric':continue
            rows=[]
            for task,z in suites.items():
                label=name(prefix,task);score=json.loads((run/'evaluation'/(label+'.score.json')).read_text())[label]['per_query']
                paths=np.load(run/'evaluation'/(label+'.npz'),allow_pickle=True)['trajectories']
                for i,r in score.items():
                    if 'collision' not in r['reasons']:continue
                    q=densify(np.asarray(paths[int(i)],np.float32));k,culprit,pos=S.first_violation(q)
                    if k is None:continue
                    rows.append(dict(task=task,query=int(i),culprit=culprit,region=S.region(pos),fraction=float(k/max(1,len(q)-1)),
                                     start_region=str(z['start_region'][int(i)]),goal_region=str(z['goal_region'][int(i)])))
            agg=lambda key:{v:sum(1 for r in rows if r[key]==v) for v in sorted({r[key] for r in rows})}
            culprit_group=lambda c:'self' if c=='self' else ('table' if c=='table' else 'cupboard_panel')
            collisions[f'{method}/{planner}']=dict(n=len(rows),culprit=agg('culprit'),
                culprit_group={g:sum(1 for r in rows if culprit_group(r['culprit'])==g) for g in ['cupboard_panel','table','self']},
                region=agg('region'),median_fraction_of_path=float(np.median([r['fraction'] for r in rows])) if rows else None,rows=rows)
            print('COLLISIONS',method,planner,len(rows),collisions[f'{method}/{planner}']['culprit_group'],collisions[f'{method}/{planner}']['region'],flush=True)
    report['collisions']=collisions
    # 4. Implied speed 1/|grad_q T| along the witness paths vs the true clearance speed.
    gate=ClearanceGate(SCENE);truth=ClearanceSpeed(gate)
    fields=dict(bellman=load_checkpoint(a.bellman/'training/checkpoint.pt','cuda:0'),metric=metric_loader()(a.metric/'training/checkpoint.pt','cuda:0'))
    implied={}
    for task,z in suites.items():
        qs,gs=[],[]
        for i,w in enumerate(z['witness_trajectories']):
            d=densify(np.asarray(w,np.float32),step=.02);qs.append(d);gs.append(np.repeat(z['goals_rad'][i][None],len(d),0))
        q=(np.concatenate(qs)-CENTER)/RANGE;g=(np.concatenate(gs)-CENTER)/RANGE
        qt=torch.tensor(q,device='cuda:0');gt=torch.tensor(g,device='cuda:0')
        true=torch.cat([truth(qt[s:s+8192]) for s in range(0,len(qt),8192)]).cpu().numpy()
        pos=S.positions(q*RANGE+CENTER);inside=((pos>=S.cupboard[:,0])&(pos<=S.cupboard[:,1])).all(1)
        far=np.linalg.norm(q-g,axis=1)>.02
        row=dict(points=int(far.sum()),points_inside=int((far&inside).sum()),true_speed_mean_inside=float(true[far&inside].mean()) if (far&inside).any() else None,
                 true_speed_mean_outside=float(true[far&~inside].mean()) if (far&~inside).any() else None)
        for name,field in fields.items():
            speeds=[]
            for s in range(0,len(qt),8192):
                x=qt[s:s+8192].clone().requires_grad_(True)
                with torch.enable_grad():grad=torch.autograd.grad(field(x,gt[s:s+8192]).sum(),x)[0]
                speeds.append((1/grad.norm(dim=1).clamp_min(1e-8)).cpu().numpy())
            sp=np.concatenate(speeds)
            for where,mask in [('inside',far&inside),('outside',far&~inside)]:
                if mask.any():
                    row[f'{name}_implied_mean_{where}']=float(sp[mask].mean());row[f'{name}_log_ratio_rmse_{where}']=float(np.sqrt(np.mean(np.log(sp[mask]/true[mask])**2)))
        implied[task]=row;print('IMPLIED',task,json.dumps(row),flush=True)
    report['implied_speed']=implied
    # 5. Query difficulty: clearance speed at the endpoints and along the certified witness paths.
    difficulty={}
    for task,z in suites.items():
        ends=torch.tensor((np.concatenate([z['starts_rad'],z['goals_rad']])-CENTER)/RANGE,device='cuda:0');speed=truth(ends).cpu().numpy()
        lowest=[float(truth(torch.tensor((densify(np.asarray(w,np.float32),step=.02)-CENTER)/RANGE,device='cuda:0')).min()) for w in z['witness_trajectories']]
        x=np.concatenate([z['ee_x_start'],z['ee_x_goal']]);comp=np.isin(np.concatenate([z['start_region'],z['goal_region']]),list(S.compartments))
        difficulty[task]=dict(endpoint_speed_mean=float(speed.mean()),endpoint_speed_floor_share=float((speed<=.1+1e-6).mean()),endpoint_speed_free_share=float((speed>=1-1e-6).mean()),
            compartment_endpoints=int(comp.sum()),compartment_endpoint_speed_mean=float(speed[comp].mean()) if comp.any() else None,
            compartment_endpoint_depth_mean=float((x[comp]-S.cupboard[0,0]).mean()) if comp.any() else None,witness_min_speed_mean=float(np.mean(lowest)),
            witness_cspace_length_median=float(np.median(z['witness_cspace_length'])),endpoint_cspace_distance_median=float(np.median(np.linalg.norm(z['goals_rad']-z['starts_rad'],axis=1))),
            direct_line=int(z['direct_line_strict_feasible'].sum()),rrt_witnesses=int((z['witness_source']=='rrtc').sum()))
        print('DIFFICULTY',task,json.dumps(difficulty[task]),flush=True)
    report['difficulty']=difficulty
    write(a.output/'diagnostics.json',report)


if __name__=='__main__':main()
