"""Certified test suites for the cabinet task (generated once, then frozen into tests/).

The obstacle-scene recipe (experiments/obstacle_scenes/make_tests.py, whose RRT-Connect and segment
checks are reused unchanged) without any orientation rule:

- Endpoints: uniform joint samples (seed 20261014) that pass the strict endpoint rule in the cabinet
  scene, obstacle clearance >= 0.005 m, self clearance >= 0.002 m and the normalization joint bounds,
  with the grasp point in the stratum's region of the sampling box.
- Pairs: random (seed 20261015), joint-space distance 0.02-4.0 rad.
- Certificate: a witness path that the unchanged release scorer (score_cross_suite.py, tilt gate off
  with --tilt_deg 180) accepts in the scene at 0.001 rad resolution: the straight joint-space line or
  bidirectional RRT-Connect (seed 20261016), never a learned method. Uncertified candidates are dropped.
- Selection: 100 queries per stratum, a seeded random choice among the certified candidates (seed 20261017).

Six strata of 100 queries. Regions of the grasp point: `free` = sampling box outside the cupboard's
outer box; `compartment` = one of the three compartment interiors (scene `compartments:`, a compartment
drawn uniformly); `box` = anywhere in the sampling box.
"""
import argparse,importlib.util,json,runpy,sys,time
from pathlib import Path
sys.path.insert(0,'/clean/method')
from common import RELEASE,sha,write
sys.path.insert(0,'/workspace')
import numpy as np
import torch
import yaml
import ur5e_plan_eval_multi as E
from curobo.geom.types import WorldConfig
from collision_utils import load_cuboid_geometry,build_self_collision_mask,sphere_cuboid_distance,sphere_sphere_distance
_spec=importlib.util.spec_from_file_location('obstacle_make_tests','/clean/experiments/obstacle_scenes/make_tests.py')
SHARED=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(SHARED)
sys.path.insert(0,str(Path(__file__).resolve().parent))
from geometry import scene_boxes

CHUNK,BATCH=SHARED.CHUNK,SHARED.BATCH
MIN_L2,MAX_L2=SHARED.MIN_L2,SHARED.MAX_L2
SEEDS=SHARED.SEEDS
TILT_OFF='180'
# (stratum, start pool, goal pool, compartment relation)
STRATA=[('free_to_free','free','free',None),('free_to_compartment','free','compartment',None),
        ('compartment_to_free','compartment','free',None),('compartment_to_compartment','compartment','compartment','different'),
        ('same_compartment','compartment','compartment','same'),('anywhere','box','box',None)]
# Zones of the frozen suites, stored only because the scorer parses them; with the tilt gate off they do not matter.
FROZEN_SUITE='repaired_eval_ur5e_mixed_c30_p30_x40_certified100.npz'


class Validator(SHARED.Validator):
    """The release scorer's per-configuration rule without orientation, batched on the GPU.
    prefix() and reach() (segment checks at the scorer's 0.001 rad resolution) are inherited."""

    def __init__(self,scene,dev):
        doc=yaml.safe_load(Path(scene).read_text())
        _,self.robot=E.setup_robot_world('/workspace/ur5e_arm.yml',WorldConfig.from_dict(doc),dev)
        self.centers,self.half=load_cuboid_geometry(str(scene),dev)
        rcfg=yaml.safe_load(Path('/workspace/ur5e_arm.yml').read_text())
        self.mask,_=build_self_collision_mask(E.SPHERE_YML_PATH,rcfg['robot_cfg']['kinematics'].get('self_collision_ignore',{}),E.COLLISION_LINK_NAMES,device=dev)
        norm=np.load(RELEASE/'configs/joint_normalization.npz');self.dev=dev
        self.lo,self.hi=torch.tensor(norm['q_min'],device=dev),torch.tensor(norm['q_max'],device=dev)

    @torch.no_grad()
    def __call__(self,q,positions=False):
        ok=torch.empty(len(q),dtype=torch.bool,device=self.dev);pos=torch.empty(len(q),3,device=self.dev)
        for s in range(0,len(q),CHUNK):
            c=q[s:s+CHUNK];st=self.robot.get_state(c);sp=st.link_spheres_tensor
            good=(sphere_cuboid_distance(sp[:,:,:3],sp[:,:,3],self.centers,self.half)>=.005)&(sphere_sphere_distance(sp[:,:,:3],sp[:,:,3],self.mask)>=.002)
            ok[s:s+CHUNK]=good&((c>=self.lo-1e-5)&(c<=self.hi+1e-5)).all(1);pos[s:s+CHUNK]=st.ee_pose.position
        return (ok,pos) if positions else ok


@torch.no_grad()
def sample_pools(V,box,cupboard,compartments,samples,generator,target_cap):
    """Valid configurations per region, and RRT targets anywhere near the table (as the obstacle scenes)."""
    names=list(compartments);found={k:[] for k in ['free','box']+names};targets=[];near_box=((-.2,1.1),(-.9,.9),(-.04,1.1))
    for _ in range(samples//BATCH):
        q=V.lo+(V.hi-V.lo)*torch.rand(BATCH,6,device=V.dev,generator=generator)
        pos=V.robot.get_state(q).ee_pose.position.clone();keep=SHARED.inside(pos,near_box)
        if not keep.any():continue
        ok,pos=V(q[keep],positions=True);q,pos=q[keep][ok],pos[ok]
        if sum(len(t) for t in targets)<target_cap:targets.append(q.cpu())
        in_box=SHARED.inside(pos,box)
        masks=dict(box=in_box,free=in_box&~SHARED.inside(pos,cupboard),**{k:SHARED.inside(pos,v) for k,v in compartments.items()})
        for k,m in masks.items():
            if m.any():found[k].append(torch.cat([q[m],pos[m]],1).cpu())
    pools={k:(torch.cat(v).numpy() if v else np.zeros((0,9),np.float32)) for k,v in found.items()}
    pools['compartment']=[pools[k] for k in names]
    return pools,torch.cat(targets)[:target_cap]


def candidates(relation,start_pool,goal_pool,count,rng):
    rows=[];parts=[]
    while sum(len(r) for r in rows)<count:
        s,sp=SHARED.draw(start_pool,rng,4096)
        if relation=='same':
            g=np.stack([goal_pool[p][rng.integers(len(goal_pool[p]))] for p in sp]);gp=sp
        else:g,gp=SHARED.draw(goal_pool,rng,4096)
        d=np.linalg.norm(s[:,:6]-g[:,:6],axis=1);keep=(d>=MIN_L2)&(d<=MAX_L2)
        if relation=='different':keep&=sp!=gp
        rows.append(np.concatenate([s,g],1)[keep]);parts.append(np.stack([sp,gp],1)[keep])
    return np.concatenate(rows)[:count],np.concatenate(parts)[:count]


def score(paths,starts,goals,scene,zones_json,work,label):
    """The unchanged release scorer, tilt gate off, on a list of paths (None = no path); per-query rows."""
    ids=[i for i,p in enumerate(paths) if p is not None]
    if not ids:return {}
    suite,traj,out=work/(label+'.suite.npz'),work/(label+'.paths.npz'),work/(label+'.score.json')
    np.savez(suite,starts_rad=starts,goals_rad=goals,corner_zones_json=np.array(zones_json))
    arr=np.empty(len(ids),dtype=object)
    for j,i in enumerate(ids):arr[j]=paths[i]
    np.savez(traj,trajectories=arr,query_ids=np.array(ids))
    argv=sys.argv
    sys.argv=['/workspace/score_cross_suite.py','--suite',str(suite),'--scene',str(scene),'--norm_file',str(RELEASE/'configs/joint_normalization.npz'),
              '--trajectories',str(traj),'--labels',label,'--json_out',str(out),'--tilt_deg',TILT_OFF]
    try:runpy.run_path('/workspace/score_cross_suite.py',run_name='__main__')
    finally:sys.argv=argv
    return {int(k):v for k,v in json.loads(out.read_text())[label]['per_query'].items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    started=time.monotonic();dev='cuda:0';name=a.scene.stem;a.output.mkdir(parents=True,exist_ok=False);generator_sha=sha(Path(__file__))
    work=a.output/'work';work.mkdir();(a.output/'suites').mkdir()
    per,round_size,max_rounds=(3,8,2) if a.smoke else (100,160,8)
    samples,iterations,capacity=(20_000_000,60,100) if a.smoke else (400_000_000,10000,10500)
    zones_json=str(np.load(RELEASE/'tests/suites'/FROZEN_SUITE,allow_pickle=True)['corner_zones_json'])
    box,cupboard,compartments=scene_boxes(a.scene);names=list(compartments)
    V=Validator(a.scene,dev);norm=np.load(RELEASE/'configs/joint_normalization.npz')
    center,scale=np.asarray(norm['q_center'],np.float32),np.asarray(norm['q_range'],np.float32)
    strata=[STRATA[1],STRATA[4]] if a.smoke else STRATA
    pools,targets=sample_pools(V,box,cupboard,compartments,samples,torch.Generator(device=dev).manual_seed(SEEDS['pool']),300_000)
    targets=targets.to(dev);timing=dict(pools=time.monotonic()-started)
    sizes={k:([len(x) for x in v] if isinstance(v,list) else len(v)) for k,v in pools.items()};print('POOLS',sizes,'targets',len(targets),flush=True)
    rng=np.random.default_rng(SEEDS['pair']);generator=torch.Generator(device=dev).manual_seed(SEEDS['rrt'])
    state={s[0]:dict(rows=[],status=[]) for s in strata};report=dict(scene=name,rounds=[])
    for r in range(max_rounds):
        todo=[s for s in strata if sum(x=='certified' for x in state[s[0]]['status'])<per]
        if not todo:break
        rows=[];labels=[];parts=[]
        for stratum,start,goal,relation in todo:
            c,pp=candidates(relation,pools[start],pools[goal],round_size,rng);rows.append(c);parts.append(pp);labels+=[stratum]*len(c)
        rows=np.concatenate(rows);parts=np.concatenate(parts);starts,goals=rows[:,:6].astype(np.float32),rows[:,9:15].astype(np.float32)
        # Straight line first, judged by the scorer itself.
        direct=score([np.stack([s,g]) for s,g in zip(starts,goals)],starts,goals,a.scene,zones_json,work,f'round{r}_direct')
        line=np.array([direct[i]['success'] for i in range(len(rows))])
        todo_ids=np.flatnonzero(~line);begin=time.monotonic()
        found,used=SHARED.rrt_connect(V,torch.tensor(starts[todo_ids],device=dev),torch.tensor(goals[todo_ids],device=dev),targets,generator,iterations,capacity=capacity)
        paths=[None]*len(rows)
        for j,i in enumerate(todo_ids):paths[i]=found[j]
        checked=score(paths,starts,goals,a.scene,zones_json,work,f'round{r}_rrt')
        report['rounds'].append(dict(round=r,candidates=len(rows),direct=int(line.sum()),rrt_queries=len(todo_ids),rrt_found=sum(x is not None for x in found),
                                     rrt_certified=sum(v['success'] for v in checked.values()),rrt_iterations=used,rrt_seconds=time.monotonic()-begin))
        print('ROUND',json.dumps(report['rounds'][-1]),flush=True)
        for i in range(len(rows)):
            if line[i]:status,witness,source,metrics=('certified',np.stack([starts[i],goals[i]]),'direct',direct[i])
            elif i in checked and checked[i]['success']:status,witness,source,metrics=('certified',paths[i],'rrtc',checked[i])
            else:status,witness,source,metrics=('uncertified' if paths[i] is None else 'rejected_by_scorer',None,'',{})
            state[labels[i]]['rows'].append(dict(start=starts[i],goal=goals[i],start_pos=rows[i,6:9],goal_pos=rows[i,15:18],parts=parts[i],
                round=r,witness=witness,source=source,direct=bool(line[i]),ee_len=metrics.get('ee_len'),cs_len=metrics.get('cs_len')))
            state[labels[i]]['status'].append(status)
    select=np.random.default_rng(SEEDS['select']);manifest=dict(scene=name,scene_sha256=sha(a.scene),generator_sha256=generator_sha,
        seeds=SEEDS,smoke=a.smoke,total=0,suites=[]);report['strata']={}
    validator=json.dumps(dict(step=SHARED.STEP,tilt_deg=None,obs_margin=.005,self_margin=.002,scorer_tilt_deg=float(TILT_OFF)))
    for stratum,start,goal,relation in strata:
        rows,status=state[stratum]['rows'],np.array(state[stratum]['status'])
        ok=np.flatnonzero(status=='certified')
        assert len(ok)>=per or (a.smoke and len(ok)),f'{stratum}: only {len(ok)} certified of {len(rows)} candidates'
        keep=select.choice(ok,min(per,len(ok)),replace=False);q=[rows[i] for i in keep];n=len(q)
        starts,goals=np.stack([x['start'] for x in q]).astype(np.float32),np.stack([x['goal'] for x in q]).astype(np.float32)
        sp,gp=np.stack([x['start_pos'] for x in q]),np.stack([x['goal_pos'] for x in q])
        part=lambda pool,k:np.array([names[x['parts'][k]] if pool=='compartment' else pool for x in q])
        witnesses=np.empty(n,dtype=object)
        for i,x in enumerate(q):witnesses[i]=x['witness']
        fields=dict(starts_rad=starts,goals_rad=goals,starts_norm=(starts-center)/scale,goals_norm=(goals-center)/scale,
            ee_x_start=sp[:,0].astype(np.float32),ee_y_start=sp[:,1].astype(np.float32),ee_z_start=sp[:,2].astype(np.float32),
            ee_x_goal=gp[:,0].astype(np.float32),ee_y_goal=gp[:,1].astype(np.float32),ee_z_goal=gp[:,2].astype(np.float32),
            start_region=part(start,0),goal_region=part(goal,1),corner_zones_json=np.array(zones_json),
            collision_margin=np.array(.005),self_collision_margin=np.array(.002),orientation_rule=np.array('none'),
            min_cspace_l2=np.array(MIN_L2),max_cspace_l2=np.array(MAX_L2),seed=np.array(SEEDS['pair']),suite_name=np.array(f'{name}_{stratum}'),
            scene=np.array(f'experiments/cabinet/scenes/{name}.yml'),scene_sha256=np.array(sha(a.scene)),sampling_box=box,
            certified=np.array(True),witness_trajectories=witnesses,witness_source=np.array([x['source'] for x in q]),
            witness_cspace_length=np.array([x['cs_len'] for x in q],np.float32),witness_ee_length=np.array([x['ee_len'] for x in q],np.float32),
            witness_n_waypoints=np.array([len(x['witness']) for x in q],np.int32),candidate_index=keep.astype(np.int32),n_candidates=np.array(len(rows)),
            n_certified=np.array(len(ok)),select_seed=np.array(SEEDS['select']),validator=np.array(validator),query_group_label=np.array([stratum]*n),
            direct_line_strict_feasible=np.array([x['direct'] for x in q]),start_pool=np.array(start),goal_pool=np.array(goal))
        path=a.output/'suites'/f'{name}_{stratum}_{n}.npz';np.savez(path,**fields)
        counts={k:int(v) for k,v in zip(*np.unique(status,return_counts=True))}
        report['strata'][stratum]=dict(start_pool=start,goal_pool=goal,relation=relation,candidates=len(rows),**counts,
            certified_direct=int(sum(rows[i]['direct'] for i in ok)),selected_direct=int(fields['direct_line_strict_feasible'].sum()))
        manifest['suites'].append(dict(task=stratum,file=path.name,sha256=sha(path),n=n,keys=sorted(fields)))
        manifest['total']+=n
        with np.load(path,allow_pickle=True) as z:
            assert np.array_equal((z['starts_rad']-center)/scale,z['starts_norm']) and np.array_equal((z['goals_rad']-center)/scale,z['goals_norm'])
    for stratum in state:
        rows=state[stratum]['rows']
        np.savez(work/f'candidates_{stratum}.npz',starts=np.stack([x['start'] for x in rows]),goals=np.stack([x['goal'] for x in rows]),
                 status=np.array(state[stratum]['status']),direct=np.array([x['direct'] for x in rows]),round=np.array([x['round'] for x in rows]))
    timing['total']=time.monotonic()-started;report.update(pool_sizes=sizes,rrt_targets=len(targets),timing=timing,
        parameters=dict(samples=samples,round_size=round_size,max_rounds=max_rounds,rrt_iterations=iterations,rrt_step=.2,rrt_capacity=capacity,goal_bias=.1,
                        min_cspace_l2=MIN_L2,max_cspace_l2=MAX_L2,per_stratum=per,scorer_tilt_deg=float(TILT_OFF)))
    write(a.output/'suite_manifest.json',manifest);write(a.output/'generation.json',dict(report,complete=True))
    print(json.dumps(report['strata'],indent=1),flush=True)


if __name__=='__main__':main()
