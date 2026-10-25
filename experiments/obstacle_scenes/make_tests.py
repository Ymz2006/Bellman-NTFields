"""Certified scene-specific test suites for one obstacle scene.

Recomputes the testing data the way the frozen certified `mixed` suite was built: endpoint pools
from uniform joint samples that pass the strict endpoint rule, random pairs 0.02-4.0 rad apart in
joint space, and a witness path for every kept query that the unchanged release scorer accepts in
the scene (0.001 rad densification, obstacle >= 0.005 m, self >= 0.002 m, 20-degree zone rule with
OR in the transition band). Witnesses come from an independent planner, the straight line or
bidirectional RRT-Connect, never from the method under test.

Six strata of 100 queries per scene. Three general strata draw endpoints from the whole cupping and
pouring regions at all heights (the frozen distribution, minus endpoints the obstacles block); three
put the endpoints at the scene's obstacles (`STRATA`, regions from targets.yml). The kept queries
are a seeded random choice among the certified candidates, not a choice by difficulty.

Runs in the pinned environment through run.py obstacle-tests; frozen copies live in tests/<scene>/.
"""
import argparse,json,runpy,sys,time
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
from ur5e_task_axes import task_axis_arrays,tilt_deg_from_quat

STEP,COARSE,CHUNK,BATCH=1e-3,1e-2,32768,50000
MIN_L2,MAX_L2=.02,4.
SEEDS=dict(pool=20261014,pair=20261015,rrt=20261016,select=20261017)
# Task regions of the frozen suites (grasp point); endpoints never lie in the transition band.
REGIONS=dict(cup=((.29,.85),(.10,.55)),pour=((.29,.85),(-.55,-.10)))
GENERAL=[('general_cupping',0,'cup','cup'),('general_pouring',1,'pour','pour'),('general_cross',2,'cup','pour')]
# (stratum, pair type 0 cupping / 1 pouring / 2 cross, start pool, goal pool)
STRATA=dict(
    platforms=GENERAL+[('platform_cupping',0,'cup_platform_top','cup_platform_top'),('platform_pouring',1,'pour_platform_top','pour_platform_top'),
                       ('platform_cross',2,'cup_platform_top','pour_platform_top')],
    shelf=GENERAL+[('shelf_from_cupping',0,'cup_outside_cupboard','shelf'),('shelf_from_pouring',2,'pour','shelf'),('shelf_to_shelf',0,'shelf','shelf')],
    clutter=GENERAL+[('low_cupping',0,'low_cupping','low_cupping'),('low_pouring',1,'low_pouring','low_pouring'),('low_cross',2,'low_cupping','low_pouring')])
# Cross strata run in both directions with equal probability, except that shelf goals stay in the shelf.
ONE_WAY={'shelf_from_cupping','shelf_from_pouring','shelf_to_shelf'}


class Validator:
    """The release scorer's per-configuration rule in one scene, batched on the GPU."""

    def __init__(self,scene,zones,dev):
        doc=yaml.safe_load(scene.read_text())
        _,self.robot=E.setup_robot_world('/workspace/ur5e_arm.yml',WorldConfig.from_dict(doc),dev)
        self.centers,self.half=load_cuboid_geometry(str(scene),dev)
        rcfg=yaml.safe_load(Path('/workspace/ur5e_arm.yml').read_text())
        self.mask,_=build_self_collision_mask(E.SPHERE_YML_PATH,rcfg['robot_cfg']['kinematics'].get('self_collision_ignore',{}),E.COLLISION_LINK_NAMES,device=dev)
        axes=E.load_task_axes(str(scene));self.cup,self.pour=task_axis_arrays(axes,'cup'),task_axis_arrays(axes,'pour')
        norm=np.load(RELEASE/'configs/joint_normalization.npz');self.zones,self.dev=zones,dev
        self.lo,self.hi=torch.tensor(norm['q_min'],device=dev),torch.tensor(norm['q_max'],device=dev)

    @torch.no_grad()
    def __call__(self,q,positions=False):
        ok=torch.empty(len(q),dtype=torch.bool,device=self.dev);pos=torch.empty(len(q),3,device=self.dev)
        for s in range(0,len(q),CHUNK):
            c=q[s:s+CHUNK];st=self.robot.get_state(c);sp=st.link_spheres_tensor
            good=(sphere_cuboid_distance(sp[:,:,:3],sp[:,:,3],self.centers,self.half)>=.005)&(sphere_sphere_distance(sp[:,:,:3],sp[:,:,3],self.mask)>=.002)
            quat,p=st.ee_pose.quaternion,st.ee_pose.position
            cup,pour=tilt_deg_from_quat(quat,*self.cup),tilt_deg_from_quat(quat,*self.pour)
            cm,pm,ov=E.corner_masks_torch(p[:,:2],self.zones,assign_outside=True)
            good&=torch.where(ov,(cup<=20)|(pour<=20),torch.where(cm,cup<=20,pour<=20))
            ok[s:s+CHUNK]=good&((c>=self.lo-1e-5)&(c<=self.hi+1e-5)).all(1);pos[s:s+CHUNK]=p
        return (ok,pos) if positions else ok

    def prefix(self,qa,qb,step):
        """Leading valid points of each segment densified like the scorer (k/n, k=1..n), and n."""
        n=torch.ceil((qb-qa).abs().amax(1)/step).clamp(min=1).long()
        seg=torch.repeat_interleave(torch.arange(len(n),device=self.dev),n)
        k=torch.arange(len(seg),device=self.dev)-(torch.cumsum(n,0)-n)[seg]+1
        ok=self(qa[seg]+(qb-qa)[seg]*(k/n[seg]).unsqueeze(1))
        return (n+1).scatter_reduce(0,seg,torch.where(ok,n[seg]+1,k),reduce='amin')-1,n

    def reach(self,qa,qb):
        """Valid fraction of each segment from qa: a coarse pass, then the scorer's resolution."""
        kc,nc=self.prefix(qa,qb,COARSE);fc=kc/nc;f=torch.zeros_like(fc);sel=kc>0
        if sel.any():
            kf,nf=self.prefix(qa[sel],qa[sel]+(qb[sel]-qa[sel])*fc[sel,None],STEP);f[sel]=fc[sel]*kf/nf
        return f


def nearest(nodes,count,target):
    d=(nodes-target[:,None]).square().sum(2)
    d.masked_fill_(torch.arange(nodes.shape[1],device=nodes.device)[None]>=count[:,None],float('inf'))
    i=d.argmin(1);return i,nodes[torch.arange(len(i),device=nodes.device),i]


@torch.no_grad()
def rrt_connect(V,starts,goals,targets,generator,iterations,eps=.2,capacity=4000,goal_bias=.1):
    """Bidirectional RRT-Connect for a batch of queries. Every tree edge passes Validator.reach."""
    dev=starts.device;B=len(starts)
    tree=torch.zeros(2,B,capacity,6,device=dev);parent=torch.full((2,B,capacity),-1,dtype=torch.long,device=dev)
    count=torch.ones(2,B,dtype=torch.long,device=dev);tree[0,:,0]=starts;tree[1,:,0]=goals
    link=torch.full((B,2),-1,dtype=torch.long,device=dev);done=torch.zeros(B,dtype=torch.bool,device=dev)
    for it in range(iterations):
        a,b=it%2,1-it%2
        ids=torch.nonzero(~done&(count.amax(0)<capacity-1)).squeeze(1)
        if not len(ids):break
        target=targets[torch.randint(len(targets),(len(ids),),device=dev,generator=generator)]
        bias=torch.rand(len(ids),device=dev,generator=generator)<goal_bias
        target=torch.where(bias[:,None],tree[b,ids,0],target)
        near,qn=nearest(tree[a,ids],count[a,ids],target)
        step=target-qn;qs=qn+step*(eps/step.norm(dim=1,keepdim=True).clamp_min(1e-9)).clamp(max=1)
        qnew=qn+(qs-qn)*V.reach(qn,qs)[:,None]
        grew=(qnew-qn).abs().amax(1)>=5*STEP
        if not grew.any():continue
        ids,near,qnew=ids[grew],near[grew],qnew[grew]
        slot=count[a,ids];tree[a,ids,slot]=qnew;parent[a,ids,slot]=near;count[a,ids]+=1
        # Connect: the other tree grows straight towards the new node as far as it is valid.
        other,qo=nearest(tree[b,ids],count[b,ids],qnew)
        f=V.reach(qo,qnew);full=f>=1
        link[ids[full],a]=slot[full];link[ids[full],b]=other[full];done[ids[full]]=True
        q2=qo+(qnew-qo)*f[:,None];part=~full&((q2-qo).abs().amax(1)>=5*STEP)
        if part.any():
            p=ids[part];s2=count[b,p];tree[b,p,s2]=q2[part];parent[b,p,s2]=other[part];count[b,p]+=1
    T,P,L=tree.cpu().numpy(),parent.cpu().numpy(),link.cpu().numpy();paths=[None]*B
    for i in np.flatnonzero(done.cpu().numpy()):
        branch=[]
        for t in (0,1):
            node,chain=L[i,t],[]
            while node>=0:chain.append(T[t,i,node]);node=P[t,i,node]
            branch.append(chain[::-1] if t==0 else chain)
        paths[i]=np.array(branch[0]+branch[1],dtype=np.float32)
    return paths,int(it)+1


def boxes_for(name,scene_doc,regions,table_top):
    """Grasp-point boxes (x, y, z ranges) of every endpoint pool the scene's strata use."""
    z=(table_top,table_top+.70)
    boxes={k:dict(box=(x,y,z)) for k,(x,y) in REGIONS.items()}
    for t in regions.get(name,[]):boxes[t['name']]=dict(box=(tuple(t['x']),tuple(t['y']),tuple(t['z'])))
    if name=='shelf':
        panels=[c for c in scene_doc['cuboids'] if c['name'].startswith('shelf_')]
        lo=np.min([np.subtract(c['pose'][:3],np.divide(c['dims'],2)) for c in panels],0);hi=np.max([np.add(c['pose'][:3],np.divide(c['dims'],2)) for c in panels],0)
        boxes['cup_outside_cupboard']=dict(box=boxes['cup']['box'],exclude=((lo[0],hi[0]),(lo[1],hi[1]),(-1.,2.)))
        boxes['shelf']=dict(parts=[t['name'] for t in regions['shelf']])
    return boxes


def inside(pos,box):
    return ((pos[:,0]>=box[0][0])&(pos[:,0]<=box[0][1])&(pos[:,1]>=box[1][0])&(pos[:,1]<=box[1][1])&(pos[:,2]>=box[2][0])&(pos[:,2]<=box[2][1]))


@torch.no_grad()
def sample_pools(V,boxes,samples,generator,target_cap):
    """Valid configurations per grasp-point box, and RRT targets anywhere near the table."""
    found={k:[] for k,v in boxes.items() if 'box' in v};targets=[];near_box=((-.2,1.1),(-.9,.9),(-.04,1.1))
    for _ in range(samples//BATCH):
        q=V.lo+(V.hi-V.lo)*torch.rand(BATCH,6,device=V.dev,generator=generator)
        pos=V.robot.get_state(q).ee_pose.position.clone();keep=inside(pos,near_box)
        if not keep.any():continue
        ok,pos=V(q[keep],positions=True);q,pos=q[keep][ok],pos[ok]
        if sum(len(t) for t in targets)<target_cap:targets.append(q.cpu())
        for k in found:
            m=inside(pos,boxes[k]['box'])
            if 'exclude' in boxes[k]:m&=~inside(pos,boxes[k]['exclude'])
            if m.any():found[k].append(torch.cat([q[m],pos[m]],1).cpu())
    pools={k:(torch.cat(v).numpy() if v else np.zeros((0,9),np.float32)) for k,v in found.items()}
    for k,v in boxes.items():
        if 'parts' in v:pools[k]=[pools[p] for p in v['parts']]
    return pools,torch.cat(targets)[:target_cap]


def draw(pool,rng,size):
    """Uniform members of a pool; a pool of parts picks a part uniformly first. Returns rows and part ids."""
    if isinstance(pool,list):
        part=rng.integers(len(pool),size=size);rows=np.stack([pool[p][rng.integers(len(pool[p]))] for p in part])
        return rows,part
    return pool[rng.integers(len(pool),size=size)],np.zeros(size,int)


def candidates(stratum,start_pool,goal_pool,count,rng):
    rows=[]
    while sum(len(r) for r in rows)<count:
        s,sp=draw(start_pool,rng,4096);g,gp=draw(goal_pool,rng,4096)
        d=np.linalg.norm(s[:,:6]-g[:,:6],axis=1);keep=(d>=MIN_L2)&(d<=MAX_L2)
        if stratum=='shelf_to_shelf':keep&=sp!=gp
        if stratum not in ONE_WAY and stratum.endswith('cross'):
            swap=rng.random(4096)<.5;s,g=np.where(swap[:,None],g,s),np.where(swap[:,None],s,g)
        rows.append(np.concatenate([s,g],1)[keep])
    return np.concatenate(rows)[:count]


def score(paths,starts,goals,pair_type,scene,zones_json,work,label):
    """The unchanged release scorer on a list of paths (None = no path); per-query rows."""
    ids=[i for i,p in enumerate(paths) if p is not None]
    if not ids:return {}
    suite,traj,out=work/(label+'.suite.npz'),work/(label+'.paths.npz'),work/(label+'.score.json')
    np.savez(suite,starts_rad=starts,goals_rad=goals,pair_type=pair_type,corner_zones_json=np.array(zones_json))
    arr=np.empty(len(ids),dtype=object)
    for j,i in enumerate(ids):arr[j]=paths[i]
    np.savez(traj,trajectories=arr,query_ids=np.array(ids))
    argv=sys.argv
    sys.argv=['/workspace/score_cross_suite.py','--suite',str(suite),'--scene',str(scene),'--norm_file',str(RELEASE/'configs/joint_normalization.npz'),
              '--trajectories',str(traj),'--labels',label,'--json_out',str(out)]
    try:runpy.run_path('/workspace/score_cross_suite.py',run_name='__main__')
    finally:sys.argv=argv
    return {int(k):v for k,v in json.loads(out.read_text())[label]['per_query'].items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--targets',type=Path,default=Path(__file__).resolve().parent/'targets.yml');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    started=time.monotonic();dev='cuda:0';name=a.scene.stem;a.output.mkdir(parents=True,exist_ok=False);generator_sha=sha(Path(__file__))
    work=a.output/'work';work.mkdir();(a.output/'suites').mkdir()
    per,round_size,max_rounds=(3,8,2) if a.smoke else (100,160,6)
    samples,iterations,capacity=(20_000_000,60,100) if a.smoke else (400_000_000,10000,10500)
    # Zones and task axes exactly as recorded in the frozen suites, which the audit ties to every scene.
    frozen=np.load(RELEASE/'tests/suites/repaired_eval_ur5e_mixed_c30_p30_x40_certified100.npz',allow_pickle=True)
    zones_json,axes_json=str(frozen['corner_zones_json']),str(frozen['task_axes_json'])
    scene_doc=yaml.safe_load(a.scene.read_text());regions=yaml.safe_load(a.targets.read_text())
    table_top=[c for c in scene_doc['cuboids'] if c['name']=='table'][0];table_top=table_top['pose'][2]+table_top['dims'][2]/2
    V=Validator(a.scene,json.loads(zones_json),dev);norm=np.load(RELEASE/'configs/joint_normalization.npz')
    center,scale=np.asarray(norm['q_center'],np.float32),np.asarray(norm['q_range'],np.float32)
    strata=[STRATA[name][0],STRATA[name][3]] if a.smoke else STRATA[name]
    boxes=boxes_for(name,scene_doc,regions,table_top)
    pools,targets=sample_pools(V,boxes,samples,torch.Generator(device=dev).manual_seed(SEEDS['pool']),300_000)
    targets=targets.to(dev);timing=dict(pools=time.monotonic()-started)
    sizes={k:([len(x) for x in v] if isinstance(v,list) else len(v)) for k,v in pools.items()};print('POOLS',sizes,'targets',len(targets),flush=True)
    rng=np.random.default_rng(SEEDS['pair']);generator=torch.Generator(device=dev).manual_seed(SEEDS['rrt'])
    state={s[0]:dict(rows=[],status=[]) for s in strata};report=dict(scene=name,rounds=[])
    for r in range(max_rounds):
        todo=[s for s in strata if sum(x=='certified' for x in state[s[0]]['status'])<per]
        if not todo:break
        rows=[];labels=[];kinds=[]
        for stratum,kind,start,goal in todo:
            c=candidates(stratum,pools[start],pools[goal],round_size,rng);rows.append(c);labels+=[stratum]*len(c);kinds+=[kind]*len(c)
        rows=np.concatenate(rows);starts,goals=rows[:,:6].astype(np.float32),rows[:,9:15].astype(np.float32);kinds=np.array(kinds,np.int32)
        # Straight line first, judged by the scorer itself.
        direct=score([np.stack([s,g]) for s,g in zip(starts,goals)],starts,goals,kinds,a.scene,zones_json,work,f'round{r}_direct')
        line=np.array([direct[i]['success'] for i in range(len(rows))])
        todo_ids=np.flatnonzero(~line);begin=time.monotonic()
        found,used=rrt_connect(V,torch.tensor(starts[todo_ids],device=dev),torch.tensor(goals[todo_ids],device=dev),targets,generator,iterations,capacity=capacity)
        paths=[None]*len(rows)
        for j,i in enumerate(todo_ids):paths[i]=found[j]
        checked=score(paths,starts,goals,kinds,a.scene,zones_json,work,f'round{r}_rrt')
        report['rounds'].append(dict(round=r,candidates=len(rows),direct=int(line.sum()),rrt_queries=len(todo_ids),rrt_found=sum(x is not None for x in found),
                                     rrt_certified=sum(v['success'] for v in checked.values()),rrt_iterations=used,rrt_seconds=time.monotonic()-begin))
        print('ROUND',json.dumps(report['rounds'][-1]),flush=True)
        for i in range(len(rows)):
            if line[i]:status,witness,source,metrics=('certified',np.stack([starts[i],goals[i]]),'direct',direct[i])
            elif i in checked and checked[i]['success']:status,witness,source,metrics=('certified',paths[i],'rrtc',checked[i])
            else:status,witness,source,metrics=('uncertified' if paths[i] is None else 'rejected_by_scorer',None,'',{})
            state[labels[i]]['rows'].append(dict(start=starts[i],goal=goals[i],kind=int(kinds[i]),start_pos=rows[i,6:9],goal_pos=rows[i,15:18],
                round=r,witness=witness,source=source,direct=bool(line[i]),ee_len=metrics.get('ee_len'),cs_len=metrics.get('cs_len')))
            state[labels[i]]['status'].append(status)
    select=np.random.default_rng(SEEDS['select']);manifest=dict(scene=name,scene_sha256=sha(a.scene),generator_sha256=generator_sha,
        targets_sha256=sha(a.targets),seeds=SEEDS,smoke=a.smoke,total=0,suites=[]);report['strata']={}
    validator=json.dumps(dict(step=STEP,tilt_deg=20.,obs_margin=.005,self_margin=.002,overlap_logic='both'))
    for stratum,kind,start,goal in strata:
        rows,status=state[stratum]['rows'],np.array(state[stratum]['status'])
        ok=np.flatnonzero(status=='certified')
        assert len(ok)>=per or (a.smoke and len(ok)),f'{stratum}: only {len(ok)} certified of {len(rows)} candidates'
        keep=select.choice(ok,min(per,len(ok)),replace=False);q=[rows[i] for i in keep];n=len(q)
        starts,goals=np.stack([x['start'] for x in q]).astype(np.float32),np.stack([x['goal'] for x in q]).astype(np.float32)
        sp,gp=np.stack([x['start_pos'] for x in q]),np.stack([x['goal_pos'] for x in q])
        witnesses=np.empty(n,dtype=object)
        for i,x in enumerate(q):witnesses[i]=x['witness']
        zone=lambda y:np.where(y>0,0,1).astype(np.int64)
        st=torch.tensor(np.concatenate([starts,goals]),device=dev)
        with torch.no_grad():
            state_=V.robot.get_state(st);quat=state_.ee_pose.quaternion
            cup,pour=tilt_deg_from_quat(quat,*V.cup).cpu().numpy(),tilt_deg_from_quat(quat,*V.pour).cpu().numpy()
        fields=dict(starts_rad=starts,goals_rad=goals,starts_norm=(starts-center)/scale,goals_norm=(goals-center)/scale,
            ee_x_start=sp[:,0].astype(np.float32),ee_y_start=sp[:,1].astype(np.float32),ee_z_start=sp[:,2].astype(np.float32),
            ee_x_goal=gp[:,0].astype(np.float32),ee_y_goal=gp[:,1].astype(np.float32),ee_z_goal=gp[:,2].astype(np.float32),
            start_cup_tilt_deg=cup[:n].astype(np.float32),start_pour_tilt_deg=pour[:n].astype(np.float32),
            goal_cup_tilt_deg=cup[n:].astype(np.float32),goal_pour_tilt_deg=pour[n:].astype(np.float32),
            start_zone=zone(sp[:,1]),goal_zone=zone(gp[:,1]),pair_type=np.full(n,kind,np.int32),
            corner_zones_json=np.array(zones_json),zone_mode=np.array('corner_table'),cup_tilt_deg=np.array(20.),pour_tilt_deg=np.array(20.),
            overlap=np.array(.1),y_divide=np.array(0.),cup_side=np.array(1),overlap_logic=np.array('both'),collision_margin=np.array(.005),
            self_collision_margin=np.array(.002),min_cspace_l2=np.array(MIN_L2),max_cspace_l2=np.array(MAX_L2),seed=np.array(SEEDS['pair']),
            suite_name=np.array(f'{name}_{stratum}'),task_axes_json=np.array(axes_json),scene=np.array(f'experiments/obstacle_scenes/scenes/{name}.yml'),
            scene_sha256=np.array(sha(a.scene)),certified=np.array(True),witness_trajectories=witnesses,
            witness_source=np.array([x['source'] for x in q]),witness_cspace_length=np.array([x['cs_len'] for x in q],np.float32),
            witness_ee_length=np.array([x['ee_len'] for x in q],np.float32),witness_n_waypoints=np.array([len(x['witness']) for x in q],np.int32),
            candidate_index=keep.astype(np.int32),n_candidates=np.array(len(rows)),n_certified=np.array(len(ok)),select_seed=np.array(SEEDS['select']),
            validator=np.array(validator),query_group_label=np.array([stratum]*n),direct_line_strict_feasible=np.array([x['direct'] for x in q]),
            start_pool=np.array(start),goal_pool=np.array(goal))
        path=a.output/'suites'/f'{name}_{stratum}_{n}.npz';np.savez(path,**fields)
        counts={k:int(v) for k,v in zip(*np.unique(status,return_counts=True))}
        report['strata'][stratum]=dict(pair_type=kind,start_pool=start,goal_pool=goal,candidates=len(rows),**counts,
            certified_direct=int(sum(rows[i]['direct'] for i in ok)),selected_direct=int(fields['direct_line_strict_feasible'].sum()))
        manifest['suites'].append(dict(task=stratum,file=path.name,sha256=sha(path),n=n,pair_types={str(kind):n},keys=sorted(fields)))
        manifest['total']+=n
        with np.load(path,allow_pickle=True) as z:
            assert np.array_equal((z['starts_rad']-center)/scale,z['starts_norm']) and np.array_equal((z['goals_rad']-center)/scale,z['goals_norm'])
    # Every candidate, kept or not, for audits.
    for stratum in state:
        rows=state[stratum]['rows']
        np.savez(work/f'candidates_{stratum}.npz',starts=np.stack([x['start'] for x in rows]),goals=np.stack([x['goal'] for x in rows]),
                 status=np.array(state[stratum]['status']),direct=np.array([x['direct'] for x in rows]),round=np.array([x['round'] for x in rows]))
    timing['total']=time.monotonic()-started;report.update(pool_sizes=sizes,rrt_targets=len(targets),timing=timing,
        parameters=dict(samples=samples,round_size=round_size,max_rounds=max_rounds,rrt_iterations=iterations,rrt_step=.2,rrt_capacity=capacity,goal_bias=.1,
                        min_cspace_l2=MIN_L2,max_cspace_l2=MAX_L2,per_stratum=per))
    write(a.output/'suite_manifest.json',manifest);write(a.output/'generation.json',dict(report,complete=True))
    print(json.dumps(report['strata'],indent=1),flush=True)


if __name__=='__main__':main()
