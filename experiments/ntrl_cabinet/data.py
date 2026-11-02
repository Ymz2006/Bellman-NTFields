"""ntrl-demo arm training pairs for a cabinet scene, with the preprocessing settings exposed.

Defaults reproduce experiments/cabinet/metric/data.py (the registered metric NTFields baseline data):
release coordinates, release clearance label (obstacle and 0.5 x self), margin 0.05, offset 0.005,
x0 window 0 < d - 0.01 < margin, x1 = x0 + unit(rand - 0.5) * U(0, 0.5), 500,000 pairs, seed 20261019.
The ntrl-demo recipe itself (dataprocessing/speed_sampling_arm_normal.py) is unchanged in structure:
x0 near an obstacle, x1 a random step from x0, speed clip(d, offset, margin) / margin, unit clearance
gradient as normal.

Settings (preprocessing only; the method, network and losses are untouched):
  --coords release  x = q / q_range (release normalization, [-0.5, 0.5] over the +-2 pi joint limits)
  --coords ntrl     x = q / (2 pi), ntrl-demo's own scale (dataprocessing: scale = pi / 0.5); the joint
                    limits are +-1 (elbow +-0.5) in these units, so the same uniform x0 draw covers them
  --label release   d = min(obstacle distance, 0.5 self distance) (the scorer also checks self contact)
  --label env       d = obstacle distance only, as in ntrl-demo (its point cloud has no self term)
  --window lo hi    x0 kept when lo < d - 0.01 < hi (default 0, margin); hi <= 0 keeps every x0 with d > 0.01
  --joint-limit L   sample configurations with |q_j| <= min(L, joint limit) (ntrl-demo's IK uses +-pi)
  --step            maximum x1 step length in --coords units (ntrl-demo 0.5); --step 0 instead draws x1 as an
                    independent uniform configuration with the grasp point in the sampling box (paper, Appendix C.1:
                    start/goal pairs formed from randomly sampled configurations). --window -1 0 keeps every x0 in the box.
  --focus F         a fraction F of the pairs has x0 with the grasp point inside the cupboard's outer box (default 0:
                    the uniform draw, where ~1% of the pairs are in the cupboard). Those x0 are the uniform draws in
                    the cupboard plus Gaussian perturbations (--focus-sigma rad per joint) of earlier ones, with the
                    same window filter; x1 is drawn from x0 as for every pair.
Writes sampled_points.npy (N, 12), speed.npy (N, 2), normal.npy (N, 12), clearance.npy, complete.json.
"""
import argparse,json,math,sys,time
from pathlib import Path
sys.path.insert(0,'/clean/method')
from common import RELEASE,sha,write
sys.path.insert(0,'/clean/data_generation/source')
import pkl_to_training_data_multi_ur5e as geom
sys.path.insert(0,'/clean/experiments/cabinet')
from geometry import scene_boxes,SELF_SCALE
import numpy as np
import torch
import yaml

EPS=1e-4


def units(coords):
    """Radians per coordinate unit for each joint."""
    n=np.load(RELEASE/'configs/joint_normalization.npz')
    assert np.all(n['q_center']==0)
    return (n['q_range'] if coords=='release' else np.full(6,2*math.pi)).astype(np.float32),n['q_max'].astype(np.float32)


class Clearance:
    """d(q) at configurations q (radians): label code's spheres, cuboids and self mask."""
    def __init__(self,scene,label,device):
        _,_,self.robot,_=geom.init_curobo('/workspace/ur5e_arm.yml',geom.URDF_PATH,device)
        self.centers,self.half=geom.cuboid_half_extents(geom.load_scene_cuboids(str(scene)),device)
        cfg=yaml.safe_load(Path('/workspace/ur5e_arm.yml').read_text())['robot_cfg']['kinematics']
        self.mask,_=geom.build_self_collision_mask(geom.SPHERE_YML_PATH,cfg.get('self_collision_ignore',{}),geom.COLLISION_LINK_NAMES,device=device)
        self.label=label

    @torch.no_grad()
    def __call__(self,q,positions=False):
        # Chunks of 50,000 bound the GPU memory of the pairwise self distances; the values are unchanged.
        parts=[self.chunk(q[i:i+50000]) for i in range(0,len(q),50000)] or [self.chunk(q)]
        d=torch.cat([x[0] for x in parts]);pos=torch.cat([x[1] for x in parts])
        return (d,pos) if positions else d

    def chunk(self,q):
        st=self.robot.get_state(q);sp=st.link_spheres_tensor;pos=st.ee_pose.position.clone()
        d=geom.sphere_cuboid_distance(sp[:,:,:3],sp[:,:,3],self.centers,self.half,ignore_sphere_indices=geom.OBSTACLE_IGNORE_SPHERE_INDICES)
        if self.label=='release':d=torch.minimum(d,SELF_SCALE*geom.sphere_sphere_distance(sp[:,:,:3],sp[:,:,3],self.mask))
        return d,pos


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--coords',choices=['release','ntrl'],default='release');p.add_argument('--label',choices=['release','env'],default='release')
    p.add_argument('--margin',type=float,default=.05);p.add_argument('--offset',type=float,default=.005);p.add_argument('--pairs',type=int,default=500000)
    p.add_argument('--window',type=float,nargs=2,default=None);p.add_argument('--step',type=float,default=.5);p.add_argument('--seed',type=int,default=20261019)
    p.add_argument('--joint-limit',type=float,help='rad; sample every joint in +-min(limit, release limit) (ntrl-demo: pi)');p.add_argument('--focus',type=float,default=0.);p.add_argument('--focus-sigma',type=float,default=.15)
    p.add_argument('--smoke',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    device='cuda:0';box,cupboard,_=scene_boxes(a.scene);box_t=torch.tensor(box,device=device)
    numsamples=4096 if a.smoke else a.pairs;margin,offset=a.margin,a.offset;window=a.window or [0.,margin];batch=200000
    unit,qmax=units(a.coords)
    if a.joint_limit:qmax=np.minimum(qmax,a.joint_limit).astype(np.float32)
    unit_t=torch.tensor(unit,device=device);bound=torch.tensor(qmax/unit,device=device)
    torch.manual_seed(a.seed);C=Clearance(a.scene,a.label,device)
    steps=torch.eye(6,device=device)*EPS

    def gradient(x):
        plus=torch.stack([C((x+s)*unit_t) for s in steps],1);minus=torch.stack([C((x-s)*unit_t) for s in steps],1)
        return (plus-minus)/(2*EPS)

    cup_t=torch.tensor(cupboard,device=device);seeds=torch.empty(0,6,device=device);sigma=a.focus_sigma/unit_t
    target=dict(base=numsamples-int(round(a.focus*numsamples)),focus=int(round(a.focus*numsamples)))
    xs,ys,ns,src=[],[],[],[];drawn=0;stats=dict(x0_in_box=0,x0_window=0,x1_inside_domain=0,nonzero_normals=0,focus_perturbed=0)
    have=lambda k:sum(len(x) for x,t in zip(xs,src) if t==k)
    while have('base')<target['base'] or have('focus')<target['focus']:
        x0=(torch.rand(batch,6,device=device)-.5)*2*bound;drawn+=batch      # uniform over the joint limits
        kind=torch.zeros(batch,dtype=torch.bool,device=device)
        if target['focus'] and len(seeds):                                   # perturbations of earlier cupboard x0
            pert=seeds[torch.randint(len(seeds),(batch,),device=device)]+torch.randn(batch,6,device=device)*sigma
            pert=pert[((pert<=bound)&(pert>=-bound)).all(1)];x0=torch.cat([x0,pert]);kind=torch.cat([kind,torch.ones(len(pert),dtype=torch.bool,device=device)])
        d0,pos=C(x0*unit_t,positions=True);keep=((pos>=box_t[:,0])&(pos<=box_t[:,1])).all(1);stats['x0_in_box']+=int(keep.sum())
        y0=d0-.01;keep&=(y0>window[0])&((y0<window[1]) if window[1]>0 else True);stats['x0_window']+=int(keep.sum())
        incup=((pos>=cup_t[:,0])&(pos<=cup_t[:,1])).all(1)
        if target['focus']:
            seeds=torch.cat([seeds,x0[keep&incup]])[-200000:];stats['focus_perturbed']+=int((keep&incup&kind).sum())
            focus=keep&incup;base=keep&~kind
            if have('focus')>=target['focus']:focus&=False
            if have('base')>=target['base']:base&=False
            keep=focus|base;kinds=focus[keep]
        x0,d0=x0[keep],d0[keep]
        if not target['focus']:kinds=torch.zeros(len(x0),dtype=torch.bool,device=device)
        if a.step>0:
            dP=torch.rand(len(x0),6,device=device)-.5;rL=torch.rand(len(x0),1,device=device)*a.step
            x1=x0+torch.nn.functional.normalize(dP,dim=1)*rL
            inside=((x1<=bound)&(x1>=-bound)).all(1)
        else:                                                                 # paper C.1: pairs of random configurations
            x1=(torch.rand(len(x0),6,device=device)-.5)*2*bound
            p1=C(x1*unit_t,positions=True)[1];inside=((p1>=box_t[:,0])&(p1<=box_t[:,1])).all(1)
        x0,x1,d0,kinds=x0[inside],x1[inside],d0[inside],kinds[inside];stats['x1_inside_domain']+=int(inside.sum())
        if len(x0)<=1:continue
        d1=C(x1*unit_t);g0=gradient(x0);g1=gradient(x1)
        n0=g0/torch.linalg.vector_norm(g0,dim=1,keepdim=True);n1=g1/torch.linalg.vector_norm(g1,dim=1,keepdim=True)
        good=~(n0.isnan().any(1)|n1.isnan().any(1)|n0.isinf().any(1)|n1.isinf().any(1));stats['nonzero_normals']+=int(good.sum())
        for k,m in (('base',~kinds),('focus',kinds)):
            m=m&good
            if m.any():xs.append(torch.cat([x0,x1],1)[m].cpu());ys.append(torch.stack([d0,d1],1)[m].cpu());ns.append(torch.cat([n0,n1],1)[m].cpu());src.append(k)
        print({'drawn':drawn,'collected':{k:have(k) for k in target},'seeds':len(seeds)},flush=True)
    take=lambda L,k:torch.cat([x for x,t in zip(L,src) if t==k])[:target[k]] if target[k] else torch.empty(0)
    if target['focus']:
        perm=torch.randperm(numsamples,generator=torch.Generator().manual_seed(a.seed))
        X=torch.cat([take(xs,'base'),take(xs,'focus')])[perm].numpy().astype(np.float32);D=torch.cat([take(ys,'base'),take(ys,'focus')])[perm].numpy();N=torch.cat([take(ns,'base'),take(ns,'focus')])[perm].numpy().astype(np.float32)
    else:
        X=torch.cat(xs)[:numsamples].numpy().astype(np.float32);D=torch.cat(ys)[:numsamples].numpy();N=torch.cat(ns)[:numsamples].numpy().astype(np.float32)
    speed=(np.clip(D,offset,margin)/margin).astype(np.float32)
    np.save(a.output/'sampled_points.npy',X);np.save(a.output/'speed.npy',speed);np.save(a.output/'normal.npy',N);np.save(a.output/'clearance.npy',D.astype(np.float32))
    with torch.no_grad():
        pos=torch.cat([C(torch.tensor(X[i:i+65536,:6],device=device)*unit_t,positions=True)[1].cpu() for i in range(0,len(X),65536)]).numpy()
    report=dict(complete=True,smoke=a.smoke,scene=str(a.scene),scene_sha256=sha(a.scene),seed=a.seed,pairs=len(X),uniform_draws=drawn,counts=stats,
        focus=a.focus,focus_sigma_rad=a.focus_sigma,coords=a.coords,radians_per_unit=unit.tolist(),domain_bound=(qmax/unit).tolist(),label=a.label,margin=margin,offset=offset,
        x0_filter=f'{window[0]} < d - 0.01 < '+(str(window[1]) if window[1]>0 else 'inf'),x1_step=f'unit(rand-0.5) * U(0, {a.step})' if a.step>0 else 'independent uniform configuration in the sampling box',normal='central difference, eps 1e-4 (coords units)',
        speed_quantiles={str(q):float(np.quantile(speed,q)) for q in (.05,.25,.5,.75,.95)},x0_cupboard_share=float(((pos>=cupboard[:,0])&(pos<=cupboard[:,1])).all(1).mean()),
        x1_speed_floor_share=float((speed[:,1]<=offset/margin+1e-6).mean()),elapsed_seconds=time.monotonic()-started,evaluation_rows_read=False,
        files={n:sha(a.output/(n+'.npy')) for n in ['sampled_points','speed','normal','clearance']})
    write(a.output/'complete.json',report);print(json.dumps(report,indent=1),flush=True)

if __name__=='__main__':main()
