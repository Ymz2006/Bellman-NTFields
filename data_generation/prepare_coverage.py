"""Configuration-only coverage augmentation; no suite trajectories or test labels."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
import yaml
ROOT=Path('/clean')
sys.path.insert(0,str(ROOT/'data_generation/source'))
import pkl_to_training_data_multi_ur5e as geom
from ur5e_task_axes import load_task_axes,tilt_deg_from_quat
sys.path.insert(0,str(Path(__file__).parent))
from labels import tilt_speed,task_speeds

SOURCE=None
ANGLE_EDGES=np.array([0,10,18,22,35,181],dtype=np.float32)

def classify(xyz,angles):
 region=np.where(xyz[:,1]>.1,0,np.where(xyz[:,1]<-.1,1,2))
 active=np.where(region==0,angles[:,0],np.where(region==1,angles[:,1],angles.min(axis=1)))
 return region*5+np.clip(np.searchsorted(ANGLE_EDGES,active,side='right')-1,0,4)

def main():
 global SOURCE
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--per-bin',type=int,default=2000);p.add_argument('--validation-per-bin',type=int,default=400);p.add_argument('--pairs',type=int,default=100000);p.add_argument('--generation-budget',type=float,default=240);p.add_argument('--scene',type=Path,default=ROOT/'runtime/configs/scene.yml');a=p.parse_args();SOURCE=a.source
 a.output.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(20260912);device='cuda:0'
 _,_,robot,_=geom.init_curobo('/workspace/ur5e_arm.yml',geom.URDF_PATH,device)
 axes=load_task_axes(a.scene)
 norm=np.load(SOURCE/'joint_normalization.npz');center=norm['q_center'];scale=norm['q_range']
 old=np.load(SOURCE/'sampled_points.npy',mmap_mode='r')
 buckets=[[] for _ in range(15)];counts=np.zeros(15,dtype=int);quota=a.per_bin+a.validation_per_bin
 start=time.monotonic();draws=0
 while counts.min()<quota:
  if time.monotonic()-start>a.generation_budget:raise RuntimeError(f'Coverage generation timeout; counts={counts.tolist()}')
  n=8192;idx=rng.integers(len(old),size=n);side=rng.integers(2,size=n)
  q=np.stack([old[idx[i],6*side[i]:6*side[i]+6] for i in range(n)])*scale+center
  sigma=rng.choice([.02,.08,.20,.50],size=(n,1));q=q+rng.normal(size=q.shape)*sigma
  uniform=rng.random(n)<.1;q[uniform]=rng.uniform(norm['q_min'],norm['q_max'],size=(uniform.sum(),6))
  # Reject joint-limit violations, rather than clamping mass onto the boundary.
  q=q[((q>norm['q_min'])&(q<norm['q_max'])).all(axis=1)].astype(np.float32)
  with torch.no_grad():
   st=robot.get_state(torch.tensor(q,device=device));xyz=st.ee_pose.position.cpu().numpy()
   angles=np.stack([tilt_deg_from_quat(st.ee_pose.quaternion,axes[t]['local_axis'],axes[t]['world_axis']).cpu().numpy() for t in ['cup','pour']],axis=1)
  bins=classify(xyz,angles)
  # Sampling focus only: these are not new planner workspace constraints.
  focus=(xyz[:,0]>.15)&(xyz[:,0]<1.)&(abs(xyz[:,1])<.7)&(xyz[:,2]>-.1)&(xyz[:,2]<.8)
  for b in range(15):
   take=np.flatnonzero((bins==b)&focus)[:max(0,quota-counts[b])]
   if len(take):buckets[b].append(np.concatenate([q[take],xyz[take],angles[take]],axis=1));counts[b]+=len(take)
  draws+=n
  if draws%(8192*20)==0:print({'draws':draws,'counts':counts.tolist()},flush=True)
 full=np.concatenate([np.concatenate(b) for b in buckets]);_,unique=np.unique(full[:,:6],axis=0,return_index=True)
 if len(unique)!=len(full):raise RuntimeError('Duplicate augmented configurations; regenerate with unique quota filling')
 q=full[:,:6];xyz=full[:,6:9];angles=full[:,9:11]
 centers,half=geom.cuboid_half_extents(geom.load_scene_cuboids(a.scene),device)
 cfg=yaml.safe_load(Path('/workspace/ur5e_arm.yml').read_text())['robot_cfg']['kinematics']
 mask,_=geom.build_self_collision_mask(geom.SPHERE_YML_PATH,cfg.get('self_collision_ignore',{}),geom.COLLISION_LINK_NAMES,device=device)
 # Establish the same obstacle/self label recipe as the source before extending it.
 probe=np.asarray(old[:256,:6])*scale+center
 po,ps,_=geom.compute_physical_distance(robot,probe,centers,half,self_coll_mask=mask,device=device)
 errors={'obstacle':float(abs(po-np.load(SOURCE/'d_obstacle.npy',mmap_mode='r')[:256,0]).max()),'self':float(abs(ps-np.load(SOURCE/'d_self.npy',mmap_mode='r')[:256,0]).max())}
 if max(errors.values())>2e-5:raise RuntimeError(f'Source geometry label mismatch: {errors}')
 do,ds,_=geom.compute_physical_distance(robot,q,centers,half,self_coll_mask=mask,device=device)
 components=np.stack([np.clip(do,.005,.05)/.05,np.clip(.5*ds,.005,.05)/.05,tilt_speed(angles[:,0],20),tilt_speed(angles[:,1],20)],axis=1).astype(np.float32)
 speeds,_,w,_,weighted=task_speeds(components,xyz[:,1]);bins=classify(xyz,angles)
 report={'source':str(SOURCE),'seed':20260912,'candidate_draws':draws,'angle_edges':ANGLE_EDGES.tolist(),'region_order':['cup_y_gt_0.1','pour_y_lt_minus0.1','overlap'],'angle_in_overlap':'minimum of both actual tilts','no_test_suite_used':True,'label_geometry_probe_max_errors':errors,'obstacle_ignored_spheres':geom.OBSTACLE_IGNORE_SPHERE_INDICES,'sampling_focus_only':{'x':[.15,1.],'y':[-.7,.7],'z':[-.1,.8]},'splits':{}}
 for split,k in [('train',a.per_bin),('validation',a.validation_per_bin)]:
  selected=[]
  for b in range(15):
   pool=np.flatnonzero(bins==b)
   selected.extend(pool[:a.per_bin] if split=='train' else pool[a.per_bin:])
  selected=np.array(selected);rng.shuffle(selected);folder=a.output/split;folder.mkdir()
  if split=='train':
   if 2*a.pairs<len(selected):raise ValueError('Not enough pairs to expose every training configuration')
   endpoints=np.concatenate([rng.permutation(selected),rng.choice(selected,2*a.pairs-len(selected))])
   rng.shuffle(endpoints);i=endpoints[:a.pairs];j=endpoints[a.pairs:]
  else:i=selected;j=rng.permutation(selected)
  arrays={'configurations_rad':q[selected],'components':components[selected],'tilts_deg':angles[selected],'stratum':bins[selected],
   'sampled_points':np.concatenate([(q[i]-center)/scale,(q[j]-center)/scale],axis=1).astype(np.float32),
   'task_speed_physical':np.stack([speeds[i],speeds[j]],axis=1),'ee_positions':np.concatenate([xyz[i],xyz[j]],axis=1)}
  for name,value in arrays.items():np.save(folder/(name+'.npy'),value)
  y=xyz[selected,1];tilt=angles[selected];f=weighted[selected]
  report['splits'][split]={'unique_configurations':len(selected),'pairs':len(i),'stratum_counts':np.bincount(bins[selected],minlength=15).tolist(),'active_tilt_violations':{'cup_exclusive':int(((y>.1)&(tilt[:,0]>20)).sum()),'pour_exclusive':int(((y<-.1)&(tilt[:,1]>20)).sum()),'overlap_both':int(((abs(y)<=.1)&(tilt.min(1)>20)).sum())},'weighted_speed_below_02':int((f<.2).sum()),'weighted_speed_above_08':int((f>.8).sum()),'obstacle_below_floor':int((do[selected]<.005).sum()),'self_below_002':int((ds[selected]<.002).sum())}
 report['elapsed_seconds']=time.monotonic()-start;report['hashes']={str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest() for f in a.output.rglob('*.npy')}
 (a.output/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
