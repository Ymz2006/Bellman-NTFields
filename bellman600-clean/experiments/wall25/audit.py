"""Endpoint eligibility under the original scene and the added wall."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,'/clean/method')
from common import ROOT,RELEASE,write,sha
sys.path.insert(0,'/workspace')
import numpy as np
import torch
import yaml
import ur5e_plan_eval_multi as E
from curobo.geom.types import WorldConfig
from collision_utils import load_cuboid_geometry,build_self_collision_mask,sphere_cuboid_distance,sphere_sphere_distance

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 scene=ROOT/'experiments/wall25/scene.yml';config=yaml.safe_load(scene.read_text())
 baseline=yaml.safe_load((RELEASE/'configs/scene.yml').read_text())
 assert config['cuboids'][:-1]==baseline['cuboids']
 for key in ['sampling','task_axes']:assert config[key]==baseline[key]
 clean_zones=json.loads(json.dumps(config['zones']));clean_zones['cup_region']['label']=baseline['zones']['cup_region']['label']
 assert clean_zones==baseline['zones'],'Only obstacle geometry may change'
 _,robot=E.setup_robot_world('/workspace/ur5e_arm.yml',WorldConfig.from_dict(config),'cuda:0')
 centers,half=load_cuboid_geometry(str(scene),'cuda:0')
 rcfg=yaml.safe_load(Path('/workspace/ur5e_arm.yml').read_text())
 mask,_=build_self_collision_mask(E.SPHERE_YML_PATH,rcfg['robot_cfg']['kinematics'].get('self_collision_ignore',{}),E.COLLISION_LINK_NAMES,device='cuda:0')
 axes=E.load_task_axes(str(scene));norm=np.load(RELEASE/'configs/joint_normalization.npz')
 lo=norm['q_min'] if 'q_min' in norm.files else norm['q_center']-norm['q_range']
 hi=norm['q_max'] if 'q_max' in norm.files else norm['q_center']+norm['q_range']
 report=dict(complete=False,scene_sha256=sha(scene),wall=config['cuboids'][-1],tasks={},total=600,
     protocol=dict(tilt_deg=20,overlap_logic='or',overlap=.10,obstacle_margin=.005,self_margin=.002),
     caveat='Valid endpoints do not certify a collision-free path exists with the added wall.')
 for item in json.loads((RELEASE/'tests/suite_manifest.json').read_text())['suites']:
  z=np.load(RELEASE/'tests/suites'/item['file'],allow_pickle=True);n=len(z['starts_rad']);assert n==100
  q=np.concatenate((z['starts_rad'],z['goals_rad']));qt=torch.tensor(q,device='cuda:0')
  state=robot.get_state(qt);spheres=state.link_spheres_tensor
  table=sphere_cuboid_distance(spheres[:,:,:3],spheres[:,:,3],centers[:1],half[:1]).cpu().numpy()
  wall=sphere_cuboid_distance(spheres[:,:,:3],spheres[:,:,3],centers[1:],half[1:]).cpu().numpy()
  own=sphere_sphere_distance(spheres[:,:,:3],spheres[:,:,3],mask).cpu().numpy()
  cup,pour,_,pos=E._zone_tilts_batch(qt,robot,return_pos=True,task_axes=axes)
  zones=json.loads(str(z['corner_zones_json']))
  cm,pm,ov=E.corner_masks_torch(pos[:,:2],zones,assign_outside=True)
  orient=torch.where(ov,(cup<=20)|(pour<=20),torch.where(cm,cup<=20,pour<=20)).cpu().numpy()
  old=(table>=.005)&(own>=.002)&orient&((q>=lo-1e-5)&(q<=hi+1e-5)).all(axis=1)
  new=old&(wall>=.005)
  rows=[]
  for i in range(n):
   rows.append(dict(query_id=i,baseline_endpoints_valid=bool(old[i] and old[n+i]),
       wall_endpoints_valid=bool(new[i] and new[n+i]),start_wall_clearance=float(wall[i]),
       goal_wall_clearance=float(wall[n+i]),start_valid=bool(new[i]),goal_valid=bool(new[n+i])))
  report['tasks'][item['task']]=dict(n=n,baseline_valid=sum(r['baseline_endpoints_valid'] for r in rows),
       wall_valid=sum(r['wall_endpoints_valid'] for r in rows),per_query=rows)
 report['baseline_valid']=sum(t['baseline_valid'] for t in report['tasks'].values())
 report['wall_valid']=sum(t['wall_valid'] for t in report['tasks'].values());report['complete']=True
 write(a.output/'endpoint_audit.json',report)
 print(json.dumps({k:v for k,v in report.items() if k!='tasks'}),flush=True)
 for task,row in report['tasks'].items():print(task,row['baseline_valid'],row['wall_valid'],flush=True)
if __name__=='__main__':main()
