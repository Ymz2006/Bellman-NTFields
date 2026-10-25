"""Metric NTFields training data for the cabinet task.

The ntrl-demo arm recipe (ntrl_demo/dataprocessing/speed_sampling_arm_normal.py, 500,000 pairs from
ntrl_demo/configs/arm.txt): x0 is a configuration near an obstacle, 0 < d - 0.01 < margin with margin
0.05 m; x1 = x0 + a random unit direction (uniform cube draw, normalized) times U(0, 0.5), kept inside
[-0.5, 0.5]^6; per endpoint the speed clip(., offset, margin) / margin with offset 0.005 m and the unit
normal, the direction of the clearance gradient. Cabinet adaptations:

- x0 candidates are uniform joint samples whose grasp point lies in the cabinet sampling box, as for
  the Bellman pool, instead of IK solutions of end-effector poses in ntrl-demo's box and orientation
  cone: the cabinet task has no orientation.
- Joint coordinates are the release normalization (q - center) / range, [-0.5, 0.5]^6 over all joint
  limits (ntrl-demo: q / 2 pi, which covers only +-pi). The planners and scorer use the same coordinates.
- d is the Bellman labels' clearance, min(obstacle distance, 0.5 x self distance), with exact
  sphere-cuboid distances (ntrl-demo: KD-tree over an obstacle point cloud, no self term). The speed is
  then exactly the Bellman clearance speed clip(d, 0.005, 0.05) / 0.05: ntrl-demo shifts d by 0.01 m to
  inflate its point cloud, which exact geometry does not need. The x0 filter keeps ntrl-demo's shift.
- The normal is the unit central-difference gradient of d in the normalized coordinates (ntrl-demo:
  J^T (p - v) of the closest sphere, the same direction in its coordinates). Rows with a zero gradient
  are dropped, as ntrl-demo drops NaN normals.

Seed 20261019. Writes sampled_points.npy (N, 12), speed.npy (N, 2), normal.npy (N, 12), the files
ntrl_demo/models/data_mlp.py reads.
"""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,'/clean/method')
from common import RELEASE,sha,write
sys.path.insert(0,'/clean/data_generation/source')
import pkl_to_training_data_multi_ur5e as geom
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from geometry import scene_boxes,MARGIN,OFFSET,SELF_SCALE
import numpy as np
import torch
import yaml

EPS=1e-4


class Clearance:
    """d(x) = min(obstacle distance, 0.5 self distance) at normalized x, label code's spheres and mask."""
    def __init__(self,scene,device):
        _,_,self.robot,_=geom.init_curobo('/workspace/ur5e_arm.yml',geom.URDF_PATH,device)
        self.centers,self.half=geom.cuboid_half_extents(geom.load_scene_cuboids(str(scene)),device)
        cfg=yaml.safe_load(Path('/workspace/ur5e_arm.yml').read_text())['robot_cfg']['kinematics']
        self.mask,_=geom.build_self_collision_mask(geom.SPHERE_YML_PATH,cfg.get('self_collision_ignore',{}),geom.COLLISION_LINK_NAMES,device=device)
        n=np.load(RELEASE/'configs/joint_normalization.npz')
        self.center=torch.tensor(n['q_center'],device=device);self.range=torch.tensor(n['q_range'],device=device)

    @torch.no_grad()
    def __call__(self,x,positions=False):
        st=self.robot.get_state(x*self.range+self.center);sp=st.link_spheres_tensor;pos=st.ee_pose.position.clone()
        do=geom.sphere_cuboid_distance(sp[:,:,:3],sp[:,:,3],self.centers,self.half,ignore_sphere_indices=geom.OBSTACLE_IGNORE_SPHERE_INDICES)
        d=torch.minimum(do,SELF_SCALE*geom.sphere_sphere_distance(sp[:,:,:3],sp[:,:,3],self.mask))
        return (d,pos) if positions else d

    @torch.no_grad()
    def gradient(self,x):
        steps=torch.eye(6,device=x.device)*EPS
        plus=torch.stack([self(x+s) for s in steps],1);minus=torch.stack([self(x-s) for s in steps],1)
        return (plus-minus)/(2*EPS)


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    device='cuda:0';box,cupboard,_=scene_boxes(a.scene);box_t=torch.tensor(box,device=device);cup_t=torch.tensor(cupboard,device=device)
    numsamples=4096 if a.smoke else 500000;margin=MARGIN;offset=OFFSET;batch=200000
    torch.manual_seed(20261019);C=Clearance(a.scene,device)
    xs,ys,ns=[],[],[];drawn=0;stats=dict(x0_in_box=0,x0_near_obstacle=0,x1_inside_domain=0,nonzero_normals=0)
    while sum(len(x) for x in xs)<numsamples:
        x0=torch.rand(batch,6,device=device)-.5;drawn+=batch
        d0,pos=C(x0,positions=True);keep=((pos>=box_t[:,0])&(pos<=box_t[:,1])).all(1);stats['x0_in_box']+=int(keep.sum())
        y0=d0-.01;keep&=(y0>0)&(y0<margin);stats['x0_near_obstacle']+=int(keep.sum());x0,d0=x0[keep],d0[keep]
        dP=torch.rand(len(x0),6,device=device)-.5;rL=torch.rand(len(x0),1,device=device)*.5
        x1=x0+torch.nn.functional.normalize(dP,dim=1)*rL
        inside=((x1<=.5)&(x1>=-.5)).all(1);x0,x1,d0=x0[inside],x1[inside],d0[inside];stats['x1_inside_domain']+=int(inside.sum())
        if len(x0)<=1:continue
        d1=C(x1);g0=C.gradient(x0);g1=C.gradient(x1)
        n0=g0/torch.linalg.vector_norm(g0,dim=1,keepdim=True);n1=g1/torch.linalg.vector_norm(g1,dim=1,keepdim=True)
        good=~(n0.isnan().any(1)|n1.isnan().any(1)|n0.isinf().any(1)|n1.isinf().any(1));stats['nonzero_normals']+=int(good.sum())
        xs.append(torch.cat([x0,x1],1)[good].cpu());ys.append(torch.stack([d0,d1],1)[good].cpu());ns.append(torch.cat([n0,n1],1)[good].cpu())
        print({'drawn':drawn,'collected':sum(len(x) for x in xs)},flush=True)
    X=torch.cat(xs)[:numsamples].numpy().astype(np.float32);D=torch.cat(ys)[:numsamples].numpy();N=torch.cat(ns)[:numsamples].numpy().astype(np.float32)
    speed=(np.clip(D,offset,margin)/margin).astype(np.float32)
    np.save(a.output/'sampled_points.npy',X);np.save(a.output/'speed.npy',speed);np.save(a.output/'normal.npy',N);np.save(a.output/'clearance.npy',D.astype(np.float32))
    with torch.no_grad():
        pos=torch.cat([C(torch.tensor(X[i:i+65536,:6],device=device),positions=True)[1].cpu() for i in range(0,len(X),65536)]).numpy()
    report=dict(complete=True,smoke=a.smoke,scene=str(a.scene),scene_sha256=sha(a.scene),seed=20261019,pairs=len(X),uniform_draws=drawn,counts=stats,
        recipe='ntrl-demo speed_sampling_arm_normal.py with cabinet adaptations (metric/data.py docstring)',margin=margin,offset=offset,
        x0_filter='0 < d - 0.01 < margin',x1_step='unit(rand-0.5) * U(0, 0.5)',normal='central difference, eps 1e-4 (normalized)',
        speed_quantiles={str(q):float(np.quantile(speed,q)) for q in (.05,.25,.5,.75,.95)},x0_cupboard_share=float(((pos>=cupboard[:,0])&(pos<=cupboard[:,1])).all(1).mean()),
        x1_speed_floor_share=float((speed[:,1]<=.1+1e-6).mean()),elapsed_seconds=time.monotonic()-started,evaluation_rows_read=False,
        files={n:sha(a.output/(n+'.npy')) for n in ['sampled_points','speed','normal','clearance']})
    write(a.output/'complete.json',report);print(json.dumps(report,indent=1),flush=True)

if __name__=='__main__':main()
