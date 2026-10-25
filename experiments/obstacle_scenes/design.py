"""Scene-design preview data: frozen-600 endpoint validity and valid example poses per target region.

Runs in the pinned environment. Uses the strict endpoint rule of transition_wall/audit.py
(obstacle >= 0.005 m, self >= 0.002 m, zone tilt <= 20 deg with OR in the transition band,
normalization joint bounds). Example poses come from uniform joint-space samples, the same
proposal as the FK training sampler. Nothing here is used for training or evaluation.
"""
import argparse,json,sys,time
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

EXAMPLES=8
CLOUD=4000


def farthest(points,count,first):
    chosen=[first];distance=np.linalg.norm(points-points[first],axis=1)
    while len(chosen)<min(count,len(points)):
        chosen.append(int(distance.argmax()));distance=np.minimum(distance,np.linalg.norm(points-points[chosen[-1]],axis=1))
    return np.array(chosen)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--scenes',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=400_000_000);p.add_argument('--seed',type=int,default=20261013)
    p.add_argument('--targets',type=Path,default=Path(__file__).resolve().parent/'targets.yml');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);dev='cuda:0';started=time.monotonic()
    base=RELEASE/'configs/scene.yml'
    _,robot=E.setup_robot_world('/workspace/ur5e_arm.yml',WorldConfig.from_dict(yaml.safe_load(base.read_text())),dev)
    rcfg=yaml.safe_load(Path('/workspace/ur5e_arm.yml').read_text())
    mask,_=build_self_collision_mask(E.SPHERE_YML_PATH,rcfg['robot_cfg']['kinematics'].get('self_collision_ignore',{}),E.COLLISION_LINK_NAMES,device=dev)
    axes=E.load_task_axes(str(base));zones=E.load_zone_config(str(base),'auto')[1]
    norm=np.load(RELEASE/'configs/joint_normalization.npz')
    lo=torch.tensor(norm['q_min'],device=dev);hi=torch.tensor(norm['q_max'],device=dev)
    docs={s.stem:yaml.safe_load(s.read_text()) for s in a.scenes};regions=yaml.safe_load(a.targets.read_text())
    geometry={s.stem:load_cuboid_geometry(str(s),dev) for s in a.scenes}

    def check(q):
        state=robot.get_state(q);spheres=state.link_spheres_tensor
        centers,radii=spheres[:,:,:3].clone(),spheres[:,:,3].clone();position=state.ee_pose.position.clone()
        own=sphere_sphere_distance(centers,radii,mask)
        cup,pour,_,_=E._zone_tilts_batch(q,robot,return_pos=True,task_axes=axes)
        cm,pm,ov=E.corner_masks_torch(position[:,:2],zones,assign_outside=True)
        orient=torch.where(ov,(cup<=20)|(pour<=20),torch.where(cm,cup<=20,pour<=20))
        bounds=((q>=lo-1e-5)&(q<=hi+1e-5)).all(1)
        return {name:(sphere_cuboid_distance(centers,radii,*geometry[name])>=.005)&(own>=.002)&orient&bounds for name in docs},position

    arrays={};summary=dict(seed=a.seed,samples=a.samples,scenes={},rule='obstacle>=0.005, self>=0.002, zone tilt<=20 OR in transition, joint bounds')
    suites=json.loads((RELEASE/'tests/suite_manifest.json').read_text())['suites']
    frozen=dict(ee=[],ok={name:[] for name in docs})
    for item in suites:
        z=np.load(RELEASE/'tests/suites'/item['file'],allow_pickle=True)
        q=torch.tensor(np.concatenate([z['starts_rad'],z['goals_rad']]),device=dev);valid,position=check(q)
        frozen['ee'].append(position.cpu().numpy())
        for name in docs:frozen['ok'][name].append(valid[name].cpu().numpy())
    arrays['frozen_ee']=np.concatenate(frozen['ee']).astype(np.float32)
    for name in docs:
        ok=np.concatenate(frozen['ok'][name]);arrays[name+'/frozen_ok']=ok
        pairs=ok.reshape(len(suites),2,100);summary['scenes'][name]=dict(scene_sha256=sha(a.scenes[[s.stem for s in a.scenes].index(name)]),
            frozen_endpoint_valid_queries=int((pairs[:,0]&pairs[:,1]).sum()),frozen_valid_endpoints=int(ok.sum()),targets={})

    generator=torch.Generator(device=dev).manual_seed(a.seed);pools={name:{t['name']:[] for t in regions.get(name,[])} for name in docs}
    boxes={name:{t['name']:torch.tensor([t['x'],t['y'],t['z']],device=dev) for t in regions.get(name,[])} for name in docs}
    batch=50000
    for _ in range(a.samples//batch):
        q=lo+(hi-lo)*torch.rand(batch,6,device=dev,generator=generator)
        position=robot.get_state(q).ee_pose.position
        near=(position[:,0]>.25)&(position[:,0]<.95)&(position[:,1].abs()<.6)&(position[:,2]>-.04)&(position[:,2]<.7)
        q=q[near]
        if not len(q):continue
        valid,position=check(q)
        for name in docs:
            for target,box in boxes[name].items():
                inside=valid[name]&((position>=box[:,0])&(position<=box[:,1])).all(1)
                if inside.any():pools[name][target].append(torch.cat([q[inside],position[inside]],1).cpu())
    rng=np.random.default_rng(a.seed)
    for name in docs:
        for target,parts in pools[name].items():
            pool=torch.cat(parts).numpy() if parts else np.zeros((0,9),np.float32)
            summary['scenes'][name]['targets'][target]=dict(valid_samples=len(pool))
            if not len(pool):continue
            cloud=pool[rng.choice(len(pool),min(CLOUD,len(pool)),replace=False)]
            # Examples spread over the target; the shelf starts from the deepest reach.
            first=int(pool[:,6].argmax()) if name=='shelf' else int(rng.integers(len(pool)))
            examples=pool[farthest(pool[:,6:9],EXAMPLES,first)]
            arrays[f'{name}/{target}/cloud']=cloud[:,6:9].astype(np.float32)
            arrays[f'{name}/{target}/examples_q']=examples[:,:6].astype(np.float32)
            arrays[f'{name}/{target}/examples_ee']=examples[:,6:9].astype(np.float32)
            print(name,target,len(pool),flush=True)
    summary['elapsed_seconds']=time.monotonic()-started
    np.savez_compressed(a.output/'preview.npz',**arrays);write(a.output/'preview.json',summary)
    print(json.dumps(summary,indent=1),flush=True)


if __name__=='__main__':main()
