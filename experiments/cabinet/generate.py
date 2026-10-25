"""Cabinet-task training data: FK pose pool in the sampling box, clearance labels, coverage augmentation.

The from-source recipe (data_generation/generate.py) with the task orientation removed:

- FK pool: the frozen sampler (data_generation/source/sample_fk_ur5e.py), uniform joint samples whose
  grasp point lies in the scene's sampling box (x, y from `sampling:`, z from the table top to the
  cupboard's top, --height 0.70 for the cabinet scene) and that are collision-free (obstacle >= 0.005 m,
  self >= 0.002 m). No tilt filter (`--task cupping` without `--max_tilt_deg` is the sampler's
  unfiltered mode). Seed 42, 2,000,000 configurations.
- Original pairs: the pool split into 1,000,000 pairs by one seeded permutation (seed 42); every pool
  configuration is an endpoint once. Cupping/pouring pair types do not exist in this task.
- Labels: the original physical components, obstacle clip(d, 0.005, 0.05) / 0.05 and self-collision
  clip(0.5 d, 0.005, 0.05) / 0.05, min-combined. No tilt component. Both SpeedField heads get this
  same clearance speed, so the speed network's FK-y blend of its two heads is the identity.
- Coverage augmentation (seed 20260912): perturb pool configurations by sigma 0.02/0.08/0.20/0.50 rad,
  10% uniform joint-limit draws, reject joint-limit violations; keep grasp points within 0.15 m of the
  sampling box; fill 15 strata (cupboard / box outside the cupboard / outside the box x five clearance
  speed bins) with 133,334 training and 13,334 validation configurations each. Half of the perturbed
  bases come from pool configurations inside the cupboard; otherwise the high-clearance cupboard strata
  are too rare to fill. Train augmentation has 1,000,005 pairs, as in the original.

No evaluation endpoint, path or success label is read.
"""
import argparse,json,pickle,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
import yaml
sys.path.insert(0,'/clean/method')
from common import ROOT,RELEASE,write,sha
sys.path.insert(0,str(ROOT/'data_generation/source'))
import pkl_to_training_data_multi_ur5e as geom
sys.path.insert(0,str(Path(__file__).resolve().parent))
from geometry import scene_boxes,MARGIN,OFFSET,SELF_SCALE

SPEED_EDGES=(.4,.7)
REGIONS=['cupboard','box_outside_cupboard','outside_box']
BINS=['floor_0.1','0.1-0.4','0.4-0.7','0.7-1','free_1.0']
FOCUS_MARGIN=.15


def inside_np(xyz,box):
    return ((xyz>=box[:,0])&(xyz<=box[:,1])).all(1)


def speed_bins(speed):
    """0: floor (within 5 mm or in collision), 1: (0.1, 0.4], 2: (0.4, 0.7], 3: (0.7, 1), 4: >= 0.05 m clearance."""
    b=np.where(speed<=.1+1e-6,0,np.where(speed<=SPEED_EDGES[0],1,np.where(speed<=SPEED_EDGES[1],2,3)))
    return np.where(speed>=1-1e-6,4,b)


def clearance(do,ds):
    return np.minimum(np.clip(do,OFFSET,MARGIN)/MARGIN,np.clip(SELF_SCALE*ds,OFFSET,MARGIN)/MARGIN).astype(np.float32)


class Labeler:
    """The original label code's distances (physical_geometry.compute_physical_distance): obstacle
    distance ignores spheres 0 and 1 (shoulder), self distance uses the release self-collision mask."""
    def __init__(self,scene,device):
        _,_,self.robot,_=geom.init_curobo('/workspace/ur5e_arm.yml',geom.URDF_PATH,device)
        self.centers,self.half=geom.cuboid_half_extents(geom.load_scene_cuboids(str(scene)),device)
        cfg=yaml.safe_load(Path('/workspace/ur5e_arm.yml').read_text())['robot_cfg']['kinematics']
        self.mask,_=geom.build_self_collision_mask(geom.SPHERE_YML_PATH,cfg.get('self_collision_ignore',{}),geom.COLLISION_LINK_NAMES,device=device)
        self.device=device

    @torch.no_grad()
    def measure(self,q,batch=65536):
        """Grasp point, obstacle distance and self distance per configuration (radians)."""
        xyz,do,ds=[],[],[]
        for s in range(0,len(q),batch):
            st=self.robot.get_state(torch.tensor(q[s:s+batch],device=self.device));sp=st.link_spheres_tensor
            xyz.append(st.ee_pose.position.cpu().numpy())
            do.append(geom.sphere_cuboid_distance(sp[:,:,:3],sp[:,:,3],self.centers,self.half,ignore_sphere_indices=geom.OBSTACLE_IGNORE_SPHERE_INDICES).cpu().numpy())
            ds.append(geom.sphere_sphere_distance(sp[:,:,:3],sp[:,:,3],self.mask).cpu().numpy())
        return np.concatenate(xyz),np.concatenate(do).astype(np.float32),np.concatenate(ds).astype(np.float32)


def coverage(pool,pool_xyz,boxes,labeler,norm,out,per_bin,validation_per_bin,pairs,budget):
    rng=np.random.default_rng(20260912);box,cupboard,_=boxes
    focus=box+np.array([[-FOCUS_MARGIN,FOCUS_MARGIN]],np.float32)
    cup_ids=np.flatnonzero(inside_np(pool_xyz,cupboard))
    if not len(cup_ids):raise RuntimeError('No pool configuration inside the cupboard')
    qmin,qmax=norm['q_min'],norm['q_max'];center,scale=norm['q_center'],norm['q_range']
    buckets=[[] for _ in range(15)];counts=np.zeros(15,dtype=int);quota=per_bin+validation_per_bin
    start=time.monotonic();draws=0;batches=0
    while counts.min()<quota:
        if time.monotonic()-start>budget:raise RuntimeError(f'Coverage generation timeout; counts={counts.tolist()}')
        n=65536;base=rng.integers(len(pool),size=n);cup=rng.random(n)<.5
        base[cup]=cup_ids[rng.integers(len(cup_ids),size=int(cup.sum()))]
        q=pool[base].astype(np.float64)
        sigma=rng.choice([.02,.08,.20,.50],size=(n,1));q=q+rng.normal(size=q.shape)*sigma
        uniform=rng.random(n)<.1;q[uniform]=rng.uniform(qmin,qmax,size=(int(uniform.sum()),6))
        # Reject joint-limit violations, rather than clamping mass onto the boundary.
        q=q[((q>qmin)&(q<qmax)).all(axis=1)].astype(np.float32)
        xyz,do,ds=labeler.measure(q);keep=inside_np(xyz,focus);q,xyz,do,ds=q[keep],xyz[keep],do[keep],ds[keep];speed=clearance(do,ds)
        region=np.where(inside_np(xyz,cupboard),0,np.where(inside_np(xyz,box),1,2))
        bins=region*5+speed_bins(speed)
        for b in range(15):
            take=np.flatnonzero(bins==b)[:max(0,quota-counts[b])]
            if len(take):buckets[b].append(np.concatenate([q[take],xyz[take],do[take,None],ds[take,None],speed[take,None]],1));counts[b]+=len(take)
        draws+=n;batches+=1
        if batches%20==0:print({'draws':draws,'counts':counts.tolist(),'seconds':round(time.monotonic()-start,1)},flush=True)
    full=np.concatenate([np.concatenate(b)[:quota] for b in buckets]);stratum=np.repeat(np.arange(15),quota)
    _,unique=np.unique(full[:,:6],axis=0,return_index=True)
    if len(unique)!=len(full):raise RuntimeError('Duplicate augmented configurations')
    q,xyz,do,ds,speed=full[:,:6],full[:,6:9],full[:,9],full[:,10],full[:,11]
    report=dict(seed=20260912,candidate_draws=draws,generation_seconds=time.monotonic()-start,regions=REGIONS,speed_bins=BINS,
        stratum_order='region*5+speed_bin',focus_box=focus.tolist(),focus_margin=FOCUS_MARGIN,cupboard_base_share=.5,
        cupboard_pool_configurations=len(cup_ids),no_test_suite_used=True,splits={})
    for split in ['train','validation']:
        selected=[]
        for b in range(15):
            pool_b=np.flatnonzero(stratum==b)
            selected.extend(pool_b[:per_bin] if split=='train' else pool_b[per_bin:])
        selected=np.array(selected);rng.shuffle(selected);folder=out/split;folder.mkdir(parents=True)
        if split=='train':
            if 2*pairs<len(selected):raise ValueError('Not enough pairs to expose every training configuration')
            endpoints=np.concatenate([rng.permutation(selected),rng.choice(selected,2*pairs-len(selected))])
            rng.shuffle(endpoints);i=endpoints[:pairs];j=endpoints[pairs:]
        else:i=selected;j=rng.permutation(selected)
        arrays={'configurations_rad':q[selected],'d_obstacle':do[selected],'d_self':ds[selected],'stratum':stratum[selected],
            'sampled_points':np.concatenate([(q[i]-center)/scale,(q[j]-center)/scale],axis=1).astype(np.float32),
            'task_speed_physical':np.stack([np.stack([speed[i],speed[i]],1),np.stack([speed[j],speed[j]],1)],axis=1).astype(np.float32),
            'ee_positions':np.concatenate([xyz[i],xyz[j]],axis=1)}
        for name,value in arrays.items():np.save(folder/(name+'.npy'),value)
        report['splits'][split]=dict(unique_configurations=len(selected),pairs=len(i),stratum_counts=np.bincount(stratum[selected],minlength=15).tolist(),
            obstacle_below_floor=int((do[selected]<.005).sum()),self_below_002=int((ds[selected]<.002).sum()))
    report['hashes']={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*.npy'))}
    write(out/'manifest.json',report);return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    out=a.output;out.mkdir(exist_ok=False);(out/'pools').mkdir();started=time.monotonic();timing={};device='cuda:0'
    doc=yaml.safe_load(a.scene.read_text());boxes=scene_boxes(a.scene);box=boxes[0]
    table=[c for c in doc['cuboids'] if c['name']=='table'][0];table_top=table['pose'][2]+table['dims'][2]/2
    height=round(doc['sampling']['z'][1]-table_top,6);assert abs(box[2,0]-table_top)<1e-6 and abs(box[2,1]-(table_top+height))<1e-6,'sampling z must start at the table top'
    pool_size=4096 if a.smoke else 2000000
    command=[sys.executable,'-u',str(ROOT/'data_generation/source/sample_fk_ur5e.py'),'--n_points',str(pool_size),'--height',str(height),
             '--scene',str(a.scene),'--device',device,'--seed','42','--task','cupping','--output',str(out/'pools/cabinet.pkl')]
    write(out/'cabinet_pool.command.json',command);begin=time.monotonic()
    with (out/'cabinet_pool.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    timing['cabinet_pool']=time.monotonic()-begin;print('SAMPLING_STAGE cabinet_pool',timing['cabinet_pool'],flush=True)
    with (out/'pools/cabinet.pkl').open('rb') as f:result=pickle.load(f)
    assert result['max_tilt_deg'] is None and result['sampling_method']=='fk' and result['n_success']==pool_size
    pool=np.asarray(result['q_solutions'],np.float32);pool_xyz=np.asarray(result['positions'],np.float32)
    assert inside_np(pool_xyz,box).all(),'pool grasp point outside the sampling box'
    norm=np.load(RELEASE/'configs/joint_normalization.npz');center,scale=norm['q_center'],norm['q_range']
    assert ((pool>=norm['q_min'])&(pool<=norm['q_max'])).all()
    labeler=Labeler(a.scene,device)
    # Original pairs: one seeded permutation of the pool, so each configuration is an endpoint once.
    begin=time.monotonic();rng=np.random.default_rng(42);order=rng.permutation(len(pool));half=len(pool)//2
    i,j=order[:half],order[half:2*half];xyz,do,ds=labeler.measure(pool);speed=clearance(do,ds)
    assert np.abs(xyz-pool_xyz).max()<1e-4,'grasp point differs from the sampler'
    original=out/'original';original.mkdir()
    arrays={'sampled_points':np.concatenate([(pool[i]-center)/scale,(pool[j]-center)/scale],1).astype(np.float32),
        'task_speed_physical':np.stack([np.stack([speed[i],speed[i]],1),np.stack([speed[j],speed[j]],1)],axis=1).astype(np.float32),
        'ee_positions':np.concatenate([pool_xyz[i],pool_xyz[j]],1),'d_obstacle':np.stack([do[i],do[j]],1),'d_self':np.stack([ds[i],ds[j]],1)}
    for name,value in arrays.items():np.save(original/(name+'.npy'),value)
    _,cupboard,compartments=boxes
    pool_stats=dict(size=len(pool),fk_draws=int(result['n_total']),acceptance=float(len(pool)/result['n_total']),
        cupboard_share=float(inside_np(pool_xyz,cupboard).mean()),compartment_counts={k:int(inside_np(pool_xyz,v).sum()) for k,v in compartments.items()},
        speed_quantiles={str(qq):float(np.quantile(speed,qq)) for qq in (.05,.25,.5,.75,.95)},speed_floor_share=float((speed<=.1+1e-6).mean()),
        speed_free_share=float((speed>=1-1e-6).mean()),obstacle_below_005=int((do<.005).sum()),self_below_002=int((ds<.002).sum()))
    timing['original_pairs']=time.monotonic()-begin;print('SAMPLING_STAGE original_pairs',timing['original_pairs'],json.dumps(pool_stats),flush=True)
    begin=time.monotonic()
    cov=coverage(pool,pool_xyz,boxes,labeler,norm,out/'coverage',128 if a.smoke else 133334,16 if a.smoke else 13334,
                 4096 if a.smoke else 1000005,1800)
    timing['coverage']=time.monotonic()-begin;print('SAMPLING_STAGE coverage',timing['coverage'],flush=True)
    inputs=out/'inputs'
    for kind,folder in [('original',original),('augmentation',out/'coverage/train')]:
        (inputs/kind).mkdir(parents=True)
        for name in ['sampled_points.npy','task_speed_physical.npy']:(inputs/kind/name).write_bytes((folder/name).read_bytes())
    files={str(f.relative_to(inputs)):sha(f) for f in sorted(inputs.rglob('*.npy'))}
    write(inputs/'source_manifest.json',dict(files=files,source='cabinet task: FK pool in the sampling box, clearance-only labels, cabinet coverage strata',
        evaluation_rows_read=False,smoke=a.smoke))
    write(out/'complete.json',dict(complete=True,smoke=a.smoke,scene=str(a.scene),scene_sha256=sha(a.scene),source_seeds=dict(fk=42,pair=42,coverage=20260912),
        sampling_box=box.tolist(),pool=pool_stats,original_pairs=half,augmentation_pairs=cov['splits']['train']['pairs'],
        coverage_counts=cov['splits']['train']['stratum_counts'],timing=timing,elapsed_seconds=time.monotonic()-started,
        evaluation_rows_read=False,training_input_hashes=files))

if __name__=='__main__':main()
