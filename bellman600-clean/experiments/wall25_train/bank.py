"""Offline goal-blind rollforward bank; fixed original sampling and batched veto."""
import time
import numpy as np
import torch
from common import sha,write

STEPS=(.01,.05,.15,.25)

def successors(source,rng):
    direction=rng.normal(size=(len(source),4,6)).astype(np.float32)
    direction/=np.maximum(np.linalg.norm(direction,axis=2,keepdims=True),1e-8)
    direction=np.concatenate((direction,-direction),axis=1)
    sizes=np.repeat(np.asarray(STEPS,np.float32),8)
    z=source[:,None,:]+sizes[None,:,None]*np.tile(direction,(1,4,1))
    return z.astype(np.float32),sizes,((z>=-.5)&(z<=.5)).all(2)

def select(safe,rng):
    # Match original RNG calls: uniformly choose a safe scale, then a safe direction.
    masks=safe.reshape(len(safe),4,8);valid=masks.any(2)
    chosen=np.empty(len(safe),np.int32)
    for row in range(len(safe)):
        scale=int(rng.choice(np.flatnonzero(valid[row])))
        direction=int(rng.choice(np.flatnonzero(masks[row,scale])))
        chosen[row]=scale*8+direction
    return chosen

@torch.no_grad()
def build(seed_source,speed,gate,output,speed_path,scene,smoke=False):
    output.mkdir(exist_ok=False);started=time.monotonic()
    walkers,generations=(512,2) if smoke else (50000,8)
    rng=np.random.default_rng(87344)
    origin=rng.choice(len(seed_source),size=walkers,replace=False).astype(np.int64)
    frontier=np.asarray(seed_source[origin],np.float32).copy()
    # Preserve the original random stream (heading was unused by uniform selection).
    rng.normal(size=frontier.shape)
    alpha=torch.linspace(0,1,17,device='cuda:0');reports=[]
    names=['source_q','successor_q','local_cost','minimum_speed','in_domain','safe','source_rows','rollforward_generation','continued_action']
    collected={name:[] for name in names}
    for generation in range(generations):
        z,sizes,domain=successors(frontier,rng)
        costs=np.empty(domain.shape,np.float32);minima=np.empty_like(costs)
        integral_start=time.monotonic()
        for begin in range(0,len(frontier),512):
            end=min(begin+512,len(frontier))
            q=torch.tensor(frontier[begin:end],device='cuda:0');zz=torch.tensor(z[begin:end],device='cuda:0')
            chord=q[:,None,None,:]+alpha[None,None,:,None]*(zz[:,:,None,:]-q[:,None,None,:])
            ss=speed(chord.reshape(-1,6)).reshape(end-begin,32,17)
            costs[begin:end]=(torch.linalg.vector_norm(zz-q[:,None,:],dim=2)*torch.trapezoid(ss.clamp_min(1e-5).reciprocal(),dim=-1)/16).cpu().numpy()
            minima[begin:end]=ss.amin(2).cpu().numpy()
        integral_seconds=time.monotonic()-integral_start
        veto_start=time.monotonic();exact=gate.edges(frontier,z,domain)
        safe=domain&exact&(minima>=.015);valid=safe.any(1);ids=np.flatnonzero(valid)
        if not len(ids):raise RuntimeError('Generation has no safe walker')
        chosen=select(safe[ids],rng)
        values=[frontier[ids].copy(),z[ids].copy(),costs[ids].copy(),minima[ids].copy(),domain[ids].copy(),safe[ids].copy(),origin[ids].copy(),np.full(len(ids),generation,np.uint8),chosen]
        for name,value in zip(names,values):collected[name].append(value)
        next_q=np.empty_like(frontier);next_origin=np.empty_like(origin)
        next_q[ids]=z[ids,chosen];next_origin[ids]=origin[ids]
        bad=np.flatnonzero(~valid)
        if len(bad):
            restart=rng.choice(len(seed_source),size=len(bad),replace=False)
            next_q[bad]=seed_source[restart];next_origin[bad]=restart
            rng.normal(size=(len(bad),6))
        report=dict(generation=generation,walkers=walkers,valid_transition_rows=len(ids),restart_rows=len(bad),
            safe_edge_fraction=float(safe.mean()),integral_seconds=integral_seconds,veto_and_selection_seconds=time.monotonic()-veto_start,
            elapsed_seconds=time.monotonic()-started)
        reports.append(report);print('BANK',report,flush=True)
        write(output/'progress.json',dict(complete=False,generations=reports))
        frontier,origin=next_q,next_origin
    files={}
    for name,parts in collected.items():
        path=output/(name+'.npy');np.save(path,np.concatenate(parts),allow_pickle=False);files[path.name]=sha(path)
    path=output/'action_step_size.npy';np.save(path,sizes);files[path.name]=sha(path)
    n=sum(x['valid_transition_rows'] for x in reports)
    assert 0<n<=walkers*generations
    manifest=dict(schema_version=1,anchors=n,anchors_requested=walkers*generations,actions_per_anchor=32,direction_pairs=4,
        step_sizes=list(STEPS),quadrature_points=17,minimum_speed=.015,local_cost='trapezoid integral of ds/frozen wall-trained scalar speed',
        scene_sha256=sha(scene),speed_checkpoint_sha256=sha(speed_path),bank_seed=87344,
        exact_veto=dict(densify_step_rad=.02,collision_margin=.005,self_collision_margin=.002,tilt_deg=20,overlap_logic='or',workspace=False),
        information_boundary=dict(evaluation_rows_read=False,goals_read=False,paths_read=False,graph_search_or_dp_used=False),
        rollforward=dict(walkers=walkers,generations=generations,selection='uniform_safe_scale_then_direction',reports=reports),
        file_sha256=files,elapsed_seconds=time.monotonic()-started,smoke=smoke)
    write(output/'manifest.json',manifest);write(output/'progress.json',dict(complete=True,**manifest))
    return manifest
