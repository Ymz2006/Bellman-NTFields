"""Rebuild the original no-wall speed and offline transition bank."""
import sys
from pathlib import Path
sys.path.insert(0,'/clean/method')
from common import ROOT,RELEASE,sha,write
from speed import SpeedField
from geometry import ExactLocalGate
from physical import PhysicalSpeed
from bank import build
import argparse,json,time
import numpy as np
import torch

EXPERIMENT=ROOT/'experiments/no_wall_rebuild'
SCENE=RELEASE/'configs/scene.yml'
INPUTS=ROOT/'training_inputs/no_wall_seed_pool'

def components(net,q):
    return net.head(net.tower(net.features(q))).clamp(-2.302585093,0).exp()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    out=a.output;out.mkdir(exist_ok=False);(out/'labels').mkdir();started=time.monotonic()
    source_manifest=json.loads((EXPERIMENT/'source_manifest.json').read_text())
    for name,digest in source_manifest['files'].items():assert sha(INPUTS/name)==digest,name
    arrays=[]
    for kind in ['original','augmentation']:
        points=np.load(INPUTS/kind/'sampled_points.npy',mmap_mode='r')
        labels=np.load(INPUTS/kind/'task_speed_physical.npy',mmap_mode='r')
        assert points.shape[1]==12 and labels.shape==(len(points),2,2)
        ids=np.arange(len(points));arrays.append((points,labels,ids[ids%100!=0],ids[ids%100==0]))
    torch.manual_seed(87358);rng=np.random.default_rng(87358)
    speed=SpeedField().cuda();torch.nn.init.normal_(speed.head.weight,std=.005);torch.nn.init.constant_(speed.head.bias,-1.15)
    opt=torch.optim.AdamW(speed.parameters(),lr=.0003,weight_decay=1e-6)
    steps=12 if a.smoke else 20000;history=[];speed_started=time.monotonic()
    for step in range(1,steps+1):
        qs=[];ss=[]
        for q,s,ids,_ in arrays:
            rows=rng.choice(ids,512);qs.append(q[rows].copy().reshape(-1,6));ss.append(s[rows].copy().reshape(-1,2))
        qt=torch.as_tensor(np.concatenate(qs),device='cuda:0');target=torch.as_tensor(np.concatenate(ss),device='cuda:0')
        pred=components(speed,qt);loss=(pred.log()-target.clamp_min(.1).log()).square().mean()
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Speed loss is nonfinite')
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(speed.parameters(),2);opt.step()
        if step%250==0 or step==steps:
            row=dict(step=step,log_mse=float(loss),seconds=time.monotonic()-speed_started);history.append(row)
            write(out/'speed_progress.json',history);print('SPEED',row,flush=True)
    speed.eval().requires_grad_(False);validation=[]
    with torch.no_grad():
        for kind,(q,s,_,ids) in zip(['original','augmentation'],arrays):
            errors=[];logs=[]
            for chunk in np.array_split(ids,max(1,len(ids)//1024)):
                qt=torch.tensor(q[chunk].copy().reshape(-1,6),device='cuda:0');target=torch.tensor(s[chunk].copy().reshape(-1,2),device='cuda:0')
                pred=components(speed,qt);errors.append((pred-target).abs().cpu().numpy());logs.append((pred.log()-target.log()).square().cpu().numpy())
            error=np.concatenate(errors);validation.append(dict(source=kind,endpoint_count=len(error),mae=float(error.mean()),p95=float(np.quantile(error,.95)),log_rmse=float(np.sqrt(np.concatenate(logs).mean()))))
    contract=dict(steps=steps,seed=87358,scene_sha256=sha(SCENE),architecture='Fourier128_harmonics4_width512_blocks6_two_training_speed_outputs',
        label_sources='frozen original and augmentation physical task speeds; 50/50 sampling',validation_split='pair index modulo100==0',
        scalar_composition='FK-Y smoothstep',evaluation_rows_read=False,smoke=a.smoke)
    speed_path=out/'speed.pt';torch.save(dict(model_state_dict=speed.state_dict(),training_contract=contract,validation=validation),speed_path)
    speed_seconds=time.monotonic()-speed_started
    write(out/'speed_audit.json',dict(training=contract,validation=validation,seconds=speed_seconds))
    # Regenerate every physical label, and audit against the original training data.
    # Fit above uses original labels to preserve the historical random/numeric stream.
    gate=ExactLocalGate(SCENE);physical=PhysicalSpeed(gate);audits=[];label_started=time.monotonic()
    for kind,(points,labels,_,_) in zip(['original','augmentation'],arrays):
        n=min(4096,len(points)) if a.smoke else len(points)
        rebuilt=np.lib.format.open_memmap(out/'labels'/(kind+'.npy'),mode='w+',dtype=np.float32,shape=(n,2,2))
        maximum=0.;equal=True
        for begin in range(0,n,4096):
            end=min(begin+4096,n);q=torch.tensor(points[begin:end].copy().reshape(-1,6),device='cuda:0')
            values,_=physical.tasks(q);values=values.cpu().numpy().reshape(end-begin,2,2)
            maximum=max(maximum,float(np.abs(values-labels[begin:end]).max()))
            equal=equal and np.array_equal(values,labels[begin:end]);rebuilt[begin:end]=values
        rebuilt.flush();assert maximum<=2e-5,('Physical label mismatch',kind,maximum)
        audits.append(dict(source=kind,pairs=n,max_absolute_error=maximum,bitwise_equal=equal))
    relabel_seconds=time.monotonic()-label_started
    write(out/'physical_label_audit.json',dict(sources=audits,seconds=relabel_seconds,evaluation_rows_read=False))
    selected=[];count=1024 if a.smoke else 100000;attempts=0
    while sum(map(len,selected))<count:
        q,_,ids,_=arrays[attempts%2];rows=rng.choice(ids,min(8192,count));candidates=q[rows].copy().reshape(-1,6);valid=[]
        for begin in range(0,len(candidates),4096):valid.append(gate.states(torch.tensor(candidates[begin:begin+4096],device='cuda:0')).cpu().numpy())
        selected.append(candidates[np.concatenate(valid)]);attempts+=1
        if attempts>1000:raise RuntimeError('Insufficient safe training poses')
    seeds=np.concatenate(selected)[:count];np.save(out/'seed_states.npy',seeds)
    manifest=build(seeds,speed,gate,out/'bank',speed_path,SCENE,a.smoke)
    source=np.load(out/'bank/source_q.npy',mmap_mode='r');ids=rng.choice(len(source),min(4096,len(source)),replace=False);errors=[]
    with torch.no_grad():
        for part in np.array_split(ids,max(1,len(ids)//512)):
            q=torch.tensor(source[part].copy(),device='cuda:0');errors.append((speed(q)-physical(q)).abs().cpu().numpy())
    errors=np.concatenate(errors);write(out/'bank_speed_audit.json',dict(rows=len(errors),mae=float(errors.mean()),p95=float(np.quantile(errors,.95)),maximum=float(errors.max()),evaluation_rows_read=False))
    write(out/'complete.json',dict(complete=True,smoke=a.smoke,speed_steps=steps,bank_anchors=manifest['anchors'],elapsed_seconds=time.monotonic()-started,
        stages=dict(relabel_seconds=relabel_seconds,speed_seconds=speed_seconds,bank_seconds=manifest['elapsed_seconds']),evaluation_rows_read=False))
    print('PREPARATION_COMPLETE',time.monotonic()-started,flush=True)
if __name__=='__main__':main()
