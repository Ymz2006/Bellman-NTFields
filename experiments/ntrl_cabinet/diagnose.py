"""Read-only fit check of a train.py checkpoint on training pairs: does the field reproduce its speed labels?

The trainer fits |d tau / d x0| to 1 / s' with s' = alpha s^2 (2 - s)^2 + 1 - alpha (model_train_metric.py), so the
field's speed at x0 is 1 / |d tau / d x0|. Reports, for pairs with the grasp point of x0 inside / outside the
cupboard's outer box, the mean label speed, the mean field speed, their MAE and correlation, and the share of
pairs whose label is at the floor while the field speed is above 0.5 (walls the field does not see).
"""
import argparse,importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,'/clean/experiments/cabinet')
from geometry import scene_boxes


def load(name):
    spec=importlib.util.spec_from_file_location('ntrl_cabinet_'+name,HERE/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--variant',required=True);p.add_argument('--coords',required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--scene',type=Path,required=True);p.add_argument('--n',type=int,default=50000);p.add_argument('--metric-scale',type=float);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();FIELD,DATA=load('field'),load('data');dev='cuda:0'
    field=FIELD.load_metric(a.checkpoint,dev,a.variant,a.coords,a.metric_scale);alpha=1.025 if a.variant=='metric' else 1.0
    X=np.load(a.data/'sampled_points.npy',mmap_mode='r');Y=np.load(a.data/'speed.npy',mmap_mode='r')
    idx=np.random.default_rng(0).choice(len(X),min(a.n,len(X)),replace=False);idx.sort()
    x=torch.tensor(np.asarray(X[idx]),device=dev);s=torch.tensor(np.asarray(Y[idx])[:,0],device=dev)
    target=alpha*s**2*(2-s)**2+1-alpha
    unit=torch.tensor(FIELD.units(a.coords),device=dev);C=DATA.Clearance(a.scene,'env',dev);_,cup,_=scene_boxes(a.scene);cup=torch.tensor(cup,device=dev)
    pred=[];inside=[]
    for i in range(0,len(x),5000):
        xb=x[i:i+5000]
        with torch.enable_grad():
            tau,_,xp=field.net.out(xb);g=torch.autograd.grad(tau.sum(),xp)[0][:,:6]
        pred.append(1/g.norm(dim=1).clamp_min(1e-6));pos=C(xb[:,:6]*unit,positions=True)[1];inside.append(((pos>=cup[:,0])&(pos<=cup[:,1])).all(1))
    pred=torch.cat(pred);inside=torch.cat(inside);out={}
    for name,m in (('all',torch.ones_like(inside)),('cupboard',inside),('outside',~inside)):
        if m.sum()<10:continue
        t,q=target[m].cpu().numpy(),pred[m].cpu().numpy()
        out[name]=dict(n=int(m.sum()),label_speed=float(t.mean()),field_speed=float(q.mean()),mae=float(np.abs(t-q).mean()),corr=float(np.corrcoef(t,q)[0,1]),
                       floor_but_fast=float(((t<=t.min()+1e-6)&(q>.5)).mean()),
                       label_quantiles=[float(x) for x in np.quantile(t,[.05,.25,.5,.75,.95])],field_quantiles=[float(x) for x in np.quantile(q,[.05,.25,.5,.75,.95])])
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'diagnose.json').write_text(json.dumps(out,indent=1));print(json.dumps(out,indent=1))


if __name__=='__main__':main()
