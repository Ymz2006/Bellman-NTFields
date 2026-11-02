"""Read-only: distribution of a checkpoint's predicted speed 1/|d tau/d x0| binned by the ground-truth speed S*.

Two sample sets: (train) the x0 of 100,000 training pairs with their own partner x1; (fresh) 100,000 new uniform
configurations with the grasp point in the sampling box, each paired with an independent uniform goal in the box.
S* = clip(d, offset, margin) / margin with the data's label (env: obstacle distance only), as in data.py; the
trainer's target s' = s^2 (2 - s)^2 (alpha 1) is reported per bin for reference.
"""
import argparse,importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,'/clean/experiments/cabinet')
from geometry import scene_boxes

EDGES=[0,.1001,.2,.3,.4,.5,.6,.7,.8,.9,.9999,1.01]
NAMES=['0.1 (floor)','0.1-0.2','0.2-0.3','0.3-0.4','0.4-0.5','0.5-0.6','0.6-0.7','0.7-0.8','0.8-0.9','0.9-1','1 (free)']


def load(name):
    spec=importlib.util.spec_from_file_location('ntrl_cabinet_'+name,HERE/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--variant',required=True);p.add_argument('--coords',required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--scene',type=Path,required=True);p.add_argument('--metric-scale',type=float)
    p.add_argument('--n',type=int,default=100000);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    FIELD,DATA=load('field'),load('data');dev='cuda:0';done=json.loads((a.data/'complete.json').read_text())
    field=FIELD.load_metric(a.checkpoint,dev,a.variant,a.coords,a.metric_scale);unit=torch.tensor(FIELD.units(a.coords),device=dev)
    C=DATA.Clearance(a.scene,done['label'],dev);box,cup,_=scene_boxes(a.scene);box_t=torch.tensor(box,device=dev);cup_t=torch.tensor(cup,device=dev)
    bound=torch.tensor(done['domain_bound'],device=dev);margin,offset=done['margin'],done['offset'];g=torch.Generator(device=dev).manual_seed(7)

    def predict(x):
        out=[]
        for i in range(0,len(x),5000):
            with torch.enable_grad():
                tau,_,xp=field.net.out(x[i:i+5000]);gr=torch.autograd.grad(tau.sum(),xp)[0][:,:6]
            out.append(1/gr.norm(dim=1).clamp_min(1e-9))
        return torch.cat(out)

    def in_box(n):
        rows=[]
        while sum(len(r) for r in rows)<n:
            q=(torch.rand(200000,6,device=dev,generator=g)-.5)*2*bound;pos=C(q*unit,positions=True)[1]
            rows.append(q[((pos>=box_t[:,0])&(pos<=box_t[:,1])).all(1)])
        return torch.cat(rows)[:n]

    X=np.load(a.data/'sampled_points.npy',mmap_mode='r');idx=np.sort(np.random.default_rng(0).choice(len(X),min(a.n,len(X)),replace=False))
    sets=dict(train=torch.tensor(np.asarray(X[idx]),device=dev))
    x0=in_box(a.n);sets['fresh']=torch.cat([x0,in_box(a.n)],1)
    report={}
    for name,x in sets.items():
        d,pos=C(x[:,:6]*unit,positions=True);s=(d.clamp(offset,margin)/margin);pred=predict(x)
        incup=((pos>=cup_t[:,0])&(pos<=cup_t[:,1])).all(1)
        s,pred,incup=s.cpu().numpy(),pred.cpu().numpy(),incup.cpu().numpy();rows=[]
        for k in range(len(NAMES)):
            m=(s>=EDGES[k])&(s<EDGES[k+1])
            if m.sum()<20:rows.append(dict(bin=NAMES[k],n=int(m.sum())));continue
            t=s[m];rows.append(dict(bin=NAMES[k],n=int(m.sum()),cupboard_share=float(incup[m].mean()),label_mean=float(t.mean()),target_mean=float((t**2*(2-t)**2).mean()),
                pred_quantiles=[float(v) for v in np.quantile(pred[m],[.05,.25,.5,.75,.95])],pred_mean=float(pred[m].mean()),
                fast_share=float((pred[m]>.5).mean()),fast_share_cupboard=float((pred[m&incup]>.5).mean()) if (m&incup).sum()>=20 else None,
                median_cupboard=float(np.median(pred[m&incup])) if (m&incup).sum()>=20 else None,median_outside=float(np.median(pred[m&~incup])) if (m&~incup).sum()>=20 else None))
        report[name]=dict(n=len(s),corr=float(np.corrcoef(s,pred)[0,1]),bins=rows)
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'speed_bins.json').write_text(json.dumps(report,indent=1))
    for name,r in report.items():
        print(f'\n[{name}] n={r["n"]} corr(S*, predicted)={r["corr"]:.3f}')
        print(f'{"S* bin":>12} {"n":>7} {"cupb%":>6} {"target":>7} | predicted speed p5 / p25 / median / p75 / p95')
        for b in r['bins']:
            if 'pred_quantiles' not in b:print(f'{b["bin"]:>12} {b["n"]:>7}');continue
            q=b['pred_quantiles'];f=lambda v:'  -  ' if v is None else f'{v:.2f}'
            print(f'{b["bin"]:>12} {b["n"]:>7} {100*b["cupboard_share"]:>5.1f}% {b["target_mean"]:>7.3f} | '+' / '.join(f'{v:.2f}' for v in q)+f' | >0.5: {100*b["fast_share"]:.0f}% | median cupboard {f(b["median_cupboard"])} outside {f(b["median_outside"])}')


if __name__=='__main__':main()
