"""A data.py/train.py metric NTFields checkpoint as a travel-time field, and ntrl-demo's own MPPI.

Same as experiments/cabinet/metric/field.py, generalized to the two ntrl-demo network variants and the two
coordinate scales of data.py. The field takes release-normalized q and goal (what the release planners
and the native RK4 pass) and feeds the network x = q_release * q_range / unit, i.e. the training
coordinates; ntrl_mppi plans directly in the training coordinates, as ntrl-demo tests/arm_plan_stat.py does.
"""
import importlib.util,math
from pathlib import Path
import numpy as np
import torch
from torch import nn

NTRL=Path('/clean/baselines/ntrl-demo')
NORM=Path('/repo/configs/joint_normalization.npz')


SCALE_LINE={'metric':'x = 0.2*(torch.sum(x,dim=1,keepdim=True))','metric_arm':'x = 0.1*(torch.sum(x,dim=1,keepdim=True))'}


def network_source(variant,metric_scale=None):
    """ntrl-demo's model_network_metric.py; metric_scale replaces the factor of its L1-of-L-inf metric (paper Eq. 10 has none)."""
    src=(NTRL/f'models/{variant}/model_network_metric.py').read_text()
    if metric_scale is not None:
        old=SCALE_LINE[variant];assert src.count(old)==1;src=src.replace(old,f'x = {metric_scale!r}*(torch.sum(x,dim=1,keepdim=True))')
    return src


def network_class(variant,metric_scale=None):
    path=NTRL/f'models/{variant}/model_network_metric.py';spec=importlib.util.spec_from_loader(f'ntrl_demo_{variant}_network',loader=None,origin=str(path))
    module=importlib.util.module_from_spec(spec);module.__file__=str(path);exec(compile(network_source(variant,metric_scale),str(path),'exec'),module.__dict__);return module.NN


def units(coords):
    n=np.load(NORM);return (n['q_range'] if coords=='release' else np.full(6,2*math.pi)).astype(np.float32)


class _Value(torch.autograd.Function):
    """tau of NN.out with its exact input gradient (NN.out detaches its input internally)."""

    @staticmethod
    def forward(ctx,coords,net):
        with torch.enable_grad():
            tau,_,xp=net.out(coords)
            gradient=torch.autograd.grad(tau.sum(),xp)[0]
        ctx.save_for_backward(gradient)
        return tau.detach()

    @staticmethod
    def backward(ctx,grad_output):
        (gradient,)=ctx.saved_tensors
        return grad_output*gradient,None


class MetricField(nn.Module):
    def __init__(self,net,coords,device):
        super().__init__();self.net=net;self.coords=coords
        n=np.load(NORM);self.unit=units(coords)
        self.register_buffer('k',torch.tensor(np.tile(n['q_range']/self.unit,2),dtype=torch.float32,device=device))

    def forward(self,q,goal):
        coords=torch.cat((q,goal),dim=1)*self.k
        if torch.is_grad_enabled() and (q.requires_grad or goal.requires_grad):
            return _Value.apply(coords,self.net)
        with torch.no_grad():
            return self.net.out(coords)[0]


def load_metric(path,device='cuda:0',variant='metric',coords='release',metric_scale=None):
    payload=torch.load(path,map_location=device,weights_only=False)
    net=network_class(variant,metric_scale)(device,6,payload['B_state_dict'])
    net.load_state_dict(payload['model_state_dict'],strict=True)
    net.to(torch.device(device)).float().eval()
    for parameter in net.parameters():parameter.requires_grad_(False)
    return MetricField(net,coords,device).eval()


@torch.no_grad()
def ntrl_mppi(field,suite,count,seed,steps=200,sample_num=50,horizon=5,step=.015,temperature=50.,tol=.01,device='cuda:0'):
    """MPPI of ntrl-demo tests/arm_plan_stat.py in the training coordinates, batched over queries (each
    query evolves as in the single-query loop). Defaults are that file's settings: 50 samples, horizon 5,
    200 steps, step 0.015, cost 10 T(first) + T(last), softmax(-50 cost), stop within 0.01 of the goal;
    the goal is appended to every path and the scorer judges the final segment."""
    torch.manual_seed(seed);unit=field.unit
    start=suite['starts_rad'][:count]/unit;goal=suite['goals_rad'][:count]/unit
    xp=torch.tensor(np.concatenate([start,goal],1),dtype=torch.float32,device=device)
    prior=torch.zeros(count,6,device=device);active=torch.ones(count,dtype=torch.bool,device=device)
    paths=[[x.copy()] for x in xp[:,:6].cpu().numpy()];converged=np.zeros(count,bool)
    for _ in range(steps):
        ids=torch.nonzero(active).squeeze(1)
        if not len(ids):break
        cur=xp[ids]
        tmp=cur[:,None,None,:].repeat(1,sample_num,horizon,1)
        dP=step*torch.randn(len(ids),sample_num,1,6,device=device)+step*torch.randn(len(ids),sample_num,horizon,6,device=device)
        dP=dP+2*prior[ids][:,None,None,:]
        dP=dP/(torch.linalg.vector_norm(dP,dim=3,keepdim=True).clamp(min=step)/step)
        tmp[...,:6]=tmp[...,:6]+torch.cumsum(dP,dim=2)
        cost=field.net.out(tmp[:,:,[0,-1],:].reshape(-1,12))[0][:,0].reshape(len(ids),sample_num,2)
        cost=10*cost[...,0]+cost[...,1]
        weight=torch.softmax(-temperature*cost,dim=1)
        update=torch.einsum('bs,bsd->bd',weight,dP[:,:,0,:])
        prior[ids]=update;xp[ids,:6]=cur[:,:6]+update
        new=xp[ids,:6].cpu().numpy();dist=torch.linalg.vector_norm(xp[ids,6:]-xp[ids,:6],dim=1)
        for k,i in enumerate(ids.tolist()):paths[i].append(new[k].copy())
        done=dist<tol;active[ids[done]]=False;converged[ids[done].cpu().numpy()]=True
    result=np.empty(count,dtype=object)
    for i,path in enumerate(paths):
        rad=np.array(path,dtype=np.float64)*unit
        rad=np.concatenate([rad,suite['goals_rad'][i][None]]);rad[0]=suite['starts_rad'][i]
        result[i]=rad.astype(np.float32)
    return result,converged
