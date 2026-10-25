"""Metric NTFields (ntrl-demo) as a travel-time field T(q, goal), and that method's own MPPI.

The vendored, unmodified network (ntrl_demo/models/metric/model_network_metric.py) is loaded by file
path, so ntrl-demo's `models` package never shadows the frozen runtime's. Its inputs are the release
joint normalization (q - center) / range, the coordinates it was trained in (metric/data.py).
"""
import importlib.util
from pathlib import Path
import numpy as np
import torch
from torch import nn

VENDOR=Path(__file__).resolve().parent/'ntrl_demo'
NORM=Path('/repo/configs/joint_normalization.npz')


def network_class():
    spec=importlib.util.spec_from_file_location('ntrl_demo_model_network_metric',VENDOR/'models/metric/model_network_metric.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module.NN


class _Value(torch.autograd.Function):
    """tau of NN.out with its exact input gradient. NN.out detaches its input internally
    (coords.clone().detach().requires_grad_), so the gradient is taken inside and passed back."""

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
    def __init__(self,net):
        super().__init__();self.net=net

    def forward(self,q,goal):
        coords=torch.cat((q,goal),dim=1)
        if torch.is_grad_enabled() and (q.requires_grad or goal.requires_grad):
            return _Value.apply(coords,self.net)
        with torch.no_grad():
            return self.net.out(coords)[0]


def load_metric(path,device='cuda:0'):
    """Model.load of ntrl-demo's model_train_metric.py: NN(device, dim, B), strict state dict."""
    payload=torch.load(path,map_location=device,weights_only=False)
    net=network_class()(device,6,payload['B_state_dict'])
    net.load_state_dict(payload['model_state_dict'],strict=True)
    net.to(torch.device(device)).float().eval()
    for parameter in net.parameters():parameter.requires_grad_(False)
    return MetricField(net).eval()


@torch.no_grad()
def ntrl_mppi(field,suite,count,seed,steps=200,sample_num=50,horizon=5,step=.015,temperature=50.,device='cuda:0'):
    """MPPI of ntrl-demo tests/arm_plan_stat.py, batched over queries (each query evolves exactly as
    in the single-query loop): 50 samples, horizon 5, 200 steps, step 0.015 (input coordinates),
    cost 10 T(first) + T(last), softmax(-50 cost), stop when the state is within 0.01 of the goal.
    As there, the goal is appended to every path; the scorer judges the final segment."""
    torch.manual_seed(seed)
    xp=torch.cat([torch.tensor(suite['starts_norm'][:count]),torch.tensor(suite['goals_norm'][:count])],1).to(device)
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
        done=dist<.01;active[ids[done]]=False;converged[ids[done].cpu().numpy()]=True
    norm=np.load(NORM);result=np.empty(count,dtype=object)
    for i,path in enumerate(paths):
        path.append(suite['goals_norm'][i].copy())
        rad=np.array(path)*norm['q_range']+norm['q_center'];rad[0]=suite['starts_rad'][i];rad[-1]=suite['goals_rad'][i]
        result[i]=rad.astype(np.float32)
    return result,converged
