"""A P-NTFields checkpoint (baselines/P-NTFields, models/model_res_sigmoid.py) as a travel-time field, and the
method's own planner.

P-NTFields predicts the factorized time tau(x0, x1); travel time T = |x1 - x0| / tau (Model.TravelTimes). The
field takes release-normalized q and goal (what the native RK4 and the release MPPI pass) and feeds the network
x = q_release * q_range / unit, the training coordinates, as experiments/ntrl_cabinet/field.py does.
plan() is test/arm_plan.py's bidirectional gradient step x <- x + beta * Gradient(x) (Model.Gradient, i.e.
S^2 grad T, paper Eq. 12) until |x_s - x_g| < dg or 300 iterations, batched with the per-query norm
(Model.Gradient takes torch.norm over the whole batch, which is the per-query norm for a batch of one).
"""
import importlib.util,math,pickle,sys,types
from pathlib import Path
import numpy as np
import torch
from torch import nn

PNT=Path('/clean/baselines/P-NTFields')
NORM=Path('/repo/configs/joint_normalization.npz')


def pnt_module():
    """models/model_res_sigmoid.py, loaded from its file with P-NTFields' own `models` package (the runtime's
    PYTHONPATH has another `models`); sys.modules is restored afterwards."""
    saved={k:sys.modules.get(k) for k in ('models','models.data_mlp','pickle5')}
    package=types.ModuleType('models');package.__path__=[str(PNT/'models')];sys.modules['models']=package;sys.modules['pickle5']=pickle
    try:
        spec=importlib.util.spec_from_file_location('models.data_mlp',PNT/'models/data_mlp.py');db=importlib.util.module_from_spec(spec);sys.modules['models.data_mlp']=db;spec.loader.exec_module(db);package.data_mlp=db
        spec=importlib.util.spec_from_file_location('pntfields_model_res_sigmoid',PNT/'models/model_res_sigmoid.py');md=importlib.util.module_from_spec(spec);spec.loader.exec_module(md)
    finally:
        for k,v in saved.items():
            if v is None:sys.modules.pop(k,None)
            else:sys.modules[k]=v
    assert Path(md.db.__file__).resolve()==(PNT/'models/data_mlp.py').resolve()
    return md


def units(coords):
    n=np.load(NORM);return (n['q_range'] if coords=='release' else np.full(6,2*math.pi)).astype(np.float32)


def travel_time(net,coords):
    """Model.TravelTimes: |x1 - x0| / tau, with x the coordinates NN.out differentiates against."""
    tau,x=net.out(coords)
    D=x[:,6:]-x[:,:6]
    return torch.sqrt((D*D).sum(1,keepdim=True))/tau,x


class _Value(torch.autograd.Function):
    """T with its exact input gradient (NN.out detaches its input internally)."""

    @staticmethod
    def forward(ctx,coords,net):
        with torch.enable_grad():
            t,x=travel_time(net,coords)
            gradient=torch.autograd.grad(t.sum(),x)[0]
        ctx.save_for_backward(gradient)
        return t.detach()

    @staticmethod
    def backward(ctx,grad_output):
        (gradient,)=ctx.saved_tensors
        return grad_output*gradient,None


class TimeNet:
    """`.out(coords) -> (T, coords)` in training coordinates, the interface field.ntrl_mppi of ntrl_cabinet costs with."""
    def __init__(self,net):self.net=net
    def out(self,coords):
        t,_=travel_time(self.net,coords)
        return t.detach(),coords


class PNTField(nn.Module):
    def __init__(self,net,coords,device):
        super().__init__();self.pnt=net;self.coords=coords;self.net=TimeNet(net)
        n=np.load(NORM);self.unit=units(coords)
        self.register_buffer('k',torch.tensor(np.tile(n['q_range']/self.unit,2),dtype=torch.float32,device=device))

    def forward(self,q,goal):
        coords=torch.cat((q,goal),dim=1)*self.k
        if torch.is_grad_enabled() and (q.requires_grad or goal.requires_grad):
            return _Value.apply(coords,self.pnt)
        return self.net.out(coords)[0]


def load_pnt(path,device='cuda:0',coords='ntrl'):
    md=pnt_module();payload=torch.load(path,map_location=device,weights_only=False)
    net=md.NN(device,6,payload['B_state_dict'])
    net.load_state_dict(payload['model_state_dict'],strict=True)
    net.to(torch.device(device)).float().eval()
    for parameter in net.parameters():parameter.requires_grad_(False)
    return PNTField(net,coords,device).eval()


def gradient_rows(net,xp):
    """Model.Gradient with the normalization per query."""
    with torch.enable_grad():
        tau,x=net.out(xp)
        dtau=torch.autograd.grad(tau,x,torch.ones_like(tau),only_inputs=True)[0]
    D=x[:,6:]-x[:,:6];T0=torch.sqrt((D*D).sum(1,keepdim=True));t=tau[:,:1];T3=t**2
    g1=-(D/(T0*t)-T0/T3*dtau[:,6:]);g0=-(-D/(T0*t)-T0/T3*dtau[:,:6])
    g1=g1/(g1*g1).sum(1,keepdim=True);g0=g0/(g0*g0).sum(1,keepdim=True)
    return torch.cat((g0,g1),1).detach()


@torch.no_grad()
def plan(field,suite,count,beta=.015,dg=.03,iters=300,device='cuda:0'):
    """test/arm_plan.py in the training coordinates; path = start branch + reversed goal branch, in radians,
    with the exact start and goal as its end points (the scorer judges every segment, including the last gap)."""
    unit=field.unit;start=suite['starts_rad'][:count]/unit;goal=suite['goals_rad'][:count]/unit
    xp=torch.tensor(np.concatenate([start,goal],1),dtype=torch.float32,device=device)
    fwd=[xp[:,:6].clone()];bwd=[xp[:,6:].clone()];steps=torch.zeros(count,dtype=torch.long,device=device)
    dist=torch.linalg.vector_norm(xp[:,6:]-xp[:,:6],dim=1);active=dist>dg;it=0
    while active.any():
        ids=torch.nonzero(active).squeeze(1)
        xp[ids]=xp[ids]+beta*gradient_rows(field.pnt,xp[ids].clone())
        fwd.append(xp[:,:6].clone());bwd.append(xp[:,6:].clone());steps[ids]+=1
        dist=torch.linalg.vector_norm(xp[:,6:]-xp[:,:6],dim=1);active&=dist>dg;it+=1
        if it>iters:break
    F=torch.stack(fwd,1).cpu().numpy();B=torch.stack(bwd,1).cpu().numpy();steps=steps.cpu().numpy();converged=(dist<=dg).cpu().numpy()
    result=np.empty(count,dtype=object)
    for i in range(count):
        rad=np.concatenate([F[i,:steps[i]+1],B[i,:steps[i]+1][::-1]]).astype(np.float64)*unit
        rad[0]=suite['starts_rad'][i];rad[-1]=suite['goals_rad'][i]
        result[i]=rad.astype(np.float32)
    return result,converged
