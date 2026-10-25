"""Fixed readouts for a bounded 20k screen; no learned parameters."""
import torch
SCREEN_NORMS = ('lp15n','lp3n','lp4n','mix_l1_25','mix_l1_50','mix_l4_50')

def normalized_lp(x, p):
    # Smooth |x_i| at epsilon=1e-6; subtract the same zero-vector formula.
    # d^(1/2-1/p) matches L2 on dense equal-magnitude vectors away from epsilon.
    eps = 1e-6
    z = (x.square()+eps**2).pow(p/2)
    zero = torch.full_like(x, eps**2).pow(p/2)
    return x.shape[-1]**(0.5-1/p)*(z.sum(-1,keepdim=True).pow(1/p)-zero.sum(-1,keepdim=True).pow(1/p))

def screen_distance(x, name):
    if name in ('lp15n','lp3n','lp4n'):
        return normalized_lp(x, {'lp15n':1.5,'lp3n':3.,'lp4n':4.}[name])
    l2=(x.square().sum(-1,keepdim=True)+1e-12).sqrt()-1e-6
    if name in ('mix_l1_25','mix_l1_50'):
        a=0.25 if name=='mix_l1_25' else 0.5
        return (1-a)*l2+a*x.abs().sum(-1,keepdim=True)/x.shape[-1]**0.5
    if name=='mix_l4_50':
        return 0.5*l2+0.5*normalized_lp(x,4.)
    raise ValueError(name)
