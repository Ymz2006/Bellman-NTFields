"""Value fields V(q, g) the guarded planner descends, all in release-normalized joint coordinates.

  bellman   a TravelCost checkpoint (method/network.py; e.g. the cabinet Bellman baseline)
  pnt       a P-NTFields checkpoint (experiments/pntfields_cabinet/field.py, ntrl coordinates inside)
  euclid    the joint-space distance in radians (control: guards without any learned field)
"""
import importlib.util, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'method') not in sys.path: sys.path.insert(0, str(ROOT/'method'))


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class Field:
    def __init__(self, kind, checkpoint=None, device='cuda:0'):
        self.kind = kind
        q_range = torch.tensor([12.5664, 12.5664, 6.2832, 12.5664, 12.5664, 12.5664], device=device)
        if kind == 'bellman':
            from network import load_checkpoint
            net = load_checkpoint(checkpoint, device); self.f = lambda q, g: net(q, g)[:, 0]
        elif kind == 'pnt':
            field = _module('pntfields_cabinet_field', ROOT/'experiments/pntfields_cabinet/field.py')
            net = field.load_pnt(checkpoint, device, 'ntrl'); self.f = lambda q, g: net(q, g).reshape(-1)
        elif kind == 'euclid':
            self.f = lambda q, g: torch.linalg.vector_norm((q-g)*q_range, dim=1)
        else: raise ValueError(kind)

    @torch.no_grad()
    def value(self, q, g, chunk=131072):
        return torch.cat([self.f(q[s:s+chunk], g[s:s+chunk]) for s in range(0, len(q), chunk)])

    def gradient(self, q, g):
        """dV/dq (B, 6) and V (B,)."""
        with torch.enable_grad():
            x = q.detach().requires_grad_(True); v = self.f(x, g.detach())
            return torch.autograd.grad(v.sum(), x)[0].detach(), v.detach()
