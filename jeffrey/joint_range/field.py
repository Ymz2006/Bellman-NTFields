"""ntrl-demo metric checkpoints behind an affine input map, and ntrl-demo's MPPI planned in release coordinates.

The planners (native RK4, the release MPPI and ntrl-demo's own MPPI) all work in release-normalized coordinates
x = q / q_range in both tests, with their unchanged settings, so their steps, noise, joint bounds and goal tolerance
are the same in radians whatever coordinates the network was trained in. Only the network input is remapped:
    network input y = (q - center) / unit = x * k - b,   k = q_range / unit,   b = center / unit
For the full-range data (center 0, unit = q_range) the map is the identity and is skipped, so the field and
ntrl_mppi below reproduce experiments/ntrl_cabinet/field.py (MetricField, ntrl_mppi with --coords release) exactly.
"""
import importlib.util
from pathlib import Path
import numpy as np
import torch
from torch import nn
_spec = importlib.util.spec_from_file_location('ntrl_cabinet_field', '/clean/experiments/ntrl_cabinet/field.py')
F = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(F)
NORM = Path('/repo/configs/joint_normalization.npz')


class AffineField(nn.Module):
    def __init__(self, net, center, unit, device):
        super().__init__(); self.net = net
        R = np.load(NORM)['q_range'].astype(np.float64); center = np.asarray(center, np.float64); unit = np.asarray(unit, np.float64)
        self.identity = bool(np.all(center == 0) and np.allclose(unit, R, rtol=0, atol=1e-6))
        self.unit = R.astype(np.float32)                    # planning unit: release coordinates
        self.register_buffer('k', torch.tensor(np.tile(R/unit, 2), dtype=torch.float32, device=device))
        self.register_buffer('b', torch.tensor(np.tile(center/unit, 2), dtype=torch.float32, device=device))

    def to_net(self, coords):
        return coords if self.identity else coords*self.k-self.b

    def forward(self, q, goal):
        coords = self.to_net(torch.cat((q, goal), dim=1))
        if torch.is_grad_enabled() and (q.requires_grad or goal.requires_grad):
            return F._Value.apply(coords, self.net)
        with torch.no_grad():
            return self.net.out(coords)[0]


def load_affine(path, center, unit, device='cuda:0', variant='metric', metric_scale=None):
    payload = torch.load(path, map_location=device, weights_only=False)
    net = F.network_class(variant, metric_scale)(device, 6, payload['B_state_dict'])
    net.load_state_dict(payload['model_state_dict'], strict=True)
    net.to(torch.device(device)).float().eval()
    for parameter in net.parameters(): parameter.requires_grad_(False)
    return AffineField(net, center, unit, device).eval()


@torch.no_grad()
def ntrl_mppi(field, suite, count, seed, steps=200, sample_num=50, horizon=5, step=.015, temperature=50., tol=.01, device='cuda:0'):
    """experiments/ntrl_cabinet/field.py ntrl_mppi line for line, planned in release coordinates (field.unit) and
    costed by the network through the field's input map."""
    torch.manual_seed(seed); unit = field.unit
    start = suite['starts_rad'][:count]/unit; goal = suite['goals_rad'][:count]/unit
    xp = torch.tensor(np.concatenate([start, goal], 1), dtype=torch.float32, device=device)
    prior = torch.zeros(count, 6, device=device); active = torch.ones(count, dtype=torch.bool, device=device)
    paths = [[x.copy()] for x in xp[:, :6].cpu().numpy()]; converged = np.zeros(count, bool)
    for _ in range(steps):
        ids = torch.nonzero(active).squeeze(1)
        if not len(ids): break
        cur = xp[ids]
        tmp = cur[:, None, None, :].repeat(1, sample_num, horizon, 1)
        dP = step*torch.randn(len(ids), sample_num, 1, 6, device=device)+step*torch.randn(len(ids), sample_num, horizon, 6, device=device)
        dP = dP+2*prior[ids][:, None, None, :]
        dP = dP/(torch.linalg.vector_norm(dP, dim=3, keepdim=True).clamp(min=step)/step)
        tmp[..., :6] = tmp[..., :6]+torch.cumsum(dP, dim=2)
        cost = field.net.out(field.to_net(tmp[:, :, [0, -1], :].reshape(-1, 12)))[0][:, 0].reshape(len(ids), sample_num, 2)
        cost = 10*cost[..., 0]+cost[..., 1]
        weight = torch.softmax(-temperature*cost, dim=1)
        update = torch.einsum('bs,bsd->bd', weight, dP[:, :, 0, :])
        prior[ids] = update; xp[ids, :6] = cur[:, :6]+update
        new = xp[ids, :6].cpu().numpy(); dist = torch.linalg.vector_norm(xp[ids, 6:]-xp[ids, :6], dim=1)
        for k, i in enumerate(ids.tolist()): paths[i].append(new[k].copy())
        done = dist < tol; active[ids[done]] = False; converged[ids[done].cpu().numpy()] = True
    result = np.empty(count, dtype=object)
    for i, path in enumerate(paths):
        rad = np.array(path, dtype=np.float64)*unit
        rad = np.concatenate([rad, suite['goals_rad'][i][None]]); rad[0] = suite['starts_rad'][i]
        result[i] = rad.astype(np.float32)
    return result, converged
