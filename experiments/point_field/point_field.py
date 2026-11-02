"""Workspace point-distance field and the hierarchical arm clearance built on it.

The release clearance of a configuration is a minimum over the arm's collision spheres of
D(c_i) - r_i, where D is the distance from a point to the nearest scene cuboid
(experiments/face_speed/clearance.py point_aabb). The sphere radius is only an offset, so the one
function the scene has to supply is D on R^3. This module learns that function:

  PointField    D_theta(p): Fourier features of the normalized point, an MLP, one output, the signed
                distance to the union of the cuboids; `distance` clamps it at 0, which is exactly the
                release's unsigned point-to-box distance outside the boxes and 0 inside.
  Hierarchical  q -> forward kinematics -> sphere centers c_i (exact) -> D_theta(c_i) - r_i -> min over the
                spheres (shoulder spheres excluded) = environment clearance; the self term 0.5 d_self is the
                exact sphere-sphere distance with the release mask (no learning). Body clearance and speed as
                the release label: min(d_env, 0.5 d_self), speed clip(d, 0.005, 0.05)/0.05.

Labels for D are exact and cheap (scene_sdf), so training draws fresh points every step: sphere centers of
configurations (the FK pool's training rows, optionally also uniform joint draws), the same centers jittered,
and uniform points in the workspace box.
"""
import math
import torch
from torch import nn
from experiments.face_speed import robot as R
from experiments.face_speed.clearance import scene_cuboids, MARGIN, OFFSET, SELF_SCALE


def scene_sdf(p, lo, hi):
    """Signed distance (N,) from points (N, 3) to the union of the cuboids lo/hi (K, 3): exact outside, the
    union's min of per-box SDFs inside. Its positive part equals min_k point_aabb(p, lo_k, hi_k)."""
    c, h = (lo+hi)/2, (hi-lo)/2
    q = (p[:, None]-c).abs()-h                                                      # (N, K, 3)
    outside = torch.linalg.vector_norm(q.clamp_min(0), dim=-1); inside = q.amax(-1).clamp_max(0)
    return (outside+inside).amin(1)


class PointField(nn.Module):
    def __init__(self, lo, hi, width=256, depth=5, features=128, sigma=8., seed=20261025):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer('box_lo', torch.as_tensor(lo, dtype=torch.float32)); self.register_buffer('box_hi', torch.as_tensor(hi, dtype=torch.float32))
        self.register_buffer('B', torch.randn(3, features, generator=g)*sigma)
        layers, n = [], 3+2*features
        for _ in range(depth): layers += [nn.Linear(n, width), nn.SiLU()]; n = width
        self.mlp = nn.Sequential(*layers, nn.Linear(n, 1))
        self.config = dict(lo=list(map(float, lo)), hi=list(map(float, hi)), width=width, depth=depth, features=features, sigma=sigma, seed=seed)

    def forward(self, p):
        """Signed distance (...,) of points (..., 3) in world metres."""
        x = 2*(p-self.box_lo)/(self.box_hi-self.box_lo)-1; z = x@self.B
        return self.mlp(torch.cat([x, torch.sin(math.pi*z), torch.cos(math.pi*z)], -1))[..., 0]

    def distance(self, p):
        return self(p).clamp_min(0)


class Hierarchical(nn.Module):
    """Arm clearance from a point-distance function (a PointField, or None for the exact scene distance)."""
    def __init__(self, scene, field=None, device='cuda:0'):
        super().__init__()
        self.field = field; self.robot = R.Robot().to(device)
        self.lo, self.hi, _ = scene_cuboids(scene, device)
        keep = torch.ones(len(self.robot.sphere_radii), dtype=torch.bool); keep[list(R.IGNORED_SPHERES)] = False
        self.register_buffer('keep', keep.to(device))

    def point_distance(self, p):
        if self.field is None: return scene_sdf(p.reshape(-1, 3), self.lo, self.hi).clamp_min(0).reshape(p.shape[:-1])
        return self.field.distance(p)

    def components(self, q_norm):
        """Per-sphere environment clearances (B, S_kept) and the exact self term 0.5 d_self (B,)."""
        spheres = self.robot.link_spheres(self.robot.radians(q_norm))
        c, r = spheres[:, self.keep, :3], spheres[:, self.keep, 3]
        return self.point_distance(c)-r, SELF_SCALE*self.robot.self_distance(spheres)

    def clearances(self, q_norm):
        """Environment clearance, self term and body clearance, each (B,)."""
        per, d_self = self.components(q_norm); env = per.amin(1)
        return env, d_self, torch.minimum(env, d_self)

    def forward(self, q_norm):
        """Body speed in [0.1, 1], the release label's formula."""
        return speed_of(self.clearances(q_norm)[2])


def speed_of(d):
    return d.clamp(OFFSET, MARGIN)/MARGIN


def load_field(path, device):
    payload = torch.load(path, map_location=device, weights_only=False); c = payload['config']
    field = PointField(c['lo'], c['hi'], c['width'], c['depth'], c['features'], c['sigma'], c['seed']).to(device)
    field.load_state_dict(payload['state_dict']); return field.eval().requires_grad_(False)
