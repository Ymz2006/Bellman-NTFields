"""Arm clearance from a hierarchical distance model, and certified (collision-free) joint-space segments.

Hierarchy: q -> exact forward kinematics -> the release collision spheres (centres c_i, padded radii r_i)
-> point distance D(c_i) to the nearest scene cuboid -> sphere clearance D(c_i) - r_i -> min over spheres.
D is either learned (a PointField trained by distance.py) or the exact point-to-box distance (the control).
Self-collision is the exact sphere-sphere distance with the release mask (no learning needed: it depends
on the arm only). All 52 spheres are checked, exactly as the release scorer does
(runtime/vendor/planner/score_cross_suite.py: obstacle clearance >= 5 mm, self clearance >= 2 mm).

Certification (conservative advancement). Rotating joint j at rate |dq_j| moves sphere centre i at most
|dq_j| * dist(c_i, axis_j) <= |dq_j| * R_ij, where R_ij is a configuration-independent bound (the length of the
kinematic chain from joint j's origin to the sphere centre, every piece of which is rigid). Along the straight
joint-space segment q(t) = a + t (b - a), t in [0, 1], sphere i therefore moves at most
t * sum_j R_ij |b_j - a_j|. Joints upstream of both spheres of a self pair move both rigidly, so the pair's
separation changes at most by the joints that move only one of them. Starting at t = 0 the segment is advanced
by the largest dt for which no sphere can lose its clearance surplus (clearance - threshold), until t = 1
(certified) or a surplus is exhausted (not certified). With a learned D, the threshold includes the model's
calibrated over-estimation bound eps, so a certified segment is collision-free whenever the model error is
within eps; the scorer's densified points are a subset of the segment.
"""
import math
from pathlib import Path
import numpy as np
import torch
import yaml
from experiments.face_speed import robot as R
from experiments.point_field.point_field import PointField, scene_sdf

OBSTACLE_MARGIN, SELF_MARGIN = .005, .002          # the release scorer's acceptance thresholds (m)


def scene_boxes(scene, device):
    cubes = yaml.safe_load(Path(scene).read_text())['cuboids']
    c = torch.tensor([x['pose'][:3] for x in cubes], dtype=torch.float32, device=device)
    h = torch.tensor([[d/2 for d in x['dims']] for x in cubes], dtype=torch.float32, device=device)
    return c-h, c+h


def point_box_distance(p, lo, hi):
    """Release point-to-cuboid distance (0 inside), min over cuboids: (..., 3) -> (...)."""
    gap = (lo-p[..., None, :]).clamp_min(0)+(p[..., None, :]-hi).clamp_min(0)
    return torch.linalg.vector_norm(gap, dim=-1).amin(-1)


def load_distance_model(path, device):
    payload = torch.load(path, map_location=device, weights_only=False); c = payload['config']
    field = PointField(c['lo'], c['hi'], c['width'], c['depth'], c['features'], c['sigma'], c['seed']).to(device)
    field.load_state_dict(payload['state_dict']); field.eval().requires_grad_(False)
    return field, payload


class ArmClearance(torch.nn.Module):
    """Per-sphere obstacle clearance (learned or exact D) and per-pair self clearance of the UR5e."""
    def __init__(self, scene, distance=None, eps=0., device='cuda:0'):
        super().__init__()
        self.device = device; self.robot = R.Robot().to(device)
        self.lo, self.hi = scene_boxes(scene, device)
        self.field = distance if distance is None or isinstance(distance, torch.nn.Module) else load_distance_model(distance, device)[0]
        self.eps = float(eps); self.mode = 'exact' if self.field is None else 'learned'
        S = len(self.robot.sphere_radii)
        i, k = torch.triu_indices(S, S, 1)
        keep = ~self.robot.self_mask.cpu()[i, k]
        self.register_buffer('pair_i', i[keep].to(device)); self.register_buffer('pair_k', k[keep].to(device))
        self.register_buffer('reach', self._reach().to(device))                                  # (S, 6) R_ij in m/rad
        up = self.reach > 0                                                                       # joint j moves sphere i
        only = up[self.pair_i] ^ up[self.pair_k]                                                  # joints moving one sphere of a pair
        self.register_buffer('pair_reach', torch.where(only, torch.maximum(self.reach[self.pair_i], self.reach[self.pair_k]), torch.zeros_like(self.reach[self.pair_i])))

    def _reach(self):
        """R_ij: chain-length bound on the distance from sphere i's centre to joint j's axis, valid at every q."""
        kin = self.robot.kinematics; dev = self.robot.sphere_centers.device
        t = kin.transforms(torch.zeros(1, 6, device=dev), kin.links)
        child_of_joint = {}
        for link, (n, parent, idx) in kin.joint_of.items():
            if idx >= 0: child_of_joint[idx] = link
        origin = {j: t[child_of_joint[j]][0, :3, 3] for j in range(6)}

        def upstream(link):
            joints = []
            while link != 'base_link':
                n, parent, idx = kin.joint_of[link]
                if idx >= 0: joints.append(idx)
                link = parent
            return joints[::-1]                                                                   # base -> tip
        sph = self.robot.link_spheres(torch.zeros(1, 6, device=dev))[0, :, :3]
        reach = torch.zeros(len(sph), 6)
        for s in range(len(sph)):
            chain = upstream(self.robot.collision_links[int(self.robot.sphere_link[s])])
            if not chain: continue
            tail = float(torch.linalg.vector_norm(sph[s]-origin[chain[-1]]))
            acc = tail
            for pos in range(len(chain)-1, -1, -1):
                if pos < len(chain)-1: acc += float(torch.linalg.vector_norm(origin[chain[pos+1]]-origin[chain[pos]]))
                reach[s, chain[pos]] = acc+1e-4
        return reach

    def spheres(self, q_norm):
        return self.robot.link_spheres(self.robot.radians(q_norm))

    def point_distance(self, p):
        if self.field is None: return point_box_distance(p, self.lo, self.hi)
        if torch.is_grad_enabled() or p.numel() <= 3*262144: return self.field(p).clamp_min(0)
        flat = p.reshape(-1, 3)                                                                   # bounded activation memory
        return torch.cat([self.field(flat[s:s+262144]) for s in range(0, len(flat), 262144)]).clamp_min(0).reshape(p.shape[:-1])

    def components(self, q_norm):
        """Obstacle clearance per sphere (B, S) and self clearance per unmasked pair (B, P), in metres."""
        s = self.spheres(q_norm); c, r = s[..., :3], s[..., 3]
        env = self.point_distance(c)-r
        d = torch.linalg.vector_norm(c[:, self.pair_i]-c[:, self.pair_k], dim=-1)-r[:, self.pair_i]-r[:, self.pair_k]
        return env, d

    def margins(self, q_norm):
        """Surplus over the guard thresholds: obstacle (5 mm + eps) and self (2 mm); min over pieces (B,)."""
        env, d = self.components(q_norm)
        return torch.minimum(env.amin(1)-OBSTACLE_MARGIN-self.eps, d.amin(1)-SELF_MARGIN)

    def clearance(self, q_norm):
        """Body clearance min(obstacle, self) (B,), metres (no thresholds)."""
        env, d = self.components(q_norm)
        return torch.minimum(env.amin(1), d.amin(1))

    @torch.no_grad()
    def certify(self, a, b, chunk=65536, max_iter=48, min_dt=1e-3):
        """Conservative advancement along a -> b (normalized joint coordinates, (B, 6)). Returns certified (B,)
        bool and the number of clearance evaluations used per segment (B,)."""
        out = torch.zeros(len(a), dtype=torch.bool, device=a.device); evals = torch.zeros(len(a), dtype=torch.long, device=a.device)
        for s in range(0, len(a), chunk):
            out[s:s+chunk], evals[s:s+chunk] = self._certify(a[s:s+chunk], b[s:s+chunk], max_iter, min_dt)
        return out, evals

    def _certify(self, a, b, max_iter, min_dt):
        n = len(a); delta = (b-a); rad = (delta*self.robot.q_range).abs()                          # (n, 6) radians
        move_env = rad@self.reach.T                                                               # (n, S) metres per unit t
        move_self = rad@self.pair_reach.T                                                         # (n, P)
        t = torch.zeros(n, device=a.device); done = torch.zeros(n, dtype=torch.bool, device=a.device)
        ok = torch.zeros(n, dtype=torch.bool, device=a.device); evals = torch.zeros(n, dtype=torch.long, device=a.device)
        active = torch.arange(n, device=a.device)
        for _ in range(max_iter):
            if not len(active): break
            q = a[active]+t[active, None]*delta[active]
            env, d = self.components(q)
            se = env-OBSTACLE_MARGIN-self.eps; ss = d-SELF_MARGIN
            evals[active] += 1
            bad = (se.amin(1) < 0) | (ss.amin(1) < 0)
            step = torch.minimum((se/move_env[active].clamp_min(1e-9)).amin(1), (ss/move_self[active].clamp_min(1e-9)).amin(1))
            new_t = t[active]+step
            finished = ~bad & (new_t >= 1)
            stuck = ~bad & ~finished & (step < min_dt)
            ok[active[finished]] = True
            t[active] = new_t.clamp(max=1.)
            keep = ~(bad | finished | stuck)
            active = active[keep]
        return ok, evals

    @torch.no_grad()
    def dense_margin(self, path_norm, step_rad=.001, chunk=65536):
        """Minimum guard surplus over the scorer's densified points of a path (T, 6) in normalized coordinates."""
        rad = path_norm*self.robot.q_range
        pts = [path_norm[:1]]
        for k in range(1, len(path_norm)):
            m = max(1, int(math.ceil(float((rad[k]-rad[k-1]).abs().max())/step_rad)))
            w = torch.arange(1, m+1, device=path_norm.device, dtype=path_norm.dtype)[:, None]/m
            pts.append(path_norm[k-1]+(path_norm[k]-path_norm[k-1])*w)
        pts = torch.cat(pts)
        return min(float(self.margins(pts[s:s+chunk]).min()) for s in range(0, len(pts), chunk))
