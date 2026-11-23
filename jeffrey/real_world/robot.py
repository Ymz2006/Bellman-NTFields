"""Collision model for the real-world cupboard scenes: one or two release UR5e + Robotiq 2F-140 arms, each on its own base pose.

Robot geometry = the release's own collision model as jeffrey/lbs_arm/export_robot.py exported it (registered run 91c5e157,
results/registered_20261121T042548Z-91c5e157/robot_model.npz): cuRobo's 53 spheres (six arm links + the gripper, rigid on wrist 3),
the self-collision mask, the grasp point. extract() turns it into the small robot_geometry.npz next to this file (sphere centres in
their link frames, fitted as in jeffrey/lbs_arm/arm.py, error < 2e-5 m), so this folder needs no container.

A scene (scenes/real_world_*.yml) lists `robots:` with base poses (x, y, z, qw, qx, qy, qz; rotations about z only) and axis-aligned
`cuboids:`. A configuration is q (N, 6 A) radians, arm a = q[:, 6 a : 6 a + 6].

    obstacle   per arm, min over its spheres (without the two shoulder spheres, which never move: the release label's rule) and every
               cuboid of |outside distance| - radius                                                      valid if >= 0.005 m
    self       per arm, min over the unmasked sphere pairs of centre distance - radii                       valid if >= 0.002 m
    inter      between arms, min over all sphere pairs (shoulder spheres included: the other arm can hit this arm's base)
                                                                                                            valid if >= 0.005 m
The obstacle and self thresholds are the release scorer's; inter-arm contact is treated like an obstacle.
"""
import math
from pathlib import Path
import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
GEOMETRY = HERE / 'robot_geometry.npz'
EXPORT = 'results/registered_20261121T042548Z-91c5e157/robot_model.npz'
OBS_VALID_M, SELF_VALID_M, INTER_VALID_M = 0.005, 0.002, 0.005
HOME = (0.0, -math.pi / 2, 0.0, -math.pi / 2, 0.0, 0.0)            # UR home: arm straight up
JOINT_NAMES = ('shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint')

# UR5e joint origins (runtime/assets/runtime/robot/ur5e_robotiq_2f_140.urdf, the arm chain): xyz, rpy; axis z
JOINTS = [((0.0, 0.0, 0.1625), (0.0, 0.0, 0.0)),
          ((0.0, 0.0, 0.0), (1.570796327, 0.0, 0.0)),
          ((-0.425, 0.0, 0.0), (0.0, 0.0, 0.0)),
          ((-0.3922, 0.0, 0.1333), (0.0, 0.0, 0.0)),
          ((0.0, -0.0997, -2.044881182297852e-11), (1.570796327, 0.0, 0.0)),
          ((0.0, 0.0996, -2.042830148012698e-11), (1.570796326589793, 3.141592653589793, 3.141592653589793))]


def _rot(axis, a):
    c, s = math.cos(a), math.sin(a)
    if axis == 'x':
        return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float64)
    if axis == 'y':
        return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float64)
    return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float64)


def _origin(xyz, rpy):
    T = torch.eye(4, dtype=torch.float64)
    T[:3, :3] = _rot('z', rpy[2]) @ _rot('y', rpy[1]) @ _rot('x', rpy[0])
    T[:3, 3] = torch.tensor(xyz, dtype=torch.float64)
    return T


def _base_inertia():
    T = torch.eye(4, dtype=torch.float64)
    T[:3, :3] = _rot('z', math.pi)                                   # base_link -> base_link_inertia
    return T


def link_frames(q, O, T0):
    """(N, 6, 4, 4) frames of the six arm links in base_link coordinates at joint angles q (N, 6) rad."""
    n = len(q)
    c, s = torch.cos(q), torch.sin(q)
    Rz = torch.zeros(n, 6, 4, 4, device=q.device, dtype=q.dtype)
    Rz[..., 0, 0], Rz[..., 0, 1], Rz[..., 1, 0], Rz[..., 1, 1] = c, -s, s, c
    Rz[..., 2, 2] = 1
    Rz[..., 3, 3] = 1
    T = T0.expand(n, 4, 4)
    out = []
    for k in range(6):
        T = T @ O[k] @ Rz[:, k]
        out.append(T)
    return torch.stack(out, 1)


def extract(export, out=GEOMETRY):
    """robot_model.npz of export_robot.py -> robot_geometry.npz (link-frame sphere centres, radii, mask, grasp point)."""
    m = np.load(export, allow_pickle=False)
    O = torch.stack([_origin(x, r) for x, r in JOINTS])
    qf = torch.tensor(m['q_fit'], dtype=torch.float64)
    sp = torch.tensor(m['spheres_fit'], dtype=torch.float64)
    frame = torch.tensor(np.minimum(m['link'], 5))                   # gripper links are rigid on wrist 3
    F = link_frames(qf, O, _base_inertia())
    F0 = F[0, frame]
    local = (F0[:, :3, :3].transpose(1, 2) @ (sp[0, :, :3] - F0[:, :3, 3])[:, :, None]).squeeze(-1)
    Fe = F[0, 5]
    ee_local = Fe[:3, :3].T @ (torch.tensor(m['ee_fit'][0], dtype=torch.float64) - Fe[:3, 3])
    Fs = F[:, frame]
    world = (Fs[..., :3, :3] @ local[None, :, :, None]).squeeze(-1) + Fs[..., :3, 3]
    ee = (F[:, 5, :3, :3] @ ee_local[None, :, None]).squeeze(-1) + F[:, 5, :3, 3]
    err = float((world - sp[..., :3]).abs().max()), float((ee - torch.tensor(m['ee_fit'], dtype=torch.float64)).abs().max())
    assert max(err) < 2e-5, err
    np.savez(out, local=local.numpy(), frame=frame.numpy(), radius=m['spheres_fit'][0, :, 3].astype(np.float64), mask=m['mask'],
             ignore=m['ignore'], ee_local=ee_local.numpy(), q_min=m['q_min'], q_max=m['q_max'], link=m['link'], link_names=m['link_names'],
             fit_error=np.array(err), source=np.array(export))
    return err


def yaw_of(wxyz):
    w, x, y, z = wxyz
    assert abs(x) < 1e-9 and abs(y) < 1e-9, 'robot bases may only be rotated about z'
    return 2 * math.atan2(z, w)


class Arms:
    """A fixed set of arm base poses and axis-aligned cuboids. All lengths in metres."""

    def __init__(self, bases, cuboids, device='cpu', dtype=torch.float32, geometry=GEOMETRY):
        g = np.load(geometry, allow_pickle=False)
        self.device, self.dtype = device, dtype
        self.A = len(bases)
        self.O = torch.stack([_origin(x, r) for x, r in JOINTS]).to(device, dtype)
        self.T0 = _base_inertia().to(device, dtype)
        self.frame = torch.tensor(g['frame'], device=device)
        self.local = torch.tensor(g['local'], device=device, dtype=dtype)
        self.radius = torch.tensor(g['radius'], device=device, dtype=dtype)
        self.ee_local = torch.tensor(g['ee_local'], device=device, dtype=dtype)
        self.S = len(self.radius)
        keep = np.ones(self.S, bool)
        keep[g['ignore']] = False
        self.moving = torch.tensor(np.nonzero(keep)[0], device=device)      # spheres checked against obstacles
        ia, ib = np.nonzero(np.triu(~g['mask'], 1))
        self.pair_a, self.pair_b = torch.tensor(ia, device=device), torch.tensor(ib, device=device)
        self.q_min = torch.tensor(g['q_min'], device=device, dtype=dtype)
        self.q_max = torch.tensor(g['q_max'], device=device, dtype=dtype)
        self.set_bases(bases)
        self.set_cuboids(cuboids)

    def set_bases(self, bases):
        """bases: list of (x, y, z, qw, qx, qy, qz) with rotation about z only."""
        assert len(bases) == self.A
        R, t = [], []
        for b in bases:
            a = yaw_of(b[3:7])
            R.append([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
            t.append(b[:3])
        self.base_R = torch.tensor(R, device=self.device, dtype=self.dtype)
        self.base_t = torch.tensor(t, device=self.device, dtype=self.dtype)

    def set_cuboids(self, cuboids):
        """cuboids: list of dicts with pose (x, y, z, 1, 0, 0, 0) and dims, axis-aligned."""
        for c in cuboids:
            assert np.allclose(c['pose'][3:], (1, 0, 0, 0)), c['name']
        self.cc = torch.tensor([c['pose'][:3] for c in cuboids], device=self.device, dtype=self.dtype).reshape(-1, 3)
        self.ch = torch.tensor(np.array([np.divide(c['dims'], 2) for c in cuboids]), device=self.device, dtype=self.dtype).reshape(-1, 3)

    @classmethod
    def from_scene(cls, path, device='cpu', dtype=torch.float32):
        doc = yaml.safe_load(Path(path).read_text())
        return cls([r['base'] for r in doc['robots']], doc['cuboids'], device, dtype)

    # --- kinematics ----------------------------------------------------------------------------------------------------------------
    def local_spheres(self, q):
        """Sphere centres (N, S, 3) and grasp point (N, 3) of one arm in its own base_link frame, q (N, 6) rad."""
        F = link_frames(q.to(self.dtype), self.O, self.T0)
        Fs = F[:, self.frame]
        p = (Fs[..., :3, :3] @ self.local[None, :, :, None]).squeeze(-1) + Fs[..., :3, 3]
        ee = (F[:, 5, :3, :3] @ self.ee_local[None, :, None]).squeeze(-1) + F[:, 5, :3, 3]
        return p, ee

    def spheres(self, q):
        """World sphere centres (N, A, S, 3) and grasp points (N, A, 3); q (N, 6 A) rad."""
        ps, es = [], []
        for a in range(self.A):
            p, e = self.local_spheres(q[:, 6 * a:6 * a + 6])
            ps.append(p @ self.base_R[a].T + self.base_t[a])
            es.append(e @ self.base_R[a].T + self.base_t[a])
        return torch.stack(ps, 1), torch.stack(es, 1)

    # --- distances -----------------------------------------------------------------------------------------------------------------
    def obstacle(self, p):
        """(N,) min over the moving spheres and the cuboids; p (N, S, 3) world centres of one arm."""
        p, r = p[:, self.moving], self.radius[self.moving]
        delta = (p[:, :, None, :] - self.cc[None, None]).abs() - self.ch[None, None]
        out = delta.clamp(min=0).norm(dim=-1)
        inside = delta.amax(-1).clamp(max=0)                            # negative depth for centres inside a cuboid
        return (out + inside).amin(2).sub(r).amin(1)

    def self_distance(self, p):
        return ((p[:, self.pair_a] - p[:, self.pair_b]).norm(dim=-1) - self.radius[self.pair_a] - self.radius[self.pair_b]).amin(1)

    def inter(self, pa, pb):
        d = torch.cdist(pa, pb) - self.radius[None, :, None] - self.radius[None, None, :]
        return d.flatten(1).amin(1)

    @torch.no_grad()
    def parts(self, q):
        """dict of tensors: obstacle (N, A), self (N, A), inter (N,) (inf for one arm), valid (N,) bool, ee (N, A, 3)."""
        p, ee = self.spheres(q)
        obs = torch.stack([self.obstacle(p[:, a]) for a in range(self.A)], 1)
        slf = torch.stack([self.self_distance(p[:, a]) for a in range(self.A)], 1)
        inter = self.inter(p[:, 0], p[:, 1]) if self.A == 2 else torch.full((len(q),), float('inf'), device=q.device, dtype=self.dtype)
        valid = (obs >= OBS_VALID_M).all(1) & (slf >= SELF_VALID_M).all(1) & (inter >= INTER_VALID_M)
        inlim = ((q.reshape(len(q), self.A, 6) >= self.q_min) & (q.reshape(len(q), self.A, 6) <= self.q_max)).flatten(1).all(1)
        return dict(obstacle=obs, self=slf, inter=inter, valid=valid & inlim, ee=ee, spheres=p)


def check_urdf(n=64, seed=0):
    """Grasp point and wrist-3 frame of this model vs yourdfpy on the release URDF (base at the origin)."""
    import yourdfpy
    urdf = yourdfpy.URDF.load(str(HERE.parents[1] / 'runtime/assets/runtime/robot/ur5e_robotiq_2f_140.urdf'),
                              build_collision_scene_graph=False, load_collision_meshes=False)
    arms = Arms([(0, 0, 0, 1, 0, 0, 0)], [], dtype=torch.float64)
    q = torch.rand(n, 6, generator=torch.Generator().manual_seed(seed), dtype=torch.float64) * 2 * math.pi - math.pi
    _, ee = arms.local_spheres(q)
    err = 0.0
    for i in range(n):
        cfg = urdf.zero_cfg.copy()
        for name, v in zip(JOINT_NAMES, q[i].tolist()):
            cfg[urdf.actuated_joint_names.index(name)] = v
        urdf.update_cfg(cfg)
        err = max(err, float(np.abs(urdf.get_transform('grasp_frame')[:3, 3] - ee[i].numpy()).max()))
    return err


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--extract', action='store_true', help=f'write robot_geometry.npz from {EXPORT}')
    a = ap.parse_args()
    if a.extract:
        print('fit error (spheres, grasp point) m:', extract(HERE.parents[1] / EXPORT))
    print('grasp point vs URDF, max error m:', check_urdf())
