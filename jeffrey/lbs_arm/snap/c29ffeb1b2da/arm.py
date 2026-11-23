"""The cabinet arm as an LBS scene: the fixed-base UR5e + Robotiq 2F-140 of the release, 6 joints, in a cabinet scene's cuboids.

Configuration x = q / q_range (the release normalization: every joint's limits map to [-0.5, 0.5], range 1; q_range = 4 pi rad, elbow 2 pi).

Geometry = the release's own (export_robot.py writes it from the label code and the scorer's functions): cuRobo's 53 collision spheres on
the six arm links (tool0 and the gripper links are rigid on wrist_3: the finger joint is locked), the scene's axis-aligned cuboids, the
self-collision mask. The kinematics here are plain torch (UR5e joint origins of the URDF), so the clearance is differentiable and can be
compiled; the sphere centres in their link frames are fitted from cuRobo's spheres at q = 0 and checked at 63 other configurations, and
the distances are checked against the label code at 200,000 configurations (check()).

    d_obs   min over spheres (without the two shoulder spheres, as the label) and cuboids of |outside distance| - radius      (metres)
    d_self  min over the unmasked sphere pairs of centre distance - radii
    valid   d_obs over all spheres >= 0.005 and d_self >= 0.002 (the release scorer's rule; the shoulder spheres do not move)
    d       min(d_obs, 0.005 + K_SELF (d_self - 0.002))          speed = clip(d, 0.005, 0.05) / 0.05

The clearance d is not the release label min(d_obs, 0.5 d_self): with that label the self term sets the speed of 86 % of the valid
configurations whose grasp point is in the sampling box (0.5 d_self never exceeds 0.055 m: wrist 1 and the forearm sit next to the
gripper in every pose) and 8 % of them reach full speed. Here self contact is a wall with a thin band: d = 0.005 m exactly where the
scorer's self rule (0.002 m) is met, K_SELF = 5 times steeper than the obstacle term, so that the self term is active only within
0.011 m of self distance. d >= 0.005 m iff the scorer accepts the configuration.

"Cells": the maze studies measure clearance in cells with margin 0.175 and offset 0.0175 cells. Here one cell is 0.05 / 0.175 m of
clearance (CELL_M), so signed_cells(x) = d / CELL_M has the same margin, offset, speed profile and lift profile as every other scene.
A configuration is in contact (invalid for the trainer) below `contact` = the offset (d < 0.005 m), where the mazes have 0.
E = cells per unit of the normalized joint space along the clearance gradient (the median |grad_x signed_cells| in the band, measured,
see measure()): the number the mazes call E (cells across the box). pitch = 1 / E.
"""
import json
import math
import os
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = next((q for q in HERE.parents if (q / 'baselines/ntrl-demo').is_dir()), Path(os.environ.get('NTFIELD_ROOT', '/home/jeffrey/bellman-ntfield')))
if str(ROOT / 'jeffrey/maze_scale') not in sys.path:
    sys.path.append(str(ROOT / 'jeffrey/maze_scale'))
from maze import MARGIN_CELLS, OFFSET_CELLS, ALPHA  # noqa: E402

D = 6
MARGIN_M, OFFSET_M, K_SELF = 0.05, 0.005, 5.0
OBS_VALID_M, SELF_VALID_M = 0.005, 0.002
CELL_M = MARGIN_M / MARGIN_CELLS                 # metres of clearance per cell (0.2857)
E_DEFAULT = 24.0                                 # cells per unit of normalized joint space (measure(): median band gradient), cabinet scene

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


def link_frames(q, O, T0):
    """(N, 6, 4, 4) frames of the six arm links in base_link coordinates (metres) at joint angles q (N, 6) rad."""
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


class CabinetArm:
    kind = 'cabinet_arm'

    def __init__(self, model, device='cuda:0', dtype=torch.float32, E=E_DEFAULT):
        m = np.load(model, allow_pickle=False)
        self.model_path = str(model)
        self.device, self.dtype = device, dtype
        self.E = float(E)
        self.pitch = 1.0 / self.E
        self.margin, self.offset = MARGIN_CELLS * self.pitch, OFFSET_CELLS * self.pitch
        self.contact = OFFSET_CELLS                                              # cells: below this the configuration is in contact
        bi = torch.eye(4, dtype=torch.float64)
        bi[:3, :3] = _rot('z', math.pi)
        O64 = torch.stack([_origin(x, r) for x, r in JOINTS])
        # sphere centres in their link frames, from cuRobo's spheres at the first fit configuration
        qf = torch.tensor(m['q_fit'], dtype=torch.float64)
        sp = torch.tensor(m['spheres_fit'], dtype=torch.float64)
        frame = torch.tensor(np.minimum(m['link'], 5))
        F = link_frames(qf, O64, bi)                                                # (K, 6, 4, 4)
        F0 = F[0, frame]
        local = (F0[:, :3, :3].transpose(1, 2) @ (sp[0, :, :3] - F0[:, :3, 3])[:, :, None]).squeeze(-1)
        Fe = F[0, 5]
        ee_local = Fe[:3, :3].T @ (torch.tensor(m['ee_fit'][0], dtype=torch.float64) - Fe[:3, 3])
        Fs = F[:, frame]
        world = (Fs[..., :3, :3] @ local[None, :, :, None]).squeeze(-1) + Fs[..., :3, 3]
        self.fit_error = float((world - sp[..., :3]).abs().max())
        ee = (F[:, 5, :3, :3] @ ee_local[None, :, None]).squeeze(-1) + F[:, 5, :3, 3]
        self.fit_error_ee = float((ee - torch.tensor(m['ee_fit'], dtype=torch.float64)).abs().max())
        assert self.fit_error < 2e-5 and self.fit_error_ee < 2e-5, (self.fit_error, self.fit_error_ee)
        self.T0, self.O = bi.to(device, dtype), O64.to(device, dtype)
        self.frame = frame.to(device)
        self.local = local.to(device, dtype)
        self.radius = torch.tensor(m['spheres_fit'][0, :, 3], device=device, dtype=dtype)
        self.S = len(self.radius)
        self.ee_local = ee_local.to(device, dtype)
        keep = np.ones(self.S, bool)
        keep[m['ignore']] = False
        self.label_idx = torch.tensor(np.nonzero(keep)[0], device=device)
        mask = m['mask']
        ia, ib = np.nonzero(np.triu(~mask, 1))
        self.pair_a, self.pair_b = torch.tensor(ia, device=device), torch.tensor(ib, device=device)
        self.pair_r = self.radius[self.pair_a] + self.radius[self.pair_b]
        self.cc = torch.tensor(m['cuboid_centre'], device=device, dtype=dtype)
        self.ch = torch.tensor(m['cuboid_half'], device=device, dtype=dtype)
        self.q_range = torch.tensor(m['q_range'], device=device, dtype=dtype)
        self.box = torch.tensor(m['box'], device=device, dtype=dtype)                 # (3, 2) grasp-point sampling box
        self.cupboard = torch.tensor(m['cupboard'], device=device, dtype=dtype)
        self.lo = torch.tensor(m['q_min'] / m['q_range'], device=device, dtype=dtype)
        self.hi = torch.tensor(m['q_max'] / m['q_range'], device=device, dtype=dtype)
        self.clamp_mask = torch.ones(D, dtype=torch.bool, device=device)
        self.phi_max = 0.5

    # --- kinematics ------------------------------------------------------------------------------------------------------------------
    def spheres(self, x):
        """Sphere centres (N, S, 3) metres and the grasp point (N, 3) at normalized configurations x (N, 6)."""
        q = x.to(self.dtype) * self.q_range
        F = link_frames(q, self.O, self.T0)
        Fs = F[:, self.frame]
        p = (Fs[..., :3, :3] @ self.local[None, :, :, None]).squeeze(-1) + Fs[..., :3, 3]
        ee = (F[:, 5, :3, :3] @ self.ee_local[None, :, None]).squeeze(-1) + F[:, 5, :3, 3]
        return p, ee

    def ee(self, x):
        return self.spheres(x)[1]

    def _obs(self, p, idx=None):
        """(N, S') signed sphere-cuboid distances minimized over the cuboids (the label code's formula)."""
        r = self.radius
        if idx is not None:
            p, r = p[:, idx], r[idx]
        delta = (p[:, :, None, :] - self.cc[None, None]).abs() - self.ch[None, None]
        dist = torch.sqrt((delta.clamp(min=0) ** 2).sum(-1) + 1e-18)
        return dist.amin(2) - r

    def _self(self, p):
        dd = torch.sqrt(((p[:, self.pair_a] - p[:, self.pair_b]) ** 2).sum(-1) + 1e-18) - self.pair_r
        return dd.amin(1)

    def parts(self, x):
        """dict of (N,) metres: obs (without the shoulder spheres), obs_all (scorer), self, d (this scene's clearance), valid margin."""
        p, ee = self.spheres(x)
        o = self._obs(p)
        obs_all = o.amin(1)
        obs = o[:, self.label_idx].amin(1)
        s = self._self(p)
        return dict(obs=obs, obs_all=obs_all, self=s, d=torch.minimum(obs, OFFSET_M + K_SELF * (s - SELF_VALID_M)), valid=torch.minimum(obs_all - OBS_VALID_M, s - SELF_VALID_M), ee=ee)

    def signed_cells(self, x):
        p, _ = self.spheres(x)
        return torch.minimum(self._obs(p, self.label_idx).amin(1), OFFSET_M + K_SELF * (self._self(p) - SELF_VALID_M)) / CELL_M

    def clearance(self, x, chunk=1 << 17):
        with torch.no_grad():
            return torch.cat([self.signed_cells(x[i:i + chunk]) for i in range(0, len(x), chunk)]) if len(x) else x.new_zeros(0)

    def valid_cells(self, x, chunk=1 << 17):
        """Scorer margin in cells: >= 0 iff the release scorer accepts the configuration (obstacle >= 0.005 m, self >= 0.002 m)."""
        with torch.no_grad():
            return torch.cat([self.parts(x[i:i + chunk])['valid'] for i in range(0, len(x), chunk)]) / CELL_M if len(x) else x.new_zeros(0)

    def clearance_normal(self, x, chunk=1 << 16):
        ds, ns = [], []
        for i in range(0, len(x), chunk):
            y = x[i:i + chunk].detach().clone().to(self.dtype).requires_grad_(True)
            with torch.enable_grad():
                d = self.signed_cells(y)
                g, = torch.autograd.grad(d.sum(), y)
            ds.append(d.detach())
            ns.append(torch.nn.functional.normalize(g, dim=1))
        return torch.cat(ds), torch.cat(ns)

    def gradient(self, x, chunk=1 << 16):
        out = []
        for i in range(0, len(x), chunk):
            y = x[i:i + chunk].detach().clone().to(self.dtype).requires_grad_(True)
            with torch.enable_grad():
                g, = torch.autograd.grad(self.signed_cells(y).sum(), y)
            out.append(g.detach())
        return torch.cat(out)

    @staticmethod
    def unit_speed(d_cells):
        return (d_cells / MARGIN_CELLS).clamp(OFFSET_CELLS / MARGIN_CELLS, 1.0)

    @staticmethod
    def train_speed(d_cells):
        s = CabinetArm.unit_speed(d_cells)
        return ALPHA * s * s * (2 - s) * (2 - s) + 1 - ALPHA

    # --- sampling --------------------------------------------------------------------------------------------------------------------
    def uniform(self, count, gen):
        return self.lo + (self.hi - self.lo) * torch.rand(count, D, generator=gen, device=self.device, dtype=self.dtype)

    def in_box(self, x, chunk=1 << 18):
        out = []
        with torch.no_grad():
            for i in range(0, len(x), chunk):
                e = self.ee(x[i:i + chunk])
                out.append(((e >= self.box[:, 0]) & (e <= self.box[:, 1])).all(1))
        return torch.cat(out)

    def free(self, count, gen, min_clear=0.0, batch=1 << 20, clear=None):
        """Uniform configurations with the grasp point in the sampling box and clearance >= contact + min_clear cells."""
        out, have = [], 0
        fn = self.clearance if clear is None else clear
        while have < count:
            x = self.uniform(batch, gen)
            x = x[self.in_box(x)]
            x = x[fn(x) >= self.contact + min_clear]
            out.append(x)
            have += len(x)
        return torch.cat(out)[:count]

    def inside(self, x):
        return ((x >= self.lo) & (x <= self.hi)).all(1)

    def clamp(self, x):
        return torch.maximum(torch.minimum(x, self.hi), self.lo)

    def spec(self):
        return dict(kind=self.kind, model=self.model_path, E=self.E, cell_m=CELL_M, margin_m=MARGIN_M, offset_m=OFFSET_M, k_self=K_SELF,
                    spheres=int(self.S), self_pairs=int(len(self.pair_a)), cuboids=int(len(self.cc)), fit_error_m=self.fit_error)

    def describe(self):
        return dict(E_cells=self.E, pitch=self.pitch, margin=self.margin, offset=self.offset, dim=D, lo=self.lo.tolist(), hi=self.hi.tolist(),
                    contact_cells=self.contact, variant=self.kind, scene=self.spec())

    # --- checks ----------------------------------------------------------------------------------------------------------------------
    def check(self):
        """Max differences to the label code's and the scorer's distances at the export's reference configurations (metres)."""
        m = np.load(self.model_path)
        x = torch.tensor(m['q_ref'] / m['q_range'], device=self.device, dtype=self.dtype)
        out = dict(obs=0.0, obs_all=0.0, self=0.0, ee=0.0)
        with torch.no_grad():
            for i in range(0, len(x), 50000):
                pt = self.parts(x[i:i + 50000])
                for k, ref in (('obs', 'd_obs'), ('obs_all', 'd_obs_all'), ('self', 'd_self'), ('ee', 'ee_ref')):
                    out[k] = max(out[k], float((pt[k].cpu() - torch.tensor(m[ref][i:i + 50000])).abs().max()))
        out.update(fit_error_m=self.fit_error, fit_error_ee_m=self.fit_error_ee, n=int(len(x)))
        return out

    def measure(self, n=400_000, seed=0):
        """|grad_x signed_cells| (cells per unit) at band configurations with the grasp point in the box: quantiles, and per joint."""
        gen = torch.Generator(self.device).manual_seed(seed)
        xs = []
        while sum(len(v) for v in xs) < n:
            x = self.uniform(1 << 20, gen)
            x = x[self.in_box(x)]
            d = self.clearance(x)
            xs.append(x[(d > OFFSET_CELLS) & (d < MARGIN_CELLS)])
        x = torch.cat(xs)[:n]
        g = self.gradient(x)
        nrm = torch.linalg.vector_norm(g, dim=1)
        q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=g.device)
        return dict(n=int(len(x)), norm_quantiles=torch.quantile(nrm, q).tolist(), norm_mean=float(nrm.mean()), per_joint_rms=torch.sqrt((g ** 2).mean(0)).tolist(),
                    per_joint_abs_median=g.abs().median(0).values.tolist())


def load_scene(prepared, device='cuda:0'):
    info = json.load(open(Path(prepared) / 'scene.json'))
    scene = CabinetArm(Path(prepared) / 'robot_model.npz', device, E=info['E_cells'])
    return scene, info
