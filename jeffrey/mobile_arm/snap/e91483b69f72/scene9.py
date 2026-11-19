"""The 9-DOF mobile-manipulator scene: robot.py's base + scaled UR5e in a Gibson house floor plan (jeffrey/lift_scale/scenes.py House).

Configuration q = (x, y, phi, psi_1 .. psi_6), normalized like the house studies' SE(2) robot: x, y in [-0.5, 0.5] (E cells across),
phi = RHO_H theta / E with theta in [-pi/2, pi/2] (RHO_H = 0.25 cells per radian, maze_scale's RHO: the bare rectangle and the tucked
robot have the same C-space as the house studies), psi_j = RHO_J[j] q_j / E with the joint box of robot.py. RHO_J = the largest lever
arm of joint j (distance from its axis to the farthest sphere centre it moves, over a million random joint vectors), rounded up, floor
0.05 cells: turning joint j by d q moves no sphere centre farther than RHO_J[j] d q cells, so every joint is charged like the base.

Clearance (cells, the one function labels, truth, planner and scorer use):
    d_env = min(base lattice clearance (the house studies' rectangle, 9 x 5 points), min over arm spheres of plan SDF(centre) - r)
    d_int = robot.Arm.internal (floor, base box, cuRobo's self pairs)
    d     = min(d_env, K_INT d_int)
ntrl-demo's speed label clip(d / margin, offset / margin, 1) with the house studies' margin 0.175 cells. Internal distances enter
K_INT = 3.5 times stretched (an internal margin of 0.05 cells): the arm's own parts sit closer than 0.175 cells in every pose (the upper
arm's shoulder end is 0.067 cells above the base box), so the wall margin would make the robot slow everywhere. d < 0 iff the robot
touches a wall, the floor, the base or itself.

TUCK = the folded pose (arm straight up, upper arm offset along the heading): every sphere's footprint lies inside the base rectangle
(0.02 cells to spare at the shoulder) and K_INT d_int = 0.23 > margin, so the tucked robot has exactly the rectangle's walls and full
speed away from them: its base motions are the house studies' SE(2) problem (their truth fields certify queries, prepare.py).
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
if str(ROOT / 'jeffrey/lift_scale') not in sys.path:
    sys.path.append(str(ROOT / 'jeffrey/lift_scale'))              # scenes.py (House); it puts embed_warp and maze_scale on the path
import scenes as HS  # noqa: E402
import robot as RB  # noqa: E402
from maze import MARGIN_CELLS, OFFSET_CELLS, ALPHA, RHO, REF_N, T_WALL  # noqa: E402

D = 9
RHO_H = RHO                                         # 0.25 cells per radian of heading (maze_scale)
RHO_J = (0.76, 0.76, 0.42, 0.10, 0.06, 0.05)        # measured lever arms 0.759, 0.752, 0.412, 0.099, 0.056, 0.001 cells
K_INT = 3.5                                         # internal margin 0.05 cells
TUCK = (math.pi / 2, -math.pi / 2, 0.0, -math.pi / 2, 0.0, 0.0)


class MobileArm:
    kind = 'mobile_arm'

    def __init__(self, house, device='cuda:0', dtype=torch.float32):
        self.house = house
        self.n, self.E = house.n, float(house.E)
        self.pitch = 1.0 / self.E
        self.phi_max = RHO_H * (math.pi / 2) / self.E
        self.margin = MARGIN_CELLS * self.pitch
        self.offset = OFFSET_CELLS * self.pitch
        self.device, self.dtype = device, dtype
        self.arm = RB.Arm(device, dtype)
        self.rho_j = torch.tensor(RHO_J, device=device, dtype=dtype)
        jl = torch.tensor(RB.JOINT_LO, device=device, dtype=dtype) * self.rho_j / self.E
        jh = torch.tensor(RB.JOINT_HI, device=device, dtype=dtype) * self.rho_j / self.E
        self.lo = torch.cat([torch.tensor([-0.5, -0.5, -self.phi_max], device=device, dtype=dtype), jl])
        self.hi = torch.cat([torch.tensor([0.5, 0.5, self.phi_max], device=device, dtype=dtype), jh])
        self.clamp_mask = torch.tensor([False, False] + [True] * 7, device=device)     # angles stay in the box; x, y are walled in
        self.tuck_psi = torch.tensor(TUCK, device=device, dtype=dtype) * self.rho_j / self.E

    # --- coordinates ---------------------------------------------------------------------------------------------------------------
    def to_cells(self, q):
        """x, y (cells), theta (rad), joints (N, 6) (rad)."""
        return q[:, 0] * self.E + self.n / 2, q[:, 1] * self.E + self.n / 2, q[:, 2] * self.E / RHO_H, q[:, 3:] * self.E / self.rho_j.to(q.dtype)

    def from_parts(self, base3, joints):
        """Normalized q from the SE(2) part (N, 3, house-study normalized) and joint angles (N, 6) rad."""
        return torch.cat([base3, joints * self.rho_j.to(joints.dtype) / self.E], 1)

    def with_tuck(self, base3):
        return torch.cat([base3, self.tuck_psi.to(base3.dtype).expand(len(base3), 6)], 1)

    def planner_scale(self):
        return self.pitch / (1.0 / (REF_N + T_WALL))

    # --- clearance -------------------------------------------------------------------------------------------------------------------
    def world_spheres(self, q):
        """Arm sphere centres (N, S, 3) in cells (x, y in the house frame, z from the floor)."""
        x, y, th, qj = self.to_cells(q)
        p = self.arm.spheres(qj)
        c, s = torch.cos(th)[:, None], torch.sin(th)[:, None]
        wx = x[:, None] + c * p[..., 0] - s * p[..., 1]
        wy = y[:, None] + s * p[..., 0] + c * p[..., 1]
        return torch.stack([wx, wy, p[..., 2]], -1)

    def parts(self, q):
        """dict of (N,) clearances in cells: rect (base lattice), arm (arm spheres vs walls), floor, base, self, env, int, d."""
        q = q.to(self.dtype)
        rect = self.house.signed_cells(q[:, :3])
        x, y, th, qj = self.to_cells(q)
        p = self.arm.spheres(qj)
        c, s = torch.cos(th)[:, None], torch.sin(th)[:, None]
        wx = x[:, None] + c * p[..., 0] - s * p[..., 1]
        wy = y[:, None] + s * p[..., 0] + c * p[..., 1]
        xy = torch.stack([(wx - self.n / 2) / self.E, (wy - self.n / 2) / self.E], -1).reshape(-1, 2)
        armd = (self.house.point_sdf_cells(xy).reshape(len(q), -1) - self.arm.radius.to(q.dtype)).amin(1)
        dint, floor, base, selfd = self.arm.internal(qj, parts=True)
        env = torch.minimum(rect, armd)
        return dict(rect=rect, arm=armd, floor=floor, base=base, self=selfd, env=env, int=dint, d=torch.minimum(env, K_INT * dint))

    def signed_cells(self, q):
        q = q.to(self.dtype)
        rect = self.house.signed_cells(q[:, :3])
        x, y, th, qj = self.to_cells(q)
        p = self.arm.spheres(qj)
        c, s = torch.cos(th)[:, None], torch.sin(th)[:, None]
        wx = x[:, None] + c * p[..., 0] - s * p[..., 1]
        wy = y[:, None] + s * p[..., 0] + c * p[..., 1]
        xy = torch.stack([(wx - self.n / 2) / self.E, (wy - self.n / 2) / self.E], -1).reshape(-1, 2)
        armd = (self.house.point_sdf_cells(xy).reshape(len(q), -1) - self.arm.radius.to(q.dtype)).amin(1)
        return torch.minimum(torch.minimum(rect, armd), K_INT * self.arm.internal(qj))

    def clearance(self, q, chunk=65536):
        with torch.no_grad():
            return torch.cat([self.signed_cells(q[i:i + chunk]) for i in range(0, len(q), chunk)]) if len(q) else q.new_zeros(0)

    def clearance_normal(self, q, chunk=65536):
        ds, ns = [], []
        for i in range(0, len(q), chunk):
            x = q[i:i + chunk].detach().clone().to(self.dtype).requires_grad_(True)
            with torch.enable_grad():
                d = self.signed_cells(x)
                g, = torch.autograd.grad(d.sum(), x)
            ds.append(d.detach())
            ns.append(torch.nn.functional.normalize(g, dim=1))
        return torch.cat(ds), torch.cat(ns)

    @staticmethod
    def unit_speed(d_cells):
        return (d_cells / MARGIN_CELLS).clamp(OFFSET_CELLS / MARGIN_CELLS, 1.0)

    @staticmethod
    def train_speed(d_cells):
        s = MobileArm.unit_speed(d_cells)
        return ALPHA * s * s * (2 - s) * (2 - s) + 1 - ALPHA

    # --- sampling --------------------------------------------------------------------------------------------------------------------
    def uniform(self, count, gen):
        return self.lo + (self.hi - self.lo) * torch.rand(count, D, generator=gen, device=self.device, dtype=self.dtype)

    def free(self, count, gen, min_clear=0.0, batch=262144):
        out, have = [], 0
        while have < count:
            q = self.uniform(batch, gen)
            q = q[self.clearance(q) >= min_clear]
            out.append(q)
            have += len(q)
        return torch.cat(out)[:count]

    def inside(self, q):
        return ((q >= self.lo) & (q <= self.hi)).all(1)

    def clamp(self, q):
        return torch.where(self.clamp_mask, torch.maximum(torch.minimum(q, self.hi), self.lo), q)

    def spec(self):
        return dict(kind='mobile_arm', house=self.house.spec(), rho_h=RHO_H, rho_j=list(RHO_J), k_int=K_INT, tuck=list(TUCK), arm_scale=RB.ARM_SCALE,
                    base=[RB.BASE_L, RB.BASE_W, RB.BASE_H], joint_lo=list(RB.JOINT_LO), joint_hi=list(RB.JOINT_HI))

    def describe(self):
        return dict(n=self.n, E_cells=self.E, pitch=self.pitch, phi_max=self.phi_max, margin=self.margin, offset=self.offset, dim=D,
                    lo=self.lo.tolist(), hi=self.hi.tolist(), planner_scale=self.planner_scale(), variant=self.kind, scene=self.spec(),
                    spheres=int(self.arm.S), self_pairs=int(len(self.arm.pair_a)), house_free_share=self.house.free_share)


def build(spec, device='cuda:0'):
    assert spec['kind'] == 'mobile_arm'
    for k, v in (('rho_j', list(RHO_J)), ('k_int', K_INT), ('tuck', list(TUCK)), ('arm_scale', RB.ARM_SCALE)):
        assert spec[k] == v, f'scene constant {k} changed: {spec[k]} vs {v}'
    return MobileArm(HS.build(spec['house'], device), device)


def load_scene(prepared, device='cuda:0'):
    info = json.load(open(Path(prepared) / 'maze.json'))
    scene = build(info['scene'], device)
    assert abs(scene.E - info['E_cells']) < 1e-6, 'prepared scene differs'
    return scene, info


def house_from_prepare(prepared, device='cuda:0'):
    """The House of a lift_scale house prepare run (its maze.json) and that run's info."""
    house, info = HS.load_scene(prepared, device)
    return house, info
