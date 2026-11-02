"""UR5e link kinematics, cuRobo-equivalent collision spheres and the box cover of the arm (pure torch).

Link transforms follow the frozen URDF (runtime/assets/runtime/robot/ur5e_robotiq_2f_140.urdf): the six
revolute arm joints move, every other joint is fixed (the cuRobo config locks finger_joint at 0 and the
other finger joints mimic it). Spheres come from runtime/vendor/planner/spheres_ur5e_arm.yml in the
release's collision_link_names order, with cuRobo's collision_sphere_buffer (0.005 m) added to every
radius and a non-positive padded radius replaced by 0.001 m (cuda_robot_generator._build_collision_model),
so `Robot.link_spheres(q)` reproduces `CudaRobotModel.get_state(q).link_spheres_tensor` and
`Robot.ee_position(q)` the grasp-frame position. Everything is differentiable in q.

Box cover (the per-face speed formulation): one axis-aligned box per rigid body in that body's reference
link frame, the smallest box containing all of the body's padded spheres. Because the boxes contain the
spheres, box-face clearance <= sphere clearance everywhere, so the face speed is a conservative surrogate
of the release label (experiments/cabinet/geometry.py clearance_speed). The shoulder is left out, as the
label leaves out spheres 0 and 1 (physical_geometry.OBSTACLE_IGNORE_SPHERE_INDICES).
"""
import importlib.util
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT/'runtime/assets/runtime/robot/ur5e_robotiq_2f_140.urdf'
PLANNER = ROOT/'runtime/vendor/planner'
SPHERES = PLANNER/'spheres_ur5e_arm.yml'
ROBOT_CFG = PLANNER/'ur5e_arm.yml'
NORMALIZATION = ROOT/'runtime/configs/joint_normalization.npz'
JOINTS = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
EE_LINK = 'grasp_frame'
IGNORED_SPHERES = (0, 1)   # physical_geometry.OBSTACLE_IGNORE_SPHERE_INDICES: the static shoulder spheres
# Rigid bodies of the box cover: name -> (reference link, links whose spheres the box must contain).
BODIES = {
    'upper_arm': ('upper_arm_link', ['upper_arm_link']),
    'forearm': ('forearm_link', ['forearm_link']),
    'wrist_1': ('wrist_1_link', ['wrist_1_link']),
    'wrist_2': ('wrist_2_link', ['wrist_2_link']),
    'hand': ('wrist_3_link', ['wrist_3_link', 'tool0', 'robotiq_arg2f_base_link', 'left_outer_knuckle', 'left_inner_knuckle',
                              'left_outer_finger', 'left_inner_finger', 'left_inner_finger_pad', 'right_outer_knuckle',
                              'right_inner_knuckle', 'right_outer_finger', 'right_inner_finger', 'right_inner_finger_pad']),
}
AXES = 'xyz'


def collision_utils():
    """The release's pure-torch collision helpers, loaded by file so no /workspace path is needed."""
    spec = importlib.util.spec_from_file_location('release_collision_utils', PLANNER/'collision_utils.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def rpy_matrix(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return torch.tensor([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr], [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


def rot_z(theta):
    c, s = torch.cos(theta), torch.sin(theta); zero = torch.zeros_like(c); one = torch.ones_like(c)
    return torch.stack([c, -s, zero, zero, s, c, zero, zero, zero, zero, one, zero, zero, zero, zero, one], -1).reshape(-1, 4, 4)


class Kinematics(torch.nn.Module):
    """Batched forward kinematics of every URDF link; only JOINTS move."""
    def __init__(self, urdf=URDF):
        super().__init__()
        root = ET.parse(urdf).getroot(); self.joint_of = {}; self.index = {}
        for n, j in enumerate(root.findall('joint')):
            origin = j.find('origin'); axis = j.find('axis')
            xyz = [float(x) for x in (origin.get('xyz', '0 0 0') if origin is not None else '0 0 0').split()]
            rpy = [float(x) for x in (origin.get('rpy', '0 0 0') if origin is not None else '0 0 0').split()]
            t = torch.eye(4); t[:3, :3] = rpy_matrix(*rpy); t[:3, 3] = torch.tensor(xyz)
            self.register_buffer(f'origin_{n}', t)
            name = j.get('name'); moving = name in JOINTS
            if moving and (axis is None or axis.get('xyz').split() != ['0', '0', '1']): raise ValueError(f'{name}: only z-axis joints are supported')
            if not moving and j.get('type') not in ('fixed', 'revolute'): raise ValueError(f'{name}: unexpected joint type {j.get("type")}')
            self.joint_of[j.find('child').get('link')] = (n, j.find('parent').get('link'), JOINTS.index(name) if moving else -1)
        self.links = ['base_link'] + list(self.joint_of)

    def transforms(self, q_rad, links):
        """World transform (B, 4, 4) of each requested link for joint angles q_rad (B, 6)."""
        memo = {'base_link': torch.eye(4, device=q_rad.device, dtype=q_rad.dtype).expand(len(q_rad), 4, 4)}
        def get(link):
            if link not in memo:
                n, parent, idx = self.joint_of[link]; t = get(parent) @ getattr(self, f'origin_{n}').to(q_rad.dtype)
                memo[link] = t @ rot_z(q_rad[:, idx]) if idx >= 0 else t
            return memo[link]
        return {link: get(link) for link in links}


class Robot(torch.nn.Module):
    """Kinematics, the release sphere model (padded radii), the self-collision mask and joint normalization."""
    def __init__(self, urdf=URDF, spheres=SPHERES, robot_cfg=ROBOT_CFG, normalization=NORMALIZATION):
        super().__init__()
        self.kinematics = Kinematics(urdf)
        cfg = yaml.safe_load(Path(robot_cfg).read_text())['robot_cfg']['kinematics']
        self.collision_links = list(cfg['collision_link_names']); buffer = float(cfg['collision_sphere_buffer'])
        data = yaml.safe_load(Path(spheres).read_text())['collision_spheres']
        centers, radii, owner, names = [], [], [], []
        for k, link in enumerate(self.collision_links):
            for i, s in enumerate(data[link]):
                r = float(s['radius'])+buffer
                if -1. < r <= 0.: r = .001
                centers.append([float(x) for x in s['center']]); radii.append(r); owner.append(k); names.append(f'{link}:{i}')
        self.register_buffer('sphere_centers', torch.tensor(centers, dtype=torch.float32))
        self.register_buffer('sphere_radii', torch.tensor(radii, dtype=torch.float32))
        self.register_buffer('sphere_link', torch.tensor(owner))
        self.sphere_names = names; self.sphere_buffer = buffer
        mask, _ = collision_utils().build_self_collision_mask(str(spheres), cfg.get('self_collision_ignore', {}), self.collision_links, device='cpu')
        self.register_buffer('self_mask', mask)
        norm = np.load(normalization)
        self.register_buffer('q_center', torch.tensor(norm['q_center'], dtype=torch.float32))
        self.register_buffer('q_range', torch.tensor(norm['q_range'], dtype=torch.float32))
        self.register_buffer('q_min', torch.tensor(norm['q_min'], dtype=torch.float32))
        self.register_buffer('q_max', torch.tensor(norm['q_max'], dtype=torch.float32))

    def radians(self, q_norm):
        return q_norm*self.q_range+self.q_center

    def transforms(self, q_rad, links=None):
        return self.kinematics.transforms(q_rad, links or self.collision_links+[EE_LINK])

    def link_spheres(self, q_rad, transforms=None):
        """(B, S, 4) world sphere centers and padded radii, the order of CudaRobotModel.link_spheres_tensor."""
        t = transforms or self.transforms(q_rad, self.collision_links)
        stacked = torch.stack([t[link] for link in self.collision_links], 1)[:, self.sphere_link]        # (B, S, 4, 4)
        world = torch.einsum('bsij,sj->bsi', stacked[:, :, :3, :3], self.sphere_centers)+stacked[:, :, :3, 3]
        return torch.cat([world, self.sphere_radii.expand(len(q_rad), -1)[..., None]], -1)

    def ee_position(self, q_rad, transforms=None):
        t = transforms or self.transforms(q_rad, [EE_LINK])
        return t[EE_LINK][:, :3, 3]

    def self_distance(self, spheres):
        """Release sphere_sphere_distance with the release self-collision mask: (B,)."""
        return collision_utils().sphere_sphere_distance(spheres[..., :3], spheres[..., 3], self.self_mask)


class BoxCover(torch.nn.Module):
    """Axis-aligned boxes in each body's reference link frame containing all its padded spheres; their faces."""
    def __init__(self, robot, bodies=BODIES, inflate=0.):
        super().__init__()
        self.bodies = dict(bodies); self.body_names = list(bodies); self.reference = [bodies[b][0] for b in self.body_names]
        with torch.no_grad():
            device = robot.sphere_centers.device
            t0 = robot.kinematics.transforms(torch.zeros(1, 6, device=device), robot.collision_links)
            boxes = []
            for body, (ref, links) in bodies.items():
                inv = torch.linalg.inv(t0[ref][0]); lo = torch.full((3,), torch.inf, device=device); hi = torch.full((3,), -torch.inf, device=device)
                for link in links:
                    rel = inv @ t0[link][0]; rows = robot.sphere_link == robot.collision_links.index(link)
                    c = robot.sphere_centers[rows] @ rel[:3, :3].T+rel[:3, 3]; r = robot.sphere_radii[rows][:, None]
                    lo = torch.minimum(lo, (c-r).amin(0)); hi = torch.maximum(hi, (c+r).amax(0))
                boxes.append(torch.stack([(lo+hi)/2, (hi-lo)/2+inflate]))
        boxes = torch.stack(boxes).cpu()                                             # (n_bodies, 2, 3) center, half
        self.register_buffer('box_center', boxes[:, 0]); self.register_buffer('box_half', boxes[:, 1])
        eye = torch.eye(3); face_body, center, U, V, a, b, n, names = [], [], [], [], [], [], [], []
        for k_body, body in enumerate(self.body_names):
            for axis in range(3):
                for sign in (1., -1.):
                    i, j = (axis+1) % 3, (axis+2) % 3
                    face_body.append(k_body); center.append(boxes[k_body, 0]+sign*boxes[k_body, 1, axis]*eye[axis])
                    U.append(eye[i]); V.append(eye[j]); a.append(boxes[k_body, 1, i]); b.append(boxes[k_body, 1, j]); n.append(sign*eye[axis])
                    names.append(f"{body}:{'+' if sign > 0 else '-'}{AXES[axis]}")
        self.face_names = names
        self.register_buffer('face_body', torch.tensor(face_body)); self.register_buffer('face_center', torch.stack(center))
        self.register_buffer('face_u', torch.stack(U)); self.register_buffer('face_v', torch.stack(V)); self.register_buffer('face_n', torch.stack(n))
        self.register_buffer('face_a', torch.stack(a)); self.register_buffer('face_b', torch.stack(b))

    def place(self, transforms):
        """World placement of every face: center, U, V, n (B, F, 3) and half sizes a, b (F,)."""
        t = torch.stack([transforms[ref] for ref in self.reference], 1)[:, self.face_body]   # (B, F, 4, 4)
        R, p = t[:, :, :3, :3], t[:, :, :3, 3]
        rotate = lambda v: torch.einsum('bfij,fj->bfi', R, v)
        return dict(center=rotate(self.face_center)+p, U=rotate(self.face_u), V=rotate(self.face_v), n=rotate(self.face_n), a=self.face_a, b=self.face_b)

    def summary(self):
        return {body: dict(reference=self.reference[k], center=self.box_center[k].tolist(), half=self.box_half[k].tolist(),
                           size=(2*self.box_half[k]).tolist()) for k, body in enumerate(self.body_names)}
