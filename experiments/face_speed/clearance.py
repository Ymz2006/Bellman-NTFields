"""Per-piece clearance and speed of the arm in a cabinet scene.

The release label is clearance d = min(d_obstacle, 0.5 d_self) with d_obstacle the minimum signed
sphere-cuboid distance over the spheres (shoulder spheres excluded) and speed clip(d, 0.005, 0.05)/0.05
(experiments/cabinet/geometry.py). Here that minimum is kept as a vector of pieces:

  cover='faces'    obstacle pieces are the faces of the box cover (robot.BoxCover): d_f = signed distance of
                   face f to its nearest cuboid; the user's per-face formulation. S = min_f s_f <= release S.
  cover='spheres'  obstacle pieces are the individual non-ignored spheres; min over pieces is exactly the
                   release clearance (the control that isolates the min structure from the geometry change).
  cover='links'    one piece per collision link, the min of its spheres (the coarse exact decomposition).
  cover='environment'  one piece, the environment clearance alone (min over the spheres); NO self-collision
                   term: the 'forget self-collision' label. Its min is >= the release clearance.
  cover='split'    two pieces: environment collision (min over the spheres) and self-collision; the min is
                   exactly the release clearance (one model per kind of collision).

The last piece is always the self-collision term 0.5 d_self (release mask). Speeds are per piece
clip(d_i, OFFSET, MARGIN)/MARGIN; the body speed is their minimum, and the gradient of the clearance
(the normal of metric NTFields' data) is autograd through the kinematics.
"""
import torch
from torch import nn
from . import robot as R
from . import faces as FACES

MARGIN, OFFSET, SELF_SCALE = .05, .005, .5


def scene_cuboids(scene, device):
    import yaml
    from pathlib import Path
    cubes = yaml.safe_load(Path(scene).read_text())['cuboids']
    c = torch.tensor([x['pose'][:3] for x in cubes], dtype=torch.float32, device=device)
    h = torch.tensor([[d/2 for d in x['dims']] for x in cubes], dtype=torch.float32, device=device)
    return c-h, c+h, [x['name'] for x in cubes]


class PieceClearance(nn.Module):
    def __init__(self, scene, cover='faces', device='cuda:0', far=.15, inflate=0., bodies=R.BODIES):
        super().__init__()
        self.cover, self.far = cover, far
        self.robot = R.Robot().to(device)
        self.lo, self.hi, self.cuboid_names = scene_cuboids(scene, device)
        keep = torch.ones(len(self.robot.sphere_radii), dtype=torch.bool); keep[list(R.IGNORED_SPHERES)] = False
        self.register_buffer('sphere_keep', keep.to(device))
        if cover == 'faces':
            self.boxes = R.BoxCover(self.robot, bodies, inflate).to(device); obstacle = list(self.boxes.face_names)
        elif cover == 'spheres':
            obstacle = [n for n, k in zip(self.robot.sphere_names, keep.tolist()) if k]
        elif cover == 'links':
            links = sorted({int(k) for k in self.robot.sphere_link[keep].tolist()}); self.piece_links = links
            obstacle = [self.robot.collision_links[k] for k in links]
        elif cover in ('split', 'environment'):
            obstacle = ['environment']
        else: raise ValueError(cover)
        self.names = obstacle+([] if cover == 'environment' else ['self'])
        self.device = device

    @property
    def n_pieces(self): return len(self.names)

    def pieces(self, q_norm):
        """(B, P) signed clearances per piece (self piece already scaled by 0.5)."""
        rad = self.robot.radians(q_norm)
        links = self.robot.collision_links if self.cover != 'faces' else list(dict.fromkeys(self.robot.collision_links+self.boxes.reference))
        t = self.robot.transforms(rad, links)
        spheres = self.robot.link_spheres(rad, t)
        d_self = SELF_SCALE*self.robot.self_distance(spheres)
        if self.cover == 'faces':
            d = FACES.face_clearance(self.boxes.place(t), self.lo, self.hi, self.far)
        else:
            c, r = spheres[..., :3], spheres[..., 3]
            per = (point_aabb(c[:, :, None], self.lo, self.hi)-r[..., None]).amin(-1)      # (B, S)
            per = per[:, self.sphere_keep]
            if self.cover == 'links':
                owner = self.robot.sphere_link[self.sphere_keep]
                d = torch.stack([per[:, owner == k].amin(1) for k in self.piece_links], 1)
            elif self.cover in ('split', 'environment'): d = per.amin(1, keepdim=True)
            else: d = per
        return d if self.cover == 'environment' else torch.cat([d, d_self[:, None]], 1)

    def clearance(self, q_norm):
        return self.pieces(q_norm).amin(1)

    @staticmethod
    def speed_of(d):
        return d.clamp(OFFSET, MARGIN)/MARGIN

    def speeds(self, q_norm):
        return self.speed_of(self.pieces(q_norm))

    def speed(self, q_norm):
        return self.speeds(q_norm).amin(1)

    def gradient(self, q_norm, create_graph=False):
        """d clearance / d q_norm by autograd (a subgradient at kinks), (B, 6)."""
        with torch.enable_grad():
            x = q_norm.detach().requires_grad_(True)
            d = self.clearance(x)
            return torch.autograd.grad(d.sum(), x, create_graph=create_graph)[0]

    def ee_position(self, q_norm):
        return self.robot.ee_position(self.robot.radians(q_norm))

    def release_components(self, q_norm):
        """The release label's own obstacle and self distances (spheres, shoulder ignored): (B,), (B,)."""
        rad = self.robot.radians(q_norm); spheres = self.robot.link_spheres(rad); c, r = spheres[..., :3], spheres[..., 3]
        per = (point_aabb(c[:, :, None], self.lo, self.hi)-r[..., None]).amin(-1)
        return per[:, self.sphere_keep].amin(1), self.robot.self_distance(spheres)


def point_aabb(p, lo, hi):
    return torch.linalg.vector_norm((lo-p).clamp_min(0)+(p-hi).clamp_min(0), dim=-1)
