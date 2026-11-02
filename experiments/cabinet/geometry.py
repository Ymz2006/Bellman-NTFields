"""Cabinet-task geometry: clearance-only labels and the clearance-only offline edge gate.

Same frozen release geometry as experiments/from_source (cuRobo spheres, axis-aligned cuboids),
with the orientation rule removed: the cabinet task has no task tilt. Speed depends on clearance only.
"""
import sys
from pathlib import Path
sys.path.insert(0,'/clean/method')
from common import RELEASE
sys.path.insert(0,'/workspace')
import numpy as np
import torch
import yaml

MARGIN,OFFSET,SELF_SCALE=.05,.005,.5


def scene_boxes(scene):
    """Sampling box, cupboard outer box and compartment boxes (x, y, z ranges) of the cabinet scene.
    A scene without a cupboard (cabinet_clutter, cabinet_platforms) names its focus box explicitly (`focus:`);
    it takes the cupboard's place in the coverage strata and the test regions."""
    doc=yaml.safe_load(Path(scene).read_text())
    box=np.array([doc['sampling'][k] for k in 'xyz'],np.float32)
    if 'focus' in doc:
        cupboard=np.array([doc['focus'][k] for k in 'xyz'],np.float32)
    else:
        panels=[c for c in doc['cuboids'] if c['name'].startswith('shelf_')]
        lo=np.min([np.subtract(c['pose'][:3],np.divide(c['dims'],2)) for c in panels],0)
        hi=np.max([np.add(c['pose'][:3],np.divide(c['dims'],2)) for c in panels],0)
        cupboard=np.stack([lo,hi],1).astype(np.float32)
    compartments={c['name']:np.array([c[k] for k in 'xyz'],np.float32) for c in doc['compartments']}
    return box,cupboard,compartments


def inside(pos,box):
    box=torch.as_tensor(box,device=pos.device,dtype=pos.dtype)
    return ((pos>=box[:,0])&(pos<=box[:,1])).all(-1)


def clearance_speed(obstacle,self_distance):
    """Original physical speed components, min-combined: clip(d, 0.005, 0.05) / 0.05 in [0.1, 1]."""
    return torch.minimum(obstacle.clamp(OFFSET,MARGIN)/MARGIN,(SELF_SCALE*self_distance).clamp(OFFSET,MARGIN)/MARGIN)


class ClearanceGate:
    """Frozen release geometry without orientation or workspace rule.

    Used ONLY to reject offline local edges and to pick bank seeds. Evaluation uses the unchanged
    release scorer at its denser 0.001-radian resolution, with the tilt gate disabled.
    """
    def __init__(self, scene):
        import ur5e_plan_eval_multi as planner
        import physical_geometry as geom
        from curobo.geom.types import WorldConfig
        self.planner, self.geom = planner, geom
        scene_doc = yaml.safe_load(Path(scene).read_text())
        _, self.robot = planner.setup_robot_world('/workspace/ur5e_arm.yml', WorldConfig.from_dict(scene_doc), 'cuda:0')
        self.centers, self.half = geom.cuboid_half_extents(geom.load_scene_cuboids(scene), 'cuda:0')
        rcfg = yaml.safe_load(Path('/workspace/ur5e_arm.yml').read_text())['robot_cfg']['kinematics']
        self.mask, _ = geom.build_self_collision_mask('/workspace/' + geom.SPHERE_YML_PATH, rcfg.get('self_collision_ignore', {}), geom.COLLISION_LINK_NAMES, device='cuda:0')
        n = np.load(RELEASE / 'configs/joint_normalization.npz')
        self.center = torch.tensor(n['q_center'], device='cuda:0')
        self.range = torch.tensor(n['q_range'], device='cuda:0')

    @torch.no_grad()
    def states(self, q):
        rad = q * self.range + self.center
        sp = self.robot.get_state(rad).link_spheres_tensor
        do = self.geom.sphere_cuboid_distance(sp[:, :, :3], sp[:, :, 3], self.centers, self.half)
        ds = self.geom.sphere_sphere_distance(sp[:, :, :3], sp[:, :, 3], self.mask)
        return ((do >= .005) & (ds >= .002)).clone() & (q.abs() <= .5).all(-1)

    def edges(self, source_norm, successor_norm, in_domain, **kwargs):
        # Same densification as experiments/from_source/geometry.py ExactLocalGate.edges.
        shape = in_domain.shape
        starts = np.broadcast_to(source_norm[:, None, :], successor_norm.shape).reshape(-1, 6)
        ends = successor_norm.reshape(-1, 6)
        valid = np.flatnonzero(in_domain.reshape(-1))
        out = np.zeros(len(starts), dtype=bool)
        qr = self.range.cpu().numpy()
        counts = np.maximum(1, np.ceil(np.abs((ends[valid] - starts[valid]) * qr).max(1) / .02).astype(int))
        for steps in np.unique(counts):
            ids = valid[counts == steps]
            chunk = max(1, 8192 // (int(steps) + 1))
            alpha = torch.linspace(0, 1, int(steps) + 1, device='cuda:0')
            for begin in range(0, len(ids), chunk):
                rows = ids[begin:begin + chunk]
                q = torch.as_tensor(starts[rows].copy(), device='cuda:0')
                z = torch.as_tensor(ends[rows].copy(), device='cuda:0')
                points = q[:, None] + alpha[None, :, None] * (z - q)[:, None]
                out[rows] = self.states(points.reshape(-1, 6)).reshape(len(rows), -1).all(1).cpu().numpy()
        return out.reshape(shape)


class ClearanceSpeed:
    """Physical clearance speed under the scene; the label recipe of generate.py, recomputed."""
    def __init__(self, gate):
        self.gate=gate

    @torch.no_grad()
    def tasks(self, q):
        """Both SpeedField heads get the same clearance speed (N, 2); also the grasp point."""
        g=self.gate;state=g.robot.get_state(q*g.range+g.center);xyz=state.ee_pose.position.clone();sp=state.link_spheres_tensor
        do=g.geom.sphere_cuboid_distance(sp[:,:,:3],sp[:,:,3],g.centers,g.half,ignore_sphere_indices=g.geom.OBSTACLE_IGNORE_SPHERE_INDICES)
        ds=g.geom.sphere_sphere_distance(sp[:,:,:3],sp[:,:,3],g.mask)
        speed=clearance_speed(do,ds)
        return torch.stack([speed,speed],-1),xyz

    @torch.no_grad()
    def __call__(self,q):
        return self.tasks(q)[0][:,0]
