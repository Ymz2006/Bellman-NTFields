"""Original batched offline geometry gate, with an explicit scene argument."""
import sys
from pathlib import Path
sys.path.insert(0,'/clean/method')
from common import RELEASE
sys.path.insert(0,'/workspace')
import numpy as np
import torch
class ExactLocalGate:
    """Frozen release geometry, 20-degree OR, no workspace constraint.

    Used ONLY to reject offline local edges. Evaluation uses the unchanged
    release scorer at its denser 0.001-radian resolution.
    """
    def __init__(self, scene):
        import yaml
        import ur5e_plan_eval_multi as planner
        import physical_geometry as geom
        from curobo.geom.types import WorldConfig
        self.planner, self.geom = planner, geom
        scene_doc = yaml.safe_load(scene.read_text())
        self.zones = planner.load_zone_config(str(scene), 'auto')[1]
        self.axes = planner.load_task_axes(str(scene))
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
        st = self.robot.get_state(rad)
        sp = st.link_spheres_tensor
        do = self.geom.sphere_cuboid_distance(sp[:, :, :3], sp[:, :, 3], self.centers, self.half)
        ds = self.geom.sphere_sphere_distance(sp[:, :, :3], sp[:, :, 3], self.mask)
        # Clone values before another cuRobo call reuses its internal buffers.
        physical = ((do >= .005) & (ds >= .002)).clone()
        cup, pour, _, pos = self.planner._zone_tilts_batch(rad, self.robot, return_pos=True, task_axes=self.axes)
        cm, pm, ov = self.planner.corner_masks_torch(pos[:, :2], self.zones, assign_outside=True)
        orient = torch.where(ov, (cup <= 20) | (pour <= 20), torch.where(cm, cup <= 20, pour <= 20))
        return physical & orient & (q.abs() <= .5).all(-1)

    def edges(self, source_norm, successor_norm, in_domain, **kwargs):
        shape = in_domain.shape
        starts = np.broadcast_to(source_norm[:, None, :], successor_norm.shape).reshape(-1, 6)
        ends = successor_norm.reshape(-1, 6)
        valid = np.flatnonzero(in_domain.reshape(-1))
        out = np.zeros(len(starts), dtype=bool)
        qr = self.range.cpu().numpy()
        # Same maximum per-joint step as historical offline local-edge veto.
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
