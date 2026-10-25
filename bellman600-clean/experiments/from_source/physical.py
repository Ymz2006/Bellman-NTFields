"""Two physical task speeds under the selected scene, matching original labels."""
import torch
from ur5e_task_axes import tilt_deg_from_quat

class PhysicalSpeed:
    def __init__(self, gate):
        self.gate=gate
        self.robot=gate.robot
        self.geom=gate.geom
        self.centers,self.half=gate.centers,gate.half
        self.mask=gate.mask
        self.axes=gate.axes
        self.tilt=tilt_deg_from_quat

    @torch.no_grad()
    def tasks(self, q):
        state = self.robot.get_state(q * self.gate.range + self.gate.center)
        xyz = state.ee_pose.position.clone()
        quat = state.ee_pose.quaternion.clone()
        sp = state.link_spheres_tensor
        do = self.geom.sphere_cuboid_distance(
            sp[:, :, :3],
            sp[:, :, 3],
            self.centers,
            self.half,
            ignore_sphere_indices=self.geom.OBSTACLE_IGNORE_SPHERE_INDICES,
        )
        ds = self.geom.sphere_sphere_distance(sp[:, :, :3], sp[:, :, 3], self.mask)
        physical = torch.minimum(
            do.clamp(0.005, 0.05) / 0.05, (0.5 * ds).clamp(0.005, 0.05) / 0.05
        )
        tilts = torch.stack(
            [
                self.tilt(quat, self.axes[t]["local_axis"], self.axes[t]["world_axis"])
                for t in ["cup", "pour"]
            ],
            dim=-1,
        )
        tilt_speed = 1 - 0.9 * ((tilts - 10) / 10).clamp(0, 1)
        return torch.minimum(physical[:, None], tilt_speed), xyz


    @torch.no_grad()
    def __call__(self,q):
        task,xyz=self.tasks(q)
        u=((xyz[:,1]+.1)/.2).clamp(0,1)
        w=u.square()*(3-2*u)
        return w*task[:,0]+(1-w)*task[:,1]
