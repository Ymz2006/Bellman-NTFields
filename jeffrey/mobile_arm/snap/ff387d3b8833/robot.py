"""The mobile manipulator: the SE(2) rectangle robot of the local maze / house studies with a scaled UR5e on top (9 DOF).

Arm = the repository's UR5e (runtime/assets/runtime/robot/ur5e.urdf joint origins, the same as cuRobo's ur_description; 6
revolute joints) with the 28 collision spheres of its six links from runtime/vendor/planner/spheres_ur5e_arm.yml (the gripper
and tool0 are left off). Everything is scaled by ARM_SCALE cells per UR5e metre and mounted at the centre of the base's top
face, base_link x along the base heading. The base is the 0.5 x 0.2-cell rectangle (box of height BASE_H cells on the floor).

Collision model (all distances in cells):
    environment   the floor plan's walls are vertical prisms: the base rectangle by the 9 x 5 point lattice of the house studies,
                  each arm sphere by the plan's signed distance at its centre minus its radius
    internal      the arm against the floor (z - r), against the base box (exact box distance - r; the shoulder link sits on the
                  base and is exempt from both), and against itself on cuRobo's UR5e self-collision pairs (ur5e_arm.yml
                  self_collision_ignore: shoulder vs forearm / wrists, upper arm vs wrists)

fk() is checked against cuRobo's kinematics in check_curobo() (max sphere-centre error 1e-6 m).
"""
import math
import torch

# --- UR5e (runtime/assets/runtime/robot/ur5e.urdf) -----------------------------------------------------------------------------
# base_link -> base_link_inertia: rpy (0, 0, pi); then per joint: origin xyz, rpy, axis z
JOINTS = [((0.0, 0.0, 0.1625), (0.0, 0.0, 0.0)),
          ((0.0, 0.0, 0.0), (1.570796327, 0.0, 0.0)),
          ((-0.425, 0.0, 0.0), (0.0, 0.0, 0.0)),
          ((-0.3922, 0.0, 0.1333), (0.0, 0.0, 0.0)),
          ((0.0, -0.0997, -2.044881182297852e-11), (1.570796327, 0.0, 0.0)),
          ((0.0, 0.0996, -2.042830148012698e-11), (1.570796326589793, 3.141592653589793, 3.141592653589793))]
LINKS = ['shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link']
# runtime/vendor/planner/spheres_ur5e_arm.yml, the six arm links (centre in the link frame, radius), metres
SPHERES = {
    'shoulder_link': [((0.0, 0.0, 0.0), 0.1), ((0.0, 0.0, -0.088), 0.1)],
    'upper_arm_link': [((-0.416, -0.0, 0.143), 0.078), ((-0.015, 0.0, 0.134), 0.077), ((-0.14, 0.0, 0.138), 0.062), ((-0.285, -0.001, 0.139), 0.061),
                       ((-0.376, 0.001, 0.138), 0.077), ((-0.222, 0.001, 0.139), 0.061), ((-0.055, 0.008, 0.14), 0.07), ((-0.001, -0.002, 0.143), 0.08)],
    'forearm_link': [((-0.01, 0.002, 0.031), 0.072), ((-0.387, 0.0, 0.014), 0.057), ((-0.121, -0.0, 0.006), 0.057), ((-0.206, 0.001, 0.007), 0.057),
                     ((-0.312, -0.001, 0.006), 0.056), ((-0.057, 0.003, 0.008), 0.065), ((-0.266, 0.0, 0.006), 0.057), ((-0.397, -0.001, -0.018), 0.052),
                     ((-0.164, -0.0, 0.007), 0.057)],
    'wrist_1_link': [((-0.0, 0.0, -0.009), 0.047), ((-0.0, 0.0, -0.052), 0.047), ((-0.002, 0.027, -0.001), 0.045), ((0.001, -0.01, 0.0), 0.046)],
    'wrist_2_link': [((0.0, -0.01, -0.001), 0.047), ((0.0, 0.008, -0.001), 0.047), ((0.001, -0.001, -0.036), 0.047), ((0.001, -0.03, -0.0), 0.047)],
    'wrist_3_link': [((0.001, 0.001, -0.029), 0.043)],
}
# cuRobo's checked self-collision link pairs for the six arm links (all pairs minus ur5e_arm.yml's self_collision_ignore)
SELF_PAIRS = [('shoulder_link', 'forearm_link'), ('shoulder_link', 'wrist_1_link'), ('shoulder_link', 'wrist_2_link'), ('shoulder_link', 'wrist_3_link'),
              ('upper_arm_link', 'wrist_1_link'), ('upper_arm_link', 'wrist_2_link'), ('upper_arm_link', 'wrist_3_link')]

# --- the mobile manipulator (cells) --------------------------------------------------------------------------------------------
BASE_L, BASE_W = 0.5, 0.2            # base rectangle (the house studies' robot), cells
BASE_H = 0.1                         # base box height, cells (the arm is mounted on its top face)
ARM_SCALE = 0.8                      # cells per UR5e metre: a UR5e at 1/5 size (0.2 m per UR5e metre) at 0.25 m per cell
# joint box (radians): full turns for the pan / elbow / wrists, shoulder lift kept above the shoulder's horizontal plane
JOINT_LO = (-math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi)
JOINT_HI = (math.pi, 0.0, math.pi, math.pi, math.pi, math.pi)


def _rot(axis, a):
    c, s = math.cos(a), math.sin(a)
    if axis == 'x':
        return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float64)
    if axis == 'y':
        return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float64)
    return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float64)


def _origin(xyz, rpy):
    """URDF origin: translation xyz, rotation Rz(yaw) Ry(pitch) Rx(roll)."""
    T = torch.eye(4, dtype=torch.float64)
    T[:3, :3] = _rot('z', rpy[2]) @ _rot('y', rpy[1]) @ _rot('x', rpy[0])
    T[:3, 3] = torch.tensor(xyz, dtype=torch.float64)
    return T


class Arm:
    """Batched forward kinematics of the scaled UR5e; sphere centres in the base frame of the mobile robot (cells)."""

    def __init__(self, device='cuda:0', dtype=torch.float32, scale=ARM_SCALE, mount_z=BASE_H):
        self.device, self.dtype, self.scale, self.mount_z = device, dtype, float(scale), float(mount_z)
        bi = torch.eye(4, dtype=torch.float64)
        bi[:3, :3] = _rot('z', math.pi)
        self.T0 = bi.to(device, dtype)                                                     # base_link -> base_link_inertia
        self.O = torch.stack([_origin(x, r) for x, r in JOINTS]).to(device, dtype)          # (6, 4, 4) fixed joint origins
        c, r, link = [], [], []
        for k, name in enumerate(LINKS):
            for cen, rad in SPHERES[name]:
                c.append(cen)
                r.append(rad)
                link.append(k)
        self.centre_link = torch.tensor(c, device=device, dtype=dtype)                    # (S, 3) metres, link frame
        self.radius_m = torch.tensor(r, device=device, dtype=dtype)
        self.link = torch.tensor(link, device=device)
        self.radius = self.radius_m * self.scale                                           # cells
        self.S = len(r)
        self.not_shoulder = self.link > 0
        ia, ib = [], []
        for la, lb in SELF_PAIRS:
            ka, kb = LINKS.index(la), LINKS.index(lb)
            for i in torch.nonzero(self.link == ka).squeeze(1).tolist():
                for j in torch.nonzero(self.link == kb).squeeze(1).tolist():
                    ia.append(i)
                    ib.append(j)
        self.pair_a = torch.tensor(ia, device=device)
        self.pair_b = torch.tensor(ib, device=device)
        self.pair_r = self.radius[self.pair_a] + self.radius[self.pair_b]
        self.lo = torch.tensor(JOINT_LO, device=device, dtype=dtype)
        self.hi = torch.tensor(JOINT_HI, device=device, dtype=dtype)

    def link_frames(self, qj):
        """(N, 6, 4, 4) frames of the six links in UR5e base_link coordinates (metres)."""
        n = len(qj)
        c, s = torch.cos(qj), torch.sin(qj)
        Rz = torch.zeros(n, 6, 4, 4, device=qj.device, dtype=qj.dtype)
        Rz[..., 0, 0], Rz[..., 0, 1], Rz[..., 1, 0], Rz[..., 1, 1] = c, -s, s, c
        Rz[..., 2, 2] = 1
        Rz[..., 3, 3] = 1
        T = self.T0.to(qj.dtype).expand(n, 4, 4)
        out = []
        for k in range(6):
            T = T @ self.O[k].to(qj.dtype) @ Rz[:, k]
            out.append(T)
        return torch.stack(out, 1)

    def spheres_m(self, qj):
        """Sphere centres (N, S, 3) in UR5e base_link coordinates, metres."""
        F = self.link_frames(qj)
        Fs = F[:, self.link]                                                               # (N, S, 4, 4)
        return (Fs[..., :3, :3] @ self.centre_link.to(qj.dtype)[None, :, :, None]).squeeze(-1) + Fs[..., :3, 3]

    def spheres(self, qj):
        """Sphere centres (N, S, 3) in the mobile base frame (cells): x along the heading, z up from the floor."""
        p = self.spheres_m(qj) * self.scale
        return p + torch.tensor([0.0, 0.0, self.mount_z], device=qj.device, dtype=qj.dtype)

    def internal(self, qj, parts=False):
        """Internal clearance (N,) in cells: floor, base box, self pairs; with parts=True the three minima."""
        p = self.spheres(qj)
        r = self.radius.to(qj.dtype)
        ns = self.not_shoulder
        floor = (p[:, ns, 2] - r[ns]).amin(1)
        half = torch.tensor([BASE_L / 2, BASE_W / 2, BASE_H / 2], device=qj.device, dtype=qj.dtype)
        rel = (p[:, ns] - torch.tensor([0.0, 0.0, BASE_H / 2], device=qj.device, dtype=qj.dtype)).abs() - half
        box = torch.sqrt((rel.clamp(min=0) ** 2).sum(-1) + 1e-12) + rel.amax(-1).clamp(max=0)
        base = (box - r[ns]).amin(1)
        dd = torch.linalg.vector_norm(p[:, self.pair_a] - p[:, self.pair_b], dim=-1) - self.pair_r.to(qj.dtype)
        selfd = dd.amin(1)
        d = torch.minimum(torch.minimum(floor, base), selfd)
        return (d, floor, base, selfd) if parts else d


def check_curobo(n=2048, seed=0, device='cuda:0'):
    """Max |sphere centre - cuRobo's| (metres) over random joint vectors, and the radii check."""
    from pathlib import Path
    import yaml
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
    from curobo.types.base import TensorDeviceType
    root = Path(__file__).resolve().parents[2]
    yml = yaml.safe_load(open(root / 'runtime/vendor/planner/spheres_ur5e_arm.yml'))['collision_spheres']
    for name in LINKS:
        assert [(tuple(s['center']), s['radius']) for s in yml[name]] == [(tuple(map(float, c)), r) for c, r in SPHERES[name]], f'{name} spheres differ from the yml'
    td = TensorDeviceType(device=torch.device(device))
    from curobo.cuda_robot_model.cuda_robot_generator import CudaRobotGeneratorConfig
    gen_cfg = CudaRobotGeneratorConfig(base_link='base_link', ee_link='wrist_3_link', tensor_args=td,
                                       urdf_path=str(root / 'runtime/assets/runtime/robot/ur5e.urdf'), asset_root_path=str(root / 'runtime/assets/runtime/robot'),
                                       collision_link_names=LINKS, collision_spheres={k: yml[k] for k in LINKS}, collision_sphere_buffer=0.0,
                                       self_collision_buffer={k: 0.0 for k in LINKS}, self_collision_ignore={})
    model = CudaRobotModel(CudaRobotModelConfig.from_config(gen_cfg))
    g = torch.Generator(device).manual_seed(seed)
    q = (torch.rand(n, 6, generator=g, device=device) * 2 - 1) * math.pi
    st = model.get_state(q.float())
    ref = st.link_spheres_tensor[..., :3].double()
    rr = st.link_spheres_tensor[0, :, 3].double()
    arm = Arm(device, torch.float64)
    mine = arm.spheres_m(q.double())
    # cuRobo orders spheres by link as listed; same order here
    return dict(max_centre_error_m=float((mine - ref).abs().max()), max_radius_error_m=float((arm.radius_m - rr).abs().max()), spheres=int(arm.S))
