"""Export the release collision model of the cabinet arm (UR5e + Robotiq 2F-140, cuRobo spheres) for arm.py's own kinematics.

Runs in the container layout (/clean, /repo, /workspace; jeffrey/segment_dp/launch.py). Uses the label code itself
(experiments/ntrl_cabinet/data.py Clearance = data_generation/source geometry: cuRobo's link spheres, the scene's cuboids, the self mask)
and the scorer's functions (collision_utils), and writes robot_model.npz:

    q_fit (K, 6) rad, spheres_fit (K, S, 4)     sphere centres and radii as cuRobo reports them at K configurations (the first is q = 0)
    ee_fit (K, 3)                               grasp-frame position
    link (S,)                                   index into the label code's COLLISION_LINK_NAMES for every sphere
    mask (S, S)                                 the self-collision ignore mask (True = ignored)
    cuboid_centre, cuboid_half (C, 3)           the scene's cuboids
    box, cupboard (3, 2), compartments          the scene's grasp-point boxes
    q_range, q_min, q_max                       the release joint normalization
    q_ref (N, 6), d_obs, d_obs_all, d_self, ee_ref    reference values at N uniform configurations: obstacle distance without the two
                                                shoulder spheres (the label), with every sphere (the scorer), self distance, grasp point
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, '/clean/method')
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/clean/experiments/ntrl_cabinet')
sys.path.insert(0, '/clean/experiments/cabinet')
import data as ND  # noqa: E402  (experiments/ntrl_cabinet/data.py: Clearance, geom)
from geometry import scene_boxes  # noqa: E402
from collision_utils import sphere_cuboid_distance, sphere_sphere_distance  # noqa: E402

NORM = Path('/repo/configs/joint_normalization.npz')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ref', type=int, default=200_000)
    p.add_argument('--fit', type=int, default=64)
    p.add_argument('--seed', type=int, default=20261120)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    dev = 'cuda:0'
    geom = ND.geom
    C = ND.Clearance(a.scene, 'release', dev)
    n = np.load(NORM)
    qmax = torch.tensor(n['q_max'], device=dev)
    g = torch.Generator(dev).manual_seed(a.seed)
    q_fit = (torch.rand(a.fit, 6, generator=g, device=dev) * 2 - 1) * qmax
    q_fit[0] = 0
    with torch.no_grad():
        st = C.robot.get_state(q_fit)
        sp_fit = st.link_spheres_tensor.clone()
        ee_fit = st.ee_pose.position.clone()
        q_ref = (torch.rand(a.ref, 6, generator=g, device=dev) * 2 - 1) * qmax
        d_obs, d_all, d_self, ee = [], [], [], []
        for i in range(0, a.ref, 20000):
            s = C.robot.get_state(q_ref[i:i + 20000])
            sp = s.link_spheres_tensor
            d_obs.append(geom.sphere_cuboid_distance(sp[:, :, :3], sp[:, :, 3], C.centers, C.half, ignore_sphere_indices=geom.OBSTACLE_IGNORE_SPHERE_INDICES))
            d_all.append(sphere_cuboid_distance(sp[:, :, :3], sp[:, :, 3], C.centers, C.half))
            d_self.append(sphere_sphere_distance(sp[:, :, :3], sp[:, :, 3], C.mask))
            ee.append(s.ee_pose.position.clone())
        d_obs, d_all, d_self, ee = torch.cat(d_obs), torch.cat(d_all), torch.cat(d_self), torch.cat(ee)
        label = C(q_ref)
        assert torch.equal(label, torch.minimum(d_obs, ND.SELF_SCALE * d_self)), 'the label is not min(obstacle, 0.5 self)'
    import yaml
    spheres = yaml.safe_load(open('/workspace/' + geom.SPHERE_YML_PATH))['collision_spheres']
    link = []
    for k, name in enumerate(geom.COLLISION_LINK_NAMES):
        link += [k] * len(spheres.get(name, []))
    assert len(link) == sp_fit.shape[1], (len(link), sp_fit.shape)
    box, cupboard, comps = scene_boxes(a.scene)
    np.savez(a.output / 'robot_model.npz', q_fit=q_fit.cpu().numpy(), spheres_fit=sp_fit.cpu().numpy(), ee_fit=ee_fit.cpu().numpy(), link=np.array(link),
             mask=C.mask.cpu().numpy(), cuboid_centre=C.centers.cpu().numpy(), cuboid_half=C.half.cpu().numpy(), box=box, cupboard=cupboard,
             compartment_names=np.array(list(comps)), compartments=np.stack([comps[k] for k in comps]),
             q_range=n['q_range'], q_min=n['q_min'], q_max=n['q_max'], q_ref=q_ref.cpu().numpy(), d_obs=d_obs.cpu().numpy(), d_obs_all=d_all.cpu().numpy(),
             d_self=d_self.cpu().numpy(), ee_ref=ee.cpu().numpy(), link_names=np.array(geom.COLLISION_LINK_NAMES),
             ignore=np.array(geom.OBSTACLE_IGNORE_SPHERE_INDICES))
    rep = dict(complete=True, scene=str(a.scene), spheres=int(sp_fit.shape[1]), cuboids=int(len(C.centers)), ref=a.ref,
               self_pairs_checked=int((~C.mask).sum().item() // 2), valid_share=float(((d_all >= 0.005) & (d_self >= 0.002)).float().mean()),
               radius_range=[float(sp_fit[0, :, 3].min()), float(sp_fit[0, :, 3].max())],
               shoulder_obstacle_distance=float(sphere_cuboid_distance(sp_fit[:, :2, :3], sp_fit[:, :2, 3], C.centers, C.half).min()))
    (a.output / 'export.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep), flush=True)


if __name__ == '__main__':
    main()
