"""The release arm's collision model as an `export` file for jeffrey/lbs_arm/arm12.build_model, without the container.

jeffrey/lbs_arm/export_robot.py needs cuRobo and the container layout, which this machine does not have. Its output for the arm itself
(sphere centres per link frame, radii, self-collision mask, the two fixed shoulder spheres, grasp point, joint limits) was already
extracted into jeffrey/real_world/robot_geometry.npz (from registered export 91c5e157, fit error 5e-7 m). This script turns that file
back into the keys arm12.build_model reads (q_fit, spheres_fit, ee_fit, link, mask, ignore, q_range, q_min, q_max): the sphere centres
at K configurations are computed with jeffrey/real_world/robot.py's kinematics in float64, so arm.py's own fit reproduces `local`.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/arm_placement/armexport.py        # writes jeffrey/arm_placement/arm_export.npz
"""
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'jeffrey/real_world'))
import robot as RW  # noqa: E402


def main():
    g = np.load(RW.GEOMETRY, allow_pickle=False)
    n = np.load(ROOT / 'runtime/configs/joint_normalization.npz')
    assert np.array_equal(g['q_min'], n['q_min']) and np.array_equal(g['q_max'], n['q_max'])
    arms = RW.Arms([(0, 0, 0, 1, 0, 0, 0)], [], 'cpu', torch.float64)
    gen = torch.Generator().manual_seed(20261123)
    q = (torch.rand(64, 6, generator=gen, dtype=torch.float64) * 2 - 1) * torch.tensor(n['q_max'], dtype=torch.float64)
    q[0] = 0
    p, ee = arms.local_spheres(q)
    sp = torch.cat([p, arms.radius[None, :, None].expand(len(q), -1, 1)], 2)
    out = HERE / 'arm_export.npz'
    np.savez(out, q_fit=q.numpy(), spheres_fit=sp.numpy(), ee_fit=ee.numpy(), link=g['link'], mask=g['mask'], ignore=g['ignore'],
             q_range=n['q_range'], q_min=n['q_min'], q_max=n['q_max'])
    print(out, {k: v.shape for k, v in np.load(out).items()})


if __name__ == '__main__':
    main()
