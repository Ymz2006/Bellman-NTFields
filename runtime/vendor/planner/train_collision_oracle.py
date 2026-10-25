"""Training-time FK + collision + tilt oracle for arbitrary C-space points.

Used by the path-level self-supervision loss to label self-rollout points as
violating (collision or over-tilt) or safe. All outputs are detached: gradients
must flow only through the tau network, never through collision geometry.
"""
import numpy as np
import torch
import yaml

from collision_utils import (
    build_self_collision_mask,
    check_collision_full,
    load_cuboid_geometry,
)
from ur5e_task_axes import load_task_axes, task_axis_arrays, tilt_deg_from_quat
from visualize import init_curobo

ROBOT_CFG_PATH = "ur5e_arm.yml"
URDF_PATH = "/opt/ur5e_description/ur5e_robotiq_2f_140.urdf"
SPHERE_YML_PATH = "spheres_ur5e_arm.yml"
COLLISION_LINK_NAMES = [
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
    "tool0",
    "robotiq_arg2f_base_link",
    "left_outer_knuckle",
    "left_inner_knuckle",
    "left_outer_finger",
    "left_inner_finger",
    "left_inner_finger_pad",
    "right_outer_knuckle",
    "right_inner_knuckle",
    "right_outer_finger",
    "right_inner_finger",
    "right_inner_finger_pad",
]


class TrainCollisionOracle:
    def __init__(self, scene_yml, norm_file, cup_tilt_max=15.0, pour_tilt_max=90.0,
                 cup_tilt_soft=10.0, pour_tilt_soft=10.0, orient_margin=0.05,
                 device="cuda:0"):
        self.device = device
        self.cup_tilt_max = cup_tilt_max
        self.pour_tilt_max = pour_tilt_max
        self.cup_tilt_soft = cup_tilt_soft
        self.pour_tilt_soft = pour_tilt_soft
        self.orient_margin = orient_margin
        _, _, self.robot_model, _ = init_curobo(ROBOT_CFG_PATH, URDF_PATH, device)
        self.cuboid_centers, self.cuboid_half = load_cuboid_geometry(scene_yml, device)
        with open(ROBOT_CFG_PATH) as f:
            robot_cfg = yaml.safe_load(f)
        self_coll_ignore = robot_cfg["robot_cfg"]["kinematics"].get(
            "self_collision_ignore", {})
        self.self_coll_mask, _ = build_self_collision_mask(
            SPHERE_YML_PATH, self_coll_ignore, COLLISION_LINK_NAMES,
            device=device)
        self.task_axes = load_task_axes(scene_yml)
        self.cup_local_axis, self.cup_world_axis = task_axis_arrays(self.task_axes, "cup")
        self.pour_local_axis, self.pour_world_axis = task_axis_arrays(self.task_axes, "pour")
        nz = np.load(norm_file)
        self.q_center = torch.tensor(nz["q_center"], dtype=torch.float32, device=device)
        self.q_range = torch.tensor(nz["q_range"], dtype=torch.float32, device=device)

    def denorm(self, q_norm):
        return q_norm * self.q_range + self.q_center

    @torch.no_grad()
    def query(self, q_norm):
        """q_norm: (M,6) normalized configs. Returns dict of (M,) detached tensors."""
        q_rad = self.denorm(q_norm.to(self.device))
        state = self.robot_model.get_state(q_rad)
        _, d_result = check_collision_full(
            self.robot_model, q_rad, self.cuboid_centers, self.cuboid_half,
            torso_cuboid_centers=None, torso_cuboid_half=None,
            self_collision_mask=self.self_coll_mask,
            torso_ignore_sphere_indices=None,
            margin=0.0, device=self.device)
        d_obs = d_result["obstacle"]                 # (M,)
        d_self = d_result["self"]                    # (M,)
        d_min = torch.minimum(d_obs, d_self)
        quat = state.ee_pose.quaternion              # (M,4) [w,x,y,z]
        tilt_cup = tilt_deg_from_quat(quat, self.cup_local_axis, self.cup_world_axis)
        tilt_pour = tilt_deg_from_quat(quat, self.pour_local_axis, self.pour_world_axis)
        cup_excess = (tilt_cup - self.cup_tilt_max).clamp(min=0.0)
        pour_excess = (tilt_pour - self.pour_tilt_max).clamp(min=0.0)
        d_orient_cup = (
            self.orient_margin
            * (1.0 - cup_excess / max(self.cup_tilt_soft, 1e-6))
        ).clamp(min=-self.orient_margin)
        d_orient_pour = (
            self.orient_margin
            * (1.0 - pour_excess / max(self.pour_tilt_soft, 1e-6))
        ).clamp(min=-self.orient_margin)
        ee_pos = state.ee_pose.position
        ee_x = ee_pos[:, 0]                          # (M,)
        ee_y = ee_pos[:, 1]                          # (M,)
        return {
            "d_obs": d_obs.detach(),
            "d_self": d_self.detach(),
            "d_collision": d_min.detach(),
            "tilt_cup": tilt_cup.detach(),
            "tilt_pour": tilt_pour.detach(),
            "d_orient_cup": d_orient_cup.detach(),
            "d_orient_pour": d_orient_pour.detach(),
            "ee_x": ee_x.detach(),
            "ee_y": ee_y.detach(),
        }
