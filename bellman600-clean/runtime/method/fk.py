"""Differentiable position-only URDF FK for training routing derivatives."""

import math
import xml.etree.ElementTree as ET
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


class PositionFK(torch.nn.Module):
    def __init__(self):
        super().__init__()
        r = ET.parse(ROOT / "assets/runtime/robot/ur5e_robotiq_2f_140.urdf").getroot()
        bychild = {j.find("child").get("link"): j for j in r.findall("joint")}
        chain = []
        link = "grasp_frame"
        while link != "base_link":
            j = bychild[link]
            chain.append(j)
            link = j.find("parent").get("link")
        self.indices = []
        for i, j in enumerate(reversed(chain)):
            origin = j.find("origin")
            xyz = (
                [float(x) for x in origin.get("xyz", "0 0 0").split()]
                if origin is not None
                else [0.0] * 3
            )
            roll, pitch, yaw = (
                [float(x) for x in origin.get("rpy", "0 0 0").split()]
                if origin is not None
                else [0.0] * 3
            )
            cr, sr, cp, sp, cy, sy = (
                math.cos(roll),
                math.sin(roll),
                math.cos(pitch),
                math.sin(pitch),
                math.cos(yaw),
                math.sin(yaw),
            )
            t = torch.eye(4)
            t[:3, :3] = torch.tensor(
                [
                    [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                    [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                    [-sp, cp * sr, cp * cr],
                ]
            )
            t[:3, 3] = torch.tensor(xyz)
            self.register_buffer("origin_" + str(i), t)
            idx = JOINTS.index(j.get("name")) if j.get("name") in JOINTS else -1
            if idx >= 0 and j.find("axis").get("xyz") != "0 0 1":
                raise ValueError("Unsupported joint axis")
            if idx < 0 and j.get("type") != "fixed":
                raise ValueError("Unexpected active FK joint")
            self.indices.append(idx)

    def forward(self, q):
        t = torch.eye(4, device=q.device, dtype=q.dtype).expand(len(q), 4, 4)
        for i, idx in enumerate(self.indices):
            t = t @ getattr(self, "origin_" + str(i)).to(q.dtype)
            if idx >= 0:
                c = q[:, idx].cos()
                s = q[:, idx].sin()
                zero = c * 0
                one = zero + 1
                rot = torch.stack(
                    [
                        c,
                        -s,
                        zero,
                        zero,
                        s,
                        c,
                        zero,
                        zero,
                        zero,
                        zero,
                        one,
                        zero,
                        zero,
                        zero,
                        zero,
                        one,
                    ],
                    dim=-1,
                ).reshape(-1, 4, 4)
                t = t @ rot
        return t[:, :3, 3]
