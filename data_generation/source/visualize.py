"""Frozen FK initialization extracted from the source sampler."""
from pathlib import Path
import torch,yaml

def init_curobo(robot_cfg_path: str, urdf_path: str, device: str = "cuda:0"):
    """Initialize cuRobo robot model. Returns (cfg_dict, robot_cfg, robot_model, tensor_args)."""
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel

    tensor_args = TensorDeviceType(device=torch.device(device))

    with open(robot_cfg_path) as f:
        cfg_dict = yaml.safe_load(f)
    p = Path(urdf_path).parent
    cfg_dict["robot_cfg"]["kinematics"]["urdf_path"] = urdf_path
    cfg_dict["robot_cfg"]["kinematics"]["asset_root_path"] = str(p)

    robot_cfg = RobotConfig.from_dict(cfg_dict, tensor_args)
    robot_model = CudaRobotModel(robot_cfg.kinematics)

    return cfg_dict, robot_cfg, robot_model, tensor_args

