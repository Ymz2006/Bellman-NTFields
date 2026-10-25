import os
import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable, grad

try:
    from models.metric_arm.zone_router import load_dataset_zone_config
except ImportError:
    from metric_arm.zone_router import load_dataset_zone_config


def _pair_type_mask(pair_type, spec):
    spec = (spec or "").strip().lower()
    if not spec or spec in ("all", "*"):
        return np.ones_like(pair_type, dtype=bool)
    aliases = {
        "cup": 0, "cupping": 0, "0": 0,
        "pour": 1, "pouring": 1, "1": 1,
        "cross": 2, "handover": 2, "2": 2,
    }
    ids = []
    for raw in spec.replace(",", " ").split():
        if raw not in aliases:
            raise ValueError(f"unknown pair type '{raw}' in pair type spec {spec!r}")
        ids.append(aliases[raw])
    return np.isin(pair_type, ids)


class _numpy2dataset_multi(torch.utils.data.Dataset):
    def __init__(self, points, speed_cup, speed_pour, normal_cup, normal_pour,
                 ee_y, d_orient_cup, d_orient_pour, zone_id, pair_type=None,
                 d_obstacle=None, d_self=None,
                 ee_xy=None, tilt_grad=None, sample_weight=None,
                 handover_grad=None, fk_features=None):
        """Multi-task dataset: concatenates all fields into a single tensor.

        Legacy layout per row (55 columns total):
          sampled_points (14) | speed_cup (2) | speed_pour (2) |
          normal_cup (14) | normal_pour (14) |
          ee_y (2) | d_orient_cup (2) | d_orient_pour (2) | zone_id (2) |
          pair_type (1)

        Optional blocks are appended after pair_type:
          d_obstacle/d_self (4) if provided, ee_xy (4) for corner-table routing,
          tilt_grad (2*dim) if provided, sample_weight (1) if provided,
          handover_grad (dim) if provided, then fk_features (2*F) if provided.
        """
        points = Variable(Tensor(points))
        speed_cup = Variable(Tensor(speed_cup))
        speed_pour = Variable(Tensor(speed_pour))
        normal_cup = Variable(Tensor(normal_cup))
        normal_pour = Variable(Tensor(normal_pour))
        ee_y = Variable(Tensor(ee_y))
        d_orient_cup = Variable(Tensor(d_orient_cup))
        d_orient_pour = Variable(Tensor(d_orient_pour))
        zone_id = Variable(Tensor(zone_id))
        self.has_ee_xy = ee_xy is not None

        tensors = [points, speed_cup, speed_pour,
                   normal_cup, normal_pour,
                   ee_y, d_orient_cup, d_orient_pour, zone_id]

        if pair_type is not None:
            pair_type_t = Variable(Tensor(pair_type.astype(np.float32)).unsqueeze(1))
            tensors.append(pair_type_t)
        else:
            # Default: all cross-zone (pair_type=2) for backward compat
            tensors.append(torch.full((points.shape[0], 1), 2.0))

        if d_obstacle is not None and d_self is not None:
            tensors.append(Variable(Tensor(d_obstacle.astype(np.float32))))
            tensors.append(Variable(Tensor(d_self.astype(np.float32))))
            self.has_safety_distance = True
        else:
            self.has_safety_distance = False

        if ee_xy is not None:
            tensors.append(Variable(Tensor(ee_xy)))

        # R58U: optional clean tilt-descent target, appended LAST so existing
        # legacy COL_* indices remain unchanged. The training loop computes the
        # optional-block offsets from has_ee_xy / has_tilt_grad.
        if tilt_grad is not None:
            tensors.append(Variable(Tensor(tilt_grad)))
            self.has_tilt_grad = True
        else:
            self.has_tilt_grad = False

        if sample_weight is not None:
            tensors.append(Variable(Tensor(sample_weight.astype(np.float32)).unsqueeze(1)))
            self.has_sample_weight = True
        else:
            self.has_sample_weight = False

        if handover_grad is not None:
            tensors.append(Variable(Tensor(handover_grad.astype(np.float32))))
            self.has_handover_grad = True
            self.handover_grad_dim = handover_grad.shape[1]
        else:
            self.has_handover_grad = False
            self.handover_grad_dim = 0

        if fk_features is not None:
            tensors.append(Variable(Tensor(fk_features.astype(np.float32))))
            self.has_fk_features = True
            self.fk_feature_dim = fk_features.shape[1] // 2
        else:
            self.has_fk_features = False
            self.fk_feature_dim = 0

        self.data = torch.cat(tensors, dim=1)

    def send_device(self, device):
        self.data = self.data.to(device)

    def __getitem__(self, index):
        data = self.data[index]
        return data, index

    def __len__(self):
        return self.data.shape[0]


def DatabaseMulti(PATH):
    """Load multi-task training data.

    Expects the following .npy files in PATH:
      sampled_points.npy  (N, 14)  - normalized [q_start, q_goal] pairs
      speed_cupping.npy   (N, 2)   - cupping speed [start, goal]
      speed_pouring.npy   (N, 2)   - pouring speed [start, goal]
      normal_cupping.npy  (N, 14)  - cupping distance field gradient
      normal_pouring.npy  (N, 14)  - pouring distance field gradient
      ee_positions.npy    (N, 6)   - EE xyz [start(3), goal(3)] -> extract Y
      d_orient_cupping.npy (N, 2)  - cupping orientation distance
      d_orient_pouring.npy (N, 2)  - pouring orientation distance
      zone_id.npy         (N, 2)   - zone assignment per endpoint

    Returns:
      database with .data tensor of shape (N, 54)
    """
    points = np.load('{}/sampled_points.npy'.format(PATH))
    speed_cup = np.load('{}/speed_cupping.npy'.format(PATH))
    speed_pour = np.load('{}/speed_pouring.npy'.format(PATH))
    normal_cup = np.load('{}/normal_cupping.npy'.format(PATH))
    normal_pour = np.load('{}/normal_pouring.npy'.format(PATH))
    ee_positions = np.load('{}/ee_positions.npy'.format(PATH))
    d_orient_cup = np.load('{}/d_orient_cupping.npy'.format(PATH))
    d_orient_pour = np.load('{}/d_orient_pouring.npy'.format(PATH))
    zone_id = np.load('{}/zone_id.npy'.format(PATH))

    # Load pair_type if available
    pair_type_path = '{}/pair_type.npy'.format(PATH)
    pair_type = None
    if os.path.exists(pair_type_path):
        pair_type = np.load(pair_type_path)

    # R58U: load clean tilt-descent target if present (optional)
    tilt_grad_path = '{}/tilt_grad.npy'.format(PATH)
    tilt_grad = np.load(tilt_grad_path) if os.path.exists(tilt_grad_path) else None

    sample_weight_path = '{}/sample_weight.npy'.format(PATH)
    sample_weight = (
        np.load(sample_weight_path) if os.path.exists(sample_weight_path) else None
    )

    teacher_dir_path = '{}/teacher_dir.npy'.format(PATH)
    if os.path.exists(teacher_dir_path):
        raise RuntimeError(
            "teacher_dir.npy is a teacher-direction hack and is disabled by "
            f"reports/ur5e-no-hack-policy.md: {teacher_dir_path}")

    fk_features_path = '{}/fk_features.npy'.format(PATH)
    fk_features = (
        np.load(fk_features_path) if os.path.exists(fk_features_path) else None
    )

    d_obstacle_path = '{}/d_obstacle.npy'.format(PATH)
    d_self_path = '{}/d_self.npy'.format(PATH)
    if os.path.exists(d_obstacle_path) and os.path.exists(d_self_path):
        d_obstacle = np.load(d_obstacle_path)
        d_self = np.load(d_self_path)
    else:
        d_obstacle = None
        d_self = None

    handover_grad_path = '{}/handover_grad.npy'.format(PATH)
    handover_grad = (
        np.load(handover_grad_path) if os.path.exists(handover_grad_path) else None
    )

    if os.environ.get("R58_DROP_BASE_CROSS", "0") == "1":
        raise RuntimeError(
            "R58_DROP_BASE_CROSS depends on teacher_dir.npy and is disabled "
            "by the UR5e no-hack policy.")

    sample_weight_pair_types = os.environ.get("R58_SAMPLE_WEIGHT_PAIR_TYPES", "")
    if sample_weight is not None and sample_weight_pair_types and pair_type is not None:
        keep_weight = _pair_type_mask(pair_type, sample_weight_pair_types)
        changed = int((~keep_weight & (sample_weight != 1.0)).sum())
        if changed > 0:
            sample_weight = sample_weight.copy()
            sample_weight[~keep_weight] = 1.0
            print(
                "  R58_SAMPLE_WEIGHT_PAIR_TYPES="
                f"{sample_weight_pair_types}: reset {changed} non-matching weights to 1")

    zone_mode, corner_zones = load_dataset_zone_config(PATH)

    # Extract Y coords from ee_positions: columns 1 (start Y) and 4 (goal Y)
    ee_y = ee_positions[:, [1, 4]]  # (N, 2)
    ee_xy = None
    if zone_mode == "corner_table":
        ee_xy = ee_positions[:, [0, 1, 3, 4]]  # start XY + goal XY

    print("Multi-task data loaded:")
    print(f"  points:      {points.shape}")
    print(f"  speed_cup:   {speed_cup.shape}")
    print(f"  speed_pour:  {speed_pour.shape}")
    print(f"  normal_cup:  {normal_cup.shape}")
    print(f"  normal_pour: {normal_pour.shape}")
    print(f"  ee_y:        {ee_y.shape}")
    if ee_xy is not None:
        print(f"  ee_xy:       {ee_xy.shape} (corner-table routing)")
    print(f"  d_orient_cup: {d_orient_cup.shape}")
    print(f"  d_orient_pour: {d_orient_pour.shape}")
    print(f"  zone_id:     {zone_id.shape}")
    if pair_type is not None:
        print(f"  pair_type:   {pair_type.shape} (0=cup, 1=pour, 2=cross)")
        unique, counts = np.unique(pair_type, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"    type={int(u)}: {c} ({100*c/len(pair_type):.1f}%)")
    else:
        print(f"  pair_type:   NOT FOUND (defaulting to cross=2)")

    rows = points.shape[0]
    print(f"  Total rows: {rows}")

    if tilt_grad is not None:
        print(f"  tilt_grad:   {tilt_grad.shape} (R58U clean tilt-descent target)")
    if sample_weight is not None:
        print(f"  sample_weight: {sample_weight.shape} "
              f"(min={sample_weight.min():.3g}, max={sample_weight.max():.3g}, "
              f"mean={sample_weight.mean():.3g})")
    if handover_grad is not None:
        active = np.linalg.norm(handover_grad, axis=1) > 1e-6
        print(f"  handover_grad: {handover_grad.shape} "
              f"(active={active.sum()}/{len(active)})")
    if d_obstacle is not None and d_self is not None:
        print(f"  d_obstacle: {d_obstacle.shape} "
              f"(min={d_obstacle.min():.4f}, mean={d_obstacle.mean():.4f})")
        print(f"  d_self:     {d_self.shape} "
              f"(min={d_self.min():.4f}, mean={d_self.mean():.4f})")
    if fk_features is not None:
        print(f"  fk_features: {fk_features.shape} "
              f"(per endpoint dim={fk_features.shape[1] // 2})")

    database = _numpy2dataset_multi(points, speed_cup, speed_pour,
                                     normal_cup, normal_pour,
                                     ee_y, d_orient_cup, d_orient_pour, zone_id,
                                     pair_type=pair_type,
                                     d_obstacle=d_obstacle, d_self=d_self,
                                     ee_xy=ee_xy,
                                     tilt_grad=tilt_grad,
                                     sample_weight=sample_weight,
                                     handover_grad=handover_grad,
                                     fk_features=fk_features)
    # Optional pure-PDE state cloud.  These q states are deliberately kept
    # outside the endpoint-pair tensor: the loss can use them only as an
    # independent collocation measure, never as endpoint/value supervision.
    collocation_path = '{}/collocation_points.npy'.format(PATH)
    if os.path.exists(collocation_path):
        collocation_points = np.load(collocation_path).astype(
            np.float32, copy=False)
        if collocation_points.ndim != 2 or collocation_points.shape[1] != points.shape[1] // 2:
            raise ValueError(
                "collocation_points.npy must have shape (M, dim), got "
                f"{collocation_points.shape} for dim={points.shape[1] // 2}")
        database.collocation_points = torch.as_tensor(collocation_points)
        collocation_speed_path = '{}/collocation_speed.npy'.format(PATH)
        if os.path.exists(collocation_speed_path):
            collocation_speed = np.load(collocation_speed_path)
            if collocation_speed.shape != (collocation_points.shape[0],):
                raise ValueError(
                    "collocation_speed.npy must have shape (M,), got "
                    f"{collocation_speed.shape}")
            database.collocation_speed = torch.as_tensor(
                collocation_speed.astype(np.float32, copy=False))
        else:
            database.collocation_speed = None
        component_path = '{}/collocation_speed_components.npy'.format(PATH)
        if os.path.exists(component_path):
            collocation_speed_components = np.load(component_path)
            if (collocation_speed_components.ndim != 2
                    or collocation_speed_components.shape[0]
                    != collocation_points.shape[0]
                    or collocation_speed_components.shape[1] < 2):
                raise ValueError(
                    "collocation_speed_components.npy must have shape "
                    f"(M,H>=2), got {collocation_speed_components.shape}")
            database.collocation_speed_components = torch.as_tensor(
                collocation_speed_components.astype(np.float32, copy=False))
        else:
            database.collocation_speed_components = None
        label_note = (
            "with scalar-speed labels" if database.collocation_speed is not None
            else "measure-only")
        if database.collocation_speed_components is not None:
            label_note += (
                f", {database.collocation_speed_components.shape[1]} signed "
                "constraint components")
        print("  collocation_points: "
              f"{tuple(database.collocation_points.shape)} ({label_note})")
    else:
        database.collocation_points = None
        database.collocation_speed = None
        database.collocation_speed_components = None
    database.zone_mode = zone_mode
    database.corner_zones = corner_zones
    return database
