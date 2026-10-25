"""
Multi-task NTField training for G1 right arm (cupping + pouring).

Uses dual-head network with zone-aware routing. Based on train_g1_exp.py.

Usage:
  # Default settings:
  python train_g1_multi.py

  # Custom zone boundary:
  python train_g1_multi.py --y_divide -0.15 --overlap 0.08

  # Full sweep:
  python train_g1_multi.py --exp_name multi_sweep_01 --route_temp 0.02
"""
import sys
import os
import time
import math
import random
import copy
import json
import argparse
import itertools

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from datetime import datetime, timedelta
from tqdm import tqdm

try:
    import pickle5 as pickle
except ImportError:
    import pickle

sys.path.append('.')
from models import data_mlp_multi as db


def _reject_teacher_hack_env():
    """Teacher trajectory/direction losses are disallowed for no-hack runs."""
    bad = []
    for key, value in os.environ.items():
        if not key.startswith("R58_TEACHER_"):
            continue
        raw = str(value).strip()
        if raw == "" or raw.lower() in {"0", "0.0", "false", "none", "null"}:
            continue
        bad.append(f"{key}={value}")
    if bad:
        raise RuntimeError(
            "Teacher trajectory/direction hacks are disabled by "
            "reports/ur5e-no-hack-policy.md. Remove: " + ", ".join(sorted(bad)))
from models.metric_arm import model_network_multi as model_network
from models.metric_arm import model_function_multi as model_function
from models.metric_arm.zone_router import load_dataset_zone_config


def _file_fingerprint(path):
    if not os.path.exists(path):
        return None
    st = os.stat(path)
    return {
        'name': os.path.basename(path),
        'size': int(st.st_size),
        'mtime_ns': int(st.st_mtime_ns),
    }


def _load_dataset_metadata(data_dir):
    """Load dataset provenance saved by pkl_to_training_data_multi_ur5e.py."""
    meta = {}
    for name in ('pipeline_config.json', 'routing_config.json'):
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                cfg = json.load(f)
        except Exception as exc:
            meta[f'{name}_error'] = str(exc)
            continue
        meta[name] = {
            'fingerprint': _file_fingerprint(path),
            'version': cfg.get('version'),
            'derived_from': cfg.get('derived_from'),
            'normal_arrays': cfg.get('normal_arrays'),
            'speed_boundary_parameterization': cfg.get(
                'speed_boundary_parameterization'),
            'zone_mode': cfg.get('zone_mode'),
            'corner_zones': cfg.get('corner_zones'),
            'scene': cfg.get('scene'),
            'task_axes': cfg.get('task_axes'),
            'task_axes_json': cfg.get('task_axes_json'),
            'pkl_cupping': cfg.get('pkl_cupping'),
            'pkl_pouring': cfg.get('pkl_pouring'),
        }
    for cfg_name in ('pipeline_config.json', 'routing_config.json'):
        cfg = meta.get(cfg_name, {})
        if cfg.get('task_axes') is not None or cfg.get('task_axes_json') is not None:
            meta['task_axes'] = cfg.get('task_axes')
            meta['task_axes_json'] = cfg.get('task_axes_json')
            meta['task_axes_source'] = cfg_name
            break
    return meta


def _infer_fk_feature_dim(data_dir):
    path = os.path.join(data_dir, "fk_features.npy")
    if not os.path.exists(path):
        return 0
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2 or arr.shape[1] % 2 != 0:
        raise ValueError(
            f"fk_features.npy must have shape (N, 2*F), got {arr.shape}")
    return int(arr.shape[1] // 2)


def _kinematic_fourier_basis(
        data_dir, dim, n_freqs, hybrid=False, seed=0,
        structured_fraction=0.75):
    """Build joint-angle frequencies matched to serial-arm kinematics.

    For revolute joints, FK is composed of sums and differences of joint
    angles.  Integer angle combinations are therefore a much better basis for
    the local speed field than arbitrary low-frequency random projections.
    The optional hybrid tail preserves non-periodic joint-coordinate and
    winding information over the finite normalization box.
    """
    norm_path = os.path.join(data_dir, "joint_normalization.npz")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(
            f"kinematic Fourier basis requires {norm_path}")
    norm = np.load(norm_path)
    q_range = np.asarray(norm["q_range"], dtype=np.float32).reshape(-1)
    if q_range.shape[0] != dim:
        raise ValueError(
            f"joint normalization dim {q_range.shape[0]} != model dim {dim}")

    # Keep one representative from each +/- pair and order from local/simple
    # joint interactions to higher-order angle combinations.
    integer_modes = []
    for values in itertools.product((-1, 0, 1), repeat=dim):
        if not any(values):
            continue
        first = next(value for value in values if value)
        if first < 0:
            continue
        integer_modes.append(values)
    integer_modes.sort(key=lambda values: (
        sum(value != 0 for value in values),
        max(i for i, value in enumerate(values) if value != 0),
        tuple(abs(value) for value in values),
        values,
    ))

    if hybrid:
        structured_n = max(
            1, int(round(float(structured_fraction) * n_freqs)))
    else:
        structured_n = n_freqs
    modes = list(integer_modes[:structured_n])
    harmonic = 2
    while len(modes) < structured_n:
        for axis in range(dim):
            values = [0] * dim
            values[axis] = harmonic
            modes.append(tuple(values))
            if len(modes) == structured_n:
                break
        harmonic += 1

    scale = q_range / (2.0 * np.pi)
    basis = np.asarray(modes, dtype=np.float32) * scale[None, :]
    if hybrid and len(basis) < n_freqs:
        rng = np.random.default_rng(seed)
        tail_n = n_freqs - len(basis)
        # Low normalized-coordinate frequencies keep track of which winding
        # of a revolute joint the endpoint occupies.
        tail = rng.normal(0.0, 0.35, size=(tail_n, dim)).astype(np.float32)
        basis = np.concatenate([basis, tail], axis=0)
    return torch.tensor(basis[:n_freqs], dtype=torch.float32)


def _multiscale_random_fourier_basis(
        dim, n_freqs, base_scale=0.2, seed=0):
    """Random Fourier modes spanning the spatial scales of a distance field.

    A single narrow Gaussian frequency band cannot simultaneously represent
    broad free-space variation and the thin transition around a clipped
    clearance boundary.  Log-spaced bands retain both, while remaining one
    fixed positional encoding rather than a bank of separately queried
    models.
    """
    rng = np.random.default_rng(seed)
    multipliers = np.asarray(
        [0.5, 1.0, 2.0, 4.0, 8.0, 16.0], dtype=np.float32)
    row_scales = np.resize(multipliers, n_freqs)
    rng.shuffle(row_scales)
    basis = rng.normal(size=(n_freqs, dim)).astype(np.float32)
    basis *= (float(base_scale) * row_scales)[:, None]
    return torch.tensor(basis, dtype=torch.float32)


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches



def configure_interior_routing_from_dataset(network, data_dir, device='cuda:0',
                                            robot_cfg_path=None, scene_path=None,
                                            use_for_inputs=True,
                                            use_for_supervision=False):
    """R63: give `network` FK so interior quadrature points carry the TRUE cond.

    Reads the constraint parameters straight out of the dataset's pipeline_config
    /routing_config so the recomputed interior d_orient matches, exactly, the
    formula that produced d_orient_*.npy for this dataset.  Returns True if it
    configured, False if the dataset does not describe a corner-table run.
    """
    import json as _json, os as _os, yaml as _yaml, torch as _torch
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel

    pc_path = _os.path.join(data_dir, 'pipeline_config.json')
    rc_path = _os.path.join(data_dir, 'routing_config.json')
    if not (_os.path.exists(pc_path) and _os.path.exists(rc_path)):
        raise FileNotFoundError(
            f"interior routing needs pipeline_config.json and routing_config.json "
            f"in {data_dir}")
    pc = _json.load(open(pc_path)); rc = _json.load(open(rc_path))
    if not pc.get('signed_orient', False):
        raise ValueError(
            "interior routing reproduces the SIGNED d_orient formula; this dataset "
            "was built without --signed_orient, so the interior values would not "
            "match the endpoint labels")
    zones = rc.get('corner_zones') or pc.get('corner_zones')
    if zones is None:
        return False

    robot_cfg_path = robot_cfg_path or './ur5e_arm.yml'
    robot_cfg_dict = _yaml.safe_load(open(robot_cfg_path))
    tensor_args = TensorDeviceType(device=_torch.device(device))
    robot_cfg = RobotConfig.from_dict(robot_cfg_dict, tensor_args)
    robot_model = CudaRobotModel(robot_cfg.kinematics)

    nz = np.load(_os.path.join(data_dir, 'joint_normalization.npz'))
    network.configure_interior_routing(
        robot_model=robot_model,
        use_for_inputs=use_for_inputs,
        use_for_supervision=use_for_supervision,
        offset=pc.get('offset', 0.005),
        q_center=nz['q_center'], q_range=nz['q_range'],
        margin=pc['margin'],
        cup_tilt_max=pc['cup_tilt_deg'], pour_tilt_max=pc['pour_tilt_deg'],
        tilt_band=pc.get('tilt_band', 15.0),
        task_axes=pc['task_axes'], corner_zones=zones,
        route_temp=float(_os.environ.get('R62_ROUTE_TEMP', '0.01')),
        smooth_routing=_os.environ.get('R62_SMOOTH_ROUTING', '0') == '1',
        route_alpha=float(_os.environ.get('R62_ROUTE_ALPHA', '0.5')),
        route_y_temp=float(_os.environ.get('R62_ROUTE_Y_TEMP', '0.3')))
    print(f"[R63/R65] interior FK attached: inputs={use_for_inputs} "
          f"supervision={use_for_supervision} "
          f"(margin={pc['margin']}, offset={pc.get('offset',0.005)}, "
          f"cup/pour tilt_max={pc['cup_tilt_deg']}/{pc['pour_tilt_deg']}, "
          f"band={pc.get('tilt_band',15.0)})")
    return True


class MultiModel():
    """Multi-task training model for dual-head NTField."""

    def __init__(self, ModelPath, DataPath, dim, source, device='cpu',
                 B_scale=0.2, B_freqs=128, nl1=2,
                 use_temporal_loss=True, use_eikonal_loss=True, use_normal_loss=True,
                 eikonal_form='legacy_sqrt',
                 eikonal_balance_power=0.0,
                 eikonal_min_speed=0.0,
                 y_divide=-0.2, overlap=0.05, route_temp=0.01,
                 cup_side=-1,
                 separate_pe_gate=False,
                 separate_fourier=False, separate_encoder_norm=False,
                 weight_mode='default',
                 use_projector=False, use_align_loss=False, use_dual_sym=False,
                 cross_mode='default', blend_mode='feature',
                 detach_routing_weights=False,
                 use_boundary_align=False, boundary_align_margin=0.05,
                 boundary_align_weight=1e-3,
                 and_overlap=0.0, and_blend='min',
                 cup_loss_weight=1.0, pour_loss_weight=1.0,
                 cross_loss_weight=1.0,
                 use_orient_head=False,
                 add_orient_to_tau=False,
                 use_task_orient_heads=False,
                 use_safety_head=False,
                 add_safety_to_tau=False,
                 use_pair_safety_heads=False,
                 pair_safety_mode='replace',
                 pair_safety_residual_scale=1.0,
                 freeze_base_for_safety_head=False,
                 freeze_base_for_value_head=False,
                 init_checkpoint=None,
                 zone_mode='y', corner_zones=None, corner_spatial_temp=0.05,
                 fk_feature_dim=None,
                 use_task_adapters=False, handover_k=0, handover_temp=0.03,
                 value_arch='metric', value_condition='task',
                 finsler_quad_points=2, finsler_task_heads=False,
                 use_cross_slowness_residual=False,
                 cross_residual_scale=0.5,
                 use_local_slowness_residual=False,
                 local_residual_scale=0.2,
                 local_residual_gate='all',
                 endpoint_task_siamese=False,
                 endpoint_dual_value_heads=False,
                 pure_scalar_q_only=False,
                 task_embedding_scale=0.25,
                 fourier_basis='random',
                 global_residual_scale=2.0,
                 use_speed_head=False,
                 separate_speed_encoder=False,
                 speed_fourier_basis='multiscale_random',
                 speed_B_scale=0.2,
                 speed_B_freqs=128,
                 speed_residual_layers=0,
                 lifted_phase=False,
                 lifted_phase_physical_dim=0,
                 lifted_phase_scale=0.1,
                 lifted_phase_width=128,
                 lifted_phase_logit_scale=8.0,
                 lifted_phase_causal=False,
                 lifted_phase_hazard_floor_scale=0.0,
                 cross_hazard_floor_scale=0.0,
                 use_clearance_head=False,
                 clearance_scale=0.05,
                 clearance_boundary=0.005,
                 clearance_speed_floor=0.001,
                 clearance_residual_layers=0,
                 clearance_fourier_skip=False,
                 couple_speed_latent_to_value=False,
                 partial_shared_embedding_dim=0,
                 pair_slowness_attention_dim=0,
                 speed_constraint_heads=0,
                 speed_constraint_softmin_temperature=0.04,
                 speed_constraint_residual_scale=1.25,
                 speed_constraint_fourier_skip=False,
                 speed_constraint_consensus_width=0,
                 speed_constraint_consensus_mode='single',
                 speed_constraint_repair_width=0,
                 global_residual_positive=False,
                 global_excess_centered_gate=False,
                 global_excess_centered_signed_gate=False,
                 chord_quad_points=3,
                 chord_quadrature='legendre',
                 chord_hazard_floor_scale=1.0,
                 use_raw_coordinate_skip=False,
                 speed_floor=0.1,
                 endpoint_barrier_speed=0.1,
                 endpoint_barrier_scale=1.0,
                 endpoint_barrier_radius=0.05,
                 endpoint_barrier_power=1.0,
                 endpoint_barrier_physical_decode=False,
                 global_pair_residual_layers=0,
                 global_pair_residual_activation='silu',
                 use_viscosity_excess_head=False,
                 viscosity_excess_gate='all',
                 viscosity_residual_mode='positive_time',
                 viscosity_stencil_radius=0.0,
                 viscosity_gate_radius=0.0,
                 viscosity_speed_latent_context=False,
                 viscosity_chord_speed_latent_context=False,
                 viscosity_chord_latent_width=32,
                 viscosity_directed_context=False,
                 viscosity_directed_odd=False,
                 viscosity_directed_odd_scale=1.0,
                 viscosity_handover_phase_context=False,
                 handover_phase_embedding_scale=0.0,
                 viscosity_activation='silu',
                 viscosity_unbounded_logit=False,
                 viscosity_minplus_modes=0,
                 viscosity_minplus_temperature=0.25,
                 viscosity_minplus_scale=1.0,
                 viscosity_minplus_mode='correction',
                 use_viscosity_gain_head=False,
                 viscosity_gain_log_scale=0.6931471805599453,
                 use_causal_handover_head=False,
                 causal_handover_gate='cross_strength',
                 causal_handover_logit_scale=3.0,
                 causal_handover_strength=1.0,
                 causal_handover_phase_context=False,
                 fermat_curve_modes=3,
                 fermat_curve_envelope_detach=False,
                 fermat_curve_quad_points=15,
                 fermat_curve_residual_layers=6,
                 fermat_curve_speed_floor=0.005,
                 fermat_curve_value_weight=1.0,
                 fermat_curve_gate='all',
                 speed_loss_weight=0.0,
                 speed_loss_balance_power=0.0,
                 speed_loss_form='speed',
                 collocation_speed_loss_weight=0.0,
                 collocation_speed_n=0,
                 collocation_speed_loss_balance_power=0.0,
                 collocation_speed_loss_form='inherit',
                 collocation_speed_component_loss_weight=0.0,
                 collocation_speed_uncomposed_loss_weight=0.0,
                 collocation_speed_consensus_identification_weight=0.0,
                 collocation_speed_sampling='uniform',
                 collocation_speed_consensus_negative_weight=1.0,
                 collocation_speed_consensus_head_component_weight=0.0,
                 collocation_speed_repair_identification_weight=0.0,
                 collocation_speed_repair_coordination=False,
                 clearance_loss_weight=0.0,
                 clearance_self_scale=2.5,
                 hopf_cole_weight=0.0,
                 hopf_cole_n=0,
                 hopf_epsilon_start=0.20,
                 hopf_epsilon_end=0.02,
                 hopf_decay_epochs=600,
                 hopf_goal_radius=0.06,
                 hopf_goal_weight=1.0,
                 hopf_adaptive=False,
                 hopf_adaptive_speed_power=2.0,
                 hopf_adaptive_sharpness_weight=1.0,
                 hopf_characteristic_fraction=0.5,
                 hopf_characteristic_step=0.02,
                 hopf_state_constraint_weight=0.0,
                 hopf_boundary_speed=0.35,
                 hopf_speed_mode='predicted',
                 hopf_residual_form='travel_time',
                 hopf_ritz_normalize=False,
                 hopf_shellwise_normalize=False,
                 hopf_shell_bins=8,
                 hopf_endpoint_measure=False,
                 hopf_goal_shell_measure=False,
                 hopf_shell_radius_min=0.03,
                 hopf_shell_radius_max=1.0,
                 viscosity_weight=0.0,
                 viscosity_n=0,
                 viscosity_dirs=16,
                 viscosity_action_mode='discrete_min',
                 viscosity_softmin_temperature=0.05,
                 viscosity_epsilon=0.005,
                 viscosity_rollout_steps=1,
                 viscosity_intrinsic_budget=0.0,
                 viscosity_step=0.02,
                 viscosity_step_max=0.0,
                 viscosity_start_epoch=1,
                 viscosity_include_descent=True,
                 viscosity_include_speed_ascent=False,
                 viscosity_causal_target=False,
                 viscosity_causal_order='euclidean',
                 viscosity_candidate_routing='cup',
                 viscosity_hard_domain=False,
                 viscosity_target_lag=0,
                 viscosity_update_mode='symmetric',
                 viscosity_collocation_pairs=False,
                 viscosity_global_cloud=False,
                 viscosity_local_boundary_fraction=0.0,
                 viscosity_boundary_min_radius=0.002,
                 viscosity_boundary_max_radius=0.03,
                 fermat_path_weight=0.0,
                 fermat_loss_transform='linear',
                 train_speed_only=False,
                 train_speed_head_only=False,
                 train_speed_constraint_heads_only=False,
                 train_speed_consensus_only=False,
                 train_speed_consensus_magnitude_only=False,
                 train_speed_consensus_specificity_magnitude_only=False,
                 train_speed_repair_only=False,
                 train_clearance_only=False,
                 train_chord_profile_only=False,
                 train_endpoint_barrier_only=False,
                 train_viscosity_excess_only=False,
                 train_viscosity_input_tail_cols=0,
                 train_value_input_tail_cols=0,
                 train_lifted_phase_speed_only=False,
                 train_last_pair_residual_only=False,
                 train_partial_shared_embedding_only=False,
                 freeze_base_for_cross_residual=False,
                 train_local_residual_only=False):

        self.Params = {}
        self.Params['ModelPath'] = ModelPath
        self.Params['DataPath'] = DataPath
        self.dim = dim

        self.source = source
        self.B_scale = B_scale
        self.B_freqs = B_freqs
        self.nl1 = nl1
        self.use_temporal_loss = use_temporal_loss
        self.use_eikonal_loss = use_eikonal_loss
        self.use_normal_loss = use_normal_loss
        self.eikonal_form = str(eikonal_form)
        self.eikonal_balance_power = float(eikonal_balance_power)
        self.eikonal_min_speed = float(eikonal_min_speed)
        self.y_divide = y_divide
        self.overlap = overlap
        self.route_temp = route_temp
        self.cup_side = 1 if cup_side >= 0 else -1
        self.separate_pe_gate = separate_pe_gate
        self.separate_fourier = separate_fourier
        self.separate_encoder_norm = separate_encoder_norm
        self.weight_mode = weight_mode
        self.use_projector = use_projector
        self.use_align_loss = use_align_loss
        self.use_dual_sym = use_dual_sym
        self.cross_mode = cross_mode
        self.blend_mode = blend_mode
        self.detach_routing_weights = detach_routing_weights
        self.use_boundary_align = use_boundary_align
        self.boundary_align_margin = boundary_align_margin
        self.boundary_align_weight = boundary_align_weight
        self.and_overlap = and_overlap
        self.and_blend = and_blend
        self.cup_loss_weight = cup_loss_weight
        self.pour_loss_weight = pour_loss_weight
        self.cross_loss_weight = cross_loss_weight
        self.use_task_orient_heads = bool(use_task_orient_heads)
        self.use_orient_head = bool(use_orient_head or self.use_task_orient_heads)
        self.zone_mode = zone_mode
        self.corner_zones = corner_zones
        self.corner_spatial_temp = corner_spatial_temp
        self.fk_feature_dim = (
            _infer_fk_feature_dim(DataPath)
            if fk_feature_dim is None else int(fk_feature_dim)
        )
        self.use_task_adapters = bool(use_task_adapters)
        self.handover_k = int(handover_k or 0)
        self.handover_temp = float(handover_temp)
        self.value_arch = value_arch
        self.value_condition = value_condition
        self.finsler_quad_points = int(finsler_quad_points)
        self.finsler_task_heads = bool(finsler_task_heads)
        self.use_cross_slowness_residual = bool(use_cross_slowness_residual)
        self.cross_residual_scale = float(cross_residual_scale)
        self.use_local_slowness_residual = bool(use_local_slowness_residual)
        self.local_residual_scale = float(local_residual_scale)
        self.local_residual_gate = str(local_residual_gate or 'all')
        self.endpoint_task_siamese = bool(endpoint_task_siamese)
        self.endpoint_dual_value_heads = bool(endpoint_dual_value_heads)
        self.pure_scalar_q_only = bool(pure_scalar_q_only)
        if self.pure_scalar_q_only:
            if not self.endpoint_task_siamese:
                raise ValueError(
                    "pure_scalar_q_only requires endpoint_task_siamese")
            if self.endpoint_dual_value_heads:
                raise ValueError(
                    "pure_scalar_q_only forbids endpoint_dual_value_heads")
            if self.value_condition != 'none':
                raise ValueError(
                    "pure_scalar_q_only requires value_condition='none'")
        self.task_embedding_scale = float(task_embedding_scale)
        self.fourier_basis = str(fourier_basis or 'random')
        self.global_residual_scale = float(global_residual_scale)
        self.use_speed_head = bool(use_speed_head)
        self.separate_speed_encoder = bool(separate_speed_encoder)
        self.speed_fourier_basis = str(
            speed_fourier_basis or 'multiscale_random')
        self.speed_B_scale = float(speed_B_scale)
        self.speed_B_freqs = int(speed_B_freqs)
        self.speed_residual_layers = int(speed_residual_layers)
        self.lifted_phase = bool(lifted_phase)
        self.lifted_phase_physical_dim = int(
            lifted_phase_physical_dim or (dim - 1))
        self.lifted_phase_scale = float(lifted_phase_scale)
        self.lifted_phase_width = int(lifted_phase_width)
        self.lifted_phase_logit_scale = float(
            lifted_phase_logit_scale)
        self.lifted_phase_causal = bool(lifted_phase_causal)
        self.lifted_phase_hazard_floor_scale = float(
            lifted_phase_hazard_floor_scale)
        self.cross_hazard_floor_scale = float(cross_hazard_floor_scale)
        self.use_clearance_head = bool(use_clearance_head)
        self.clearance_scale = float(clearance_scale)
        self.clearance_boundary = float(clearance_boundary)
        self.clearance_speed_floor = float(clearance_speed_floor)
        self.clearance_residual_layers = int(clearance_residual_layers)
        self.clearance_fourier_skip = bool(clearance_fourier_skip)
        self.couple_speed_latent_to_value = bool(
            couple_speed_latent_to_value)
        self.partial_shared_embedding_dim = int(
            partial_shared_embedding_dim)
        self.pair_slowness_attention_dim = int(
            pair_slowness_attention_dim)
        self.speed_constraint_heads = int(speed_constraint_heads)
        self.speed_constraint_softmin_temperature = float(
            speed_constraint_softmin_temperature)
        self.speed_constraint_residual_scale = float(
            speed_constraint_residual_scale)
        self.speed_constraint_fourier_skip = bool(
            speed_constraint_fourier_skip)
        self.speed_constraint_consensus_width = int(
            speed_constraint_consensus_width)
        self.speed_constraint_consensus_mode = str(
            speed_constraint_consensus_mode)
        self.speed_constraint_repair_width = int(
            speed_constraint_repair_width)
        if self.partial_shared_embedding_dim < 0:
            raise ValueError(
                "partial_shared_embedding_dim must be non-negative")
        if self.pair_slowness_attention_dim < 0:
            raise ValueError(
                "pair_slowness_attention_dim must be non-negative")
        if (self.speed_constraint_heads != 0
                and self.speed_constraint_heads < 2):
            raise ValueError(
                "speed_constraint_heads must be zero or at least two")
        if self.speed_constraint_softmin_temperature <= 0.0:
            raise ValueError(
                "speed_constraint_softmin_temperature must be positive")
        if self.speed_constraint_residual_scale <= 0.0:
            raise ValueError(
                "speed_constraint_residual_scale must be positive")
        if self.speed_constraint_heads > 0 and not self.use_speed_head:
            raise ValueError(
                "speed_constraint_heads requires use_speed_head")
        if (self.speed_constraint_fourier_skip
                and self.speed_constraint_heads == 0):
            raise ValueError(
                "speed_constraint_fourier_skip requires constraint heads")
        if self.speed_constraint_consensus_width < 0:
            raise ValueError(
                "speed_constraint_consensus_width must be non-negative")
        if self.speed_constraint_consensus_mode not in (
                'single', 'hurdle', 'corridor_hurdle',
                'corridor_hurdle_compensated',
                'corridor_hurdle_compensated_balanced',
                'corridor_hurdle_specificity',
                'corridor_hurdle_specificity_odds',
                'corridor_hurdle_specificity_orthogonal',
                'corridor_hurdle_semantic_veto_min_orthogonal',
                'corridor_hurdle_semantic_veto_min_odds',
                'corridor_hurdle_semantic_veto_min_odds_residual',
                'corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend',
                'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8',
                *model_network.ROUTE_CHART_MODES):
            raise ValueError(
                "speed_constraint_consensus_mode must be single, hurdle, "
                "corridor_hurdle, corridor_hurdle_compensated, or "
                "corridor_hurdle_compensated_balanced, or "
                "corridor_hurdle_specificity, or "
                "corridor_hurdle_specificity_odds, or "
                "corridor_hurdle_specificity_orthogonal, or "
                "corridor_hurdle_semantic_veto_min_orthogonal, or "
                "corridor_hurdle_semantic_veto_min_odds, or "
                "corridor_hurdle_semantic_veto_min_odds_residual, or "
                "corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend, or "
                "corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8, or "
                f"one of {model_network.ROUTE_CHART_MODES}")
        if (self.speed_constraint_consensus_width > 0
                and self.speed_constraint_heads == 0):
            raise ValueError(
                "speed constraint consensus requires constraint heads")
        if (self.speed_constraint_consensus_width == 0
                and self.speed_constraint_consensus_mode != 'single'):
            raise ValueError(
                "non-single speed constraint consensus mode requires a "
                "positive consensus width")
        if self.speed_constraint_repair_width < 0:
            raise ValueError(
                "speed_constraint_repair_width must be non-negative")
        if (self.speed_constraint_repair_width > 0
                and self.speed_constraint_heads == 0):
            raise ValueError(
                "speed constraint repair requires constraint heads")
        if (self.speed_constraint_consensus_mode in (
                'corridor_hurdle', 'corridor_hurdle_compensated',
                'corridor_hurdle_compensated_balanced',
                'corridor_hurdle_specificity',
                'corridor_hurdle_specificity_odds',
                'corridor_hurdle_specificity_orthogonal',
                'corridor_hurdle_semantic_veto_min_orthogonal',
                'corridor_hurdle_semantic_veto_min_odds',
                'corridor_hurdle_semantic_veto_min_odds_residual',
                'corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend',
                'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8',
                *model_network.ROUTE_CHART_MODES)
                and self.speed_constraint_repair_width < 1):
            raise ValueError(
                "corridor hurdle consensus requires component repair")
        if (self.speed_constraint_repair_width > 0
                and self.speed_constraint_consensus_width > 0
                and self.speed_constraint_consensus_mode not in (
                    'corridor_hurdle', 'corridor_hurdle_compensated',
                    'corridor_hurdle_compensated_balanced',
                    'corridor_hurdle_specificity',
                    'corridor_hurdle_specificity_odds',
                    'corridor_hurdle_specificity_orthogonal',
                    'corridor_hurdle_semantic_veto_min_orthogonal',
                    'corridor_hurdle_semantic_veto_min_odds',
                    'corridor_hurdle_semantic_veto_min_odds_residual',
                    'corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend',
                    'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8',
                    *model_network.ROUTE_CHART_MODES)):
            raise ValueError(
                "speed constraint repair and final consensus are isolated "
                "architectures unless mode is a corridor hurdle")
        if self.separate_speed_encoder and not self.use_speed_head:
            raise ValueError(
                "separate_speed_encoder requires use_speed_head")
        if self.speed_B_freqs < 1:
            raise ValueError("speed_B_freqs must be positive")
        if self.speed_fourier_basis not in (
                'random', 'kinematic', 'hybrid_kinematic',
                'mixed_kinematic', 'multiscale_random'):
            raise ValueError(
                f"unknown speed_fourier_basis="
                f"{self.speed_fourier_basis!r}")
        self.global_residual_positive = bool(global_residual_positive)
        self.global_excess_centered_gate = bool(global_excess_centered_gate)
        self.global_excess_centered_signed_gate = bool(
            global_excess_centered_signed_gate)
        self.chord_quad_points = int(chord_quad_points)
        self.chord_quadrature = str(chord_quadrature or 'legendre')
        self.chord_hazard_floor_scale = float(chord_hazard_floor_scale)
        if not 0.0 <= self.chord_hazard_floor_scale <= 1.0:
            raise ValueError(
                "chord_hazard_floor_scale must lie in [0, 1]")
        self.use_raw_coordinate_skip = bool(use_raw_coordinate_skip)
        self.speed_floor = float(speed_floor)
        self.endpoint_barrier_speed = float(endpoint_barrier_speed)
        self.endpoint_barrier_scale = float(endpoint_barrier_scale)
        self.endpoint_barrier_radius = float(endpoint_barrier_radius)
        self.endpoint_barrier_power = float(endpoint_barrier_power)
        self.endpoint_barrier_physical_decode = bool(
            endpoint_barrier_physical_decode)
        self.global_pair_residual_layers = int(global_pair_residual_layers)
        self.global_pair_residual_activation = str(
            global_pair_residual_activation)
        if self.global_pair_residual_activation not in ('silu', 'maxmin'):
            raise ValueError(
                "global_pair_residual_activation must be 'silu' or "
                f"'maxmin', got {self.global_pair_residual_activation!r}")
        self.use_viscosity_excess_head = bool(use_viscosity_excess_head)
        self.viscosity_excess_gate = str(viscosity_excess_gate)
        if self.viscosity_excess_gate not in (
                'all', 'cross_strength', 'cross_transverse',
                'cross_transverse_saturated', 'cross_chord_evidence',
                'cross_chord_sqrt_evidence',
                'cross_chord_saturated_evidence', 'cross_certain'):
            raise ValueError(
                f"unknown viscosity_excess_gate="
                f"{self.viscosity_excess_gate!r}")
        self.viscosity_residual_mode = str(viscosity_residual_mode)
        if self.viscosity_residual_mode not in (
                'positive_time', 'signed_fraction', 'causal_log_time'):
            raise ValueError(
                f"unknown viscosity_residual_mode="
                f"{self.viscosity_residual_mode!r}")
        self.viscosity_stencil_radius = float(viscosity_stencil_radius)
        if not 0.0 <= self.viscosity_stencil_radius <= 0.25:
            raise ValueError(
                "viscosity_stencil_radius must lie in [0, 0.25]")
        self.viscosity_gate_radius = float(viscosity_gate_radius)
        if not 0.0 <= self.viscosity_gate_radius <= 0.25:
            raise ValueError(
                "viscosity_gate_radius must lie in [0, 0.25]")
        if (self.viscosity_stencil_radius > 0.0
                and self.viscosity_gate_radius > 0.0
                and abs(self.viscosity_stencil_radius
                        - self.viscosity_gate_radius) > 1e-9):
            raise ValueError(
                "viscosity_stencil_radius and viscosity_gate_radius must "
                "match when both are enabled")
        if (self.viscosity_excess_gate in (
                'cross_transverse', 'cross_transverse_saturated')
                and max(self.viscosity_stencil_radius,
                        self.viscosity_gate_radius) <= 0.0):
            raise ValueError(
                "cross_transverse viscosity gating requires a positive "
                "stencil or gate radius")
        self.viscosity_speed_latent_context = bool(
            viscosity_speed_latent_context)
        self.viscosity_chord_speed_latent_context = bool(
            viscosity_chord_speed_latent_context)
        self.viscosity_chord_latent_width = int(
            viscosity_chord_latent_width)
        if self.viscosity_chord_latent_width < 1:
            raise ValueError(
                "viscosity_chord_latent_width must be positive")
        self.viscosity_directed_context = bool(
            viscosity_directed_context)
        self.viscosity_directed_odd = bool(viscosity_directed_odd)
        self.viscosity_directed_odd_scale = float(
            viscosity_directed_odd_scale)
        self.viscosity_handover_phase_context = bool(
            viscosity_handover_phase_context)
        self.handover_phase_embedding_scale = float(
            handover_phase_embedding_scale)
        self.viscosity_activation = str(viscosity_activation)
        if self.viscosity_activation not in ('silu', 'maxmin'):
            raise ValueError(
                "viscosity_activation must be 'silu' or 'maxmin', got "
                f"{self.viscosity_activation!r}")
        self.viscosity_unbounded_logit = bool(viscosity_unbounded_logit)
        self.viscosity_minplus_modes = int(viscosity_minplus_modes)
        self.viscosity_minplus_temperature = float(
            viscosity_minplus_temperature)
        self.viscosity_minplus_scale = float(viscosity_minplus_scale)
        self.viscosity_minplus_mode = str(viscosity_minplus_mode)
        self.use_viscosity_gain_head = bool(use_viscosity_gain_head)
        self.viscosity_gain_log_scale = float(viscosity_gain_log_scale)
        self.use_causal_handover_head = bool(use_causal_handover_head)
        self.causal_handover_gate = str(causal_handover_gate)
        self.causal_handover_logit_scale = float(
            causal_handover_logit_scale)
        self.causal_handover_strength = float(causal_handover_strength)
        self.causal_handover_phase_context = bool(
            causal_handover_phase_context)
        self.fermat_curve_modes = int(fermat_curve_modes)
        self.fermat_curve_envelope_detach = bool(
            fermat_curve_envelope_detach)
        self.fermat_curve_quad_points = int(fermat_curve_quad_points)
        self.fermat_curve_residual_layers = int(
            fermat_curve_residual_layers)
        self.fermat_curve_speed_floor = float(fermat_curve_speed_floor)
        self.fermat_curve_value_weight = float(fermat_curve_value_weight)
        self.fermat_curve_gate = str(fermat_curve_gate)
        self.speed_loss_weight = float(speed_loss_weight)
        self.speed_loss_balance_power = float(speed_loss_balance_power)
        self.speed_loss_form = str(speed_loss_form)
        if self.speed_loss_form not in ('speed', 'log_speed', 'slowness'):
            raise ValueError(
                f"unknown speed_loss_form={self.speed_loss_form!r}")
        self.collocation_speed_loss_weight = float(
            collocation_speed_loss_weight)
        self.collocation_speed_n = int(collocation_speed_n)
        self.collocation_speed_loss_balance_power = float(
            collocation_speed_loss_balance_power)
        self.collocation_speed_loss_form = str(collocation_speed_loss_form)
        self.collocation_speed_component_loss_weight = float(
            collocation_speed_component_loss_weight)
        self.collocation_speed_uncomposed_loss_weight = float(
            collocation_speed_uncomposed_loss_weight)
        self.collocation_speed_consensus_identification_weight = float(
            collocation_speed_consensus_identification_weight)
        self.collocation_speed_sampling = str(collocation_speed_sampling)
        self.collocation_speed_consensus_negative_weight = float(
            collocation_speed_consensus_negative_weight)
        self.collocation_speed_consensus_head_component_weight = float(
            collocation_speed_consensus_head_component_weight)
        self.collocation_speed_repair_identification_weight = float(
            collocation_speed_repair_identification_weight)
        self.collocation_speed_repair_coordination = bool(
            collocation_speed_repair_coordination)
        if self.collocation_speed_loss_form not in (
                'inherit', 'speed', 'log_speed', 'slowness'):
            raise ValueError(
                "collocation_speed_loss_form must be 'inherit', 'speed', "
                "'log_speed', or 'slowness'")
        if self.collocation_speed_loss_weight < 0.0:
            raise ValueError(
                "collocation_speed_loss_weight must be non-negative")
        if self.collocation_speed_n < 0:
            raise ValueError("collocation_speed_n must be non-negative")
        if self.collocation_speed_loss_balance_power < 0.0:
            raise ValueError(
                "collocation_speed_loss_balance_power must be non-negative")
        if self.collocation_speed_component_loss_weight < 0.0:
            raise ValueError(
                "collocation_speed_component_loss_weight must be non-negative")
        if self.collocation_speed_uncomposed_loss_weight < 0.0:
            raise ValueError(
                "collocation_speed_uncomposed_loss_weight must be non-negative")
        if self.collocation_speed_consensus_identification_weight < 0.0:
            raise ValueError(
                "collocation_speed_consensus_identification_weight must be "
                "non-negative")
        if self.collocation_speed_sampling not in (
                'uniform', 'consensus_balanced'):
            raise ValueError(
                "collocation_speed_sampling must be 'uniform' or "
                "'consensus_balanced'")
        if self.collocation_speed_consensus_negative_weight <= 0.0:
            raise ValueError(
                "collocation_speed_consensus_negative_weight must be "
                "positive")
        if self.collocation_speed_consensus_head_component_weight < 0.0:
            raise ValueError(
                "collocation_speed_consensus_head_component_weight must be "
                "non-negative")
        if (self.collocation_speed_sampling == 'consensus_balanced'
                and self.collocation_speed_consensus_identification_weight
                <= 0.0):
            raise ValueError(
                "consensus-balanced sampling requires consensus "
                "identification supervision")
        if self.collocation_speed_repair_identification_weight < 0.0:
            raise ValueError(
                "collocation_speed_repair_identification_weight must be "
                "non-negative")
        if (self.collocation_speed_component_loss_weight > 0.0
                and self.speed_constraint_heads < 2):
            raise ValueError(
                "component supervision requires speed_constraint_heads")
        if (self.collocation_speed_uncomposed_loss_weight > 0.0
                and self.speed_constraint_consensus_width < 1):
            raise ValueError(
                "uncomposed speed supervision requires scalar consensus")
        if (self.collocation_speed_consensus_identification_weight > 0.0
                and self.speed_constraint_consensus_width < 1):
            raise ValueError(
                "consensus identification requires scalar consensus")
        if (self.collocation_speed_repair_identification_weight > 0.0
                and self.speed_constraint_repair_width < 1):
            raise ValueError(
                "repair identification requires scalar component repair")
        if (self.collocation_speed_repair_coordination
                and self.collocation_speed_repair_identification_weight <= 0.0):
            raise ValueError(
                "repair coordination requires repair identification")
        if ((self.collocation_speed_loss_weight > 0.0
             or self.collocation_speed_component_loss_weight > 0.0
             or self.collocation_speed_uncomposed_loss_weight > 0.0
             or self.collocation_speed_consensus_identification_weight > 0.0
             or self.collocation_speed_repair_identification_weight > 0.0)
                and self.collocation_speed_n < 1):
            raise ValueError(
                "collocation_speed_n must be positive when collocation "
                "speed supervision is enabled")
        self.clearance_loss_weight = float(clearance_loss_weight)
        self.clearance_self_scale = float(clearance_self_scale)
        self.hopf_cole_weight = float(hopf_cole_weight)
        self.hopf_cole_n = int(hopf_cole_n)
        self.hopf_epsilon_start = float(hopf_epsilon_start)
        self.hopf_epsilon_end = float(hopf_epsilon_end)
        self.hopf_decay_epochs = int(hopf_decay_epochs)
        self.hopf_goal_radius = float(hopf_goal_radius)
        self.hopf_goal_weight = float(hopf_goal_weight)
        if self.hopf_goal_weight <= 0.0:
            raise ValueError("hopf_goal_weight must be positive")
        self.hopf_adaptive = bool(hopf_adaptive)
        self.hopf_adaptive_speed_power = float(
            hopf_adaptive_speed_power)
        self.hopf_adaptive_sharpness_weight = float(
            hopf_adaptive_sharpness_weight)
        self.hopf_characteristic_fraction = float(
            hopf_characteristic_fraction)
        self.hopf_characteristic_step = float(hopf_characteristic_step)
        self.hopf_state_constraint_weight = float(
            hopf_state_constraint_weight)
        self.hopf_boundary_speed = float(hopf_boundary_speed)
        self.hopf_speed_mode = str(hopf_speed_mode)
        if self.hopf_speed_mode not in ('predicted', 'target'):
            raise ValueError(
                "hopf_speed_mode must be 'predicted' or 'target'")
        self.hopf_residual_form = str(hopf_residual_form)
        if self.hopf_residual_form not in (
                'travel_time', 'desirability_ritz'):
            raise ValueError(
                "hopf_residual_form must be 'travel_time' or "
                "'desirability_ritz'")
        self.hopf_ritz_normalize = bool(hopf_ritz_normalize)
        self.hopf_shellwise_normalize = bool(hopf_shellwise_normalize)
        self.hopf_shell_bins = int(hopf_shell_bins)
        self.hopf_endpoint_measure = bool(hopf_endpoint_measure)
        self.hopf_goal_shell_measure = bool(hopf_goal_shell_measure)
        self.hopf_shell_radius_min = float(hopf_shell_radius_min)
        self.hopf_shell_radius_max = float(hopf_shell_radius_max)
        if self.hopf_cole_weight < 0.0 or self.hopf_cole_n < 0:
            raise ValueError("Hopf--Cole weight and sample count must be non-negative")
        if not 0.0 < self.hopf_epsilon_end <= self.hopf_epsilon_start:
            raise ValueError("require 0 < hopf_epsilon_end <= hopf_epsilon_start")
        if self.hopf_decay_epochs < 1 or self.hopf_goal_radius <= 0.0:
            raise ValueError("invalid Hopf--Cole decay or goal radius")
        self.viscosity_weight = float(viscosity_weight)
        self.viscosity_n = int(viscosity_n)
        self.viscosity_dirs = int(viscosity_dirs)
        self.viscosity_action_mode = str(viscosity_action_mode)
        if self.viscosity_action_mode not in (
                'discrete_min', 'softmin', 'stochastic_vv'):
            raise ValueError(
                f"unknown viscosity_action_mode="
                f"{self.viscosity_action_mode!r}")
        self.viscosity_softmin_temperature = float(
            viscosity_softmin_temperature)
        if self.viscosity_softmin_temperature <= 0.0:
            raise ValueError(
                "viscosity_softmin_temperature must be positive")
        self.viscosity_epsilon = float(viscosity_epsilon)
        if self.viscosity_epsilon <= 0.0:
            raise ValueError("viscosity_epsilon must be positive")
        if (self.viscosity_action_mode == 'stochastic_vv'
                and (self.viscosity_dirs < 2
                     or self.viscosity_dirs % 2 != 0)):
            raise ValueError(
                "stochastic_vv requires an even --viscosity_dirs >= 2")
        self.viscosity_rollout_steps = int(viscosity_rollout_steps)
        if self.viscosity_rollout_steps < 1:
            raise ValueError("viscosity_rollout_steps must be >= 1")
        self.viscosity_intrinsic_budget = float(
            viscosity_intrinsic_budget)
        if self.viscosity_intrinsic_budget < 0.0:
            raise ValueError(
                "viscosity_intrinsic_budget must be non-negative")
        if (self.viscosity_rollout_steps > 1
                and self.viscosity_action_mode not in (
                    'stochastic_vv', 'softmin')):
            raise ValueError(
                "multi-step viscosity rollout requires stochastic_vv or "
                "softmin")
        if (self.viscosity_rollout_steps > 1
                and self.viscosity_action_mode == 'stochastic_vv'
                and self.viscosity_intrinsic_budget <= 0.0):
            raise ValueError(
                "multi-step viscosity rollout requires a positive "
                "intrinsic budget")
        self.viscosity_step = float(viscosity_step)
        self.viscosity_step_max = float(viscosity_step_max)
        self.viscosity_start_epoch = int(viscosity_start_epoch)
        self.viscosity_include_descent = bool(viscosity_include_descent)
        self.viscosity_include_speed_ascent = bool(
            viscosity_include_speed_ascent)
        self.viscosity_causal_target = bool(viscosity_causal_target)
        self.viscosity_causal_order = str(viscosity_causal_order)
        if self.viscosity_causal_order not in ('euclidean', 'value', 'none'):
            raise ValueError(
                f"unknown viscosity_causal_order="
                f"{self.viscosity_causal_order!r}")
        self.viscosity_candidate_routing = str(
            viscosity_candidate_routing)
        if self.viscosity_candidate_routing not in ('cup', 'local'):
            raise ValueError(
                f"unknown viscosity_candidate_routing="
                f"{self.viscosity_candidate_routing!r}")
        self.viscosity_hard_domain = bool(viscosity_hard_domain)
        if self.viscosity_hard_domain and self.eikonal_min_speed <= 0:
            raise ValueError(
                "viscosity_hard_domain requires --eikonal_min_speed > 0")
        self.viscosity_target_lag = int(viscosity_target_lag)
        if self.viscosity_target_lag < 0:
            raise ValueError("viscosity_target_lag must be non-negative")
        if self.viscosity_target_lag > 0 and self.viscosity_weight <= 0:
            raise ValueError(
                "viscosity_target_lag requires --viscosity_weight > 0")
        self.viscosity_update_mode = str(viscosity_update_mode)
        if self.viscosity_update_mode not in ('symmetric', 'raise_only'):
            raise ValueError(
                f"unknown viscosity_update_mode="
                f"{self.viscosity_update_mode!r}")
        self.viscosity_collocation_pairs = bool(
            viscosity_collocation_pairs)
        if (self.viscosity_collocation_pairs
                and not self.pure_scalar_q_only):
            raise ValueError(
                "viscosity_collocation_pairs is restricted to the pure "
                "q-only scalar PDE")
        self.viscosity_global_cloud = bool(viscosity_global_cloud)
        if (self.viscosity_global_cloud
                and not self.viscosity_collocation_pairs):
            raise ValueError(
                "viscosity_global_cloud requires independent scalar "
                "viscosity_collocation_pairs")
        if (self.viscosity_global_cloud
                and self.viscosity_action_mode != 'softmin'):
            raise ValueError(
                "viscosity_global_cloud requires the smooth softmin "
                "Bellman operator")
        self.viscosity_local_boundary_fraction = float(
            viscosity_local_boundary_fraction)
        self.viscosity_boundary_min_radius = float(
            viscosity_boundary_min_radius)
        self.viscosity_boundary_max_radius = float(
            viscosity_boundary_max_radius)
        if not 0.0 <= self.viscosity_local_boundary_fraction <= 1.0:
            raise ValueError(
                "viscosity_local_boundary_fraction must lie in [0, 1]")
        if (self.viscosity_boundary_min_radius <= 0.0
                or self.viscosity_boundary_max_radius
                < self.viscosity_boundary_min_radius):
            raise ValueError(
                "invalid viscosity local-boundary radius range")
        if (self.viscosity_local_boundary_fraction > 0.0
                and not self.viscosity_collocation_pairs):
            raise ValueError(
                "viscosity_local_boundary_fraction requires "
                "viscosity_collocation_pairs")
        self.fermat_path_weight = float(fermat_path_weight)
        self.fermat_loss_transform = str(fermat_loss_transform)
        if self.fermat_loss_transform not in ('linear', 'log'):
            raise ValueError(
                f"unknown fermat_loss_transform="
                f"{self.fermat_loss_transform!r}")
        self.train_speed_only = bool(train_speed_only)
        self.train_speed_head_only = bool(train_speed_head_only)
        self.train_speed_constraint_heads_only = bool(
            train_speed_constraint_heads_only)
        self.train_speed_consensus_only = bool(
            train_speed_consensus_only)
        self.train_speed_consensus_magnitude_only = bool(
            train_speed_consensus_magnitude_only)
        self.train_speed_consensus_specificity_magnitude_only = bool(
            train_speed_consensus_specificity_magnitude_only)
        self.train_speed_repair_only = bool(
            train_speed_repair_only)
        self.train_clearance_only = bool(train_clearance_only)
        self.train_chord_profile_only = bool(train_chord_profile_only)
        self.train_endpoint_barrier_only = bool(
            train_endpoint_barrier_only)
        self.train_viscosity_excess_only = bool(
            train_viscosity_excess_only)
        self.train_viscosity_input_tail_cols = int(
            train_viscosity_input_tail_cols)
        self.train_value_input_tail_cols = int(
            train_value_input_tail_cols)
        self.train_lifted_phase_speed_only = bool(
            train_lifted_phase_speed_only)
        if self.train_viscosity_input_tail_cols < 0:
            raise ValueError(
                "train_viscosity_input_tail_cols must be non-negative")
        if self.train_value_input_tail_cols < 0:
            raise ValueError(
                "train_value_input_tail_cols must be non-negative")
        if (self.train_value_input_tail_cols > 0
                and not self.couple_speed_latent_to_value):
            raise ValueError(
                "train_value_input_tail_cols requires "
                "--couple_speed_latent_to_value")
        if self.train_lifted_phase_speed_only and not self.lifted_phase:
            raise ValueError(
                "train_lifted_phase_speed_only requires --lifted_phase")
        self.train_last_pair_residual_only = bool(
            train_last_pair_residual_only)
        self.train_partial_shared_embedding_only = bool(
            train_partial_shared_embedding_only)
        if (self.train_partial_shared_embedding_only
                and self.partial_shared_embedding_dim <= 0):
            raise ValueError(
                "train_partial_shared_embedding_only requires "
                "--partial_shared_embedding_dim > 0")
        if self.train_speed_only and not (
                self.endpoint_task_siamese and self.use_speed_head):
            raise ValueError(
                "train_speed_only requires endpoint_task_siamese and "
                "use_speed_head")
        if self.train_speed_head_only and not self.use_speed_head:
            raise ValueError("train_speed_head_only requires use_speed_head")
        if (self.train_speed_constraint_heads_only
                and self.speed_constraint_heads < 2):
            raise ValueError(
                "train_speed_constraint_heads_only requires "
                "--speed_constraint_heads >= 2")
        if (self.train_speed_consensus_only
                and self.speed_constraint_consensus_width < 1):
            raise ValueError(
                "train_speed_consensus_only requires "
                "--speed_constraint_consensus_width > 0")
        if (self.train_speed_consensus_magnitude_only
                and self.speed_constraint_consensus_mode not in (
                    'corridor_hurdle_compensated',
                    'corridor_hurdle_compensated_balanced',
                    'corridor_hurdle_semantic_veto_min_odds',
                    'corridor_hurdle_semantic_veto_min_odds_residual',
                    'corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend',
                    'corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8',
                    *model_network.ROUTE_CHART_MODES)):
            raise ValueError(
                "train_speed_consensus_magnitude_only requires "
                "a compensated corridor hurdle or semantic-veto odds mode")
        if (self.train_speed_consensus_magnitude_only
                and self.speed_constraint_consensus_width < 1):
            raise ValueError(
                "train_speed_consensus_magnitude_only requires "
                "--speed_constraint_consensus_width > 0")
        if (self.train_speed_consensus_specificity_magnitude_only
                and self.speed_constraint_consensus_mode not in (
                    'corridor_hurdle_specificity',
                    'corridor_hurdle_specificity_odds',
                    'corridor_hurdle_specificity_orthogonal',
                    'corridor_hurdle_semantic_veto_min_orthogonal')):
            raise ValueError(
                "train_speed_consensus_specificity_magnitude_only requires "
                "--speed_constraint_consensus_mode "
                "corridor_hurdle_specificity or "
                "corridor_hurdle_specificity_odds or "
                "corridor_hurdle_specificity_orthogonal or "
                "corridor_hurdle_semantic_veto_min_orthogonal")
        if (self.train_speed_consensus_specificity_magnitude_only
                and self.speed_constraint_consensus_width < 1):
            raise ValueError(
                "train_speed_consensus_specificity_magnitude_only requires "
                "--speed_constraint_consensus_width > 0")
        if (self.train_speed_repair_only
                and self.speed_constraint_repair_width < 1):
            raise ValueError(
                "train_speed_repair_only requires "
                "--speed_constraint_repair_width > 0")
        if self.train_clearance_only and not self.use_clearance_head:
            raise ValueError(
                "train_clearance_only requires use_clearance_head")
        if self.clearance_loss_weight > 0 and not self.use_clearance_head:
            raise ValueError(
                "clearance_loss_weight requires use_clearance_head")
        if (self.train_chord_profile_only
                and self.value_arch != "bounded_chord_profile_symmetric"):
            raise ValueError(
                "train_chord_profile_only requires "
                "bounded_chord_profile_symmetric")
        if (self.train_endpoint_barrier_only
                and self.value_arch not in (
                    "bounded_chord_speed_barrier_symmetric",
                    "bounded_chord_local_speed_barrier_symmetric",
                    "bounded_chord_global_excess_symmetric",
                    "bounded_chord_smooth_excess_symmetric",
                    "bounded_chord_separate_smooth_excess_symmetric",
                    "bounded_chord_separate_global_excess_symmetric",
                    "bounded_chord_signed_clearance_symmetric",
                    "bounded_chord_eikonal_refine_symmetric")):
            raise ValueError(
                "train_endpoint_barrier_only requires a supported "
                "bounded-chord auxiliary architecture")
        if (self.train_viscosity_excess_only
                and not self.use_viscosity_excess_head
                and self.value_arch !=
                "bounded_chord_neural_viscosity_symmetric"):
            raise ValueError(
                "train_viscosity_excess_only requires "
                "a viscosity excess head")
        if (self.train_viscosity_input_tail_cols > 0
                and not self.use_viscosity_excess_head
                and self.value_arch !=
                "bounded_chord_neural_viscosity_symmetric"):
            raise ValueError(
                "train_viscosity_input_tail_cols requires "
                "a viscosity excess head")
        if (self.train_viscosity_input_tail_cols > 0
                and self.train_viscosity_excess_only):
            raise ValueError(
                "train_viscosity_input_tail_cols cannot be combined with "
                "train_viscosity_excess_only")
        if (self.train_last_pair_residual_only
                and self.global_pair_residual_layers < 1):
            raise ValueError(
                "train_last_pair_residual_only requires "
                "--global_pair_residual_layers >= 1")
        if (self.train_last_pair_residual_only
                and self.train_endpoint_barrier_only):
            raise ValueError(
                "train_last_pair_residual_only cannot be combined with "
                "train_endpoint_barrier_only")
        if self.train_clearance_only and any((
                self.train_speed_only,
                self.train_speed_head_only,
                self.train_speed_constraint_heads_only,
                self.train_speed_consensus_only,
                self.train_speed_consensus_magnitude_only,
                self.train_speed_consensus_specificity_magnitude_only,
                self.train_speed_repair_only,
                self.train_chord_profile_only,
                self.train_endpoint_barrier_only,
                self.train_viscosity_excess_only,
                self.train_viscosity_input_tail_cols > 0,
                self.train_value_input_tail_cols > 0,
                self.train_lifted_phase_speed_only,
                self.train_last_pair_residual_only,
                self.train_partial_shared_embedding_only)):
            raise ValueError(
                "train_clearance_only cannot be combined with another "
                "parameter-isolation mode")
        if (self.viscosity_weight > 0 and not self.use_speed_head
                and self.value_arch != 'metric'):
            raise ValueError(
                "viscosity loss without --use_speed_head is supported only "
                "for vanilla --value_arch metric, where the supplied scalar "
                "speed label defines the local Bellman stage cost")
        if (self.viscosity_weight > 0 and not self.use_speed_head
                and self.viscosity_hard_domain):
            raise ValueError(
                "--viscosity_hard_domain probes off-grid speeds and therefore "
                "requires --use_speed_head; vanilla label-speed Bellman uses "
                "the first-order local stage cost instead")
        if self.fourier_basis not in (
                'random', 'kinematic', 'hybrid_kinematic',
                'mixed_kinematic', 'multiscale_random'):
            raise ValueError(
                f"unknown fourier_basis={self.fourier_basis!r}")
        self.freeze_base_for_cross_residual = bool(freeze_base_for_cross_residual)
        self.train_local_residual_only = bool(train_local_residual_only)
        if self.train_local_residual_only and not self.use_local_slowness_residual:
            raise ValueError(
                "train_local_residual_only requires use_local_slowness_residual")
        if (self.freeze_base_for_cross_residual
                and not (self.use_cross_slowness_residual
                         or self.use_local_slowness_residual)):
            raise ValueError(
                "freeze_base_for_cross_residual requires at least one slowness residual")
        # R58G: bake orientation cost into tau. When set, force use_orient_head on
        # (tau needs the head) and record the lambda actually used (env-overridable).
        self.add_orient_to_tau = add_orient_to_tau
        if self.add_orient_to_tau:
            self.use_orient_head = True
        self.orient_tau_lambda = float(
            os.environ.get('R58_ORIENT_TAU_LAMBDA', 0.1))
        self.use_safety_head = bool(use_safety_head)
        self.use_pair_safety_heads = bool(use_pair_safety_heads)
        self.pair_safety_mode = str(pair_safety_mode or 'replace')
        self.pair_safety_residual_scale = float(pair_safety_residual_scale)
        self.add_safety_to_tau = bool(add_safety_to_tau)
        if self.add_safety_to_tau:
            self.use_safety_head = True
        self.freeze_base_for_safety_head = bool(freeze_base_for_safety_head)
        if self.freeze_base_for_safety_head and not self.use_safety_head:
            raise ValueError("freeze_base_for_safety_head requires use_safety_head")
        self.freeze_base_for_value_head = bool(freeze_base_for_value_head)
        train_only_modes = sum((
            self.train_speed_only,
            self.train_speed_head_only,
            self.train_speed_constraint_heads_only,
            self.train_speed_consensus_only,
            self.train_speed_consensus_magnitude_only,
            self.train_speed_consensus_specificity_magnitude_only,
            self.train_speed_repair_only,
            self.train_chord_profile_only,
            self.train_viscosity_excess_only,
            self.train_viscosity_input_tail_cols > 0,
            self.train_lifted_phase_speed_only,
            self.train_last_pair_residual_only,
            self.train_partial_shared_embedding_only,
            self.freeze_base_for_value_head,
            self.freeze_base_for_safety_head,
            self.freeze_base_for_cross_residual,
        ))
        # A frozen-speed scalar value head and its viscosity branch are a
        # deliberate joint PDE fine-tune: both remain inside the same scalar
        # endpoint value, while the speed trunk stays fixed.  Other isolation
        # combinations remain mutually exclusive.
        joint_value_viscosity = (
            self.freeze_base_for_value_head
            and self.train_viscosity_excess_only
            and train_only_modes == 2)
        if train_only_modes > 1 and not joint_value_viscosity:
            raise ValueError(
                "freeze/train-only modes cannot be combined")
        self.safety_tau_lambda = float(
            os.environ.get('R58_SAFETY_TAU_LAMBDA', 0.1))
        self.init_checkpoint = init_checkpoint
        self.cross_start_epoch = 0  # default: cross pairs active from epoch 0

        current_time = datetime.utcnow() - timedelta(hours=5)
        self.folder = self.Params['ModelPath'] + "/" + DataPath.split('/')[-2] + '_' + current_time.strftime("%m_%d_%H_%M")

        self.Params['Device'] = device
        self.Params['Pytorch Amp (bool)'] = False

        self.Params['Network'] = {}
        self.Params['Network']['Normlisation'] = 'OffsetMinMax'

        self.Params['Training'] = {}
        self.Params['Training']['Number of sample points'] = 2e5
        self.Params['Training']['Batch Size'] = 2000
        self.Params['Training']['Validation Percentage'] = 10
        self.Params['Training']['Number of Epochs'] = 3000
        self.Params['Training']['Resampling Bounds'] = [0.1, 0.9]
        self.Params['Training']['Print Every * Epoch'] = 1
        self.Params['Training']['Save Every * Epoch'] = 1000
        self.Params['Training']['Learning Rate'] = 1e-3
        self.Params['Training']['Random Distance Sampling'] = True
        self.Params['Training']['Use Scheduler (bool)'] = False

        self.total_train_loss = []
        self.total_val_loss = []

    def _zone_params_dict(self):
        return {
            'y_divide': self.y_divide,
            'overlap': self.overlap,
            'route_temp': self.route_temp,
            'cup_side': self.cup_side,
            'zone_mode': self.zone_mode,
            'corner_zones': self.corner_zones,
            'corner_spatial_temp': self.corner_spatial_temp,
            'smooth_routing': os.environ.get('R62_SMOOTH_ROUTING', '0') == '1',
            'route_alpha': float(os.environ.get('R62_ROUTE_ALPHA', '0.5')),
            'route_y_temp': float(os.environ.get('R62_ROUTE_Y_TEMP', '0.3')),
            'interior_routing': os.environ.get('R63_INTERIOR_ROUTING', '0') == '1',
            'interior_supervision': float(os.environ.get('R65_INTERIOR_SUP_WEIGHT', '0.0')) > 0.0,
            'separate_pe_gate': self.separate_pe_gate,
            'separate_fourier': self.separate_fourier,
            'separate_encoder_norm': self.separate_encoder_norm,
            'nl1': self.nl1,
            'weight_mode': self.weight_mode,
            'use_projector': self.use_projector,
            'use_align_loss': self.use_align_loss,
            'use_dual_sym': self.use_dual_sym,
            'cross_mode': self.cross_mode,
            'cross_loss_weight': self.cross_loss_weight,
            'blend_mode': self.blend_mode,
            'detach_routing_weights': self.detach_routing_weights,
            'use_orient_head': self.use_orient_head,
            'add_orient_to_tau': self.add_orient_to_tau,
            'orient_tau_lambda': self.orient_tau_lambda,
            'use_task_orient_heads': self.use_task_orient_heads,
            'use_safety_head': self.use_safety_head,
            'add_safety_to_tau': self.add_safety_to_tau,
            'use_pair_safety_heads': self.use_pair_safety_heads,
            'pair_safety_mode': self.pair_safety_mode,
            'pair_safety_residual_scale': self.pair_safety_residual_scale,
            'safety_tau_lambda': self.safety_tau_lambda,
            'freeze_base_for_safety_head': self.freeze_base_for_safety_head,
            'freeze_base_for_value_head': self.freeze_base_for_value_head,
            'fk_feature_dim': self.fk_feature_dim,
            'use_task_adapters': self.use_task_adapters,
            'handover_k': self.handover_k,
            'handover_temp': self.handover_temp,
            'value_arch': self.value_arch,
            'value_condition': self.value_condition,
            'finsler_quad_points': self.finsler_quad_points,
            'finsler_task_heads': self.finsler_task_heads,
            'use_cross_slowness_residual': self.use_cross_slowness_residual,
            'cross_residual_scale': self.cross_residual_scale,
            'use_local_slowness_residual': self.use_local_slowness_residual,
            'local_residual_scale': self.local_residual_scale,
            'local_residual_gate': self.local_residual_gate,
            'endpoint_task_siamese': self.endpoint_task_siamese,
            'endpoint_dual_value_heads': self.endpoint_dual_value_heads,
            'pure_scalar_q_only': self.pure_scalar_q_only,
            'task_embedding_scale': self.task_embedding_scale,
            'fourier_basis': self.fourier_basis,
            'global_residual_scale': self.global_residual_scale,
            'use_speed_head': self.use_speed_head,
            'separate_speed_encoder': self.separate_speed_encoder,
            'speed_fourier_basis': self.speed_fourier_basis,
            'speed_B_scale': self.speed_B_scale,
            'speed_B_freqs': self.speed_B_freqs,
            'speed_residual_layers': self.speed_residual_layers,
            'lifted_phase': self.lifted_phase,
            'lifted_phase_physical_dim':
                self.lifted_phase_physical_dim,
            'lifted_phase_scale': self.lifted_phase_scale,
            'lifted_phase_width': self.lifted_phase_width,
            'lifted_phase_logit_scale':
                self.lifted_phase_logit_scale,
            'lifted_phase_causal': self.lifted_phase_causal,
            'lifted_phase_hazard_floor_scale':
                self.lifted_phase_hazard_floor_scale,
            'cross_hazard_floor_scale': self.cross_hazard_floor_scale,
            'use_clearance_head': self.use_clearance_head,
            'clearance_scale': self.clearance_scale,
            'clearance_boundary': self.clearance_boundary,
            'clearance_speed_floor': self.clearance_speed_floor,
            'clearance_residual_layers': self.clearance_residual_layers,
            'clearance_fourier_skip': self.clearance_fourier_skip,
            'couple_speed_latent_to_value':
                self.couple_speed_latent_to_value,
            'partial_shared_embedding_dim':
                self.partial_shared_embedding_dim,
            'pair_slowness_attention_dim':
                self.pair_slowness_attention_dim,
            'speed_constraint_heads': self.speed_constraint_heads,
            'speed_constraint_softmin_temperature':
                self.speed_constraint_softmin_temperature,
            'speed_constraint_residual_scale':
                self.speed_constraint_residual_scale,
            'speed_constraint_fourier_skip':
                self.speed_constraint_fourier_skip,
            'speed_constraint_consensus_width':
                self.speed_constraint_consensus_width,
            'speed_constraint_consensus_mode':
                self.speed_constraint_consensus_mode,
            'speed_constraint_repair_width':
                self.speed_constraint_repair_width,
            'global_residual_positive': self.global_residual_positive,
            'global_excess_centered_gate': self.global_excess_centered_gate,
            'global_excess_centered_signed_gate':
                self.global_excess_centered_signed_gate,
            'chord_quad_points': self.chord_quad_points,
            'chord_quadrature': self.chord_quadrature,
            'chord_hazard_floor_scale': self.chord_hazard_floor_scale,
            'use_raw_coordinate_skip': self.use_raw_coordinate_skip,
            'speed_floor': self.speed_floor,
            'endpoint_barrier_speed': self.endpoint_barrier_speed,
            'endpoint_barrier_scale': self.endpoint_barrier_scale,
            'endpoint_barrier_radius': self.endpoint_barrier_radius,
            'endpoint_barrier_power': self.endpoint_barrier_power,
            'endpoint_barrier_physical_decode':
                self.endpoint_barrier_physical_decode,
            'global_pair_residual_layers': self.global_pair_residual_layers,
            'global_pair_residual_activation':
                self.global_pair_residual_activation,
            'use_viscosity_excess_head': self.use_viscosity_excess_head,
            'viscosity_excess_gate': self.viscosity_excess_gate,
            'viscosity_residual_mode': self.viscosity_residual_mode,
            'viscosity_stencil_radius': self.viscosity_stencil_radius,
            'viscosity_gate_radius': self.viscosity_gate_radius,
            'viscosity_speed_latent_context':
                self.viscosity_speed_latent_context,
            'viscosity_chord_speed_latent_context':
                self.viscosity_chord_speed_latent_context,
            'viscosity_chord_latent_width':
                self.viscosity_chord_latent_width,
            'viscosity_directed_context':
                self.viscosity_directed_context,
            'viscosity_directed_odd': self.viscosity_directed_odd,
            'viscosity_directed_odd_scale':
                self.viscosity_directed_odd_scale,
            'viscosity_handover_phase_context':
                self.viscosity_handover_phase_context,
            'handover_phase_embedding_scale':
                self.handover_phase_embedding_scale,
            'viscosity_activation': self.viscosity_activation,
            'viscosity_unbounded_logit':
                self.viscosity_unbounded_logit,
            'viscosity_minplus_modes': self.viscosity_minplus_modes,
            'viscosity_minplus_temperature':
                self.viscosity_minplus_temperature,
            'viscosity_minplus_scale': self.viscosity_minplus_scale,
            'viscosity_minplus_mode': self.viscosity_minplus_mode,
            'use_viscosity_gain_head': self.use_viscosity_gain_head,
            'viscosity_gain_log_scale': self.viscosity_gain_log_scale,
            'use_causal_handover_head': self.use_causal_handover_head,
            'causal_handover_gate': self.causal_handover_gate,
            'causal_handover_logit_scale':
                self.causal_handover_logit_scale,
            'causal_handover_strength': self.causal_handover_strength,
            'causal_handover_phase_context':
                self.causal_handover_phase_context,
            'fermat_curve_modes': self.fermat_curve_modes,
            'fermat_curve_envelope_detach':
                self.fermat_curve_envelope_detach,
            'fermat_curve_quad_points': self.fermat_curve_quad_points,
            'fermat_curve_residual_layers':
                self.fermat_curve_residual_layers,
            'fermat_curve_speed_floor': self.fermat_curve_speed_floor,
            'fermat_curve_value_weight': self.fermat_curve_value_weight,
            'fermat_curve_gate': self.fermat_curve_gate,
            'freeze_base_for_cross_residual': self.freeze_base_for_cross_residual,
            'train_local_residual_only': self.train_local_residual_only,
            'eikonal_form': self.eikonal_form,
            'eikonal_balance_power': self.eikonal_balance_power,
            'eikonal_min_speed': self.eikonal_min_speed,
            'viscosity_weight': self.viscosity_weight,
            'viscosity_n': self.viscosity_n,
            'viscosity_dirs': self.viscosity_dirs,
            'viscosity_action_mode': self.viscosity_action_mode,
            'viscosity_softmin_temperature':
                self.viscosity_softmin_temperature,
            'viscosity_epsilon': self.viscosity_epsilon,
            'viscosity_rollout_steps': self.viscosity_rollout_steps,
            'viscosity_intrinsic_budget': self.viscosity_intrinsic_budget,
            'viscosity_step': self.viscosity_step,
            'viscosity_step_max': self.viscosity_step_max,
            'viscosity_start_epoch': self.viscosity_start_epoch,
            'viscosity_include_descent': self.viscosity_include_descent,
            'viscosity_include_speed_ascent':
                self.viscosity_include_speed_ascent,
            'viscosity_causal_target': self.viscosity_causal_target,
            'viscosity_causal_order': self.viscosity_causal_order,
            'viscosity_candidate_routing':
                self.viscosity_candidate_routing,
            'viscosity_hard_domain': self.viscosity_hard_domain,
            'viscosity_target_lag': self.viscosity_target_lag,
            'viscosity_update_mode': self.viscosity_update_mode,
            'viscosity_collocation_pairs':
                self.viscosity_collocation_pairs,
            'viscosity_global_cloud': self.viscosity_global_cloud,
            'viscosity_local_boundary_fraction':
                self.viscosity_local_boundary_fraction,
            'viscosity_boundary_min_radius':
                self.viscosity_boundary_min_radius,
            'viscosity_boundary_max_radius':
                self.viscosity_boundary_max_radius,
            'fermat_path_weight': self.fermat_path_weight,
            'fermat_loss_transform': self.fermat_loss_transform,
            'train_speed_only': self.train_speed_only,
            'train_speed_head_only': self.train_speed_head_only,
            'train_speed_constraint_heads_only':
                self.train_speed_constraint_heads_only,
            'train_speed_consensus_only':
                self.train_speed_consensus_only,
            'train_speed_consensus_magnitude_only':
                self.train_speed_consensus_magnitude_only,
            'train_speed_consensus_specificity_magnitude_only':
                self.train_speed_consensus_specificity_magnitude_only,
            'train_speed_repair_only':
                self.train_speed_repair_only,
            'train_clearance_only': self.train_clearance_only,
            'train_chord_profile_only': self.train_chord_profile_only,
            'train_endpoint_barrier_only':
                self.train_endpoint_barrier_only,
            'train_viscosity_excess_only':
                self.train_viscosity_excess_only,
            'train_viscosity_input_tail_cols':
                self.train_viscosity_input_tail_cols,
            'train_value_input_tail_cols':
                self.train_value_input_tail_cols,
            'train_lifted_phase_speed_only':
                self.train_lifted_phase_speed_only,
            'train_last_pair_residual_only':
                self.train_last_pair_residual_only,
            'train_partial_shared_embedding_only':
                self.train_partial_shared_embedding_only,
        }

    def train(self):
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        import shutil
        source_folder = './models/metric_arm'
        destination_folder = self.folder + '/models'
        os.makedirs(destination_folder, exist_ok=True)
        for filename in os.listdir(source_folder):
            source_file = os.path.join(source_folder, filename)
            destination_file = os.path.join(destination_folder, filename)
            if os.path.isfile(source_file):
                shutil.copy2(source_file, destination_file)
        print("Files copied successfully!")

        # Save training config
        dataset_metadata = _load_dataset_metadata(self.Params['DataPath'])
        config = {
            'ModelPath': self.Params['ModelPath'],
            'DataPath': self.Params['DataPath'],
            'dim': self.dim,
            'source': self.source,
            'Device': self.Params['Device'],
            'Dataset': dataset_metadata,
            'Training': {
                'Batch Size': self.Params['Training']['Batch Size'],
                'Number of Epochs': self.Params['Training']['Number of Epochs'],
                'Learning Rate': self.Params['Training']['Learning Rate'],
                'Max Batches Per Epoch': self.Params['Training'].get('Max Batches Per Epoch', 5),
                'Loss Ratio Threshold': self.Params['Training'].get('Loss Ratio Threshold', 1.2),
                'Base LR': self.Params['Training'].get('Base LR', 5e-4),
                'Cross LR': self.Params['Training'].get('Cross LR', None),
                'cross_start_epoch': self.cross_start_epoch,
                'Dist Condition Only': self.Params['Training'].get(
                    'Dist Condition Only', False),
                'Pair Condition Only': self.Params['Training'].get(
                    'Pair Condition Only', False),
            },
            'Experiment': {
                'B_scale': self.B_scale,
                'B_freqs': self.B_freqs,
                'use_temporal_loss': self.use_temporal_loss,
                'use_eikonal_loss': self.use_eikonal_loss,
                'use_normal_loss': self.use_normal_loss,
                'eikonal_form': self.eikonal_form,
                'y_divide': self.y_divide,
                'overlap': self.overlap,
                'route_temp': self.route_temp,
                'cup_side': self.cup_side,
                'zone_mode': self.zone_mode,
                'corner_zones': self.corner_zones,
                'corner_spatial_temp': self.corner_spatial_temp,
            'smooth_routing': os.environ.get('R62_SMOOTH_ROUTING', '0') == '1',
            'route_alpha': float(os.environ.get('R62_ROUTE_ALPHA', '0.5')),
            'route_y_temp': float(os.environ.get('R62_ROUTE_Y_TEMP', '0.3')),
            'interior_routing': os.environ.get('R63_INTERIOR_ROUTING', '0') == '1',
            'interior_supervision': float(os.environ.get('R65_INTERIOR_SUP_WEIGHT', '0.0')) > 0.0,
                'separate_pe_gate': self.separate_pe_gate,
                'separate_fourier': self.separate_fourier,
                'separate_encoder_norm': self.separate_encoder_norm,
                'weight_mode': self.weight_mode,
                'use_projector': self.use_projector,
                'use_align_loss': self.use_align_loss,
                'use_dual_sym': self.use_dual_sym,
                'cross_mode': self.cross_mode,
                'cup_loss_weight': self.cup_loss_weight,
                'pour_loss_weight': self.pour_loss_weight,
                'cross_loss_weight': self.cross_loss_weight,
                'blend_mode': self.blend_mode,
                'detach_routing_weights': self.detach_routing_weights,
                'use_boundary_align': self.use_boundary_align,
                'boundary_align_margin': self.boundary_align_margin,
                'boundary_align_weight': self.boundary_align_weight,
                'and_overlap': self.and_overlap,
                'and_blend': self.and_blend,
                'use_orient_head': self.use_orient_head,
                'add_orient_to_tau': self.add_orient_to_tau,
                'orient_tau_lambda': self.orient_tau_lambda,
                'use_task_orient_heads': self.use_task_orient_heads,
                'use_safety_head': self.use_safety_head,
                'add_safety_to_tau': self.add_safety_to_tau,
                'use_pair_safety_heads': self.use_pair_safety_heads,
                'pair_safety_mode': self.pair_safety_mode,
                'pair_safety_residual_scale': self.pair_safety_residual_scale,
                'safety_tau_lambda': self.safety_tau_lambda,
                'freeze_base_for_safety_head': self.freeze_base_for_safety_head,
                'freeze_base_for_value_head': self.freeze_base_for_value_head,
                'safety_weight': float(os.environ.get('R58_SAFETY_WEIGHT', '1e-2')),
                'safety_pos_weight': float(os.environ.get('R58_SAFETY_POS_WEIGHT', '1.0')),
                'safety_margin': float(os.environ.get('R58_SAFETY_MARGIN', '0.03')),
                'safety_self_margin': float(os.environ.get('R58_SAFETY_SELF_MARGIN', '0.002')),
                'fk_feature_dim': self.fk_feature_dim,
                'use_task_adapters': self.use_task_adapters,
                'handover_k': self.handover_k,
                'handover_temp': self.handover_temp,
                'value_arch': self.value_arch,
                'value_condition': self.value_condition,
                'finsler_quad_points': self.finsler_quad_points,
                'finsler_task_heads': self.finsler_task_heads,
                'use_cross_slowness_residual': self.use_cross_slowness_residual,
                'cross_residual_scale': self.cross_residual_scale,
                'use_local_slowness_residual': self.use_local_slowness_residual,
                'local_residual_scale': self.local_residual_scale,
                'local_residual_gate': self.local_residual_gate,
                'endpoint_task_siamese': self.endpoint_task_siamese,
                'endpoint_dual_value_heads': self.endpoint_dual_value_heads,
                'pure_scalar_q_only': self.pure_scalar_q_only,
                'task_embedding_scale': self.task_embedding_scale,
                'fourier_basis': self.fourier_basis,
                'global_residual_scale': self.global_residual_scale,
                'use_speed_head': self.use_speed_head,
                'separate_speed_encoder': self.separate_speed_encoder,
                'speed_fourier_basis': self.speed_fourier_basis,
                'speed_B_scale': self.speed_B_scale,
                'speed_B_freqs': self.speed_B_freqs,
                'speed_residual_layers': self.speed_residual_layers,
                'lifted_phase': self.lifted_phase,
                'lifted_phase_physical_dim':
                    self.lifted_phase_physical_dim,
                'lifted_phase_scale': self.lifted_phase_scale,
                'lifted_phase_width': self.lifted_phase_width,
                'lifted_phase_logit_scale':
                    self.lifted_phase_logit_scale,
                'lifted_phase_causal': self.lifted_phase_causal,
                'lifted_phase_hazard_floor_scale':
                    self.lifted_phase_hazard_floor_scale,
                'cross_hazard_floor_scale': self.cross_hazard_floor_scale,
                'use_clearance_head': self.use_clearance_head,
                'clearance_scale': self.clearance_scale,
                'clearance_boundary': self.clearance_boundary,
                'clearance_speed_floor': self.clearance_speed_floor,
                'clearance_residual_layers': self.clearance_residual_layers,
                'clearance_fourier_skip': self.clearance_fourier_skip,
                'couple_speed_latent_to_value':
                    self.couple_speed_latent_to_value,
                'partial_shared_embedding_dim':
                    self.partial_shared_embedding_dim,
                'pair_slowness_attention_dim':
                    self.pair_slowness_attention_dim,
                'speed_constraint_heads': self.speed_constraint_heads,
                'speed_constraint_softmin_temperature':
                    self.speed_constraint_softmin_temperature,
                'speed_constraint_residual_scale':
                    self.speed_constraint_residual_scale,
                'speed_constraint_fourier_skip':
                    self.speed_constraint_fourier_skip,
                'speed_constraint_consensus_width':
                    self.speed_constraint_consensus_width,
                'speed_constraint_consensus_mode':
                    self.speed_constraint_consensus_mode,
                'speed_constraint_repair_width':
                    self.speed_constraint_repair_width,
                'global_residual_positive': self.global_residual_positive,
                'global_excess_centered_gate': self.global_excess_centered_gate,
                'global_excess_centered_signed_gate':
                    self.global_excess_centered_signed_gate,
                'chord_quad_points': self.chord_quad_points,
                'chord_quadrature': self.chord_quadrature,
                'chord_hazard_floor_scale': self.chord_hazard_floor_scale,
                'use_raw_coordinate_skip': self.use_raw_coordinate_skip,
                'speed_floor': self.speed_floor,
                'endpoint_barrier_speed': self.endpoint_barrier_speed,
                'endpoint_barrier_scale': self.endpoint_barrier_scale,
                'endpoint_barrier_radius': self.endpoint_barrier_radius,
                'endpoint_barrier_power': self.endpoint_barrier_power,
                'endpoint_barrier_physical_decode':
                    self.endpoint_barrier_physical_decode,
                'global_pair_residual_layers': self.global_pair_residual_layers,
                'global_pair_residual_activation':
                    self.global_pair_residual_activation,
                'use_viscosity_excess_head': self.use_viscosity_excess_head,
                'viscosity_excess_gate': self.viscosity_excess_gate,
                'viscosity_residual_mode': self.viscosity_residual_mode,
                'viscosity_stencil_radius': self.viscosity_stencil_radius,
                'viscosity_gate_radius': self.viscosity_gate_radius,
                'viscosity_speed_latent_context':
                    self.viscosity_speed_latent_context,
                'viscosity_chord_speed_latent_context':
                    self.viscosity_chord_speed_latent_context,
                'viscosity_chord_latent_width':
                    self.viscosity_chord_latent_width,
                'viscosity_directed_context':
                    self.viscosity_directed_context,
                'viscosity_directed_odd':
                    self.viscosity_directed_odd,
                'viscosity_directed_odd_scale':
                    self.viscosity_directed_odd_scale,
                'viscosity_handover_phase_context':
                    self.viscosity_handover_phase_context,
                'handover_phase_embedding_scale':
                    self.handover_phase_embedding_scale,
                'viscosity_activation': self.viscosity_activation,
                'viscosity_unbounded_logit':
                    self.viscosity_unbounded_logit,
                'viscosity_minplus_modes': self.viscosity_minplus_modes,
                'viscosity_minplus_temperature':
                    self.viscosity_minplus_temperature,
                'viscosity_minplus_scale': self.viscosity_minplus_scale,
                'viscosity_minplus_mode': self.viscosity_minplus_mode,
                'viscosity_collocation_pairs':
                    self.viscosity_collocation_pairs,
                'viscosity_global_cloud': self.viscosity_global_cloud,
                'viscosity_local_boundary_fraction':
                    self.viscosity_local_boundary_fraction,
                'viscosity_boundary_min_radius':
                    self.viscosity_boundary_min_radius,
                'viscosity_boundary_max_radius':
                    self.viscosity_boundary_max_radius,
                'use_viscosity_gain_head': self.use_viscosity_gain_head,
                'viscosity_gain_log_scale': self.viscosity_gain_log_scale,
                'use_causal_handover_head':
                    self.use_causal_handover_head,
                'causal_handover_gate': self.causal_handover_gate,
                'causal_handover_logit_scale':
                    self.causal_handover_logit_scale,
                'causal_handover_strength':
                    self.causal_handover_strength,
                'causal_handover_phase_context':
                    self.causal_handover_phase_context,
                'fermat_curve_modes': self.fermat_curve_modes,
                'fermat_curve_envelope_detach':
                    self.fermat_curve_envelope_detach,
                'fermat_curve_quad_points': self.fermat_curve_quad_points,
                'fermat_curve_residual_layers':
                    self.fermat_curve_residual_layers,
                'fermat_curve_speed_floor': self.fermat_curve_speed_floor,
                'fermat_curve_value_weight':
                    self.fermat_curve_value_weight,
                'fermat_curve_gate': self.fermat_curve_gate,
                'eikonal_balance_power': self.eikonal_balance_power,
                'eikonal_min_speed': self.eikonal_min_speed,
                'speed_loss_weight': self.speed_loss_weight,
                'speed_loss_balance_power': self.speed_loss_balance_power,
                'speed_loss_form': self.speed_loss_form,
                'collocation_speed_loss_weight':
                    self.collocation_speed_loss_weight,
                'collocation_speed_n': self.collocation_speed_n,
                'collocation_speed_loss_balance_power':
                    self.collocation_speed_loss_balance_power,
                'collocation_speed_loss_form':
                    self.collocation_speed_loss_form,
                'collocation_speed_component_loss_weight':
                    self.collocation_speed_component_loss_weight,
                'collocation_speed_uncomposed_loss_weight':
                    self.collocation_speed_uncomposed_loss_weight,
                'collocation_speed_consensus_identification_weight':
                    self.collocation_speed_consensus_identification_weight,
                'collocation_speed_sampling':
                    self.collocation_speed_sampling,
                'collocation_speed_consensus_negative_weight':
                    self.collocation_speed_consensus_negative_weight,
                'collocation_speed_consensus_head_component_weight':
                    self.collocation_speed_consensus_head_component_weight,
                'collocation_speed_repair_identification_weight':
                    self.collocation_speed_repair_identification_weight,
                'collocation_speed_repair_coordination':
                    self.collocation_speed_repair_coordination,
                'clearance_loss_weight': self.clearance_loss_weight,
                'clearance_self_scale': self.clearance_self_scale,
                'hopf_cole_weight': self.hopf_cole_weight,
                'hopf_cole_n': self.hopf_cole_n,
                'hopf_epsilon_start': self.hopf_epsilon_start,
                'hopf_epsilon_end': self.hopf_epsilon_end,
                'hopf_decay_epochs': self.hopf_decay_epochs,
                'hopf_goal_radius': self.hopf_goal_radius,
                'hopf_goal_weight': self.hopf_goal_weight,
                'hopf_adaptive': self.hopf_adaptive,
                'hopf_adaptive_speed_power':
                    self.hopf_adaptive_speed_power,
                'hopf_adaptive_sharpness_weight':
                    self.hopf_adaptive_sharpness_weight,
                'hopf_characteristic_fraction':
                    self.hopf_characteristic_fraction,
                'hopf_characteristic_step':
                    self.hopf_characteristic_step,
                'hopf_state_constraint_weight':
                    self.hopf_state_constraint_weight,
                'hopf_boundary_speed': self.hopf_boundary_speed,
                'hopf_speed_mode': self.hopf_speed_mode,
                'hopf_residual_form': self.hopf_residual_form,
                'hopf_ritz_normalize': self.hopf_ritz_normalize,
                'hopf_shellwise_normalize': self.hopf_shellwise_normalize,
                'hopf_shell_bins': self.hopf_shell_bins,
                'hopf_endpoint_measure': self.hopf_endpoint_measure,
                'hopf_goal_shell_measure': self.hopf_goal_shell_measure,
                'hopf_shell_radius_min': self.hopf_shell_radius_min,
                'hopf_shell_radius_max': self.hopf_shell_radius_max,
                'viscosity_weight': self.viscosity_weight,
                'viscosity_n': self.viscosity_n,
                'viscosity_dirs': self.viscosity_dirs,
                'viscosity_action_mode': self.viscosity_action_mode,
                'viscosity_softmin_temperature':
                    self.viscosity_softmin_temperature,
                'viscosity_epsilon': self.viscosity_epsilon,
                'viscosity_rollout_steps': self.viscosity_rollout_steps,
                'viscosity_intrinsic_budget':
                    self.viscosity_intrinsic_budget,
                'viscosity_step': self.viscosity_step,
                'viscosity_step_max': self.viscosity_step_max,
                'viscosity_start_epoch': self.viscosity_start_epoch,
                'viscosity_include_descent': self.viscosity_include_descent,
                'viscosity_include_speed_ascent':
                    self.viscosity_include_speed_ascent,
                'viscosity_causal_target': self.viscosity_causal_target,
                'viscosity_causal_order': self.viscosity_causal_order,
                'viscosity_candidate_routing':
                    self.viscosity_candidate_routing,
                'viscosity_hard_domain': self.viscosity_hard_domain,
                'viscosity_target_lag': self.viscosity_target_lag,
                'viscosity_update_mode': self.viscosity_update_mode,
                'fermat_path_weight': self.fermat_path_weight,
                'fermat_loss_transform': self.fermat_loss_transform,
                'train_speed_only': self.train_speed_only,
                'train_speed_head_only': self.train_speed_head_only,
                'train_speed_constraint_heads_only':
                    self.train_speed_constraint_heads_only,
                'train_speed_consensus_only':
                    self.train_speed_consensus_only,
                'train_speed_consensus_magnitude_only':
                    self.train_speed_consensus_magnitude_only,
                'train_speed_consensus_specificity_magnitude_only':
                    self.train_speed_consensus_specificity_magnitude_only,
                'train_speed_repair_only':
                    self.train_speed_repair_only,
                'train_clearance_only': self.train_clearance_only,
                'train_chord_profile_only': self.train_chord_profile_only,
                'train_endpoint_barrier_only':
                    self.train_endpoint_barrier_only,
                'train_viscosity_excess_only':
                    self.train_viscosity_excess_only,
                'train_viscosity_input_tail_cols':
                    self.train_viscosity_input_tail_cols,
                'train_value_input_tail_cols':
                    self.train_value_input_tail_cols,
                'train_lifted_phase_speed_only':
                    self.train_lifted_phase_speed_only,
                'train_last_pair_residual_only':
                    self.train_last_pair_residual_only,
                'train_partial_shared_embedding_only':
                    self.train_partial_shared_embedding_only,
                'seed': int(getattr(self, 'seed', 0)),
                'freeze_base_for_cross_residual': self.freeze_base_for_cross_residual,
                'train_local_residual_only': self.train_local_residual_only,
                'cross_align_weight': float(os.environ.get('R58_ALIGN_WEIGHT', '0.0')),
                'tilt_grad_tau_weight': float(os.environ.get('R58_TILT_GRAD_TAU_WEIGHT', '0.0')),
                'tilt_grad_tau_n': int(os.environ.get('R58_TILT_GRAD_TAU_N', '0')),
                'tilt_grad_pair_types': os.environ.get('R58_TILT_GRAD_PAIR_TYPES', ''),
                'handover_grad_weight': float(os.environ.get('R58_HANDOVER_GRAD_WEIGHT', '0.0')),
                'handover_grad_n': int(os.environ.get('R58_HANDOVER_GRAD_N', '0')),
                'handover_grad_pair_types': os.environ.get('R58_HANDOVER_GRAD_PAIR_TYPES', ''),
                'handover_grad_mode': os.environ.get('R58_HANDOVER_GRAD_MODE', 'hinge'),
                'handover_grad_margin': float(os.environ.get('R58_HANDOVER_GRAD_MARGIN', '0.1')),
                'ignore_sample_weight': os.environ.get('R58_IGNORE_SAMPLE_WEIGHT', '0') == '1',
                'sample_weight_cap': (
                    float(os.environ['R58_SAMPLE_WEIGHT_CAP'])
                    if os.environ.get('R58_SAMPLE_WEIGHT_CAP') else None),
                'sample_weight_pair_types': os.environ.get('R58_SAMPLE_WEIGHT_PAIR_TYPES', ''),
                'drop_base_cross': os.environ.get('R58_DROP_BASE_CROSS', '0') == '1',
                'select_total_loss': os.environ.get('R58_SELECT_TOTAL_LOSS', '0') == '1',
                'disable_rollback': (
                    bool(self.Params['Training'].get('Disable Rollback', False))
                    or os.environ.get('R58_DISABLE_ROLLBACK', '0') == '1'),
                'init_checkpoint': self.init_checkpoint,
                'path_loss_args': getattr(self, '_path_loss_args', None),
                'segment_barrier_args': getattr(
                    self, '_segment_barrier_args', None),
                'line_preserve_args': getattr(
                    self, '_line_preserve_args', None),
            },
            'folder': self.folder,
            'timestamp': datetime.utcnow().isoformat(),
        }
        config_path = os.path.join(self.folder, 'train_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Config saved to {config_path}")

        # Initialize network. If an init checkpoint is provided, reuse its
        # Fourier basis but keep the architecture requested by the current CLI
        # flags. This allows adding new heads (for example orient_head) while
        # still warm-starting all compatible existing weights.
        init_checkpoint_state = None
        if self.init_checkpoint:
            print(f"Preparing initialization from checkpoint: {self.init_checkpoint}")
            init_checkpoint_state = torch.load(
                self.init_checkpoint, map_location=torch.device(self.Params['Device']))
            self.B = init_checkpoint_state['B_state_dict'].to(self.Params['Device'])
            if (self.lifted_phase and self.B.ndim == 2
                    and self.B.shape[1] + 1 == self.dim):
                # Preserve every inherited Fourier feature exactly.  The new
                # phase coordinate is read only by the dedicated scalar-speed
                # restriction until a later value-capacity experiment says
                # otherwise.
                self.B = torch.cat([
                    self.B,
                    torch.zeros(
                        self.B.shape[0], 1, device=self.B.device,
                        dtype=self.B.dtype),
                ], dim=1)
                print(
                    f"  [init] lifted B from physical dim {self.dim - 1} "
                    f"to model dim {self.dim} with an exact-zero phase column")
            if self.B.shape[0] != self.B_freqs:
                print(f"  [init] overriding B_freqs {self.B_freqs} -> {self.B.shape[0]} "
                      "to match checkpoint")
                self.B_freqs = int(self.B.shape[0])
        elif self.fourier_basis == 'multiscale_random':
            self.B = _multiscale_random_fourier_basis(
                self.dim, self.B_freqs,
                base_scale=self.B_scale, seed=0).to(
                    self.Params['Device'])
            print(
                f"[fourier] multiscale_random: B={tuple(self.B.shape)} "
                f"base_scale={self.B_scale:g}")
        elif self.fourier_basis in (
                'kinematic', 'hybrid_kinematic', 'mixed_kinematic'):
            self.B = _kinematic_fourier_basis(
                self.Params['DataPath'], self.dim, self.B_freqs,
                hybrid=self.fourier_basis != 'kinematic', seed=0,
                structured_fraction=(
                    0.5 if self.fourier_basis == 'mixed_kinematic'
                    else 0.75)).to(self.Params['Device'])
            print(
                f"[fourier] {self.fourier_basis}: "
                f"B={tuple(self.B.shape)}")
        else:
            self.B = self.B_scale * torch.normal(0, 1, size=(self.B_freqs, self.dim))

        speed_B = None
        if self.separate_speed_encoder:
            source_speed_B = None
            source_speed_basis = None
            if init_checkpoint_state is not None:
                source_speed_B = init_checkpoint_state.get(
                    'model_state_dict', {}).get('B_speed')
                if (self.lifted_phase and source_speed_B is not None
                        and source_speed_B.ndim == 2
                        and source_speed_B.shape[0] + 1 == self.dim):
                    source_speed_B = torch.cat([
                        source_speed_B,
                        torch.zeros(
                            1, source_speed_B.shape[1],
                            device=source_speed_B.device,
                            dtype=source_speed_B.dtype),
                    ], dim=0)
                source_speed_basis = init_checkpoint_state.get(
                    'zone_params', {}).get(
                        'speed_fourier_basis', 'multiscale_random')
            source_speed_shape_matches = (
                source_speed_B is not None
                and source_speed_B.ndim == 2
                and tuple(source_speed_B.shape) == (
                    self.dim, self.speed_B_freqs))
            if (source_speed_B is not None
                    and source_speed_basis == self.speed_fourier_basis
                    and source_speed_shape_matches):
                speed_B = source_speed_B.T.contiguous().to(
                    self.Params['Device'])
                print(
                    f"[speed_fourier] restored {self.speed_fourier_basis} basis "
                    f"{tuple(speed_B.shape)} from checkpoint")
            elif self.speed_fourier_basis == 'multiscale_random':
                speed_B = _multiscale_random_fourier_basis(
                    self.dim, self.speed_B_freqs,
                    base_scale=self.speed_B_scale, seed=17).to(
                        self.Params['Device'])
                print(
                    f"[speed_fourier] separate multiscale_random basis "
                    f"{tuple(speed_B.shape)} "
                    f"base_scale={self.speed_B_scale:g}")
            elif self.speed_fourier_basis in (
                    'kinematic', 'hybrid_kinematic', 'mixed_kinematic'):
                speed_B = _kinematic_fourier_basis(
                    self.Params['DataPath'], self.dim, self.speed_B_freqs,
                    hybrid=self.speed_fourier_basis != 'kinematic', seed=17,
                    structured_fraction=(
                        0.5 if self.speed_fourier_basis == 'mixed_kinematic'
                        else 0.75)).to(self.Params['Device'])
                print(
                    f"[speed_fourier] separate "
                    f"{self.speed_fourier_basis} basis "
                    f"{tuple(speed_B.shape)}")
            else:
                rng = np.random.default_rng(17)
                speed_B = torch.tensor(
                    rng.normal(
                        0.0, self.speed_B_scale,
                        size=(self.speed_B_freqs, self.dim)),
                    dtype=torch.float32, device=self.Params['Device'])
                print(
                    f"[speed_fourier] separate random basis "
                    f"{tuple(speed_B.shape)} "
                    f"base_scale={self.speed_B_scale:g}")

        self.network = model_network.MultiNN(
            self.Params['Device'], self.dim, self.B,
            h_size=2 * self.B_freqs, nl1=self.nl1,
            y_divide=self.y_divide, overlap=self.overlap,
            route_temp=self.route_temp,
            cup_side=self.cup_side,
            separate_pe_gate=self.separate_pe_gate,
            separate_fourier=self.separate_fourier,
            separate_encoder_norm=self.separate_encoder_norm,
            weight_mode=self.weight_mode,
            use_projector=self.use_projector,
            use_dual_sym=self.use_dual_sym, blend_mode=self.blend_mode,
            detach_routing_weights=self.detach_routing_weights,
            use_orient_head=self.use_orient_head,
            add_orient_to_tau=self.add_orient_to_tau,
            orient_tau_lambda=self.orient_tau_lambda,
            use_task_orient_heads=self.use_task_orient_heads,
            use_safety_head=self.use_safety_head,
            add_safety_to_tau=self.add_safety_to_tau,
            use_pair_safety_heads=self.use_pair_safety_heads,
            pair_safety_mode=self.pair_safety_mode,
            pair_safety_residual_scale=self.pair_safety_residual_scale,
            safety_tau_lambda=self.safety_tau_lambda,
            zone_mode=self.zone_mode,
            corner_zones=self.corner_zones,
            corner_spatial_temp=self.corner_spatial_temp,
            fk_feature_dim=self.fk_feature_dim,
            use_task_adapters=self.use_task_adapters,
            handover_k=self.handover_k,
            handover_temp=self.handover_temp,
            value_arch=self.value_arch,
            value_condition=self.value_condition,
            finsler_quad_points=self.finsler_quad_points,
            finsler_task_heads=self.finsler_task_heads,
            use_cross_slowness_residual=self.use_cross_slowness_residual,
            cross_residual_scale=self.cross_residual_scale,
            use_local_slowness_residual=self.use_local_slowness_residual,
            local_residual_scale=self.local_residual_scale,
            local_residual_gate=self.local_residual_gate,
            endpoint_task_siamese=self.endpoint_task_siamese,
            endpoint_dual_value_heads=self.endpoint_dual_value_heads,
            task_embedding_scale=self.task_embedding_scale,
            global_residual_scale=self.global_residual_scale,
            use_speed_head=self.use_speed_head,
            separate_speed_encoder=self.separate_speed_encoder,
            speed_B=speed_B,
            speed_residual_layers=self.speed_residual_layers,
            lifted_phase=self.lifted_phase,
            lifted_phase_physical_dim=
                self.lifted_phase_physical_dim,
            lifted_phase_scale=self.lifted_phase_scale,
            lifted_phase_width=self.lifted_phase_width,
            lifted_phase_logit_scale=
                self.lifted_phase_logit_scale,
            lifted_phase_causal=self.lifted_phase_causal,
            lifted_phase_hazard_floor_scale=
                self.lifted_phase_hazard_floor_scale,
            cross_hazard_floor_scale=self.cross_hazard_floor_scale,
            use_clearance_head=self.use_clearance_head,
            clearance_scale=self.clearance_scale,
            clearance_boundary=self.clearance_boundary,
            clearance_speed_floor=self.clearance_speed_floor,
            clearance_residual_layers=self.clearance_residual_layers,
            clearance_fourier_skip=self.clearance_fourier_skip,
            couple_speed_latent_to_value=
                self.couple_speed_latent_to_value,
            partial_shared_embedding_dim=
                self.partial_shared_embedding_dim,
            pair_slowness_attention_dim=
                self.pair_slowness_attention_dim,
            speed_constraint_heads=self.speed_constraint_heads,
            speed_constraint_softmin_temperature=
                self.speed_constraint_softmin_temperature,
            speed_constraint_residual_scale=
                self.speed_constraint_residual_scale,
            speed_constraint_fourier_skip=
                self.speed_constraint_fourier_skip,
            speed_constraint_consensus_width=
                self.speed_constraint_consensus_width,
            speed_constraint_consensus_mode=
                self.speed_constraint_consensus_mode,
            speed_constraint_repair_width=
                self.speed_constraint_repair_width,
            global_residual_positive=self.global_residual_positive,
            global_excess_centered_gate=self.global_excess_centered_gate,
            global_excess_centered_signed_gate=(
                self.global_excess_centered_signed_gate),
            chord_quad_points=self.chord_quad_points,
            chord_quadrature=self.chord_quadrature,
            chord_hazard_floor_scale=self.chord_hazard_floor_scale,
            use_raw_coordinate_skip=self.use_raw_coordinate_skip,
            speed_floor=self.speed_floor,
            endpoint_barrier_speed=self.endpoint_barrier_speed,
            endpoint_barrier_scale=self.endpoint_barrier_scale,
            endpoint_barrier_radius=self.endpoint_barrier_radius,
            endpoint_barrier_power=self.endpoint_barrier_power,
            endpoint_barrier_physical_decode=
                self.endpoint_barrier_physical_decode,
            global_pair_residual_layers=self.global_pair_residual_layers,
            global_pair_residual_activation=
                self.global_pair_residual_activation,
            use_viscosity_excess_head=self.use_viscosity_excess_head,
            viscosity_excess_gate=self.viscosity_excess_gate,
            viscosity_residual_mode=self.viscosity_residual_mode,
            viscosity_stencil_radius=self.viscosity_stencil_radius,
            viscosity_gate_radius=self.viscosity_gate_radius,
            viscosity_speed_latent_context=
                self.viscosity_speed_latent_context,
            viscosity_chord_speed_latent_context=
                self.viscosity_chord_speed_latent_context,
            viscosity_chord_latent_width=
                self.viscosity_chord_latent_width,
            viscosity_directed_context=
                self.viscosity_directed_context,
            viscosity_directed_odd=self.viscosity_directed_odd,
            viscosity_directed_odd_scale=
                self.viscosity_directed_odd_scale,
            viscosity_handover_phase_context=
                self.viscosity_handover_phase_context,
            handover_phase_embedding_scale=
                self.handover_phase_embedding_scale,
            viscosity_activation=self.viscosity_activation,
            viscosity_unbounded_logit=self.viscosity_unbounded_logit,
            viscosity_minplus_modes=self.viscosity_minplus_modes,
            viscosity_minplus_temperature=
                self.viscosity_minplus_temperature,
            viscosity_minplus_scale=self.viscosity_minplus_scale,
            viscosity_minplus_mode=self.viscosity_minplus_mode,
            use_viscosity_gain_head=self.use_viscosity_gain_head,
            viscosity_gain_log_scale=self.viscosity_gain_log_scale,
            use_causal_handover_head=self.use_causal_handover_head,
            causal_handover_gate=self.causal_handover_gate,
            causal_handover_logit_scale=
                self.causal_handover_logit_scale,
            causal_handover_strength=self.causal_handover_strength,
            causal_handover_phase_context=
                self.causal_handover_phase_context,
            fermat_curve_modes=self.fermat_curve_modes,
            fermat_curve_envelope_detach=
                self.fermat_curve_envelope_detach,
            fermat_curve_quad_points=self.fermat_curve_quad_points,
            fermat_curve_residual_layers=
                self.fermat_curve_residual_layers,
            fermat_curve_speed_floor=self.fermat_curve_speed_floor,
            fermat_curve_value_weight=self.fermat_curve_value_weight,
            fermat_curve_gate=self.fermat_curve_gate)
        self.network.apply(self.network.init_weights)
        self.network.init_task_adapter_weights()
        self.network.init_raw_coordinate_skip_weights()
        self.network.init_cross_residual_weights()
        self.network.init_global_residual_weights()
        self.network.init_global_excess_weights()
        self.network.init_global_pair_residual_weights()
        self.network.init_partial_shared_embedding_weights()
        self.network.init_fermat_curve_weights()
        self.network.init_speed_residual_weights()
        self.network.init_speed_constraint_weights()
        self.network.init_speed_constraint_consensus_weights()
        self.network.init_speed_constraint_repair_weights()
        self.network.init_lifted_phase_weights()
        self.network.init_viscosity_excess_weights()
        self.network.init_viscosity_minplus_weights()
        self.network.init_viscosity_gain_weights()
        self.network.init_causal_handover_weights()
        self.network.init_clearance_residual_weights()
        if getattr(self, 'no_lip_norm', False):
            self.network.disable_lip_norm = True
        self.network.to(self.Params['Device'])
        # R63: interior quadrature points get FK-recomputed cond instead of a
        # linear interpolation of the endpoints.  Opt-in via R63_INTERIOR_ROUTING=1.
        _r63 = os.environ.get('R63_INTERIOR_ROUTING', '0') == '1'
        _r65 = float(os.environ.get('R65_INTERIOR_SUP_WEIGHT', '0.0')) > 0.0
        if _r63 or _r65:
            configure_interior_routing_from_dataset(
                self.network, self.Params['DataPath'],
                device=self.Params['Device'],
                use_for_inputs=_r63, use_for_supervision=_r65)

        self.function = model_function.MultiFunction(
            self.folder, self.Params['Device'], self.network, self.dim,
            use_temporal_loss=self.use_temporal_loss,
            use_eikonal_loss=self.use_eikonal_loss,
            use_normal_loss=self.use_normal_loss,
            eikonal_form=self.eikonal_form,
            eikonal_balance_power=self.eikonal_balance_power,
            eikonal_min_speed=self.eikonal_min_speed,
            speed_loss_weight=self.speed_loss_weight,
            speed_loss_balance_power=self.speed_loss_balance_power,
            speed_loss_form=self.speed_loss_form,
            collocation_speed_loss_weight=
                self.collocation_speed_loss_weight,
            collocation_speed_n=self.collocation_speed_n,
            collocation_speed_loss_balance_power=
                self.collocation_speed_loss_balance_power,
            collocation_speed_loss_form=self.collocation_speed_loss_form,
            collocation_speed_component_loss_weight=
                self.collocation_speed_component_loss_weight,
            collocation_speed_uncomposed_loss_weight=
                self.collocation_speed_uncomposed_loss_weight,
            collocation_speed_consensus_identification_weight=
                self.collocation_speed_consensus_identification_weight,
            collocation_speed_sampling=self.collocation_speed_sampling,
            collocation_speed_consensus_negative_weight=
                self.collocation_speed_consensus_negative_weight,
            collocation_speed_consensus_head_component_weight=
                self.collocation_speed_consensus_head_component_weight,
            collocation_speed_repair_identification_weight=
                self.collocation_speed_repair_identification_weight,
            collocation_speed_repair_coordination=
                self.collocation_speed_repair_coordination,
            clearance_loss_weight=self.clearance_loss_weight,
            clearance_self_scale=self.clearance_self_scale,
            hopf_cole_weight=self.hopf_cole_weight,
            hopf_cole_n=self.hopf_cole_n,
            hopf_epsilon_start=self.hopf_epsilon_start,
            hopf_epsilon_end=self.hopf_epsilon_end,
            hopf_decay_epochs=self.hopf_decay_epochs,
            hopf_goal_radius=self.hopf_goal_radius,
            hopf_goal_weight=self.hopf_goal_weight,
            hopf_adaptive=self.hopf_adaptive,
            hopf_adaptive_speed_power=self.hopf_adaptive_speed_power,
            hopf_adaptive_sharpness_weight=
                self.hopf_adaptive_sharpness_weight,
            hopf_characteristic_fraction=
                self.hopf_characteristic_fraction,
            hopf_characteristic_step=self.hopf_characteristic_step,
            hopf_state_constraint_weight=
                self.hopf_state_constraint_weight,
            hopf_boundary_speed=self.hopf_boundary_speed,
            hopf_speed_mode=self.hopf_speed_mode,
            hopf_residual_form=self.hopf_residual_form,
            hopf_ritz_normalize=self.hopf_ritz_normalize,
            hopf_shellwise_normalize=self.hopf_shellwise_normalize,
            hopf_shell_bins=self.hopf_shell_bins,
            hopf_endpoint_measure=self.hopf_endpoint_measure,
            hopf_goal_shell_measure=self.hopf_goal_shell_measure,
            hopf_shell_radius_min=self.hopf_shell_radius_min,
            hopf_shell_radius_max=self.hopf_shell_radius_max,
            viscosity_weight=self.viscosity_weight,
            viscosity_n=self.viscosity_n,
            viscosity_dirs=self.viscosity_dirs,
            viscosity_action_mode=self.viscosity_action_mode,
            viscosity_softmin_temperature=
                self.viscosity_softmin_temperature,
            viscosity_epsilon=self.viscosity_epsilon,
            viscosity_rollout_steps=self.viscosity_rollout_steps,
            viscosity_intrinsic_budget=self.viscosity_intrinsic_budget,
            viscosity_step=self.viscosity_step,
            viscosity_step_max=self.viscosity_step_max,
            viscosity_start_epoch=self.viscosity_start_epoch,
            viscosity_include_descent=self.viscosity_include_descent,
            viscosity_include_speed_ascent=
                self.viscosity_include_speed_ascent,
            viscosity_causal_target=self.viscosity_causal_target,
            viscosity_causal_order=self.viscosity_causal_order,
            viscosity_candidate_routing=
                self.viscosity_candidate_routing,
            viscosity_hard_domain=self.viscosity_hard_domain,
            viscosity_update_mode=self.viscosity_update_mode,
            viscosity_collocation_pairs=self.viscosity_collocation_pairs,
            viscosity_global_cloud=self.viscosity_global_cloud,
            viscosity_local_boundary_fraction=
                self.viscosity_local_boundary_fraction,
            viscosity_boundary_min_radius=self.viscosity_boundary_min_radius,
            viscosity_boundary_max_radius=self.viscosity_boundary_max_radius,
            fermat_path_weight=self.fermat_path_weight,
            fermat_loss_transform=self.fermat_loss_transform,
            use_align_loss=self.use_align_loss,
            cross_mode=self.cross_mode,
            use_boundary_align=self.use_boundary_align,
            boundary_align_margin=self.boundary_align_margin,
            boundary_align_weight=self.boundary_align_weight,
            and_overlap=self.and_overlap,
            and_blend=self.and_blend,
            cup_loss_weight=self.cup_loss_weight,
            pour_loss_weight=self.pour_loss_weight,
            cross_loss_weight=self.cross_loss_weight,
            pure_scalar_q_only=self.pure_scalar_q_only,
        )
        self.function.travel_time_weight = self.Params['Training'].get('Travel Time Weight', 0.5)

        # --- Optional train-time oracle wiring (default OFF) ---
        # _path_loss_args, _segment_barrier_args, and _line_preserve_args are
        # set by main() only when their corresponding flags are passed.
        self._path_loss_cfg = None
        self._segment_barrier_cfg = None
        self._line_preserve_cfg = None
        pl_args = getattr(self, '_path_loss_args', None)
        sb_args = getattr(self, '_segment_barrier_args', None)
        lp_args = getattr(self, '_line_preserve_args', None)
        oracle_args = pl_args or sb_args or lp_args
        if oracle_args is not None:
            # ensure curobo dir importable for the oracle (mounted at /workspace)
            import sys as _sys
            if "/workspace" not in _sys.path:
                _sys.path.insert(0, "/workspace")
            from train_collision_oracle import TrainCollisionOracle

            device = oracle_args['device']
            data_dir = oracle_args['data_dir']
            norm_file = f"{data_dir}/joint_normalization.npz"
            oracle = TrainCollisionOracle(
                scene_yml=oracle_args['scene'], norm_file=norm_file,
                cup_tilt_max=oracle_args['cup_tilt_max'],
                pour_tilt_max=oracle_args['pour_tilt_max'],
                cup_tilt_soft=oracle_args['cup_tilt_soft'],
                pour_tilt_soft=oracle_args['pour_tilt_soft'],
                device=device)

            nz = np.load(norm_file)
            q_min_norm = (nz["q_min"] - nz["q_center"]) / nz["q_range"]
            q_max_norm = (nz["q_max"] - nz["q_center"]) / nz["q_range"]
            q_min_norm_t = torch.tensor(q_min_norm, dtype=torch.float32, device=device)
            q_max_norm_t = torch.tensor(q_max_norm, dtype=torch.float32, device=device)

            self.function._oracle = oracle
            self.function._q_min_norm = q_min_norm_t
            self.function._q_max_norm = q_max_norm_t

            if pl_args is not None:
                from scipy.spatial import cKDTree
                idx_dir = pl_args['safe_index_dir'] or data_dir
                z = np.load(f"{idx_dir}/safe_index.npz")
                safe_q_np = np.asarray(z["q_norm"], dtype=np.float32)
                safe_mask = np.ones(safe_q_np.shape[0], dtype=bool)
                if "d_obs" in z.files:
                    safe_mask &= np.asarray(z["d_obs"]) >= pl_args['margin']
                tilt_filter = "none"
                if not pl_args.get('collision_only', False):
                    has_tilt = "tilt_cup" in z.files and "tilt_pour" in z.files
                    has_xy = "ee_x" in z.files and "ee_y" in z.files
                    if pl_args['handover_accept'] and has_tilt and has_xy:
                        from morph_orient import corner_blend_param
                        s = corner_blend_param(
                            np.asarray(z["ee_x"]), np.asarray(z["ee_y"]),
                            overlap_deg=pl_args['corner_overlap_deg'])
                        tilt_cup = np.asarray(z["tilt_cup"])
                        tilt_pour = np.asarray(z["tilt_pour"])
                        cup_bad = (s <= 1e-6) & (tilt_cup > pl_args['cup_tilt_max'])
                        pour_bad = (
                            (s >= 1.0 - 1e-6)
                            & (tilt_pour > pl_args['pour_tilt_max']))
                        safe_mask &= ~(cup_bad | pour_bad)
                        tilt_filter = "corner_handover_wings"
                    else:
                        if "tilt_cup" in z.files:
                            safe_mask &= (
                                np.asarray(z["tilt_cup"]) <= pl_args['cup_tilt_max'])
                        if "tilt_pour" in z.files:
                            safe_mask &= (
                                np.asarray(z["tilt_pour"]) <= pl_args['pour_tilt_max'])
                        tilt_filter = "dual_tilt"
                if not np.any(safe_mask):
                    raise ValueError(
                        f"No path-loss safe neighbors in {idx_dir}/safe_index.npz "
                        f"for margin={pl_args['margin']}, "
                        f"cup_tilt_max={pl_args['cup_tilt_max']}, "
                        f"pour_tilt_max={pl_args['pour_tilt_max']}")
                safe_q_np = safe_q_np[safe_mask]
                print(
                    f"[path-loss] safe neighbors: {safe_q_np.shape[0]} / "
                    f"{z['q_norm'].shape[0]} (tilt_filter={tilt_filter})")
                safe_q = torch.tensor(safe_q_np, dtype=torch.float32, device=device)
                tree = cKDTree(safe_q_np)

                def safe_neighbor(q_query):
                    qn = q_query.detach().cpu().numpy()
                    _, nn = tree.query(qn, k=1)
                    q_safe = safe_q[torch.as_tensor(nn, device=device)]
                    found = torch.ones(q_safe.shape[0], dtype=torch.bool, device=device)
                    return q_safe, found

                self.function._safe_neighbor = safe_neighbor
                self.function._path_pair_pool_start = None
                self.function._path_pair_pool_goal = None
                self.function._path_pair_pool_pair_type = None
                pool_path = pl_args.get('pair_pool')
                if pool_path:
                    pool = np.load(pool_path, allow_pickle=True)
                    if "starts_norm" in pool.files and "goals_norm" in pool.files:
                        starts_np = np.asarray(pool["starts_norm"], dtype=np.float32)
                        goals_np = np.asarray(pool["goals_norm"], dtype=np.float32)
                    elif "q_start_norm" in pool.files and "q_goal_norm" in pool.files:
                        starts_np = np.asarray(pool["q_start_norm"], dtype=np.float32)
                        goals_np = np.asarray(pool["q_goal_norm"], dtype=np.float32)
                    elif "q_start" in pool.files and "q_goal" in pool.files:
                        starts_np = np.asarray(pool["q_start"], dtype=np.float32)
                        goals_np = np.asarray(pool["q_goal"], dtype=np.float32)
                    else:
                        raise ValueError(
                            f"{pool_path} must contain starts_norm/goals_norm "
                            "or q_start_norm/q_goal_norm")
                    if "pair_type" not in pool.files:
                        raise ValueError(
                            f"{pool_path} must contain pair_type for path-loss "
                            "pair-pool sampling")
                    pair_np = np.asarray(pool["pair_type"], dtype=np.int64)
                    if starts_np.shape != goals_np.shape:
                        raise ValueError(
                            f"path-loss pair pool shape mismatch: "
                            f"{starts_np.shape} vs {goals_np.shape}")
                    if starts_np.shape[0] != pair_np.shape[0]:
                        raise ValueError(
                            f"path-loss pair pool pair_type length mismatch: "
                            f"{starts_np.shape[0]} vs {pair_np.shape[0]}")
                    pool_n = int(pl_args.get('pair_pool_n') or 0)
                    if pool_n > 0 and starts_np.shape[0] > pool_n:
                        rng = np.random.default_rng(
                            int(pl_args.get('pair_pool_seed', 6262)))
                        take = rng.choice(
                            starts_np.shape[0], size=pool_n, replace=False)
                        starts_np = starts_np[take]
                        goals_np = goals_np[take]
                        pair_np = pair_np[take]
                    self.function._path_pair_pool_start = torch.tensor(
                        starts_np, dtype=torch.float32, device=device)
                    self.function._path_pair_pool_goal = torch.tensor(
                        goals_np, dtype=torch.float32, device=device)
                    self.function._path_pair_pool_pair_type = torch.tensor(
                        pair_np, dtype=torch.long, device=device)
                    print(
                        f"[path_loss] pair pool: {starts_np.shape[0]} rows "
                        f"from {pool_path}")
                self._path_loss_cfg = dict(
                    steps=pl_args['steps'], step_size=pl_args['step_size'],
                    rollout_n=pl_args['rollout_n'], margin=pl_args['margin'],
                    self_collision_margin=pl_args['self_collision_margin'],
                    cup_tilt_max=pl_args['cup_tilt_max'],
                    pour_tilt_max=pl_args['pour_tilt_max'],
                    rank_margin=pl_args['rank_margin'],
                    path_eik_weight=pl_args['path_eik_weight'],
                    rank_weight=pl_args['rank_weight'],
                    start_epoch=pl_args['start_epoch'],
                    rank_start_epoch=pl_args['rank_start_epoch'],
                    pair_types=pl_args['pair_types'],
                    handover_accept=pl_args['handover_accept'],
                    corner_overlap_deg=pl_args['corner_overlap_deg'],
                    handover_mono_tol=pl_args['handover_mono_tol'],
                    handover_weight=pl_args['handover_weight'],
                    collision_only=pl_args.get('collision_only', False),
                    offset=None,
                )
                print(f"[path_loss] ENABLED: scene={pl_args['scene']} "
                      f"rollout_n={pl_args['rollout_n']} steps={pl_args['steps']} "
                      f"pair_types={pl_args['pair_types']} "
                      f"handover_accept={pl_args['handover_accept']} "
                      f"collision_only={pl_args.get('collision_only', False)} "
                      f"eik_warmup={pl_args['start_epoch']} rank_warmup={pl_args['rank_start_epoch']}")

            if sb_args is not None:
                if sb_args.get('safe_pool', False):
                    idx_dir = sb_args['safe_index_dir'] or data_dir
                    z = np.load(f"{idx_dir}/safe_index.npz")
                    safe_q_np = np.asarray(z["q_norm"], dtype=np.float32)
                    safe_mask = np.ones(safe_q_np.shape[0], dtype=bool)
                    if "d_obs" in z.files:
                        safe_mask &= np.asarray(z["d_obs"]) >= sb_args['margin']
                    if not sb_args.get('collision_only', False):
                        has_tilt = "tilt_cup" in z.files and "tilt_pour" in z.files
                        has_xy = "ee_x" in z.files and "ee_y" in z.files
                        if sb_args['handover_accept'] and has_tilt and has_xy:
                            from morph_orient import corner_blend_param
                            s = corner_blend_param(
                                np.asarray(z["ee_x"]), np.asarray(z["ee_y"]),
                                overlap_deg=sb_args['corner_overlap_deg'])
                            tilt_cup = np.asarray(z["tilt_cup"])
                            tilt_pour = np.asarray(z["tilt_pour"])
                            cup_bad = (
                                (s <= 1e-6)
                                & (tilt_cup > sb_args['cup_tilt_max']))
                            pour_bad = (
                                (s >= 1.0 - 1e-6)
                                & (tilt_pour > sb_args['pour_tilt_max']))
                            safe_mask &= ~(cup_bad | pour_bad)
                        else:
                            if "tilt_cup" in z.files:
                                safe_mask &= (
                                    np.asarray(z["tilt_cup"])
                                    <= sb_args['cup_tilt_max'])
                            if "tilt_pour" in z.files:
                                safe_mask &= (
                                    np.asarray(z["tilt_pour"])
                                    <= sb_args['pour_tilt_max'])
                    if not np.any(safe_mask):
                        raise ValueError(
                            f"No segment-barrier safe pool in "
                            f"{idx_dir}/safe_index.npz for "
                            f"margin={sb_args['margin']}")
                    safe_q_np = safe_q_np[safe_mask]
                    pool_n = int(sb_args.get('safe_pool_n', 0))
                    if pool_n > 0 and safe_q_np.shape[0] > pool_n:
                        rng = np.random.default_rng(
                            int(sb_args.get('safe_pool_seed', 6262)))
                        take = rng.choice(
                            safe_q_np.shape[0], size=pool_n, replace=False)
                        safe_q_np = safe_q_np[take]
                    self.function._segment_barrier_safe_q = torch.tensor(
                        safe_q_np, dtype=torch.float32, device=device)
                    print(
                        f"[segment_barrier] safe pool: "
                        f"{safe_q_np.shape[0]} / {z['q_norm'].shape[0]} "
                        f"from {idx_dir}/safe_index.npz")

                self._segment_barrier_cfg = dict(
                    anchor_n=sb_args['anchor_n'],
                    n_dirs=sb_args['n_dirs'],
                    step_size=sb_args['step_size'],
                    margin=sb_args['margin'],
                    cup_tilt_max=sb_args['cup_tilt_max'],
                    pour_tilt_max=sb_args['pour_tilt_max'],
                    rank_margin=sb_args['rank_margin'],
                    weight=sb_args['weight'],
                    self_collision_margin=sb_args['self_collision_margin'],
                    start_epoch=sb_args['start_epoch'],
                    pair_types=sb_args['pair_types'],
                    anchor_mode=sb_args['anchor_mode'],
                    handover_accept=sb_args['handover_accept'],
                    corner_overlap_deg=sb_args['corner_overlap_deg'],
                    check_midpoint=sb_args['check_midpoint'],
                    collision_only=sb_args['collision_only'],
                    safe_pool=sb_args.get('safe_pool', False),
                )
                print(f"[segment_barrier] ENABLED: scene={sb_args['scene']} "
                      f"anchors={sb_args['anchor_n']} dirs={sb_args['n_dirs']} "
                      f"step={sb_args['step_size']} pair_types={sb_args['pair_types']} "
                      f"safe_pool={sb_args.get('safe_pool', False)} "
                      f"weight={sb_args['weight']} start={sb_args['start_epoch']}")

            if lp_args is not None:
                self.function._line_pair_pool_start = None
                self.function._line_pair_pool_goal = None
                self.function._line_pair_pool_pair_type = None
                if lp_args.get('safe_pool', False):
                    idx_dir = lp_args['safe_index_dir'] or data_dir
                    z = np.load(f"{idx_dir}/safe_index.npz")
                    safe_q_np = np.asarray(z["q_norm"], dtype=np.float32)
                    safe_mask = np.ones(safe_q_np.shape[0], dtype=bool)
                    if "d_obs" in z.files:
                        safe_mask &= np.asarray(z["d_obs"]) >= lp_args['margin']
                    if not np.any(safe_mask):
                        raise ValueError(
                            f"No line-preserve safe pool in "
                            f"{idx_dir}/safe_index.npz for "
                            f"margin={lp_args['margin']}")
                    safe_q_np = safe_q_np[safe_mask]
                    pool_n = int(lp_args.get('safe_pool_n', 0))
                    if pool_n > 0 and safe_q_np.shape[0] > pool_n:
                        rng = np.random.default_rng(
                            int(lp_args.get('safe_pool_seed', 6262)))
                        take = rng.choice(
                            safe_q_np.shape[0], size=pool_n, replace=False)
                        safe_q_np = safe_q_np[take]
                    self.function._line_preserve_safe_q = torch.tensor(
                        safe_q_np, dtype=torch.float32, device=device)
                    print(
                        f"[line_preserve] safe pool: "
                        f"{safe_q_np.shape[0]} / {z['q_norm'].shape[0]} "
                        f"from {idx_dir}/safe_index.npz")

                pool_path = lp_args.get('pair_pool')
                if pool_path:
                    pool = np.load(pool_path, allow_pickle=True)
                    if "starts_norm" in pool.files and "goals_norm" in pool.files:
                        starts_np = np.asarray(pool["starts_norm"], dtype=np.float32)
                        goals_np = np.asarray(pool["goals_norm"], dtype=np.float32)
                    elif "q_start_norm" in pool.files and "q_goal_norm" in pool.files:
                        starts_np = np.asarray(pool["q_start_norm"], dtype=np.float32)
                        goals_np = np.asarray(pool["q_goal_norm"], dtype=np.float32)
                    elif "q_start" in pool.files and "q_goal" in pool.files:
                        starts_np = np.asarray(pool["q_start"], dtype=np.float32)
                        goals_np = np.asarray(pool["q_goal"], dtype=np.float32)
                    else:
                        raise ValueError(
                            f"{pool_path} must contain starts_norm/goals_norm "
                            "or q_start_norm/q_goal_norm")
                    if "pair_type" not in pool.files:
                        raise ValueError(
                            f"{pool_path} must contain pair_type for "
                            "line-preserve pair-pool sampling")
                    pair_np = np.asarray(pool["pair_type"], dtype=np.int64)
                    if starts_np.shape != goals_np.shape:
                        raise ValueError(
                            f"line-preserve pair pool shape mismatch: "
                            f"{starts_np.shape} vs {goals_np.shape}")
                    if starts_np.shape[0] != pair_np.shape[0]:
                        raise ValueError(
                            f"line-preserve pair pool pair_type length mismatch: "
                            f"{starts_np.shape[0]} vs {pair_np.shape[0]}")
                    pool_n = int(lp_args.get('pair_pool_n') or 0)
                    if pool_n > 0 and starts_np.shape[0] > pool_n:
                        rng = np.random.default_rng(
                            int(lp_args.get('pair_pool_seed', 6262)))
                        take = rng.choice(
                            starts_np.shape[0], size=pool_n, replace=False)
                        starts_np = starts_np[take]
                        goals_np = goals_np[take]
                        pair_np = pair_np[take]
                    self.function._line_pair_pool_start = torch.tensor(
                        starts_np, dtype=torch.float32, device=device)
                    self.function._line_pair_pool_goal = torch.tensor(
                        goals_np, dtype=torch.float32, device=device)
                    self.function._line_pair_pool_pair_type = torch.tensor(
                        pair_np, dtype=torch.long, device=device)
                    print(
                        f"[line_preserve] pair pool: {starts_np.shape[0]} "
                        f"rows from {pool_path}")

                self._line_preserve_cfg = dict(
                    sample_n=lp_args['sample_n'],
                    checks=lp_args['checks'],
                    margin=lp_args['margin'],
                    self_collision_margin=lp_args['self_collision_margin'],
                    cup_tilt_max=lp_args['cup_tilt_max'],
                    pour_tilt_max=lp_args['pour_tilt_max'],
                    align_weight=lp_args['align_weight'],
                    mono_weight=lp_args['mono_weight'],
                    mono_margin=lp_args['mono_margin'],
                    step_frac=lp_args['step_frac'],
                    start_epoch=lp_args['start_epoch'],
                    pair_types=lp_args['pair_types'],
                    handover_accept=lp_args['handover_accept'],
                    corner_overlap_deg=lp_args['corner_overlap_deg'],
                    safe_pool=lp_args.get('safe_pool', False),
                    local_step_min=lp_args['local_step_min'],
                    local_step_max=lp_args['local_step_max'],
                )
                print(f"[line_preserve] ENABLED: scene={lp_args['scene']} "
                      f"samples={lp_args['sample_n']} checks={lp_args['checks']} "
                      f"pair_types={lp_args['pair_types']} "
                      f"safe_pool={lp_args.get('safe_pool', False)} "
                      f"local_step=[{lp_args['local_step_min']}, "
                      f"{lp_args['local_step_max']}] "
                      f"align_w={lp_args['align_weight']} "
                      f"mono_w={lp_args['mono_weight']} start={lp_args['start_epoch']}")

        if init_checkpoint_state is not None:
            print(f"Initializing compatible weights from checkpoint: {self.init_checkpoint}")
            current_state = self.network.state_dict()
            source_state = dict(init_checkpoint_state['model_state_dict'])
            if self.lifted_phase:
                source_B = source_state.get('B')
                if (source_B is not None and source_B.ndim == 2
                        and source_B.shape[0] + 1 == self.dim):
                    source_state['B'] = torch.cat([
                        source_B,
                        torch.zeros(
                            1, source_B.shape[1],
                            device=source_B.device, dtype=source_B.dtype),
                    ], dim=0)
                source_speed_B = source_state.get('B_speed')
                if (source_speed_B is not None
                        and source_speed_B.ndim == 2
                        and source_speed_B.shape[0] + 1 == self.dim):
                    source_state['B_speed'] = torch.cat([
                        source_speed_B,
                        torch.zeros(
                            1, source_speed_B.shape[1],
                            device=source_speed_B.device,
                            dtype=source_speed_B.dtype),
                    ], dim=0)
            compatible = {
                k: v for k, v in source_state.items()
                if k in current_state and tuple(current_state[k].shape) == tuple(v.shape)
            }
            source_value_arch = init_checkpoint_state.get(
                'zone_params', {}).get('value_arch', 'metric')
            source_endpoint_dual = bool(init_checkpoint_state.get(
                'zone_params', {}).get(
                    'endpoint_dual_value_heads', False))
            source_separate_speed = bool(init_checkpoint_state.get(
                'zone_params', {}).get('separate_speed_encoder', False))
            source_speed_basis = init_checkpoint_state.get(
                'zone_params', {}).get(
                    'speed_fourier_basis', 'multiscale_random')
            source_speed_B = source_state.get('B_speed')
            source_speed_shape_matches = (
                source_speed_B is not None
                and source_speed_B.ndim == 2
                and tuple(source_speed_B.shape) == (
                    self.dim, self.speed_B_freqs))
            reset_speed_branch = (
                self.separate_speed_encoder
                and (not source_separate_speed
                     or source_speed_basis != self.speed_fourier_basis
                     or not source_speed_shape_matches))
            if reset_speed_branch:
                # A shared speed head or a different fixed basis has different
                # feature semantics even when every tensor shape matches.
                # Keep the value field, but initialize the complete scalar
                # branch (including its non-trainable basis) consistently.
                for key in list(compatible):
                    if key == 'B_speed' or key.startswith('speed_'):
                        compatible.pop(key)
                print(
                    "  [init] reset scalar-speed branch for "
                    f"basis={self.speed_fourier_basis}; preserved the smooth "
                    "value trunk/head")
            # ``bounded_chord_endpoint_eikonal_symmetric`` keeps exactly the
            # same sigmoid chord-fraction output as the proven base
            # ``bounded_chord_symmetric`` field and only adds a scalar
            # endpoint state-constraint extension below.  Preserve that
            # fraction on a warm start; resetting it would turn a controlled
            # boundary-condition ablation into a new randomly initialized
            # value field.  Other architecture transitions retain the
            # conservative reset because their output semantics differ.
            preserve_bounded_fraction = (
                self.value_arch in (
                    'bounded_chord_endpoint_eikonal_symmetric',
                    'bounded_chord_local_endpoint_eikonal_symmetric')
                and source_value_arch == 'bounded_chord_symmetric')
            if (self.value_arch in (
                    'bounded_chord_symmetric',
                    'bounded_chord_log_symmetric',
                    'bounded_chord_eikonal_symmetric',
                    'bounded_chord_endpoint_eikonal_symmetric',
                    'factored_source_eikonal_symmetric',
                    'bounded_chord_local_endpoint_eikonal_symmetric',
                    'local_endpoint_eikonal_residual_symmetric')
                    and source_value_arch != self.value_arch
                    and not preserve_bounded_fraction):
                # The positive global-residual model squares tanh(raw), so
                # the sign of its last-layer output is intentionally
                # unidentifiable.  Reusing that signed output as a sigmoid
                # chord fraction can saturate the new architecture at zero.
                # Keep the learned trunk/speed field, but retain the new
                # head's semantically correct initialization.
                last_idx = len(self.network.global_residual_net) - 1
                for suffix in ('weight', 'bias'):
                    compatible.pop(
                        f'global_residual_net.{last_idx}.{suffix}', None)
                print(
                    "  [init] reset value output layer because "
                    f"source value_arch={source_value_arch}")
            # Warm-start input-expanded Linear layers, e.g. task_scene ->
            # task_scene_dist adds one condition column to local_finsler/asym
            # value MLPs. Copy the old columns exactly and zero the new columns
            # so the initial function matches the source checkpoint.
            if self.lifted_phase:
                # bounded-chord pair input layout is
                # [3*latent, |dq|, distance, speed_sum, speed_diff].  The
                # seventh |dq| coordinate is inserted before the last three
                # scalars, so a naive right-padding would silently move their
                # learned weights onto the phase column.
                key = 'global_residual_net.0.weight'
                old = source_state.get(key)
                cur = current_state.get(key)
                h_size = 2 * self.B_freqs
                insert = 3 * h_size + (self.dim - 1)
                if (old is not None and cur is not None
                        and old.ndim == 2 and cur.ndim == 2
                        and cur.shape[0] == old.shape[0]
                        and cur.shape[1] == old.shape[1] + 1):
                    expanded = cur.new_zeros(cur.shape)
                    expanded[:, :insert] = old[:, :insert]
                    expanded[:, insert + 1:] = old[:, insert:]
                    compatible[key] = expanded
            for k, v in source_state.items():
                if k in compatible or k not in current_state:
                    continue
                if (reset_speed_branch
                        and (k == 'B_speed' or k.startswith('speed_'))):
                    continue
                cur = current_state[k]
                if (torch.is_tensor(v) and torch.is_tensor(cur)
                        and v.ndim == 2 and cur.ndim == 2
                        and cur.shape[0] == v.shape[0]
                        and cur.shape[1] > v.shape[1]
                        and v.dtype.is_floating_point):
                    expanded = cur.clone()
                    expanded[:, :v.shape[1]] = v
                    expanded[:, v.shape[1]:] = 0.0
                    compatible[k] = expanded
            if (self.value_arch in (
                    'bounded_chord_speed_barrier_symmetric',
                    'bounded_chord_local_speed_barrier_symmetric',
                    'bounded_chord_global_excess_symmetric',
                    'bounded_chord_smooth_excess_symmetric',
                    'bounded_chord_signed_clearance_symmetric',
                    'bounded_chord_eikonal_refine_symmetric')
                    and source_value_arch in (
                        'bounded_chord_symmetric',
                        'bounded_chord_endpoint_eikonal_symmetric',
                        'bounded_chord_local_endpoint_eikonal_symmetric',
                        'bounded_chord_eikonal_refine_symmetric',
                        'bounded_chord_speed_barrier_symmetric',
                        'bounded_chord_local_speed_barrier_symmetric')):
                # Preserve E217's learned fraction exactly in output row 0;
                # retain the new auxiliary initialization in output row 1.
                last_idx = len(self.network.global_residual_net) - 1
                for suffix in ('weight', 'bias'):
                    key = f'global_residual_net.{last_idx}.{suffix}'
                    source = source_state.get(key)
                    target = current_state.get(key)
                    if (source is None or target is None
                            or target.ndim != source.ndim
                            or target.shape[0] != source.shape[0] + 1
                            or tuple(target.shape[1:]) !=
                            tuple(source.shape[1:])):
                        raise ValueError(
                            f"cannot expand bounded-chord output {key}: "
                            f"source={getattr(source, 'shape', None)} "
                            f"target={getattr(target, 'shape', None)}")
                    expanded = target.clone()
                    expanded[:source.shape[0]] = source
                    compatible[key] = expanded
            if self.use_task_adapters:
                # Task-adapter runs replace separate cup/pour encoders with one
                # shared trunk. Warm-start that trunk from the average of the
                # already-trained task encoders instead of leaving it random.
                for k in list(current_state.keys()):
                    if k.startswith('shared_encoder.'):
                        suffix = k[len('shared_encoder.'):]
                        ck = f'cup_encoder.{suffix}'
                        pk = f'pour_encoder.{suffix}'
                        if (ck in source_state and pk in source_state
                                and tuple(source_state[ck].shape) == tuple(current_state[k].shape)
                                and tuple(source_state[pk].shape) == tuple(current_state[k].shape)):
                            compatible[k] = 0.5 * (source_state[ck] + source_state[pk])
                    elif k.startswith('shared_gate.'):
                        suffix = k[len('shared_gate.'):]
                        ck = f'cup_gate.{suffix}'
                        pk = f'pour_gate.{suffix}'
                        if (ck in source_state and pk in source_state
                                and tuple(source_state[ck].shape) == tuple(current_state[k].shape)
                                and tuple(source_state[pk].shape) == tuple(current_state[k].shape)):
                            compatible[k] = 0.5 * (source_state[ck] + source_state[pk])
            if self.endpoint_dual_value_heads and not source_endpoint_dual:
                # The pure scalar warm start used one trained endpoint trunk;
                # its legacy cup/pour modules were present only for checkpoint
                # compatibility and remained random.  Seed both active value
                # heads from the trained trunk instead of accidentally loading
                # those unused tensors.  Task-specific batches immediately
                # give the two copies different gradients.
                copied = 0
                for target_prefix, source_prefix in (
                        ('cup_encoder.', 'siamese_encoder.'),
                        ('pour_encoder.', 'siamese_encoder.'),
                        ('cup_gate.', 'siamese_gate.'),
                        ('pour_gate.', 'siamese_gate.')):
                    for key in current_state:
                        if not key.startswith(target_prefix):
                            continue
                        suffix = key[len(target_prefix):]
                        source_key = source_prefix + suffix
                        source = source_state.get(source_key)
                        if (source is not None
                                and tuple(source.shape)
                                == tuple(current_state[key].shape)):
                            compatible[key] = source.clone()
                            copied += 1
                if self.separate_pe_gate:
                    for target_prefix in ('cup_pe_gate.', 'pour_pe_gate.'):
                        for key in current_state:
                            if not key.startswith(target_prefix):
                                continue
                            suffix = key[len(target_prefix):]
                            source = source_state.get('pe_gate.' + suffix)
                            if (source is not None
                                    and tuple(source.shape)
                                    == tuple(current_state[key].shape)):
                                compatible[key] = source.clone()
                                copied += 1
                if self.separate_fourier:
                    source_B = source_state.get('B')
                    for key in ('B_cup', 'B_pour'):
                        if (source_B is not None and key in current_state
                                and tuple(source_B.shape)
                                == tuple(current_state[key].shape)):
                            compatible[key] = source_B.clone()
                            copied += 1
                if self.use_projector and 'projector.weight' not in source_state:
                    weight = current_state['projector.weight'].new_zeros(
                        current_state['projector.weight'].shape)
                    diagonal = min(weight.shape)
                    weight[:diagonal, :diagonal] = torch.eye(
                        diagonal, device=weight.device, dtype=weight.dtype)
                    compatible['projector.weight'] = weight
                    compatible['projector.bias'] = torch.zeros_like(
                        current_state['projector.bias'])
                    copied += 2
                print(
                    "  [init] seeded endpoint dual value heads from the "
                    f"trained scalar trunk ({copied} tensors)")
            skipped = sorted(set(source_state.keys()) - set(compatible.keys()))
            missing = sorted(set(current_state.keys()) - set(compatible.keys()))
            self.network.load_state_dict(compatible, strict=False)
            print(f"  [init] loaded {len(compatible)} tensors; "
                  f"skipped {len(skipped)} source tensors; "
                  f"new/uninitialized {len(missing)} tensors")
            if skipped[:5]:
                print(f"  [init] skipped examples: {skipped[:5]}")
            if missing[:5]:
                print(f"  [init] new examples: {missing[:5]}")
            source_constraint_heads = int(init_checkpoint_state.get(
                'zone_params', {}).get('speed_constraint_heads', 0))
            if (self.speed_constraint_heads > 0
                    and source_constraint_heads == 0):
                # Structural, evaluator-free warm-start certificate.  Disable
                # only the analytic reduction briefly; every inherited tensor
                # remains fixed.  Zero private residuals must make the new
                # field indistinguishable from the source field.
                sample_count = 64
                audit_points = torch.linspace(
                    -0.45, 0.45, sample_count * self.dim,
                    device=self.Params['Device'], dtype=self.B.dtype
                ).reshape(sample_count, self.dim)
                audit_goals = torch.flip(audit_points, dims=[0])
                audit_pairs = torch.cat([audit_points, audit_goals], dim=1)
                audit_reverse = torch.cat(
                    [audit_goals, audit_points], dim=1)
                audit_diagonal = torch.cat(
                    [audit_points, audit_points], dim=1)
                enabled_count = self.network.speed_constraint_head_count
                with torch.no_grad():
                    try:
                        self.network.speed_constraint_head_count = 0
                        inherited_speed = self.network.predict_speed(
                            audit_points)
                        inherited_tau = self.network.out(
                            audit_pairs, detach_input=False)[0]
                    finally:
                        self.network.speed_constraint_head_count = enabled_count
                    factored_speed = self.network.predict_speed(audit_points)
                    factored_tau = self.network.out(
                        audit_pairs, detach_input=False)[0]
                    reverse_tau = self.network.out(
                        audit_reverse, detach_input=False)[0]
                    diagonal_tau = self.network.out(
                        audit_diagonal, detach_input=False)[0]
                warm_audit = {
                    'source_constraint_heads': source_constraint_heads,
                    'target_constraint_heads': enabled_count,
                    'sample_count': sample_count,
                    'speed_max_abs_change': float(
                        (factored_speed - inherited_speed).abs().max()),
                    'travel_time_max_abs_change': float(
                        (factored_tau - inherited_tau).abs().max()),
                    'swap_max_abs_error': float(
                        (factored_tau - reverse_tau).abs().max()),
                    'diagonal_max_abs_value': float(diagonal_tau.abs().max()),
                    'tolerance': 1e-6,
                }
                audit_path = os.path.join(
                    self.folder, 'constraint_warm_start_audit.json')
                with open(audit_path, 'w') as handle:
                    json.dump(warm_audit, handle, indent=2)
                print(f"  [constraint_warm_start] {warm_audit}")
                if any(warm_audit[key] > warm_audit['tolerance'] for key in (
                        'speed_max_abs_change',
                        'travel_time_max_abs_change',
                        'swap_max_abs_error',
                        'diagonal_max_abs_value')):
                    raise RuntimeError(
                        "constraint-head warm start failed the preregistered "
                        f"function-preservation audit: {warm_audit}")
            self.network.train()
            self.function.travel_time_weight = self.Params['Training'].get('Travel Time Weight', 0.5)

        if self.train_speed_only:
            if self.separate_speed_encoder:
                train_prefixes = ["speed_"]
            else:
                train_prefixes = [
                    "siamese_encoder.",
                    "siamese_gate.",
                    "pe_gate.",
                    "speed_head.",
                ]
            if self.use_raw_coordinate_skip:
                train_prefixes.append("raw_coordinate_projection.")
            if self.use_clearance_head:
                # Jointly adapt the one scalar-geometry trunk to both the
                # historical task speed and signed clearance.  A frozen trunk
                # trained on clipped F cannot recover depth information that
                # its old objective encouraged it to discard.
                train_prefixes.append("clearance_head.")
                train_prefixes.append("clearance_residual_blocks.")
            for name, param in self.network.named_parameters():
                param.requires_grad = any(
                    name.startswith(prefix) for prefix in train_prefixes)
            trainable = sum(
                p.numel() for p in self.network.parameters()
                if p.requires_grad)
            frozen = sum(
                p.numel() for p in self.network.parameters()
                if not p.requires_grad)
            print(
                "[speed_only] training the shared scalar-geometry trunk"
                f"{' and clearance head' if self.use_clearance_head else ''}; "
                f"trainable={trainable}, frozen={frozen}")

        if self.train_speed_head_only:
            for name, param in self.network.named_parameters():
                # A newly appended zero-initialized speed residual stack is
                # part of the scalar speed parameterization, not of the
                # shared value encoder.  Let the calibration-only mode fit
                # that stack together with the final speed head while the
                # inherited encoder and endpoint value parameters remain
                # frozen.  With no residual stack this is exactly the
                # historical speed-head-only mask.
                param.requires_grad = (
                    name.startswith("speed_head.")
                    or name.startswith("speed_residual_blocks."))
            trainable = sum(
                p.numel() for p in self.network.parameters()
                if p.requires_grad)
            frozen = sum(
                p.numel() for p in self.network.parameters()
                if not p.requires_grad)
            print(
                "[speed_head_only] calibrating only the scalar output head; "
                f"trainable={trainable}, frozen={frozen}")

        if self.train_speed_constraint_heads_only:
            prefix = "speed_constraint_heads."
            for name, param in self.network.named_parameters():
                param.requires_grad = name.startswith(prefix)
            trainable = sum(
                param.numel() for param in self.network.parameters()
                if param.requires_grad)
            frozen = sum(
                param.numel() for param in self.network.parameters()
                if not param.requires_grad)
            if trainable == 0:
                raise RuntimeError(
                    "network has no scalar constraint-head parameters")
            print(
                "[speed_constraint_heads_only] preserving every inherited "
                "speed/value tensor and training only the semantic private "
                f"tails; heads={self.speed_constraint_heads}, "
                f"trainable={trainable}, frozen={frozen}")

        if self.train_speed_consensus_only:
            prefixes = (
                "speed_constraint_consensus_head.",
                "speed_constraint_consensus_trunk.",
                "speed_constraint_consensus_heads.",
            )
            for name, param in self.network.named_parameters():
                param.requires_grad = name.startswith(prefixes)
            trainable = sum(
                parameter.numel() for parameter in self.network.parameters()
                if parameter.requires_grad)
            frozen = sum(
                parameter.numel() for parameter in self.network.parameters()
                if not parameter.requires_grad)
            if trainable == 0:
                raise RuntimeError(
                    "network has no scalar consensus-head parameters")
            print(
                "[speed_consensus_only] preserving both fixed scalar "
                "experts and training only q-dependent alpha; "
                f"mode={self.speed_constraint_consensus_mode}, "
                f"trainable={trainable}, frozen={frozen}")

        if self.train_speed_consensus_magnitude_only:
            if (self.speed_constraint_consensus_mode
                    == "corridor_hurdle_semantic_veto_min_odds_residual"):
                prefix = "speed_constraint_consensus_magnitude_residual."
            elif (self.speed_constraint_consensus_mode
                    == "corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend"):
                prefix = (
                    "speed_constraint_consensus_evidence_blend_residual.")
            elif (self.speed_constraint_consensus_mode
                    == "corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8"):
                prefix = (
                    "speed_constraint_consensus_head_logit_residual.")
            elif self.speed_constraint_consensus_mode in model_network.ROUTE_CHART_MODES:
                if (self.speed_constraint_consensus_mode
                        in model_network.ROUTE_CHART_PARTIAL_SHARED_MODES):
                    prefix = (
                        "speed_constraint_consensus_route_experts.",
                        "speed_constraint_route_fourier_branch.")
                else:
                    prefix = "speed_constraint_consensus_route_experts."
            else:
                prefix = "speed_constraint_consensus_heads.magnitude."
            for name, param in self.network.named_parameters():
                param.requires_grad = name.startswith(prefix)
            trainable = sum(
                parameter.numel() for parameter in self.network.parameters()
                if parameter.requires_grad)
            frozen = sum(
                parameter.numel() for parameter in self.network.parameters()
                if not parameter.requires_grad)
            if trainable == 0:
                raise RuntimeError(
                "network has no selected scalar consensus private-tail "
                "parameters")
            print(
                "[speed_consensus_magnitude_only] preserving the fixed "
                "release head, shared trunk, both scalar experts, and all "
                "other tensors; training only the registered compensated "
                f"private tail ({prefix}); trainable={trainable}, "
                f"frozen={frozen}")

        if self.train_speed_consensus_specificity_magnitude_only:
            prefixes = (
                "speed_constraint_consensus_heads.specificity.",
                "speed_constraint_consensus_heads.magnitude.",
            )
            for name, param in self.network.named_parameters():
                param.requires_grad = name.startswith(prefixes)
            trainable = sum(
                parameter.numel() for parameter in self.network.parameters()
                if parameter.requires_grad)
            frozen = sum(
                parameter.numel() for parameter in self.network.parameters()
                if not parameter.requires_grad)
            if trainable == 0:
                raise RuntimeError(
                    "network has no specificity/magnitude parameters")
            print(
                "[speed_consensus_specificity_magnitude_only] preserving "
                "the fixed release head, shared trunk, both scalar experts, "
                "and all other tensors; training only specificity and "
                f"magnitude private tails; trainable={trainable}, "
                f"frozen={frozen}")

        if self.train_speed_repair_only:
            prefixes = (
                "speed_constraint_repair_trunk.",
                "speed_constraint_repair_heads.",
            )
            for name, param in self.network.named_parameters():
                param.requires_grad = name.startswith(prefixes)
            trainable = sum(
                parameter.numel() for parameter in self.network.parameters()
                if parameter.requires_grad)
            frozen = sum(
                parameter.numel() for parameter in self.network.parameters()
                if not parameter.requires_grad)
            if trainable == 0:
                raise RuntimeError(
                    "network has no scalar component-repair parameters")
            print(
                "[speed_repair_only] preserving both fixed scalar experts "
                "and training only shared-trunk/private-tail per-factor "
                f"repair; trainable={trainable}, frozen={frozen}")

        if self.train_clearance_only:
            for name, param in self.network.named_parameters():
                param.requires_grad = (
                    name.startswith("clearance_head.")
                    or name.startswith("clearance_residual_blocks."))
            trainable = sum(
                p.numel() for p in self.network.parameters()
                if p.requires_grad)
            frozen = sum(
                p.numel() for p in self.network.parameters()
                if not p.requires_grad)
            print(
                "[clearance_only] fitting signed scalar clearance; "
                f"trainable={trainable}, frozen={frozen}")

        self._chord_profile_hooks = []
        if self.train_chord_profile_only:
            profile_cols = self.chord_quad_points + 1
            target_name = "global_residual_net.0.weight"
            for name, param in self.network.named_parameters():
                param.requires_grad = name == target_name
                if param.requires_grad:
                    mask = torch.zeros_like(param)
                    mask[:, -profile_cols:] = 1.0
                    self._chord_profile_hooks.append(
                        param.register_hook(
                            lambda grad, mask=mask: grad * mask))
            trainable = sum(
                p.numel() for p in self.network.parameters()
                if p.requires_grad)
            frozen = sum(
                p.numel() for p in self.network.parameters()
                if not p.requires_grad)
            print(
                "[chord_profile_only] training only appended scalar-profile "
                f"columns={profile_cols}; matrix_params={trainable}, "
                f"effective_params={self.network.global_residual_net[0].weight.shape[0] * profile_cols}, "
                f"frozen={frozen}")

        if self.freeze_base_for_safety_head:
            train_prefixes = []
            if self.use_safety_head:
                train_prefixes.append("safety_head.")
            if self.use_orient_head:
                train_prefixes.append("orient_head.")
            for name, param in self.network.named_parameters():
                param.requires_grad = any(name.startswith(prefix) for prefix in train_prefixes)
            trainable = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
            frozen = sum(p.numel() for p in self.network.parameters() if not p.requires_grad)
            print(f"[safety] freeze_base_for_safety_head=True; "
                  f"trainable={trainable}, frozen={frozen}")

        if self.freeze_base_for_value_head:
            train_prefixes = []
            if self.value_arch == "local_finsler":
                train_prefixes.extend([
                    "local_slowness_net.",
                    "local_slowness_cup.",
                    "local_slowness_pour.",
                ])
            elif self.value_arch == "asym_goal":
                train_prefixes.append("asym_value_net.")
            elif self.value_arch in (
                    "global_residual", "global_residual_symmetric",
                    "bounded_chord_symmetric",
                    "bounded_chord_fermat_curve_symmetric",
                    "bounded_chord_endpoint_fermat_curve_symmetric",
                    "bounded_chord_endpoint_speed_fermat_curve_symmetric",
                    "bounded_chord_log_symmetric",
                    "chord_integral_symmetric",
                    "bounded_chord_hazard_symmetric",
                    "bounded_chord_hazard_floor_symmetric",
                    "bounded_chord_eikonal_symmetric",
                    "bounded_chord_eikonal_refine_symmetric",
                    "bounded_chord_endpoint_eikonal_symmetric",
                    "factored_source_eikonal_symmetric",
                    "bounded_chord_local_endpoint_eikonal_symmetric",
                    "bounded_chord_speed_barrier_symmetric",
                    "bounded_chord_local_speed_barrier_symmetric",
                    "bounded_chord_global_excess_symmetric",
                    "bounded_chord_smooth_excess_symmetric",
                    "bounded_chord_separate_smooth_excess_symmetric",
                    "bounded_chord_separate_global_excess_symmetric",
                    "bounded_chord_neural_viscosity_symmetric",
                    "bounded_chord_signed_clearance_symmetric",
                    "bounded_chord_clearance_eikonal_symmetric",
                    "bounded_chord_clearance_state_symmetric",
                    "endpoint_eikonal_residual_symmetric",
                    "local_endpoint_eikonal_residual_symmetric",
                    "bounded_chord_profile_symmetric",
                    "fermat_waypoint_symmetric",
                    "fermat_curve_symmetric",
                    "fermat_curve_equivariant"):
                train_prefixes.append("global_residual_net.")
                train_prefixes.append("global_pair_residual_blocks.")
                if self.value_arch in (
                        "bounded_chord_fermat_curve_symmetric",
                        "bounded_chord_endpoint_fermat_curve_symmetric",
                        "bounded_chord_endpoint_speed_fermat_curve_symmetric"):
                    train_prefixes.append("fermat_curve_net.")
                    train_prefixes.append("fermat_curve_residual_blocks.")
                    if (self.value_arch ==
                            "bounded_chord_endpoint_speed_fermat_curve_symmetric"):
                        train_prefixes.append("fermat_speed_profile_net.")
            else:
                train_prefixes.append("sym_op.")
            for name, param in self.network.named_parameters():
                param.requires_grad = any(
                    name.startswith(prefix) for prefix in train_prefixes)
            trainable = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
            frozen = sum(p.numel() for p in self.network.parameters() if not p.requires_grad)
            print(f"[value_head] freeze_base_for_value_head=True; "
                  f"value_arch={self.value_arch} trainable={trainable}, "
                  f"frozen={frozen}")

        self._endpoint_barrier_hooks = []
        if self.train_endpoint_barrier_only:
            separate_excess = self.value_arch in (
                "bounded_chord_separate_smooth_excess_symmetric",
                "bounded_chord_separate_global_excess_symmetric")
            if separate_excess:
                targets = {
                    name for name, _ in self.network.named_parameters()
                    if name.startswith("global_excess_net.")
                }
            else:
                last_idx = len(self.network.global_residual_net) - 1
                targets = {
                    f"global_residual_net.{last_idx}.weight",
                    f"global_residual_net.{last_idx}.bias",
                }
            for name, param in self.network.named_parameters():
                param.requires_grad = name in targets
                if param.requires_grad and not separate_excess:
                    mask = torch.zeros_like(param)
                    mask[1] = 1.0
                    self._endpoint_barrier_hooks.append(
                        param.register_hook(
                            lambda grad, mask=mask: grad * mask))
            trainable = sum(
                p.numel() for p in self.network.parameters()
                if p.requires_grad)
            if separate_excess:
                effective = trainable
            else:
                effective = sum(
                    int(p[1].numel()) for name, p in
                    self.network.named_parameters() if name in targets)
            frozen = sum(
                p.numel() for p in self.network.parameters()
                if not p.requires_grad)
            if separate_excess:
                print(
                    "[endpoint_barrier_only] preserving the inherited scalar "
                    "field and training the independent smooth excess MLP; "
                    f"trainable={trainable}, effective={effective}, "
                    f"frozen={frozen}")
            else:
                print(
                    "[endpoint_barrier_only] preserving the E217 fraction and "
                    f"training only auxiliary row; matrix_params={trainable}, "
                    f"effective_params={effective}, frozen={frozen}")

        if self.train_viscosity_excess_only:
            branch = getattr(
                self.network, 'viscosity_excess_net', None)
            if branch is None:
                raise RuntimeError(
                    "network has no viscosity_excess_net to train")
            for name, param in self.network.named_parameters():
                param.requires_grad = (
                    name.startswith("viscosity_excess_net.")
                    or name.startswith("viscosity_minplus_trunk.")
                    or name.startswith("viscosity_minplus_heads.")
                    or name.startswith(
                        "viscosity_chord_latent_projection.")
                    # When the value-head isolation flag is also set, retain
                    # the frozen-speed scalar endpoint value head as a second
                    # trainable PDE component.  This is still one scalar
                    # field; the combination simply avoids the all-network
                    # drift of an unconstrained joint fine-tune.
                    or (self.freeze_base_for_value_head and (
                        name.startswith("global_residual_net.")
                        or name.startswith("global_pair_residual_blocks."))))
            trainable = sum(
                p.numel() for p in self.network.parameters()
                if p.requires_grad)
            frozen = sum(
                p.numel() for p in self.network.parameters()
                if not p.requires_grad)
            mode_note = (
                " plus the isolated scalar endpoint value head"
                if self.freeze_base_for_value_head else "")
            print(
                "[viscosity_excess_only] preserving the complete E260 "
                "value/speed field and training only the conditional "
                f"viscosity branch{mode_note} (including any internal "
                f"min-plus envelope); trainable={trainable}, "
                f"frozen={frozen}")

        self._viscosity_input_hooks = []
        if self.train_viscosity_input_tail_cols > 0:
            branch = getattr(
                self.network, 'viscosity_excess_net', None)
            if branch is None or not isinstance(branch[0], torch.nn.Linear):
                raise RuntimeError(
                    "network has no Linear viscosity input layer to train")
            target_name = "viscosity_excess_net.0.weight"
            input_width = int(branch[0].weight.shape[1])
            tail_cols = self.train_viscosity_input_tail_cols
            if tail_cols > input_width:
                raise ValueError(
                    "train_viscosity_input_tail_cols exceeds viscosity "
                    f"input width: {tail_cols} > {input_width}")
            target = None
            for name, param in self.network.named_parameters():
                param.requires_grad = name == target_name
                if name == target_name:
                    target = param
            if target is None:
                raise RuntimeError(
                    f"could not find trainable tensor {target_name}")
            mask = torch.zeros_like(target)
            mask[:, -tail_cols:] = 1.0
            self._viscosity_input_hooks.append(
                target.register_hook(
                    lambda grad, mask=mask: grad * mask))
            matrix_params = target.numel()
            effective_params = target.shape[0] * tail_cols
            frozen = sum(
                p.numel() for p in self.network.parameters()
                if not p.requires_grad)
            print(
                "[viscosity_input_tail] preserving the complete warm-started "
                "field and training only the newly appended scalar-context "
                f"columns; input_width={input_width}, tail_cols={tail_cols}, "
                f"matrix_params={matrix_params}, "
                f"effective_params={effective_params}, frozen={frozen}")

        self._value_input_hooks = []
        if self.train_value_input_tail_cols > 0:
            branch = getattr(self.network, 'global_residual_net', None)
            if branch is None or not isinstance(branch[0], torch.nn.Linear):
                raise RuntimeError(
                    "network has no Linear value input layer to train")
            target_name = "global_residual_net.0.weight"
            input_width = int(branch[0].weight.shape[1])
            tail_cols = self.train_value_input_tail_cols
            if tail_cols > input_width:
                raise ValueError(
                    "train_value_input_tail_cols exceeds value input width: "
                    f"{tail_cols} > {input_width}")
            target = None
            for name, param in self.network.named_parameters():
                param.requires_grad = name == target_name
                if name == target_name:
                    target = param
            if target is None:
                raise RuntimeError(
                    f"could not find trainable tensor {target_name}")
            mask = torch.zeros_like(target)
            mask[:, -tail_cols:] = 1.0
            self._value_input_hooks.append(
                target.register_hook(
                    lambda grad, mask=mask: grad * mask))
            matrix_params = target.numel()
            effective_params = target.shape[0] * tail_cols
            frozen = sum(
                p.numel() for p in self.network.parameters()
                if not p.requires_grad)
            print(
                "[value_input_tail] preserving the complete warm-started "
                "field and training only the newly appended symmetric "
                f"speed-latent columns; input_width={input_width}, "
                f"tail_cols={tail_cols}, matrix_params={matrix_params}, "
                f"effective_params={effective_params}, frozen={frozen}")

        if self.train_lifted_phase_speed_only:
            prefix = "lifted_phase_speed_net."
            for name, param in self.network.named_parameters():
                param.requires_grad = name.startswith(prefix)
            trainable = sum(
                param.numel() for param in self.network.parameters()
                if param.requires_grad)
            frozen = sum(
                param.numel() for param in self.network.parameters()
                if not param.requires_grad)
            if trainable == 0:
                raise RuntimeError(
                    "network has no lifted_phase_speed_net parameters")
            print(
                "[lifted_phase_speed_only] preserving the complete q-only "
                "value/speed field and training only the one-scalar lifted "
                f"phase restriction; trainable={trainable}, frozen={frozen}")

        if self.train_last_pair_residual_only:
            blocks = getattr(
                self.network, 'global_pair_residual_blocks', None)
            if blocks is None or len(blocks) < 1:
                raise RuntimeError(
                    "network has no global pair residual block to train")
            last_idx = len(blocks) - 1
            train_prefix = f"global_pair_residual_blocks.{last_idx}."
            for name, param in self.network.named_parameters():
                param.requires_grad = name.startswith(train_prefix)
            trainable = sum(
                p.numel() for p in self.network.parameters()
                if p.requires_grad)
            frozen = sum(
                p.numel() for p in self.network.parameters()
                if not p.requires_grad)
            print(
                "[last_pair_residual_only] preserving the warm-started "
                f"value/speed field and training only block {last_idx}; "
                f"trainable={trainable}, frozen={frozen}")

        if self.train_partial_shared_embedding_only:
            prefix = "partial_shared_"
            for name, param in self.network.named_parameters():
                param.requires_grad = name.startswith(prefix)
            trainable = sum(
                p.numel() for p in self.network.parameters()
                if p.requires_grad)
            frozen = sum(
                p.numel() for p in self.network.parameters()
                if not p.requires_grad)
            if trainable == 0:
                raise RuntimeError(
                    "network has no partial_shared embedding parameters")
            print(
                "[partial_shared_embedding_only] preserving both inherited "
                "private branches and training only the common stem plus E/F "
                f"adapters; dim={self.partial_shared_embedding_dim}, "
                f"trainable={trainable}, frozen={frozen}")

        if self.freeze_base_for_cross_residual:
            for name, param in self.network.named_parameters():
                if self.train_local_residual_only:
                    param.requires_grad = name.startswith("local_slowness_residual.")
                else:
                    param.requires_grad = (
                        name.startswith("cross_slowness_residual.")
                        or name.startswith("local_slowness_residual."))
            trainable = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
            frozen = sum(p.numel() for p in self.network.parameters() if not p.requires_grad)
            print(f"[cross_residual] freeze_base_for_cross_residual=True; "
                  f"trainable={trainable}, frozen={frozen}, "
                  f"cross_scale={self.cross_residual_scale}, "
                  f"local_scale={self.local_residual_scale}, "
                  f"local_gate={self.local_residual_gate}, "
                  f"local_only={self.train_local_residual_only}")

        self._dist_condition_hooks = []
        if self.Params['Training'].get('Dist Condition Only', False):
            if self.value_condition != "task_scene_dist":
                raise ValueError(
                    "--dist_condition_only requires --value_condition task_scene_dist")
            train_names = {
                "local_slowness_net.0.weight",
                "local_slowness_cup.0.weight",
                "local_slowness_pour.0.weight",
                "asym_value_net.0.weight",
            }
            for name, param in self.network.named_parameters():
                param.requires_grad = name in train_names
                if param.requires_grad:
                    mask = torch.zeros_like(param)
                    mask[:, -1] = 1.0
                    self._dist_condition_hooks.append(
                        param.register_hook(
                            lambda grad, mask=mask: grad * mask))
            trainable = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
            frozen = sum(p.numel() for p in self.network.parameters() if not p.requires_grad)
            print(f"[dist_condition] training only appended distance columns; "
                  f"trainable_matrix_params={trainable}, frozen={frozen}")

        self._pair_condition_hooks = []
        if self.Params['Training'].get('Pair Condition Only', False):
            if self.value_condition not in ("task_scene_pair", "task_scene_pair_dist"):
                raise ValueError(
                    "--pair_condition_only requires --value_condition "
                    "task_scene_pair or task_scene_pair_dist")
            train_names = {
                "local_slowness_net.0.weight",
                "local_slowness_cup.0.weight",
                "local_slowness_pour.0.weight",
                "asym_value_net.0.weight",
            }
            n_new_cols = 14 if self.value_condition == "task_scene_pair" else 15
            for name, param in self.network.named_parameters():
                param.requires_grad = name in train_names
                if param.requires_grad:
                    if param.shape[1] < n_new_cols:
                        raise ValueError(
                            f"{name} has too few columns for pair-condition mask: "
                            f"{tuple(param.shape)}")
                    mask = torch.zeros_like(param)
                    mask[:, -n_new_cols:] = 1.0
                    self._pair_condition_hooks.append(
                        param.register_hook(
                            lambda grad, mask=mask: grad * mask))
            trainable = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
            frozen = sum(p.numel() for p in self.network.parameters() if not p.requires_grad)
            print(f"[pair_condition] training only appended pair-context columns; "
                  f"new_cols={n_new_cols} trainable_matrix_params={trainable}, "
                  f"frozen={frozen}")

        print('train (multi-task)')

        # Fitted value iteration uses a periodically synchronized, frozen
        # target only while constructing Bellman labels.  It is deliberately
        # attached to the loss object rather than the deployed network, so it
        # is absent from every saved checkpoint and from inference.
        self._viscosity_target_network = None
        if self.viscosity_target_lag > 0:
            self._viscosity_target_network = copy.deepcopy(self.network)
            self._viscosity_target_network.eval()
            for param in self._viscosity_target_network.parameters():
                param.requires_grad_(False)
            self.function.viscosity_target_network = (
                self._viscosity_target_network)
            print(
                "[viscosity_target] fitted value iteration enabled; "
                f"sync every {self.viscosity_target_lag} epochs; "
                "target is training-only and is not checkpointed")

        optimizer_params = [p for p in self.network.parameters() if p.requires_grad]
        if not optimizer_params:
            raise RuntimeError("No trainable parameters selected")
        weight_decay_override = self.Params['Training'].get('Weight Decay Override', None)
        weight_decay = (
            float(weight_decay_override) if weight_decay_override is not None
            else 0.0 if (
                self.Params['Training'].get('Dist Condition Only', False)
                or self.Params['Training'].get('Pair Condition Only', False)
                or self.train_speed_only
                or self.train_speed_head_only
                or self.train_speed_constraint_heads_only
                or self.train_speed_consensus_only
                or self.train_speed_consensus_magnitude_only
                or self.train_speed_consensus_specificity_magnitude_only
                or self.train_speed_repair_only
                or self.train_clearance_only
                or self.train_chord_profile_only
                or self.train_endpoint_barrier_only
                or self.train_viscosity_excess_only
                or self.train_viscosity_input_tail_cols > 0
                or self.train_value_input_tail_cols > 0
                or self.train_lifted_phase_speed_only
                or self.train_last_pair_residual_only
                or self.train_partial_shared_embedding_only
                or self.freeze_base_for_value_head
                or self.freeze_base_for_cross_residual)
            else 0.5)
        self.optimizer = torch.optim.AdamW(
            optimizer_params, lr=self.Params['Training']['Learning Rate'], weight_decay=weight_decay)

        if self.Params['Training']['Use Scheduler (bool)']:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)

        # Load multi-task dataset
        self.dataset = db.DatabaseMulti(self.Params['DataPath'])
        collocation_points = getattr(self.dataset, 'collocation_points', None)
        collocation_speed = getattr(self.dataset, 'collocation_speed', None)
        collocation_speed_components = getattr(
            self.dataset, 'collocation_speed_components', None)
        if collocation_points is not None:
            # The endpoint tensor remains CPU-shuffled by FastTensorDataLoader;
            # the independent PDE cloud is sampled inside the loss and kept on
            # the training device for the whole run.
            collocation_points = collocation_points.to(
                self.Params['Device'])
            if collocation_speed is not None:
                collocation_speed = collocation_speed.to(
                    self.Params['Device'])
            if collocation_speed_components is not None:
                collocation_speed_components = (
                    collocation_speed_components.to(self.Params['Device']))
            print(f"[PDE] independent collocation cloud enabled: "
                  f"{tuple(collocation_points.shape)} "
                  f"({'with scalar-speed labels' if collocation_speed is not None else 'measure-only'})")

        len_dataset = len(self.dataset)
        training_start_time = time.time()

        dataloader = FastTensorDataLoader(
            self.dataset.data,
            batch_size=int(self.Params['Training']['Batch Size']),
            shuffle=True)

        # Data layout: 14 + 2 + 2 + 14 + 14 + 2 + 2 + 2 + 2 + 1 = 55
        dim = self.dim  # 7
        # Column offsets
        COL_POINTS = 0                        # 0:14
        COL_SPEED_CUP = 2 * dim              # 14:16
        COL_SPEED_POUR = 2 * dim + 2         # 16:18
        COL_NORMAL_CUP = 2 * dim + 4        # 18:32
        COL_NORMAL_POUR = 2 * dim + 4 + 2 * dim  # 32:46
        COL_EE_Y = 2 * dim + 4 + 4 * dim    # 46:48
        COL_D_ORIENT_CUP = 2 * dim + 4 + 4 * dim + 2  # 48:50
        COL_D_ORIENT_POUR = 2 * dim + 4 + 4 * dim + 4  # 50:52
        COL_ZONE_ID = 2 * dim + 4 + 4 * dim + 6  # 52:54
        COL_PAIR_TYPE = 2 * dim + 4 + 4 * dim + 8  # 54:55
        COL_OPTIONAL = 2 * dim + 4 + 4 * dim + 9  # 55:
        has_safety_distance = getattr(self.dataset, 'has_safety_distance', False)
        has_tilt_grad = getattr(self.dataset, 'has_tilt_grad', False)
        has_sample_weight = getattr(self.dataset, 'has_sample_weight', False)
        has_handover_grad = getattr(self.dataset, 'has_handover_grad', False)
        handover_grad_dim = getattr(self.dataset, 'handover_grad_dim', dim)
        has_ee_xy = getattr(self.dataset, 'has_ee_xy', False)
        has_fk_features = getattr(self.dataset, 'has_fk_features', False)
        fk_feature_dim = getattr(self.dataset, 'fk_feature_dim', 0)
        if has_fk_features and self.fk_feature_dim != fk_feature_dim:
            raise ValueError(
                f"checkpoint/config fk_feature_dim={self.fk_feature_dim} "
                f"but dataset has {fk_feature_dim}")

        optional_cursor = COL_OPTIONAL
        COL_D_OBSTACLE = optional_cursor
        COL_D_SELF = COL_D_OBSTACLE + 2
        if has_safety_distance:
            optional_cursor += 4
        COL_EE_XY = optional_cursor
        if has_ee_xy:
            optional_cursor += 4
        COL_TILT_GRAD = optional_cursor
        if has_tilt_grad:
            optional_cursor += 2 * dim
        COL_SAMPLE_WEIGHT = optional_cursor
        if has_sample_weight:
            optional_cursor += 1
        COL_HANDOVER_GRAD = optional_cursor
        if has_handover_grad:
            optional_cursor += handover_grad_dim
        COL_FK_FEATURES = optional_cursor
        if has_safety_distance:
            print(f"[safety] d_obstacle block at columns "
                  f"{COL_D_OBSTACLE}:{COL_D_OBSTACLE + 2}; "
                  f"d_self block at columns {COL_D_SELF}:{COL_D_SELF + 2}")
        if has_ee_xy:
            print(f"[routing] corner-table ee_xy block enabled at columns {COL_EE_XY}:{COL_EE_XY + 4}")
        if has_tilt_grad:
            print(f"[routing] tilt_grad block at columns {COL_TILT_GRAD}:{COL_TILT_GRAD + 2 * dim}")
        if has_sample_weight:
            print(f"[routing] sample_weight column at {COL_SAMPLE_WEIGHT}")
        if has_handover_grad:
            print(f"[routing] handover_grad block at columns "
                  f"{COL_HANDOVER_GRAD}:{COL_HANDOVER_GRAD + handover_grad_dim}")
        if has_fk_features:
            print(f"[routing] fk_features block at columns "
                  f"{COL_FK_FEATURES}:{COL_FK_FEATURES + 2 * fk_feature_dim} "
                  f"(per endpoint dim={fk_feature_dim})")
        ignore_sample_weight = os.environ.get('R58_IGNORE_SAMPLE_WEIGHT', '0') == '1'
        sample_weight_cap = (
            float(os.environ['R58_SAMPLE_WEIGHT_CAP'])
            if os.environ.get('R58_SAMPLE_WEIGHT_CAP') else None)
        if ignore_sample_weight:
            print("[routing] sample_weight ignored by R58_IGNORE_SAMPLE_WEIGHT=1")
        elif sample_weight_cap is not None:
            print(f"[routing] sample_weight capped at {sample_weight_cap:g}")

        weights = Tensor(torch.ones(len(self.dataset))).to(
            torch.device(self.Params['Device']))
        beta = 1.0
        prev_diff = 1.0
        current_diff = 1000.0
        step = -1000.0 / 4000.0
        tt = time.time()

        current_state = pickle.loads(pickle.dumps(self.network.state_dict()))
        current_optimizer = pickle.loads(pickle.dumps(self.optimizer.state_dict()))
        prev_state_queue = []
        prev_optimizer_queue = []
        self.best_val_loss = float('inf')
        n_epochs = self.Params['Training']['Number of Epochs']
        if self.cross_start_epoch > 0:
            print(f"[Curriculum] Cross-zone pairs suppressed until epoch {self.cross_start_epoch}")
        pbar = tqdm(range(1, n_epochs + 1), desc="Training (multi)", unit="ep")

        for epoch in pbar:
            if (self._viscosity_target_network is not None
                    and epoch > 1
                    and (epoch - 1) % self.viscosity_target_lag == 0):
                with torch.no_grad():
                    self._viscosity_target_network.load_state_dict(
                        self.network.state_dict(), strict=True)
                self._viscosity_target_network.eval()
                tqdm.write(
                    f"[viscosity_target] synchronized before epoch {epoch}")
            print_every = 1
            start_time = time.time()
            total_train_loss = 0
            total_diff = 0
            last_path_eik = 0.0
            last_rank = 0.0

            alpha = 1.0
            step += 1.0 / 4000 / ((int)(epoch / 4000) + 1.)
            gamma = 0.001

            prev_state_queue.append(current_state)
            prev_optimizer_queue.append(current_optimizer)
            if len(prev_state_queue) > 5:
                prev_state_queue.pop(0)
                prev_optimizer_queue.pop(0)

            current_state = pickle.loads(pickle.dumps(self.network.state_dict()))
            current_optimizer = pickle.loads(pickle.dumps(self.optimizer.state_dict()))

            base_lr = self.Params['Training'].get('Base LR', 5e-4)
            cross_lr = self.Params['Training'].get('Cross LR', None)
            if cross_lr is not None and self.cross_start_epoch > 0 and epoch >= self.cross_start_epoch:
                self.optimizer.param_groups[0]['lr'] = cross_lr
            else:
                self.optimizer.param_groups[0]['lr'] = base_lr

            # Curriculum transition: reset rollback state when cross pairs activate
            if self.cross_start_epoch > 0 and epoch == self.cross_start_epoch:
                tqdm.write(f"[Curriculum] Cross-zone pairs now active (epoch {epoch}). Resetting rollback state.")
                prev_diff = 1.0
                current_diff = 1000.0
                prev_state_queue.clear()
                prev_optimizer_queue.clear()
                prev_state_queue.append(current_state)
                prev_optimizer_queue.append(current_optimizer)
                # Optionally freeze encoders during cross phase
                if self.Params['Training'].get('Freeze Encoders On Cross', False):
                    tqdm.write("[Curriculum] Freezing encoder weights. Only pe_gate/encoder_norm/sym_op trainable.")
                    for param in self.network.cup_encoder.parameters():
                        param.requires_grad = False
                    for param in self.network.pour_encoder.parameters():
                        param.requires_grad = False
                    for param in self.network.cup_gate.parameters():
                        param.requires_grad = False
                    for param in self.network.pour_gate.parameters():
                        param.requires_grad = False
                    # Rebuild optimizer with only trainable params
                    trainable_params = [p for p in self.network.parameters() if p.requires_grad]
                    tqdm.write(f"  Trainable params: {sum(p.numel() for p in trainable_params)} "
                               f"(frozen: {sum(p.numel() for p in self.network.parameters() if not p.requires_grad)})")
                    self.optimizer = torch.optim.AdamW(trainable_params, lr=cross_lr or base_lr, weight_decay=0.5)
            else:
                prev_diff = current_diff

            rollback_enabled = not (
                bool(self.Params['Training'].get('Disable Rollback', False))
                or os.environ.get('R58_DISABLE_ROLLBACK', '0') == '1')
            while True:
                total_train_loss = 0
                total_diff = 0
                max_batches = self.Params['Training'].get('Max Batches Per Epoch', 5)
                ii = 0
                last_aux_terms = {}
                for i, wholedata in enumerate(dataloader, 0):
                    if ii >= max_batches:
                        break
                    ii += 1

                    data = wholedata[0].to(self.Params['Device'])

                    # Unpack multi-task data tensor
                    points = data[:, COL_POINTS:COL_POINTS + 2 * dim]
                    speed_cup = data[:, COL_SPEED_CUP:COL_SPEED_CUP + 2]
                    speed_pour = data[:, COL_SPEED_POUR:COL_SPEED_POUR + 2]
                    normal_cup = data[:, COL_NORMAL_CUP:COL_NORMAL_CUP + 2 * dim]
                    normal_pour = data[:, COL_NORMAL_POUR:COL_NORMAL_POUR + 2 * dim]
                    ee_y = data[:, COL_EE_Y:COL_EE_Y + 2]
                    ee_xy = data[:, COL_EE_XY:COL_EE_XY + 4] if has_ee_xy else None
                    d_obstacle = (
                        data[:, COL_D_OBSTACLE:COL_D_OBSTACLE + 2]
                        if has_safety_distance else None)
                    d_self = (
                        data[:, COL_D_SELF:COL_D_SELF + 2]
                        if has_safety_distance else None)
                    d_orient_cup = data[:, COL_D_ORIENT_CUP:COL_D_ORIENT_CUP + 2]
                    d_orient_pour = data[:, COL_D_ORIENT_POUR:COL_D_ORIENT_POUR + 2]
                    zone_id = data[:, COL_ZONE_ID:COL_ZONE_ID + 2]
                    pair_type = data[:, COL_PAIR_TYPE].long()  # (B,) int: 0=cup, 1=pour, 2=cross
                    tilt_grad = data[:, COL_TILT_GRAD:COL_TILT_GRAD + 2 * dim] if has_tilt_grad else None  # R58U
                    sample_weight = data[:, COL_SAMPLE_WEIGHT] if has_sample_weight else None
                    if sample_weight is not None:
                        if ignore_sample_weight:
                            sample_weight = None
                        elif sample_weight_cap is not None:
                            sample_weight = torch.clamp(sample_weight, max=sample_weight_cap)
                    handover_grad = (
                        data[:, COL_HANDOVER_GRAD:COL_HANDOVER_GRAD + handover_grad_dim]
                        if has_handover_grad else None
                    )
                    fk_features = (
                        data[:, COL_FK_FEATURES:COL_FK_FEATURES + 2 * fk_feature_dim]
                        if has_fk_features else None
                    )

                    # Curriculum: suppress cross pairs before cross_start_epoch
                    if self.cross_start_epoch > 0 and epoch < self.cross_start_epoch:
                        is_cross = (pair_type == 2)
                        # Reclassify cross pairs by start zone: cupping zone → intra-cup, else → intra-pour
                        start_zone = zone_id[:, 0].long()
                        pair_type = pair_type.clone()
                        pair_type[is_cross & (start_zone == 0)] = 0  # → intra-cupping
                        pair_type[is_cross & (start_zone == 1)] = 1  # → intra-pouring

                    # Apply speed transform
                    speed_transform = self.Params['Training'].get('Speed Transform', 'quartic')
                    if speed_transform == 'quartic':
                        speed_cup = speed_cup * speed_cup * (2 - speed_cup) * (2 - speed_cup)
                        speed_pour = speed_pour * speed_pour * (2 - speed_pour) * (2 - speed_pour)
                    elif speed_transform == 'quadratic':
                        speed_cup = speed_cup * speed_cup
                        speed_pour = speed_pour * speed_pour

                    speed_cup = alpha * speed_cup + 1 - alpha
                    speed_pour = alpha * speed_pour + 1 - alpha

                    loss_value, loss_n, wv, path_eik_term, rank_term = self.function.Loss(
                        points, speed_cup, speed_pour,
                        normal_cup, normal_pour,
                        ee_y, d_orient_cup, d_orient_pour, zone_id,
                        beta, gamma, epoch,
                        ee_xy=ee_xy,
                        pair_type=pair_type, path_loss_cfg=self._path_loss_cfg,
                        segment_barrier_cfg=self._segment_barrier_cfg,
                        line_preserve_cfg=self._line_preserve_cfg,
                        tilt_grad=tilt_grad,
                        sample_weight=sample_weight,
                        handover_grad=handover_grad,
                        fk_features=fk_features,
                        d_obstacle=d_obstacle,
                        d_self=d_self,
                        collocation_points=collocation_points,
                        collocation_speed_labels=collocation_speed,
                        collocation_speed_component_labels=
                            collocation_speed_components)

                    # Capture last-batch path terms for epoch-end logging
                    last_path_eik = float(path_eik_term)
                    last_rank = float(rank_term)
                    last_aux_terms = getattr(self.function, 'last_aux_terms', {})

                    # Optional optimizer-side numerical guard.  Pure-PDE
                    # experiments may use gradient clipping to keep rare
                    # second-order batches from corrupting the deployed
                    # field; it never supplies a target or inference signal.
                    # A non-finite loss/gradient is a hard run failure rather
                    # than something that may silently enter a checkpoint.
                    loss_value.backward()
                    grad_clip_raw = os.environ.get('R58_GRAD_CLIP_NORM', '0')
                    try:
                        grad_clip_norm = float(grad_clip_raw)
                    except (TypeError, ValueError):
                        raise ValueError(
                            f"R58_GRAD_CLIP_NORM must be numeric, got {grad_clip_raw!r}")
                    if not torch.isfinite(loss_value.detach()):
                        self.optimizer.zero_grad(set_to_none=True)
                        raise FloatingPointError(
                            f"non-finite loss at epoch={epoch} batch={ii}; "
                            "run is invalid")
                    trainable_params = [
                        p for p in self.network.parameters() if p.requires_grad]
                    if grad_clip_norm > 0.0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            trainable_params, grad_clip_norm)
                    else:
                        grad_sq = torch.zeros(
                            (), device=loss_value.device, dtype=loss_value.dtype)
                        for param in trainable_params:
                            if param.grad is not None:
                                grad_sq = grad_sq + param.grad.detach().square().sum()
                        grad_norm = torch.sqrt(grad_sq)
                    if not torch.isfinite(grad_norm):
                        self.optimizer.zero_grad(set_to_none=True)
                        raise FloatingPointError(
                            f"non-finite gradient at epoch={epoch} "
                            f"batch={ii}; run is invalid")
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    total_train_loss += loss_value.item()
                    total_diff += loss_n.item()

                    del points, speed_cup, speed_pour, loss_value, loss_n

                total_train_loss /= max(ii, 1)
                total_diff /= max(ii, 1)

                use_total_select = (
                    os.environ.get('R58_SELECT_TOTAL_LOSS', '0') == '1')
                current_diff = total_train_loss if use_total_select else total_diff
                diff_ratio = current_diff / prev_diff

                ratio_threshold = self.Params['Training'].get('Loss Ratio Threshold', 1.2)
                diff_ratio_ok = (
                    torch.isfinite(torch.tensor(diff_ratio)).item()
                    and diff_ratio > 0
                    and diff_ratio < ratio_threshold)
                if (not rollback_enabled) or diff_ratio_ok:
                    break
                else:
                    with torch.no_grad():
                        random_number = random.randint(0, len(prev_state_queue) - 1)
                        self.network.load_state_dict(prev_state_queue[random_number], strict=True)
                        self.optimizer.load_state_dict(prev_optimizer_queue[random_number])
                    tqdm.write(f"  Rollback ep={epoch} loss={total_diff:.4e}")

            self.total_train_loss.append(total_train_loss)
            beta = 1.0 / total_diff

            if self.Params['Training']['Use Scheduler (bool)']:
                self.scheduler.step(total_train_loss)

            t_tmp = tt
            tt = time.time()

            if self._path_loss_cfg is not None and epoch % 50 == 0:
                path_handover = last_aux_terms.get('path_handover_mono', 0.0)
                print(f"  [path] eik={last_path_eik:.4e} rank={last_rank:.4e} "
                      f"handover_mono={path_handover:.4e} "
                      f"epoch_time={tt - start_time:.1f}s")
            if (last_aux_terms
                    and (epoch % 25 == 0 or epoch == n_epochs)):
                _d = getattr(self.function, '_interior_sup_diag', None)
                _isup_diag = ("pred %.2f+-%.2f tgt %.2f+-%.2f corr %.3f" % _d) if _d else "off"
                print(
                    "  [aux] "
                    f"segment_barrier={last_aux_terms.get('segment_barrier', 0.0):.4e} "
                    f"seg_invalid={last_aux_terms.get('segment_invalid_rate', 0.0):.3f} "
                    f"line_preserve={last_aux_terms.get('line_preserve', 0.0):.4e} "
                    f"line_valid={last_aux_terms.get('line_valid_rate', 0.0):.3f} "
                    f"slowness_sup={last_aux_terms.get('slowness_sup', 0.0):.4e} "
                    f"interior_sup={last_aux_terms.get('interior_sup', 0.0):.4e} "
                    f"isup=[{_isup_diag}] "
                    f"slow_pred={last_aux_terms.get('slowness_pred_mean', 0.0):.3f} "
                    f"slow_std={last_aux_terms.get('slowness_pred_std', 0.0):.3f} "
                    f"slow_tgt={last_aux_terms.get('slowness_tgt_mean', 0.0):.3f} "
                    f"slow_corr={last_aux_terms.get('slowness_corr', 0.0):.3f} "
                    f"yobs=[{last_aux_terms.get('slowness_yobs_min', 0.0):.4f},{last_aux_terms.get('slowness_yobs_max', 0.0):.4f}] "
                    f"safety_head={last_aux_terms.get('safety_head', 0.0):.4e} "
                    f"speed={last_aux_terms.get('speed_supervision', 0.0):.4e} "
                    f"colloc_speed={last_aux_terms.get('collocation_speed_supervision', 0.0):.4e} "
                    f"colloc_mae={last_aux_terms.get('collocation_speed_mae', 0.0):.4e} "
                    f"colloc_corr={last_aux_terms.get('collocation_speed_corr', 0.0):.3f} "
                    f"colloc_slow_recall={last_aux_terms.get('collocation_speed_slow_recall', 0.0):.3f} "
                    f"component={last_aux_terms.get('collocation_speed_component_supervision', 0.0):.4e} "
                    f"component_mae={last_aux_terms.get('collocation_speed_component_mae', 0.0):.4e} "
                    f"component_corr_min={last_aux_terms.get('collocation_speed_component_corr_min', 0.0):.3f} "
                    f"direct={last_aux_terms.get('collocation_speed_uncomposed_supervision', 0.0):.4e} "
                    f"direct_mae={last_aux_terms.get('collocation_speed_uncomposed_mae', 0.0):.4e} "
                    f"direct_corr={last_aux_terms.get('collocation_speed_uncomposed_corr', 0.0):.3f} "
                    f"consensus_id={last_aux_terms.get('collocation_speed_consensus_identification', 0.0):.4e} "
                    f"consensus_mae={last_aux_terms.get('collocation_speed_consensus_alpha_mae', 0.0):.4e} "
                    f"consensus_target={last_aux_terms.get('collocation_speed_consensus_alpha_target_mean', 0.0):.3f} "
                    f"consensus_pred={last_aux_terms.get('collocation_speed_consensus_alpha_prediction_mean', 0.0):.3f} "
                    f"release_bce={last_aux_terms.get('collocation_speed_consensus_release_bce', 0.0):.4e} "
                    f"release_target={last_aux_terms.get('collocation_speed_consensus_release_target_mean', 0.0):.3f} "
                    f"release_pred={last_aux_terms.get('collocation_speed_consensus_release_prediction_mean', 0.0):.3f} "
                    f"magnitude_mse={last_aux_terms.get('collocation_speed_consensus_magnitude_mse', 0.0):.4e} "
                    f"magnitude_target={last_aux_terms.get('collocation_speed_consensus_magnitude_target_mean', 0.0):.3f} "
                    f"magnitude_pred={last_aux_terms.get('collocation_speed_consensus_magnitude_prediction_mean', 0.0):.3f} "
                    f"magnitude_neg_mse={last_aux_terms.get('collocation_speed_consensus_negative_magnitude_mse', 0.0):.4e} "
                    f"magnitude_neg_frac={last_aux_terms.get('collocation_speed_consensus_negative_fraction', 0.0):.3f} "
                    f"magnitude_neg_pred={last_aux_terms.get('collocation_speed_consensus_negative_magnitude_prediction_mean', 0.0):.3f} "
                    f"specificity_bce={last_aux_terms.get('collocation_speed_consensus_specificity_bce', 0.0):.4e} "
                    f"specificity_pos={last_aux_terms.get('collocation_speed_consensus_specificity_positive_mean', 0.0):.3f} "
                    f"specificity_neg={last_aux_terms.get('collocation_speed_consensus_specificity_negative_mean', 0.0):.3f} "
                    f"repair_id={last_aux_terms.get('collocation_speed_repair_identification', 0.0):.4e} "
                    f"repair_rho_mae={last_aux_terms.get('collocation_speed_repair_rho_mae', 0.0):.4e} "
                    f"repair_bce={last_aux_terms.get('collocation_speed_repair_release_bce', 0.0):.4e} "
                    f"repair_support={last_aux_terms.get('collocation_speed_repair_support_target_mean', 0.0):.3f} "
                    f"repair_release={last_aux_terms.get('collocation_speed_repair_release_prediction_mean', 0.0):.3f} "
                    f"repair_mag_target={last_aux_terms.get('collocation_speed_repair_magnitude_target_mean', 0.0):.3f} "
                    f"repair_mag_pred={last_aux_terms.get('collocation_speed_repair_magnitude_prediction_mean', 0.0):.3f} "
                    f"repair_beta={last_aux_terms.get('collocation_speed_repair_coordination_beta_mean', 0.0):.3f} "
                    f"repair_beta0={last_aux_terms.get('collocation_speed_repair_coordination_zero_fraction', 0.0):.3f} "
                    f"coherent_eik={last_aux_terms.get('coherent_eikonal', 0.0):.4e} "
                    f"coherent_res={last_aux_terms.get('coherent_eikonal_residual', 0.0):.4e} "
                    f"coherent_goals={last_aux_terms.get('coherent_goal_count', 0.0):.0f} "
                    f"monotone_fd={last_aux_terms.get('monotone_fd', 0.0):.4e} "
                    f"monotone_res={last_aux_terms.get('monotone_fd_residual', 0.0):.4e} "
                    f"monotone_valid={last_aux_terms.get('monotone_fd_valid_rate', 0.0):.3f} "
                    f"monotone_norm={last_aux_terms.get('monotone_fd_norm_mean', 0.0):.3f} "
                    f"trusted_max={last_aux_terms.get('trusted_anchor_maximal', 0.0):.4e} "
                    f"trusted_hard={last_aux_terms.get('trusted_anchor_hard_count', 0.0):.0f} "
                    f"trusted_direct_hard={last_aux_terms.get('trusted_anchor_direct_hard_count', 0.0):.0f} "
                    f"trusted_routes={last_aux_terms.get('trusted_anchor_route_pair_count', 0.0):.0f} "
                    f"trusted_ratio={last_aux_terms.get('trusted_anchor_ratio', 0.0):.3f} "
                    f"trusted_vp95={last_aux_terms.get('trusted_anchor_violation_p95', 0.0):.3f} "
                    f"clearance={last_aux_terms.get('clearance_supervision', 0.0):.4e} "
                    f"hopf={last_aux_terms.get('hopf_cole', 0.0):.4e} "
                    f"hopf_boundary={last_aux_terms.get('hopf_boundary_fraction', 0.0):.3f} "
                    f"hopf_state_bad={last_aux_terms.get('hopf_state_violation_rate', 0.0):.3f} "
                    f"viscosity={last_aux_terms.get('viscosity_bellman', 0.0):.4e} "
                    f"domain_valid={last_aux_terms.get('viscosity_domain_valid_rate', 0.0):.3f} "
                    f"upwind={last_aux_terms.get('viscosity_upwind_rate', 0.0):.3f} "
                    f"bellman_signed={last_aux_terms.get('viscosity_signed_residual', 0.0):.3e} "
                    f"target_above={last_aux_terms.get('viscosity_target_above_rate', 0.0):.3f} "
                    f"fermat={last_aux_terms.get('fermat_path', 0.0):.4e} "
                    f"tilt_grad_tau={last_aux_terms.get('tilt_grad_tau', 0.0):.4e} "
                    f"handover_grad={last_aux_terms.get('handover_grad', 0.0):.4e} "
                    f"cross_align={last_aux_terms.get('cross_align', 0.0):.4e} "
                    f"total={total_train_loss:.4e} diff={total_diff:.4e}"
                )

            if epoch % self.Params['Training']['Print Every * Epoch'] == 0:
                with torch.no_grad():
                    pbar.set_postfix(loss=f"{current_diff:.4e}",
                                     diff=f"{total_diff:.4e}",
                                     alpha=f"{alpha:.2f}",
                                     best=f"{self.best_val_loss:.4e}")

            if (epoch % self.Params['Training']['Save Every * Epoch'] == 0) or \
               (epoch == self.Params['Training']['Number of Epochs']) or (epoch == 1):
                self.function.plot(epoch, total_diff, alpha, self.source)
                with torch.no_grad():
                    self.save(epoch=epoch, val_loss=current_diff)

            # Track best model
            if current_diff < self.best_val_loss:
                self.best_val_loss = current_diff
                best_path = '{}/Best_Model.pt'.format(self.folder)
                with torch.no_grad():
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.network.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'B_state_dict': self.B,
                        'B_pour_state_dict': getattr(self.network, 'B_pour', None),
                        'train_loss': self.total_train_loss,
                        'val_loss': self.total_val_loss,
                        'zone_params': self._zone_params_dict(),
                    }, best_path)

    def save(self, epoch='', val_loss=''):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'B_state_dict': self.B,
            'B_pour_state_dict': getattr(self.network, 'B_pour', None),
            'train_loss': self.total_train_loss,
            'val_loss': self.total_val_loss,
            'zone_params': self._zone_params_dict(),
        }, '{}/Model_Epoch_{}_ValLoss_{:.6e}.pt'.format(
            self.folder, str(epoch).zfill(5), val_loss))

    def load(self, filepath):
        checkpoint = torch.load(
            filepath, map_location=torch.device(self.Params['Device']))
        self.B = checkpoint['B_state_dict']
        checkpoint_dim = int(self.B.shape[1])
        if checkpoint_dim != self.dim:
            print(
                f"  [load] model dimension {self.dim} -> {checkpoint_dim} "
                "from checkpoint Fourier basis")
            self.dim = checkpoint_dim

        # Restore zone params if present
        zone_params = checkpoint.get('zone_params', {})
        y_divide = zone_params.get('y_divide', self.y_divide)
        overlap = zone_params.get('overlap', self.overlap)
        route_temp = zone_params.get('route_temp', self.route_temp)
        cup_side = zone_params.get('cup_side', self.cup_side)
        zone_mode = zone_params.get('zone_mode', self.zone_mode)
        corner_zones = zone_params.get('corner_zones', self.corner_zones)
        corner_spatial_temp = zone_params.get('corner_spatial_temp', self.corner_spatial_temp)
        # R62: the routing SHAPE must follow the checkpoint, never the ambient
        # environment -- otherwise a smooth-trained model silently evaluates with
        # legacy (discontinuous) routing when the env var is absent.
        _ck_smooth = zone_params.get('smooth_routing', None)
        _ck_alpha = zone_params.get('route_alpha', None)
        _ck_ytemp = zone_params.get('route_y_temp', None)
        _ck_interior = zone_params.get('interior_routing', False)
        separate_pe_gate = zone_params.get('separate_pe_gate', False)
        separate_fourier = zone_params.get('separate_fourier', False)
        separate_encoder_norm = zone_params.get('separate_encoder_norm', False)
        nl1 = zone_params.get('nl1', self.nl1)
        weight_mode = zone_params.get('weight_mode', 'default')
        use_projector = zone_params.get('use_projector', False)
        use_align_loss = zone_params.get('use_align_loss', False)
        use_dual_sym = zone_params.get('use_dual_sym', False)
        cross_mode = zone_params.get('cross_mode', 'default')
        cup_loss_weight = zone_params.get('cup_loss_weight', 1.0)
        pour_loss_weight = zone_params.get('pour_loss_weight', 1.0)
        cross_loss_weight = zone_params.get('cross_loss_weight', 1.0)
        blend_mode = zone_params.get('blend_mode', 'feature')
        detach_routing_weights = zone_params.get('detach_routing_weights', False)
        use_task_orient_heads = bool(zone_params.get('use_task_orient_heads', False))
        use_orient_head = bool(zone_params.get('use_orient_head', False) or use_task_orient_heads)
        self.use_orient_head = use_orient_head
        self.use_task_orient_heads = use_task_orient_heads
        # R58G: restore the bake-into-tau flag + lambda so out() returns tau_total.
        add_orient_to_tau = zone_params.get('add_orient_to_tau', False)
        orient_tau_lambda = zone_params.get('orient_tau_lambda', 0.1)
        self.add_orient_to_tau = add_orient_to_tau
        self.orient_tau_lambda = orient_tau_lambda
        use_safety_head = zone_params.get('use_safety_head', False)
        add_safety_to_tau = zone_params.get('add_safety_to_tau', False)
        use_pair_safety_heads = bool(zone_params.get('use_pair_safety_heads', False))
        pair_safety_mode = zone_params.get('pair_safety_mode', 'replace')
        pair_safety_residual_scale = float(
            zone_params.get('pair_safety_residual_scale', 1.0))
        safety_tau_lambda = zone_params.get('safety_tau_lambda', 0.1)
        self.use_safety_head = use_safety_head
        self.add_safety_to_tau = add_safety_to_tau
        self.use_pair_safety_heads = use_pair_safety_heads
        self.pair_safety_mode = pair_safety_mode
        self.pair_safety_residual_scale = pair_safety_residual_scale
        self.safety_tau_lambda = safety_tau_lambda
        use_boundary_align = zone_params.get('use_boundary_align', self.use_boundary_align)
        boundary_align_margin = zone_params.get('boundary_align_margin', self.boundary_align_margin)
        boundary_align_weight = zone_params.get('boundary_align_weight', self.boundary_align_weight)
        fk_feature_dim = int(zone_params.get('fk_feature_dim', 0))
        use_task_adapters = bool(zone_params.get('use_task_adapters', False))
        handover_k = int(zone_params.get('handover_k', 0))
        handover_temp = float(zone_params.get('handover_temp', 0.03))
        value_arch = zone_params.get('value_arch', 'metric')
        value_condition = zone_params.get('value_condition', 'task')
        finsler_quad_points = int(zone_params.get('finsler_quad_points', 2))
        finsler_task_heads = bool(zone_params.get('finsler_task_heads', False))
        use_cross_slowness_residual = bool(
            zone_params.get('use_cross_slowness_residual', False))
        cross_residual_scale = float(zone_params.get('cross_residual_scale', 0.5))
        use_local_slowness_residual = bool(
            zone_params.get('use_local_slowness_residual', False))
        local_residual_scale = float(zone_params.get('local_residual_scale', 0.2))
        local_residual_gate = zone_params.get('local_residual_gate', 'all')
        endpoint_task_siamese = bool(
            zone_params.get('endpoint_task_siamese', False))
        endpoint_dual_value_heads = bool(
            zone_params.get('endpoint_dual_value_heads', False))
        task_embedding_scale = float(
            zone_params.get('task_embedding_scale', 0.25))
        fourier_basis = zone_params.get('fourier_basis', 'random')
        global_residual_scale = float(
            zone_params.get('global_residual_scale', 2.0))
        use_speed_head = bool(zone_params.get('use_speed_head', False))
        separate_speed_encoder = bool(
            zone_params.get('separate_speed_encoder', False))
        speed_fourier_basis = zone_params.get(
            'speed_fourier_basis', 'multiscale_random')
        speed_B_scale = float(zone_params.get('speed_B_scale', 0.2))
        speed_B_freqs = int(zone_params.get('speed_B_freqs', 128))
        speed_residual_layers = int(
            zone_params.get('speed_residual_layers', 0))
        lifted_phase = bool(zone_params.get('lifted_phase', False))
        lifted_phase_physical_dim = int(
            zone_params.get(
                'lifted_phase_physical_dim', self.dim - 1))
        lifted_phase_scale = float(
            zone_params.get('lifted_phase_scale', 0.1))
        lifted_phase_width = int(
            zone_params.get('lifted_phase_width', 128))
        lifted_phase_logit_scale = float(
            zone_params.get('lifted_phase_logit_scale', 8.0))
        lifted_phase_causal = bool(
            zone_params.get('lifted_phase_causal', False))
        lifted_phase_hazard_floor_scale = float(
            zone_params.get('lifted_phase_hazard_floor_scale', 0.0))
        cross_hazard_floor_scale = float(
            zone_params.get('cross_hazard_floor_scale', 0.0))
        speed_correction_width = int(
            zone_params.get('speed_correction_width', 0))
        speed_correction_scale = float(
            zone_params.get('speed_correction_scale', 1.0))
        speed_correction_mode = zone_params.get(
            'speed_correction_mode', 'signed_logit')
        use_clearance_head = bool(
            zone_params.get('use_clearance_head', False))
        clearance_scale = float(
            zone_params.get('clearance_scale', 0.05))
        clearance_boundary = float(
            zone_params.get('clearance_boundary', 0.005))
        clearance_speed_floor = float(
            zone_params.get('clearance_speed_floor', 0.001))
        clearance_residual_layers = int(
            zone_params.get('clearance_residual_layers', 0))
        clearance_fourier_skip = bool(
            zone_params.get('clearance_fourier_skip', False))
        couple_speed_latent_to_value = bool(
            zone_params.get('couple_speed_latent_to_value', False))
        partial_shared_embedding_dim = int(
            zone_params.get('partial_shared_embedding_dim', 0))
        pair_slowness_attention_dim = int(
            zone_params.get('pair_slowness_attention_dim', 0))
        speed_constraint_heads = int(
            zone_params.get('speed_constraint_heads', 0))
        speed_constraint_softmin_temperature = float(
            zone_params.get('speed_constraint_softmin_temperature', 0.04))
        speed_constraint_residual_scale = float(
            zone_params.get('speed_constraint_residual_scale', 1.25))
        speed_constraint_fourier_skip = bool(
            zone_params.get('speed_constraint_fourier_skip', False))
        speed_constraint_consensus_width = int(
            zone_params.get('speed_constraint_consensus_width', 0))
        speed_constraint_consensus_mode = str(
            zone_params.get('speed_constraint_consensus_mode', 'single'))
        speed_constraint_repair_width = int(
            zone_params.get('speed_constraint_repair_width', 0))
        global_residual_positive = bool(
            zone_params.get('global_residual_positive', False))
        global_excess_centered_gate = bool(
            zone_params.get('global_excess_centered_gate', False))
        global_excess_centered_signed_gate = bool(
            zone_params.get('global_excess_centered_signed_gate', False))
        chord_quad_points = int(zone_params.get('chord_quad_points', 3))
        chord_quadrature = zone_params.get('chord_quadrature', 'legendre')
        chord_hazard_floor_scale = float(
            zone_params.get('chord_hazard_floor_scale', 1.0))
        use_raw_coordinate_skip = bool(
            zone_params.get('use_raw_coordinate_skip', False))
        speed_floor = float(zone_params.get('speed_floor', 0.1))
        endpoint_barrier_speed = float(
            zone_params.get('endpoint_barrier_speed', 0.1))
        endpoint_barrier_scale = float(
            zone_params.get('endpoint_barrier_scale', 1.0))
        endpoint_barrier_radius = float(
            zone_params.get('endpoint_barrier_radius', 0.05))
        endpoint_barrier_power = float(
            zone_params.get('endpoint_barrier_power', 1.0))
        endpoint_barrier_physical_decode = bool(
            zone_params.get('endpoint_barrier_physical_decode', False))
        endpoint_barrier_contrast = bool(
            zone_params.get('endpoint_barrier_contrast', False))
        global_pair_residual_layers = int(
            zone_params.get('global_pair_residual_layers', 0))
        global_pair_residual_activation = zone_params.get(
            'global_pair_residual_activation', 'silu')
        use_viscosity_excess_head = bool(
            zone_params.get('use_viscosity_excess_head', False))
        viscosity_excess_gate = zone_params.get(
            'viscosity_excess_gate', 'all')
        viscosity_residual_mode = zone_params.get(
            'viscosity_residual_mode', 'positive_time')
        viscosity_stencil_radius = float(
            zone_params.get('viscosity_stencil_radius', 0.0))
        viscosity_gate_radius = float(
            zone_params.get('viscosity_gate_radius', 0.0))
        viscosity_speed_latent_context = bool(
            zone_params.get('viscosity_speed_latent_context', False))
        viscosity_chord_speed_latent_context = bool(
            zone_params.get(
                'viscosity_chord_speed_latent_context', False))
        viscosity_chord_latent_width = int(
            zone_params.get('viscosity_chord_latent_width', 32))
        viscosity_directed_context = bool(
            zone_params.get('viscosity_directed_context', False))
        viscosity_directed_odd = bool(
            zone_params.get('viscosity_directed_odd', False))
        viscosity_directed_odd_scale = float(
            zone_params.get('viscosity_directed_odd_scale', 1.0))
        viscosity_handover_phase_context = bool(zone_params.get(
            'viscosity_handover_phase_context', False))
        handover_phase_embedding_scale = float(zone_params.get(
            'handover_phase_embedding_scale', 0.0))
        viscosity_activation = zone_params.get(
            'viscosity_activation', 'silu')
        viscosity_unbounded_logit = bool(
            zone_params.get('viscosity_unbounded_logit', False))
        viscosity_minplus_modes = int(
            zone_params.get('viscosity_minplus_modes', 0))
        viscosity_minplus_temperature = float(
            zone_params.get('viscosity_minplus_temperature', 0.25))
        viscosity_minplus_scale = float(
            zone_params.get('viscosity_minplus_scale', 1.0))
        viscosity_minplus_mode = zone_params.get(
            'viscosity_minplus_mode', 'correction')
        use_viscosity_gain_head = bool(
            zone_params.get('use_viscosity_gain_head', False))
        viscosity_gain_log_scale = float(
            zone_params.get(
                'viscosity_gain_log_scale', 0.6931471805599453))
        use_causal_handover_head = bool(
            zone_params.get('use_causal_handover_head', False))
        causal_handover_gate = zone_params.get(
            'causal_handover_gate', 'cross_strength')
        causal_handover_logit_scale = float(
            zone_params.get('causal_handover_logit_scale', 3.0))
        causal_handover_strength = float(
            zone_params.get('causal_handover_strength', 1.0))
        causal_handover_phase_context = bool(
            zone_params.get('causal_handover_phase_context', False))
        viscosity_logit_scale = float(zone_params.get(
            'viscosity_logit_scale', global_residual_scale))
        if viscosity_logit_scale <= 0.0:
            raise ValueError("viscosity_logit_scale must be positive")
        self.viscosity_logit_scale = viscosity_logit_scale
        fermat_curve_modes = int(zone_params.get('fermat_curve_modes', 3))
        fermat_curve_envelope_detach = bool(
            zone_params.get('fermat_curve_envelope_detach', False))
        fermat_curve_quad_points = int(
            zone_params.get('fermat_curve_quad_points', 15))
        fermat_curve_residual_layers = int(
            zone_params.get('fermat_curve_residual_layers', 6))
        fermat_curve_speed_floor = float(
            zone_params.get('fermat_curve_speed_floor', 0.005))
        fermat_curve_value_weight = float(
            zone_params.get('fermat_curve_value_weight', 1.0))
        fermat_curve_gate = zone_params.get('fermat_curve_gate', 'all')
        eikonal_form = zone_params.get('eikonal_form', 'legacy_sqrt')
        eikonal_balance_power = float(
            zone_params.get('eikonal_balance_power', 0.0))
        eikonal_min_speed = float(
            zone_params.get('eikonal_min_speed', 0.0))
        freeze_base_for_cross_residual = bool(
            zone_params.get('freeze_base_for_cross_residual', False))
        train_local_residual_only = bool(
            zone_params.get('train_local_residual_only', False))

        # Restore flags so function object picks them up
        self.use_align_loss = use_align_loss
        self.cross_mode = cross_mode
        self.cup_loss_weight = cup_loss_weight
        self.pour_loss_weight = pour_loss_weight
        self.cross_loss_weight = cross_loss_weight
        self.blend_mode = blend_mode
        self.use_boundary_align = use_boundary_align
        self.boundary_align_margin = boundary_align_margin
        self.boundary_align_weight = boundary_align_weight
        self.zone_mode = zone_mode
        self.corner_zones = corner_zones
        self.corner_spatial_temp = corner_spatial_temp
        self.fk_feature_dim = fk_feature_dim
        self.use_task_adapters = use_task_adapters
        self.handover_k = handover_k
        self.handover_temp = handover_temp
        self.value_arch = value_arch
        self.value_condition = value_condition
        self.finsler_quad_points = finsler_quad_points
        self.finsler_task_heads = finsler_task_heads
        self.use_cross_slowness_residual = use_cross_slowness_residual
        self.cross_residual_scale = cross_residual_scale
        self.use_local_slowness_residual = use_local_slowness_residual
        self.local_residual_scale = local_residual_scale
        self.local_residual_gate = local_residual_gate
        self.endpoint_task_siamese = endpoint_task_siamese
        self.endpoint_dual_value_heads = endpoint_dual_value_heads
        self.task_embedding_scale = task_embedding_scale
        self.fourier_basis = fourier_basis
        self.global_residual_scale = global_residual_scale
        self.use_speed_head = use_speed_head
        self.separate_speed_encoder = separate_speed_encoder
        self.speed_fourier_basis = speed_fourier_basis
        self.speed_B_scale = speed_B_scale
        self.speed_B_freqs = speed_B_freqs
        self.speed_residual_layers = speed_residual_layers
        self.lifted_phase = lifted_phase
        self.lifted_phase_physical_dim = (
            lifted_phase_physical_dim)
        self.lifted_phase_scale = lifted_phase_scale
        self.lifted_phase_width = lifted_phase_width
        self.lifted_phase_logit_scale = (
            lifted_phase_logit_scale)
        self.lifted_phase_causal = lifted_phase_causal
        self.lifted_phase_hazard_floor_scale = (
            lifted_phase_hazard_floor_scale)
        self.cross_hazard_floor_scale = cross_hazard_floor_scale
        self.speed_correction_width = speed_correction_width
        self.speed_correction_scale = speed_correction_scale
        self.speed_correction_mode = speed_correction_mode
        self.use_clearance_head = use_clearance_head
        self.clearance_scale = clearance_scale
        self.clearance_boundary = clearance_boundary
        self.clearance_speed_floor = clearance_speed_floor
        self.clearance_residual_layers = clearance_residual_layers
        self.clearance_fourier_skip = clearance_fourier_skip
        self.couple_speed_latent_to_value = couple_speed_latent_to_value
        self.partial_shared_embedding_dim = partial_shared_embedding_dim
        self.pair_slowness_attention_dim = pair_slowness_attention_dim
        self.speed_constraint_heads = speed_constraint_heads
        self.speed_constraint_softmin_temperature = (
            speed_constraint_softmin_temperature)
        self.speed_constraint_residual_scale = (
            speed_constraint_residual_scale)
        self.speed_constraint_fourier_skip = (
            speed_constraint_fourier_skip)
        self.speed_constraint_consensus_width = (
            speed_constraint_consensus_width)
        self.speed_constraint_consensus_mode = (
            speed_constraint_consensus_mode)
        self.speed_constraint_repair_width = (
            speed_constraint_repair_width)
        self.global_residual_positive = global_residual_positive
        self.global_excess_centered_gate = global_excess_centered_gate
        self.global_excess_centered_signed_gate = (
            global_excess_centered_signed_gate)
        self.chord_quad_points = chord_quad_points
        self.chord_quadrature = chord_quadrature
        self.chord_hazard_floor_scale = chord_hazard_floor_scale
        self.use_raw_coordinate_skip = use_raw_coordinate_skip
        self.speed_floor = speed_floor
        self.endpoint_barrier_speed = endpoint_barrier_speed
        self.endpoint_barrier_scale = endpoint_barrier_scale
        self.endpoint_barrier_radius = endpoint_barrier_radius
        self.endpoint_barrier_power = endpoint_barrier_power
        self.endpoint_barrier_physical_decode = (
            endpoint_barrier_physical_decode)
        self.global_pair_residual_layers = global_pair_residual_layers
        self.global_pair_residual_activation = (
            global_pair_residual_activation)
        self.use_viscosity_excess_head = use_viscosity_excess_head
        self.viscosity_excess_gate = viscosity_excess_gate
        self.viscosity_residual_mode = viscosity_residual_mode
        self.viscosity_stencil_radius = viscosity_stencil_radius
        self.viscosity_gate_radius = viscosity_gate_radius
        self.viscosity_speed_latent_context = (
            viscosity_speed_latent_context)
        self.viscosity_chord_speed_latent_context = (
            viscosity_chord_speed_latent_context)
        self.viscosity_chord_latent_width = (
            viscosity_chord_latent_width)
        self.viscosity_directed_context = (
            viscosity_directed_context)
        self.viscosity_directed_odd = viscosity_directed_odd
        self.viscosity_directed_odd_scale = viscosity_directed_odd_scale
        self.viscosity_handover_phase_context = (
            viscosity_handover_phase_context)
        self.handover_phase_embedding_scale = (
            handover_phase_embedding_scale)
        self.viscosity_activation = viscosity_activation
        self.viscosity_unbounded_logit = viscosity_unbounded_logit
        self.viscosity_minplus_modes = viscosity_minplus_modes
        self.viscosity_minplus_temperature = (
            viscosity_minplus_temperature)
        self.viscosity_minplus_scale = viscosity_minplus_scale
        self.viscosity_minplus_mode = viscosity_minplus_mode
        self.use_viscosity_gain_head = use_viscosity_gain_head
        self.viscosity_gain_log_scale = viscosity_gain_log_scale
        self.use_causal_handover_head = use_causal_handover_head
        self.causal_handover_gate = causal_handover_gate
        self.causal_handover_logit_scale = causal_handover_logit_scale
        self.causal_handover_strength = causal_handover_strength
        self.causal_handover_phase_context = (
            causal_handover_phase_context)
        self.fermat_curve_modes = fermat_curve_modes
        self.fermat_curve_envelope_detach = fermat_curve_envelope_detach
        self.fermat_curve_quad_points = fermat_curve_quad_points
        self.fermat_curve_residual_layers = fermat_curve_residual_layers
        self.fermat_curve_speed_floor = fermat_curve_speed_floor
        self.fermat_curve_value_weight = fermat_curve_value_weight
        self.fermat_curve_gate = fermat_curve_gate
        self.eikonal_form = eikonal_form
        self.eikonal_balance_power = eikonal_balance_power
        self.eikonal_min_speed = eikonal_min_speed
        self.freeze_base_for_cross_residual = freeze_base_for_cross_residual
        self.train_local_residual_only = train_local_residual_only

        h_size = 2 * self.B.shape[0]
        speed_B = None
        if separate_speed_encoder:
            stored_speed_B = checkpoint['model_state_dict'].get('B_speed')
            if stored_speed_B is None:
                raise ValueError(
                    "checkpoint enables separate_speed_encoder but has no "
                    "B_speed buffer")
            speed_B = stored_speed_B.T.contiguous()
            speed_B_freqs = int(speed_B.shape[0])
            self.speed_B_freqs = speed_B_freqs
        self.network = model_network.MultiNN(
            self.Params['Device'], self.dim, self.B, h_size=h_size, nl1=nl1,
            y_divide=y_divide, overlap=overlap, route_temp=route_temp,
            cup_side=cup_side,
            separate_pe_gate=separate_pe_gate,
            separate_fourier=separate_fourier,
            separate_encoder_norm=separate_encoder_norm,
            weight_mode=weight_mode,
            use_projector=use_projector,
            use_dual_sym=use_dual_sym,
            blend_mode=blend_mode,
            detach_routing_weights=detach_routing_weights,
            use_orient_head=use_orient_head,
            add_orient_to_tau=add_orient_to_tau,
            orient_tau_lambda=orient_tau_lambda,
            use_task_orient_heads=use_task_orient_heads,
            use_safety_head=use_safety_head,
            add_safety_to_tau=add_safety_to_tau,
            use_pair_safety_heads=use_pair_safety_heads,
            pair_safety_mode=pair_safety_mode,
            pair_safety_residual_scale=pair_safety_residual_scale,
            safety_tau_lambda=safety_tau_lambda,
            zone_mode=zone_mode,
            corner_zones=corner_zones,
            corner_spatial_temp=corner_spatial_temp,
            fk_feature_dim=fk_feature_dim,
            use_task_adapters=use_task_adapters,
            handover_k=handover_k,
            handover_temp=handover_temp,
            value_arch=value_arch,
            value_condition=value_condition,
            finsler_quad_points=finsler_quad_points,
            finsler_task_heads=finsler_task_heads,
            use_cross_slowness_residual=use_cross_slowness_residual,
            cross_residual_scale=cross_residual_scale,
            use_local_slowness_residual=use_local_slowness_residual,
            local_residual_scale=local_residual_scale,
            local_residual_gate=local_residual_gate,
            endpoint_task_siamese=endpoint_task_siamese,
            endpoint_dual_value_heads=endpoint_dual_value_heads,
            task_embedding_scale=task_embedding_scale,
            global_residual_scale=global_residual_scale,
            use_speed_head=use_speed_head,
            separate_speed_encoder=separate_speed_encoder,
            speed_B=speed_B,
            speed_residual_layers=speed_residual_layers,
            lifted_phase=lifted_phase,
            lifted_phase_physical_dim=
                lifted_phase_physical_dim,
            lifted_phase_scale=lifted_phase_scale,
            lifted_phase_width=lifted_phase_width,
            lifted_phase_logit_scale=
                lifted_phase_logit_scale,
            lifted_phase_causal=lifted_phase_causal,
            lifted_phase_hazard_floor_scale=
                lifted_phase_hazard_floor_scale,
            cross_hazard_floor_scale=cross_hazard_floor_scale,
            speed_correction_width=speed_correction_width,
            speed_correction_scale=speed_correction_scale,
            speed_correction_mode=speed_correction_mode,
            use_clearance_head=use_clearance_head,
            clearance_scale=clearance_scale,
            clearance_boundary=clearance_boundary,
            clearance_speed_floor=clearance_speed_floor,
            clearance_residual_layers=clearance_residual_layers,
            clearance_fourier_skip=clearance_fourier_skip,
            couple_speed_latent_to_value=couple_speed_latent_to_value,
            partial_shared_embedding_dim=partial_shared_embedding_dim,
            pair_slowness_attention_dim=pair_slowness_attention_dim,
            speed_constraint_heads=speed_constraint_heads,
            speed_constraint_softmin_temperature=
                speed_constraint_softmin_temperature,
            speed_constraint_residual_scale=
                speed_constraint_residual_scale,
            speed_constraint_fourier_skip=
                speed_constraint_fourier_skip,
            speed_constraint_consensus_width=
                speed_constraint_consensus_width,
            speed_constraint_consensus_mode=
                speed_constraint_consensus_mode,
            speed_constraint_repair_width=
                speed_constraint_repair_width,
            global_residual_positive=global_residual_positive,
            global_excess_centered_gate=global_excess_centered_gate,
            global_excess_centered_signed_gate=(
                global_excess_centered_signed_gate),
            chord_quad_points=chord_quad_points,
            chord_quadrature=chord_quadrature,
            chord_hazard_floor_scale=chord_hazard_floor_scale,
            use_raw_coordinate_skip=use_raw_coordinate_skip,
            speed_floor=speed_floor,
            endpoint_barrier_speed=endpoint_barrier_speed,
            endpoint_barrier_scale=endpoint_barrier_scale,
            endpoint_barrier_radius=endpoint_barrier_radius,
            endpoint_barrier_power=endpoint_barrier_power,
            endpoint_barrier_physical_decode=
                endpoint_barrier_physical_decode,
            endpoint_barrier_contrast=endpoint_barrier_contrast,
            global_pair_residual_layers=global_pair_residual_layers,
            global_pair_residual_activation=
                global_pair_residual_activation,
            use_viscosity_excess_head=use_viscosity_excess_head,
            viscosity_excess_gate=viscosity_excess_gate,
            viscosity_residual_mode=viscosity_residual_mode,
            viscosity_stencil_radius=viscosity_stencil_radius,
            viscosity_gate_radius=viscosity_gate_radius,
            viscosity_speed_latent_context=
                viscosity_speed_latent_context,
            viscosity_chord_speed_latent_context=
                viscosity_chord_speed_latent_context,
            viscosity_chord_latent_width=
                viscosity_chord_latent_width,
            viscosity_directed_context=
                viscosity_directed_context,
            viscosity_directed_odd=viscosity_directed_odd,
            viscosity_directed_odd_scale=viscosity_directed_odd_scale,
            viscosity_handover_phase_context=
                viscosity_handover_phase_context,
            handover_phase_embedding_scale=
                handover_phase_embedding_scale,
            viscosity_activation=viscosity_activation,
            viscosity_unbounded_logit=viscosity_unbounded_logit,
            viscosity_minplus_modes=viscosity_minplus_modes,
            viscosity_minplus_temperature=viscosity_minplus_temperature,
            viscosity_minplus_scale=viscosity_minplus_scale,
            viscosity_minplus_mode=viscosity_minplus_mode,
            use_viscosity_gain_head=use_viscosity_gain_head,
            viscosity_gain_log_scale=viscosity_gain_log_scale,
            use_causal_handover_head=use_causal_handover_head,
            causal_handover_gate=causal_handover_gate,
            causal_handover_logit_scale=causal_handover_logit_scale,
            causal_handover_strength=causal_handover_strength,
            causal_handover_phase_context=
                causal_handover_phase_context,
            fermat_curve_modes=fermat_curve_modes,
            fermat_curve_envelope_detach=fermat_curve_envelope_detach,
            fermat_curve_quad_points=fermat_curve_quad_points,
            fermat_curve_residual_layers=fermat_curve_residual_layers,
            fermat_curve_speed_floor=fermat_curve_speed_floor,
            fermat_curve_value_weight=fermat_curve_value_weight,
            fermat_curve_gate=fermat_curve_gate)

        # R62: apply the checkpoint's routing shape before the weights are used, so a
        # smooth-trained model never silently falls back to legacy (discontinuous)
        # routing just because the ambient environment lacks R62_SMOOTH_ROUTING.
        if _ck_smooth is not None:
            self.network.smooth_routing = bool(_ck_smooth)
        if _ck_alpha is not None:
            self.network.route_alpha = float(_ck_alpha)
        if _ck_ytemp is not None:
            self.network.route_y_temp = float(_ck_ytemp)
        # R63: record that this checkpoint was TRAINED with FK-recomputed interior
        # cond.  The caller must then attach a robot model via
        # configure_interior_routing_from_dataset, or inference silently reverts to
        # the interpolated interior the model was never trained against.
        self.network.interior_routing_requested = bool(_ck_interior)
        self.network.load_state_dict(checkpoint['model_state_dict'], strict=True)
        # This is a fixed scalar architecture constant rather than a tensor;
        # restore it after constructing the backward-compatible network.
        self.network.viscosity_logit_scale = viscosity_logit_scale
        # Restore B_pour if separate_fourier was used
        if separate_fourier and 'B_pour_state_dict' in checkpoint and checkpoint['B_pour_state_dict'] is not None:
            self.network.B_pour = checkpoint['B_pour_state_dict'].to(self.Params['Device'])
        self.network.to(torch.device(self.Params['Device']))
        self.network.float()
        self.network.eval()

        self.function = model_function.MultiFunction(
            self.folder, self.Params['Device'], self.network, self.dim,
            use_temporal_loss=self.use_temporal_loss,
            use_eikonal_loss=self.use_eikonal_loss,
            use_normal_loss=self.use_normal_loss,
            eikonal_form=self.eikonal_form,
            eikonal_balance_power=self.eikonal_balance_power,
            eikonal_min_speed=self.eikonal_min_speed,
            use_align_loss=self.use_align_loss,
            cross_mode=self.cross_mode,
            use_boundary_align=self.use_boundary_align,
            boundary_align_margin=self.boundary_align_margin,
            boundary_align_weight=self.boundary_align_weight,
            and_overlap=self.and_overlap,
            and_blend=self.and_blend,
            cup_loss_weight=self.cup_loss_weight,
            pour_loss_weight=self.pour_loss_weight,
            cross_loss_weight=self.cross_loss_weight,
        )


# --- CLI entry point ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-task NTField training for G1")
    parser.add_argument("--data_dir", default="./datasets/arm/G1_mt_v2")
    parser.add_argument("--exp_name", default="G1_multi")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0,
                        help="Reproducible model, data-order, and CUDA seed")
    parser.add_argument("--init_checkpoint", type=str, default=None,
                        help="Optional checkpoint used to initialize model weights before training")
    parser.add_argument("--dim", type=int, default=7,
                        help="Joint-space dimension (G1=7, UR5e=6)")
    parser.add_argument("--max_batches", type=int, default=5,
                        help="Max batches per epoch (default 5)")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--weight_decay", type=float, default=None,
                        help="Override the AdamW weight_decay (default: 0.5 unless "
                             "--freeze_base_for_cross_residual, which uses 0.0). The "
                             "0.5 default is tuned for from-scratch full-network "
                             "training over thousands of epochs; it is too aggressive "
                             "for a short unfrozen fine-tune from a warm-started "
                             "checkpoint, where it erodes the pretrained weights "
                             "faster than the eikonal loss can compensate.")
    parser.add_argument("--loss_ratio", type=float, default=1.5,
                        help="Loss ratio threshold for rollback")
    parser.add_argument("--no_rollback", action="store_true",
                        help="Disable legacy stochastic rollback during training")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Base learning rate")
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--B_scale", type=float, default=0.2,
                        help="Fourier basis magnitude")
    parser.add_argument("--B_freqs", type=int, default=128,
                        help="Number of Fourier bands, hidden size = 2*B_freqs")
    parser.add_argument("--fk_feature_dim", type=int, default=None,
                        help="Per-endpoint FK feature dimension. Default auto-detects "
                             "data_dir/fk_features.npy when present.")
    parser.add_argument("--nl1", type=int, default=2,
                        help="Number of gated residual blocks")
    parser.add_argument("--no_normal_loss", action="store_true",
                        help="Disable normal loss (recommended)")
    parser.add_argument("--no_temporal_loss", action="store_true",
                        help="Disable temporal loss")
    parser.add_argument("--no_eikonal_loss", action="store_true",
                        help="Disable eikonal loss")
    parser.add_argument(
        "--eikonal_form", choices=["legacy_sqrt", "standard"],
        default="legacy_sqrt",
        help="PDE residual form. standard enforces speed*|grad T|=1 directly.")
    parser.add_argument(
        "--eikonal_balance_power", type=float, default=0.0,
        help="Reweight Eikonal samples by speed**(-power), normalized per batch.")
    parser.add_argument(
        "--eikonal_min_speed", type=float, default=0.0,
        help="Apply the Eikonal residual only where target speed is at least "
             "this value. Lower-speed obstacle interiors still affect chord "
             "and Bellman costs. 0 preserves the historical all-domain loss.")
    parser.add_argument("--speed_transform", type=str, default="quartic",
                        choices=["quartic", "quadratic", "linear"],
                        help="Speed nonlinearity")
    parser.add_argument("--tt_weight", type=float, default=0.5,
                        help="Travel time weighting in loss")
    # Multi-task specific args
    parser.add_argument("--y_divide", type=float, default=-0.2,
                        help="Y-coordinate zone boundary (default -0.2)")
    parser.add_argument("--overlap", type=float, default=0.05,
                        help="Half-width of overlap zone (default 0.05)")
    parser.add_argument("--route_temp", type=float, default=0.01,
                        help="Softmax temperature for constraint-aware routing (default 0.01)")
    parser.add_argument("--cup_side", type=int, choices=[-1, 1], default=-1,
                        help="Y side occupied by cupping: -1=lower-y/G1, 1=higher-y/UR5e")
    parser.add_argument("--zone_mode", type=str, default="auto",
                        choices=["auto", "y", "corner_table"],
                        help="Routing geometry. auto reads data_dir metadata; y keeps legacy "
                             "ee_y split; corner_table uses scene/data rectangular XY zones.")
    parser.add_argument("--corner_spatial_temp", type=float, default=0.05,
                        help="Softmax spatial temperature for corner-table fallback routing "
                             "outside explicit task boxes.")
    parser.add_argument("--separate_pe_gate", action="store_true",
                        help="Give each subnet its own pe_gate (removes shared-modulation bottleneck)")
    parser.add_argument("--separate_fourier", action="store_true",
                        help="Give each subnet its own Fourier B matrix (independent frequency basis)")
    parser.add_argument("--separate_encoder_norm", action="store_true",
                        help="Give each subnet its own InstanceNorm (independent feature normalization)")
    parser.add_argument("--separate_all", action="store_true",
                        help="Shorthand for --separate_pe_gate --separate_fourier --separate_encoder_norm")
    # Architectural feature flags
    parser.add_argument("--use_projector", action="store_true",
                        help="Apply shared linear projector to encoder outputs before blending")
    parser.add_argument("--use_task_adapters", action="store_true",
                        help="Use one shared geometric trunk with small cup/pour residual adapters")
    parser.add_argument("--handover_k", type=int, default=0,
                        help="Number of learned latent handover prototypes. 0 disables handover routing.")
    parser.add_argument("--handover_temp", type=float, default=0.03,
                        help="Soft-min temperature for latent handover prototypes")
    parser.add_argument("--value_arch", type=str, default="metric",
                        choices=["metric", "asym_goal", "local_finsler",
                                 "global_residual",
                                 "global_residual_symmetric",
                                 "bounded_chord_symmetric",
                                 "bounded_chord_fermat_curve_symmetric",
                                 "bounded_chord_endpoint_fermat_curve_symmetric",
                                 "bounded_chord_endpoint_speed_fermat_curve_symmetric",
                                 "bounded_chord_log_symmetric",
                                 "chord_integral_symmetric",
                                 "bounded_chord_hazard_symmetric",
                                 "bounded_chord_hazard_floor_symmetric",
                                 "bounded_chord_eikonal_symmetric",
                                 "bounded_chord_eikonal_refine_symmetric",
                                 "bounded_chord_endpoint_eikonal_symmetric",
                                 "factored_source_eikonal_symmetric",
                                 "bounded_chord_local_endpoint_eikonal_symmetric",
                                 "bounded_chord_speed_barrier_symmetric",
                                 "bounded_chord_local_speed_barrier_symmetric",
                                 "bounded_chord_global_excess_symmetric",
                                 "bounded_chord_smooth_excess_symmetric",
                                 "bounded_chord_separate_smooth_excess_symmetric",
                                 "bounded_chord_separate_global_excess_symmetric",
                                 "bounded_chord_neural_viscosity_symmetric",
                                 "bounded_chord_signed_clearance_symmetric",
                                 "bounded_chord_clearance_eikonal_symmetric",
                                 "bounded_chord_clearance_state_symmetric",
                                 "endpoint_eikonal_residual_symmetric",
                                 "local_endpoint_eikonal_residual_symmetric",
                                 "bounded_chord_profile_symmetric",
                                 "fermat_waypoint_symmetric",
                                 "fermat_curve_symmetric",
                                 "fermat_curve_equivariant"],
                        help="Travel-time architecture: metric uses legacy sym_op; "
                             "asym_goal uses direct ordered V(q, goal, condition); "
                             "local_finsler integrates a local directional slowness field; "
                             "chord_integral_symmetric integrates scalar slowness along "
                             "the endpoint chord without a learned value head; "
                             "bounded_chord_log_symmetric interpolates the physical "
                             "distance/chord bounds in log time for large slowness ranges; "
                             "bounded_chord_fermat_curve_symmetric keeps the inherited "
                             "bounded-chord value and continuously replaces its straight "
                             "upper bound with one scalar-speed-trained Fermat curve; "
                             "bounded_chord_hazard_symmetric suppresses learned detours "
                             "to second order on free chords; "
                             "bounded_chord_hazard_floor_symmetric lower-bounds the "
                             "chord penalty using harmonic-mean speed; "
                             "bounded_chord_eikonal_symmetric enforces the local "
                             "T~distance/F limit while learning global attenuation; "
                             "bounded_chord_eikonal_refine_symmetric preserves a "
                             "warm-started global chord fraction and adds an exact "
                             "local Eikonal envelope; "
                             "bounded_chord_endpoint_eikonal_symmetric also preserves "
                             "the near-zero-speed endpoint boundary under attenuation; "
                             "factored_source_eikonal_symmetric uses differentiable "
                             "endpoint-speed source factorization with exact local slope; "
                             "bounded_chord_local_endpoint_eikonal_symmetric caps that "
                             "boundary extension at one local Eikonal step; "
                             "bounded_chord_speed_barrier_symmetric adds one learned "
                             "speed-gradient gate to an otherwise unchanged chord value; "
                             "bounded_chord_local_speed_barrier_symmetric caps that "
                             "endpoint term at one local Eikonal step, removing the "
                             "spurious long-range radial gradient; "
                             "bounded_chord_global_excess_symmetric adds a chord-hazard "
                             "gated O(distance^2) viscosity correction while preserving "
                             "the base model's local Eikonal limit; "
                             "bounded_chord_smooth_excess_symmetric removes that jagged "
                             "straight-chord gate and learns a smooth non-negative "
                             "O(distance^2) viscosity correction; "
                             "bounded_chord_separate_smooth_excess_symmetric keeps the "
                             "inherited endpoint/chord field frozen and learns the "
                             "same correction with an independent symmetric scalar MLP; "
                             "bounded_chord_separate_global_excess_symmetric gates that "
                             "independent correction by the same scalar chord hazard; "
                             "bounded_chord_neural_viscosity_symmetric keeps the inherited "
                             "bounded-chord value intact and learns that smooth correction "
                             "with a dedicated symmetric endpoint/chord-speed MLP; "
                             "bounded_chord_signed_clearance_symmetric sharpens only "
                             "the chord's physical barrier using a learned signed scalar "
                             "distance and an exact-no-op auxiliary gate; "
                             "bounded_chord_clearance_eikonal_symmetric makes the "
                             "signed scalar clearance the physical propagation speed "
                             "of the main Eikonal field rather than an auxiliary loss; "
                             "bounded_chord_clearance_state_symmetric extends the "
                             "feasible viscosity solution into invalid endpoints with "
                             "a signed-clearance state-constraint boundary; "
                             "endpoint_eikonal_residual_symmetric uses endpoint speed "
                             "only for the exact local PDE limit and learns a smooth "
                             "global viscosity residual without a privileged chord; "
                             "local_endpoint_eikonal_residual_symmetric restricts "
                             "that endpoint term to one fixed local radius and leaves "
                             "all long-range shaping to a non-negative D^2 excess; "
                             "bounded_chord_profile_symmetric lets that attenuation "
                             "see the same scalar-speed samples used by the chord bound; "
                             "fermat_waypoint_symmetric integrates the same scalar speed "
                             "along one feed-forward two-segment detour; "
                             "fermat_curve_symmetric integrates one midpoint-symmetric "
                             "reversible curve; fermat_curve_equivariant keeps all sine "
                             "modes with exact endpoint-swap parity")
    parser.add_argument("--endpoint_task_siamese", action="store_true",
                        help="Use one shared endpoint encoder phi(q, task) and "
                             "a zero-boundary latent metric for travel time.")
    parser.add_argument(
        "--endpoint_dual_value_heads", action="store_true",
        help="Use the legacy cup/pour endpoint encoders and pointwise "
             "feature/SLERP routing inside the modern single scalar value "
             "head. Requires --endpoint_task_siamese.")
    parser.add_argument(
        "--pure_scalar_q_only", action="store_true",
        help=(
            "Use one symmetric q-only scalar loss: ignore pair/task routing "
            "and consume the identical scalar speed copies. Requires "
            "--endpoint_task_siamese --value_condition none and forbids "
            "--endpoint_dual_value_heads."))
    parser.add_argument("--task_embedding_scale", type=float, default=0.25,
                        help="Amplitude of the cup/pour token added to the shared "
                             "endpoint Fourier representation.")
    parser.add_argument("--fourier_basis", type=str, default="random",
                        choices=["random", "kinematic", "hybrid_kinematic",
                                 "mixed_kinematic", "multiscale_random"],
                        help="Fourier frequency construction. Kinematic modes use "
                             "integer revolute-joint angle combinations; "
                             "multiscale_random spans six log-spaced bands.")
    parser.add_argument("--global_residual_scale", type=float, default=2.0,
                        help="Bound on log travel-time detour correction for "
                             "--value_arch global_residual.")
    parser.add_argument("--use_speed_head", action="store_true",
                        help="Regress the scalar speed F(q) inside the same value model.")
    parser.add_argument(
        "--separate_speed_encoder", action="store_true",
        help="Use a local multiscale Fourier encoder for scalar speed while "
             "keeping the smooth travel-time encoder in the same model.")
    parser.add_argument(
        "--speed_fourier_basis", type=str, default="multiscale_random",
        choices=["random", "kinematic", "hybrid_kinematic",
                 "mixed_kinematic", "multiscale_random"],
        help="Fixed positional basis for the scalar-speed branch. Kinematic "
             "modes follow the integer joint-angle combinations induced by "
             "serial-arm forward kinematics.")
    parser.add_argument(
        "--speed_B_scale", type=float, default=0.2,
        help="Base scale for the separate multiscale speed Fourier basis.")
    parser.add_argument(
        "--speed_B_freqs", type=int, default=128,
        help="Number of Fourier rows in the separate scalar-speed encoder.")
    parser.add_argument(
        "--speed_residual_layers", type=int, default=0,
        help="Zero-initialized residual refinement blocks in the one scalar "
             "speed field. Zero exactly preserves historical checkpoints.")
    parser.add_argument(
        "--lifted_phase", action="store_true",
        help="Lift the scalar Eikonal field by one continuous handover-state "
             "coordinate while retaining one checkpoint and one speed output.")
    parser.add_argument(
        "--lifted_phase_physical_dim", type=int, default=0,
        help="Number of physical q coordinates before the final lifted phase.")
    parser.add_argument(
        "--lifted_phase_scale", type=float, default=0.1,
        help="Metric length represented by a complete p=0 to p=1 handover.")
    parser.add_argument(
        "--lifted_phase_width", type=int, default=128,
        help="Hidden width of the in-model scalar phase restriction.")
    parser.add_argument(
        "--lifted_phase_logit_scale", type=float, default=8.0,
        help="Maximum downward logit correction from phase inconsistency.")
    parser.add_argument(
        "--lifted_phase_causal", action="store_true",
        help=("Use the lifted coordinate as a causal handover clock: cross "
              "motions may advance it but cannot run it backward."))
    parser.add_argument(
        "--lifted_phase_hazard_floor_scale", type=float, default=0.0,
        help=("Strength in [0,1] of the same-scalar chord-hazard floor "
              "for phase-changing value queries. Zero exactly preserves "
              "the inherited q-only value head."))
    parser.add_argument(
        "--cross_hazard_floor_scale", type=float, default=0.0,
        help=("Strength in [0,1] of a cross-gated lower bound on the "
              "same scalar-speed chord hazard. Zero exactly preserves the "
              "inherited value field and within-task pairs are unchanged."))
    parser.add_argument(
        "--use_clearance_head", action="store_true",
        help="Predict one signed physical-clearance scalar from the shared "
             "speed trunk; no normal or collision query is used at inference.")
    parser.add_argument(
        "--clearance_scale", type=float, default=0.05,
        help="Meters represented by normalized signed clearance 1.0.")
    parser.add_argument(
        "--clearance_boundary", type=float, default=0.005,
        help="Strict obstacle boundary in meters for signed-clearance speed.")
    parser.add_argument(
        "--clearance_speed_floor", type=float, default=0.001,
        help="Positive numerical floor for the signed-clearance Eikonal speed.")
    parser.add_argument(
        "--clearance_residual_layers", type=int, default=0,
        help="Zero-initialized clearance-specific residual blocks after the "
             "shared scalar encoder. They add signed-depth capacity without "
             "changing the historical speed field.")
    parser.add_argument(
        "--clearance_fourier_skip", action="store_true",
        help="Concatenate the shared raw kinematic Fourier coordinate into "
             "the clearance head, bypassing information discarded by the "
             "clipped-speed trunk while leaving that trunk unchanged.")
    parser.add_argument(
        "--couple_speed_latent_to_value", action="store_true",
        help="Expose reversal-invariant endpoint features from the same "
             "scalar-speed trunk to a learned bounded-chord value head.")
    parser.add_argument(
        "--partial_shared_embedding_dim", type=int, default=0,
        help="Width of a q-only embedding shared through zero-initialized "
             "adapters by the otherwise-private E and F encoders.")
    parser.add_argument(
        "--pair_slowness_attention_dim", type=int, default=0,
        help="Width of one symmetric Q/K attention lane from the endpoint "
             "pair into chord-local scalar-speed tokens. V contains only "
             "slowness-derived quantities; zero disables the lane.")
    parser.add_argument(
        "--speed_constraint_heads", type=int, default=0,
        help="Number of q-only signed-clearance residual heads analytically "
             "reduced to the one deployed scalar F; zero disables them.")
    parser.add_argument(
        "--speed_constraint_softmin_temperature", type=float, default=0.04,
        help="Dimensionless normalized-softmin temperature for the internal "
             "constraint components (tau_m/free_distance_m).")
    parser.add_argument(
        "--speed_constraint_residual_scale", type=float, default=1.25,
        help="Maximum tanh residual amplitude around the inherited scalar "
             "base for each signed component head.")
    parser.add_argument(
        "--speed_constraint_fourier_skip", action="store_true",
        help="Concatenate the fixed raw speed Fourier coordinate to each "
             "semantic private head while retaining the shared latent.")
    parser.add_argument(
        "--speed_constraint_consensus_width", type=int, default=0,
        help="Width of a q-dependent alpha gate that blends the semantic "
             "soft-min with the shared direct scalar estimate; zero disables it.")
    parser.add_argument(
        "--speed_constraint_consensus_mode",
        choices=(
            "single", "hurdle", "corridor_hurdle",
            "corridor_hurdle_compensated",
            "corridor_hurdle_compensated_balanced",
            "corridor_hurdle_specificity",
            "corridor_hurdle_specificity_odds",
            "corridor_hurdle_specificity_orthogonal",
            "corridor_hurdle_semantic_veto_min_orthogonal",
            "corridor_hurdle_semantic_veto_min_odds",
            "corridor_hurdle_semantic_veto_min_odds_residual",
            "corridor_hurdle_semantic_veto_min_odds_residual_multihead_blend",
            "corridor_hurdle_semantic_veto_min_odds_residual_headwise_logit8",
            *model_network.ROUTE_CHART_MODES),
        default="single",
        help="Consensus parameterization: one sigmoid alpha; a shared trunk "
             "with private release/magnitude heads; or an upward-only "
             "corridor hurdle stacked after component repair; compensated "
             "uses the same deployed product but release-aware magnitude "
             "supervision; balanced adds threshold-dangerous negatives; "
             "specificity separates that classifier into a private tail; "
             "specificity_odds uses a neutral log-odds fusion; orthogonal "
             "keeps specificity outside magnitude compensation; semantic "
             "veto uses one private permission per component and their min; "
             "semantic veto odds updates coarse release with that minimum; "
             "the residual variant adds a zero-warm-started 64-wide private "
             "nonlinear magnitude tail; headwise_logit8 adds one bounded "
             "q-conditioned correction per semantic permission head; "
             "routechart4_soft0125 adds four private release experts under "
             "one fixed differentiable q-space configuration gate; "
             "routechart4_unionlme005 keeps the same training charts but "
             "deploys their constant-preserving smooth corridor union; "
             "the Fourier physicalrisk4 variant adds four internal shared/"
             "private semantic specialists whose weakest negative evidence "
             "can only veto that union; the partial-shared Fourier variant "
             "also gives the four route charts one shared raw-q Fourier "
             "branch while retaining private hidden/fusion tails. Public F "
             "remains one scalar in both variants.")
    parser.add_argument(
        "--speed_constraint_repair_width", type=int, default=0,
        help="Width of a shared q-only trunk whose private release/magnitude "
             "tails repair each semantic component toward the direct scalar "
             "before the one analytic soft minimum; zero disables it.")
    parser.add_argument("--global_residual_positive", action="store_true",
                        help="Enforce T(q,g) >= Euclidean distance structurally.")
    parser.add_argument(
        "--global_excess_centered_gate", action="store_true",
        help=(
            "For bounded-chord global/smooth excess architectures, use a "
            "zero-valued centered-softplus detour gate with nonzero gradient "
            "at the neutral warm start. The correction remains one-sided."))
    parser.add_argument(
        "--global_excess_centered_signed_gate", action="store_true",
        help=(
            "For bounded-chord global/smooth excess architectures, use a "
            "zero-valued centered-softplus signed correction. It has a "
            "nonzero derivative at raw=-6 and can raise or lower travel time."))
    parser.add_argument("--chord_quad_points", type=int, default=3,
                        choices=[2, 3, 5, 7, 15],
                        help="Gauss-Legendre points per segment for the learned "
                             "straight-chord or Fermat-path integral.")
    parser.add_argument(
        "--chord_quadrature", type=str, default="legendre",
        choices=["legendre", "lobatto"],
        help="Symmetric chord quadrature. 'legendre' preserves the historical "
             "open rule; 'lobatto' includes both endpoints so the scalar "
             "speed at q and g enters the physical travel-time bound.")
    parser.add_argument(
        "--use_raw_coordinate_skip", action="store_true",
        help="Add one learned raw-q residual to the shared Fourier encoder, "
             "preserving absolute joint position inside finite limits.")
    parser.add_argument(
        "--speed_floor", type=float, default=0.1,
        help="Minimum normalized scalar speed. Values below the historical "
             "0.1 make strict obstacle boundaries closer to impassable.")
    parser.add_argument(
        "--chord_hazard_floor_scale", type=float, default=1.0,
        help="Strength in [0,1] of the scalar-speed hazard lower bound used "
             "by bounded_chord_hazard_floor_symmetric. It is stored in the "
             "single checkpoint; zero exactly recovers the inherited "
             "bounded-chord value and one is the parameter-free full bound.")
    parser.add_argument(
        "--endpoint_barrier_speed", type=float, default=0.1,
        help="Speed below which the endpoint-Eikonal architecture applies its "
             "log-speed signed-depth extension. Choose a conservative interior "
             "value when the learned scalar boundary has calibration error.")
    parser.add_argument(
        "--endpoint_barrier_scale", type=float, default=1.0,
        help="Non-negative amplitude of the learned endpoint-speed barrier. "
             "One preserves the original architecture; zero is an exact "
             "bounded-chord ablation.")
    parser.add_argument(
        "--endpoint_barrier_radius", type=float, default=0.05,
        help="Positive local joint-space length of the saturated endpoint "
             "speed barrier. It should match the largest local Eikonal step.")
    parser.add_argument(
        "--endpoint_barrier_power", type=float, default=1.0,
        help="Power >=1 shaping normalized log-speed penetration. One is the "
             "linear signed-depth extension; larger values retain the same "
             "floor depth while smoothly weakening the feasible boundary.")
    parser.add_argument(
        "--endpoint_barrier_physical_decode", action="store_true",
        help="Apply endpoint state constraints to the physical sigmoid speed "
             "before its historical affine 0.1 storage floor.")
    parser.add_argument(
        "--global_pair_residual_layers", type=int, default=0,
        help="Zero-initialized residual reasoning layers in the symmetric "
             "endpoint-pair value head.")
    parser.add_argument(
        "--global_pair_residual_activation", type=str, default="silu",
        choices=["silu", "maxmin"],
        help="Activation in optional symmetric pair-residual blocks.")
    parser.add_argument(
        "--use_viscosity_excess_head", action="store_true",
        help=("Add an exact-zero, non-negative O(distance^2) viscosity "
              "correction while preserving the base value architecture."))
    parser.add_argument(
        "--viscosity_excess_gate", type=str, default="all",
        choices=[
            "all", "cross_strength", "cross_transverse",
            "cross_transverse_saturated", "cross_chord_evidence",
            "cross_certain"],
        help=("Structural support of the viscosity correction. "
              "cross_strength leaves within-task values exactly inherited "
              "and activates continuously on cup<->pour pairs; "
              "cross_transverse additionally requires asymmetric scalar "
              "speed on the two sides of the endpoint chord; its saturated "
              "variant passes strong evidence without attenuation. "
              "cross_chord_evidence trusts local graph-scale pairs and "
              "requires integrated scalar barrier evidence for long pairs; "
              "cross_certain removes ambiguous same-task routing mass before "
              "activating the cross branch."))
    parser.add_argument(
        "--viscosity_residual_mode", type=str, default="positive_time",
        choices=["positive_time", "signed_fraction", "causal_log_time"],
        help=("How the dedicated in-model viscosity branch changes T. "
              "positive_time is the historical non-negative O(distance^2) "
              "excess; signed_fraction adds a bounded signed correction to "
              "the scalar chord-mixing logit, preserving T inside the same "
              "physical distance/chord bounds; causal_log_time learns one "
              "ordered multiplicative correction for a directed state-"
              "constrained Eikonal value."))
    parser.add_argument(
        "--viscosity_stencil_radius", type=float, default=0.0,
        help=("Normalized joint-space radius of the fixed transverse "
              "scalar-speed stencil used by the in-model viscosity branch. "
              "Zero preserves the historical chord-only input."))
    parser.add_argument(
        "--viscosity_gate_radius", type=float, default=0.0,
        help=("Normalized joint-space radius used only by a transverse "
              "scalar-speed support gate. Unlike viscosity_stencil_radius, "
              "these samples are not appended to the learned MLP input."))
    parser.add_argument(
        "--viscosity_speed_latent_context", action="store_true",
        help=("Expose the complete speed-supervised endpoint latent to the "
              "dedicated in-model viscosity branch without changing the "
              "inherited value head or adding scalar-field evaluations."))
    parser.add_argument(
        "--viscosity_chord_speed_latent_context", action="store_true",
        help=("Reuse the speed-supervised latents at the existing chord "
              "quadrature points as a compact reversal-invariant profile "
              "inside the viscosity branch; this adds no field queries."))
    parser.add_argument(
        "--viscosity_chord_latent_width", type=int, default=32,
        help=("Per-chord-point projection width for the compact latent "
              "profile used by --viscosity_chord_speed_latent_context."))
    parser.add_argument(
        "--viscosity_directed_context", action="store_true",
        help=("Expose ordered endpoint latent/joint/routing differences to "
              "one causal scalar-value branch. Requires "
              "--viscosity_residual_mode causal_log_time."))
    parser.add_argument(
        "--viscosity_handover_phase_context", action="store_true",
        help=("Append the deterministic FK handover phase at both endpoints "
              "to the one directed scalar-value branch."))
    parser.add_argument(
        "--handover_phase_embedding_scale", type=float, default=0.0,
        help=("Scale of the deterministic FK handover coordinate inside one "
              "product-space scalar Eikonal distance. Zero disables it."))
    parser.add_argument(
        "--viscosity_activation", type=str, default="silu",
        choices=["silu", "maxmin"],
        help=("Hidden activation of the single scalar viscosity branch. "
              "maxmin is a pairwise GroupSort activation that can represent "
              "the nonsmooth Eikonal cut locus without multiple outputs or "
              "candidate paths."))
    parser.add_argument(
        "--viscosity_unbounded_logit", action="store_true",
        help=("Do not tanh-cap the viscosity branch before the final sigmoid. "
              "The scalar travel time remains structurally bounded between "
              "Euclidean distance and chord-integrated time."))
    parser.add_argument(
        "--viscosity_minplus_modes", type=int, default=0,
        choices=[0, 2, 3, 4],
        help=("Number of internal scalar modes in the normalized soft-min "
              "viscosity envelope. Zero disables it; modes remain internal "
              "and inference still returns one symmetric scalar value."))
    parser.add_argument(
        "--viscosity_minplus_temperature", type=float, default=0.25,
        help="Temperature of the internal viscosity soft minimum.")
    parser.add_argument(
        "--viscosity_minplus_scale", type=float, default=1.0,
        help="Amplitude of the internal viscosity soft-min correction.")
    parser.add_argument(
        "--viscosity_minplus_mode", type=str, default="correction",
        choices=["correction", "tau_envelope"],
        help=("Where to apply the internal soft minimum. 'correction' "
              "keeps the historical normalized correction envelope; "
              "'tau_envelope' takes a normalized smooth minimum over the "
              "base and internal scalar travel-time branches."))
    parser.add_argument(
        "--fermat_curve_modes", type=int, default=3,
        help="Number of odd sine modes in the single reversible Fermat curve.")
    parser.add_argument(
        "--fermat_curve_envelope_detach", action="store_true",
        help="At eval only, use the Fermat envelope derivative by holding the "
                             "single decoded curve coefficients fixed while differentiating "
                             "its scalar-speed path integral with respect to the endpoints.")
    parser.add_argument(
        "--fermat_curve_quad_points", type=int, default=15,
        choices=[3, 5, 7, 15],
        help="Independent Gauss-Legendre rule for the single Fermat curve.")
    parser.add_argument(
        "--fermat_curve_residual_layers", type=int, default=6,
        help="Identity-initialized reasoning blocks in the one-curve decoder.")
    parser.add_argument(
        "--fermat_curve_speed_floor", type=float, default=0.005,
        help="Numerical floor after undoing the scalar speed's affine storage code.")
    parser.add_argument(
        "--fermat_curve_value_weight", type=float, default=1.0,
        help="Continuous [0,1] replacement strength of the straight upper bound.")
    parser.add_argument(
        "--fermat_curve_gate",
        choices=["all", "cross_strength", "cross_certain"], default="all",
        help="Restrict the one-curve correction to the cross routing boundary.")
    parser.add_argument("--speed_loss_weight", type=float, default=0.0,
                        help="Pointwise scalar-speed regression weight.")
    parser.add_argument("--speed_loss_balance_power", type=float, default=0.0,
                        help="Reweight speed MSE by speed**(-power), normalized per batch.")
    parser.add_argument(
        "--speed_loss_form", type=str, default="speed",
        choices=["speed", "log_speed", "slowness"],
        help="Coordinate for fitting the same scalar speed label. slowness "
             "fits normalized (1/F-1), the coefficient used by Eikonal "
             "propagation; no extra supervision is introduced.")
    parser.add_argument(
        "--collocation_speed_loss_weight", type=float, default=0.0,
        help="Pointwise scalar-speed regression weight on independent "
             "collocation states. Requires collocation_speed.npy.")
    parser.add_argument(
        "--collocation_speed_n", type=int, default=0,
        help="Independent collocation states sampled per endpoint batch for "
             "scalar-speed supervision.")
    parser.add_argument(
        "--collocation_speed_loss_balance_power", type=float, default=0.0,
        help="Reweight independent collocation speed loss by target "
             "speed**(-power), normalized per cloud minibatch.")
    parser.add_argument(
        "--collocation_speed_loss_form", type=str, default="inherit",
        choices=["inherit", "speed", "log_speed", "slowness"],
        help="Regression coordinate for independent collocation speed; "
             "inherit uses --speed_loss_form.")
    parser.add_argument(
        "--collocation_speed_component_loss_weight", type=float, default=0.0,
        help="MSE weight for training-only signed q-only component labels in "
             "collocation_speed_components.npy.")
    parser.add_argument(
        "--collocation_speed_uncomposed_loss_weight", type=float, default=0.0,
        help="Regression weight applying the same aggregate scalar F(q) "
             "labels directly to the shared pre-consensus estimate. Requires "
             "a scalar consensus head and introduces no new labels.")
    parser.add_argument(
        "--collocation_speed_consensus_identification_weight",
        type=float,
        default=0.0,
        help="MSE weight for the algebraically identified best convex "
             "coefficient between fixed direct and semantic scalar experts. "
             "Uses only the existing exact aggregate F(q) label.")
    parser.add_argument(
        "--collocation_speed_sampling",
        choices=["uniform", "consensus_balanced"],
        default="uniform",
        help="Sampling measure for scalar collocation supervision. The "
             "consensus-balanced mode draws equally from exact rescue support "
             "and threshold-flippable dangerous lows, both derived only from "
             "the training scalar field and frozen scalar experts.")
    parser.add_argument(
        "--collocation_speed_consensus_negative_weight",
        type=float,
        default=1.0,
        help="Relative dangerous-low weight in the head-wise effective-release "
             "objective. One preserves the historical equal-class loss.")
    parser.add_argument(
        "--collocation_speed_consensus_head_component_weight",
        type=float,
        default=0.0,
        help="Relative exact per-component permission BCE added to the "
             "head-wise effective-release objective. This identifies the "
             "safe/cup/pour private corrections without changing inference.")
    parser.add_argument(
        "--collocation_speed_repair_identification_weight",
        type=float,
        default=0.0,
        help="Hurdle-loss weight for algebraically identified one-sided "
             "per-component repair using existing exact component labels.")
    parser.add_argument(
        "--collocation_speed_repair_coordination", action="store_true",
        help="Scale all exact per-component repair targets by one monotone "
             "bisection coefficient so their analytic soft minimum cannot "
             "overshoot the existing exact aggregate F(q) label.")
    parser.add_argument(
        "--clearance_loss_weight", type=float, default=0.0,
        help="Regression weight for normalized signed physical clearance.")
    parser.add_argument(
        "--clearance_self_scale", type=float, default=2.5,
        help="Scale mapping self clearance's 2mm boundary to the obstacle "
             "clearance head's 5mm boundary.")
    parser.add_argument(
        "--hopf_cole_weight", type=float, default=0.0,
        help="Weight of the squared-viscous-Eikonal / Hopf--Cole elliptic "
             "residual. Zero disables it.")
    parser.add_argument(
        "--hopf_cole_n", type=int, default=0,
        help="Pairs per batch receiving the six-dimensional Laplacian loss.")
    parser.add_argument("--hopf_epsilon_start", type=float, default=0.20)
    parser.add_argument("--hopf_epsilon_end", type=float, default=0.02)
    parser.add_argument("--hopf_decay_epochs", type=int, default=600)
    parser.add_argument(
        "--hopf_goal_radius", type=float, default=0.06,
        help="Radius of the finite absorbing goal cell in normalized C-space.")
    parser.add_argument(
        "--hopf_goal_weight", type=float, default=1.0,
        help="Relative Dirichlet penalty of the finite Hopf--Cole goal cell.")
    parser.add_argument(
        "--hopf_adaptive", action="store_true",
        help="Concentrate Hopf--Cole PDE points using only scalar speed and its gradient.")
    parser.add_argument("--hopf_adaptive_speed_power", type=float, default=2.0)
    parser.add_argument("--hopf_adaptive_sharpness_weight", type=float, default=1.0)
    parser.add_argument("--hopf_characteristic_fraction", type=float, default=0.5)
    parser.add_argument("--hopf_characteristic_step", type=float, default=0.02)
    parser.add_argument("--hopf_state_constraint_weight", type=float, default=0.0)
    parser.add_argument("--hopf_boundary_speed", type=float, default=0.35)
    parser.add_argument(
        "--hopf_speed_mode", type=str, default="predicted",
        choices=["predicted", "target"],
        help="Hopf PDE propagation coefficient: learned scalar speed or the "
             "detached independent collocation speed label. The target arm "
             "requires collocation_speed.npy and isolates value-branch error.")
    parser.add_argument(
        "--hopf_residual_form", type=str, default="travel_time",
        choices=["travel_time", "desirability_ritz"],
        help="Strong-form residual in travel time, or the positive weak "
             "Dirichlet energy of the linear Hopf--Cole desirability PDE.")
    parser.add_argument(
        "--hopf_ritz_normalize", action="store_true",
        help="Normalize Ritz interior energy by detached mean z^2 to avoid "
             "small-epsilon gradient underflow without changing the PDE "
             "stationary solution.")
    parser.add_argument(
        "--hopf_shellwise_normalize", action="store_true",
        help="Normalize desirability Ritz energy independently in fixed "
             "log-distance goal shells so far-field PDE gradients do not "
             "underflow behind near-goal samples.")
    parser.add_argument(
        "--hopf_shell_bins", type=int, default=8,
        help="Number of equally weighted log-distance shells used by "
             "--hopf_shellwise_normalize.")
    parser.add_argument(
        "--hopf_endpoint_measure", action="store_true",
        help="Collocate the q-only Hopf PDE on supplied endpoint pairs instead "
             "of independently pairing scalar-cloud states with random goals. "
             "No pair label or trajectory information is consumed.")
    parser.add_argument(
        "--hopf_goal_shell_measure", action="store_true",
        help="Collocate the q-only Hopf PDE on random multiscale shells around "
             "each goal; this changes only the PDE measure and supplies no path.")
    parser.add_argument("--hopf_shell_radius_min", type=float, default=0.03)
    parser.add_argument("--hopf_shell_radius_max", type=float, default=1.0)
    parser.add_argument("--viscosity_weight", type=float, default=0.0,
                        help="Semi-Lagrangian Bellman/viscosity loss weight. "
                             "For vanilla value_arch=metric without a speed "
                             "head, the supplied endpoint scalar speed is "
                             "used as the local one-step cost.")
    parser.add_argument("--viscosity_n", type=int, default=0,
                        help="Pairs per batch used by the viscosity loss.")
    parser.add_argument("--viscosity_dirs", type=int, default=16,
                        help="Local action directions in the Bellman minimum.")
    parser.add_argument(
        "--viscosity_action_mode",
        choices=["discrete_min", "softmin", "stochastic_vv"],
        default="discrete_min",
        help="discrete_min uses a finite action minimum; softmin uses a "
             "normalized smooth log-mean-exp minimum; stochastic_vv uses "
             "the continuous characteristic plus antithetic Brownian walkers "
             "for the vanishing-viscosity Eikonal fixed point.")
    parser.add_argument(
        "--viscosity_softmin_temperature", type=float, default=0.05,
        help="Temperature for the normalized smooth Bellman soft minimum.")
    parser.add_argument(
        "--viscosity_epsilon", type=float, default=0.005,
        help="Diffusion epsilon for stochastic_vv. Walker noise has scale "
             "sqrt(2*epsilon*viscosity_step).")
    parser.add_argument(
        "--viscosity_rollout_steps", type=int, default=1,
        help="Maximum stochastic characteristic steps in one Feynman--Kac "
             "target. Values above one recompute the same scalar field's "
             "characteristic after every step.")
    parser.add_argument(
        "--viscosity_intrinsic_budget", type=float, default=0.0,
        help="Travel-time budget sum(ds/F) for a multi-step stochastic "
             "rollout. The final spatial horizon therefore adapts to speed; "
             "zero retains the historical one-step operator.")
    parser.add_argument("--viscosity_step", type=float, default=0.02,
                        help="Normalized C-space step for Bellman consistency.")
    parser.add_argument("--viscosity_step_max", type=float, default=0.0,
                        help="If larger than viscosity_step, sample Bellman "
                             "steps log-uniformly over this multi-scale range.")
    parser.add_argument("--viscosity_start_epoch", type=int, default=1,
                        help="Epoch at which Bellman consistency starts.")
    parser.add_argument(
        "--viscosity_no_self_descent", action="store_true",
        help="Exclude the value field's own descent direction from Bellman "
             "proposals, preventing a wrong branch from validating itself.")
    parser.add_argument(
        "--viscosity_speed_ascent", action="store_true",
        help="Add clearance-ascent and transverse Fermat-bending proposals "
             "derived by differentiating the same scalar speed head. Uses no "
             "normal labels or external geometry query.")
    parser.add_argument(
        "--viscosity_causal_target", action="store_true",
        help="Bootstrap Bellman targets only from causally earlier states and "
             "stop their gradient, enforcing a one-way update.")
    parser.add_argument(
        "--viscosity_causal_order", type=str, default="euclidean",
        choices=["euclidean", "value", "none"],
        help="Ordering used with --viscosity_causal_target. 'euclidean' "
             "reproduces the historical closer-to-goal filter; 'value' uses "
             "a lower-arrival-time filter; 'none' performs an unrestricted "
             "detached semi-Lagrangian backup so true detours may compete.")
    parser.add_argument(
        "--viscosity_candidate_routing", type=str, default="cup",
        choices=["cup", "local"],
        help="Routing used for Bellman neighbor values. 'cup' reproduces the "
             "historical all-cup override; 'local' preserves the current and "
             "goal endpoint routing so both sides use one value field.")
    parser.add_argument(
        "--viscosity_hard_domain", action="store_true",
        help="Solve a state-constraint Eikonal problem: Bellman segments with "
             "any sampled scalar speed below eikonal_min_speed are outside "
             "the feasible domain. Uses five-point Boole quadrature so a "
             "finite step cannot jump across a thin low-speed band.")
    parser.add_argument(
        "--viscosity_target_lag", type=int, default=0,
        help="If positive, form Bellman labels with a frozen copy synchronized "
             "every N epochs (fitted value iteration). The copy exists only "
             "during training and is never saved or used at inference.")
    parser.add_argument(
        "--viscosity_update_mode", type=str, default="symmetric",
        choices=["symmetric", "raise_only"],
        help="Bellman update direction. raise_only performs monotone fitted "
             "value iteration: it corrects underestimated travel time but "
             "does not propagate spuriously low sampled minima.")
    parser.add_argument(
        "--viscosity_collocation_pairs", action="store_true",
        help=("For pure_scalar_q_only, draw symmetric q--goal pairs from the "
              "independent strict-feasible scalar-speed collocation cloud "
              "instead of reusing endpoint-pair goals. This is a PDE measure "
              "change only; no task, pair, path, or teacher information is "
              "used."))
    parser.add_argument(
        "--viscosity_global_cloud", action="store_true",
        help=("For the pure scalar softmin Bellman operator, mix local "
              "stencil candidates with states from the independent scalar-"
              "speed cloud. Transition costs use the same q-only F(q), with "
              "no graph/path/teacher labels."))
    parser.add_argument(
        "--viscosity_local_boundary_fraction", type=float, default=0.0,
        help=("Fraction of each symmetric independent scalar-PDE pair batch "
              "closed by an isotropic local point-source cell. The displaced "
              "state is generated from the q-only collocation cloud; no task "
              "or pair metadata is used."))
    parser.add_argument(
        "--viscosity_boundary_min_radius", type=float, default=0.002,
        help="Minimum normalized radius of the local scalar PDE cell.")
    parser.add_argument(
        "--viscosity_boundary_max_radius", type=float, default=0.03,
        help="Maximum normalized radius of the local scalar PDE cell.")
    parser.add_argument("--fermat_path_weight", type=float, default=0.0,
                        help="Weight for label-free Fermat path minimization. "
                             "Use with fermat_waypoint_symmetric and normally "
                             "--freeze_base_for_value_head.")
    parser.add_argument(
        "--fermat_loss_transform", choices=["linear", "log"],
        default="linear",
        help="Conditioning of the Fermat training objective. log uses log1p "
             "of path cost; it preserves the exact minimizer while keeping "
             "near-zero-speed obstacle crossings numerically stable.")
    parser.add_argument("--value_condition", type=str, default="task",
                        choices=["none", "task", "task_scene",
                                 "task_scene_dist", "task_scene_pair",
                                 "task_scene_pair_dist"],
                        help="Condition inputs for goal-conditioned value heads")
    parser.add_argument("--finsler_quad_points", type=int, default=2,
                        choices=[1, 2, 3, 5, 9],
                        help="Quadrature samples for --value_arch local_finsler. "
                             "1 uses start-local left rule; 2 preserves the "
                             "endpoint-average C1 behavior.")
    parser.add_argument("--finsler_task_heads", action="store_true",
                        help="Use separate cup/pour local slowness heads for "
                             "--value_arch local_finsler and blend slowness values.")
    parser.add_argument("--use_cross_slowness_residual", action="store_true",
                        help="Add a zero-initialized residual local slowness head "
                             "that activates only for inferred cup<->pour cross pairs.")
    parser.add_argument("--cross_residual_scale", type=float, default=0.5,
                        help="Max log-slowness residual magnitude for "
                             "--use_cross_slowness_residual.")
    parser.add_argument("--use_local_slowness_residual", action="store_true",
                        help="Add a zero-initialized residual local slowness head "
                             "that applies to every local Finsler segment.")
    parser.add_argument("--local_residual_scale", type=float, default=0.2,
                        help="Max log-slowness residual magnitude for "
                             "--use_local_slowness_residual.")
    parser.add_argument("--local_residual_gate", type=str, default="all",
                        choices=["all", "cross_strength"],
                        help="Where to apply --use_local_slowness_residual. "
                             "'cross_strength' gates it by inferred endpoint "
                             "cup<->pour routing strength.")
    parser.add_argument("--freeze_base_for_cross_residual", action="store_true",
                        help="Train only slowness residual parameters.")
    parser.add_argument("--train_local_residual_only", action="store_true",
                        help="With --freeze_base_for_cross_residual, train only "
                             "local_slowness_residual and keep cross residual frozen.")
    parser.add_argument("--dist_condition_only", action="store_true",
                        help="Freeze the warm-started field and train only the "
                             "new final distance-condition column in the value "
                             "head first layer. Requires task_scene_dist.")
    parser.add_argument("--pair_condition_only", action="store_true",
                        help="Freeze the warm-started field and train only the "
                             "new pair-context condition columns in the value "
                             "head first layer. Requires task_scene_pair or "
                             "task_scene_pair_dist.")
    parser.add_argument("--use_align_loss", action="store_true",
                        help="Add feature alignment loss for cross-zone pairs (pair_type==2)")
    parser.add_argument("--use_dual_sym", action="store_true",
                        help="Compute tau as max(tau_blended, tau_cup, tau_pour)")
    parser.add_argument("--cross_mode", type=str, default="default",
                        help="Cross-zone mode in loss: default, hard_cross, sharp, half_cross, "
                             "cross_equal_weights, zone_cross, cross_maxspeed, cross_highspeed, "
                             "cross_downweight, no_cross")
    parser.add_argument("--cross_loss_weight", type=float, default=1.0,
                        help="Multiplier for pair_type==2 samples in the training loss")
    parser.add_argument("--cup_loss_weight", type=float, default=1.0,
                        help="Multiplier for pair_type==0 samples in the training loss")
    parser.add_argument("--pour_loss_weight", type=float, default=1.0,
                        help="Multiplier for pair_type==1 samples in the training loss")
    parser.add_argument("--blend_mode", type=str, default="feature",
                        choices=["feature", "tau_blend", "tau_blend_perpoint", "tau_softmax",
                                 "cross_attention", "film", "slerp"],
                        help="How to combine cup/pour outputs: feature (default, FB-NTFields), "
                             "tau_blend (blend tau values), tau_softmax (soft-min of taus)")
    parser.add_argument("--cross_start_epoch", type=int, default=0,
                        help="Epoch to start training on cross-zone pairs (0=always, >0=curriculum)")
    parser.add_argument("--cross_lr", type=float, default=None,
                        help="Learning rate for cross phase (after cross_start_epoch). "
                             "If None, uses --lr for all epochs.")
    parser.add_argument("--freeze_encoders_on_cross", action="store_true",
                        help="Freeze encoder weights when cross phase starts. "
                             "Only pe_gate/encoder_norm/sym_op remain trainable.")
    parser.add_argument("--detach_routing_weights", action="store_true",
                        help="Detach routing weights before feature blending to prevent "
                             "boundary-seeking gradients through dw/dq")
    parser.add_argument("--no_lip_norm", action="store_true",
                        help="Disable Lipschitz normalization on all linear layers")
    parser.add_argument("--add_orient_to_tau", action="store_true",
                        help="R58G: bake orientation cost Phi(start) into tau "
                             "(tau_total = tau + lambda*Phi). Forces use_orient_head on. "
                             "lambda from env R58_ORIENT_TAU_LAMBDA (default 0.1).")
    parser.add_argument("--use_orient_head", action="store_true",
                        help="R58F: add a decoupled additive orientation head Phi_orient(q). "
                             "Off by default (behavior identical to current code when off).")
    parser.add_argument("--use_task_orient_heads", action="store_true",
                        help="Use separate cup/pour orientation scalar heads blended by routing.")
    parser.add_argument("--use_safety_head", action="store_true",
                        help="Add a learned config-only collision/self-collision potential Phi_safety(q).")
    parser.add_argument("--add_safety_to_tau", action="store_true",
                        help="Bake learned Phi_safety(start) into tau. Forces --use_safety_head on; "
                             "lambda from env R58_SAFETY_TAU_LAMBDA (default 0.1).")
    parser.add_argument("--use_pair_safety_heads", action="store_true",
                        help="Use separate cup/pour/cross safety scalar heads selected by pair_type.")
    parser.add_argument("--pair_safety_mode", type=str, default="replace",
                        choices=["replace", "residual", "constraint_residual"],
                        help="Pair heads either replace the shared potential or add a "
                             "zero-centred correction to it.")
    parser.add_argument("--pair_safety_residual_scale", type=float, default=1.0,
                        help="Scale of task-conditioned safety residuals.")
    parser.add_argument("--freeze_base_for_safety_head", action="store_true",
                        help="Train only safety_head parameters while keeping the base NTField frozen.")
    parser.add_argument("--freeze_base_for_value_head", action="store_true",
                        help="Train only the value/slowness head while keeping encoders, "
                             "routing, orientation, and safety heads frozen.")
    parser.add_argument("--train_speed_only", action="store_true",
                        help="Train only the one shared scalar-speed trunk/head; "
                             "all travel-time/value parameters remain frozen.")
    parser.add_argument("--train_speed_head_only", action="store_true",
                        help="Calibrate only speed_head while freezing the encoder and "
                             "all travel-time/value parameters.")
    parser.add_argument(
        "--train_speed_constraint_heads_only", action="store_true",
        help="Freeze every inherited tensor and train only the private signed "
             "constraint-component residual heads.")
    parser.add_argument(
        "--train_speed_consensus_only", action="store_true",
        help="Freeze both scalar-speed experts and every inherited tensor; "
             "train only the q-dependent semantic/direct consensus head.")
    parser.add_argument(
        "--train_speed_consensus_magnitude_only", action="store_true",
        help="Freeze the corridor release/trunk and all inherited tensors; "
             "train only the mode's registered private magnitude/evidence "
             "tail with release-aware supervision.")
    parser.add_argument(
        "--train_speed_consensus_specificity_magnitude_only",
        action="store_true",
        help="Freeze release, the shared consensus trunk, and all inherited "
             "tensors; train only specificity and magnitude private tails.")
    parser.add_argument(
        "--train_speed_repair_only", action="store_true",
        help="Freeze both scalar experts and train only the shared-trunk, "
             "private-tail per-component repair heads.")
    parser.add_argument(
        "--train_clearance_only", action="store_true",
        help="Train only the signed scalar clearance head.")
    parser.add_argument("--train_chord_profile_only", action="store_true",
                        help="Train only the newly appended chord-speed-profile "
                             "columns of bounded_chord_profile_symmetric.")
    parser.add_argument(
        "--train_endpoint_barrier_only", action="store_true",
        help="Freeze a warm-started bounded-chord model and train only row 1 "
             "of a supported two-row auxiliary value head.")
    parser.add_argument(
        "--train_viscosity_excess_only", action="store_true",
        help="Freeze the warm-started value/speed field and train only the "
             "conditional neural viscosity-excess branch.")
    parser.add_argument(
        "--train_viscosity_input_tail_cols", type=int, default=0,
        help="Freeze the warm-started model and train only this many newly "
             "appended columns of the viscosity branch's first input layer.")
    parser.add_argument(
        "--train_value_input_tail_cols", type=int, default=0,
        help="Freeze the warm-started model and train only this many newly "
             "appended columns of the value branch's first input layer.")
    parser.add_argument(
        "--train_lifted_phase_speed_only", action="store_true",
        help="Freeze the inherited q-only model and train only the scalar "
             "speed restriction for the added handover coordinate.")
    parser.add_argument(
        "--train_last_pair_residual_only", action="store_true",
        help="Freeze the warm-started model and train only its final "
             "zero-initialized symmetric pair-residual block.")
    parser.add_argument(
        "--train_partial_shared_embedding_only", action="store_true",
        help="Freeze both inherited E/F private branches and train only the "
             "new common stem and its two private adapters.")
    parser.add_argument("--cross_align_weight", type=float, default=0.0,
                        help="R58I: weight for cross gradient-direction alignment loss "
                             "(aligns -grad(tau) at start with goal-start direction for "
                             "cross pairs). Sets env R58_ALIGN_WEIGHT. 0.0 (default) => off.")
    parser.add_argument("--use_boundary_align", action="store_true",
                        help="Add feature alignment loss for points near zone boundary")
    parser.add_argument("--boundary_align_margin", type=float, default=0.05,
                        help="Half-width of boundary alignment zone in Y (default 0.05)")
    parser.add_argument("--boundary_align_weight", type=float, default=1e-3,
                        help="Weight for boundary alignment loss (default 1e-3)")
    parser.add_argument("--and_overlap", type=float, default=0.0,
                        help="AND-logic overlap half-width: in overlap zone, speed target "
                             "is blended from speed_cup and speed_pour for cross pairs. "
                             "0 = disabled (default). E.g. 0.05 for overlap [-0.25, -0.15].")
    parser.add_argument("--and_blend", type=str, default="min",
                        choices=["min", "geomean", "avg", "harmonic",
                                 "max", "const1", "const05", "max_floor03", "y_interp"],
                        help="How to blend speed_cup/speed_pour in overlap for cross pairs.")
    parser.add_argument("--weight_mode", type=str, default="default",
                        help="Routing weight computation mode (default, sharp_boundary, sigmoid_y, "
                             "y_distance_scaled, yds_scale010, yds_scale020, yds_scale025, "
                             "yds_scale030, yds_ydiv175, yds_ydiv225, yds_ydiv175_s020, "
                             "yds_orient_boost, yds_quadratic, yds_asymmetric, etc.)")

    # --- Path-level self-supervision (self-rollout) loss (default OFF) ---
    parser.add_argument("--use_path_loss", action="store_true",
                        help="Enable self-rollout path losses")
    parser.add_argument("--path_loss_pair_types", type=str, default="cross",
                        help="Pair types for path loss: all, cup, pour, cross, or comma-separated list")
    parser.add_argument("--path_loss_pair_pool", type=str, default=None,
                        help="Optional NPZ of normalized start/goal pairs used as "
                             "path-loss anchors. Uses only endpoints and pair_type; "
                             "collision/tilt labels still come from the scene oracle.")
    parser.add_argument("--path_loss_pair_pool_n", type=int, default=0,
                        help="Maximum number of pair-pool rows to keep. <=0 keeps all.")
    parser.add_argument("--path_loss_pair_pool_seed", type=int, default=6262)
    parser.add_argument("--path_handover_accept", action="store_true",
                        help="Match corner-table handover_accept metric inside path loss: "
                             "strict wing tilt, corner monotonic handover ranking")
    parser.add_argument("--path_handover_weight", type=float, default=0.0,
                        help="Weight for no-teacher handover monotonicity ranking in path loss")
    parser.add_argument("--path_handover_mono_tol", type=float, default=0.10,
                        help="Allowed corner handover-coordinate decrease per rollout step")
    parser.add_argument("--path_corner_overlap_deg", type=float, default=45.0,
                        help="Corner blend angular half-width for handover-aware path loss")
    parser.add_argument("--path_collision_only", action="store_true",
                        help="In path loss, rank only collision/self-collision "
                             "violations and build safe neighbors without tilt "
                             "filtering. Useful for corner-table handover where "
                             "safe_index lacks ee_x for exact corner tilt filtering.")
    parser.add_argument("--rollout_steps", type=int, default=12)
    parser.add_argument("--rollout_step_size", type=float, default=0.03)
    parser.add_argument("--rollout_n", type=int, default=256)
    parser.add_argument("--path_eik_weight", type=float, default=1e-2)
    parser.add_argument("--rank_weight", type=float, default=1e-1)
    parser.add_argument("--rank_margin", type=float, default=0.02)
    parser.add_argument("--rank_warmup_epochs", type=int, default=600)
    parser.add_argument("--path_eik_warmup_epochs", type=int, default=200)
    parser.add_argument("--path_margin", type=float, default=0.05,
                        help="Obstacle margin (m) for violation/speed in path loss")
    parser.add_argument("--path_self_margin", type=float, default=0.002,
                        help="Self-collision margin (m) for path-loss violation labels")
    parser.add_argument("--cup_tilt_max", type=float, default=20.0)
    parser.add_argument("--pour_tilt_max", type=float, default=20.0)
    parser.add_argument("--cup_tilt_soft", type=float, default=10.0)
    parser.add_argument("--pour_tilt_soft", type=float, default=10.0)

    # --- No-teacher local segment barrier (default OFF) ---
    parser.add_argument("--use_segment_barrier", action="store_true",
                        help="Enable random local segment feasibility ranking. "
                             "Uses only detached collision/tilt labels, no teacher "
                             "trajectory or direction.")
    parser.add_argument("--segment_barrier_pair_types", type=str,
                        default="cross",
                        help="Pair types whose endpoints provide anchors: all, cup, "
                             "pour, cross, or comma-separated list")
    parser.add_argument("--segment_barrier_anchor_n", type=int, default=128)
    parser.add_argument("--segment_barrier_dirs", type=int, default=12)
    parser.add_argument("--segment_barrier_step_size", type=float, default=0.02)
    parser.add_argument("--segment_barrier_margin", type=float, default=0.03)
    parser.add_argument("--segment_barrier_self_margin", type=float, default=0.002,
                        help="Self-collision margin for segment barrier labels.")
    parser.add_argument("--segment_barrier_rank_margin", type=float, default=0.01)
    parser.add_argument("--segment_barrier_weight", type=float, default=0.05)
    parser.add_argument("--segment_barrier_start_epoch", type=int, default=1)
    parser.add_argument("--segment_barrier_anchor_mode", type=str, default="both",
                        choices=["start", "goal", "both", "line", "all"])
    parser.add_argument("--segment_barrier_handover_accept", action="store_true",
                        help="Use corner-table wing tilt acceptance for local "
                             "segment labels; corner band waypoints are not "
                             "angle-pinned.")
    parser.add_argument("--segment_barrier_corner_overlap_deg", type=float,
                        default=45.0)
    parser.add_argument("--segment_barrier_no_midpoint", action="store_true",
                        help="Check only the local segment endpoint, not midpoint.")
    parser.add_argument("--segment_barrier_collision_only", action="store_true",
                        help="Ignore tilt labels inside segment barrier and rank "
                             "only obstacle-margin violations.")
    parser.add_argument("--segment_barrier_safe_pool", action="store_true",
                        help="Sample segment-barrier anchors from safe_index.npz "
                             "instead of current batch endpoints. Labels are still "
                             "computed by the detached train-time oracle.")
    parser.add_argument("--segment_barrier_safe_index_dir", type=str, default=None,
                        help="Directory containing safe_index.npz for "
                             "--segment_barrier_safe_pool. Defaults to data_dir.")
    parser.add_argument("--segment_barrier_pool_n", type=int, default=500000,
                        help="Deterministic subsample size for the segment "
                             "safe pool; <=0 keeps all matching rows.")
    parser.add_argument("--segment_barrier_pool_seed", type=int, default=6262)

    # --- No-teacher direct-line preservation (default OFF) ---
    parser.add_argument("--use_line_preserve", action="store_true",
                        help="Enable direct-line-valid preservation. Random batch "
                             "pairs whose straight joint chord is collision/tilt "
                             "valid get a small gradient-alignment/monotonicity "
                             "loss. Uses only scene feasibility labels, no teacher "
                             "trajectory or direction.")
    parser.add_argument("--line_preserve_pair_types", type=str, default="all",
                        help="Pair types eligible for line preservation: all, cup, "
                             "pour, cross, or comma-separated list")
    parser.add_argument("--line_preserve_pair_pool", type=str, default=None,
                        help="Optional NPZ of normalized direct-line candidate "
                             "start/goal pairs. Uses endpoints only; each sampled "
                             "line is still checked by the scene oracle before loss.")
    parser.add_argument("--line_preserve_pair_pool_n", type=int, default=0,
                        help="Maximum line-preserve pair-pool rows to keep. <=0 keeps all.")
    parser.add_argument("--line_preserve_pair_pool_seed", type=int, default=6262)
    parser.add_argument("--line_preserve_n", type=int, default=256)
    parser.add_argument("--line_preserve_checks", type=int, default=7,
                        help="Number of straight-line samples used to accept a pair")
    parser.add_argument("--line_preserve_margin", type=float, default=0.012,
                        help="Obstacle margin for accepting direct-line pairs")
    parser.add_argument("--line_preserve_self_margin", type=float, default=0.002)
    parser.add_argument("--line_preserve_align_weight", type=float, default=0.0,
                        help="Weight for aligning -grad tau with the direct chord")
    parser.add_argument("--line_preserve_mono_weight", type=float, default=0.0,
                        help="Weight for monotonic tau-to-goal along the direct chord")
    parser.add_argument("--line_preserve_mono_margin", type=float, default=0.0,
                        help="Optional progress margin for line monotonicity")
    parser.add_argument("--line_preserve_step_frac", type=float, default=0.20,
                        help="Fraction of the chord used for monotonic line steps")
    parser.add_argument("--line_preserve_start_epoch", type=int, default=1)
    parser.add_argument("--line_preserve_handover_accept", action="store_true",
                        help="Use corner-table handover_accept wing tilt rule for "
                             "line-valid labels.")
    parser.add_argument("--line_preserve_corner_overlap_deg", type=float,
                        default=45.0)
    parser.add_argument("--line_preserve_safe_pool", action="store_true",
                        help="Sample local straight segments from safe_index.npz "
                             "instead of using only batch start-goal pairs. "
                             "Candidate segments are still accepted only after "
                             "oracle collision/tilt line checks.")
    parser.add_argument("--line_preserve_pool_n", type=int, default=500000,
                        help="Maximum number of safe_index q samples to keep on GPU "
                             "for line-preserve local segment sampling. <=0 keeps all.")
    parser.add_argument("--line_preserve_pool_seed", type=int, default=6262)
    parser.add_argument("--line_preserve_local_step_min", type=float, default=0.02)
    parser.add_argument("--line_preserve_local_step_max", type=float, default=0.06)
    parser.add_argument("--scene", type=str, default="scene_ur5e.yml",
                        help="Scene yaml for the train-time collision oracle")
    parser.add_argument("--safe_index_dir", type=str, default=None,
                        help="Dir with safe_index.npz (default: data_dir)")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --separate_all enables all three separation flags
    if args.separate_all:
        args.separate_pe_gate = True
        args.separate_fourier = True
        args.separate_encoder_norm = True
    if args.handover_k > 0:
        args.use_task_adapters = True

    # R58I: --cross_align_weight (if > 0) sets env R58_ALIGN_WEIGHT, which the
    # loss reads. An explicitly-set env var still wins if the flag is left at 0.0.
    if args.cross_align_weight > 0.0:
        os.environ["R58_ALIGN_WEIGHT"] = repr(args.cross_align_weight)

    _reject_teacher_hack_env()

    source = [0.0] * args.dim
    dataset_zone_mode, dataset_corner_zones = load_dataset_zone_config(args.data_dir)
    zone_mode = dataset_zone_mode if args.zone_mode == "auto" else args.zone_mode
    corner_zones = dataset_corner_zones if zone_mode == "corner_table" else None
    if zone_mode == "corner_table" and corner_zones is None:
        raise ValueError(
            "--zone_mode corner_table requires routing_config.json or pipeline_config.json "
            "with corner_zones in the data_dir")
    print(f"[routing] zone_mode={zone_mode}")
    if corner_zones is not None:
        print(f"[routing] corner_zones={corner_zones}")

    model = MultiModel(
        f'./Experiments/multi/{args.exp_name}', args.data_dir, args.dim, source,
        device=args.device,
        B_scale=args.B_scale, B_freqs=args.B_freqs, nl1=args.nl1,
        use_normal_loss=not args.no_normal_loss,
        use_temporal_loss=not args.no_temporal_loss,
        use_eikonal_loss=not args.no_eikonal_loss,
        eikonal_form=args.eikonal_form,
        eikonal_balance_power=args.eikonal_balance_power,
        eikonal_min_speed=args.eikonal_min_speed,
        y_divide=args.y_divide, overlap=args.overlap, route_temp=args.route_temp,
        cup_side=args.cup_side,
        separate_pe_gate=args.separate_pe_gate,
        separate_fourier=args.separate_fourier,
        separate_encoder_norm=args.separate_encoder_norm,
        weight_mode=args.weight_mode,
        use_projector=args.use_projector,
        use_align_loss=args.use_align_loss,
        use_dual_sym=args.use_dual_sym,
        cross_mode=args.cross_mode,
        cup_loss_weight=args.cup_loss_weight,
        pour_loss_weight=args.pour_loss_weight,
        cross_loss_weight=args.cross_loss_weight,
        blend_mode=args.blend_mode,
        detach_routing_weights=args.detach_routing_weights,
        use_boundary_align=args.use_boundary_align,
        boundary_align_margin=args.boundary_align_margin,
        boundary_align_weight=args.boundary_align_weight,
        and_overlap=args.and_overlap,
        and_blend=args.and_blend,
        use_orient_head=args.use_orient_head,
        add_orient_to_tau=args.add_orient_to_tau,
        use_task_orient_heads=args.use_task_orient_heads,
        use_safety_head=args.use_safety_head,
        add_safety_to_tau=args.add_safety_to_tau,
        use_pair_safety_heads=args.use_pair_safety_heads,
        pair_safety_mode=args.pair_safety_mode,
        pair_safety_residual_scale=args.pair_safety_residual_scale,
        freeze_base_for_safety_head=args.freeze_base_for_safety_head,
        freeze_base_for_value_head=args.freeze_base_for_value_head,
        init_checkpoint=args.init_checkpoint,
        zone_mode=zone_mode,
        corner_zones=corner_zones,
        corner_spatial_temp=args.corner_spatial_temp,
        fk_feature_dim=args.fk_feature_dim,
        use_task_adapters=args.use_task_adapters,
        handover_k=args.handover_k,
        handover_temp=args.handover_temp,
        value_arch=args.value_arch,
        value_condition=args.value_condition,
        finsler_quad_points=args.finsler_quad_points,
        finsler_task_heads=args.finsler_task_heads,
        use_cross_slowness_residual=args.use_cross_slowness_residual,
        cross_residual_scale=args.cross_residual_scale,
        use_local_slowness_residual=args.use_local_slowness_residual,
        local_residual_scale=args.local_residual_scale,
        local_residual_gate=args.local_residual_gate,
        endpoint_task_siamese=args.endpoint_task_siamese,
        endpoint_dual_value_heads=args.endpoint_dual_value_heads,
        pure_scalar_q_only=args.pure_scalar_q_only,
        task_embedding_scale=args.task_embedding_scale,
        fourier_basis=args.fourier_basis,
        global_residual_scale=args.global_residual_scale,
        use_speed_head=args.use_speed_head,
        separate_speed_encoder=args.separate_speed_encoder,
        speed_fourier_basis=args.speed_fourier_basis,
        speed_B_scale=args.speed_B_scale,
        speed_B_freqs=args.speed_B_freqs,
        speed_residual_layers=args.speed_residual_layers,
        lifted_phase=args.lifted_phase,
        lifted_phase_physical_dim=args.lifted_phase_physical_dim,
        lifted_phase_scale=args.lifted_phase_scale,
        lifted_phase_width=args.lifted_phase_width,
        lifted_phase_logit_scale=args.lifted_phase_logit_scale,
        lifted_phase_causal=args.lifted_phase_causal,
        lifted_phase_hazard_floor_scale=
            args.lifted_phase_hazard_floor_scale,
        cross_hazard_floor_scale=args.cross_hazard_floor_scale,
        use_clearance_head=args.use_clearance_head,
        clearance_scale=args.clearance_scale,
        clearance_boundary=args.clearance_boundary,
        clearance_speed_floor=args.clearance_speed_floor,
        clearance_residual_layers=args.clearance_residual_layers,
        clearance_fourier_skip=args.clearance_fourier_skip,
        couple_speed_latent_to_value=args.couple_speed_latent_to_value,
        partial_shared_embedding_dim=args.partial_shared_embedding_dim,
        pair_slowness_attention_dim=args.pair_slowness_attention_dim,
        speed_constraint_heads=args.speed_constraint_heads,
        speed_constraint_softmin_temperature=
            args.speed_constraint_softmin_temperature,
        speed_constraint_residual_scale=
            args.speed_constraint_residual_scale,
        speed_constraint_fourier_skip=
            args.speed_constraint_fourier_skip,
        speed_constraint_consensus_width=
            args.speed_constraint_consensus_width,
        speed_constraint_consensus_mode=
            args.speed_constraint_consensus_mode,
        speed_constraint_repair_width=
            args.speed_constraint_repair_width,
        global_residual_positive=args.global_residual_positive,
        global_excess_centered_gate=args.global_excess_centered_gate,
        global_excess_centered_signed_gate=(
            args.global_excess_centered_signed_gate),
        chord_quad_points=args.chord_quad_points,
        chord_quadrature=args.chord_quadrature,
        chord_hazard_floor_scale=args.chord_hazard_floor_scale,
        use_raw_coordinate_skip=args.use_raw_coordinate_skip,
        speed_floor=args.speed_floor,
        endpoint_barrier_speed=args.endpoint_barrier_speed,
        endpoint_barrier_scale=args.endpoint_barrier_scale,
        endpoint_barrier_radius=args.endpoint_barrier_radius,
        endpoint_barrier_power=args.endpoint_barrier_power,
        endpoint_barrier_physical_decode=
            args.endpoint_barrier_physical_decode,
        global_pair_residual_layers=args.global_pair_residual_layers,
        global_pair_residual_activation=
            args.global_pair_residual_activation,
        use_viscosity_excess_head=args.use_viscosity_excess_head,
        viscosity_excess_gate=args.viscosity_excess_gate,
        viscosity_residual_mode=args.viscosity_residual_mode,
        viscosity_stencil_radius=args.viscosity_stencil_radius,
        viscosity_gate_radius=args.viscosity_gate_radius,
        viscosity_speed_latent_context=
            args.viscosity_speed_latent_context,
        viscosity_chord_speed_latent_context=
            args.viscosity_chord_speed_latent_context,
        viscosity_chord_latent_width=
            args.viscosity_chord_latent_width,
        viscosity_directed_context=
            args.viscosity_directed_context,
        viscosity_handover_phase_context=
            args.viscosity_handover_phase_context,
        handover_phase_embedding_scale=
            args.handover_phase_embedding_scale,
        viscosity_activation=args.viscosity_activation,
        viscosity_unbounded_logit=args.viscosity_unbounded_logit,
        viscosity_minplus_modes=args.viscosity_minplus_modes,
        viscosity_minplus_temperature=args.viscosity_minplus_temperature,
        viscosity_minplus_scale=args.viscosity_minplus_scale,
        viscosity_minplus_mode=args.viscosity_minplus_mode,
        fermat_curve_modes=args.fermat_curve_modes,
        fermat_curve_envelope_detach=args.fermat_curve_envelope_detach,
        fermat_curve_quad_points=args.fermat_curve_quad_points,
        fermat_curve_residual_layers=args.fermat_curve_residual_layers,
        fermat_curve_speed_floor=args.fermat_curve_speed_floor,
        fermat_curve_value_weight=args.fermat_curve_value_weight,
        fermat_curve_gate=args.fermat_curve_gate,
        speed_loss_weight=args.speed_loss_weight,
        speed_loss_balance_power=args.speed_loss_balance_power,
        speed_loss_form=args.speed_loss_form,
        collocation_speed_loss_weight=args.collocation_speed_loss_weight,
        collocation_speed_n=args.collocation_speed_n,
        collocation_speed_loss_balance_power=
            args.collocation_speed_loss_balance_power,
        collocation_speed_loss_form=args.collocation_speed_loss_form,
        collocation_speed_component_loss_weight=
            args.collocation_speed_component_loss_weight,
        collocation_speed_uncomposed_loss_weight=
            args.collocation_speed_uncomposed_loss_weight,
        collocation_speed_consensus_identification_weight=
            args.collocation_speed_consensus_identification_weight,
        collocation_speed_sampling=args.collocation_speed_sampling,
        collocation_speed_consensus_negative_weight=
            args.collocation_speed_consensus_negative_weight,
        collocation_speed_consensus_head_component_weight=
            args.collocation_speed_consensus_head_component_weight,
        collocation_speed_repair_identification_weight=
            args.collocation_speed_repair_identification_weight,
        collocation_speed_repair_coordination=
            args.collocation_speed_repair_coordination,
        clearance_loss_weight=args.clearance_loss_weight,
        clearance_self_scale=args.clearance_self_scale,
        hopf_cole_weight=args.hopf_cole_weight,
        hopf_cole_n=args.hopf_cole_n,
        hopf_epsilon_start=args.hopf_epsilon_start,
        hopf_epsilon_end=args.hopf_epsilon_end,
        hopf_decay_epochs=args.hopf_decay_epochs,
        hopf_goal_radius=args.hopf_goal_radius,
        hopf_goal_weight=args.hopf_goal_weight,
        hopf_adaptive=args.hopf_adaptive,
        hopf_adaptive_speed_power=args.hopf_adaptive_speed_power,
        hopf_adaptive_sharpness_weight=
            args.hopf_adaptive_sharpness_weight,
        hopf_characteristic_fraction=args.hopf_characteristic_fraction,
        hopf_characteristic_step=args.hopf_characteristic_step,
        hopf_state_constraint_weight=args.hopf_state_constraint_weight,
        hopf_boundary_speed=args.hopf_boundary_speed,
        hopf_speed_mode=args.hopf_speed_mode,
        hopf_residual_form=args.hopf_residual_form,
        hopf_ritz_normalize=args.hopf_ritz_normalize,
        hopf_shellwise_normalize=args.hopf_shellwise_normalize,
        hopf_shell_bins=args.hopf_shell_bins,
        hopf_endpoint_measure=args.hopf_endpoint_measure,
        hopf_goal_shell_measure=args.hopf_goal_shell_measure,
        hopf_shell_radius_min=args.hopf_shell_radius_min,
        hopf_shell_radius_max=args.hopf_shell_radius_max,
        viscosity_weight=args.viscosity_weight,
        viscosity_n=args.viscosity_n,
        viscosity_dirs=args.viscosity_dirs,
        viscosity_action_mode=args.viscosity_action_mode,
        viscosity_softmin_temperature=args.viscosity_softmin_temperature,
        viscosity_epsilon=args.viscosity_epsilon,
        viscosity_rollout_steps=args.viscosity_rollout_steps,
        viscosity_intrinsic_budget=args.viscosity_intrinsic_budget,
        viscosity_step=args.viscosity_step,
        viscosity_step_max=args.viscosity_step_max,
        viscosity_start_epoch=args.viscosity_start_epoch,
        viscosity_include_descent=not args.viscosity_no_self_descent,
        viscosity_include_speed_ascent=args.viscosity_speed_ascent,
        viscosity_causal_target=args.viscosity_causal_target,
        viscosity_causal_order=args.viscosity_causal_order,
        viscosity_candidate_routing=args.viscosity_candidate_routing,
        viscosity_hard_domain=args.viscosity_hard_domain,
        viscosity_target_lag=args.viscosity_target_lag,
        viscosity_update_mode=args.viscosity_update_mode,
        viscosity_collocation_pairs=args.viscosity_collocation_pairs,
        viscosity_global_cloud=args.viscosity_global_cloud,
        viscosity_local_boundary_fraction=
            args.viscosity_local_boundary_fraction,
        viscosity_boundary_min_radius=args.viscosity_boundary_min_radius,
        viscosity_boundary_max_radius=args.viscosity_boundary_max_radius,
        fermat_path_weight=args.fermat_path_weight,
        fermat_loss_transform=args.fermat_loss_transform,
        train_speed_only=args.train_speed_only,
        train_speed_head_only=args.train_speed_head_only,
        train_speed_constraint_heads_only=
            args.train_speed_constraint_heads_only,
        train_speed_consensus_only=args.train_speed_consensus_only,
        train_speed_consensus_magnitude_only=
            args.train_speed_consensus_magnitude_only,
        train_speed_consensus_specificity_magnitude_only=
            args.train_speed_consensus_specificity_magnitude_only,
        train_speed_repair_only=args.train_speed_repair_only,
        train_clearance_only=args.train_clearance_only,
        train_chord_profile_only=args.train_chord_profile_only,
        train_endpoint_barrier_only=args.train_endpoint_barrier_only,
        train_viscosity_excess_only=
            args.train_viscosity_excess_only,
        train_viscosity_input_tail_cols=
            args.train_viscosity_input_tail_cols,
        train_value_input_tail_cols=
            args.train_value_input_tail_cols,
        train_lifted_phase_speed_only=
            args.train_lifted_phase_speed_only,
        train_last_pair_residual_only=
            args.train_last_pair_residual_only,
        train_partial_shared_embedding_only=
            args.train_partial_shared_embedding_only,
        freeze_base_for_cross_residual=args.freeze_base_for_cross_residual,
        train_local_residual_only=args.train_local_residual_only)

    model.seed = args.seed
    model.cross_start_epoch = args.cross_start_epoch
    model.no_lip_norm = args.no_lip_norm
    model.Params['Training']['Batch Size'] = args.batch_size
    model.Params['Training']['Max Batches Per Epoch'] = args.max_batches
    model.Params['Training']['Number of Epochs'] = args.epochs
    model.Params['Training']['Loss Ratio Threshold'] = args.loss_ratio
    model.Params['Training']['Disable Rollback'] = args.no_rollback
    model.Params['Training']['Weight Decay Override'] = args.weight_decay
    # The initial optimizer reads "Learning Rate".  "Base LR" is consulted
    # only when rebuilding the optimizer for a later curriculum phase, so the
    # CLI value must populate both fields.
    model.Params['Training']['Learning Rate'] = args.lr
    model.Params['Training']['Base LR'] = args.lr
    if args.cross_lr is not None:
        model.Params['Training']['Cross LR'] = args.cross_lr
    model.Params['Training']['Freeze Encoders On Cross'] = args.freeze_encoders_on_cross
    model.Params['Training']['Speed Transform'] = args.speed_transform
    model.Params['Training']['Travel Time Weight'] = args.tt_weight
    model.Params['Training']['Dist Condition Only'] = args.dist_condition_only
    model.Params['Training']['Pair Condition Only'] = args.pair_condition_only

    # Path-loss wiring args (consumed inside train() once self.function exists).
    # Default OFF: when --use_path_loss is absent, this is None and behavior is
    # byte-for-byte identical to the prior pipeline.
    if args.use_path_loss:
        model._path_loss_args = dict(
            scene=args.scene,
            data_dir=args.data_dir,
            safe_index_dir=args.safe_index_dir,
            device=args.device,
            cup_tilt_max=args.cup_tilt_max,
            pour_tilt_max=args.pour_tilt_max,
            cup_tilt_soft=args.cup_tilt_soft,
            pour_tilt_soft=args.pour_tilt_soft,
            steps=args.rollout_steps,
            step_size=args.rollout_step_size,
            rollout_n=args.rollout_n,
            margin=args.path_margin,
            self_collision_margin=args.path_self_margin,
            rank_margin=args.rank_margin,
            path_eik_weight=args.path_eik_weight,
            rank_weight=args.rank_weight,
            start_epoch=args.path_eik_warmup_epochs,
            rank_start_epoch=args.rank_warmup_epochs,
            pair_types=args.path_loss_pair_types,
            pair_pool=args.path_loss_pair_pool,
            pair_pool_n=args.path_loss_pair_pool_n,
            pair_pool_seed=args.path_loss_pair_pool_seed,
            handover_accept=args.path_handover_accept,
            corner_overlap_deg=args.path_corner_overlap_deg,
            handover_mono_tol=args.path_handover_mono_tol,
            handover_weight=args.path_handover_weight,
            collision_only=args.path_collision_only,
        )
    else:
        model._path_loss_args = None

    if args.use_segment_barrier:
        model._segment_barrier_args = dict(
            scene=args.scene,
            data_dir=args.data_dir,
            device=args.device,
            cup_tilt_max=args.cup_tilt_max,
            pour_tilt_max=args.pour_tilt_max,
            cup_tilt_soft=args.cup_tilt_soft,
            pour_tilt_soft=args.pour_tilt_soft,
            anchor_n=args.segment_barrier_anchor_n,
            n_dirs=args.segment_barrier_dirs,
            step_size=args.segment_barrier_step_size,
            margin=args.segment_barrier_margin,
            self_collision_margin=args.segment_barrier_self_margin,
            rank_margin=args.segment_barrier_rank_margin,
            weight=args.segment_barrier_weight,
            start_epoch=args.segment_barrier_start_epoch,
            pair_types=args.segment_barrier_pair_types,
            anchor_mode=args.segment_barrier_anchor_mode,
            handover_accept=args.segment_barrier_handover_accept,
            corner_overlap_deg=args.segment_barrier_corner_overlap_deg,
            check_midpoint=not args.segment_barrier_no_midpoint,
            collision_only=args.segment_barrier_collision_only,
            safe_pool=args.segment_barrier_safe_pool,
            safe_index_dir=args.segment_barrier_safe_index_dir,
            safe_pool_n=args.segment_barrier_pool_n,
            safe_pool_seed=args.segment_barrier_pool_seed,
        )
    else:
        model._segment_barrier_args = None

    if args.use_line_preserve:
        model._line_preserve_args = dict(
            scene=args.scene,
            data_dir=args.data_dir,
            safe_index_dir=args.safe_index_dir,
            device=args.device,
            cup_tilt_max=args.cup_tilt_max,
            pour_tilt_max=args.pour_tilt_max,
            cup_tilt_soft=args.cup_tilt_soft,
            pour_tilt_soft=args.pour_tilt_soft,
            sample_n=args.line_preserve_n,
            checks=args.line_preserve_checks,
            margin=args.line_preserve_margin,
            self_collision_margin=args.line_preserve_self_margin,
            align_weight=args.line_preserve_align_weight,
            mono_weight=args.line_preserve_mono_weight,
            mono_margin=args.line_preserve_mono_margin,
            step_frac=args.line_preserve_step_frac,
            start_epoch=args.line_preserve_start_epoch,
            pair_types=args.line_preserve_pair_types,
            pair_pool=args.line_preserve_pair_pool,
            pair_pool_n=args.line_preserve_pair_pool_n,
            pair_pool_seed=args.line_preserve_pair_pool_seed,
            handover_accept=args.line_preserve_handover_accept,
            corner_overlap_deg=args.line_preserve_corner_overlap_deg,
            safe_pool=args.line_preserve_safe_pool,
            safe_pool_n=args.line_preserve_pool_n,
            safe_pool_seed=args.line_preserve_pool_seed,
            local_step_min=args.line_preserve_local_step_min,
            local_step_max=args.line_preserve_local_step_max,
        )
    else:
        model._line_preserve_args = None

    model.train()
