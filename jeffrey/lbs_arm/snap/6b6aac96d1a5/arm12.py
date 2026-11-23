"""Several release arms in one scene (real_world_12dof: two UR5e + Robotiq 2F-140 side by side, 12 joints) as an LBS scene.

Configuration x = q / q_range per joint (arm.py's normalization), x[:, 6 a : 6 a + 6] = arm a (the scene file's order: left, right).
Every arm is arm.py's arm (same sphere fit, same self-collision mask, same joint limits) on its own base pose; the cuboids are the
scene's. model = build_model(): the single-arm geometry of an export run's robot_model.npz + the scene file's bases, cuboids and
grasp-point boxes.

    d_obs    min over both arms' spheres (without each arm's two shoulder spheres, as the label) and the cuboids        (metres)
    d_inter  min over all sphere pairs of two different arms (shoulder spheres included: the other arm can hit a base)
    d_self   min over both arms of the arm's own self distance
    valid    d_obs over all spheres >= 0.005, d_inter >= 0.005, d_self >= 0.002: the release scorer's rule for each arm, and the
             other arm treated as an obstacle (jeffrey/real_world/robot.py's rule; there the shoulder spheres are left out of the
             obstacle check, which is the same set of configurations here: they do not move and are 9.5 mm from the pedestals)
    d        min(d_obs, d_inter, 0.005 + K_SELF (d_self - 0.002))          speed = clip(d, 0.005, 0.05) / 0.05

Everything else (cells, E, contact, speed and lift profiles, sampling) is arm.py's with the dimension 6 A. The grasp points (N, A, 3):
a configuration is "in the box" when every arm's grasp point is in the scene's sampling box.
"""
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arm as A1  # noqa: E402
from arm import MARGIN_M, OFFSET_M, K_SELF, OBS_VALID_M, SELF_VALID_M, CELL_M, MARGIN_CELLS, OFFSET_CELLS  # noqa: E402,F401

INTER_VALID_M = 0.005
D = 12                                            # set by load_scene (6 x arms)
ARM_KEYS = ('q_fit', 'spheres_fit', 'ee_fit', 'link', 'mask', 'ignore', 'q_range', 'q_min', 'q_max')


def scene_boxes(doc):
    """experiments/cabinet/geometry.py scene_boxes: sampling box, cupboard outer box (the shelf_ panels), compartments."""
    box = np.array([doc['sampling'][k] for k in 'xyz'], np.float32)
    panels = [c for c in doc['cuboids'] if c['name'].startswith('shelf_')]
    lo = np.min([np.subtract(c['pose'][:3], np.divide(c['dims'], 2)) for c in panels], 0)
    hi = np.max([np.add(c['pose'][:3], np.divide(c['dims'], 2)) for c in panels], 0)
    return box, np.stack([lo, hi], 1).astype(np.float32), {c['name']: np.array([c[k] for k in 'xyz'], np.float32) for c in doc['compartments']}


def build_model(export, scene_file, out):
    """robot_model.npz of a multi-arm scene: the arm of `export` (a robot_model.npz of export_robot.py) + the scene file's own
    bases (x, y, z, qw, qx, qy, qz; rotation about z only), axis-aligned cuboids and boxes."""
    m = np.load(export, allow_pickle=False)
    doc = yaml.safe_load(Path(scene_file).read_text())
    for c in doc['cuboids']:
        assert np.allclose(c['pose'][3:], (1, 0, 0, 0)), c['name']
    bases = np.array([r['base'] for r in doc['robots']], np.float64)
    assert np.abs(bases[:, 4:6]).max() < 1e-9, 'bases may only be rotated about z'
    box, cupboard, comp = scene_boxes(doc)
    np.savez(out, **{k: m[k] for k in ARM_KEYS}, bases=bases, arm_names=np.array([r['name'] for r in doc['robots']]),
             cuboid_centre=np.array([c['pose'][:3] for c in doc['cuboids']], np.float32), cuboid_half=np.array([np.divide(c['dims'], 2) for c in doc['cuboids']], np.float32),
             cuboid_names=np.array([c['name'] for c in doc['cuboids']]), box=box, cupboard=cupboard, compartment_names=np.array(list(comp)), compartments=np.stack(list(comp.values())))
    return out


class ArmSet(A1.CabinetArm):
    kind = 'arm_set'

    def __init__(self, model, device='cuda:0', dtype=torch.float32, E=A1.E_DEFAULT):
        super().__init__(model, device, dtype, E)                        # one arm's fit, mask, limits; this scene's cuboids and boxes
        m = np.load(model, allow_pickle=False)
        bases = m['bases']
        self.A = len(bases)
        self.D = 6 * self.A
        yaw = 2 * np.arctan2(bases[:, 6], bases[:, 3])
        R = np.stack([np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]) for a in yaw])
        self.base_R = torch.tensor(R, device=device, dtype=dtype)
        self.base_t = torch.tensor(bases[:, :3], device=device, dtype=dtype)
        self.q_range1 = self.q_range
        self.q_range = self.q_range.repeat(self.A)
        self.lo, self.hi = self.lo.repeat(self.A), self.hi.repeat(self.A)
        self.clamp_mask = torch.ones(self.D, dtype=torch.bool, device=device)
        S = self.S
        off = torch.arange(self.A, device=device)[:, None] * S
        self.radius_all = self.radius.repeat(self.A)
        self.label_all = (self.label_idx[None] + off).reshape(-1)        # every arm's moving spheres, in the flat (A S) order
        self.pair_a_all = (self.pair_a[None] + off).reshape(-1)
        self.pair_b_all = (self.pair_b[None] + off).reshape(-1)
        self.pair_r_all = self.pair_r.repeat(self.A)
        self.inter_r = self.radius[:, None] + self.radius[None, :]
        self.inter_chunk = 1 << 15

    # --- kinematics ------------------------------------------------------------------------------------------------------------------
    def spheres(self, x):
        """World sphere centres (N, A, S, 3) metres and grasp points (N, A, 3) at normalized configurations x (N, 6 A)."""
        q = (x.to(self.dtype) * self.q_range).reshape(len(x) * self.A, 6)
        F = A1.link_frames(q, self.O, self.T0)
        Fs = F[:, self.frame]
        p = (Fs[..., :3, :3] @ self.local[None, :, :, None]).squeeze(-1) + Fs[..., :3, 3]
        ee = (F[:, 5, :3, :3] @ self.ee_local[None, :, None]).squeeze(-1) + F[:, 5, :3, 3]
        p = p.reshape(len(x), self.A, self.S, 3)
        ee = ee.reshape(len(x), self.A, 3)
        Rt = self.base_R.transpose(1, 2)
        return p @ Rt[None] + self.base_t[None, :, None, :], (ee[:, :, None, :] @ Rt[None]).squeeze(2) + self.base_t[None]

    def _obs_flat(self, p, idx=None):
        """(N, S') sphere-cuboid distances minimized over the cuboids; p (N, A, S, 3)."""
        P, r = p.reshape(len(p), self.A * self.S, 3), self.radius_all
        if idx is not None:
            P, r = P[:, idx], r[idx]
        delta = (P[:, :, None, :] - self.cc[None, None]).abs() - self.ch[None, None]
        return torch.sqrt((delta.clamp(min=0) ** 2).sum(-1) + 1e-18).amin(2) - r

    def _self_all(self, p):
        P = p.reshape(len(p), self.A * self.S, 3)
        return (torch.sqrt(((P[:, self.pair_a_all] - P[:, self.pair_b_all]) ** 2).sum(-1) + 1e-18) - self.pair_r_all).amin(1)

    def _inter(self, p):
        """(N,) min over all sphere pairs of different arms, in pieces of inter_chunk configurations (S x S distances each)."""
        out = []
        for i in range(0, len(p), self.inter_chunk):
            q = p[i:i + self.inter_chunk]
            best = None
            for a in range(self.A):
                for b in range(a + 1, self.A):
                    d = (torch.sqrt(((q[:, a][:, :, None, :] - q[:, b][:, None, :, :]) ** 2).sum(-1) + 1e-18) - self.inter_r).flatten(1).amin(1)
                    best = d if best is None else torch.minimum(best, d)
            out.append(best)
        return out[0] if len(out) == 1 else torch.cat(out)

    def parts(self, x):
        """dict of (N,) metres: obs (label: moving spheres against the cuboids), obs_only (all spheres against the cuboids), inter,
        obs_all = min(obs_only, inter) (everything the scorer treats as an obstacle), self, d, valid margin; ee (N, A, 3)."""
        p, ee = self.spheres(x)
        o = self._obs_flat(p)
        obs_only = o.amin(1)
        obs = o[:, self.label_all].amin(1)
        s = self._self_all(p)
        inter = self._inter(p) if self.A > 1 else torch.full_like(s, float('inf'))
        return dict(obs=obs, obs_only=obs_only, inter=inter, obs_all=torch.minimum(obs_only, inter), self=s,
                    d=torch.minimum(torch.minimum(obs, inter), OFFSET_M + K_SELF * (s - SELF_VALID_M)),
                    valid=torch.minimum(torch.minimum(obs_only - OBS_VALID_M, inter - INTER_VALID_M), s - SELF_VALID_M), ee=ee)

    def valid_margin(self, x):
        """(N,) metres, >= 0 iff valid (parts(x)['valid'] without the dictionary, for torch.compile)."""
        p, _ = self.spheres(x)
        v = torch.minimum(self._obs_flat(p).amin(1) - OBS_VALID_M, self._self_all(p) - SELF_VALID_M)
        return torch.minimum(v, self._inter(p) - INTER_VALID_M) if self.A > 1 else v

    def signed_cells(self, x):
        p, _ = self.spheres(x)
        d = torch.minimum(self._obs_flat(p, self.label_all).amin(1), OFFSET_M + K_SELF * (self._self_all(p) - SELF_VALID_M))
        if self.A > 1:
            d = torch.minimum(d, self._inter(p))
        return d / CELL_M

    # arm.py's chunked helpers with smaller pieces (twice the spheres, and the S x S arm-to-arm distances)
    def clearance(self, x, chunk=1 << 15):
        return super().clearance(x, chunk)

    def valid_cells(self, x, chunk=1 << 15):
        return super().valid_cells(x, chunk)

    def clearance_normal(self, x, chunk=1 << 14):
        return super().clearance_normal(x, chunk)

    def gradient(self, x, chunk=1 << 14):
        return super().gradient(x, chunk)

    def arm_parts(self, a, x6):
        """One arm alone in the scene (the others absent): valid margin (metres) and grasp point, x6 (N, 6) normalized."""
        q = x6.to(self.dtype) * self.q_range1
        F = A1.link_frames(q, self.O, self.T0)
        Fs = F[:, self.frame]
        p = ((Fs[..., :3, :3] @ self.local[None, :, :, None]).squeeze(-1) + Fs[..., :3, 3]) @ self.base_R[a].T + self.base_t[a]
        ee = ((F[:, 5, :3, :3] @ self.ee_local[None, :, None]).squeeze(-1) + F[:, 5, :3, 3]) @ self.base_R[a].T + self.base_t[a]
        delta = (p[:, :, None, :] - self.cc[None, None]).abs() - self.ch[None, None]
        obs = (torch.sqrt((delta.clamp(min=0) ** 2).sum(-1) + 1e-18).amin(2) - self.radius).amin(1)
        s = (torch.sqrt(((p[:, self.pair_a] - p[:, self.pair_b]) ** 2).sum(-1) + 1e-18) - self.pair_r).amin(1)
        return torch.minimum(obs - OBS_VALID_M, s - SELF_VALID_M), ee

    # --- sampling --------------------------------------------------------------------------------------------------------------------
    def uniform(self, count, gen):
        return self.lo + (self.hi - self.lo) * torch.rand(count, self.D, generator=gen, device=self.device, dtype=self.dtype)

    def in_box(self, x, chunk=1 << 18):
        out = []
        with torch.no_grad():
            for i in range(0, len(x), chunk):
                e = self.ee(x[i:i + chunk])
                out.append(((e >= self.box[:, 0]) & (e <= self.box[:, 1])).all(2).all(1))
        return torch.cat(out) if out else torch.zeros(0, dtype=torch.bool, device=x.device)

    def in_cupboard(self, x, chunk=1 << 18):
        """(N, A) bool: the arm's grasp point is inside the cupboard's outer box."""
        out = []
        with torch.no_grad():
            for i in range(0, len(x), chunk):
                e = self.ee(x[i:i + chunk])
                out.append(((e >= self.cupboard[:, 0]) & (e <= self.cupboard[:, 1])).all(2))
        return torch.cat(out)

    def spec(self):
        return dict(kind=self.kind, model=self.model_path, arms=self.A, E=self.E, cell_m=CELL_M, margin_m=MARGIN_M, offset_m=OFFSET_M, k_self=K_SELF, inter_valid_m=INTER_VALID_M,
                    spheres=int(self.S) * self.A, self_pairs=int(len(self.pair_a_all)), inter_pairs=int(self.S * self.S * self.A * (self.A - 1) // 2), cuboids=int(len(self.cc)),
                    fit_error_m=self.fit_error)

    def describe(self):
        return dict(E_cells=self.E, pitch=self.pitch, margin=self.margin, offset=self.offset, dim=self.D, lo=self.lo.tolist(), hi=self.hi.tolist(),
                    contact_cells=self.contact, variant=self.kind, scene=self.spec())

    # --- checks ----------------------------------------------------------------------------------------------------------------------
    def check(self, scene_file, n=200_000, seed=0):
        """Against jeffrey/real_world/robot.py (written separately, checked there against the URDF) in the same scene file, at n
        uniform configurations: the valid flag everywhere, and each distance where robot.py's is positive (inside a cuboid robot.py
        counts depth, the label code does not)."""
        import importlib.util
        spec = importlib.util.spec_from_file_location('real_world_robot', A1.ROOT / 'jeffrey/real_world/robot.py')
        RW = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(RW)
        ref = RW.Arms.from_scene(scene_file, self.device, torch.float64)
        gen = torch.Generator(self.device).manual_seed(seed)
        x = self.uniform(n, gen)
        out = dict(n=n, valid_agree=0, valid=0, obs=0.0, self=0.0, inter=0.0, ee=0.0)
        with torch.no_grad():
            for i in range(0, n, 20000):
                xi = x[i:i + 20000]
                mine = self.parts(xi)
                r = ref.parts((xi * self.q_range).double())
                v = mine['valid'] >= 0
                out['valid_agree'] += int((v.cpu() == r['valid'].cpu()).sum())
                out['valid'] += int(v.sum())
                ro, rs = r['obstacle'].amin(1), r['self'].amin(1)
                pos = ro > 0
                out['obs'] = max(out['obs'], float((mine['obs'].double() - ro)[pos].abs().max()))
                out['self'] = max(out['self'], float((mine['self'].double() - rs).abs().max()))
                if self.A > 1:
                    out['inter'] = max(out['inter'], float((mine['inter'].double() - r['inter']).abs().max()))
                out['ee'] = max(out['ee'], float((mine['ee'].double() - r['ee']).abs().max()))
        out.update(fit_error_m=self.fit_error, fit_error_ee_m=self.fit_error_ee)
        return out


def load_scene(prepared, device='cuda:0'):
    global D
    info = json.load(open(Path(prepared) / 'scene.json'))
    scene = ArmSet(Path(prepared) / 'robot_model.npz', device, E=info['E_cells'])
    D = scene.D
    return scene, info
