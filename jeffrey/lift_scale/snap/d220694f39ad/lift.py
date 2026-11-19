"""The travel-time lift as an input coordinate of ntrl-demo's network, for any scene with a differentiable signed clearance.

    phi(q) = (q, h(q))      h(q) = gain * G(d(q)) / E      G(d) = int_d^margin sqrt(1 / S(d')^2 - 1) dd'   (cells; 0 where d >= margin)

d is the scene's signed clearance in cells (the function the speed labels are computed from), S the speed the metric trainer fits
(ALPHA s^2 (2 - s)^2 + 1 - ALPHA, s = clip(d / margin, offset / margin, 1)). Below the offset the speed is the floor and G continues
with the floor slope down to d = -depth (the depth the training pairs reach, offset by default), constant below. Along a path that
leaves the band straight up the clearance gradient, the lifted arc length is ds sqrt(1 + G'^2 nu^2) with nu = |grad_q d| / E; for a
point robot nu = 1 and this is ds / S: coordinate speed 1 at every depth. In the SE(2) C-space of these scenes nu is between 1 and
about 1.5 (the heading term), so the coordinate speed along the normal is between 1 and nu instead of between 0.012 and 1.

The network is ntrl-demo's (jeffrey/embed_warp/features.EmbedNN, which is the package's NN when no octave is requested) with one more
column in its Fourier matrix for h: proj = 2 pi (q B^T + h b_h^T). B is drawn from the global RNG exactly as ntrl-demo draws it and b_h
from a private generator, so the network's own draws are those of the plain run of the same seed. Every loss term is still
differentiated with respect to q: h enters the graph as h(q0) + G'(d) grad_q d . (q - q0) (value and first derivative exact; the loss
never needs second derivatives of the input map with respect to q).

kinds:  none   no extra column: ntrl-demo
        time   the travel-time lift above
        clear  the same range and input length, but linear in the clearance (control: input length without the time metric)
        dead   the extra column with h = 0 (control: the plumbing alone)
        vec    the vector lift: three extra columns for h(q) n(q), n the unit clearance gradient in C-space. Along a wall normal it
               is the travel-time lift (n is constant there); across a wall the two faces get opposite vectors, 2 G apart in the
               input (1.4 cells at the offset, 4.3 at the surface) where the scalar lift gives them the same height. The chain rule
               uses the full Jacobian of h n with respect to q (second derivatives of the clearance by autograd).
"""
import math
import os
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = next((q for q in HERE.parents if (q / 'baselines/ntrl-demo').is_dir()), Path(os.environ.get('NTFIELD_ROOT', '/home/jeffrey/bellman-ntfield')))
sys.path.insert(0, str(HERE))
import scenes as SN  # noqa: E402  (appends embed_warp, puts maze_scale first)
import features as FE  # noqa: E402  (embed_warp/features.py)
from maze import MARGIN_CELLS, OFFSET_CELLS, ALPHA  # noqa: E402


def train_speed_np(d):
    s = np.clip(d / MARGIN_CELLS, OFFSET_CELLS / MARGIN_CELLS, 1.0)
    return ALPHA * s * s * (2 - s) * (2 - s) + 1 - ALPHA


class Profile:
    """G(d) and G'(d) (cells) on tensors of clearances in cells."""

    def __init__(self, kind='time', depth=OFFSET_CELLS, nodes=8193, device='cuda:0'):
        self.kind, self.depth = ('time' if kind == 'vec' else kind), float(depth)
        d = np.linspace(OFFSET_CELLS, MARGIN_CELLS, nodes)
        f = np.sqrt(np.clip(1 / train_speed_np(d) ** 2 - 1, 0, None))
        seg = 0.5 * (f[1:] + f[:-1]) * np.diff(d)
        G = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])                 # int_d^margin
        self.floor_slope = float(f[0])
        self.G_offset = float(G[0])
        self.G_max = self.G_offset + (OFFSET_CELLS + self.depth) * self.floor_slope
        self.table = torch.tensor(G, device=device, dtype=torch.float32)
        self.nodes = nodes
        self.band_length = float(np.trapz(1 / train_speed_np(d), d))            # int dd / S over the trained band (cells)

    def __call__(self, d):
        """(G, dG/dd) at clearances d (cells)."""
        if self.kind == 'dead':
            z = torch.zeros_like(d)
            return z, z
        if self.kind == 'clear':
            # linear in the clearance over [-depth, margin], same range as the time lift
            k = self.G_max / (MARGIN_CELLS + self.depth)
            dc = d.clamp(-self.depth, MARGIN_CELLS)
            inside = (d > -self.depth) & (d < MARGIN_CELLS)
            return k * (MARGIN_CELLS - dc), torch.where(inside, torch.full_like(d, -k), torch.zeros_like(d))
        dc = d.clamp(OFFSET_CELLS, MARGIN_CELLS)
        u = (dc - OFFSET_CELLS) / (MARGIN_CELLS - OFFSET_CELLS) * (self.nodes - 1)
        i = u.floor().long().clamp(0, self.nodes - 2)
        w = u - i
        G = (1 - w) * self.table[i] + w * self.table[i + 1]
        s = (dc / MARGIN_CELLS)
        S = ALPHA * s * s * (2 - s) * (2 - s) + 1 - ALPHA
        slope = -torch.sqrt((1 / S ** 2 - 1).clamp(min=0))
        below = d < OFFSET_CELLS
        deep = d <= -self.depth
        G = torch.where(below, self.G_offset + (OFFSET_CELLS - d.clamp(min=-self.depth)) * self.floor_slope, G)
        slope = torch.where(d >= MARGIN_CELLS, torch.zeros_like(slope), slope)
        slope = torch.where(below, torch.full_like(slope, -self.floor_slope), slope)
        slope = torch.where(deep, torch.zeros_like(slope), slope)
        return G, slope


def lift_class(mn):
    Base = FE.embed_class(mn)

    class LiftNN(Base):
        """cfg: features.EmbedNN's keys (octaves, rows, base_scale, whiten, gate) plus lift = dict(kind, gain, depth, freq, seed)."""

        def __init__(self, device, dim, B, cfg):
            super().__init__(device, dim, B, cfg)
            lc = dict(cfg.get('lift') or {})
            self.lift_kind = lc.get('kind', 'none')
            self.lift_gain = float(lc.get('gain', 1.0))
            self.scene = None
            if self.lift_kind != 'none':
                col = FE.draw_rows(int(lc.get('seed', 0)) + 4242, B.shape[0], 3 if self.lift_kind == 'vec' else 1, float(lc.get('freq', 1.0)))
                self.register_buffer('b_h', (col if self.lift_kind == 'vec' else col[:, 0]).to(device))
                self.profile = Profile(self.lift_kind, float(lc.get('depth', OFFSET_CELLS)), device=device)

        def attach(self, scene):
            self.scene = scene

        def height(self, x):
            """h(x) in normalized units (N,), with its first derivative with respect to x in the graph when x requires grad."""
            if self.lift_kind == 'dead':
                return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
            sc = self.scene
            if torch.is_grad_enabled() and x.requires_grad:
                xd = x.detach().clone().requires_grad_(True)
                with torch.enable_grad():
                    d = sc.signed_cells(xd)
                    g, = torch.autograd.grad(d.sum(), xd)
                d, g = d.detach(), torch.nan_to_num(g.detach())
                G, Gp = self.profile(d)
                h = G + Gp * ((x - x.detach()) * g).sum(1)
            else:
                with torch.no_grad():
                    h, _ = self.profile(sc.signed_cells(x))
            return self.lift_gain * h / sc.E

        def vector(self, x):
            """h(x) n(x) in normalized units (N, 3), with its Jacobian with respect to x in the graph when x requires grad."""
            sc = self.scene
            if torch.is_grad_enabled() and x.requires_grad:
                xd = x.detach().clone().requires_grad_(True)
                with torch.enable_grad():
                    d = sc.signed_cells(xd)
                    g, = torch.autograd.grad(d.sum(), xd, create_graph=True)
                    n = g / (torch.linalg.vector_norm(g, dim=1, keepdim=True) + 1e-9)
                    G0, Gp = self.profile(d.detach())
                    v = (G0 + Gp * (d - d.detach()))[:, None] * n
                    J = torch.stack([torch.autograd.grad(v[:, k].sum(), xd, retain_graph=True)[0] for k in range(3)], 1)   # (N, 3, 3)
                v, J = torch.nan_to_num(v.detach()), torch.nan_to_num(J.detach())
                h = v + torch.einsum('nkj,nj->nk', J, x - x.detach())
            else:
                xd = x.detach().clone().requires_grad_(True)
                with torch.enable_grad():
                    d = sc.signed_cells(xd)
                    g, = torch.autograd.grad(d.sum(), xd)
                G0, _ = self.profile(d.detach())
                h = torch.nan_to_num(G0[:, None] * g / (torch.linalg.vector_norm(g, dim=1, keepdim=True) + 1e-9)).detach()
            return self.lift_gain * h / sc.E

        def input_mapping(self, x):
            if self.lift_kind == 'none':
                return super().input_mapping(x)
            if self.K:
                proj = x @ (2.0 * math.pi * self.B_eff)
            else:
                proj = x @ (2. * np.pi * self.B)
            if self.lift_kind == 'vec':
                proj = proj + (2.0 * math.pi) * (self.vector(x) @ self.b_h.T)
            else:
                proj = proj + (2.0 * math.pi) * self.height(x)[:, None] * self.b_h[None, :]
            if self.K and self.embed_gate is not None and self.embed_gate.kind != 'const':
                m = self.row_scale(x, self.base_mapping(x) if getattr(self.embed_gate, 'inp', None) == 'u0' else None)
                return torch.cat([torch.sin(proj) * m, torch.cos(proj) * m], dim=-1)
            return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    return LiftNN


def build_network(mn, device, B, cfg, scene):
    """As the trainer constructs it: package init first (same RNG draws as ntrl-demo), then the scene handle."""
    net = lift_class(mn)(device, 3, B, cfg)
    net.apply(net.init_weights)
    net.to(device)
    gate_cfg = cfg.get('gate', {}) or {}
    gate = FE.make_gate(gate_cfg.get('kind', 'none'), net.K, gate_cfg, gate_cfg.get('lo'), gate_cfg.get('hi'), gate_cfg.get('shape'))
    if gate is not None:
        gate.to(device)
        net.set_gate(gate)
    net.attach(scene)
    return net


def load_field(checkpoint, scene, device='cuda:0'):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    mn = FE.network_module()
    net = build_network(mn, device, payload['B_state_dict'], payload['embed_config'], scene)
    net.load_state_dict(payload['model_state_dict'], strict=True)
    net.to(torch.device(device)).float().eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


@torch.no_grad()
def segment_blocked(scene, pairs, spacing_cells=0.1, points=None, depth=OFFSET_CELLS, chunk=1_000_000):
    """Does the straight C-space segment between the two configurations of each pair (N, 6) cross an obstacle? True if the clearance
    at any sample of the segment is below -depth cells (deeper than any training endpoint, so the skin an endpoint may sit in does not
    count). Samples: `points` equally spaced interior points (the page's check: 48), or one every spacing_cells cells of xy length."""
    a, b = pairs[:, :3], pairs[:, 3:]
    n = len(pairs)
    if points is not None:
        k = torch.full((n,), int(points), device=pairs.device, dtype=torch.long)
    else:
        k = torch.ceil(torch.linalg.vector_norm(a[:, :2] - b[:, :2], dim=1) * scene.E / spacing_cells).long().clamp(min=2)
    out = torch.zeros(n, dtype=torch.bool, device=pairs.device)
    order = torch.argsort(k)
    i = 0
    while i < n:
        budget, j = 0, i
        while j < n and budget + int(k[order[j]]) <= chunk:
            budget += int(k[order[j]])
            j += 1
        j = max(j, i + 1)
        idx = order[i:j]
        kk = k[idx]
        rep = torch.repeat_interleave(torch.arange(len(idx), device=pairs.device), kk)
        first = torch.cumsum(kk, 0) - kk
        t = (torch.arange(int(kk.sum()), device=pairs.device) - first[rep] + 1).to(pairs.dtype) / (kk[rep] + 1).to(pairs.dtype)
        q = a[idx][rep] + t[:, None] * (b[idx] - a[idx])[rep]
        d = scene.clearance(q)
        hit = torch.zeros(len(idx), dtype=torch.bool, device=pairs.device)
        hit.index_put_((rep[d < -depth],), torch.tensor(True, device=pairs.device))
        out[idx] = hit
        i = j
    return out
