"""ntrl-demo's network on the travel-time lift (jeffrey/lift_scale/lift.py) with one more input that belongs to the PAIR, not to a point.

Why. With the lift the network sees phi(q) = (q, h(q)). Two configurations on opposite faces of a thin wall have nearly the same q and
the same height h, so their embeddings are nearly the same and no travel-time step can be represented between them. Here the two
ends of a pair get an extra coordinate c computed from the straight segment between them:

    kind 'bit'   psi = 1 if the straight C-space segment crosses an obstacle (clearance below -depth at a sample every `spacing` cells,
                 end points excluded), else 0                                                                           (Task 2)
    kind 'count' psi = norm(number of obstacles the segment enters) (the bit that keeps counting; norm raw with scale s: count / s)
    kind 'len'   psi = norm(e),  e = length of the lifted curve of the straight segment minus the segment's own length, in cells:
                 e = sum over samples of sqrt(|dq|^2 + dh^2) - |dq|.  0 for a clear segment outside every speed band, about
                 2 G_max = 7.2 cells for every wall the segment crosses                                                 (Task 1)
                 norm: tanh (e / s) | frac e / (e + s) | log log1p(e / s) | raw e / s        (s = `scale` cells; raw with s = E is box lengths)
                 what = 'length' uses the whole lifted length instead of the excess; what = 'updown' keeps only the part of the curve that
                 goes up and comes back down: sum |dh| - |h(end) - h(start)|, which is 0 for any segment along which h is monotone
                 (every clear segment that leaves a wall) and 2 (G_max - the higher end) across a wall

    sign 'anti'  c = -psi at the first configuration, +psi at the second: the two ends are 2 psi apart in the new coordinate
    sign 'same'  c = psi at both ends (the literal 'append the distance to the input of both points')

    place 'input'  one more column of the Fourier matrix:  proj = 2 pi (q B^T + h b_h^T + c b_c^T)   (b_c: N(0, 1) truncated, times `freq`)
          'rows'   `rows` of the 128 Fourier rows are given to c alone (their q and h columns are zero), frequencies freq 2^[-2 .. 2]
          'pre'    `copies` copies of amp c / 2 appended to the 256-wide embedding before the instance norm
          'post'   the same after the instance norm: with sign 'anti' and copies a multiple of 16 the metric gains exactly
                   (copies / 80) amp psi. Other counts are filled with zeros to whole groups of 16 (a group of k copies and 16 - k zeros
                   adds 0.2 (amp psi + ln(k / 16) / 10) once amp psi is well above 0.1)

The clearance along segments comes from a table (ClearTable: the scene's exact clearance on a grid of `per_cell` nodes per cell and
`nt` headings, trilinear), the same in training and planning; the network's own h and the speed labels use the exact clearance as before.
For kind 'len' the feature's first derivative with respect to both configurations is in the graph (linearized, as the lift does for h),
so grad_q T is the total derivative of the field the planner reads; for 'bit' it is zero almost everywhere.
With kind 'none' this is jeffrey/lift_scale's network.
"""
import math
import os
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = next((q for q in HERE.parents if (q / 'baselines/ntrl-demo').is_dir()), Path(os.environ.get('NTFIELD_ROOT', '/home/jeffrey/bellman-ntfield')))
if str(ROOT / 'jeffrey/lift_scale') not in sys.path:
    sys.path.insert(0, str(ROOT / 'jeffrey/lift_scale'))
import lift as LF  # noqa: E402  (jeffrey/lift_scale/lift.py)
import features as FE  # noqa: E402  (jeffrey/embed_warp/features.py)
from maze import OFFSET_CELLS  # noqa: E402

assert Path(LF.__file__).resolve() == ROOT / 'jeffrey/lift_scale/lift.py', 'lift module shadowed'


class ClearTable:
    """Signed clearance (cells) of the scene on a regular grid over the C-space box, trilinear, differentiable in q."""

    def __init__(self, scene, per_cell=16, nt=49):
        dev = scene.device
        self.nx = int(round(scene.E * per_cell)) + 1
        self.nt = int(nt)
        xs = torch.linspace(-0.5, 0.5, self.nx, device=dev)
        ts = torch.linspace(-scene.phi_max, scene.phi_max, self.nt, device=dev)
        tab = torch.empty(self.nx, self.nx, self.nt, device=dev)
        rows = max(1, 400_000 // (self.nx * self.nt))
        for i in range(0, self.nx, rows):
            X, Y, Tt = torch.meshgrid(xs[i:i + rows], xs, ts, indexing='ij')
            tab[i:i + rows] = scene.clearance(torch.stack([X, Y, Tt], -1).reshape(-1, 3)).reshape(X.shape)
        self.tab = tab.clamp(max=2.0).reshape(-1).contiguous()          # scenes report 1e4 where no obstacle is near; the lift ends at the margin
        self.lo, self.hi = scene.lo, scene.hi
        self.size = torch.tensor([self.nx - 1, self.nx - 1, self.nt - 1], device=dev, dtype=torch.float32)

    def __call__(self, q):
        c = (q - self.lo) / (self.hi - self.lo) * self.size
        c = torch.minimum(torch.maximum(c, torch.zeros_like(self.size)), self.size)
        k = torch.minimum(torch.floor(c.detach()), self.size - 1).long()
        f = c - k.to(c.dtype)
        base = (k[:, 0] * self.nx + k[:, 1]) * self.nt + k[:, 2]
        out = 0
        for dx in (0, 1):
            wx = f[:, 0] if dx else 1 - f[:, 0]
            for dy in (0, 1):
                wy = f[:, 1] if dy else 1 - f[:, 1]
                for dz in (0, 1):
                    wz = f[:, 2] if dz else 1 - f[:, 2]
                    out = out + wx * wy * wz * self.tab[base + (dx * self.nx + dy) * self.nt + dz]
        return out


class SpeedTable(ClearTable):
    """The speed a trained field predicts, on the ClearTable grid: log |grad_q T(q, g)| (the predicted slowness 1 / S; the eikonal term trains
    S |grad T| = 1 at both ends of every pair) averaged over `goals` random free partners g per node. Trilinear in the log."""

    def __init__(self, net, scene, per_cell=32, nt=49, goals=4, seed=0, chunk=150_000):
        dev = scene.device
        self.nx = int(round(scene.E * per_cell)) + 1
        self.nt = int(nt)
        xs = torch.linspace(-0.5, 0.5, self.nx, device=dev)
        ts = torch.linspace(-scene.phi_max, scene.phi_max, self.nt, device=dev)
        gen = torch.Generator(dev).manual_seed(seed)
        pool = scene.free(8192, gen, 0.1)
        tab = torch.empty(self.nx, self.nx, self.nt, device=dev)
        rows = max(1, chunk // (self.nx * self.nt))
        for i in range(0, self.nx, rows):
            X, Y, Tt = torch.meshgrid(xs[i:i + rows], xs, ts, indexing='ij')
            q = torch.stack([X, Y, Tt], -1).reshape(-1, 3)
            acc = torch.zeros(len(q), device=dev)
            for _ in range(goals):
                g = pool[torch.randint(len(pool), (len(q),), generator=gen, device=dev)]
                with torch.enable_grad():
                    tau, _, xp = net.out(torch.cat([q, g], 1))
                    gr, = torch.autograd.grad(tau.sum(), xp)
                acc += torch.log(torch.linalg.vector_norm(gr[:, :3], dim=1).clamp(min=1e-6))
            tab[i:i + rows] = (acc / goals).reshape(X.shape)
        self.tab = tab.reshape(-1).contiguous()
        self.lo, self.hi = scene.lo, scene.hi
        self.size = torch.tensor([self.nx - 1, self.nx - 1, self.nt - 1], device=dev, dtype=torch.float32)

    def slowness(self, q):
        return torch.exp(self(q))

    def step_time(self, a, b, samples=33):
        """Cost of the straight step a -> b (..., 3): its length x the mean (trapezoid rule) of the predicted slowness along it."""
        shape = a.shape[:-1]
        ts = torch.linspace(0, 1, samples, device=a.device)
        p = a[..., None, :] + ts[:, None] * (b - a)[..., None, :]
        s = self.slowness(p.reshape(-1, 3)).reshape(*shape, samples)
        mean = (0.5 * (s[..., 1:] + s[..., :-1])).mean(-1)
        return torch.linalg.vector_norm(b - a, dim=-1) * mean


def _chunks(k, budget):
    """Index groups (in order of the per-pair sample count k) whose total sample count stays below budget."""
    order = torch.argsort(k)
    ks = k[order]
    cs = torch.cumsum(ks, 0)
    i, n = 0, len(k)
    while i < n:
        base = int(cs[i - 1]) if i else 0
        j = int(torch.searchsorted(cs, torch.tensor(base + budget, device=k.device), right=True))
        j = min(max(j, i + 1), n)
        yield order[i:j]
        i = j


def segment_counts(pairs, E, spacing):
    return torch.ceil(torch.linalg.vector_norm(pairs[:, :3] - pairs[:, 3:], dim=1) * E / spacing).long().clamp(min=2)


@torch.no_grad()
def segment_min(table, pairs, E, spacing=0.1, budget=3_000_000):
    """Smallest clearance (cells) at the interior samples of the straight segment of each pair (N, 6)."""
    n = len(pairs)
    out = torch.full((n,), float('inf'), device=pairs.device)
    if not n:
        return out
    k = segment_counts(pairs, E, spacing)
    a, b = pairs[:, :3], pairs[:, 3:]
    for idx in _chunks(k, budget):
        kk = k[idx]
        rep = torch.repeat_interleave(torch.arange(len(idx), device=pairs.device), kk)
        first = torch.cumsum(kk, 0) - kk
        t = (torch.arange(int(kk.sum()), device=pairs.device) - first[rep] + 1).to(pairs.dtype) / (kk[rep] + 1).to(pairs.dtype)
        d = table(a[idx][rep] + t[:, None] * (b[idx] - a[idx])[rep])
        out[idx] = torch.full((len(idx),), float('inf'), device=pairs.device).scatter_reduce(0, rep, d, 'amin')
    return out


@torch.no_grad()
def segment_count(table, pairs, E, spacing=0.1, depth=OFFSET_CELLS, budget=3_000_000):
    """Number of obstacles the straight segment of each pair (N, 6) enters: runs of interior samples with clearance below -depth cells."""
    n = len(pairs)
    out = torch.zeros(n, device=pairs.device)
    if not n:
        return out
    k = segment_counts(pairs, E, spacing)
    a, b = pairs[:, :3], pairs[:, 3:]
    for idx in _chunks(k, budget):
        kk = k[idx]
        rep = torch.repeat_interleave(torch.arange(len(idx), device=pairs.device), kk)
        first = torch.cumsum(kk, 0) - kk
        j = torch.arange(int(kk.sum()), device=pairs.device) - first[rep]
        t = (j + 1).to(pairs.dtype) / (kk[rep] + 1).to(pairs.dtype)
        ins = table(a[idx][rep] + t[:, None] * (b[idx] - a[idx])[rep]) < -depth
        prev = torch.cat([ins.new_zeros(1), ins[:-1]]) & (j > 0)
        out[idx] = torch.zeros(len(idx), device=pairs.device).index_add(0, rep, (ins & ~prev).float())
    return out


def lifted_feature(table, profile, gain, E, pairs, fn, spacing=0.1, need_grad=False, what='excess', budget=3_000_000):
    """psi = fn(e) per pair (N,), e the lifted-curve excess (or length) of the straight segment in cells; with need_grad also d psi / d pairs (N, 6)."""
    n = len(pairs)
    dev = pairs.device
    psi = torch.zeros(n, device=dev)
    grad = torch.zeros(n, 6, device=dev) if need_grad else None
    if not n:
        return psi, grad
    pairs = pairs.detach()
    k = segment_counts(pairs, E, spacing)
    for idx in _chunks(k, budget // (3 if need_grad else 1)):
        kk = k[idx]
        cnt = kk + 1                                               # samples 0 .. k (both ends)
        m = len(idx)
        rep = torch.repeat_interleave(torch.arange(m, device=dev), cnt)
        first = torch.cumsum(cnt, 0) - cnt
        j = torch.arange(int(cnt.sum()), device=dev) - first[rep]
        t = j.to(pairs.dtype) / kk[rep].to(pairs.dtype)
        iv = torch.nonzero(j < kk[rep]).squeeze(1)                 # sample that starts an interval
        with torch.set_grad_enabled(need_grad):
            pr = pairs[idx].clone().requires_grad_(True) if need_grad else pairs[idx]
            a, b = pr[:, :3], pr[:, 3:]
            p = a[rep] + t[:, None] * (b - a)[rep]
            G, _ = profile(table(p))
            h = gain * G / E
            dp2 = ((p[iv + 1] - p[iv]) ** 2).sum(1)
            dh = h[iv + 1] - h[iv]
            root = torch.sqrt(dp2 + dh * dh + 1e-18)
            if what == 'updown':
                # total variation of h along the segment minus |h(end) - h(start)|: 0 when h is monotone (a segment that leaves a wall), twice the
                # barrier above the higher end when the segment climbs over something
                seg = torch.sqrt(dh * dh + 1e-18)
                last = first + kk
                e = (torch.zeros(m, device=dev, dtype=seg.dtype).index_add(0, rep[iv], seg) - torch.sqrt((h[last] - h[first]) ** 2 + 1e-18)).clamp(min=0) * E
            else:
                seg = root if what == 'length' else dh * dh / (root + torch.sqrt(dp2 + 1e-18))
                e = torch.zeros(m, device=dev, dtype=seg.dtype).index_add(0, rep[iv], seg) * E
            val = fn(e)
            if need_grad:
                g, = torch.autograd.grad(val.sum(), pr)
                grad[idx] = torch.nan_to_num(g.detach())
        psi[idx] = val.detach()
    return psi, grad


def norm_fn(kind, scale):
    s = float(scale)
    if kind == 'tanh':
        return lambda e: torch.tanh(e / s)
    if kind == 'frac':
        return lambda e: e / (e + s)
    if kind == 'log':
        return lambda e: torch.log1p(e / s)
    if kind == 'raw':
        return lambda e: e / s
    raise ValueError(kind)


def pair_class(mn):
    Base = LF.lift_class(mn)

    class PairNN(Base):
        """cfg: lift.LiftNN's keys plus pair = dict(kind, sign, norm, scale, what, place, copies, amp, freq, rows, inside, spacing, depth, grad, per_cell, nt, seed)."""

        def __init__(self, device, dim, B, cfg):
            super().__init__(device, dim, B, cfg)
            pc = dict(cfg.get('pair') or {})
            self.pk = pc.get('kind', 'none')
            self.p_sign = pc.get('sign', 'anti')
            self.p_norm, self.p_scale, self.p_what = pc.get('norm', 'tanh'), float(pc.get('scale', 2.0)), pc.get('what', 'excess')
            self.p_place = pc.get('place', 'input')
            self.p_copies, self.p_amp = int(pc.get('copies', 16)), float(pc.get('amp', 1.0))
            self.p_freq, self.p_rows = float(pc.get('freq', 1.0)), int(pc.get('rows', 16))
            self.p_inside = pc.get('inside', 'seg')
            self.p_spacing, self.p_depth = float(pc.get('spacing', 0.1)), float(pc.get('depth', OFFSET_CELLS))
            self.p_grad = bool(pc.get('grad', True))
            self.p_table = (int(pc.get('per_cell', 16)), int(pc.get('nt', 49)))
            self.table = None
            assert self.pk in ('none', 'bit', 'count', 'len') and self.p_sign in ('anti', 'same') and self.p_place in ('input', 'rows', 'pre', 'post')
            if self.pk != 'none' and self.p_place in ('input', 'rows'):
                M = B.shape[0]
                seed = int(pc.get('seed', 0)) + 777
                if self.p_place == 'input':
                    col = FE.draw_rows(seed, M, 1, self.p_freq)[:, 0]
                else:
                    r = self.p_rows
                    col = torch.zeros(M)
                    col[M - r:] = self.p_freq * 2.0 ** torch.linspace(-2, 2, r)
                    self.B[:, M - r:] = 0
                    if hasattr(self, 'B_eff'):
                        self.B_eff[:, M - r:] = 0
                    if hasattr(self, 'b_h'):
                        self.b_h[M - r:] = 0
                self.register_buffer('b_c', col.to(device))
            if self.pk in ('len', 'count'):
                self.p_fn = norm_fn(self.p_norm, self.p_scale)
            if self.pk == 'len':
                if not hasattr(self, 'profile'):
                    self.profile = LF.Profile('time', float((cfg.get('lift') or {}).get('depth', OFFSET_CELLS)), device=device)

        def attach(self, scene):
            self.scene = scene
            if self.pk != 'none':
                self.table = ClearTable(scene, *self.p_table)

        def pair_feature(self, coords):
            """psi (N,) for pairs (N, 6); for kind 'len' with its first derivative with respect to coords in the graph when coords requires grad."""
            sc = self.scene
            flat = coords.detach()
            if self.pk == 'bit':
                with torch.no_grad():
                    blk = segment_min(self.table, flat, sc.E, self.p_spacing) < -self.p_depth
                    if self.p_inside == 'zero':
                        blk = blk & (sc.clearance(flat[:, :3]) >= 0) & (sc.clearance(flat[:, 3:]) >= 0)
                return blk.to(coords.dtype)
            if self.pk == 'count':
                return self.p_fn(segment_count(self.table, flat, sc.E, self.p_spacing, self.p_depth)).to(coords.dtype)
            need = bool(self.p_grad and torch.is_grad_enabled() and coords.requires_grad)
            gain = self.lift_gain if self.lift_kind != 'none' else 1.0
            psi, g = lifted_feature(self.table, self.profile, gain, sc.E, flat, self.p_fn, self.p_spacing, need, self.p_what)
            if need:
                psi = psi + (g * (coords - flat)).sum(1)
            return psi

        def pair_mapping(self, x, c):
            """LiftNN.input_mapping with the pair column (c is None for places 'pre' / 'post')."""
            if self.K:
                proj = x @ (2.0 * math.pi * self.B_eff)
            else:
                proj = x @ (2. * np.pi * self.B)
            if self.lift_kind == 'vec':
                proj = proj + (2.0 * math.pi) * (self.vector(x) @ self.b_h.T)
            elif self.lift_kind != 'none':
                proj = proj + (2.0 * math.pi) * self.height(x)[:, None] * self.b_h[None, :]
            if c is not None:
                proj = proj + (2.0 * math.pi) * c[:, None] * self.b_c[None, :]
            if self.K and self.embed_gate is not None and self.embed_gate.kind != 'const':
                m = self.row_scale(x, self.base_mapping(x) if getattr(self.embed_gate, 'inp', None) == 'u0' else None)
                return torch.cat([torch.sin(proj) * m, torch.cos(proj) * m], dim=-1)
            return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

        def out(self, coords):
            if self.pk == 'none':
                return super().out(coords)
            # ntrl-demo's NN.out (baselines/ntrl-demo/models/metric/model_network_metric.py) with the pair coordinate
            coords = coords.clone().detach().requires_grad_(True)
            size = coords.shape[0]
            x0 = coords[:, :self.dim]
            x1 = coords[:, self.dim:]
            psi = self.pair_feature(coords)
            c = torch.cat([-psi, psi]) if self.p_sign == 'anti' else torch.cat([psi, psi])
            x = torch.vstack((x0, x1))
            x = self.pair_mapping(x, c if self.p_place in ('input', 'rows') else None)

            w = self.lip_norm(self.pe_gate[0].weight)
            u = torch.sin(x @ w.T + self.pe_gate[0].bias)
            w = self.lip_norm(self.pe_gate[1].weight)
            v = torch.sin(x @ w.T + self.pe_gate[1].bias)
            for ii in range(0, self.nl1):
                x_tmp = x
                w = self.lip_norm(self.encoder[3 * ii + 1].weight)
                y = x @ w.T + self.encoder[3 * ii + 1].bias
                x = u * torch.sin(y) + v * (1 - torch.sin(y))
                w = self.lip_norm(self.encoder[3 * ii + 2].weight)
                y = x @ w.T + self.encoder[3 * ii + 2].bias
                x = u * torch.sin(y) + v * (1 - torch.sin(y))
                w = self.lip_norm(self.encoder[3 * ii + 3].weight)
                y = x @ w.T + self.encoder[3 * ii + 3].bias
                weight = torch.sigmoid(0.1 * self.gate[ii].weight)
                x = (1 - weight) * x_tmp + (weight) * torch.sin(y)
            w = self.lip_norm(self.encoder[-1].weight)
            y = x @ w.T + self.encoder[-1].bias
            block = None
            if self.p_place in ('pre', 'post'):
                # `copies` copies of amp c / 2, filled with zeros to whole groups of 16 (the metric reads the embedding in groups of 16)
                block = (0.5 * self.p_amp) * c[:, None].expand(-1, self.p_copies)
                pad = (-self.p_copies) % 16
                if pad:
                    block = torch.cat([block, block.new_zeros(block.shape[0], pad)], 1)
            if self.p_place == 'pre':
                y = torch.cat([y, block], 1)
            y = self.encoder_norm(y)
            if self.p_place == 'post':
                y = torch.cat([y, block], 1)
            x0 = y[:size, ...]
            x1 = y[size:, ...]
            x = torch.sqrt((x0 - x1) ** 2 + 1e-6)
            x = x.view(x.shape[0], -1, 16)
            x = (torch.logsumexp(10 * x, 2) - np.log(16)) / 10
            x = 0.2 * (torch.sum(x, dim=1, keepdim=True))
            return x, w, coords

    return PairNN


def build_network(mn, device, B, cfg, scene):
    """As lift.build_network: package init first (the global RNG draws of ntrl-demo), then the gate and the scene handle."""
    net = pair_class(mn)(device, 3, B, cfg)
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
