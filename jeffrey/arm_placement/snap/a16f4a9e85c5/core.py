"""LBS in a joint space of any dimension: the pieces of jeffrey/lbs (lift.py, liftfast.py, bell.py, traverse.py) with the dimension taken
from the configurations and the scene's own clearance in place of the 3-D clearance table. Formulas are unchanged.

    Profile        G(d), G'(d): the travel-time height of a clearance (jeffrey/lbs/lift.py Profile, kind 'time')
    lift_class     ntrl-demo's NN with one more Fourier column for h(x) = gain G(d(x)) / E (lift.py LiftNN without octaves); the Fourier
                   matrix of the joints can be scaled (one factor, or one per joint)
    Compiled       no-grad evaluator on rows [x, h] (liftfast.py Compiled)
    mppi           the traversal planner (liftfast.mppi; every coordinate is clamped to the joint box)
    collisions     exact clearance at every waypoint and segment midpoint (traverse.collisions; contact at scene.contact)
    step_time      travel time of a straight step from the true speed along it, and whether it stays free (bell.step_time)
    bellman_term   the free-step Bellman backup (bell.bellman_term), on a subset of the batch if asked
    plan / score   ntrl-demo's MPPI on queries and the release scorer's rule with this scene's own geometry (validation during training)

A scene provides: device, E, pitch, contact (cells), lo, hi, signed_cells(x), clearance(x), train_speed(d), free(count, gen, min_clear).
"""
import math
import os
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = next((q for q in HERE.parents if (q / 'baselines/ntrl-demo').is_dir()), Path(os.environ.get('NTFIELD_ROOT', '/home/jeffrey/bellman-ntfield')))
if str(ROOT / 'jeffrey/maze_scale') not in sys.path:
    sys.path.append(str(ROOT / 'jeffrey/maze_scale'))
sys.path.insert(0, str(HERE))
from maze import MARGIN_CELLS, OFFSET_CELLS, ALPHA, REF_N, T_WALL  # noqa: E402
import fastnet as F  # noqa: E402  (verbatim copy of jeffrey/lbs/fastnet.py)

BASE = dict(sample_num=50, horizon=5, step=0.015, temperature=50.0, tol=0.01)


def planner_settings(scene, **override):
    """ntrl-demo's MPPI settings scaled with the cell as in the maze studies (maze_scale/plan.py settings: step and tolerance x s,
    temperature / s, s = pitch / pitch of the reference maze): a step is 0.0615 cells in every scene."""
    s = scene.pitch * (REF_N + T_WALL)
    out = dict(BASE, step=BASE['step'] * s, temperature=BASE['temperature'] / s, tol=BASE['tol'] * s)
    out.update(override)
    return out


def train_speed_np(d):
    s = np.clip(d / MARGIN_CELLS, OFFSET_CELLS / MARGIN_CELLS, 1.0)
    return ALPHA * s * s * (2 - s) * (2 - s) + 1 - ALPHA


def train_speed(d):
    s = (d / MARGIN_CELLS).clamp(OFFSET_CELLS / MARGIN_CELLS, 1.0)
    return ALPHA * s * s * (2 - s) * (2 - s) + 1 - ALPHA


# --- the lift ------------------------------------------------------------------------------------------------------------------------
class Profile:
    """G(d) = int_d^margin sqrt(1 / S^2 - 1) dd' (cells; 0 above the margin) and G'(d). Below the offset the speed is the floor and G
    continues with the floor slope down to d = -depth, constant below (jeffrey/lbs/lift.py Profile)."""

    def __init__(self, depth=OFFSET_CELLS, nodes=8193, device='cuda:0'):
        self.depth = float(depth)
        d = np.linspace(OFFSET_CELLS, MARGIN_CELLS, nodes)
        f = np.sqrt(np.clip(1 / train_speed_np(d) ** 2 - 1, 0, None))
        seg = 0.5 * (f[1:] + f[:-1]) * np.diff(d)
        G = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
        self.floor_slope = float(f[0])
        self.G_offset = float(G[0])
        self.G_max = self.G_offset + (OFFSET_CELLS + self.depth) * self.floor_slope
        self.table = torch.tensor(G, device=device, dtype=torch.float32)
        self.nodes = nodes

    def __call__(self, d):
        dc = d.clamp(OFFSET_CELLS, MARGIN_CELLS)
        u = (dc - OFFSET_CELLS) / (MARGIN_CELLS - OFFSET_CELLS) * (self.nodes - 1)
        i = u.floor().long().clamp(0, self.nodes - 2)
        w = u - i
        G = (1 - w) * self.table[i] + w * self.table[i + 1]
        s = dc / MARGIN_CELLS
        S = ALPHA * s * s * (2 - s) * (2 - s) + 1 - ALPHA
        slope = -torch.sqrt((1 / S ** 2 - 1).clamp(min=0))
        below = d < OFFSET_CELLS
        deep = d <= -self.depth
        G = torch.where(below, self.G_offset + (OFFSET_CELLS - d.clamp(min=-self.depth)) * self.floor_slope, G)
        slope = torch.where(d >= MARGIN_CELLS, torch.zeros_like(slope), slope)
        slope = torch.where(below, torch.full_like(slope, -self.floor_slope), slope)
        slope = torch.where(deep, torch.zeros_like(slope), slope)
        return G, slope


def draw_rows(seed, rows, dim=1, scale=1.0):
    """embed_warp/features.py draw_rows: N(0, 1) truncated at +-2 times scale, from a private generator."""
    gen = torch.Generator().manual_seed(int(seed))
    out = torch.randn(rows, dim, generator=gen)
    for _ in range(20):
        bad = out.abs() > 2
        if not bool(bad.any()):
            break
        out[bad] = torch.randn(int(bad.sum()), generator=gen)
    return out.clamp(-2, 2) * scale


def lift_class(mn):
    class LiftNN(mn.NN):
        """cfg: lift = dict(kind 'none' | 'time', gain, depth, seed), freq (list, one factor per input coordinate, already applied to B)."""

        def __init__(self, device, dim, B, cfg):
            super().__init__(device, dim, B)
            self.cfg = cfg
            lc = dict(cfg.get('lift') or {})
            self.lift_kind = lc.get('kind', 'none')
            self.lift_gain = float(lc.get('gain', 1.0))
            self.scene = None
            if self.lift_kind != 'none':
                self.register_buffer('b_h', draw_rows(int(lc.get('seed', 0)) + 4242, B.shape[0], 1, 1.0)[:, 0].to(device))
                self.profile = Profile(float(lc.get('depth', OFFSET_CELLS)), device=device)

        def attach(self, scene):
            self.scene = scene

        def height(self, x):
            """h(x) in normalized units (N,), with its first derivative with respect to x in the graph when x requires grad."""
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

        def input_mapping(self, x):
            proj = x @ (2. * np.pi * self.B)
            if self.lift_kind != 'none':
                proj = proj + (2.0 * math.pi) * self.height(x)[:, None] * self.b_h[None, :]
            return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    return LiftNN


def build_network(mn, device, dim, B, cfg, scene):
    net = lift_class(mn)(device, dim, B, cfg)
    net.apply(net.init_weights)
    net.to(device)
    net.attach(scene)
    return net


def load_field(mn, checkpoint, scene, device='cuda:0'):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    B = payload['B_state_dict']
    net = build_network(mn, device, B.shape[1], B, payload['embed_config'], scene)
    net.load_state_dict(payload['model_state_dict'], strict=True)
    net.to(torch.device(device)).float().eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


# --- fast evaluation -----------------------------------------------------------------------------------------------------------------
class Clearance:
    """scene.signed_cells (optionally torch.compile'd) on fixed-size chunks; counts the queries."""

    def __init__(self, scene, chunk=1 << 18, compile=True):
        self.scene, self.chunk, self.n = scene, int(chunk), 0
        self.fn = torch.compile(scene.signed_cells, dynamic=False) if compile else scene.signed_cells
        self.buckets = [self.chunk >> k for k in (0, 2, 4, 6) if (self.chunk >> k) >= 1024]

    @torch.no_grad()
    def __call__(self, q):
        self.n += len(q)
        out = []
        for i in range(0, len(q), self.chunk):
            x = q[i:i + self.chunk]
            b = next((b for b in reversed(self.buckets) if len(x) <= b), self.chunk)
            pad = torch.cat([x, x[:1].expand(b - len(x), x.shape[1])]) if b > len(x) else x
            out.append(self.fn(pad)[:len(x)].clone())
        return torch.cat(out) if out else q.new_zeros(0)


def _rows_weight(net):
    B2pi = (2.0 * math.pi * net.B).detach().clone()
    if getattr(net, 'lift_kind', 'none') == 'none':
        return B2pi
    return torch.cat([B2pi, (2.0 * math.pi * net.b_h.detach().clone())[None, :]], 0)


class Compiled:
    """Frozen (lifted) evaluator; update(net) copies new weights in place. clear: a shared Clearance."""

    def __init__(self, net, scene, clear, chunk=1 << 19, compile=True):
        self.lift = getattr(net, 'lift_kind', 'none') != 'none'
        self.W = _rows_weight(net)
        fz = F.Frozen(net)
        self.layers, self.gates, self.nl1 = fz.layers, fz.gates, fz.nl1
        self.scene, self.clear = scene, clear
        if self.lift:
            self.profile, self.gain, self.E = net.profile, float(net.lift_gain), float(scene.E)
        W, layers, gates, nl1 = self.W, self.layers, self.gates, self.nl1

        def kernel(x, g):
            return F.metric(F._forward(W, layers, gates, nl1, x), g)

        def enc(x):
            return F._forward(W, layers, gates, nl1, x)

        self.kernel = torch.compile(kernel, dynamic=False) if compile else kernel
        self.enc = torch.compile(enc, dynamic=False) if compile else enc
        self.set_chunk(chunk)

    @torch.no_grad()
    def update(self, net):
        fz = F.Frozen(net)
        for (w, b), (w2, b2) in zip(self.layers, fz.layers):
            w.copy_(w2)
            b.copy_(b2)
        for g, g2 in zip(self.gates, fz.gates):
            g.copy_(g2)

    def set_chunk(self, chunk):
        self.chunk = int(chunk)
        self.buckets = [self.chunk >> k for k in (0, 2, 4, 6, 8) if (self.chunk >> k) >= 4096]

    def _bucket(self, n):
        for b in reversed(self.buckets):
            if n <= b:
                return b
        return self.chunk

    @property
    def n_clear(self):
        return self.clear.n

    @torch.no_grad()
    def clearance(self, q):
        return self.clear(q)

    @torch.no_grad()
    def rows(self, q, d=None):
        if not self.lift:
            return q
        if d is None:
            d = self.clear(q)
        G, _ = self.profile(d)
        return torch.cat([q, (self.gain * G / self.E)[:, None]], 1)

    @torch.no_grad()
    def encode_rows(self, x):
        out = []
        for i in range(0, len(x), self.chunk):
            xs = x[i:i + self.chunk]
            b = self._bucket(len(xs))
            pad = torch.cat([xs, xs.new_zeros(b - len(xs), xs.shape[1])]) if b > len(xs) else xs
            out.append(self.enc(pad)[:len(xs)].clone())
        return torch.cat(out) if out else x.new_zeros(0, 256)

    @torch.no_grad()
    def encode(self, q, d=None):
        return self.encode_rows(self.rows(q, d))

    @torch.no_grad()
    def to_goal(self, states, goal_feat, d=None):
        x = self.rows(states, d)
        out = []
        for i in range(0, len(x), self.chunk):
            xs, gs = x[i:i + self.chunk], goal_feat[i:i + self.chunk]
            b = self._bucket(len(xs))
            if b > len(xs):
                xs = torch.cat([xs, xs.new_zeros(b - len(xs), xs.shape[1])])
                gs = torch.cat([gs, gs.new_zeros(b - len(gs), gs.shape[1])])
            out.append(self.kernel(xs, gs)[:min(self.chunk, len(x) - i)].clone())
        return torch.cat(out) if out else states.new_zeros(0)

    @torch.no_grad()
    def travel(self, pairs):
        D = pairs.shape[1] // 2
        return F.metric(self.encode(pairs[:, :D]), self.encode(pairs[:, D:]))


def travel_train(net, W, ra, rb):
    """T(a, b) from input rows, differentiable in the network's parameters."""
    gates = [torch.sigmoid(0.1 * net.gate[ii].weight) for ii in range(net.nl1)]
    f = F._forward(W, F._layers(net), gates, net.nl1, torch.cat([ra, rb]))
    return F.metric(f[:len(ra)], f[len(ra):])


@torch.no_grad()
def mppi(ev, start, goal, gen, budget, lo, hi, sample_num=50, horizon=5, step=0.015, temperature=50.0, tol=0.01,
         stall_window=0, stall_drop=0.0, safe=False, safe_margin=0.0, safe_cost=100.0, **_):
    """liftfast.mppi for any dimension: paths (P, L + 1, D) (rows past a pair's stop repeat its last position), stop, converged, stalled.
    safe: a rollout whose first state has clearance below scene.contact + safe_margin cells gets safe_cost (traversals only)."""
    n, dev = len(start), start.device
    D = start.shape[1]
    L = int(budget.max())
    paths = start.new_empty(n, L + 1, D)
    paths[:, 0] = start
    cur_all = start.clone()
    prior = torch.zeros(n, D, device=dev)
    goal_feat = ev.encode(goal)
    budget = budget.to(dev).long()
    stop = budget.clone()
    active = torch.ones(n, dtype=torch.bool, device=dev)
    converged = torch.zeros(n, dtype=torch.bool, device=dev)
    stalled = torch.zeros(n, dtype=torch.bool, device=dev)
    best = torch.full((n,), float('inf'), device=dev)
    best_at = torch.zeros(n, dtype=torch.long, device=dev)
    limit = ev.scene.contact + safe_margin
    for it in range(L):
        active = active & (budget > it)
        ids = torch.nonzero(active).squeeze(1)
        if not len(ids):
            break
        k = len(ids)
        cur = cur_all[ids]
        dp = step * torch.randn(k, sample_num, 1, D, device=dev, generator=gen) + step * torch.randn(k, sample_num, horizon, D, device=dev, generator=gen)
        dp = dp + 2 * prior[ids][:, None, None, :]
        dp = dp / (torch.linalg.vector_norm(dp, dim=3, keepdim=True).clamp(min=step) / step)
        roll = torch.maximum(torch.minimum(cur[:, None, None, :] + torch.cumsum(dp, dim=2), hi), lo)
        ends = roll[:, :, [0, -1], :]
        flat = ends.reshape(-1, D)
        d = ev.clearance(flat) if ev.lift else None
        gf = goal_feat[ids][:, None, None, :].expand(k, sample_num, 2, goal_feat.shape[1])
        c = ev.to_goal(flat, gf.reshape(-1, goal_feat.shape[1]), d=d).reshape(k, sample_num, 2)
        if safe:
            d0 = d.reshape(k, sample_num, 2)[:, :, 0] if d is not None else ev.clearance(ends[:, :, 0, :].reshape(-1, D)).reshape(k, sample_num)
            pen = (d0 < limit).float() * safe_cost
        c = 10 * c[..., 0] + c[..., 1]
        if safe:
            c = c + pen
        weight = torch.softmax(-temperature * c, dim=1)
        update = torch.einsum('bs,bsd->bd', weight, dp[:, :, 0, :])
        prior[ids] = update
        new = torch.maximum(torch.minimum(cur + update, hi), lo)
        cur_all[ids] = new
        paths[ids, it + 1] = new
        done = torch.linalg.vector_norm(goal[ids] - new, dim=1) < tol
        stop[ids[done]] = it + 1
        converged[ids[done]] = True
        active[ids[done]] = False
        if stall_window:
            tnow = (weight * (c if not safe else c - pen)).sum(1) / 11.0
            better = tnow < best[ids] - stall_drop
            best[ids] = torch.where(better, tnow, best[ids])
            best_at[ids] = torch.where(better, torch.full_like(best_at[ids], it), best_at[ids])
            st = (~done) & (it - best_at[ids] >= stall_window)
            if bool(st.any()):
                stop[ids[st]] = it + 1
                stalled[ids[st]] = True
                active[ids[st]] = False
    idx = torch.arange(L + 1, device=dev)[None, :].expand(n, -1)
    last = torch.minimum(idx, stop[:, None])
    paths = torch.gather(paths, 1, last[..., None].expand(-1, -1, D))
    return paths, stop, converged, stalled


@torch.no_grad()
def collisions(scene, paths, stop, clear, chunk=1 << 20):
    """Clearance (cells) at every waypoint, and the index of the first waypoint i whose position or the midpoint before it is in
    contact (clearance < scene.contact); stop + 1 if none."""
    n, L1, D = paths.shape
    flat = paths.reshape(-1, D)
    d = torch.cat([clear(flat[i:i + chunk]) for i in range(0, len(flat), chunk)]).reshape(n, L1)
    mf = (0.5 * (paths[:, 1:] + paths[:, :-1])).reshape(-1, D)
    dm = torch.cat([clear(mf[i:i + chunk]) for i in range(0, len(mf), chunk)]).reshape(n, L1 - 1)
    bad = d < scene.contact
    bad[:, 1:] |= dm < scene.contact
    idx = torch.arange(L1, device=paths.device)[None].expand(n, -1)
    bad &= idx <= stop[:, None]
    first = torch.where(bad.any(1), torch.argmax(bad.int(), 1), stop + 1)
    return d, first


# --- the Bellman backup ----------------------------------------------------------------------------------------------------------------
class Slowness:
    """F(d) = int_d^margin dd' / S(d') (cells), continued linearly below the offset and above the margin (jeffrey/lbs/bell.py)."""

    def __init__(self, device, nodes=16385):
        d = np.linspace(OFFSET_CELLS, MARGIN_CELLS, nodes)
        inv = 1 / train_speed_np(d)
        seg = 0.5 * (inv[1:] + inv[:-1]) * np.diff(d)
        Fv = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
        self.table = torch.tensor(Fv, device=device, dtype=torch.float64)
        self.nodes, self.inv_floor, self.F0 = nodes, float(inv[0]), float(Fv[0])

    def F(self, d):
        d = d.double()
        dc = d.clamp(OFFSET_CELLS, MARGIN_CELLS)
        u = (dc - OFFSET_CELLS) / (MARGIN_CELLS - OFFSET_CELLS) * (self.nodes - 1)
        i = u.floor().long().clamp(0, self.nodes - 2)
        w = u - i
        out = (1 - w) * self.table[i] + w * self.table[i + 1]
        out = torch.where(d < OFFSET_CELLS, self.F0 + (OFFSET_CELLS - d) * self.inv_floor, out)
        return torch.where(d > MARGIN_CELLS, MARGIN_CELLS - d, out)

    def mean(self, d1, d2):
        dd = (d2 - d1).double()
        flat = dd.abs() < 1e-5
        m = (self.F(d1) - self.F(d2)) / torch.where(flat, torch.ones_like(dd), dd)
        return torch.where(flat, 1 / train_speed(0.5 * (d1 + d2)).double(), m).float()


_slow = {}


def step_time(scene, a, b, samples, clear, exact=True, chunk=4_000_000):
    """Straight step a -> b (..., D): its travel time at the true speed (length x mean of 1 / S along it) and whether it stays free
    (clearance above scene.contact - offset at every sample: the mazes' rule, where contact is 0)."""
    shape, D = a.shape[:-1], a.shape[-1]
    ts = torch.linspace(0, 1, samples, device=a.device)
    af, bf = a.reshape(-1, D), b.reshape(-1, D)
    rows = max(1, chunk // samples)
    ds = []
    for i in range(0, len(af), rows):
        p = af[i:i + rows, None, :] + ts[:, None] * (bf[i:i + rows] - af[i:i + rows])[:, None, :]
        ds.append(clear(p.reshape(-1, D)).reshape(-1, samples))
    d = torch.cat(ds).reshape(*shape, samples)
    if exact:
        if a.device not in _slow:
            _slow[a.device] = Slowness(a.device)
        mean = _slow[a.device].mean(d[..., :-1], d[..., 1:]).mean(-1)
    else:
        inv = 1 / train_speed(d)
        mean = (0.5 * (inv[..., 1:] + inv[..., :-1])).mean(-1)
    return torch.linalg.vector_norm(b - a, dim=-1) * mean, d.amin(-1) > scene.contact - OFFSET_CELLS


def bellman_term(net, scene, points, T, grads, radius, K, gen, clear, cone=6, long=8.0, subset=0, samples_per_radius=8.5, goal=False):
    """jeffrey/lbs/bell.py bellman_term: mean squared backup residual over both ends of the pairs, the share of pair ends used, the mean
    T - y. subset > 0: the backup is computed on that many random pairs of the batch (the step cost needs samples x K exact clearances
    per pair end). Samples along a step: samples_per_radius per radius of the longest step (bell.py: 17 per radius, at most 137).
    goal: the last candidate is the straight step towards the pair's other end (its drawn length, at most the distance to that end).
    The backup takes the minimum over sampled steps, which is never below the true backup; when no sampled step is close to the
    best direction, the fixed point is an inflated T. The straight step is the best one wherever the way to the goal is open."""
    N, dev = len(points), points.device
    D = points.shape[1] // 2
    if subset and subset < N:
        pick = torch.randperm(N, generator=gen, device=dev)[:subset]
        points, T, grads = points[pick], T[pick], (grads[0][pick], grads[1][pick])
        N = subset
    total, used, gap = 0.0, 0.0, 0.0
    samples = int(max(9, min(round(samples_per_radius * max(long, 1.0)) + 1, 137)))
    for end in (0, 1):
        sl = slice(D * end, D * end + D)
        other = slice(D - D * end, 2 * D - D * end)
        a = points[:, sl]
        dirs = torch.randn(N, K, D, generator=gen, device=dev)
        g = -grads[end]
        g = g / torch.linalg.vector_norm(g, dim=1, keepdim=True).clamp(min=1e-12)
        length = radius * (0.5 + 0.5 * torch.rand(N, K, 1, generator=gen, device=dev))
        dirs[:, 0] = g
        dirs[:, 1] = g
        length[:, 0] = radius
        length[:, 1] = 0.5 * radius
        if cone:
            dirs = dirs / torch.linalg.vector_norm(dirs, dim=2, keepdim=True).clamp(min=1e-12)
            dirs[:, 2:2 + cone] = g[:, None, :] + 0.5 * dirs[:, 2:2 + cone]
            dirs[:, 2] = g
            length[:, 2] = 0.25 * radius
            length[:, 3:2 + cone] = radius * (0.25 + 0.75 * torch.rand(N, cone - 1, 1, generator=gen, device=dev))
        dirs = dirs / torch.linalg.vector_norm(dirs, dim=2, keepdim=True).clamp(min=1e-12)
        if long > 1:
            lo, hi = np.log(0.5 * radius), np.log(long * radius)
            length[:, 2:] = torch.exp(lo + (hi - lo) * torch.rand(N, K - 2, 1, generator=gen, device=dev))
        if goal:
            to = points[:, other] - a
            dist = torch.linalg.vector_norm(to, dim=1, keepdim=True)
            dirs[:, -1] = to / dist.clamp(min=1e-12)
            length[:, -1] = torch.minimum(length[:, -1], dist)
        with torch.no_grad():
            b = torch.minimum(torch.maximum(a[:, None, :] + dirs * length, scene.lo), scene.hi)
            tau, free = step_time(scene, a[:, None, :].expand(-1, K, -1), b, samples, clear)
            new = points[:, None, :].repeat(1, K, 1)
            new[:, :, sl] = b
            Tn = net.out(new.reshape(-1, 2 * D))[0][:, 0].reshape(N, K)
            cand = torch.where(free, tau + Tn, torch.full_like(Tn, float('inf')))
            y, _ = cand.min(1)
            ok = torch.isfinite(y) & (torch.linalg.vector_norm(a - points[:, other], dim=1) > 2 * radius)
        if bool(ok.any()):
            r = (T - y)[ok]
            total = total + (r ** 2).mean()
            gap += float(r.mean()) / 2
        used += float(ok.float().mean()) / 2
    return total, used, gap


# --- planning the queries and scoring them ------------------------------------------------------------------------------------------------
@torch.no_grad()
def plan(ev, start, goal, seed, steps, sample_num=50, horizon=5, step=0.015, temperature=50.0, tol=0.01, **_):
    """ntrl-demo's MPPI (tests/arm_plan_stat.py as experiments/ntrl_cabinet/field.py ntrl_mppi batches it) in the training coordinates,
    with every joint clamped to its limits; a fixed number of steps for every query. Returns a list of paths (start ... last, goal
    appended, as the evaluator does) and converged."""
    gen = torch.Generator(start.device).manual_seed(int(seed))
    budget = torch.full((len(start),), int(steps), device=start.device)
    paths, stop, conv, _ = mppi(ev, start, goal, gen, budget, ev.scene.lo, ev.scene.hi, sample_num=sample_num, horizon=horizon, step=step,
                                temperature=temperature, tol=tol)
    return [torch.cat([paths[i, :int(stop[i]) + 1], goal[i:i + 1]]) for i in range(len(start))], conv


def densify(path, res):
    """Interpolate so that no coordinate moves more than res[j] between consecutive points (the scorer's rule, per joint)."""
    seg = path[1:] - path[:-1]
    k = torch.clamp(torch.ceil((seg.abs() / res).amax(1)), min=1).long()
    j = torch.repeat_interleave(torch.arange(len(seg), device=path.device), k)
    first = torch.cumsum(k, 0) - k
    t = (torch.arange(int(k.sum()), device=path.device) - first[j] + 1).to(path.dtype) / k[j].to(path.dtype)
    return torch.cat([path[:1], path[j] + t[:, None] * seg[j]])


@torch.no_grad()
def score(scene, paths, converged, step_rad=0.002):
    """Per path: success = every configuration of the path, densified to step_rad per joint (the release scorer: 0.001), is valid by the
    scorer's rule (scene.valid_cells >= 0). The goal is appended to every path, so a path that did not converge fails by collision or
    counts as 'not arrived'."""
    res = step_rad / scene.q_range
    rows = []
    for i, path in enumerate(paths):
        dense = densify(path, res)
        v = scene.valid_cells(dense)
        bad = torch.nonzero(v < 0).squeeze(1)
        row = dict(converged=bool(converged[i]), collision=bool(len(bad)), min_margin_cells=float(v.min()), clen=float(torch.linalg.vector_norm(dense[1:] - dense[:-1], dim=1).sum()),
                   waypoints=int(len(path)))
        if len(bad):
            pt = scene.parts(dense[int(bad[0]):int(bad[0]) + 1])
            row['contact'] = 'self' if float(pt['self'][0]) - 0.002 < float(pt['obs_all'][0]) - 0.005 else 'obstacle'
        row['success'] = not row['collision']
        rows.append(row)
    return rows
