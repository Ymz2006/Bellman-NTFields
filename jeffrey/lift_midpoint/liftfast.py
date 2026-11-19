"""Batched evaluation of the lifted ntrl-demo network (lift.py LiftNN, kind 'time' or 'none', no octaves) for the midpoint loss.

The lifted network's input mapping is  proj = 2 pi (q B^T + h(q) b_h^T),  h(q) = gain G(d(q)) / E  (lift.py): one more input column,
the travel-time height h of the configuration, a closed-form function of its exact clearance d. Written as one matrix product,
proj = [q, h] W with W = 2 pi [B; b_h] (4 x 128), so midpoint_dp's fastnet._forward (Fourier features, gated sine blocks, last layer,
per-row InstanceNorm) runs unchanged on 4-column rows [q, h]; with kind 'none' the rows are q and W = 2 pi B (ntrl-demo, as fastnet).

    Compiled     no-grad evaluator (torch.compile on fixed-size chunks, as midpoint_dp/traverse.Compiled) that computes h from the
                 compiled exact clearance (traverse.Clearance) of the states it is given, or takes a clearance already computed
    mppi         midpoint_dp/traverse.mppi line for line (same sampling, step clipping, cost 10 T(first) + T(last), softmax weights, phi
                 clamp, tolerance stop, stall stop, 'safe' rollouts) except that the clearance of both rollout ends is computed once per
                 step and serves both the lift and the safety probe
    travel_train T(a, b) with gradients in the network's parameters, from rows [q, h] with h precomputed (the bank's pairs are fixed
                 between refreshes); equals NN.out's T
    check        all of the above against NN.out on random free pairs
"""
import math
import torch
import fastnet as F
import traverse as TV


def _rows_weight(net):
    """W (in_dim, 128) such that rows @ W is the network's Fourier projection."""
    B2pi = (2.0 * math.pi * net.B).detach().clone()
    if getattr(net, 'lift_kind', 'none') == 'none':
        return B2pi
    assert net.lift_kind == 'time' and not getattr(net, 'K', 0), 'only the scalar travel-time lift without octaves is supported'
    return torch.cat([B2pi, (2.0 * math.pi * net.b_h.detach().clone())[None, :]], 0)


class Compiled:
    """Frozen lifted evaluator. update(net) copies new weights in place (the compiled graphs close over these tensors)."""

    def __init__(self, net, scene, chunk=1 << 20, compile=True, clear_chunk=1 << 18):
        self.lift = getattr(net, 'lift_kind', 'none') != 'none'
        self.W = _rows_weight(net)
        fz = F.Frozen(net)
        self.layers, self.gates, self.nl1 = fz.layers, fz.gates, fz.nl1
        self.scene = scene
        if self.lift:
            self.profile, self.gain, self.E = net.profile, float(net.lift_gain), float(scene.E)
        self.clear = TV.Clearance(scene, chunk=clear_chunk, compile=compile)
        self.n_clear = 0
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

    @torch.no_grad()
    def clearance(self, q):
        self.n_clear += len(q)
        return self.clear(q)

    @torch.no_grad()
    def height(self, q, d=None):
        """h(q) in normalized units (lift.LiftNN.height without the derivative)."""
        if d is None:
            d = self.clearance(q)
        G, _ = self.profile(d)
        return self.gain * G / self.E

    @torch.no_grad()
    def rows(self, q, d=None):
        if not self.lift:
            return q
        return torch.cat([q, self.height(q, d)[:, None]], 1)

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
        """T(states[k], goal) with goal_feat (N, 256) row-aligned with states (N, 3); d = their clearance if already known."""
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
        fa = self.encode(pairs[:, :3])
        fb = self.encode(pairs[:, 3:])
        return F.metric(fa, fb)


def train_weight(net):
    """W for travel_train (constant: B and b_h are not trained)."""
    return _rows_weight(net)


def travel_train(net, W, ra, rb):
    """T(a, b) from input rows ra, rb (N, 3 or 4), differentiable in the network's parameters (training)."""
    gates = [torch.sigmoid(0.1 * net.gate[ii].weight) for ii in range(net.nl1)]
    f = F._forward(W, F._layers(net), gates, net.nl1, torch.cat([ra, rb]))
    return F.metric(f[:len(ra)], f[len(ra):])


@torch.no_grad()
def mppi(ev, start, goal, gen, budget, lo, hi, sample_num=50, horizon=5, step=0.015, temperature=50.0, tol=0.01,
         stall_window=0, stall_drop=0.0, safe=False, safe_margin=0.0, safe_mode='first', safe_cost=100.0, **_):
    """midpoint_dp/traverse.mppi with the lifted evaluator: returns paths (P, L + 1, 3) (rows past a pair's stop repeat its last
    position), stop (P,), converged (P,), stalled (P,). safe: a rollout whose first state (safe_mode 'first') or first and last states
    ('ends') has exact clearance < safe_margin cells gets safe_cost added (traversals only; the evaluated planner never does this)."""
    n, dev = len(start), start.device
    L = int(budget.max())
    paths = start.new_empty(n, L + 1, 3)
    paths[:, 0] = start
    cur_all = start.clone()
    prior = torch.zeros(n, 3, device=dev)
    goal_feat = ev.encode(goal)
    budget = budget.to(dev).long()
    stop = budget.clone()
    active = torch.ones(n, dtype=torch.bool, device=dev)
    converged = torch.zeros(n, dtype=torch.bool, device=dev)
    stalled = torch.zeros(n, dtype=torch.bool, device=dev)
    best = torch.full((n,), float('inf'), device=dev)
    best_at = torch.zeros(n, dtype=torch.long, device=dev)
    lo2, hi2 = float(lo[2]), float(hi[2])
    for it in range(L):
        active = active & (budget > it)
        ids = torch.nonzero(active).squeeze(1)
        if not len(ids):
            break
        k = len(ids)
        cur = cur_all[ids]
        dp = step * torch.randn(k, sample_num, 1, 3, device=dev, generator=gen) + step * torch.randn(k, sample_num, horizon, 3, device=dev, generator=gen)
        dp = dp + 2 * prior[ids][:, None, None, :]
        dp = dp / (torch.linalg.vector_norm(dp, dim=3, keepdim=True).clamp(min=step) / step)
        roll = cur[:, None, None, :] + torch.cumsum(dp, dim=2)
        roll[..., 2] = roll[..., 2].clamp(lo2, hi2)
        ends = roll[:, :, [0, -1], :]                                            # (k, S, 2, 3)
        flat = ends.reshape(-1, 3)
        d = ev.clearance(flat) if (ev.lift or (safe and safe_mode != 'first')) else None
        gf = goal_feat[ids][:, None, None, :].expand(k, sample_num, 2, goal_feat.shape[1])
        c = ev.to_goal(flat, gf.reshape(-1, goal_feat.shape[1]), d=d).reshape(k, sample_num, 2)
        if safe:
            if d is not None:
                dd = d.reshape(k, sample_num, 2)
                blocked = (dd[:, :, 0] < safe_margin) if safe_mode == 'first' else (dd < safe_margin).any(2)
            else:
                blocked = (ev.clearance(ends[:, :, 0, :].reshape(-1, 3)) < safe_margin).reshape(k, sample_num)
            pen = blocked.float() * safe_cost
        c = 10 * c[..., 0] + c[..., 1]
        if safe:
            c = c + pen
        weight = torch.softmax(-temperature * c, dim=1)
        update = torch.einsum('bs,bsd->bd', weight, dp[:, :, 0, :])
        prior[ids] = update
        new = cur + update
        new[:, 2] = new[:, 2].clamp(lo2, hi2)
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
    paths = torch.gather(paths, 1, last[..., None].expand(-1, -1, 3))
    return paths, stop, converged, stalled


def check(net, scene, n=4096, seed=0, compile=False):
    """Max |T - NN.out| on random free pairs for the compiled evaluator (travel, to_goal) and the training path."""
    gen = torch.Generator(scene.device).manual_seed(seed)
    a = scene.free(n, gen, 0.0)
    b = scene.free(n, gen, 0.0)
    # include band points: half of the second ends within the band of the first set's walls
    pairs = torch.cat([a, b], 1)
    with torch.no_grad():
        ref = net.out(pairs)[0][:, 0]
        ev = Compiled(net, scene, chunk=1 << 14, compile=compile)
        t1 = ev.travel(pairs)
        t2 = ev.to_goal(a, ev.encode(b))
        W = train_weight(net)
        t3 = travel_train(net, W, ev.rows(a), ev.rows(b))
    d = scene.clearance(a)
    return dict(travel=float((t1 - ref).abs().max()), to_goal=float((t2 - ref).abs().max()), train=float((t3 - ref).abs().max()),
                ref_mean=float(ref.mean()), band_share=float((d < 0.175).float().mean()))
