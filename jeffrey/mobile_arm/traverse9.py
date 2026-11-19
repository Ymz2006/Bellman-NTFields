"""Batched MPPI traversals for the midpoint loss in the 9-D C-space: jeffrey/midpoint_dp/traverse.py with the dimension taken from the
configurations (there: 3) and the scene's clamp mask (there: phi only). Same sampling, step clipping, cost 10 T(first) + T(last), softmax
weights, weighted first step, tolerance stop, stall rule and 'safe' option (a rollout whose first state has exact clearance below
safe_margin gets +100 cost). Compiled: the frozen network's encoder through torch.compile on fixed chunk sizes; collisions(): exact
clearance at every waypoint and segment midpoint.
"""
import torch
import fastnet as F


class Compiled:
    """Frozen evaluator (midpoint_dp/traverse.py Compiled) for configurations of any dimension."""

    def __init__(self, frozen, chunk=1 << 20, compile=True):
        self.fz, self.chunk = frozen, int(chunk)
        fz = frozen

        def kernel(x, g):
            return F.metric(F._forward(fz.B2pi, fz.layers, fz.gates, fz.nl1, x), g)

        def enc(x):
            return F._forward(fz.B2pi, fz.layers, fz.gates, fz.nl1, x)

        self.kernel = torch.compile(kernel, dynamic=False) if compile else kernel
        self.enc = torch.compile(enc, dynamic=False) if compile else enc
        self.set_chunk(self.chunk)

    def update(self, network):
        frozen = F.Frozen(network)
        for (w, b), (w2, b2) in zip(self.fz.layers, frozen.layers):
            w.copy_(w2)
            b.copy_(b2)
        for g, g2 in zip(self.fz.gates, frozen.gates):
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
    def encode(self, x):
        out = []
        for i in range(0, len(x), self.chunk):
            xs = x[i:i + self.chunk]
            b = self._bucket(len(xs))
            pad = torch.cat([xs, xs.new_zeros(b - len(xs), xs.shape[1])]) if b > len(xs) else xs
            out.append(self.enc(pad)[:len(xs)].clone())
        return torch.cat(out) if out else x.new_zeros(0, 256)

    @torch.no_grad()
    def to_goal(self, states, goal_feat):
        out = []
        for i in range(0, len(states), self.chunk):
            xs, gs = states[i:i + self.chunk], goal_feat[i:i + self.chunk]
            b = self._bucket(len(xs))
            if b > len(xs):
                xs = torch.cat([xs, xs.new_zeros(b - len(xs), xs.shape[1])])
                gs = torch.cat([gs, gs.new_zeros(b - len(gs), gs.shape[1])])
            out.append(self.kernel(xs, gs)[:min(self.chunk, len(states) - i)].clone())
        return torch.cat(out) if out else states.new_zeros(0)

    @torch.no_grad()
    def travel(self, pairs):
        D = pairs.shape[1] // 2
        fa = self.encode(pairs[:, :D])
        fb = self.encode(pairs[:, D:])
        return F.metric(fa, fb)


@torch.no_grad()
def mppi(ev, start, goal, gen, budget, scene, sample_num=50, horizon=5, step=0.015, temperature=50.0, tol=0.01,
         stall_window=0, stall_drop=0.0, clearance=None, safe_margin=0.0, safe_mode='first', safe_cost=100.0, **_):
    """Returns paths (P, L + 1, D) (rows past a pair's stop repeat its last position), stop (P,), converged (P,), stalled (P,)."""
    n, dev = len(start), start.device
    D = start.shape[1]
    lo, hi, mask = scene.lo, scene.hi, scene.clamp_mask
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
        roll = cur[:, None, None, :] + torch.cumsum(dp, dim=2)
        roll = torch.where(mask, torch.maximum(torch.minimum(roll, hi), lo), roll)
        ends = roll[:, :, [0, -1], :]                                            # (k, S, 2, D)
        gf = goal_feat[ids][:, None, None, :].expand(k, sample_num, 2, goal_feat.shape[1])
        c = ev.to_goal(ends.reshape(-1, D), gf.reshape(-1, goal_feat.shape[1])).reshape(k, sample_num, 2)
        if clearance is not None:
            probe = ends[:, :, 0, :] if safe_mode == 'first' else ends
            blocked = (clearance(probe.reshape(-1, D)) < safe_margin).reshape(probe.shape[:-1])
            if safe_mode != 'first':
                blocked = blocked.any(2)
            pen = blocked.float() * safe_cost
        c = 10 * c[..., 0] + c[..., 1]
        if clearance is not None:
            c = c + pen
        weight = torch.softmax(-temperature * c, dim=1)
        update = torch.einsum('bs,bsd->bd', weight, dp[:, :, 0, :])
        prior[ids] = update
        new = cur + update
        new = torch.where(mask, torch.maximum(torch.minimum(new, hi), lo), new)
        cur_all[ids] = new
        paths[ids, it + 1] = new
        done = torch.linalg.vector_norm(goal[ids] - new, dim=1) < tol
        stop[ids[done]] = it + 1
        converged[ids[done]] = True
        active[ids[done]] = False
        if stall_window:
            tnow = (weight * (c if clearance is None else c - pen)).sum(1) / 11.0
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


@torch.no_grad()
def collisions(scene, paths, stop, chunk=1 << 20, clear=None):
    """Exact clearance (cells) at every waypoint and segment midpoint: (P, L + 1) waypoint clearance, (P,) index of the first waypoint i
    whose position or the midpoint before it collides (stop + 1 if none)."""
    n, L1, D = paths.shape
    flat = paths.reshape(-1, D)
    fn = clear if clear is not None else scene.clearance
    d = torch.cat([fn(flat[i:i + chunk]) for i in range(0, len(flat), chunk)]).reshape(n, L1)
    mid = 0.5 * (paths[:, 1:] + paths[:, :-1])
    mf = mid.reshape(-1, D)
    dm = torch.cat([fn(mf[i:i + chunk]) for i in range(0, len(mf), chunk)]).reshape(n, L1 - 1)
    bad = d < 0
    bad[:, 1:] |= dm < 0
    idx = torch.arange(L1, device=paths.device)[None].expand(n, -1)
    bad &= idx <= stop[:, None]
    first = torch.where(bad.any(1), torch.argmax(bad.int(), 1), stop + 1)
    return d, first
