"""Batched MPPI traversals for the midpoint loss: ntrl-demo's MPPI (jeffrey/maze_scale/plan.py mppi, same sampling,
step clipping, cost 10 T(first) + T(last), softmax weights, weighted first step, phi clamp, tolerance stop) run on many
(start, goal) pairs at once with the network being trained.

Differences from plan.mppi, all for throughput: the goal's features are computed once per pair (fastnet.Frozen) and the
encoder is torch.compile'd on fixed-size chunks; paths are written to one padded (P, L + 1, 3) buffer; a pair stops
early when it stalls (its T(current, goal) has not dropped by --stall-drop over the last --stall-window steps) or when
its step budget runs out. Nothing here reads the truth. collisions() checks waypoints and segment midpoints with the
exact clearance (the same geometry ntrl-demo's speed labels come from).
"""
import math
import torch
import fastnet as F


class Compiled:
    """Frozen evaluator whose encoder+metric runs through torch.compile on fixed chunk sizes (padding the tail)."""

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
        """T(states[k], goal) with goal_feat (N, 256) row-aligned with states (N, 3)."""
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
        fa = self.encode(pairs[:, :3])
        fb = self.encode(pairs[:, 3:])
        return F.metric(fa, fb)


@torch.no_grad()
def mppi(ev, start, goal, gen, budget, lo, hi, sample_num=50, horizon=5, step=0.015, temperature=50.0, tol=0.01,
         stall_window=0, stall_drop=0.0, clearance=None, safe_margin=0.0, safe_mode='first', safe_cost=100.0, **_):
    """Returns paths (P, L + 1, 3) (rows past a pair's stop repeat its last position), stop (P,) = index of the last
    position, converged (P,), stalled (P,). With clearance (a function of (N, 3) configurations -> cells), a rollout
    whose first state (safe_mode 'first') or first and last states ('ends') has clearance < safe_margin gets
    safe_cost added: the traversal keeps out of walls (the geometry ntrl-demo's labels come from); the evaluated
    planner never does this."""
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
        gf = goal_feat[ids][:, None, None, :].expand(k, sample_num, 2, goal_feat.shape[1])
        c = ev.to_goal(ends.reshape(-1, 3), gf.reshape(-1, goal_feat.shape[1])).reshape(k, sample_num, 2)
        if clearance is not None:
            probe = ends[:, :, 0, :] if safe_mode == 'first' else ends
            blocked = (clearance(probe.reshape(-1, 3)) < safe_margin).reshape(probe.shape[:-1])
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
        new[:, 2] = new[:, 2].clamp(lo2, hi2)
        cur_all[ids] = new
        paths[ids, it + 1] = new
        done = torch.linalg.vector_norm(goal[ids] - new, dim=1) < tol
        stop[ids[done]] = it + 1
        converged[ids[done]] = True
        active[ids[done]] = False
        if stall_window:
            # T(current, goal) ~ the cost of the best rollout's first state; use the weighted first-state cost proxy
            tnow = (weight * (c if clearance is None else c - pen)).sum(1) / 11.0
            better = tnow < best[ids] - stall_drop
            best[ids] = torch.where(better, tnow, best[ids])
            best_at[ids] = torch.where(better, torch.full_like(best_at[ids], it), best_at[ids])
            st = (~done) & (it - best_at[ids] >= stall_window)
            if bool(st.any()):
                stop[ids[st]] = it + 1
                stalled[ids[st]] = True
                active[ids[st]] = False
    # pad rows after each stop with the last position
    idx = torch.arange(L + 1, device=dev)[None, :].expand(n, -1)
    last = torch.minimum(idx, stop[:, None])
    paths = torch.gather(paths, 1, last[..., None].expand(-1, -1, 3))
    return paths, stop, converged, stalled


class Clearance:
    """maze.signed_cells through torch.compile on fixed-size chunks (the MPPI safety check runs it every step)."""

    def __init__(self, maze, chunk=1 << 18, compile=True):
        self.maze, self.chunk = maze, int(chunk)
        self.fn = torch.compile(maze.signed_cells, dynamic=False) if compile else maze.signed_cells
        self.buckets = [self.chunk >> k for k in (0, 2, 4, 6) if (self.chunk >> k) >= 1024]

    @torch.no_grad()
    def __call__(self, q):
        out = []
        for i in range(0, len(q), self.chunk):
            x = q[i:i + self.chunk]
            b = next((b for b in reversed(self.buckets) if len(x) <= b), self.chunk)
            pad = torch.cat([x, x[:1].expand(b - len(x), 3)]) if b > len(x) else x
            out.append(self.fn(pad)[:len(x)].clone())
        return torch.cat(out) if out else q.new_zeros(0)


@torch.no_grad()
def collisions(maze, paths, stop, chunk=1 << 20, clear=None):
    """Exact clearance (cells) at every waypoint and segment midpoint: (P, L + 1) waypoint clearance, (P,) index of the
    first waypoint i whose position or the midpoint before it collides (stop + 1 if none)."""
    n, L1, _ = paths.shape
    flat = paths.reshape(-1, 3)
    fn = clear if clear is not None else maze.clearance
    d = torch.cat([fn(flat[i:i + chunk]) for i in range(0, len(flat), chunk)]).reshape(n, L1)
    mid = 0.5 * (paths[:, 1:] + paths[:, :-1])
    mf = mid.reshape(-1, 3)
    dm = torch.cat([fn(mf[i:i + chunk]) for i in range(0, len(mf), chunk)]).reshape(n, L1 - 1)
    bad = d < 0
    bad[:, 1:] |= dm < 0
    idx = torch.arange(L1, device=paths.device)[None].expand(n, -1)
    bad &= idx <= stop[:, None]
    first = torch.where(bad.any(1), torch.argmax(bad.int(), 1), stop + 1)
    return d, first
