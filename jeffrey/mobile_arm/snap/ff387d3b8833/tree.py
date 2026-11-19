"""Travel-time midpoint recursion along traversed paths, batched over all paths.

A path w_0 .. w_e (MPPI waypoints, cut where it first collides or stalls) is parametrized by its travel time
cum_i = sum_{k < i} |w_(k+1) - w_k| (1 / S(w_k) + 1 / S(w_(k+1))) / 2 (S = the trained speed of the exact clearance,
the speed ntrl-demo's labels come from). A node is a waypoint interval [s, e]; it is a leaf when its travel time
cum_e - cum_s <= tau (the local scale the network is trusted at) or e = s + 1, otherwise it splits at its travel-time
midpoint m (the waypoint whose cum is nearest (cum_s + cum_e) / 2). Every path is split recursively until all nodes are
leaves; the leaves partition the path. Leaf value = the network's T(w_s, w_e) ('net'; the user's "recurse until you
hit the local model"), the path's own travel time over the leaf ('path') or the smaller of the two ('min'). The target
of an inner node [s, e] is the sum of the leaf values inside it: T(w_s, w_e) should equal T(w_s, w_m) + T(w_m, w_e),
recursively, down to the local model. Nodes are processed level by level with one flat searchsorted for all paths.
"""
import torch


def path_time(maze, paths, d=None, chunk=1 << 20):
    """cum (P, L + 1) travel time along each padded path (normalized units, as T) from the exact clearance d (cells)."""
    n, L1, _ = paths.shape
    if d is None:
        flat = paths.reshape(-1, 3)
        d = torch.cat([maze.clearance(flat[i:i + chunk]) for i in range(0, len(flat), chunk)]).reshape(n, L1)
    inv = 1.0 / maze.train_speed(d)
    seg = torch.linalg.vector_norm(paths[:, 1:] - paths[:, :-1], dim=2)
    dt = seg * 0.5 * (inv[:, 1:] + inv[:, :-1])
    return torch.cat([torch.zeros(n, 1, device=paths.device), torch.cumsum(dt, 1)], 1)


def split_tree(cum, end, tau, max_depth=40):
    """Recursive travel-time midpoint split of every path [0, end_p] (end (P,) long, last usable waypoint).
    Returns leaves and inner nodes as (pid, s, e, depth) long tensors."""
    dev = cum.device
    P, L1 = cum.shape
    big = float(cum.max()) * 2 + 1.0
    flat = (cum + torch.arange(P, device=dev, dtype=cum.dtype)[:, None] * big).reshape(-1)
    pid = torch.arange(P, device=dev)
    keep = end >= 1
    pid, s, e = pid[keep], torch.zeros(int(keep.sum()), dtype=torch.long, device=dev), end[keep]
    depth = torch.zeros_like(s)
    leaves, inner = [], []
    for lvl in range(max_depth + 1):
        if not len(pid):
            break
        t = cum[pid, e] - cum[pid, s]
        leaf = (t <= tau) | (e - s <= 1) | (lvl == max_depth)
        leaves.append(torch.stack([pid[leaf], s[leaf], e[leaf], depth[leaf]], 1))
        pid, s, e, depth = pid[~leaf], s[~leaf], e[~leaf], depth[~leaf]
        if not len(pid):
            break
        inner.append(torch.stack([pid, s, e, depth], 1))
        half = 0.5 * (cum[pid, s] + cum[pid, e]) + pid.to(cum.dtype) * big
        m = torch.searchsorted(flat, half) - pid * L1                     # first waypoint with cum >= half
        m = m.clamp(min=1)
        lower = (m - 1).clamp(min=0)
        pick_lower = (half - flat[pid * L1 + lower]) < (flat[pid * L1 + m.clamp(max=L1 - 1)] - half)
        m = torch.where(pick_lower, lower, m)
        m = torch.maximum(torch.minimum(m, e - 1), s + 1)
        pid, s, e, depth = torch.cat([pid, pid]), torch.cat([s, m]), torch.cat([m, e]), torch.cat([depth, depth]) + 1
    z = torch.zeros(0, 4, dtype=torch.long, device=dev)
    return (torch.cat(leaves) if leaves else z), (torch.cat(inner) if inner else z)


def node_targets(paths, cum, leaves, inner, leaf_value):
    """Sum of leaf values inside each inner node. leaf_value (Nl,) aligned with leaves."""
    P, L1, _ = paths.shape
    acc = torch.zeros(P, L1, device=paths.device, dtype=leaf_value.dtype)
    acc.index_put_((leaves[:, 0], leaves[:, 2]), leaf_value, accumulate=True)
    pre = torch.cumsum(acc, 1)
    return pre[inner[:, 0], inner[:, 2]] - pre[inner[:, 0], inner[:, 1]]


def leaf_values(ev, paths, cum, leaves, mode='net'):
    a = paths[leaves[:, 0], leaves[:, 1]]
    b = paths[leaves[:, 0], leaves[:, 2]]
    along = cum[leaves[:, 0], leaves[:, 2]] - cum[leaves[:, 0], leaves[:, 1]]
    if mode == 'path':
        return along
    net = ev.travel(torch.cat([a, b], 1))
    return net if mode == 'net' else torch.minimum(net, along)


def sub_intervals(leaves, count, gen, min_leaves=2):
    """count random waypoint intervals [s, e] whose ends are leaf boundaries of one path and that span >= min_leaves
    leaves (any sub-path, not only the dyadic midpoint nodes); paths are drawn in proportion to their interval count.
    Returns (pid, s, e, -1) rows like split_tree's inner nodes."""
    dev = leaves.device
    order = torch.argsort(leaves[:, 0] * (int(leaves[:, 2].max()) + 2) + leaves[:, 1])
    lv = leaves[order]
    pid_u, counts = torch.unique_consecutive(lv[:, 0], return_counts=True)
    ok = counts >= min_leaves
    if not bool(ok.any()):
        return torch.zeros(0, 4, dtype=torch.long, device=dev)
    first = torch.cumsum(counts, 0) - counts                       # index of each path's first leaf in lv
    K = counts.to(torch.float64)
    w = torch.where(ok, (K - min_leaves + 1) * (K - min_leaves + 2) / 2, torch.zeros_like(K))
    pick = torch.multinomial(w.float(), count, replacement=True, generator=gen)
    Kp = counts[pick]
    # i in [0, K - min_leaves], j in [i + min_leaves, K] (boundary indices), uniform over valid pairs via rejection
    i = (torch.rand(count, generator=gen, device=dev) * (Kp + 1).float()).long().clamp(max=Kp)
    j = (torch.rand(count, generator=gen, device=dev) * (Kp + 1).float()).long().clamp(max=Kp)
    lo, hi = torch.minimum(i, j), torch.maximum(i, j)
    good = (hi - lo) >= min_leaves
    lo, hi, pick, Kp = lo[good], hi[good], pick[good], Kp[good]
    base = first[pick]
    s = lv[base + lo.clamp(max=Kp - 1), 1]
    s = torch.where(lo == Kp, lv[base + Kp - 1, 2], s)
    e = torch.where(hi == Kp, lv[base + Kp - 1, 2], lv[base + hi.clamp(max=Kp - 1), 1])
    return torch.stack([pid_u[pick], s, e, torch.full_like(s, -1)], 1)
