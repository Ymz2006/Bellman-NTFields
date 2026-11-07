"""The line loss: pick a line in the warped space, read the free / obstacle step function along it, equalize its runs.

For a warp z = phi(x) and a line in z (axis-aligned here: z_perp = c, running along axis a), the graph g(t) =
1[phi^-1(c, t) is in obstacle space] is a step function. Its breakpoints are where the line crosses the image of the obstacle
boundary, so nothing has to be inverted or smoothed: the boundary polylines (obstacles.py) go through phi once, and each
image segment (A, B) that straddles the line gives the crossing

    t = A_a + (c - A_perp) / (B_perp - A_perp) * (B_a - A_a),

a differentiable function of phi's output. Sorted along the line, consecutive crossings bound the runs l_1 .. l_r (free,
obstacle, free, ... free; the unbounded outside is not a run). Runs are compared inside one line and in units of that
line's own mean run, because a loss on raw lengths has the trivial minimum phi = constant:

    hard   (max l - min l) / mean l                   the proposal: difference between the largest and the smallest segment
    hard2  ((max l - min l) / mean l)^2
    logr   log max l - log min l                       the same on a log scale (ratio instead of difference)
    logr2  (log max l - log min l)^2
    var    variance of log l over the line's runs      every run gets a gradient, not only the two extreme ones

A line with a single run has nothing to compare and gives no loss. Runs shorter than `floor` x the line's median are left out
of the minimum: a curved preimage can clip a corner of an obstacle and leave a sliver of any length.
A crossing where the line runs nearly parallel to the image segment moves arbitrarily fast with phi: its position is kept but
its gradient dropped (`graze`, the sine of the angle between line and segment).
"""
import math
import torch


def crossings(Z, seg, c, axis, graze=0.05):
    """Crossings of the lines z[1 - axis] = c (L,) with the polylines Z (N, 2), seg (S, 2). Returns (line index (M,), t (M,))
    sorted by line, then along it."""
    A, B = Z[seg[:, 0]], Z[seg[:, 1]]
    p = 1 - axis
    u, v = A[:, p], B[:, p]
    with torch.no_grad():
        hit = ((u[None] - c[:, None]) * (v[None] - c[:, None])) < 0                    # (L, S)
        line, s = hit.nonzero(as_tuple=True)
    du = v[s] - u[s]
    t = A[s, axis] + (c[line] - u[s]) / du * (B[s, axis] - A[s, axis])
    with torch.no_grad():
        steep = du.abs() / (B[s] - A[s]).norm(dim=1).clamp(min=1e-12)
    t = torch.where(steep >= graze, t, t.detach())
    order = torch.argsort(t.detach())
    line, t = line[order], t[order]
    order = torch.argsort(line, stable=True)
    return line[order], t[order]


def runs(line, t, count):
    """Closed runs between consecutive crossings of each line, padded: lengths (L, R) and mask (L, R). Lines with an odd
    number of crossings (a fold, or a line through a vertex) are dropped."""
    n = torch.bincount(line, minlength=count)
    start = torch.cumsum(n, 0) - n
    pos = torch.arange(len(line), device=t.device) - start[line]
    R = max(int(n.max()) - 1, 1) if len(line) else 1
    ell = t.new_zeros(count, R)
    mask = torch.zeros(count, R, dtype=torch.bool, device=t.device)
    if len(line) > 1:
        same = line[1:] == line[:-1]
        ell[line[:-1][same], pos[:-1][same]] = (t[1:] - t[:-1])[same]
        mask[line[:-1][same], pos[:-1][same]] = True
    even = (n % 2) == 0
    return ell, mask & even[:, None], n


def per_line(ell, mask, kind='hard', floor=0.1, big_runs='all'):
    """Loss of each line with at least two runs: (loss (K,), statistics dict of (K,) tensors). big_runs='free': the largest run is taken
    over the free runs only (the even positions: a line starts in a free run), so a long stretch inside an obstacle costs nothing."""
    use = mask.sum(1) >= 2
    ell, mask = ell[use], mask[use]
    if not len(ell):
        z = ell.new_zeros(0)
        return z, dict(ratio=z, runs=z)
    k = mask.sum(1).clamp(min=1)
    mean = (ell * mask).sum(1) / k
    with torch.no_grad():
        med = torch.nanquantile(torch.where(mask, ell, torch.full_like(ell, float('nan'))), 0.5, dim=1)
    ok = mask & (ell >= floor * med[:, None])
    can_be_big = ok if big_runs == 'all' else ok & (torch.arange(ell.shape[1], device=ell.device)[None] % 2 == 0)
    big = torch.where(can_be_big, ell, torch.full_like(ell, -1.0)).amax(1)
    small = torch.where(ok, ell, torch.full_like(ell, float('inf'))).amin(1)
    if kind == 'hard':
        loss = (big - small) / mean
    elif kind == 'hard2':
        loss = ((big - small) / mean) ** 2
    elif kind == 'logr':
        loss = torch.log(big) - torch.log(small)
    elif kind == 'logr2':
        loss = (torch.log(big) - torch.log(small)) ** 2
    elif kind == 'var':
        lg = torch.log(ell.clamp(min=1e-9))
        m = (lg * ok).sum(1, keepdim=True) / ok.sum(1, keepdim=True).clamp(min=1)
        loss = (((lg - m) ** 2) * ok).sum(1) / ok.sum(1).clamp(min=1)
    else:
        raise ValueError(kind)
    return loss, dict(ratio=(big / small).detach(), runs=k.float())


def line_batch(Z, seg, lines, axis, gen, kind='hard', floor=0.1, graze=0.05, angle=0.0, big_runs='all'):
    """`lines` lines along `axis` of the frame turned by `angle` (radians; 0: axis-aligned), uniform over the image's extent.
    Returns the loss of each line and its statistics."""
    if angle:
        c_, s_ = math.cos(angle), math.sin(angle)
        Z = Z @ torch.tensor([[c_, -s_], [s_, c_]], device=Z.device, dtype=Z.dtype)
    p = 1 - axis
    with torch.no_grad():
        lo, hi = Z[:, p].min(), Z[:, p].max()
        c = lo + (hi - lo) * torch.rand(lines, generator=gen, device=Z.device, dtype=Z.dtype)
    line, t = crossings(Z, seg, c, axis, graze)
    ell, mask, n = runs(line, t, lines)
    loss, stats = per_line(ell, mask, kind, floor, big_runs)
    stats['crossings_odd'] = float(((n % 2) == 1).float().mean())
    return loss, stats
