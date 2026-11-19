"""Losses on the runs of a batch of lines. Every line's runs l_1 .. l_r come from scan.py, flat: length, line of each run, and per-line counts.

The proposal: every run has the same value. How 'the same' is read is what this file varies:

    form   lv    (log l - log m)^2     ratio to the target: a run half the target costs what a run twice the target costs
           cv    (l / m - 1)^2         difference from the target, in units of it: a long run costs far more than a short one
    scope  line  m is the line's own mean: the runs of a line agree with each other. A line with a single run has nothing to compare and costs nothing.
           batch m is one value for all lines of the batch: every run of every line agrees, so a long run costs wherever it lies, a single-run line included.
    weight run   every run counts once. A long run is one vote among many short ones.
           len   a run counts for its length (at least one target length): a random point on a random line lands in it with that weight. With this weight the
                 long-thin-stripes solution of line_warp costs what it should: a line along a stripe is one very long run and it is as heavy as all the runs across the stripes.

    hard   the old loss: (max l - min l) / mean l per line
    cap    one-sided: only runs longer than the target cost, (log l - log m)^2 for l > m, weighted by length. Nothing else is pushed.
    pair   the same idea without finding a single run: if every run has length m, then shifting any point by m along an axis lands in the other state, so
           g(z) + g(z + m e_a) = 1 for all z. With u = tanh(clearance / sigma) the loss is the mean of ((u(z) + u(z + m e_a)) / 2)^2 over random z and axes, which is
           (1 + the autocorrelation of the step function at lag m) / 2. Unlike a loss on crossings it can see a wall before the line touches it, it needs no scan, and it
           costs one forward pass in any dimension.

The target m is detached in scope batch unless fixed by --target, so the map cannot lower the loss by moving the target.
A run shorter than `floor` x the target is left out of the push: a curved preimage clips a corner of an obstacle and leaves a sliver of any length, and a sliver pushed to grow is a ripple pushed to grow.

Focus: the segment with the most error is scaled hardest, the others less.
    --top, --rest    runs of the batch ranked by error: the worst `top` share keeps its weight, every other run is scaled by `rest`
    --power p        inside a line: a run's weight is (its error / the line's largest error)^p. 0: all equal; large: only the worst run of each line is pushed
"""
import torch


def scatter_mean(v, lid, count):
    out = v.new_zeros(count).index_add_(0, lid, v)
    n = v.new_zeros(count).index_add_(0, lid, torch.ones_like(v))
    return out / n.clamp(min=1), n


def focus(err, top, rest):
    """Weights of the runs: 1 for the worst `top` share of `err`, `rest` for the others."""
    if top >= 1.0 or rest >= 1.0:
        return torch.ones_like(err)
    k = max(1, int(top * len(err)))
    cut = err.detach().topk(k).values[-1]
    return torch.where(err.detach() >= cut, torch.ones_like(err), torch.full_like(err, rest))


def line_focus(err, lid, lines, power):
    """(err / the largest err of the same line)^power."""
    if power <= 0:
        return torch.ones_like(err)
    worst = err.new_zeros(lines).scatter_reduce(0, lid, err, 'amax', include_self=True)
    return (err / worst[lid].clamp(min=1e-12)).clamp(min=0) ** power


def equal(ell, lid, lines, form='lv', scope='batch', weight='len', target=0.0, top=1.0, rest=1.0, floor=0.0, power=0.0):
    v = ell.clamp(min=1e-6)
    if scope == 'line':
        m, n = scatter_mean(v.detach(), lid, lines)
        lg, _ = scatter_mean(torch.log(v.detach()), lid, lines)
        use = (n >= 2)[lid]
        ref_log, ref = lg[lid], m[lid]
    else:
        use = torch.ones_like(v, dtype=torch.bool)
        with torch.no_grad():
            med = v.median() if target <= 0 else v.new_tensor(target)
            w0 = torch.clamp(v / med, min=1.0) if weight == 'len' else torch.ones_like(v)
            if target > 0:
                ref_log, ref = torch.log(med).expand_as(v), med.expand_as(v)
            else:
                ref_log = ((w0 * torch.log(v)).sum() / w0.sum()).expand_as(v)
                ref = ((w0 * v).sum() / w0.sum()).expand_as(v)
    err = (torch.log(v) - ref_log) ** 2 if form == 'lv' else (v / ref - 1) ** 2
    with torch.no_grad():
        if weight == 'len':
            w = torch.clamp(v / ref, min=1.0) if scope == 'batch' else v / ref
        else:
            w = torch.ones_like(v)
        w = w * focus(err * w, top, rest) * use * (v >= floor * ref) * line_focus((err * w).detach(), lid, lines, power)
    if scope == 'line':
        s = v.new_zeros(lines).index_add_(0, lid, w * err)
        z = v.new_zeros(lines).index_add_(0, lid, w)
        have = z > 0
        return (s[have] / z[have]).mean() if bool(have.any()) else s.sum()
    return (w * err).sum() / w.sum().clamp(min=1e-9)


def anchored(ell, lid, count, ell0, count0, keep):
    """Mean squared change of log run length against the same lines at the start of the update, over the runs selected by keep (flat, of ell0). A run is matched by its rank on a line whose number of runs has not changed."""
    start = torch.cumsum(count, 0) - count
    start0 = torch.cumsum(count0, 0) - count0
    rank = torch.arange(len(ell), device=ell.device) - start[lid]
    same = (count == count0)[lid]
    i = (start0[lid] + rank)[same]
    k = keep[i]
    if not bool(k.any()):
        return ell.new_zeros(())
    return ((torch.log(ell[same].clamp(min=1e-6)) - torch.log(ell0[i].clamp(min=1e-6)))[k] ** 2).mean()


def hard(ell, lid, lines, floor=0.1):
    """(max - min) / mean per line, without slivers below `floor` x the line's median."""
    v = ell.clamp(min=1e-6)
    with torch.no_grad():
        m, n = scatter_mean(v, lid, lines)
        # a line's median is costly with flat arrays; its mean is a stable stand-in for the sliver floor
        ok = v >= floor * m[lid] * 0.5
    big = v.new_full((lines,), -1.0).scatter_reduce(0, lid[ok], v[ok], 'amax', include_self=True)
    small = v.new_full((lines,), 1e9).scatter_reduce(0, lid[ok], v[ok], 'amin', include_self=True)
    mean = v.new_zeros(lines).index_add_(0, lid, v) / n.clamp(min=1)
    use = n >= 2
    return ((big[use] - small[use]) / mean[use]).mean()


def cap(ell, lid, lines, target, top=1.0, rest=1.0, power=0.0):
    """Only runs longer than the target are pushed, each by its log excess, weighted by its length in target units."""
    v = ell.clamp(min=1e-6)
    over = torch.relu(torch.log(v / target))
    with torch.no_grad():
        w = (v / target).clamp(min=1.0)
        w = w * focus(over * over * w, top, rest) * line_focus(over * over * w, lid, lines, power)
    return (w * over * over).sum() / w.sum().clamp(min=1e-9)


def pair(psi, scene, count, lag, sigma, gen, dev):
    """Mean of ((u(z) + u(z + lag e_a)) / 2)^2, u = tanh(clearance / sigma), over random z, axes and slices: 0 when the step function flips at every shift by lag, 1 when it never does."""
    d = scene.dim
    a = torch.randint(d, (count,), generator=gen, device=dev)
    k = torch.randint(scene.planes, (count,), generator=gen, device=dev)
    z = torch.rand(count, d, generator=gen, device=dev) - 0.5
    z[torch.arange(count, device=dev), a] = (torch.rand(count, generator=gen, device=dev) - 0.5) * (1 - lag) - 0.5 * lag
    y = z.clone()
    y[torch.arange(count, device=dev), a] += lag
    both = psi(torch.cat([z, y]))
    s = scene.sdf(both, torch.cat([k, k]))
    u = torch.tanh(s / sigma)
    return (((u[:count] + u[count:]) / 2) ** 2).mean()
