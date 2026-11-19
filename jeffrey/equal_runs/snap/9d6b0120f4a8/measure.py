"""What a map did, on held-out lines (never the batches of the fit) and on a grid of the box: are the runs equal, is the signal periodic, is the map valid, are the stripes back."""
import math
import numpy as np
import torch

import maps as M
import scan as S


def q(v, a):
    return float(torch.quantile(v[:2_000_000], a)) if len(v) else float('nan')


@torch.no_grad()
def run_stats(psi, scene, lines=4096, samples=2048, seed=7):
    dev = scene.table.v.device
    gen = torch.Generator(dev).manual_seed(seed)
    base, axis, plane = S.lines_uniform(lines, scene.dim, scene.planes, gen, dev)
    r = S.read(psi, scene, base, axis, plane, samples, differentiate=False)
    v, lid, free, count = r['ell'].clamp(min=1e-6), r['lid'], r['free'], r['count']
    w = v / v.sum()
    lg = torch.log(v)
    mean_len = (w * v).sum()                                  # expected run length under a random point: long runs dominate
    out = dict(lines=lines, samples=samples, runs_per_line=float(count.float().mean()), single_run_lines=float((count == 1).float().mean()),
               run_q10_50_90=[q(v, 0.1), q(v, 0.5), q(v, 0.9)], log_std_run=float(lg.std()), cv_run=float(v.std() / v.mean()))
    mu = (w * lg).sum()
    out['log_std_len'] = float(((w * (lg - mu) ** 2).sum()).sqrt())
    out['cv_len'] = float(((w * (v - mean_len) ** 2).sum()).sqrt() / mean_len)
    out['mean_len'] = float(mean_len)
    # the share of the box in runs longer than 4 x the typical (median) run
    out['long_share'] = float(w[v > 4 * v.median()].sum())
    # per line: spread inside a line
    n = torch.zeros(lines, device=dev).index_add_(0, lid, torch.ones_like(v))
    s1 = torch.zeros(lines, device=dev).index_add_(0, lid, v)
    s2 = torch.zeros(lines, device=dev).index_add_(0, lid, v * v)
    use = n >= 2
    m = s1[use] / n[use]
    out['cv_line'] = float((((s2[use] / n[use] - m * m).clamp(min=0)).sqrt() / m).mean())
    hi = torch.full((lines,), -1.0, device=dev).scatter_reduce(0, lid, v, 'amax')
    lo = torch.full((lines,), 1e9, device=dev).scatter_reduce(0, lid, v, 'amin')
    ratio = hi[use] / lo[use]
    out['ratio_q50_90'] = [q(ratio, 0.5), q(ratio, 0.9)]
    # per axis: a stripe map has very different run lengths along the two axes
    by = []
    for a in range(scene.dim):
        k = axis[lid] == a
        ww = v[k] / v[k].sum()
        by.append(float((ww * lg[k]).sum()))
    out['log_len_by_axis'] = by
    out['axis_gap'] = max(by) - min(by)
    out['free_share'] = float(v[free].sum() / v.sum())
    out['free_log_mean'] = float(lg[free].mean())
    out['obstacle_log_mean'] = float(lg[~free].mean())
    return out


@torch.no_grad()
def stripe_index(psi, scene, points=2048, samples=2048, seed=11):
    """At a random point p of the box: the run through p along each axis. Mean |log l_a(p) - log l_b(p)| over axis pairs: 0 for a checkerboard, large for stripes (long along one axis, short across)."""
    dev = scene.table.v.device
    gen = torch.Generator(dev).manual_seed(seed)
    p = torch.rand(points, scene.dim, generator=gen, device=dev) - 0.5
    plane = torch.randint(scene.planes, (points,), generator=gen, device=dev)
    logs = []
    for a in range(scene.dim):
        axis = torch.full((points,), a, device=dev, dtype=torch.long)
        line, t, free0 = S.crossings(psi, scene, p, axis, plane, samples)
        ell, lid, free, count = S.runs(line, t, free0, points)
        # position of p in its line: count the crossings below it
        below = torch.zeros(points, device=dev, dtype=torch.long).index_add_(0, line, (t < p[line, a]).long())
        first = torch.cumsum(count, 0) - count
        logs.append(torch.log(ell[first + below].clamp(min=1e-6)))
    logs = torch.stack(logs, 1)
    gaps = [(logs[:, i] - logs[:, j]).abs() for i in range(scene.dim) for j in range(i + 1, scene.dim)]
    return dict(stripe_index=float(torch.stack(gaps, 1).mean()), point_log_std=float(logs.std()), point_log_mean=float(logs.mean()))


@torch.no_grad()
def periodicity(psi, scene, lines=1024, samples=2048, seed=13):
    """Is the step function periodic: share of the signal's power (without the mean) in the three strongest frequencies of each line, averaged over lines. Near 1: one dominant period. Spread: an irregular signal."""
    dev = scene.table.v.device
    gen = torch.Generator(dev).manual_seed(seed)
    base, axis, plane = S.lines_uniform(lines, scene.dim, scene.planes, gen, dev)
    t = torch.linspace(-0.5, 0.5, samples, device=dev)
    z = S.line_points(base, axis, t)
    x = torch.cat([psi(z[i:i + 128].reshape(-1, scene.dim)) for i in range(0, lines, 128)])
    k = plane[:, None].expand(lines, samples).reshape(-1)
    g = (scene.sdf(x, k) <= 0).float().view(lines, samples)
    g = g - g.mean(1, keepdim=True)
    P = torch.fft.rfft(g, dim=1).abs() ** 2
    P = P[:, 1:]
    tot = P.sum(1)
    use = tot > 1e-6
    top = P.topk(3, dim=1).values.sum(1)
    peak = P.argmax(1) + 1
    return dict(peak_share=float((top[use] / tot[use]).mean()), peak_cycles_median=float(peak[use].float().median()), peak_cycles_iqr=[float(peak[use].float().quantile(0.25)), float(peak[use].float().quantile(0.75))])


@torch.no_grad()
def distortion(psi, scene, res=None, seed=3, count=1 << 18):
    """Singular values and sign of det J of psi at random points of the box. In line_warp's terms these are psi = phi^-1, so a large value here is compression in z."""
    dev = scene.table.v.device
    gen = torch.Generator(dev).manual_seed(seed)
    eps = 0.02 / max(scene.E, 1.0)
    z = (torch.rand(count, scene.dim, generator=gen, device=dev) - 0.5) * (1 - 2 * eps)
    Js = torch.cat([M.jacobian_fd(psi, z[i:i + 65536], eps) for i in range(0, count, 65536)])
    sv = M.singular_values(Js)
    det = torch.linalg.det(Js)
    l1, l2 = torch.log(sv[:, 0]), torch.log(sv[:, -1].clamp(min=1e-9))
    return dict(det_positive=float((det > 0).float().mean()), det_min=float(det.min()), sv_q01=q(sv[:, -1], 0.01), sv_q99=q(sv[:, 0], 0.99), sv_min=float(sv[:, -1].min()), sv_max=float(sv[:, 0].max()),
                aniso_q50=q(sv[:, 0] / sv[:, -1].clamp(min=1e-9), 0.5), aniso_q99=q(sv[:, 0] / sv[:, -1].clamp(min=1e-9), 0.99), log_area_std=float(torch.log(det.abs().clamp(min=1e-12)).std()), log_stretch_rms=float(((l1 ** 2 + l2 ** 2) / 2).mean().sqrt()))


@torch.no_grad()
def peak_spacing(psi, scene, lines=4096, samples=2048, seed=17):
    """'Peaking more periodically', measured: speed peaks in the middle of a free run, so the spacing of the speed peaks along a line is the distance between the midpoints of neighbouring free runs.
    Coefficient of variation of that spacing over all neighbours of all lines, and inside a line (lines with at least three peaks)."""
    dev = scene.table.v.device
    gen = torch.Generator(dev).manual_seed(seed)
    base, axis, plane = S.lines_uniform(lines, scene.dim, scene.planes, gen, dev)
    r = S.read(psi, scene, base, axis, plane, samples, differentiate=False)
    v, lid, free = r['ell'], r['lid'], r['free']
    c = torch.cumsum(v, 0)
    first = torch.cumsum(r['count'], 0) - r['count']
    off = (c - v)[first]
    mid = c - off[lid] - 0.5 * v
    m, l = mid[free], lid[free]
    same = l[1:] == l[:-1]
    gap, gl = (m[1:] - m[:-1])[same], l[:-1][same]
    n = torch.zeros(lines, device=dev).index_add_(0, gl, torch.ones_like(gap))
    s1 = torch.zeros(lines, device=dev).index_add_(0, gl, gap)
    s2 = torch.zeros(lines, device=dev).index_add_(0, gl, gap * gap)
    use = n >= 2
    mean = s1[use] / n[use]
    return dict(spacing_cv=float(gap.std() / gap.mean()), spacing_cv_line=float((((s2[use] / n[use] - mean ** 2).clamp(min=0)).sqrt() / mean).mean()), spacing_median=q(gap, 0.5), peaks_per_line=float(free.sum()) / lines)


@torch.no_grad()
def speed_evenness(psi, scene, res=512, blocks=16):
    """'More evenly distributed': mean speed label in each of blocks^2 equal squares of the z box (a 2-D slice), coefficient of variation over the squares, averaged over slices."""
    dev = scene.table.v.device
    t = torch.linspace(-0.5, 0.5, res, device=dev)
    X, Y = torch.meshgrid(t, t, indexing='ij')
    z = torch.stack([X.reshape(-1), Y.reshape(-1)] + [torch.zeros_like(X.reshape(-1))] * (scene.dim - 2), 1)
    x = torch.cat([psi(z[i:i + 262144]) for i in range(0, len(z), 262144)])
    out = []
    for k in range(scene.planes):
        s = scene.sdf(x, torch.full((len(x),), k, device=dev, dtype=torch.long))
        sp = (s / 0.175).clamp(0.1, 1.0).view(1, 1, res, res)
        b = torch.nn.functional.avg_pool2d(sp, res // blocks)
        out.append(float(b.std() / b.mean()))
    return dict(block_speed_cv=float(np.mean(out)))


def all_measures(psi, scene, samples=2048):
    out = run_stats(psi, scene, samples=samples)
    out.update(stripe_index(psi, scene, samples=samples))
    out.update(periodicity(psi, scene, samples=samples))
    out.update(peak_spacing(psi, scene, samples=samples))
    out.update(speed_evenness(psi, scene))
    out.update(distortion(psi, scene))
    return out
