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
    rays = S.lines_uniform(lines, scene.dim, scene.planes, gen, dev)
    axis = rays.u.abs().argmax(1)
    r = S.read(psi, scene, rays, samples, differentiate=False)
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
        rays = S.axis_rays(p, axis, plane)
        line, t, free0 = S.crossings(psi, scene, rays, samples)
        ell, lid, free, count = S.runs(line, t, free0, rays)
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
    rays = S.lines_uniform(lines, scene.dim, scene.planes, gen, dev)
    t = torch.linspace(-0.5, 0.5, samples, device=dev)
    z = rays.p[:, None, :] + t[None, :, None] * rays.u[:, None, :]
    x = torch.cat([psi(z[i:i + 128].reshape(-1, scene.dim)) for i in range(0, lines, 128)])
    k = rays.plane[:, None].expand(lines, samples).reshape(-1)
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
def read_oblique(psi, scene, p, u, plane, samples):
    """Sign changes along chords of the box through p with direction u, at the resolution of the samples. Returns line, position t (cells of the chord parameter), start state, tlo, thi."""
    tlo, thi = S.chord(p, u)
    s = torch.linspace(0, 1, samples, device=p.device, dtype=p.dtype)
    lines, ts, free0 = [], [], []
    per = max(1, (1 << 19) // samples)
    for i in range(0, len(p), per):
        t = tlo[i:i + per, None] + s[None] * (thi - tlo)[i:i + per, None]
        z = p[i:i + per, None, :] + t[..., None] * u[i:i + per, None, :]
        n = z.shape[0]
        x = psi(z.reshape(-1, scene.dim))
        k = plane[i:i + per, None].expand(n, samples).reshape(-1)
        free = (scene.sdf(x, k) > 0).view(n, samples)
        free0.append(free[:, 0])
        l, j = (free[:, 1:] != free[:, :-1]).nonzero(as_tuple=True)
        lines.append(l + i)
        ts.append(0.5 * (t[l, j] + t[l, j + 1]))
    return torch.cat(lines), torch.cat(ts), torch.cat(free0), tlo, thi


@torch.no_grad()
def any_angle(psi, scene, points=2048, dirs=8, samples=1024, seed=19):
    """The same reading along lines at every angle, not only along the axes: the fit never saw these. Diagonal stripes have equal runs along both axes and a very long run along their own direction, so only this reading shows them.
    At a random point: the run through it along `dirs` directions (evenly spaced in the plane, random in higher dimension). any_stripe = mean over points of (largest - smallest log run over the directions).
    any_log_std: spread of log run over points and directions. any_long: share of (point, direction) in runs longer than 4 x the median."""
    dev = scene.table.v.device
    gen = torch.Generator(dev).manual_seed(seed)
    d = scene.dim
    p = torch.rand(points, d, generator=gen, device=dev) - 0.5
    plane = torch.randint(scene.planes, (points,), generator=gen, device=dev)
    if d == 2:
        ang = torch.arange(dirs, device=dev, dtype=p.dtype) * (math.pi / dirs)
        U = torch.stack([torch.cos(ang), torch.sin(ang)], 1)
    else:
        U = torch.randn(dirs, d, generator=gen, device=dev)
        U = U / U.norm(dim=1, keepdim=True)
    logs = []
    for j in range(dirs):
        u = U[j].expand(points, d).contiguous()
        line, t, free0, tlo, thi = read_oblique(psi, scene, p, u, plane, samples)
        # the crossing just below and just above t = 0 (the point itself): the run through p
        below = torch.where(t < 0, t, torch.full_like(t, -1e9))
        above = torch.where(t >= 0, t, torch.full_like(t, 1e9))
        lo = tlo.clone().scatter_reduce(0, line, below, 'amax', include_self=True)
        hi = thi.clone().scatter_reduce(0, line, above, 'amin', include_self=True)
        logs.append(torch.log((hi - lo).clamp(min=1e-6)))
    L = torch.stack(logs, 1)
    v = L.exp().reshape(-1)
    return dict(any_stripe=float((L.amax(1) - L.amin(1)).mean()), any_log_std=float(L.std()), any_long=float((v > 4 * v.median()).float().mean()), any_median=float(v.median()))


@torch.no_grad()
def run_stats_any(psi, scene, lines=4096, samples=2048, seed=23):
    """run_stats' main numbers along chords at every angle (held-out: seed apart from the fit)."""
    dev = scene.table.v.device
    gen = torch.Generator(dev).manual_seed(seed)
    r = S.read(psi, scene, S.any_rays(lines, scene.dim, scene.planes, gen, dev), samples, differentiate=False)
    v, lid = r['ell'].clamp(min=1e-6), r['lid']
    w = v / v.sum()
    lg = torch.log(v)
    mu = (w * lg).sum()
    mean_len = (w * v).sum()
    n = torch.zeros(lines, device=dev).index_add_(0, lid, torch.ones_like(v))
    s1 = torch.zeros(lines, device=dev).index_add_(0, lid, v)
    s2 = torch.zeros(lines, device=dev).index_add_(0, lid, v * v)
    use = n >= 2
    m = s1[use] / n[use]
    return dict(any_log_std_len=float((w * (lg - mu) ** 2).sum().sqrt()), any_cv_len=float(((w * (v - mean_len) ** 2).sum()).sqrt() / mean_len), any_long_share=float(w[v > 4 * v.median()].sum()),
                any_cv_line=float((((s2[use] / n[use] - m * m).clamp(min=0)).sqrt() / m).mean()), any_runs_per_line=float(r['count'].float().mean()))


@torch.no_grad()
def peak_spacing(psi, scene, lines=4096, samples=2048, seed=17):
    """'Peaking more periodically', measured: speed peaks in the middle of a free run, so the spacing of the speed peaks along a line is the distance between the midpoints of neighbouring free runs.
    Coefficient of variation of that spacing over all neighbours of all lines, and inside a line (lines with at least three peaks)."""
    dev = scene.table.v.device
    gen = torch.Generator(dev).manual_seed(seed)
    rays = S.lines_uniform(lines, scene.dim, scene.planes, gen, dev)
    r = S.read(psi, scene, rays, samples, differentiate=False)
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
        sp = (s / scene.margin).clamp(0.1, 1.0).view(1, 1, res, res)
        b = torch.nn.functional.avg_pool2d(sp, res // blocks)
        out.append(float(b.std() / b.mean()))
    return dict(block_speed_cv=float(np.mean(out)))


@torch.no_grad()
def wall_points(scene, lines=8192, samples=2048, seed=29):
    """Points of the obstacle boundary in x with their normals: where lines through the identity cross it. The same set for every map of a scene."""
    dev = scene.table.v.device
    gen = torch.Generator(dev).manual_seed(seed)
    r = S.lines_uniform(lines, scene.dim, scene.planes, gen, dev)
    line, t, _ = S.crossings(lambda z: z, scene, r, samples, refine=6)
    x = S.at(r, line, t)
    k = r.plane[line]
    keep = (x.abs() < 0.5 - 1.5 / scene.E / 2).all(1)                # not the border of the box
    x, k = x[keep], k[keep]
    n = scene.grad(x, k)
    n = n / n.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return x, k, n


@torch.no_grad()
def wall_zone(phi, scene, pts=None):
    """How thick is each wall's slow zone in z? phi: x -> z. At a boundary point the zone is `margin` thick along the normal in x, so in z it is margin x |J_phi n|.
    line_warp found runs failing where this got thin or uneven (a slow zone under two planner steps). Returns the normal stretch |J_phi n| as quantiles, its spread in log, and the share under half the median. A map is good for walls when the stretch is even."""
    x, k, n = pts if pts is not None else wall_points(scene)
    h = 0.02 / scene.E
    st = (phi(x + h * n) - phi(x - h * n)).norm(dim=1) / (2 * h)
    ok = torch.isfinite(st) & (st > 0)
    st = st[ok]
    lg = torch.log(st)
    med = st.median()
    return dict(zone_q05=q(st, 0.05), zone_q50=float(med), zone_q95=q(st, 0.95), zone_log_std=float(lg.std()), zone_thin=float((st < 0.5 * med).float().mean()), zone_points=int(len(st)), zone_q05_over_q50=q(st, 0.05) / float(med))


def all_measures(psi, scene, samples=2048):
    out = run_stats(psi, scene, samples=samples)
    out.update(stripe_index(psi, scene, samples=samples))
    out.update(periodicity(psi, scene, samples=samples))
    out.update(peak_spacing(psi, scene, samples=samples))
    out.update(any_angle(psi, scene))
    out.update(run_stats_any(psi, scene, samples=samples))
    out.update(speed_evenness(psi, scene))
    out.update(distortion(psi, scene))
    if hasattr(psi, 'inverse'):
        out.update(wall_zone(psi.inverse, scene))
    return out
