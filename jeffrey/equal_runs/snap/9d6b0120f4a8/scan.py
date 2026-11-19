"""Lines in the warped space, read through the collision oracle.

A line in z runs along one axis at fixed other coordinates. Its preimage psi(line) is a curve in x, and g(t) = 1[psi(line(t)) is an obstacle] is a step function, the graph of the proposal.
Samples find the steps; regula falsi on the clearance pins them down. Nothing here knows the dimension or what an obstacle looks like.

A run length is then a difference of crossing positions t*, and t* depends on the map through s(psi_theta(line(t*))) = 0. Implicit differentiation gives, with n the gradient of the clearance at the crossing,

    d t* = - n . d psi_theta / (n . d psi / d t),

so the gradient needs only the network at the crossing and the oracle's normal: the sampled indicator is never differentiated or smoothed. A crossing where the curve runs almost along the boundary moves
arbitrarily fast with theta: its value is kept and its gradient dropped (`graze`, the sine of the angle between curve and boundary).
"""
import torch


def line_points(base, axis, t):
    """(L, T, d) points of L lines: copies of base (L, d) with coordinate axis (L,) replaced by t (T,)."""
    L, d = base.shape
    z = base[:, None, :].repeat(1, len(t), 1)
    z.scatter_(2, axis[:, None, None].expand(L, len(t), 1), t[None, :, None].expand(L, len(t), 1))
    return z


def put(base, axis, t):
    z = base.clone()
    z[torch.arange(len(z), device=z.device), axis] = t
    return z


@torch.no_grad()
def crossings(psi, scene, base, axis, plane, samples, refine=3, chunk=1 << 19):
    """Sign changes of the clearance along each line. Returns line index (M,), crossing t (M,), sorted by line then t, and start state free0 (L,).
    A bracket from the samples, then `refine` steps of regula falsi on the clearance itself (Illinois weights), which reach the boundary in far fewer map evaluations than bisection."""
    L, d = base.shape
    dev = base.device
    t = torch.linspace(-0.5, 0.5, samples, device=dev, dtype=base.dtype)
    per = max(1, chunk // samples)
    lines, lo_all, hi_all, slo, shi, free0 = [], [], [], [], [], []
    for i in range(0, L, per):
        z = line_points(base[i:i + per], axis[i:i + per], t)
        n = z.shape[0]
        x = psi(z.reshape(-1, d))
        k = plane[i:i + per, None].expand(n, samples).reshape(-1)
        s = scene.sdf(x, k).view(n, samples)
        free = s > 0
        free0.append(free[:, 0])
        l, j = (free[:, 1:] != free[:, :-1]).nonzero(as_tuple=True)
        lines.append(l + i)
        lo_all.append(t[j])
        hi_all.append(t[j + 1])
        slo.append(s[l, j])
        shi.append(s[l, j + 1])
    line, lo, hi, a_s, b_s = torch.cat(lines), torch.cat(lo_all), torch.cat(hi_all), torch.cat(slo), torch.cat(shi)
    b, a, k = base[line], axis[line], plane[line]
    for _ in range(refine):
        w = (a_s / (a_s - b_s).where((a_s - b_s).abs() > 1e-12, torch.ones_like(a_s))).clamp(0.02, 0.98)
        mid = lo + w * (hi - lo)
        s = scene.sdf(psi(put(b, a, mid)), k)
        left = (s > 0) == (a_s > 0)
        lo, a_s, b_s = torch.where(left, mid, lo), torch.where(left, s, a_s), torch.where(left, 0.5 * b_s, s)
        hi, a_s = torch.where(left, hi, mid), torch.where(left, a_s, 0.5 * a_s)
    w = (a_s / (a_s - b_s).where((a_s - b_s).abs() > 1e-12, torch.ones_like(a_s))).clamp(0.0, 1.0)
    return line, lo + w * (hi - lo), torch.cat(free0)


def differentiable(psi, scene, base, axis, plane, line, tstar, graze=0.05, eps=2e-4):
    """Crossing positions with the implicit gradient: the value is t*, the gradient that of the crossing with respect to psi's parameters."""
    b, a, k = base[line], axis[line], plane[line]
    z = put(b, a, tstar)
    x = psi(z)
    with torch.no_grad():
        n = scene.grad(x, k)
        tangent = (psi(put(b, a, (tstar + eps).clamp(max=0.5))) - psi(put(b, a, (tstar - eps).clamp(min=-0.5)))) / ((tstar + eps).clamp(max=0.5) - (tstar - eps).clamp(min=-0.5))[:, None]
        along = (n * tangent).sum(1)
        sine = along.abs() / (n.norm(dim=1) * tangent.norm(dim=1)).clamp(min=1e-12)
        keep = sine >= graze
        scale = torch.where(keep, -1.0 / along.where(along.abs() > 1e-12, torch.ones_like(along)), torch.zeros_like(along))
    return tstar + scale * (n * (x - x.detach())).sum(1), keep


def runs(line, t, free0, count):
    """Runs of every line, flat. Every line reads -0.5, t_1 .. t_m, 0.5: m + 1 runs, each a difference of neighbouring positions.
    Returns length (R,), line of each run (R,), run is free (R,), number of runs per line (L,)."""
    dev = t.device
    m = torch.bincount(line, minlength=count)
    per = m + 2
    start = torch.cumsum(per, 0) - per
    N = int(per.sum())
    rank = torch.arange(len(line), device=dev) - (torch.cumsum(m, 0) - m)[line]
    pos = t.new_zeros(N)
    pos = pos.index_put((start[line] + 1 + rank,), t)
    idx = torch.arange(count, device=dev)
    pos = pos.index_put((start[idx],), t.new_full((count,), -0.5))
    pos = pos.index_put((start[idx] + per[idx] - 1,), t.new_full((count,), 0.5))
    owner = torch.repeat_interleave(idx, per)
    same = owner[1:] == owner[:-1]
    ell = (pos[1:] - pos[:-1])[same]
    lid = owner[:-1][same]
    r = torch.arange(N - 1, device=dev)[same] - start[lid]
    free = free0[lid] ^ (r % 2 == 1)
    return ell, lid, free, m + 1


def lines_uniform(count, d, planes, gen, dev, dtype=torch.float32):
    """Base points uniform in the box, axes and slices at random."""
    base = torch.rand(count, d, generator=gen, device=dev, dtype=dtype) - 0.5
    axis = torch.randint(d, (count,), generator=gen, device=dev)
    plane = torch.randint(planes, (count,), generator=gen, device=dev)
    return base, axis, plane


def read(psi, scene, base, axis, plane, samples, graze=0.05, differentiate=True):
    """Runs along the given lines. Returns dict(ell, lid, free, count, keep)."""
    line, tstar, free0 = crossings(psi, scene, base, axis, plane, samples)
    if differentiate:
        t, keep = differentiable(psi, scene, base, axis, plane, line, tstar, graze)
    else:
        t, keep = tstar, torch.ones_like(tstar, dtype=torch.bool)
    ell, lid, free, count = runs(line, t, free0, len(base))
    return dict(ell=ell, lid=lid, free=free, count=count, keep=keep, crossings=len(line))
