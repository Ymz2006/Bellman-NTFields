"""Lines in the warped space, read through the collision oracle.

A line in z is a chord of the box: z(t) = p + t u, t in [tlo, thi], u a unit vector (an axis, or any direction). Its preimage psi(line) is a curve in x, and g(t) = 1[psi(line(t)) is an obstacle] is a step function, the graph of the proposal.
Samples find the steps; regula falsi on the clearance pins them down. Nothing here knows the dimension or what an obstacle looks like.

A run length is then a difference of crossing positions t*, and t* depends on the map through s(psi_theta(line(t*))) = 0. Implicit differentiation gives, with n the gradient of the clearance at the crossing,

    d t* = - n . d psi_theta / (n . d psi / d t),

so the gradient needs only the network at the crossing and the oracle's normal: the sampled indicator is never differentiated or smoothed. A crossing where the curve runs almost along the boundary moves
arbitrarily fast with theta: its value is kept and its gradient dropped (`graze`, the sine of the angle between curve and boundary).
"""
from collections import namedtuple
import torch

Rays = namedtuple('Rays', 'p u lo hi plane')


def axis_rays(base, axis, plane):
    """Axis-aligned chords through base (L, d) along axis (L,): t runs from -0.5 to 0.5 and z(t) has coordinate t on that axis."""
    u = torch.zeros_like(base)
    i = torch.arange(len(base), device=base.device)
    u[i, axis] = 1.0
    p = base.clone()
    p[i, axis] = 0.0
    return Rays(p, u, torch.full_like(base[:, 0], -0.5), torch.full_like(base[:, 0], 0.5), plane)


def chord(p, u):
    """Parameter range of the line p + t u inside the box."""
    safe = torch.where(u.abs() < 1e-9, torch.full_like(u, 1e-9), u)
    a, b = (-0.5 - p) / safe, (0.5 - p) / safe
    return torch.minimum(a, b).amax(1), torch.maximum(a, b).amin(1)


def any_rays(count, d, planes, gen, dev, dtype=torch.float32):
    """Chords through points uniform in the box, directions uniform on the sphere, slices at random."""
    p = torch.rand(count, d, generator=gen, device=dev, dtype=dtype) - 0.5
    u = torch.randn(count, d, generator=gen, device=dev, dtype=dtype)
    u = u / u.norm(dim=1, keepdim=True).clamp(min=1e-9)
    lo, hi = chord(p, u)
    return Rays(p, u, lo, hi, torch.randint(planes, (count,), generator=gen, device=dev))


def lines_uniform(count, d, planes, gen, dev, dtype=torch.float32):
    """Axis-aligned chords: base points uniform in the box, axes and slices at random."""
    base = torch.rand(count, d, generator=gen, device=dev, dtype=dtype) - 0.5
    axis = torch.randint(d, (count,), generator=gen, device=dev)
    plane = torch.randint(planes, (count,), generator=gen, device=dev)
    return axis_rays(base, axis, plane)


def at(r, line, t):
    return r.p[line] + t[:, None] * r.u[line]


@torch.no_grad()
def crossings(psi, scene, r, samples, refine=3, chunk=1 << 19):
    """Sign changes of the clearance along each chord. Returns line index (M,), crossing t (M,), sorted by line then t, and start state free0 (L,).
    A bracket from the samples, then `refine` steps of regula falsi on the clearance itself (Illinois weights), which reach the boundary in far fewer map evaluations than bisection."""
    L, d = r.p.shape
    dev = r.p.device
    s01 = torch.linspace(0, 1, samples, device=dev, dtype=r.p.dtype)
    per = max(1, chunk // samples)
    lines, lo_all, hi_all, slo, shi, free0 = [], [], [], [], [], []
    for i in range(0, L, per):
        t = r.lo[i:i + per, None] + s01[None] * (r.hi - r.lo)[i:i + per, None]
        z = r.p[i:i + per, None, :] + t[..., None] * r.u[i:i + per, None, :]
        n = z.shape[0]
        x = psi(z.reshape(-1, d))
        k = r.plane[i:i + per, None].expand(n, samples).reshape(-1)
        s = scene.sdf(x, k).view(n, samples)
        free = s > 0
        free0.append(free[:, 0])
        l, j = (free[:, 1:] != free[:, :-1]).nonzero(as_tuple=True)
        lines.append(l + i)
        lo_all.append(t[l, j])
        hi_all.append(t[l, j + 1])
        slo.append(s[l, j])
        shi.append(s[l, j + 1])
    line, lo, hi, a_s, b_s = torch.cat(lines), torch.cat(lo_all), torch.cat(hi_all), torch.cat(slo), torch.cat(shi)
    k = r.plane[line]
    for _ in range(refine):
        w = (a_s / (a_s - b_s).where((a_s - b_s).abs() > 1e-12, torch.ones_like(a_s))).clamp(0.02, 0.98)
        mid = lo + w * (hi - lo)
        s = scene.sdf(psi(at(r, line, mid)), k)
        left = (s > 0) == (a_s > 0)
        lo, a_s, b_s = torch.where(left, mid, lo), torch.where(left, s, a_s), torch.where(left, 0.5 * b_s, s)
        hi, a_s = torch.where(left, hi, mid), torch.where(left, a_s, 0.5 * a_s)
    w = (a_s / (a_s - b_s).where((a_s - b_s).abs() > 1e-12, torch.ones_like(a_s))).clamp(0.0, 1.0)
    return line, lo + w * (hi - lo), torch.cat(free0)


def differentiable(psi, scene, r, line, tstar, graze=0.05, eps=2e-4):
    """Crossing positions with the implicit gradient: the value is t*, the gradient that of the crossing with respect to psi's parameters."""
    k = r.plane[line]
    x = psi(at(r, line, tstar))
    with torch.no_grad():
        n = scene.grad(x, k)
        up, dn = torch.minimum(tstar + eps, r.hi[line]), torch.maximum(tstar - eps, r.lo[line])
        tangent = (psi(at(r, line, up)) - psi(at(r, line, dn))) / (up - dn).clamp(min=1e-9)[:, None]
        along = (n * tangent).sum(1)
        sine = along.abs() / (n.norm(dim=1) * tangent.norm(dim=1)).clamp(min=1e-12)
        keep = sine >= graze
        scale = torch.where(keep, -1.0 / along.where(along.abs() > 1e-12, torch.ones_like(along)), torch.zeros_like(along))
    return tstar + scale * (n * (x - x.detach())).sum(1), keep


def runs(line, t, free0, r):
    """Runs of every chord, flat. Every chord reads lo, t_1 .. t_m, hi: m + 1 runs, each a difference of neighbouring positions.
    Returns length (R,), line of each run (R,), run is free (R,), number of runs per line (L,)."""
    dev = t.device
    count = len(r.p)
    m = torch.bincount(line, minlength=count)
    per = m + 2
    start = torch.cumsum(per, 0) - per
    N = int(per.sum())
    rank = torch.arange(len(line), device=dev) - (torch.cumsum(m, 0) - m)[line]
    pos = t.new_zeros(N)
    pos = pos.index_put((start[line] + 1 + rank,), t)
    idx = torch.arange(count, device=dev)
    pos = pos.index_put((start[idx],), r.lo.to(t.dtype))
    pos = pos.index_put((start[idx] + per[idx] - 1,), r.hi.to(t.dtype))
    owner = torch.repeat_interleave(idx, per)
    same = owner[1:] == owner[:-1]
    ell = (pos[1:] - pos[:-1])[same]
    lid = owner[:-1][same]
    q = torch.arange(N - 1, device=dev)[same] - start[lid]
    free = free0[lid] ^ (q % 2 == 1)
    return ell, lid, free, m + 1


def read(psi, scene, r, samples, graze=0.05, differentiate=True):
    """Runs along the chords r. Returns dict(ell, lid, free, count, keep, crossings)."""
    line, tstar, free0 = crossings(psi, scene, r, samples)
    if differentiate:
        t, keep = differentiable(psi, scene, r, line, tstar, graze)
    else:
        t, keep = tstar, torch.ones_like(tstar, dtype=torch.bool)
    ell, lid, free, count = runs(line, t, free0, r)
    return dict(ell=ell, lid=lid, free=free, count=count, keep=keep, crossings=len(line))
