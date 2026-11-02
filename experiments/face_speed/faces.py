"""Signed distance between oriented rectangles (box faces) and axis-aligned cuboids, batched and differentiable.

For a face (center c, in-plane unit axes U, V, half sizes a, b) and a cuboid [lo, hi]:
- disjoint: the exact Euclidean distance, the minimum over the closest-feature pairs of two convex
  polytopes (rectangle vertex to cuboid, cuboid vertex to rectangle, rectangle edge to cuboid edge);
- intersecting: minus the separating-axis penetration depth (the smallest overlap over the 15 SAT axes),
  continuous with the distance at contact, so the clearance keeps a gradient inside obstacles the way the
  release's signed sphere-cuboid distance does.
The SAT separation (maximum over the axes of the projected gap) is a lower bound of the distance and is
exact for a face parallel to a panel; pairs whose bound already exceeds `far` keep the bound, since only
distances below the speed margin matter, which keeps the exact feature search small.
"""
import torch

EPS = 1e-9


def point_aabb(p, lo, hi):
    """|p - clamp(p, lo, hi)| for p (..., 3) against cuboids lo, hi broadcastable to p."""
    return torch.linalg.vector_norm((lo-p).clamp_min(0)+(p-hi).clamp_min(0), dim=-1)


def point_rect(w, c, U, V, a, b):
    """Distance from points w to rectangles (c, U, V, a, b); all broadcastable over leading dims, a, b (...,)."""
    d = w-c; u = (d*U).sum(-1).clamp(-a, a); v = (d*V).sum(-1).clamp(-b, b)
    return torch.linalg.vector_norm(d-u[..., None]*U-v[..., None]*V, dim=-1)


def segment_segment(p1, q1, p2, q2):
    """Closest distance between segments p1q1 and p2q2 (Ericson 5.1.9), broadcastable over leading dims."""
    d1, d2, r = q1-p1, q2-p2, p1-p2
    a, e, f = (d1*d1).sum(-1), (d2*d2).sum(-1), (d2*r).sum(-1)
    c, b = (d1*r).sum(-1), (d1*d2).sum(-1)
    denom = a*e-b*b
    s = torch.where(denom > EPS, ((b*f-c*e)/denom.clamp_min(EPS)).clamp(0, 1), torch.zeros_like(denom))
    t = (b*s+f)/e.clamp_min(EPS)
    s = torch.where(t < 0, (-c/a.clamp_min(EPS)).clamp(0, 1), torch.where(t > 1, ((b-c)/a.clamp_min(EPS)).clamp(0, 1), s))
    t = t.clamp(0, 1)
    return torch.linalg.vector_norm(p1+d1*s[..., None]-p2-d2*t[..., None], dim=-1)


def sat_separation(c, U, V, a, b, lo, hi):
    """Max over the 15 separating axes of (projected gap): > 0 lower-bounds the distance, <= 0 is minus the
    penetration depth. c, U, V (N, 3), a, b (N,), lo, hi (K, 3) -> (N, K)."""
    m, h = (lo+hi)/2, (hi-lo)/2
    eye = torch.eye(3, device=c.device, dtype=c.dtype)
    n = torch.cross(U, V, dim=-1)
    axes = [eye[k].expand_as(U) for k in range(3)]+[U, V, n]
    axes += [torch.cross(w, eye[k].expand_as(U), dim=-1) for w in (U, V, n) for k in range(3)]
    L = torch.stack(axes, 1)                                                    # (N, 15, 3)
    norm = torch.linalg.vector_norm(L, dim=-1, keepdim=True); L = L/norm.clamp_min(EPS)
    valid = norm[..., 0] > 1e-6                                                 # (N, 15)
    T = c[:, None, :]-m[None, :, :]                                             # (N, K, 3)
    gap = torch.einsum('nkj,naj->nka', T, L).abs()                              # (N, K, 15)
    rA = torch.einsum('kj,naj->nka', h, L.abs())
    rB = (a[:, None]*(U[:, None]*L).sum(-1).abs()+b[:, None]*(V[:, None]*L).sum(-1).abs())[:, None]  # (N, 1, 15)
    sep = (gap-rA-rB).masked_fill(~valid[:, None, :], -torch.inf)
    return sep.amax(-1)


def exact_distance(c, U, V, a, b, lo, hi):
    """Exact distance of disjoint pairs, one cuboid per rectangle: all (M, ...) -> (M,)."""
    P = torch.stack([c+a[:, None]*U+b[:, None]*V, c-a[:, None]*U+b[:, None]*V, c-a[:, None]*U-b[:, None]*V, c+a[:, None]*U-b[:, None]*V], 1)  # (M, 4, 3)
    d_vertex = point_aabb(P, lo[:, None], hi[:, None]).amin(1)
    bits = torch.tensor([[(k >> 2) & 1, (k >> 1) & 1, k & 1] for k in range(8)], device=c.device, dtype=c.dtype)
    W = lo[:, None]+(hi-lo)[:, None]*bits                                        # (M, 8, 3)
    d_corner = point_rect(W, c[:, None], U[:, None], V[:, None], a[:, None], b[:, None]).amin(1)
    start, end = [], []
    for axis in range(3):
        for k in range(8):
            if not (k >> (2-axis)) & 1:
                start.append(k); end.append(k | (1 << (2-axis)))
    E1, E2 = W[:, start], W[:, end]                                              # (M, 12, 3)
    R1, R2 = P, P.roll(-1, 1)                                                    # (M, 4, 3)
    d_edge = segment_segment(R1[:, :, None], R2[:, :, None], E1[:, None], E2[:, None]).amin((1, 2))
    return torch.minimum(torch.minimum(d_vertex, d_corner), d_edge)


def rect_aabb_signed(c, U, V, a, b, lo, hi, far=.15):
    """Signed face-cuboid distance for every pair: (N, 3)... x (K, 3) -> (N, K)."""
    sep = sat_separation(c, U, V, a, b, lo, hi)
    close = (sep > 0) & (sep < far)
    if not bool(close.any()): return sep
    n_idx, k_idx = torch.nonzero(close, as_tuple=True)
    exact = exact_distance(c[n_idx], U[n_idx], V[n_idx], a[n_idx], b[n_idx], lo[k_idx], hi[k_idx])
    return sep.index_put((n_idx, k_idx), exact)


def face_clearance(faces, lo, hi, far=.15, chunk=256):
    """Per-face signed clearance to the nearest cuboid: placement dict (B, F, ...) -> (B, F)."""
    B, F = faces['center'].shape[:2]; out = []
    for s in range(0, B, chunk):
        sl = slice(s, s+chunk); n = faces['center'][sl].shape[0]
        flat = lambda x: x[sl].reshape(n*F, 3)
        d = rect_aabb_signed(flat(faces['center']), flat(faces['U']), flat(faces['V']), faces['a'].repeat(n), faces['b'].repeat(n), lo, hi, far)
        out.append(d.amin(1).reshape(n, F))
    return torch.cat(out)
