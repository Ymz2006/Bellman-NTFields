"""The field trained on gap x1, used on a wider gap: T_k(x0, x1) = T_1(psi(x0), psi(x1)).

psi is the inverse of gapprepare.map_from_k1, for every base: along x and along y, the quadrants and the band's edge
cells (one cell inside its outline) keep their place relative to their walls, and the band's interior, 6 k - 2 cells on
gap x k, is squeezed linearly onto the 4 cells it has on gap x1. The heading angle is kept. All walls lie in the rigid
part or on the outer border, which the map only slides along, so psi carries the free poses of gap x k onto those of
gap x1, and a path on one is a path on the other.

A stretched field is a z-trained field whose warp is psi (followed by the field's own warp, if it has one), so zplan and
zq plan it as they plan those: 'x' = plan.mppi on maze k, the field looked up at psi(x); 'z' = plan.mppi in the field's
own coordinates, the path mapped back. t_scale (zq) converts the time unit: E_k / E_1 for ntrl-demo, so the softmax is
as selective per cell inside the quadrants as on gap x k's own fields.

kind 'uniform' is the control without the plus-shaped stretch: the checkpoint evaluated at the same normalized
coordinates, so the x1 maze is simply blown up to fill the wider box and its walls miss gap x k's.
"""
import torch

import gap as GP
import warpplan as WP
import zplan as ZP


class Stretch(torch.nn.Module):
    """x on gap x k (normalized, (N, 3)) -> x on gap x1, and back (inverse_fn)."""

    def __init__(self, maze, k, kind='plus'):
        super().__init__()
        assert kind in ('plus', 'uniform')
        self.kind, self.k = kind, k
        self.n, self.E = maze.n, maze.E
        self.n1 = 2 * GP.BLOCK + GP.BASE_WIDTH
        self.E1 = self.n1 + (maze.E - maze.n)
        self.a = GP.BLOCK + 1.0                      # the band's interior starts here on both mazes
        self.f = (GP.BASE_WIDTH * k - 2) / (GP.BASE_WIDTH - 2)
        self.b_k, self.b_1 = self.a + GP.BASE_WIDTH * k - 2, self.a + GP.BASE_WIDTH - 2

    def _in(self, c):
        return torch.where(c <= self.a, c, torch.where(c < self.b_k, self.a + (c - self.a) / self.f, c - (self.b_k - self.b_1)))

    def _out(self, c):
        return torch.where(c <= self.a, c, torch.where(c < self.b_1, self.a + (c - self.a) * self.f, c + (self.b_k - self.b_1)))

    def forward(self, q):
        if self.kind == 'uniform':
            return torch.stack([q[:, 0], q[:, 1], q[:, 2] * self.E / self.E1], 1)
        c = q[:, :2].double() * self.E + self.n / 2
        z = (self._in(c) - self.n1 / 2) / self.E1
        return torch.cat([z.to(q.dtype), q[:, 2:3] * self.E / self.E1], 1)

    def inverse_fn(self, z):
        if self.kind == 'uniform':
            return torch.stack([z[:, 0], z[:, 1], z[:, 2] * self.E1 / self.E], 1)
        c = z[:, :2].double() * self.E1 + self.n1 / 2
        q = (self._out(c) - self.n / 2) / self.E
        return torch.cat([q.to(z.dtype), z[:, 2:3] * self.E1 / self.E], 1)


class Composed(torch.nn.Module):
    """own(psi(x)) with its inverse; cfg z_space: the field's time unit is that of these coordinates."""
    cfg = dict(z_space=True)

    def __init__(self, psi, own, own_inverse):
        super().__init__()
        self.psi, self.own, self.own_inverse = psi, own, own_inverse

    def identity(self):
        return False

    def forward(self, q):
        return self.own(self.psi(q))

    def inverse_fn(self, z):
        return self.psi.inverse_fn(self.own_inverse(z))


class Stretched:
    """A trained field on gap x1 as a field on gap x k. out(x): the field at psi(x); bypass: inputs are already in the
    field's own coordinates (own warp of psi(x))."""

    def __init__(self, net, psi):
        self.base, self.psi = net, psi
        self.has_own = getattr(net, 'warp', None) is not None
        self.warp = Composed(psi, ZP.field_warp(net), ZP.inverse(getattr(net, 'warp', None)))
        self.bypass = False

    def out(self, pairs):
        if self.bypass:
            prev = getattr(self.base, 'bypass', False)
            if self.has_own:
                self.base.bypass = True
            try:
                return self.base.out(pairs)
            finally:
                if self.has_own:
                    self.base.bypass = prev
        return self.base.out(torch.cat([self.psi(pairs[:, :3]), self.psi(pairs[:, 3:])], 1))
