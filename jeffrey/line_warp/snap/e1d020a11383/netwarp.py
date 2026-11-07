"""A fitted plane warp (fit.py) as a fixed warp of the C-space, with the interface of cross_warp/warp.py's Warp.

    z = phi(x, y, p) = ( s * (psi(x, y) - centre) / unit,  s_p * p )

centre and unit normalize the image of the maze box to the unit box, so s means what it means for ntrl-demo's B x s: the
number of unit boxes across the warped maze. install() makes cross_warp's trainer, loader and planners build this class from a
checkpoint's warp config, so they run unchanged: the network, its loss, the planner and the scorer are ntrl-demo's and
maze_scale's, the warp only changes the coordinates that reach the encoder.
"""
import numpy as np
import torch
import torch.nn as nn

import warpnet as WN


class NetWarp(nn.Module):
    spatial = heading = xy = sep = False

    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        self.psi = WN.build(cfg['net'])
        self.s, self.s_p = float(cfg.get('s', 1.0)), float(cfg.get('s_p', cfg.get('s', 1.0)))
        self.register_buffer('centre', torch.zeros(2))
        self.register_buffer('unit', torch.ones(()))
        for p in self.parameters():
            p.requires_grad_(False)
        self._table = None

    def identity(self):
        return False

    def plane(self, xy):
        return self.s * (self.psi(xy) - self.centre) / self.unit

    def forward(self, q):
        return torch.cat([self.plane(q[:, :2]), self.s_p * q[:, 2:3]], 1)

    @torch.no_grad()
    def set_state(self, state, maze):
        self.psi.load_state_dict(state)
        t = torch.linspace(-0.5, 0.5, 4 * maze.n + 1, device=self.centre.device)
        edge = torch.cat([torch.stack([t, torch.full_like(t, v)], 1) for v in (-0.5, 0.5)] + [torch.stack([torch.full_like(t, v), t], 1) for v in (-0.5, 0.5)])
        u = self.psi(edge)
        lo, hi = u.amin(0), u.amax(0)
        self.centre.copy_(0.5 * (lo + hi))
        self.unit.copy_((hi - lo).max())

    @torch.no_grad()
    def inverse_table(self, res=1536):
        """Start points of the inverse: the image of a res x res grid of the box, rasterized on a res x res grid over the image's
        bounding box; empty raster cells take the nearest filled cell (iterated dilation)."""
        dev = self.centre.device
        t = torch.linspace(-0.5, 0.5, res, device=dev)
        X, Y = torch.meshgrid(t, t, indexing='ij')
        x = torch.stack([X.reshape(-1), Y.reshape(-1)], 1)
        z = torch.cat([self.plane(x[i:i + 262144]) for i in range(0, len(x), 262144)])
        lo, hi = z.amin(0), z.amax(0)
        ij = ((z - lo) / (hi - lo) * (res - 1)).round().long()
        tab = torch.full((res * res,), -1, device=dev, dtype=torch.long)
        tab[ij[:, 0] * res + ij[:, 1]] = torch.arange(len(x), device=dev)
        tab = tab.view(res, res)
        for _ in range(res):
            empty = tab < 0
            if not bool(empty.any()):
                break
            n = tab.clone()
            for d, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
                sh = torch.roll(tab, d, ax)
                n = torch.where((n < 0) & (sh >= 0), sh, n)
            tab = n
        self._table = (lo, hi, tab, x, res)

    @torch.no_grad()
    def inverse_plane(self, z, iters=12):
        """x with plane(x) = z by Newton from the table's start (damped; central-difference Jacobian). Rows whose z lies outside the
        image converge to the continuation of the map beyond the box."""
        if self._table is None:
            self.inverse_table()
        lo, hi, tab, grid, res = self._table
        ij = ((z - lo) / (hi - lo) * (res - 1)).round().long().clamp(0, res - 1)
        x = grid[tab[ij[:, 0], ij[:, 1]]].clone()
        eps = 2e-4
        for _ in range(iters):
            r = self.plane(x) - z
            cols = [(self.plane(x + e) - self.plane(x - e)) / (2 * eps) for e in (torch.tensor([eps, 0.0], device=z.device), torch.tensor([0.0, eps], device=z.device))]
            J = torch.stack(cols, 2)
            det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
            det = torch.where(det.abs() < 1e-8, torch.full_like(det, 1e-8), det)
            dx = torch.stack([(J[:, 1, 1] * r[:, 0] - J[:, 0, 1] * r[:, 1]) / det, (-J[:, 1, 0] * r[:, 0] + J[:, 0, 0] * r[:, 1]) / det], 1)
            n = dx.norm(dim=1, keepdim=True)
            x = x - dx * (0.02 / n.clamp(min=0.02))          # a Newton step, at most 0.02 normalized units
        return x

    def inverse_fn(self, z):
        return torch.cat([self.inverse_plane(z[:, :2]), z[:, 2:3] / self.s_p], 1)


def install(W, P=None):
    """cross_warp's warp module builds NetWarp from a config with a 'net' key; plan.load_field reads any warp checkpoint."""
    plain = W.Warp

    def dispatch(cfg):
        return NetWarp(cfg) if 'net' in cfg else plain(cfg)
    W.Warp = dispatch
    if P is not None:
        base = P.load_field

        def load(checkpoint, *a, **k):
            payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
            return W.load_field(checkpoint) if 'warp_config' in payload else base(checkpoint, *a, **k)
        P.load_field = load


def load_fit(fit_dir, device):
    """(state, net config, args) of a fit run."""
    d = torch.load(fit_dir / 'phi.pt', map_location=device, weights_only=False)
    return d['state'], d['cfg'], d.get('args', {})
