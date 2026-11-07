"""A warp of the robot's C-space in front of ntrl-demo's Fourier encoder (the README_warp.md idea, on the maze robot).

ntrl-demo's travel time is T(x0, x1) = H(x0, x1), H = its models/metric network: Fourier features sin/cos(2 pi B q)
of each configuration q = (x, y, p), a shared encoder, and a metric between the two encodings. Here every
configuration first goes through a map phi, so T(x0, x1) = H(phi(x0), phi(x1)). By the chain rule
grad_x T = J_phi^T grad_u H, so where phi stretches by sigma the same encoder can change sigma times faster in x
(sharper walls), and where it compresses the field is smoother. phi is applied inside NN.input_mapping, i.e. to both
endpoints, to the TD-shifted copies and to every MPPI rollout state, and the eikonal / normal / TD terms are taken
in x (autograd through phi), exactly as ntrl-demo takes them without a warp.

    phi(x, y, p) = ( s * psi(x, y),  s_p * exp(g(x', y')) * p )        (x', y') = psi(x, y)

* psi: the xy box [-0.5, 0.5]^2 onto itself, a diffeomorphism by construction (two coupling steps, as in a
  normalizing flow): x' = F(x | y), then y' = G(y | x'), each a monotone conditional map of one coordinate given the
  other (cond_map). Each step is the normalized integral of a positive density rho = exp(h) that is bilinear on a
  node grid, so dx'/dx = rho(x, y) / R(y) > 0 with R(y) the row integral, and every row (column) is mapped onto
  [-0.5, 0.5]. Box-preserving: psi only moves resolution around (compressing one region stretches the others);
  it cannot add resolution overall. h = hmax tanh(raw / hmax) bounds each step's stretch to [1 / k, k], k = exp(2 hmax).
  det J_psi = rho_F / R_F * rho_G / R_G (analytic, used by the penalties).
* g: an optional heading log-scale on a node grid over (x', y') (|g| <= log k_p): the heading axis stretched by a
  position-dependent factor (the C-space heading slab is thin: phi_max = 0.39 / E).
* s, s_p: global factors. With psi = id and g = 0, phi = s q exactly reproduces jeffrey/maze_scale/fourier.py's
  B x s (s_p = s), since (s q) B^T = q (s B)^T; s = 1 is plain ntrl-demo.

Learned (raw grids are parameters, the README's 'learned diffeomorphism') or designed from the geometry
(design_open: high density where the robot is near walls, low where it is far from every wall, i.e. in the open +).
"""
import math
import numpy as np
import torch
import torch.nn as nn

ROBOT_CIRC = math.hypot(0.25, 0.1)          # robot's circumscribed radius (cells): the free radius is clearance minus this


def cond_map(h, z, c):
    """z' = -0.5 + int_{-0.5}^{z} rho(t, c) dt / R(c) and dz'/dz, rho = exp(h) bilinear on the (Nc + 1) x (Nz + 1) node
    grid over (c, z) in [-0.5, 0.5]^2 (rows: conditioning coordinate c; columns: mapped coordinate z), R(c) = the
    integral over the whole row, so [-0.5, 0.5] maps onto itself for every c. The integral is exact (rho is linear in
    z within a grid cell for fixed c, and the map is linear in c between node rows). Outside [-0.5, 0.5], z continues
    with the edge slope and c is clamped."""
    Nc, Nz = h.shape[0] - 1, h.shape[1] - 1
    rho = torch.exp(h)
    dz = 1.0 / Nz
    C = torch.cat([rho.new_zeros(Nc + 1, 1), torch.cumsum(0.5 * (rho[:, :-1] + rho[:, 1:]) * dz, dim=1)], 1)
    R = C[:, -1]
    tc = ((c + 0.5) * Nc).clamp(0, Nc)
    j0 = torch.floor(tc).clamp(max=Nc - 1).long()
    v = tc - j0
    zc = z.clamp(-0.5, 0.5)
    tz = (zc + 0.5) * Nz
    i0 = torch.floor(tz).clamp(max=Nz - 1).long()
    u = tz - i0

    def row(j):
        r0, r1 = rho[j, i0], rho[j, i0 + 1]
        return C[j, i0] + dz * (r0 * u + 0.5 * (r1 - r0) * u * u), r0 + (r1 - r0) * u, R[j]

    c0, d0, R0 = row(j0)
    c1, d1, R1 = row(j0 + 1)
    Rz = (1 - v) * R0 + v * R1
    slope = ((1 - v) * d0 + v * d1) / Rz
    return -0.5 + ((1 - v) * c0 + v * c1) / Rz + (z - zc) * slope, slope


def interp2(g, x, y):
    """Bilinear interpolation of a node grid g (Nx + 1, Ny + 1) over [-0.5, 0.5]^2 at (x, y) (clamped)."""
    Nx, Ny = g.shape[0] - 1, g.shape[1] - 1
    tx = ((x + 0.5) * Nx).clamp(0, Nx)
    ty = ((y + 0.5) * Ny).clamp(0, Ny)
    i = torch.floor(tx).clamp(max=Nx - 1).long()
    j = torch.floor(ty).clamp(max=Ny - 1).long()
    a, b = tx - i, ty - j
    return (1 - a) * (1 - b) * g[i, j] + a * (1 - b) * g[i + 1, j] + (1 - a) * b * g[i, j + 1] + a * b * g[i + 1, j + 1]


class Warp(nn.Module):
    """phi of the module docstring. cfg keys: s, s_p, xy (bool), nodes (grid intervals per axis), k (xy stretch bound per
    step), heading (bool), k_p (heading scale bound), learn (bool), design (dict or None)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        self.s, self.s_p = float(cfg.get('s', 1.0)), float(cfg.get('s_p', cfg.get('s', 1.0)))
        self.xy, self.heading = bool(cfg.get('xy', False)), bool(cfg.get('heading', False))
        self.sep = bool(cfg.get('sep', False))
        n = int(cfg.get('nodes', 22))
        n_sep = int(cfg.get('sep_nodes', 8 * n))
        self.hmax_sep = 0.5 * math.log(float(cfg.get('k_sep', 4.0)))
        self.hmax = 0.5 * math.log(float(cfg.get('k', 4.0)))
        self.hmax_p = math.log(float(cfg.get('k_p', 4.0)))
        learn = bool(cfg.get('learn', False))
        if self.xy:
            self.raw_f = nn.Parameter(torch.zeros(n + 1, n + 1), requires_grad=learn)     # F: rows over y, columns over x
            self.raw_g = nn.Parameter(torch.zeros(n + 1, n + 1), requires_grad=learn)     # G: rows over x', columns over y
        if self.heading:
            self.raw_p = nn.Parameter(torch.zeros(n + 1, n + 1), requires_grad=learn)     # over (x', y')
        if self.sep:
            self.raw_sx = nn.Parameter(torch.zeros(n_sep + 1), requires_grad=learn)        # separable 1-D log-densities
            self.raw_sy = nn.Parameter(torch.zeros(n_sep + 1), requires_grad=learn)

    # bounded grids
    def h_f(self):
        return self.hmax * torch.tanh(self.raw_f / self.hmax)

    def h_g(self):
        return self.hmax * torch.tanh(self.raw_g / self.hmax)

    def g_p(self):
        return self.hmax_p * torch.tanh(self.raw_p / self.hmax_p)

    def h_sx(self):
        return self.hmax_sep * torch.tanh(self.raw_sx / self.hmax_sep)

    def h_sy(self):
        return self.hmax_sep * torch.tanh(self.raw_sy / self.hmax_sep)

    def psi(self, x, y):
        """(x', y', a, b) of the xy steps: the separable step (x -> F(x), y -> G(y), 1-D monotone maps of the box, no
        shear) and then the coupling steps; a + b = log det J_psi (a, b: the steps' log stretches summed)."""
        la = torch.zeros_like(x)
        lb = torch.zeros_like(x)
        if self.sep:
            zero = torch.zeros_like(x)
            hx, hy = self.h_sx(), self.h_sy()
            x, sx = cond_map(torch.stack([hx, hx]), x, zero)
            y, sy = cond_map(torch.stack([hy, hy]), y, zero)
            la, lb = la + torch.log(sx), lb + torch.log(sy)
        if self.xy:
            x1, sf = cond_map(self.h_f(), x, y)
            y1, sg = cond_map(self.h_g(), y, x1)
            x, y = x1, y1
            la, lb = la + torch.log(sf), lb + torch.log(sg)
        return x, y, la, lb

    def forward(self, q):
        x1, y1, _, _ = self.psi(q[:, 0], q[:, 1])
        p = q[:, 2]
        if self.heading:
            p = p * torch.exp(interp2(self.g_p(), x1, y1))
        return torch.stack([self.s * x1, self.s * y1, self.s_p * p], 1)

    def logdet(self, q):
        """log det J of psi (xy, without s) and the heading log-scale g at q (both (N,))."""
        x1, y1, lf, lg = self.psi(q[:, 0], q[:, 1])
        gp = interp2(self.g_p(), x1, y1) if self.heading else torch.zeros_like(lf)
        return lf + lg, gp

    def smoothness(self):
        """Mean squared second difference of the bounded grids (a curvature penalty on the log-densities)."""
        out = 0.0
        grids = ([self.h_f(), self.h_g()] if self.xy else []) + ([self.g_p()] if self.heading else [])
        for h in grids:
            out = out + (h[2:] - 2 * h[1:-1] + h[:-2]).pow(2).mean() + (h[:, 2:] - 2 * h[:, 1:-1] + h[:, :-2]).pow(2).mean()
        for h in ([self.h_sx(), self.h_sy()] if self.sep else []):
            out = out + (h[2:] - 2 * h[1:-1] + h[:-2]).pow(2).mean()
        return out

    def identity(self):
        return self.s == 1.0 and self.s_p == 1.0 and not self.xy and not self.heading and not self.sep

    @property
    def spatial(self):
        return self.xy or self.sep


def jacobian(warp, q):
    """Full 3 x 3 Jacobian of phi at q (N, 3) by autograd (diagnostics)."""
    q = q.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        out = warp(q)
        rows = [torch.autograd.grad(out[:, k].sum(), q, retain_graph=k < 2)[0] for k in range(3)]
    return torch.stack(rows, 1).detach()


# --- geometry ---------------------------------------------------------------------------------------------------------
def wall_distance_cells(maze, xy_cells, chunk=65536):
    """Distance (cells) from points (N, 2) in cell units to the nearest wall box (0 inside a wall)."""
    boxes = torch.tensor(maze.boxes, device=xy_cells.device, dtype=xy_cells.dtype)
    out = []
    for i in range(0, len(xy_cells), chunk):
        p = xy_cells[i:i + chunk]
        dx = torch.clamp(torch.maximum(boxes[None, :, 0] - p[:, None, 0], p[:, None, 0] - boxes[None, :, 2]), min=0)
        dy = torch.clamp(torch.maximum(boxes[None, :, 1] - p[:, None, 1], p[:, None, 1] - boxes[None, :, 3]), min=0)
        out.append(torch.sqrt(dx * dx + dy * dy).amin(1))
    return torch.cat(out)


def free_radius(maze, q):
    """Free radius (cells) of configurations q (N, 3): wall distance of the robot centre minus its circumscribed radius."""
    x, y, _ = maze.to_cells(q)
    return wall_distance_cells(maze, torch.stack([x, y], 1)) - ROBOT_CIRC


def openness(maze, xy_norm, r0, r1):
    """0 (near walls) .. 1 (open) smoothstep of the free radius between r0 and r1 cells, at normalized xy (N, 2)."""
    q = torch.cat([xy_norm, torch.zeros_like(xy_norm[:, :1])], 1)
    t = ((free_radius(maze, q) - r0) / (r1 - r0)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def design_open(warp, maze, ratio, r0=0.5, r1=1.5, heading_ratio=None, fine=4097):
    """Set the warp's grids from the geometry: density ratio `ratio` between the maze (free radius <= r0 cells) and the
    open area (>= r1), log-linear in between, for both coupling steps (G's grid is laid over (x', y): its nodes are
    mapped back through F^-1 before the geometry is evaluated); heading scale ratio heading_ratio likewise over (x', y')
    (maze: high, open: low). Densities are centred (|h| <= log(ratio) / 2), and the raw grids are set so that the
    bounded grids equal them (needs ratio < k)."""
    dev = next(warp.parameters()).device
    lr = math.log(ratio)
    assert lr / 2 < warp.hmax, f'ratio {ratio} needs k > {ratio}'
    with torch.no_grad():
        n = warp.raw_f.shape[0] - 1
        nodes = torch.linspace(-0.5, 0.5, n + 1, device=dev)
        Y, X = torch.meshgrid(nodes, nodes, indexing='ij')            # F grid: rows y, columns x
        o = openness(maze, torch.stack([X.reshape(-1), Y.reshape(-1)], 1), r0, r1).reshape(n + 1, n + 1)
        hf = lr * (0.5 - o)
        warp.raw_f.copy_(warp.hmax * torch.atanh((hf / warp.hmax).clamp(-0.999999, 0.999999)))
        # G grid: rows over x', columns over y. For each (x'_i, y_j) find x with F(x | y_j) = x'_i (monotone: interpolate)
        fx = torch.linspace(-0.5, 0.5, fine, device=dev)
        xs = torch.empty(n + 1, n + 1, device=dev)                  # [i over x', j over y]
        for j in range(n + 1):
            xp, _ = cond_map(warp.h_f(), fx, torch.full_like(fx, float(nodes[j])))
            xs[:, j] = torch.tensor(np.interp(nodes.cpu().numpy(), xp.cpu().numpy(), fx.cpu().numpy()), device=dev)
        Yg = nodes[None, :].expand(n + 1, n + 1)
        o2 = openness(maze, torch.stack([xs.reshape(-1), Yg.reshape(-1)], 1), r0, r1).reshape(n + 1, n + 1)
        hg = lr * (0.5 - o2)
        warp.raw_g.copy_(warp.hmax * torch.atanh((hg / warp.hmax).clamp(-0.999999, 0.999999)))
        if heading_ratio is not None and warp.heading:
            design_heading(warp, maze, heading_ratio, r0, r1)


def design_heading(warp, maze, ratio, r0=0.5, r1=1.5, base=1.0):
    """Heading scale grid over (x', y') = psi(x, y): log-scale log(base) + log(ratio) / 2 in the maze (free radius <= r0
    cells), log(base) - log(ratio) / 2 in the open area (>= r1), smoothstep between; the geometry of each node is taken
    at its preimage under psi (nearest of a 401 x 401 sample of the box)."""
    dev = next(warp.parameters()).device
    with torch.no_grad():
        n = warp.raw_p.shape[0] - 1
        nodes = torch.linspace(-0.5, 0.5, n + 1, device=dev)
        m = 401
        g1 = torch.linspace(-0.5, 0.5, m, device=dev)
        GX, GY = torch.meshgrid(g1, g1, indexing='ij')
        src = torch.stack([GX.reshape(-1), GY.reshape(-1)], 1)
        x1, y1, _, _ = warp.psi(src[:, 0], src[:, 1])
        tgt = torch.stack(torch.meshgrid(nodes, nodes, indexing='ij'), -1).reshape(-1, 2)
        img = torch.stack([x1, y1], 1)
        idx = torch.cat([torch.cdist(tgt[i:i + 512], img).argmin(1) for i in range(0, len(tgt), 512)])
        o = openness(maze, src[idx], r0, r1).reshape(n + 1, n + 1)
        gp = math.log(base) + math.log(ratio) * (0.5 - o)
        assert float(gp.abs().max()) < warp.hmax_p, 'heading design needs a larger k_p'
        warp.raw_p.copy_(warp.hmax_p * torch.atanh((gp / warp.hmax_p).clamp(-0.999999, 0.999999)))


def axis_wall_distance(maze, xy_cells, axis, chunk=65536):
    """Distance (cells) from points (N, 2) to the nearest wall that crosses the given axis: axis 0 -> vertical walls
    (the V boxes, which a move along x runs into), axis 1 -> horizontal walls (H boxes)."""
    import maze as MZ
    V, H = (maze.V, np.zeros_like(maze.H)) if axis == 0 else (np.zeros_like(maze.V), maze.H)
    boxes = torch.tensor(MZ.wall_boxes(V, H), device=xy_cells.device, dtype=xy_cells.dtype)
    out = []
    for i in range(0, len(xy_cells), chunk):
        p = xy_cells[i:i + chunk]
        dx = torch.clamp(torch.maximum(boxes[None, :, 0] - p[:, None, 0], p[:, None, 0] - boxes[None, :, 2]), min=0)
        dy = torch.clamp(torch.maximum(boxes[None, :, 1] - p[:, None, 1], p[:, None, 1] - boxes[None, :, 3]), min=0)
        out.append(torch.sqrt(dx * dx + dy * dy).amin(1))
    return torch.cat(out)


def design_slow(warp, maze, band, kd, alpha=1.0, open_ratio=None, r0=0.5, r1=1.5, fine=8193):
    """README_warp section 10 'anisotropic, geometry-tied stretch', on the box-preserving coupling grids: step F (along x)
    gets density (band / d_x)^alpha clipped to [1, kd], d_x = distance (cells) to the nearest vertical wall, step G (along
    y) the same with horizontal walls, so each axis is stretched across the walls that block it, within `band` cells of
    them (where the robot is slowed). open_ratio (optional) also divides the density by open_ratio in the open area
    (design_open's openness). Log-densities are centred per grid and must fit the bound (needs k > the spread)."""
    dev = next(warp.parameters()).device
    with torch.no_grad():
        n = warp.raw_f.shape[0] - 1
        nodes = torch.linspace(-0.5, 0.5, n + 1, device=dev)

        def logdens(xy_norm, axis):
            x = xy_norm[:, 0] * maze.E + maze.n / 2
            y = xy_norm[:, 1] * maze.E + maze.n / 2
            d = axis_wall_distance(maze, torch.stack([x, y], 1), axis)
            h = alpha * torch.log((band / d.clamp(min=1e-3)).clamp(1.0, kd))
            if open_ratio is not None:
                h = h - math.log(open_ratio) * openness(maze, xy_norm, r0, r1)
            return h

        def setgrid(raw, h):
            h = h - 0.5 * (h.max() + h.min())
            assert float(h.abs().max()) < warp.hmax, f'design spread {2 * float(h.abs().max()):.2f} needs k > {math.exp(2 * float(h.abs().max())):.1f}'
            raw.copy_(warp.hmax * torch.atanh((h / warp.hmax).clamp(-0.999999, 0.999999)))

        Y, X = torch.meshgrid(nodes, nodes, indexing='ij')                # F grid: rows y, columns x
        setgrid(warp.raw_f, logdens(torch.stack([X.reshape(-1), Y.reshape(-1)], 1), 0).reshape(n + 1, n + 1))
        fx = torch.linspace(-0.5, 0.5, fine, device=dev)
        xs = torch.empty(n + 1, n + 1, device=dev)                      # G grid [i over x', j over y]: x = F^-1(x'_i | y_j)
        for j in range(n + 1):
            xp, _ = cond_map(warp.h_f(), fx, torch.full_like(fx, float(nodes[j])))
            xs[:, j] = torch.tensor(np.interp(nodes.cpu().numpy(), xp.cpu().numpy(), fx.cpu().numpy()), device=dev)
        Yg = nodes[None, :].expand(n + 1, n + 1)
        setgrid(warp.raw_g, logdens(torch.stack([xs.reshape(-1), Yg.reshape(-1)], 1), 1).reshape(n + 1, n + 1))


def design_grid(warp, maze, band, kd, open_ratio=None, plus_width=None):
    """Separable design: along x, density kd within `band` cells of every vertical grid line x = 0..n (cell units)
    falling linearly to 1 at 2 band (walls only ever lie on grid lines; a doorway's line is stretched too), divided by
    open_ratio over the + band's interior columns (plus_width cells centred, one cell in from its outline); the same
    along y. Log-densities centred; needs k_sep > kd * open_ratio."""
    dev = next(warp.parameters()).device
    with torch.no_grad():
        m = warp.raw_sx.shape[0] - 1
        z = torch.linspace(-0.5, 0.5, m + 1, device=dev) * maze.E + maze.n / 2      # cell units
        d = (z - torch.round(z)).abs()
        h = math.log(kd) * ((2 * band - d) / band).clamp(0, 1)
        if open_ratio is not None and plus_width:
            c0 = (maze.n - plus_width) // 2
            inner = (z > c0 + 1) & (z < c0 + plus_width - 1)
            h = h - math.log(open_ratio) * inner.float()
        h = h - 0.5 * (h.max() + h.min())
        assert float(h.abs().max()) < warp.hmax_sep, 'design spread needs a larger k_sep'
        raw = warp.hmax_sep * torch.atanh((h / warp.hmax_sep).clamp(-0.999999, 0.999999))
        warp.raw_sx.copy_(raw)
        warp.raw_sy.copy_(raw)


# --- network ----------------------------------------------------------------------------------------------------------
def warped_class(mn):
    """Subclass of the package's models.metric NN whose input mapping applies self.warp first (None: unchanged)."""
    class WarpNN(mn.NN):
        def __init__(self, device, dim, B):
            super().__init__(device, dim, B)
            self.warp = None

        def input_mapping(self, x):
            if self.warp is not None:
                x = self.warp(x)
            return super().input_mapping(x)
    return WarpNN


def network_module():
    """ntrl-demo's models/metric network module (unchanged on disk), imported under its package name."""
    import sys
    from pathlib import Path
    ntrl = Path(__file__).resolve().parents[2] / 'baselines/ntrl-demo'
    if not (ntrl / 'models').is_dir():
        import os
        ntrl = Path(os.environ.get('NTFIELD_ROOT', '/home/jeffrey/bellman-ntfield')) / 'baselines/ntrl-demo'
    if str(ntrl) not in sys.path:
        sys.path.insert(0, str(ntrl))
    from models.metric import model_network_metric as mn
    assert Path(mn.__file__).resolve() == ntrl / 'models/metric/model_network_metric.py', 'ntrl-demo models package shadowed'
    return mn


def load_field(checkpoint, device='cuda:0'):
    """A trained warped field (checkpoint written by trainer.py), frozen, for planning and diagnostics."""
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    mn = network_module()
    net = warped_class(mn)(device, 3, payload['B_state_dict'])
    net.warp = Warp(payload['warp_config'])
    net.load_state_dict(payload['model_state_dict'], strict=True)
    net.to(torch.device(device)).float().eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net
