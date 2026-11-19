"""Tc: a separate close-range travel-time network that stands in for the exact speed (ground truth) in the midpoint loss.

The midpoint loss needs travel times along short stretches of a traversal: to place travel-time midpoints and leaf
boundaries (cum, the running travel time along a path) and to cap each leaf (min(T, driven time)). The main recipe gets
them from the exact clearance at every waypoint (about 1.6 M speed labels per refresh). Here a second, small network
Tc(a, b) is trained once, before the midpoint term starts, on a fixed budget of close pairs, and the loss queries Tc
instead of the geometry. T (ntrl-demo's network) stays the network that plans.

Tc(a, b) = |b - a| (1 + softplus(h(a, b))), h symmetric in (a, b): Fourier features of both poses through an MLP. The
factor |b - a| makes it exact in scale for any length (a waypoint step of 0.06 cells or a 1-cell leaf) and the
1 + softplus keeps it at or above the straight-line distance (the trained speed is at most 1).

Labels: straight-line travel time, the trapezoid of 1 / S over poses every --tc-spacing cells along the segment
(S = ntrl-demo's trained speed of the exact clearance). It equals the true close-range travel time when nothing blocks
the straight line. Two ways to make sure the sampled pairs are such pairs (the user's (a) and (b)):

  (a) geometry: both ends have clearance >= d cells and |b - a| <= 2 d, so no wall can lie between them.
  (b) ntrl-demo: ends anywhere in the box (no clearance filter), |b - a| <= R cells, kept when an early ntrl-demo
      checkpoint's T(a, b) <= tau cells of travel time (it says the pair is close).
"""
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as Fn


class CloseNet(nn.Module):
    def __init__(self, maze, feats=128, width=256, depth=3, sigma_cells=1.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        B = torch.randn(feats, 3, generator=g) * sigma_cells * maze.E          # cycles per normalized unit
        self.register_buffer('B2pi', 2 * math.pi * B)
        layers, w = [], 4 * feats + 6
        for _ in range(depth):
            layers += [nn.Linear(w, width), nn.SiLU()]
            w = width
        layers.append(nn.Linear(w, 1))
        self.mlp = nn.Sequential(*layers)

    def feat(self, x):
        z = x @ self.B2pi.T
        return torch.cat([torch.sin(z), torch.cos(z), x], 1)

    def slowness(self, a, b):
        fa, fb = self.feat(a), self.feat(b)
        h = 0.5 * (self.mlp(torch.cat([fa, fb], 1)) + self.mlp(torch.cat([fb, fa], 1)))[:, 0]
        return 1.0 + Fn.softplus(h)

    def forward(self, a, b):
        return torch.linalg.vector_norm(b - a, dim=1) * self.slowness(a, b)

    @torch.no_grad()
    def time(self, a, b, chunk=1 << 20):
        out = [self(a[i:i + chunk], b[i:i + chunk]) for i in range(0, len(a), chunk)]
        return torch.cat(out) if out else a.new_zeros(0)

    @torch.no_grad()
    def path_time(self, paths, chunk=1 << 20):
        """cum (P, L + 1) along padded paths from Tc of consecutive waypoints (the tree.path_time analogue)."""
        P, L1, _ = paths.shape
        a = paths[:, :-1].reshape(-1, 3)
        b = paths[:, 1:].reshape(-1, 3)
        dt = self.time(a, b, chunk).reshape(P, L1 - 1)
        return torch.cat([torch.zeros(P, 1, device=paths.device), torch.cumsum(dt, 1)], 1), dt


class Counter:
    """Exact-clearance queries made for Tc (sampling filters and labels)."""

    def __init__(self, maze):
        self.maze, self.n = maze, 0

    def __call__(self, q, chunk=1 << 18):
        self.n += len(q)
        return torch.cat([self.maze.clearance(q[i:i + chunk]) for i in range(0, len(q), chunk)]) if len(q) else q.new_zeros(0)


def offsets(maze, count, r_cells, gen):
    """Random steps: uniform direction in the normalized (x, y, phi) metric, length half uniform in [0, r] and half
    log-uniform in [0.01, r] cells (the chain needs many waypoint-sized steps of ~0.06 cells)."""
    dev = maze.device
    u = Fn.normalize(torch.randn(count, 3, generator=gen, device=dev), dim=1)
    lo = math.log(min(0.01, r_cells / 2))
    v = torch.rand(count, generator=gen, device=dev)
    lin = torch.rand(count, generator=gen, device=dev) * r_cells
    log = torch.exp(lo + v * (math.log(r_cells) - lo))
    half = torch.rand(count, generator=gen, device=dev) < 0.5
    length = torch.where(half, lin, log) * maze.pitch
    return u * length[:, None]


@torch.no_grad()
def line_time(maze, clear, a, b, spacing_cells=0.025, chunk=1 << 16):
    """Straight-line travel time (normalized units, like T) and minimum clearance along the segment."""
    ys, mins = [], []
    for i in range(0, len(a), chunk):
        aa, bb = a[i:i + chunk], b[i:i + chunk]
        L = torch.linalg.vector_norm(bb - aa, dim=1)
        K = max(1, int(math.ceil(float(L.max()) / maze.pitch / spacing_cells)))
        t = torch.linspace(0, 1, K + 1, device=a.device)
        pts = aa[:, None] + t[None, :, None] * (bb - aa)[:, None]
        d = clear(pts.reshape(-1, 3)).reshape(len(aa), K + 1)
        inv = 1.0 / maze.train_speed(d)
        ys.append(L * (0.5 * (inv[:, 1:] + inv[:, :-1])).mean(1))
        mins.append(d.amin(1))
    return torch.cat(ys), torch.cat(mins)


@torch.no_grad()
def sample_a(maze, clear, count, d_cells, gen, batch=1 << 18):
    """(a): both ends with clearance >= d cells, |b - a| <= 2 d."""
    A, Bs, have = [], [], 0
    while have < count:
        a = maze.uniform(batch, gen)
        a = a[clear(a) >= d_cells]
        b = a + offsets(maze, len(a), 2 * d_cells, gen)
        ok = maze.inside(b)
        a, b = a[ok], b[ok]
        ok = clear(b) >= d_cells
        A.append(a[ok])
        Bs.append(b[ok])
        have += int(ok.sum())
    return torch.cat(A)[:count], torch.cat(Bs)[:count]


@torch.no_grad()
def sample_b(maze, frozen, count, r_cells, tau_cells, gen, batch=1 << 18):
    """(b): ends anywhere in the box, |b - a| <= R cells, kept when the early ntrl-demo field says T(a, b) <= tau."""
    A, Bs, have, seen = [], [], 0, 0
    while have < count:
        a = maze.uniform(batch, gen)
        b = a + offsets(maze, batch, r_cells, gen)
        ok = maze.inside(b)
        a, b = a[ok], b[ok]
        seen += len(a)
        keep = frozen.travel(torch.cat([a, b], 1)) <= tau_cells * maze.pitch
        A.append(a[keep])
        Bs.append(b[keep])
        have += int(keep.sum())
    return torch.cat(A)[:count], torch.cat(Bs)[:count], seen


def train_tc(maze, cfg, log, frozen_early=None, device='cuda:0'):
    """Sample, label and fit Tc. Returns (Tc, stats). Every exact-clearance query is counted."""
    t0 = time.time()
    gen = torch.Generator(device).manual_seed(cfg['seed'] + 707)
    clear = Counter(maze)
    N = int(cfg['tc_pairs'])
    if cfg['tc_mode'] == 'a':
        a, b = sample_a(maze, clear, N, cfg['tc_clear'], gen)
        seen = None
    else:
        a, b, seen = sample_b(maze, frozen_early, N, cfg['tc_radius'], cfg['tc_tau'], gen)
    q_sample = clear.n
    y, dmin = line_time(maze, clear, a, b, cfg['tc_spacing'])
    q_label = clear.n - q_sample
    blocked = dmin < 0
    t_label = time.time() - t0
    nval = max(1000, N // 20)
    perm = torch.randperm(N, generator=gen, device=device)
    tr, va = perm[nval:], perm[:nval]
    net = CloseNet(maze, cfg['tc_feats'], cfg['tc_width'], cfg['tc_depth'], cfg['tc_sigma'], seed=cfg['seed']).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg['tc_lr'])
    steps = int(cfg['tc_steps'])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg['tc_lr'], total_steps=steps, pct_start=0.05)
    ly = torch.log(y.clamp(min=1e-9))
    for s in range(steps):
        i = tr[torch.randint(len(tr), (cfg['tc_batch'],), generator=gen, device=device)]
        loss = (torch.log(net(a[i], b[i]).clamp(min=1e-9)) - ly[i]).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if (s + 1) % max(1, steps // 5) == 0:
            with torch.no_grad():
                ev = (torch.log(net.time(a[va], b[va])) - ly[va]).abs()
            log(f'Tc step {s + 1}/{steps}: train |dlog| {float(loss):.4f}, held-out median {float(ev.median()):.4f}, mean {float(ev.mean()):.4f}')
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        ev = (torch.log(net.time(a[va], b[va])) - ly[va]).abs()
        euclid = (torch.log(torch.linalg.vector_norm(b[va] - a[va], dim=1).clamp(min=1e-9)) - ly[va]).abs()
    cells = torch.linalg.vector_norm(b - a, dim=1) / maze.pitch
    stats = dict(mode=cfg['tc_mode'], pairs=N, candidates_seen=seen, clearance_queries_sampling=q_sample, clearance_queries_labels=q_label,
                 clearance_queries=clear.n, blocked_share=float(blocked.float().mean()), pair_cells_median=float(cells.median()),
                 pair_cells_max=float(cells.max()), label_cells_median=float((y / maze.pitch).median()),
                 heldout_abs_dlog_median=float(ev.median()), heldout_abs_dlog_mean=float(ev.mean()),
                 euclid_abs_dlog_median=float(euclid.median()), params=sum(p.numel() for p in net.parameters()),
                 label_seconds=t_label, seconds=time.time() - t0)
    log(f'Tc ({cfg["tc_mode"]}): {N} pairs, {clear.n / 1e6:.2f} M clearance queries ({q_sample / 1e6:.2f} M sampling, {q_label / 1e6:.2f} M labels), '
        f'blocked straight lines {100 * stats["blocked_share"]:.2f} %, held-out |dlog| median {stats["heldout_abs_dlog_median"]:.4f} '
        f'(Euclidean {stats["euclid_abs_dlog_median"]:.4f}), {stats["seconds"]:.0f} s')
    return net, stats
