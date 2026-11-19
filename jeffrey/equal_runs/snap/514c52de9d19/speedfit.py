"""Is the speed easier to learn through this map? A learnability score that needs no travel-time field.

The request's reason for equal runs: 'you want to learn speed, so it would be better if the speed was more evenly distributed, peaking more periodically'. So this measures that directly: a network of ntrl-demo's
shape (the same Fourier encoder size and softplus layers) regresses the speed label S(x, theta) from the warped coordinates z = phi(x) and the heading, for a fixed budget, and is scored on held-out points. Same network, same budget, same
points per unit of z for every map. A regression is not a planner: speedfit_calibrate.py checks this score against the maps of line_warp, whose planning results are known.

    fit(xy_of_z, maze, s)      xy_of_z: z (N, 2) -> x (N, 2) and a validity mask; z is drawn uniformly in the box, as in training on the warped maze
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as Fn


class SpeedNet(nn.Module):
    """ntrl-demo's encoder at its own size: 128 Fourier rows, 256 wide, softplus(beta 10)."""

    def __init__(self, B, width=256, depth=4):
        super().__init__()
        self.register_buffer('B', B)
        self.inp = nn.Linear(2 * B.shape[0], width)
        self.mid = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.out = nn.Linear(width, 1)

    def forward(self, u):
        p = (2 * math.pi) * (u @ self.B.T)
        h = Fn.softplus(self.inp(torch.cat([torch.sin(p), torch.cos(p)], -1)), beta=10)
        for m in self.mid:
            h = h + Fn.softplus(m(h), beta=10)
        return torch.sigmoid(self.out(h))[:, 0]


def draw(xy_of_z, maze, count, gen):
    """count points: z uniform in the box, heading uniform; returns inputs (z, p) and the speed label."""
    dev = maze.device
    zs, ss = [], []
    have = 0
    while have < count:
        z = torch.rand(count, 2, generator=gen, device=dev) - 0.5
        x, ok = xy_of_z(z)
        th = (torch.rand(len(z), generator=gen, device=dev) - 0.5) * math.pi
        p = th * 0.25 / maze.E
        d = maze.clearance(torch.cat([x, p[:, None]], 1))
        s = maze.unit_speed(d)
        keep = ok
        zs.append(torch.cat([z[keep], (p[keep] / maze.phi_max * 0.5)[:, None]], 1))
        ss.append(s[keep])
        have += int(keep.sum())
    return torch.cat(zs)[:count], torch.cat(ss)[:count]


def fit(xy_of_z, maze, s=1.0, steps=3000, batch=16384, lr=1e-3, seed=0, rows=128, test=200000, keep=None):
    """Returns held-out scores. s scales the Fourier rows (ntrl-demo's B x s). keep: a list that receives the trained network."""
    dev = maze.device
    gen = torch.Generator(dev).manual_seed(seed + 100)
    gcpu = torch.Generator().manual_seed(seed)
    B = (torch.randn(rows, 3, generator=gcpu) * s).to(dev)
    state = torch.get_rng_state()
    torch.manual_seed(seed)
    net = SpeedNet(B).to(dev)
    torch.set_rng_state(state)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.02)
    # a fixed pool, drawn once, so that a slow inverse is paid once: a map is scored on what it shows, not on how fast it runs
    pool_z, pool_s = draw(xy_of_z, maze, 1 << 21, gen)
    for _ in range(steps):
        i = torch.randint(len(pool_z), (batch,), generator=gen, device=dev)
        loss = (net(pool_z[i]) - pool_s[i]).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    if keep is not None:
        keep.append(net)
    with torch.no_grad():
        tz, ts = draw(xy_of_z, maze, test, torch.Generator(dev).manual_seed(seed + 200))
        pr = torch.cat([net(tz[i:i + 65536]) for i in range(0, len(tz), 65536)])
        err = (pr - ts).pow(2).mean()
        floor, top = ts <= 0.2, ts >= 0.99
        r2 = 1 - err / ts.var()
        lp, lt = torch.log(pr.clamp(min=1e-3)), torch.log(ts)
        corr = torch.corrcoef(torch.stack([lp, lt]))[0, 1]
        return dict(r2=float(r2), log_corr=float(corr), rmse=float(err.sqrt()), wall_pred=float(pr[floor].median()) if bool(floor.any()) else None, free_pred=float(pr[top].median()) if bool(top.any()) else None,
                    wall_share=float(floor.float().mean()), s=s, steps=steps)


def fit_scene(psi, scene, s=1.0, steps=3000, batch=16384, lr=1e-3, seed=0, rows=128, test=200000, pool=1 << 21, keep=None):
    """The same score for any scene (the whole C-space, a 3-D workspace, a floor plan): inputs are z alone, the label is clip(clearance / margin, 0.1, 1) with the oracle's own margin."""
    dev = scene.table.v.device
    gen = torch.Generator(dev).manual_seed(seed + 100)
    gcpu = torch.Generator().manual_seed(seed)
    B = (torch.randn(rows, scene.dim, generator=gcpu) * s).to(dev)
    state = torch.get_rng_state()
    torch.manual_seed(seed)
    net = SpeedNet(B).to(dev)
    torch.set_rng_state(state)

    @torch.no_grad()
    def draw3(count, g):
        z = torch.rand(count, scene.dim, generator=g, device=dev) - 0.5
        x = torch.cat([psi(z[i:i + 262144]) for i in range(0, count, 262144)])
        k = torch.zeros(count, device=dev, dtype=torch.long)
        return z, (scene.sdf(x, k) / scene.margin).clamp(0.1, 1.0)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.02)
    pool_z, pool_s = draw3(pool, gen)
    for _ in range(steps):
        i = torch.randint(pool, (batch,), generator=gen, device=dev)
        loss = (net(pool_z[i]) - pool_s[i]).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    if keep is not None:
        keep.append(net)
    with torch.no_grad():
        tz, ts = draw3(test, torch.Generator(dev).manual_seed(seed + 200))
        pr = torch.cat([net(tz[i:i + 65536]) for i in range(0, len(tz), 65536)])
        err = (pr - ts).pow(2).mean()
        floor, top = ts <= 0.2, ts >= 0.99
        lp, lt = torch.log(pr.clamp(min=1e-3)), torch.log(ts)
        return dict(r2=float(1 - err / ts.var()), log_corr=float(torch.corrcoef(torch.stack([lp, lt]))[0, 1]), rmse=float(err.sqrt()), wall_pred=float(pr[floor].median()) if bool(floor.any()) else None,
                    free_pred=float(pr[top].median()) if bool(top.any()) else None, wall_share=float(floor.float().mean()), s=s, steps=steps)


@torch.no_grad()
def wall_band(maze, count=200000, seed=9, band=2.0):
    """A fixed test set in x, the same for every map: positions and headings with clearance inside `band` margins of a wall (and a little inside it), with their speed labels. Uniform in z favours a map that leaves walls little room; this does not."""
    dev = maze.device
    gen = torch.Generator(dev).manual_seed(seed)
    xs, ps, ss, have = [], [], [], 0
    while have < count:
        x = torch.rand(1 << 20, 2, generator=gen, device=dev) - 0.5
        th = (torch.rand(1 << 20, generator=gen, device=dev) - 0.5) * math.pi
        p = th * 0.25 / maze.E
        d = maze.clearance(torch.cat([x, p[:, None]], 1))
        k = (d > -0.05) & (d < band * 0.175)
        xs.append(x[k])
        ps.append(p[k])
        ss.append(maze.unit_speed(d[k]))
        have += int(k.sum())
    return torch.cat(xs)[:count], torch.cat(ps)[:count], torch.cat(ss)[:count]


@torch.no_grad()
def wall_score(net, phi, maze, test):
    """R2 and mean absolute error of the speed on the fixed set, read through phi: x -> z."""
    x, p, s = test
    z = torch.cat([phi(x[i:i + 65536]) for i in range(0, len(x), 65536)])
    u = torch.cat([z, (p / maze.phi_max * 0.5)[:, None]], 1)
    pr = torch.cat([net(u[i:i + 65536]) for i in range(0, len(u), 65536)])
    out = dict(r2=float(1 - (pr - s).pow(2).mean() / s.var()), mae=float((pr - s).abs().mean()), corr=float(torch.corrcoef(torch.stack([pr, s]))[0, 1]))
    info = maze.describe()
    if info.get('variant') == 'gap':                      # walls inside the quadrants, and walls along the open band
        import gap
        cells = (x * maze.E + maze.n / 2).cpu().numpy()
        quad = torch.tensor(gap.region_of(maze.n, cells[:, 0], cells[:, 1]) < 4, device=x.device)
        for name, k in (('quadrants', quad), ('band', ~quad)):
            out[f'r2_{name}'] = float(1 - (pr[k] - s[k]).pow(2).mean() / s[k].var())
            out[f'mae_{name}'] = float((pr[k] - s[k]).abs().mean())
            out[f'share_{name}'] = float(k.float().mean())
    return out


@torch.no_grad()
def wall_band_scene(scene, count=200000, seed=9, band=2.0):
    """wall_band for any scene with no headings: fixed points of the scene within `band` margins of a wall, with their labels."""
    dev = scene.table.v.device
    gen = torch.Generator(dev).manual_seed(seed)
    xs, ss, have, tries = [], [], 0, 0
    while have < count and tries < 400:
        x = torch.rand(1 << 19, scene.dim, generator=gen, device=dev) - 0.5
        d = scene.sdf(x, torch.zeros(len(x), device=dev, dtype=torch.long))
        k = (d > -0.3 * scene.margin) & (d < band * scene.margin)
        xs.append(x[k])
        ss.append((d[k] / scene.margin).clamp(0.1, 1.0))
        have += int(k.sum())
        tries += 1
    return torch.cat(xs)[:count], torch.cat(ss)[:count]


@torch.no_grad()
def wall_score_scene(net, phi, test):
    x, s = test
    z = torch.cat([phi(x[i:i + 65536]) for i in range(0, len(x), 65536)])
    pr = torch.cat([net(z[i:i + 65536]) for i in range(0, len(z), 65536)])
    return dict(r2=float(1 - (pr - s).pow(2).mean() / s.var()), mae=float((pr - s).abs().mean()))


def near_wall(psi, phi, scene, s=4.0, steps=3000, seed=0):
    """Train the regression on z as usual, score it on the fixed near-wall points of the scene. phi: x -> z. Every map of a scene sees the same points, so a map is not marked down for giving walls more of the box."""
    nets = []
    if scene.dim == 3 or not hasattr(scene, 'maze'):
        r = fit_scene(psi, scene, s, steps, seed=seed, keep=nets)
        return dict(z_r2=r['r2'], **wall_score_scene(nets[0], phi, wall_band_scene(scene)))
    if hasattr(scene, 'deg'):
        def turned(z):
            x, _ = of_psi(psi)(z)
            x = scene.to_inner(x)
            return x, (x.abs() <= 0.5).all(1)
        r = fit(turned, scene.maze, s, steps, seed=seed, keep=nets)
        return dict(z_r2=r['r2'], **wall_score(nets[0], lambda x: phi(scene.from_inner(x)), scene.maze, wall_band(scene.maze)))
    r = fit(of_psi(psi), scene.maze, s, steps, seed=seed, keep=nets)
    return dict(z_r2=r['r2'], **wall_score(nets[0], phi, scene.maze, wall_band(scene.maze)))


def score(psi, scene, ss, steps):
    """Learnability scores of a map on any scene: {s: scores}. A plane maze (headings as an extra input), a turned maze (only points inside the turned square), or any other scene through its oracle."""
    if scene.dim == 3 or not hasattr(scene, 'maze'):
        return {s: fit_scene(psi, scene, float(s), steps) for s in ss}
    if hasattr(scene, 'deg'):
        def turned(z):                                  # outside the turned square there is no maze: those z are not drawn
            x, _ = of_psi(psi)(z)
            x = scene.to_inner(x)
            return x, (x.abs() <= 0.5).all(1)
        return {s: fit(turned, scene.maze, float(s), steps) for s in ss}
    return {s: fit(of_psi(psi), scene.maze, float(s), steps) for s in ss}


def of_psi(psi):
    """xy_of_z for a map of this study (a self-map of the box: every z is valid)."""
    def f(z):
        with torch.no_grad():
            x = torch.cat([psi(z[i:i + 262144]) for i in range(0, len(z), 262144)])
        return x, torch.ones(len(z), device=z.device, dtype=torch.bool)
    return f
