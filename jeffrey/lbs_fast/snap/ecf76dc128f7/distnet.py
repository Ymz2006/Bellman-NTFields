"""Task 2: a learned clearance field d^(q), so that the lift needs no geometric distance query at planning time.

The lift height of a configuration is h(q) = gain G(d(q)) / E with d the exact signed clearance in cells (rectangle against the maze
walls, or the trilinear voxel table of a 3-D house). G is constant outside [-depth, margin] (depth 0.0175, margin 0.175 cells), so only
the clearance inside that band matters. The model regresses the clipped clearance clamp(d, LO, HI) (cells) and the planner applies the
closed-form G to its output; regressing d rather than h keeps the target Lipschitz (G is 80 times steeper at the wall than d).

    kind 'mlp'   Fourier features of the configuration in cell units (random Gaussian matrix, sigma cycles per cell) -> ReLU MLP -> d
    kind 'grid'  a learned multi-resolution grid (dense while it fits, hashed above that; the Instant-NGP encoding) with trilinear
                 interpolation, levels concatenated -> small ReLU MLP -> d

Training samples are drawn on the fly: uniform in the C-space box, labelled by the scene's exact clearance (half of every batch is
redrawn near obstacles: points with |d| < HI kept). Adam, cosine decay, L1 loss on the clipped clearance.

    train(scene, cfg, device, log) -> model, history      load(path, scene, device) -> callable q (N, 3) -> clearance in cells
"""
import math
import time
import torch

LO, HI = -0.05, 0.25           # clipping of the regression target (cells); G only needs [-0.0175, 0.175]
PRIMES = (1, 2654435761, 805459861)


class DistMLP(torch.nn.Module):
    def __init__(self, scale, width=128, depth=3, feats=64, sigma=1.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float32))           # normalized -> cells, per coordinate
        self.register_buffer('B', 2 * math.pi * sigma * torch.randn(3, feats, generator=g))
        dims = [2 * feats + 3] + [width] * depth
        self.layers = torch.nn.ModuleList([torch.nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.out = torch.nn.Linear(width, 1)

    def forward(self, q):
        x = q * self.scale
        p = x @ self.B
        x = torch.cat([torch.sin(p), torch.cos(p), x * 0.1], 1)
        for lin in self.layers:
            x = torch.relu(lin(x))
        return self.out(x)[:, 0]


class DistGrid(torch.nn.Module):
    """Multi-resolution grid over the C-space box [lo, hi]: level l has res_l = round(base * growth^l) nodes per cell in x, y (and the
    same step in the third coordinate, at least 2 nodes); dense when the level has at most 2^log2_size nodes, hashed otherwise."""

    def __init__(self, lo, hi, cells, levels=8, feat=2, base=1.0, growth=1.6, log2_size=21, width=32, depth=1, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        lo, hi = torch.as_tensor(lo, dtype=torch.float32), torch.as_tensor(hi, dtype=torch.float32)
        ext = (hi - lo) * float(cells)                                                       # extent in cells per coordinate
        self.register_buffer('lo', lo)
        self.register_buffer('inv', 1.0 / (hi - lo))
        size = 1 << log2_size
        res, dense, offs, total = [], [], [], 0
        for l in range(levels):
            per = base * growth ** l
            r = torch.clamp(torch.ceil(ext * per), min=1).long() + 1                         # nodes per coordinate
            n = int(r.prod())
            res.append(r)
            dense.append(n <= size)
            offs.append(total)
            total += n if n <= size else size
        self.levels, self.feat, self.size = levels, feat, size
        self.register_buffer('res', torch.stack(res).float())                               # (L, 3)
        self.register_buffer('resl', torch.stack(res))                                      # (L, 3) long
        self.register_buffer('dense', torch.tensor(dense))
        self.register_buffer('offs', torch.tensor(offs))
        self.table = torch.nn.Parameter(1e-4 * (2 * torch.rand(total, feat) - 1))
        self.register_buffer('corner', torch.tensor([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]))   # (8, 3)
        dims = [levels * feat] + [width] * depth
        self.layers = torch.nn.ModuleList([torch.nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.out = torch.nn.Linear(dims[-1], 1)
        self.n_table = total

    def encode(self, q):
        u = ((q - self.lo) * self.inv).clamp(0, 1)                                           # (N, 3)
        x = u[:, None, :] * (self.res[None] - 1)                                             # (N, L, 3) node coordinates
        i0 = x.floor().long()
        i0 = torch.minimum(i0, (self.resl - 2).clamp(min=0)[None])
        w = x - i0                                                                           # (N, L, 3)
        c = i0[:, :, None, :] + self.corner[None, None]                                      # (N, L, 8, 3)
        c = torch.minimum(c, (self.resl - 1)[None, :, None, :])
        r = self.resl[None, :, None, :]
        lin = (c[..., 0] * r[..., 1] + c[..., 1]) * r[..., 2] + c[..., 2]                    # dense index
        hsh = ((c[..., 0] * PRIMES[0]) ^ (c[..., 1] * PRIMES[1]) ^ (c[..., 2] * PRIMES[2])) % self.size
        idx = torch.where(self.dense[None, :, None], lin, hsh) + self.offs[None, :, None]    # (N, L, 8)
        f = self.table[idx]                                                                  # (N, L, 8, F)
        cw = torch.where(self.corner[None, None].bool(), w[:, :, None, :], 1 - w[:, :, None, :]).prod(-1)   # (N, L, 8)
        return (f * cw[..., None]).sum(2).reshape(len(q), -1)

    def forward(self, q):
        x = self.encode(q)
        for lin in self.layers:
            x = torch.relu(lin(x))
        return self.out(x)[:, 0]


def build(cfg, scene):
    kind = cfg.get('kind', 'mlp')
    if kind == 'mlp':
        return DistMLP([scene.E] * 3, cfg.get('width', 128), cfg.get('depth', 3), cfg.get('feats', 64), cfg.get('sigma', 1.0), cfg.get('seed', 0))
    return DistGrid(scene.lo.cpu(), scene.hi.cpu(), scene.E, cfg.get('levels', 8), cfg.get('feat', 2), cfg.get('base', 1.0), cfg.get('growth', 1.6),
                    cfg.get('log2_size', 21), cfg.get('width', 32), cfg.get('depth', 1), cfg.get('seed', 0))


def draw(scene, n, gen, near_share=0.5):
    """n configurations: uniform in the box, and a share redrawn until they lie near an obstacle surface (|d| < HI)."""
    q = scene.uniform(n, gen)
    d = scene.clearance(q)
    k = int(n * near_share)
    if k:
        need = k
        out_q, out_d = [], []
        while need > 0:
            c = scene.uniform(4 * n, gen)
            dc = scene.clearance(c)
            m = dc.abs() < HI
            out_q.append(c[m][:need])
            out_d.append(dc[m][:need])
            need -= int(m.sum())
        q = torch.cat([q[:n - k]] + out_q)[:n]
        d = torch.cat([d[:n - k]] + out_d)[:n]
    return q, d


@torch.no_grad()
def evaluate(model, scene, field_G=None, n=400_000, seed=123):
    """Errors on fresh uniform samples: over all, inside the lift band (-0.0175 <= d < 0.175) and in the steep skin (d < 0.035)."""
    gen = torch.Generator(scene.device).manual_seed(seed)
    q = scene.uniform(n, gen)
    d = scene.clearance(q)
    p = model(q)
    e = (p - d.clamp(LO, HI)).abs()
    band = (d >= -0.0175) & (d < 0.175)
    skin = (d >= 0.0) & (d < 0.035)
    free = d >= 0
    out = dict(mae=float(e.mean()), mae_band=float(e[band].mean()), p95_band=float(torch.quantile(e[band][:200_000], 0.95)), max_band=float(e[band].max()),
               mae_skin=float(e[skin].mean()), band_share=float(band.float().mean()), free_share=float(free.float().mean()),
               free_called_blocked=float(((p < 0) & free).float().sum() / free.float().sum()), blocked_called_free=float(((p >= 0) & ~free).float().sum() / (~free).float().sum()))
    if field_G is not None:
        hg, hp = field_G(d), field_G(p)
        eh = (hg - hp).abs()
        out.update(h_mae_cells=float(eh[free].mean()), h_mae_band_cells=float(eh[band & free].mean()), h_p95_band_cells=float(torch.quantile(eh[band & free][:200_000], 0.95)))
    return out


def train(scene, cfg, device, log=print):
    torch.manual_seed(cfg.get('seed', 0))
    model = build(cfg, scene).to(device)
    steps, batch, lr = int(cfg.get('steps', 20000)), int(cfg.get('batch', 65536)), float(cfg.get('lr', 2e-3))
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-15 if cfg.get('kind') == 'grid' else 1e-8)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.02, final_div_factor=100)
    gen = torch.Generator(device).manual_seed(cfg.get('seed', 0) + 1)
    hist, t0 = [], time.time()
    for it in range(steps):
        q, d = draw(scene, batch, gen, cfg.get('near_share', 0.5))
        loss = (model(q) - d.clamp(LO, HI)).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if it % max(1, steps // 20) == 0 or it == steps - 1:
            ev = evaluate(model, scene, n=100_000)
            hist.append(dict(step=it, loss=float(loss), seconds=time.time() - t0, **ev))
            log(f'dist {it}/{steps} loss {float(loss):.5f} mae_band {ev["mae_band"]:.5f} skin {ev["mae_skin"]:.5f} ({time.time() - t0:.0f}s)')
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, hist


def save(path, model, cfg):
    torch.save(dict(cfg=cfg, state=model.state_dict()), path)


def load(path, scene, device):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    model = build(payload['cfg'], scene)
    model.load_state_dict(payload['state'])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, payload['cfg']
