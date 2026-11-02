"""Pair regression of the geodesic distance with different point embeddings.

Network (P-NTFields shape): per-point encoder on [x, features(x)] -> symmetric combination [h_s + h_x, |h_s - h_x|]
-> generator -> rho = softplus(.) = log(d / ||s - x||). Features are frozen grids (masked bilinear lookup), Fourier
mappings, or learnable grids. The masked lookup uses free corner cells only, so a wall is a discontinuity of the input.
"""
import itertools
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedGridFeature(nn.Module):
    def __init__(self, grid, free, learnable=False):
        super().__init__()
        self.shape = tuple(grid.shape[1:])
        self.dim = len(self.shape)
        self.res = self.shape[0]
        self.channels = int(grid.shape[0])
        flat = grid.reshape(grid.shape[0], -1).T.contiguous()
        if learnable:
            self.grid = nn.Parameter(flat.float())
        else:
            self.register_buffer('grid', flat.half())
        self.register_buffer('free', free.reshape(-1).float())
        self.register_buffer('strides', torch.tensor([int(np.prod(self.shape[i + 1:])) for i in range(self.dim)]))
        self.register_buffer('corners', torch.tensor(list(itertools.product([0, 1], repeat=self.dim))))

    def forward(self, x):
        u = (x + 0.5) * self.res - 0.5
        i0 = torch.floor(u)
        t = u - i0
        i0 = i0.long()
        acc, den = 0.0, 0.0
        for c in self.corners:
            idx = (i0 + c).clamp(0, self.res - 1)
            w = torch.prod(torch.where(c.bool(), t, 1 - t), 1)
            lin = (idx * self.strides).sum(1)
            m = self.free[lin] * w
            acc = acc + m[:, None] * self.grid[lin].float()
            den = den + m
        return acc / den.clamp(min=1e-6)[:, None]


class FourierFeature(nn.Module):
    def __init__(self, dim, freqs, sigma, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer('B', sigma * torch.randn(dim, freqs, generator=g))
        self.channels = 2 * freqs

    def forward(self, x):
        p = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(p), torch.cos(p)], 1)


class Block(nn.Module):
    def __init__(self, width, act):
        super().__init__()
        self.l1, self.l2, self.act = nn.Linear(width, width), nn.Linear(width, width), act

    def forward(self, h):
        return self.act(self.l2(self.act(self.l1(h))) + h)


class PairNet(nn.Module):
    def __init__(self, dim, features, hidden=128, blocks=3, beta=10.0):
        super().__init__()
        self.features = nn.ModuleList(features)
        self.act = nn.Softplus(beta=beta)
        width = dim + sum(f.channels for f in features)
        self.inp = nn.Linear(width, hidden)
        self.enc = nn.ModuleList([Block(hidden, self.act) for _ in range(blocks)])
        self.gen_in = nn.Linear(2 * hidden, 2 * hidden)
        self.gen = nn.ModuleList([Block(2 * hidden, self.act) for _ in range(blocks)])
        self.out1 = nn.Linear(2 * hidden, hidden)
        self.out2 = nn.Linear(hidden, 1)

    def encode(self, x):
        h = self.act(self.inp(torch.cat([x] + [f(x) for f in self.features], 1)))
        for b in self.enc:
            h = b(h)
        return h

    def forward(self, s, x):
        hs, hx = self.encode(s), self.encode(x)
        z = self.act(self.gen_in(torch.cat([hs + hx, (hs - hx).abs()], 1)))
        for b in self.gen:
            z = b(z)
        # linear head: a softplus head dies (logit -> -16, zero gradient) under the L1 loss whose median target is ~0.07
        return self.out2(self.act(self.out1(z))).squeeze(-1)


class Sampler:
    """Pairs (source, free point): half uniform over free cells, half within local_radius cells of the source."""

    def __init__(self, free, clearance, dist, sources, local_radius, seed):
        self.free = free
        self.res = free.shape[0]
        self.dim = free.ndim
        self.cells = torch.stack(torch.nonzero(free, as_tuple=True), 1)
        self.clearance = clearance
        self.dist = dist.view(dist.shape[0], -1)
        self.sources = sources
        self.local_radius = local_radius
        self.strides = torch.tensor([int(np.prod(free.shape[i + 1:])) for i in range(self.dim)], device=free.device)
        self.corners = torch.tensor(list(itertools.product([0, 1], repeat=self.dim)), device=free.device)
        self.free_flat = free.reshape(-1).float()
        self.g = torch.Generator(device=free.device).manual_seed(seed)

    def interp_dist(self, s, x):
        u = (x + 0.5) * self.res - 0.5
        i0 = torch.floor(u)
        t = u - i0
        i0 = i0.long()
        acc, den = 0.0, 0.0
        for c in self.corners:
            idx = (i0 + c).clamp(0, self.res - 1)
            w = torch.prod(torch.where(c.bool(), t, 1 - t), 1)
            lin = (idx * self.strides).sum(1)
            m = self.free_flat[lin] * w
            acc = acc + m * self.dist[s, lin]
            den = den + m
        return acc / den.clamp(min=1e-6)

    def sample(self, n, source_ids, local_fraction=0.5):
        dev = self.free.device
        s = source_ids[torch.randint(len(source_ids), (n,), generator=self.g, device=dev)]
        sc = self.sources[s]
        n_loc = int(n * local_fraction)
        off = torch.randint(-self.local_radius, self.local_radius + 1, (n_loc, self.dim), generator=self.g, device=dev)
        cand = (sc[:n_loc] + off).clamp(0, self.res - 1)
        uni = self.cells[torch.randint(len(self.cells), (n,), generator=self.g, device=dev)]
        ok = self.free[tuple(cand.T)]
        cells = uni.clone()
        cells[:n_loc][ok] = cand[ok]
        x = (cells.float() + torch.rand((n, self.dim), generator=self.g, device=dev)) / self.res - 0.5
        s_pos = (sc.float() + 0.5) / self.res - 0.5
        d = self.interp_dist(s, x)
        euclid = (s_pos - x).norm(dim=1)
        keep = torch.isfinite(d) & (euclid > 0) & (d > 0)
        rho = torch.log(torch.maximum(d, euclid) / euclid)
        clear = self.clearance[tuple(cells.T)]
        return dict(s=s_pos[keep], x=x[keep], rho=rho[keep], euclid=euclid[keep], d=d[keep], clearance=clear[keep], source=s[keep])


def train(net, sampler, source_ids, steps, batch, lr, grid_lr, log, label=''):
    grid_params = [p for n, p in net.named_parameters() if n.endswith('.grid')]
    other = [p for n, p in net.named_parameters() if not n.endswith('.grid')]
    groups = [dict(params=other, lr=lr)] + ([dict(params=grid_params, lr=grid_lr)] if grid_params else [])
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.01)
    history = []
    t0 = time.time()
    for step in range(1, steps + 1):
        b = sampler.sample(batch, source_ids)
        loss = (net(b['s'], b['x']) - b['rho']).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 200 == 0 or step == steps:
            history.append(dict(step=step, loss=float(loss)))
        if step % 1000 == 0 or step == steps:
            log(f'  {label} step {step}/{steps} loss {float(loss):.4f} ({time.time() - t0:.0f}s)')
    return history


@torch.no_grad()
def evaluate(net, sampler, source_ids, n, batch, local_radius_units, wall_ratio=2.5, corridor_ratio=1.15):
    rows = {k: [] for k in ['pred', 'rho', 'euclid', 'd', 'clearance']}
    for _ in range(math.ceil(n / batch)):
        b = sampler.sample(batch, source_ids)
        rows['pred'].append(net(b['s'], b['x']).clamp(min=0))
        for k in ['rho', 'euclid', 'd', 'clearance']:
            rows[k].append(b[k])
    r = {k: torch.cat(v).cpu().numpy() for k, v in rows.items()}
    err = np.abs(r['pred'] - r['rho'])
    ratio = np.exp(r['rho'])
    pred_ratio = np.exp(r['pred'])
    local = r['euclid'] <= local_radius_units
    cats = dict(all=np.ones_like(local), local=local, far=~local, wall=local & (ratio >= wall_ratio), detour=local & (ratio >= 1.5),
                corridor=local & (ratio <= corridor_ratio), near_wall=r['clearance'] <= 2, open=r['clearance'] > 8)
    out = dict(n=int(len(err)))
    for name, sel in cats.items():
        if sel.sum():
            rel = np.abs(pred_ratio[sel] - ratio[sel]) / ratio[sel]
            out[name] = dict(n=int(sel.sum()), mean_abs_log=float(err[sel].mean()), median_abs_log=float(np.median(err[sel])),
                             mean_rel=float(rel.mean()), p90_abs_log=float(np.percentile(err[sel], 90)))
    label = local & (ratio >= 2.0)
    if label.sum() and (local & ~label).sum():
        from probe import auc
        score = r['pred'][local]
        out['detect_wall'] = dict(auc=auc(score[label[local]], score[~label[local]]),
                                  recall_at_log2=float((r['pred'][label] >= math.log(2)).mean()),
                                  false_alarm_at_log2=float((r['pred'][local & ~label] >= math.log(2)).mean()))
    return out


@torch.no_grad()
def render(net, sampler, source_id, batch=65536):
    """Predicted distance at every free cell centre from one source: (res, res) image (2-D) with nan outside free."""
    cells = sampler.cells
    s_pos = (sampler.sources[source_id].float() + 0.5) / sampler.res - 0.5
    x = (cells.float() + 0.5) / sampler.res - 0.5
    pred = torch.cat([net(s_pos.expand(len(xb), -1), xb) for xb in x.split(batch)])
    img = torch.full(sampler.free.shape, float('nan'), device=cells.device)
    img[tuple(cells.T)] = (s_pos - x).norm(dim=1) * torch.exp(pred.clamp(min=0))
    return img
