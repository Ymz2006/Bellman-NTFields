"""The cartogram as a normalizing flow: the Fourier network learns the density-equalizing map by maximum likelihood.

A map phi of the box onto itself pushes a density p to uniform exactly when p(x) = |det J_phi(x)|. So the map of carto.py is the maximum-likelihood normalizing flow of

    p = (blurred boundary density + eps) / (1 + eps),

and its loss needs only samples of p: crossing points of random lines through the identity (the oracle, any dimension), each moved by Gaussian noise of standard deviation h (that is the blur), mixed with uniform points at share eps / (1 + eps):

    L = - E_{x ~ p} log |det J_phi(x)|

No grid, no scan during training, so it scales to dimensions where carto.py's tables cannot go. phi is the network; psi = phi^-1 is what the line reader uses.
"""
import math
import time
import torch

import maps as M
import measure as ME
import scan as S


@torch.no_grad()
def cloud(scene, lines=1 << 16, samples=2048, frame=1.0, seed=37, directions='any'):
    """Crossing points (M, d) of random lines through the identity, without those on the frame of the box."""
    dev = scene.table.v.device
    gen = torch.Generator(dev).manual_seed(seed)
    out, done = [], 0
    while done < lines:
        n = min(8192, lines - done)
        r = S.any_rays(n, scene.dim, scene.planes, gen, dev) if directions == 'any' else S.lines_uniform(n, scene.dim, scene.planes, gen, dev)
        line, t, _ = S.crossings(lambda z: z, scene, r, samples)
        x = S.at(r, line, t)
        out.append(x[x.abs().amax(1) < 0.5 - frame / scene.E])
        done += n
    return torch.cat(out)


def reflect(x):
    """Fold points back into the box: the blurred density has no flux through the faces."""
    y = torch.remainder(x + 0.5, 2.0)
    return torch.where(y > 1.0, 2.0 - y, y) - 0.5


class Inverted(torch.nn.Module):
    """phi as the psi of the rest of the study: calling it is z -> x."""

    def __init__(self, phi):
        super().__init__()
        self.phi = phi
        self.cfg = dict(kind='inverted', inner=phi.cfg)

    def forward(self, z):
        with torch.no_grad():
            return torch.cat([self.phi.inverse(z[i:i + 262144]) for i in range(0, len(z), 262144)]) if len(z) else z

    def inverse(self, x):
        return self.phi(x)

    def window(self, alpha):
        self.phi.window(alpha)


def fit(phi, scene, pts, h=1.0, eps=1.0, steps=6000, batch=8192, lr=1e-3, kappa=0.0, log=print, log_every=500, seed=0):
    """Maximum likelihood on the mixture. phi: a map of this study, used as x -> z. Returns the history."""
    dev = pts.device
    gen = torch.Generator(dev).manual_seed(seed + 3)
    d = pts.shape[1]
    opt = torch.optim.Adam([p for p in phi.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.02)
    fd = 0.02 / scene.E
    share = eps / (1 + eps)
    nu = int(batch * share)
    hb = h / scene.E
    groups = phi.feat.groups if hasattr(phi.feat, 'groups') else phi.feat[0].groups
    t0, hist = time.time(), []
    for step in range(1, steps + 1):
        phi.window(min(groups, 1 + (groups - 1) * step / (0.5 * steps)))
        i = torch.randint(len(pts), (batch - nu,), generator=gen, device=dev)
        xb = reflect(pts[i] + hb * torch.randn(batch - nu, d, generator=gen, device=dev))
        xu = torch.rand(nu, d, generator=gen, device=dev) - 0.5
        x = (torch.cat([xb, xu]) * (1 - 4 * fd))
        if hasattr(phi, 'forward_logdet'):
            _, ld = phi.forward_logdet(x)
            loss = -ld.mean()
        else:
            J = M.jacobian_fd(phi, x, fd)
            sign, ld = torch.linalg.slogdet(J + 1e-6 * torch.eye(d, device=dev))
            loss = -ld.clamp(min=-30).mean() + 10.0 * torch.relu(-sign).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for p in phi.parameters() if p.requires_grad], 5.0)
        if torch.isfinite(norm):                        # a near-singular Jacobian on one batch must not poison the network
            opt.step()
        sched.step()
        if step % log_every == 0 or step == 1:
            hist.append(dict(step=step, nll=float(loss), seconds=time.time() - t0))
            log(f'step {step}: nll {float(loss):.4f}, {time.time() - t0:.0f}s')
    phi.window(groups)
    return hist
