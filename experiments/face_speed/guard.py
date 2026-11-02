"""Per-piece clearance step filter in joint space (cspace_face_filter.pdf, Sections 1.2 and 3).

After a planner picks a step dq at q, the filter returns the closest step dq' (Euclidean, W = I) that satisfies
one half-space per active piece and the joint limits:

    g_i . dq' >= -alpha (s_i(q) - s_floor)      for every piece i with s_i(q) < 1,   g_i = grad_q s_i(q)
    e_k . dq' >= -alpha (q_k - q_min,k),  -e_k . dq' >= -alpha (q_max,k - q_k)       for every joint k

with s_i = min(d_i / margin, 1): the piece speed without the lower clip, so a piece closer than the offset keeps
a gradient and is pushed back out. Hildreth's dual coordinate ascent, 30 Gauss-Seidel sweeps, batched over
cases. A post-step check evaluates every piece at q + dq' and halves the step up to three times when the
linear prediction s_i(q + dq') >= (1 - alpha) s_i(q) + alpha s_floor fails (arm lever arms break the
linearisation sooner than a rigid body). `rows='pieces'` is the per-piece filter; `rows='body'` is the
one-speed control that uses only the gradient of the minimum.
"""
import torch
from .clearance import PieceClearance, MARGIN


class StepGuard:
    def __init__(self, clearance: PieceClearance, alpha=.5, floor=.15, sweeps=30, rows='pieces', tolerance=1e-3, retries=3):
        self.clear, self.alpha, self.floor, self.sweeps, self.rows, self.tol, self.retries = clearance, alpha, floor, sweeps, rows, tolerance, retries
        self.summary = dict(steps=0, active_steps=0, rows_active=0, relative_change=0., retries=0, stalled=0)

    def speeds(self, q):
        """Piece speeds without the lower clip, (B, P)."""
        return (self.clear.pieces(q)/MARGIN).clamp(max=1.)

    def speeds_and_gradients(self, q):
        """Piece speeds (B, P) and their joint gradients (B, P, 6); each case depends only on its own q, so the
        Jacobian of the batch sum gives every case's gradient in one vectorized backward."""
        f = (lambda x: self.speeds(x).amin(1, keepdim=True)) if self.rows == 'body' else self.speeds
        with torch.enable_grad():
            x = q.detach().requires_grad_(True); s = f(x)
            G = torch.autograd.functional.jacobian(lambda y: f(y).sum(0), x, vectorize=True).permute(1, 0, 2)
        return s.detach(), G.detach()

    @torch.no_grad()
    def project(self, dq, G, h, mask):
        """min |dq' - dq|^2 s.t. G dq' >= h on the masked rows: Hildreth, (B, R, 6), (B, R), (B, R) -> (B, 6)."""
        w = dq.clone(); lam = torch.zeros_like(h); gg = (G*G).sum(-1).clamp_min(1e-12)
        for _ in range(self.sweeps):
            for i in range(G.shape[1]):
                gi = G[:, i]; new = ((lam[:, i]+(h[:, i]-(gi*w).sum(-1))/gg[:, i]).clamp_min(0))*mask[:, i]
                w = w+(new-lam[:, i])[:, None]*gi; lam[:, i] = new
        return w

    @torch.no_grad()
    def filter(self, q, dq):
        """Filtered step for cases q (B, 6) and planner steps dq (B, 6), normalized joint coordinates."""
        s, G = self.speeds_and_gradients(q); B, P = s.shape
        eye = torch.eye(6, device=q.device, dtype=q.dtype)
        G_rows = torch.cat([G, eye.expand(B, 6, 6), -eye.expand(B, 6, 6)], 1)
        h_rows = torch.cat([-self.alpha*(s-self.floor), -self.alpha*(q+.5), -self.alpha*(.5-q)], 1)
        mask = torch.cat([s < 1., torch.ones(B, 12, device=q.device, dtype=torch.bool)], 1)
        cols = mask.any(0)                                             # rows active in at least one case
        w = self.project(dq, G_rows[:, cols], h_rows[:, cols], mask[:, cols].to(q.dtype))
        target = (1-self.alpha)*s+self.alpha*self.floor-self.tol
        for attempt in range(self.retries):
            bad = (self.speeds(q+w) < target).any(1)
            if not bool(bad.any()): break
            w[bad] *= .5; self.summary['retries'] += int(bad.sum())
        active = (s < 1.).any(1); change = (w-dq).norm(dim=1)/dq.norm(dim=1).clamp_min(1e-12)
        self.summary['steps'] += B; self.summary['active_steps'] += int(active.sum()); self.summary['rows_active'] += int((s < 1.).sum())
        self.summary['relative_change'] += float(change.sum()); self.summary['stalled'] += int((w.norm(dim=1) < 1e-3*dq.norm(dim=1)).sum())
        return w

    def report(self):
        n = max(1, self.summary['steps'])
        return dict(alpha=self.alpha, floor=self.floor, sweeps=self.sweeps, rows=self.rows, steps=self.summary['steps'], filter_active_frac=self.summary['active_steps']/n,
                    mean_rows_active=self.summary['rows_active']/n, mean_relative_change=self.summary['relative_change']/n, retries=self.summary['retries'], stalled_steps=self.summary['stalled'])
