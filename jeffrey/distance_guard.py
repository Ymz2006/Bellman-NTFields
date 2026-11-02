"""A step guard that uses only the learned distance model's predicted obstacle clearance.

The planner (RK4 descent or ntrl-demo MPPI on a travel-time field) proposes a step dq at q; the guard returns the
step that is actually taken. The only geometric information is the model's prediction: per-sphere obstacle
clearance e_i(q) = D(c_i(q)) - r_i (exact kinematics -> learned point distance D) and its joint gradient by autograd.
No Lipschitz bounds, no exact distance, no self-collision term, no search.

Per sphere the floor is f_i = min(5 mm + eps, max(e_end - tol, min(e_end, 5 mm))) with e_end = min(e_i(start), e_i(goal)):
the scorer threshold plus the model's calibrated over-estimation bound, lowered only for spheres that are already
tighter at a valid endpoint, and then never by the tolerance below 5 mm.

  mode 'reject'   check the step's points at the scorer's resolution (0.001 rad per joint): a step is taken only
                  if e_i >= f_i at all of them; otherwise it is halved up to `retries` times, then not taken.
  mode 'project'  first project dq onto the half-spaces  grad e_i . dq' >= -alpha (e_i - f_i)  of the spheres
                  closer than `activate` to their floor, plus the joint limits (Hildreth), so the step slides
                  along the obstacle instead of into it; then the same dense check and halving. With buffer b the
                  rows are  grad e_i . dq' >= -alpha (e_i - f_i - b): a sphere within b of its floor must move out.
"""
import torch
from jeffrey.geometry import OBSTACLE_MARGIN


class DistanceGuard:
    def __init__(self, clear, mode='project', alpha=.5, activate=.03, retries=4, tol=2.5e-4, buffer=0., sweeps=30, step_rad=1e-3):
        assert mode in ('reject', 'project')
        self.clear, self.mode, self.alpha, self.activate, self.retries, self.tol, self.buffer, self.sweeps, self.step_rad = clear, mode, alpha, activate, retries, tol, buffer, sweeps, step_rad
        self.q_range = clear.robot.q_range
        self.reset()

    def reset(self):
        self.summary = dict(steps=0, active_steps=0, projected_change=0., halvings=0, refused=0, checked_points=0)

    def env(self, q):
        """Predicted obstacle clearance per sphere (B, S) in metres."""
        s = self.clear.spheres(q)
        return self.clear.point_distance(s[..., :3])-s[..., 3]

    @torch.no_grad()
    def set_endpoints(self, starts, goals):
        thr = OBSTACLE_MARGIN+self.clear.eps
        end = torch.minimum(self.env(starts), self.env(goals))
        self.floor = torch.maximum(end-self.tol, end.clamp(max=OBSTACLE_MARGIN)).clamp(max=thr)

    def env_and_jacobian(self, q, chunk=128):
        J = []
        for s in range(0, len(q), chunk):                                                         # bounded activation memory
            with torch.enable_grad():
                x = q[s:s+chunk].detach().requires_grad_(True)
                J.append(torch.autograd.functional.jacobian(lambda y: self.env(y).sum(0), x, vectorize=True).permute(1, 0, 2).detach())    # (b, S, 6)
        with torch.no_grad(): e = self.env(q)
        return e, torch.cat(J)

    @torch.no_grad()
    def project(self, dq, G, h, mask):
        """min |dq' - dq|^2 s.t. G dq' >= h on the masked rows (Hildreth): (B, R, 6), (B, R), (B, R) -> (B, 6)."""
        w = dq.clone(); lam = torch.zeros_like(h); gg = (G*G).sum(-1).clamp_min(1e-12)
        for _ in range(self.sweeps):
            for i in range(G.shape[1]):
                gi = G[:, i]; new = ((lam[:, i]+(h[:, i]-(gi*w).sum(-1))/gg[:, i]).clamp_min(0))*mask[:, i]
                w = w+(new-lam[:, i])[:, None]*gi; lam[:, i] = new
        return w

    @torch.no_grad()
    def segment_ok(self, a, b, floor, chunk=32768):
        """True where every scorer-resolution point of a -> b keeps every sphere at or above its floor."""
        n = ((b-a)*self.q_range).abs().amax(1).div(self.step_rad).ceil().clamp_min(1)                  # points per segment
        M = int(n.max()); ok = torch.ones(len(a), dtype=torch.bool, device=a.device)
        k = torch.arange(1, M+1, device=a.device, dtype=a.dtype)
        rows = max(1, chunk//M)
        for s in range(0, len(a), rows):
            t = (torch.minimum(k[None], n[s:s+rows, None])/n[s:s+rows, None])                           # (b, M), repeats past n
            pts = a[s:s+rows, None]+t[..., None]*(b-a)[s:s+rows, None]
            e = self.env(pts.reshape(-1, 6)).reshape(len(t), M, -1)
            ok[s:s+rows] = (e >= floor[s:s+rows, None]).all(2).all(1)
        self.summary['checked_points'] += int(n.sum())
        return ok

    @torch.no_grad()
    def filter(self, q, dq, ids):
        """Guarded step for cases q (B, 6) with planner steps dq (B, 6) and query indices ids (B,)."""
        floor = self.floor[ids]; B = len(q)
        if self.mode == 'project':
            e, J = self.env_and_jacobian(q); slack = e-floor
            eye = torch.eye(6, device=q.device, dtype=q.dtype)
            G = torch.cat([J, eye.expand(B, 6, 6), -eye.expand(B, 6, 6)], 1)
            h = torch.cat([-self.alpha*(slack-self.buffer), -self.alpha*(q+.5), -self.alpha*(.5-q)], 1)
            mask = torch.cat([slack < self.activate, torch.ones(B, 12, dtype=torch.bool, device=q.device)], 1)
            cols = mask.any(0)
            w = self.project(dq, G[:, cols], h[:, cols], mask[:, cols].to(q.dtype))
            active = (slack < self.activate).any(1)
            self.summary['active_steps'] += int(active.sum())
            self.summary['projected_change'] += float(((w-dq).norm(dim=1)/dq.norm(dim=1).clamp_min(1e-12)).sum())
        else: w = dq.clone()
        w = (q+w).clamp(-.5, .5)-q
        todo = torch.arange(B, device=q.device)
        for attempt in range(self.retries+1):
            ok = self.segment_ok(q[todo], q[todo]+w[todo], floor[todo])
            todo = todo[~ok]
            if not len(todo): break
            if attempt < self.retries: w[todo] *= .5; self.summary['halvings'] += len(todo)
        w[todo] = 0.; self.summary['refused'] += len(todo); self.summary['steps'] += B
        return w

    def report(self):
        n = max(1, self.summary['steps'])
        return dict(mode=self.mode, alpha=self.alpha, activate=self.activate, retries=self.retries, tol=self.tol, buffer=self.buffer, eps=self.clear.eps, steps=self.summary['steps'],
                    active_frac=self.summary['active_steps']/n, mean_projected_change=self.summary['projected_change']/n, halvings=self.summary['halvings'],
                    refused_steps=self.summary['refused'], refused_frac=self.summary['refused']/n, checked_points=self.summary['checked_points'])
