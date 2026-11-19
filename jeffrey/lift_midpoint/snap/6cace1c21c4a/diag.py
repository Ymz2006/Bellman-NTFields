"""Field diagnostics of the lift-scale study against the prepared ground truth (no solver is run here: truth_fields.npy of the prepare
run holds T*(node, goal) for its 48 goals; Truth.time() extends it to off-grid points as maze_scale/truth.Grid.query_time does).

wall_fit      S |grad_q0 T| at held-out band points by speed bin (1 = the eikonal equation holds; below 1 = the slope falls short)
accuracy      random (free point, goal) pairs: share of learned times within 10 % of the truth, median learned / true; the same on
              line-of-sight pairs (T* <= 1.05 D, D the straight C-space distance) and the detour index rho = (T - D) / (T* - D) on
              detour pairs (T* >= 1.3 D), overall and by the true detour's length T* - D in cells
stalls        planning seed 0: where each run that did not arrive stopped: learned / true time from there, clearance, distance to the
              goal, whether the straight segment to the goal is blocked, the eikonal residual there
nu            |grad_q d| / E at held-out band points (1 for a point robot; the lift equalizes the normal speed up to this factor)
"""
import math
from pathlib import Path
import numpy as np
import torch

SPEED_BINS = [0.0, 0.02, 0.06, 0.15, 0.4, 0.8, 1.01]
DETOUR_BINS = [0.0, 1.0, 3.0, 6.0, 12.0, 24.0, 1e9]


def speed_labels():
    return [f'{SPEED_BINS[i]:g}-{min(SPEED_BINS[i + 1], 1):g}' for i in range(len(SPEED_BINS) - 1)]


def detour_labels():
    return [f'{DETOUR_BINS[i]:g}-{DETOUR_BINS[i + 1]:g}' if DETOUR_BINS[i + 1] < 1e8 else f'>{DETOUR_BINS[i]:g}' for i in range(len(DETOUR_BINS) - 1)]


class Truth:
    def __init__(self, scene, prepared):
        self.scene = scene
        self.fields = np.load(Path(prepared) / 'truth_fields.npy', mmap_mode='r')          # (G, nx, ny, nt) float16
        self.goals = torch.tensor(np.load(Path(prepared) / 'goals.npy'), device=scene.device)
        _, nx, ny, nt = self.fields.shape
        self.shape = (nx, ny, nt)
        dev = scene.device
        self.xs = torch.linspace(-0.5, 0.5, nx, device=dev)
        self.ts = torch.linspace(-scene.phi_max, scene.phi_max, nt, device=dev)
        self.hx, self.ht = 1.0 / (nx - 1), 2 * scene.phi_max / (nt - 1)
        self._cache = {}

    def field(self, g):
        if g not in self._cache:
            if len(self._cache) > 3:
                self._cache.pop(next(iter(self._cache)))
            self._cache[g] = torch.tensor(np.asarray(self.fields[g], dtype=np.float32), device=self.scene.device).reshape(-1)
        return self._cache[g]

    @torch.no_grad()
    def time(self, gi, q, chunk=20000):
        """T*(q, goal gi) for q (P, 3), gi (P,) long (numpy or tensor)."""
        gi = torch.as_tensor(gi, device=q.device).long()
        out = torch.full((len(q),), float('inf'), device=q.device)
        nx, ny, nt = self.shape
        off = torch.stack(torch.meshgrid(*[torch.arange(-1, 3, device=q.device)] * 3, indexing='ij'), -1).reshape(-1, 3)
        for g in torch.unique(gi).tolist():
            sel = torch.nonzero(gi == g).squeeze(1)
            f = self.field(g)
            for i in range(0, len(sel), chunk):
                s = sel[i:i + chunk]
                src = q[s]
                c = torch.stack([(src[:, 0] + 0.5) / self.hx, (src[:, 1] + 0.5) / self.hx, (src[:, 2] + self.scene.phi_max) / self.ht], 1)
                k = torch.floor(c).long()[:, None] + off[None]
                ok = (k >= 0).all(-1) & (k[..., 0] < nx) & (k[..., 1] < ny) & (k[..., 2] < nt)
                kc = torch.stack([k[..., 0].clamp(0, nx - 1), k[..., 1].clamp(0, ny - 1), k[..., 2].clamp(0, nt - 1)], -1)
                node = torch.stack([self.xs[kc[..., 0]], self.xs[kc[..., 1]], self.ts[kc[..., 2]]], -1)
                s0 = 1 / self.scene.train_speed(self.scene.clearance(src))
                sn = 1 / self.scene.train_speed(self.scene.clearance(node.reshape(-1, 3))).reshape(len(s), -1)
                cost = torch.linalg.vector_norm(node - src[:, None], dim=-1) * 0.5 * (s0[:, None] + sn)
                cost = torch.where(ok, cost, torch.full_like(cost, float('inf')))
                out[s] = (cost + f[(kc[..., 0] * ny + kc[..., 1]) * nt + kc[..., 2]]).amin(1)
        return out


def wall_fit(scene, prepared, grad, n=50000):
    hx = torch.tensor(np.load(Path(prepared) / 'heldout/sampled_points.npy')[:n], device=scene.device)
    hd = torch.tensor(np.load(Path(prepared) / 'heldout/clearance_cells.npy')[:n], device=scene.device)[:, 0]
    S = scene.train_speed(hd)
    gap = torch.linalg.vector_norm(hx[:, :3] - hx[:, 3:], dim=1) / scene.pitch
    _, g = grad(hx)
    slope = S * torch.linalg.vector_norm(g, dim=1)
    _, normal = scene.clearance_normal(hx[:, :3])
    cos = torch.nn.functional.cosine_similarity(g, -normal, dim=1)
    out = dict(by_speed={}, by_speed_far={})
    for name, far in (('by_speed', gap >= 0), ('by_speed_far', gap >= 2.0)):
        for i, lab in enumerate(speed_labels()):
            m = (S >= SPEED_BINS[i]) & (S < SPEED_BINS[i + 1]) & far
            if int(m.sum()) > 20:
                out[name][lab] = dict(n=int(m.sum()), slope=float(slope[m].median()), q10=float(slope[m].quantile(0.1)), q90=float(slope[m].quantile(0.9)),
                                      cos=float(cos[m].median()), attract=float((cos[m] < 0).float().mean()))
    m = S < 0.06
    out['innermost'] = float(slope[m].median()) if int(m.sum()) > 20 else None
    out['overshoot_share'] = float((slope > 1.2).float().mean())
    return out


def nu_stats(scene, prepared, n=20000):
    hx = torch.tensor(np.load(Path(prepared) / 'heldout/sampled_points.npy')[:n, :3], device=scene.device)
    x = hx.clone().requires_grad_(True)
    with torch.enable_grad():
        d = scene.signed_cells(x)
        g, = torch.autograd.grad(d.sum(), x)
    nu = torch.linalg.vector_norm(g, dim=1) / scene.E
    return dict(median=float(nu.median()), q10=float(nu.quantile(0.1)), q90=float(nu.quantile(0.9)), max=float(nu.max()))


def truth_pairs(scene, truth, n=24000, seed=5, min_clear=0.1):
    """Random (free configuration, goal) pairs with a finite true time: pairs (P, 6), goal index, T*, D (normalized units)."""
    gen = torch.Generator(scene.device).manual_seed(seed)
    q = scene.free(n, gen, min_clear)
    gi = torch.randint(0, len(truth.goals), (n,), generator=gen, device=scene.device)
    t = truth.time(gi, q)
    g = truth.goals[gi]
    D = torch.linalg.vector_norm(q - g, dim=1)
    keep = torch.isfinite(t) & (D > 0.25 * scene.pitch)
    return torch.cat([q, g], 1)[keep], gi[keep], t[keep], D[keep]


def accuracy(scene, cost, pairs, t_star, D, blocked=None):
    t = cost(pairs)
    ratio = t / t_star
    out = dict(n=int(len(t)), within10=float(((ratio - 1).abs() <= 0.1).float().mean()), ratio_median=float(ratio.median()),
               ratio_q10=float(ratio.quantile(0.1)), ratio_q90=float(ratio.quantile(0.9)), t_max=float(t.max()), t_star_max=float(t_star.max()),
               short_share=float((ratio < 0.9).float().mean()), long_share=float((ratio > 1.1).float().mean()))
    los = t_star <= 1.05 * D
    det = t_star >= 1.3 * D
    if int(los.sum()) > 20:
        out['los'] = dict(n=int(los.sum()), within10=float(((ratio[los] - 1).abs() <= 0.1).float().mean()), ratio_median=float(ratio[los].median()))
    if int(det.sum()) > 20:
        rho = (t[det] - D[det]) / (t_star[det] - D[det])
        true_detour = (t_star[det] - D[det]) / scene.pitch
        learned_detour = (t[det] - D[det]) / scene.pitch
        out['detour'] = dict(n=int(det.sum()), rho_median=float(rho.median()), rho_q10=float(rho.quantile(0.1)), rho_q90=float(rho.quantile(0.9)),
                             seen_through=float((rho < 0.5).float().mean()), within10=float(((ratio[det] - 1).abs() <= 0.1).float().mean()),
                             ratio_median=float(ratio[det].median()), by_length={})
        for i, lab in enumerate(detour_labels()):
            m = (true_detour >= DETOUR_BINS[i]) & (true_detour < DETOUR_BINS[i + 1])
            if int(m.sum()) > 10:
                out['detour']['by_length'][lab] = dict(n=int(m.sum()), true_cells=float(true_detour[m].median()), learned_cells=float(learned_detour[m].median()),
                                                       rho_median=float(rho[m].median()), seen_through=float((rho[m] < 0.5).float().mean()))
    if blocked is not None:
        for name, m in (('blocked', blocked), ('clear', ~blocked)):
            if int(m.sum()) > 20:
                out['segment_' + name] = dict(n=int(m.sum()), within10=float(((ratio[m] - 1).abs() <= 0.1).float().mean()), ratio_median=float(ratio[m].median()))
    # by true travel distance in cells
    dc = t_star / scene.pitch
    out['by_distance'] = {}
    for lo, hi in ((0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 1e9)):
        m = (dc >= lo) & (dc < hi)
        if int(m.sum()) > 20:
            out['by_distance'][f'{lo:g}-{hi:g}' if hi < 1e8 else f'>{lo:g}'] = dict(n=int(m.sum()), within10=float(((ratio[m] - 1).abs() <= 0.1).float().mean()),
                                                                                  ratio_median=float(ratio[m].median()))
    return out


def stalls(scene, truth, cost, grad, prepared, out_dir, blocked_fn):
    """Planning seed 0 (paths_seed0.npy, rows.npz): one row per run that did not arrive."""
    Q = np.load(Path(prepared) / 'queries.npz')
    rows = np.load(Path(out_dir) / 'rows.npz')
    paths = np.load(Path(out_dir) / 'paths_seed0.npy', allow_pickle=True)
    conv, coll, ok = rows['converged'][0], rows['collision'][0], rows['success'][0]
    idx = np.nonzero(~conv)[0]
    res = dict(runs=int(len(conv)), success=int(ok.sum()), arrived_wall=int((conv & coll).sum()), not_arrived=int(len(idx)))
    if not len(idx):
        return res, None
    dev = scene.device
    end = torch.tensor(np.stack([np.asarray(paths[i])[-2] for i in idx]), device=dev, dtype=torch.float32)
    goal = torch.tensor(Q['goal'][idx], device=dev, dtype=torch.float32)
    gi = torch.tensor(Q['goal_index'][idx], device=dev)
    pairs = torch.cat([end, goal], 1)
    d_end = scene.clearance(end)
    inside = d_end >= 0
    t_true = truth.time(gi, end)
    t_net, g = grad(pairs)
    S = scene.train_speed(d_end)
    resid = S * torch.linalg.vector_norm(g, dim=1)
    dist = torch.linalg.vector_norm(end[:, :2] - goal[:, :2], dim=1) / scene.pitch
    blk = blocked_fn(pairs)
    ratio = t_net / t_true
    # how far the run got: remaining true time over the query's true time
    t0 = torch.tensor(Q['t_star'][idx], device=dev)
    progress = 1 - t_true / t0
    # the clearance trace of the last tenth of the path: pressed against a wall?
    fin = torch.isfinite(ratio) & inside
    res.update(end_in_free=int(inside.sum()), facing_blocked=float(blk.float().mean()), ratio_median=float(ratio[fin].median()) if bool(fin.any()) else None,
               ratio_q10=float(ratio[fin].quantile(0.1)) if bool(fin.any()) else None, ratio_q90=float(ratio[fin].quantile(0.9)) if bool(fin.any()) else None,
               clearance_median=float(d_end.median()), near_wall=float((d_end < 0.175).float().mean()), open_space=float((d_end > 0.5).float().mean()),
               dist_goal_median=float(dist.median()), slope_median=float(resid.median()), progress_median=float(progress[torch.isfinite(progress)].median()) if bool(torch.isfinite(progress).any()) else None,
               too_short=float((ratio[fin] < 0.8).float().mean()) if bool(fin.any()) else None,
               bins={str(b): int((Q['bin'][idx] == b).sum()) for b in sorted(set(Q['bin'][idx].tolist()))})
    table = dict(query=idx, end=end.cpu().numpy(), t_net=t_net.cpu().numpy(), t_true=t_true.cpu().numpy(), clearance=d_end.cpu().numpy(), dist_goal=dist.cpu().numpy(),
                 blocked=blk.cpu().numpy(), slope=resid.cpu().numpy(), progress=progress.cpu().numpy())
    return res, table


def descent_trap(scene, cost, grad, pairs, t_star, steps=400, step_cells=0.05):
    """Follow -grad T (unit steps of step_cells cells in C-space, stopping inside obstacles) from the first configuration of each pair
    towards the second and report where it ends: the share that reaches the goal's neighbourhood (0.5 cells), that ends inside an
    obstacle, and that stops moving (a local minimum of the field that is not the goal)."""
    x = pairs[:, :3].clone()
    goal = pairs[:, 3:]
    h = step_cells * scene.pitch
    alive = torch.ones(len(x), dtype=torch.bool, device=x.device)
    arrived = torch.zeros_like(alive)
    hit = torch.zeros_like(alive)
    start_t = cost(pairs)
    for _ in range(steps):
        ids = torch.nonzero(alive).squeeze(1)
        if not len(ids):
            break
        _, g = grad(torch.cat([x[ids], goal[ids]], 1))
        new = x[ids] - h * torch.nn.functional.normalize(g, dim=1)
        new = torch.minimum(torch.maximum(new, scene.lo), scene.hi)
        x[ids] = new
        near = torch.linalg.vector_norm(new[:, :2] - goal[ids, :2], dim=1) < 0.5 * scene.pitch
        arrived[ids[near]] = True
        inside = scene.clearance(new) < 0
        hit[ids[inside & ~near]] = True
        alive[ids[near | inside]] = False
    stuck = alive
    return dict(n=int(len(x)), arrived=float(arrived.float().mean()), hit_obstacle=float(hit.float().mean()), stuck=float(stuck.float().mean()))
