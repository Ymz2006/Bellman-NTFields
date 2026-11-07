"""MPPI in a warp's stretched space (the encoder's input coordinates z = phi(x)), for trained cross-warp fields.

maze_scale/plan.mppi draws its perturbations isotropically in the robot's C-space (normalized units, a fixed share of
a cell). mppi_stretched draws them in z instead. Each iteration it takes the warp's Jacobian J(x) at every active
query's pose (central differences of phi), draws plan.mppi's perturbations in z-units, maps every sampled step back
to C-space with J(x)^-1, scores the rollouts with the field exactly as plan.mppi does (10 T(after step 1) + T(after
step H), softmax weights) and moves the pose by J(x)^-1 times the weighted first step. Everything else is plan.mppi
line for line: the momentum term (prior, here in z-units), the cap on each step's norm (in z-units), the heading
clamp, convergence |goal - x| < tol in C-space, the returned paths.

Step: step_z = step * m_q, with step plan.settings' C-space step and m_q the warp's median area stretch (sqrt det J of
the x, y map) over the four quadrant mazes, so a step inside a quadrant is about plan.mppi's step there and a step
where the warp magnifies less (the open band, for every warp here) is longer, with a longer look-ahead. Near the
goal the cap is min(step_z, |phi(goal) - phi(x)|), so the planner does not step past it. Budget: plan.mppi's rule
with the certified true path measured in z: max(min_steps, ceil(budget_factor * L_z / step_z)).
With a constant Jacobian (ntrl-demo, uniform stretch) this is plan.mppi itself: the controls.
"""
import numpy as np
import torch


class Identity(torch.nn.Module):
    """The warp of a field without one (maze_scale ntrl-demo runs)."""
    s, s_p, xy, sep = 1.0, 1.0, False, False

    def forward(self, q):
        return q


@torch.no_grad()
def jacobian(warp, x, eps=1e-4):
    """(N, 3, 3) d phi / d x by central differences (x: (N, 3) normalized poses)."""
    cols = []
    for a in range(3):
        e = torch.zeros(1, 3, device=x.device)
        e[0, a] = eps
        cols.append((warp(x + e) - warp(x - e)) / (2 * eps))
    return torch.stack(cols, 2)


@torch.no_grad()
def quadrant_stretch(warp, maze, block=8, res=0.25):
    """Median sqrt(det J_xy) over poses inside the four block x block quadrant mazes (heading 0)."""
    n, E = maze.n, maze.E
    c = torch.arange(res / 2, n, res, device=maze.device)
    X, Y = torch.meshgrid(c, c, indexing='ij')
    inq = lambda z: (z < block) | (z >= n - block)
    m = inq(X) & inq(Y)
    q = torch.stack([(X[m] - n / 2) / E, (Y[m] - n / 2) / E, torch.zeros(int(m.sum()), device=maze.device)], 1)
    J = jacobian(warp, q)
    det = (J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]).abs()
    return float(det.sqrt().median())


@torch.no_grad()
def path_length_z(warp, maze, paths, res_cells=0.02):
    """Length of each (L, 3) path measured in z (densified to res_cells as plan.densify)."""
    import plan as P
    out = []
    for p in paths:
        d = P.densify(torch.as_tensor(np.asarray(p, np.float32), device=maze.device), res_cells * maze.pitch)
        z = torch.cat([warp(d[i:i + 65536]) for i in range(0, len(d), 65536)])
        out.append(float(torch.linalg.vector_norm(z[1:] - z[:-1], dim=1).sum()))
    return np.array(out)


@torch.no_grad()
def mppi_stretched(cost, warp, start, goal, seed, budget, lo, hi, step_z, sample_num=50, horizon=5, temperature=50.0, tol=0.01, **_):
    """plan.mppi with its perturbations in z = warp(x) (see the module docstring). Returns (paths, stop, converged)."""
    torch.manual_seed(seed)
    n, dev = len(start), start.device
    xp = torch.cat([start, goal], 1).clone()
    prior = torch.zeros(n, 3, device=dev)
    active = torch.ones(n, dtype=torch.bool, device=dev)
    budget = budget.to(dev).long()
    stop = budget.clone()
    converged = torch.zeros(n, dtype=torch.bool, device=dev)
    zg = warp(goal)
    trace = [xp[:, :3].clone()]
    for it in range(int(budget.max())):
        active = active & (budget > it)
        ids = torch.nonzero(active).squeeze(1)
        if not len(ids):
            break
        cur = xp[ids]
        A = torch.linalg.inv(jacobian(warp, cur[:, :3]))
        zc = warp(cur[:, :3])
        cap = torch.minimum(torch.full((len(ids),), float(step_z), device=dev), torch.linalg.vector_norm(zg[ids] - zc, dim=1).clamp(min=1e-12))[:, None, None, None]
        tmp = cur[:, None, None, :].repeat(1, sample_num, horizon, 1)
        dz = cap * torch.randn(len(ids), sample_num, 1, 3, device=dev) + cap * torch.randn(len(ids), sample_num, horizon, 3, device=dev)
        dz = dz + 2 * prior[ids][:, None, None, :]
        dz = dz / (torch.maximum(torch.linalg.vector_norm(dz, dim=3, keepdim=True), cap) / cap)
        dx = torch.einsum('mij,mshj->mshi', A, dz)
        tmp[..., :3] = tmp[..., :3] + torch.cumsum(dx, dim=2)
        tmp[..., 2] = tmp[..., 2].clamp(float(lo[2]), float(hi[2]))
        c = cost(tmp[:, :, [0, -1], :].reshape(-1, 6), ids).reshape(len(ids), sample_num, 2)
        c = 10 * c[..., 0] + c[..., 1]
        weight = torch.softmax(-temperature * c, dim=1)
        update = torch.einsum('bs,bsd->bd', weight, dz[:, :, 0, :])
        prior[ids] = update
        new = cur[:, :3] + torch.einsum('mij,mj->mi', A, update)
        new[:, 2] = new[:, 2].clamp(float(lo[2]), float(hi[2]))
        xp[ids, :3] = new
        trace.append(xp[:, :3].clone())
        done = torch.linalg.vector_norm(xp[ids, 3:] - xp[ids, :3], dim=1) < tol
        stop[ids[done]] = it + 1
        converged[ids[done]] = True
        active[ids[done]] = False
    trace = torch.stack(trace)
    paths = {k: torch.cat([start[k:k + 1], trace[1:int(stop[k]) + 1, k], goal[k:k + 1]]) for k in range(n)}
    return paths, stop, converged
