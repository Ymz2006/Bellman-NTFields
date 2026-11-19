"""Scene-agnostic field diagnostics (near_wall and bumps are jeffrey/cross_warp/diag.py's, copied so a run imports no other study's
modules; failures_by_arrival applies the scorer pitfall rule: plan.mppi appends the goal to every path, so a stall's collision flag covers
the final jump; count wall contacts only among runs that arrived)."""
import numpy as np
import torch


def near_wall(scene, prepared, grad):
    hx = torch.tensor(np.load(prepared / 'heldout/sampled_points.npy'), device=scene.device)
    hd = torch.tensor(np.load(prepared / 'heldout/clearance_cells.npy'), device=scene.device)[:, 0]
    _, normal = scene.clearance_normal(hx[:, :3])
    S = scene.train_speed(hd)
    gap = torch.linalg.vector_norm(hx[:, :3] - hx[:, 3:], dim=1) / scene.pitch
    _, g = grad(hx)
    slope = S * torch.linalg.vector_norm(g, dim=1)
    cos = torch.nn.functional.cosine_similarity(g, -normal, dim=1)
    out = {}
    for lo, hi in [(0, 0.5), (0.5, 2), (2, 8), (8, 1e9)]:
        m = (gap >= lo) & (gap < hi) & (S < 0.3)
        if int(m.sum()) > 20:
            out[f'{lo:g}-{hi:g}' if hi < 1e8 else f'>{lo:g}'] = dict(n=int(m.sum()), slope=float(slope[m].median()), cos=float(cos[m].median()),
                                                                   attract=float((cos[m] < 0).float().mean()))
    m = S < 0.3
    out['all'] = dict(n=int(m.sum()), slope=float(slope[m].median()), cos=float(cos[m].median()), attract=float((cos[m] < 0).float().mean()))
    return out


def bumps(scene, prepared, cost, P, min_bin=3):
    q = np.load(prepared / 'queries.npz')
    tp = np.load(prepared / 'truth_paths.npy', allow_pickle=True)
    counts = []
    for k in np.nonzero(q['bin'] >= min_bin)[0]:
        p = P.densify(torch.tensor(np.asarray(tp[k], np.float32), device=scene.device), 0.02 * scene.pitch)
        g = torch.tensor(q['goal'][k], device=scene.device, dtype=torch.float32).expand(len(p), 3)
        t = cost(torch.cat([p, g], 1)).cpu().numpy()
        dt = np.diff(t)
        keep = np.abs(dt) > 1e-6 * t[0]
        sgn = np.sign(dt[keep])
        counts.append(int(((sgn[:-1] < 0) & (sgn[1:] > 0)).sum()))
    return dict(paths=len(counts), bumps_per_path=float(np.mean(counts)) if counts else None, paths_with_bump=int(np.sum(np.array(counts) > 0)))


def failures_by_arrival(rows):
    conv, coll, ok = rows['converged'], rows['collision'], rows['success']
    return dict(arrived_wall=int((~ok & conv & coll).sum()), not_arrived=int((~ok & ~conv).sum()), runs=int(ok.size))


def by_mask(rows, bins, mask, name):
    """Success counts for query subsets (mask over queries)."""
    s = rows['success'][:, mask]
    return {name: dict(success=int(s.sum()), runs=int(s.size))}
