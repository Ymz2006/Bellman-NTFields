"""Embedding-only diagnostics (no training): does ||phi(x) - phi(y)|| tell wall-straddling pairs (Euclidean-close,
geodesic-far) from corridor pairs at the same Euclidean separation? AUC is stratified by Euclidean distance."""
import numpy as np
import torch
from scipy.stats import rankdata, spearmanr


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def stratified_auc(score, label, euclid, bins=6):
    edges = np.quantile(euclid, np.linspace(0, 1, bins + 1))
    vals, weights = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (euclid >= lo) & (euclid <= hi)
        a = auc(score[sel & label], score[sel & ~label])
        if not np.isnan(a):
            vals.append(a)
            weights.append(min((sel & label).sum(), (sel & ~label).sum()))
    return float(np.average(vals, weights=weights)) if vals else float('nan')


def local_pairs(free, sources, dist, radius_cells, per_source, seed):
    """Pairs (source cell, random free cell within a box of radius_cells) with Euclidean and geodesic distances."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    S, dim = sources.shape
    res = free.shape[0]
    off = torch.randint(-radius_cells, radius_cells + 1, (S, per_source, dim), generator=g).to(free.device)
    y = (sources[:, None] + off).clamp(0, res - 1)
    sidx = torch.arange(S, device=free.device)[:, None].expand(S, per_source)
    strides = torch.tensor([int(np.prod(free.shape[i + 1:])) for i in range(dim)], device=free.device)
    lin = (y * strides).sum(-1)
    geo = dist.view(S, -1)[sidx, lin]
    euclid = (y - sources[:, None]).float().norm(dim=-1) / res
    keep = free.view(-1)[lin] & torch.isfinite(geo) & (euclid > 0)
    return sidx[keep], y[keep], euclid[keep], geo[keep]


def probe(features, free, sources, dist, radius_cells, per_source=2000, seed=0, wall_ratio=2.5, corridor_ratio=1.15):
    sidx, y, euclid, geo = local_pairs(free, sources, dist, radius_cells, per_source, seed)
    ratio = (geo / euclid).cpu().numpy()
    euclid_np, geo_np = euclid.cpu().numpy(), geo.cpu().numpy()
    wall, corridor = ratio >= wall_ratio, ratio <= corridor_ratio
    sel = wall | corridor
    results = dict(pairs=int(len(ratio)), wall_pairs=int(wall.sum()), corridor_pairs=int(corridor.sum()), radius_cells=radius_cells, features={})
    s_cells = sources[sidx]
    for name, phi in features.items():
        fs = phi[(slice(None),) + tuple(s_cells.T)].float().T
        fy = phi[(slice(None),) + tuple(y.T)].float().T
        dphi = ((fs - fy).norm(dim=1) / np.sqrt(phi.shape[0])).cpu().numpy()
        results['features'][name] = dict(
            channels=int(phi.shape[0]),
            auc_wall_vs_corridor=auc(dphi[wall], dphi[corridor]),
            auc_stratified=stratified_auc(dphi[sel], wall[sel], euclid_np[sel]),
            spearman_geodesic=float(spearmanr(dphi, geo_np).statistic),
            spearman_ratio=float(spearmanr(dphi, ratio).statistic),
            spearman_euclid=float(spearmanr(dphi, euclid_np).statistic),
            wall_over_corridor_median=float(np.median(dphi[wall]) / max(np.median(dphi[corridor]), 1e-12)))
    return results
