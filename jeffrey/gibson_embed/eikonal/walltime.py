"""Minimal travel time on the grid under the P-NTFields speed model, and whether the optimum crosses a wall.

Speed s = clip(d_mesh, m/10, m)/m (m = 0.5/12) never vanishes: 0.1 at the surface. A thin wall is therefore a costly
slab, not a barrier, and the Eikonal (viscosity) solution crosses it whenever the detour costs more than the crossing
(about 6.6 m = 0.28 time units for a zero-thickness wall). For every evaluation pair this computes the minimal time
(a) through free cells only (line-of-sight diagonals, as the geodesic) and (b) through all cells (occupied cells at
speed 0.1, the wall thickness being that of the conservative rasterisation). If (b) < (a) the optimum crosses a wall
and a planner that follows it must collide.

    python jeffrey/gibson_embed/eikonal/walltime.py results/registered_<id> out.json [model,...]
"""
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffuse import Stencil  # noqa: E402

MARGIN = 0.5 / 12


def speed_field(occ, margin=MARGIN):
    """P-NTFields speed on the grid from the Euclidean distance transform (units of the box)."""
    d = ndimage.distance_transform_edt(~occ) / occ.shape[0]
    return np.clip(d, margin / 10, margin) / margin


def weighted_time(inv_s, support, los, sources, chunk=25, max_iter=20000, check_every=16, log=print):
    """(S, *shape) minimal travel time from each source cell; a move x -> x+o costs |o| (1/s(x) + 1/s(x+o)) / 2."""
    st = Stencil(support, r=math.sqrt(3), los=los)
    masks = [m.bool() for m in st.masks]
    cell = 1.0 / support.shape[0]
    weights = [math.sqrt(sum(c * c for c in o)) * cell for o in st.offsets]
    shape = tuple(support.shape)
    inf = float('inf')
    strides = torch.tensor([int(np.prod(shape[i + 1:])) for i in range(3)], device=support.device)
    padded_inv = st.pad_field(inv_s[None], value=inf)[0]
    half_cost = [None if w == 0 else w * 0.5 * (inv_s + st.shifted(padded_inv, o)) for o, w in zip(st.offsets, weights)]
    out = []
    for start in range(0, len(sources), chunk):
        src = sources[start:start + chunk]
        S = len(src)
        d = torch.full((S,) + shape, inf, device=support.device)
        d.view(S, -1)[torch.arange(S, device=support.device), (src * strides).sum(1)] = 0.0
        for it in range(1, max_iter + 1):
            padded = st.pad_field(d, value=inf)
            new = d.clone()
            for o, m, c in zip(st.offsets, masks, half_cost):
                if c is None:
                    continue
                cand = torch.where(m, st.shifted(padded, o) + c, inf)
                torch.minimum(new, cand, out=new)
            new[:, support == 0] = inf
            done = it % check_every == 0 and torch.equal(new, d)
            d = new
            if done:
                break
        log(f'  travel time sources {start}-{start + S}: {it} sweeps')
        out.append(d.cpu())
    return torch.cat(out)


def pair_optima(occ, pairs, device, log=print):
    """T_free, T_all (per pair) and crosses = T_all < T_free: the optimum passes through occupied cells."""
    res = occ.shape[0]
    inv_s = torch.tensor(1.0 / speed_field(occ), dtype=torch.float32, device=device)
    to_cell = lambda p: np.clip(np.floor((p + 0.5) * res).astype(np.int64), 0, res - 1)  # noqa: E731
    sc, gc = to_cell(pairs['start']), to_cell(pairs['goal'])
    uniq, inverse = np.unique(sc, axis=0, return_inverse=True)
    src = torch.tensor(uniq, device=device)
    free_t = torch.tensor(~occ, device=device)
    T_free = weighted_time(inv_s, free_t, True, src, log=log)
    T_all = weighted_time(inv_s, torch.ones_like(free_t), False, src, log=log)
    idx = (inverse.ravel(), gc[:, 0], gc[:, 1], gc[:, 2])
    tf, ta = T_free[idx].numpy(), T_all[idx].numpy()
    return tf, ta, ta < tf * (1 - 1e-3)


def main():
    run_dir, out_path = sys.argv[1], sys.argv[2]
    models = sys.argv[3].split(',') if len(sys.argv) > 3 else []
    dev = torch.device('cuda')
    occ = np.load(f'{run_dir}/grid.npz')['occupancy_geodesic']
    pairs = np.load(f'{run_dir}/pairs.npz')
    tf, ta, crosses = pair_optima(occ, pairs, dev)
    results = json.load(open(f'{run_dir}/results.json'))
    summary = dict(run=run_dir, per_pair=dict(T_free=tf.tolist(), T_all=ta.tolist(), crosses=crosses.tolist()))
    for kind in ['uniform', 'wall']:
        sel = pairs['kind'] == kind
        summary[kind] = dict(n=int(sel.sum()), optimum_crosses_wall=float(crosses[sel].mean()),
                             time_ratio_free_over_all_median=float(np.median((tf / ta)[sel & crosses])) if (sel & crosses).any() else None)
        print(f'{kind}: n {sel.sum()} optimum crosses a wall for {crosses[sel].mean():.3f}; '
              f'T_free/T_all median among those {summary[kind]["time_ratio_free_over_all_median"]}', flush=True)
        for name in models:
            if name not in results:
                continue
            succ = np.array(results[name]['planning']['per_pair']['success'])
            a, b = succ[sel & crosses], succ[sel & ~crosses]
            summary[kind][name] = dict(success_when_crosses=float(a.mean()) if len(a) else None, success_when_detour=float(b.mean()) if len(b) else None)
            print(f'   {name}: success {a.mean() if len(a) else float("nan"):.3f} when the optimum crosses (n={len(a)}), '
                  f'{b.mean() if len(b) else float("nan"):.3f} when it detours (n={len(b)})', flush=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out_path, 'w'), indent=2)


if __name__ == '__main__':
    main()
