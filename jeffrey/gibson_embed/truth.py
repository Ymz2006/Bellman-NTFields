"""Geodesic distance through free space from source cells: min-plus relaxation on the GPU.

Moves: the 8 (2-D) / 26 (3-D) neighbours with Euclidean weights; a diagonal move is allowed only when the cells of
its unit box are free (line-of-sight), so no path cuts a wall corner. Converges when a full sweep changes nothing.
"""
import math
import numpy as np
import torch
from diffuse import Stencil


def choose_sources(free_np, clearance_np, labels_np, count, min_clearance, seed, max_clearance=float('inf')):
    """Random free cells of the largest free component with min_clearance <= clearance <= max_clearance (cells).

    The upper bound keeps sources near walls, where the sharp (wall-straddling) pairs are."""
    rng = np.random.default_rng(seed)
    sizes = np.bincount(labels_np.ravel())[1:]
    main = int(np.argmax(sizes)) + 1
    ok = free_np & (clearance_np >= min_clearance) & (clearance_np <= max_clearance) & (labels_np == main)
    cells = np.stack(np.nonzero(ok), 1)
    return cells[rng.choice(len(cells), count, replace=False)]


def geodesic(free, sources, chunk=32, max_iter=50000, check_every=16, log=print):
    """dist (S, *shape): grid-geodesic distance (units of the [-0.5, 0.5] box) from each source cell; inf if unreachable."""
    st = Stencil(free, r=math.sqrt(free.ndim), los=True)
    masks = [m.bool() for m in st.masks]
    cell = 1.0 / free.shape[0]
    weights = [math.sqrt(sum(c * c for c in o)) * cell for o in st.offsets]
    shape = tuple(free.shape)
    inf = float('inf')
    strides = torch.tensor([int(np.prod(shape[i + 1:])) for i in range(free.ndim)], device=free.device)
    out = []
    for start in range(0, len(sources), chunk):
        src = sources[start:start + chunk]
        S = len(src)
        d = torch.full((S,) + shape, inf, device=free.device)
        d.view(S, -1)[torch.arange(S, device=free.device), (src * strides).sum(1)] = 0.0
        for it in range(1, max_iter + 1):
            padded = st.pad_field(d, value=inf)
            new = d.clone()
            for o, m, w in zip(st.offsets, masks, weights):
                if w == 0:
                    continue
                cand = torch.where(m, st.shifted(padded, o) + w, inf)
                torch.minimum(new, cand, out=new)
            new[:, ~free] = inf
            done = it % check_every == 0 and torch.equal(new, d)
            d = new
            if done:
                break
        log(f'  geodesic sources {start}-{start + S}: {it} sweeps')
        out.append(d)
    return torch.cat(out)
