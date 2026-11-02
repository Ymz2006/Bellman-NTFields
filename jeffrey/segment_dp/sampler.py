"""Configuration pool and distance-segment pair sampling for the bank-free (eikonal / segment-DP) trainers.

No transitions are used. The pool is the set of collision-checked configurations the bank was built on
(bank/source_q.npy positions only: on the no-wall task the fast region is a thin manifold, 99 % of the joint box
sits at the 0.1 speed floor, so configurations are not drawn from the box); their speeds come from the frozen speed
network. A pair for segment [lo, hi] takes a source q uniformly among the sources that reach lo (their farthest
partner is at least lo away), a target distance t uniformly in [lo, hi], and the goal g whose distance to q is
closest to t, searched among the 64 nearest neighbours of q and a random subset of the pool (shared by the rows of
one call). q itself and its duplicates (distance < 2e-3: the float32 expansion |x|^2 + |p|^2 - 2 x.p is only good to
~1e-3 near zero) are never chosen; neighbour distances are recomputed exactly.
"""
import json
import numpy as np
import torch
from common import sha
from speed import load_speed


def load_pool(prepared, device):
    """Positions and frozen-network speeds of the pool, with the manifest checks of the original trainer."""
    bank = prepared/'bank'; manifest = json.loads((bank/'manifest.json').read_text())
    assert sha(bank/'source_q.npy') == manifest['file_sha256']['source_q.npy']
    assert sha(prepared/'speed.pt') == manifest['speed_checkpoint_sha256']
    pool = torch.as_tensor(np.load(bank/'source_q.npy'), device=device)
    speed = load_speed(prepared/'speed.pt', device)
    with torch.no_grad():
        s = torch.cat([speed(pool[i:i+65536]).reshape(-1) for i in range(0, len(pool), 65536)])
    return pool, s, speed, dict(source_q_sha256=manifest['file_sha256']['source_q.npy'], speed_sha256=manifest['speed_checkpoint_sha256'],
                                scene_sha256=manifest['scene_sha256'], pool_size=len(pool))


DUPLICATE = 2e-3


def pairwise(a, b, square_b=None):
    square_b = b.square().sum(1) if square_b is None else square_b
    return torch.addmm(square_b[None, :], a, b.T, alpha=-2.0).add_(a.square().sum(1, keepdim=True)).clamp_min_(0.0).sqrt_()


class PairSampler:
    def __init__(self, pool, edges, blend, generator, subset=16384, neighbours=64, reach_subset=32768, floor=0.004):
        self.pool, self.square, self.generator, self.subset = pool, pool.square().sum(1), generator, subset
        edges = torch.tensor(edges, device=pool.device, dtype=pool.dtype)
        self.lo = (edges[:-1]-blend).clamp_min(floor); self.hi = edges[1:]+blend
        n = len(pool); cols = torch.randint(n, (reach_subset,), generator=generator, device=pool.device)
        self.reach = torch.empty(n, device=pool.device); self.knn_d = torch.empty((n, neighbours), device=pool.device)
        self.knn_i = torch.empty((n, neighbours), dtype=torch.long, device=pool.device)
        for i in range(0, n, 256):
            x = pool[i:i+256]; self.reach[i:i+256] = pairwise(x, pool[cols], self.square[cols]).amax(1)
            d = pairwise(x, pool, self.square); d[torch.arange(len(x), device=pool.device), torch.arange(i, i+len(x), device=pool.device)] = torch.inf
            d.masked_fill_(d < DUPLICATE, torch.inf); top = d.topk(neighbours, dim=1, largest=False)
            exact = torch.linalg.vector_norm(pool[top.indices]-x[:, None], dim=2)
            self.knn_i[i:i+256] = top.indices; self.knn_d[i:i+256] = exact.masked_fill_(exact < DUPLICATE, torch.inf)
        self.eligible = [torch.nonzero(self.reach >= lo).flatten() for lo in self.lo.tolist()]
        assert all(len(e) > 1000 for e in self.eligible), [len(e) for e in self.eligible]

    def sample(self, segments):
        """segments: LongTensor [n] of segment ids -> (q index, g index, achieved distance, target distance)."""
        n = len(segments); qi = torch.empty(n, dtype=torch.long, device=self.pool.device)
        for m in torch.unique(segments).tolist():
            rows = torch.nonzero(segments == m).flatten(); pick = self.eligible[m]
            qi[rows] = pick[torch.randint(len(pick), (len(rows),), generator=self.generator, device=self.pool.device)]
        target = self.lo[segments]+torch.rand(n, generator=self.generator, device=self.pool.device)*(self.hi[segments]-self.lo[segments])
        near = (self.knn_d[qi]-target[:, None]).abs(); a = near.argmin(1); error_a = near.gather(1, a[:, None])[:, 0]
        cols = torch.randint(len(self.pool), (self.subset,), generator=self.generator, device=self.pool.device)
        d = pairwise(self.pool[qi], self.pool[cols], self.square[cols]); d.masked_fill_((d < 2.5*DUPLICATE) | (cols[None, :] == qi[:, None]), torch.inf)
        far = d.sub_(target[:, None]).abs_(); b = far.argmin(1); error_b = far.gather(1, b[:, None])[:, 0]
        gi = torch.where(error_a <= error_b, self.knn_i[qi, a], cols[b])
        return qi, gi, torch.linalg.vector_norm(self.pool[qi]-self.pool[gi], dim=1), target
