"""Certified test suites for a scene outside experiments/cabinet/scenes (the real-world cupboard), container layout.

The cabinet generator (experiments/cabinet/make_tests.py: strict endpoints, random pairs 0.02-4 rad, witness = the straight line or
RRT-Connect, certified by the unchanged release scorer, 100 per stratum, six strata, the same seeds) is imported and run unchanged except
for one function: its pool sampler keeps only grasp points "near the table" (z >= -0.04), which would drop this scene's lower zones.
Here the pre-filter is the scene's own sampling box widened by 0.2 m on every side, and the stored free / box pools are capped at
POOL_CAP rows (a uniform prefix of the same stream) to bound memory. The suite files' `scene` field still reads experiments/cabinet/...
(written by the generator); the manifest's scene name and sha256 are those of the scene file given.
"""
import importlib.util
import sys
import numpy as np
import torch

_spec = importlib.util.spec_from_file_location('cabinet_make_tests', '/clean/experiments/cabinet/make_tests.py')
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)
SHARED = M.SHARED
POOL_CAP = 4_000_000
WIDEN = 0.2


@torch.no_grad()
def sample_pools(V, box, cupboard, compartments, samples, generator, target_cap):
    names = list(compartments)
    found = {k: [] for k in ['free', 'box'] + names}
    count = {k: 0 for k in found}
    targets = []
    near_box = tuple((float(lo) - WIDEN, float(hi) + WIDEN) for lo, hi in np.asarray(box))
    for _ in range(samples // M.BATCH):
        q = V.lo + (V.hi - V.lo) * torch.rand(M.BATCH, 6, device=V.dev, generator=generator)
        pos = V.robot.get_state(q).ee_pose.position.clone()
        keep = SHARED.inside(pos, near_box)
        if not keep.any():
            continue
        ok, pos = V(q[keep], positions=True)
        q, pos = q[keep][ok], pos[ok]
        if sum(len(t) for t in targets) < target_cap:
            targets.append(q.cpu())
        in_box = SHARED.inside(pos, box)
        masks = dict(box=in_box, free=in_box & ~SHARED.inside(pos, cupboard), **{k: SHARED.inside(pos, v) for k, v in compartments.items()})
        for k, m in masks.items():
            if m.any() and count[k] < POOL_CAP:
                found[k].append(torch.cat([q[m], pos[m]], 1).cpu())
                count[k] += int(m.sum())
    pools = {k: (torch.cat(v).numpy()[:POOL_CAP] if v else np.zeros((0, 9), np.float32)) for k, v in found.items()}
    pools['compartment'] = [pools[k] for k in names]
    return pools, torch.cat(targets)[:target_cap]


M.sample_pools = sample_pools
if __name__ == '__main__':
    M.main()
