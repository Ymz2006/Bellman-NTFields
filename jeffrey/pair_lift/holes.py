"""Holes in a collision check made from a model's speed table: wall-crossing steps (start free, cross an obstacle, end free) that the
check does not flag, by threshold, and where they are. Read-only."""
import sys, json
import numpy as np, torch
sys.path.insert(0, '/home/jeffrey/bellman-ntfield/jeffrey/pair_lift')
import forensics as FX
PT, PN = FX.PT, FX.PN
torch.set_grad_enabled(False)
spec, seed = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
net, scene, prepared, res, row = FX.load(spec)
dev = FX.DEV
with torch.enable_grad():
    table = PN.SpeedTable(net, scene, 32, 49, goals=4, seed=seed)
clear = PN.ClearTable(scene, 32, 49)
gen = torch.Generator(dev).manual_seed(31)
E, r = scene.E, 0.03
THR = (0.04, 0.06, 0.08, 0.12, 0.2)
n_cross, miss, where = 0, {t: 0 for t in THR}, {t: [] for t in THR}
for it in range(400):
    a = scene.uniform(20_000, gen)
    d0 = clear(a)
    a = a[(d0 > 0.0) & (d0 < 0.6)]                       # starts in free space near obstacles
    N, K = len(a), 16
    dirs = torch.randn(N, K, 3, generator=gen, device=dev)
    dirs = dirs / torch.linalg.vector_norm(dirs, dim=2, keepdim=True)
    b = torch.minimum(torch.maximum(a[:, None] + dirs * r * (0.5 + 0.5 * torch.rand(N, K, 1, generator=gen, device=dev)), scene.lo), scene.hi)
    A = a[:, None].expand(-1, K, -1)
    ts = torch.linspace(0, 1, 33, device=dev)
    p = A[..., None, :] + ts[:, None] * (b - A)[..., None, :]
    dm = clear(p.reshape(-1, 3)).reshape(N, K, 33)
    cross = (dm.amin(-1) < -0.05) & (dm[..., -1] > 0.0)   # goes at least 0.05 cells deep and ends free: a real crossing, not a grazed corner
    _, smax = table.step_time(A, b, 33, with_max=True)
    n_cross += int(cross.sum())
    for t in THR:
        m = cross & (smax < 1.0 / t)
        miss[t] += int(m.sum())
        if int(m.sum()) and len(where[t]) < 4000:
            mid = 0.5 * (A[m] + b[m])
            where[t] += [[round(float(x), 2) for x in ((q[:2] + 0.5) * E).tolist()] + [round(float(q[2] / scene.phi_max), 2)] for q in mid[:200]]
print(spec, 'table seed', seed, 'crossings', n_cross, {t: (miss[t], f'{100 * miss[t] / max(n_cross, 1):.4f} %') for t in THR})
for t in THR[:3]:
    if where[t]:
        w = np.array(where[t])
        cells, counts = np.unique(np.floor(w[:, :2]).astype(int), axis=0, return_counts=True)
        order = np.argsort(-counts)[:8]
        print(f'  unflagged at {t}: {len(w)} listed, in {len(cells)} cells; busiest cells (x, y: count)', [(cells[i].tolist(), int(counts[i])) for i in order], 'heading/phi_max quantiles', np.round(np.quantile(w[:, 2], [0.1, 0.5, 0.9]), 2).tolist())
json.dump(dict(crossings=n_cross, missed={str(t): miss[t] for t in THR}, where={str(t): where[t][:500] for t in THR}), open(f'/home/jeffrey/bellman-ntfield/jeffrey/pair_lift/report/holes_{row["run"][-8:]}_s{seed}.json', 'w'))
