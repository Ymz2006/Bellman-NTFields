"""Cheapest wall crossings by a model's speed table against the true ones: the backup takes a min over candidate steps, so a wall is as
strong as its cheapest crossing. Read-only."""
import sys, json
import numpy as np, torch
sys.path.insert(0, '/home/jeffrey/bellman-ntfield/jeffrey/pair_lift')
import forensics as FX
PT, PN = FX.PT, FX.PN
torch.set_grad_enabled(False)
out = {}
for spec in sys.argv[1:]:
    net, scene, prepared, res, row = FX.load(spec)
    dev = FX.DEV
    with torch.enable_grad():
        table = PN.SpeedTable(net, scene, 32, 49, goals=4, seed=0)
    clear = PN.ClearTable(scene, 32, 49)
    gen = torch.Generator(dev).manual_seed(21)
    E, r = scene.E, 0.03
    tt_all, tl_all, start_d = [], [], []
    for _ in range(60):
        a = scene.uniform(12_000, gen)
        a = a[clear(a) > 0.02]                              # starts in free space
        N, K = len(a), 16
        dirs = torch.randn(N, K, 3, generator=gen, device=dev)
        dirs = dirs / torch.linalg.vector_norm(dirs, dim=2, keepdim=True)
        b = torch.minimum(torch.maximum(a[:, None] + dirs * r * (0.5 + 0.5 * torch.rand(N, K, 1, generator=gen, device=dev)), scene.lo), scene.hi)
        A = a[:, None].expand(-1, K, -1)
        tt, free = PT.step_time(scene, A, b, 137, clear=clear)
        tl = table.step_time(A, b, 33)
        endfree = clear(b.reshape(-1, 3)).reshape(N, K) > 0.02
        m = ~free & endfree                                  # steps that start free, cross an obstacle and end free: a wall crossing
        tt_all.append(tt[m]); tl_all.append(tl[m])
    tt, tl = torch.cat(tt_all) * E, torch.cat(tl_all) * E
    qs = torch.tensor([0.001, 0.01, 0.05, 0.25, 0.5], device=dev)
    rec = dict(n=int(len(tt)), true_cells=[round(float(v), 2) for v in torch.quantile(tt, qs)], table_cells=[round(float(v), 2) for v in torch.quantile(tl, qs)],
               ratio=[round(float(v), 3) for v in torch.quantile(tl / tt, qs)], below_half=float((tl < 0.5 * tt).float().mean()), below_10_cells=float((tl < 10).float().mean()), true_below_10_cells=float((tt < 10).float().mean()),
               min_true=float(tt.min()), min_table=float(tl.min()))
    print(spec, res.get('scene_name'), json.dumps(rec))
    out[row['run']] = rec
json.dump(out, open('/home/jeffrey/bellman-ntfield/jeffrey/pair_lift/report/crossings.json', 'w'), indent=1)
