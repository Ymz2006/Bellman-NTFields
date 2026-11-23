"""Step cost from the field's own speed against the true step time (lifted length), on random steps from training pairs. Read-only."""
import sys, json
import numpy as np, torch
sys.path.insert(0, '/home/jeffrey/bellman-ntfield/jeffrey/pair_lift')
import forensics as FX
PT, PN = FX.PT, FX.PN
out = {}
for spec in sys.argv[1:]:
    net, scene, prepared, res, row = FX.load(spec)
    pts = np.load(prepared / 'data/sampled_points.npy', mmap_mode='r')
    idx = np.sort(np.random.default_rng(0).choice(len(pts), 6000, replace=False))
    P = torch.as_tensor(np.asarray(pts[idx])[:, :6], dtype=torch.float32, device=FX.DEV)
    table = PN.ClearTable(scene, 32, 49)
    gen = torch.Generator(FX.DEV).manual_seed(7)
    N, K, r, E = len(P), 8, 0.03, scene.E
    a = P[:, :3]
    dirs = torch.randn(N, K, 3, generator=gen, device=FX.DEV)
    dirs = dirs / torch.linalg.vector_norm(dirs, dim=2, keepdim=True)
    b = torch.minimum(torch.maximum(a[:, None] + dirs * r * (0.5 + 0.5 * torch.rand(N, K, 1, generator=gen, device=FX.DEV)), scene.lo), scene.hi)
    with torch.no_grad():
        A = a[:, None].expand(-1, K, -1)
        tt, free = PT.step_time(scene, A, b, 137, clear=table)
        p = A[..., None, :] + torch.linspace(0, 1, 137, device=FX.DEV)[:, None] * (b - A)[..., None, :]
        dmin = table(p.reshape(-1, 3)).reshape(N, K, 137).amin(-1)
    sl = slice(0, 3)
    t4 = PT.learned_step_time(net, P, sl, a, b, 4, gen)
    t32 = PT.learned_step_time(net, P, sl, a, b, 32, gen)
    q = lambda x, m: [round(float(v), 3) for v in torch.quantile((x[m]), torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=x.device))]  # noqa: E731
    groups = {'free, open (clearance above 0.175 cells)': free & (dmin > 0.175), 'free, in the speed band': free & (dmin <= 0.175) & (dmin > 0.05), 'free, within 0.05 cells of an obstacle': free & (dmin <= 0.05),
              'through an obstacle': ~free}
    print(spec, res.get("scene_name"), "steps", N * K, "free", float(free.float().mean()), "dmin", float(dmin.min()), float(dmin.median()), float(dmin.max()), {k: int(v.sum()) for k, v in groups.items()})
    rows = {}
    for name, m in groups.items():
        if int(m.sum()) < 20:
            continue
        rows[name] = dict(share=float(m.float().mean()), ratio32=q(t32 / tt, m), ratio4=q(t4 / tt, m), within10=float((((t32 / tt)[m] - 1).abs() < 0.1).float().mean()), noise4=float(((t4 / t32)[m] - 1).abs().median()),
                          true_cells=float(tt[m].median() * E), learned_cells=float(t32[m].median() * E))
        print(f"  {name:42s} share {rows[name]['share']:.3f}  learned / true (5, 25, 50, 75, 95 %) {rows[name]['ratio32']}  within 10 % {rows[name]['within10']:.2f}  4-sample noise {rows[name]['noise4']:.3f}  median true {rows[name]['true_cells']:.2f} learned {rows[name]['learned_cells']:.2f} cells")
    out[spec] = rows
json.dump(out, open('/tmp/claude-1000/-home-jeffrey-bellman-ntfield/328cb179-735e-4288-a77f-99051bf68464/scratchpad/lspeed.json', 'w'), indent=1)
