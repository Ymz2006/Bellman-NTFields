"""Read-only forensics on registered pair-lift fields (plus maze 22 unless the run says otherwise); results go to report/forensics.json.

    CUDA_VISIBLE_DEVICES=0 python jeffrey/pair_lift/forensics.py sensitivity <run>[@epoch] ...   how much the field depends on its pair column
    ... cliffs <run> ...        |dT| over 0.1-cell steps in open space, split by whether the number of obstacles on the segment to the goal changes
    ... bycount <run> ...       detour index of detour pairs by the number of obstacles their straight segment enters
    ... quadrature              the 6-point trapezoid step time and the exact one against a 4,001-point reference
    ... backup <run> ...        the Bellman backup of a trained field: T - y at band and open-space pairs, share of free descent steps
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pairnet as PN  # noqa: E402
import pairtrain as PT  # noqa: E402
import scenes as SN  # noqa: E402
import diag as DG  # noqa: E402
LF = PN.LF
ROOT = PN.ROOT
OUT = HERE / 'report/forensics.json'
DEV = 'cuda:0'
_scene = {}


def scene_of(prepared):
    if str(prepared) not in _scene:
        _scene[str(prepared)] = SN.load_scene(prepared)[0]
    return _scene[str(prepared)]


def load(spec):
    path, _, ep = spec.partition('@')
    run = Path(path) if '/' in path else ROOT / f'results/registered_{path}'
    res = json.load(open(run / 'results.json'))
    prepared = Path(res['prepared'])
    scene = scene_of(prepared)
    ck = run / 'training/checkpoint.pt' if not ep else run / f'training/checkpoints/epoch_{int(ep):05d}.pt'
    return PN.load_field(ck, scene), scene, prepared, res, dict(run=run.name.replace('registered_', ''), epoch=int(ep) if ep else None, args=res['args'].get('pair', 'none'))


def save(section, rows):
    OUT.parent.mkdir(exist_ok=True)
    data = json.load(open(OUT)) if OUT.exists() else {}
    prev = {(r.get('run'), r.get('epoch')): r for r in data.get(section, [])}
    for r in rows:
        prev[(r.get('run'), r.get('epoch'))] = r
    data[section] = list(prev.values())
    json.dump(data, open(OUT, 'w'), indent=1, default=float)


def cost_of(net):
    import plan as P
    return P.field_cost(net)


def sensitivity(specs):
    rows = []
    for spec in specs:
        net, scene, prepared, res, row = load(spec)
        truth = DG.Truth(scene, prepared)
        pairs, gi, t_star, D = DG.truth_pairs(scene, truth, 20000)
        blk = LF.segment_blocked(scene, pairs)
        cost = cost_of(net)
        t1 = cost(pairs)
        keep = net.b_c.clone()
        net.b_c.zero_()
        t0 = cost(pairs)
        net.b_c.copy_(keep)
        E = scene.E
        row.update(t_blocked=float(t1[blk].mean() * E), t_blocked_zeroed=float(t0[blk].mean() * E), t_true=float(t_star[blk].mean() * E),
                   change_median=float((t1 - t0)[blk].abs().median() * E), change_q90=float((t1 - t0)[blk].abs().quantile(0.9) * E))
        rows.append(row)
        print(row)
    save('sensitivity', rows)


def cliffs(specs):
    rows = []
    for spec in specs:
        net, scene, prepared, res, row = load(spec)
        truth = DG.Truth(scene, prepared)
        gen = torch.Generator(DEV).manual_seed(3)
        n = 200000
        q = scene.free(n, gen, 0.2)
        gi = torch.randint(0, len(truth.goals), (n,), generator=gen, device=DEV)
        step = 0.1 * scene.pitch
        d = torch.randn(n, 3, generator=gen, device=DEV)
        d[:, 2] = 0
        q2 = q + d / torch.linalg.vector_norm(d, dim=1, keepdim=True) * step
        _, free = PT.step_time(scene, q, q2)
        ok = free & (scene.clearance(q2) > 0.2)
        q, q2, gi = q[ok], q2[ok], gi[ok]
        g = truth.goals[gi]
        tab = PN.ClearTable(scene)
        ch = PN.segment_count(tab, torch.cat([q, g], 1), scene.E) != PN.segment_count(tab, torch.cat([q2, g], 1), scene.E)
        cost = cost_of(net)
        dT = (cost(torch.cat([q, g], 1)) - cost(torch.cat([q2, g], 1))).abs() / step
        dTs = (truth.time(gi, q) - truth.time(gi, q2)).abs() / step
        for name, m in (('same', ~ch), ('changes', ch)):
            row[name] = dict(n=int(m.sum()), median=float(dT[m].median()), q90=float(dT[m].quantile(0.9)), q99=float(dT[m].quantile(0.99)), over2=float((dT[m] > 2).float().mean()),
                             truth_q99=float(dTs[m].quantile(0.99)))
        row['share_changes'] = float(ch.float().mean())
        rows.append(row)
        print(row)
    save('cliffs', rows)


def bycount(specs):
    rows = []
    for spec in specs:
        net, scene, prepared, res, row = load(spec)
        truth = DG.Truth(scene, prepared)
        pairs, gi, t_star, D = DG.truth_pairs(scene, truth, 40000)
        cnt = PN.segment_count(PN.ClearTable(scene), pairs, scene.E)
        det = t_star >= 1.3 * D
        t = cost_of(net)(pairs)
        rho = (t - D) / (t_star - D)
        row['by_count'] = {}
        for k in range(0, 7):
            m = cnt == k
            dm = det & m
            row['by_count'][str(k)] = dict(share=float(m.float().mean()), ratio=float((t / t_star)[m].median()) if int(m.sum()) > 20 else None, detours=int(dm.sum()),
                                           rho=float(rho[dm].median()) if int(dm.sum()) > 20 else None, seen_through=float((rho[dm] < 0.5).float().mean()) if int(dm.sum()) > 20 else None)
        rows.append(row)
        print(row)
    save('bycount', rows)


def quadrature():
    prepared = ROOT / 'results/registered_20261102T042454Z-62864e61'
    scene = scene_of(prepared)
    x = torch.tensor(np.load(prepared / 'data/sampled_points.npy', mmap_mode='r')[:20000, :3], device=DEV)
    gen = torch.Generator(DEV).manual_seed(1)
    rows = []
    for r in (0.03, 0.015):
        d = torch.randn(len(x), 3, generator=gen, device=DEV)
        d = d / torch.linalg.vector_norm(d, dim=1, keepdim=True) * r * (0.5 + 0.5 * torch.rand(len(x), 1, generator=gen, device=DEV))
        b = torch.minimum(torch.maximum(x + d, scene.lo), scene.hi)
        ref, free = PT.step_time(scene, x, b, 4001, exact=False)
        for name, kw in (('trapezoid, 6 points', dict(samples=6, exact=False)), ('antiderivative, 9 points', dict(samples=9)), ('antiderivative, 17 points', dict(samples=17))):
            t, _ = PT.step_time(scene, x, b, **kw)
            q = (t / ref)[free]
            rows.append(dict(run=f'{name} r={r}', epoch=None, radius=r, rule=name, median=float(q.median()), mean=float(q.mean()), q10=float(q.quantile(0.1)), q90=float(q.quantile(0.9)), q99=float(q.quantile(0.99))))
            print(rows[-1])
    save('quadrature', rows)


def backup(specs, K=16):
    rows = []
    for spec in specs:
        net, scene, prepared, res, row = load(spec)
        x = torch.tensor(np.load(prepared / 'data/sampled_points.npy', mmap_mode='r')[:6000], device=DEV)
        gen = torch.Generator(DEV).manual_seed(1)
        open_pairs = torch.cat([scene.free(6000, gen, 0.3), scene.free(6000, gen, 0.3)], 1)
        for name, pts in (('band', x), ('open', open_pairs)):
            with torch.enable_grad():
                tau, _, xp = net.out(pts)
                gg, = torch.autograd.grad(tau.sum(), xp)
            T = tau.detach()[:, 0]
            a = pts[:, :3]
            S = scene.train_speed(scene.clearance(a))
            g = -gg[:, :3]
            g = g / torch.linalg.vector_norm(g, dim=1, keepdim=True)
            dirs = torch.randn(len(pts), K, 3, generator=gen, device=DEV)
            dirs[:, 0] = g
            dirs[:, 1] = g
            dirs = dirs / torch.linalg.vector_norm(dirs, dim=2, keepdim=True)
            length = 0.03 * (0.5 + 0.5 * torch.rand(len(pts), K, 1, generator=gen, device=DEV))
            length[:, 0] = 0.03
            length[:, 1] = 0.015
            b = torch.minimum(torch.maximum(a[:, None] + dirs * length, scene.lo), scene.hi)
            new = pts[:, None].repeat(1, K, 1)
            new[:, :, :3] = b
            with torch.no_grad():
                Tn = net.out(new.reshape(-1, 6))[0][:, 0].reshape(len(pts), K)
            out = dict(slope=float((S * torch.linalg.vector_norm(gg[:, :3], dim=1)).median()))
            for rule, kw in (('trapezoid', dict(samples=6, exact=False)), ('exact', dict(samples=17))):
                tau_s, free = PT.step_time(scene, a[:, None].expand(-1, K, -1), b, **kw)
                y, arg = torch.where(free, tau_s + Tn, torch.full_like(Tn, float('inf'))).min(1)
                ok = torch.isfinite(y)
                out[rule] = dict(gap_mean=float((T - y)[ok].mean() * scene.E), gap_median=float((T - y)[ok].median() * scene.E), descent_free=float(free[:, 0].float().mean()),
                                 best_is_descent=float((arg <= 1)[ok].float().mean()))
            row[name] = out
        rows.append(row)
        print(row)
    save('backup', rows)


if __name__ == '__main__':
    cmd, specs = sys.argv[1], sys.argv[2:]
    with torch.no_grad():
        dict(sensitivity=lambda: sensitivity(specs), cliffs=lambda: cliffs(specs), bycount=lambda: bycount(specs), quadrature=quadrature, backup=lambda: backup(specs))[cmd]()
