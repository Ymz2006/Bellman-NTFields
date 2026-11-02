"""Stage 3 of the maze-scale test: segment DP on a maze, on top of ntrl-demo's trained field, planned and diagnosed.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/maze_scale/segdp.py --prepared <stage-1 dir> --base <ntrl checkpoint.pt> --seed 87358 --out <dir>

The interval-DP transition of jeffrey/segment_dp (dp.py), on maze C-space distances:
    T_(m+1)(i, j) = min over k of  T_<=m(i, k) + T_<=m(k, j),   |i - k| <= R_m and |k - j| <= R_m,
where T_<=m is the frozen chain of the base and DP levels 1..m (fields.ChainField) and R_m = e_m (1 + blend), the
largest gap those models answer. Segments are geometric in cells: the base (ntrl-demo's own field) is trusted below
e_1 = --base-cells, then e_(k+1) = --ratio x e_k up to the largest gap in the maze; level m is a fresh ntrl-demo
models/metric NN (or, --init warm, the previous level's weights) trained on pool pairs of its segment widened by the
blend. Split points k are pool configurations (stage-1 pool.npy) in the lens: half of --candidates the lens points
nearest the straight line (smallest |i - k| + |k - j|), half uniform over the lens; pairs with an empty lens are
dropped. --check-pieces: a piece shorter than the base range (answered by the base) counts only if its straight segment
is collision-free (exact clearance every 0.02 cells), so the min cannot pick a through-wall underestimate of the base.
Regression loss --loss (logl1 = |log T - log y|, scale-free across maze sizes), AdamW --lr, weight decay 1e-6,
batch --batch, --updates per level. Distances are exact (no |a|^2 + |b|^2 - 2ab expansion).
Diagnostics per level (truth): probe pairs (pool pose, truth goal) in the segment; DP target / T*, the trained level /
T*, the base (ntrl-demo alone) / T*. Then the chain is planned and diagnosed exactly as stage 2 (evaluate.py).
Outputs: chain/field.pt (+ chain/level_XX.pt), progress.json, results.json, rows.npz, paths_seed0.npy, slice_goal0.npy.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import maze as MZ  # noqa: E402
import plan as P  # noqa: E402
import truth as TR  # noqa: E402
import evaluate as EV  # noqa: E402
from fields import ChainField, save_chain  # noqa: E402
from common import Logger, dump  # noqa: E402


def dists(a, pool, chunk=64):
    """Exact Euclidean distances (len(a), len(pool))."""
    return torch.cat([torch.linalg.vector_norm(a[s:s + chunk, None, :] - pool[None], dim=2) for s in range(0, len(a), chunk)])


@torch.no_grad()
def reach(pool, gen, subset=8192):
    """Largest gap from every pool pose to a random subset of the pool (who can have a partner at a given gap)."""
    cols = pool[torch.randint(len(pool), (subset,), generator=gen, device=pool.device)]
    return torch.cat([dists(pool[s:s + 4096], cols).amax(1) for s in range(0, len(pool), 4096)])


@torch.no_grad()
def sample_pairs(pool, count, lo, hi, gen, eligible, chunk=256, max_tries=4000):
    """i uniform over the eligible pool poses (reach >= lo), j uniform over the pool points at gap in [lo, hi] from i
    (rows without one dropped; at most max_tries chunks)."""
    out_i, out_j = [], []
    tries = 0
    while sum(len(x) for x in out_i) < count and tries < max_tries:
        tries += 1
        i = eligible[torch.randint(len(eligible), (chunk,), generator=gen, device=pool.device)]
        d = dists(pool[i], pool)
        ok = (d >= lo) & (d <= hi)
        keys = torch.rand(d.shape, generator=gen, device=pool.device).masked_fill_(~ok, -1.0)
        best, j = keys.max(1)
        keep = best >= 0
        out_i.append(i[keep])
        out_j.append(j[keep])
    return torch.cat(out_i)[:count], torch.cat(out_j)[:count]


@torch.no_grad()
def segment_free(maze, x, y, res_cells=0.02):
    """True where the straight C-space segment x -> y (rows) is collision-free (exact clearance every <= res_cells)."""
    steps = int(math.ceil(float(torch.linalg.vector_norm(x - y, dim=1).max()) / (res_cells * maze.pitch))) + 1
    t = torch.linspace(0, 1, steps + 1, device=x.device)
    free = torch.ones(len(x), dtype=torch.bool, device=x.device)
    for s in range(0, len(x), 16384):
        a, b = x[s:s + 16384], y[s:s + 16384]
        pts = a[:, None] + t[None, :, None] * (b - a)[:, None]
        free[s:s + 16384] = (maze.clearance(pts.reshape(-1, 3)).view(len(a), -1) >= 0).all(1)
    return free


@torch.no_grad()
def targets(teacher, pool, a, b, radius, candidates, gen, chunk=128, check=None):
    """DP target min_k T(a, k) + T(k, b) over pool poses k in the lens (both pieces <= radius, k distinct from a, b).
    check = (maze, gap): pieces shorter than gap (the ones the base answers) only count if their straight segment is
    collision-free."""
    half = candidates // 2
    out = dict(target=[], lens=[], near_won=[], detour=[])
    for s in range(0, len(a), chunk):
        x, y = a[s:s + chunk], b[s:s + chunk]
        n = len(x)
        da, db = dists(x, pool), dists(y, pool)
        lens = (da <= radius) & (db <= radius) & (da > 1e-6) & (db > 1e-6)
        ell = (da + db).masked_fill(~lens, float('inf'))
        near = ell.topk(min(half, ell.shape[1]), dim=1, largest=False)
        keys = torch.rand(lens.shape, generator=gen, device=pool.device).masked_fill_(~lens, -1.0)
        rnd = keys.topk(min(half, keys.shape[1]), dim=1)
        index = torch.cat([near.indices, rnd.indices], 1)
        valid = torch.cat([torch.isfinite(near.values), rnd.values >= 0], 1)
        c = index.shape[1]
        k = pool[index.reshape(-1)]
        left = teacher.cost(torch.cat([x.repeat_interleave(c, 0), k], 1)).view(n, c)
        right = teacher.cost(torch.cat([k, y.repeat_interleave(c, 0)], 1)).view(n, c)
        if check is not None:
            maze, short = check
            xa, kk, yb = x.repeat_interleave(c, 0), k, y.repeat_interleave(c, 0)
            for u, v, piece in ((xa, kk, left), (kk, yb, right)):
                g = torch.linalg.vector_norm(u - v, dim=1)
                sel = torch.nonzero(g <= short).squeeze(1)
                if len(sel):
                    bad = ~segment_free(maze, u[sel], v[sel])
                    piece.view(-1)[sel[bad]] = float('inf')
        total = (left + right).masked_fill(~valid, float('inf'))
        best, which = total.min(1)
        rows = torch.arange(n, device=pool.device)
        w = index[rows, which]
        gap = torch.linalg.vector_norm(x - y, dim=1)
        out['target'].append(best)
        out['lens'].append(lens.sum(1))
        out['near_won'].append(which < half)
        out['detour'].append((da[rows, w] + db[rows, w]) / gap)
    return {key: torch.cat(v) for key, v in out.items()}


def fresh_net(seed, device):
    """An ntrl-demo models/metric NN initialized as its Model.train() does (B ~ N(0, 1) truncated at 2)."""
    torch.manual_seed(seed)
    B = torch.normal(0, 1, size=(128, 3))
    torch.nn.init.trunc_normal_(B, mean=0.0, std=1, a=-2.0, b=2.0)
    net = P.network_class('metric')(device, 3, B)
    net.apply(net.init_weights)
    return net.to(device), B


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--seed', type=int, default=87358)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--base-cells', type=float, default=0.5)
    p.add_argument('--ratio', type=float, default=1.35)
    p.add_argument('--blend', type=float, default=0.05)
    p.add_argument('--pairs', type=int, default=100_000)
    p.add_argument('--candidates', type=int, default=128)
    p.add_argument('--updates', type=int, default=6000)
    p.add_argument('--batch', type=int, default=2000)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--init', choices=['warm', 'fresh'], default='warm')
    p.add_argument('--loss', choices=['logl1', 'l1', 'smoothl1'], default='logl1')
    p.add_argument('--probes', type=int, default=2000)
    p.add_argument('--check-pieces', action='store_true', help='base-range pieces count only if their straight segment is collision-free')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    (a.out / 'chain').mkdir(parents=True, exist_ok=True)
    log = Logger(a.out / 'log.txt')
    dev = 'cuda:0'
    t0 = time.time()
    if a.smoke:
        a.pairs, a.updates, a.probes = 2000, 30, 200
    info = json.load(open(a.prepared / 'maze.json'))
    maze = MZ.Maze(info['n'], dev)
    assert np.array_equal(maze.V.astype(int), np.array(info['V'])), 'prepared maze differs'
    torch.manual_seed(a.seed)
    gen = torch.Generator(dev).manual_seed(a.seed)
    pool = torch.tensor(np.load(a.prepared / 'pool.npy'), device=dev)
    goals = torch.tensor(np.load(a.prepared / 'goals.npy'), device=dev)
    base = P.load_field(a.base, 'metric', dev)
    grid = TR.Grid(maze)
    del grid.costs, grid.cost_flat
    torch.cuda.empty_cache()
    fields = torch.tensor(np.load(a.prepared / 'truth_fields.npy'), device=dev).float()
    # edges (cells -> normalized): base below e_1, geometric after; the last level runs to the 99.9th percentile gap
    pg = dists(goals, pool)                                              # (G, P) probe gaps
    reach_all = reach(pool, gen)
    gaps = dists(pool[torch.randint(len(pool), (512,), generator=gen, device=dev)], pool)
    max_gap = float(torch.quantile(gaps.flatten()[::31], 0.999))
    del gaps
    edges_cells = [a.base_cells]
    while edges_cells[-1] * a.ratio * maze.pitch < max_gap:
        edges_cells.append(edges_cells[-1] * a.ratio)
    edges = [e * maze.pitch for e in edges_cells]
    L = len(edges) if not a.smoke else min(3, len(edges))
    res = dict(stage='segdp', maze=maze.describe(), seed=a.seed, base=str(a.base), prepared=str(a.prepared), smoke=a.smoke, edges_cells=edges_cells, levels=L, max_gap_cells=max_gap / maze.pitch,
               config={k: v for k, v in vars(a).items() if k not in ('prepared', 'base', 'out')}, gpu=torch.cuda.get_device_name(0), level_stats=[])
    log(f'segment DP n={maze.n}: {L} levels over edges (cells) {[round(e, 2) for e in edges_cells]} to {max_gap / maze.pitch:.2f}, pool {len(pool)}')
    models, Bs, states = [base], [], []
    teacher = ChainField(models, [], a.blend)
    for m in range(1, L + 1):
        tl = time.time()
        lo = edges[m - 1] * (1 - a.blend)
        hi = (edges[m] * (1 + a.blend)) if m < len(edges) else max_gap
        radius = edges[m - 1] * (1 + a.blend)
        i, j = sample_pairs(pool, a.pairs, lo, hi, gen, torch.nonzero(reach_all >= lo).squeeze(1))
        check = (maze, edges[0] * (1 + a.blend)) if a.check_pieces else None
        t = targets(teacher, pool, pool[i], pool[j], radius, a.candidates, gen, check=check)
        keep = torch.isfinite(t['target'])
        gap = torch.linalg.vector_norm(pool[i] - pool[j], dim=1)
        st = dict(level=m, segment_cells=[lo / maze.pitch, hi / maze.pitch], radius_cells=radius / maze.pitch, pairs=len(i), kept=int(keep.sum()),
                  empty_lens=float((~keep).float().mean()), lens_median=float(t['lens'].float().median()), near_won=float(t['near_won'][keep].float().mean()),
                  detour_median=float(t['detour'][keep].median()), target_over_gap_median=float((t['target'][keep] / gap[keep]).median()), target_seconds=time.time() - tl)
        i, j, y = i[keep], j[keep], t['target'][keep]
        # model: warm start from the previous level (level 1 from the base), or fresh
        if a.init == 'warm':
            prev = models[-1]
            B = (prev.B.T).detach().cpu().clone()
            net = P.network_class('metric')(dev, 3, B).to(dev)
            net.load_state_dict(prev.state_dict(), strict=True)
        else:
            net, B = fresh_net(a.seed * 1000 + m, dev)
        net.train()
        for prm in net.parameters():
            prm.requires_grad_(True)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-6)
        order = torch.randperm(len(i), generator=gen, device=dev)
        hold, train = order[:len(i) // 50], order[len(i) // 50:]

        def loss_fn(v, target):
            if a.loss == 'logl1':
                return (v.clamp(min=1e-6).log() - target.log()).abs().mean()
            if a.loss == 'l1':
                return (v - target).abs().mean()
            return torch.nn.functional.smooth_l1_loss(v, target)

        with torch.no_grad():
            v0 = torch.cat([net.out(torch.cat([pool[i[hold[s:s + 8192]]], pool[j[hold[s:s + 8192]]]], 1))[0][:, 0] for s in range(0, len(hold), 8192)])
            st['init_mean_abs_log'] = float((v0.clamp(min=1e-6) / y[hold]).log().abs().mean())
        for step in range(a.updates):
            rows = train[torch.randint(len(train), (a.batch,), generator=gen, device=dev)]
            v = net.out(torch.cat([pool[i[rows]], pool[j[rows]]], 1))[0][:, 0]
            loss = loss_fn(v, y[rows])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('non-finite loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        net.eval()
        for prm in net.parameters():
            prm.requires_grad_(False)
        with torch.no_grad():
            def fit(rows):
                v = torch.cat([net.out(torch.cat([pool[i[rows[s:s + 8192]]], pool[j[rows[s:s + 8192]]]], 1))[0][:, 0] for s in range(0, len(rows), 8192)])
                return float((v.clamp(min=1e-6) / y[rows]).log().abs().mean())
            st.update(train_mean_abs_log=fit(train[:len(hold)]), held_out_mean_abs_log=fit(hold))
        # truth probes for this segment
        gi, pi = torch.nonzero((pg >= lo) & (pg <= hi), as_tuple=True)
        if len(gi):
            sel = torch.randperm(len(gi), generator=gen, device=dev)[:a.probes]
            gi, pi = gi[sel], pi[sel]
            tstar = grid.query_time(fields, gi, pool[pi])
            tp = targets(teacher, pool, pool[pi], goals[gi], radius, a.candidates, gen, check=check)
            ok = torch.isfinite(tp['target']) & torch.isfinite(tstar)
            with torch.no_grad():
                pr = torch.cat([pool[pi], goals[gi]], 1)
                v_level = net.out(pr)[0][:, 0]
                v_base = base.out(pr)[0][:, 0]
            r = lambda v: dict(median=float((v[ok] / tstar[ok]).median()), mean_abs_log=float((v[ok].clamp(min=1e-6) / tstar[ok]).log().abs().mean()))
            st['probe'] = dict(n=int(ok.sum()), empty_lens=float((~torch.isfinite(tp['target'])).float().mean()), target_over_truth=r(tp['target']),
                               level_over_truth=r(v_level), base_over_truth=r(v_base), detour_over_gap_truth_median=float((tstar[ok] / torch.linalg.vector_norm(pool[pi] - goals[gi], dim=1)[ok]).median()))
        st['seconds'] = time.time() - tl
        res['level_stats'].append(st)
        models.append(net)
        Bs.append(B)
        states.append({k: v.detach().clone() for k, v in net.state_dict().items()})
        teacher = ChainField(models, edges[:m], a.blend)
        save_chain(a.out / 'chain' / f'level_{m:02d}.pt', a.base, edges[:m], a.blend, states, Bs, levels=m, edges_cells=edges_cells[:m])
        pr = st.get('probe', {})
        log(f'level {m}/{L} cells [{st["segment_cells"][0]:.2f}, {st["segment_cells"][1]:.2f}] kept {st["kept"]}/{st["pairs"]} lens {st["lens_median"]:.0f} '
            f'detour {st["detour_median"]:.3f}; fit held-out |log| {st["held_out_mean_abs_log"]:.4f} (init {st["init_mean_abs_log"]:.4f}); '
            + (f'probes {pr["n"]}: target/T* {pr["target_over_truth"]["median"]:.3f}, level/T* {pr["level_over_truth"]["median"]:.3f}, '
               f'base/T* {pr["base_over_truth"]["median"]:.3f} ' if pr else '') + f'({st["seconds"]:.0f}s)')
        dump(a.out / 'progress.json', dict(complete=False, level_stats=res['level_stats'], elapsed_seconds=time.time() - t0))
    save_chain(a.out / 'chain' / 'field.pt', a.base, edges[:L], a.blend, states, Bs, levels=L, edges_cells=edges_cells[:L])
    del fields, grid
    torch.cuda.empty_cache()
    chain = teacher
    res.update(EV.evaluate(maze, a.prepared, chain.cost, chain.cost_grad, a.out, log, label=f'segdp n={maze.n} s{a.seed}'))
    np.save(a.out / 'slice_goal0.npy', EV.slice_values(maze, chain.cost, goals[0]))
    res['elapsed_seconds'] = time.time() - t0
    res['complete'] = True
    dump(a.out / 'results.json', res)
    log(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
