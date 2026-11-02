"""Segment DP on a zoomed ntrl-demo base (zoom.py control): segdp.py's main with the base loaded as the zoomed field.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/maze_scale/segdp2.py --prepared <stage-1 dir> --base <zoom run checkpoint.pt> --zoom 5.648 --seed 87358 --out <dir>

Identical to segdp.py (same targets, sampling, losses, diagnostics and evaluation, imported from it) except: the base is
zoomfield.load_zoomed(--base, --zoom) (the network trained on coordinates x --zoom, used on the original coordinates
with B' = zoom B and output / zoom), and every DP level is a deep copy of the previous level (warm start), so it keeps
the zoomed input matrix B' and the output scale; --init fresh builds a fresh NN with B ~ N(0, 1) x zoom. results.json
has stage 'segdp' and config.zoom; chain/*.pt store the levels' B' and the output factor 1 / zoom.
"""
import argparse
import copy
import json
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
from segdp import dists, reach, sample_pairs, targets, fresh_net  # noqa: E402
from zoomfield import Scaled, load_zoomed  # noqa: E402
from common import Logger, dump  # noqa: E402


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
    p.add_argument('--zoom', type=float, required=True, help='coordinate zoom the base checkpoint was trained with (zoom.py)')
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
    base = load_zoomed(a.base, a.zoom, dev)
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
            net = copy.deepcopy(models[-1])                              # Scaled(NN(B' = zoom B), 1 / zoom), same weights
        else:
            raw, B0 = fresh_net(a.seed * 1000 + m, dev)
            raw.B = (B0 * a.zoom).T.to(dev)
            net = Scaled(raw, 1.0 / a.zoom)
        B = net.B.T.detach().cpu().clone()                                # the zoomed input matrix B'
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
        save_chain(a.out / 'chain' / f'level_{m:02d}.pt', a.base, edges[:m], a.blend, states, Bs, levels=m, edges_cells=edges_cells[:m], zoom=a.zoom, output_factor=1.0 / a.zoom)
        pr = st.get('probe', {})
        log(f'level {m}/{L} cells [{st["segment_cells"][0]:.2f}, {st["segment_cells"][1]:.2f}] kept {st["kept"]}/{st["pairs"]} lens {st["lens_median"]:.0f} '
            f'detour {st["detour_median"]:.3f}; fit held-out |log| {st["held_out_mean_abs_log"]:.4f} (init {st["init_mean_abs_log"]:.4f}); '
            + (f'probes {pr["n"]}: target/T* {pr["target_over_truth"]["median"]:.3f}, level/T* {pr["level_over_truth"]["median"]:.3f}, '
               f'base/T* {pr["base_over_truth"]["median"]:.3f} ' if pr else '') + f'({st["seconds"]:.0f}s)')
        dump(a.out / 'progress.json', dict(complete=False, level_stats=res['level_stats'], elapsed_seconds=time.time() - t0))
    save_chain(a.out / 'chain' / 'field.pt', a.base, edges[:L], a.blend, states, Bs, levels=L, edges_cells=edges_cells[:L], zoom=a.zoom, output_factor=1.0 / a.zoom)
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
