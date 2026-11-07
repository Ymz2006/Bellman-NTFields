"""One run of the anisotropic-eikonal study: train a field on a gap maze (atrain.py), plan the gap study's queries with
it on the maze and in the warp's coordinates, and measure how well it tells the maze's travel time.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/aniso_eikonal/pipeline.py --prepared <cross_warp gapprepare run> --seed 87358 \
        --mode zaniso --unit-cells 22.1 --out <dir>

1. maze     the gap maze of the prepared run (cross_warp/gap.py), checked against its maze.json.
2. warp     cross_warp's constructed warp (warp.constructed: x m_hi = 6 E / 22.1 over the quadrants and the band's edge
            cells, x m_lo over the band's interior, heading x m_hi), fixed; none for mode id.
3. train    atrain.train: ntrl-demo's loop, eikonal term only. Modes id / ziso / zaniso / zsplit (zaniso with the closed-form
            deficit term, aniso.split_class; with --checkpoint of a ziso run: that field plus the term, untrained control);
            for zaniso / zsplit the time unit is
            --unit-cells U cells of free travel (c = E / U; 'box' = E, ntrl-demo's own unit; 'quad' = E / m_hi, the unit
            in which the quadrants' equation is the warped maze's).
4. plan     the prepared run's queries (60 per region pair), planning seeds 0/1/2, maze_scale's MPPI on the maze (x) and
            in the warp's coordinates (z; cross_warp/zplan.py), exact scorer. The softmax sees the field in ntrl-demo's
            time unit inside the quadrants in every case (t_scale: 1 for id, the quadrant stretch for ziso, c for zaniso).
5. truth    how the field tells the maze's travel time: T(start, goal) over T* per query; the real travel time of the
            solved paths over T*; the descent direction against the true one; the speed the field implies on the maze,
            t_scale / |grad_x T|, over the maze's speed at held-out poses, by region (quadrants, band edge cells, arms,
            centre).
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
ROOT = next((q for q in HERE.parents if (q / 'baselines/ntrl-demo').is_dir()), Path(os.environ.get('NTFIELD_ROOT', '/home/jeffrey/bellman-ntfield')))
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
if not (HERE / 'warp.py').exists():                # working copy (a registered run's snapshot holds the cross_warp modules itself)
    sys.path.insert(0, str(ROOT / 'jeffrey/cross_warp'))
sys.path.insert(0, str(HERE))


def unit_c(spec, maze, m_hi):
    """c = E / U for --unit-cells: a number of cells, 'box' (U = E, c = 1) or 'quad' (c = m_hi)."""
    if spec == 'box':
        return 1.0
    if spec == 'quad':
        return float(m_hi)
    return maze.E / float(spec)


def band_z(maze, band, m_hi, m_lo, inset=1.0, trans=1.0):
    """Width on the warp of the band's low-density interval (warp.design_plusmag: the band less `inset` cells at each side, log-linear
    ramps of `trans` cells inside it, m_lo between them)."""
    ramp = trans * ((m_hi - m_lo) / math.log(m_hi / m_lo) if abs(m_hi - m_lo) > 1e-12 else m_hi)
    return (2 * ramp + (band - 2 * inset - 2 * trans) * m_lo) / maze.E


def m_lo_for(maze, band, m_hi, width):
    """The band density whose band is `width` wide on the warp (bisection; band_z grows with m_lo)."""
    lo, hi = 1e-6, m_hi
    assert band_z(maze, band, m_hi, lo) < width < band_z(maze, band, m_hi, hi), f'band width {width} out of reach'
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if band_z(maze, band, m_hi, mid) < width else (lo, mid)
    return math.sqrt(lo * hi)


def parse_groups(spec):
    """'8x1,8x6' -> [1]*8 + [6]*8 (16 factors, one per group of the distance head); '' -> None."""
    if not spec:
        return None
    out = []
    for part in spec.split(','):
        n, v = part.split('x') if 'x' in part else (1, part)
        out += [float(v)] * int(n)
    assert len(out) == 16, f'--group-scales needs 16 factors, got {len(out)}'
    return out


def summarize(ok, coll, conv, steps, tr, lr, cat, categories):
    med = lambda a, m: float(np.nanmedian(a[m])) if np.isfinite(a[m]).any() else None
    by = lambda arr: {name: [int(arr[:, cat == c].sum()), int(arr[:, cat == c].size)] for c, name in enumerate(categories) if (cat == c).any()}
    return dict(success_total=[int(ok.sum()), int(ok.size)], by_category=by(ok), arrived_wall=int((~ok & conv & coll).sum()), not_arrived=int((~ok & ~conv).sum()),
                arrived_wall_by_category=by(~ok & conv & coll), not_arrived_by_category=by(~ok & ~conv),
                steps_median=float(np.median(steps)), time_ratio_median=med(tr, ok), length_ratio_median=med(lr, ok),
                time_ratio_by_category={name: med(tr, ok & (cat == c)[None]) for c, name in enumerate(categories) if (cat == c).any()},
                time_ratio_p90=float(np.nanquantile(tr[ok], 0.9)) if ok.any() else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=['id', 'ziso', 'zaniso', 'zsplit', 'zboth', 'zres'], required=True,
                   help="zres: the head trained with the plain length (the stretched space) plus the far term trained with the inner product on the sum (aniso.residual_loss)")
    p.add_argument('--unit-cells', default='box', help="zaniso: cells of free travel per unit of the network's time (a number, 'box' or 'quad')")
    p.add_argument('--m-lo', type=float, default=None, help="the warp's density over the band's interior (default: that of the warp the pairs were drawn on, else 1)")
    p.add_argument('--band-z', type=float, default=None, help="choose the band density so that the band's interior is this wide on the warp (the same warped maze at every gap)")
    p.add_argument('--swap-warp', action='store_true', help="--checkpoint from another gap: evaluate it with this maze's warp (and deficit term) in place of its own")
    p.add_argument('--messy', type=float, default=0.0, help='generality test: follow the constructed warp by a smooth non-separable displacement whose Jacobian is at most this in norm (aniso.MessyWarp)')
    p.add_argument('--messy-seed', type=int, default=0)
    p.add_argument('--messy-modes', type=int, default=8)
    p.add_argument('--head-scale', type=float, default=1.0, help="output range: the distance head's scalar 0.2 becomes 0.2 x this")
    p.add_argument('--emb-scale', type=float, default=1.0, help='output range: the normalized embeddings are multiplied by this before the distance head')
    p.add_argument('--extra-data', type=Path, default=None, help='a second prepared run of the same maze whose training pairs are appended (e.g. pairs drawn on the maze itself)')
    p.add_argument('--group-scales', default='', help="output range: a factor per group of the distance head, e.g. '8x1,8x6' (multi-scale head)")
    p.add_argument('--far', default='', help="output range: add a learned smooth metric without an upper limit to the head (aniso.FarMetric); 'auto' or its scale")
    p.add_argument('--sum-groups', type=int, default=0, help='the last N groups of the distance head are added up instead of entering the soft maximum (aniso.set_scales)')
    p.add_argument('--sum-scale', type=float, default=1.0)
    p.add_argument('--head-from', type=Path, default=None, help='zres: take the head from this trained stretched-space field and freeze it; only the correction is trained')
    p.add_argument('--far-width', type=int, default=128)
    p.add_argument('--far-dim', type=int, default=8)
    p.add_argument('--pair-weight', choices=['tau', 'z', 'tauz', 'none'], default='tau', help="pair weight exp(-tau_weight .): on the field's output (ntrl-demo), on the distance between the inputs, or none")
    p.add_argument('--tag', default='', help='free label recorded with the run (variant_tag), e.g. the origin of a --checkpoint')
    p.add_argument('--tau-weight', type=float, default=0.5, help='pair weight exp(-tau_weight T) (ntrl-demo: 0.5)')
    p.add_argument('--lb-weight', type=float, default=0.0, help='straight-line lower bound H(a, b) >= c |x_a - x_b| on shuffled batch pairs (relative hinge)')
    p.add_argument('--epochs', type=int, default=4000)
    p.add_argument('--planners', default='x,z')
    p.add_argument('--plan-seeds', default='0,1,2')
    p.add_argument('--checkpoint', type=Path, default=None, help='skip training: evaluate this checkpoint (same mode / unit)')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from common import Logger, dump
    log = Logger(a.out / 'log.txt')
    info = json.load(open(a.prepared / 'maze.json'))
    assert info.get('variant') == 'gap', '--prepared must be a cross_warp gapprepare run'
    import gap as GP
    k, base = info['gap_scale'], GP.base_name(info)
    GP.install(k, base)
    import maze as MZ
    import plan as P
    import warp as W
    import warpplan as WP
    import zplan as ZP
    import aniso as AN
    import atrain as AT
    dev = 'cuda:0'
    t0 = time.time()
    maze = MZ.Maze(info['n'], dev)
    assert np.array_equal(maze.V.astype(int), np.array(info['V'])) and np.array_equal(maze.H.astype(int), np.array(info['H'])), 'prepared maze differs'
    pres = json.load(open(a.prepared / 'results.json'))
    drawn = (pres.get('data') or {}).get('warp') or {}
    if a.extra_data is not None:
        i2 = json.load(open(a.extra_data / 'maze.json'))
        assert i2['n'] == info['n'] and i2['V'] == info['V'] and i2['H'] == info['H'], '--extra-data is another maze'
    band = GP.BASE_WIDTH * k
    m_hi = 6 * maze.E / 22.1
    m_lo = a.m_lo if a.m_lo is not None else float(drawn.get('m_lo', 1.0))
    if a.band_z is not None:
        assert a.m_lo is None, '--band-z or --m-lo'
        m_lo = m_lo_for(maze, band, m_hi, a.band_z)
    if a.mode == 'id':
        warp = W.Warp(dict(s=1.0, s_p=1.0, xy=False, heading=False, nodes=maze.n, learn=False, sep=False)).to(dev)
        c = 1.0
    else:
        warp = W.constructed(maze, band, m_hi, m_lo).to(dev)
        if a.messy > 0:
            assert a.mode in ('ziso', 'zaniso'), '--messy: ziso or zaniso'
            warp = AN.MessyWarp(warp, a.messy, a.messy_modes, a.messy_seed).to(dev)
        c = unit_c(a.unit_cells, maze, m_hi) if a.mode in ('zaniso', 'zsplit', 'zboth', 'zres') else 1.0
        assert a.mode != 'zres' or (a.unit_cells == 'quad' and a.far), "zres: --unit-cells quad (the warp's largest stretch, where both equations agree) and --far"
    epochs = 3 if a.smoke else a.epochs
    res = dict(stage='aniso-eikonal', maze=maze.describe(), gap_scale=k, gap_base=base, seed=a.seed, epochs=epochs, mode=a.mode, prepared=str(a.prepared),
               gpu=torch.cuda.get_device_name(0), smoke=a.smoke, warp=dict(kind='none' if a.mode == 'id' else 'constructed (plusmag)', m_hi=m_hi, m_lo=m_lo, config=warp.cfg),
               pairs_drawn_on=dict(drawn, drawn_on=(pres.get('data') or {}).get('drawn_on', 'maze')), unit_cells=a.unit_cells, c=c, tau_weight=a.tau_weight, lb_weight=a.lb_weight, head_scale=a.head_scale, emb_scale=a.emb_scale, pair_weight=a.pair_weight, group_scales=a.group_scales, head_from=str(a.head_from) if a.head_from else None, far=a.far or None, sum_groups=a.sum_groups, sum_scale=a.sum_scale, extra_data=str(a.extra_data) if a.extra_data else None,
               open_pairs=int(((pres.get('data') or {}).get('open') or {}).get('n', 0)), messy=dict(amp=a.messy, seed=a.messy_seed, modes=a.messy_modes) if a.messy > 0 else None,
               loss=('eikonal term only (no TD, no normal term)' if a.head_from is None else 'head: as its checkpoint was trained (frozen); correction: inner-product eikonal term') if a.checkpoint is None else 'as the checkpoint was trained', variant_tag=a.tag, band_z_request=a.band_z, band_z=None if a.mode == 'id' else band_z(maze, band, m_hi, m_lo), swap_warp=a.swap_warp)
    if a.mode != 'id':
        res['metric_full_check'] = AN.check_metric_full(warp, maze)
        log(f'M = J J^T (autograd, general form) against central differences of the warp: {res["metric_full_check"]}')
        assert res['metric_full_check']['max_rel'] < 1e-3 and res['metric_full_check']['det_min'] > 0      # (central differences are rough where the warp's density has a kink)
        if a.messy == 0:
            res['metric_check'] = AN.check_metric(warp, maze)
            assert res['metric_check']['max_rel_diag'] < 1e-6 and res['metric_check']['max_rel_offdiag'] < 1e-9
    log(f'gap maze k={k} base {base} n={maze.n}; mode {a.mode}, warp m_hi {m_hi:.4f} m_lo {m_lo:.5f}, c {c:.4f} (unit {a.unit_cells} cells), pairs drawn on {res["pairs_drawn_on"]}')
    if a.checkpoint is None:
        ck, hist = AT.train(a.prepared / 'data', a.out / 'training', a.seed, epochs, warp, a.mode, c, log, tau_weight=a.tau_weight, split_m=m_hi, lb_weight=a.lb_weight,
                            head_scale=a.head_scale, emb_scale=a.emb_scale, pair_weight=a.pair_weight, group_scales=parse_groups(a.group_scales),
                            extra_data=(a.extra_data / 'data') if a.extra_data is not None else None,
                            far=dict(scale='auto' if a.far == 'auto' else float(a.far), width=a.far_width, dim=a.far_dim) if a.far else None, sum_groups=a.sum_groups, sum_scale=a.sum_scale, head_from=a.head_from)
        res.update(train_seconds=hist['seconds'], final_loss=hist['final_loss'], rollbacks=int(sum(hist['rollbacks'])), train_info=hist['info'], stopped_at_epoch=hist['stopped_at_epoch'])
        log(f'trained: {hist["seconds"] / 60:.1f} min, final loss {hist["final_loss"]:.4e}, rollbacks {res["rollbacks"]}')
    else:
        ck = a.checkpoint
        res['checkpoint_from'] = str(ck)
    if a.swap_warp:
        # a field trained on another gap's warped maze, read on this maze: its network on this maze's warp coordinates
        assert a.checkpoint is not None and a.mode != 'id'
        net = AN.load_field(ck, dev, split=False)
        res['swap_from'] = dict(warp_config=dict(net.warp.cfg), trained=torch.load(ck, map_location='cpu', weights_only=False).get('aniso'))
        res['swap_from_gap'] = int(round((int(net.warp.cfg['nodes']) - 2 * GP.BLOCK) / GP.BASE_WIDTH))
        warp.cfg['z_space'] = True
        net.warp = warp
        if a.mode == 'zsplit':
            net.set_split(c, m_hi)
    else:
        net = AN.load_field(ck, dev, split=(c, m_hi) if (a.mode == 'zsplit' and a.checkpoint is not None) else None)
    m = WP.quadrant_stretch(net.warp, maze, block=GP.BLOCK) if a.mode != 'id' else 1.0
    t_scale = {'id': 1.0, 'ziso': m, 'zaniso': c, 'zsplit': c, 'zboth': c, 'zres': c}[a.mode]
    res['split'] = dict(coeff=net.split, m_ref=net.m_ref) if net.split else None
    res.update(quadrant_stretch=m, t_scale=t_scale)

    # 4. plan
    q = np.load(a.prepared / 'queries.npz')
    S, G = torch.tensor(q['start'], device=dev), torch.tensor(q['goal'], device=dev)
    cat, clen, t_star, path_cells = q['category'], q['path_clen'], q['t_star'], q['path_cells']
    truth_paths = np.load(a.prepared / 'truth_paths.npy', allow_pickle=True)
    true_dir = torch.tensor(q['true_dir'], device=dev)
    if a.smoke:
        sel = np.concatenate([np.nonzero(cat == c_)[0][:3] for c_ in range(len(GP.CATEGORIES))])
        S, G, cat, clen, t_star, path_cells, truth_paths, true_dir = S[sel], G[sel], cat[sel], clen[sel], t_star[sel], path_cells[sel], truth_paths[sel], true_dir[sel]
    seeds = [int(s) for s in a.plan_seeds.split(',')][: 1 if a.smoke else None]
    st = P.settings(maze)
    planners = [x for x in a.planners.split(',') if x and not (x == 'z' and a.mode == 'id')]
    Lz = WP.path_length_z(net.warp, maze, list(truth_paths)) if 'z' in planners else None
    if a.mode == 'zres' and 'z' in planners:
        planners.append('zh')                 # the head alone on the warp: the stretched-space field inside this run (the in-run reference)
    res['plans'] = {}
    arrays = dict(category=cat)
    for planner in planners:
        ok = np.zeros((len(seeds), len(S)), bool)
        coll, conv_all = np.zeros_like(ok), np.zeros_like(ok)
        tr, lr = np.full(ok.shape, np.nan), np.full(ok.shape, np.nan)
        steps = []
        tpl = time.time()
        keep_far = net.far
        if planner == 'zh':
            net.far = None
        for i, ps in enumerate(seeds):
            if planner == 'x':
                paths, stop, conv, used = ZP.plan_x(net, maze, S, G, ps, clen, t_scale, st)
            else:
                paths, stop, conv, used, _ = ZP.plan_z(net, maze, S, G, ps, Lz, m if planner == 'zh' else t_scale, st)
            rows = P.score(maze, paths, conv)
            ok[i] = [r['success'] for r in rows]
            coll[i] = [r['collision'] for r in rows]
            conv_all[i] = [r['converged'] for r in rows]
            tr[i] = [r['time'] / max(float(t_star[j]), 1e-9) for j, r in enumerate(rows)]
            lr[i] = [r['length_cells'] / max(float(path_cells[j]), 1e-9) for j, r in enumerate(rows)]
            steps.append(float(stop.float().median()))
            if i == 0 and not a.smoke:
                arr = np.empty(len(paths), dtype=object)
                arr[:] = [pp.detach().cpu().numpy().astype(np.float32) for pp in paths]
                np.save(a.out / f'paths_{planner}_seed{ps}.npy', arr, allow_pickle=True)
        net.far = keep_far
        e = summarize(ok, coll, conv_all, steps, tr, lr, cat, GP.CATEGORIES)
        e.update(settings={kk: (float(v) if isinstance(v, (int, float)) else v) for kk, v in used.items()}, seconds=time.time() - tpl)
        res['plans'][planner] = e
        arrays.update({f'{planner}_success': ok, f'{planner}_collision': coll, f'{planner}_converged': conv_all, f'{planner}_time_ratio': tr, f'{planner}_length_ratio': lr})
        log(f'MPPI in {planner}: {ok.sum()}/{ok.size} by category {e["by_category"]} (arrived + wall {e["arrived_wall"]}, never arrived {e["not_arrived"]}, '
            f'path time / T* median {e["time_ratio_median"]}, {e["seconds"]:.0f}s)')
        dump(a.out / 'results.json', res)

    # 5. truth: the field's travel time against the maze's
    pairs = torch.cat([S, G], 1)
    T = P.field_cost(net)(pairs) / t_scale
    ratio = (T.cpu().numpy() / np.maximum(t_star, 1e-9))
    arrays['T_pred'] = T.cpu().numpy()
    arrays['t_star'] = t_star
    tt, g = P.field_grad(net, pairs)
    d = torch.nn.functional.normalize(-g, dim=1)
    ang = torch.rad2deg(torch.acos((d * true_dir).sum(1).clamp(-1, 1))).cpu().numpy()
    arrays['direction_error_deg'] = ang
    names = GP.CATEGORIES
    res['time_accuracy'] = dict(ratio_median=float(np.median(ratio)), abs_log_ratio_median=float(np.median(np.abs(np.log(np.maximum(ratio, 1e-9))))),
                                within_10pct=float((np.abs(ratio - 1) < 0.1).mean()), within_25pct=float((np.abs(ratio - 1) < 0.25).mean()),
                                by_category={names[c_]: dict(ratio_median=float(np.median(ratio[cat == c_])), ratio_p10=float(np.quantile(ratio[cat == c_], 0.1)),
                                                             ratio_p90=float(np.quantile(ratio[cat == c_], 0.9)), within_10pct=float((np.abs(ratio[cat == c_] - 1) < 0.1).mean()))
                                             for c_ in range(len(names)) if (cat == c_).any()},
                                note='T(start, goal) / t_scale over the true travel time T* (both in ntrl-demo\'s normalized unit of the maze)')
    res['direction'] = dict(median_deg=float(np.median(ang)), within_30deg=float((ang < 30).mean()),
                            by_category={names[c_]: float(np.median(ang[cat == c_])) for c_ in range(len(names)) if (cat == c_).any()},
                            note='angle between -grad_x T(start, goal) and the true descent direction at the start')
    log(f'T / T*: {res["time_accuracy"]}')
    log(f'direction: {res["direction"]}')
    hx = np.load(a.prepared / 'heldout/sampled_points.npy')
    hd = np.load(a.prepared / 'heldout/clearance_cells.npy')
    if a.smoke:
        hx, hd = hx[:2000], hd[:2000]
    hp = torch.tensor(hx, device=dev)
    ev = torch.cat([hp, torch.cat([hp[:, 3:], hp[:, :3]], 1)])            # evaluate at x0 (goal x1) and at x1 (goal x0)
    dcl = torch.tensor(np.concatenate([hd[:, 0], hd[:, 1]]), device=dev)
    keep = dcl > 0.02
    ev, dcl = ev[keep], dcl[keep]
    _, gg = P.field_grad(net, ev)
    s_pred = t_scale / torch.linalg.vector_norm(gg, dim=1).clamp(min=1e-12)
    s_true = maze.train_speed(dcl)
    rr = (s_pred / s_true).cpu().numpy()
    reg = AN.region4(maze, GP.BLOCK, ev[:, :3])
    res['speed_on_maze'] = dict(note='t_scale / |grad_x T| over the maze\'s speed at held-out poses (both ends of 50,000 held-out pairs, clearance > 0.02 cells)',
                                all=dict(median=float(np.median(rr)), abs_log_median=float(np.median(np.abs(np.log(rr)))), poses=int(len(rr))),
                                by_region={AN.REGIONS[r]: dict(median=float(np.median(rr[reg == r])), p10=float(np.quantile(rr[reg == r], 0.1)), p90=float(np.quantile(rr[reg == r], 0.9)),
                                                               abs_log_median=float(np.median(np.abs(np.log(rr[reg == r])))), poses=int((reg == r).sum()))
                                           for r in range(4) if (reg == r).any()})
    log(f'speed on the maze / true: {res["speed_on_maze"]}')
    # the straight-line lower bound: the travel time between two free poses is at least their distance on the maze (speed <= 1).
    # A field that ignores a compressed direction breaks it; the share of random free pairs below 0.9 x the distance measures that.
    gen = torch.Generator(dev).manual_seed(101)
    fa, fb = maze.free(2000 if a.smoke else 20000, gen, 0.1), maze.free(2000 if a.smoke else 20000, gen, 0.1)
    dist = torch.linalg.vector_norm(fa - fb, dim=1)
    lbr = ((P.field_cost(net)(torch.cat([fa, fb], 1)) / t_scale) / dist.clamp(min=1e-9)).cpu().numpy()
    rga, rgb = AN.region4(maze, GP.BLOCK, fa), AN.region4(maze, GP.BLOCK, fb)
    band = (rga >= 2) & (rgb >= 2)
    res['lower_bound'] = dict(note='T(a, b) / |x_a - x_b| over random pairs of free poses (the truth is >= 1)', pairs=int(len(lbr)),
                              share_below_0_9=float((lbr < 0.9).mean()), share_below_0_5=float((lbr < 0.5).mean()), p10=float(np.quantile(lbr, 0.1)), median=float(np.median(lbr)),
                              band_pairs=dict(n=int(band.sum()), share_below_0_9=float((lbr[band] < 0.9).mean()) if band.any() else None,
                                              median=float(np.median(lbr[band])) if band.any() else None, p10=float(np.quantile(lbr[band], 0.1)) if band.any() else None))
    log(f'lower bound T / |dx|: {res["lower_bound"]}')
    np.savez(a.out / 'rows.npz', **arrays)
    res['elapsed_seconds'] = time.time() - t0
    res['complete'] = True
    dump(a.out / 'results.json', res)
    log(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
