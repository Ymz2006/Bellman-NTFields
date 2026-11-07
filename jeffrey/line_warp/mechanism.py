"""Measurements behind the findings' mechanisms, from registered runs (writes report/mechanism.json; wallhits.py adds what needs a field or a replan).

* slow zone (default): how wide the wall's slow zone (the speed margin, 0.175 cells) is in the warped space, inside a quadrant and on the band side of
  an outline wall, on the 22 x 22 quadrants at gap x12, with and without the dilation, against one planner step.
* stalls: where runs that never arrived stopped, on the plus maze: in a quadrant or the band, how near a wall, how far from the goal.
* zones, zones12 <c22|m10>: the slow zone in planner steps at near-wall poses, on the plus maze and at gap x12 (zones12 runs in its own process per maze).
* paths: true paths measured in z. variation, shape: how the log-density of a fit varies. ratios: largest / smallest run, one procedure for all fits.

    CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/line_warp/mechanism.py [stalls|zones|zones12 <base>|paths|variation|shape|ratios]
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / 'jeffrey/maze_scale'), str(ROOT / 'jeffrey/cross_warp')]
import figs as G  # noqa: E402

PLUS = ROOT / 'results/registered_20261102T042454Z-62864e61'
K12 = ROOT / 'results/registered_20261103T233747Z-3c48b7dc'
M12 = ROOT / 'results/registered_20261105T070402Z-98f3cab7'
FIELDS = {'separable, no TV': 'sep125_m6', 'recipe, seed 87358': 'bt10_m6_87358'}
CONTROLS = {'ntrl-demo': '20261103T163539Z-48369bc1', 'hand-built x 6': '20261103T171757Z-79906972'}


def slow_zone(B):
    """Runs in its own process per maze: gap x12."""
    import fit as FT
    import warpplan as WP
    maze, _ = FT.load_maze(K12, 'cuda:0')
    n, E = maze.n, maze.E
    m = 6 * E / 22.1
    step = 0.015 * maze.planner_scale() * m
    out = dict(step=step)
    for name, job in (('no dilation', 'c22k12t10'), ('dilation', 'c22k12dil')):
        w = G.load_warp(ROOT / 'results' / f'registered_{B[job]}', maze)
        w.s = 1.0
        w.s = m / WP.quadrant_stretch(w, maze, block=8)

        def length(a, b, y=3.5):
            t = torch.tensor([[(a - n / 2) / E, (y - n / 2) / E], [(b - n / 2) / E, (y - n / 2) / E]], device='cuda:0')
            with torch.no_grad():
                z = w.plane(t)
            return float((z[1, 0] - z[0, 0]).abs())
        # heading 0: the free side of the outline wall at x = 8 starts 0.05 + 0.25 cells past the grid line
        out[name] = dict(quadrant=length(1.3, 1.475), band_side=length(8.30, 8.475), s=w.s)
    return out


def zone_widths(B):
    """Width in z of the wall's slow zone (0.175 cells along the wall normal) at held-out near-wall poses of the plus maze, in planner steps, for each
    fitted warp at the quadrant stretch of its field: |J n| 0.175 / E / (step m), with n the clearance normal in (x, y)."""
    import fit as FT
    import warpplan as WP
    import netwarp as NW
    import warpnet as WN
    from maze import MARGIN_CELLS
    maze, _ = FT.load_maze(PLUS, 'cuda:0')
    x = torch.tensor(np.load(PLUS / 'heldout/sampled_points.npy')[:20000, :3], device='cuda:0')
    _, nrm = maze.clearance_normal(x)
    n = torch.nn.functional.normalize(nrm[:, :2], dim=1)
    keep = nrm[:, :2].norm(dim=1) > 0.5                     # the wall normal is mostly in the plane
    x, n = x[keep], n[keep]
    m = 6.0
    step = 0.015 * maze.planner_scale() * m
    out = dict(step=step)
    for name, job in (('free ResNet (stripes)', 'fitA'), ('ResNet, limit 12', 'rsA12'), ('ResNet, limit 4', 'rsA4'), ('separable, no TV', 'sep125'), ('separable + TV', 'bt10'), ('the recipe (+ dilation)', 'dil'), ('bound 2, no TV', 'sep200'), ('bound 2, dilation, no TV', 'ntdil'), ('bound 0.70, no TV', 'sep070')):
        w = G.load_warp(ROOT / 'results' / f'registered_{B[job]}', maze)
        w.s = 1.0
        w.s = m / WP.quadrant_stretch(w, maze, block=8)
        J = WN.jacobian_fd(w.plane, x[:, :2].contiguous(), 0.02 / maze.E)
        g = torch.einsum('nij,nj->ni', J, n).norm(dim=1)          # stretch along the wall normal
        width = g * (MARGIN_CELLS / maze.E) / step
        q = lambda a: float(width.quantile(a))
        s1, s2, _ = WN.singular_values(J)
        out[name] = dict(median=q(0.5), q10=q(0.1), below_one_step=float((width < 1).float().mean()), sv_small_median=float(s2.median()))
    return out


WIDE = {'c22': (K12, dict(both='c22k12dil', tv='c22k12t10', dil='c22k12nt', none='c22k12n0', tight='c22k12h07', range='c22k12r', b125='c22k12b125', b250='c22k12b250')),
        'm10': (M12, dict(both='m10k12d', tv='m10k12t10', dil='m10k12nt', none='m10k12n0', tight='m10k12h07', range='m10k12r', b125='m10k12b125', b250='m10k12b250'))}


def zones_wide(B, base):
    """zone_widths at gap x12, for the fits of report.GRID, one process per maze: how many near-wall held-out poses have a slow zone thinner than one planner step, inside a quadrant and
    outside one (the band side of the outline walls). Wall hits should follow this; stalls need not."""
    import fit as FT
    import warpplan as WP
    import netwarp as NW
    import warpnet as WN
    from maze import MARGIN_CELLS
    prep, cells = WIDE[base]
    maze, info = FT.load_maze(prep, 'cuda:0')
    n, E = maze.n, maze.E
    block = maze.describe().get('block', 8)
    x = torch.tensor(np.load(prep / 'heldout/sampled_points.npy')[:, :3], device='cuda:0')
    c = torch.tensor(np.load(prep / 'heldout/clearance_cells.npy'), device='cuda:0').flatten()[:len(x)]
    near = (c > 0) & (c < MARGIN_CELLS)
    x = x[near][:40000]
    _, nrm = maze.clearance_normal(x)
    nn = torch.nn.functional.normalize(nrm[:, :2], dim=1)
    keep = nrm[:, :2].norm(dim=1) > 0.5                     # the wall normal is mostly in the plane
    x, nn = x[keep], nn[keep]
    X, Y = x[:, 0] * E + n / 2, x[:, 1] * E + n / 2
    inq = lambda z: (z < block) | (z >= n - block)
    quad = inq(X) & inq(Y)
    m = 6 * E / 22.1
    step = 0.015 * maze.planner_scale() * m
    out = dict(step=step, poses=int(len(x)), in_quadrant=float(quad.float().mean()))
    for name, job in cells.items():
        if job not in B:
            continue
        w = G.load_warp(ROOT / 'results' / f'registered_{B[job]}', maze)
        w.s = 1.0
        w.s = m / WP.quadrant_stretch(w, maze, block=block)
        J = WN.jacobian_fd(w.plane, x[:, :2].contiguous(), 0.02 / E)
        width = torch.einsum('nij,nj->ni', J, nn).norm(dim=1) * (MARGIN_CELLS / E) / step
        thin = (width < 1).float()
        two = (width < 2).float()
        out[name] = dict(job=job, s=w.s, median=float(width.median()), q10=float(width.quantile(0.1)), below_one_step=float(thin.mean()), below_in_quadrant=float(thin[quad].mean()), below_outside=float(thin[~quad].mean()) if (~quad).any() else None,
                         below_two=float(two.mean()), two_in_quadrant=float(two[quad].mean()), two_outside=float(two[~quad].mean()))
    return out


def path_lengths(B):
    """How long the certified true paths of the region query set are in z, against the straight z-distance between their ends and against the same paths at a
    uniform stretch m: a warp that stretches corridors along their length makes the field's travel times long."""
    import fit as FT
    import warpplan as WP
    maze, _ = FT.load_maze(PLUS, 'cuda:0')
    Q = ROOT / 'results/registered_20261103T233014Z-ae84b379'
    tp = list(np.load(Q / 'truth_paths.npy', allow_pickle=True))[::3]
    m = 6.0
    base = np.array([np.linalg.norm(np.diff(np.asarray(p)[:, :2], axis=0), axis=1).sum() for p in tp]) * m
    out = {}
    for name, job in (('free ResNet (stripes)', 'fitA'), ('ResNet, limit 4', 'rsA4'), ('the recipe (+ dilation)', 'dil')):
        w = G.load_warp(ROOT / 'results' / f'registered_{B[job]}', maze)
        w.s = 1.0
        w.s = m / WP.quadrant_stretch(w, maze, block=8)
        w.s_p = m
        L = WP.path_length_z(w, maze, tp)
        ends = np.array([float(torch.linalg.vector_norm(w(torch.tensor(np.asarray(p)[[0, -1]], device='cuda:0', dtype=torch.float32))[1, :2] - w(torch.tensor(np.asarray(p)[[0, -1]], device='cuda:0', dtype=torch.float32))[0, :2])) for p in tp])
        t = torch.linspace(-0.5, 0.5, 2001, device='cuda:0')
        edge = torch.cat([torch.stack([t, torch.full_like(t, v)], 1) for v in (-0.5, 0.5)] + [torch.stack([torch.full_like(t, v), t], 1) for v in (-0.5, 0.5)])
        with torch.no_grad():
            z = w.plane(edge)
        box = float((z.amax(0) - z.amin(0)).max())
        out[name] = dict(length_median=float(np.median(L)), length_q90=float(np.quantile(L, 0.9)), over_uniform_median=float(np.median(L / base)), detour_median=float(np.median(L / np.maximum(ends, 1e-6))), box=box, length_q90_over_box=float(np.quantile(L, 0.9) / box))
    return out


def stalls(B):
    import fit as FT
    maze, _ = FT.load_maze(PLUS, 'cuda:0')
    import gap
    gap.use_base('cross22')
    out = {}
    runs = {k: f'registered_{B[j]}' for k, j in FIELDS.items() if j in B}
    runs.update({k: f'registered_{v}' for k, v in CONTROLS.items()})
    for name, rid in runs.items():
        r = ROOT / 'results' / rid
        rows = np.load(r / 'rows.npz')
        conv = rows['converged'][0]
        paths = np.load(r / 'paths_seed0.npy', allow_pickle=True)
        last = np.array([np.asarray(p)[-2] for i, p in enumerate(paths) if not conv[i]])
        goal = np.array([np.asarray(p)[-1] for i, p in enumerate(paths) if not conv[i]])
        c = last[:, :2] * maze.E + maze.n / 2
        d = maze.clearance(torch.tensor(last, device='cuda:0', dtype=torch.float32)).cpu().numpy()
        reg = gap.region_of(22, c[:, 0], c[:, 1])
        out[name] = dict(not_arrived=int(len(last)), of=int(len(conv)), in_quadrant=float((reg < 4).mean()), near_wall=float((d < 0.2).mean()), goal_distance_median=float(np.median(np.linalg.norm((last - goal)[:, :2], axis=1) * maze.E)))
    return out


VARIATION = [('sep070', 'sep070_m6'), ('sep100', 'sep100_m6'), ('sep125', 'sep125_m6'), ('sep160', 'sep160_m6'), ('sep200', 'sep200_m6'), ('sepL125', 'sepL125_m6'), ('sepL200', 'sepL200_m6'),
             ('tv02', 'tv02_m6'), ('tv05', 'tv05_m6'), ('tv10', 'tv10_m6'), ('tv20', 'tv20_m6'), ('tv40', 'tv40_m6'), ('bt10', 'bt10_m6_87358'), ('dil', 'dil_m6_87358'), ('ntdil', 'ntdil_m6_87358')]


def variation(B):
    """How much the log-density of each separable fit on the plus maze varies, next to what its field did (training seed 87358): the total rise and fall of log density
    along an axis, its range, and how much of the variation is finer than a cell."""
    out = {}
    for fit, field in VARIATION:
        if fit not in B or field not in B:
            continue
        r = ROOT / 'results' / f'registered_{B[fit]}'
        H = torch.load(r / 'phi.pt', map_location='cpu', weights_only=False)['state']['Hfrozen'][:, 0, :].numpy()     # (axes, nodes + 1): separable, so every row is the same
        per_cell = (H.shape[1] - 1) / 22.1
        w = int(round(per_cell))
        box = np.ones(2 * w + 1) / (2 * w + 1)
        fine = [h - np.convolve(np.pad(h, w, mode='edge'), box, mode='valid') for h in H]
        rise = np.abs(H[:, w:] - H[:, :-w])                    # change of log density over one cell
        edges = [int(((r[1:] >= 0.5) & (r[:-1] < 0.5)).sum()) for r in rise]
        d = json.load(open(ROOT / 'results' / f'registered_{B[field]}' / 'results.json'))
        a = json.load(open(r / 'results.json'))['warp']['area_stretch']
        t = d['success_total']
        out[fit] = dict(tv=float(np.abs(np.diff(H, axis=1)).sum(1).mean()), range=float((H.max(1) - H.min(1)).mean()), fine=float(np.mean([f.std() for f in fine])), edges=float(np.mean(edges)), band_over_quadrant=a['arms'] / a['quadrants'],
                        pct=100 * t[0] / t[1], bumps=d['bumps']['bumps_per_path'], tilt=quadrant_tilt(H, 22.1, 22, 8))
    return out


def ratios(B):
    """Largest / smallest run of plus-maze fits measured with one procedure (checks.run_stats as it stands; the first fits were measured before the crossing test was fixed)."""
    import fit as FT
    import checks as CK
    maze, _ = FT.load_maze(PLUS, 'cuda:0')
    stat = lambda r: dict(ratio=r['ratio_median'], log_std=r['log_std'])
    out = {'identity': stat(CK.run_stats(lambda x: x, maze, 'cspace', 9))}
    for job in ('fitA', 'fitA2', 'fitB', 'shear30', 'gmix', 'triFree', 'rsA12', 'rsA4', 'resTV', 'sep125', 'sep070', 'sepL125', 'tv05', 'bt10', 'dil', 'ntdil', 'inset1', 'triTV', 'tv40'):
        if job in B:
            w = G.load_warp(ROOT / 'results' / f'registered_{B[job]}', maze)
            with torch.no_grad():
                out[job] = stat(CK.run_stats(lambda x: w.plane(x), maze, 'cspace', 9))
    return out


def quadrant_tilt(H, E, n, block):
    """How far the log-density moves inside a quadrant: largest minus smallest one-cell mean over the block cells of each quadrant side, averaged over the sides and axes."""
    c = np.linspace(-0.5, 0.5, H.shape[1]) * E + n / 2
    out = []
    for h in H:
        for side in (range(0, block), range(n - block, n)):
            m = [h[(c >= k) & (c < k + 1)].mean() for k in side]
            out.append(max(m) - min(m))
    return float(np.mean(out))


SHAPES = ['c22k12t10', 'c22k12dil', 'c22k12nt', 'c22k12n0', 'c22k12h07', 'c22k12b125', 'c22k12r', 'c22k12b250', 'm10k12t10', 'm10k12d', 'm10k12nt', 'm10k12n0', 'm10k12h07', 'm10k12b125', 'm10k12r', 'm10k12b250', 'c22k1d', 'c22k2d', 'c22k4d', 'c22k8d', 'c22k1nt',
          'm10k1d', 'm10k2d', 'm10k4d', 'm10k8d', 'dil', 'ntdil', 'sep200', 'sep070']


def density_shape(B):
    """The shape of the log-density of separable fits: its range, total rise and fall, and how many steps of at least 0.5 within one cell it has per axis."""
    out = {}
    for job in SHAPES:
        if job not in B:
            continue
        r = ROOT / 'results' / f'registered_{B[job]}'
        H = torch.load(r / 'phi.pt', map_location='cpu', weights_only=False)['state']['Hfrozen'][:, 0, :].numpy()
        d = json.load(open(r / 'results.json'))
        w = max(1, int(round((H.shape[1] - 1) / d['maze']['E_cells'])))
        rise = np.abs(H[:, w:] - H[:, :-w])
        a = d['warp']['area_stretch']
        out[job] = dict(range=float((H.max(1) - H.min(1)).mean()), tv=float(np.abs(np.diff(H, axis=1)).sum(1).mean()), edges=float(np.mean([((x[1:] >= 0.5) & (x[:-1] < 0.5)).sum() for x in rise])),
                        band_over_quadrant=a['arms'] / a['quadrants'], sv=d['warp']['sv_min_max'], ratio=d['warp']['runs']['ratio_median'], tilt=quadrant_tilt(H, d['maze']['E_cells'], d['maze']['n'], d['maze'].get('block', 8)))
    return out


if __name__ == '__main__':
    B = json.load(open(HERE / 'report/runs.json'))
    path = HERE / 'report/mechanism.json'
    res = json.load(open(path)) if path.exists() else {}
    if len(sys.argv) > 1 and sys.argv[1] == 'stalls':
        res['stalls'] = stalls(B)
    elif len(sys.argv) > 1 and sys.argv[1] == 'paths':
        res['paths'] = path_lengths(B)
        print(json.dumps(res['paths'], indent=0))
    elif len(sys.argv) > 1 and sys.argv[1] == 'zones':
        res['zones'] = zone_widths(B)
    elif len(sys.argv) > 2 and sys.argv[1] == 'zones12':
        res.setdefault('zones12', {})[sys.argv[2]] = zones_wide(B, sys.argv[2])
        print(json.dumps(res['zones12'][sys.argv[2]], indent=0))
    elif len(sys.argv) > 1 and sys.argv[1] == 'ratios':
        res['ratios'] = ratios(B)
        print(res['ratios'])
    elif len(sys.argv) > 1 and sys.argv[1] == 'shape':
        res['shape'] = density_shape(B)
    elif len(sys.argv) > 1 and sys.argv[1] == 'variation':
        res['variation'] = variation(B)
        print(json.dumps(res['variation'], indent=0))
    else:
        res['slow_zone'] = slow_zone(B)
    json.dump(res, open(path, 'w'), indent=1)
    print(json.dumps(res.get('zones', res), indent=0)[:1400])
