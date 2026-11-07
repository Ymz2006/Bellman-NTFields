"""Report (from registered runs): training pairs drawn on the warped maze (gapprepare.py --z-draw) and the 10 x 10 base
with the same gaps (gap.py base 'maze10'). Writes report/zdraw10.json, report/zdraw10.md and report/gap_mazes10.png.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/cross_warp/zdraw10.py

Fields (run labels): id = ntrl-demo (trainer.py with the identity warp), pm = the constructed warp trained on the maze
(the gap study's), pmz = the constructed warp trained on the warped maze (trainer.py z_space) on pairs drawn on the real
maze, pmzw = the same on pairs drawn on the warped maze; controls on the warped-maze pairs: idw = ntrl-demo, pmw = the constructed warp
trained on the maze (the gap study's loss); the shrunk-band control (the warp's band interior shrunk to its x1 width, m_lo < 1)
and its 2 x 2: pmzs = the shrunk warp with pairs drawn on it, pmzp = pairs drawn on the shrunk warp trained on the unshrunk
warped maze, pmzg = pairs drawn on the unshrunk warp trained on the shrunk warped maze. Planners: x = MPPI on the maze (pipeline.py's evaluation, the
gap study's planner, cost / t_scale for z-trained fields), z = MPPI on the warped maze (zq.py --planners z). The
22 x 22 study's own fields (ntrl-demo, constructed, pmz) come from report/fieldspaces/summary.json (fieldspaces.py).
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gapreport as GR  # noqa: E402

OUT = HERE / 'report'
CATS = GR.CATS
BLOCK = {'c22': 8, 'm10': 5}
S0 = 87358


def base_of(d):
    return ((d.get('maze') or {}).get('base') or {}).get('name', 'cross22')


def zdrawn(d):
    return str((d.get('data') or {}).get('drawn_on', '')).startswith('warped')


def m_lo(d):
    """The band-interior density of the warp a z-drawn prepare drew on (1: the gap study's constructed warp; < 1: shrunk)."""
    return float(((d.get('data') or {}).get('warp') or {}).get('m_lo', 1.0))


def regions(points, n, block, chunk=1 << 20):
    """Shares of training pairs (N, 6) with x0 in a quadrant, x1 in a quadrant, both ends in quadrants (any), both in the
    same quadrant."""
    E = n + 0.1
    G = n - 2 * block
    tot = np.zeros(8)
    N = len(points)
    for i in range(0, N, chunk):
        p = np.asarray(points[i:i + chunk], dtype=np.float64)

        def reg(x, y):
            cx, cy = np.floor(x * E + n / 2), np.floor(y * E + n / 2)
            q = ((cx < block) | (cx >= block + G)) & ((cy < block) | (cy >= block + G))
            r = np.where(q, (cx >= block + G).astype(int) + 2 * (cy >= block + G).astype(int), 4)
            return r
        r0, r1 = reg(p[:, 0], p[:, 1]), reg(p[:, 3], p[:, 4])
        q0, q1 = r0 < 4, r1 < 4
        diag = q0 & q1 & (r0 + r1 == 3) & (r0 != r1)
        tot += [q0.sum(), q1.sum(), (q0 & q1).sum(), (q0 & (r0 == r1)).sum(), (q0 & q1 & (r0 != r1) & ~diag).sum(), diag.sum(), (q0 ^ q1).sum(), (~q0 & ~q1).sum()]
    return dict(x0_quadrant=tot[0] / N, x1_quadrant=tot[1] / N, both_quadrants=tot[2] / N, same_quadrant=tot[3] / N,
                kinds={c: tot[4 + i - 1] / N if i else tot[3] / N for i, c in enumerate(CATS)})


def band_z(base, k, m_lo=1.0, inset=1.0, trans=1.0):
    """Width in the warp (units of the network's input) of the band's interior, design_plusmag's low-density interval (the
    band less its edge cells, log-linear ramps of trans cells), for the constructed warp at gap k (m_hi = 6 E / 22.1)."""
    n = (16 if base == 'c22' else 10) + 6 * k
    E = n + 0.1
    m_hi = 6 * E / 22.1
    ramp = trans * ((m_hi - m_lo) / math.log(m_hi / m_lo) if abs(m_hi - m_lo) > 1e-12 else m_hi)
    return (2 * ramp + (6 * k - 2 * inset - 2 * trans) * m_lo) / E


def prepare_warp(R, key, k):
    """The warp record (m_hi, m_lo, band, ...) of a z-drawn prepare in R['prepares'][key]."""
    name = (R['prepares'].get(key) or {}).get(str(k))
    if not name:
        return None
    return (json.loads((GR.ROOT / 'results' / name / 'results.json').read_text()).get('data') or {}).get('warp')


def arrival(rows_file, planner):
    """Failures split by arrival from a run's rows.npz: arrived_wall = arrived but touched a wall on the way, not_arrived =
    never arrived within the step budget. (plan.mppi appends the goal to every path, so the scorer's collision flag of a
    run that never arrived includes the jump from where it stopped to the goal: 'collision' / 'stalled' mix the two.)"""
    f = Path(rows_file)
    if not f.exists():
        return {}
    z = np.load(f)
    key = lambda name: f'{planner}_{name}' if planner else name          # planner None: a training run's own rows (MPPI on the maze)
    if key('converged') not in z.files:
        return {}
    ok, co, cv = z[key('success')], z[key('collision')], z[key('converged')]
    return dict(arrived_wall=int((~ok & cv & co).sum()), not_arrived=int((~ok & ~cv).sum()))


def entry_x(d):
    bc = d.get('by_category') or {}
    return dict(success_total=list(d['success_total']), by_category={c: [v['success'], v['runs']] for c, v in bc.items()})


def collect():
    runs = GR.load_all()
    _, preps22 = GR.collect()
    main22 = {k: preps22[k]['_dir'].name for k in preps22}
    main10, z22, z10, s22, s10 = {}, {}, {}, {}, {}
    for rid, d in sorted(runs.items()):
        if d.get('stage') == 'gapprepare' and base_of(d) == 'maze10' and not d.get('queries_from'):
            main10[d['scale']] = d['_dir'].name                     # the latest one per k
    for rid, d in sorted(runs.items()):
        if d.get('stage') != 'gapprepare' or not d.get('queries_from') or not zdrawn(d):
            continue
        k, src = d['scale'], Path(d['queries_from']).name
        shr = m_lo(d) < 1.0
        if base_of(d) == 'cross22' and main22.get(k) == src:
            (s22 if shr else z22)[k] = d['_dir'].name
        if base_of(d) == 'maze10' and main10.get(k) == src:
            (s10 if shr else z10)[k] = d['_dir'].name
    role = {}
    for k, v in main10.items():
        role[v] = ('m10', k, 'real')
    for k, v in z10.items():
        role[v] = ('m10', k, 'warped')
    for k, v in z22.items():
        role[v] = ('c22', k, 'warped')
    for k, v in s22.items():
        role[v] = ('c22', k, 'shrunk')
    for k, v in s10.items():
        role[v] = ('m10', k, 'shrunk')
    fields = defaultdict(lambda: defaultdict(dict))               # 'base/k' -> label -> seed -> entry
    index = {}
    for rid, d in sorted(runs.items()):
        if d.get('stage') != 'cross-warp':
            continue
        r = role.get(Path(d['prepared']).name)
        if not r:
            continue
        base, k, draw = r
        lab = GR.label_of(rid)
        want = {('m10', 'real'): ('id', 'pm', 'pmz'), ('m10', 'warped'): ('pmzw', 'pmw', 'idw', 'pmzg'), ('c22', 'warped'): ('pmzw', 'pmw', 'idw', 'pmzg'),
                ('c22', 'shrunk'): ('pmzs', 'pmzp'), ('m10', 'shrunk'): ('pmzs', 'pmzp')}[(base, draw)]
        if lab not in want:
            continue
        e = dict(_id=rid, seed=d['seed'], x=dict(entry_x(d), **arrival(d['_dir'] / 'rows.npz', None)), final_loss=d.get('final_loss'), rollbacks=d.get('rollbacks'), train_seconds=d.get('train_seconds'),
                 t_scale=(d.get('z_space') or {}).get('t_scale'), failures=d.get('failures'))
        fields[f'{base}/{k}'][lab][d['seed']] = e
        index[rid] = e
    planwarp = {}                                                  # field run id -> MPPI in the constructed warp's coordinates (zq --plan-warp constructed)
    for rid, d in sorted(runs.items()):                            # zq: MPPI on the warped maze (later runs win)
        if d.get('stage') != 'zq' or d.get('smoke'):
            continue
        for frid, z in d.get('runs', {}).items():
            if 'z' not in z.get('plans', {}):
                continue
            p = z['plans']['z']
            e = dict(success_total=p['success_total'], by_category=p['by_category'], zq=rid, time_ratio_median=p.get('time_ratio_median'),
                     collision=p.get('collision'), stalled=p.get('stalled'), **arrival(d['_dir'] / frid / 'rows.npz', 'z'))
            if z.get('plan_warp') == 'constructed':
                planwarp[frid] = dict(e, quadrant_stretch=z.get('quadrant_stretch'))
            elif frid in index:
                index[frid]['z'] = e
                index[frid]['inverse_roundtrip_max'] = z.get('inverse_roundtrip_max')
    # pair distributions: drawn on the real maze (main prepares; k = 1 of the 22 x 22 study: the maze_scale prepare) and on the warped maze
    pairs = {}
    for base, mains, zs, ss in (('c22', main22, z22, s22), ('m10', main10, z10, s10)):
        for k in sorted(set(mains) | set(zs) | set(ss)):
            for draw, name in (('real', mains.get(k)), ('warped', zs.get(k)), ('shrunk', ss.get(k))):
                if not name:
                    continue
                if base == 'c22' and k == 1 and draw == 'real':
                    name = GR.PLUS22
                d = GR.ROOT / 'results' / name
                n = json.loads((d / 'maze.json').read_text())['n']
                reg = regions(np.load(d / 'data/sampled_points.npy', mmap_mode='r'), n, BLOCK[base])
                pairs[f'{base}/{k}/{draw}'] = dict(reg, run=name)
    controls = {}
    for k, name in main10.items():
        res = json.loads((GR.ROOT / 'results' / name / 'results.json').read_text())
        controls[str(k)] = dict(n=res['maze']['n'], oracle=res.get('oracle', {}).get('by_category'), straight=res.get('straight', {}).get('by_category'),
                                queries=res.get('queries', {}).get('per_category'), walls=res['maze'].get('walls'))
    zs = json.loads((OUT / 'fieldspaces/summary.json').read_text()) if (OUT / 'fieldspaces/summary.json').exists() else {}
    c22 = {}
    for k, g in (zs.get('gaps') or {}).items():                  # the 22 x 22 study's fields: per seed
        c22[k] = {}
        for key, lab in (('ntrl', 'id'), ('x_trained', 'pm'), ('z_trained', 'pmz'), ('z_trained_k1data', 'pmzd')):
            if key not in g:
                continue
            for planner, e in g[key].items():
                if planner not in ('x', 'z'):
                    continue
                for s, ps in (e.get('per_seed') or {}).items():
                    c22[k].setdefault(lab, {}).setdefault(s, {})[planner] = dict(success_total=ps['success_total'], by_category=ps['by_category'])
    # the 22 x 22 study's ntrl-demo fields (trained on its main prepares): run ids per gap, for their MPPI in the constructed warp's coordinates
    data22, _ = GR.collect()
    id22 = {str(k): {str(s): e['_id'] for s, e in data22[k].get('ntrl-demo', {}).items()} for k in data22}
    id22_arrival = {}                                              # their failures by arrival, MPPI on the maze (rows on the gap's 300 queries only)
    for k, by in id22.items():
        for s, rid in by.items():
            f = GR.ROOT / 'results' / f'registered_{rid}' / 'rows.npz'
            if f.exists() and np.load(f)['success'].shape[-1] == 300:
                id22_arrival.setdefault(k, {})[s] = arrival(f, None)
    return dict(fields={b: {lab: {str(s): e for s, e in v.items()} for lab, v in f.items()} for b, f in fields.items()}, pairs=pairs, controls=controls,
                c22_study=c22, planwarp=planwarp, id22=id22, id22_arrival=id22_arrival,
                prepares=dict(c22_main=main22, c22_zdraw=z22, m10_main=main10, m10_zdraw=z10, c22_shrunk=s22, m10_shrunk=s10))


def rate(R, base, k, lab, planner, seed=S0, cat=None, pooled=False):
    """Success % of one field (or pooled over its training seeds); the 22 x 22 study's own fields from c22_study."""
    if base == 'c22' and lab in ('id', 'pm', 'pmz', 'pmzd'):
        by = R['c22_study'].get(str(k), {}).get(lab, {})
        ents = [v.get(planner) for v in by.values()] if pooled else [by.get(str(seed), {}).get(planner)]
    else:
        by = R['fields'].get(f'{base}/{k}', {}).get(lab, {})
        ents = [v.get(planner) for v in by.values()] if pooled else [by.get(str(seed), {}).get(planner)]
    ents = [e for e in ents if e]
    if not ents:
        return None
    a = [e['success_total'] if cat is None else e['by_category'].get(cat) for e in ents]
    a = [x for x in a if x]
    return 100 * sum(x[0] for x in a) / sum(x[1] for x in a) if a else None


def field_id(R, base, k, lab, seed=S0):
    """The registered run of a field: the 22 x 22 study's ntrl-demo from the gap study, the rest from this study's fields."""
    if base == 'c22' and lab == 'id':
        return (R.get('id22') or {}).get(str(k), {}).get(str(seed))
    return (R['fields'].get(f'{base}/{k}', {}).get(lab, {}).get(str(seed)) or {}).get('_id')


def planwarp_entry(R, base, k, lab, seed=S0):
    """MPPI in the constructed warp's coordinates for a field without a warp of its own (zq --plan-warp constructed)."""
    rid = field_id(R, base, k, lab, seed)
    return (R.get('planwarp') or {}).get(rid) if rid else None


def planwarp_rate(R, base, k, lab, seed=S0, cat=None):
    e = planwarp_entry(R, base, k, lab, seed)
    if not e:
        return None
    a = e['success_total'] if cat is None else e['by_category'].get(cat)
    return 100 * a[0] / a[1] if a else None


def test_count(R, base, k, lab, planner, seed=S0, cat=None):
    """[solved, runs] of one field under one planner for the three tests: 'x' = MPPI on the maze (test 2), 'z' = MPPI on
    the warped maze (test 3; for ntrl-demo's fields, which have no warp of their own, zq --plan-warp constructed)."""
    if planner == 'z' and lab in ('id', 'idw'):
        e = planwarp_entry(R, base, k, lab, seed)
    elif base == 'c22' and lab in ('id', 'pm', 'pmz', 'pmzd'):
        e = R['c22_study'].get(str(k), {}).get(lab, {}).get(str(seed), {}).get(planner)
    else:
        e = (R['fields'].get(f'{base}/{k}', {}).get(lab, {}).get(str(seed)) or {}).get(planner)
    if not e:
        return None
    return e['success_total'] if cat is None else e['by_category'].get(cat)


def x_arrival(R, base, k, lab, seed=S0):
    """Failures by arrival under MPPI on the maze (arrived_wall, not_arrived), or {}."""
    if base == 'c22' and lab == 'id':
        return (R.get('id22_arrival') or {}).get(str(k), {}).get(str(seed), {})
    e = R['fields'].get(f'{base}/{k}', {}).get(lab, {}).get(str(seed)) or {}
    return {key: e['x'][key] for key in ('arrived_wall', 'not_arrived') if key in e.get('x', {})}


def seeds(R, base, k, lab, planner):
    if base == 'c22' and lab in ('id', 'pm', 'pmz', 'pmzd'):
        by = R['c22_study'].get(str(k), {}).get(lab, {})
    else:
        by = R['fields'].get(f'{base}/{k}', {}).get(lab, {})
    return sorted(int(s) for s, v in by.items() if v.get(planner))


def fig_mazes(R):
    """The 10 x 10 base's gap mazes, each panel at its own scale (walls as lines), quadrants shaded, with the longest
    diagonal-quadrant query's truth path."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    m10 = R['prepares']['m10_main']
    ks = sorted(int(k) for k in m10)
    if not ks:
        return
    fig, axes = plt.subplots(1, len(ks), figsize=(2.9 * len(ks), 3.25))
    for ax, k in zip(np.atleast_1d(axes), ks):
        d = GR.ROOT / 'results' / m10[k]
        info = json.loads((d / 'maze.json').read_text())
        n = info['n']
        G = n - 10
        for (x0, y0) in ((0, 0), (5 + G, 0), (0, 5 + G), (5 + G, 5 + G)):
            ax.add_patch(plt.Rectangle((x0, y0), 5, 5, color='#eb6834', alpha=0.13, lw=0))
        V, H = np.array(info['V']), np.array(info['H'])
        segs = [((i, j), (i, j + 1)) for i, j in zip(*np.nonzero(V))] + [((i, j), (i + 1, j)) for i, j in zip(*np.nonzero(H))]
        ax.add_collection(LineCollection(segs, colors='#3b3a37', linewidths=1.1 if n <= 34 else 0.8, capstyle='projecting'))
        q = np.load(d / 'queries.npz')
        tp = np.load(d / 'truth_paths.npy', allow_pickle=True)
        idx = np.nonzero(q['category'] == 2)[0]
        if len(idx):
            i = idx[np.argmax(q['d_cells'][idx])]
            p = np.asarray(tp[i])
            ax.plot(p[:, 0] * (n + 0.1) + n / 2, p[:, 1] * (n + 0.1) + n / 2, color='#2a78d6', lw=1.2)
        ax.set_xlim(-0.6, n + 0.6)
        ax.set_ylim(-0.6, n + 0.6)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f'gap ×{k}: {n} × {n} cells (band {6 * k})', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / 'gap_mazes10.png', dpi=110)
    plt.close(fig)


def markdown(R):
    f = lambda v: '–' if v is None else f'{v:.1f}'
    L = ['# Pairs drawn on the warped maze; the 10 × 10 maze with the same gaps (generated by zdraw10.py)', '']
    ks = [1, 2, 4, 8, 12]
    L += ['## 22 × 22 study: success (%), training seed 87358', '',
          '| gap | ntrl-demo | constructed, trained on the maze | warped-maze training, real-maze pairs: MPPI on maze / warped | warped-maze training, warped-maze pairs: maze / warped |',
          '|---|---|---|---|---|']
    for k in ks:
        L.append(f'| ×{k} | {f(rate(R, "c22", k, "id", "x"))} | {f(rate(R, "c22", k, "pm", "x"))} | {f(rate(R, "c22", k, "pmz", "x"))} / {f(rate(R, "c22", k, "pmz", "z"))} | '
                 f'{f(rate(R, "c22", k, "pmzw", "x"))} / {f(rate(R, "c22", k, "pmzw", "z"))} |')
    L += ['', '## 10 × 10 maze, same gaps: success (%), training seed 87358', '',
          '| gap | maze | ntrl-demo | constructed, trained on the maze | warped-maze training, real-maze pairs: maze / warped | warped-maze training, warped-maze pairs: maze / warped |',
          '|---|---|---|---|---|---|']
    for k in ks:
        n = (R['controls'].get(str(k)) or {}).get('n')
        L.append(f'| ×{k} | {n} | {f(rate(R, "m10", k, "id", "x"))} | {f(rate(R, "m10", k, "pm", "x"))} | {f(rate(R, "m10", k, "pmz", "x"))} / {f(rate(R, "m10", k, "pmz", "z"))} | '
                 f'{f(rate(R, "m10", k, "pmzw", "x"))} / {f(rate(R, "m10", k, "pmzw", "z"))} |')
    L += ['', '## Pooled over training seeds (seeds), MPPI on the warped maze for warped-maze training, on the maze otherwise', '']
    for base in ('c22', 'm10'):
        for k in ks:
            cells = []
            for lab, pl in (('id', 'x'), ('pmz', 'z'), ('pmzw', 'z')):
                s = seeds(R, base, k, lab, pl)
                if len(s) > 1:
                    cells.append(f'{lab} {f(rate(R, base, k, lab, pl, pooled=True))} ({len(s)})')
            if cells:
                L.append(f'- {base} ×{k}: ' + ', '.join(cells))
    L += ['', '## Where the training pairs lie', '', '| maze | gap | drawn on | x0 in a quadrant | x1 in a quadrant | both ends in quadrants | both in one quadrant |', '|---|---|---|---|---|---|---|']
    for key, v in sorted(R['pairs'].items(), key=lambda t: (t[0].split('/')[0], int(t[0].split('/')[1]), t[0].split('/')[2])):
        b, k, dr = key.split('/')
        L.append(f'| {b} | ×{k} | {dr} maze | {100 * v["x0_quadrant"]:.1f} | {100 * v["x1_quadrant"]:.1f} | {100 * v["both_quadrants"]:.1f} | {100 * v["same_quadrant"]:.1f} |')
    L += ['', '## By query kind, training seed 87358 (MPPI on the warped maze for warped-maze training)', '']
    for base in ('c22', 'm10'):
        for k in (1, 12):
            L.append(f'- {base} ×{k}: ' + '; '.join(f'{c}: ' + ', '.join(f'{lab} {f(rate(R, base, k, lab, pl, cat=c))}' for lab, pl in (('id', 'x'), ('pm', 'x'), ('pmz', 'z'), ('pmzw', 'z'))) for c in CATS))
    return '\n'.join(L) + '\n'


CATS5 = ['same quadrant', 'adjacent quadrants', 'diagonal quadrants', 'quadrant - open', 'open - open']   # gap.py's region-pair categories, in their code order


def planwarp_runs(R):
    """Run by run, MPPI on the maze against MPPI on the warped maze for ntrl-demo's fields (test 2 vs test 3): per
    'base/k/label' and category, solved by each, the runs the warped planner loses (and how: arrived after a wall, or never
    arrived) and the runs it gains. Same queries and planning seeds in both rows files."""
    out = {}
    for base in ('c22', 'm10'):
        for k in (1, 2, 4, 8, 12):
            for lab in ('id', 'idw'):
                frid, e = field_id(R, base, k, lab), planwarp_entry(R, base, k, lab)
                if not frid or not e:
                    continue
                fx = GR.ROOT / 'results' / f'registered_{frid}' / 'rows.npz'
                fz = GR.ROOT / 'results' / f'registered_{e["zq"]}' / frid / 'rows.npz'
                if not fx.exists() or not fz.exists():
                    continue
                x, z = np.load(fx), np.load(fz)
                if x['success'].shape != z['z_success'].shape:
                    continue                                       # the 22 x 22 maze's k = 1 field: its own rows are maze_scale's 240 queries
                xs, zs = x['success'].astype(bool), z['z_success'].astype(bool)
                zc, zv = z['z_collision'].astype(bool), z['z_converged'].astype(bool)
                d = {}
                for ci, c in enumerate(CATS5):
                    m = np.broadcast_to(z['category'] == ci, xs.shape)
                    lost, gain = m & xs & ~zs, m & ~xs & zs
                    d[c] = dict(runs=int(m.sum()), x=int((m & xs).sum()), z=int((m & zs).sum()), lost=int(lost.sum()),
                                lost_wall=int((lost & zv & zc).sum()), lost_stall=int((lost & ~zv).sum()), gained=int(gain.sum()))
                out[f'{base}/{k}/{lab}'] = d
    return out


def main():
    R = collect()
    R['planwarp_runs'] = planwarp_runs(R)
    (OUT / 'zdraw10.json').write_text(json.dumps(R, indent=1, default=float) + '\n')
    md = markdown(R)
    (OUT / 'zdraw10.md').write_text(md)
    fig_mazes(R)
    print(md)


if __name__ == '__main__':
    main()
