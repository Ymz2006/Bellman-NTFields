"""Tables and figures for the midpoint-loss write-up (report/*.md, *.json, *.png), from registered runs only.

    ~/.venvs/bellman-ntfield/bin/python jeffrey/midpoint_dp/report.py [--figs] [--wall]

Runs: the registered ntrl-demo runs of jeffrey/maze_scale (the baseline, epoch 5000) and every registered, succeeded,
non-smoke midpoint run (stage 'midpoint'). A midpoint run's variant is named from its config (weight, leaf values,
traversal, nodes, and any non-default setting); '--weight 0' is the control (the same resumed loop, no midpoint term).
--wall adds near-wall slope / direction (jeffrey/maze_scale/wallfeel.py measure) per run (GPU, cached in report/wall.json);
--figs draws the PNGs (inference only for the path figure).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / 'results'
REG = Path('~/.local/share/ntfields-experiments/runs').expanduser()
OUT = HERE / 'report'
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
from common import bin_labels  # noqa: E402

SIZES = [10, 13, 22]
SEEDS = [87358, 1, 2]
BASE = {(10, 87358): '20261102T032216Z-e76476bf', (10, 1): '20261102T033305Z-5d802f80', (10, 2): '20261102T032238Z-09a41392',
        (13, 87358): '20261101T203437Z-ebc95225', (13, 1): '20261101T204630Z-58badec5', (13, 2): '20261101T205900Z-c57aa0f1',
        (22, 87358): '20261101T193703Z-dd44f02b', (22, 1): '20261101T194908Z-fa2678e3', (22, 2): '20261101T200614Z-92e8b351'}
FOURIER = {(10, 87358): '20261102T040322Z-06099883', (10, 1): '20261102T050643Z-d64a530a', (10, 2): '20261102T051954Z-3166a3d5',
           (13, 87358): '20261102T014741Z-dbca8e69', (13, 1): '20261102T023943Z-95b7bd4e', (13, 2): '20261102T025340Z-27b1eeb6',
           (22, 87358): '20261102T032133Z-9d61cff5', (22, 1): '20261102T050535Z-7879ad1e', (22, 2): '20261102T051820Z-834b2691'}
BSCALE = {10: 3, 13: 3, 22: 6}                    # the best Fourier scale per maze (jeffrey/maze_scale fourier.py sweep)
DEFAULTS = dict(start=2000, ramp=500, refresh=100, retarget=0, traversals=8192, batch=8192, leaf_cells=1.0, loss='l1', reached_only=False,
                local_share=0.0, budget_factor=2.0, stall_window=60, trim_hover=0.25, safe_margin=0.02, safe_mode='first', sub_factor=1.0)
MAIN = 'safe, min leaves, tree + sub-paths'
COLORS = {'ntrl-demo': '#2a78d6', 'control': '#8a8984', MAIN: '#eb6834', 'plain, min leaves, tree': '#1baf7a', 'plain, net leaves, tree': '#6250d6',
          'safe, net leaves, tree': '#eda100', 'safe, min leaves, tree': '#e87ba4'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e6e5e0'


def variant(cfg, b_scale=1.0):
    if b_scale != 1.0:
        return ('control' if cfg['weight'] == 0 else variant(cfg)) + f' on B×{b_scale:g}'
    if cfg['weight'] == 0:
        return 'control'
    if cfg.get('anchor', 'truth') == 'tc' or cfg.get('cut', 'truth') == 'tc':
        return tc_name(cfg)
    name = f'{cfg.get("traverse", "plain")}, {cfg["leaf"]} leaves, ' + {'tree': 'tree', 'sub': 'sub-paths', 'both': 'tree + sub-paths'}[cfg.get('nodes', 'tree')]
    extra = [f'{k} {cfg[k]}' for k, v in DEFAULTS.items() if k in cfg and cfg[k] != v and not (k == 'sub_factor' and cfg.get('nodes', 'tree') == 'tree')]
    if cfg['weight'] != 1:
        extra.insert(0, f'weight {cfg["weight"]:g}')
    return name + (' (' + ', '.join(extra) + ')' if extra else '')


def tc_name(cfg):
    """Runs where the close-range network Tc replaces the exact speed (closenet.py)."""
    k = cfg['tc_pairs']
    size = f'{k // 1000}k' if k < 1_000_000 else f'{k / 1e6:g}M'
    if cfg['tc_mode'] == 'a':
        name = f'Tc (a): d {cfg["tc_clear"]:g}, {size} pairs'
    else:
        name = f'Tc (b): epoch {cfg["tc_epoch"]}, tau {cfg["tc_tau"]:g}, {size} pairs'
    extra = []
    if cfg['leaf'] in ('tc', 'mintc'):
        extra.append('direct leaves' if cfg['leaf'] == 'mintc' else 'Tc leaves')
    if cfg.get('cut') == 'tc' and cfg.get('ends') == 'data':
        extra.append('no truth in loop')
    elif cfg.get('cut') == 'tc':
        extra.append('Tc cut')
    if cfg.get('traverse', 'plain') != 'safe':
        extra.append(cfg.get('traverse', 'plain'))
    if cfg['weight'] != 1:
        extra.append(f'weight {cfg["weight"]:g}')
    return name + (', ' + ', '.join(extra) if extra else '')


def state(run_id):
    p = REG / run_id / 'record.json'
    return json.load(open(p))['state'] if p.exists() else None


def collect():
    runs = defaultdict(dict)                      # (n, variant) -> seed -> results (+ _dir, _id)
    for (n, s), rid in BASE.items():
        d = RESULTS / f'registered_{rid}'
        r = json.load(open(d / 'results.json'))
        r['_dir'], r['_id'] = d, rid
        runs[(n, 'ntrl-demo')][s] = r
    for (n, s), rid in FOURIER.items():
        d = RESULTS / f'registered_{rid}'
        r = json.load(open(d / 'results.json'))
        r['_dir'], r['_id'] = d, rid
        runs[(n, f'ntrl-demo B×{BSCALE[n]}')][s] = r
    for d in sorted(RESULTS.glob('registered_*')):
        f = d / 'results.json'
        if not f.exists():
            continue
        try:
            r = json.load(open(f))
        except ValueError:
            continue
        if r.get('stage') != 'midpoint' or r.get('smoke') or not r.get('complete'):
            continue
        rid = d.name.replace('registered_', '')
        if state(rid) != 'succeeded':
            continue
        r['_dir'], r['_id'] = d, rid
        key = (r['maze']['n'], variant(r['config'], r.get('b_scale', 1.0)))
        if r['seed'] in runs[key]:
            continue                               # first registered run of a (maze, variant, seed) counts
        runs[key][r['seed']] = r
    return runs


def pct(a, b):
    return f'{100 * a / b:.0f}' if b else '-'


def by_bin_pooled(rs):
    out = {}
    for r in rs:
        for k, (a, b) in r['success_by_bin'].items():
            x = out.setdefault(k, [0, 0])
            x[0] += a
            x[1] += b
    return out


def tables(runs, wall):
    labels = bin_labels()
    md = []
    variants_main = ['ntrl-demo', 'control', MAIN]
    fourier_pair = lambda n: [f'ntrl-demo B×{BSCALE[n]}', f'{MAIN} on B×{BSCALE[n]}']
    # 1. headline: per seed and pooled
    md += ['## Success by maze and training seed', '',
           'MPPI success (runs that reach the goal without touching a wall), planning seeds 0/1/2 pooled; per training seed and over all seeds.', '',
           '| maze | field | ' + ' | '.join(f'seed {s}' for s in SEEDS) + ' | all | % |', '|---|---|' + '---|' * (len(SEEDS) + 2)]
    head = {}
    for n in SIZES:
        for v in variants_main + fourier_pair(n):
            rs = runs.get((n, v), {})
            if not rs:
                continue
            cells = [f'{rs[s]["success_total"][0]}/{rs[s]["success_total"][1]}' if s in rs else '' for s in SEEDS]
            a = sum(rs[s]['success_total'][0] for s in rs)
            b = sum(rs[s]['success_total'][1] for s in rs)
            head[(n, v)] = (a, b, len(rs))
            md.append(f'| {n}×{n} | {"**" + v + "**" if v == MAIN else v} | ' + ' | '.join(cells) + f' | {a}/{b} | {pct(a, b)} |')
    md.append('')
    # 2. by bin pooled
    md += ['## Success by true travel distance', '', 'Pooled over the training seeds listed above and planning seeds 0/1/2 (%).', '',
           '| maze | field | ' + ' | '.join(labels[:6]) + ' |', '|---|---|' + '---|' * 6]
    for n in SIZES:
        for v in variants_main + fourier_pair(n):
            rs = runs.get((n, v), {})
            if not rs:
                continue
            bb = by_bin_pooled(rs.values())
            md.append(f'| {n}×{n} | {v} | ' + ' | '.join(pct(*bb[k]) if k in bb else '' for k in labels[:6]) + ' |')
    md.append('')
    # 3. field diagnostics
    md += ['## Field diagnostics', '',
           'Median T(start, goal) / T* per distance bin and median cosine between the field\'s descent direction at the start and the truth\'s (mean over training seeds); '
           'held-out log-speed correlation (speed fit); near-wall slope S|∇T| / cos(∇T, into the wall) at held-out near-wall poses, gap 0.5-2 and 2-8 cells (training seed 87358).', '',
           '| maze | field | T/T* by bin | descent cos | speed fit | near wall 0.5-2 | near wall 2-8 |', '|---|---|---|---|---|---|---|']
    for n in SIZES:
        for v in variants_main + [k[1] for k in runs if k[0] == n and k[1] not in variants_main]:
            rs = runs.get((n, v), {})
            if not rs:
                continue
            ratio = defaultdict(list)
            for r in rs.values():
                for k, x in r['field']['by_bin'].items():
                    ratio[k].append(x['t_over_tstar_median'])
            rt = ' / '.join(f'{np.mean(ratio[k]):.2f}' for k in labels if k in ratio)
            cos = np.mean([r['field']['descent_cos_median'] for r in rs.values()])
            sf = np.mean([r['speed_fit']['log_speed_corr'] for r in rs.values()])
            w = wall.get(rs[87358]['_id']) if 87358 in rs else None
            wtxt = [f'{w[k]["slope"]:.2f} / {w[k]["cos"]:.2f}' if w and k in w else '' for k in ('0.5-2', '2-8')]
            md.append(f'| {n}×{n} | {v} | {rt} | {cos:.3f} | {sf:.2f} | {wtxt[0]} | {wtxt[1]} |')
    md.append('')
    # 4. variants
    md += ['## Variants', '', 'Every registered midpoint variant (planning seeds pooled; training seeds as listed).', '',
           '| maze | variant | seeds | success | % | ' + ' | '.join(labels[:6]) + ' |', '|---|---|---|---|---|' + '---|' * 6]
    for (n, v), rs in sorted(runs.items(), key=lambda t: (t[0][0], t[0][1] != 'ntrl-demo', t[0][1] != 'control', t[0][1])):
        a = sum(r['success_total'][0] for r in rs.values())
        b = sum(r['success_total'][1] for r in rs.values())
        bb = by_bin_pooled(rs.values())
        md.append(f'| {n}×{n} | {v} | {", ".join(str(s) for s in sorted(rs))} | {a}/{b} | {pct(a, b)} | ' + ' | '.join(pct(*bb[k]) if k in bb else '' for k in labels[:6]) + ' |')
    md.append('')
    # 5. failure kinds
    md += ['## Failure kinds', '', 'Per planning run: collision = touched a wall; stall = never reached the goal without touching a wall (planning seeds and training seeds pooled).', '',
           '| maze | field | runs | success | collision | stall only |', '|---|---|---|---|---|---|']
    for n in SIZES:
        for v in variants_main + fourier_pair(n):
            rs = runs.get((n, v), {})
            if not rs:
                continue
            tot = succ = coll = stall = 0
            for r in rs.values():
                z = np.load(r['_dir'] / 'rows.npz')
                tot += z['success'].size
                succ += int(z['success'].sum())
                coll += int(z['collision'].sum())
                stall += int((~z['converged'] & ~z['collision']).sum())
            md.append(f'| {n}×{n} | {v} | {tot} | {succ} | {coll} | {stall} |')
    md.append('')
    return '\n'.join(md), head


def training_table(runs):
    md = ['## Training traversals', '', 'Share of the refresh\'s MPPI traversals (random collision-free pairs, current field) that reach their goal without touching a wall, '
          'at the first refresh (epoch 2001, the baseline\'s field) and the last; and the median log(T / target) over the bank at those refreshes.', '',
          '| maze | variant | seed | reached first | reached last | stalled first | stalled last | log(T/target) first | last |', '|---|---|---|---|---|---|---|---|---|']
    for (n, v), rs in sorted(runs.items()):
        if v in ('ntrl-demo', 'control'):
            continue
        for s in sorted(rs):
            ref = rs[s].get('refresh', [])
            if not ref:
                continue
            f, l = ref[0], ref[-1]
            md.append(f'| {n}×{n} | {v} | {s} | {pct(f["reached"], f["traversals"])} % | {pct(l["reached"], l["traversals"])} % | {pct(f["stalled"], f["traversals"])} % | '
                      f'{pct(l["stalled"], l["traversals"])} % | {f["log_net_over_target_median"]:.3f} | {l["log_net_over_target_median"]:.3f} |')
    md.append('')
    return '\n'.join(md)


def wall_cache(runs, compute):
    path = OUT / 'wall.json'
    cache = json.load(open(path)) if path.exists() else {}
    if not compute:
        return cache
    import torch
    import maze as MZ
    import plan as P
    import wallfeel as WF
    for (n, v), rs in runs.items():
        r = rs.get(87358)
        if r is None or r['_id'] in cache:
            continue
        maze = MZ.Maze(n)
        net = P.load_field(r['_dir'] / 'training/checkpoint.pt')
        cache[r['_id']] = WF.measure(maze, Path(r['prepared']), lambda x: P.field_grad(net, x))
        del net
        torch.cuda.empty_cache()
        print(f'wall {n} {v}: {cache[r["_id"]]}', flush=True)
    path.write_text(json.dumps(cache, indent=1))
    return cache


RANGES = [(0.5, 2), (2, 8), (8, 1e9)]


def grad_cache(runs, compute, extra=()):
    """Descent direction near walls by goal distance: 400,000 uniform poses paired with the 48 truth goals; for goals at
    T* in [0.5, 2), [2, 8) and >= 8 cells, the share of near-wall poses (offset < clearance < margin) whose descent
    direction -grad_x T has a component of more than 0.3 into the wall, and the free-space |grad_x T| (clearance >= 0.25
    cells, truth 1). Every run of the headline variants and every 13 x 13 training-seed-87358 variant; inference only;
    cached in report/grad.json (key: run id)."""
    path = OUT / 'grad.json'
    cache = json.load(open(path)) if path.exists() else {}
    cache = {k: v for k, v in cache.items() if 'ranges' in v}
    if not compute:
        return cache
    import torch
    import maze as MZ
    import plan as P
    import truth as TR
    for n in SIZES:
        heads = ('ntrl-demo', 'control', MAIN, f'ntrl-demo B×{BSCALE[n]}', f'{MAIN} on B×{BSCALE[n]}') + tuple(extra)
        todo = [(v, s, r) for (m, v), rs in runs.items() if m == n for s, r in rs.items()
                if r['_id'] not in cache and (v in heads or (n == 13 and s == 87358))]
        if not todo:
            continue
        prep = Path(todo[0][2]['prepared'])
        maze = MZ.Maze(n)
        grid = TR.Grid(maze)
        del grid.costs, grid.cost_flat
        fields = torch.tensor(np.load(prep / 'truth_fields.npy'), device='cuda:0').float()
        goals = torch.tensor(np.load(prep / 'goals.npy'), device='cuda:0')
        gen = torch.Generator('cuda:0').manual_seed(0)
        q = maze.uniform(400_000, gen)
        d_c, normal = maze.clearance_normal(q)
        gi = torch.randint(len(goals), (len(q),), generator=gen, device='cuda:0')
        tstar = grid.query_time(fields, gi, q) / maze.pitch
        nearw = (d_c > MZ.OFFSET_CELLS) & (d_c < MZ.MARGIN_CELLS)
        free = d_c >= 0.25
        pairs = torch.cat([q, goals[gi]], 1)
        for v, s, r in todo:
            net = P.load_field(r['_dir'] / 'training/checkpoint.pt')
            t, g = P.field_grad(net, pairs)
            gn = torch.linalg.vector_norm(g, dim=1)
            into = ((-g / gn[:, None].clamp(min=1e-9)) * normal).sum(1) < -0.3
            rec = dict(ranges={})
            for lo, hi in RANGES:
                band = (tstar >= lo) & (tstar < hi)
                rec['ranges'][f'{lo:g}-{hi:g}' if hi < 1e8 else f'>{lo:g}'] = dict(into_wall=float(into[band & nearw].float().mean()),
                                                                                  free_grad=float(gn[band & free].median()), n_near=int((band & nearw).sum()))
            far = rec['ranges'][f'>{RANGES[-1][0]:g}']
            rec.update(into_wall=far['into_wall'], free_grad=far['free_grad'])
            cache[r['_id']] = rec
            print(f'grad {n} {v} seed {s}: ' + ', '.join(f'{k} {100 * x["into_wall"]:.1f} %' for k, x in rec['ranges'].items()), flush=True)
            del net
            torch.cuda.empty_cache()
        del fields, grid
        torch.cuda.empty_cache()
        path.write_text(json.dumps(cache, indent=1))
    return cache


def grad_table(runs, cache):
    keys = [f'{lo:g}-{hi:g}' if hi < 1e8 else f'>{lo:g}' for lo, hi in RANGES]
    md = ['## Descent direction near walls', '', 'Share of near-wall poses whose descent direction points into the wall (component > 0.3 along the into-wall normal), '
          'by the goal\'s true distance in cells, and median free-space |∇T| for goals >= 8 cells (truth 1); mean over training seeds.', '',
          '| maze | field | seeds | ' + ' | '.join(f'into wall, {k} (%)' for k in keys) + ' | free-space abs grad T |', '|---|---|---|' + '---|' * (len(keys) + 1)]
    for n in SIZES:
        names = ['ntrl-demo', 'control', MAIN, f'ntrl-demo B×{BSCALE[n]}', f'{MAIN} on B×{BSCALE[n]}']
        if n == 13:
            names += sorted(v for (m, v) in runs if m == 13 and v not in names)
        for v in names:
            got = [cache[r['_id']] for r in runs.get((n, v), {}).values() if r['_id'] in cache]
            if got:
                md.append(f'| {n}×{n} | {v} | {len(got)} | ' + ' | '.join(f'{100 * np.mean([x["ranges"][k]["into_wall"] for x in got]):.1f}' for k in keys)
                          + f' | {np.mean([x["free_grad"] for x in got]):.2f} |')
    md.append('')
    return '\n'.join(md)


UTURN = {10: '20261102T032220Z-de32a8ec', 13: '20261102T013501Z-64a34273', 22: '20261102T013558Z-5d2a1408'}


def uturn_cache(runs, compute):
    """U-turn queries of jeffrey/maze_scale/uturn.py (start and goal one cell apart on either side of a wall end, true path
    around it; registered uturn runs) planned exactly as uturneval.py does (stage-2 MPPI settings and budget rule,
    planning seeds 0/1/2, exact scorer), for every run of the headline variants. Inference only; cached in report/uturn.json."""
    path = OUT / 'uturn.json'
    cache = json.load(open(path)) if path.exists() else {}
    if not compute:
        return cache
    import torch
    import maze as MZ
    import plan as P
    from uturn import budget_of
    for (n, v), rs in sorted(runs.items()):
        if n not in UTURN or v not in ('ntrl-demo', 'control', MAIN, f'ntrl-demo B×{BSCALE[n]}', f'{MAIN} on B×{BSCALE[n]}'):
            continue
        prep = RESULTS / f'registered_{UTURN[n]}'
        z = np.load(prep / 'uturn.npz')
        maze = MZ.Maze(n)
        S, G = torch.tensor(z['start'], device='cuda:0'), torch.tensor(z['goal'], device='cuda:0')
        st, budget = budget_of(maze, z['path_clen'])
        budget = torch.tensor(budget, device='cuda:0')
        for s, r in rs.items():
            if r['_id'] in cache:
                continue
            net = P.load_field(r['_dir'] / 'training/checkpoint.pt')
            cost = P.field_cost(net)
            ok, coll = 0, 0
            for ps in (0, 1, 2):
                paths, stop, conv = P.mppi(cost, S, G, ps, budget, maze.lo, maze.hi, **st)
                rows = P.score(maze, [paths[i] for i in range(len(S))], conv)
                ok += sum(q['success'] for q in rows)
                coll += sum(q['collision'] for q in rows)
            cache[r['_id']] = dict(success=int(ok), total=3 * len(S), collision=int(coll))
            print(f'uturn {n} {v} seed {s}: {ok}/{3 * len(S)}', flush=True)
            del net
            torch.cuda.empty_cache()
        path.write_text(json.dumps(cache, indent=1))
    return cache


def uturn_table(runs, cache):
    md = ['## U-turns around a wall end', '', 'jeffrey/maze_scale u-turn queries (40 per maze: start and goal 1 cell apart on either side of a wall end, true path ~2.2 cells around it), '
          'planning seeds 0/1/2; per training seed and pooled. Inference on the registered checkpoints.', '',
          '| maze | field | ' + ' | '.join(f'seed {s}' for s in SEEDS) + ' | all | % |', '|---|---|' + '---|' * (len(SEEDS) + 2)]
    for n in SIZES:
        for v in ['ntrl-demo', 'control', MAIN, f'ntrl-demo B×{BSCALE[n]}', f'{MAIN} on B×{BSCALE[n]}']:
            rs = runs.get((n, v), {})
            got = {s: cache[r['_id']] for s, r in rs.items() if r['_id'] in cache}
            if not got:
                continue
            a = sum(x['success'] for x in got.values())
            b = sum(x['total'] for x in got.values())
            md.append(f'| {n}×{n} | {v} | ' + ' | '.join(f'{got[s]["success"]}/{got[s]["total"]}' if s in got else '' for s in SEEDS) + f' | {a}/{b} | {pct(a, b)} |')
    md.append('')
    return '\n'.join(md)


def figures(runs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = bin_labels()
    show = ['ntrl-demo', 'control', MAIN]

    def style(ax):
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color(INK2)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.grid(axis='y', color=GRID, lw=0.8)
        ax.set_axisbelow(True)

    # success by bin
    fig, axes = plt.subplots(1, len(SIZES), figsize=(4.2 * len(SIZES), 3.4), sharey=True)
    for ax, n in zip(axes, SIZES):
        for v in show:
            rs = runs.get((n, v), {})
            if not rs:
                continue
            ks = [k for k in labels if any(k in r['success_by_bin'] for r in rs.values())]
            per = np.array([[100 * r['success_by_bin'][k][0] / r['success_by_bin'][k][1] for k in ks] for r in rs.values()])
            x = np.arange(len(ks))
            ls = '--' if v == 'control' else '-'
            ax.plot(x, per.mean(0), ls, color=COLORS[v], lw=2, marker='o', ms=5, label=f'{v} ({len(rs)} seeds)')
            if len(rs) > 1:
                ax.vlines(x, per.min(0), per.max(0), color=COLORS[v], lw=1.2, alpha=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(ks, fontsize=8)
        ax.set_title(f'{n} × {n} maze', fontsize=10, color=INK)
        ax.set_xlabel('true travel distance (cells)', fontsize=8, color=INK2)
        ax.set_ylim(0, 102)
        style(ax)
    axes[0].set_ylabel('MPPI success (%)', fontsize=9, color=INK2)
    axes[0].legend(fontsize=7, loc='lower left', frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / 'success.png', dpi=130)
    plt.close(fig)
    # validation curves
    fig, axes = plt.subplots(1, len(SIZES), figsize=(4.2 * len(SIZES), 3.2), sharey=False)
    for ax, n in zip(axes, SIZES):
        base = runs.get((n, 'ntrl-demo'), {})
        for v in ['control', MAIN]:
            rs = runs.get((n, v), {})
            if not rs:
                continue
            curves = []
            for r in rs.values():
                vv = {x['epoch']: x['success'] for x in r.get('validate', []) if 'success' in x}
                b0 = r['planning']['0']['success']
                vv[5000] = b0
                curves.append(vv)
            eps = sorted(set.intersection(*[set(c) for c in curves]))
            tot = list(rs.values())[0]['planning']['0']
            q = len(np.load(list(rs.values())[0]['_dir'] / 'rows.npz')['success'][0])
            y = np.array([[100 * c[e] / q for e in eps] for c in curves])
            ls = '--' if v == 'control' else '-'
            ax.plot(eps, y.mean(0), ls, color=COLORS[v], lw=2, marker='o', ms=4, label=f'{v} ({len(rs)} seeds)')
        if base:
            q = len(np.load(list(base.values())[0]['_dir'] / 'rows.npz')['success'][0])
            b = np.mean([100 * r['planning']['0']['success'] / q for r in base.values()])
            ax.axhline(b, color=COLORS['ntrl-demo'], lw=1.5, ls=':', label='ntrl-demo epoch 5000')
        ax.set_title(f'{n} × {n}: planning seed 0 during training', fontsize=9, color=INK)
        ax.set_xlabel('epoch (midpoint term from 2000, ramp to 2500)', fontsize=8, color=INK2)
        style(ax)
    axes[0].set_ylabel('success (%)', fontsize=9, color=INK2)
    axes[0].legend(fontsize=7, loc='lower right', frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / 'curves.png', dpi=130)
    plt.close(fig)
    # T / T* by bin
    fig, axes = plt.subplots(1, len(SIZES), figsize=(4.2 * len(SIZES), 3.2), sharey=True)
    for ax, n in zip(axes, SIZES):
        for v in show:
            rs = runs.get((n, v), {})
            if not rs:
                continue
            ks = [k for k in labels if k in list(rs.values())[0]['field']['by_bin']]
            y = np.array([[r['field']['by_bin'][k]['t_over_tstar_median'] for k in ks] for r in rs.values()])
            ls = '--' if v == 'control' else '-'
            ax.plot(np.arange(len(ks)), y.mean(0), ls, color=COLORS[v], lw=2, marker='o', ms=5, label=v)
            ax.set_xticks(np.arange(len(ks)))
            ax.set_xticklabels(ks, fontsize=8)
        ax.axhline(1, color=INK2, lw=1)
        ax.set_title(f'{n} × {n}: learned T / true T*', fontsize=9, color=INK)
        ax.set_xlabel('true travel distance (cells)', fontsize=8, color=INK2)
        style(ax)
    axes[0].set_ylabel('median T / T*', fontsize=9, color=INK2)
    axes[0].legend(fontsize=7, loc='upper left', frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / 'ratio.png', dpi=130)
    plt.close(fig)


def fig_paths(runs, n=13, seed=87358, bins=('8-16', '16-32', '16-32')):
    """Planned paths (planning seed 0) on the n-cell maze for queries in the given bins where ntrl-demo fails and the
    midpoint field succeeds (first such queries), with the truth path."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    base, mid = runs.get((n, 'ntrl-demo'), {}).get(seed), runs.get((n, MAIN), {}).get(seed)
    if base is None or mid is None:
        return
    prep = Path(mid['prepared'])
    info = json.load(open(prep / 'maze.json'))
    q = np.load(prep / 'queries.npz')
    truth = np.load(prep / 'truth_paths.npy', allow_pickle=True)
    labels = bin_labels()
    ok_b = np.load(base['_dir'] / 'rows.npz')['success'][0]
    ok_m = np.load(mid['_dir'] / 'rows.npz')['success'][0]
    picks, used = [], set()
    for lab in bins:
        b = labels.index(lab)
        cand = [i for i in np.nonzero(q['bin'] == b)[0] if ok_m[i] and not ok_b[i] and i not in used]
        if cand:
            picks.append(int(cand[0]))
            used.add(int(cand[0]))
    E = info['n'] + 0.1
    fig, axes = plt.subplots(1, len(picks), figsize=(3.5 * len(picks), 3.8))
    for ax, i in zip(np.atleast_1d(axes), picks):
        for x0, y0, x1, y1 in info['boxes_cells']:
            ax.add_patch(plt.Rectangle(((x0 - info['n'] / 2) / E, (y0 - info['n'] / 2) / E), (x1 - x0) / E, (y1 - y0) / E, color='#9c9b97', lw=0))
        ax.plot(truth[i][:, 0], truth[i][:, 1], color=INK, lw=1.2, ls='--', label='truth')
        for nm, r, col in (('ntrl-demo', base, COLORS['ntrl-demo']), ('midpoint loss', mid, COLORS[MAIN])):
            p = np.load(r['_dir'] / 'paths_seed0.npy', allow_pickle=True)[i]
            z = np.load(r['_dir'] / 'rows.npz')
            okk, conv, hit = bool(z['success'][0, i]), bool(z['converged'][0, i]), bool(z['collision'][0, i])
            if not conv:
                p = p[:-1]
            how = 'reaches goal' if okk else ('hits a wall' if hit else 'stalls')
            ax.plot(p[:, 0], p[:, 1], color=col, lw=1.5, label=f'{nm} ({how})')
        ax.plot(*truth[i][0, :2], 'o', color=INK, ms=5)
        ax.plot(*truth[i][-1, :2], 's', color=INK, ms=5)
        ax.set_xlim(-0.505, 0.505)
        ax.set_ylim(-0.505, 0.505)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f'{n} × {n}, query {i}: {q["d_cells"][i]:.1f} cells', fontsize=9, color=INK)
        ax.legend(fontsize=6.5, loc='lower left', frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / f'paths{n}.png', dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--figs', action='store_true')
    p.add_argument('--wall', action='store_true')
    p.add_argument('--uturn', action='store_true')
    p.add_argument('--grad', action='store_true')
    a = p.parse_args()
    OUT.mkdir(exist_ok=True)
    runs = collect()
    wall = wall_cache(runs, a.wall)
    md, head = tables(runs, wall)
    md += '\n' + training_table(runs)
    md += '\n' + uturn_table(runs, uturn_cache(runs, a.uturn))
    md += '\n' + grad_table(runs, grad_cache(runs, a.grad))
    index = ['## Run index', '', '| maze | field | seed | run |', '|---|---|---|---|']
    for (n, v), rs in sorted(runs.items()):
        for s in sorted(rs):
            index.append(f'| {n}×{n} | {v} | {s} | `{rs[s]["_id"]}` |')
    md += '\n' + '\n'.join(index) + '\n'
    (OUT / 'tables.md').write_text('# Midpoint loss: tables (generated by report.py)\n\n' + md)
    data = {}
    for (n, v), rs in runs.items():
        data.setdefault(str(n), {})[v] = {str(s): dict(id=r['_id'], success=r['success_total'], by_bin=r['success_by_bin'],
                                                      ratio={k: x['t_over_tstar_median'] for k, x in r['field']['by_bin'].items()},
                                                      cos=r['field']['descent_cos_median'], speed_fit=r['speed_fit']['log_speed_corr'],
                                                      planning={k: x['success'] for k, x in r['planning'].items()},
                                                      validate={str(x['epoch']): x.get('success') for x in r.get('validate', []) if 'success' in x},
                                                      config=r.get('config'), b_scale=r.get('b_scale', 1.0))
                                          for s, r in rs.items()}
    (OUT / 'data.json').write_text(json.dumps(dict(runs=data, wall=wall, uturn=json.load(open(OUT / 'uturn.json')) if (OUT / 'uturn.json').exists() else {},
                                                   grad=json.load(open(OUT / 'grad.json')) if (OUT / 'grad.json').exists() else {}), indent=1))
    (OUT / 'summary.json').write_text(json.dumps({f'{n}|{v}': dict(success=a, runs=b, seeds=k) for (n, v), (a, b, k) in head.items()}, indent=1))
    if a.figs:
        figures(runs)
        for n in SIZES:
            fig_paths(runs, n)
    print(md)


if __name__ == '__main__':
    main()
