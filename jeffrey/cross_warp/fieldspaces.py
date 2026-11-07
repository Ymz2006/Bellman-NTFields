"""Data for the travel-time section of the warp-space viewer (report/warpspaces.html): learned and true T(., goal) of the
constructed warp trained in x (the gap study) and trained in z (trainer.py z_space), and ntrl-demo, on a grid of the
maze, with the grid's position in the warp's space, so the viewer can draw every field on the maze and on the warped
maze. Inference and ground truth only.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ~/.venvs/bellman-ntfield/bin/python jeffrey/cross_warp/fieldspaces.py [k ...]

Per gap scale k (one process per k): three goals at the same place relative to the quadrants at every gap (heading 0):
the band's centre, the lower band arm's centre (between the two bottom quadrants) and a cell deep in the top-right
quadrant. Fields on the nodes of maze_scale's truth grid (1/15 cell, 13 headings) taken every 8th node (0.53 cells at
every gap), the minimum over the 13 headings at each (x, y):
  true_x  T* of the maze (truthfast.FastGrid), the time every x-trained field learns, in cells;
  true_z  T* of the warped maze (warptruth.WarpGrid: lengths in z, the maze's speed), the time the z-trained field
          learns, in quadrant cells (z-units / (m / E): a quadrant cell counts 1, a band cell 1 / m);
  ntrl, x_trained (cells) and z_trained (quadrant cells): the fields (training seed 87358).
u, v: phi(x, y, 0) of the constructed warp at the nodes. Errors (median |T - T*| / T* by region, nodes farther than one
cell from the goal and outside walls) against each field's own truth, and the z-trained field also against true_x.
report/fieldspaces/k<k>.json.
"""
import base64
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
sys.path.insert(0, str(HERE))
OUT = HERE / 'report/fieldspaces'
BUILD = Path('/tmp/claude-1000/-home-jeffrey-bellman-ntfield/0579a4a5-2910-4b5c-b9c3-fbb9de47e385/scratchpad/zs/build')
SEED = 87358


def q16(a):
    """uint16 codes of a float array (NaN -> 65535) and its range, base64."""
    a = np.asarray(a, np.float64)
    ok = np.isfinite(a)
    lo, hi = (float(a[ok].min()), float(a[ok].max())) if ok.any() else (0.0, 1.0)
    c = np.full(a.shape, 65535, np.uint16)
    c[ok] = np.round((a[ok] - lo) / max(hi - lo, 1e-12) * 65534).astype(np.uint16)
    return dict(lo=lo, hi=hi, b64=base64.b64encode(c.tobytes()).decode())


PLUS22 = 'registered_20261102T042454Z-62864e61'          # the 22 x 22 cross maze's prepare run (k = 1 training data)
VARIANTS = {'z_trained': 'trained on the warped maze', 'z_trained_k1data': 'trained on the warped maze, the 22 x 22 pairs',
            'x_trained_k1data': 'trained on the maze, the 22 x 22 pairs', 'z_trained_zdraw': 'trained on the warped maze, pairs drawn on the warped maze',
            'x_trained_zdraw': 'trained on the maze, pairs drawn on the warped maze'}


def own_runs(k, preps):
    """The constructed warp's runs of this study on gap maze k: {field key: {seed: run dir}}. z_trained = warped-space
    training on k's own data (k = 1: the 22 x 22 prepare run); *_k1data = training on the 22 x 22 pairs mapped into k
    (gapprepare --data-from), in z (z_trained_k1data) or on the maze (x_trained_k1data); *_zdraw = training on pairs drawn on
    the warped maze (gapprepare --z-draw), in z (z_trained_zdraw) or on the maze (x_trained_zdraw)."""
    n = 16 + 6 * k
    main = PLUS22 if k == 1 else preps[k]['_dir'].name
    mapped, zdraw = set(), set()
    cross22 = lambda d: ((d.get('maze') or {}).get('base') or {}).get('name', 'cross22') == 'cross22'
    for p in (ROOT / 'results').glob('registered_*/results.json'):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if d.get('stage') == 'gapprepare' and d.get('complete') and not d.get('smoke') and (d.get('data') or {}).get('mapped_from') and d.get('scale') == k:
            mapped.add(p.parent.name)
        if (d.get('stage') == 'gapprepare' and d.get('complete') and not d.get('smoke') and d.get('scale') == k and cross22(d)
                and str((d.get('data') or {}).get('drawn_on', '')).startswith('warped') and ((d.get('data') or {}).get('warp') or {}).get('m_lo', 1.0) == 1.0):
            zdraw.add(p.parent.name)
    out = {}
    for p in sorted((ROOT / 'results').glob('registered_*/results.json')):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if d.get('stage') != 'cross-warp' or not d.get('complete') or d.get('smoke') or d['maze']['n'] != n or not cross22(d):
            continue
        if (d.get('design') or {}).get('kind') != 'plusmag' or float((d.get('design') or {}).get('m_lo') or 1.0) != 1.0:
            continue                                                  # the gap study's warp only (not the shrunk-band control's m_lo < 1)
        prep, zs = Path(d['prepared']).name, bool((d.get('trainer') or {}).get('z_space'))
        key = (('z_trained' if zs else None) if prep == main else ('z_trained_k1data' if zs else 'x_trained_k1data') if prep in mapped
               else ('z_trained_zdraw' if zs else 'x_trained_zdraw') if prep in zdraw else None)
        if key:
            out.setdefault(key, {})[d['seed']] = p.parent
    return out


def compute(k):
    import gapreport as GR
    import gap
    gap.install(k)
    import maze as MZ
    import plan as P
    import truthfast as TF
    import warp as W
    import warptruth as WT
    dev = 'cuda:0'
    data, preps = GR.collect()
    n = 16 + 6 * k
    maze = MZ.Maze(n, dev)
    info = json.loads((preps[k]['_dir'] / 'maze.json').read_text())
    assert np.array_equal(maze.V.astype(int), np.array(info['V'])) and np.array_equal(maze.H.astype(int), np.array(info['H']))
    E = maze.E
    runs = dict(ntrl=ROOT / 'results' / f'registered_{data[k]["ntrl-demo"][SEED]["_id"]}', x_trained=ROOT / 'results' / f'registered_{data[k]["constructed"][SEED]["_id"]}')
    for key, by in own_runs(k, preps).items():
        if SEED in by:
            runs[key] = by[SEED]
    nets = {}
    for key, r in runs.items():
        d = json.loads((r / 'results.json').read_text())
        nets[key] = W.load_field(r / 'training/checkpoint.pt', dev) if d.get('stage') == 'cross-warp' else P.load_field(r / 'training/checkpoint.pt')
    warp = nets['x_trained'].warp
    for key in VARIANTS:
        if key in nets:
            a, b = warp.state_dict(), nets[key].warp.state_dict()
            assert all(torch.equal(a[q], b[q]) for q in a), f'{key} uses a different warp'
            assert bool(nets[key].warp.cfg.get('z_space')) == key.startswith('z_')
    import warpplan as WP
    m = WP.quadrant_stretch(warp, maze)
    cells = torch.tensor([[n / 2, n / 2], [n / 2, 4.0], [n - 2.5, n - 2.5]], device=dev)
    names = ['band centre', 'lower band arm', 'top-right quadrant']
    goals = maze.from_cells(cells[:, 0], cells[:, 1], torch.zeros(len(cells), device=dev))
    assert bool((maze.clearance(goals) > 0.1).all()), 'goal not free'
    BUILD.mkdir(parents=True, exist_ok=True)
    fg = TF.FastGrid(maze, BUILD, threads=3)
    tx, _ = fg.solve(goals)
    wg = WT.WarpGrid(maze, warp, BUILD, threads=3)
    tz, _ = wg.solve(goals)
    stride = 8                                      # 0.53 cells at every gap, so sinks are counted at the same resolution in cells
    ix = torch.arange(0, fg.shape[0], stride, device=dev)
    nd = len(ix)
    X, Y = torch.meshgrid(fg.xs[ix], fg.xs[ix], indexing='ij')
    nt = fg.shape[2]
    poses = torch.stack([X[..., None].expand(nd, nd, nt), Y[..., None].expand(nd, nd, nt), fg.ts[None, None].expand(nd, nd, nt)], -1).reshape(-1, 3)
    clear = maze.clearance(poses).reshape(nd, nd, nt)
    wall = (clear.amax(2) < 0)
    xc, yc = X * E + n / 2, Y * E + n / 2
    with torch.no_grad():
        uv = warp(torch.stack([X.reshape(-1), Y.reshape(-1), torch.zeros(nd * nd, device=dev)], 1))[:, :2].reshape(nd, nd, 2)
    region = torch.tensor(gap.region_of(n, xc.cpu().numpy(), yc.cpu().numpy()), device=dev)       # 0..3 quadrants, 4 band
    with torch.no_grad():                           # widths as the encoder sees them: a quadrant, the band, the whole maze
        lines = torch.tensor([0, 8, 8 + 6 * k, n, 9, 8 + 6 * k - 1], dtype=torch.float32, device=dev)
        zl = warp(torch.stack([(lines - n / 2) / E, torch.zeros_like(lines), torch.zeros_like(lines)], 1))[:, 0].cpu().numpy()
    geometry = dict(quadrant_z=float(zl[1] - zl[0]), band_z=float(zl[2] - zl[1]), maze_z=float(zl[3] - zl[0]), band_interior_z=float(zl[5] - zl[4]),
                    quadrant_time=8 / E, band_time=6 * k / E,
                    note='widths along x at y = 0 in encoder units; times in ntrl-demo units (normalized: a cell is 1 / E) to cross 8 quadrant cells and the band')
    out = dict(k=k, n=n, E=E, m=m, geometry=geometry, stride_cells=float(stride * fg.hx * E), nd=nd, boxes=info['boxes_cells'], band=6 * k,
               x=[float(v) for v in (fg.xs[ix] * E + n / 2).cpu()], runs={key: r.name.replace('registered_', '') for key, r in runs.items()},
               u=q16(uv[..., 0].cpu().numpy().ravel()), v=q16(uv[..., 1].cpu().numpy().ravel()), wall=base64.b64encode(np.packbits(wall.cpu().numpy().ravel())).decode(),
               goals=[], units=dict(x='cells', z='quadrant cells (z-units x E / m)'))
    for g, name in enumerate(names):
        fields = dict(true_x=tx[g][ix][:, ix].amin(2) * E, true_z=tz[g][ix][:, ix].amin(2) * E / m)
        for key, net in nets.items():
            gp = goals[g][None].expand(len(poses), 3)
            t = P.field_cost(net)(torch.cat([poses, gp], 1)).reshape(nd, nd, nt).amin(2)
            fields[key] = t * E / (m if key.startswith('z_') else 1.0)
        dist = torch.sqrt((xc - cells[g, 0]) ** 2 + (yc - cells[g, 1]) ** 2)
        keep = ~wall & (dist > 1.0)
        err = {}
        pairs = [('ntrl', 'true_x'), ('x_trained', 'true_x'), ('z_trained', 'true_z'), ('z_trained', 'true_x'), ('z_trained_k1data', 'true_z'),
                 ('z_trained_k1data', 'true_x'), ('x_trained_k1data', 'true_x'), ('z_trained_zdraw', 'true_z'), ('z_trained_zdraw', 'true_x'),
                 ('x_trained_zdraw', 'true_x')]
        for f, t in pairs:
            if f not in fields:
                continue
            rel = (fields[f] - fields[t]).abs() / fields[t].clamp(min=1e-6)
            err[f'{f}|{t}'] = {rn: float(rel[keep & sel].median()) for rn, sel in (('quadrants', region < 4), ('band', region == 4), ('all', torch.ones_like(keep)))}
        entry = dict(name=name, cell=[float(cells[g, 0]), float(cells[g, 1])], z=[float(v) for v in warp(goals[g][None])[0, :2]], errors=err,
                     fields={key: q16(v.cpu().numpy().ravel()) for key, v in fields.items()})
        out['goals'].append(entry)
        print(f'k={k} goal {name}: ' + '; '.join(f'{key} q {v["quadrants"]:.3f} band {v["band"]:.3f}' for key, v in err.items()), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'k{k}.json').write_text(json.dumps(out))
    print(f'k={k}: n={n}, display grid {nd}^2 (every {stride} truth nodes, {out["stride_cells"]:.3f} cells), runs {out["runs"]}, '
          f'{(OUT / f"k{k}.json").stat().st_size / 1e6:.2f} MB', flush=True)


def pool(entries):
    """Sum success_total and by_category over entries (training seeds)."""
    if not entries:
        return None
    tot = [sum(e['success_total'][i] for e in entries) for i in (0, 1)]
    cats = {}
    for e in entries:
        for c, v in (e.get('by_category') or {}).items():
            a = cats.setdefault(c, [0, 0])
            a[0] += v[0] if isinstance(v, list) else v['success']
            a[1] += v[1] if isinstance(v, list) else v['runs']
    return dict(success_total=tot, by_category=cats)


def rate(e, cat=None, seed=None):
    """Success % of a summary entry (pooled, or one training seed), all queries or one category; None if absent."""
    if not e:
        return None
    if seed is not None:
        e = (e.get('per_seed') or {}).get(str(seed))
        if not e:
            return None
    a = e['success_total'] if cat is None else e['by_category'].get(cat)
    return 100 * a[0] / a[1] if a and a[1] else None


def why_html(out):
    """The viewer's answer section, from the numbers (summary + report/fieldspaces/k<k>.json geometry)."""
    G = out['gaps']
    geo = {}
    for k in (1, 12):
        if (OUT / f'k{k}.json').exists():
            d = json.loads((OUT / f'k{k}.json').read_text())
            if 'geometry' in d:
                geo[k] = d['geometry']
    f1 = lambda v: '–' if v is None else f'{v:.1f}'
    f0 = lambda v: '–' if v is None else f'{v:.0f}'
    get = lambda k, key, pl, seed=None, cat=None: rate((G.get(str(k), {}).get(key) or {}).get(pl), cat, seed)
    paras = []
    if 1 in geo and 12 in geo:
        a, b = geo[1], geo[12]
        paras.append(
            f'<p><b>1. Is the constructed warp the same at every gap? Yes.</b> It is one design, scaled so that a quadrant cell looks the same to the encoder at every gap, '
            f'and training never changes it: its grids are created without gradients, they are not in the optimizer, and every checkpoint holds the design bit for bit '
            f'(&times;1 to &times;12; epoch 1,000 equals epoch 5,000). In the encoder&rsquo;s input a quadrant is {a["quadrant_z"]:.2f} units wide at every gap, and the band '
            f'grows only from {a["band_z"]:.2f} to {b["band_z"]:.2f} units while it grows from 6 to 72 cells (the whole maze: {a["maze_z"]:.2f} &rarr; {b["maze_z"]:.2f} units). '
            f'The warped mazes are almost the same picture.</p>')
        paras.append(
            f'<p><b>2. Did the model know it was warped? In the gap study, yes: through its loss.</b> The gap study took ntrl-demo&rsquo;s loss on the maze: its eikonal, TD and normal terms '
            f'are derivatives with respect to the pose x, so the chain rule runs through &phi; on every step, although &phi; itself never changes. The network had to output the robot&rsquo;s real '
            f'travel time, and in the warped coordinates that target changes with the gap while the input does not. Crossing a quadrant takes {a["quadrant_time"]:.2f} time units at &times;1 '
            f'and {b["quadrant_time"]:.2f} at &times;12 over the same {a["quadrant_z"]:.2f} input units; crossing the band takes {a["band_time"]:.2f} over {a["band_z"]:.2f} units at &times;1 '
            f'and {b["band_time"]:.2f} over {b["band_z"]:.2f} at &times;12. At &times;12 the quadrants are almost flat and the band is steep (the field &ldquo;trained on the maze&rdquo; at every gap, below). '
            f'That is the function the gap study&rsquo;s network failed to fit: all queries {f1(get(1, "x_trained", "x"))} % at &times;1, {f1(get(12, "x_trained", "x"))} % at &times;12 '
            f'(three training seeds; ntrl-demo {f1(get(1, "ntrl", "x"))} &rarr; {f1(get(12, "ntrl", "x"))} %).</p>')
    zk = [k for k in (1, 2, 4, 8, 12) if get(k, 'z_trained', 'z', 87358) is not None]
    if zk:
        row = ', '.join(f'&times;{k} {f1(get(k, "z_trained", "z", 87358))} %' for k in zk)
        rowx = ', '.join(f'&times;{k} {f1(get(k, "x_trained", "x", 87358))} %' for k in zk)
        paras.append(
            f'<p><b>3. The test: train and plan on the warped maze.</b> Here the network never sees x. Every training pose is mapped once to z = &phi;(x) under no_grad, so no gradient '
            f'reaches the warp or passes through its Jacobian; the clearance normals are carried into z; and ntrl-demo&rsquo;s loss is taken in z exactly as it is taken in x without a warp. '
            f'MPPI then runs on z with the same code and the same step per quadrant cell, and each path is mapped back to the maze exactly and scored there. All queries solved, training seed 87358: '
            f'{row} (the gap study&rsquo;s constructed warp, same seed, planned on the maze: {rowx}).'
            + (lambda ks3: '' if not ks3 else (' Over three training seeds (87358, 1, 2), MPPI on the warped maze: ' + '; '.join(
                f'&times;{k} {f1(rate(G[str(k)]["z_trained"]["z"]))} % against ntrl-demo {f1(rate(G[str(k)]["ntrl"]["x"]))} % and the gap study&rsquo;s constructed warp '
                f'{f1(rate(G[str(k)]["x_trained"]["x"]))} %' for k in ks3) + ' (both pooled over the same three seeds, planned on the maze).'))(
                [k for k in (1, 2, 4, 8, 12) if len(((G.get(str(k), {}).get('z_trained') or {}).get('z') or {}).get('seeds', [])) >= 3])
            + '</p>')
    pr = out.get('pairs', {})
    if '1' in pr and '12' in pr:
        paras.append(
            f'<p><b>4. What is still different: the training pairs.</b> Training on the warped maze makes the input and the loss the same at every gap, but not the data. '
            f'ntrl-demo&rsquo;s recipe draws its pairs on the maze in units of the maze&rsquo;s side: x0 within the wall margin of any wall, x1 a random distance away. As the band widens, '
            f'its outer border takes a growing share of the walls and the pairs get longer: {100 * pr["1"]["x0_in_quadrant"]:.0f} % of the starts lie in a quadrant at &times;1 and '
            f'{100 * pr["12"]["x0_in_quadrant"]:.0f} % at &times;12, and {100 * pr["1"]["same_quadrant"]:.0f} % of the pairs have both ends in one quadrant at &times;1 against '
            f'{100 * pr["12"]["same_quadrant"]:.0f} % at &times;12. The quadrants are identical in the warped space, but at &times;12 they get about a fifth of the pairs.</p>')
    c = G.get('12', {})
    CAT = [('same quadrant', 'same quadrant'), ('adjacent quadrants', 'adjacent'), ('diagonal quadrants', 'diagonal'), ('quadrant - open', 'quadrant&ndash;open'), ('open - open', 'open&ndash;open')]
    if c.get('z_trained_k1data'):
        mz, mzx = get(12, 'z_trained_k1data', 'z', 87358), get(12, 'z_trained_k1data', 'x', 87358)
        mx, mxz = get(12, 'x_trained_k1data', 'x', 87358), get(12, 'x_trained_k1data', 'z', 87358)
        paras.append(
            f'<p><b>5. The same pairs.</b> Every 22 &times; 22 training pair moved into the &times;12 maze, with the quadrants and the band&rsquo;s edge cells unchanged and only the band&rsquo;s '
            f'interior stretched, is exactly the 22 &times; 22 pair inside the quadrants in the warped space (largest difference 7&middot;10<sup>&minus;6</sup> units). Trained on these pairs in the warped space '
            f'(training seed 87358, as every control in this and the next paragraph), '
            f'&times;12 solves {f1(mz)} % with MPPI on the warped maze ({f1(mzx)} % on the maze) instead of {f1(get(12, "z_trained", "z", 87358))} %. Trained on the same pairs the gap study&rsquo;s way, '
            f'with the loss on the maze, it stays at {f1(mx)} % ({f1(mxz)} % with MPPI on the warped maze): the pairs only help once the network is no longer asked for the real travel time.</p>')
    mq = (out.get('mapped_queries') or {}).get('12', {})
    if mq.get('z_trained_k1data', {}).get('z') and G.get('1', {}).get('z_trained', {}).get('z'):
        a = mq['z_trained_k1data']['z']
        ref = (G['1']['z_trained']['z'].get('per_seed') or {}).get('87358')
        pc = lambda e, cat: 100 * e['by_category'][cat][0] / e['by_category'][cat][1]
        cats = '; '.join(f'{lab} {pc(a, key):.0f} against {pc(ref, key):.0f} %' for key, lab in CAT)
        others = []
        for key, lab in (('z_trained', 'trained on the warped maze with &times;12&rsquo;s own pairs'), ('x_trained', 'the gap study&rsquo;s constructed warp'), ('ntrl', 'ntrl-demo')):
            e = mq.get(key, {}).get('z')
            if e:
                others.append(f'{lab} {100 * e["success_total"][0] / e["success_total"][1]:.1f} %')
        paras.append(
            f'<p><b>6. The same queries.</b> The 22 &times; 22 queries carried into the &times;12 maze the same way are the same queries in the warped space wherever they stay out of the band&rsquo;s interior. '
            f'The &times;12 field trained on the warped maze with the 22 &times; 22 pairs solves {100 * a["success_total"][0] / a["success_total"][1]:.1f} % of them with MPPI on the warped maze; '
            f'the 22 &times; 22 field solved {100 * ref["success_total"][0] / ref["success_total"][1]:.1f} % of the originals. By kind: {cats}.'
            + (f' On the same carried queries: {"; ".join(others)}.' if others else '') + '</p>')
    zd = json.loads((OUT.parent / 'zdraw10.json').read_text()) if (OUT.parent / 'zdraw10.json').exists() else None
    if zd:
        import zdraw10 as ZD
        zr = lambda base, k, lab, pl, **kw: ZD.rate(zd, base, k, lab, pl, **kw)
    zp = (pr.get('12') or {}).get('zdraw')
    if zp and get(12, 'z_trained_zdraw', 'z') is not None:
        n3 = len(((G.get('12', {}).get('z_trained_zdraw') or {}).get('z') or {}).get('seeds', []))
        paras.append(
            f'<p><b>7. Pairs drawn on the warped maze.</b> ntrl-demo&rsquo;s recipe run on the warped maze instead (its draws read as coordinates of the warped box and mapped back '
            f'exactly before its wall rules apply; <code>gapprepare.py --z-draw</code>) spreads the pairs over the warped maze as it spreads them over the 22 &times; 22 maze: at &times;12, '
            f'{100 * zp["same_quadrant"]:.0f} % of the pairs have both ends in one quadrant (drawn on the maze {100 * pr["12"]["same_quadrant"]:.0f} %; the 22 &times; 22 maze '
            f'{100 * pr["1"]["same_quadrant"]:.0f} %). Trained and planned on the warped maze, &times;12 solves {f1(get(12, "z_trained_zdraw", "z"))} % of all queries'
            + (f' over {["", "one", "two", "three"][n3] if n3 < 4 else n3} training seeds' if n3 > 1 else ' (training seed 87358)') +
            f', against {f1(get(12, "z_trained", "z"))} % with the pairs drawn on the maze and ntrl-demo&rsquo;s {f1(get(12, "ntrl", "x"))} %; &times;1&rsquo;s field solves '
            f'{f1(get(1, "z_trained", "z"))} %. Given to the gap study&rsquo;s constructed warp, trained on the maze, the same pairs give {f1(get(12, "x_trained_zdraw", "x", 87358))} % '
            f'({f1(get(12, "x_trained", "x", 87358))} % with its own; training seed 87358): they help only with the loss taken in the warped space. The field is under '
            '&ldquo;trained on the warped maze, pairs drawn on the warped maze&rdquo; above.</p>')
    if mq.get('z_trained_k1data', {}).get('z') and G.get('1', {}).get('z_trained', {}).get('z') and geo.get(1, {}).get('band_interior_z') and geo.get(12, {}).get('band_interior_z'):
        a, ref = mq['z_trained_k1data']['z'], (G['1']['z_trained']['z'].get('per_seed') or {}).get('87358')
        pc = lambda e, cat: 100 * e['by_category'][cat][0] / e['by_category'][cat][1]
        bi1, bi12 = geo[1]['band_interior_z'], geo[12]['band_interior_z']
        t = (f'<p><b>So why does it fall off?</b> Not because of the warp. The warped mazes are nearly the same, no gradient ever reaches the warp, and a network given the same '
             f'thing in the warped space plans at &times;12 about as well as at &times;1 on the queries that stay in the quadrants or only cross a band arm (same quadrant {pc(a, "same quadrant"):.0f} against '
             f'{pc(ref, "same quadrant"):.0f} %, adjacent {pc(a, "adjacent quadrants"):.0f} against {pc(ref, "adjacent quadrants"):.0f} %). The gap study&rsquo;s version fell off for three reasons, '
             f'and the warp removes none of them. Its loss asked for the robot&rsquo;s real travel time ({f1(get(12, "x_trained", "z", 87358))} &rarr; {f1(get(12, "z_trained", "z", 87358))} % at &times;12 '
             'once the loss is taken in the warped space). Its training pairs are drawn on the real maze '
             + (f'({f1(get(12, "z_trained", "z", 87358))} &rarr; {f1(get(12, "z_trained_zdraw", "z", 87358))} % with the pairs drawn on the warped maze; these steps on training seed 87358). '
                if get(12, 'z_trained_zdraw', 'z', 87358) is not None else
                f'({f1(get(12, "z_trained", "z", 87358))} &rarr; {f1(get(12, "z_trained_k1data", "z", 87358))} % with the 22 &times; 22 pairs; these steps on training seed 87358). ') +
             f'And it leaves the band&rsquo;s interior at its own scale: {bi12:.2f} units wide in the warped space at &times;12 against {bi1:.2f} at &times;1, so the &times;12 warped maze is not quite '
             'the &times;1 warped maze, and the queries that cross the band&rsquo;s centre are the ones still behind')
        if zd and zr('c22', 12, 'pmzs', 'z') is not None:
            DG = 'diagonal quadrants'
            t += (f' (diagonal {f1(zr("c22", 12, "pmzw", "z", cat=DG))} % with the pairs drawn on the warped maze, against {f1(zr("c22", 1, "pmz", "z", cat=DG))} % at &times;1). '
                  f'Shrink the band&rsquo;s interior to its &times;1 width as well (density {((ZD.prepare_warp(zd, "c22_shrunk", 12) or {}).get("m_lo") or 0):.3f} instead of 1) and the two are the '
                  f'same maze in the warped space. Then &times;12 solves {f1(zr("c22", 12, "pmzs", "z"))} % (diagonal {f1(zr("c22", 12, "pmzs", "z", cat=DG))} %), as the &times;1 field does '
                  f'({f1(zr("c22", 1, "pmz", "z"))} %, diagonal {f1(zr("c22", 1, "pmz", "z", cat=DG))} %; training seed 87358)')
            if zr('c22', 12, 'pmzg', 'z') is not None and zr('c22', 12, 'pmzp', 'z') is not None:
                t += (f', whichever warp its pairs were drawn on ({f1(zr("c22", 12, "pmzg", "z"))} %), while the shrunk draw&rsquo;s pairs trained on the unshrunk warp do not get there '
                      f'({f1(zr("c22", 12, "pmzp", "z"))} %): the width decides, not the pairs')
            t += '.'
            if zr('m10', 12, 'pmzw', 'z') is not None and zr('m10', 1, 'id', 'x') is not None:
                t += (f' On the 10 &times; 10 maze&rsquo;s quadrants with the same gaps (findings page): ntrl-demo {f1(zr("m10", 1, "id", "x"))} &rarr; {f1(zr("m10", 12, "id", "x"))} % from &times;1 to &times;12, '
                      f'trained on the warped maze with the pairs drawn there {f1(zr("m10", 1, "pmzw", "z"))} &rarr; {f1(zr("m10", 12, "pmzw", "z"))} %'
                      + (f', with the band shrunk {f1(zr("m10", 12, "pmzs", "z"))} %' if zr('m10', 12, 'pmzs', 'z') is not None else '') + '.')
            t += ' These fields need the planner on the warped maze: MPPI on the maze stalls in the shrunk band.</p>'
        else:
            t += (f' (diagonal {pc(a, "diagonal quadrants"):.0f} against {pc(ref, "diagonal quadrants"):.0f} %). Shrinking the band&rsquo;s interior to its &times;1 width as well would make the two '
                  'problems identical in the warped space.</p>')
        paras.append(t)
    return ''.join(paras)


def summary():
    """report/fieldspaces/summary.json: per gap, ntrl-demo and the constructed warp trained on the maze (the gap study's
    MPPI on the maze, gapreport.collect) and, from the zq runs, both fields planned in z and the constructed warp trained
    on the warped maze planned in x and in z; per training seed and pooled."""
    import gapreport as GR
    data, preps = GR.collect()
    k_of = {str(Path(preps[k]['_dir']).resolve()): k for k in preps}
    ids = {k: {'ntrl': {e['_id']: s for s, e in data[k].get('ntrl-demo', {}).items()},
               'x_trained': {e['_id']: s for s, e in data[k].get('constructed', {}).items()}} for k in preps}
    for k in preps:
        for key, by in own_runs(k, preps).items():
            ids[k][key] = {r.name.replace('registered_', ''): s for s, r in by.items()}
    per = {}                                    # (k, field, planner) -> {seed: entry}
    for k in preps:
        for key, name in (('ntrl', 'ntrl-demo'), ('x_trained', 'constructed')):
            for s, e in data[k].get(name, {}).items():
                per.setdefault((k, key, 'x'), {})[s] = dict(success_total=e['success_total'], by_category=e['by_category'], source=e['_id'])
    for k in preps:
        for key, by in own_runs(k, preps).items():
            if not key.endswith('_zdraw'):
                continue
            for s, r in by.items():
                d = json.loads((r / 'results.json').read_text())
                per.setdefault((k, key, 'x'), {})[s] = dict(success_total=d['success_total'], by_category={c: [v['success'], v['runs']] for c, v in d['by_category'].items()},
                                                            source=r.name.replace('registered_', ''), from_='pipeline')
    checks = []
    out_mq = {}
    for p in sorted((ROOT / 'results').glob('registered_*/results.json')):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if d.get('stage') != 'zq' or d.get('smoke') or not d.get('complete'):
            continue
        k = k_of.get(str(Path(d['queries']).resolve()))
        qres = json.loads((Path(d['queries']) / 'results.json').read_text()) if (Path(d['queries']) / 'results.json').exists() else {}
        if k is None and qres.get('queries_mapped_from'):          # the 22 x 22 queries mapped into gap maze k (gapprepare --map-queries)
            kq = qres['scale']
            for rid, e in d['runs'].items():
                key = next((f for f, by in ids.get(kq, {}).items() if rid in by), None)
                if key is None:
                    continue
                for planner, v in e['plans'].items():
                    out_mq.setdefault(str(kq), {}).setdefault(key, {})[planner] = dict(success_total=v['success_total'], by_category=v['by_category'], seed=e['seed'],
                                                                                    source=p.parent.name, queries=Path(d['queries']).name)
            continue
        if k is None:
            continue
        for rid, e in d['runs'].items():
            key = next((f for f, by in ids[k].items() if rid in by), None)
            if key is None:
                continue
            for planner, v in e['plans'].items():
                if key in ('ntrl', 'x_trained') and planner == 'x':
                    old = per.get((k, key, 'x'), {}).get(e['seed'])
                    if old:
                        checks.append(dict(k=k, field=key, seed=e['seed'], gap_study=old['success_total'], zq=v['success_total'], zq_run=p.parent.name))
                    continue
                per.setdefault((k, key, planner), {})[e['seed']] = dict(success_total=v['success_total'], by_category=v['by_category'], source=p.parent.name,
                                                                         collision=v.get('collision'), stalled=v.get('stalled'),
                                                                         time_ratio_median=v.get('time_ratio_median'), length_ratio_median=v.get('length_ratio_median'))
    out = dict(gaps={}, checks=checks, pairs={}, mapped_queries=out_mq)
    for k in preps:                                 # where the training pairs lie (first 200,000 of each main prepare run)
        n = 16 + 6 * k
        src = ROOT / 'results' / (PLUS22 if k == 1 else preps[k]['_dir'].name) / 'data/sampled_points.npy'
        x = np.load(src, mmap_mode='r')[:200_000]
        c0, c1 = x[:, :2] * (n + 0.1) + n / 2, x[:, 3:5] * (n + 0.1) + n / 2
        import gap
        r0, r1 = gap.region_of(n, c0[:, 0], c0[:, 1]), gap.region_of(n, c1[:, 0], c1[:, 1])
        out['pairs'][str(k)] = dict(x0_in_quadrant=float((r0 < 4).mean()), same_quadrant=float(((r0 == r1) & (r0 < 4)).mean()),
                                    length_cells_median=float(np.median(np.linalg.norm(c0 - c1, axis=1))))
        zr = [r for r in own_runs(k, preps).get('z_trained_zdraw', {}).values()]
        if zr:                                      # the pairs drawn on the warped maze (the first such run's prepare)
            zp = ROOT / json.loads((zr[0] / 'results.json').read_text())['prepared']
            x = np.load(Path(zp) / 'data/sampled_points.npy', mmap_mode='r')[:200_000]
            c0, c1 = x[:, :2] * (n + 0.1) + n / 2, x[:, 3:5] * (n + 0.1) + n / 2
            r0, r1 = gap.region_of(n, c0[:, 0], c0[:, 1]), gap.region_of(n, c1[:, 0], c1[:, 1])
            out['pairs'][str(k)]['zdraw'] = dict(x0_in_quadrant=float((r0 < 4).mean()), same_quadrant=float(((r0 == r1) & (r0 < 4)).mean()),
                                                 length_cells_median=float(np.median(np.linalg.norm(c0 - c1, axis=1))), prepare=Path(zp).name)
    for (k, key, planner), by in sorted(per.items()):
        g = out['gaps'].setdefault(str(k), {}).setdefault(key, dict(seeds=[]))
        g[planner] = dict(pool(list(by.values())), seeds=sorted(by), per_seed={str(s): v for s, v in by.items()})
        g['seeds'] = sorted(set(g['seeds']) | set(by))
    OUT.mkdir(parents=True, exist_ok=True)
    out['why'] = why_html(out)
    (OUT / 'summary.json').write_text(json.dumps(out, indent=1))
    for k in sorted(out['gaps'], key=int):
        row = []
        for key in ('ntrl', 'x_trained', 'z_trained', 'z_trained_k1data', 'x_trained_k1data', 'z_trained_zdraw', 'x_trained_zdraw'):
            for planner in ('x', 'z'):
                e = out['gaps'][k].get(key, {}).get(planner)
                if e:
                    t = e['success_total']
                    oo = e['by_category'].get('open - open', [0, 1])
                    sq = e['by_category'].get('same quadrant', [0, 1])
                    row.append(f'{key}/{planner} {100 * t[0] / t[1]:.1f} (sq {100 * sq[0] / sq[1]:.0f}, oo {100 * oo[0] / oo[1]:.0f}; seeds {e["seeds"]})')
        print(f'x{k}: ' + ' | '.join(row))
    for kq, by in out_mq.items():
        for key, pl in by.items():
            print(f'x{kq} on the 22 x 22 queries mapped: ' + ' | '.join(f'{key}/{planner} {100 * v["success_total"][0] / v["success_total"][1]:.1f}' for planner, v in pl.items()))
    for c in checks:
        print('check', c)


def main():
    if '--summary' in sys.argv:
        summary()
        return
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if '--child' in sys.argv:
        compute(int(args[0]))
        return
    for k in [int(a) for a in args] or [1, 2, 4, 8, 12]:
        subprocess.run([sys.executable, __file__, str(k), '--child'], check=True)


if __name__ == '__main__':
    main()
