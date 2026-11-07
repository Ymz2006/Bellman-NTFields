"""Mechanism diagnostics of the grid-sync fields (inference only): report/mechanism.md, mechanism.json, profiles.json.

    CUDA_VISIBLE_DEVICES=0 ~/.venvs/bellman-ntfield/bin/python jeffrey/grid_sync/mechanism.py [--seeds 87358]

Per field (ntrl-demo, the earlier B x s runs, every grid-sync mode; training seed 87358 unless --seeds):

1. Grid-line profiles. Learned speed S = 1 / |grad_q0 T(q0, g)| (ntrl-demo's speed, as evaluate.speed_fit), median
   over 8 prepared goals, on the perpendicular through the midpoint of every interior grid-line segment, at offsets
   u = -0.5 .. 0.5 cells, robot heading along the crossing (theta = 0 across vertical lines, pi/2 across horizontal
   ones). Doorways (segments without a wall) are free along the whole profile (trained speed 1 everywhere, checked
   with the exact clearance), so a dip there is a phantom wall: the 'doorway dip' is median S at u = 0 over median S
   at |u| = 0.5 (1 = no phantom wall). Wall segments: S at |u| = 0.35 (inside the slow band, truth = the trained
   label there) and at 0.5.
2. Near-wall slope and direction (maze_scale/wallfeel.measure on the held-out pairs): median S |grad T| (target 1)
   and cos(grad T, -n) (target 1) at near-wall x0, by pair gap.
3. Local minima ('bumps') of T(x, goal) along the true paths of queries >= 8 cells (paths densified to 0.02 cells),
   as in the maze_scale scaled-DP check: an MPPI run can stall in each.
"""
import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
_spec = importlib.util.spec_from_file_location('grid_sync_report', HERE / 'report.py')   # maze_scale has its own report.py
GR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GR)
import maze as MZ  # noqa: E402
import plan as P  # noqa: E402
from wallfeel import measure as wall_measure  # noqa: E402

OFFSETS = np.round(np.linspace(-0.5, 0.5, 21), 3)


def segments(maze):
    """Interior grid-line segments: (centre x, y in cells, crossing heading, has wall)."""
    out = []
    n = maze.n
    for i in range(1, n):
        for j in range(n):
            out.append((float(i), j + 0.5, 0.0, bool(maze.V[i, j])))
    for i in range(n):
        for j in range(1, n):
            out.append((i + 0.5, float(j), math.pi / 2, bool(maze.H[i, j])))
    return out


def profiles(maze, prepared, grad, goals_k=8):
    dev = maze.device
    goals = torch.tensor(np.load(prepared / 'goals.npy')[:goals_k], device=dev)
    seg = segments(maze)
    cx = torch.tensor([s[0] for s in seg], device=dev)
    cy = torch.tensor([s[1] for s in seg], device=dev)
    th = torch.tensor([s[2] for s in seg], device=dev)
    wall = torch.tensor([s[3] for s in seg], device=dev)
    u = torch.tensor(OFFSETS, device=dev, dtype=torch.float32)
    vert = (th == 0)[:, None]
    x = torch.where(vert, cx[:, None] + u[None], cx[:, None].expand(-1, len(u)))
    y = torch.where(vert, cy[:, None].expand(-1, len(u)), cy[:, None] + u[None])
    # theta = pi/2 is the box edge (phi = phi_max): use the edge itself, inside the C-space box
    q = maze.from_cells(x.reshape(-1), y.reshape(-1), th[:, None].expand(-1, len(u)).reshape(-1))
    q[:, 2] = q[:, 2].clamp(-maze.phi_max, maze.phi_max)
    clear = maze.clearance(q).reshape(len(seg), len(u))
    label = maze.train_speed(clear).reshape(len(seg), len(u))
    S = []
    for g in goals:
        pairs = torch.cat([q, g[None].expand(len(q), 3)], 1)
        _, gr = grad(pairs)
        S.append(1 / torch.linalg.vector_norm(gr, dim=1).clamp(min=1e-6))
    S = torch.stack(S).median(0).values.reshape(len(seg), len(u))
    door = ~wall
    free_door = door & (label > 0.999).all(1)
    mid = len(u) // 2
    edge = [0, len(u) - 1]
    k35 = [int(np.argmin(np.abs(OFFSETS + 0.35))), int(np.argmin(np.abs(OFFSETS - 0.35)))]
    res = dict(doorways=int(door.sum()), doorways_free=int(free_door.sum()), walls=int(wall.sum()))
    sd = S[free_door]
    dip = sd[:, mid] / sd[:, edge].mean(1)
    res['doorway_dip_median'] = float(dip.median())
    res['doorway_dip_q10'] = float(dip.quantile(0.1))
    res['doorway_phantom_share'] = float((dip < 0.5).float().mean())     # doorways the field slows to half speed or less
    sw = S[wall]
    res['wall_S_at_035'] = float(sw[:, k35].median())
    res['wall_label_at_035'] = float(label[wall][:, k35].median())
    res['wall_S_at_05'] = float(sw[:, edge].median())
    res['profile_doorway_median'] = sd.median(0).values.tolist()
    res['profile_wall_median'] = sw.median(0).values.tolist()
    res['profile_wall_label'] = label[wall].median(0).values.tolist()
    return res


def bumps(maze, prepared, cost, min_bin=3):
    q = np.load(prepared / 'queries.npz')
    tp = np.load(prepared / 'truth_paths.npy', allow_pickle=True)
    counts = []
    for k in np.nonzero(q['bin'] >= min_bin)[0]:
        p = P.densify(torch.tensor(np.asarray(tp[k], np.float32), device=maze.device), 0.02 * maze.pitch)
        g = torch.tensor(q['goal'][k], device=maze.device, dtype=torch.float32).expand(len(p), 3)
        t = cost(torch.cat([p, g], 1)).cpu().numpy()
        dt = np.diff(t)
        keep = np.abs(dt) > 1e-6 * t[0]
        sgn = np.sign(dt[keep])
        counts.append(int(((sgn[:-1] < 0) & (sgn[1:] > 0)).sum()))
    return dict(paths=len(counts), bumps_per_path=float(np.mean(counts)) if counts else None, paths_with_bump=int(np.sum(np.array(counts) > 0)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', default='87358')
    p.add_argument('--fields', default='', help='comma-separated field keys to analyse (default: all)')
    a = p.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    keep = set(a.fields.split(',')) if a.fields else None
    out_json = GR.OUT / 'mechanism.json'
    res = json.loads(out_json.read_text()) if out_json.exists() else {}
    for d in GR.runs():
        n, key = d['maze']['n'], GR.field_key(d)
        if d['seed'] not in seeds or n not in GR.SIZES or (keep and key not in keep):
            continue
        tag = f'{n}|{key}|{d["seed"]}|{d["_id"]}'
        if tag in res:
            continue
        maze = MZ.Maze(n)
        prepared = Path(d['prepared'])
        net = P.load_field(d['_dir'] / 'training/checkpoint.pt')
        cost, grad = P.field_cost(net), (lambda pairs, net=net: P.field_grad(net, pairs))
        r = dict(n=n, field=key, seed=d['seed'], id=d['_id'])
        r['profiles'] = profiles(maze, prepared, grad)
        r['wall'] = wall_measure(maze, prepared, grad)
        r['bumps'] = bumps(maze, prepared, cost)
        res[tag] = r
        print(tag, 'doorway dip', round(r['profiles']['doorway_dip_median'], 3), 'phantom share', round(r['profiles']['doorway_phantom_share'], 3),
              'wall S@0.35', round(r['profiles']['wall_S_at_035'], 3), '(label', round(r['profiles']['wall_label_at_035'], 3), ') bumps/path',
              r['bumps']['bumps_per_path'], 'wall', {k: (round(v['slope'], 2), round(v['cos'], 2)) for k, v in r['wall'].items()}, flush=True)
        out_json.write_text(json.dumps(res, indent=1))
        del net
        torch.cuda.empty_cache()
    # markdown
    order = ['ntrl-demo', 'B × 1.5', 'B × 3', 'B × 6'] + GR.MODES
    md = ['# Mechanism diagnostics (generated by mechanism.py)', '',
          'Training seed(s) ' + ', '.join(map(str, seeds)) + '. Doorway dip: learned speed at the middle of a free doorway over the speed half a cell either side '
          '(1 = no phantom wall; truth 1). Phantom share: doorways slowed to half speed or less. Wall S@0.35: learned speed 0.35 cells from a wall line '
          '(truth: the trained label, given). Bumps: local minima of T(x, goal) along true paths of queries >= 8 cells. Near-wall slope / cos at '
          'gap 2-8 cells (targets 1 / 1).', '',
          '| maze | field | doorway dip (median, q10) | phantom share | wall S@0.35 (label) | bumps per long path | near-wall slope / cos (gap 2-8) |', '|---|---|---|---|---|---|---|']
    for tag, r in sorted(res.items(), key=lambda kv: (kv[1]['n'], order.index(kv[1]['field']) if kv[1]['field'] in order else 99, kv[1]['seed'])):
        if r['seed'] not in seeds:
            continue
        pr, w = r['profiles'], r['wall'].get('2-8')
        md.append(f'| {r["n"]}×{r["n"]} | {GR.MODE_LABEL.get(r["field"], r["field"])} (seed {r["seed"]}) | {pr["doorway_dip_median"]:.2f} ({pr["doorway_dip_q10"]:.2f}) | '
                  f'{100 * pr["doorway_phantom_share"]:.0f} % | {pr["wall_S_at_035"]:.3f} ({pr["wall_label_at_035"]:.3f}) | {r["bumps"]["bumps_per_path"]:.2f} | '
                  + (f'{w["slope"]:.2f} / {w["cos"]:.2f}' if w else '–') + ' |')
    (GR.OUT / 'mechanism.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
