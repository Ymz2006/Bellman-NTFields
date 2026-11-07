"""What the speed rows change in a trained field (inference only): report/mechanism.json, report/mechanism.md.

    CUDA_VISIBLE_DEVICES=0 ~/.venvs/bellman-ntfield/bin/python jeffrey/speed_fourier/mechanism.py --ids <run id> ...

Per run, from jeffrey/grid_sync/mechanism.py: doorway dip and phantom share, near-wall slope and direction (maze_scale/wallfeel), local minima
of T along true paths. New here, the separation of the size of grad T from its direction:

  magnitude    held-out log-speed correlation (evaluate.speed_fit: 1 / |grad T| against the label), goal-independent
  direction    cosine between -grad_q T(q, goal) and the tangent of the true path, at points every 0.1 cells along the true paths of queries of
               4 cells or more, xy components only; median and share of points with a negative cosine, over all points and over the points
               with trained speed below 0.9 (true paths stay on the corridor centre lines, so they rarely enter the slow band). Goal-dependent.
  barrier      does T have a ridge on the walls? For every interior wall segment and 8 prepared goals with a true difference of at least 2 cells
               between the two sides, T(q, goal) on the perpendicular through the segment's midpoint, u = -0.4 .. 0.4 cells, heading along the
               wall. Truth: T* rises from the high side into the wall before falling to the low side, because crossing costs time; a
               field without that rise slides the planner from the high side through the wall. leak = share of (segment, goal) with no
               rise above 0.05 cells on the high side; rise = median rise in cells, for the field and the truth; jump = median of
               (T(high) - T(low)) / (T*(high) - T*(low)).
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
sys.path.insert(0, str(ROOT / 'jeffrey/grid_sync'))
_spec = importlib.util.spec_from_file_location('grid_sync_mechanism', ROOT / 'jeffrey/grid_sync/mechanism.py')
GM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GM)
import embed as EM  # noqa: E402
import speedfourier as SF  # noqa: E402
import maze as MZ  # noqa: E402
import plan as P  # noqa: E402
from wallfeel import measure as wall_measure  # noqa: E402


def direction(maze, prepared, grad, min_bin=2, spacing=0.1, per_path=400):
    q = np.load(prepared / 'queries.npz')
    tp = np.load(prepared / 'truth_paths.npy', allow_pickle=True)
    cos, speed = [], []
    for k in np.nonzero(q['bin'] >= min_bin)[0]:
        p = P.densify(torch.tensor(np.asarray(tp[k], np.float32), device=maze.device), spacing * maze.pitch)
        if len(p) < 3:
            continue
        p = p[:: max(1, len(p) // per_path)]
        tangent = torch.nn.functional.normalize((p[1:] - p[:-1])[:, :2], dim=1)
        x = p[:-1]
        goal = torch.tensor(q['goal'][k], device=maze.device, dtype=torch.float32).expand(len(x), 3)
        _, g = grad(torch.cat([x, goal], 1))
        cos.append(torch.nn.functional.cosine_similarity(-g[:, :2], tangent, dim=1))
        speed.append(maze.train_speed(maze.clearance(x)))
    cos, speed = torch.cat(cos), torch.cat(speed)
    near = speed < 0.9
    return dict(points=int(len(cos)), near_points=int(near.sum()), cos_median=float(cos.median()), cos_negative=float((cos < 0).float().mean()),
                near_cos_median=float(cos[near].median()), near_cos_negative=float((cos[near] < 0).float().mean()))


def barrier(maze, prepared, cost, goals_k=8, min_gap=2.0, leak_rise=0.05):
    dev = maze.device
    seg = [s for s in GM.segments(maze) if s[3]]
    u = torch.linspace(-0.4, 0.4, 17, device=dev)
    cx = torch.tensor([s[0] for s in seg], device=dev)
    cy = torch.tensor([s[1] for s in seg], device=dev)
    vert = torch.tensor([s[2] == 0.0 for s in seg], device=dev)[:, None]
    x = torch.where(vert, cx[:, None] + u[None], cx[:, None].expand(-1, len(u)))
    y = torch.where(vert, cy[:, None].expand(-1, len(u)), cy[:, None] + u[None])
    th = torch.where(vert, torch.full_like(x, np.pi / 2), torch.zeros_like(x))     # heading along the wall
    q = maze.from_cells(x.reshape(-1), y.reshape(-1), th.reshape(-1))
    q[:, 2] = q[:, 2].clamp(-maze.phi_max, maze.phi_max)
    goals = torch.tensor(np.load(prepared / 'goals.npy')[:goals_k], device=dev)
    fields = np.load(prepared / 'truth_fields.npy', mmap_mode='r')
    nx, _, nt = fields.shape[1:]
    ix = ((q[:, 0] + 0.5) * (nx - 1)).round().long().cpu().numpy()
    iy = ((q[:, 1] + 0.5) * (nx - 1)).round().long().cpu().numpy()
    it = ((q[:, 2] / maze.phi_max + 1) / 2 * (nt - 1)).round().long().cpu().numpy()
    rise_f, rise_t, jump = [], [], []
    for k, g in enumerate(goals):
        tt = torch.tensor(np.array(fields[k][ix, iy, it], np.float32), device=dev).reshape(len(seg), len(u)) / maze.pitch
        tf = cost(torch.cat([q, g[None].expand(len(q), 3)], 1)).reshape(len(seg), len(u)) / maze.pitch
        flip = tt[:, 0] < tt[:, -1]                                                # profile starts on the high side
        tt = torch.where(flip[:, None], tt.flip(1), tt)
        tf = torch.where(flip[:, None], tf.flip(1), tf)
        use = (tt[:, 0] - tt[:, -1]) >= min_gap
        rise_f.append((tf.amax(1) - tf[:, 0])[use])
        rise_t.append((tt.amax(1) - tt[:, 0])[use])
        jump.append(((tf[:, 0] - tf[:, -1]) / (tt[:, 0] - tt[:, -1]))[use])
    rise_f, rise_t, jump = torch.cat(rise_f), torch.cat(rise_t), torch.cat(jump)
    return dict(pairs=int(len(rise_f)), leak=float((rise_f < leak_rise).float().mean()), truth_leak=float((rise_t < leak_rise).float().mean()), rise_median=float(rise_f.median()),
                truth_rise_median=float(rise_t.median()), jump_median=float(jump.median()), jump_wrong_sign=float((jump < 0).float().mean()))


def load(run):
    """(maze, prepared, net, results) of a registered ntrl / gridsync / speedfourier run."""
    d = json.load(open(run / 'results.json'))
    maze = MZ.Maze(d['maze']['n'])
    net, weight = SF.load_field(run / 'training/checkpoint.pt')
    if d['stage'] == 'speedfourier':
        e = d['embedding']
        phi, info, _ = SF.build(maze, d['seed'], e['form'], e['scope'], e['terms'], e['walls'], None, None, e['rows_from'], target=e.get('target', 'obstacle'), reach=e.get('reach'), norm=e.get('norm'))
        assert info.get('table_digest') == e.get('table_digest'), 'rebuilt embedding differs from the run'
        EM.install(net, phi)
        if e.get('head', 'none') != 'none':
            EM.install_head(net, phi, phi.raw, e['head'], e['head_scale'], weight=weight)
    return maze, Path(d['prepared']), net, d


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ids', nargs='+', required=True)
    a = p.parse_args()
    out = HERE / 'report/mechanism.json'
    res = json.loads(out.read_text()) if out.exists() else {}
    for rid in a.ids:
        if rid in res:
            continue
        maze, prepared, net, d = load(ROOT / f'results/registered_{rid}')
        cost, grad = P.field_cost(net), (lambda pairs, net=net: P.field_grad(net, pairs))
        r = dict(n=maze.n, seed=d['seed'], stage=d['stage'], embedding=d.get('embedding'), mode=d.get('mode'), success=d['success_total'], speed_fit=d['speed_fit'],
                 collisions=sum(v['collision'] for v in d['planning'].values()), descent_cos=d['field']['descent_cos_median'])
        r['direction'] = direction(maze, prepared, grad)
        r['barrier'] = barrier(maze, prepared, cost)
        r['profiles'] = {k: v for k, v in GM.profiles(maze, prepared, grad).items() if not k.startswith('profile_')}
        r['wall'] = wall_measure(maze, prepared, grad)
        r['bumps'] = GM.bumps(maze, prepared, cost)
        res[rid] = r
        out.write_text(json.dumps(res, indent=1) + '\n')
        print(rid, {k: round(v, 3) for k, v in r['direction'].items() if 'cos' in k}, {k: round(v, 3) for k, v in r['barrier'].items()}, 'phantom', round(r['profiles']['doorway_phantom_share'], 3), 'bumps', round(r['bumps']['bumps_per_path'], 2), flush=True)
        del net
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
