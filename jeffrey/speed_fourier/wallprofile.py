"""T across a wall, and how much the rows themselves differ across it. Inference only: report/profile.json.

    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/speed_fourier/wallprofile.py

  profile   median over (interior wall, prepared goal) with a true gap of at least 2 cells of T(u) - T(far side), u from -0.4 (high side) to 0.4 (low side)
            on the perpendicular through the wall's midpoint, heading along the wall, in cells, as in mechanism.barrier; for the truth and each field
  rowdiff   mean over rows of the median over walls with at least 2 corridor steps between the cells of |row(A) - row(B)| / RMS of the row, at the two cell centres
            (heading along the wall): do the rows tell the two sides of a wall apart? 0 = same value on both sides.
  rowtrack  Spearman over all interior walls of |rows(A) - rows(B)| (Euclidean over the speed rows, both cell centres) against the corridor steps from A to B: does
            the difference grow with how far apart the two sides really are? A fit's ripple can differ across a wall (rowdiff above zero) and still say nothing.
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import report as R  # noqa: E402 (first: jeffrey/grid_sync and maze_scale have a report.py too, and ablate puts them on the path)
import ablate as A  # noqa: E402
M = A.M
ROOT = A.ROOT
ARMS = ['octave', 'topzero', 'top line M30 true', 'top line M30 flip', 'oracle true', 'oracle flip', 'geo true', 'geo flip', 'mix ridge M30 true', 'mix line M30 true',
        'top integral M30 true slow', 'top integral M30 flip slow', 'top exact-integral true slow', 'top exact-integral flip slow', 'top both M30 true slow', 'top both M30 flip slow',
        'mix iterms M31 true slow', 'mix iterms M31 flip slow', 'top line M30 true slow', 'top share M30 true slow', 'top share M30 flip slow', 'half']


def wall_profile(maze, prepared, cost, goals_k=8, min_gap=2.0):
    dev = maze.device
    seg = [s for s in M.GM.segments(maze) if s[3]]
    u = torch.linspace(-0.4, 0.4, 17, device=dev)
    cx, cy = torch.tensor([s[0] for s in seg], device=dev), torch.tensor([s[1] for s in seg], device=dev)
    vert = torch.tensor([s[2] == 0.0 for s in seg], device=dev)[:, None]
    x = torch.where(vert, cx[:, None] + u[None], cx[:, None].expand(-1, len(u)))
    y = torch.where(vert, cy[:, None].expand(-1, len(u)), cy[:, None] + u[None])
    th = torch.where(vert, torch.full_like(x, np.pi / 2), torch.zeros_like(x))
    q = maze.from_cells(x.reshape(-1), y.reshape(-1), th.reshape(-1))
    q[:, 2] = q[:, 2].clamp(-maze.phi_max, maze.phi_max)
    goals = torch.tensor(np.load(prepared / 'goals.npy')[:goals_k], device=dev)
    fields = np.load(prepared / 'truth_fields.npy', mmap_mode='r')
    nx, _, nt = fields.shape[1:]
    ix = ((q[:, 0] + 0.5) * (nx - 1)).round().long().cpu().numpy()
    iy = ((q[:, 1] + 0.5) * (nx - 1)).round().long().cpu().numpy()
    it = ((q[:, 2] / maze.phi_max + 1) / 2 * (nt - 1)).round().long().cpu().numpy()
    pf, pt = [], []
    for k, g in enumerate(goals):
        tt = torch.tensor(np.array(fields[k][ix, iy, it], np.float32), device=dev).reshape(len(seg), len(u)) / maze.pitch
        tf = cost(torch.cat([q, g[None].expand(len(q), 3)], 1)).reshape(len(seg), len(u)) / maze.pitch
        flip = tt[:, 0] < tt[:, -1]
        tt = torch.where(flip[:, None], tt.flip(1), tt)
        tf = torch.where(flip[:, None], tf.flip(1), tf)
        use = (tt[:, 0] - tt[:, -1]) >= min_gap
        pf.append((tf - tf[:, -1:])[use])
        pt.append((tt - tt[:, -1:])[use])
    return u.cpu().tolist(), torch.cat(pf).median(0).values.cpu().tolist(), torch.cat(pt).median(0).values.cpu().tolist()


def row_difference(maze, phi, D, min_steps=2):
    n, k = maze.n, A.speed_columns(phi)
    pr = [p for p in A.pairs(maze) if p[3] and D[p[0][0] * n + p[0][1], p[1][0] * n + p[1][1]] >= min_steps]
    a = torch.tensor([[p[0][0] + 0.5, p[0][1] + 0.5] for p in pr], device=maze.device)
    b = torch.tensor([[p[1][0] + 0.5, p[1][1] + 0.5] for p in pr], device=maze.device)
    th = torch.tensor([p[2] for p in pr], device=maze.device)
    qa, qb = maze.from_cells(a[:, 0], a[:, 1], th), maze.from_cells(b[:, 0], b[:, 1], th)
    qa[:, 2] = qa[:, 2].clamp(-maze.phi_max, maze.phi_max)
    qb[:, 2] = qa[:, 2]
    d = (phi(qa)[:, -k:] - phi(qb)[:, -k:]).abs()
    steps = np.array([D[p[0][0] * n + p[0][1], p[1][0] * n + p[1][1]] for p in pr])
    return float((d.median(0).values / M.EM.RMS).mean()), float(spearmanr(torch.linalg.vector_norm(d, dim=1).cpu().numpy(), steps)[0])


def main():
    t = R.collect()
    out = HERE / 'report/profile.json'
    res = json.loads(out.read_text()) if out.exists() else {}
    D = {}
    for n in (22, 13):
        for arm in ARMS:
            for seed, r in t[n].get(arm, {}).items():
                if seed not in R.SEEDS or 'rowtrack' in res.get(r['id'], {}) or (r['id'] in res and 'rowdiff' not in res[r['id']]):
                    continue
                maze, prepared, net, d = M.load(ROOT / f'results/registered_{r["id"]}')
                if n not in D:
                    D[n] = M.SF.LF.corridor_distance(maze)
                cost = M.P.field_cost(net)
                u, pf, pt = wall_profile(maze, prepared, cost)
                e = dict(n=n, arm=arm, seed=seed, u=u, field=pf, truth=pt)
                if d['stage'] == 'speedfourier' and (d['embedding']['form'] in ('top', 'oracle', 'geo', 'mix')):
                    phi = net.input_mapping.__func__.__closure__[0].cell_contents
                    e['rowdiff'], e['rowtrack'] = row_difference(maze, phi, D[n])
                    EM = M.EM
                    EM.install(net, A.freeze(maze, phi))
                    e['frozen'] = wall_profile(maze, prepared, cost)[1]
                res[r['id']] = e
                out.write_text(json.dumps(res) + '\n')
                print(n, arm, seed, 'rowdiff', round(e.get('rowdiff', float('nan')), 3), 'track', round(e.get('rowtrack', float('nan')), 3), 'peak', round(max(pf), 2), 'far', round(pf[0], 2), flush=True)
                del net
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
