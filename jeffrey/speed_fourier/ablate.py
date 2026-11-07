"""Where does a trained field keep the walls? Inference only: report/ablate.json.

    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/speed_fourier/ablate.py --ids <run id> ...

(OMP_NUM_THREADS=4 is what the queued runs had; the table digest that mechanism.load checks is bit-exact and changes with the thread count.)

T(q, g) = 0.2 sum over groups of smoothmax |f_j(q) - f_j(g)|, with one encoder f of a single endpoint: a distance between two embeddings. Two points
are far apart only if f separates them. A speed row has the same value at the centres of the two cells on either side of a wall, so its input to f
does not differ between them. It can raise a ridge on the wall, but the difference between the two sides has to come from the other rows.

  sep      for walls between two cells more than 3 corridor steps apart: median of T(A, B) / D(A, B), A and B the two cell centres, heading along the wall,
           T in cells, D the corridor distance in cells (breadth-first search over the cell graph)
  door     median T(A, B) in cells for two neighbouring cells with no wall between them, D = 1: the same reading where nothing separates them
  frozen   the same trained network with the speed rows of both endpoints pinned to their median over cell centres and headings, i.e. the value at a
           corridor centre. What is left of rise and jump (mechanism.barrier) and of sep is what the other rows carry.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
import sys
sys.path.insert(0, str(HERE))
import mechanism as M  # noqa: E402
EM, P = M.EM, M.P


def pairs(maze):
    """Neighbouring cell pairs (cell a, cell b, along-wall heading, has wall)."""
    n = maze.n
    V, H = np.asarray(maze.V.cpu() if torch.is_tensor(maze.V) else maze.V), np.asarray(maze.H.cpu() if torch.is_tensor(maze.H) else maze.H)
    out = []
    for i in range(1, n):
        for j in range(n):
            out.append(((i - 1, j), (i, j), math.pi / 2, bool(V[i, j])))
    for i in range(n):
        for j in range(1, n):
            out.append(((i, j - 1), (i, j), 0.0, bool(H[i, j])))
    return out


def separation(maze, cost, D, min_steps=4):
    n = maze.n
    pr = pairs(maze)
    a = torch.tensor([[p[0][0] + 0.5, p[0][1] + 0.5] for p in pr], device=maze.device)
    b = torch.tensor([[p[1][0] + 0.5, p[1][1] + 0.5] for p in pr], device=maze.device)
    th = torch.tensor([p[2] for p in pr], device=maze.device)
    qa, qb = maze.from_cells(a[:, 0], a[:, 1], th), maze.from_cells(b[:, 0], b[:, 1], th)
    qa[:, 2] = qa[:, 2].clamp(-maze.phi_max, maze.phi_max)
    qb[:, 2] = qa[:, 2]
    t = (cost(torch.cat([qa, qb], 1)) / maze.pitch).cpu().numpy()
    d = np.array([D[p[0][0] * n + p[0][1], p[1][0] * n + p[1][1]] for p in pr])
    wall = np.array([p[3] for p in pr])
    use = wall & (d >= min_steps)
    return dict(pairs=int(use.sum()), steps_median=float(np.median(d[use])), sep=float(np.median(t[use] / d[use])), sep_cells=float(np.median(t[use])), door=float(np.median(t[~wall])), doors=int((~wall).sum()))


class Frozen:
    """The embedding with its last k columns pinned to c."""
    def __init__(self, phi, k, c):
        self.phi, self.k, self.c = phi, k, c

    def __call__(self, q):
        x = self.phi(q)
        return torch.cat([x[:, :-self.k], self.c.expand(len(x), -1)], 1)


def speed_columns(phi):
    return phi.table.shape[1] if phi.table is not None else phi.copies


def freeze(maze, phi):
    n, k = maze.n, speed_columns(phi)
    ij = torch.stack(torch.meshgrid(torch.arange(n, device=maze.device) + 0.5, torch.arange(n, device=maze.device) + 0.5, indexing='ij'), -1).reshape(-1, 2)
    rows = []
    for th in np.linspace(0, 2 * math.pi, 8, endpoint=False):
        q = maze.from_cells(ij[:, 0], ij[:, 1], torch.full_like(ij[:, 0], float(th)))
        q[:, 2] = q[:, 2].clamp(-maze.phi_max, maze.phi_max)
        rows.append(phi(q)[:, -k:])
    return Frozen(phi, k, torch.cat(rows).median(0).values[None])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ids', nargs='+', required=True)
    a = p.parse_args()
    out = HERE / 'report/ablate.json'
    res = json.loads(out.read_text()) if out.exists() else {}
    D = {}
    for rid in a.ids:
        if rid in res:
            continue
        maze, prepared, net, d = M.load(ROOT / f'results/registered_{rid}')
        if maze.n not in D:
            D[maze.n] = M.SF.LF.corridor_distance(maze)
        cost = P.field_cost(net)
        e = d.get('embedding') or {}
        r = dict(n=maze.n, seed=d['seed'], stage=d['stage'], form=e.get('form'), walls=e.get('walls'), scope=e.get('scope'), terms=e.get('terms'), intact=separation(maze, cost, D[maze.n]))
        r['intact'].update(M.barrier(maze, prepared, cost))
        if e.get('form') in ('top', 'oracle', 'mix', 'geo'):
            phi = net.input_mapping.__func__.__closure__[0].cell_contents
            EM.install(net, freeze(maze, phi))
            r['frozen'] = separation(maze, cost, D[maze.n])
            r['frozen'].update(M.barrier(maze, prepared, cost))
        res[rid] = r
        out.write_text(json.dumps(res, indent=1) + '\n')
        show = lambda x: {k: round(x[k], 3) for k in ('sep', 'door', 'leak', 'rise_median', 'jump_median')}
        print(rid, maze.n, e.get('form'), e.get('walls'), show(r['intact']), show(r['frozen']) if 'frozen' in r else '', flush=True)
        del net
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
