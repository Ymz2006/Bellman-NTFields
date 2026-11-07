"""Near-wall attraction and open-area speckle of every trained plus-maze field (inference only): report/attract.json.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ~/.venvs/bellman-ntfield/bin/python jeffrey/cross_warp/attract.py

Per field (registered plus-ntrl runs and cross-warp runs on the plus prepare run; cached by run id):
* attract   diag.near_wall over the held-out pairs whose x0 is near a wall (trained speed S < 0.3): share of poses whose
            descent direction -grad T points into the wall, median S |grad T| (target 1) and cos(grad T, -n). At the
            first wall contact of failed MPPI runs the descent points into the wall in 88-100 % of cases, so this is
            the field defect behind the collisions.
* speckle   learned speed 1 / |grad_x T(x, goal)| over 20,000 poses of the open + (free radius >= 1 cell, true speed 1)
            towards 8 prepared goals: median, and share below 0.7 (spurious slow patches in open space).
"""
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
sys.path.insert(0, str(HERE))
PREP = ROOT / 'results/registered_20261102T042454Z-62864e61'
REG = Path(os.environ.get('NTFIELDS_REGISTRY', Path.home() / '.local/share/ntfields-experiments')) / 'runs'


def main():
    import plus as PL
    PL.install(6)
    import maze as MZ
    import plan as P
    import warp as W
    import diag as DG
    dev = 'cuda:0'
    maze = MZ.Maze(22, dev)
    out_path = HERE / 'report/attract.json'
    C = json.loads(out_path.read_text()) if out_path.exists() else {}
    gen = torch.Generator(dev).manual_seed(3)
    u = maze.uniform(400_000, gen)
    openq = u[W.free_radius(maze, u) >= 1.0][:20_000]
    goals = torch.tensor(np.load(PREP / 'goals.npy')[:8], device=dev)
    for r in sorted(ROOT.glob('results/registered_*/results.json')):
        rid = r.parent.name.replace('registered_', '')
        if rid in C:
            continue
        d = json.loads(r.read_text())
        if d.get('stage') not in ('plus-ntrl', 'cross-warp') or d.get('smoke') or not d.get('complete'):
            continue
        if Path(d['prepared']).name != PREP.name:
            continue
        lab = 'ntrl-demo' if d['stage'] == 'plus-ntrl' else json.loads((REG / rid / 'spec.json').read_text())['config']['label']
        ck = r.parent / 'training/checkpoint.pt'
        net = W.load_field(ck, dev) if d['stage'] == 'cross-warp' else P.load_field(ck)
        grad = lambda x: P.field_grad(net, x)
        nw = DG.near_wall(maze, PREP, grad)
        S = []
        for g in goals:
            _, gr = grad(torch.cat([openq, g[None].expand(len(openq), 3)], 1))
            S.append(1 / torch.linalg.vector_norm(gr, dim=1))
        S = torch.stack(S)
        C[rid] = dict(label=lab, seed=d['seed'], attract=nw['all']['attract'], slope=nw['all']['slope'], cos=nw['all']['cos'],
                      open_speed_median=float(S.median()), open_slow_share=float((S < 0.7).float().mean()))
        out_path.write_text(json.dumps(C, indent=1))
        print(lab, d['seed'], C[rid], flush=True)
        del net
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
