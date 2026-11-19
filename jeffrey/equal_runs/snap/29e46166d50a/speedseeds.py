"""How much of a difference in the learnability score is noise? The key maps of a scene scored with several regression seeds: same map, same budget, different network seed.

    speedseeds.py --scene <spec> --res 16 --runs label=<run id>,... [--old label=<line_warp fit id>,...] --out <dir>
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

import maps as M
import remeasure as RM
import scene as SC
import speedfit as SF


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--res', type=int, default=32)
    p.add_argument('--runs', default='')
    p.add_argument('--old', default='')
    p.add_argument('--seeds', default='0,1,2,3,4')
    p.add_argument('--s', type=float, default=4.0)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = open(out / 'log.txt', 'a', buffering=1)
    dev = 'cuda:0'
    scene = SC.build(a.scene, dev, 9, a.res)
    maze = scene.maze
    ident = M.Identity()
    jobs = [('identity', SF.of_psi(ident), None, ident)]
    for q in filter(None, a.runs.split(',')):
        label, rid = q.split('=')
        m = RM.load(rid, dev)
        jobs.append((label, SF.of_psi(m), rid, m.inverse))
    if a.old:
        sys.path.insert(0, str(SC.ROOT / 'jeffrey/line_warp'))
        import netwarp as NW
        import speedcal as CAL
        for q in a.old.split(','):
            label, rid = q.split('=')
            state, cfg, _ = NW.load_fit(SC.ROOT / 'results' / f'registered_{rid}', dev)
            w = NW.NetWarp(dict(net=cfg, s=1.0)).to(dev)
            w.set_state(state, maze)
            jobs.append((label, CAL.of_netwarp(w), rid, w.plane))
    res = dict(scene=a.scene, s=a.s, rows={})
    test = SF.wall_band(maze)
    for label, fn, rid, phi in jobs:
        v, wall = [], []
        for k in a.seeds.split(','):
            nets = []
            v.append(SF.fit(fn, maze, a.s, a.steps, seed=int(k), keep=nets)['r2'])
            wall.append(SF.wall_score(nets[0], phi, maze, test))
        res['rows'][label] = dict(run=rid, r2=v, mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), wall_r2=[w['r2'] for w in wall], wall_mean=float(np.mean([w['r2'] for w in wall])), wall_sd=float(np.std([w['r2'] for w in wall], ddof=1)), wall_mae=float(np.mean([w['mae'] for w in wall])), **{k: float(np.mean([w[k] for w in wall])) for k in wall[0] if k.startswith(('r2_', 'mae_', 'share_'))})
        f.write(f"{label}: r2 {np.mean(v):.4f} sd {np.std(v, ddof=1):.4f} | near walls r2 {res['rows'][label]['wall_mean']:.4f} sd {res['rows'][label]['wall_sd']:.4f} mae {res['rows'][label]['wall_mae']:.4f} | { {k: round(v, 3) for k, v in res['rows'][label].items() if k.startswith(('r2_', 'share_'))} }\n")
        (out / 'results.json').write_text(json.dumps(res, indent=1))
    res['complete'] = True
    (out / 'results.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
