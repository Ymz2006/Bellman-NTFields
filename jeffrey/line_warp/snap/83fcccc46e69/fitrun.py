"""One registered fit (fit.py) with its measurements (checks.py) and pictures (view.py)."""
import json
import sys
import time
from pathlib import Path
import torch

import fit as F
import checks as CK
import netwarp as NW
import view as V


def main():
    p = F.arguments()
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if a.smoke:
        a.steps = 30
    a.out.mkdir(parents=True, exist_ok=True)
    from common import Logger, dump
    log = Logger(a.out / 'log.txt')
    dev = 'cuda:0'
    t0 = time.time()
    maze, info = F.load_maze(a.prepared, dev)
    phi, hist, _ = F.fit(a, maze, log)
    if a.window > 0:
        phi.window(phi.groups)
    if a.arch != 'resnet':
        phi.freeze(int(round(a.dilate * a.nodes_per_cell)))
    torch.save(dict(state=phi.state_dict(), cfg=phi.cfg, args={k: str(v) for k, v in vars(a).items()}), a.out / 'phi.pt')
    json.dump(hist, open(a.out / 'history.json', 'w'))
    w = NW.NetWarp(dict(net=phi.cfg, s=1.0)).to(dev)
    w.set_state(phi.state_dict(), maze)
    ident = NW.NetWarp(dict(net=phi.cfg, s=1.0))
    res = dict(stage='line-warp-fit', maze=maze.describe(), args={k: str(v) for k, v in vars(a).items()}, cfg=phi.cfg, gpu=torch.cuda.get_device_name(0), smoke=a.smoke, final=hist[-1], fit_seconds=time.time() - t0)
    res['identity_runs'] = CK.run_stats(lambda x: x, maze, a.mode, a.headings, floor=a.floor)
    res['warp'] = CK.measure(w, maze)
    moved = res['warp']['sv_min_max'][1] / res['warp']['sv_min_max'][0]
    if moved < 1.5 and not a.smoke:      # a map that stayed at the identity must not reach a field run as if it were a learned warp
        raise SystemExit(f'fit did not move: largest / smallest singular value {moved:.3f}')
    log(f"identity runs {res['identity_runs']}")
    log(f"warp {res['warp']}")
    with torch.no_grad():
        V.overview(lambda x: w.plane(x), maze, a.out / 'overview.png', 0.0, a.kind)
    res['complete'] = True
    res['elapsed_seconds'] = time.time() - t0
    dump(a.out / 'results.json', res)


if __name__ == '__main__':
    main()
