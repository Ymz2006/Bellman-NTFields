"""Does the learnability score track planning? Scores line_warp's fitted maps, whose planning results are known, with speedfit.py. Read-only on those runs.

    speedcal.py --maze plus:22:6 --fits label=<fit run id>,... --out <dir>     (label 'identity' needs no run)

speedfit.py draws z uniformly in the box. These maps have an inverse by Newton (NetWarp.inverse_plane); z outside the image of the box is rejected.
"""
import argparse
import json
import sys
from pathlib import Path
import torch

import scene as SC
import speedfit as SF
import maps as M

sys.path.insert(0, str(SC.ROOT / 'jeffrey/line_warp'))
import netwarp as NW  # noqa: E402


def of_netwarp(w):
    w.inverse_table()

    @torch.no_grad()
    def f(z):
        xs, oks = [], []
        for i in range(0, len(z), 262144):
            c = z[i:i + 262144]
            x = w.inverse_plane(c)
            xs.append(x)
            oks.append((x.abs() <= 0.5).all(1) & ((w.plane(x) - c).norm(dim=1) < 3e-3))
        return torch.cat(xs), torch.cat(oks)
    return f


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--maze', required=True)
    p.add_argument('--fits', required=True)
    p.add_argument('--speed-s', default='1,4,8')
    p.add_argument('--speed-steps', type=int, default=3000)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = open(out / 'log.txt', 'a', buffering=1)
    dev = 'cuda:0'
    maze = SC.make_maze(a.maze, dev)
    res = dict(maze=a.maze, rows={})
    jobs = [('identity', None)] + [tuple(q.split('=')) for q in a.fits.split(',')]
    for label, rid in jobs:
        if rid is None:
            fn = SF.of_psi(M.Identity())
            valid = 1.0
        else:
            state, cfg, _ = NW.load_fit(SC.ROOT / 'results' / f'registered_{rid}', dev)
            w = NW.NetWarp(dict(net=cfg, s=1.0)).to(dev)
            w.set_state(state, maze)
            fn = of_netwarp(w)
            z = torch.rand(200000, 2, device=dev) - 0.5
            valid = float(fn(z)[1].float().mean())
        row = dict(run=rid, valid_share=valid, speed={s: SF.fit(fn, maze, float(s), a.speed_steps) for s in a.speed_s.split(',')})
        res['rows'][label] = row
        f.write(f'{label}: valid {valid:.3f} ' + ' '.join(f"s{s} r2 {v['r2']:.3f} corr {v['log_corr']:.3f} wall {v['wall_pred']:.2f}" for s, v in row['speed'].items()) + '\n')
        (out / 'results.json').write_text(json.dumps(res, indent=1))
    res['complete'] = True
    (out / 'results.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
