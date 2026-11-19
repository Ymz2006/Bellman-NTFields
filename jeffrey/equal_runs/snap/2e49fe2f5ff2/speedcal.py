"""Does the learnability score track planning? Scores line_warp's fitted maps, whose planning results are known, with speedfit.py. Read-only on those runs.

    speedcal.py --maze plus:22:6 --fits label=<fit run id>,... --out <dir>     (label 'identity' needs no run)

speedfit.py draws z uniformly in the box. These maps have an inverse by Newton (NetWarp.inverse_plane); z outside the image of the box is rejected.
"""
import argparse
import json
import sys
from pathlib import Path
import torch

import measure as ME
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


class Inverse:
    """A line_warp map as psi: z -> x by Newton, with its forward map as .inverse, so measure.all_measures reads it like a map of this study."""

    def __init__(self, w):
        self.w = w
        w.inverse_table()

    def __call__(self, z):
        with torch.no_grad():
            return torch.cat([self.w.inverse_plane(z[i:i + 262144]) for i in range(0, len(z), 262144)]) if len(z) else z

    def inverse(self, x):
        return self.w.plane(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--maze', required=True)
    p.add_argument('--fits', required=True)
    p.add_argument('--speed-s', default='1,4,8')
    p.add_argument('--speed-steps', type=int, default=3000)
    p.add_argument('--zone-only', action='store_true', help='only the wall-zone measure (measure.wall_zone), no speed regression')
    p.add_argument('--measures', action='store_true', help='also every measure of measure.all_measures, for a table next to this study')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = open(out / 'log.txt', 'a', buffering=1)
    dev = 'cuda:0'
    scene = SC.build(a.maze, dev, 9, 16)
    maze = scene.maze
    pts = ME.wall_points(scene)
    res = dict(maze=a.maze, rows={})
    jobs = [('identity', None)] + [tuple(q.split('=')) for q in a.fits.split(',')]
    for label, rid in jobs:
        if rid is None:
            fn = SF.of_psi(M.Identity())
            valid = 1.0
            plane = lambda x: x
        else:
            state, cfg, _ = NW.load_fit(SC.ROOT / 'results' / f'registered_{rid}', dev)
            w = NW.NetWarp(dict(net=cfg, s=1.0)).to(dev)
            w.set_state(state, maze)
            plane = w.plane
            fn = None if a.zone_only else of_netwarp(w)
            valid = 1.0
            if fn is not None:
                z = torch.rand(200000, 2, device=dev) - 0.5
                valid = float(fn(z)[1].float().mean())
        row = dict(run=rid, valid_share=valid, zone=ME.wall_zone(plane, scene, pts), speed={} if a.zone_only else {s: SF.fit(fn, maze, float(s), a.speed_steps) for s in a.speed_s.split(',')})
        if a.measures:
            row['warp'] = ME.all_measures(Inverse(w) if rid is not None else M.Identity(), scene, max(2048, 32 * int(round(scene.E))))
        res['rows'][label] = row
        f.write(f"{label}: zone {row['zone']} valid {valid:.3f} " + ' '.join(f"s{s} r2 {v['r2']:.3f} corr {v['log_corr']:.3f} wall {v['wall_pred']:.2f}" for s, v in row['speed'].items()) + '\n')
        (out / 'results.json').write_text(json.dumps(res, indent=1))
    res['complete'] = True
    (out / 'results.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
