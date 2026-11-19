"""One registered normalizing-flow cartogram: the Fourier network learns phi from crossing points of the oracle (nf.py)."""
import argparse
import json
import time
from pathlib import Path
import torch

torch.set_float32_matmul_precision('high')
import maps as M
import measure as ME
import nf as N
import scene as SC
import speedfit as SF
import view as V


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--res', type=int, default=32)
    p.add_argument('--headings', type=int, default=9)
    p.add_argument('--arch', choices=['coupling', 'flow', 'res'], default='coupling')
    p.add_argument('--h', type=float, default=1.0)
    p.add_argument('--eps', type=float, default=1.0)
    p.add_argument('--frame', type=float, default=1.0)
    p.add_argument('--lines', type=int, default=1 << 16)
    p.add_argument('--samples', type=int, default=2048)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--batch', type=int, default=8192)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--layers', type=int, default=6)
    p.add_argument('--knots-per-cell', type=float, default=2.0)
    p.add_argument('--hmax', type=float, default=4.0)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--per-scale', type=int, default=32)
    p.add_argument('--scales', default='0.05,0.1,0.2,0.4,0.8', help='cycles per cell')
    p.add_argument('--flow-steps', type=int, default=8)
    p.add_argument('--amp', type=float, default=4.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--speed-s', default='1,4')
    p.add_argument('--speed-steps', type=int, default=3000)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if a.smoke:
        a.steps, a.speed_steps, a.lines = 30, 30, 4096
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = open(out / 'log.txt', 'a', buffering=1)

    def log(s):
        print(s, flush=True)
        f.write(s + '\n')

    dev = 'cuda:0'
    t0 = time.time()
    scene = SC.build(a.scene, dev, a.headings, a.res)
    pts = N.cloud(scene, a.lines, a.samples, a.frame)
    log(f'scene {a.scene}: dim {scene.dim}, slices {scene.planes}; {len(pts)} crossing points, {time.time() - t0:.0f}s')
    scales = [float(s) * scene.E for s in a.scales.split(',')]
    if a.arch == 'coupling':
        phi = M.CouplingMap(scene.dim, a.layers, int(round(a.knots_per_cell * scene.E)), a.hidden, a.depth, scales, a.per_scale, a.hmax, a.seed).to(dev)
    else:
        phi = M.FlowMap(scene.dim, a.flow_steps, a.hidden, a.depth, scales, a.per_scale, a.seed, a.arch == 'flow', a.amp).to(dev)
    hist = N.fit(phi, scene, pts, a.h, a.eps, a.steps, a.batch, a.lr, log=log, seed=a.seed)
    torch.save(dict(state=phi.state_dict(), cfg=phi.cfg, args={k: str(v) for k, v in vars(a).items()}), out / 'psi.pt')
    psi = N.Inverted(phi)
    res = dict(stage='equal-runs-flow', scene=a.scene, args={k: str(v) for k, v in vars(a).items()}, cfg=phi.cfg, gpu=torch.cuda.get_device_name(0), smoke=a.smoke, history=hist, fit_seconds=time.time() - t0)
    samples = max(2048, 32 * int(round(scene.E)))
    res['warp'] = ME.all_measures(psi, scene, samples)
    res['identity'] = ME.all_measures(M.Identity(), scene, samples)
    log(f"warp {res['warp']}")
    ss = a.speed_s.split(',')
    res['speed'] = SF.score(psi, scene, ss, a.speed_steps)
    res['speed_identity'] = SF.score(M.Identity(), scene, ss, a.speed_steps)
    log(f"speed {res['speed']} identity {res['speed_identity']}")
    V.overview(psi, scene, out / 'overview.png', f'flow {a.arch} {a.scene} h {a.h} eps {a.eps}')
    res['complete'] = True
    res['elapsed_seconds'] = time.time() - t0
    (out / 'results.json').write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
