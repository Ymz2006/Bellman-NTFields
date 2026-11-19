"""One registered cartogram of a scene: the map (carto.py), its measurements (measure.py), its learnability score (speedfit.py), its picture (view.py)."""
import argparse
import json
import time
from pathlib import Path
import torch

torch.set_float32_matmul_precision('high')
import carto as C
import maps as M
import measure as ME
import scene as SC
import speedfit as SF
import view as V


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--res', type=int, default=32, help='oracle table steps per cell')
    p.add_argument('--headings', type=int, default=9)
    p.add_argument('--G', type=int, default=512, help='density grid')
    p.add_argument('--N', type=int, default=1024, help='nodes of the output tables')
    p.add_argument('--h', type=float, default=1.0, help='blur, cells')
    p.add_argument('--eps', type=float, default=1.0, help='open space share: the boundary gets 1 / (1 + eps) of the z box')
    p.add_argument('--frame', type=float, default=1.0, help='crossings this close to the box (cells) are left out')
    p.add_argument('--lines', type=int, default=1 << 16)
    p.add_argument('--samples', type=int, default=2048)
    p.add_argument('--flow-steps', type=int, default=320)
    p.add_argument('--directions', default='any')
    p.add_argument('--speed-s', default='1,4')
    p.add_argument('--speed-steps', type=int, default=3000)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if a.smoke:
        a.flow_steps, a.speed_steps, a.lines = 40, 30, 4096
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = open(out / 'log.txt', 'a', buffering=1)

    def log(s):
        print(s, flush=True)
        f.write(s + '\n')

    dev = 'cuda:0'
    t0 = time.time()
    scene = SC.build(a.scene, dev, a.headings, a.res)
    log(f'scene {a.scene}: dim {scene.dim}, slices {scene.planes}, table {scene.table.G}, {time.time() - t0:.0f}s')
    t1 = time.time()
    psi, rho = C.build(scene, a.G, a.N, a.h, a.eps, a.lines, a.samples, a.flow_steps, 1.0, a.directions, a.frame, log)
    res = dict(stage='equal-runs-cartogram', scene=a.scene, args={k: str(v) for k, v in vars(a).items()}, gpu=torch.cuda.get_device_name(0), smoke=a.smoke, build_seconds=time.time() - t1, rho_range=[float(rho.min()), float(rho.max())])
    torch.save(dict(psi=psi.psi_table.cpu(), phi=psi.phi_table.cpu(), args={k: str(v) for k, v in vars(a).items()}), out / 'psi.pt')
    samples = max(2048, 32 * int(round(scene.E)))
    res['warp'] = ME.all_measures(psi, scene, samples)
    res['identity'] = ME.all_measures(M.Identity(), scene, samples)
    log(f"warp {res['warp']}")
    ss = a.speed_s.split(',')
    res['speed'] = SF.score(psi, scene, ss, a.speed_steps)
    res['speed_identity'] = SF.score(M.Identity(), scene, ss, a.speed_steps)
    log(f"speed {res['speed']} identity {res['speed_identity']}")
    V.overview(psi, scene, out / 'overview.png', f'cartogram {a.scene} h {a.h} eps {a.eps}')
    res['complete'] = True
    res['elapsed_seconds'] = time.time() - t0
    (out / 'results.json').write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
