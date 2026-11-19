"""One registered fit: the map (fit.py), its measurements on held-out lines (measure.py), its learnability score (speedfit.py), its picture (view.py)."""
import json
import time
from pathlib import Path
import torch

import fit as F
import maps as M
import measure as ME
import scene as SC
import speedfit as SF
import view as V


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=float))


def main():
    p = F.arguments()
    p.add_argument('--speed-steps', type=int, default=3000)
    p.add_argument('--speed-s', default='1,4', help='Fourier scale(s) of the learnability score')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if a.smoke:
        a.steps, a.speed_steps = 30, 30
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
    psi, hist = F.fit(a, scene, log)
    torch.save(dict(state=psi.state_dict(), cfg=psi.cfg, args={k: str(v) for k, v in vars(a).items()}), out / 'psi.pt')
    dump(out / 'history.json', hist)
    res = dict(stage='equal-runs-fit', scene=a.scene, args={k: str(v) for k, v in vars(a).items()}, cfg=psi.cfg, gpu=torch.cuda.get_device_name(0), smoke=a.smoke, final=hist[-1] if hist else None, fit_seconds=time.time() - t0)
    samples = max(2048, 32 * int(round(scene.E)))
    res['warp'] = ME.all_measures(psi, scene, samples)
    res['identity'] = ME.all_measures(M.Identity(), scene, samples)
    log(f"warp {res['warp']}")
    if scene.dim == 3:
        res['speed'] = {s: SF.fit_scene(psi, scene, float(s), a.speed_steps) for s in a.speed_s.split(',')}
        res['speed_identity'] = {s: SF.fit_scene(M.Identity(), scene, float(s), a.speed_steps) for s in a.speed_s.split(',')}
    elif hasattr(scene, 'exact') and not hasattr(scene, 'deg'):
        res['speed'] = {s: SF.fit(SF.of_psi(psi), scene.maze, float(s), a.speed_steps) for s in a.speed_s.split(',')}
        res['speed_identity'] = {s: SF.fit(SF.of_psi(M.Identity()), scene.maze, float(s), a.speed_steps) for s in a.speed_s.split(',')}
    elif hasattr(scene, 'deg'):                    # a turned maze: z -> x -> the maze's own frame
        def turned(g):
            def f(z):                              # outside the turned square there is no maze: those z are not drawn
                x, _ = SF.of_psi(g)(z)
                x = scene.to_inner(x)
                return x, (x.abs() <= 0.5).all(1)
            return f
        res['speed'] = {s: SF.fit(turned(psi), scene.maze, float(s), a.speed_steps) for s in a.speed_s.split(',')}
        res['speed_identity'] = {s: SF.fit(turned(M.Identity()), scene.maze, float(s), a.speed_steps) for s in a.speed_s.split(',')}
    log(f"speed {res.get('speed')} identity {res.get('speed_identity')}")
    V.overview(psi, scene, out / 'overview.png', f"{a.scene} {a.arch} {a.kind} {a.form} {a.scope} {a.weight} target {a.target} kappa {a.kappa}")
    res['complete'] = True
    res['elapsed_seconds'] = time.time() - t0
    dump(out / 'results.json', res)


if __name__ == '__main__':
    main()
