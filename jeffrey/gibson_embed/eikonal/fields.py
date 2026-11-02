"""Travel-time fields T(start, .) on the z-slice of a start point, for several trained models, with the planned path.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/gibson_embed/eikonal/fields.py <out.png> --models label=run_dir:variant ...

Picks one wall pair whose speed-model optimum detours and one whose optimum crosses the wall (from an analysis.json of
the same scene) and draws, per model, the contours of T from the start over the slice, the walls, and the planned path.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
import diffuse  # noqa: E402
import model as M  # noqa: E402
import plan as P  # noqa: E402

_spec = importlib.util.spec_from_file_location('eikonal_pipeline', HERE.parent / 'pipeline.py')
_pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pipeline)
EMBEDDINGS, VARIANTS = _pipeline.EMBEDDINGS, dict(_pipeline.VARIANTS, **{'fourier-visc': _pipeline.VARIANTS['fourier']})
ROOT = HERE.parents[3]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out')
    p.add_argument('--run', required=True, help='run dir providing grid.npz, pairs.npz and the scene')
    p.add_argument('--analysis', required=True, help='analysis.json of the same pairs (optimum per pair)')
    p.add_argument('--models', nargs='+', required=True, help='label=run_dir:variant, or label=pretrained')
    p.add_argument('--pairs', default='', help='explicit pair indices i,j (default: auto-pick detour-optimal and crossing-optimal wall pairs)')
    p.add_argument('--slice-res', type=int, default=256)
    a = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dev = torch.device('cuda')
    run = Path(a.run)
    import ast
    args = ast.literal_eval(open(run / 'log.txt').readline().split('args ', 1)[1])
    g = np.load(run / 'grid.npz')
    occ_g, occ_c = g['occupancy_geodesic'], g['occupancy_collision']
    pairs = dict(np.load(run / 'pairs.npz'))
    an = json.load(open(a.analysis))
    crosses = np.array(an['optimum']['crosses'])
    free_g = torch.from_numpy(~occ_g).to(dev)
    occ_c_t = torch.from_numpy(occ_c).to(dev)
    B = np.load(ROOT / args['bdir'] / args['scene'] / 'B.npy')

    if a.pairs:
        chosen = [int(s) for s in a.pairs.split(',')]
    else:
        wall = pairs['kind'] == 'wall'
        dz = np.abs(pairs['goal'][:, 2] - pairs['start'][:, 2])
        flat = dz < 0.02
        rng = np.random.default_rng(1)
        chosen = [int(rng.choice(np.nonzero(wall & flat & ~crosses)[0])), int(rng.choice(np.nonzero(wall & flat & crosses)[0]))]
    print('pairs', chosen, [(pairs['kind'][i], bool(crosses[i]), float(pairs['euclid'][i]), float(pairs['geodesic'][i])) for i in chosen])

    feats = {}

    def build(spec):
        label, src = spec.split('=', 1)
        if src == 'pretrained':
            ck = torch.load(ROOT / args['pretrained'], map_location='cpu', weights_only=False)
            net = M.EikNet(3, [M.FourierB(B)]).to(dev)
            M.load_pntfields_state(net, ck['model_state_dict'])
        else:
            rdir, variant = src.rsplit(':', 1)
            use_fourier, emb = VARIANTS[variant][:2]
            for e in emb:
                if e not in feats:
                    s = dict(EMBEDDINGS[e])
                    kind = s.pop('kind')
                    snaps = s.pop('snapshots', tuple(int(x) for x in args['snapshots'].split(',')))
                    chans = s.pop('channels', args['channels'])
                    f, _ = diffuse.embed(free_g, kind=kind, r=1.0, snapshots=snaps, channels=chans, seed=args['seed'], log=lambda *_: None, **s)
                    feats[e] = f.half()
                    del f
                    torch.cuda.empty_cache()
            features = ([M.FourierB(B)] if use_fourier else []) + [M.MaskedGridFeature(feats[e].float(), free_g) for e in emb]
            net = M.EikNet(3, features).to(dev)
            net.load_state_dict(torch.load(Path(rdir) / f'model_{variant}.pt', map_location=dev), strict=False)
        net.eval()
        return label, net

    nets = [build(s) for s in a.models]
    R = a.slice_res
    lin = (np.arange(R) + 0.5) / R - 0.5
    fig, axes = plt.subplots(len(chosen), len(nets), figsize=(3.2 * len(nets), 3.2 * len(chosen)), squeeze=False)
    for r, i in enumerate(chosen):
        s, gl = pairs['start'][i], pairs['goal'][i]
        z = float(s[2])
        kz = int(np.clip(np.floor((z + 0.5) * occ_c.shape[0]), 0, occ_c.shape[0] - 1))
        wall_slice = occ_c[:, :, kz]
        xx, yy = np.meshgrid(lin, lin, indexing='ij')
        pts = torch.tensor(np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, z)], 1), dtype=torch.float32, device=dev)
        start = torch.tensor(s, device=dev)[None]
        goal = torch.tensor(gl, device=dev)[None]
        win = 0.12
        for c, (label, net) in enumerate(nets):
            with torch.no_grad():
                T = torch.cat([net.travel_time(start.expand(len(ch), 3), ch) for ch in pts.split(65536)]).cpu().numpy().reshape(R, R)
            traj0, traj1, steps, met = P.plan(lambda X: M.descent_direction(net, X), start, goal)
            path = P.path_points(traj0, traj1, steps, 0).cpu().numpy()
            ok = bool(met[0]) and P.collision_free(torch.tensor(path, device=dev), occ_c_t)
            ax = axes[r, c]
            Tm = np.ma.masked_where(wall_slice, T)
            ax.contourf(lin, lin, Tm.T, levels=np.linspace(0, 0.6, 25), cmap='viridis', extend='max')
            ax.contour(lin, lin, Tm.T, levels=np.linspace(0, 0.6, 13), colors='white', linewidths=0.4, alpha=0.7)
            ax.imshow(np.ma.masked_where(~wall_slice, wall_slice).T, extent=[-0.5, 0.5, -0.5, 0.5], origin='lower', cmap='gray_r', vmin=0, vmax=1.3, interpolation='nearest')
            ax.plot(path[:, 0], path[:, 1], color='#ff5a36' if not ok else '#ffffff', lw=1.6)
            ax.plot(s[0], s[1], 'o', color='#ffd500', ms=5)
            ax.plot(gl[0], gl[1], 's', color='#ff5a36', ms=5)
            cx, cy = (s[0] + gl[0]) / 2, (s[1] + gl[1]) / 2
            ax.set_xlim(cx - win, cx + win)
            ax.set_ylim(cy - win, cy + win)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(label, fontsize=9)
            if c == 0:
                ax.set_ylabel(f'pair {i} ({"optimum crosses" if crosses[i] else "optimum detours"})', fontsize=8)
            ax.text(0.02, 0.02, 'collision' if not ok else 'free', transform=ax.transAxes, fontsize=8, color='white',
                    bbox=dict(facecolor='black', alpha=0.5, pad=1.5, lw=0))
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
