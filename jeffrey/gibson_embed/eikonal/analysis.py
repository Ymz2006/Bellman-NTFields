"""Re-plan with the saved models of a run and score every path against the speed-model optimum.

For each pair: success (met + collision-free, as in the run), the travel time of the planned polyline under the
P-NTFields speed model (line integral of 1/s on the 256^3 collision grid, occupied cells at speed 0.1), and the grid
optimum through all cells (T_all) and through free cells only (T_free, see walltime.py). A path that crosses a wall
can still be close to the optimum when the optimum crosses too. Prints a markdown table and writes analysis.json.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/gibson_embed/eikonal/analysis.py results/registered_<id> out_dir [--models a,b]
"""
import argparse
import ast
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
import diffuse  # noqa: E402
import model as M  # noqa: E402
import plan as P  # noqa: E402
import walltime as W  # noqa: E402
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location('eikonal_pipeline', HERE.parent / 'pipeline.py')  # gibson_embed/pipeline.py shadows it
_pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pipeline)
EMBEDDINGS, VARIANTS = _pipeline.EMBEDDINGS, _pipeline.VARIANTS

ROOT = HERE.parents[3]
VARIANTS = dict(VARIANTS, **{'fourier-visc': VARIANTS['fourier']})  # round-1 name of the viscous Fourier baseline


def path_time(points, inv_s, spacing=0.5):
    """Travel time of a polyline: line integral of 1/s sampled at <= spacing cells (trapezoid)."""
    res = inv_s.shape[0]
    seg = points[1:] - points[:-1]
    length = seg.norm(dim=1)
    total = 0.0
    for k in range(len(seg)):
        n_sub = max(1, int(np.ceil(float(length[k]) * res / spacing)))
        t = torch.linspace(0, 1, n_sub + 1, device=points.device)
        samples = points[k] + t[:, None] * seg[k]
        cells = ((samples + 0.5) * res).floor().long().clamp(0, res - 1)
        v = inv_s[tuple(cells.T)]
        total += float(length[k]) * float((0.5 * (v[1:] + v[:-1])).mean())
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_dir')
    p.add_argument('out_dir')
    p.add_argument('--models', default='')
    p.add_argument('--step', type=float, default=0.03)
    p.add_argument('--tol', type=float, default=0.06)
    p.add_argument('--max-iter', type=int, default=500)
    a = p.parse_args()
    dev = torch.device('cuda')
    run = Path(a.run_dir)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = print
    results = json.load(open(run / 'results.json'))
    args = ast.literal_eval(open(run / 'log.txt').readline().split('args ', 1)[1])
    g = np.load(run / 'grid.npz')
    occ_g, occ_c = g['occupancy_geodesic'], g['occupancy_collision']
    pairs = dict(np.load(run / 'pairs.npz'))
    models = a.models.split(',') if a.models else [k for k in results if k in VARIANTS]
    free_g = torch.from_numpy(~occ_g).to(dev)
    occ_c_t = torch.from_numpy(occ_c).to(dev)
    inv_s_c = torch.tensor(1.0 / W.speed_field(occ_c), dtype=torch.float32, device=dev)

    tf, ta, crosses = W.pair_optima(occ_g, pairs, dev, log=log)
    B = np.load(ROOT / args['bdir'] / args['scene'] / 'B.npy')
    snapshots = tuple(int(s) for s in args['snapshots'].split(','))
    needed = sorted({e for v in models for e in VARIANTS[v][1]})
    feats = {}
    for name in needed:
        spec = dict(EMBEDDINGS[name])
        kind = spec.pop('kind')
        snaps = spec.pop('snapshots', snapshots)
        chans = spec.pop('channels', args['channels'])
        feat, _ = diffuse.embed(free_g, kind=kind, r=1.0, snapshots=snaps, channels=chans, seed=args['seed'], log=lambda *_: None, **spec)
        feats[name] = feat.half()
        del feat
        torch.cuda.empty_cache()
        log(f'embedding {name} recomputed')

    start = torch.tensor(pairs['start'], device=dev)
    goal = torch.tensor(pairs['goal'], device=dev)
    summary = dict(run=str(run), optimum=dict(T_free=tf.tolist(), T_all=ta.tolist(), crosses=crosses.tolist()), models={})
    rows = []

    def score(name, direction_fn):
        t0 = time.time()
        traj0, traj1, steps, met = P.plan(direction_fn, start, goal, a.step, a.tol, a.max_iter)
        n = len(start)
        free_path, ptime = np.zeros(n, dtype=bool), np.zeros(n)
        for i in range(n):
            pts = P.path_points(traj0, traj1, steps, i)
            free_path[i] = P.collision_free(pts, occ_c_t)
            ptime[i] = path_time(pts, inv_s_c)
        met = met.cpu().numpy()
        success = met & free_path
        summary['models'][name] = dict(per_pair=dict(success=success.tolist(), met=met.tolist(), free=free_path.tolist(), time=ptime.tolist()))
        for kind in ['uniform', 'wall']:
            row = score_rows(name, kind, pairs['kind'], crosses, ta, success, met, ptime)
            rows.append(row)
            summary['models'][name][kind] = row
        log(f'  {name}: {time.time() - t0:.0f}s')

    pre = ROOT / args['pretrained']
    if pre.exists():
        ck = torch.load(pre, map_location='cpu', weights_only=False)
        net = M.EikNet(3, [M.FourierB(B)]).to(dev)
        M.load_pntfields_state(net, ck['model_state_dict'])
        net.eval()
        score('pretrained', lambda X: M.descent_direction(net, X))
        del net
    for v in models:
        use_fourier, emb = VARIANTS[v][:2]
        features = ([M.FourierB(B)] if use_fourier else []) + [M.MaskedGridFeature(feats[e].float(), free_g) for e in emb]
        net = M.EikNet(3, features).to(dev)
        net.load_state_dict(torch.load(run / f'model_{v}.pt', map_location=dev), strict=False)
        net.eval()
        score(v, lambda X: M.descent_direction(net, X))
        del net, features
        torch.cuda.empty_cache()

    json.dump(summary, open(out / 'analysis.json', 'w'))
    write_table(rows, out / 'analysis.md')


def score_rows(name, kind, kinds, crosses, ta, success, met, ptime):
    """One table row: success overall and split by whether the optimum detours or crosses; path time / optimum
    (median over met pairs) overall and on the two subsets; fraction within 5 % of the optimum."""
    rel = ptime / ta
    sel = kinds == kind
    det, cro = sel & ~crosses, sel & crosses
    med = lambda m: float(np.median(rel[m & met])) if (m & met).any() else None  # noqa: E731
    return dict(model=name, kind=kind, n=int(sel.sum()), success=float(success[sel].mean()),
                success_detour_optimal=float(success[det].mean()) if det.any() else None, n_detour=int(det.sum()),
                success_cross_optimal=float(success[cro].mean()) if cro.any() else None, n_cross=int(cro.sum()),
                time_over_optimum_median=med(sel), time_over_optimum_p90=float(np.percentile(rel[sel & met], 90)),
                time_over_optimum_detour=med(det), time_over_optimum_cross=med(cro), within_5pct=float((rel[sel & met] < 1.05).mean()))


def write_table(rows, path=None):
    lines = ['| model | kind | n | success | success, detour-optimal (n) | success, crossing-optimal (n) | time/opt median | time/opt p90 | time/opt detour-opt | time/opt crossing-opt | within 5 % |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    f = lambda x: '–' if x is None else f'{x:.3f}'  # noqa: E731
    for r in rows:
        lines.append(f"| {r['model']} | {r['kind']} | {r['n']} | {f(r['success'])} | {f(r['success_detour_optimal'])} ({r['n_detour']}) | "
                     f"{f(r['success_cross_optimal'])} ({r['n_cross']}) | {f(r['time_over_optimum_median'])} | {f(r['time_over_optimum_p90'])} | "
                     f"{f(r['time_over_optimum_detour'])} | {f(r['time_over_optimum_cross'])} | {f(r['within_5pct'])} |")
    text = '\n'.join(lines) + '\n'
    if path:
        Path(path).write_text(text)
    print(text)


def table_from_json(path, pairs_path):
    """Regenerate the table from a saved analysis.json (per-pair data) and the run's pairs.npz."""
    s = json.load(open(path))
    kinds = np.load(pairs_path)['kind']
    crosses, ta = np.array(s['optimum']['crosses']), np.array(s['optimum']['T_all'])
    rows = []
    for name, m in s['models'].items():
        pp = m['per_pair']
        for kind in ['uniform', 'wall']:
            rows.append(score_rows(name, kind, kinds, crosses, ta, np.array(pp['success']), np.array(pp['met']), np.array(pp['time'])))
    write_table(rows, Path(path).with_suffix('.md'))


if __name__ == '__main__':
    if sys.argv[1] == '--from-json':
        table_from_json(sys.argv[2], sys.argv[3])
    else:
        main()
