"""Ripples of the learned travel time along the true paths: where the local minima are and at what wavelength.
-> report/ripple.json (one record per run id, cached).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/sharp_features/ripple.py --ids a,b

Per run: T(x, goal) along every true path of the queries >= 8 cells (densified to 0.01 cell); local minima (as
diagnostics.bumps, finer sampling); their depth (rise of T after the minimum before it falls again, in cells of travel
time: T / pitch); the spacing between consecutive minima (cells of path length); the spectrum of the detrended T
(T minus its 1-cell moving average) by wavelength band: share of ripple power at wavelength < 1/4 cell, 1/4-1/2,
1/2-1, 1-2 cells. The powers sin^p of a top-octave row (half period one cell, wavelength 2 cells) put harmonics at
wavelength 2 / k cells, k = 2 ... p.
Eikonal against monotone: uphill share = share of the true path's length on which the learned T rises toward the goal
(the true T falls everywhere along it); eikonal residual = median |S |grad T| - 1| along the paths (S = trained speed
label), overall and on the uphill steps. A sawtooth T satisfies the eikonal equation as well as the true travel time,
so a low residual with a large uphill share is a spurious (folded) solution, not a poor fit.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
import sharpnet as SN  # noqa: E402
import maze as MZ  # noqa: E402
import plan as P  # noqa: E402

OUT = HERE / 'report' / 'ripple.json'
PREP = {13: ROOT / 'results/registered_20261101T192504Z-13cf1d04', 22: ROOT / 'results/registered_20261101T193403Z-cea05796'}
BANDS = [(0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0)]


def analyse(maze, prepared, cost, grad=None, res=0.01):
    q = np.load(prepared / 'queries.npz')
    tp = np.load(prepared / 'truth_paths.npy', allow_pickle=True)
    depths, spacings, counts, power = [], [], [], np.zeros(len(BANDS))
    uphill_len, total_len, eik_res, eik_up = 0.0, 0.0, [], []
    for k in np.nonzero(q['bin'] >= 3)[0]:
        p = P.densify(torch.tensor(np.asarray(tp[k], np.float32), device=maze.device), res * maze.pitch)
        g = torch.tensor(q['goal'][k], device=maze.device, dtype=torch.float32).expand(len(p), 3)
        t = cost(torch.cat([p, g], 1)).cpu().numpy() / maze.pitch          # travel time in cells
        seg_c = np.linalg.norm(np.diff(p[:, :2].cpu().numpy(), axis=0), axis=1) * maze.E
        up = np.diff(t) > 0                                                  # T rises toward the goal on this step
        uphill_len += float(seg_c[up].sum())
        total_len += float(seg_c.sum())
        if grad is not None:
            _, gr = grad(torch.cat([p, g], 1))
            S = maze.train_speed(maze.clearance(p))
            r = (S * torch.linalg.vector_norm(gr, dim=1)).cpu().numpy()      # eikonal: S |grad T| = 1
            eik_res += np.abs(r - 1).tolist()
            mid = np.concatenate([up, [False]])
            eik_up += np.abs(r[mid] - 1).tolist()
        seg = np.linalg.norm(np.diff(p[:, :2].cpu().numpy(), axis=0), axis=1) * maze.E
        s = np.concatenate([[0], np.cumsum(seg)])                            # path length in cells
        dt = np.diff(t)
        sgn = np.sign(dt)
        mins = np.nonzero((sgn[:-1] < 0) & (sgn[1:] > 0))[0] + 1
        counts.append(len(mins))
        for m in mins:
            j = m
            while j + 1 < len(t) and t[j + 1] >= t[j]:
                j += 1
            depths.append(float(t[j] - t[m]))
        if len(mins) > 1:
            spacings += np.diff(s[mins]).tolist()
        # ripple spectrum: resample on a uniform 0.01-cell grid, subtract a 1-cell moving average
        if s[-1] > 4:
            u = np.arange(0, s[-1], res)
            tu = np.interp(u, s, t)
            w = int(round(1.0 / res))
            trend = np.convolve(tu, np.ones(w) / w, mode='same')
            r = (tu - trend)[w:-w]
            if len(r) > 16:
                F = np.abs(np.fft.rfft(r * np.hanning(len(r)))) ** 2
                lam = 1.0 / np.maximum(np.fft.rfftfreq(len(r), d=res), 1e-12)
                for b, (lo, hi) in enumerate(BANDS):
                    power[b] += F[(lam >= lo) & (lam < hi)].sum()
    power = power / power.sum() if power.sum() else power
    extra = dict(uphill_share=uphill_len / total_len if total_len else None)
    if eik_res:
        extra.update(eikonal_residual_median=float(np.median(eik_res)), eikonal_residual_uphill_median=float(np.median(eik_up)) if eik_up else None)
    return dict(**extra, paths=len(counts), minima_per_path=float(np.mean(counts)), depth_median=float(np.median(depths)) if depths else None,
                depth_q90=float(np.quantile(depths, 0.9)) if depths else None,
                spacing_median=float(np.median(spacings)) if spacings else None,
                spacing_quartiles=[float(x) for x in np.quantile(spacings, [0.25, 0.5, 0.75])] if spacings else None,
                ripple_power_by_wavelength={f'{lo:g}-{hi:g}': float(v) for (lo, hi), v in zip(BANDS, power)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids', default='')
    ap.add_argument('--all', action='store_true', help="every finished run of this study and every reference run in report/summary.json")
    a = ap.parse_args()
    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    dev = 'cuda:0'
    ids = [x for x in a.ids.split(',') if x]
    if a.all:
        ids += [rid for rid in json.load(open(HERE / 'queue/state.json'))['done'].values() if rid]
        summ = json.loads((HERE / 'report/summary.json').read_text())
        for n, v in summ['sizes'].items():
            for ref in v['reference'].values():
                ids += ref['runs']
    for rid in dict.fromkeys(ids):
        if rid in old:
            print(rid, json.dumps(old[rid]))
            continue
        d = ROOT / f'results/registered_{rid}'
        r = json.loads((d / 'results.json').read_text())
        n = r['maze']['n']
        spec = r.get('spec') if r.get('stage') == 'sharp' else 'octave'
        maze = MZ.Maze(n, dev)
        if r.get('stage') == 'sharp':
            net = SN.load_field(d / 'training/checkpoint.pt', spec, maze.E, n, r['seed'], dev)
        else:
            net = P.load_field(d / 'training/checkpoint.pt', 'metric', dev)
        rec = dict(spec=spec if r.get('stage') == 'sharp' else r.get('spec', r.get('mode', 'octave')), n=n, seed=r['seed'],
                   **analyse(maze, PREP[n], P.field_cost(net), lambda pairs: P.field_grad(net, pairs)))
        old[rid] = rec
        OUT.write_text(json.dumps(old, indent=1))
        print(rid, json.dumps(rec), flush=True)
        del net
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
