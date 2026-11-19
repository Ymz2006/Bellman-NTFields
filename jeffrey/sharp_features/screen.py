"""Properties of every spec's input mapping before any training (CPU, no GPU): report/screen.json.

    OMP_NUM_THREADS=4 ~/.venvs/bellman-ntfield/bin/python jeffrey/sharp_features/screen.py [--specs a,b,...]

Per maze (13x13, 22x22) and spec, with training seed 87358's draw:
  feature_metrics (diagnostics.py): arc length of the embedding near walls vs in the open (and the share carried by
      the sharp columns), fold across grid lines vs inside cells
  rank      singular values of the centred feature matrix over 20,000 held-out x0: effective rank exp(entropy of
            s^2 / sum s^2) and the count above 1e-3 of the largest (sin^2 + cos^2 = 1 makes a pow2 pair one feature)
  probe     ridge regression (lambda 1e-3 x trace / D) of the trained speed label, log S(x0), on the features: R^2 on
            10,000 held-out x0 not used in the fit; a linear readout of the walls, not what the network does, but it
            shows whether a feature makes the speed map more directly readable
  rms       RMS of the features and of the sharp columns (scale entering the first layer)
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
import features as FT  # noqa: E402
import sharpnet as SN  # noqa: E402
import maze as MZ  # noqa: E402
import diagnostics as DG  # noqa: E402
from sharp import sharp_columns  # noqa: E402

PREP = {13: ROOT / 'results/registered_20261101T192504Z-13cf1d04', 22: ROOT / 'results/registered_20261101T193403Z-cea05796'}
OUT = HERE / 'report' / 'screen.json'


def mapping_for(spec, n, E, seed, dev):
    mn = SN.network_module()
    Bo = FT.octave(FT.draw(seed), E)
    pl = FT.plan(spec, Bo, E, n, seed)
    net = SN.build(mn.NN, dev, Bo.clone(), pl).to(dev)
    return net.input_mapping, pl


@torch.no_grad()
def probe(maze, prepared, mapping, count=30000, fit=20000):
    hx = torch.tensor(np.load(prepared / 'heldout/sampled_points.npy')[:count], device=maze.device)
    hd = torch.tensor(np.load(prepared / 'heldout/clearance_cells.npy')[:count], device=maze.device)[:, 0]
    y = torch.log(maze.train_speed(hd)).double()
    F = mapping(hx[:, :3].contiguous()).double()
    F = torch.cat([F, torch.ones(len(F), 1, dtype=F.dtype)], 1)
    A, b = F[:fit], y[:fit]
    G = A.T @ A
    lam = 1e-3 * torch.trace(G) / G.shape[0]
    w = torch.linalg.solve(G + lam * torch.eye(G.shape[0], dtype=G.dtype), A.T @ b)
    pred = F[fit:] @ w
    r2 = 1 - float(((pred - y[fit:]) ** 2).mean() / y[fit:].var())
    Fc = F[:fit, :-1] - F[:fit, :-1].mean(0)
    s = torch.linalg.svdvals(Fc)
    pw = s ** 2 / (s ** 2).sum()
    eff = float(torch.exp(-(pw * torch.log(pw.clamp(min=1e-30))).sum()))
    return dict(speed_probe_r2=r2, rank_effective=eff, rank_1e3=int((s > 1e-3 * s[0]).sum()), width=int(F.shape[1] - 1),
                rms=float(F[:, :-1].pow(2).mean().sqrt()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--specs', default='')
    p.add_argument('--seed', type=int, default=87358)
    a = p.parse_args()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    specs = [s for s in a.specs.split(',') if s] or default_specs()
    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    dev = 'cpu'
    for n in (13, 22):
        maze = MZ.Maze(n, dev)
        for spec in specs:
            key = f'{n}|{a.seed}|{spec}'
            if key in old:
                continue
            mapping, pl = mapping_for(spec, n, maze.E, a.seed, dev)
            cols = torch.tensor(sharp_columns(pl))
            rec = DG.feature_metrics(maze, PREP[n], mapping, cols if cols.any() else None)
            rec.update(probe(maze, PREP[n], mapping))
            if cols.any():
                hx = torch.tensor(np.load(PREP[n] / 'heldout/sampled_points.npy')[:5000])
                rec['rms_sharp'] = float(mapping(hx[:, :3].contiguous())[:, cols].pow(2).mean().sqrt())
            old[key] = rec
            print(key, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
            OUT.write_text(json.dumps(old, indent=1))


def default_specs():
    jobs = [l.strip() for l in open(HERE / 'queue/jobs.txt') if l.strip() and not l.startswith('#')]
    out = []
    for j in jobs:
        s = j.split('|')[3]
        if s not in out:
            out.append(s)
    return out


if __name__ == '__main__':
    main()
