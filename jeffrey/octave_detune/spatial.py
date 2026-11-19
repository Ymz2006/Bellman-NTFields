"""Where in the maze does a detuned field differ? (inference only; report/spatial.json)

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/octave_detune/spatial.py

Every row of an exact octave matrix has its phase locked to the grid lines; a detuned row (shift or jitter) agrees with
it at the maze centre (x = y = 0, where every phase is 0) and drifts linearly away from it: its phase error at a grid
line is pi |d - d*| times the line's distance from the centre in cells. So if the lock matters, a detuned field should
lose its accuracy and its wall barrier mostly in the OUTER part of the maze and keep them near the centre. Rings by the
Chebyshev radius r = max(|x|, |y|) / 0.5 (0 = centre, 1 = border): inner r < 1/3, middle 1/3 - 2/3, outer > 2/3.

Per run, final checkpoint, by ring of x0 on the 50,000 held-out pairs: median near-wall slope S |grad T| and cosine
with -n (x0 with trained speed S < 0.3, gap 0.5 - 8 cells), median |log S_pred - log S_label| over all x0; by ring of
the midpoint of the local queries (true distance <= 8 cells; rows.npz, no GPU): success over planning seeds 0/1/2 and
median |log T/T*|.
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import octdetune as OD  # noqa: E402
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location('octave_detune_report', HERE / 'report.py')   # maze_scale has its own report.py
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

OUT = HERE / 'report' / 'spatial.json'
EDGES = [0, 1 / 3, 2 / 3, 1.0001]
RINGS = ['inner', 'middle', 'outer']


def ring_of(xy):
    r = np.abs(np.asarray(xy)).max(1) / 0.5
    return np.digitize(r, EDGES[1:-1])


def field_part(maze, prepared, net):
    hx = torch.tensor(np.load(prepared / 'heldout/sampled_points.npy'), device=maze.device)
    hd = torch.tensor(np.load(prepared / 'heldout/clearance_cells.npy'), device=maze.device)[:, 0]
    _, normal = maze.clearance_normal(hx[:, :3])
    S = maze.train_speed(hd)
    gap = torch.linalg.vector_norm(hx[:, :3] - hx[:, 3:], dim=1) / maze.pitch
    _, g = OD.P.field_grad(net, hx)
    gn = torch.linalg.vector_norm(g, dim=1)
    slope = (S * gn).cpu().numpy()
    cos = torch.nn.functional.cosine_similarity(g, -normal, dim=1).cpu().numpy()
    res = np.abs(np.log(1 / gn.clamp(min=1e-6).cpu().numpy()) - np.log(S.cpu().numpy()))
    S, gap = S.cpu().numpy(), gap.cpu().numpy()
    ring = ring_of(hx[:, :2].cpu().numpy())
    out = {}
    for k, name in enumerate(RINGS):
        near = (ring == k) & (S < 0.3) & (gap >= 0.5) & (gap < 8)
        out[name] = dict(near_n=int(near.sum()), slope=float(np.median(slope[near])) if near.any() else None,
                         cos=float(np.median(cos[near])) if near.any() else None,
                         speed_res=float(np.median(res[ring == k])), n=int((ring == k).sum()))
    return out


def query_part(d):
    q = np.load(Path(d['prepared']) / 'queries.npz')
    r = np.load(d['_dir'] / 'rows.npz')
    mid = (q['start'][:, :2] + q['goal'][:, :2]) / 2
    ring = ring_of(mid)
    local = q['bin'] <= 2
    err = np.abs(np.log(np.maximum(r['t_pred'], 1e-9) / q['t_star']))
    out = {}
    for k, name in enumerate(RINGS):
        m = local & (ring == k)
        out[name] = dict(queries=int(m.sum()), success=int(r['success'][:, m].sum()), runs=int(r['success'][:, m].size),
                         log_err=float(np.median(err[m])) if m.any() else None)
    return out


def main():
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for d in R.runs():
        if d['_id'] in res:
            continue
        maze = OD.MZ.Maze(d['_n'], 'cuda:0')
        net = OD.P.load_field(d['_dir'] / 'training/checkpoint.pt', 'metric', 'cuda:0')
        res[d['_id']] = dict(key=d['_key'], n=d['_n'], seed=d['seed'], field=field_part(maze, Path(d['prepared']), net), queries=query_part(d))
        print(d['_id'], d['_key'], d['_n'], d['seed'], {k: round(v['slope'], 3) if v['slope'] else None for k, v in res[d['_id']]['field'].items()}, flush=True)
        del net
        torch.cuda.empty_cache()
        OUT.write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
