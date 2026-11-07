"""Where along the cell do the local minima of T(x, goal) sit? (report/bumpphase.json, bumpphase.md; inference only)

    CUDA_VISIBLE_DEVICES=0 ~/.venvs/bellman-ntfield/bin/python jeffrey/grid_sync/bumpphase.py [--seeds 87358,1,2]

Test of the coherent-ripple explanation: rows locked to the grid make the field's residual periodic with the grid, so
its local minima along a path should recur at the same position within a cell. For every local minimum of T(x, goal)
along the true paths of queries >= 8 cells (as mechanism.bumps), take the robot's position along its direction of
travel there (x if the path moves mostly along x, else y) modulo one cell: phase 0 = on a grid line, 0.5 = the cell
centre. Concentration R = |mean exp(2 pi i phase)| (0 = spread evenly; about 1 / sqrt(count) by chance) and the mean
phase. The same is computed for the wall-crossing positions of the true paths themselves (the paths' own geometry is
not uniform in phase: corridors and doorways), as the reference a field without grid-locked error would follow.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'jeffrey/maze_scale'))
_spec = importlib.util.spec_from_file_location('grid_sync_report', HERE / 'report.py')
GR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GR)
import maze as MZ  # noqa: E402
import plan as P  # noqa: E402


def phases(maze, prepared, cost, min_bin=3):
    q = np.load(prepared / 'queries.npz')
    tp = np.load(prepared / 'truth_paths.npy', allow_pickle=True)
    bump, ref, clear, allclear = [], [], [], []
    for k in np.nonzero(q['bin'] >= min_bin)[0]:
        p = P.densify(torch.tensor(np.asarray(tp[k], np.float32), device=maze.device), 0.02 * maze.pitch)
        g = torch.tensor(q['goal'][k], device=maze.device, dtype=torch.float32).expand(len(p), 3)
        t = cost(torch.cat([p, g], 1)).cpu().numpy()
        x, y, _ = (v.cpu().numpy() for v in maze.to_cells(p))
        dx, dy = np.gradient(x), np.gradient(y)
        along = np.where(np.abs(dx) >= np.abs(dy), x, y)          # coordinate along the direction of travel (cells)
        ref.append(np.mod(along, 1.0))
        dt = np.diff(t)
        keep = np.nonzero(np.abs(dt) > 1e-6 * t[0])[0]
        sgn = np.sign(dt[keep])
        idx = keep[1:][(sgn[:-1] < 0) & (sgn[1:] > 0)]           # index of the sample at the minimum
        bump.append(np.mod(along[idx], 1.0))
        clear.append(maze.clearance(p[torch.tensor(idx, device=maze.device, dtype=torch.long)]).cpu().numpy() if len(idx) else np.zeros(0))
        allclear.append(maze.clearance(p).cpu().numpy())
    phases.clearance = (np.concatenate(clear) if clear else np.zeros(0), np.concatenate(allclear))
    return np.concatenate(bump) if bump else np.zeros(0), np.concatenate(ref)


def ripple(maze, prepared, cost, min_bin=3, bins=20):
    """Cell-scale ripple of the field's error along the true paths and the share of it that is locked to the grid.

    Along each path (densified to 0.02 cells): r = T(x, goal) - T*_path(x), where T*_path is the travel time left along
    the true path (sum of |dq| / S, S = the trained speed label). The ripple is r minus its moving average over two
    cells of path; 'locked' = share of the ripple variance explained by the phase along the direction of travel (20
    phase bins, between-bin variance / total variance), pooled over all paths. Path geometry is the same for every
    field on a maze, so fields are compared on the same samples."""
    q = np.load(prepared / 'queries.npz')
    tp = np.load(prepared / 'truth_paths.npy', allow_pickle=True)
    rip, phs, scale = [], [], []
    for k in np.nonzero(q['bin'] >= min_bin)[0]:
        p = P.densify(torch.tensor(np.asarray(tp[k], np.float32), device=maze.device), 0.02 * maze.pitch)
        g = torch.tensor(q['goal'][k], device=maze.device, dtype=torch.float32).expand(len(p), 3)
        t = cost(torch.cat([p, g], 1)).cpu().numpy().astype(np.float64)
        S = maze.train_speed(maze.clearance(p)).clamp(min=1e-3).cpu().numpy().astype(np.float64)
        dq = torch.linalg.vector_norm(p[1:] - p[:-1], dim=1).cpu().numpy().astype(np.float64)
        dtau = dq * 0.5 * (1 / S[1:] + 1 / S[:-1])
        left = np.concatenate([np.cumsum(dtau[::-1])[::-1], [0.0]])     # travel time left along the path
        r = t - left
        x, y, _ = (v.cpu().numpy() for v in maze.to_cells(p))
        dx, dy = np.gradient(x), np.gradient(y)
        along = np.where(np.abs(dx) >= np.abs(dy), x, y)
        w = max(3, int(round(2.0 / 0.02)))                               # two cells of samples
        if len(r) <= w:
            continue
        kern = np.ones(w) / w
        trend = np.convolve(np.pad(r, (w // 2, w - 1 - w // 2), mode='edge'), kern, mode='valid')
        rip.append(r - trend)
        phs.append(np.mod(along, 1.0))
        scale.append(np.full(len(r), left[0]))
    rip, phs, scale = np.concatenate(rip), np.concatenate(phs), np.concatenate(scale)
    rel = rip / scale                                                    # relative to the query's travel time
    b = np.minimum((phs * bins).astype(int), bins - 1)
    means = np.bincount(b, weights=rel, minlength=bins) / np.maximum(np.bincount(b, minlength=bins), 1)
    counts = np.bincount(b, minlength=bins)
    between = float(np.sum(counts * (means - rel.mean()) ** 2) / counts.sum())
    total = float(rel.var())
    return dict(samples=int(len(rel)), ripple_rms_rel=float(np.sqrt(total)), locked_share=between / total if total > 0 else None,
                locked_rms_rel=float(np.sqrt(between)), phase_profile=[float(m) for m in means])


def stats(ph):
    if len(ph) == 0:
        return dict(count=0)
    z = np.exp(2j * np.pi * ph).mean()
    hist = np.histogram(ph, bins=10, range=(0, 1))[0]
    return dict(count=int(len(ph)), R=float(abs(z)), chance_R=float(1 / np.sqrt(len(ph))), mean_phase=float(np.mod(np.angle(z) / (2 * np.pi), 1.0)),
                hist10=hist.tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', default='87358')
    p.add_argument('--fields', default='ntrl-demo,B × 6,ring,octave,octavecircle,octaveperm')
    p.add_argument('--sizes', default='13,22')
    a = p.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    fields = a.fields.split(',')
    sizes = [int(s) for s in a.sizes.split(',')]
    out_json = GR.OUT / 'bumpphase.json'
    res = json.loads(out_json.read_text()) if out_json.exists() else {}
    for d in GR.runs():
        n, key = d['maze']['n'], GR.field_key(d)
        if d['seed'] not in seeds or n not in sizes or key not in fields:
            continue
        tag = f'{n}|{key}|{d["seed"]}|{d["_id"]}'
        if tag in res and 'near_wall' in res[tag]:
            continue
        maze = MZ.Maze(n)
        net = P.load_field(d['_dir'] / 'training/checkpoint.pt')
        b, ref = phases(maze, Path(d['prepared']), P.field_cost(net))
        cl, allcl = phases.clearance
        margin = MZ.MARGIN_CELLS
        res[tag] = dict(n=n, field=key, seed=d['seed'], id=d['_id'], bumps=stats(b), path=stats(ref),
                        ripple=ripple(maze, Path(d['prepared']), P.field_cost(net)),
                        near_wall=dict(minima_within_margin=int((cl < margin).sum()), minima=int(len(cl)),
                                       path_share_within_margin=float((allcl < margin).mean())))
        print(tag, res[tag]['bumps'].get('R'), {k: v for k, v in res[tag]['ripple'].items() if k != 'phase_profile'}, flush=True)
        out_json.write_text(json.dumps(res, indent=1))
        del net
        torch.cuda.empty_cache()
    md = ['# Local-minimum position within the cell (generated by bumpphase.py)', '',
          'Local minima of T(x, goal) along the true paths of queries >= 8 cells; phase = position along the direction of travel modulo one cell '
          '(0 = grid line, 0.5 = cell centre). R = concentration of the phases (0 = even; chance level about 1/sqrt(count)). Path reference: all '
          'path samples. Ripple: the field error along the true paths minus its two-cell moving average, relative to the query travel time; locked share = '
          'fraction of the ripple variance explained by the position within the cell.', '',
          '| maze | field | seed | minima | R (chance) | mean phase | histogram (10 bins from the grid line) | ripple rms | locked share | minima inside the wall margin |', '|---|---|---|---|---|---|---|---|---|---|']
    for tag, r in sorted(res.items(), key=lambda kv: (kv[1]['n'], kv[1]['field'], kv[1]['seed'])):
        b, rp = r['bumps'], r.get('ripple', {})
        nw = r.get('near_wall', {})
        tail = (f' {rp["ripple_rms_rel"]:.4f} | {rp["locked_share"]:.3f} |' if rp else ' – | – |') + \
            (f' {nw["minima_within_margin"]}/{nw["minima"]} (path {100 * nw["path_share_within_margin"]:.0f} %) |' if nw else ' – |')
        if b['count'] == 0:
            md.append(f'| {r["n"]}×{r["n"]} | {r["field"]} | {r["seed"]} | 0 | – | – | – |' + tail)
            continue
        md.append(f'| {r["n"]}×{r["n"]} | {r["field"]} | {r["seed"]} | {b["count"]} | {b["R"]:.2f} ({b["chance_R"]:.2f}) | {b["mean_phase"]:.2f} | {" ".join(map(str, b["hist10"]))} |' + tail)
    ref = next(iter(res.values()))['path'] if res else None
    if ref:
        md += ['', f'Path samples (reference, {next(iter(res.values()))["n"]}×{next(iter(res.values()))["n"]}): R {ref["R"]:.2f}, histogram {" ".join(map(str, ref["hist10"]))}']
    (GR.OUT / 'bumpphase.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
