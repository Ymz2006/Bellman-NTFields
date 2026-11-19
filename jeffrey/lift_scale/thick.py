"""Read-only forensics: does a field see through an obstacle according to how much obstacle its straight segment crosses?

For the detour pairs of diag.truth_pairs (free configuration, goal, T* >= 1.3 D): the length of the straight C-space segment that lies
inside obstacles (clearance < 0; xy length in cells, sampled every 0.05 cells), the longest contiguous part of it (the thickest single
obstacle the segment crosses) and the detour index rho = (T - D) / (T* - D) of the
field. Reported by that length: pairs, median rho, share seen through (rho < 0.5).

    CUDA_VISIBLE_DEVICES=1 python jeffrey/lift_scale/thick.py --out jeffrey/lift_scale/report/thick.json <run dir> ...   (one scene per process)
"""
import argparse
import json
import sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lift as LF  # noqa: E402
import scenes as SN  # noqa: E402
import diag as DG  # noqa: E402
import plan as P  # noqa: E402

BINS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 1e9]
TB = [0.0, 0.35, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 8.0, 1e9]
_cache = {}


def labels():
    return [f'{BINS[i]:g}-{BINS[i + 1]:g}' if BINS[i + 1] < 1e8 else f'>{BINS[i]:g}' for i in range(len(BINS) - 1)]


def pairs_of(scene, prep):
    key = str(prep)
    if key not in _cache:
        truth = DG.Truth(scene, prep)
        pairs, gi, t_star, D = DG.truth_pairs(scene, truth, 24000)
        det = t_star >= 1.3 * D
        pairs, t_star, D = pairs[det], t_star[det], D[det]
        inside = torch.zeros(len(pairs), device=scene.device)
        thickest = torch.zeros(len(pairs), device=scene.device)
        step = 0.05
        n = torch.ceil(torch.linalg.vector_norm(pairs[:, :2] - pairs[:, 3:5], dim=1) * scene.E / step).long().clamp(min=2)
        for i in range(0, len(pairs), 2000):
            idx = torch.arange(i, min(i + 2000, len(pairs)), device=scene.device)
            k = n[idx]
            rep = torch.repeat_interleave(torch.arange(len(idx), device=scene.device), k)
            first = torch.cumsum(k, 0) - k
            t = (torch.arange(int(k.sum()), device=scene.device) - first[rep] + 0.5) / k[rep]
            q = pairs[idx][rep, :3] + t[:, None] * (pairs[idx][rep, 3:] - pairs[idx][rep, :3])
            d = scene.clearance(q)
            fl = d < 0
            cnt = torch.zeros(len(idx), device=scene.device).index_add_(0, rep, fl.float())
            seg = torch.linalg.vector_norm(pairs[idx][:, :2] - pairs[idx][:, 3:5], dim=1) * scene.E
            inside[idx] = cnt / k * seg
            # the longest contiguous run inside an obstacle (the thickest single obstacle the segment crosses)
            prev = torch.cat([torch.zeros(1, dtype=torch.bool, device=scene.device), fl[:-1]])
            newseg = torch.cat([torch.ones(1, dtype=torch.bool, device=scene.device), rep[1:] != rep[:-1]])
            start = fl & (~prev | newseg)
            run = torch.cumsum(start.long(), 0) - 1
            if bool(fl.any()):
                length = torch.bincount(run[fl], minlength=int(start.sum())).float()
                owner = rep[start]
                best = torch.zeros(len(idx), device=scene.device).scatter_reduce_(0, owner, length, reduce='amax')
                thickest[idx] = best / k * seg
        _cache[key] = (pairs, t_star, D, inside, thickest)
    return _cache[key]


def analyse(run):
    res = json.load(open(run / 'results.json'))
    prep = Path(res['prepared'])
    scene, info = SN.load_scene(prep)
    pairs, t_star, D, inside, thickest = pairs_of(scene, prep)
    net = LF.load_field(run / 'training/checkpoint.pt', scene)
    t = P.field_cost(net)(pairs)
    rho = (t - D) / (t_star - D)
    out = dict(run=run.name.replace('registered_', ''), scene=res['scene_name'], lift=res['args']['lift'], push=res['args']['push'], octaves=res['args'].get('octaves', ''), pairs=int(len(pairs)),
               inside_median=float(inside.median()), inside_q10=float(inside.quantile(0.1)), inside_q90=float(inside.quantile(0.9)), by_inside={})
    for i, lab in enumerate(labels()):
        m = (inside >= BINS[i]) & (inside < BINS[i + 1])
        if int(m.sum()) > 15:
            out['by_inside'][lab] = dict(n=int(m.sum()), share=float(m.float().mean()), rho=float(rho[m].median()), seen_through=float((rho[m] < 0.5).float().mean()),
                                         detour_cells=float(((t_star[m] - D[m]) / scene.pitch).median()))
    out.update(thickest_median=float(thickest.median()), thickest_q10=float(thickest.quantile(0.1)), thickest_q90=float(thickest.quantile(0.9)), by_thickest={})
    for i in range(len(TB) - 1):
        m = (thickest >= TB[i]) & (thickest < TB[i + 1])
        if int(m.sum()) > 15:
            out['by_thickest'][f'{TB[i]:g}-{TB[i + 1]:g}' if TB[i + 1] < 1e8 else f'>{TB[i]:g}'] = dict(n=int(m.sum()), share=float(m.float().mean()), mid=float(thickest[m].median()), rho=float(rho[m].median()),
                                                                                                    seen_through=float((rho[m] < 0.5).float().mean()), detour_cells=float(((t_star[m] - D[m]) / scene.pitch).median()))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+')
    p.add_argument('--out', type=Path, default=None)
    a = p.parse_args()
    rows = []
    for path in a.runs:
        r = analyse(Path(path))
        rows.append(r)
        print(f"{r['scene']} | lift {r['lift']} push {r['push']} oct {r['octaves']} | {r['pairs']} detour pairs, obstacle length on the straight segment: median {r['inside_median']:.2f} cells (q10 {r['inside_q10']:.2f}, q90 {r['inside_q90']:.2f})")
        print(f"   thickest single obstacle crossed: median {r['thickest_median']:.2f} cells (q10 {r['thickest_q10']:.2f}, q90 {r['thickest_q90']:.2f}); " + '; '.join(f"{k}: n {v['n']}, rho {v['rho']:.2f}, seen {100 * v['seen_through']:.0f} %" for k, v in r['by_thickest'].items()))
        print('   by obstacle length: ' + '; '.join(f"{k}: n {v['n']}, rho {v['rho']:.2f}, seen through {100 * v['seen_through']:.0f} %" for k, v in r['by_inside'].items()))
    if a.out:
        prev = json.load(open(a.out)) if a.out.exists() else []
        keys = {r['run'] for r in rows}
        json.dump([r for r in prev if r['run'] not in keys] + rows, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
