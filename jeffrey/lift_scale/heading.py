"""Read-only forensics on registered lift-scale fields: where the slope of the field goes on line-of-sight pairs.

For random free configurations q (clearance >= margin) and goals g with T* <= 1.05 D (D the straight C-space distance), by D in cells:
learned / true time, |grad_q T|, the radial slope grad T . u (u the unit vector from the goal to q; 1 for the true field), the planar
slope |grad_xy T| and the heading slope |dT/dphi|. A field can keep |grad T| = 1 while its radial slope falls if it puts slope into
the heading coordinate, whose whole range is 0.39 cells.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/lift_scale/heading.py --out jeffrey/lift_scale/report/heading.json <run dir> ...
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

BINS = [(0, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 200)]


def analyse(run, n=30000):
    res = json.load(open(run / 'results.json'))
    prep = Path(res['prepared'])
    scene, info = SN.load_scene(prep)
    truth = DG.Truth(scene, prep)
    net = LF.load_field(run / 'training/checkpoint.pt', scene)
    gen = torch.Generator(scene.device).manual_seed(11)
    q = scene.free(n, gen, 0.175)
    gi = torch.randint(0, len(truth.goals), (len(q),), generator=gen, device=scene.device)
    g = truth.goals[gi]
    ts = truth.time(gi, q)
    D = torch.linalg.vector_norm(q - g, dim=1)
    t, gr = P.field_grad(net, torch.cat([q, g], 1))
    los = torch.isfinite(ts) & (ts <= 1.05 * D)
    u = torch.nn.functional.normalize(q - g, dim=1)
    out = dict(run=run.name.replace('registered_', ''), scene=res['scene_name'], lift=res['args']['lift'], push=res['args']['push'], zero_heading=bool(res['args'].get('zero_heading')), by_distance={})
    for lo, hi in BINS:
        m = los & (D / scene.pitch >= lo) & (D / scene.pitch < hi)
        if int(m.sum()) < 30:
            continue
        gm = gr[m]
        out['by_distance'][f'{lo}-{hi}'] = dict(n=int(m.sum()), ratio=float((t[m] / ts[m]).median()), grad=float(torch.linalg.vector_norm(gm, dim=1).median()),
                                                radial=float((gm * u[m]).sum(1).median()), planar=float(torch.linalg.vector_norm(gm[:, :2], dim=1).median()),
                                                heading=float(gm[:, 2].abs().median()))
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
        print(f"{r['scene']} | lift {r['lift']} push {r['push']}" + (' heading-blind' if r['zero_heading'] else ''))
        for k, v in r['by_distance'].items():
            print(f"   D {k}: n {v['n']}, T/T* {v['ratio']:.3f}, |grad| {v['grad']:.3f}, radial {v['radial']:.3f}, planar {v['planar']:.3f}, heading {v['heading']:.3f}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        prev = json.load(open(a.out)) if a.out.exists() else []
        keys = {r['run'] for r in rows}
        json.dump([r for r in prev if r['run'] not in keys] + rows, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
