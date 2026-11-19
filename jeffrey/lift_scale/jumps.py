"""Read-only forensics on registered lift-scale fields: how large a step in travel time each field holds across a wall.

For band configurations a (held-out x0 of the prepare run) the mirror configuration b is found on the other side of the nearest wall:
b = a - s n_xy (same heading; n_xy the planar part of the clearance gradient), s the smallest step in [0.2, 1.6] cells for which b is
free with at least 0.8 of a's clearance and the straight segment a-b is blocked. For goals g of the prepare run the truth has a step
D* = |T*(a, g) - T*(b, g)| across that wall (0 when the goal is reached the same way from both sides, the whole detour when the wall
separates them); the field's step is D = |T(a, g) - T(b, g)|. Reported by D* (cells): median D / D*, the median and the 95th percentile
of D, and the distance between a and b in the network's input (cells), which is what the step has to be built over.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/lift_scale/jumps.py --out jeffrey/lift_scale/report/jumps.json <run dir>[@epoch] ...
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lift as LF  # noqa: E402
import scenes as SN  # noqa: E402
import diag as DG  # noqa: E402

BINS = [0.0, 0.5, 1.5, 3.0, 6.0, 12.0, 24.0, 1e9]
_cache = {}


def labels():
    return [f'{BINS[i]:g}-{BINS[i + 1]:g}' if BINS[i + 1] < 1e8 else f'>{BINS[i]:g}' for i in range(len(BINS) - 1)]


def wall_pairs(scene, prepared, n=6000, goals_per=6, seed=3):
    key = str(prepared)
    if key in _cache:
        return _cache[key]
    dev = scene.device
    truth = DG.Truth(scene, prepared)
    a = torch.tensor(np.load(Path(prepared) / 'heldout/sampled_points.npy')[:n, :3], device=dev)
    x = a.clone().requires_grad_(True)
    with torch.enable_grad():
        d = scene.signed_cells(x)
        g, = torch.autograd.grad(d.sum(), x)
    da = d.detach()
    nxy = torch.nn.functional.normalize(g[:, :2].detach(), dim=1)
    b = torch.full_like(a, float('nan'))
    found = torch.zeros(len(a), dtype=torch.bool, device=dev)
    for s in torch.linspace(0.2, 1.6, 29).tolist():
        cand = a.clone()
        cand[:, :2] = a[:, :2] - s * scene.pitch * nxy
        dc = scene.clearance(cand)
        ok = ~found & (dc >= 0.8 * da) & (dc > 0.02) & scene.inside(cand)
        if bool(ok.any()):
            idx = torch.nonzero(ok).squeeze(1)
            blk = LF.segment_blocked(scene, torch.cat([a[idx], cand[idx]], 1), spacing_cells=0.02)
            idx = idx[blk]
            b[idx] = cand[idx]
            found[idx] = True
    a, b, da = a[found], b[found], da[found]
    gen = torch.Generator(dev).manual_seed(seed)
    gi = torch.randint(0, len(truth.goals), (len(a), goals_per), generator=gen, device=dev)
    A = a[:, None].expand(-1, goals_per, -1).reshape(-1, 3)
    Bq = b[:, None].expand(-1, goals_per, -1).reshape(-1, 3)
    gi = gi.reshape(-1)
    ta, tb = truth.time(gi, A), truth.time(gi, Bq)
    keep = torch.isfinite(ta) & torch.isfinite(tb)
    out = dict(A=A[keep], B=Bq[keep], goal=truth.goals[gi[keep]], step_true=(ta - tb).abs()[keep] / scene.pitch,
               planar_gap=torch.linalg.vector_norm(A[keep][:, :2] - Bq[keep][:, :2], dim=1) / scene.pitch, share_found=float(found.float().mean()))
    _cache[key] = out
    return out


def analyse(run, epoch=None):
    res = json.load(open(run / 'results.json'))
    prepared = Path(res['prepared'])
    scene, info = SN.load_scene(prepared)
    W = wall_pairs(scene, prepared)
    ck = run / 'training/checkpoint.pt' if epoch is None else run / f'training/checkpoints/epoch_{epoch:05d}.pt'
    net = LF.load_field(ck, scene)
    with torch.no_grad():
        ta = net.out(torch.cat([W['A'], W['goal']], 1))[0][:, 0]
        tb = net.out(torch.cat([W['B'], W['goal']], 1))[0][:, 0]
        # distance between the two sides in the network's input coordinates (cells)
        if net.lift_kind == 'none':
            extra = torch.zeros(len(W['A']), device=scene.device)
        elif net.lift_kind == 'vec':
            extra = torch.linalg.vector_norm(net.vector(W['A']) - net.vector(W['B']), dim=1) * scene.E
        else:
            extra = (net.height(W['A']) - net.height(W['B'])).abs() * scene.E
    step = (ta - tb).abs() / scene.pitch
    st = W['step_true']
    input_gap = torch.sqrt(W['planar_gap'] ** 2 + extra ** 2)
    out = dict(run=run.name.replace('registered_', ''), epoch=epoch, scene=res['scene_name'], E=scene.E, lift=res['args']['lift'], push=res['args']['push'],
               pairs=int(len(st)), share_found=W['share_found'], planar_gap_median=float(W['planar_gap'].median()), input_gap_median=float(input_gap.median()),
               step_q95=float(step.quantile(0.95)), step_q99=float(step.quantile(0.99)), by_true={})
    for i, lab in enumerate(labels()):
        m = (st >= BINS[i]) & (st < BINS[i + 1])
        if int(m.sum()) > 15:
            out['by_true'][lab] = dict(n=int(m.sum()), true=float(st[m].median()), learned=float(step[m].median()), learned_q90=float(step[m].quantile(0.9)),
                                       ratio=float((step[m] / st[m].clamp(min=1e-6)).median()) if BINS[i] > 0 else None)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+')
    p.add_argument('--out', type=Path, default=None)
    a = p.parse_args()
    rows = []
    for spec in a.runs:
        path, _, ep = spec.partition('@')
        r = analyse(Path(path), int(ep) if ep else None)
        rows.append(r)
        print(f"{r['scene']} | lift {r['lift']} push {r['push']} | epoch {r['epoch']} | {r['pairs']} pairs, planar gap {r['planar_gap_median']:.2f} cells, input gap {r['input_gap_median']:.2f}, "
              f"step q95 {r['step_q95']:.1f} q99 {r['step_q99']:.1f} cells")
        print('   true step -> learned (median, q90): ' + '; '.join(f"{k}: {v['true']:.1f} -> {v['learned']:.1f} ({v['learned_q90']:.1f})" for k, v in r['by_true'].items()))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        prev = json.load(open(a.out)) if a.out.exists() else []
        keys = {(r['run'], r['epoch']) for r in rows}
        json.dump([r for r in prev if (r['run'], r['epoch']) not in keys] + rows, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
