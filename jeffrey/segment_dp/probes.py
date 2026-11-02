"""Field probes behind README section 6 (diagnostics on the host; they read finished runs and write nothing).

barrier  the eikonal slope near obstacles: start configurations of a truth file's segment-3/4 pairs perturbed by
         N(0, 0.04^2) per joint (the scale of MPPI's rollouts), the goal kept (gap still inside the segment); per bin
         of the frozen speed network S at the perturbed state, the median S|grad_q T| (eikonal: 1) and T/gap
ridge    monotonicity of T along rays: goals of the truth file's long pairs (gap >= 0.96), T(g + d u, g) for
         d = 0.04 .. 0.96 along the unit direction u towards the pair's start; the share of 0.02 steps where T falls
         going outwards, overall and at the 0.48 seam between segments 6 and 7 (a fall on most rays at one gap is a
         ridge that gradient descent from far away cannot cross)

    python3 jeffrey/segment_dp/probes.py barrier --truth results/<truth run>/truth/truth.npz \
        --prepared results/<pipeline run>/prepared --fields ES=results/<run>/training/field.pt DP=...
    python3 jeffrey/segment_dp/probes.py ridge --truth ... --fields ...
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(HERE/'method'), str(HERE/'runtime/method'), str(HERE/'jeffrey/segment_dp')]
import argparse, json
import numpy as np
import torch
import common
common.RELEASE = HERE/'runtime'   # the container's /repo
from fields import load_field
from fit import evaluate
from speed import load_speed


def barrier(t, fields, prepared, dev):
    speed = load_speed(prepared/'speed.pt', dev); torch.manual_seed(0); out = {}
    for seg in (2, 3):
        sel = np.where(t['segment'] == seg)[0]
        q0, g0 = torch.tensor(t['q'][sel], device=dev).repeat(16, 1), torch.tensor(t['g'][sel], device=dev).repeat(16, 1)
        q = q0+0.04*torch.randn_like(q0); gap = (q-g0).norm(dim=1)
        keep = (gap > 0.08*seg+0.01) & (gap < 0.08*(seg+1)-0.01); q, g = q[keep], g0[keep]
        with torch.no_grad(): s = speed(q).reshape(-1)
        b = np.digitize(s.cpu().numpy(), [0.1001, 0.5, 0.9]); row = dict(states=len(q), share=[float((b == k).mean()) for k in range(4)], fields={})
        for label, path in fields:
            f = load_field(path, dev); v, gr = evaluate(f, q, g)
            e = (s*gr.norm(dim=1)).cpu().numpy(); r = (v/(q-g).norm(dim=1)).cpu().numpy()
            row['fields'][label] = dict(slope=[float(np.median(e[b == k])) for k in range(4)], t_over_gap=[float(np.median(r[b == k])) for k in range(4)])
            del f; torch.cuda.empty_cache()
        out[f'segment {seg+1}'] = row
    return dict(bins=['speed floor 0.1', '0.1-0.5', '0.5-0.9', '> 0.9'], sigma=0.04, by_segment=out)


def ridge(t, fields, dev):
    sel = np.where(t['segment'] >= 12)[0][:2000]
    g = torch.tensor(t['g'][sel], device=dev); u = torch.tensor(t['q'][sel]-t['g'][sel], device=dev); u = u/u.norm(dim=1, keepdim=True)
    d = torch.arange(0.04, 0.97, 0.02, device=dev); out = {}
    for label, path in fields:
        f = load_field(path, dev)
        with torch.no_grad(): T = torch.stack([f(g+x*u, g)[:, 0] for x in d], 1).cpu().numpy()
        falls = np.diff(T, axis=1) <= 0
        out[label] = dict(t_over_gap=np.median(T/d.cpu().numpy(), 0)[::4].round(3).tolist(),
                          falls={'0.04-0.32': float(falls[:, :14].mean()), '0.32-0.64': float(falls[:, 14:30].mean()), '0.64-0.96': float(falls[:, 30:].mean())},
                          seam_0_48=float(falls[:, 21:23].mean()), monotone_rays=float((~falls).all(1).mean()))   # steps 0.46 -> 0.50: the edge between segments 6 and 7
        del f; torch.cuda.empty_cache()
    return dict(rays=len(sel), gaps=d[::4].round(decimals=2).tolist(), fields=out)


def main():
    p = argparse.ArgumentParser(); p.add_argument('probe', choices=['barrier', 'ridge']); p.add_argument('--truth', type=Path, required=True)
    p.add_argument('--prepared', type=Path); p.add_argument('--fields', nargs='+', required=True); a = p.parse_args()
    t = np.load(a.truth); fields = [(x.split('=', 1)[0], Path(x.split('=', 1)[1])) for x in a.fields]
    report = barrier(t, fields, a.prepared, 'cuda:0') if a.probe == 'barrier' else ridge(t, fields, 'cuda:0')
    print(json.dumps(dict(probe=a.probe, truth=str(a.truth), **report)))


if __name__ == '__main__': main()
