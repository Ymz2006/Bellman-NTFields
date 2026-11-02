"""Does the field invite leaving the free manifold? (unregistered diagnostic) For pool pairs (q, g) with gap in --range,
q is moved by a random step of norm --step. Steps that lower the speed S(q) by more than --drop leave the fast region
(off the orientation manifold on no-wall, toward an obstacle on the cabinet). Reports, per field, the fraction of such
steps on which T(q + d, g) < T(q, g) (the planner would be rewarded for leaving), the mean dT/|d| on leaving steps, and
the same for steps that keep S (|dS| < drop/4)."""
import sys
sys.path.insert(0, '/clean/method'); sys.path.insert(0, '/clean/jeffrey/segment_dp')
import argparse, json
from pathlib import Path
import torch
from fields import load_field
from sampler import PairSampler, load_pool


def main():
    p = argparse.ArgumentParser(); p.add_argument('--prepared', type=Path, required=True); p.add_argument('--fields', type=Path, nargs='+', required=True)
    p.add_argument('--labels', nargs='+', required=True); p.add_argument('--output', type=Path, required=True); p.add_argument('--pairs', type=int, default=16384)
    p.add_argument('--range', default='0.2,0.6'); p.add_argument('--step', type=float, default=0.03); p.add_argument('--drop', type=float, default=0.2); a = p.parse_args()
    dev = 'cuda:0'; generator = torch.Generator(device=dev).manual_seed(0); pool, speeds, speed, _ = load_pool(a.prepared, dev)
    lo, hi = (float(x) for x in a.range.split(',')); sampler = PairSampler(pool, [lo, hi], 0.0, generator)
    qi, gi, gap, _ = sampler.sample(torch.zeros(a.pairs, dtype=torch.long, device=dev)); q, g = pool[qi], pool[gi]
    d = torch.randn((a.pairs, 6), generator=generator, device=dev); d = d/torch.linalg.vector_norm(d, dim=1, keepdim=True)*a.step
    q2 = (q+d).clamp(-.5, .5)
    with torch.no_grad():
        s1 = speeds[qi]; s2 = speed(q2).reshape(-1); ds = s2-s1
    leave = ds < -a.drop; keep = ds.abs() < a.drop/4
    report = dict(pairs=a.pairs, range=[lo, hi], step=a.step, drop=a.drop, leaving_fraction=float(leave.float().mean()), keeping_fraction=float(keep.float().mean()), fields={})
    for label, path in zip(a.labels, a.fields):
        field = load_field(path, dev)
        with torch.no_grad():
            t1 = field(q, g)[:, 0]; t2 = field(q2, g)[:, 0]; dt = (t2-t1)/a.step
        row = dict(leave_rewarded=float((dt[leave] < 0).float().mean()), leave_mean_dT=float(dt[leave].mean()), keep_rewarded=float((dt[keep] < 0).float().mean()), keep_mean_dT=float(dt[keep].mean()),
                   leave_median_dT=float(dt[leave].median()))
        report['fields'][label] = row; print(label, json.dumps({k: round(v, 3) for k, v in row.items()}), flush=True)
        del field; torch.cuda.empty_cache()
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(report, indent=1))


if __name__ == '__main__': main()
