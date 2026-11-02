"""Segment-edge consistency of a segment field (unregistered diagnostic): at every inner edge e_m, pool pairs with
|q - g| within 0.004 of e_m are evaluated by the model below and the model above the edge; reports mean and spread of
log(T_below / T_above), and the fraction of pairs where the inner (shorter-range) model reads higher (a barrier for
descent toward the goal). Also the field's eikonal residual per segment on random pool pairs (mean |log S|grad T||).
"""
import sys
sys.path.insert(0, '/clean/method'); sys.path.insert(0, '/clean/jeffrey/segment_dp')
import argparse, json
from pathlib import Path
import torch
from fields import load_field, SegmentField
from sampler import PairSampler, load_pool


def main():
    p = argparse.ArgumentParser(); p.add_argument('--prepared', type=Path, required=True); p.add_argument('--fields', type=Path, nargs='+', required=True)
    p.add_argument('--labels', nargs='+', required=True); p.add_argument('--output', type=Path, required=True); p.add_argument('--pairs', type=int, default=4096); a = p.parse_args()
    dev = 'cuda:0'; generator = torch.Generator(device=dev).manual_seed(0); pool, speeds, _, _ = load_pool(a.prepared, dev); report = {}
    for label, path in zip(a.labels, a.fields):
        field = load_field(path, dev); row = dict(edges=[], eikonal=[])
        if isinstance(field, SegmentField) and len(field.models) > 1:
            edges = field.edges; windows = [x for e in edges[1:-1] for x in (e-0.004, e+0.004)]
            sampler = PairSampler(pool, windows, 0.0, generator)          # even segments are the windows around the inner edges
            for m in range(1, len(edges)-1):
                e = edges[m]
                qi, gi, gap, _ = sampler.sample(torch.full((a.pairs,), 2*(m-1), dtype=torch.long, device=dev)); q, g = pool[qi], pool[gi]
                with torch.no_grad():
                    below = field.models[m-1](q, g)[:, 0]; above = field.models[m](q, g)[:, 0]
                r = (below/above).log()
                row['edges'].append(dict(edge=e, mean_log_below_over_above=float(r.mean()), std=float(r.std()), inner_higher=float((r > 0).float().mean()), gap_mean=float(gap.mean())))
        # eikonal residual of the whole field per 0.08 segment
        seg_edges = [0.08*i for i in range(21)]; sampler = PairSampler(pool, seg_edges, 0.0, generator)
        for s in range(20):
            qi, gi, gap, _ = sampler.sample(torch.full((a.pairs//4,), s, dtype=torch.long, device=dev)); q = pool[qi].clone().requires_grad_(True); g = pool[gi]
            v = field(q, g)[:, 0]; grad = torch.autograd.grad(v.sum(), q)[0]
            row['eikonal'].append(dict(segment=[seg_edges[s], seg_edges[s+1]], mean_abs_log=float(torch.log((speeds[qi]*torch.linalg.vector_norm(grad, dim=1)).clamp_min(1e-8)).abs().mean())))
        report[label] = row; print(label, json.dumps(row['edges'][:3]), flush=True)
        del field; torch.cuda.empty_cache()
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(report, indent=1))



if __name__ == '__main__': main()
