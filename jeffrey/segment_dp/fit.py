"""Fit of travel-cost fields to the ground truth of truth.py, per distance segment (diagnostic; never selects a model).

For every field (segment-field or TravelCost checkpoint) and every segment of the truth file:
  rel_error  mean |log(T / T*)|            (0.10: 10 % off on average)
  bias       median T / T*
  spearman   rank correlation of T / |q - g| with T* / |q - g| within the segment (does it know which pairs are slow)
  cosine     mean cos(-grad_q T, initial direction of the best path at q)
  eikonal    mean |log(S(q) |grad_q T|)|   (secondary)
Reference rows: 'straight' (the straight-line travel time, the best field without detours: its rel_error is how
much the optimal paths gain by bending) and 'euclid' (T = |q - g|: bias T*/|q-g| inverted, cosine of the straight
line). --by-model also scores every model of a segment field on every segment (how specific each model's fit is).
"""
import sys
sys.path.insert(0, '/clean/method'); sys.path.insert(0, '/clean/jeffrey/segment_dp')
import argparse, json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from common import sha, write
from fields import SegmentField, load_field
from speed import load_speed


def evaluate(field, q, g, chunk=4096):
    values, grads = [], []
    for i in range(0, len(q), chunk):
        x = q[i:i+chunk].clone().requires_grad_(True); v = field(x, g[i:i+chunk])[:, 0]
        grads.append(torch.autograd.grad(v.sum(), x)[0].detach()); values.append(v.detach())
    return torch.cat(values), torch.cat(grads)


def metrics(value, grad, truth, gap, segment, direction, speed_q, segments):
    cos = (-grad*direction).sum(1)/torch.linalg.vector_norm(grad, dim=1).clamp_min(1e-12)
    eik = torch.log((speed_q*torch.linalg.vector_norm(grad, dim=1)).clamp_min(1e-8)).abs()
    value, cos, eik = value.cpu().numpy(), cos.cpu().numpy(), eik.cpu().numpy()
    out = dict(rel_error=[], bias=[], spearman=[], cosine=[], eikonal=[])
    for s in range(segments):
        sel = segment == s
        if sel.sum() < 10:
            for k in out: out[k].append(None)
            continue
        ratio = value[sel]/truth[sel]
        out['rel_error'].append(float(np.abs(np.log(ratio)).mean())); out['bias'].append(float(np.median(ratio)))
        out['spearman'].append(float(spearmanr(value[sel]/gap[sel], truth[sel]/gap[sel]).statistic)); out['cosine'].append(float(cos[sel].mean()))
        out['eikonal'].append(float(eik[sel].mean()))
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('--truth', type=Path, required=True); p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--fields', type=Path, nargs='+', required=True); p.add_argument('--labels', nargs='+', required=True)
    p.add_argument('--by-model', nargs='*', default=[], help='labels of segment fields to also score model by model'); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); assert len(a.fields) == len(a.labels)
    data = np.load(a.truth); edges = data['edges']; segments = len(edges)-1; segment = data['segment']; gap = data['gap']; truth = data['truth']
    q = torch.as_tensor(data['q'], device='cuda:0'); g = torch.as_tensor(data['g'], device='cuda:0'); direction = torch.as_tensor(data['direction'], device='cuda:0')
    speed = load_speed(a.prepared/'speed.pt', 'cuda:0')
    with torch.no_grad():
        speed_q = torch.cat([speed(q[i:i+65536]).reshape(-1) for i in range(0, len(q), 65536)])
    t_truth = torch.as_tensor(truth, device='cuda:0'); t_gap = torch.as_tensor(gap, device='cuda:0')
    report = dict(truth=str(a.truth), truth_sha256=sha(a.truth), edges=edges.tolist(), pairs=[int((segment == s).sum()) for s in range(segments)],
                  truth_over_gap=[float(np.median(truth[segment == s]/gap[segment == s])) if (segment == s).any() else None for s in range(segments)], fields={}, by_model={})
    straight = torch.as_tensor(data['straight'], device='cuda:0'); line = (g-q)/t_gap[:, None]
    report['fields']['straight'] = metrics(straight, -line, truth, gap, segment, direction, speed_q, segments)
    report['fields']['euclid'] = metrics(t_gap, -line, truth, gap, segment, direction, speed_q, segments)
    for label, path in zip(a.labels, a.fields):
        field = load_field(path, 'cuda:0'); value, grad = evaluate(field, q, g)
        row = metrics(value, grad, truth, gap, segment, direction, speed_q, segments); row.update(path=str(path), sha256=sha(path))
        report['fields'][label] = row; print(label, json.dumps({k: [None if x is None else round(x, 3) for x in v] for k, v in row.items() if isinstance(v, list)}), flush=True)
        if label in a.by_model and isinstance(field, SegmentField):
            report['by_model'][label] = []
            for m, model in enumerate(field.models):
                value, grad = evaluate(model, q, g); r = metrics(value, grad, truth, gap, segment, direction, speed_q, segments)
                report['by_model'][label].append(dict(model=m+1, rel_error=r['rel_error'], bias=r['bias'], spearman=r['spearman'], cosine=r['cosine']))
        write(a.output, report)
        del field; torch.cuda.empty_cache()
    report['complete'] = True; write(a.output, report)


if __name__ == '__main__': main()
