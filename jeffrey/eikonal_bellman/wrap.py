"""Where does a field fail? (unregistered diagnostic) Probe pairs of the ground truth are split, per gap range, into
'direct' pairs (the straight line is nearly optimal: straight-line cost / T* <= --direct) and 'detour' pairs (the optimal
path is much cheaper than the straight line: on no-wall these wrap around the orientation manifold, on the cabinet they
go around the cupboard), and, independently, into 'fast' pairs (T* / gap <= --slow) and 'slow' pairs (T* / gap > --slow:
on no-wall these are the pairs whose optimal path wraps around the orientation manifold). Reports the median field / T*
and the mean |log| error for every group and field.
"""
import sys
sys.path.insert(0, '/clean/method'); sys.path.insert(0, '/clean/jeffrey/segment_dp')
import argparse, json
from pathlib import Path
import numpy as np
import torch
from fields import load_field


def main():
    p = argparse.ArgumentParser(); p.add_argument('--truth', type=Path, required=True); p.add_argument('--fields', type=Path, nargs='+', required=True)
    p.add_argument('--labels', nargs='+', required=True); p.add_argument('--output', type=Path, required=True); p.add_argument('--direct', type=float, default=1.15); p.add_argument('--slow', type=float, default=1.8)
    p.add_argument('--ranges', nargs='*', default=['0.16,0.4', '0.4,0.56', '0.56,0.8', '0.8,1.6']); a = p.parse_args()
    t = np.load(a.truth); gap, truth, straight = t['gap'], t['truth'], t['straight']; ratio_s = straight/truth
    q = torch.as_tensor(t['q'], device='cuda:0'); g = torch.as_tensor(t['g'], device='cuda:0')
    ranges = [tuple(float(x) for x in r.split(',')) for r in a.ranges]; report = dict(direct_threshold=a.direct, ranges=ranges, fields={})
    speed = truth/gap
    for lo, hi in ranges:
        sel = (gap > lo) & (gap <= hi); d = sel & (ratio_s <= a.direct); w = sel & (ratio_s > a.direct); f = sel & (speed <= a.slow); sl = sel & (speed > a.slow)
        report.setdefault('groups', []).append(dict(range=[lo, hi], pairs=int(sel.sum()), detour_fraction=float(w.sum()/max(1, sel.sum())), slow_fraction=float(sl.sum()/max(1, sel.sum())),
                                                    truth_over_gap_direct=float(np.median(speed[d])) if d.any() else None, truth_over_gap_detour=float(np.median(speed[w])) if w.any() else None,
                                                    truth_over_gap_fast=float(np.median(speed[f])) if f.any() else None, truth_over_gap_slow=float(np.median(speed[sl])) if sl.any() else None))
    for label, path in zip(a.labels, a.fields):
        field = load_field(path, 'cuda:0')
        with torch.no_grad():
            v = torch.cat([field(q[i:i+8192], g[i:i+8192])[:, 0] for i in range(0, len(q), 8192)]).cpu().numpy()
        rows = []
        def group(mask, r):
            return (float(np.median(r[mask])), float(np.abs(np.log(r[mask])).mean())) if mask.any() else (None, None)
        for lo, hi in ranges:
            sel = (gap > lo) & (gap <= hi); r = v/truth; row = dict(range=[lo, hi])
            for name, mask in (('direct', sel & (ratio_s <= a.direct)), ('detour', sel & (ratio_s > a.direct)), ('fast', sel & (speed <= a.slow)), ('slow', sel & (speed > a.slow))):
                row[name+'_median'], row[name+'_abs_log'] = group(mask, r)
            rows.append(row)
        report['fields'][label] = rows; print(label, json.dumps([{k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items()} for r in rows]), flush=True)
        del field; torch.cuda.empty_cache()
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(report, indent=1))


if __name__ == '__main__': main()
