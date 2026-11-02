"""Does a segment field keep the direction term (1.001 - S)|S grad T + n|^2 at the pool poses, level by level?
Pairs (q, g) of pool poses at distances in the field's level bins; residual at q with the frozen speed net's unit
gradient as n (the eikonal specialists' --normal-weight term), reported per distance bin for every field.
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/easy_cabinet/direction_residual.py \
      --prepared results/registered_.../prepared label=field.pt ...
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'method'), str(ROOT/'runtime/method'), str(ROOT/'jeffrey/segment_dp')]
import common; common.RELEASE = ROOT/'runtime'
from sampler import load_pool
from fields import load_field


def main():
    p = argparse.ArgumentParser(); p.add_argument('--prepared', type=Path, required=True); p.add_argument('fields', nargs='+')
    p.add_argument('--bins', default='0.02,0.08,0.16,0.4,0.8,1.6'); p.add_argument('--per-bin', type=int, default=20000); p.add_argument('--out', type=Path)
    a = p.parse_args(); dev = 'cuda:0'; torch.manual_seed(0)
    pool, speeds, speed_net, _ = load_pool(a.prepared, dev); fields = {k: load_field(Path(v), dev) for k, v in (f.split('=', 1) for f in a.fields)}
    slow = (speeds < 0.9).nonzero().reshape(-1)                      # poses near an obstacle: where the term is active
    x = pool[slow].clone().requires_grad_(True)
    with torch.enable_grad(): gs = torch.autograd.grad(speed_net(x).sum(), x)[0]
    normals = gs/gs.norm(dim=1, keepdim=True).clamp_min(1e-8); S = speeds[slow]
    bins = [float(b) for b in a.bins.split(',')]; report = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        # random pool goals at distance in [lo, hi) from slow poses: sample candidates and keep those in range
        qi = torch.randint(len(slow), (a.per_bin*20,), device=dev); gi = torch.randint(len(pool), (a.per_bin*20,), device=dev)
        d = (pool[slow][qi]-pool[gi]).norm(dim=1); keep = ((d >= lo) & (d < hi)).nonzero().reshape(-1)[:a.per_bin]
        qi, gi = qi[keep], gi[keep]; q = pool[slow][qi]; g = pool[gi]; n = normals[qi]; s = S[qi].view(-1, 1)
        row = dict(n=int(len(keep)))
        for name, field in fields.items():
            res, cosv = [], []
            for i in range(0, len(q), 8192):
                xq = q[i:i+8192].clone().requires_grad_(True)
                with torch.enable_grad(): grad = torch.autograd.grad(field(xq, g[i:i+8192]).sum(), xq)[0]
                res.append(((1.001-s[i:i+8192])*((s[i:i+8192]*grad+n[i:i+8192])**2).sum(1, keepdim=True)).reshape(-1))
                u = grad/grad.norm(dim=1, keepdim=True).clamp_min(1e-8); cosv.append(-(u*n[i:i+8192]).sum(1))   # cos(angle between -grad T and the wall-away normal)
            res = torch.cat(res); cosv = torch.cat(cosv)
            row[name] = dict(residual_mean=float(res.mean()), cos_mean=float(cosv.mean()), into_wall_share=float((cosv < 0).float().mean()))
        report[f'{lo}-{hi}'] = row; print(f'{lo}-{hi}', json.dumps(row), flush=True)
    if a.out: a.out.write_text(json.dumps(report, indent=1))


if __name__ == '__main__': main()
