"""Close pairs: the travel time a field predicts against the true one. No training.

For a prepared scene (6 or 12 joints) and field runs of it: a = a valid configuration with its grasp point(s) in the task box
(scene.free, clearance >= contact), b = a + a random direction times a length of --cells cells (1 cell = 1 / E of the joint range),
kept if b is inside the joint limits and the straight step a -> b stays free. Truth = the travel time of that straight step at the
training speed (core.step_time, the step cost the Bellman update uses: length x mean of 1 / S). For steps this short the straight
step is the fastest path or within a few percent of it, so the truth is an upper bound that is tight for the shortest lengths.
Prediction = the network's T(a, b).

    CUDA_VISIBLE_DEVICES=1 python jeffrey/lbs_arm/closepairs.py --prepared <prepare run> --fields ntrl=<field run> lbs=<field run> --out <dir>
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--fields', nargs='+', required=True, help='name=<field run directory>')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cells', type=float, nargs='+', default=[0.25, 0.5, 1.0, 2.0, 4.0])
    p.add_argument('--pairs', type=int, default=5000, help='pairs kept per length')
    p.add_argument('--tries', type=int, default=40, help='batches of 2^16 starts tried per length at most')
    p.add_argument('--samples', type=int, default=65)
    p.add_argument('--seed', type=int, default=20261121)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    info = json.loads((a.prepared / 'scene.json').read_text())
    if info['describe']['dim'] > 6:
        import arm12
        sys.modules['arm'] = arm12
    import arm as A
    import core as C
    import trainer6 as TRN
    from common import Logger, dump
    a.out.mkdir(parents=True, exist_ok=True)
    log = Logger(a.out / 'log.txt')
    dev, t0 = 'cuda:0', time.time()
    scene, info = A.load_scene(a.prepared, dev)
    D = len(scene.lo)
    clear = C.Clearance(scene, compile=False)
    mn = TRN.ntrl_modules()[2]
    nets = {}
    for item in a.fields:
        name, run = item.split('=', 1)
        nets[name] = C.load_field(mn, Path(run) / 'training/checkpoint.pt', scene, dev)
    if a.smoke:
        a.pairs, a.tries = 200, 2
    gen = torch.Generator(dev).manual_seed(a.seed)
    res = dict(stage='lbsarm-closepairs', scene=scene.describe(), prepared=str(a.prepared), fields={i.split('=', 1)[0]: i.split('=', 1)[1] for i in a.fields}, smoke=a.smoke,
               E=scene.E, cell=1 / scene.E, samples=a.samples, lengths=[])
    keep_arrays = {}
    for cells in a.cells:
        r = cells / scene.E
        As, Bs, Ts, tried = [], [], [], 0
        for _ in range(a.tries):
            x = scene.free(1 << 16, gen, 0.0, clear=clear)
            dirs = torch.nn.functional.normalize(torch.randn(len(x), D, generator=gen, device=dev), dim=1)
            y = x + dirs * r
            tau, free = C.step_time(scene, x, y, a.samples, clear)
            ok = free & scene.inside(y)
            tried += len(x)
            As.append(x[ok]), Bs.append(y[ok]), Ts.append(tau[ok])
            if sum(len(v) for v in As) >= a.pairs:
                break
        xa, xb, tau = torch.cat(As)[:a.pairs], torch.cat(Bs)[:a.pairs], torch.cat(Ts)[:a.pairs]
        row = dict(cells=cells, length=r, pairs=len(xa), starts_tried=tried, kept_share=sum(len(v) for v in As) / tried)
        if len(xa):
            slow = tau / r                                   # mean of 1 / S along the step: 1 = full speed all the way
            q = lambda v: [float(t) for t in torch.quantile(v.double(), torch.tensor([.1, .5, .9], dtype=torch.float64, device=dev))]  # noqa: E731
            row.update(truth_quantiles=q(tau), slowness_quantiles=q(slow), full_speed_share=float((slow < 1.02).float().mean()), fields={})
            keep_arrays[f'a_{cells}'], keep_arrays[f'b_{cells}'], keep_arrays[f'truth_{cells}'] = xa.cpu().numpy(), xb.cpu().numpy(), tau.cpu().numpy()
            groups = dict(all=torch.ones_like(slow, dtype=torch.bool), full_speed=slow < 1.02, slowed=(slow >= 1.02) & (slow < 2), slow=slow >= 2)
            row['groups'] = {k: int(m.sum()) for k, m in groups.items()}
            for name, net in nets.items():
                with torch.no_grad():
                    T = torch.cat([net.out(torch.cat([xa[i:i + 4096], xb[i:i + 4096]], 1))[0][:, 0] for i in range(0, len(xa), 4096)])
                keep_arrays[f'pred_{name}_{cells}'] = T.cpu().numpy()
                ratio = T / tau
                out = {}
                for k, m in groups.items():
                    if int(m.sum()) >= 20:
                        out[k] = dict(n=int(m.sum()), predicted_quantiles=q(T[m]), ratio_quantiles=q(ratio[m]), within_20pct=float(((ratio[m] > 0.8) & (ratio[m] < 1.25)).float().mean()),
                                      mean_abs_log_ratio=float(torch.log(ratio[m].clamp(min=1e-6)).abs().mean()), nonpositive=int((T[m] <= 0).sum()))
                out['euclid_ratio_quantiles'] = q(r / tau)
                row['fields'][name] = out
                o = out['all']
                log(f'{cells:g} cells ({r:.4f}), {len(xa)} pairs (kept {row["kept_share"]:.4f}), {name}: predicted / true 10/50/90 % {[round(v, 3) for v in o["ratio_quantiles"]]}, within 20 % {o["within_20pct"]:.3f}; '
                    + '; '.join(f'{k} (n {out[k]["n"]}) median {out[k]["ratio_quantiles"][1]:.3f}' for k in ('full_speed', 'slowed', 'slow') if k in out))
        res['lengths'].append(row)
    np.savez_compressed(a.out / 'pairs.npz', **keep_arrays)
    res['elapsed_seconds'] = time.time() - t0
    dump(a.out / 'results.json', res)
    log(f'done in {res["elapsed_seconds"] / 60:.1f} min')


if __name__ == '__main__':
    main()
