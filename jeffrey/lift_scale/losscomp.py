"""Read-only forensics: ntrl-demo's loss, term by term (eikonal, normal, temporal difference; each with its exp(-0.5 T) weight), of a
registered field on the first 20,000 training pairs of its prepare run, for all pairs and for the pairs whose straight segment is
blocked / clear. Shows which term a field pays when it holds its detours.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/lift_scale/losscomp.py --out jeffrey/lift_scale/report/losscomp.json <run dir>[@epoch] ...
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


def terms(net, points, Yobs, normal, td_len=0.03, dim=3):
    tau, w, Xp = net.out(points)
    dtau, = torch.autograd.grad(tau.sum(), Xp)
    DT0, DT1 = dtau[:, :dim], dtau[:, dim:]
    S0, S1 = (DT0 ** 2).sum(1), (DT1 ** 2).sum(1)
    with torch.no_grad():
        length0 = td_len / Yobs[:, 0:1]
        X0 = Xp.clone().detach()
        X0[:, :dim] -= length0 * DT0 * Yobs[:, 0:1] ** 2
        t0 = net.out(X0)[0]
        length1 = td_len / Yobs[:, 1:2]
        X1 = Xp.clone().detach()
        X1[:, dim:] -= length1 * DT1 * Yobs[:, 1:2] ** 2
        t1 = net.out(X1)[0]
        tl0 = 1e-3 * ((tau - (t0 + length0)) ** 2).squeeze()
        tl1 = 1e-3 * ((tau - (t1 + length1)) ** 2).squeeze()
        tl0[tau[:, 0] < length0.squeeze()] = 0
        tl1[tau[:, 0] < length1.squeeze()] = 0
        r0 = torch.sqrt(Yobs[:, 0] * torch.sqrt(S0 + 1e-8))
        r1 = torch.sqrt(Yobs[:, 1] * torch.sqrt(S1 + 1e-8))
        eik = 1e-2 * ((r0 - 1) ** 2 + (r1 - 1) ** 2)
        n0 = (1.001 - Yobs[:, 0:1]) * (Yobs[:, 0:1] * DT0 + normal[:, :dim]) ** 2
        n1 = (1.001 - Yobs[:, 1:2]) * (Yobs[:, 1:2] * DT1 + normal[:, dim:]) ** 2
        nrm = 1e-3 * (n0.sum(1) + n1.sum(1))
        wgt = torch.exp(-0.5 * tau[:, 0])
        # where the TD step of the first end lands: clearance there
        land = net.scene.clearance(X0[:, :dim])
    return dict(eik=eik * wgt, normal=nrm * wgt, td=(tl0 + tl1) * wgt, T=tau[:, 0].detach(), td0=tl0 * wgt, land=land, eik_lo=1e-2 * (torch.relu(1 - r0) ** 2 + torch.relu(1 - r1) ** 2) * wgt,
                eik_hi=1e-2 * (torch.relu(r0 - 1) ** 2 + torch.relu(r1 - 1) ** 2) * wgt)


def analyse(run, epoch=None, n=20000):
    res = json.load(open(run / 'results.json'))
    prep = Path(res['prepared'])
    scene, info = SN.load_scene(prep)
    ck = run / 'training/checkpoint.pt' if epoch is None else run / f'training/checkpoints/epoch_{epoch:05d}.pt'
    net = LF.load_field(ck, scene)
    x = torch.tensor(np.load(prep / 'data/sampled_points.npy', mmap_mode='r')[:n], device=scene.device)
    sp = torch.tensor(np.load(prep / 'data/speed.npy', mmap_mode='r')[:n], device=scene.device)
    nr = torch.tensor(np.load(prep / 'data/normal.npy', mmap_mode='r')[:n], device=scene.device)
    sp = 1.025 * sp * sp * (2 - sp) * (2 - sp) + 1 - 1.025
    blk = LF.segment_blocked(scene, x)
    acc = {}
    for i in range(0, n, 4000):
        t = terms(net, x[i:i + 4000], sp[i:i + 4000], nr[i:i + 4000])
        for k, v in t.items():
            acc.setdefault(k, []).append(v)
    t = {k: torch.cat(v) for k, v in acc.items()}
    out = dict(run=run.name.replace('registered_', ''), epoch=epoch, scene=res['scene_name'], lift=res['args']['lift'], push=res['args']['push'])
    for name, m in (('all', torch.ones_like(blk)), ('blocked', blk), ('clear', ~blk)):
        out[name] = dict(share=float(m.float().mean()), eik=float(t['eik'][m].mean()), eik_low=float(t['eik_lo'][m].mean()), eik_high=float(t['eik_hi'][m].mean()),
                         normal=float(t['normal'][m].mean()), td=float(t['td'][m].mean()), T=float(t['T'][m].mean()),
                         td_first_end_lands_in_obstacle=float((t['land'][m] < 0).float().mean()))
        out[name]['total'] = out[name]['eik'] + out[name]['normal'] + out[name]['td']
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
        print(f"{r['scene']} | lift {r['lift']} push {r['push']} | epoch {r['epoch']}")
        for k in ('all', 'blocked', 'clear'):
            v = r[k]
            print(f"   {k:8s} share {v['share']:.2f}  total {v['total']:.2e} = eikonal {v['eik']:.2e} (slope low {v['eik_low']:.2e}, high {v['eik_high']:.2e}) + normal {v['normal']:.2e} + TD {v['td']:.2e};  mean T {v['T']:.3f};  TD step lands in obstacle {v['td_first_end_lands_in_obstacle']:.2f}")
    if a.out:
        prev = json.load(open(a.out)) if a.out.exists() else []
        keys = {(r['run'], r['epoch']) for r in rows}
        json.dump([r for r in prev if (r['run'], r['epoch']) not in keys] + rows, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
