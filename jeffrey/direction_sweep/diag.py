"""Accuracy diagnostics of one TravelCost field trained with D direction pairs (never selects a model).

On --samples bank sources q with random bank goals g (fixed seeds, the same pairs for every field):
  speed_ratio      S(q) |grad_q T(q,g)|: the field's implied speed 1/|grad T| against the speed network S it is meant
                   to integrate (1 = exact; > 1: the field is too steep, i.e. it predicts a slower speed than S)
  backup_ratio_D'  T(q,g) / min_a [c(q,a) + T(z_a,g)] over the safe actions of the first D' pairs, for D' in --pairs:
                   self-consistency with a Bellman min over D' directions (1 = fixed point of that min)
  winner_cosine    cos(-grad_q T, z_win - q) for the D'-pair winner
On the ground-truth pairs of jeffrey/segment_dp/truth.py (sources are bank rows):
  coverage         best cosine between the optimal initial direction at q and a safe action direction of the first D'
                   pairs (field independent: how well D' random directions can follow the optimal path)
"""
import sys
sys.path.insert(0, '/clean/method'); sys.path.insert(0, '/clean/jeffrey/segment_dp')
import argparse, json
from pathlib import Path
import numpy as np
import torch
from common import sha, write
from fields import load_field
from speed import load_speed

PAIRS = (1, 2, 4, 8, 16)


def stats(x):
    x = np.asarray(x, np.float64); x = x[np.isfinite(x)]
    return dict(n=int(len(x)), median=float(np.median(x)), mean_abs_log=float(np.abs(np.log(x)).mean()), p10=float(np.quantile(x, .1)), p90=float(np.quantile(x, .9)),
                within_5pct=float((np.abs(np.log(x)) <= np.log(1.05)).mean()))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--checkpoint', type=Path, required=True); p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--ext', type=Path, required=True); p.add_argument('--truth', type=Path, required=True); p.add_argument('--samples', type=int, default=16384)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    ext = a.ext/'ext'; P = json.loads((ext/'manifest.json').read_text())['pairs']; pairs = [d for d in PAIRS if d <= P]
    source = np.load(a.prepared/'bank/source_q.npy', mmap_mode='r'); succ, cost, safe = [np.load(ext/(k+'.npy'), mmap_mode='r') for k in ('successor_q', 'local_cost', 'safe')]
    field = load_field(a.checkpoint, 'cuda:0'); speed = load_speed(a.prepared/'speed.pt', 'cuda:0')
    rng = np.random.default_rng(5); n_ext = len(succ); rows = np.sort(rng.choice(n_ext, a.samples, replace=False)); goals = rng.choice(len(source), a.samples, replace=False)
    report = dict(checkpoint=str(a.checkpoint), checkpoint_sha256=sha(a.checkpoint), samples=a.samples, pairs=pairs)
    ratio, backup, cosine, gaps = [], {d: [] for d in pairs}, {d: [] for d in pairs}, []
    for b in range(0, a.samples, 512):
        r = rows[b:b+512]; n = len(r)
        q = torch.tensor(np.asarray(source[r]), device='cuda:0').requires_grad_(True); g = torch.tensor(np.asarray(source[goals[b:b+512]]), device='cuda:0')
        t = field(q, g)[:, 0]; grad = torch.autograd.grad(t.sum(), q)[0]; t = t.detach(); q = q.detach()
        with torch.no_grad():
            ratio.append((speed(q).reshape(-1)*torch.linalg.vector_norm(grad, dim=1)).cpu().numpy()); gaps.append(torch.linalg.vector_norm(g-q, dim=1).cpu().numpy())
            zall = torch.tensor(np.asarray(succ[r]), device='cuda:0'); call = torch.tensor(np.asarray(cost[r]), device='cuda:0'); sall = torch.tensor(np.asarray(safe[r]), device='cuda:0')
            tail = field(zall.reshape(-1, 6), g[:, None, :].expand(n, 8*P, 6).reshape(-1, 6)).reshape(n, 4, 2, P)
            for d in pairs:
                cand = (call[..., :d]+tail[..., :d]).masked_fill(~sall[..., :d], torch.inf).reshape(n, -1)
                best, win = cand.min(1); ok = torch.isfinite(best)
                delta = zall[..., :d, :].reshape(n, -1, 6)[torch.arange(n), win]-q
                cos = (-grad*delta).sum(1)/(torch.linalg.vector_norm(grad, dim=1)*torch.linalg.vector_norm(delta, dim=1)).clamp_min(1e-12)
                backup[d].append(torch.where(ok, t/best, torch.nan).cpu().numpy()); cosine[d].append(torch.where(ok, cos, torch.nan).cpu().numpy())
    ratio = np.concatenate(ratio); gaps = np.concatenate(gaps)
    report['speed_ratio'] = stats(ratio); report['speed_ratio_by_gap'] = {}
    for lo, hi in ((0, .3), (.3, .6), (.6, .9), (.9, 9)):
        sel = (gaps >= lo) & (gaps < hi)
        if sel.sum() >= 50: report['speed_ratio_by_gap'][f'{lo}-{hi}'] = stats(ratio[sel])
    report['backup_ratio'] = {d: stats(np.concatenate(backup[d])) for d in pairs}
    report['winner_cosine'] = {d: float(np.nanmean(np.concatenate(cosine[d]))) for d in pairs}
    # Field-independent direction coverage of the ground-truth optimal directions.
    data = np.load(a.truth); keep = data['source_index'] < n_ext   # all pairs unless a smoke extension
    src = data['source_index'][keep]; direction = data['direction'][keep]; seg = data['segment'][keep]; edges = data['edges']
    uq, inv = np.unique(src, return_inverse=True); q0 = np.asarray(source[uq])[inv]; assert np.allclose(q0, data['q'][keep], atol=1e-6)
    z = np.asarray(succ[uq])[inv][:, 3]; s = np.asarray(safe[uq])[inv].any(1)   # (m,2,P,6) at the largest step; safe at any step
    unit = z-q0[:, None, None, :]; unit /= np.linalg.norm(unit, axis=-1, keepdims=True)
    cos = np.einsum('mspk,mk->msp', unit, direction/np.linalg.norm(direction, axis=1, keepdims=True))
    report['coverage'] = {}
    for d in pairs:
        c = np.where(s[:, :, :d], cos[:, :, :d], -1.).reshape(len(src), -1).max(1)
        report['coverage'][d] = dict(mean_best_cosine=float(c.mean()), by_segment=[float(c[seg == k].mean()) if (seg == k).any() else None for k in range(len(edges)-1)])
    report['complete'] = True; write(a.output, report); print(json.dumps({k: v for k, v in report.items() if k in ('speed_ratio', 'winner_cosine')}), flush=True)


if __name__ == '__main__': main()
