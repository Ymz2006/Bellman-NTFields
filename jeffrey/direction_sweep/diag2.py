"""Post-hoc version of diag.py for several checkpoints: D' up to 32, plus how often the direction term is active.

Same pairs and seeds as diag.py; extra per D': confident = fraction of sources whose two best candidates differ by
>= 10 % of the winner's local cost (the gate of method/loss.py direction_loss), winner_step = mean step size of the
winner. Output: {label: report}.

diag.py (original description):
Accuracy diagnostics of one TravelCost field trained with D direction pairs (never selects a model).

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

PAIRS = (1, 2, 4, 8, 16, 32)
STEPS = torch.tensor((.01, .05, .15, .25))


def stats(x):
    x = np.asarray(x, np.float64); x = x[np.isfinite(x)]
    return dict(n=int(len(x)), median=float(np.median(x)), mean_abs_log=float(np.abs(np.log(x)).mean()), p10=float(np.quantile(x, .1)), p90=float(np.quantile(x, .9)),
                within_5pct=float((np.abs(np.log(x)) <= np.log(1.05)).mean()))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--checkpoints', type=Path, nargs='+', required=True); p.add_argument('--labels', nargs='+', required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--ext', type=Path, required=True); p.add_argument('--truth', type=Path, required=True); p.add_argument('--samples', type=int, default=16384)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args(); assert len(a.checkpoints) == len(a.labels)
    out = {}
    for label, checkpoint in zip(a.labels, a.checkpoints):
        out[label] = one(a, checkpoint); write(a.output, out); print(label, json.dumps({d: round(v, 3) for d, v in out[label]['confident'].items()}), flush=True)


def one(a, checkpoint):
    ext = a.ext/'ext'; P = json.loads((ext/'manifest.json').read_text())['pairs']; pairs = [d for d in PAIRS if d <= P]
    source = np.load(a.prepared/'bank/source_q.npy', mmap_mode='r'); succ, cost, safe = [np.load(ext/(k+'.npy'), mmap_mode='r') for k in ('successor_q', 'local_cost', 'safe')]
    field = load_field(checkpoint, 'cuda:0'); speed = load_speed(a.prepared/'speed.pt', 'cuda:0')
    rng = np.random.default_rng(5); n_ext = len(succ); rows = np.sort(rng.choice(n_ext, a.samples, replace=False)); goals = rng.choice(len(source), a.samples, replace=False)
    report = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), samples=a.samples, pairs=pairs)
    ratio, backup, cosine, gaps, speeds, away = [], {d: [] for d in pairs}, {d: [] for d in pairs}, [], [], []; confident, step = {d: [] for d in pairs}, {d: [] for d in pairs}
    for b in range(0, a.samples, 512):
        r = rows[b:b+512]; n = len(r)
        q = torch.tensor(np.asarray(source[r]), device='cuda:0').requires_grad_(True); g = torch.tensor(np.asarray(source[goals[b:b+512]]), device='cuda:0')
        t = field(q, g)[:, 0]; grad = torch.autograd.grad(t.sum(), q)[0]; t = t.detach(); q = q.detach()
        with torch.enable_grad():   # direction of increasing speed (clearance) at q
            qs = q.clone().requires_grad_(True); gs = torch.autograd.grad(speed(qs).sum(), qs)[0]
        away.append(((-grad*gs).sum(1)/(torch.linalg.vector_norm(grad, dim=1)*torch.linalg.vector_norm(gs, dim=1)).clamp_min(1e-12)).cpu().numpy())
        with torch.no_grad():
            sq = speed(q).reshape(-1); speeds.append(sq.cpu().numpy()); ratio.append((sq*torch.linalg.vector_norm(grad, dim=1)).cpu().numpy()); gaps.append(torch.linalg.vector_norm(g-q, dim=1).cpu().numpy())
            zall = torch.tensor(np.asarray(succ[r]), device='cuda:0'); call = torch.tensor(np.asarray(cost[r]), device='cuda:0'); sall = torch.tensor(np.asarray(safe[r]), device='cuda:0')
            tail = field(zall.reshape(-1, 6), g[:, None, :].expand(n, 8*P, 6).reshape(-1, 6)).reshape(n, 4, 2, P)
            for d in pairs:
                cand = (call[..., :d]+tail[..., :d]).masked_fill(~sall[..., :d], torch.inf).reshape(n, -1)
                best, win = cand.min(1); ok = torch.isfinite(best)
                delta = zall[..., :d, :].reshape(n, -1, 6)[torch.arange(n), win]-q
                cos = (-grad*delta).sum(1)/(torch.linalg.vector_norm(grad, dim=1)*torch.linalg.vector_norm(delta, dim=1)).clamp_min(1e-12)
                backup[d].append(torch.where(ok, t/best, torch.nan).cpu().numpy()); cosine[d].append(torch.where(ok, cos, torch.nan).cpu().numpy())
                two = cand.topk(2, largest=False, dim=1).values; cw = call[..., :d].reshape(n, -1)[torch.arange(n), win]
                conf = ok & torch.isfinite(two[:, 1]) & ((two[:, 1]-two[:, 0])/cw.clamp_min(1e-8) >= .10)
                confident[d].append(conf[ok].float().cpu().numpy()); step[d].append(STEPS.to(win.device)[win[ok]//(2*d)].cpu().numpy())
    ratio = np.concatenate(ratio); gaps = np.concatenate(gaps); speeds = np.concatenate(speeds)
    away = np.concatenate(away)
    # cos(-grad T, grad S): does the descent direction head toward higher speed (away from obstacles)?
    report['descent_toward_speed'] = {f'{lo}-{hi}': float(away[(speeds >= lo) & (speeds < hi)].mean()) for lo, hi in ((0, .3), (.3, .6), (.6, .9), (.9, 2))
                                      if ((speeds >= lo) & (speeds < hi)).sum() >= 50}
    # Near obstacles (low speed) fewer sampled directions are safe: is the field steeper there?
    report['speed_ratio_by_speed'] = {f'{lo}-{hi}': stats(ratio[(speeds >= lo) & (speeds < hi)]) for lo, hi in ((0, .3), (.3, .6), (.6, .9), (.9, 2))
                                      if ((speeds >= lo) & (speeds < hi)).sum() >= 50}
    report['backup_ratio_by_speed'] = {d: {f'{lo}-{hi}': float(np.nanmedian(np.concatenate(backup[d])[(speeds >= lo) & (speeds < hi)]))
                                           for lo, hi in ((0, .3), (.3, .6), (.6, .9), (.9, 2)) if ((speeds >= lo) & (speeds < hi)).sum() >= 50} for d in pairs}
    report['speed_ratio'] = stats(ratio); report['speed_ratio_by_gap'] = {}
    for lo, hi in ((0, .3), (.3, .6), (.6, .9), (.9, 9)):
        sel = (gaps >= lo) & (gaps < hi)
        if sel.sum() >= 50: report['speed_ratio_by_gap'][f'{lo}-{hi}'] = stats(ratio[sel])
    report['backup_ratio'] = {d: stats(np.concatenate(backup[d])) for d in pairs}
    report['winner_cosine'] = {d: float(np.nanmean(np.concatenate(cosine[d]))) for d in pairs}
    report['confident'] = {d: float(np.concatenate(confident[d]).mean()) for d in pairs}
    report['winner_step'] = {d: float(np.concatenate(step[d]).mean()) for d in pairs}
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
    del field; torch.cuda.empty_cache(); return report


if __name__ == '__main__': main()
