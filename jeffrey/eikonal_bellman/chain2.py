"""Bellman (segment-DP) extension of a frozen short-range eikonal foundation to longer travel times.

Foundation F: a travel-cost field trained with the eikonal loss on short pairs only (|q - g| <= R_E = --first-range),
frozen. Segment 0 of the field, [0, e_1], is answered by F. Every further segment m = [e_m, e_(m+1)] of |q - g| is
learned from the Bellman / interval-DP target
    --scheme efl        T(i, j) = min_k  F(i, k) + T_<m(k, j)        |i - k| <= R_E,         |k - j| <= e_m + blend
    --scheme symmetric  T(i, j) = min_k  T_<m(i, k) + T_<m(k, j)     |i - k| <= e_m + blend, |k - j| <= e_m + blend
where T_<m is the frozen composite of everything trained so far (fields.SegmentField): every entry is built from
shorter entries that are already final. In the efl scheme the first leg is always the foundation's accurate
short-range travel time (a Bellman backup with the eikonal field as the one-step cost); in the symmetric scheme both
legs may come from learned segments (jeffrey/segment_dp/dp.py).

--scheme vi         CONTROL for the curriculum: value iteration on natural pairs. Every round draws pairs from all
                    segments, the first leg is the foundation (|i - k| <= R_E) and the tail T_(r-1)(k, j) is the previous
                    round's field at ANY gap (no trusted-range restriction; the field starts as the foundation's
                    extrapolation); --models single|finetune, rounds = the number of levels
--models multi      a fresh model per segment (several networks, one per range); the foundation is never changed
--models single     one long-range network L for every gap > e_1, warm-started from F and trained level by level on
                    the new segment's targets plus --old-share of the earlier segments' targets (rehearsal)
--models finetune   F itself is fine-tuned level by level (one network for every gap; the frozen ORIGINAL F still
                    provides the first leg of efl targets); --rehearsal N adds N short pairs with the eikonal loss to
                    every batch (keeps the close-range field), 0 = none
--split pool        split points k are pool configurations (collision-checked poses): --candidates per pair, half
                    nearest the straight line, half uniform over the lens; empty lens -> pair dropped
--split cont        plus --continuous points along the straight segment with noise (dp2.py), the best --refine-count
                    of them refined by --refine normalized gradient steps on the frozen composite
--split gated       cont, but off-pool points are valid only where the foundation was trained: within --support of a
                    pool pose (0 = off; 'auto' = the 95th percentile of the pool's nearest-neighbour distance) and,
                    with --clearance SCENE, collision-free by the exact release clearance (jeffrey/geometry.py)
--estimator double  two models per level on disjoint pairs; k is selected with composite A and the target read with
                    composite B (independent regression noise: no winner's-curse bias from the min)
--weights B,E,D     loss of the learned segments: B SmoothL1(T, target) + E eikonal residual (method/loss.py) +
                    D (1 - cos(-grad_q T, d*)) with d* = -grad_q of the first leg at q toward the winning split point
--truth truth.npz   per-level diagnostic against path-optimized ground truth (segment_dp/truth.py): median
                    target / T* and model / T* on the truth pairs of each segment (never selects anything)
Optimizer as the original method: AdamW 2e-4, weight decay 1e-6, gradient clip 2, batch 256, --updates per level.
Writes field.pt (the composite), progress.json (per-level statistics), targets_segXX.npz, config.json.
"""
import sys
sys.path.insert(0, '/clean'); sys.path.insert(0, '/clean/method'); sys.path.insert(0, '/clean/jeffrey/segment_dp')
import argparse, copy, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from common import sha, write
from fields import SegmentField, load_field, save_field
from network import TravelCost
from optimizer import stable_clip_grad_norm_
from sampler import DUPLICATE, PairSampler, load_pool, pairwise

EDGES = '0,0.16,'+','.join(f'{0.08*i:g}' for i in range(3, 21))      # [0, 0.16] foundation, then 0.08 segments to 1.6
QUANT = torch.tensor([.1, .5, .9])


def project(k, a, b, ra, rb):
    """k [n, c, 6] onto {|k - a| <= ra} n {|k - b| <= rb} n [-0.5, 0.5]^6 by alternating projections."""
    for _ in range(4):
        for c, r in ((a, ra), (b, rb)):
            d = k-c[:, None]; dist = torch.linalg.vector_norm(d, dim=2, keepdim=True)
            k = c[:, None]+d*(r/dist.clamp_min(1e-12)).clamp(max=1.0)
        k = k.clamp(-.5, .5)
    return k


class Gate:
    """Validity of off-pool split points: support (nearest pool pose within rho) and exact release clearance."""

    def __init__(self, pool, square, rho, clearance, sub=1024):
        self.pool, self.square, self.rho, self.clearance, self.sub = pool, square, float(rho), clearance, sub

    @torch.no_grad()
    def __call__(self, k):
        ok = torch.ones(len(k), dtype=torch.bool, device=k.device); stats = {}
        if self.rho > 0:
            nn = torch.cat([pairwise(k[s:s+self.sub], self.pool, self.square).amin(1) for s in range(0, len(k), self.sub)])
            ok &= nn <= self.rho; stats['support_rejected'] = float((nn > self.rho).float().mean())
        if self.clearance is not None:
            margin = torch.cat([self.clearance.margins(k[s:s+8192]) for s in range(0, len(k), 8192)])
            ok &= margin >= 0; stats['clearance_rejected'] = float((margin < 0).float().mean())
        return ok, stats


def values(first, tail, a, b, k):
    """first(a, k) + tail(k, b) for k [n, c, 6] -> [n, c]."""
    n, c, _ = k.shape; flat = k.reshape(-1, 6)
    return (first(a.repeat_interleave(c, 0), flat)[:, 0]+tail(flat, b.repeat_interleave(c, 0))[:, 0]).view(n, c)


class Targets:
    def __init__(self, a, pool, square, gate, generator):
        self.a, self.pool, self.square, self.gate, self.generator = a, pool, square, gate, generator

    @torch.no_grad()
    def pool_candidates(self, a, b, r_first, r_tail):
        da = pairwise(a, self.pool, self.square); db = pairwise(b, self.pool, self.square)
        lens = (da <= r_first) & (db <= r_tail) & (da > DUPLICATE) & (db > DUPLICATE)      # never i or j themselves
        half = self.a.candidates//2
        near = (da+db).masked_fill_(~lens, torch.inf).topk(half, dim=1, largest=False)
        keys = torch.rand(lens.shape, generator=self.generator, device=a.device).masked_fill_(~lens, -1.0).topk(half, dim=1)
        index = torch.cat((near.indices, keys.indices), 1); valid = torch.cat((torch.isfinite(near.values), keys.values >= 0), 1)
        return self.pool[index], valid, lens.sum(1)

    @torch.no_grad()
    def continuous_candidates(self, a, b, r_first, r_tail):
        n, c = len(a), self.a.continuous; gap = torch.linalg.vector_norm(b-a, dim=1)
        t = torch.rand((n, c), generator=self.generator, device=a.device)*0.7+0.15; t[:, 0] = 0.5
        scale = torch.where(torch.arange(c, device=a.device) % 2 == 0, 0.03, 0.12)[None, :, None]*gap[:, None, None]
        noise = torch.randn((n, c, 6), generator=self.generator, device=a.device)*scale; noise[:, 0] = 0
        return project(a[:, None]+t[..., None]*(b-a)[:, None]+noise, a, b, r_first, r_tail)

    def build(self, sel, val, i, j, r_first, r_tail, want_direction):
        """sel = (first, tail) composites that select k; val = the composites that read the target (same unless double)."""
        a_ = self.a; out = dict(target=[], source=[], lens=[], split=[], detour=[], direction=[]); gate_stats = []
        c_pool, c_cont, c_ref = a_.candidates, (a_.continuous if a_.split != 'pool' else 0), (a_.refine_count if a_.split != 'pool' and a_.refine else 0)
        for s in range(0, len(i), a_.chunk):
            a, b = self.pool[i[s:s+a_.chunk]], self.pool[j[s:s+a_.chunk]]; n = len(a); rows = torch.arange(n, device=a.device)
            gap = torch.linalg.vector_norm(b-a, dim=1)
            with torch.no_grad():
                k_all, ok_all, lens = self.pool_candidates(a, b, r_first, r_tail); src = [torch.zeros((n, c_pool), dtype=torch.int8, device=a.device)]
                if c_cont:
                    k_c = self.continuous_candidates(a, b, r_first, r_tail)
                    ok_c = torch.ones((n, c_cont), dtype=torch.bool, device=a.device)
                    if a_.split == 'gated':
                        ok_c, st = self.gate(k_c.reshape(-1, 6)); ok_c = ok_c.view(n, c_cont); gate_stats.append(st)
                    k_all = torch.cat((k_all, k_c), 1); ok_all = torch.cat((ok_all, ok_c), 1); src.append(torch.ones((n, c_cont), dtype=torch.int8, device=a.device))
                v_sel = values(*sel, a, b, k_all).masked_fill(~ok_all, torch.inf)
            if c_ref:
                start = v_sel[:, c_pool:].topk(c_ref, dim=1, largest=False).indices; k_ref = k_all[:, c_pool:][rows[:, None], start]
                for step in range(a_.refine):
                    with torch.enable_grad():
                        x = k_ref.detach().requires_grad_(True); grad = torch.autograd.grad(values(*sel, a, b, x).sum(), x)[0]
                    with torch.no_grad():
                        size = (0.05*(0.1**(step/max(1, a_.refine-1))))*gap[:, None, None]
                        k_ref = project(k_ref-size*grad/torch.linalg.vector_norm(grad, dim=2, keepdim=True).clamp_min(1e-12), a, b, r_first, r_tail)
                with torch.no_grad():
                    ok_r = torch.ones((n, c_ref), dtype=torch.bool, device=a.device)
                    if a_.split == 'gated':
                        ok_r, st = self.gate(k_ref.reshape(-1, 6)); ok_r = ok_r.view(n, c_ref); gate_stats.append(st)
                    v_r = values(*sel, a, b, k_ref).masked_fill(~ok_r, torch.inf)
                    k_all = torch.cat((k_all, k_ref), 1); v_sel = torch.cat((v_sel, v_r), 1); src.append(torch.full((n, c_ref), 2, dtype=torch.int8, device=a.device))
            with torch.no_grad():
                best, which = v_sel.min(dim=1); k_best = k_all[rows, which]; feasible = torch.isfinite(best)
                target = values(*val, a, b, k_best[:, None])[:, 0] if val is not sel else best
                target = torch.where(feasible, target, torch.full_like(target, torch.inf))
                ra = torch.linalg.vector_norm(k_best-a, dim=1); rb = torch.linalg.vector_norm(k_best-b, dim=1)
            if want_direction:
                with torch.enable_grad():
                    x = a.detach().clone().requires_grad_(True); g = torch.autograd.grad(sel[0](x, k_best.detach())[:, 0].sum(), x)[0]
                d = -g/torch.linalg.vector_norm(g, dim=1, keepdim=True).clamp_min(1e-12)
            else:
                d = torch.zeros_like(a)
            out['target'].append(target); out['source'].append(torch.cat(src, 1)[rows, which]); out['lens'].append(lens)
            out['split'].append(ra/(ra+rb).clamp_min(1e-12)); out['detour'].append((ra+rb)/gap); out['direction'].append(d)
        out = {k: torch.cat(v) for k, v in out.items()}
        if gate_stats:
            out['gate'] = {k: float(np.mean([g[k] for g in gate_stats if k in g])) for k in gate_stats[0]}
        return out


def eikonal_terms(dq, dg, sq, sg):
    nq = torch.sqrt(dq.square().sum(1)+1e-12); ng = torch.sqrt(dg.square().sum(1)+1e-12)
    return torch.cat((torch.log((sq*nq).clamp_min(1e-8)), torch.log((sg*ng).clamp_min(1e-8)))).square().mean()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--prepared', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True); p.add_argument('--first-range', type=float, default=0.176)
    p.add_argument('--edges', default=EDGES); p.add_argument('--blend', type=float, default=0.016); p.add_argument('--last-edge', type=float, default=None)
    p.add_argument('--scheme', choices=['efl', 'symmetric', 'vi'], default='efl'); p.add_argument('--models', choices=['multi', 'single', 'finetune'], default='multi')
    p.add_argument('--split', choices=['pool', 'cont', 'gated'], default='pool'); p.add_argument('--candidates', type=int, default=128)
    p.add_argument('--continuous', type=int, default=32); p.add_argument('--refine', type=int, default=4); p.add_argument('--refine-count', type=int, default=2)
    p.add_argument('--support', default='auto'); p.add_argument('--clearance', type=Path, default=None)
    p.add_argument('--estimator', choices=['single', 'double'], default='single'); p.add_argument('--init', choices=['foundation', 'random'], default='foundation')
    p.add_argument('--weights', default='1,0,0'); p.add_argument('--rehearsal', type=int, default=0); p.add_argument('--rehearsal-weight', type=float, default=2.0)
    p.add_argument('--old-share', type=float, default=0.5); p.add_argument('--pairs', type=int, default=400000); p.add_argument('--updates', type=int, default=15620)
    p.add_argument('--chunk', type=int, default=128); p.add_argument('--truth', type=Path, default=None)
    p.add_argument('--seed', type=int, default=87358); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False); started = time.time(); dev = 'cuda:0'
    torch.manual_seed(a.seed); generator = torch.Generator(device=dev).manual_seed(a.seed)
    weights = [float(x) for x in a.weights.split(',')]; assert len(weights) == 3; wB, wE, wD = weights
    edges = [float(x) for x in a.edges.split(',')]
    if a.last_edge is not None: edges = [e for e in edges if e <= a.last_edge+1e-9]
    assert len(edges) >= 3 and abs(edges[1]+a.blend-a.first_range) < 1e-6, 'segment 1 must end at the foundation range minus the blend'
    levels = list(range(1, len(edges)-1)); levels = levels[:2] if a.smoke else levels
    pairs, updates = (2048, 40) if a.smoke else (a.pairs, a.updates)
    pool, speeds, speed, pool_info = load_pool(a.prepared, dev); square = pool.square().sum(1)
    sampler = PairSampler(pool, edges, a.blend, generator)
    foundation = load_field(a.base, dev)
    base_payload = torch.load(a.base, map_location='cpu', weights_only=False)
    assert base_payload.get('kind') == 'segment_field' and len(base_payload['model_state_dicts']) == 1 and base_payload['arch'] == 'pair', 'the foundation must be a one-model pair field'
    foundation_state = base_payload['model_state_dicts'][0]
    rho = 0.0
    if a.support != '0':
        idx = torch.randint(len(pool), (4096,), generator=generator, device=dev); d = pairwise(pool[idx], pool, square); d.masked_fill_(d < DUPLICATE, torch.inf)
        nn1 = d.amin(1); rho = float(torch.quantile(nn1, 0.95)) if a.support == 'auto' else float(a.support); del d
    clearance = None
    if a.clearance is not None:
        from jeffrey.geometry import ArmClearance
        clearance = ArmClearance(a.clearance, None, 0.0, dev)
        with torch.no_grad():
            sample = pool[torch.randint(len(pool), (8192,), generator=generator, device=dev)]
            pool_margin = clearance.margins(sample); pool_clear = float((pool_margin >= 0).float().mean())
    gate = Gate(pool, square, rho, clearance)
    targets = Targets(a, pool, square, gate, generator)
    truth = None
    if a.truth is not None:
        t = np.load(a.truth); truth = dict(i=torch.as_tensor(t['source_index'], device=dev), j=torch.as_tensor(t['goal_index'], device=dev),
                                           value=torch.as_tensor(t['truth'], device=dev), gap=torch.as_tensor(t['gap'], device=dev))
        assert bool((pool[truth['i']]-torch.as_tensor(t['q'], device=dev)).abs().max() < 1e-6), 'truth pairs are not pool poses of this scene'
    count = 2 if a.estimator == 'double' else 1

    def fresh():
        net = TravelCost().to(dev)
        if a.init == 'foundation': net.load_state_dict(foundation_state)
        return net.train()
    persistent = None
    if a.models in ('single', 'finetune'):
        persistent = [TravelCost().to(dev) for _ in range(count)]
        for net in persistent: net.load_state_dict(foundation_state); net.train()
        optimizers = [torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-6) for net in persistent]
    learned = [[] for _ in range(count)]          # multi: one model per level per estimator
    config = dict(scheme=a.scheme, models=a.models, split=a.split, estimator=a.estimator, init=a.init, edges=edges, blend=a.blend, first_range=a.first_range,
                  base=str(a.base), base_sha256=sha(a.base), candidates=a.candidates, continuous=a.continuous if a.split != 'pool' else 0,
                  refine=a.refine if a.split != 'pool' else 0, refine_count=a.refine_count, support_rho=rho if a.split == 'gated' else None,
                  clearance=str(a.clearance) if a.clearance else None, pool_clear_fraction=pool_clear if clearance else None,
                  weights=dict(bellman=wB, eikonal=wE, direction=wD), rehearsal=a.rehearsal, rehearsal_weight=a.rehearsal_weight, old_share=a.old_share,
                  pairs_per_level=pairs, updates_per_level=updates, batch=256, learning_rate=2e-4, weight_decay=1e-6, gradient_clip=2.0, seed=a.seed, smoke=a.smoke,
                  truth=str(a.truth) if a.truth else None, truth_sha256=sha(a.truth) if a.truth else None, levels=[edges[m:m+2] for m in levels], **pool_info)
    write(a.output/'config.json', config); progress = []
    old = [dict(i=[], j=[], y=[], d=[]) for _ in range(count)]          # accumulated earlier levels (single / finetune rehearsal)

    def composite(e, which):
        """Frozen composite of the foundation and everything learned so far for estimator `which`, up to edge index e."""
        if a.models == 'multi':
            models = [foundation]+learned[which]
            return SegmentField(models, edges[:e+1], a.blend).eval().requires_grad_(False) if len(models) > 1 else foundation
        frozen = copy.deepcopy(persistent[which]).eval().requires_grad_(False)
        if a.models == 'single':
            return SegmentField([foundation, frozen], [edges[0], edges[1], edges[e]], a.blend).eval().requires_grad_(False) if e > 1 else foundation
        return frozen                                                                   # finetune: one model answers every gap

    vi = a.scheme == 'vi'
    if vi: assert a.models in ('single', 'finetune'), 'vi: one persistent network'
    for m in levels:
        t0 = time.time(); r_tail = edges[-1]+a.blend if vi else edges[m]+a.blend; r_first = a.first_range if a.scheme in ('efl', 'vi') else r_tail
        comps = [composite(len(edges)-1 if vi else m, w) for w in range(count)]
        first = [(foundation if a.scheme in ('efl', 'vi') else c) for c in comps]
        sel, val = (first[0], comps[0]), ((first[1], comps[1]) if count == 2 else None)
        val = val or sel
        i, j = [], []
        for s in range(0, pairs, 8192):
            n_ = min(8192, pairs-s); seg = torch.randint(1, len(edges)-1, (n_,), generator=generator, device=dev) if vi else torch.full((n_,), m, device=dev)
            qi, gi, _, _ = sampler.sample(seg); i.append(qi); j.append(gi)
        i, j = torch.cat(i), torch.cat(j)
        t = targets.build(sel, val, i, j, r_first, r_tail, wD > 0); keep = torch.isfinite(t['target'])
        gap = torch.linalg.vector_norm(pool[i]-pool[j], dim=1)
        stats = dict(level=m, segment=[edges[m], edges[m+1]], r_first=r_first, r_tail=r_tail, pairs=pairs, kept=int(keep.sum()), infeasible=float((~keep).float().mean()),
                     lens_median=float(t['lens'].float().median()), won=dict(zip(('pool', 'continuous', 'refined'), torch.bincount(t['source'][keep].long(), minlength=3).div(max(1, int(keep.sum()))).tolist())),
                     split_quantiles=torch.quantile(t['split'][keep], QUANT.to(dev)).tolist(), detour_median=float(t['detour'][keep].median()),
                     target_over_gap_median=float((t['target'][keep]/gap[keep]).median()), gate=t.get('gate'), target_seconds=time.time()-t0)
        if truth is not None and vi:
            tt = targets.build(sel, val, truth['i'], truth['j'], r_first, r_tail, False); ok = torch.isfinite(tt['target']); ratio = (tt['target']/truth['value'])
            stats['truth_by_segment'] = [dict(segment=[edges[k], edges[k+1]], target_over_truth_median=float(ratio[ok & (truth['gap'] > edges[k]) & (truth['gap'] <= edges[k+1])].median()))
                                         for k in range(1, len(edges)-1)]
        if truth is not None:
            sel_t = (truth['gap'] > edges[m]) & (truth['gap'] <= edges[m+1]) if not vi else (truth['gap'] > edges[1]) & (truth['gap'] <= edges[-1])
            if int(sel_t.sum()) >= 10:
                tt = targets.build(sel, val, truth['i'][sel_t], truth['j'][sel_t], r_first, r_tail, False); ok = torch.isfinite(tt['target'])
                ratio = tt['target'][ok]/truth['value'][sel_t][ok]
                stats['truth'] = dict(pairs=int(sel_t.sum()), feasible=int(ok.sum()), target_over_truth_median=float(ratio.median()), target_abs_log_error=float(ratio.log().abs().mean()),
                                      detour_median=float(tt['detour'][ok].median()))
                if count == 2:      # the single-estimator target on the same pairs: its excess over the double one is the selection bias of the min
                    ts = targets.build(sel, sel, truth['i'][sel_t], truth['j'][sel_t], r_first, r_tail, False); oks = torch.isfinite(ts['target'])
                    stats['truth']['single_estimator_target_over_truth_median'] = float((ts['target'][oks]/truth['value'][sel_t][oks]).median())
        i, j, y, d = i[keep], j[keep], t['target'][keep], t['direction'][keep]
        order = torch.randperm(len(i), generator=generator, device=dev); hold = order[:len(i)//50]; train_rows = order[len(i)//50:]
        halves = [train_rows[w::count] for w in range(count)]
        need_grad = wE > 0 or wD > 0; fits = []
        for w in range(count):
            if a.models == 'multi':
                net = fresh(); optimizer = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-6)
            else:
                net, optimizer = persistent[w], optimizers[w]; net.train()
            rows_new = halves[w]
            have_old = a.models != 'multi' and len(old[w]['i']) > 0 and not vi
            if have_old:
                oi, oj, oy, od = (torch.cat(old[w][k]) for k in ('i', 'j', 'y', 'd'))
            n_old = int(256*a.old_share) if have_old else 0
            for step in range(1, updates+1):
                r = rows_new[torch.randint(len(rows_new), (256-n_old,), generator=generator, device=dev)]
                qi, gi, yb, db = i[r], j[r], y[r], d[r]
                if n_old:
                    ro = torch.randint(len(oi), (n_old,), generator=generator, device=dev)
                    qi, gi, yb, db = torch.cat((qi, oi[ro])), torch.cat((gi, oj[ro])), torch.cat((yb, oy[ro])), torch.cat((db, od[ro]))
                q, g = pool[qi], pool[gi]
                if need_grad: q = q.clone().requires_grad_(True); g = g.clone().requires_grad_(True)
                value = net(q, g)[:, 0]; loss = wB*F.smooth_l1_loss(value, yb); terms = dict(bellman=float(loss))
                if need_grad:
                    dq, dg = torch.autograd.grad(value.sum(), (q, g), create_graph=True)
                    if wE > 0:
                        eik = eikonal_terms(dq, dg, speeds[qi], speeds[gi]); loss = loss+wE*eik; terms['eikonal'] = float(eik)
                    if wD > 0:
                        cos = (-dq*db).sum(1)/torch.linalg.vector_norm(dq, dim=1).clamp_min(1e-8); direction = (1-cos).mean(); loss = loss+wD*direction; terms['direction'] = float(direction)
                if a.rehearsal > 0:
                    si, sg_, _, _ = sampler.sample(torch.zeros(a.rehearsal, dtype=torch.long, device=dev))
                    q2 = pool[si].clone().requires_grad_(True); g2 = pool[sg_].clone().requires_grad_(True)
                    v2 = net(q2, g2)[:, 0]; dq2, dg2 = torch.autograd.grad(v2.sum(), (q2, g2), create_graph=True)
                    reh = eikonal_terms(dq2, dg2, speeds[si], speeds[sg_]); loss = loss+a.rehearsal_weight*reh; terms['rehearsal'] = float(reh)
                if not bool(torch.isfinite(loss)): raise FloatingPointError('Non-finite training loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                norm = stable_clip_grad_norm_(net.parameters(), 2.0)
                if not bool(torch.isfinite(norm)): raise FloatingPointError('Non-finite gradient')
                optimizer.step()
            net.eval()
            with torch.no_grad():
                def fit(rows):
                    v = torch.cat([net(pool[i[rows[s:s+8192]]], pool[j[rows[s:s+8192]]])[:, 0] for s in range(0, len(rows), 8192)])
                    return dict(smooth_l1=float(F.smooth_l1_loss(v, y[rows])), mean_abs_log_ratio=float((v/y[rows]).log().abs().mean()))
                row = dict(estimator=w, train=fit(rows_new[:max(1, len(hold))]), held_out=fit(hold), last_terms=terms)
                if truth is not None and 'truth' in stats:
                    sel_t = (truth['gap'] > edges[m]) & (truth['gap'] <= edges[m+1]) if not vi else (truth['gap'] > edges[1]) & (truth['gap'] <= edges[-1])
                    v = net(pool[truth['i'][sel_t]], pool[truth['j'][sel_t]])[:, 0]; ratio = v/truth['value'][sel_t]
                    row['model_over_truth_median'] = float(ratio.median()); row['model_abs_log_error'] = float(ratio.log().abs().mean())
                    if vi:
                        gaps = truth['gap'][sel_t]
                        row['model_over_truth_by_segment'] = [float(ratio[(gaps > edges[k]) & (gaps <= edges[k+1])].median()) for k in range(1, len(edges)-1)]
                fits.append(row)
            if a.models == 'multi':
                learned[w].append(net.requires_grad_(False))
            else:
                old[w]['i'].append(i[halves[w]]); old[w]['j'].append(j[halves[w]]); old[w]['y'].append(y[halves[w]]); old[w]['d'].append(d[halves[w]])
        stats.update(fits=fits, seconds=time.time()-t0, elapsed_seconds=time.time()-started); progress.append(stats)
        print(json.dumps({k: v for k, v in stats.items() if k not in ('gate',)}), flush=True)
        np.savez(a.output/f'targets_seg{m:02d}.npz', i=i.cpu().numpy(), j=j.cpu().numpy(), target=y.cpu().numpy(), source=t['source'][keep].cpu().numpy())
        write(a.output/'progress.json', dict(complete=False, levels=progress, elapsed_seconds=time.time()-started))
    # final field (estimator 0)
    if a.models == 'multi':
        save_field(a.output/'field.pt', 'pair', edges[:len(levels)+2], a.blend, [foundation_state]+[n.state_dict() for n in learned[0]], models='multi', scheme=a.scheme, seed=a.seed, smoke=a.smoke)
    elif a.models == 'single':
        save_field(a.output/'field.pt', 'pair', [edges[0], edges[1], edges[-1] if vi else edges[levels[-1]+1]], a.blend, [foundation_state, persistent[0].state_dict()], models='single', scheme=a.scheme, seed=a.seed, smoke=a.smoke)
    else:
        save_field(a.output/'field.pt', 'pair', [edges[0], edges[-1] if vi else edges[levels[-1]+1]], 0.0, [persistent[0].state_dict()], models='finetune', scheme=a.scheme, seed=a.seed, smoke=a.smoke)
    if count == 2:
        if a.models == 'multi': save_field(a.output/'field_b.pt', 'pair', edges[:len(levels)+2], a.blend, [foundation_state]+[n.state_dict() for n in learned[1]], models='multi-b', seed=a.seed, smoke=a.smoke)
        else: torch.save(dict(model_state_dict=persistent[1].state_dict()), a.output/'model_b.pt')
    write(a.output/'progress.json', dict(complete=True, levels=progress, elapsed_seconds=time.time()-started))


if __name__ == '__main__': main()
