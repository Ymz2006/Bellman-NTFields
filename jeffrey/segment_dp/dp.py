"""Segment-DP training (part b): the transition of interval DP between distance segments.

Model m+1 (segment [e_m, e_(m+1)]) is trained to
    T_(m+1)(i, j) = min over k of  T_<=m(i, k) + T_<=m(k, j),   |i - k| <= R_m and |k - j| <= R_m,
where T_<=m is the frozen composite (fields.SegmentField) of models 1..m and R_m = e_m + blend, the largest distance
those models were trained on: every entry is built from two strictly shorter entries that are already final, as
dp[i][j] = min_k dp[i][k] + dp[k][j]. The chain starts from the first --base-models models of a trained segment
field (--base, the eikonal specialists of part a): the pool has no split points for the second segment
(|i - j| up to 0.18 with both halves <= 0.10: 70-100 % of the lenses are empty), so DP starts at segment 3.

Split points k are pool configurations (no transitions; sampler.py) inside the lens |i - k| <= R, |k - j| <= R:
half of --candidates are the lens points nearest to the straight line (smallest |i - k| + |k - j|), half are
uniform over the lens (detours). Pairs whose lens is empty are dropped. Per segment: --pairs pairs (shell sampling
as part a), targets computed once with the frozen composite, then a fresh model trained --updates steps of 256 on
SmoothL1(T, target) (the Bellman term's loss), optionally + --eikonal-weight times the eikonal term. AdamW 2e-4,
weight decay 1e-6, clip 2, as the original method. Writes field.pt (all segments), field_segXX.pt after each
segment, progress.json (target statistics, train/held-out loss per segment), config.json.
"""
import sys
sys.path.insert(0, '/clean/method'); sys.path.insert(0, '/clean/jeffrey/segment_dp')
import argparse, json, time
from pathlib import Path
import torch
from torch.nn import functional as F
from common import sha, write
from fields import SegmentField, build, load_field, save_field
from optimizer import stable_clip_grad_norm_
from sampler import DUPLICATE, PairSampler, load_pool, pairwise


@torch.no_grad()
def targets(teacher, pool, square, i, j, radius, candidates, generator, chunk=128):
    out = dict(target=[], lens=[], near_won=[], split=[], detour=[])
    half = candidates//2
    for s in range(0, len(i), chunk):
        a, b = pool[i[s:s+chunk]], pool[j[s:s+chunk]]; n = len(a)
        da = pairwise(a, pool, square); db = pairwise(b, pool, square)
        lens = (da <= radius) & (db <= radius) & (da > DUPLICATE) & (db > DUPLICATE)   # never i or j themselves
        ellipse = (da+db).masked_fill_(~lens, torch.inf); near = ellipse.topk(half, dim=1, largest=False)
        keys = torch.rand(lens.shape, generator=generator, device=pool.device).masked_fill_(~lens, -1.0); rand = keys.topk(half, dim=1)
        index = torch.cat((near.indices, rand.indices), dim=1); valid = torch.cat((torch.isfinite(near.values), rand.values >= 0), dim=1)
        k = pool[index.reshape(-1)]
        left = teacher(a.repeat_interleave(candidates, 0), k)[:, 0].view(n, candidates)
        right = teacher(k, b.repeat_interleave(candidates, 0))[:, 0].view(n, candidates)
        total = (left+right).masked_fill(~valid, torch.inf); best, which = total.min(dim=1)
        rows = torch.arange(n, device=pool.device); w = index[rows, which]
        gap = torch.linalg.vector_norm(a-b, dim=1)
        out['target'].append(best); out['lens'].append(lens.sum(1)); out['near_won'].append(which < half)
        out['split'].append(da[rows, w]/(da[rows, w]+db[rows, w])); out['detour'].append((da[rows, w]+db[rows, w])/gap)
    return {k: torch.cat(v) for k, v in out.items()}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--prepared', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True); p.add_argument('--base-models', type=int, default=2)
    p.add_argument('--pairs', type=int, default=400000); p.add_argument('--candidates', type=int, default=384); p.add_argument('--updates', type=int, default=15620)
    p.add_argument('--eikonal-weight', type=float, default=0.0); p.add_argument('--seed', type=int, default=87358); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False); started = time.time()
    torch.manual_seed(a.seed); generator = torch.Generator(device='cuda:0').manual_seed(a.seed)
    base = torch.load(a.base, map_location='cuda:0', weights_only=False); assert base['kind'] == 'segment_field'
    edges, blend, arch = base['edges'], base['blend'], base['arch']; segments = len(edges)-1
    last = min(segments, a.base_models+2) if a.smoke else segments
    pairs, updates = (2048, 40) if a.smoke else (a.pairs, a.updates)
    pool, speeds, _, pool_info = load_pool(a.prepared, 'cuda:0'); square = pool.square().sum(1)
    sampler = PairSampler(pool, edges, blend, generator)
    full = load_field(a.base, 'cuda:0'); models = list(full.models[:a.base_models])
    config = dict(arch=arch, edges=edges, blend=blend, base=str(a.base), base_sha256=sha(a.base), base_models=a.base_models, pairs_per_segment=pairs,
                  candidates=a.candidates, updates_per_model=updates, batch=256, eikonal_weight=a.eikonal_weight, seed=a.seed, smoke=a.smoke,
                  learning_rate=2e-4, weight_decay=1e-6, gradient_clip=2.0, loss='SmoothL1(T, min_k T(i,k)+T(k,j))' + (f' + {a.eikonal_weight} eikonal' if a.eikonal_weight else ''),
                  **pool_info)
    write(a.output/'config.json', config); levels = []
    for m in range(a.base_models, last):
        level_start = time.time(); teacher = SegmentField(models, edges[:m+1], blend).eval().requires_grad_(False); radius = edges[m]+blend
        i, j = [], []
        for s in range(0, pairs, 8192):
            qi, gi, _, _ = sampler.sample(torch.full((min(8192, pairs-s),), m, device='cuda:0')); i.append(qi); j.append(gi)
        i, j = torch.cat(i), torch.cat(j)
        t = targets(teacher, pool, square, i, j, radius, a.candidates, generator); keep = torch.isfinite(t['target'])
        gap = torch.linalg.vector_norm(pool[i]-pool[j], dim=1)
        stats = dict(segment=m+1, edges=[edges[m], edges[m+1]], radius=radius, pairs=pairs, kept=int(keep.sum()), empty_lens=float((~keep).float().mean()),
                     lens_median=float(t['lens'].float().median()), near_won=float(t['near_won'][keep].float().mean()),
                     split_quantiles=torch.quantile(t['split'][keep], torch.tensor([.1, .5, .9], device='cuda:0')).tolist(),
                     detour_median=float(t['detour'][keep].median()), target_over_gap_median=float((t['target'][keep]/gap[keep]).median()),
                     target_seconds=time.time()-level_start)
        i, j, y = i[keep], j[keep], t['target'][keep]
        order = torch.randperm(len(i), generator=generator, device='cuda:0'); hold = order[:len(i)//50]; train = order[len(i)//50:]
        model = build(arch).cuda().train(); optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-6)
        for step in range(1, updates+1):
            rows = train[torch.randint(len(train), (256,), generator=generator, device='cuda:0')]
            q, g = pool[i[rows]], pool[j[rows]]
            if a.eikonal_weight:
                q = q.clone().requires_grad_(True); g = g.clone().requires_grad_(True)
            value = model(q, g)[:, 0]; loss = F.smooth_l1_loss(value, y[rows])
            if a.eikonal_weight:
                dq, dg = torch.autograd.grad(value.sum(), (q, g), create_graph=True)
                nq = torch.sqrt(dq.square().sum(1)+1e-12); ng = torch.sqrt(dg.square().sum(1)+1e-12)
                residual = torch.cat((torch.log((speeds[i[rows]]*nq).clamp_min(1e-8)), torch.log((speeds[j[rows]]*ng).clamp_min(1e-8))))
                loss = loss+a.eikonal_weight*residual.square().mean()
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Non-finite training loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            norm = stable_clip_grad_norm_(model.parameters(), 2.0)
            if not bool(torch.isfinite(norm)): raise FloatingPointError('Non-finite gradient')
            optimizer.step()
        model.eval().requires_grad_(False)
        with torch.no_grad():
            def fit(rows):
                v = torch.cat([model(pool[i[rows[s:s+8192]]], pool[j[rows[s:s+8192]]])[:, 0] for s in range(0, len(rows), 8192)])
                return dict(smooth_l1=float(F.smooth_l1_loss(v, y[rows])), mean_abs_log_ratio=float((v/y[rows]).log().abs().mean()))
            stats.update(train=fit(train[:len(hold)]), held_out=fit(hold), seconds=time.time()-level_start)
        models.append(model); levels.append(stats); print(json.dumps(stats), flush=True)
        save_field(a.output/f'field_seg{m+1:02d}.pt', arch, edges[:m+2], blend, [x.state_dict() for x in models], segments_trained=m+1, seed=a.seed, smoke=a.smoke)
        write(a.output/'progress.json', dict(complete=False, levels=levels, elapsed_seconds=time.time()-started))
    if last < segments:   # smoke: pad with the base field's remaining models so the field covers every segment
        models += list(full.models[last:])
    save_field(a.output/'field.pt', arch, edges, blend, [x.state_dict() for x in models], segments_trained=last, seed=a.seed, smoke=a.smoke)
    write(a.output/'progress.json', dict(complete=True, levels=levels, elapsed_seconds=time.time()-started))


if __name__ == '__main__': main()
