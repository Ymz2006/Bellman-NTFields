"""Segment-DP training with continuous split points (part b, DP-c): dp.py with k no longer restricted to the pool.

Model m+1 (segment [e_m, e_(m+1)]) is trained to
    T_(m+1)(i, j) = min over k of  T_<=m(i, k) + T_<=m(k, j),   |i - k| <= R_m and |k - j| <= R_m,
where T_<=m is the frozen composite (fields.SegmentField) of models 1..m and R_m = e_m + blend, the largest distance
those models were trained on: every entry is built from two strictly shorter entries that are already final, as
dp[i][j] = min_k dp[i][k] + dp[k][j]. The chain starts from the first --base-models models of a trained segment
field (--base, the eikonal specialists of part a): the pool has no split points for the second segment
(|i - j| up to 0.18 with both halves <= 0.10: 70-100 % of the lenses are empty), so DP starts at segment 3.

Split points k (the min of the transition is over all configurations; dp.py restricts it to the pool, whose
sparsity adds a detour that compounds over the segments: target / truth 1.22, 1.27, 1.29 for segments 3-5 of the
cabinet): --candidates pool configurations in the lens |i - k| <= R, |k - j| <= R (2/3 nearest to the straight line,
1/3 uniform over the lens), --continuous points k = i + t (j - i) + noise (t uniform in [0.2, 0.8], t = 0.5 noiseless;
noise per coordinate 0.03 or 0.12 x |i - j|), and the best 4 of all refined by --refine normalized gradient steps on
T_<=m(i, k) + T_<=m(k, j) (step 0.05 -> 0.005 x |i - j|); continuous and refined points are projected onto the lens
and the joint box. Pairs whose pool lens is empty keep the continuous candidates. Per segment: --pairs pairs (shell sampling
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


def project(k, a, b, radius):
    """Onto {|k - a| <= R} n {|k - b| <= R} n [-0.5, 0.5]^6 by alternating projections (k [n, c, 6], a, b [n, 6])."""
    for _ in range(4):
        for c in (a, b):
            d = k-c[:, None]; r = torch.linalg.vector_norm(d, dim=2, keepdim=True)
            k = c[:, None]+d*(radius/r.clamp_min(1e-12)).clamp(max=1.0)
        k = k.clamp(-.5, .5)
    return k


def pair_values(teacher, a, b, k):
    n, c, _ = k.shape; flat = k.reshape(-1, 6)
    return (teacher(a.repeat_interleave(c, 0), flat)[:, 0]+teacher(flat, b.repeat_interleave(c, 0))[:, 0]).view(n, c)


def targets(teacher, pool, square, i, j, radius, candidates, generator, continuous=64, refine=10, chunk=128):
    out = dict(target=[], lens=[], won=[], split=[], detour=[])
    near_count = 2*candidates//3; rand_count = candidates-near_count
    for s in range(0, len(i), chunk):
        a, b = pool[i[s:s+chunk]], pool[j[s:s+chunk]]; n = len(a); gap = torch.linalg.vector_norm(a-b, dim=1); rows = torch.arange(n, device=pool.device)
        with torch.no_grad():
            da = pairwise(a, pool, square); db = pairwise(b, pool, square)
            lens = (da <= radius) & (db <= radius) & (da > DUPLICATE) & (db > DUPLICATE)   # never i or j themselves
            near = (da+db).masked_fill_(~lens, torch.inf).topk(near_count, dim=1, largest=False)
            keys = torch.rand(lens.shape, generator=generator, device=pool.device).masked_fill_(~lens, -1.0); rand = keys.topk(rand_count, dim=1)
            index = torch.cat((near.indices, rand.indices), dim=1); valid = torch.cat((torch.isfinite(near.values), rand.values >= 0), dim=1)
            k_pool = pool[index]
            t = torch.rand((n, continuous), generator=generator, device=pool.device)*0.6+0.2; t[:, 0] = 0.5
            scale = torch.where(torch.arange(continuous, device=pool.device) % 2 == 0, 0.03, 0.12)[None, :, None]*gap[:, None, None]
            noise = torch.randn((n, continuous, 6), generator=generator, device=pool.device)*scale; noise[:, 0] = 0
            k_cont = project(a[:, None]+t[..., None]*(b-a)[:, None]+noise, a, b, radius)
            k_all = torch.cat((k_pool, k_cont), dim=1); ok = torch.cat((valid, torch.ones((n, continuous), dtype=torch.bool, device=pool.device)), dim=1)
            values = pair_values(teacher, a, b, k_all).masked_fill(~ok, torch.inf)
            start = values.topk(4, dim=1, largest=False).indices; k_ref = k_all[rows[:, None], start]
        for step in range(refine):
            with torch.enable_grad():
                x = k_ref.detach().requires_grad_(True); grad = torch.autograd.grad(pair_values(teacher, a, b, x).sum(), x)[0]
            with torch.no_grad():
                size = (0.05*(0.1**(step/max(1, refine-1))))*gap[:, None, None]
                k_ref = project(k_ref-size*grad/torch.linalg.vector_norm(grad, dim=2, keepdim=True).clamp_min(1e-12), a, b, radius)
        with torch.no_grad():
            refined = pair_values(teacher, a, b, k_ref); total = torch.cat((values, refined), dim=1); best, which = total.min(dim=1)
            k_best = torch.cat((k_all, k_ref), dim=1)[rows, which]; ra = torch.linalg.vector_norm(k_best-a, dim=1); rb = torch.linalg.vector_norm(k_best-b, dim=1)
            source = torch.bucketize(which, torch.tensor([near_count, candidates, candidates+continuous], device=pool.device), right=True)   # 0 near pool, 1 random pool, 2 continuous, 3 refined
        out['target'].append(best); out['lens'].append(lens.sum(1)); out['won'].append(source)
        out['split'].append(ra/(ra+rb)); out['detour'].append((ra+rb)/gap)
    return {k: torch.cat(v) for k, v in out.items()}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--prepared', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True); p.add_argument('--base-models', type=int, default=2)
    p.add_argument('--pairs', type=int, default=400000); p.add_argument('--candidates', type=int, default=192); p.add_argument('--continuous', type=int, default=64); p.add_argument('--refine', type=int, default=10); p.add_argument('--updates', type=int, default=15620)
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
                  candidates=a.candidates, continuous=a.continuous, refine=a.refine, split_points='pool + continuous + refined', updates_per_model=updates, batch=256, eikonal_weight=a.eikonal_weight, seed=a.seed, smoke=a.smoke,
                  learning_rate=2e-4, weight_decay=1e-6, gradient_clip=2.0, loss='SmoothL1(T, min_k T(i,k)+T(k,j))' + (f' + {a.eikonal_weight} eikonal' if a.eikonal_weight else ''),
                  **pool_info)
    write(a.output/'config.json', config); levels = []
    for m in range(a.base_models, last):
        level_start = time.time(); teacher = SegmentField(models, edges[:m+1], blend).eval().requires_grad_(False); radius = edges[m]+blend
        i, j = [], []
        for s in range(0, pairs, 8192):
            qi, gi, _, _ = sampler.sample(torch.full((min(8192, pairs-s),), m, device='cuda:0')); i.append(qi); j.append(gi)
        i, j = torch.cat(i), torch.cat(j)
        t = targets(teacher, pool, square, i, j, radius, a.candidates, generator, a.continuous, a.refine); keep = torch.isfinite(t['target'])
        gap = torch.linalg.vector_norm(pool[i]-pool[j], dim=1)
        stats = dict(segment=m+1, edges=[edges[m], edges[m+1]], radius=radius, pairs=pairs, kept=int(keep.sum()), empty_lens=float((~keep).float().mean()),
                     pool_lens_empty=float((t['lens'] == 0).float().mean()),
                     lens_median=float(t['lens'].float().median()), won=dict(zip(('pool_near', 'pool_random', 'continuous', 'refined'), torch.bincount(t['won'][keep], minlength=4).div(int(keep.sum())).tolist())),
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
