"""Bank-free eikonal training of travel-cost models on distance segments (segment-DP experiment, part a).

--edges e_0,...,e_M   segments of |q - g| in normalized joint units (default: 20 segments of 0.08 over [0, 1.6])
--mode specialists    one model per segment, trained only on pairs of its segment widened by --blend; the M models
                      train together as an Ensemble (one batched pass; each model has its own batch of 256, loss,
                      gradient clipping and AdamW state, exactly as M separate runs)
--mode generalist     one model on the union: every row of a batch draws its segment uniformly
--arch pair|offset    TravelCost (two positions) or OffsetCost (position + offset), fields.py
Loss: the eikonal term of method/loss.py, mean of (log S(q)|grad_q T|)^2 and (log S(g)|grad_g T|)^2; no Bellman
target, no transitions (pairs from sampler.py). --normal-weight w > 0 adds ntrl-demo's normal term at both ends,
w (1.001 - S) |S grad T + n|^2 with n the unit gradient of the frozen speed network at the pose (the direction away
from the nearest obstacle; the eikonal term fixes only |grad T|, this pins its direction towards the obstacle;
2026-11-01 representability test: without it TravelCost puts the wall's slope orthogonal to the approach). Optimizer as the original method: AdamW 2e-4, weight decay 1e-6,
gradient-norm clip 2 per model, batch 256, 15,620 updates per model.
Writes field.pt (segment field of the M specialists, or the generalist as a one-segment field over [0, e_M]),
field_uXXXXX.pt at every fifth of the run, progress.json (per-segment loss and gap-matching statistics), config.json.
"""
import sys
sys.path.insert(0, '/clean/method'); sys.path.insert(0, '/clean/jeffrey/segment_dp')
import argparse, json, time
from pathlib import Path
import torch
from common import write
from fields import ARCHS, Ensemble, build, clip_per_model, save_field, valid_arch
from sampler import PairSampler, load_pool

EDGES = ','.join(f'{0.08*i:g}' for i in range(21))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--prepared', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['specialists', 'generalist'], required=True); p.add_argument('--arch', default='pair', help="pair | offset | pair-hH[-iI] (H harmonics, input scale I)")
    p.add_argument('--edges', default=EDGES); p.add_argument('--blend', type=float, default=0.016); p.add_argument('--updates', type=int, default=15620)
    p.add_argument('--batch', type=int, default=256); p.add_argument('--seed', type=int, default=87358); p.add_argument('--normal-weight', type=float, default=0.)
    p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    assert valid_arch(a.arch), a.arch; a.output.mkdir(parents=True, exist_ok=False); started = time.time()
    edges = [float(x) for x in a.edges.split(',')]; segments = len(edges)-1; updates = 40 if a.smoke else a.updates
    torch.manual_seed(a.seed); generator = torch.Generator(device='cuda:0').manual_seed(a.seed)
    pool, speeds, speed_net, pool_info = load_pool(a.prepared, 'cuda:0'); normals = None
    if a.normal_weight > 0:   # unit speed gradient at every pool pose (zero where the speed is flat)
        parts = []
        for i in range(0, len(pool), 32768):
            x = pool[i:i+32768].detach().clone().requires_grad_(True)
            with torch.enable_grad(): gs = torch.autograd.grad(speed_net(x).sum(), x)[0]
            parts.append(gs/torch.linalg.vector_norm(gs, dim=1, keepdim=True).clamp_min(1e-6)*(torch.linalg.vector_norm(gs, dim=1, keepdim=True) > 1e-4))
        normals = torch.cat(parts).detach()
    count = segments if a.mode == 'specialists' else 1
    ensemble = Ensemble([build(a.arch).cuda() for _ in range(count)]).cuda().train()
    optimizer = torch.optim.AdamW(ensemble.parameters(), lr=2e-4, weight_decay=1e-6)
    sampler = PairSampler(pool, edges, a.blend, generator)
    config = dict(mode=a.mode, arch=a.arch, edges=edges, blend=a.blend, models=count, updates_per_model=updates, batch_per_model=a.batch, seed=a.seed,
                  learning_rate=2e-4, weight_decay=1e-6, gradient_clip=2.0, loss='eikonal (method/loss.py eikonal term only)'+(f' + {a.normal_weight:g} x normal term (1.001 - S)|S grad T + n|^2' if a.normal_weight > 0 else ''),
                  normal_weight=a.normal_weight, smoke=a.smoke,
                  eligible_sources=[len(e) for e in sampler.eligible], **pool_info, setup_seconds=time.time()-started)
    write(a.output/'config.json', config)
    fixed = torch.arange(count, device='cuda:0').repeat_interleave(a.batch)
    totals = torch.zeros(segments, device='cuda:0', dtype=torch.float64); counts = torch.zeros(segments, device='cuda:0', dtype=torch.float64)
    gap_error = torch.zeros(segments, device='cuda:0', dtype=torch.float64); outside = torch.zeros(segments, device='cuda:0', dtype=torch.float64)
    history = []
    for step in range(1, updates+1):
        seg = fixed if a.mode == 'specialists' else torch.randint(segments, (a.batch,), generator=generator, device='cuda:0')
        qi, gi, gap, target = sampler.sample(seg)
        assert bool((gap > 1e-3).all()), 'degenerate pair'
        q = pool[qi].view(count, -1, 6).requires_grad_(True); g = pool[gi].view(count, -1, 6).requires_grad_(True)
        value = ensemble(q, g)
        dq, dg = torch.autograd.grad(value.sum(), (q, g), create_graph=True)
        nq = torch.sqrt(dq.square().sum(2)+1e-12); ng = torch.sqrt(dg.square().sum(2)+1e-12)
        residual = torch.cat((torch.log((speeds[qi].view(count, -1)*nq).clamp_min(1e-8)), torch.log((speeds[gi].view(count, -1)*ng).clamp_min(1e-8))), dim=1)
        per_model = residual.square().mean(1)
        if normals is not None:
            sq = speeds[qi].view(count, -1, 1); sg = speeds[gi].view(count, -1, 1)
            n_term = ((1.001-sq)*(sq*dq+normals[qi].view(count, -1, 6)).square()).sum(2).mean(1)+((1.001-sg)*(sg*dg+normals[gi].view(count, -1, 6)).square()).sum(2).mean(1)
            per_model = per_model+a.normal_weight*n_term
        loss = per_model.sum()
        if not bool(torch.isfinite(loss)): raise FloatingPointError('Non-finite training loss')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm = clip_per_model(ensemble.parameters(), 2.0)
        if not bool(torch.isfinite(norm).all()): raise FloatingPointError('Non-finite gradient')
        optimizer.step()
        with torch.no_grad():
            per_row = residual.detach().view(count, 2, -1).square().mean(1).reshape(-1)
            totals.index_add_(0, seg, per_row.double()); counts.index_add_(0, seg, torch.ones_like(per_row, dtype=torch.float64))
            gap_error.index_add_(0, seg, (gap-target).abs().double())
            outside.index_add_(0, seg, ((gap < sampler.lo[seg]-1e-3) | (gap > sampler.hi[seg]+1e-3)).double())
        if step % 500 == 0 or step == updates:
            row = dict(step=step, elapsed_seconds=time.time()-started, loss=(totals/counts.clamp_min(1)).tolist(), gap_error=(gap_error/counts.clamp_min(1)).tolist(),
                       outside_segment=(outside/counts.clamp_min(1)).tolist())
            history.append(row); print(json.dumps(dict(step=step, seconds=round(row['elapsed_seconds']), loss=[round(x, 4) for x in row['loss']])), flush=True)
            totals.zero_(); counts.zero_(); gap_error.zero_(); outside.zero_()
            write(a.output/'progress.json', dict(complete=False, history=history))
        if step % max(1, updates//5) == 0 or step == updates:
            field_edges = edges if a.mode == 'specialists' else [0.0, edges[-1]]
            save_field(a.output/f'field_u{step:05d}.pt', a.arch, field_edges, a.blend, [ensemble.state_dict_of(m) for m in range(count)],
                       mode=a.mode, updates=step, seed=a.seed, smoke=a.smoke)
    (a.output/'field.pt').write_bytes((a.output/f'field_u{updates:05d}.pt').read_bytes())
    write(a.output/'progress.json', dict(complete=True, history=history, elapsed_seconds=time.time()-started))


if __name__ == '__main__': main()
