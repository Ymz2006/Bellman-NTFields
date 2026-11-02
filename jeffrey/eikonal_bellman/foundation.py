"""Eikonal foundation with augmented support: the short-range eikonal model (segment_dp/eikonal.py --mode generalist on
one segment) trained on pool pairs PLUS pairs with one off-pool endpoint, so the field is defined where off-pool split
points of the Bellman extension are read (the blind-spot pitfall of continuous split points: the pool-only foundation was
never trained inside obstacles or off the free manifold and reads low there).

Rows of every batch: (1 - --augment) pool pairs of segment [0, e_1] (sampler.py, gaps 0.004 .. e_1 + blend) with the
pool's speeds; --augment rows take such a pair and move the goal off the pool, g' = g + N(0, --sigma^2 I) clamped to the
joint box (with probability --swap the source is moved instead), with the frozen speed network evaluated at the moved
point (its speed floor is what the eikonal residual sees inside obstacles). Loss, optimizer, updates and the saved
one-segment field are those of eikonal.py (eikonal residual only; AdamW 2e-4, weight decay 1e-6, clip 2, batch 256).
"""
import sys
sys.path.insert(0, '/clean/method'); sys.path.insert(0, '/clean/jeffrey/segment_dp')
import argparse, json, time
from pathlib import Path
import torch
from common import write
from fields import save_field
from network import TravelCost
from optimizer import stable_clip_grad_norm_
from sampler import PairSampler, load_pool


def main():
    p = argparse.ArgumentParser(); p.add_argument('--prepared', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--edges', default='0,0.16'); p.add_argument('--blend', type=float, default=0.016); p.add_argument('--updates', type=int, default=15620)
    p.add_argument('--augment', type=float, default=0.5); p.add_argument('--sigma', type=float, default=0.05); p.add_argument('--swap', type=float, default=0.5)
    p.add_argument('--batch', type=int, default=256); p.add_argument('--seed', type=int, default=87358); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False); started = time.time(); dev = 'cuda:0'
    edges = [float(x) for x in a.edges.split(',')]; assert len(edges) == 2; updates = 40 if a.smoke else a.updates
    torch.manual_seed(a.seed); generator = torch.Generator(device=dev).manual_seed(a.seed)
    pool, speeds, speed, pool_info = load_pool(a.prepared, dev)
    net = TravelCost().to(dev).train(); optimizer = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-6)
    sampler = PairSampler(pool, edges, a.blend, generator); n_aug = int(round(a.batch*a.augment))
    config = dict(mode='generalist-augmented', arch='pair', edges=edges, blend=a.blend, models=1, updates_per_model=updates, batch_per_model=a.batch, augment_rows=n_aug, sigma=a.sigma, swap=a.swap,
                  seed=a.seed, learning_rate=2e-4, weight_decay=1e-6, gradient_clip=2.0, loss='eikonal (method/loss.py eikonal term only), off-pool endpoints with on-the-fly speeds', smoke=a.smoke,
                  eligible_sources=[len(e) for e in sampler.eligible], **pool_info)
    write(a.output/'config.json', config); history = []; acc = dict(loss_pool=0.0, loss_aug=0.0, speed_aug=0.0, gap_aug=0.0, n=0)
    for step in range(1, updates+1):
        qi, gi, gap, _ = sampler.sample(torch.zeros(a.batch, dtype=torch.long, device=dev))
        q, g = pool[qi].clone(), pool[gi].clone(); sq, sg = speeds[qi].clone(), speeds[gi].clone()
        if n_aug:
            noise = torch.randn((n_aug, 6), generator=generator, device=dev)*a.sigma
            swap = torch.rand(n_aug, generator=generator, device=dev) < a.swap
            moved_g = (g[:n_aug]+noise).clamp(-.5, .5); moved_q = (q[:n_aug]+noise).clamp(-.5, .5)
            g[:n_aug] = torch.where(swap[:, None], g[:n_aug], moved_g); q[:n_aug] = torch.where(swap[:, None], moved_q, q[:n_aug])
            with torch.no_grad():
                s_moved = speed(torch.where(swap[:, None], q[:n_aug], g[:n_aug])).reshape(-1)
            sq[:n_aug] = torch.where(swap, s_moved, sq[:n_aug]); sg[:n_aug] = torch.where(swap, sg[:n_aug], s_moved)
        q.requires_grad_(True); g.requires_grad_(True)
        value = net(q, g)[:, 0]; dq, dg = torch.autograd.grad(value.sum(), (q, g), create_graph=True)
        nq = torch.sqrt(dq.square().sum(1)+1e-12); ng = torch.sqrt(dg.square().sum(1)+1e-12)
        residual = torch.stack((torch.log((sq*nq).clamp_min(1e-8)), torch.log((sg*ng).clamp_min(1e-8))), 1)   # [B, 2]
        per_row = residual.square().mean(1); loss = per_row.mean()
        if not bool(torch.isfinite(loss)): raise FloatingPointError('Non-finite training loss')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm = stable_clip_grad_norm_(net.parameters(), 2.0)
        if not bool(torch.isfinite(norm)): raise FloatingPointError('Non-finite gradient')
        optimizer.step()
        with torch.no_grad():
            acc['loss_aug'] += float(per_row[:n_aug].mean()) if n_aug else 0.0; acc['loss_pool'] += float(per_row[n_aug:].mean())
            if n_aug: acc['speed_aug'] += float(s_moved.mean()); acc['gap_aug'] += float(torch.linalg.vector_norm(q[:n_aug]-g[:n_aug], dim=1).mean())
            acc['n'] += 1
        if step % 500 == 0 or step == updates:
            row = dict(step=step, elapsed_seconds=time.time()-started, **{k: v/acc['n'] for k, v in acc.items() if k != 'n'}); history.append(row)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True); acc = dict.fromkeys(acc, 0.0); acc['n'] = 0
            write(a.output/'progress.json', dict(complete=False, history=history))
    save_field(a.output/'field.pt', 'pair', edges, a.blend, [net.state_dict()], mode='generalist-augmented', updates=updates, seed=a.seed, smoke=a.smoke)
    write(a.output/'progress.json', dict(complete=True, history=history, elapsed_seconds=time.time()-started))


if __name__ == '__main__': main()
