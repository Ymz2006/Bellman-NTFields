"""Segment-DP chain trainer with the knobs of tasks 1-5 (dp.py otherwise unchanged; with the defaults it is dp.py bit for bit).

Model m+1 (segment [e_m, e_(m+1)]) is a fresh network trained on pool pairs (i, j) of its segment with any combination
of three losses (--losses, comma list; weights --w-dp --w-td --w-bell):
  dp    regression on the segment-DP target  y(i,j) = min_k T_<=m(i,k) + T_<=m(k,j), |i-k|, |k-j| <= e_m + blend,
        k over --candidates pool poses (dp.targets, unchanged); the regression loss is --loss (task 1):
        smoothl1 (dp.py), l1, l2, logl1 |log T/y|, logl2 (log T/y)^2, rel ((T-y)/y)^2
  td    the TD step of the eikonal field (task 4): along the descent direction u* = -grad_q T / |grad_q T| of the
        model, T(q, g) = T(q + dt u*, g) + dt / S(q), with S(q) read from the gradient of the frozen base specialist:
        S(q) = 1 / |grad_q T_base(q, q + delta v)| for a random unit v (--speed-probe delta). SmoothL1 to the
        detached target; the step value comes from the live chain (frozen models 1..m and the model being trained).
  bell  a Bellman backup in a ball of radius --ball-step around q (task 4/5): T(q, g) = min_u T_base(q, q + rho u)
        + T(q + rho u, g) over --directions random unit directions plus u*; the local cost T_base is the frozen
        base specialist (its trained range), the step value the live chain, the target detached. SmoothL1.
--decoder (task 2) picks the output map of the new models: mult T = |d| (1 + softplus r) (dp.py), add T = |d| +
softplus r (an addition over the lower bound |d|, since speed <= 1), exp T = |d| exp r (no bound; control).
--truth (task 3 and per-level accuracy): the probe pairs of a truth stage; per level the DP targets of its segment's
pairs are recomputed with every candidate count of --sweep (median target / T*, mean |log|, empty fraction), and the
trained model is scored against T* (mean |log T/T*|, median T/T*). Diagnostics use their own generator, so the
dp-only chain is unchanged by them.
--last-segment L trains the chain up to segment L; the remaining segments are padded with the base field's models
(never used by queries shorter than e_L). Writes field.pt, field_segXX.pt, progress.json, config.json.
"""
import sys
sys.path.insert(0, '/clean/method'); sys.path.insert(0, '/clean/jeffrey/segment_dp')
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from common import sha, write
from dp import targets
from fields import SegmentField, build, load_field, save_field, valid_arch
from optimizer import stable_clip_grad_norm_
from sampler import PairSampler, load_pool

LOSSES = ('smoothl1', 'l1', 'l2', 'logl1', 'logl2', 'rel')
DECODERS = dict(mult='pair', add='add', exp='exp')


def regression(kind, value, y):
    if kind == 'smoothl1': return F.smooth_l1_loss(value, y)
    if kind == 'l1': return (value-y).abs().mean()
    if kind == 'l2': return (value-y).square().mean()
    if kind == 'logl1': return (value.clamp_min(1e-6).log()-y.log()).abs().mean()
    if kind == 'logl2': return (value.clamp_min(1e-6).log()-y.log()).square().mean()
    if kind == 'rel': return ((value-y)/y).square().mean()
    raise ValueError(kind)


def unit(x):
    return x/torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-12)


def speed_from_base(base, q, delta, generator):
    """S(q) = 1 / |grad_q T_base(q, q + delta v)|, v a random unit direction (the eikonal condition of the base)."""
    v = unit(torch.randn(q.shape, generator=generator, device=q.device)); other = (q+delta*v).clamp(-.5, .5)
    with torch.enable_grad():
        x = q.detach().clone().requires_grad_(True); grad = torch.autograd.grad(base(x, other)[:, 0].sum(), x)[0]
    return 1.0/torch.linalg.vector_norm(grad, dim=1).clamp_min(1e-6)


@torch.no_grad()
def td_target(live, q, g, grad, step, speed):
    q2 = (q+step*unit(-grad)).clamp(-.5, .5); moved = torch.linalg.vector_norm(q2-q, dim=1)
    return live(q2, g)[:, 0]+moved/speed


@torch.no_grad()
def ball_target(live, base, q, g, grad, rho, directions, generator):
    n = len(q); d = unit(torch.randn((n, directions, 6), generator=generator, device=q.device))
    dirs = torch.cat((unit(-grad)[:, None], d), dim=1); count = directions+1
    p = (q[:, None]+rho*dirs).clamp(-.5, .5).reshape(-1, 6)
    qq, gg = q.repeat_interleave(count, 0), g.repeat_interleave(count, 0)
    total = (base(qq, p)[:, 0]+live(p, gg)[:, 0]).view(n, count)
    best, which = total.min(dim=1)
    return best, (which == 0).float().mean()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--prepared', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True); p.add_argument('--base-models', type=int, default=2)
    p.add_argument('--pairs', type=int, default=400000); p.add_argument('--candidates', type=int, default=384); p.add_argument('--updates', type=int, default=15620)
    p.add_argument('--losses', default='dp'); p.add_argument('--loss', choices=LOSSES, default='smoothl1'); p.add_argument('--decoder', choices=list(DECODERS), default='mult')
    p.add_argument('--w-dp', type=float, default=1.0); p.add_argument('--w-td', type=float, default=1.0); p.add_argument('--w-bell', type=float, default=1.0)
    p.add_argument('--td-step', type=float, default=0.01); p.add_argument('--ball-step', type=float, default=0.05); p.add_argument('--directions', type=int, default=8)
    p.add_argument('--speed-probe', type=float, default=0.04); p.add_argument('--last-segment', type=int, default=20)
    p.add_argument('--truth', type=Path); p.add_argument('--sweep', default='8,32,128,384,1536')
    p.add_argument('--new-arch', help="architecture of the chain's models (default: the decoder's, mult = pair); e.g. pair-h16 or pair-h16-i4")
    p.add_argument('--init', choices=['fresh', 'warm'], default='fresh', help='warm: each new model starts from the previous level\'s weights (level 3 from the last base model) when the architectures match')
    p.add_argument('--normal-weight', type=float, default=0., help='w > 0: add the eikonal specialists\' direction term w (1.001 - S)|S grad T + n|^2 at both ends of every pair (n = unit speed-net gradient)')
    p.add_argument('--seed', type=int, default=87358); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    losses = a.losses.split(','); assert losses and all(x in ('dp', 'td', 'bell') for x in losses), a.losses
    a.output.mkdir(parents=True, exist_ok=False); started = time.time()
    torch.manual_seed(a.seed); generator = torch.Generator(device='cuda:0').manual_seed(a.seed); aux = torch.Generator(device='cuda:0').manual_seed(a.seed+1)
    base_payload = torch.load(a.base, map_location='cuda:0', weights_only=False); assert base_payload['kind'] == 'segment_field'
    edges, blend, base_arch = base_payload['edges'], base_payload['blend'], base_payload['arch']; segments = len(edges)-1
    last = min(segments, a.base_models+2, a.last_segment) if a.smoke else min(segments, a.last_segment)
    pairs, updates = (2048, 40) if a.smoke else (a.pairs, a.updates)
    pool, speeds, speed_net, pool_info = load_pool(a.prepared, 'cuda:0'); square = pool.square().sum(1); normals = None
    if a.normal_weight > 0:   # unit speed gradient at every pool pose (zero where the speed is flat), as in eikonal.py
        parts = []
        for s0 in range(0, len(pool), 32768):
            x = pool[s0:s0+32768].detach().clone().requires_grad_(True)
            with torch.enable_grad(): gs = torch.autograd.grad(speed_net(x).sum(), x)[0]
            parts.append(gs/torch.linalg.vector_norm(gs, dim=1, keepdim=True).clamp_min(1e-6)*(torch.linalg.vector_norm(gs, dim=1, keepdim=True) > 1e-4))
        normals = torch.cat(parts).detach(); del parts
    sampler = PairSampler(pool, edges, blend, generator)
    full = load_field(a.base, 'cuda:0'); models = list(full.models[:a.base_models]); archs = [base_arch]*a.base_models
    base = SegmentField(models[:a.base_models], edges[:a.base_models+1], blend).eval().requires_grad_(False)
    truth = None
    if a.truth:
        z = np.load(a.truth); truth = dict(i=torch.as_tensor(z['source_index'], device='cuda:0'), j=torch.as_tensor(z['goal_index'], device='cuda:0'),
                                            t=torch.as_tensor(z['truth'], device='cuda:0'), gap=torch.as_tensor(z['gap'], device='cuda:0'), sha256=sha(a.truth))
        assert bool((torch.linalg.vector_norm(pool[truth['i']]-pool[truth['j']], dim=1)-truth['gap']).abs().max() < 1e-3), 'truth pairs are not pool pairs of this scene'
    sweep = [int(x) for x in a.sweep.split(',')] if truth is not None else []
    new_arch = a.new_arch or DECODERS[a.decoder]; assert valid_arch(new_arch), new_arch
    config = dict(arch=new_arch, decoder=a.decoder, base_arch=base_arch, edges=edges, blend=blend, base=str(a.base), base_sha256=sha(a.base), base_models=a.base_models,
                  pairs_per_segment=pairs, candidates=a.candidates, updates_per_model=updates, batch=256, losses=losses, regression_loss=a.loss,
                  weights=dict(dp=a.w_dp, td=a.w_td, bell=a.w_bell), td_step=a.td_step, ball_step=a.ball_step, directions=a.directions, speed_probe=a.speed_probe,
                  last_segment=last, truth=str(a.truth) if a.truth else None, truth_sha256=truth['sha256'] if truth else None, sweep=sweep, seed=a.seed, smoke=a.smoke,
                  learning_rate=2e-4, weight_decay=1e-6, gradient_clip=2.0, init=a.init, normal_weight=a.normal_weight,
                  loss=' + '.join({'dp': f'{a.w_dp:g} {a.loss}(T, min_k T(i,k)+T(k,j))', 'td': f'{a.w_td:g} SmoothL1(T, T(q+dt u*)+dt/S_base)',
                                   'bell': f'{a.w_bell:g} SmoothL1(T, min_u T_base(q,q+rho u)+T(q+rho u))'}[x] for x in losses), **pool_info)
    write(a.output/'config.json', config); levels = []
    for m in range(a.base_models, last):
        level_start = time.time(); teacher = SegmentField(models, edges[:m+1], blend).eval().requires_grad_(False); radius = edges[m]+blend
        i, j = [], []
        for s in range(0, pairs, 8192):
            qi, gi, _, _ = sampler.sample(torch.full((min(8192, pairs-s),), m, device='cuda:0')); i.append(qi); j.append(gi)
        i, j = torch.cat(i), torch.cat(j); gap = torch.linalg.vector_norm(pool[i]-pool[j], dim=1)
        stats = dict(segment=m+1, edges=[edges[m], edges[m+1]], radius=radius, pairs=pairs)
        if 'dp' in losses:
            t = targets(teacher, pool, square, i, j, radius, a.candidates, generator); keep = torch.isfinite(t['target'])
            stats.update(kept=int(keep.sum()), empty_lens=float((~keep).float().mean()), lens_median=float(t['lens'].float().median()), near_won=float(t['near_won'][keep].float().mean()),
                         split_quantiles=torch.quantile(t['split'][keep], torch.tensor([.1, .5, .9], device='cuda:0')).tolist(),
                         detour_median=float(t['detour'][keep].median()), target_over_gap_median=float((t['target'][keep]/gap[keep]).median()), target_seconds=time.time()-level_start)
            i, j, y = i[keep], j[keep], t['target'][keep]
        else:
            y = None; stats.update(kept=len(i))
        if truth is not None:   # task 3: the same level's targets for the probe pairs, per candidate count
            sel = torch.nonzero((truth['gap'] >= edges[m]) & (truth['gap'] < edges[m+1])).flatten(); ti, tj, ts = truth['i'][sel], truth['j'][sel], truth['t'][sel]
            sweep_stats = {}
            for c in sweep:
                tc = targets(teacher, pool, square, ti, tj, radius, c, aux)['target']; ok = torch.isfinite(tc); r = tc[ok]/ts[ok]
                sweep_stats[str(c)] = dict(median_over_truth=float(r.median()), mean_abs_log=float(r.log().abs().mean()), empty=float((~ok).float().mean()))
            stats.update(truth_pairs=int(len(sel)), candidate_sweep=sweep_stats)
        order = torch.randperm(len(i), generator=generator, device='cuda:0'); hold = order[:len(i)//50]; train = order[len(i)//50:]
        model = build(new_arch).cuda().train()
        stats['warm_start'] = a.init == 'warm' and archs[-1] == new_arch   # false for the first level of a chain whose decoder differs from the base's
        if stats['warm_start']:
            model.load_state_dict(models[-1].state_dict())
        if y is not None:   # held-out fit of the initial weights, before any update (a warm start's inherited level)
            with torch.no_grad():
                v0 = torch.cat([model(pool[i[hold[s:s+8192]]], pool[j[hold[s:s+8192]]])[:, 0] for s in range(0, len(hold), 8192)])
                stats['init_fit'] = dict(smooth_l1=float(F.smooth_l1_loss(v0, y[hold])), mean_abs_log_ratio=float((v0/y[hold]).log().abs().mean()))
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-6)
        live = SegmentField(models+[model], edges[:m+2], blend); need_grad = ('td' in losses) or ('bell' in losses)
        acc = dict(dp=0.0, td=0.0, bell=0.0, normal=0.0, n=0, speed_ratio=[], descent_won=0.0); history = []
        for step in range(1, updates+1):
            rows = train[torch.randint(len(train), (256,), generator=generator, device='cuda:0')]
            q, g = pool[i[rows]], pool[j[rows]]
            if need_grad or normals is not None:
                q = q.clone().requires_grad_(True)
            if normals is not None:
                g = g.clone().requires_grad_(True)
            value = model(q, g)[:, 0]; loss = value.new_zeros(()); parts = {}
            if 'dp' in losses:
                parts['dp'] = regression(a.loss, value, y[rows]); loss = loss+a.w_dp*parts['dp']
            if normals is not None:   # direction term at both ends (eikonal.py), differentiated through the gradient
                dq, dg = torch.autograd.grad(value.sum(), (q, g), create_graph=True)
                sq = speeds[i[rows]].view(-1, 1); sg = speeds[j[rows]].view(-1, 1)
                parts['normal'] = (((1.001-sq)*(sq*dq+normals[i[rows]]).square()).sum(1)+((1.001-sg)*(sg*dg+normals[j[rows]]).square()).sum(1)).mean()
                loss = loss+a.normal_weight*parts['normal']
            if need_grad:
                grad = torch.autograd.grad(value.sum(), q, retain_graph=True)[0].detach(); qd = q.detach(); g = g.detach()
                s_est = speed_from_base(base, qd, a.speed_probe, aux)
                if 'td' in losses:
                    parts['td'] = F.smooth_l1_loss(value, td_target(live, qd, g, grad, a.td_step, s_est)); loss = loss+a.w_td*parts['td']
                if 'bell' in losses:
                    tb, won = ball_target(live, base, qd, g, grad, a.ball_step, a.directions, aux)
                    parts['bell'] = F.smooth_l1_loss(value, tb); loss = loss+a.w_bell*parts['bell']; acc['descent_won'] += float(won)
                if step % 50 == 0:
                    acc['speed_ratio'].append(float((s_est/speeds[i[rows]]).log().abs().median()))
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Non-finite training loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            norm = stable_clip_grad_norm_(model.parameters(), 2.0)
            if not bool(torch.isfinite(norm)): raise FloatingPointError('Non-finite gradient')
            optimizer.step()
            for k, v in parts.items(): acc[k] += float(v)
            acc['n'] += 1
            if step % 500 == 0 or step == updates:
                row = dict(step=step, **{k: acc[k]/acc['n'] for k in ('dp', 'td', 'bell', 'normal') if k in parts})
                if need_grad: row['speed_median_abs_log_ratio'] = float(np.median(acc['speed_ratio'])) if acc['speed_ratio'] else None
                if 'bell' in losses: row['descent_direction_won'] = acc['descent_won']/acc['n']
                history.append(row); acc = dict(dp=0.0, td=0.0, bell=0.0, normal=0.0, n=0, speed_ratio=[], descent_won=0.0)
        model.eval().requires_grad_(False)
        with torch.no_grad():
            def fit(rows):
                v = torch.cat([model(pool[i[rows[s:s+8192]]], pool[j[rows[s:s+8192]]])[:, 0] for s in range(0, len(rows), 8192)])
                return dict(smooth_l1=float(F.smooth_l1_loss(v, y[rows])), mean_abs_log_ratio=float((v/y[rows]).log().abs().mean()))
            if y is not None:
                stats.update(train=fit(train[:len(hold)]), held_out=fit(hold))
            if truth is not None:
                v = model(pool[ti], pool[tj])[:, 0]; r = v/ts
                stats['truth_fit'] = dict(mean_abs_log=float(r.log().abs().mean()), median_over_truth=float(r.median()))
            stats.update(history=history, seconds=time.time()-level_start)
        models.append(model); archs.append(new_arch); levels.append(stats); print(json.dumps(stats), flush=True)
        save_field(a.output/f'field_seg{m+1:02d}.pt', base_arch, edges[:m+2], blend, [x.state_dict() for x in models], archs=archs, segments_trained=m+1, seed=a.seed, smoke=a.smoke)
        write(a.output/'progress.json', dict(complete=False, levels=levels, elapsed_seconds=time.time()-started))
    if last < segments:   # pad with the base field's remaining models so the field covers every segment
        models += list(full.models[last:]); archs += [base_arch]*(segments-last)
    save_field(a.output/'field.pt', base_arch, edges, blend, [x.state_dict() for x in models], archs=archs, segments_trained=last, seed=a.seed, smoke=a.smoke)
    write(a.output/'progress.json', dict(complete=True, levels=levels, elapsed_seconds=time.time()-started))


if __name__ == '__main__': main()
