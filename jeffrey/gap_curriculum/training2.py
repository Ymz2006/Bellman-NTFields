"""training.py plus --split: which loss terms see the curriculum goals (a decomposition of the curriculum's effect).

  --split none        training.py exactly (all three terms on the drawn goals)
  --split eikonal     Bellman + winner direction on the natural goals, Eikonal on the curriculum goals (same sources)
  --split bellman     Bellman + winner direction on the curriculum goals, Eikonal on the natural goals
The losses, weights and their accumulation order are method/loss.py's; with a split the direction term takes the
gradient at its own (Bellman) pairs from one extra forward pass. Everything below is training.py's description.

Bellman TravelCost training whose goal pairs follow a distance (gap) curriculum, as in interval DP.

Interval DP fills dp[i][j] by increasing gap j - i, so every entry is built from entries of smaller gap that are
already final. Here round k trains only pairs (q, g) with |q - g| <= r_k (normalized joint units) against the
frozen round k-1 network, so the Bellman tail T_{k-1}(successor, g) is queried mostly at gaps the previous round
has already fitted; r_k grows until the last round covers every pair of the bank.

Everything else is experiments/from_source/training.py unchanged: the 338,177-parameter TravelCost, the three
losses 6/2/2 (method/loss.py), AdamW 2e-4, clip 2, batch 256, seed 87358, the target frozen per round. The source
permutation and the natural goal draws come from the same numpy generator in the same order; a curriculum also
draws (and discards) the natural goals, so every schedule visits the same sources in the same order and
--goals natural --rounds 5 is the baseline itself.

  --goals natural   goals uniform over the bank sources (the baseline)
  --goals ball      round k: goal uniform over the bank sources within r_k of the source (not the source itself)
  --goals shell     round k: gap uniform in [0, r_k], goal = the bank source whose distance is closest to it
  --goals value     round k: a goal whose travel time under the frozen round k-1 network is <= r_k (Dijkstra-like
                    order; candidates from the Euclidean ball of radius r_k, since T >= |q - g|)
  --mix             control: every pair draws its radius uniformly from --radii, in every round (the same
                    mixture of gaps over the run as the curriculum, without the order)
  --min-step S      mechanism test, changes the Bellman target: the backup (and the winner direction) only use
                    safe actions of step >= S, so successors are genuinely closer subproblems; rows without such
                    an action keep all safe actions
  --grow-mix        curriculum with rehearsal: in round k every pair draws its radius uniformly from r_1..r_k, so
                    the radii introduced so far keep equal shares and the last round is the --mix distribution
"""
import sys
sys.path.insert(0, "/clean/method")
import argparse
import copy
import json
import random
import time
import numpy as np
import torch
from pathlib import Path
from common import sha, write
from network import TravelCost, CONFIG
from speed import load_speed
from loss import training_loss, WEIGHTS, bellman_backup, eikonal_loss, direction_loss
from torch.nn import functional as F
from optimizer import stable_clip_grad_norm_


class GapSampler:
    """Goals among the bank sources by distance to each source, on the GPU.

    subset > 0: each batch first looks in a random subset of the pool (drawn with replacement, shared by the rows);
    rows whose ball misses the subset look in the whole pool. Every source in the ball stays equally likely (the
    subset is exchangeable), so ball goals keep the full-pool distribution at a fraction of the memory traffic."""

    def __init__(self, pool, seed, subset=0):
        self.pool = pool
        self.square = pool.square().sum(dim=1)
        self.generator = torch.Generator(device=pool.device).manual_seed(seed)
        self.subset = subset

    def distances(self, q, rows, columns=None):
        # |q|^2 + |p|^2 - 2 q.p in float32 (TF32 off): error ~1e-7 in d^2, far below the 0.01 smallest step.
        pool, square = (self.pool, self.square) if columns is None else (self.pool[columns], self.square[columns])
        d = torch.addmm(square[None, :], q, pool.T, alpha=-2.0).add_(q.square().sum(dim=1, keepdim=True)).clamp_min_(0.0).sqrt_()
        if columns is None:
            d[torch.arange(len(q), device=q.device), rows] = torch.inf
        else:
            d.masked_fill_(columns[None, :] == rows[:, None], torch.inf)
        return d.masked_fill_(d < 1e-4, torch.inf)        # never the source itself or one of its duplicates

    def _ball(self, q, rows, radius, columns):
        d = self.distances(q, rows, columns)
        inside = d <= radius[:, None]
        keys = torch.rand(d.shape, generator=self.generator, device=d.device).masked_fill_(~inside, -1.0)
        pick = keys.argmax(dim=1)
        return (pick if columns is None else columns[pick]), ~inside.any(dim=1), d

    def ball(self, q, rows, radius):
        if self.subset:
            columns = torch.randint(len(self.pool), (self.subset,), generator=self.generator, device=q.device)
            pick, missed, _ = self._ball(q, rows, radius, columns); empty = 0
            if bool(missed.any()):
                pick[missed], empty = self.ball_full(q[missed], rows[missed], radius[missed])
            return pick, empty
        return self.ball_full(q, rows, radius)

    def ball_full(self, q, rows, radius):
        pick, empty, d = self._ball(q, rows, radius, None)
        if bool(empty.any()):
            pick[empty] = d[empty].argmin(dim=1)        # no source within the radius: the nearest one
        return pick, int(empty.sum())

    def shell(self, q, rows, radius):
        d = self.distances(q, rows)
        gap = torch.rand(len(q), generator=self.generator, device=d.device)*radius
        return d.sub_(gap[:, None]).abs_().argmin(dim=1), 0

    def value(self, q, rows, radius, target, candidates=16):
        """Dijkstra-like order: a goal whose travel time under the frozen previous round is within the radius.
        Candidates are random sources inside the Euclidean ball of the same radius (T >= |q - g| by construction);
        the first in random order with T_{k-1}(q, g) <= radius is taken, else the candidate with the smallest T."""
        d = self.distances(q, rows)
        keys = torch.rand(d.shape, generator=self.generator, device=d.device).masked_fill_(d > radius[:, None], -1.0)
        top = keys.topk(candidates, dim=1); index = top.indices; inside = top.values >= 0
        with torch.no_grad():
            value = target(q.repeat_interleave(candidates, dim=0), self.pool[index.reshape(-1)])[:, 0].reshape(index.shape)
        value = value.masked_fill(~inside, torch.inf); accept = value <= radius[:, None]
        rows_ = torch.arange(len(q), device=q.device)
        pick = torch.where(accept.any(dim=1), index[rows_, accept.float().argmax(dim=1)], index[rows_, value.argmin(dim=1)])
        empty = ~inside.any(dim=1)
        if bool(empty.any()):
            pick[empty] = d[empty].argmin(dim=1)
        return pick, int((~accept.any(dim=1)).sum())


def split_loss(model, target, speed, q, bellman_goal, eikonal_goal, successors, cost, safe):
    backup, candidates = bellman_backup(target, successors, bellman_goal, cost, safe)
    value = model(q, bellman_goal)[:, 0]
    bellman = F.smooth_l1_loss(value, backup)
    x = q.detach().requires_grad_(True)
    gradient = torch.autograd.grad(model(x, bellman_goal)[:, 0].sum(), x, create_graph=True)[0]
    direction = direction_loss(gradient, q, successors, cost, safe, candidates)
    eikonal, _ = eikonal_loss(model, speed, q, eikonal_goal)
    total = sum((6.0*bellman, 2.0*eikonal, 2.0*direction), value.new_zeros(()))
    return total, dict(bellman=bellman, eikonal=eikonal, direction=direction)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True, help='directory with bank/ and speed.pt')
    p.add_argument('--rounds', type=int, default=10)
    p.add_argument('--epochs-per-round', type=int, default=2)
    p.add_argument('--goals', choices=['natural', 'ball', 'shell', 'value'], default='natural')
    p.add_argument('--radii', default='', help='comma-separated r_k, one per round (ball/shell)')
    p.add_argument('--mix', action='store_true', help='control: radius per pair drawn from --radii in every round')
    p.add_argument('--grow-mix', action='store_true', help='round k: radius per pair drawn from the first k radii')
    p.add_argument('--min-step', type=float, default=0.0, help='mechanism test: backup only over safe actions with step >= this')
    p.add_argument('--split', choices=['none', 'eikonal', 'bellman'], default='none', help='which terms see the curriculum goals')
    p.add_argument('--seed', type=int, default=87358)
    p.add_argument('--pool-subset', type=int, default=0, help='ball goals: look in a random subset of this size first (same distribution, faster)')
    p.add_argument('--smoke', action='store_true', help='Two rounds, two minibatches per epoch; not a benchmark result')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    bank = args.prepared/'bank'
    manifest = json.loads((bank/'manifest.json').read_text())
    for name, expected in manifest['file_sha256'].items():
        assert sha(bank/name)==expected, name
    assert sha(args.prepared/'speed.pt')==manifest['speed_checkpoint_sha256']
    arrays = [np.load(bank/name, mmap_mode='r') for name in
              ('source_q.npy', 'successor_q.npy', 'local_cost.npy', 'safe.npy')]
    source, successor, cost, safe = arrays
    rounds = 2 if args.smoke else args.rounds
    radii = [float(x) for x in args.radii.split(',') if x]
    if args.goals == 'natural':
        assert not radii and not args.mix and not args.grow_mix and args.split == 'none', 'natural goals take no radii or split'
    else:
        assert len(radii) == args.rounds, f'{len(radii)} radii for {args.rounds} rounds'
        if args.smoke:
            radii = [radii[0], radii[-1]]
    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    generator = np.random.default_rng(seed)
    # Construction order preserves the baseline initialization RNG sequence.
    speed = load_speed(args.prepared/'speed.pt', 'cuda:0')
    model = TravelCost().cuda().train()
    target = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-6)
    sampler = None if args.goals == 'natural' else GapSampler(torch.as_tensor(np.asarray(source).copy(), device='cuda:0'), seed+1, args.pool_subset)
    radius_table = torch.tensor(radii, device='cuda:0') if radii else None
    large = torch.as_tensor(np.load(bank/'action_step_size.npy') >= args.min_step-1e-9, device='cuda:0') if args.min_step else None
    mix_generator = torch.Generator(device='cuda:0').manual_seed(seed+2)
    steps_per_epoch = 2 if args.smoke else (len(source)+255)//256
    config = dict(model=CONFIG, losses=WEIGHTS, seed=seed, rounds=rounds,
                  epochs_per_round=1 if args.smoke else args.epochs_per_round, batch_size=256,
                  learning_rate=2e-4, weight_decay=1e-6, gradient_clip=2.0, goals=args.goals, radii=radii, mix=args.mix, grow_mix=args.grow_mix, pool_subset=args.pool_subset, min_step=args.min_step, split=args.split,
                  bank_manifest_sha256=sha(bank/'manifest.json'), speed_sha256=sha(args.prepared/'speed.pt'), smoke=args.smoke,
                  source_count=len(source), scene_sha256=manifest['scene_sha256'],
                  expected_updates=rounds*(1 if args.smoke else args.epochs_per_round)*steps_per_epoch)
    write(args.output/'config.json', config)
    started = time.time(); step = 0; history = []
    for round_id in range(1, rounds+1):
        gaps = []; fallback = 0; sampling = 0.0; restricted = 0
        for epoch in range(config['epochs_per_round']):
            order = generator.permutation(len(source))
            for batch in range(steps_per_epoch):
                indices = order[batch*256:min((batch+1)*256, len(source))]
                goals = generator.choice(len(source), size=len(indices), replace=False)
                q,z,c,s = [torch.as_tensor(np.asarray(a[indices]).copy(),device='cuda:0') for a in arrays]
                if sampler is None:
                    g = torch.as_tensor(np.asarray(source[goals]).copy(),device='cuda:0')
                else:
                    begin = time.time()
                    if args.mix or args.grow_mix:
                        k = len(radii) if args.mix else round_id
                        radius = radius_table[torch.randint(k, (len(indices),), generator=mix_generator, device='cuda:0')]
                    else:
                        radius = radius_table[round_id-1].expand(len(indices))
                    extra = (target,) if args.goals == 'value' else ()
                    pick, empty = getattr(sampler, args.goals)(q, torch.as_tensor(indices, device='cuda:0'), radius, *extra)
                    g = sampler.pool[pick]; fallback += empty; sampling += time.time()-begin
                gaps.append(torch.linalg.vector_norm(q-g, dim=1).detach())
                s = s.bool()
                if large is not None:
                    allowed = s & large[None, :]; keep = allowed.any(dim=1)
                    s = torch.where(keep[:, None], allowed, s); restricted += int(keep.sum())
                optimizer.zero_grad(set_to_none=True)
                if args.split == 'none':
                    total, terms = training_loss(model,target,speed,q,g,z,c,s)
                else:
                    natural = torch.as_tensor(np.asarray(source[goals]).copy(),device='cuda:0')
                    bellman_goal, eikonal_goal = (natural, g) if args.split == 'eikonal' else (g, natural)
                    total, terms = split_loss(model,target,speed,q,bellman_goal,eikonal_goal,z,c,s)
                if not bool(torch.isfinite(total)):raise FloatingPointError('Non-finite training loss')
                total.backward()
                norm=stable_clip_grad_norm_(model.parameters(),2.0)
                if not bool(torch.isfinite(norm)):raise FloatingPointError('Non-finite gradient')
                optimizer.step();step+=1
                if args.smoke or step%100==0:
                    print(json.dumps(dict(step=step,round=round_id,loss=float(total.detach()),
                          **{k:float(v.detach()) for k,v in terms.items()},gap=float(gaps[-1].mean()),elapsed_seconds=time.time()-started)),flush=True)
        target.load_state_dict(model.state_dict())
        gap = torch.cat(gaps)
        history.append(dict(round=round_id, radius=None if not radii or args.mix else (radii[:round_id] if args.grow_mix else radii[round_id-1]), pairs=len(gap),
                            gap_mean=float(gap.mean()), gap_quantiles={str(x): float(v) for x, v in
                            zip((.05, .25, .5, .75, .95), torch.quantile(gap[:1_000_000], torch.tensor((.05, .25, .5, .75, .95), device=gap.device)))},
                            nearest_fallback_rows=fallback, sampling_seconds=sampling, restricted_rows=restricted))
        payload=dict(model_config=CONFIG,model_state_dict=model.state_dict(),loss_weights=WEIGHTS,
                     round=round_id,optimizer_steps=step,training_seed=seed,smoke=args.smoke)
        torch.save(payload,args.output/f'round_{round_id:02d}.pt')
        torch.save(dict(**payload,optimizer_state_dict=optimizer.state_dict(),numpy_generator_state=generator.bit_generator.state),args.output/'recovery.pt')
        write(args.output/'progress.json',dict(complete=round_id==rounds,round=round_id,optimizer_steps=step,elapsed_seconds=time.time()-started,rounds=history))
    assert step==config['expected_updates']
    torch.save(payload,args.output/'checkpoint.pt')


if __name__=='__main__':main()
