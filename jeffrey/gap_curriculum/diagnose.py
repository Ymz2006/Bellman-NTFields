"""How far each round has filled the travel-cost table, by gap (diagnostic only; never selects a checkpoint).

A fixed probe set (seed 20261027, independent of training): 8192 bank sources, each paired with the bank source
whose distance is closest to a gap drawn uniformly in [0, 2]. For every round checkpoint T_k and gap bin:
  ratio     T_k(q, g) / |q - g|
  fixed     (T_k - B T_k) / B T_k, B the bank's Bellman backup with T_k itself as the tail (0 at a fixed point)
  fit       (T_k - B T_{k-1}) / B T_{k-1}, the error against the target round k was trained on
  eikonal   mean of log(S(q) |grad_q T_k|)^2 at the probe pairs
"""
import sys
sys.path.insert(0, '/clean/method')
import argparse, importlib.util, json
from pathlib import Path
import numpy as np
import torch
from common import write
from network import TravelCost, load_checkpoint
from speed import load_speed
from loss import bellman_backup

_spec = importlib.util.spec_from_file_location('gap_training', Path(__file__).with_name('training.py'))
TRAINING = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(TRAINING)
BINS = [0, .1, .2, .3, .4, .5, .6, .8, 1.0, 1.2, 1.4, 2.0]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--training', type=Path, required=True); p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--probes', type=int, default=8192); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    bank = a.prepared/'bank'
    source, successor, cost, safe = [np.load(bank/n, mmap_mode='r') for n in ('source_q.npy', 'successor_q.npy', 'local_cost.npy', 'safe.npy')]
    rng = np.random.default_rng(20261027); rows = np.sort(rng.choice(len(source), 512 if a.smoke else a.probes, replace=False))
    pool = torch.as_tensor(np.asarray(source).copy(), device='cuda:0'); sampler = TRAINING.GapSampler(pool, 20261027)
    q, z, c, s = [torch.as_tensor(np.asarray(x[rows]).copy(), device='cuda:0') for x in (source, successor, cost, safe)]; s = s.bool()
    picks = torch.cat([sampler.shell(q[i:i+256], torch.as_tensor(rows[i:i+256], device='cuda:0'), torch.full((len(q[i:i+256]),), 2.0, device='cuda:0'))[0] for i in range(0, len(q), 256)])
    g = pool[picks]; gap = torch.linalg.vector_norm(q-g, dim=1)
    edges = torch.tensor(BINS, device='cuda:0'); which = torch.bucketize(gap, edges[1:-1])
    speed = load_speed(a.prepared/'speed.pt', 'cuda:0'); sq = speed(q).reshape(-1)
    checkpoints = sorted(a.training.glob('round_*.pt')); rounds = []
    previous = TravelCost().cuda().eval().requires_grad_(False)   # T_0: zero output layer, exactly 1.05 |q - g| for any init

    def per_bin(values):
        return [None if not bool((which == b).any()) else float(values[which == b].mean()) for b in range(len(BINS)-1)]

    for path in checkpoints:
        net = load_checkpoint(path, 'cuda:0')
        with torch.no_grad():
            value = torch.cat([net(q[i:i+1024], g[i:i+1024])[:, 0] for i in range(0, len(q), 1024)])
            backup = torch.cat([bellman_backup(net, z[i:i+256], g[i:i+256], c[i:i+256], s[i:i+256])[0] for i in range(0, len(q), 256)])
            fit = torch.cat([bellman_backup(previous, z[i:i+256], g[i:i+256], c[i:i+256], s[i:i+256])[0] for i in range(0, len(q), 256)])
        x = q.clone().requires_grad_(True); grad = torch.autograd.grad(net.requires_grad_(False)(x, g).sum(), x)[0]
        eik = torch.log((sq*grad.norm(dim=1)).clamp_min(1e-8)).square().detach()
        row = dict(checkpoint=path.name, ratio=per_bin(value/gap), fixed=per_bin((value-backup)/backup), abs_fixed=per_bin(((value-backup)/backup).abs()),
                   eikonal=per_bin(eik), fit=per_bin((value-fit)/fit), abs_fit=per_bin(((value-fit)/fit).abs()))
        rounds.append(row); previous = net
    counts = [int((which == b).sum()) for b in range(len(BINS)-1)]
    write(a.output, dict(bins=BINS, counts=counts, probes=len(rows), probe_seed=20261027, rounds=rounds))


if __name__ == '__main__': main()
