"""Original Bellman TravelCost training with the Bellman min taken over D random-walk direction pairs.

jeffrey/segment_dp/ablation/train.py (the original method, weights 6,2,2) with one change: the candidate actions of
every source come from the nested extension bank of extend.py, pairs[:D] (8*D actions: 4 step sizes x 2 signs x D
directions), instead of the frozen 32. Source poses, batches, goals, seeds, network, losses, rounds and the 15,620
updates are unchanged. --pairs 4 reads exactly the frozen bank arrays in the frozen order, so it reproduces the
baseline checkpoint bit for bit.

With D < 4 some sources have no safe action (1.8 % at D = 1, 0.4 % at D = 2). Those rows are left out of the
Bellman and direction terms of their batch (they keep the eikonal term); batches without such rows use the original
loss code unchanged.
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
from torch.nn import functional as F
from common import sha, write
from network import TravelCost, CONFIG
from speed import load_speed
from loss import bellman_backup, eikonal_loss, direction_loss, training_loss
from optimizer import stable_clip_grad_norm_


def masked_loss(model, target, speed, q, goal, successors, cost, safe):
    valid = safe.any(dim=1)
    if bool(valid.all()):
        return training_loss(model, target, speed, q, goal, successors, cost, safe)
    backup, candidates = bellman_backup(target, successors[valid], goal[valid], cost[valid], safe[valid])
    value = model(q, goal)[:, 0]
    bellman = F.smooth_l1_loss(value[valid], backup)
    eikonal, gradient = eikonal_loss(model, speed, q, goal)
    direction = direction_loss(gradient[valid], q[valid], successors[valid], cost[valid], safe[valid], candidates)
    total = sum((6.0*bellman, 2.0*eikonal, 2.0*direction), value.new_zeros(()))
    return total, dict(bellman=bellman, eikonal=eikonal, direction=direction)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--ext', type=Path, required=True, help='extend.py output folder (holds ext/)')
    p.add_argument('--pairs', type=int, required=True)
    p.add_argument('--seed', type=int, default=87358)
    p.add_argument('--smoke', action='store_true', help='Two rounds, two minibatches per epoch; not a benchmark result')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    bank = args.prepared/'bank'; ext = args.ext/'ext'
    manifest = json.loads((bank/'manifest.json').read_text()); extm = json.loads((ext/'manifest.json').read_text())
    for name, expected in manifest['file_sha256'].items():
        if name in ('source_q.npy',): assert sha(bank/name)==expected, name
    for name, expected in extm['file_sha256'].items():
        assert sha(ext/name)==expected, name
    assert extm['source_q_sha256']==sha(bank/'source_q.npy') and extm['speed_sha256']==sha(args.prepared/'speed.pt')
    assert sha(args.prepared/'speed.pt')==manifest['speed_checkpoint_sha256']
    D = args.pairs; assert 1 <= D <= extm['pairs']
    source = np.load(bank/'source_q.npy', mmap_mode='r')
    successor, cost, safe = [np.load(ext/(name+'.npy'), mmap_mode='r') for name in ('successor_q', 'local_cost', 'safe')]
    assert extm['sources']==len(successor)==(len(source) if not extm['smoke'] else extm['sources'])
    source = source[:extm['sources']]   # a smoke extension covers only the first sources
    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    generator = np.random.default_rng(seed)
    # Construction order preserves the baseline initialization RNG sequence.
    speed = load_speed(args.prepared/'speed.pt', 'cuda:0')
    model = TravelCost().cuda().train()
    target = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-6)
    config = dict(model=CONFIG, losses=dict(bellman=6.0, eikonal=2.0, direction=2.0), seed=seed, rounds=2 if args.smoke else 5,
                  epochs_per_round=1 if args.smoke else 2, batch_size=256, direction_pairs=D, actions=8*D,
                  learning_rate=2e-4, weight_decay=1e-6, gradient_clip=2.0, ext_manifest_sha256=sha(ext/'manifest.json'),
                  bank_manifest_sha256=sha(bank/'manifest.json'), speed_sha256=sha(args.prepared/'speed.pt'), smoke=args.smoke, source_count=len(source),
                  scene_sha256=manifest['scene_sha256'], expected_updates=4 if args.smoke else 10*((len(source)+255)//256))
    write(args.output/'config.json', config)
    steps_per_epoch = 2 if args.smoke else (len(source)+255)//256
    started = time.time(); step = 0; masked_rows = 0
    for round_id in range(1, config['rounds']+1):
        for epoch in range(config['epochs_per_round']):
            order = generator.permutation(len(source))
            for batch in range(steps_per_epoch):
                indices = order[batch*256:min((batch+1)*256, len(source))]
                goals = generator.choice(len(source), size=len(indices), replace=False)
                n = len(indices)
                q = torch.as_tensor(np.asarray(source[indices]).copy(), device='cuda:0')
                z = torch.as_tensor(np.ascontiguousarray(np.asarray(successor[indices])[:, :, :, :D].reshape(n, 8*D, 6)), device='cuda:0')
                c = torch.as_tensor(np.ascontiguousarray(np.asarray(cost[indices])[:, :, :, :D].reshape(n, 8*D)), device='cuda:0')
                s = torch.as_tensor(np.ascontiguousarray(np.asarray(safe[indices])[:, :, :, :D].reshape(n, 8*D)), device='cuda:0')
                g = torch.as_tensor(np.asarray(source[goals]).copy(), device='cuda:0')
                masked_rows += int((~s.any(1)).sum())
                optimizer.zero_grad(set_to_none=True)
                total, terms = masked_loss(model, target, speed, q, g, z, c, s)
                if not bool(torch.isfinite(total)):raise FloatingPointError('Non-finite training loss')
                total.backward()
                norm=stable_clip_grad_norm_(model.parameters(),2.0)
                if not bool(torch.isfinite(norm)):raise FloatingPointError('Non-finite gradient')
                optimizer.step();step+=1
                if args.smoke or step%100==0:
                    print(json.dumps(dict(step=step,round=round_id,loss=float(total.detach()),
                          **{k:float(v.detach()) for k,v in terms.items()},elapsed_seconds=time.time()-started)),flush=True)
        target.load_state_dict(model.state_dict())
        payload=dict(model_config=CONFIG,model_state_dict=model.state_dict(),loss_weights=config['losses'],
                     round=round_id,optimizer_steps=step,training_seed=seed,smoke=args.smoke)
        torch.save(payload,args.output/f'round_{round_id:02d}.pt')
        write(args.output/'progress.json',dict(complete=round_id==config['rounds'],round=round_id,optimizer_steps=step,
              rows_without_safe_action=masked_rows,elapsed_seconds=time.time()-started))
    assert step==config['expected_updates']
    torch.save(payload,args.output/'checkpoint.pt')


if __name__=='__main__':main()
