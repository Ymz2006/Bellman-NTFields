"""Train the fixed Bellman + Eikonal + direction baseline from scratch."""
import argparse
import copy
import json
import random
import time
import numpy as np
import torch
from pathlib import Path
from common import ROOT, sha, write
from network import TravelCost, CONFIG
from speed import load_speed
from loss import training_loss, WEIGHTS
from optimizer import stable_clip_grad_norm_


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true', help='Two rounds, two minibatches per epoch; not a benchmark result')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    bank = ROOT/'assets/bank'
    manifest = json.loads((bank/'manifest.json').read_text())
    for name, expected in manifest['file_sha256'].items():
        assert sha(bank/name)==expected, name
    assert sha(ROOT/'assets/speed.pt')==manifest['speed_checkpoint_sha256']
    arrays = [np.load(bank/name, mmap_mode='r') for name in
              ('source_q.npy', 'successor_q.npy', 'local_cost.npy', 'safe.npy')]
    source, successor, cost, safe = arrays
    random.seed(87358); np.random.seed(87358); torch.manual_seed(87358); torch.cuda.manual_seed_all(87358)
    generator = np.random.default_rng(87358)
    # Construction order preserves the baseline initialization RNG sequence.
    speed = load_speed(ROOT/'assets/speed.pt', 'cuda:0')
    model = TravelCost().cuda().train()
    target = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-6)
    config = dict(model=CONFIG, losses=WEIGHTS, seed=87358, rounds=2 if args.smoke else 5,
                  epochs_per_round=1 if args.smoke else 2, batch_size=256,
                  learning_rate=2e-4, weight_decay=1e-6, gradient_clip=2.0,
                  bank_manifest_sha256=sha(bank/'manifest.json'), speed_sha256=sha(ROOT/'assets/speed.pt'), smoke=args.smoke)
    write(args.output/'config.json', config)
    steps_per_epoch = 2 if args.smoke else (len(source)+255)//256
    started = time.time(); step = 0
    for round_id in range(1, config['rounds']+1):
        for epoch in range(config['epochs_per_round']):
            order = generator.permutation(len(source))
            for batch in range(steps_per_epoch):
                indices = order[batch*256:min((batch+1)*256, len(source))]
                goals = generator.choice(len(source), size=len(indices), replace=False)
                q,z,c,s = [torch.as_tensor(np.asarray(a[indices]).copy(),device='cuda:0') for a in arrays]
                g = torch.as_tensor(np.asarray(source[goals]).copy(),device='cuda:0')
                optimizer.zero_grad(set_to_none=True)
                total, terms = training_loss(model,target,speed,q,g,z,c,s.bool())
                if not bool(torch.isfinite(total)):raise FloatingPointError('Non-finite training loss')
                total.backward()
                norm=stable_clip_grad_norm_(model.parameters(),2.0)
                if not bool(torch.isfinite(norm)):raise FloatingPointError('Non-finite gradient')
                optimizer.step();step+=1
                if args.smoke or step%100==0:
                    print(json.dumps(dict(step=step,round=round_id,loss=float(total.detach()),
                          **{k:float(v.detach()) for k,v in terms.items()},elapsed_seconds=time.time()-started)),flush=True)
        target.load_state_dict(model.state_dict())
        payload=dict(model_config=CONFIG,model_state_dict=model.state_dict(),loss_weights=WEIGHTS,
                     round=round_id,optimizer_steps=step,training_seed=87358,smoke=args.smoke)
        torch.save(payload,args.output/f'round_{round_id:02d}.pt')
        torch.save(dict(**payload,optimizer_state_dict=optimizer.state_dict(),numpy_generator_state=generator.bit_generator.state),args.output/'recovery.pt')
        write(args.output/'progress.json',dict(complete=round_id==config['rounds'],round=round_id,optimizer_steps=step,elapsed_seconds=time.time()-started))
    assert step==(4 if args.smoke else 15620)
    torch.save(payload,args.output/'checkpoint.pt')


if __name__=='__main__':main()
