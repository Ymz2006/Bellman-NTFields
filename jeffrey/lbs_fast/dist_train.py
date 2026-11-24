"""Train the learned clearance field of one prepared scene (Task 2; distnet.py) and store it with its error statistics.

    python dist_train.py --prepared <prep run> --field <LBS field run> --lbs-src <jeffrey/lbs snapshot> --cfg '{"kind": "grid"}' --out <dir>

The field run is only used for its lift profile G (the h error in cells is reported next to the clearance error).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--field', type=Path, required=True)
    p.add_argument('--lbs-src', required=True)
    p.add_argument('--cfg', default='{}')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, a.lbs_src)
    sys.path.insert(0, str(HERE))
    import torch
    import seeds as SD
    import scenes3 as S3
    import lift as LF
    import fast as FA
    import distnet as DN
    from common import Logger
    log = Logger(a.out / 'log.txt')
    cfg = json.loads(a.cfg)
    info = json.load(open(a.prepared / 'maze.json'))
    instance = SD.apply(info)
    scene, info = S3.load_scene(a.prepared, a.device)
    net = LF.load_field(a.field / 'training/checkpoint.pt', scene, a.device)
    field = FA.Field(net, scene)
    t0 = time.time()
    model, hist = DN.train(scene, cfg, a.device, log)
    ev = DN.evaluate(model, scene, field.G)
    DN.save(a.out / 'dist.pt', model, cfg)
    res = dict(stage='lbs-fast-dist', prepared=str(a.prepared), field=str(a.field), scene_name=S3.scene_name(info), instance=instance, cfg=cfg,
               parameters=int(sum(q.numel() for q in model.parameters())), train_seconds=time.time() - t0, error=ev, history=hist, clip=[DN.LO, DN.HI])
    json.dump(res, open(a.out / 'results.json', 'w'), indent=1)
    log(f'done: {json.dumps(ev)}')
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
