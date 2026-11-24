"""Planners and planner settings on trained fields of one prepared placement: the frozen queries planned with planning seeds 0/1/2 and
scored by core.score at 0.001 rad. No training. One result per (field, planner, steps, samples, horizon); per-query flags are saved.

    --planner forward   ntrl-demo's MPPI start -> goal (core.plan; the field run's own count is this at 600 steps, 50 samples, horizon 5)
    --planner reverse   the same planner goal -> start, path reversed (bidir.plan_reverse)
    --planner bidir     two heads walking towards each other (bidir.plan_bidir); --steps is per head, so 300 here = 600 moves

    CUDA_VISIBLE_DEVICES=0 python evalfield.py --prepared <prepare run> --fields lift=<field run> ntrl=<field run> --steps 600 2400 --samples 50 400 --out <dir>
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
LBS = HERE if (HERE / 'pipeline6.py').is_file() else HERE.parent / 'lbs_arm'
sys.path.insert(0, str(LBS))
import arm12  # noqa: E402
sys.modules['arm'] = arm12
import arm as A  # noqa: E402
import core as C  # noqa: E402
import bidir as BD  # noqa: E402
import trainer6 as TRN  # noqa: E402
from pipeline6 import summarize  # noqa: E402
from common import Logger, dump  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--fields', nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, nargs='+', default=[600])
    p.add_argument('--samples', type=int, nargs='+', default=[50])
    p.add_argument('--horizon', type=int, nargs='+', default=[5])
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--planner', nargs='+', default=['forward'], choices=['forward', 'reverse', 'bidir'])
    p.add_argument('--checkpoint', default='training/checkpoint.pt')
    p.add_argument('--save-paths', action='store_true')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    log = Logger(a.out / 'log.txt')
    dev, t0 = 'cuda:0', time.time()
    scene, info = A.load_scene(a.prepared, dev)
    Q = dict(np.load(a.prepared / 'queries.npz'))
    qs, qg = torch.tensor(Q['start'], device=dev), torch.tensor(Q['goal'], device=dev)
    if a.smoke:
        keep = np.arange(0, len(qs), 10)
        Q = {k: (v[keep] if getattr(v, 'shape', (0,))[:1] == (len(qs),) else v) for k, v in Q.items()}
        qs, qg = qs[keep], qg[keep]
    mn = TRN.ntrl_modules()[2]
    clear = C.Clearance(scene, compile=True)
    res = dict(stage='armplace-evalfield', scene=scene.describe(), prepared=str(a.prepared), smoke=a.smoke, runs=[])
    flags = {}
    for item in a.fields:
        name, run = item.split('=', 1)
        net = C.load_field(mn, Path(run) / a.checkpoint, scene, dev)
        ev = C.Compiled(net, scene, clear, compile=True)
        for kind in a.planner:
            for steps in a.steps:
                for samples in a.samples:
                    for horizon in a.horizon:
                        planner = C.planner_settings(scene, sample_num=samples, horizon=horizon)
                        row = dict(field=name, run=str(run), planner=dict(planner, steps=steps, kind=kind), plan={})
                        total = np.zeros(len(qs), int)
                        tp = time.time()
                        for seed in a.seeds:
                            if kind == 'forward':
                                paths, conv = C.plan(ev, qs, qg, seed, steps, **planner)
                            elif kind == 'reverse':
                                paths, conv = BD.plan_reverse(ev, qs, qg, seed, steps, **planner)
                            else:
                                paths, conv, _ = BD.plan_bidir(ev, qs, qg, seed, steps, **planner)
                            rows = C.score(scene, paths, conv.cpu().numpy(), step_rad=1e-3)
                            out, ok = summarize(rows, Q)
                            total += ok
                            row['plan'][str(seed)] = out
                            flags[f'{name}|{kind}|{steps}|{samples}|{horizon}|{seed}'] = ok
                            if a.save_paths:
                                arr = np.empty(len(paths), dtype=object)
                                arr[:] = [(pth * scene.q_range).cpu().numpy().astype(np.float32) for pth in paths]
                                np.save(a.out / f'paths_{name}_{kind}_{steps}_{samples}_{horizon}_seed{seed}.npy', arr, allow_pickle=True)
                        row['success'] = [int(total.sum()), int(len(qs) * len(a.seeds))]
                        row['success_by_task'] = {n: int(sum(row['plan'][s]['by_task'][n] for s in row['plan'])) for n in [str(t) for t in Q['task_names']]}
                        row['not_arrived'] = int(sum(p['not_arrived'] for p in row['plan'].values()))
                        row['arrived_contact'] = int(sum(p['arrived_contact'] for p in row['plan'].values()))
                        row['seconds'] = time.time() - tp
                        res['runs'].append(row)
                        log(f'{name} {kind} steps {steps} samples {samples} horizon {horizon}: {row["success"][0]}/{row["success"][1]} by task {row["success_by_task"]}; did not arrive {row["not_arrived"]}, arrived in contact {row["arrived_contact"]} ({row["seconds"]:.0f}s)')
    np.savez(a.out / 'flags.npz', **flags)
    res['elapsed_seconds'] = time.time() - t0
    dump(a.out / 'results.json', res)
    log(f'done in {res["elapsed_seconds"] / 60:.1f} min')


if __name__ == '__main__':
    main()
