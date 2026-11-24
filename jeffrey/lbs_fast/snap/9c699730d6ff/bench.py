"""Timing benchmark of one prepared scene of the LBS final benchmark: single-query MPPI on the scene's LBS field, one query at a time,
for a list of planner variants on one device. Only the generation of the path is timed; the exact scorer (maze_scale/plan.score) runs
afterwards, off the clock, as a control of what the faster planners return.

    python bench.py --prepared <prep run> --field <LBS field run> --lbs-src <jeffrey/lbs snapshot> --device cpu --threads 2 \
        --variant 'ref|v=ref' --variant 'fast|c=default' --limit 100 --out <dir>

A variant is 'label|key=value,...':
    v      ref    bench.py of jeffrey/lbs_final: plan.mppi with plan.field_cost(net) (the baseline)
           goal   fast.Planner, one agent start -> goal
           meet   two agents chasing each other (bidirectional, moving targets)
           both   two independent agents, start -> goal and goal -> start (bidirectional, fixed targets, first arrival or meeting)
    c      none | default | ro (torch.compile mode 'reduce-overhead' = CUDA graphs) | native (native.cpp: the whole loop in C++ on the
           CPU, t = its thread count; v = goal or meet; exact clearance only; its own random numbers)
    q      number of queries for this variant (the first q of the run's queries)
    rev    1 = plan every query from its goal to its start (control for the bidirectional planners; the field is symmetric)
    k      MPPI steps per compiled call (the host reads the positions back once per block)
    n      ref (the reference's random numbers, call for call) | bulk (one randn per block)
    s      rollout samples (default: the scene's 50);   m   step multiplier (step x m, temperature / m; tolerance unchanged)
    dist   exact (the scene's clearance) | <name> of a --dist name=<dist run> learned clearance (Task 2)
    bf     step budget relative to the reference's max(200, 3 x true path length / step); default 1 for one agent, 0.5 for two agents
           (the same total number of agent steps); divided by m
    mt     meeting tolerance of 'both' in units of the goal tolerance (default 1)
Time = wall time of plan() for one query: from the start / goal tensors on the device to the finished path on the CPU.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def parse(spec):
    label, _, rest = spec.partition('|')
    kw = dict(x.split('=', 1) for x in rest.split(',') if x)
    return label, kw


def summarize(rows):
    ok = np.array([r['success'] for r in rows])
    t = np.array([r['wall'] for r in rows])
    st = np.array([r['steps'] for r in rows])
    out = dict(n=len(rows), success=int(ok.sum()), rate=float(ok.mean()), converged=int(sum(r['converged'] for r in rows)),
               wall_all=dict(mean=float(t.mean()), median=float(np.median(t)), std=float(t.std())), steps_mean=float(st.mean()),
               ms_per_step=float(1e3 * t.sum() / st.sum()))
    if ok.any():
        out['wall_solved'] = dict(mean=float(t[ok].mean()), median=float(np.median(t[ok])), std=float(t[ok].std()))
        out['steps_solved'] = float(st[ok].mean())
        out['length_solved'] = float(np.mean([r['length_cells'] for r, o in zip(rows, ok) if o]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--field', type=Path, required=True)
    p.add_argument('--lbs-src', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--variant', action='append', default=[])
    p.add_argument('--dist', action='append', default=[], help='name=<dist run directory>')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, a.lbs_src)
    sys.path.insert(0, str(HERE))
    import torch
    torch.set_num_threads(a.threads)
    import seeds as SD
    import scenes3 as S3
    import lift as LF
    import plan as P
    import fast as FA
    import distnet as DN
    import native as NA
    from common import Logger, save_paths
    log = Logger(a.out / 'log.txt')
    dev = a.device
    cuda = dev.startswith('cuda')
    info = json.load(open(a.prepared / 'maze.json'))
    instance = SD.apply(info)
    scene, info = S3.load_scene(a.prepared, dev)
    Q = dict(np.load(a.prepared / 'queries.npz'))
    n = len(Q['start']) if a.limit <= 0 else min(a.limit, len(Q['start']))
    S = torch.tensor(Q['start'][:n], device=dev)
    G = torch.tensor(Q['goal'][:n], device=dev)
    st = P.settings(scene)
    budget = np.maximum(st['min_steps'], np.ceil(st['budget_factor'] * Q['path_clen'][:n] / st['step'])).astype(int)
    net = LF.load_field(a.field / 'training/checkpoint.pt', scene, dev)
    fres = json.load(open(a.field / 'results.json'))
    dists = {}
    for d in a.dist:
        name, run = d.split('=', 1)
        model, cfg = DN.load(Path(run) / 'dist.pt', scene, dev)
        dists[name] = dict(model=model, cfg=cfg, run=run)
    fields = {'exact': FA.Field(net, scene)}
    for name, d in dists.items():
        fields[name] = FA.Field(net, scene, dist=d['model'])
    with torch.no_grad():
        pairs = torch.cat([S, G], 1)
        ref_T = net.out(pairs)[0][:, 0]
        check = {name: float((f.travel(pairs) - ref_T).abs().max()) for name, f in fields.items()}
    log(f'{S3.scene_name(info)} ({instance}), {n} queries on {dev} ({a.threads} threads); max |T - NN.out| over the queries: {check}')
    res = dict(stage='lbs-fast-bench', prepared=str(a.prepared), field=str(a.field), arm=fres.get('arm'), scene_name=S3.scene_name(info), instance=instance, n_queries=n,
               device=dev, gpu=torch.cuda.get_device_name(torch.device(dev)) if cuda else None, threads=a.threads, planner=st, torch=torch.__version__, seed=a.seed,
               field_check=check, dists={k: dict(cfg=v['cfg'], run=v['run']) for k, v in dists.items()},
               truth=dict(path_cells=Q['path_cells'][:n].tolist(), d_cells=Q['d_cells'][:n].tolist()), budget=budget.tolist(), variants={}, summary={})

    native = [None]

    def sync():
        if cuda:
            torch.cuda.synchronize()

    for spec in a.variant:
        label, kw = parse(spec)
        v = kw.get('v', 'goal')
        t_v = time.time()
        if v == 'ref':
            plan, setup, bf, m = (lambda s, g, b: FA.reference(net, scene, st, s, g, b, a.seed)), 0.0, 1.0, 1.0
            plan(S[0], G[0], 8)
        elif kw.get('c') == 'native':
            assert not cuda and kw.get('dist', 'exact') == 'exact'
            m = float(kw.get('m', 1.0))
            t0 = time.perf_counter()
            if native[0] is None:
                native[0] = NA.NativeField(fields['exact'], a.threads)
                res['native_check'] = NA.check(fields['exact'], native[0])
                log(f'native kernel against the torch field: {res["native_check"]}')
            pl = NA.NativePlanner(fields['exact'], st, variant=v, threads=int(kw.get('t', a.threads)), samples=int(kw['s']) if 's' in kw else None, step_mult=m, native=native[0])
            pl.warm(S[0], G[0])
            setup = time.perf_counter() - t0
            bf = float(kw.get('bf', 1.0 if v == 'goal' else 0.5))
            plan = lambda s, g, b: pl.plan(s, g, b, a.seed)
        else:
            comp = {'none': None, 'default': 'default', 'ro': 'reduce-overhead'}[kw.get('c', 'none')]
            m = float(kw.get('m', 1.0))
            pl = FA.Planner(fields[kw.get('dist', 'exact')], st, variant=v, samples=int(kw['s']) if 's' in kw else None, step_mult=m, compile=comp,
                            block=int(kw.get('k', 1)), noise=kw.get('n', 'ref'), meet_tol=float(kw['mt']) * st['tol'] if 'mt' in kw else None)
            pl.warm(S[0], G[0])
            setup = pl.setup_seconds
            bf = float(kw.get('bf', 1.0 if v == 'goal' else 0.5))
            plan = lambda s, g, b: pl.plan(s, g, b, a.seed)
        nq = min(n, int(kw['q'])) if 'q' in kw else n
        rev = kw.get('rev') == '1'
        rows, paths = [], []
        for i in range(nq):
            b = int(math.ceil(budget[i] * bf / m))
            qs, qg = (G[i], S[i]) if rev else (S[i], G[i])
            sync()
            t0 = time.perf_counter()
            r = plan(qs, qg, b)
            sync()
            wall = time.perf_counter() - t0
            paths.append(r['path'])
            rows.append(dict(wall=wall, steps=r['steps'], converged=r['converged'], how=r['how'], budget=b))
            if i % 100 == 0:
                log(f'{label} {i}/{n}: {wall * 1e3:.1f} ms, {r["steps"]} steps')
        for i in range(0, nq, 50):                                        # off the clock: the exact scorer
            sc = P.score(scene, [x.to(dev) for x in paths[i:i + 50]], [r['converged'] for r in rows[i:i + 50]])
            for r, s_ in zip(rows[i:i + 50], sc):
                r.update(success=bool(s_['success']), collision=bool(s_['collision']), length_cells=float(s_['length_cells']), min_clearance=float(s_['min_clearance']),
                         waypoints=int(s_['waypoints']))
        res['variants'][label] = dict(spec=kw, setup_seconds=setup, rows=rows, seconds=time.time() - t_v)
        res['summary'][label] = summarize(rows)
        save_paths(a.out / f'paths_{label}.npy', paths)
        log(f'{label}: {json.dumps(res["summary"][label])} (setup {setup:.0f}s, {time.time() - t_v:.0f}s)')
        json.dump(res, open(a.out / 'bench.json', 'w'))
    res['complete'] = True
    json.dump(res, open(a.out / 'bench.json', 'w'))
    log('done')
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
