"""Parking thin-wall test for ntrl-demo: one divider thickness, one or more training seeds.

    CUDA_VISIBLE_DEVICES=1 python jeffrey/parking/pipeline.py --wall 0.05 --seeds 87358,1,2 --out <dir>

1. data      ntrl-demo point-recipe pairs in the car's C-space (data.py), fixed data seed 0.
2. queries   spot_to_spot: 50 bay A -> bay B and 50 bay B -> bay A parked poses; aisle_to_spot (control): 50 aisle ->
             bay and 50 bay -> aisle. Every query is certified by a collision-free witness path (back out, move in
             the aisle, drive in); start and goal keep >= 0.01 clearance.
3. truth     T*(., goal) grids for the spot_to_spot goals (truth.py), T*(start, goal), its descent direction at the
             start, and the bay-A profile / theta=0 slice.
4. controls  straight line (both suites); oracle = the same ntrl-demo MPPI on the T* grid (spot_to_spot).
5. per seed  train (ntrl-demo models/<variant>, unchanged) -> ntrl-demo MPPI at its defaults and the --settings variants
             (wide: 200 samples, horizon 10, 400 steps; fine: step 0.005, 600 steps, for tight bays), planning seeds
             0/1/2 -> exact scorer -> field diagnostics
             (T vs T*, start descent direction vs truth, profile across the divider, speed fit on held-out pairs).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scene as SC  # noqa: E402
from scene import Scene  # noqa: E402
import data as D  # noqa: E402
import plan as P  # noqa: E402
import truth as TR  # noqa: E402
import train as TN  # noqa: E402

PLAN_SEEDS = [0, 1, 2]


def make_queries(scene, seed=12345, n=50):
    dev = scene.device
    gen = torch.Generator(dev).manual_seed(seed)
    w = scene.w
    need = min(0.01, 0.3 * scene.side_gap)          # end clearance: 0.01, or 0.3 x the side gap in tight bays

    def certified(draw_start, draw_goal, count):
        starts, goals, drawn = [], [], 0
        while sum(len(s) for s in starts) < count:
            s, g = draw_start(4 * count), draw_goal(4 * count)
            drawn += len(s)
            ok = (scene.clearance(s)[0] >= need) & (scene.clearance(g)[0] >= need)
            path = SC.witness(s, g)
            for i in torch.nonzero(ok).squeeze(1).tolist():
                if not P.score(scene, [path[i]], [True])[0]['collision']:
                    starts.append(s[i:i + 1])
                    goals.append(g[i:i + 1])
                    if len(starts) == count:
                        break
        return torch.cat(starts), torch.cat(goals), drawn

    A = lambda k: SC.parked(w, 'A', k, gen, dev, scene.car_w, scene.bay_w)
    B = lambda k: SC.parked(w, 'B', k, gen, dev, scene.car_w, scene.bay_w)
    aisle = lambda k: SC.aisle(w, k, gen, dev)

    def bay(k):
        pick = torch.rand(k, generator=gen, device=dev) < 0.5
        return torch.where(pick[:, None], A(k), B(k))

    s1, g1, _ = certified(A, B, n)
    s2, g2, _ = certified(B, A, n)
    s3, g3, _ = certified(aisle, bay, n)
    s4, g4, _ = certified(bay, aisle, n)
    return dict(spot_to_spot=(torch.cat([s1, s2]), torch.cat([g1, g2])), aisle_to_spot=(torch.cat([s3, s4]), torch.cat([g3, g4])))


def summarize(rows):
    n = len(rows)
    out = dict(n=n, success=sum(r['success'] for r in rows), converged=sum(r['converged'] for r in rows),
               collision=sum(r['collision'] for r in rows), divider_contact=sum(r.get('divider_contact', False) for r in rows))
    out['first_contact'] = {}
    for r in rows:
        if r['collision']:
            k = SC.NAMES[r['first_contact_box']]
            out['first_contact'][k] = out['first_contact'].get(k, 0) + 1
    ok = [r for r in rows if r['success']]
    if ok and 'time' in ok[0]:
        out['median_success_time'] = float(np.median([r['time'] for r in ok]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wall', type=float, required=True)
    p.add_argument('--seeds', default='87358,1,2')
    p.add_argument('--variant', choices=['metric', 'metric_arm'], default='metric')
    p.add_argument('--epochs', type=int, default=5000)
    p.add_argument('--n-data', type=int, default=500_000)
    p.add_argument('--car-width', type=float, default=SC.CAR_W, help='car width W (C-space gap = wall + W); default 0.13')
    p.add_argument('--bay-width', type=float, default=None, help='fixed bay width (default car width + 0.10, i.e. 0.05 side gap)')
    p.add_argument('--settings', default='default,wide', help='MPPI settings (plan.SETTINGS) for the trained field; the oracle runs each too')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / 'log.txt', 'a')

    def log(msg):
        line = f'[{time.strftime("%H:%M:%S")}] {msg}'
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()

    seeds = [int(s) for s in a.seeds.split(',')]
    epochs = 3 if a.smoke else a.epochs
    n_data = 20_000 if a.smoke else a.n_data
    dev = 'cuda:0'
    car_w = a.car_width
    scene = Scene(a.wall, dev, car_w=car_w, bay_w=a.bay_width)
    speed_fn = scene.train_speed
    results = dict(wall=a.wall, variant=a.variant, epochs=epochs, seeds=seeds, smoke=a.smoke, car=[SC.CAR_L, car_w], car_width=car_w, bay_width=scene.bay_w,
                   side_gap=scene.side_gap, bay_width_fixed=a.bay_width is not None,
                   margin=SC.MARGIN, offset=SC.OFFSET, cspace_gap=a.wall + car_w, gpu=torch.cuda.get_device_name(0), mppi={k: P.SETTINGS[k] for k in a.settings.split(',')})
    t0 = time.time()

    # 1. data
    x, speed, normal, d, drawn = D.sample(scene, n_data, 0)
    (out / 'data').mkdir(exist_ok=True)
    np.save(out / 'data/sampled_points.npy', x)
    np.save(out / 'data/speed.npy', speed)
    np.save(out / 'data/normal.npy', normal)
    near_div = None
    results['data'] = dict(n=len(x), drawn=drawn, x1_blocked=float((d[:, 1] <= SC.OFFSET).mean()), x1_free=float((d[:, 1] >= SC.MARGIN).mean()))
    with torch.no_grad():
        _, idx0 = scene.clearance(torch.tensor(x[:, :3], device=dev))
    results['data']['x0_closest_box'] = {SC.NAMES[j]: int((idx0 == j).sum()) for j in range(len(SC.NAMES))}
    log(f'data {len(x)} pairs ({drawn} drawn), x0 closest box {results["data"]["x0_closest_box"]}')
    hx, hspeed, _, hd, _ = D.sample(scene, 5_000 if a.smoke else 50_000, 1)   # held-out pairs for the speed fit
    torch.cuda.empty_cache()

    # 2. queries
    Q = make_queries(scene)
    np.savez(out / 'queries.npz', **{f'{k}_{j}': v[i].cpu().numpy() for k, v in Q.items() for i, j in enumerate(['start', 'goal'])})
    log('queries certified: ' + ', '.join(f'{k} {len(v[0])}' for k, v in Q.items()))

    # 3. truth
    grid = TR.Grid(scene)
    s2s, g2s = Q['spot_to_spot']
    fields, sweeps = [], []
    for i in range(0, len(g2s), 25):
        f, it = grid.solve(g2s[i:i + 25])
        fields.append(f)
        sweeps.append(it)
    fields = torch.cat(fields)
    t_star = grid.interp(fields, s2s[:, None])[:, 0]
    true_dir = torch.nn.functional.normalize(grid.descent(fields, s2s), dim=1)
    qa = torch.tensor([[SC.PARK_X, SC.bay_y(a.wall, 'A', car_w, scene.bay_w), 0.0]], device=dev)
    fa, _ = grid.solve(qa)
    line_y = torch.linspace(-0.45, 0.45, 361, device=dev)
    line = torch.stack([torch.full_like(line_y, SC.PARK_X), line_y, torch.zeros_like(line_y)], 1)
    prof_star = grid.interp(fa, line[None])[0]
    gx = torch.linspace(-0.5, 0.5, 201, device=dev)
    plane = torch.stack(torch.meshgrid(gx, gx, indexing='ij') + (torch.zeros(201, 201, device=dev),), -1).reshape(-1, 3)
    slice_star = grid.interp(fa, plane[None])[0].reshape(201, 201)
    y_top, y_bot = a.wall / 2 + car_w / 2 + 0.005, -(a.wall / 2 + car_w / 2 + 0.005)
    edge = torch.tensor([[SC.PARK_X, y_top, 0.0], [SC.PARK_X, y_bot, 0.0]], device=dev)
    edge_star = grid.interp(fa, edge[None])[0]
    euclid = torch.linalg.vector_norm(s2s - g2s, dim=1)
    results['truth'] = dict(sweeps=sweeps, t_star_median=float(t_star.median()), euclid_median=float(euclid.median()),
                            detour_ratio_median=float((t_star / euclid).median()), edge_star=edge_star.tolist(),
                            jump_star=float(edge_star[1] - edge_star[0]))
    np.savez(out / 'truth.npz', t_star=t_star.cpu().numpy(), true_dir=true_dir.cpu().numpy(), profile_y=line_y.cpu().numpy(),
             profile_star=prof_star.cpu().numpy(), slice_star=slice_star.cpu().numpy(), euclid=euclid.cpu().numpy())
    log(f'truth: sweeps {sweeps}, T* median {float(t_star.median()):.3f}, Euclid median {float(euclid.median()):.3f}, '
        f'jump across divider {results["truth"]["jump_star"]:.3f}')

    # 4. controls
    results['straight'] = {k: summarize(P.score(scene, P.straight(*v), [True] * len(v[0]), speed_fn)) for k, v in Q.items()}

    def oracle(pairs, ids):
        q = pairs.reshape(len(ids), -1, 6)[..., :3]
        return grid.interp(fields[ids], q).reshape(-1)

    results['oracle'], results['oracle_settings'] = {}, {}
    for setting in dict.fromkeys(['default'] + a.settings.split(',')):
        orc = {}
        for ps in PLAN_SEEDS:
            trace, stop, conv = P.mppi(oracle, s2s, g2s, ps, **P.SETTINGS[setting])
            orc[str(ps)] = summarize(P.score(scene, P.paths_from_trace(trace, stop, s2s, g2s), conv, speed_fn))
        results['oracle_settings'][setting] = orc
    results['oracle'] = results['oracle_settings']['default']
    log('controls: straight ' + json.dumps({k: v['success'] for k, v in results['straight'].items()}) +
        ', oracle MPPI spot_to_spot ' + json.dumps({st: [v['success'] for v in o.values()] for st, o in results['oracle_settings'].items()}))
    del fields, grid
    torch.cuda.empty_cache()
    json.dump(results, open(out / 'results.json', 'w'), indent=2)

    # 5. per training seed
    results['seeds_results'] = {}
    for seed in seeds:
        sdir = out / f'seed{seed}'
        ts = time.time()
        ck = TN.train(out / 'data', sdir / 'training', a.variant, epochs, seed, source=qa[0].tolist())
        train_s = time.time() - ts
        net = P.load_field(ck, a.variant, dev)
        cost = P.field_cost(net)
        res = dict(train_seconds=train_s, final_loss=json.load(open(sdir / 'training/progress.json'))['final_loss'], planning={})
        saved = {}
        for setting, kw in ((k, P.SETTINGS[k]) for k in a.settings.split(',')):
            for suite, (s, g) in Q.items():
                for ps in PLAN_SEEDS:
                    trace, stop, conv = P.mppi(cost, s, g, ps, **kw)
                    paths = P.paths_from_trace(trace, stop, s, g)
                    rows = P.score(scene, paths, conv, speed_fn)
                    res['planning'][f'{setting}/{suite}/{ps}'] = summarize(rows)
                    res['planning'][f'{setting}/{suite}/{ps}']['success_ids'] = [i for i, r in enumerate(rows) if r['success']]
                    if setting == 'default' or ps == 0:
                        arr = np.empty(len(paths), dtype=object)
                        arr[:] = [x.cpu().numpy() for x in paths]
                        saved[f'{setting}_{suite}_{ps}'] = arr
        np.savez(sdir / 'paths.npz', **saved)
        # field diagnostics
        pairs = torch.cat([s2s, g2s], 1)
        t_pred, grad = P.field_grad(net, pairs)
        pred_dir = torch.nn.functional.normalize(-grad, dim=1)
        cos = (pred_dir * true_dir).sum(1)
        toward = torch.sign(g2s[:, 1] - s2s[:, 1])
        through = (pred_dir[:, 1] * toward > pred_dir[:, 0].abs()) & (pred_dir[:, 0] > -0.5)
        with torch.no_grad():
            prof = P.field_cost(net)(torch.cat([qa.expand(len(line), 3), line], 1))
            sl = P.field_cost(net)(torch.cat([plane, qa.expand(len(plane), 3)], 1)).reshape(201, 201)
            e = P.field_cost(net)(torch.cat([qa.expand(2, 3), edge], 1))
        hp = torch.tensor(hx, device=dev)
        _, hg = P.field_grad(net, hp)
        s_pred = 1 / torch.linalg.vector_norm(hg, dim=1).clamp(min=1e-6)
        s_true = scene.train_speed(torch.tensor(hd[:, 0], device=dev))
        _, hidx = scene.clearance(hp[:, :3])
        div = hidx == 4
        corr = lambda u, v: float(torch.corrcoef(torch.stack([u, v]))[0, 1])
        res['field'] = dict(
            t_over_tstar_median=float((t_pred / t_star).median()), t_over_euclid_median=float((t_pred / euclid).median()),
            t_pred_median=float(t_pred.median()), descent_cos_median=float(cos.median()), descent_cos_mean=float(cos.mean()),
            start_points_through_wall=int(through.sum()), edge_pred=e.tolist(), jump_pred=float(e[1] - e[0]),
            jump_ratio=float((e[1] - e[0]) / (edge_star[1] - edge_star[0])),
            speed_corr=corr(s_pred, s_true), speed_mae=float((s_pred - s_true).abs().mean()),
            speed_corr_divider=corr(s_pred[div], s_true[div]) if int(div.sum()) > 2 else None,
            speed_mae_divider=float((s_pred[div] - s_true[div]).abs().mean()) if int(div.sum()) else None,
            speed_pred_at_contact_divider=float(s_pred[div & (s_true < 0.05)].median()) if int((div & (s_true < 0.05)).sum()) else None)
        np.savez(sdir / 'field.npz', t_pred=t_pred.cpu().numpy(), pred_dir=pred_dir.cpu().numpy(), cos=cos.cpu().numpy(),
                 profile_pred=prof.cpu().numpy(), slice_pred=sl.cpu().numpy())
        results['seeds_results'][str(seed)] = res
        s = res['planning']
        log(f'seed {seed}: trained {train_s / 60:.1f} min, loss {res["final_loss"]:.3e}; MPPI default spot_to_spot ' +
            str([s[f'default/spot_to_spot/{k}']['success'] for k in PLAN_SEEDS]) + ' aisle ' + str([s[f'default/aisle_to_spot/{k}']['success'] for k in PLAN_SEEDS]) +
            ''.join(f'; {st} spot_to_spot ' + str([s[f'{st}/spot_to_spot/{k}']['success'] for k in PLAN_SEEDS]) for st in a.settings.split(',') if st != 'default') +
            f'; T/T* {res["field"]["t_over_tstar_median"]:.2f}, jump ratio {res["field"]["jump_ratio"]:.2f}, descent cos {res["field"]["descent_cos_median"]:.2f}, '
            f'through-wall starts {res["field"]["start_points_through_wall"]}')
        del net
        torch.cuda.empty_cache()
        json.dump(results, open(out / 'results.json', 'w'), indent=2)
    results['elapsed_seconds'] = time.time() - t0
    results['complete'] = True
    json.dump(results, open(out / 'results.json', 'w'), indent=2)
    log(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
