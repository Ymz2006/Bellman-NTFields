"""Build the 3-D viewer of the study: report/viewer/index.html plus the data files it fetches.

    CUDA_VISIBLE_DEVICES=0 ~/.venvs/bellman-ntfield/bin/python jeffrey/arm_placement/viewer.py

Reads (no training, no planning): scenes/*.yml, models/<placement>/ (scene, test queries), report/summary.json, report/depth.json,
report/demo*.json + .npz (the picked configurations and their saved plans), and the registered `viewer_<placement>` evaluations of
queue/state.json (first lift model and the ntrl-demo field, two-ended planner, 2,400 steps per end, planning seed 0, paths saved).

Writes report/viewer/
    index.html        the page (viewer_template.html with the small data inlined)
    data.json         arm model, every placement's geometry and numbers, the picked configurations
    demo_<k>.txt      plans of the k-th set of picked configurations
    pl/<name>.json    test queries of one placement and where their plans are in the .txt
    pl/<name>.txt     start, goal and plans of that placement

A .txt is base64 of little-endian int16 rows of 15 (artifacts do not serve binary files): twelve joint angles in units of 2e-4 rad ([left arm 6, right arm 6]) and three clearances in units of
0.1 mm (the moving spheres to the nearest cuboid, i.e. without the two fixed shoulder spheres that always sit about 10 mm above their
pedestal; the other arm; the arm's own links). A plan is resampled to equal steps of joint-space arc length for display;
its clearances are the minimum over the scorer's own samples (0.001 rad per joint) that fall into each display step, and its verdict is
recomputed from those samples (asserted to agree with the registered runs' flags).
"""
import argparse
import base64
import json
import sys
from pathlib import Path
import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / 'lbs_arm'))
import arm12  # noqa: E402
sys.modules['arm'] = arm12
import core as C  # noqa: E402

DEV = 'cuda:0'
ANG, MM = 2e-4, 1e-4          # int16 units: radians, metres
KIND = ['cupboard', 'other arm', 'own links']


def rows_of(scene, x):
    """(N, 15) int16 rows of normalized configurations x (N, 12)."""
    pt = parts(scene, x)
    return np.concatenate([q16(x * scene.q_range), c16(torch.stack([pt['obs'], pt['inter'], pt['self']], 1))], 1)


def q16(rad):
    return torch.round(rad / ANG).clamp(-32767, 32767).short().cpu().numpy()


def c16(m):
    return torch.round(m / MM).clamp(-32767, 32767).short().cpu().numpy()


@torch.no_grad()
def parts(scene, x, chunk=1 << 14):
    out = {}
    for i in range(0, len(x), chunk):
        pt = scene.parts(x[i:i + chunk])
        for k in ('obs', 'obs_only', 'inter', 'self', 'valid'):
            out.setdefault(k, []).append(pt[k])
    return {k: torch.cat(v) for k, v in out.items()}


@torch.no_grad()
def pack(scene, path_rad, step, lo, hi):
    """A plan (K, 12) in radians -> its display rows and verdict."""
    x = torch.tensor(np.asarray(path_rad), device=DEV, dtype=torch.float32) / scene.q_range
    dense = C.densify(x, 1e-3 / scene.q_range)
    pt = parts(scene, dense)
    rad = dense * scene.q_range
    s = torch.cat([rad.new_zeros(1), torch.cumsum(torch.linalg.vector_norm(rad[1:] - rad[:-1], dim=1), 0)])
    total = float(s[-1])
    n = int(min(hi, max(lo, np.ceil(total / step) + 1)))
    t = torch.linspace(0, total, n, device=DEV)
    i = torch.searchsorted(s, t, right=True).clamp(1, len(s) - 1)
    w = ((t - s[i - 1]) / (s[i] - s[i - 1]).clamp(min=1e-12))[:, None]
    frames = rad[i - 1] + w * (rad[i] - rad[i - 1])
    b = torch.round(s / max(total, 1e-9) * (n - 1)).long().clamp(0, n - 1)
    cl = torch.full((n, 3), float('inf'), device=DEV)
    for j, k in enumerate(('obs', 'inter', 'self')):
        cl[:, j] = cl[:, j].scatter_reduce(0, b, pt[k], 'amin')
    for j in range(1, n):                                           # a display step without a scorer sample (cannot happen at these steps)
        cl[j] = torch.where(torch.isinf(cl[j]), cl[j - 1], cl[j])
    bad = torch.nonzero(pt['valid'] < 0).squeeze(1)
    info = dict(n=n, ok=not len(bad), waypoints=int(len(x)), length=round(total, 3),
                min_mm=[round(float(pt[k].min()) * 1000, 1) for k in ('obs', 'inter', 'self')])
    if len(bad):
        f = int(bad[0])
        m = [float(pt['obs_only'][f]) - 0.005, float(pt['inter'][f]) - 0.005, float(pt['self'][f]) - 0.002]
        info.update(fail_at=round(float(s[f]) / max(total, 1e-9), 4), fail_kind=KIND[int(np.argmin(m))])
        if len(path_rad) > 1:                                       # the goal is appended to a plan that did not arrive: a long last segment
            info['jump_rad'] = round(float(np.linalg.norm(np.asarray(path_rad[-1]) - np.asarray(path_rad[-2]))), 2)
    return np.concatenate([q16(frames), c16(cl)], 1), info


def geometry(doc):
    """Cuboids, zones and bases of a scene (the yml of place.py or the scene block of a results file), as plain lists."""
    cub = [dict(name=c['name'], c=[round(float(v), 4) for v in c['pose'][:3]], d=[round(float(v), 4) for v in c['dims']]) for c in doc['cuboids']]
    comp = [dict(name=c['name'], x=c['x'], y=c['y'], z=c['z']) for c in doc.get('compartments', [])]
    bases = [[round(float(v), 4) for v in r['base'][:3]] for r in doc['robots']]
    assert all(list(map(float, r['base'][3:])) == [1.0, 0.0, 0.0, 0.0] for r in doc['robots']), 'the viewer assumes unrotated bases'
    return dict(cuboids=cub, zones=comp, bases=bases)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--best', default='p_d100_s20_h80')
    p.add_argument('--demos', nargs='+', default=['report/demo', 'report/demo_p_d055_s46_h60', 'report/demo_p_d100_s30_h60'])
    p.add_argument('--demo-scenes', nargs='+', default=['p_d100_s20_h80', 'p_d055_s46_h60', 'p_d100_s30_h60'])
    p.add_argument('--step', type=float, default=0.08, help='display step of a test-query plan, radians of joint-space arc length')
    p.add_argument('--only', nargs='*', help='placements to rebuild (default: all)')
    a = p.parse_args()
    out = HERE / 'report/viewer'
    (out / 'pl').mkdir(parents=True, exist_ok=True)
    S = json.loads((HERE / 'report/summary.json').read_text())
    D = json.loads((HERE / 'report/depth.json').read_text())
    idx = json.loads((HERE / 'models/index.json').read_text())['placements']
    state = json.loads((HERE / 'queue/state.json').read_text())
    by = {pl['name']: pl for pl in S['placements']}
    C2C = 'compartment_to_compartment'

    # ---------------------------------------------------------------- the arm
    scene, _ = arm12.load_scene(HERE / 'models' / a.best, DEV)
    arm = dict(O=[[round(float(v), 9) for v in T.flatten()] for T in scene.O.double().cpu()], T0=[round(float(v), 9) for v in scene.T0.double().cpu().flatten()],
               frame=scene.frame.cpu().tolist(), local=[[round(float(v), 5) for v in r] for r in scene.local.cpu()], radius=[round(float(v), 5) for v in scene.radius.cpu()],
               ee_local=[round(float(v), 5) for v in scene.ee_local.cpu()], ang=ANG, mm=MM * 1000)
    g = torch.Generator().manual_seed(7)
    xc = (torch.rand(4, 12, generator=g) - 0.5).to(DEV) * 0.5
    sp, ee = scene.spheres(xc)
    arm['check'] = dict(q=[[round(float(v), 6) for v in r] for r in (xc * scene.q_range).cpu()], bases=scene.base_t.cpu().tolist(),
                        spheres=[[round(float(v), 6) for v in r] for r in sp.reshape(4, -1).cpu()], ee=[[round(float(v), 6) for v in r] for r in ee.reshape(4, -1).cpu()])

    # ---------------------------------------------------------------- placements
    P = []
    for f in sorted((HERE / 'scenes').glob('p_d*.yml')):
        name = f.stem
        key = 'base' if name == 'p_d025_s30_h60' and 'base' in idx and name not in idx else name
        doc = yaml.safe_load(f.read_text())
        dx, s, h = int(name[3:6]) / 100, int(name[8:10]) / 100, int(name[12:]) / 100
        row = dict(name=name, key=key, dx=dx, s=s, h=h, **geometry(doc), trained=key in idx)
        pl = by.get(key)
        if pl and pl.get('reach'):
            row['reach'] = [round(100 * pl['reach']['union_mean'], 1), round(100 * pl['reach']['both_mean'], 1)]
        if key in idx:
            runs = {}
            for fr in sorted(pl['fields'], key=lambda x: x['seed']):
                if fr['label'] not in ('lift', 'ntrl'):
                    continue
                runs.setdefault(fr['label'] + '_forward', []).append([fr['success'][0], fr['success'][1], fr['by_task'][C2C], fr['run'][-8:]])
                for q in fr.get('planners', []):
                    if q['kind'] == 'bidir' and q['steps'] == 2400 and q['samples'] == 50 and q.get('suite') == 'placement':
                        runs.setdefault(fr['label'] + '_two_ended', []).append([q['success'][0], q['success'][1], q['by_task'][C2C], q['run'][-8:]])
            row['runs'] = runs
            row['zone_runs'] = pl['fields'][0]['per_task_runs']
            if pl.get('euclid'):
                row['euclid'] = [pl['euclid']['success'][0], pl['euclid']['success'][1], pl['euclid']['run'][-8:]]
            if pl.get('prepare'):
                row['line'] = [pl['prepare']['straight_line'], pl['prepare']['queries']]
            if key in D:
                row['depth_cm'] = [round(100 * D[key]['zone_to_zone']['median'], 1), round(100 * D[key]['zone_to_zone']['max'], 1)]
            row['suite'] = pl.get('suite')
        else:
            st = state.get(f'tests_{name}', {})
            row['why'] = ('an arm reaches fewer than two zones: no zone to zone query exists' if (name.startswith('p_d105') or name == 'p_d100_s46_h80')
                          else 'building the test queries took over two hours and was stopped' if st.get('status') == 'failed' else 'not run')
        P.append(row)

    # ---------------------------------------------------------------- test queries and plans per placement
    for row in P:
        if not row['trained'] or (a.only and row['name'] not in a.only):
            if row['trained'] and (out / 'pl' / f'{row["name"]}.json').exists():
                row['queries'] = json.loads((out / 'pl' / f'{row["name"]}.json').read_text())['n']
            continue
        key = row['key']
        sc, _ = arm12.load_scene(HERE / 'models' / key, DEV)
        m = np.load(HERE / 'models' / key / 'robot_model.npz')
        row['bases'] = [[round(float(v), 4) for v in r[:3]] for r in m['bases']]
        row['cuboids'] = [dict(name=str(n), c=[round(float(v), 4) for v in c], d=[round(float(2 * v), 4) for v in hh]) for n, c, hh in zip(m['cuboid_names'], m['cuboid_centre'], m['cuboid_half'])]
        row['zones'] = [dict(name=str(n), x=[round(float(v), 4) for v in b[0]], y=[round(float(v), 4) for v in b[1]], z=[round(float(v), 4) for v in b[2]]) for n, b in zip(m['compartment_names'], m['compartments'])]
        Q = np.load(HERE / 'models' / key / 'queries.npz', allow_pickle=True)
        n = len(Q['start'])
        xs, xg = torch.tensor(Q['start'], device=DEV), torch.tensor(Q['goal'], device=DEV)
        rows = [rows_of(sc, xs), rows_of(sc, xg)]
        at = 2 * n
        meta = dict(name=row['name'], n=n, tasks=[str(t) for t in Q['task_names']], task=Q['task'].tolist(), line=Q['line_here' if 'line_here' in Q else 'line'].astype(int).tolist(),
                    dist=[round(float(v) * 1, 3) for v in torch.linalg.vector_norm((xg - xs) * sc.q_range, dim=1).cpu()], plans={})
        # how many of the registered headline runs (lift, two-ended, 2,400 steps, every training and planning seed) solved each query
        F = []
        for fr in by[key]['fields']:
            for q in fr.get('planners', []) if fr['label'] == 'lift' else []:
                if q['kind'] == 'bidir' and q['steps'] == 2400 and q['samples'] == 50 and q.get('suite') == 'placement':
                    z = np.load(ROOT / f'results/registered_{q["run"]}/flags.npz')
                    F += [z[f'lift|bidir|2400|50|5|{k}'] for k in (0, 1, 2)]
        meta['solved_runs'], meta['runs'] = np.stack(F).sum(0).astype(int).tolist(), len(F)
        job = state.get(f'viewer_{key}', {})
        if job.get('status') == 'done':
            run = ROOT / f'results/registered_{job["run"]}'
            meta['run'] = job['run']
            flags = np.load(run / 'flags.npz')
            for field in ('lift', 'ntrl'):
                f = run / f'paths_{field}_bidir_2400_50_5_seed0.npy'
                if not f.exists():
                    continue
                paths = np.load(f, allow_pickle=True)
                ok = flags[f'{field}|bidir|2400|50|5|0']
                plan, wrong = [], 0
                for i, pth in enumerate(paths):
                    r, info = pack(sc, pth, a.step if field == 'lift' else 1.5 * a.step, 12, 160 if field == 'lift' else 110)
                    wrong += int(info['ok'] != bool(ok[i]))
                    plan.append([at, info['n'], int(info['ok']), info.get('fail_at', -1), KIND.index(info['fail_kind']) if 'fail_kind' in info else -1, info['length'], info['waypoints']])
                    rows.append(r)
                    at += info['n']
                assert wrong == 0, (row['name'], field, wrong, 'verdicts differ from the registered run')
                meta['plans'][field] = plan
                meta[f'solved_{field}'] = int(ok.sum())
        arr = np.concatenate(rows).astype('<i2')
        (out / 'pl' / f'{row["name"]}.txt').write_bytes(base64.b64encode(arr.tobytes()))
        (out / 'pl' / f'{row["name"]}.json').write_text(json.dumps(meta, separators=(',', ':')))
        row['queries'] = n
        row['viewer_run'] = job.get('run', '')[-8:] if job.get('status') == 'done' else None
        row['viewer_solved'] = {k: meta.get(f'solved_{k}') for k in ('lift', 'ntrl') if f'solved_{k}' in meta}
        print(row['name'], n, 'queries;', {k: f'{meta.get("solved_" + k)}/{n}' for k in meta['plans']}, f'{arr.nbytes / 1e6:.2f} MB', flush=True)

    # ---------------------------------------------------------------- the picked configurations
    sets = []
    for stem, sname in zip(a.demos, a.demo_scenes):
        if not (HERE / f'{stem}.json').exists():
            continue
        k = len(sets)
        J = json.loads((HERE / f'{stem}.json').read_text())
        Z = np.load(HERE / f'{stem}.npz', allow_pickle=True)
        sc, _ = arm12.load_scene(HERE / 'models' / sname, DEV)
        rows, at, picks = [], 0, []
        for pk in J['picked']:
            ends = torch.tensor(np.array([pk['start_rad'], pk['goal_rad']]), device=DEV, dtype=torch.float32) / sc.q_range
            rows.append(rows_of(sc, ends))
            deep = float(sc.ee(ends)[..., 0].max()) - idx[sname]['dx']                 # deepest grasp point behind the cupboard front, either end
            item = dict(query=pk['query'], line=bool(pk['straight_line_feasible']), task=pk['task'], deep_cm=round(100 * deep, 1), start=at, goal=at + 1, start_zones=pk['start_zones'], goal_zones=pk['goal_zones'], dist=round(pk['joint_distance_rad'], 3),
                        start_rad=[round(v, 4) for v in pk['start_rad']], goal_rad=[round(v, 4) for v in pk['goal_rad']], plans={})
            at += 2
            for field in J['fields']:
                for seed in (0, 1, 2):
                    r, info = pack(sc, Z[f'q{pk["query"]}_{field}_seed{seed}'], 0.045, 24, 360)
                    assert info['ok'] == bool(pk['plans'][field]['solved_seeds'][seed]), (stem, pk['query'], field, seed, 'verdict differs from the registered demo run')
                    item['plans'].setdefault(field, []).append(dict(at=at, **info))
                    rows.append(r)
                    at += info['n']
            picks.append(item)
        arr = np.concatenate(rows).astype('<i2')
        (out / f'demo_{k}.txt').write_bytes(base64.b64encode(arr.tobytes()))
        sets.append(dict(placement=sname, file=f'demo_{k}.txt', fields=list(J['fields']), picks=picks,
                         runs={f: v.split('registered_')[1].split(':')[0][-8:] for f, v in J['fields'].items()},
                         candidates=J['hard']['n'], solved_all={f: v['all_seeds'] for f, v in J['hard']['solved'].items()}))
        print('set', k, sname, len(picks), 'configurations', f'{arr.nbytes / 1e6:.2f} MB', flush=True)

    data = dict(arm=arm, best=a.best, placements=P, sets=sets)
    (out / 'data.json').write_text(json.dumps(data, separators=(',', ':')))
    tpl = (HERE / 'viewer_template.html').read_text()
    (out / 'index.html').write_text(tpl)
    files = sorted(f for f in out.rglob('*') if f.is_file())
    print(f'{len(files)} files, {sum(f.stat().st_size for f in files) / 1e6:.1f} MB; data.json {(out / "data.json").stat().st_size / 1e3:.0f} KB')


if __name__ == '__main__':
    main()
